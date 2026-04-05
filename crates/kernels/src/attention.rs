//! Scaled Dot-Product Attention kernel.
//!
//! Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
//!
//! This is the heart of every transformer. We implement:
//! 1. Naive attention (materializes full N×N attention matrix) — correctness baseline
//! 2. Flash-style attention (online softmax, O(N) memory) — the real deal
//!
//! The naive kernel is memory-bound at O(N²) — Flash fixes this by never
//! materializing the full attention matrix, computing softmax in tiles
//! with online normalization (the Milakov-Gimelshein trick).

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Naive scaled dot-product attention.
/// Q: [batch, seq_len, head_dim]
/// K: [batch, seq_len, head_dim]
/// V: [batch, seq_len, head_dim]
/// out: [batch, seq_len, head_dim]
///
/// This materializes the full seq_len × seq_len attention matrix.
/// Fine for short sequences, death for long ones.
const NAIVE_ATTENTION_SRC: &str = r#"
extern "C" __global__ void warp_attention_naive(
    float *out,           // [B, N, D]
    const float *Q,       // [B, N, D]
    const float *K,       // [B, N, D]
    const float *V,       // [B, N, D]
    unsigned int B,       // batch size
    unsigned int N,       // sequence length
    unsigned int D,       // head dimension
    float scale,          // 1/sqrt(D)
    unsigned int causal   // 1 for causal mask, 0 for full
) {
    // Each thread handles one output element: out[b][i][d]
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = B * N * D;
    if (idx >= total) return;

    unsigned int d = idx % D;
    unsigned int i = (idx / D) % N;
    unsigned int b = idx / (N * D);

    // Compute attention scores for row i: score[j] = Q[i] · K[j] * scale
    // Then softmax, then weighted sum of V

    // Step 1: Find max score (for numerical stability in softmax)
    float max_score = -1e30f;
    unsigned int end_j = causal ? (i + 1) : N;
    for (unsigned int j = 0; j < end_j; j++) {
        float score = 0.0f;
        for (unsigned int dd = 0; dd < D; dd++) {
            score += Q[b * N * D + i * D + dd] * K[b * N * D + j * D + dd];
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }

    // Step 2: Compute exp(score - max) and sum
    float sum_exp = 0.0f;
    // We'll also accumulate the output on the fly
    float out_val = 0.0f;

    for (unsigned int j = 0; j < end_j; j++) {
        float score = 0.0f;
        for (unsigned int dd = 0; dd < D; dd++) {
            score += Q[b * N * D + i * D + dd] * K[b * N * D + j * D + dd];
        }
        score *= scale;
        float w = expf(score - max_score);
        sum_exp += w;
        out_val += w * V[b * N * D + j * D + d];
    }

    // Step 3: Normalize
    out[b * N * D + i * D + d] = out_val / sum_exp;
}
"#;

/// Flash-style attention with online softmax.
/// Processes K/V in tiles, maintaining running softmax statistics.
/// Memory: O(N*D) instead of O(N²). Much faster for long sequences.
///
/// This is a simplified single-pass Flash Attention:
/// - One thread block per query row
/// - Iterates over K/V tiles sequentially
/// - Maintains running max and sum for online softmax correction
const FLASH_ATTENTION_SRC: &str = r#"
extern "C" __global__ void warp_flash_attention(
    float *out,           // [B, N, D]
    const float *Q,       // [B, N, D]
    const float *K,       // [B, N, D]
    const float *V,       // [B, N, D]
    unsigned int B,
    unsigned int N,
    unsigned int D,
    float scale,
    unsigned int causal
) {
    // Each block handles one query position: (batch, query_idx)
    unsigned int b = blockIdx.y;
    unsigned int i = blockIdx.x;  // query index
    unsigned int d = threadIdx.x; // head dim index

    if (b >= B || i >= N || d >= D) return;

    const float *q_row = Q + b * N * D + i * D;
    float *out_row = out + b * N * D + i * D;

    // Online softmax state
    float running_max = -1e30f;
    float running_sum = 0.0f;
    float running_out = 0.0f;

    unsigned int end_j = causal ? (i + 1) : N;

    // Iterate over all key/value positions
    for (unsigned int j = 0; j < end_j; j++) {
        const float *k_row = K + b * N * D + j * D;

        // Compute Q[i] · K[j] — all threads contribute their dimension
        // Use warp-level reduction for the dot product
        float partial = q_row[d] * k_row[d];

        // Warp reduction for dot product
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset);

        // Broadcast score from lane 0
        float score = __shfl_sync(0xffffffff, partial, 0) * scale;

        // Online softmax update (Milakov-Gimelshein trick)
        float new_max = fmaxf(running_max, score);
        float correction = expf(running_max - new_max);
        float weight = expf(score - new_max);

        // Correct running sum and output for new max
        running_sum = running_sum * correction + weight;
        running_out = running_out * correction + weight * V[b * N * D + j * D + d];
        running_max = new_max;
    }

    // Final normalization
    out_row[d] = running_out / running_sum;
}
"#;

/// Launch naive attention.
pub fn attention_naive(
    cache: &KernelCache,
    device: &WarpDevice,
    q: &GpuTensor<f32>,
    k: &GpuTensor<f32>,
    v: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    batch: u32,
    seq_len: u32,
    head_dim: u32,
    causal: bool,
) -> Result<(), DeviceError> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let causal_u32 = if causal { 1u32 } else { 0u32 };
    let total = batch * seq_len * head_dim;

    let f = cache.get_or_compile(device, NAIVE_ATTENTION_SRC, "warp_attention_naive")?;
    let cfg = LaunchConfig::for_num_elems(total);

    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&q.data)
            .arg(&k.data)
            .arg(&v.data)
            .arg(&batch)
            .arg(&seq_len)
            .arg(&head_dim)
            .arg(&scale)
            .arg(&causal_u32)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// Launch flash-style attention with online softmax.
/// head_dim must be <= 32 (one warp handles the dot product).
/// For larger head_dim, we'd need multi-warp reduction.
pub fn attention_flash(
    cache: &KernelCache,
    device: &WarpDevice,
    q: &GpuTensor<f32>,
    k: &GpuTensor<f32>,
    v: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    batch: u32,
    seq_len: u32,
    head_dim: u32,
    causal: bool,
) -> Result<(), DeviceError> {
    // For head_dim > 32, delegate to attention_ext which handles larger dims
    if head_dim > 32 {
        return crate::attention_ext::attention_best(
            cache, device, q, k, v, out, batch, seq_len, head_dim, causal);
    }

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let causal_u32 = if causal { 1u32 } else { 0u32 };

    let f = cache.get_or_compile(device, FLASH_ATTENTION_SRC, "warp_flash_attention")?;

    // One block per (query_position, batch)
    // head_dim threads per block (one warp)
    let cfg = LaunchConfig {
        grid_dim: (seq_len, batch, 1),
        block_dim: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&q.data)
            .arg(&k.data)
            .arg(&v.data)
            .arg(&batch)
            .arg(&seq_len)
            .arg(&head_dim)
            .arg(&scale)
            .arg(&causal_u32)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// Flash Attention v2 — tiled Q×K with online softmax.
///
/// This is the real deal: processes queries in BLOCK_M tiles, keys/values in
/// BLOCK_N tiles, using shared memory for K tiles so every thread in the block
/// benefits from the cached KV data. Online softmax (Milakov-Gimelshein) means
/// we never materialize the full N×N attention matrix.
///
/// Architecture:
/// - Each thread block handles BLOCK_M queries for one batch*head
/// - One thread per query row (threadIdx.x = query within tile)
/// - K tile loaded cooperatively into shared memory
/// - Each thread computes full dot products Q[qi] . K[j] over HEAD_DIM
/// - Online softmax: running max + sum correction across KV tiles
/// - V accumulation in registers, corrected each tile
///
/// HEAD_DIM is baked in via JIT (#define) for maximum performance.
fn flash_attn_v2_src(head_dim: u32) -> String {
    format!(r#"
#define HEAD_DIM {head_dim}
#define BLOCK_M 64
#define BLOCK_N 64

extern "C" __global__ void warp_flash_attn_v2(
    float *out,        // [B*H, N, D]
    const float *Q,    // [B*H, N, D]
    const float *K,    // [B*H, N, D]
    const float *V,    // [B*H, N, D]
    unsigned int N,    // sequence length
    unsigned int D,    // head dimension (== HEAD_DIM)
    float scale,       // 1/sqrt(D)
    unsigned int causal
) {{
    // blockIdx.x = which batch*head
    // blockIdx.y = which query tile (query_start = blockIdx.y * BLOCK_M)
    unsigned int bh = blockIdx.x;
    unsigned int q_start = blockIdx.y * BLOCK_M;
    unsigned int tid = threadIdx.x;  // 0..BLOCK_M-1, one thread per query row

    unsigned int qi = q_start + tid;  // global query index

    // Pointers for this batch*head
    const float *Q_bh = Q + bh * N * D;
    const float *K_bh = K + bh * N * D;
    const float *V_bh = V + bh * N * D;
    float *O_bh = out + bh * N * D;

    // Shared memory: K and V tiles [BLOCK_N][HEAD_DIM] each
    extern __shared__ float smem[];  // (2 * BLOCK_N * HEAD_DIM) floats
    float *K_smem = smem;
    float *V_smem = smem + BLOCK_N * HEAD_DIM;

    // Load this thread's query row into registers
    float q_reg[HEAD_DIM];
    if (qi < N) {{
        for (unsigned int d = 0; d < HEAD_DIM; d++) {{
            q_reg[d] = Q_bh[qi * D + d];
        }}
    }}

    // Per-query accumulators
    float m_i = -1e30f;   // running max
    float l_i = 0.0f;     // running sum of exp
    float O_acc[HEAD_DIM]; // output accumulator
    for (unsigned int d = 0; d < HEAD_DIM; d++) {{
        O_acc[d] = 0.0f;
    }}

    // Number of KV tiles
    unsigned int kv_end = (causal && qi < N) ? (qi + 1) : N;
    unsigned int num_kv_tiles = (N + BLOCK_N - 1) / BLOCK_N;  // iterate all, mask per-query

    for (unsigned int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {{
        unsigned int k_start = kv_tile * BLOCK_N;

        // Early exit: if causal and entire tile is past our query position
        if (causal && qi < N && k_start > qi) {{
            __syncthreads();  // still need to sync for other threads
            continue;
        }}

        // Cooperative load of K tile into shared memory
        // BLOCK_M threads load BLOCK_N * HEAD_DIM elements
        unsigned int total_k_elems = BLOCK_N * HEAD_DIM;
        for (unsigned int idx = tid; idx < total_k_elems; idx += BLOCK_M) {{
            unsigned int row = idx / HEAD_DIM;
            unsigned int col = idx % HEAD_DIM;
            unsigned int gk = k_start + row;
            K_smem[row * HEAD_DIM + col] = (gk < N) ? K_bh[gk * D + col] : 0.0f;
            V_smem[row * HEAD_DIM + col] = (gk < N) ? V_bh[gk * D + col] : 0.0f;
        }}
        __syncthreads();

        if (qi < N) {{
            // Compute scores and online softmax for this KV tile
            float tile_max = -1e30f;
            float scores[BLOCK_N];

            // Compute Q[qi] . K[j] for all j in this tile
            for (unsigned int j = 0; j < BLOCK_N; j++) {{
                unsigned int gk = k_start + j;
                if (gk < N && (!causal || gk <= qi)) {{
                    float dot = 0.0f;
                    for (unsigned int d = 0; d < HEAD_DIM; d++) {{
                        dot += q_reg[d] * K_smem[j * HEAD_DIM + d];
                    }}
                    scores[j] = dot * scale;
                    if (scores[j] > tile_max) tile_max = scores[j];
                }} else {{
                    scores[j] = -1e30f;
                }}
            }}

            // Online softmax correction
            float m_new = fmaxf(m_i, tile_max);
            float correction = expf(m_i - m_new);

            // Correct running state
            l_i *= correction;
            for (unsigned int d = 0; d < HEAD_DIM; d++) {{
                O_acc[d] *= correction;
            }}

            // Accumulate P @ V
            float tile_sum = 0.0f;
            for (unsigned int j = 0; j < BLOCK_N; j++) {{
                unsigned int gk = k_start + j;
                float p = expf(scores[j] - m_new);
                tile_sum += p;

                if (gk < N && (!causal || gk <= qi)) {{
                    for (unsigned int d = 0; d < HEAD_DIM; d++) {{
                        O_acc[d] += p * V_smem[j * HEAD_DIM + d];
                    }}
                }}
            }}

            l_i += tile_sum;
            m_i = m_new;
        }}

        __syncthreads();
    }}

    // Final normalization and write output
    if (qi < N && l_i > 0.0f) {{
        for (unsigned int d = 0; d < HEAD_DIM; d++) {{
            O_bh[qi * D + d] = O_acc[d] / l_i;
        }}
    }}
}}
"#)
}

/// Launch Flash Attention v2 with tiled Q×K and online softmax.
///
/// This is the high-performance path for head_dim >= 32. Uses BLOCK_M=64 query
/// tiles, BLOCK_N=64 KV tiles, shared memory for K tiles, and online softmax
/// so the full N×N attention matrix is never materialized.
///
/// HEAD_DIM is baked in as a compile-time constant via JIT for register-level
/// optimization of the inner dot product loops.
pub fn flash_attention_v2(
    cache: &KernelCache,
    device: &WarpDevice,
    q: &GpuTensor<f32>,   // [B*H, N, D]
    k: &GpuTensor<f32>,
    v: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    batch_heads: u32,      // B * num_heads
    seq_len: u32,
    head_dim: u32,
    causal: bool,
) -> Result<(), DeviceError> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let causal_u32 = if causal { 1u32 } else { 0u32 };

    let src = flash_attn_v2_src(head_dim);
    let f = cache.get_or_compile(device, &src, "warp_flash_attn_v2")?;

    let block_m = 64u32;
    let block_n = 64u32;
    let num_q_tiles = (seq_len + block_m - 1) / block_m;

    // Shared memory: K + V tiles [BLOCK_N][HEAD_DIM] each
    let smem_bytes = 2 * block_n * head_dim * 4;

    let cfg = LaunchConfig {
        grid_dim: (batch_heads, num_q_tiles, 1),
        block_dim: (block_m, 1, 1),  // one thread per query row in tile
        shared_mem_bytes: smem_bytes,
    };

    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&q.data)
            .arg(&k.data)
            .arg(&v.data)
            .arg(&seq_len)
            .arg(&head_dim)
            .arg(&scale)
            .arg(&causal_u32)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// Smart attention dispatch: picks the best kernel for the given parameters.
///
/// - head_dim >= 32: Flash Attention v2 (tiled, online softmax, shared memory)
/// - head_dim < 32: Original flash attention (warp-level, single-pass)
pub fn attention_best(
    cache: &KernelCache,
    device: &WarpDevice,
    q: &GpuTensor<f32>,
    k: &GpuTensor<f32>,
    v: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    batch_heads: u32,
    seq_len: u32,
    head_dim: u32,
    causal: bool,
) -> Result<(), DeviceError> {
    if head_dim >= 32 {
        flash_attention_v2(cache, device, q, k, v, out, batch_heads, seq_len, head_dim, causal)
    } else {
        attention_flash(cache, device, q, k, v, out, batch_heads, seq_len, head_dim, causal)
    }
}

/// CPU reference attention for validation.
pub fn cpu_attention(
    q: &[f32], k: &[f32], v: &[f32], out: &mut [f32],
    batch: usize, seq_len: usize, head_dim: usize, causal: bool,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();

    for b in 0..batch {
        for i in 0..seq_len {
            let end_j = if causal { i + 1 } else { seq_len };

            // Compute scores
            let mut scores = vec![0.0f32; end_j];
            for j in 0..end_j {
                let mut dot = 0.0f32;
                for dd in 0..head_dim {
                    dot += q[b * seq_len * head_dim + i * head_dim + dd]
                        * k[b * seq_len * head_dim + j * head_dim + dd];
                }
                scores[j] = dot * scale;
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            for s in &mut scores {
                *s /= sum_exp;
            }

            // Weighted sum of V
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for j in 0..end_j {
                    val += scores[j] * v[b * seq_len * head_dim + j * head_dim + d];
                }
                out[b * seq_len * head_dim + i * head_dim + d] = val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::{DType, Shape};

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        let dev = WarpDevice::new(0).ok()?;
        Some((dev, KernelCache::new()))
    }

    #[test]
    fn cpu_attention_basic() {
        let (b, n, d) = (1, 4, 8);
        let q: Vec<f32> = (0..b*n*d).map(|i| (i as f32 * 0.1) - 1.6).collect();
        let k = q.clone();
        let v: Vec<f32> = (0..b*n*d).map(|i| i as f32 * 0.05).collect();
        let mut out = vec![0.0f32; b * n * d];

        cpu_attention(&q, &k, &v, &mut out, b, n, d, false);

        // Output should be valid (not NaN, not zero)
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(out.iter().any(|x| *x != 0.0));
    }

    #[test]
    fn gpu_naive_attention() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (b, n, d) = (2u32, 16u32, 8u32);
        let total = (b * n * d) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, d as usize]);

        let q_data: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let k_data: Vec<f32> = (0..total).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let v_data: Vec<f32> = (0..total).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

        // CPU reference
        let mut cpu_out = vec![0.0f32; total];
        cpu_attention(&q_data, &k_data, &v_data, &mut cpu_out, b as usize, n as usize, d as usize, false);

        // GPU
        let q = GpuTensor::from_host(&dev, &q_data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &k_data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &v_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        attention_naive(&cache, &dev, &q, &k, &v, &mut out, b, n, d, false).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();
        let max_err: f32 = gpu_out.iter().zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Naive attention B={b} N={n} D={d}: max error = {max_err:.6}");
        assert!(max_err < 1e-3, "Max error {max_err} too high");
    }

    #[test]
    fn gpu_naive_attention_causal() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (b, n, d) = (1u32, 8u32, 16u32);
        let total = (b * n * d) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, d as usize]);

        let q_data: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let k_data: Vec<f32> = (0..total).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let v_data: Vec<f32> = (0..total).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

        let mut cpu_out = vec![0.0f32; total];
        cpu_attention(&q_data, &k_data, &v_data, &mut cpu_out, b as usize, n as usize, d as usize, true);

        let q = GpuTensor::from_host(&dev, &q_data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &k_data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &v_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        attention_naive(&cache, &dev, &q, &k, &v, &mut out, b, n, d, true).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();
        let max_err: f32 = gpu_out.iter().zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Causal attention B={b} N={n} D={d}: max error = {max_err:.6}");
        assert!(max_err < 1e-3, "Max error {max_err} too high");
    }

    #[test]
    fn gpu_flash_attention() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (b, n, d) = (2u32, 32u32, 32u32); // d=32 fits in one warp
        let total = (b * n * d) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, d as usize]);

        let q_data: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let k_data: Vec<f32> = (0..total).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let v_data: Vec<f32> = (0..total).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

        // CPU reference
        let mut cpu_out = vec![0.0f32; total];
        cpu_attention(&q_data, &k_data, &v_data, &mut cpu_out, b as usize, n as usize, d as usize, false);

        // GPU flash
        let q = GpuTensor::from_host(&dev, &q_data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &k_data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &v_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        attention_flash(&cache, &dev, &q, &k, &v, &mut out, b, n, d, false).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();
        let max_err: f32 = gpu_out.iter().zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Flash attention B={b} N={n} D={d}: max error = {max_err:.6}");
        assert!(max_err < 1e-2, "Max error {max_err} too high");
    }

    #[test]
    fn gpu_flash_attention_causal() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (b, n, d) = (1u32, 64u32, 32u32);
        let total = (b * n * d) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, d as usize]);

        let q_data: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.08).collect();
        let k_data: Vec<f32> = (0..total).map(|i| ((i % 11) as f32 - 5.0) * 0.08).collect();
        let v_data: Vec<f32> = (0..total).map(|i| ((i % 7) as f32 - 3.0) * 0.08).collect();

        let mut cpu_out = vec![0.0f32; total];
        cpu_attention(&q_data, &k_data, &v_data, &mut cpu_out, b as usize, n as usize, d as usize, true);

        let q = GpuTensor::from_host(&dev, &q_data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &k_data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &v_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        attention_flash(&cache, &dev, &q, &k, &v, &mut out, b, n, d, true).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();
        let max_err: f32 = gpu_out.iter().zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Flash causal attention B={b} N={n} D={d}: max error = {max_err:.6}");
        assert!(max_err < 1e-2, "Max error {max_err} too high");
    }

    #[test]
    fn attention_perf_comparison() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (b, n, d) = (1u32, 128u32, 32u32);
        let total = (b * n * d) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, d as usize]);

        let q_data: Vec<f32> = (0..total).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let k_data: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let v_data: Vec<f32> = (0..total).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();

        let q = GpuTensor::from_host(&dev, &q_data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &k_data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &v_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        let iters = 100;

        // Naive warmup + bench
        attention_naive(&cache, &dev, &q, &k, &v, &mut out, b, n, d, true).unwrap();
        dev.synchronize().unwrap();
        let start = std::time::Instant::now();
        for _ in 0..iters {
            attention_naive(&cache, &dev, &q, &k, &v, &mut out, b, n, d, true).unwrap();
        }
        dev.synchronize().unwrap();
        let naive_time = start.elapsed();

        // Flash warmup + bench
        attention_flash(&cache, &dev, &q, &k, &v, &mut out, b, n, d, true).unwrap();
        dev.synchronize().unwrap();
        let start = std::time::Instant::now();
        for _ in 0..iters {
            attention_flash(&cache, &dev, &q, &k, &v, &mut out, b, n, d, true).unwrap();
        }
        dev.synchronize().unwrap();
        let flash_time = start.elapsed();

        println!("\nAttention perf (B={b}, N={n}, D={d}, causal, {iters} iters):");
        println!("  Naive: {:.3}ms avg", naive_time.as_secs_f64() * 1000.0 / iters as f64);
        println!("  Flash: {:.3}ms avg", flash_time.as_secs_f64() * 1000.0 / iters as f64);
        println!("  Speedup: {:.2}x", naive_time.as_secs_f64() / flash_time.as_secs_f64());
    }

    #[test]
    fn flash_attn_v2_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // seq_len=128, head_dim=64, batch_heads=4, non-causal
        let (bh, n, d) = (4u32, 128u32, 64u32);
        let total = (bh * n * d) as usize;
        let shape = Shape::from_static(&[bh as usize, n as usize, d as usize]);

        let q_data: Vec<f32> = (0..total).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let k_data: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let v_data: Vec<f32> = (0..total).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();

        // CPU reference
        let mut cpu_out = vec![0.0f32; total];
        cpu_attention(&q_data, &k_data, &v_data, &mut cpu_out, bh as usize, n as usize, d as usize, false);

        // Naive GPU reference
        let q = GpuTensor::from_host(&dev, &q_data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &k_data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &v_data, shape.clone(), DType::F32).unwrap();
        let mut naive_out = GpuTensor::<f32>::zeros(&dev, shape.clone(), DType::F32).unwrap();
        attention_naive(&cache, &dev, &q, &k, &v, &mut naive_out, bh, n, d, false).unwrap();
        dev.synchronize().unwrap();
        let naive_result = naive_out.to_host(&dev).unwrap();

        // Flash Attention v2
        let mut flash_out = GpuTensor::<f32>::zeros(&dev, shape.clone(), DType::F32).unwrap();
        flash_attention_v2(&cache, &dev, &q, &k, &v, &mut flash_out, bh, n, d, false).unwrap();
        dev.synchronize().unwrap();
        let flash_result = flash_out.to_host(&dev).unwrap();

        // Compare flash v2 vs CPU
        let max_err_cpu: f32 = flash_result.iter().zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Flash v2 vs CPU (BH={bh} N={n} D={d} non-causal): max error = {max_err_cpu:.6}");
        assert!(max_err_cpu < 0.01, "Flash v2 vs CPU max error {max_err_cpu} too high");

        // Compare flash v2 vs naive GPU
        let max_err_naive: f32 = flash_result.iter().zip(naive_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Flash v2 vs naive GPU: max error = {max_err_naive:.6}");
        assert!(max_err_naive < 0.01, "Flash v2 vs naive max error {max_err_naive} too high");
    }

    #[test]
    fn flash_attn_v2_causal() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (bh, n, d) = (4u32, 128u32, 64u32);
        let total = (bh * n * d) as usize;
        let shape = Shape::from_static(&[bh as usize, n as usize, d as usize]);

        let q_data: Vec<f32> = (0..total).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let k_data: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let v_data: Vec<f32> = (0..total).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();

        // CPU reference with causal
        let mut cpu_out = vec![0.0f32; total];
        cpu_attention(&q_data, &k_data, &v_data, &mut cpu_out, bh as usize, n as usize, d as usize, true);

        // Flash Attention v2 causal
        let q = GpuTensor::from_host(&dev, &q_data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &k_data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &v_data, shape.clone(), DType::F32).unwrap();
        let mut flash_out = GpuTensor::<f32>::zeros(&dev, shape.clone(), DType::F32).unwrap();
        flash_attention_v2(&cache, &dev, &q, &k, &v, &mut flash_out, bh, n, d, true).unwrap();
        dev.synchronize().unwrap();
        let flash_result = flash_out.to_host(&dev).unwrap();

        let max_err: f32 = flash_result.iter().zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Flash v2 causal (BH={bh} N={n} D={d}): max error = {max_err:.6}");
        assert!(max_err < 0.01, "Flash v2 causal max error {max_err} too high");
    }

    #[test]
    fn flash_attn_v2_head_dim_128() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // LLaMA-style: head_dim=128
        let (bh, n, d) = (4u32, 128u32, 128u32);
        let total = (bh * n * d) as usize;
        let shape = Shape::from_static(&[bh as usize, n as usize, d as usize]);

        let q_data: Vec<f32> = (0..total).map(|i| ((i % 17) as f32 - 8.0) * 0.03).collect();
        let k_data: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.03).collect();
        let v_data: Vec<f32> = (0..total).map(|i| ((i % 7) as f32 - 3.0) * 0.03).collect();

        // CPU reference
        let mut cpu_out = vec![0.0f32; total];
        cpu_attention(&q_data, &k_data, &v_data, &mut cpu_out, bh as usize, n as usize, d as usize, false);

        // Flash v2
        let q = GpuTensor::from_host(&dev, &q_data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &k_data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &v_data, shape.clone(), DType::F32).unwrap();
        let mut flash_out = GpuTensor::<f32>::zeros(&dev, shape.clone(), DType::F32).unwrap();
        flash_attention_v2(&cache, &dev, &q, &k, &v, &mut flash_out, bh, n, d, false).unwrap();
        dev.synchronize().unwrap();
        let flash_result = flash_out.to_host(&dev).unwrap();

        let max_err: f32 = flash_result.iter().zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Flash v2 D=128 (BH={bh} N={n}): max error = {max_err:.6}");
        assert!(max_err < 0.01, "Flash v2 D=128 max error {max_err} too high");

        // Also test causal with D=128
        let mut cpu_out_causal = vec![0.0f32; total];
        cpu_attention(&q_data, &k_data, &v_data, &mut cpu_out_causal, bh as usize, n as usize, d as usize, true);

        let mut flash_out_causal = GpuTensor::<f32>::zeros(&dev, shape.clone(), DType::F32).unwrap();
        flash_attention_v2(&cache, &dev, &q, &k, &v, &mut flash_out_causal, bh, n, d, true).unwrap();
        dev.synchronize().unwrap();
        let flash_causal_result = flash_out_causal.to_host(&dev).unwrap();

        let max_err_c: f32 = flash_causal_result.iter().zip(cpu_out_causal.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Flash v2 D=128 causal: max error = {max_err_c:.6}");
        assert!(max_err_c < 0.01, "Flash v2 D=128 causal max error {max_err_c} too high");
    }

    #[test]
    fn attention_best_dispatch() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // D=64 should use Flash v2
        let (bh, n, d) = (2u32, 32u32, 64u32);
        let total = (bh * n * d) as usize;
        let shape = Shape::from_static(&[bh as usize, n as usize, d as usize]);

        let q_data: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.08).collect();
        let k_data: Vec<f32> = (0..total).map(|i| ((i % 11) as f32 - 5.0) * 0.08).collect();
        let v_data: Vec<f32> = (0..total).map(|i| ((i % 7) as f32 - 3.0) * 0.08).collect();

        let mut cpu_out = vec![0.0f32; total];
        cpu_attention(&q_data, &k_data, &v_data, &mut cpu_out, bh as usize, n as usize, d as usize, false);

        let q = GpuTensor::from_host(&dev, &q_data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &k_data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &v_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        attention_best(&cache, &dev, &q, &k, &v, &mut out, bh, n, d, false).unwrap();
        dev.synchronize().unwrap();
        let result = out.to_host(&dev).unwrap();

        let max_err: f32 = result.iter().zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("attention_best D=64: max error = {max_err:.6}");
        assert!(max_err < 0.01, "attention_best D=64 max error {max_err} too high");
        assert!(result.iter().all(|v| v.is_finite()));
        println!("attention_best D=64: dispatched to Flash v2");
    }
}

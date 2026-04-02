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
}

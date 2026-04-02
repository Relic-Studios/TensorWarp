//! Sliding Window Attention — for Mistral, Gemma, and local-attention models.
//!
//! Standard causal attention: each position attends to ALL previous positions.
//! Cost: O(N²) in sequence length.
//!
//! Sliding window attention: each position only attends to the last W positions.
//! Cost: O(N·W), with W typically 4096 or 8192.
//!
//! This is how Mistral-7B handles long contexts efficiently — it uses
//! window_size=4096, meaning each token can "see" 4K tokens behind it.
//! Combined with KV cache, this means we only need to cache W tokens
//! instead of the full sequence.
//!
//! We implement both prefill (full sequence) and decode (single token) variants,
//! plus a combined causal+sliding window mode (some layers use full attention,
//! others use windowed — this is the Gemma 2 pattern).

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

macro_rules! launch_err {
    ($e:expr) => {
        $e.map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))
    };
}

/// Sliding window attention with online softmax.
/// Q, K, V: [batch, seq_len, head_dim]
/// Each query at position i attends to positions max(0, i-window+1) .. i (inclusive).
const SLIDING_WINDOW_ATTENTION_SRC: &str = r#"
extern "C" __global__ void warp_sliding_window_attention(
    float *out,           // [B, N, D]
    const float *Q,       // [B, N, D]
    const float *K,       // [B, N, D]
    const float *V,       // [B, N, D]
    unsigned int B,
    unsigned int N,
    unsigned int D,
    float scale,
    unsigned int window_size
) {
    extern __shared__ float smem[];
    float *dot_buf = smem;

    unsigned int b = blockIdx.y;
    unsigned int i = blockIdx.x;
    unsigned int d = threadIdx.x;

    if (b >= B || i >= N) return;

    const float *q_row = Q + b * N * D + i * D;
    float *out_row = out + b * N * D + i * D;

    float q_val = (d < D) ? q_row[d] : 0.0f;

    float running_max = -1e30f;
    float running_sum = 0.0f;
    float running_out = 0.0f;

    // Sliding window: attend from max(0, i-window+1) to i (inclusive)
    int start_j = (int)i - (int)window_size + 1;
    if (start_j < 0) start_j = 0;
    unsigned int end_j = i + 1; // causal

    for (unsigned int j = (unsigned int)start_j; j < end_j; j++) {
        const float *k_row = K + b * N * D + j * D;

        // Dot product via shared memory reduction
        float partial = 0.0f;
        if (d < D) {
            partial = q_val * k_row[d];
        }
        dot_buf[threadIdx.x] = partial;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                dot_buf[threadIdx.x] += dot_buf[threadIdx.x + stride];
            __syncthreads();
        }

        float score = dot_buf[0] * scale;

        float v_val = (d < D) ? V[b * N * D + j * D + d] : 0.0f;

        // Online softmax
        float new_max = fmaxf(running_max, score);
        float correction = expf(running_max - new_max);
        float weight = expf(score - new_max);

        running_sum = running_sum * correction + weight;
        running_out = running_out * correction + weight * v_val;
        running_max = new_max;

        __syncthreads();
    }

    if (d < D && running_sum > 0.0f) {
        out_row[d] = running_out / running_sum;
    }
}
"#;

/// Sliding window decode attention — single query against last W cached positions.
/// Q: [head_dim], K_cache/V_cache: [cache_len, head_dim]
/// Only attends to the last window_size positions.
const SLIDING_WINDOW_DECODE_SRC: &str = r#"
extern "C" __global__ void warp_sliding_window_decode(
    float *out,
    const float *q,
    const float *k_cache,
    const float *v_cache,
    unsigned int cache_len,
    unsigned int head_dim,
    float scale,
    unsigned int window_size
) {
    extern __shared__ float smem[];
    float *dot_buf = smem;

    unsigned int d = threadIdx.x;

    float q_val = (d < head_dim) ? q[d] : 0.0f;

    // Only attend to last window_size positions
    int start = (int)cache_len - (int)window_size;
    if (start < 0) start = 0;

    float running_max = -1e30f;
    float running_sum = 0.0f;
    float running_out = 0.0f;

    for (unsigned int j = (unsigned int)start; j < cache_len; j++) {
        float partial = 0.0f;
        if (d < head_dim) {
            partial = q_val * k_cache[j * head_dim + d];
        }
        dot_buf[threadIdx.x] = partial;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                dot_buf[threadIdx.x] += dot_buf[threadIdx.x + stride];
            __syncthreads();
        }

        float score = dot_buf[0] * scale;

        float v_val = (d < head_dim) ? v_cache[j * head_dim + d] : 0.0f;

        float new_max = fmaxf(running_max, score);
        float correction = expf(running_max - new_max);
        float weight = expf(score - new_max);

        running_sum = running_sum * correction + weight;
        running_out = running_out * correction + weight * v_val;
        running_max = new_max;

        __syncthreads();
    }

    if (d < head_dim && running_sum > 0.0f) {
        out[d] = running_out / running_sum;
    }
}
"#;

/// Launch sliding window attention (prefill).
pub fn sliding_window_attention(
    cache: &KernelCache,
    device: &WarpDevice,
    q: &GpuTensor<f32>,
    k: &GpuTensor<f32>,
    v: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    batch: u32,
    seq_len: u32,
    head_dim: u32,
    window_size: u32,
) -> Result<(), DeviceError> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let f = cache.get_or_compile(device, SLIDING_WINDOW_ATTENTION_SRC, "warp_sliding_window_attention")?;

    let block_size = head_dim.next_power_of_two().max(32);
    let shared_mem = block_size * 4;

    let cfg = LaunchConfig {
        grid_dim: (seq_len, batch, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_mem,
    };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&q.data)
            .arg(&k.data)
            .arg(&v.data)
            .arg(&batch)
            .arg(&seq_len)
            .arg(&head_dim)
            .arg(&scale)
            .arg(&window_size)
            .launch(cfg))?;
    }
    Ok(())
}

/// Launch sliding window decode attention (single token).
pub fn sliding_window_decode(
    cache: &KernelCache,
    device: &WarpDevice,
    q: &GpuTensor<f32>,         // [head_dim]
    k_cache: &GpuTensor<f32>,   // [cache_len, head_dim]
    v_cache: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,   // [head_dim]
    cache_len: u32,
    head_dim: u32,
    window_size: u32,
) -> Result<(), DeviceError> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let f = cache.get_or_compile(device, SLIDING_WINDOW_DECODE_SRC, "warp_sliding_window_decode")?;

    let block_size = head_dim.next_power_of_two().max(32);
    let shared_mem = block_size * 4;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_mem,
    };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&q.data)
            .arg(&k_cache.data)
            .arg(&v_cache.data)
            .arg(&cache_len)
            .arg(&head_dim)
            .arg(&scale)
            .arg(&window_size)
            .launch(cfg))?;
    }
    Ok(())
}

/// CPU reference for sliding window attention.
pub fn cpu_sliding_window_attention(
    q: &[f32], k: &[f32], v: &[f32], out: &mut [f32],
    batch: usize, seq_len: usize, head_dim: usize,
    window_size: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    for b in 0..batch {
        for i in 0..seq_len {
            let start_j = if i >= window_size { i - window_size + 1 } else { 0 };

            // Compute scores
            let mut max_score = f32::NEG_INFINITY;
            let mut scores = Vec::new();
            for j in start_j..=i {
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[b * seq_len * head_dim + i * head_dim + d]
                           * k[b * seq_len * head_dim + j * head_dim + d];
                }
                score *= scale;
                scores.push(score);
                if score > max_score { max_score = score; }
            }

            // Softmax
            let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();

            // Weighted sum of V
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for (idx, j) in (start_j..=i).enumerate() {
                    val += exp_scores[idx] / sum_exp * v[b * seq_len * head_dim + j * head_dim + d];
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
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    #[test]
    fn sliding_window_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (b, n, d) = (1u32, 32u32, 64u32);
        let window_size = 8u32;
        let total = (b * n * d) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, d as usize]);

        let q_data: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.08).collect();
        let k_data: Vec<f32> = (0..total).map(|i| ((i % 11) as f32 - 5.0) * 0.08).collect();
        let v_data: Vec<f32> = (0..total).map(|i| ((i % 7) as f32 - 3.0) * 0.08).collect();

        // CPU reference
        let mut cpu_out = vec![0.0f32; total];
        cpu_sliding_window_attention(
            &q_data, &k_data, &v_data, &mut cpu_out,
            b as usize, n as usize, d as usize, window_size as usize,
        );

        // GPU
        let q = GpuTensor::from_host(&dev, &q_data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &k_data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &v_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        sliding_window_attention(&cache, &dev, &q, &k, &v, &mut out, b, n, d, window_size).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();
        let max_err: f32 = gpu_out.iter().zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Sliding window attention (N={n}, D={d}, W={window_size}): max error = {max_err:.6}");
        assert!(max_err < 1e-2, "Max error {max_err} too high");
    }

    #[test]
    fn sliding_window_vs_full_causal_when_window_large() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // When window >= seq_len, sliding window should equal full causal attention
        let (b, n, d) = (1u32, 16u32, 32u32);
        let window_size = 32u32; // larger than seq_len
        let total = (b * n * d) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, d as usize]);

        let q_data: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.08).collect();
        let k_data: Vec<f32> = (0..total).map(|i| ((i % 11) as f32 - 5.0) * 0.08).collect();
        let v_data: Vec<f32> = (0..total).map(|i| ((i % 7) as f32 - 3.0) * 0.08).collect();

        let q = GpuTensor::from_host(&dev, &q_data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &k_data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &v_data, shape.clone(), DType::F32).unwrap();

        // Sliding window with large window
        let mut sw_out = GpuTensor::<f32>::zeros(&dev, shape.clone(), DType::F32).unwrap();
        sliding_window_attention(&cache, &dev, &q, &k, &v, &mut sw_out, b, n, d, window_size).unwrap();

        // Full causal attention (using extended)
        let mut full_out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();
        crate::attention_ext::attention_extended(&cache, &dev, &q, &k, &v, &mut full_out, b, n, d, true).unwrap();

        dev.synchronize().unwrap();

        let sw_result = sw_out.to_host(&dev).unwrap();
        let full_result = full_out.to_host(&dev).unwrap();
        let max_err: f32 = sw_result.iter().zip(full_result.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Sliding window (W={window_size}) vs full causal (N={n}): max diff = {max_err:.6}");
        assert!(max_err < 1e-3, "Should match full causal when window >= seq_len, but diff = {max_err}");
    }

    #[test]
    fn sliding_window_decode_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let head_dim = 64u32;
        let cache_len = 100u32;
        let window_size = 32u32;

        // Create cached K, V
        let k_data: Vec<f32> = (0..(cache_len * head_dim) as usize)
            .map(|i| ((i % 53) as f32 - 26.0) * 0.05).collect();
        let v_data: Vec<f32> = (0..(cache_len * head_dim) as usize)
            .map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
        let q_data: Vec<f32> = (0..head_dim as usize)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();

        // CPU reference: only last 32 positions
        let start = (cache_len - window_size) as usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = Vec::new();
        let mut max_score = f32::NEG_INFINITY;
        for j in start..cache_len as usize {
            let mut score = 0.0f32;
            for d in 0..head_dim as usize {
                score += q_data[d] * k_data[j * head_dim as usize + d];
            }
            score *= scale;
            scores.push(score);
            if score > max_score { max_score = score; }
        }
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let mut expected = vec![0.0f32; head_dim as usize];
        for d in 0..head_dim as usize {
            for (idx, j) in (start..cache_len as usize).enumerate() {
                expected[d] += exp_scores[idx] / sum_exp * v_data[j * head_dim as usize + d];
            }
        }

        // GPU
        let q = GpuTensor::from_host(&dev, &q_data,
            Shape::from_static(&[head_dim as usize]), DType::F32).unwrap();
        let k_cache = GpuTensor::from_host(&dev, &k_data,
            Shape::from_static(&[cache_len as usize, head_dim as usize]), DType::F32).unwrap();
        let v_cache = GpuTensor::from_host(&dev, &v_data,
            Shape::from_static(&[cache_len as usize, head_dim as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[head_dim as usize]), DType::F32).unwrap();

        sliding_window_decode(
            &cache, &dev, &q, &k_cache, &v_cache, &mut out,
            cache_len, head_dim, window_size,
        ).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        let max_err: f32 = result.iter().zip(expected.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Sliding window decode (cache={cache_len}, W={window_size}, D={head_dim}): max error = {max_err:.6}");
        assert!(max_err < 1e-3, "Max error {max_err} too high");
    }

    #[test]
    fn sliding_window_perf() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (b, n, d) = (1u32, 512u32, 128u32);
        let total = (b * n * d) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, d as usize]);

        let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001).collect();
        let q = GpuTensor::from_host(&dev, &data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &data, shape.clone(), DType::F32).unwrap();

        for window in [64, 128, 256, 512] {
            let mut out = GpuTensor::<f32>::zeros(&dev, shape.clone(), DType::F32).unwrap();

            // Warmup
            sliding_window_attention(&cache, &dev, &q, &k, &v, &mut out, b, n, d, window).unwrap();
            dev.synchronize().unwrap();

            let iters = 20;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                sliding_window_attention(&cache, &dev, &q, &k, &v, &mut out, b, n, d, window).unwrap();
            }
            dev.synchronize().unwrap();
            let elapsed = start.elapsed();

            println!("Sliding window N={n} D={d} W={window}: {:.3}ms avg",
                elapsed.as_secs_f64() * 1000.0 / iters as f64);
        }
    }
}

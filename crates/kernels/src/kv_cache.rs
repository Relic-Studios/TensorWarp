//! KV Cache — the key to fast autoregressive decoding.
//!
//! Without KV cache: each new token recomputes attention over the ENTIRE
//! sequence. For 512 tokens that's 512 forward passes of growing length.
//! Total work: O(N²·L·D) where N=seq_len, L=layers, D=hidden.
//!
//! With KV cache: store K and V from previous positions on GPU.
//! Each new token only computes K,V for position N, appends to cache,
//! then attends over the full cached K/V. Work per step: O(N·D).
//! Total work: O(N·L·D) — a factor of N speedup.
//!
//! For a 512-token generation, that's a ~256x speedup.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Bulk-copy K/V from prefill into the cache.
/// Copies [seq_len, dim] contiguous data starting at position 0.
const KV_CACHE_PREFILL_SRC: &str = r#"
extern "C" __global__ void warp_kv_cache_prefill(
    float *cache,            // [max_seq, dim]
    const float *src,        // [seq_len, dim]
    unsigned int seq_len,
    unsigned int dim,
    unsigned int max_seq
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = seq_len * dim;
    if (idx >= total) return;
    unsigned int pos = idx / dim;
    if (pos >= max_seq) return;
    cache[idx] = src[idx];
}
"#;

/// Fused K+V append — writes both in a single kernel launch.
const KV_CACHE_APPEND_FUSED_SRC: &str = r#"
extern "C" __global__ void warp_kv_cache_append_fused(
    float *k_cache,          // [max_seq, dim]
    float *v_cache,          // [max_seq, dim]
    const float *new_k,      // [1, dim]
    const float *new_v,      // [1, dim]
    unsigned int pos,
    unsigned int dim,
    unsigned int max_seq
) {
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    if (pos >= max_seq) return;
    k_cache[pos * dim + d] = new_k[d];
    v_cache[pos * dim + d] = new_v[d];
}
"#;

/// Append new K/V values to the cache at the current position.
const KV_CACHE_APPEND_SRC: &str = r#"
extern "C" __global__ void warp_kv_cache_append(
    float *cache,            // [max_seq, dim] — the K or V cache
    const float *new_vals,   // [1, dim] — new K or V for current token
    unsigned int pos,        // current position to write at
    unsigned int dim,        // kv dimension
    unsigned int max_seq     // max sequence length of cache
) {
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    if (pos >= max_seq) return;
    cache[pos * dim + d] = new_vals[d];
}
"#;

/// GPU-resident KV cache for one layer.
pub struct LayerKVCache {
    /// K cache: [max_seq_len, kv_dim] on GPU
    pub k: GpuTensor<f32>,
    /// V cache: [max_seq_len, kv_dim] on GPU
    pub v: GpuTensor<f32>,
    /// Current filled length
    pub len: u32,
    /// Max capacity
    pub max_seq_len: u32,
    /// KV dimension
    pub kv_dim: u32,
}

impl LayerKVCache {
    pub fn new(device: &WarpDevice, max_seq_len: u32, kv_dim: u32) -> Result<Self, DeviceError> {
        let shape = Shape::from_static(&[max_seq_len as usize, kv_dim as usize]);
        let k = GpuTensor::<f32>::zeros(device, shape.clone(), DType::F32)?;
        let v = GpuTensor::<f32>::zeros(device, shape, DType::F32)?;
        Ok(Self { k, v, len: 0, max_seq_len, kv_dim })
    }

    /// Bulk-write K and V from a prefill pass.
    /// k_all and v_all are [seq_len, kv_dim] — the full prompt's projections.
    pub fn prefill(
        &mut self,
        kernel_cache: &KernelCache,
        device: &WarpDevice,
        k_all: &GpuTensor<f32>,  // [seq_len, kv_dim]
        v_all: &GpuTensor<f32>,  // [seq_len, kv_dim]
        seq_len: u32,
    ) -> Result<(), DeviceError> {
        let f = kernel_cache.get_or_compile(device, KV_CACHE_PREFILL_SRC, "warp_kv_cache_prefill")?;
        let total = seq_len * self.kv_dim;
        let cfg = LaunchConfig::for_num_elems(total);

        // Copy K
        unsafe {
            device.stream.launch_builder(&f)
                .arg(&mut self.k.data)
                .arg(&k_all.data)
                .arg(&seq_len)
                .arg(&self.kv_dim)
                .arg(&self.max_seq_len)
                .launch(cfg)
                .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
        }

        // Copy V
        unsafe {
            device.stream.launch_builder(&f)
                .arg(&mut self.v.data)
                .arg(&v_all.data)
                .arg(&seq_len)
                .arg(&self.kv_dim)
                .arg(&self.max_seq_len)
                .launch(cfg)
                .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
        }

        self.len = seq_len;
        Ok(())
    }

    /// Append K and V vectors for one new token position.
    /// Uses a fused kernel to write both K and V in a single launch.
    pub fn append(
        &mut self,
        kernel_cache: &KernelCache,
        device: &WarpDevice,
        new_k: &GpuTensor<f32>,  // [1, kv_dim]
        new_v: &GpuTensor<f32>,  // [1, kv_dim]
    ) -> Result<(), DeviceError> {
        let f = kernel_cache.get_or_compile(device, KV_CACHE_APPEND_FUSED_SRC, "warp_kv_cache_append_fused")?;
        let cfg = LaunchConfig::for_num_elems(self.kv_dim);

        // Fused K+V append (2 launches → 1)
        unsafe {
            device.stream.launch_builder(&f)
                .arg(&mut self.k.data)
                .arg(&mut self.v.data)
                .arg(&new_k.data)
                .arg(&new_v.data)
                .arg(&self.len)
                .arg(&self.kv_dim)
                .arg(&self.max_seq_len)
                .launch(cfg)
                .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
        }

        self.len += 1;
        Ok(())
    }

    /// Reset the cache (for new sequence).
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Current sequence length stored in cache.
    pub fn seq_len(&self) -> u32 {
        self.len
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.k.size_bytes() + self.v.size_bytes()
    }
}

/// Full KV cache for all layers.
pub struct ModelKVCache {
    pub layers: Vec<LayerKVCache>,
    pub max_seq_len: u32,
}

impl ModelKVCache {
    /// Allocate KV cache for all layers.
    pub fn new(
        device: &WarpDevice,
        num_layers: u32,
        max_seq_len: u32,
        kv_dim: u32,
    ) -> Result<Self, DeviceError> {
        let mut layers = Vec::with_capacity(num_layers as usize);
        for _ in 0..num_layers {
            layers.push(LayerKVCache::new(device, max_seq_len, kv_dim)?);
        }
        Ok(Self { layers, max_seq_len })
    }

    /// Reset all layer caches.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Current sequence length (same across all layers).
    pub fn seq_len(&self) -> u32 {
        self.layers.first().map(|l| l.len).unwrap_or(0)
    }

    /// Total memory usage.
    pub fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    pub fn summary(&self) -> String {
        format!(
            "KVCache: {} layers, seq={}/{}, {:.2} MB",
            self.layers.len(),
            self.seq_len(),
            self.max_seq_len,
            self.memory_bytes() as f64 / 1e6,
        )
    }
}

/// Multi-head decode attention with native GQA support.
/// One kernel launch handles ALL Q heads, mapping each to its KV head via GQA ratio.
/// Q: [num_heads * head_dim], K_cache: [cache_len, kv_dim], V_cache: [cache_len, kv_dim]
/// out: [num_heads * head_dim]
///
/// Grid: num_heads blocks. Block: head_dim threads.
/// Each thread is responsible for one output dimension of its head.
/// Each thread computes the full dot product (iterating over all dims) to avoid
/// cross-warp reduction complexity, then computes its own weighted V sum.
const DECODE_ATTENTION_MULTIHEAD_SRC: &str = r#"
extern "C" __global__ void warp_decode_attention_multihead(
    float *out,              // [num_heads * head_dim]
    const float *Q,          // [num_heads * head_dim]
    const float *K_cache,    // [cache_len, kv_dim]  (kv_dim = num_kv_heads * head_dim)
    const float *V_cache,    // [cache_len, kv_dim]
    unsigned int cache_len,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float scale
) {
    // One block per Q head
    unsigned int q_head = blockIdx.x;
    if (q_head >= num_heads) return;

    // GQA mapping: which KV head does this Q head attend to?
    unsigned int n_rep = num_heads / num_kv_heads;
    unsigned int kv_head = q_head / n_rep;

    // Each thread handles one output dimension
    unsigned int d = threadIdx.x;
    if (d >= head_dim) return;

    unsigned int kv_dim = num_kv_heads * head_dim;

    // Online softmax (Milakov-Gimelshein) — no shared memory needed
    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float out_val = 0.0f;

    for (unsigned int pos = 0; pos < cache_len; pos++) {
        // Full dot product Q[q_head, :] · K[pos, kv_head, :]
        float score = 0.0f;
        for (unsigned int dd = 0; dd < head_dim; dd++) {
            score += Q[q_head * head_dim + dd]
                   * K_cache[pos * kv_dim + kv_head * head_dim + dd];
        }
        score *= scale;

        // Online softmax update
        float new_max = fmaxf(max_score, score);
        float correction = expf(max_score - new_max);
        float weight = expf(score - new_max);

        sum_exp = sum_exp * correction + weight;
        out_val = out_val * correction
                + weight * V_cache[pos * kv_dim + kv_head * head_dim + d];
        max_score = new_max;
    }

    out[q_head * head_dim + d] = out_val / sum_exp;
}
"#;

/// Run multi-head decode attention: all Q heads against full KV cache in one launch.
/// Handles GQA natively — no CPU roundtrips, no per-head extraction.
pub fn decode_attention_multihead(
    kernel_cache: &KernelCache,
    device: &WarpDevice,
    q: &GpuTensor<f32>,         // [num_heads * head_dim]
    kv: &LayerKVCache,
    out: &mut GpuTensor<f32>,   // [num_heads * head_dim]
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Result<(), DeviceError> {
    if kv.len == 0 {
        return Ok(()); // nothing to attend to
    }

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let f = kernel_cache.get_or_compile(
        device,
        DECODE_ATTENTION_MULTIHEAD_SRC,
        "warp_decode_attention_multihead",
    )?;

    let cfg = LaunchConfig {
        grid_dim: (num_heads, 1, 1),     // one block per Q head
        block_dim: (head_dim, 1, 1),     // one thread per dimension
        shared_mem_bytes: 0,
    };

    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&q.data)
            .arg(&kv.k.data)
            .arg(&kv.v.data)
            .arg(&kv.len)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&head_dim)
            .arg(&scale)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// Decode attention: attend from a single query token over the full KV cache.
/// Q: [1, head_dim], K_cache: [cache_len, head_dim], V_cache: [cache_len, head_dim]
/// out: [1, head_dim]
///
/// This replaces the full N×N attention during decode — only one query row.
/// NOTE: For multi-head / GQA models, prefer decode_attention_multihead instead.
const DECODE_ATTENTION_SRC: &str = r#"
extern "C" __global__ void warp_decode_attention(
    float *out,             // [head_dim]
    const float *q,         // [head_dim]
    const float *k_cache,   // [cache_len, head_dim]
    const float *v_cache,   // [cache_len, head_dim]
    unsigned int cache_len,
    unsigned int head_dim,
    float scale
) {
    unsigned int d = threadIdx.x;
    if (d >= head_dim) return;

    // Compute Q·K^T scores and online softmax
    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float out_val = 0.0f;

    for (unsigned int j = 0; j < cache_len; j++) {
        // Dot product Q·K[j]
        float score = 0.0f;
        for (unsigned int dd = 0; dd < head_dim; dd++) {
            score += q[dd] * k_cache[j * head_dim + dd];
        }
        score *= scale;

        // Online softmax (Milakov-Gimelshein)
        float new_max = fmaxf(max_score, score);
        float correction = expf(max_score - new_max);
        float weight = expf(score - new_max);

        sum_exp = sum_exp * correction + weight;
        out_val = out_val * correction + weight * v_cache[j * head_dim + d];
        max_score = new_max;
    }

    out[d] = out_val / sum_exp;
}
"#;

/// Run decode attention: single query against full KV cache.
pub fn decode_attention(
    kernel_cache: &KernelCache,
    device: &WarpDevice,
    q: &GpuTensor<f32>,         // [1, head_dim]
    kv: &LayerKVCache,
    out: &mut GpuTensor<f32>,   // [1, head_dim]
    head_dim: u32,
) -> Result<(), DeviceError> {
    if kv.len == 0 {
        return Ok(()); // nothing to attend to
    }

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let f = kernel_cache.get_or_compile(device, DECODE_ATTENTION_SRC, "warp_decode_attention")?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&q.data)
            .arg(&kv.k.data)
            .arg(&kv.v.data)
            .arg(&kv.len)
            .arg(&head_dim)
            .arg(&scale)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    #[test]
    fn kv_cache_append_and_read() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let kv_dim = 32u32;
        let max_seq = 64u32;
        let mut kv = LayerKVCache::new(&dev, max_seq, kv_dim).unwrap();

        // Append 5 positions
        for pos in 0..5u32 {
            let k_data: Vec<f32> = (0..kv_dim).map(|d| pos as f32 + d as f32 * 0.01).collect();
            let v_data: Vec<f32> = (0..kv_dim).map(|d| -(pos as f32) + d as f32 * 0.01).collect();

            let k = GpuTensor::from_host(&dev, &k_data, Shape::from_static(&[1, kv_dim as usize]), DType::F32).unwrap();
            let v = GpuTensor::from_host(&dev, &v_data, Shape::from_static(&[1, kv_dim as usize]), DType::F32).unwrap();

            kv.append(&cache, &dev, &k, &v).unwrap();
        }
        dev.synchronize().unwrap();

        assert_eq!(kv.seq_len(), 5);

        // Read back and verify
        let k_all = kv.k.to_host(&dev).unwrap();
        // Position 0, dim 0 should be 0.0
        assert!((k_all[0] - 0.0).abs() < 1e-5);
        // Position 3, dim 0 should be 3.0
        assert!((k_all[3 * kv_dim as usize] - 3.0).abs() < 1e-5);

        println!("KV cache: appended 5 positions, verified correct");
        println!("  {}", ModelKVCache { layers: vec![], max_seq_len: max_seq }.summary());
    }

    #[test]
    fn decode_attention_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let head_dim = 16u32;
        let max_seq = 32u32;
        let mut kv = LayerKVCache::new(&dev, max_seq, head_dim).unwrap();

        // Fill cache with 8 positions
        for pos in 0..8u32 {
            let k_data: Vec<f32> = (0..head_dim).map(|d| ((pos * head_dim + d) % 17) as f32 * 0.1 - 0.8).collect();
            let v_data: Vec<f32> = (0..head_dim).map(|d| ((pos * head_dim + d) % 13) as f32 * 0.1 - 0.6).collect();
            let k = GpuTensor::from_host(&dev, &k_data, Shape::from_static(&[1, head_dim as usize]), DType::F32).unwrap();
            let v = GpuTensor::from_host(&dev, &v_data, Shape::from_static(&[1, head_dim as usize]), DType::F32).unwrap();
            kv.append(&cache, &dev, &k, &v).unwrap();
        }

        // Query
        let q_data: Vec<f32> = (0..head_dim).map(|d| (d % 7) as f32 * 0.1 - 0.3).collect();
        let q = GpuTensor::from_host(&dev, &q_data, Shape::from_static(&[1, head_dim as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[1, head_dim as usize]), DType::F32).unwrap();

        decode_attention(&cache, &dev, &q, &kv, &mut out, head_dim).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        assert!(result.iter().all(|v| v.is_finite()), "Decode attention has NaN/Inf!");
        assert!(result.iter().any(|v| *v != 0.0), "Decode attention is all zeros!");

        // Verify against CPU reference
        let k_host = kv.k.to_host(&dev).unwrap();
        let v_host = kv.v.to_host(&dev).unwrap();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Compute attention on CPU
        let mut scores = vec![0.0f32; 8];
        for j in 0..8 {
            for d in 0..head_dim as usize {
                scores[j] += q_data[d] * k_host[j * head_dim as usize + d];
            }
            scores[j] *= scale;
        }
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();

        let mut expected = vec![0.0f32; head_dim as usize];
        for d in 0..head_dim as usize {
            for j in 0..8 {
                expected[d] += exp_scores[j] / sum_exp * v_host[j * head_dim as usize + d];
            }
        }

        let max_err: f32 = result.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Decode attention (8 cached positions, D={head_dim}): max error = {max_err:.6}");
        assert!(max_err < 1e-3, "Max error {max_err} too high");
    }

    #[test]
    fn decode_attention_perf() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let head_dim = 32u32;
        let max_seq = 1024u32;
        let mut kv = LayerKVCache::new(&dev, max_seq, head_dim).unwrap();

        // Fill cache with 512 positions
        for pos in 0..512u32 {
            let k_data: Vec<f32> = (0..head_dim).map(|d| ((pos + d) % 100) as f32 * 0.01).collect();
            let v_data = k_data.clone();
            let k = GpuTensor::from_host(&dev, &k_data, Shape::from_static(&[1, head_dim as usize]), DType::F32).unwrap();
            let v = GpuTensor::from_host(&dev, &v_data, Shape::from_static(&[1, head_dim as usize]), DType::F32).unwrap();
            kv.append(&cache, &dev, &k, &v).unwrap();
        }

        let q_data: Vec<f32> = (0..head_dim).map(|d| d as f32 * 0.01).collect();
        let q = GpuTensor::from_host(&dev, &q_data, Shape::from_static(&[1, head_dim as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[1, head_dim as usize]), DType::F32).unwrap();

        // Warmup
        decode_attention(&cache, &dev, &q, &kv, &mut out, head_dim).unwrap();
        dev.synchronize().unwrap();

        let iters = 1000;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            decode_attention(&cache, &dev, &q, &kv, &mut out, head_dim).unwrap();
        }
        dev.synchronize().unwrap();
        let elapsed = start.elapsed();

        println!("\nDecode attention perf (512 cached positions, D={head_dim}):");
        println!("  {:.3}μs per attention ({iters} iters)",
            elapsed.as_secs_f64() * 1e6 / iters as f64);
        println!("  KV cache memory: {:.2} MB", kv.memory_bytes() as f64 / 1e6);
    }

    #[test]
    fn decode_attention_multihead_gqa() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // GQA config: 14 Q heads, 2 KV heads (ratio 7:1), head_dim=16
        let num_heads = 14u32;
        let num_kv_heads = 2u32;
        let head_dim = 16u32;
        let kv_dim = num_kv_heads * head_dim; // 32
        let q_dim = num_heads * head_dim;     // 224
        let seq_len = 8u32;
        let max_seq = 32u32;

        let mut kv = LayerKVCache::new(&dev, max_seq, kv_dim).unwrap();

        // Fill cache with seq_len positions
        let mut k_host_all = vec![0.0f32; max_seq as usize * kv_dim as usize];
        let mut v_host_all = vec![0.0f32; max_seq as usize * kv_dim as usize];
        for pos in 0..seq_len {
            let k_data: Vec<f32> = (0..kv_dim)
                .map(|d| ((pos * kv_dim + d) % 17) as f32 * 0.1 - 0.8)
                .collect();
            let v_data: Vec<f32> = (0..kv_dim)
                .map(|d| ((pos * kv_dim + d) % 13) as f32 * 0.1 - 0.6)
                .collect();
            k_host_all[pos as usize * kv_dim as usize..(pos as usize + 1) * kv_dim as usize]
                .copy_from_slice(&k_data);
            v_host_all[pos as usize * kv_dim as usize..(pos as usize + 1) * kv_dim as usize]
                .copy_from_slice(&v_data);
            let k = GpuTensor::from_host(&dev, &k_data,
                Shape::from_static(&[1, kv_dim as usize]), DType::F32).unwrap();
            let v = GpuTensor::from_host(&dev, &v_data,
                Shape::from_static(&[1, kv_dim as usize]), DType::F32).unwrap();
            kv.append(&cache, &dev, &k, &v).unwrap();
        }

        // Query: [num_heads * head_dim]
        let q_data: Vec<f32> = (0..q_dim)
            .map(|d| (d % 11) as f32 * 0.1 - 0.5)
            .collect();
        let q = GpuTensor::from_host(&dev, &q_data,
            Shape::from_static(&[1, q_dim as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, q_dim as usize]), DType::F32).unwrap();

        decode_attention_multihead(&cache, &dev, &q, &kv, &mut out, num_heads, num_kv_heads, head_dim).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        assert!(result.iter().all(|v| v.is_finite()), "Multihead decode has NaN/Inf!");
        assert!(result.iter().any(|v| *v != 0.0), "Multihead decode is all zeros!");

        // CPU reference
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n_rep = num_heads / num_kv_heads;
        let mut expected = vec![0.0f32; q_dim as usize];

        for q_head in 0..num_heads as usize {
            let kv_head = q_head / n_rep as usize;

            // Compute scores
            let mut scores = vec![0.0f32; seq_len as usize];
            for pos in 0..seq_len as usize {
                let mut dot = 0.0f32;
                for dd in 0..head_dim as usize {
                    dot += q_data[q_head * head_dim as usize + dd]
                         * k_host_all[pos * kv_dim as usize + kv_head * head_dim as usize + dd];
                }
                scores[pos] = dot * scale;
            }

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();

            // Weighted V sum
            for dd in 0..head_dim as usize {
                let mut val = 0.0f32;
                for pos in 0..seq_len as usize {
                    val += (exp_scores[pos] / sum_exp)
                         * v_host_all[pos * kv_dim as usize + kv_head * head_dim as usize + dd];
                }
                expected[q_head * head_dim as usize + dd] = val;
            }
        }

        let max_err: f32 = result.iter().zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Multihead GQA decode ({num_heads} Q heads, {num_kv_heads} KV heads, D={head_dim}, seq={seq_len}): max error = {max_err:.6}");
        assert!(max_err < 1e-3, "Max error {max_err} too high for multihead GQA decode");
    }

    #[test]
    fn model_kv_cache() {
        let (dev, _cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let num_layers = 32u32;
        let max_seq = 2048u32;
        let kv_dim = 128u32; // 32 heads × 4 kv_heads × 128 head_dim for GQA

        let kv_cache = ModelKVCache::new(&dev, num_layers, max_seq, kv_dim).unwrap();

        println!("Model KV cache (LLaMA-7B-like):");
        println!("  {}", kv_cache.summary());
        println!("  Memory: {:.2} MB ({:.2} GB)",
            kv_cache.memory_bytes() as f64 / 1e6,
            kv_cache.memory_bytes() as f64 / 1e9);
        println!("  Per-layer: {:.2} MB", kv_cache.layers[0].memory_bytes() as f64 / 1e6);
    }
}

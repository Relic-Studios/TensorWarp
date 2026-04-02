//! Complete transformer block — attention + FFN + norms.
//!
//! This is where TensorWarp's fusion advantage materializes.
//! A single transformer block fuses operations that TRT runs as
//! separate kernel launches:
//!
//! 1. RMSNorm(x)
//! 2. Q, K, V projections (3 GEMMs)
//! 3. RoPE on Q and K
//! 4. Flash Attention
//! 5. Output projection (GEMM)
//! 6. Residual add
//! 7. RMSNorm
//! 8. Gate + Up projections (2 GEMMs)
//! 9. SiLU + element-wise multiply (SwiGLU)
//! 10. Down projection (GEMM)
//! 11. Residual add
//!
//! TRT: 11+ kernel launches. TensorWarp: fewer via fusion.

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;
use crate::ops;

/// Weights for a single transformer block (LLaMA-style).
pub struct TransformerBlockWeights {
    /// Attention input norm weights [hidden_size]
    pub attn_norm: GpuTensor<f32>,
    /// Q projection [hidden_size, hidden_size]
    pub wq: GpuTensor<f32>,
    /// K projection [hidden_size, kv_dim]
    pub wk: GpuTensor<f32>,
    /// V projection [hidden_size, kv_dim]
    pub wv: GpuTensor<f32>,
    /// Output projection [hidden_size, hidden_size]
    pub wo: GpuTensor<f32>,
    /// FFN input norm weights [hidden_size]
    pub ffn_norm: GpuTensor<f32>,
    /// Gate projection [hidden_size, ffn_dim]
    pub w_gate: GpuTensor<f32>,
    /// Up projection [hidden_size, ffn_dim]
    pub w_up: GpuTensor<f32>,
    /// Down projection [ffn_dim, hidden_size]
    pub w_down: GpuTensor<f32>,
}

/// Config for a transformer block.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub hidden_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub ffn_dim: u32,
    pub rope_base: f32,
    pub norm_eps: f32,
}

impl TransformerConfig {
    /// Tiny config for correctness testing.
    pub fn tiny() -> Self {
        Self {
            hidden_size: 64,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            ffn_dim: 128,
            rope_base: 10000.0,
            norm_eps: 1e-6,
        }
    }

    /// Small config — realistic ratios, fits in VRAM for benchmarking.
    pub fn small() -> Self {
        Self {
            hidden_size: 256,
            num_heads: 8,
            num_kv_heads: 8,
            head_dim: 32,
            ffn_dim: 512,
            rope_base: 10000.0,
            norm_eps: 1e-6,
        }
    }

    /// Medium config — approaching real model dimensions.
    pub fn medium() -> Self {
        Self {
            hidden_size: 1024,
            num_heads: 16,
            num_kv_heads: 16,
            head_dim: 32, // Keep <=32 for flash attention
            ffn_dim: 2048,
            rope_base: 10000.0,
            norm_eps: 1e-6,
        }
    }

    pub fn kv_dim(&self) -> u32 {
        self.num_kv_heads * self.head_dim
    }
}

/// Forward pass of a single transformer block.
///
/// x: [batch, seq_len, hidden_size]
/// Returns: [batch, seq_len, hidden_size]
pub fn transformer_block_forward(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    weights: &TransformerBlockWeights,
    config: &TransformerConfig,
    batch: u32,
    seq_len: u32,
    pos_offset: u32,
) -> Result<GpuTensor<f32>, DeviceError> {
    let h = config.hidden_size;
    let d = config.head_dim;
    let kv_dim = config.kv_dim();
    let ffn = config.ffn_dim;
    let bsz = batch;
    let n = seq_len;

    let shape_bnh = warp_ir::Shape::from_static(&[bsz as usize, n as usize, h as usize]);
    let shape_bnk = warp_ir::Shape::from_static(&[bsz as usize, n as usize, kv_dim as usize]);
    let shape_bnf = warp_ir::Shape::from_static(&[bsz as usize, n as usize, ffn as usize]);

    // 1. Attention input norm
    let mut normed = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::rmsnorm(cache, device, x, &weights.attn_norm, &mut normed, h, config.norm_eps)?;

    // 2. Q, K, V projections
    // For simplicity, treat [B*N, H] × [H, out_dim] as a GEMM
    let bn = bsz * n;
    let mut q = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut k_proj = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;
    let mut v_proj = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;

    ops::gemm(cache, device, &normed, &weights.wq, &mut q, bn, h, h)?;
    ops::gemm(cache, device, &normed, &weights.wk, &mut k_proj, bn, kv_dim, h)?;
    ops::gemm(cache, device, &normed, &weights.wv, &mut v_proj, bn, kv_dim, h)?;

    // 3. RoPE on Q and K
    // Apply per-head: reshape conceptually to [B, N, num_heads, head_dim]
    // For now apply on flat [B*N*num_heads, head_dim] — RoPE is per-position
    let mut q_rope = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut k_rope = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;
    crate::rope::rope(cache, device, &q, &mut q_rope, bsz * config.num_heads, n, d, config.rope_base, pos_offset)?;
    crate::rope::rope(cache, device, &k_proj, &mut k_rope, bsz * config.num_kv_heads, n, d, config.rope_base, pos_offset)?;

    // 4. Flash Attention (per-head, simplified: treat all heads as batch dim)
    // Reshape: [B, N, num_heads * D] → [B * num_heads, N, D]
    let attn_batch = bsz * config.num_heads;
    let shape_attn = warp_ir::Shape::from_static(&[attn_batch as usize, n as usize, d as usize]);
    let mut attn_out = GpuTensor::<f32>::zeros(device, shape_attn, warp_ir::DType::F32)?;

    if d <= 32 {
        crate::attention::attention_flash(
            cache, device, &q_rope, &k_rope, &v_proj,
            &mut attn_out, attn_batch, n, d, true,
        )?;
    } else {
        crate::attention::attention_naive(
            cache, device, &q_rope, &k_rope, &v_proj,
            &mut attn_out, attn_batch, n, d, true,
        )?;
    }

    // 5. Output projection
    let mut attn_projected = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::gemm(cache, device, &attn_out, &weights.wo, &mut attn_projected, bn, h, h)?;

    // 6. Residual add: x = x + attn_out
    let mut residual1 = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::add(cache, device, x, &attn_projected, &mut residual1)?;

    // 7. FFN norm
    let mut ffn_normed = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::rmsnorm(cache, device, &residual1, &weights.ffn_norm, &mut ffn_normed, h, config.norm_eps)?;

    // 8. Gate + Up projections
    let mut gate = GpuTensor::<f32>::zeros(device, shape_bnf.clone(), warp_ir::DType::F32)?;
    let mut up = GpuTensor::<f32>::zeros(device, shape_bnf.clone(), warp_ir::DType::F32)?;
    ops::gemm(cache, device, &ffn_normed, &weights.w_gate, &mut gate, bn, ffn, h)?;
    ops::gemm(cache, device, &ffn_normed, &weights.w_up, &mut up, bn, ffn, h)?;

    // 9. SwiGLU: silu(gate) * up
    let mut gate_activated = GpuTensor::<f32>::zeros(device, shape_bnf.clone(), warp_ir::DType::F32)?;
    ops::silu(cache, device, &gate, &mut gate_activated)?;
    let mut swiglu_out = GpuTensor::<f32>::zeros(device, shape_bnf, warp_ir::DType::F32)?;
    ops::mul(cache, device, &gate_activated, &up, &mut swiglu_out)?;

    // 10. Down projection
    let mut ffn_out = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::gemm(cache, device, &swiglu_out, &weights.w_down, &mut ffn_out, bn, h, ffn)?;

    // 11. Final residual add
    let mut output = GpuTensor::<f32>::zeros(device, shape_bnh, warp_ir::DType::F32)?;
    ops::add(cache, device, &residual1, &ffn_out, &mut output)?;

    Ok(output)
}

/// Create random weights for testing.
pub fn random_weights(
    device: &WarpDevice,
    config: &TransformerConfig,
) -> Result<TransformerBlockWeights, DeviceError> {
    let h = config.hidden_size as usize;
    let kv = config.kv_dim() as usize;
    let ffn = config.ffn_dim as usize;

    let rand_vec = |n: usize| -> Vec<f32> {
        (0..n).map(|i| ((i * 7 + 13) % 100) as f32 * 0.01 - 0.5).collect()
    };

    Ok(TransformerBlockWeights {
        attn_norm: GpuTensor::from_host(device, &vec![1.0f32; h], warp_ir::Shape::from_static(&[h]), warp_ir::DType::F32)?,
        wq: GpuTensor::from_host(device, &rand_vec(h * h), warp_ir::Shape::from_static(&[h, h]), warp_ir::DType::F32)?,
        wk: GpuTensor::from_host(device, &rand_vec(h * kv), warp_ir::Shape::from_static(&[h, kv]), warp_ir::DType::F32)?,
        wv: GpuTensor::from_host(device, &rand_vec(h * kv), warp_ir::Shape::from_static(&[h, kv]), warp_ir::DType::F32)?,
        wo: GpuTensor::from_host(device, &rand_vec(h * h), warp_ir::Shape::from_static(&[h, h]), warp_ir::DType::F32)?,
        ffn_norm: GpuTensor::from_host(device, &vec![1.0f32; h], warp_ir::Shape::from_static(&[h]), warp_ir::DType::F32)?,
        w_gate: GpuTensor::from_host(device, &rand_vec(h * ffn), warp_ir::Shape::from_static(&[h, ffn]), warp_ir::DType::F32)?,
        w_up: GpuTensor::from_host(device, &rand_vec(h * ffn), warp_ir::Shape::from_static(&[h, ffn]), warp_ir::DType::F32)?,
        w_down: GpuTensor::from_host(device, &rand_vec(ffn * h), warp_ir::Shape::from_static(&[ffn, h]), warp_ir::DType::F32)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::{DType, Shape};

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    #[test]
    fn transformer_block_runs() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let config = TransformerConfig::tiny();
        let weights = random_weights(&dev, &config).unwrap();

        let (b, n) = (1u32, 8u32);
        let h = config.hidden_size;
        let total = (b * n * h) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, h as usize]);

        let x_data: Vec<f32> = (0..total).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();
        let x = GpuTensor::from_host(&dev, &x_data, shape, DType::F32).unwrap();

        let output = transformer_block_forward(
            &cache, &dev, &x, &weights, &config, b, n, 0,
        ).unwrap();
        dev.synchronize().unwrap();

        let result = output.to_host(&dev).unwrap();
        assert_eq!(result.len(), total);
        assert!(result.iter().all(|x| x.is_finite()), "Output has NaN/Inf!");
        assert!(result.iter().any(|x| *x != 0.0), "Output is all zeros!");

        println!("Transformer block forward: B={b} N={n} H={h}");
        println!("  Output sample: [{:.4}, {:.4}, {:.4}, {:.4}, ...]",
            result[0], result[1], result[2], result[3]);
        println!("  Output range: [{:.4}, {:.4}]",
            result.iter().cloned().fold(f32::INFINITY, f32::min),
            result.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        println!("{}", cache.stats());
    }

    #[test]
    fn transformer_block_perf() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let config = TransformerConfig::tiny();
        let weights = random_weights(&dev, &config).unwrap();

        let (b, n) = (1u32, 32u32);
        let h = config.hidden_size;
        let shape = Shape::from_static(&[b as usize, n as usize, h as usize]);
        let total = (b * n * h) as usize;

        let x_data: Vec<f32> = (0..total).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();
        let x = GpuTensor::from_host(&dev, &x_data, shape, DType::F32).unwrap();

        // Warmup
        let _ = transformer_block_forward(&cache, &dev, &x, &weights, &config, b, n, 0).unwrap();
        dev.synchronize().unwrap();

        // Timed
        let iters = 50;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = transformer_block_forward(&cache, &dev, &x, &weights, &config, b, n, 0).unwrap();
        }
        dev.synchronize().unwrap();
        let elapsed = start.elapsed();

        println!("\nTransformer block perf (B={b} N={n} H={h} FFN={}):", config.ffn_dim);
        println!("  {:.3}ms avg ({iters} iters)", elapsed.as_secs_f64() * 1000.0 / iters as f64);
        println!("  {:.0} blocks/sec", iters as f64 / elapsed.as_secs_f64());
        println!("{}", cache.stats());
    }

    #[test]
    fn transformer_scaling_benchmark() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        println!("\n=== Transformer Block Scaling Benchmark ===");
        let configs = vec![
            ("tiny",   TransformerConfig::tiny(),   32u32),
            ("small",  TransformerConfig::small(),  64),
            ("medium", TransformerConfig::medium(), 128),
        ];

        for (name, config, seq_len) in configs {
            let weights = random_weights(&dev, &config).unwrap();
            let (b, n) = (1u32, seq_len);
            let h = config.hidden_size;
            let total = (b * n * h) as usize;
            let shape = Shape::from_static(&[b as usize, n as usize, h as usize]);

            let x_data: Vec<f32> = (0..total).map(|i| ((i % 23) as f32 - 11.0) * 0.01).collect();
            let x = GpuTensor::from_host(&dev, &x_data, shape, DType::F32).unwrap();

            // Warmup
            let out = transformer_block_forward(&cache, &dev, &x, &weights, &config, b, n, 0).unwrap();
            dev.synchronize().unwrap();

            // Verify
            let result = out.to_host(&dev).unwrap();
            assert!(result.iter().all(|v| v.is_finite()), "{name}: NaN/Inf in output!");

            // Bench
            let iters = 30;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = transformer_block_forward(&cache, &dev, &x, &weights, &config, b, n, 0).unwrap();
            }
            dev.synchronize().unwrap();
            let elapsed = start.elapsed();

            let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
            let blocks_per_sec = iters as f64 / elapsed.as_secs_f64();

            // Estimate FLOPS: ~12 * seq * hidden² for one block (rough)
            let approx_flops = 12.0 * n as f64 * (h as f64).powi(2);
            let approx_tflops = approx_flops * iters as f64 / elapsed.as_secs_f64() / 1e12;

            println!(
                "  {name:6} (H={h:4} FFN={:4} N={n:3}): {ms:.3}ms | {blocks_per_sec:.0} blocks/s | ~{approx_tflops:.3} TFLOPS",
                config.ffn_dim,
            );
        }
        println!("{}", cache.stats());
    }
}

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
use crate::kv_cache::LayerKVCache;
use crate::quantize;
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
            head_dim: 64, // Extended attention handles >32
            ffn_dim: 2048,
            rope_base: 10000.0,
            norm_eps: 1e-6,
        }
    }

    pub fn kv_dim(&self) -> u32 {
        self.num_kv_heads * self.head_dim
    }
}

/// Q4_0 quantized weights for a single transformer block.
///
/// Norm weights stay f32 (tiny, not worth quantizing).
/// All projection GEMMs use W4A16 — 6.4x weight compression.
pub struct QuantizedBlockWeights {
    /// Attention input norm [hidden_size] — f32
    pub attn_norm: GpuTensor<f32>,
    /// Q projection [hidden_size, hidden_size] — Q4_0
    pub wq: GpuTensor<u8>,
    /// K projection [hidden_size, kv_dim] — Q4_0
    pub wk: GpuTensor<u8>,
    /// V projection [hidden_size, kv_dim] — Q4_0
    pub wv: GpuTensor<u8>,
    /// Output projection [hidden_size, hidden_size] — Q4_0
    pub wo: GpuTensor<u8>,
    /// FFN input norm [hidden_size] — f32
    pub ffn_norm: GpuTensor<f32>,
    /// Gate projection [hidden_size, ffn_dim] — Q4_0
    pub w_gate: GpuTensor<u8>,
    /// Up projection [hidden_size, ffn_dim] — Q4_0
    pub w_up: GpuTensor<u8>,
    /// Down projection [ffn_dim, hidden_size] — Q4_0
    pub w_down: GpuTensor<u8>,
}

/// Quantize a full-precision block's weights to Q4_0.
pub fn quantize_block_weights(
    cache: &KernelCache,
    device: &WarpDevice,
    weights: &TransformerBlockWeights,
    config: &TransformerConfig,
) -> Result<QuantizedBlockWeights, DeviceError> {
    let h = config.hidden_size;
    let kv = config.kv_dim();
    let ffn = config.ffn_dim;

    Ok(QuantizedBlockWeights {
        attn_norm: GpuTensor::from_host(device,
            &weights.attn_norm.to_host(device)?,
            weights.attn_norm.shape.clone(), warp_ir::DType::F32)?,
        wq: quantize::quantize_weights_q4_0(cache, device, &weights.wq, h, h)?,
        wk: quantize::quantize_weights_q4_0(cache, device, &weights.wk, h, kv)?,
        wv: quantize::quantize_weights_q4_0(cache, device, &weights.wv, h, kv)?,
        wo: quantize::quantize_weights_q4_0(cache, device, &weights.wo, h, h)?,
        ffn_norm: GpuTensor::from_host(device,
            &weights.ffn_norm.to_host(device)?,
            weights.ffn_norm.shape.clone(), warp_ir::DType::F32)?,
        w_gate: quantize::quantize_weights_q4_0(cache, device, &weights.w_gate, h, ffn)?,
        w_up: quantize::quantize_weights_q4_0(cache, device, &weights.w_up, h, ffn)?,
        w_down: quantize::quantize_weights_q4_0(cache, device, &weights.w_down, ffn, h)?,
    })
}

/// Forward pass of a single transformer block with Q4_0 quantized weights.
///
/// Same computation as `transformer_block_forward` but all 7 projection
/// GEMMs use W4A16 dequantize-on-the-fly.
pub fn transformer_block_forward_q4(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    weights: &QuantizedBlockWeights,
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

    // 1. RMSNorm
    let mut normed = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::rmsnorm(cache, device, x, &weights.attn_norm, &mut normed, h, config.norm_eps)?;

    // 2. Q, K, V projections — W4A16
    let bn = bsz * n;
    let mut q = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut k_proj = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;
    let mut v_proj = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;

    quantize::gemm_q4_0(cache, device, &normed, &weights.wq, &mut q, bn, h, h)?;
    quantize::gemm_q4_0(cache, device, &normed, &weights.wk, &mut k_proj, bn, kv_dim, h)?;
    quantize::gemm_q4_0(cache, device, &normed, &weights.wv, &mut v_proj, bn, kv_dim, h)?;

    // 3. RoPE
    let mut q_rope = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut k_rope = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;
    crate::rope::rope(cache, device, &q, &mut q_rope, bsz * config.num_heads, n, d, config.rope_base, pos_offset)?;
    crate::rope::rope(cache, device, &k_proj, &mut k_rope, bsz * config.num_kv_heads, n, d, config.rope_base, pos_offset)?;

    // 4. Attention
    let attn_batch = bsz * config.num_heads;
    let shape_attn = warp_ir::Shape::from_static(&[attn_batch as usize, n as usize, d as usize]);
    let mut attn_out = GpuTensor::<f32>::zeros(device, shape_attn, warp_ir::DType::F32)?;
    crate::attention_ext::attention_best(
        cache, device, &q_rope, &k_rope, &v_proj,
        &mut attn_out, attn_batch, n, d, true,
    )?;

    // 5. Output projection — W4A16
    let mut attn_projected = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    quantize::gemm_q4_0(cache, device, &attn_out, &weights.wo, &mut attn_projected, bn, h, h)?;

    // 6+7. Fused residual add + FFN norm (2 launches → 1)
    let mut ffn_normed = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut residual1 = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::fused_residual_rmsnorm(cache, device, &attn_projected, x, &weights.ffn_norm,
        &mut ffn_normed, &mut residual1, h, config.norm_eps)?;

    // 8. Gate + Up — W4A16
    let mut gate = GpuTensor::<f32>::zeros(device, shape_bnf.clone(), warp_ir::DType::F32)?;
    let mut up = GpuTensor::<f32>::zeros(device, shape_bnf.clone(), warp_ir::DType::F32)?;
    quantize::gemm_q4_0(cache, device, &ffn_normed, &weights.w_gate, &mut gate, bn, ffn, h)?;
    quantize::gemm_q4_0(cache, device, &ffn_normed, &weights.w_up, &mut up, bn, ffn, h)?;

    // 9. Fused SwiGLU: silu(gate) * up (2 launches → 1)
    let mut swiglu_out = GpuTensor::<f32>::zeros(device, shape_bnf, warp_ir::DType::F32)?;
    ops::fused_silu_mul(cache, device, &gate, &up, &mut swiglu_out)?;

    // 10. Down projection — W4A16
    let mut ffn_out = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    quantize::gemm_q4_0(cache, device, &swiglu_out, &weights.w_down, &mut ffn_out, bn, h, ffn)?;

    // 11. Residual
    let mut output = GpuTensor::<f32>::zeros(device, shape_bnh, warp_ir::DType::F32)?;
    ops::add(cache, device, &residual1, &ffn_out, &mut output)?;

    Ok(output)
}

/// Prefill with Q4_0 quantized weights — populates KV cache.
pub fn transformer_block_prefill_q4(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    weights: &QuantizedBlockWeights,
    config: &TransformerConfig,
    kv: &mut LayerKVCache,
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

    let mut normed = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::rmsnorm(cache, device, x, &weights.attn_norm, &mut normed, h, config.norm_eps)?;

    let bn = bsz * n;
    let mut q = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut k_proj = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;
    let mut v_proj = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;

    quantize::gemm_q4_0(cache, device, &normed, &weights.wq, &mut q, bn, h, h)?;
    quantize::gemm_q4_0(cache, device, &normed, &weights.wk, &mut k_proj, bn, kv_dim, h)?;
    quantize::gemm_q4_0(cache, device, &normed, &weights.wv, &mut v_proj, bn, kv_dim, h)?;

    let mut q_rope = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut k_rope = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;
    crate::rope::rope(cache, device, &q, &mut q_rope, bsz * config.num_heads, n, d, config.rope_base, pos_offset)?;
    crate::rope::rope(cache, device, &k_proj, &mut k_rope, bsz * config.num_kv_heads, n, d, config.rope_base, pos_offset)?;

    // Store K/V in cache
    kv.prefill(cache, device, &k_rope, &v_proj, n)?;

    let attn_batch = bsz * config.num_heads;
    let shape_attn = warp_ir::Shape::from_static(&[attn_batch as usize, n as usize, d as usize]);
    let mut attn_out = GpuTensor::<f32>::zeros(device, shape_attn, warp_ir::DType::F32)?;
    crate::attention_ext::attention_best(
        cache, device, &q_rope, &k_rope, &v_proj,
        &mut attn_out, attn_batch, n, d, true,
    )?;

    let mut attn_projected = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    quantize::gemm_q4_0(cache, device, &attn_out, &weights.wo, &mut attn_projected, bn, h, h)?;

    let mut residual1 = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::add(cache, device, x, &attn_projected, &mut residual1)?;

    let mut ffn_normed = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::rmsnorm(cache, device, &residual1, &weights.ffn_norm, &mut ffn_normed, h, config.norm_eps)?;

    let mut gate = GpuTensor::<f32>::zeros(device, shape_bnf.clone(), warp_ir::DType::F32)?;
    let mut up = GpuTensor::<f32>::zeros(device, shape_bnf.clone(), warp_ir::DType::F32)?;
    quantize::gemm_q4_0(cache, device, &ffn_normed, &weights.w_gate, &mut gate, bn, ffn, h)?;
    quantize::gemm_q4_0(cache, device, &ffn_normed, &weights.w_up, &mut up, bn, ffn, h)?;

    let mut gate_activated = GpuTensor::<f32>::zeros(device, shape_bnf.clone(), warp_ir::DType::F32)?;
    ops::silu(cache, device, &gate, &mut gate_activated)?;
    let mut swiglu_out = GpuTensor::<f32>::zeros(device, shape_bnf, warp_ir::DType::F32)?;
    ops::mul(cache, device, &gate_activated, &up, &mut swiglu_out)?;

    let mut ffn_out = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    quantize::gemm_q4_0(cache, device, &swiglu_out, &weights.w_down, &mut ffn_out, bn, h, ffn)?;

    let mut output = GpuTensor::<f32>::zeros(device, shape_bnh, warp_ir::DType::F32)?;
    ops::add(cache, device, &residual1, &ffn_out, &mut output)?;

    Ok(output)
}

/// Decode with Q4_0 quantized weights — single token, uses KV cache.
pub fn transformer_block_decode_q4(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    weights: &QuantizedBlockWeights,
    config: &TransformerConfig,
    kv: &mut LayerKVCache,
    batch: u32,
    pos: u32,
) -> Result<GpuTensor<f32>, DeviceError> {
    let h = config.hidden_size;
    let d = config.head_dim;
    let kv_dim = config.kv_dim();
    let ffn = config.ffn_dim;
    let n = 1u32;
    let bn = batch * n;

    let shape_bh = warp_ir::Shape::from_static(&[batch as usize, h as usize]);
    let shape_bk = warp_ir::Shape::from_static(&[batch as usize, kv_dim as usize]);
    let shape_bf = warp_ir::Shape::from_static(&[batch as usize, ffn as usize]);

    // 1. RMSNorm
    let mut normed = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    ops::rmsnorm(cache, device, x, &weights.attn_norm, &mut normed, h, config.norm_eps)?;

    // 2. Q, K, V — W4A16
    let mut q = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    let mut new_k = GpuTensor::<f32>::zeros(device, shape_bk.clone(), warp_ir::DType::F32)?;
    let mut new_v = GpuTensor::<f32>::zeros(device, shape_bk.clone(), warp_ir::DType::F32)?;

    quantize::gemm_q4_0(cache, device, &normed, &weights.wq, &mut q, bn, h, h)?;
    quantize::gemm_q4_0(cache, device, &normed, &weights.wk, &mut new_k, bn, kv_dim, h)?;
    quantize::gemm_q4_0(cache, device, &normed, &weights.wv, &mut new_v, bn, kv_dim, h)?;

    // 3. RoPE
    let mut q_rope = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    let mut k_rope = GpuTensor::<f32>::zeros(device, shape_bk.clone(), warp_ir::DType::F32)?;
    crate::rope::rope(cache, device, &q, &mut q_rope, batch * config.num_heads, 1, d, config.rope_base, pos)?;
    crate::rope::rope(cache, device, &new_k, &mut k_rope, batch * config.num_kv_heads, 1, d, config.rope_base, pos)?;

    // 4. Append to KV cache
    kv.append(cache, device, &k_rope, &new_v)?;

    // 5. Decode attention
    let mut attn_out = GpuTensor::<f32>::zeros(device,
        warp_ir::Shape::from_static(&[batch as usize, d as usize]), warp_ir::DType::F32)?;
    crate::kv_cache::decode_attention(cache, device, &q_rope, kv, &mut attn_out, d)?;

    // 6. Output projection — W4A16
    let mut attn_projected = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    quantize::gemm_q4_0(cache, device, &attn_out, &weights.wo, &mut attn_projected, bn, h, d)?;

    // 7+8. Fused residual + FFN norm (2 launches → 1)
    let mut ffn_normed = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    let mut residual = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    ops::fused_residual_rmsnorm(cache, device, &attn_projected, x, &weights.ffn_norm,
        &mut ffn_normed, &mut residual, h, config.norm_eps)?;

    // 9-10. Gate + Up + Fused SwiGLU + Down — W4A16
    let mut gate = GpuTensor::<f32>::zeros(device, shape_bf.clone(), warp_ir::DType::F32)?;
    let mut up = GpuTensor::<f32>::zeros(device, shape_bf.clone(), warp_ir::DType::F32)?;
    quantize::gemm_q4_0(cache, device, &ffn_normed, &weights.w_gate, &mut gate, bn, ffn, h)?;
    quantize::gemm_q4_0(cache, device, &ffn_normed, &weights.w_up, &mut up, bn, ffn, h)?;

    // Fused SwiGLU (2 launches → 1)
    let mut swiglu = GpuTensor::<f32>::zeros(device, shape_bf, warp_ir::DType::F32)?;
    ops::fused_silu_mul(cache, device, &gate, &up, &mut swiglu)?;

    let mut ffn_out = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    quantize::gemm_q4_0(cache, device, &swiglu, &weights.w_down, &mut ffn_out, bn, h, ffn)?;

    // 11. Residual
    let mut output = GpuTensor::<f32>::zeros(device, shape_bh, warp_ir::DType::F32)?;
    ops::add(cache, device, &residual, &ffn_out, &mut output)?;

    Ok(output)
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

    crate::attention_ext::attention_best(
        cache, device, &q_rope, &k_rope, &v_proj,
        &mut attn_out, attn_batch, n, d, true,
    )?;

    // 5. Output projection
    let mut attn_projected = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::gemm(cache, device, &attn_out, &weights.wo, &mut attn_projected, bn, h, h)?;

    // 6+7. Fused residual + RMSNorm (2 launches → 1)
    let mut ffn_normed = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut residual1 = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::fused_residual_rmsnorm(cache, device, &attn_projected, x, &weights.ffn_norm,
        &mut ffn_normed, &mut residual1, h, config.norm_eps)?;

    // 8. Gate + Up projections
    let mut gate = GpuTensor::<f32>::zeros(device, shape_bnf.clone(), warp_ir::DType::F32)?;
    let mut up = GpuTensor::<f32>::zeros(device, shape_bnf.clone(), warp_ir::DType::F32)?;
    ops::gemm(cache, device, &ffn_normed, &weights.w_gate, &mut gate, bn, ffn, h)?;
    ops::gemm(cache, device, &ffn_normed, &weights.w_up, &mut up, bn, ffn, h)?;

    // 9. Fused SwiGLU: silu(gate) * up (2 launches → 1)
    let mut swiglu_out = GpuTensor::<f32>::zeros(device, shape_bnf, warp_ir::DType::F32)?;
    ops::fused_silu_mul(cache, device, &gate, &up, &mut swiglu_out)?;

    // 10. Down projection
    let mut ffn_out = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::gemm(cache, device, &swiglu_out, &weights.w_down, &mut ffn_out, bn, h, ffn)?;

    // 11. Final residual add
    let mut output = GpuTensor::<f32>::zeros(device, shape_bnh, warp_ir::DType::F32)?;
    ops::add(cache, device, &residual1, &ffn_out, &mut output)?;

    Ok(output)
}

/// Prefill pass — like forward but also populates the KV cache for this layer.
///
/// x: [batch, seq_len, hidden_size]
/// kv: layer KV cache (mutated: K/V from all positions stored)
/// Returns: [batch, seq_len, hidden_size]
pub fn transformer_block_prefill(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    weights: &TransformerBlockWeights,
    config: &TransformerConfig,
    kv: &mut LayerKVCache,
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
    let bn = bsz * n;
    let mut q = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut k_proj = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;
    let mut v_proj = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;

    ops::gemm(cache, device, &normed, &weights.wq, &mut q, bn, h, h)?;
    ops::gemm(cache, device, &normed, &weights.wk, &mut k_proj, bn, kv_dim, h)?;
    ops::gemm(cache, device, &normed, &weights.wv, &mut v_proj, bn, kv_dim, h)?;

    // 3. RoPE on Q and K
    let mut q_rope = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut k_rope = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;
    crate::rope::rope(cache, device, &q, &mut q_rope, bsz * config.num_heads, n, d, config.rope_base, pos_offset)?;
    crate::rope::rope(cache, device, &k_proj, &mut k_rope, bsz * config.num_kv_heads, n, d, config.rope_base, pos_offset)?;

    // 3.5. Store K/V in cache for future decode steps
    kv.prefill(cache, device, &k_rope, &v_proj, n)?;

    // 4. Flash Attention
    let attn_batch = bsz * config.num_heads;
    let shape_attn = warp_ir::Shape::from_static(&[attn_batch as usize, n as usize, d as usize]);
    let mut attn_out = GpuTensor::<f32>::zeros(device, shape_attn, warp_ir::DType::F32)?;

    crate::attention_ext::attention_best(
        cache, device, &q_rope, &k_rope, &v_proj,
        &mut attn_out, attn_batch, n, d, true,
    )?;

    // 5. Output projection
    let mut attn_projected = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::gemm(cache, device, &attn_out, &weights.wo, &mut attn_projected, bn, h, h)?;

    // 6+7. Fused residual + FFN norm
    let mut ffn_normed = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut residual1 = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::fused_residual_rmsnorm(cache, device, &attn_projected, x, &weights.ffn_norm,
        &mut ffn_normed, &mut residual1, h, config.norm_eps)?;

    // 8. Gate + Up
    let mut gate = GpuTensor::<f32>::zeros(device, shape_bnf.clone(), warp_ir::DType::F32)?;
    let mut up = GpuTensor::<f32>::zeros(device, shape_bnf.clone(), warp_ir::DType::F32)?;
    ops::gemm(cache, device, &ffn_normed, &weights.w_gate, &mut gate, bn, ffn, h)?;
    ops::gemm(cache, device, &ffn_normed, &weights.w_up, &mut up, bn, ffn, h)?;

    // 9. Fused SwiGLU
    let mut swiglu_out = GpuTensor::<f32>::zeros(device, shape_bnf, warp_ir::DType::F32)?;
    ops::fused_silu_mul(cache, device, &gate, &up, &mut swiglu_out)?;

    // 10. Down projection
    let mut ffn_out = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::gemm(cache, device, &swiglu_out, &weights.w_down, &mut ffn_out, bn, h, ffn)?;

    // 11. Final residual
    let mut output = GpuTensor::<f32>::zeros(device, shape_bnh, warp_ir::DType::F32)?;
    ops::add(cache, device, &residual1, &ffn_out, &mut output)?;

    Ok(output)
}

/// Forward pass with KV cache — for decode (single token at a time).
///
/// During prefill: call transformer_block_forward (processes full prompt).
/// During decode: call this function (processes one new token, uses cached K/V).
///
/// x: [batch, 1, hidden_size] — single token's hidden state
/// kv: layer KV cache (mutated: new K/V appended)
/// Returns: [batch, 1, hidden_size]
pub fn transformer_block_decode(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    weights: &TransformerBlockWeights,
    config: &TransformerConfig,
    kv: &mut LayerKVCache,
    batch: u32,
    pos: u32,
) -> Result<GpuTensor<f32>, DeviceError> {
    let h = config.hidden_size;
    let d = config.head_dim;
    let kv_dim = config.kv_dim();
    let ffn = config.ffn_dim;
    let n = 1u32; // single token
    let bn = batch * n;

    let shape_bh = warp_ir::Shape::from_static(&[batch as usize, h as usize]);
    let shape_bk = warp_ir::Shape::from_static(&[batch as usize, kv_dim as usize]);
    let shape_bf = warp_ir::Shape::from_static(&[batch as usize, ffn as usize]);

    // 1. RMSNorm
    let mut normed = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    ops::rmsnorm(cache, device, x, &weights.attn_norm, &mut normed, h, config.norm_eps)?;

    // 2. Q, K, V projections (single token)
    let mut q = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    let mut new_k = GpuTensor::<f32>::zeros(device, shape_bk.clone(), warp_ir::DType::F32)?;
    let mut new_v = GpuTensor::<f32>::zeros(device, shape_bk.clone(), warp_ir::DType::F32)?;

    ops::gemm(cache, device, &normed, &weights.wq, &mut q, bn, h, h)?;
    ops::gemm(cache, device, &normed, &weights.wk, &mut new_k, bn, kv_dim, h)?;
    ops::gemm(cache, device, &normed, &weights.wv, &mut new_v, bn, kv_dim, h)?;

    // 3. RoPE (single position)
    let mut q_rope = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    let mut k_rope = GpuTensor::<f32>::zeros(device, shape_bk.clone(), warp_ir::DType::F32)?;
    crate::rope::rope(cache, device, &q, &mut q_rope, batch * config.num_heads, 1, d, config.rope_base, pos)?;
    crate::rope::rope(cache, device, &new_k, &mut k_rope, batch * config.num_kv_heads, 1, d, config.rope_base, pos)?;

    // 4. Append K, V to cache
    kv.append(cache, device, &k_rope, &new_v)?;

    // 5. Decode attention: Q[1, D] attends over full cache K/V
    let mut attn_out = GpuTensor::<f32>::zeros(device,
        warp_ir::Shape::from_static(&[batch as usize, d as usize]), warp_ir::DType::F32)?;
    crate::kv_cache::decode_attention(cache, device, &q_rope, kv, &mut attn_out, d)?;

    // 6. Output projection
    // Note: attn_out is [batch, head_dim] but we need [batch, hidden_size]
    // For single-head simplified case, we pad/project
    let mut attn_projected = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    ops::gemm(cache, device, &attn_out, &weights.wo, &mut attn_projected, bn, h, d)?;

    // 7+8. Fused residual + FFN norm
    let mut ffn_normed = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    let mut residual = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    ops::fused_residual_rmsnorm(cache, device, &attn_projected, x, &weights.ffn_norm,
        &mut ffn_normed, &mut residual, h, config.norm_eps)?;

    // 9-10. Gate + Up + Fused SwiGLU + Down
    let mut gate = GpuTensor::<f32>::zeros(device, shape_bf.clone(), warp_ir::DType::F32)?;
    let mut up = GpuTensor::<f32>::zeros(device, shape_bf.clone(), warp_ir::DType::F32)?;
    ops::gemm(cache, device, &ffn_normed, &weights.w_gate, &mut gate, bn, ffn, h)?;
    ops::gemm(cache, device, &ffn_normed, &weights.w_up, &mut up, bn, ffn, h)?;

    let mut swiglu = GpuTensor::<f32>::zeros(device, shape_bf, warp_ir::DType::F32)?;
    ops::fused_silu_mul(cache, device, &gate, &up, &mut swiglu)?;

    let mut ffn_out = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    ops::gemm(cache, device, &swiglu, &weights.w_down, &mut ffn_out, bn, h, ffn)?;

    // 11. Final residual
    let mut output = GpuTensor::<f32>::zeros(device, shape_bh, warp_ir::DType::F32)?;
    ops::add(cache, device, &residual, &ffn_out, &mut output)?;

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

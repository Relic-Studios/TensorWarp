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
use crate::cublas_gemm;
use crate::device::{DeviceError, WarpDevice};
use crate::fp16;
use crate::kv_cache::LayerKVCache;
use crate::quantize;
use crate::tensor::GpuTensor;
use crate::ops;

/// Mixed-precision weights for a single transformer block.
/// Weight matrices are FP16 (2x bandwidth savings), norms and biases stay F32.
/// During decode: cast activations F32→F16, HGEMM with F16 weights, cast output F16→F32.
pub struct TransformerBlockWeightsF16 {
    /// Attention input norm weights [hidden_size] — F32 for stability
    pub attn_norm: GpuTensor<f32>,
    /// Q projection [hidden_size, hidden_size] — FP16
    pub wq: GpuTensor<half::f16>,
    /// K projection [hidden_size, kv_dim] — FP16
    pub wk: GpuTensor<half::f16>,
    /// V projection [hidden_size, kv_dim] — FP16
    pub wv: GpuTensor<half::f16>,
    /// Output projection [hidden_size, hidden_size] — FP16
    pub wo: GpuTensor<half::f16>,
    /// FFN input norm weights [hidden_size] — F32 for stability
    pub ffn_norm: GpuTensor<f32>,
    /// Gate projection [hidden_size, ffn_dim] — FP16
    pub w_gate: GpuTensor<half::f16>,
    /// Up projection [hidden_size, ffn_dim] — FP16
    pub w_up: GpuTensor<half::f16>,
    /// Down projection [ffn_dim, hidden_size] — FP16
    pub w_down: GpuTensor<half::f16>,
    /// Optional Q/K/V biases — F32 for stability
    pub bq: Option<GpuTensor<f32>>,
    pub bk: Option<GpuTensor<f32>>,
    pub bv: Option<GpuTensor<f32>>,
    /// Pre-computed FP16 norm weights (avoid per-call casting)
    pub attn_norm_f16: Option<GpuTensor<half::f16>>,
    pub ffn_norm_f16: Option<GpuTensor<half::f16>>,
    /// Pre-computed FP16 biases (avoid per-call casting)
    pub bq_f16: Option<GpuTensor<half::f16>>,
    pub bk_f16: Option<GpuTensor<half::f16>>,
    pub bv_f16: Option<GpuTensor<half::f16>>,
}

impl TransformerBlockWeightsF16 {
    /// Returns total GPU memory used by this layer's weights.
    pub fn total_memory_bytes(&self) -> usize {
        let mut total = self.attn_norm.size_bytes()
            + self.wq.size_bytes()
            + self.wk.size_bytes()
            + self.wv.size_bytes()
            + self.wo.size_bytes()
            + self.ffn_norm.size_bytes()
            + self.w_gate.size_bytes()
            + self.w_up.size_bytes()
            + self.w_down.size_bytes();
        if let Some(ref bq) = self.bq { total += bq.size_bytes(); }
        if let Some(ref bk) = self.bk { total += bk.size_bytes(); }
        if let Some(ref bv) = self.bv { total += bv.size_bytes(); }
        if let Some(ref n) = self.attn_norm_f16 { total += n.size_bytes(); }
        if let Some(ref n) = self.ffn_norm_f16 { total += n.size_bytes(); }
        total
    }

    /// Get FP16 attention norm weight (pre-computed or panic).
    pub fn attn_norm_f16(&self) -> &GpuTensor<half::f16> {
        self.attn_norm_f16.as_ref().expect("FP16 norm weights not prepared — call prepare_f16_norms first")
    }

    /// Get FP16 FFN norm weight (pre-computed or panic).
    pub fn ffn_norm_f16(&self) -> &GpuTensor<half::f16> {
        self.ffn_norm_f16.as_ref().expect("FP16 norm weights not prepared — call prepare_f16_norms first")
    }

    /// Pre-compute FP16 norm weights from F32 (call once at load time).
    pub fn prepare_f16_norms(
        &mut self,
        cache: &crate::cache::KernelCache,
        device: &crate::device::WarpDevice,
    ) -> Result<(), crate::device::DeviceError> {
        let mut attn_f16 = GpuTensor::<half::f16>::zeros(device, self.attn_norm.shape.clone(), warp_ir::DType::F16)?;
        fp16::cast_f32_to_f16(cache, device, &self.attn_norm, &mut attn_f16)?;
        self.attn_norm_f16 = Some(attn_f16);

        let mut ffn_f16 = GpuTensor::<half::f16>::zeros(device, self.ffn_norm.shape.clone(), warp_ir::DType::F16)?;
        fp16::cast_f32_to_f16(cache, device, &self.ffn_norm, &mut ffn_f16)?;
        self.ffn_norm_f16 = Some(ffn_f16);

        Ok(())
    }
}

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
    /// Optional Q/K/V biases (Qwen, Phi, etc. — not present in LLaMA/Mistral)
    pub bq: Option<GpuTensor<f32>>,
    pub bk: Option<GpuTensor<f32>>,
    pub bv: Option<GpuTensor<f32>>,
    /// Fused QKV weight matrix [hidden_size, hidden_size + 2*kv_dim] — optional optimization.
    /// When present, a single GEMM replaces three separate Q/K/V projections.
    pub wqkv: Option<GpuTensor<f32>>,
    /// Fused QKV bias [hidden_size + 2*kv_dim] — optional, only when wqkv is Some and biases exist.
    pub bqkv: Option<GpuTensor<f32>>,
    /// Fused gate+up weight matrix [hidden_size, 2*ffn_dim] — optional optimization.
    /// When present, a single GEMM replaces the separate gate and up projections.
    pub w_gate_up: Option<GpuTensor<f32>>,
}

/// Attention dispatch mode for a transformer block.
///
/// Standard: uses the contiguous KV cache with decode_attention (default).
/// PagedAttention: memory-efficient scattered block KV via vLLM-style paging.
/// SlidingWindow: only attend to the last `window_size` tokens (Mistral/Gemma).
#[derive(Debug, Clone)]
pub enum AttentionMode {
    /// Current behavior — contiguous KV cache, full causal attention.
    Standard,
    /// Paged attention — KV blocks allocated on demand from a pool.
    /// Requires a `PagedKVPool` and `PagedSequence` per request.
    PagedAttention,
    /// Sliding window — only attend to the last `window_size` positions.
    /// Used by Mistral-7B (W=4096), Gemma 2, etc.
    SlidingWindow { window_size: u32 },
}

impl Default for AttentionMode {
    fn default() -> Self {
        AttentionMode::Standard
    }
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
    /// Attention mode — controls which attention kernel is dispatched during decode.
    pub attention_mode: AttentionMode,
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
            attention_mode: AttentionMode::Standard,
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
            attention_mode: AttentionMode::Standard,
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
            attention_mode: AttentionMode::Standard,
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
    // GQA: repeat K/V heads to match Q head count when num_kv_heads < num_heads
    let attn_batch = bsz * config.num_heads;
    let k_for_attn = repeat_kv(device, &k_rope, bsz * config.num_kv_heads, attn_batch, n, d)?;
    let v_for_attn = repeat_kv(device, &v_proj, bsz * config.num_kv_heads, attn_batch, n, d)?;

    let shape_attn = warp_ir::Shape::from_static(&[attn_batch as usize, n as usize, d as usize]);
    let mut attn_out = GpuTensor::<f32>::zeros(device, shape_attn, warp_ir::DType::F32)?;
    crate::attention_ext::attention_best(
        cache, device, &q_rope, &k_for_attn, &v_for_attn,
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

    // GQA: repeat K/V heads to match Q head count when num_kv_heads < num_heads
    let attn_batch = bsz * config.num_heads;
    let k_for_attn = repeat_kv(device, &k_rope, bsz * config.num_kv_heads, attn_batch, n, d)?;
    let v_for_attn = repeat_kv(device, &v_proj, bsz * config.num_kv_heads, attn_batch, n, d)?;

    let shape_attn = warp_ir::Shape::from_static(&[attn_batch as usize, n as usize, d as usize]);
    let mut attn_out = GpuTensor::<f32>::zeros(device, shape_attn, warp_ir::DType::F32)?;
    crate::attention_ext::attention_best(
        cache, device, &q_rope, &k_for_attn, &v_for_attn,
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

    // 5. Decode attention (multihead with GQA support)
    let mut attn_out = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    crate::kv_cache::decode_attention_multihead(
        cache, device, &q_rope, kv, &mut attn_out,
        config.num_heads, config.num_kv_heads, d,
    )?;

    // 6. Output projection — W4A16 (K=h, not K=d)
    let mut attn_projected = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    quantize::gemm_q4_0(cache, device, &attn_out, &weights.wo, &mut attn_projected, bn, h, h)?;

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

/// Repeat KV heads to match Q heads for GQA (Grouped Query Attention).
///
/// When num_kv_heads < num_heads, each KV head serves multiple Q heads.
/// This function repeats K or V from [bsz*num_kv_heads, seq_len, head_dim]
/// to [bsz*num_heads, seq_len, head_dim] by duplicating each KV head
/// `num_heads / num_kv_heads` times.
///
/// If num_kv_heads == num_heads (MHA), returns the input unchanged.
fn repeat_kv(
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    num_kv_heads: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
) -> Result<GpuTensor<f32>, DeviceError> {
    if num_kv_heads == num_heads {
        // MHA — no repetition needed; clone the tensor to avoid lifetime issues
        let data = x.to_host(device)?;
        return GpuTensor::from_host(device, &data, x.shape.clone(), warp_ir::DType::F32);
    }

    let n_rep = num_heads / num_kv_heads;
    let kv_total = (num_kv_heads * seq_len * head_dim) as usize;
    let out_total = (num_heads * seq_len * head_dim) as usize;

    // CPU repeat: for each KV head, copy its seq_len*head_dim block n_rep times
    let src = x.to_host(device)?;
    assert_eq!(src.len(), kv_total, "repeat_kv: input size mismatch");
    let mut dst = vec![0.0f32; out_total];

    let head_stride = (seq_len * head_dim) as usize;
    for kv_h in 0..num_kv_heads as usize {
        let src_off = kv_h * head_stride;
        for r in 0..n_rep as usize {
            let dst_h = kv_h * n_rep as usize + r;
            let dst_off = dst_h * head_stride;
            dst[dst_off..dst_off + head_stride].copy_from_slice(&src[src_off..src_off + head_stride]);
        }
    }

    let out_shape = warp_ir::Shape::from_static(&[num_heads as usize, seq_len as usize, head_dim as usize]);
    GpuTensor::from_host(device, &dst, out_shape, warp_ir::DType::F32)
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
    let mut q: GpuTensor<f32> = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut k_proj: GpuTensor<f32> = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;
    let mut v_proj: GpuTensor<f32> = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;

    // 2b. Fused GEMM+bias when biases present (1 launch instead of 2),
    //     plain GEMM otherwise.
    if let Some(ref bq) = weights.bq {
        ops::gemm_bias(cache, device, &normed, &weights.wq, bq, &mut q, bn, h, h)?;
    } else {
        ops::gemm(cache, device, &normed, &weights.wq, &mut q, bn, h, h)?;
    }
    if let Some(ref bk) = weights.bk {
        ops::gemm_bias(cache, device, &normed, &weights.wk, bk, &mut k_proj, bn, kv_dim, h)?;
    } else {
        ops::gemm(cache, device, &normed, &weights.wk, &mut k_proj, bn, kv_dim, h)?;
    }
    if let Some(ref bv) = weights.bv {
        ops::gemm_bias(cache, device, &normed, &weights.wv, bv, &mut v_proj, bn, kv_dim, h)?;
    } else {
        ops::gemm(cache, device, &normed, &weights.wv, &mut v_proj, bn, kv_dim, h)?;
    }

    // 3. RoPE on Q and K
    // Apply per-head: reshape conceptually to [B, N, num_heads, head_dim]
    // For now apply on flat [B*N*num_heads, head_dim] — RoPE is per-position
    let mut q_rope = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut k_rope = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;
    crate::rope::rope(cache, device, &q, &mut q_rope, bsz * config.num_heads, n, d, config.rope_base, pos_offset)?;
    crate::rope::rope(cache, device, &k_proj, &mut k_rope, bsz * config.num_kv_heads, n, d, config.rope_base, pos_offset)?;

    // 4. Flash Attention (per-head, simplified: treat all heads as batch dim)
    // Reshape: [B, N, num_heads * D] → [B * num_heads, N, D]
    // GQA: repeat K/V heads to match Q head count when num_kv_heads < num_heads
    let attn_batch = bsz * config.num_heads;
    let k_for_attn = repeat_kv(device, &k_rope, bsz * config.num_kv_heads, attn_batch, n, d)?;
    let v_for_attn = repeat_kv(device, &v_proj, bsz * config.num_kv_heads, attn_batch, n, d)?;

    let shape_attn = warp_ir::Shape::from_static(&[attn_batch as usize, n as usize, d as usize]);
    let mut attn_out = GpuTensor::<f32>::zeros(device, shape_attn, warp_ir::DType::F32)?;

    crate::attention_ext::attention_best(
        cache, device, &q_rope, &k_for_attn, &v_for_attn,
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
    let mut q: GpuTensor<f32> = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut k_proj: GpuTensor<f32> = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;
    let mut v_proj: GpuTensor<f32> = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;

    // 2b. Fused GEMM+bias when biases present (1 launch instead of 2),
    //     plain GEMM otherwise.
    if let Some(ref bq) = weights.bq {
        ops::gemm_bias(cache, device, &normed, &weights.wq, bq, &mut q, bn, h, h)?;
    } else {
        ops::gemm(cache, device, &normed, &weights.wq, &mut q, bn, h, h)?;
    }
    if let Some(ref bk) = weights.bk {
        ops::gemm_bias(cache, device, &normed, &weights.wk, bk, &mut k_proj, bn, kv_dim, h)?;
    } else {
        ops::gemm(cache, device, &normed, &weights.wk, &mut k_proj, bn, kv_dim, h)?;
    }
    if let Some(ref bv) = weights.bv {
        ops::gemm_bias(cache, device, &normed, &weights.wv, bv, &mut v_proj, bn, kv_dim, h)?;
    } else {
        ops::gemm(cache, device, &normed, &weights.wv, &mut v_proj, bn, kv_dim, h)?;
    }

    // 3. RoPE on Q and K
    // GEMM outputs [seq, heads*dim] (positions-first). RoPE needs [heads, seq, dim] (heads-first).
    // GPU transpose — no CPU roundtrips.
    let ns = n as usize;
    let nh = config.num_heads as usize;
    let nkv = config.num_kv_heads as usize;
    let hd = d as usize;

    // Transpose Q: [seq, num_heads*head_dim] → [num_heads, seq, head_dim] (GPU)
    let mut q_hf = GpuTensor::<f32>::zeros(device,
        warp_ir::Shape::from_static(&[nh, ns, hd]), warp_ir::DType::F32)?;
    ops::transpose_to_heads_first(cache, device, &q, &mut q_hf, config.num_heads, n, d)?;

    // Transpose K: [seq, kv_dim] → [num_kv_heads, seq, head_dim] (GPU)
    let mut k_hf = GpuTensor::<f32>::zeros(device,
        warp_ir::Shape::from_static(&[nkv, ns, hd]), warp_ir::DType::F32)?;
    ops::transpose_to_heads_first(cache, device, &k_proj, &mut k_hf, config.num_kv_heads, n, d)?;

    // Apply RoPE (now input is heads-first, matching kernel expectation)
    let mut q_rope = GpuTensor::<f32>::zeros(device,
        warp_ir::Shape::from_static(&[nh, ns, hd]), warp_ir::DType::F32)?;
    let mut k_rope = GpuTensor::<f32>::zeros(device,
        warp_ir::Shape::from_static(&[nkv, ns, hd]), warp_ir::DType::F32)?;
    crate::rope::rope(cache, device, &q_hf, &mut q_rope, bsz * config.num_heads, n, d, config.rope_base, pos_offset)?;
    crate::rope::rope(cache, device, &k_hf, &mut k_rope, bsz * config.num_kv_heads, n, d, config.rope_base, pos_offset)?;

    // 3.5. Store K/V in cache
    // k_rope is [num_kv_heads, seq, head_dim] (heads-first after RoPE)
    // KV cache needs [seq, kv_dim] (positions-first) — GPU transpose
    // V is already [seq, kv_dim] from GEMM output
    {
        let kv_d = nkv * hd;
        let mut k_for_cache = GpuTensor::<f32>::zeros(device,
            warp_ir::Shape::from_static(&[ns, kv_d]), warp_ir::DType::F32)?;
        ops::transpose_to_positions_first(cache, device, &k_rope, &mut k_for_cache, config.num_kv_heads, n, d)?;
        kv.prefill(cache, device, &k_for_cache, &v_proj, n)?;
    }

    // 4. Flash Attention
    // q_rope and k_rope are already heads-first [heads, seq, dim] — perfect for attention
    let attn_batch = bsz * config.num_heads;
    let k_for_attn = repeat_kv(device, &k_rope, bsz * config.num_kv_heads, attn_batch, n, d)?;
    // V needs transpose to heads-first for repeat_kv (GPU)
    let mut v_hf = GpuTensor::<f32>::zeros(device,
        warp_ir::Shape::from_static(&[nkv, ns, hd]), warp_ir::DType::F32)?;
    ops::transpose_to_heads_first(cache, device, &v_proj, &mut v_hf, config.num_kv_heads, n, d)?;
    let v_for_attn = repeat_kv(device, &v_hf, bsz * config.num_kv_heads, attn_batch, n, d)?;

    let shape_attn = warp_ir::Shape::from_static(&[attn_batch as usize, n as usize, d as usize]);
    let mut attn_out = GpuTensor::<f32>::zeros(device, shape_attn, warp_ir::DType::F32)?;

    crate::attention_ext::attention_best(
        cache, device, &q_rope, &k_for_attn, &v_for_attn,
        &mut attn_out, attn_batch, n, d, true,
    )?;

    // 5. Output projection
    // attn_out is [num_heads, seq, head_dim] (heads-first from attention)
    // GEMM needs [seq, hidden] (positions-first) — GPU transpose
    let mut attn_out_pf = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::transpose_to_positions_first(cache, device, &attn_out, &mut attn_out_pf, config.num_heads, n, d)?;

    let mut attn_projected = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::gemm(cache, device, &attn_out_pf, &weights.wo, &mut attn_projected, bn, h, h)?;

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
    let mut q: GpuTensor<f32>;
    let mut new_k: GpuTensor<f32>;
    let mut new_v: GpuTensor<f32>;

    if let Some(ref wqkv) = weights.wqkv {
        // ── Fused QKV path: ONE GEMM instead of THREE ──
        // wqkv is [hidden_size, h + kv_dim + kv_dim]
        let qkv_dim = h + kv_dim + kv_dim;
        let shape_qkv = warp_ir::Shape::from_static(&[batch as usize, qkv_dim as usize]);
        let mut qkv = GpuTensor::<f32>::zeros(device, shape_qkv, warp_ir::DType::F32)?;

        if let Some(ref bqkv) = weights.bqkv {
            // Fused GEMM + bias: QKV = normed @ Wqkv + Bqkv
            ops::gemm_bias(cache, device, &normed, wqkv, bqkv, &mut qkv, bn, qkv_dim, h)?;
        } else {
            ops::gemm(cache, device, &normed, wqkv, &mut qkv, bn, qkv_dim, h)?;
        }

        // Split QKV on GPU (1 cheap kernel instead of GPU→CPU→GPU roundtrip)
        q = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
        new_k = GpuTensor::<f32>::zeros(device, shape_bk.clone(), warp_ir::DType::F32)?;
        new_v = GpuTensor::<f32>::zeros(device, shape_bk.clone(), warp_ir::DType::F32)?;
        ops::split_qkv(cache, device, &qkv, &mut q, &mut new_k, &mut new_v, h, kv_dim, bn)?;
    } else {
        // ── Unfused fallback: 3 separate GEMMs ──
        q = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
        new_k = GpuTensor::<f32>::zeros(device, shape_bk.clone(), warp_ir::DType::F32)?;
        new_v = GpuTensor::<f32>::zeros(device, shape_bk.clone(), warp_ir::DType::F32)?;

        ops::gemm(cache, device, &normed, &weights.wq, &mut q, bn, h, h)?;
        ops::gemm(cache, device, &normed, &weights.wk, &mut new_k, bn, kv_dim, h)?;
        ops::gemm(cache, device, &normed, &weights.wv, &mut new_v, bn, kv_dim, h)?;

        // Biases handled by fused kernel below
    }

    // 3+4. Fused bias + RoPE + KV cache append (1 launch replaces 6)
    let mut q_rope = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    let mut k_rope = GpuTensor::<f32>::zeros(device, shape_bk.clone(), warp_ir::DType::F32)?;

    let has_bias = weights.bq.is_some() && weights.wqkv.is_none();
    let bq_ref = if has_bias { weights.bq.as_ref().unwrap() } else { &q };
    let bk_ref = if has_bias { weights.bk.as_ref().unwrap() } else { &new_k };
    let bv_ref = if has_bias { weights.bv.as_ref().unwrap() } else { &new_v };

    ops::fused_bias_rope_append(
        cache, device,
        &q, &new_k, &new_v,
        &mut q_rope, &mut k_rope,
        bq_ref, bk_ref, bv_ref,
        &mut kv.k, &mut kv.v,
        config.num_heads, config.num_kv_heads, d, kv_dim,
        pos, kv.max_seq_len, config.rope_base, has_bias,
    )?;
    kv.len = pos + 1;

    // 5. Run attention, dispatching by attention mode.
    // attn_out must hold ALL heads' output: [batch, num_heads * head_dim] = [batch, hidden_size]
    let mut attn_out = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;

    // V for cache was already written by fused kernel above — use new_v only
    // for non-standard paths that still call kv.append separately.
    match &config.attention_mode {
        AttentionMode::Standard => {
            // Cache already populated by fused kernel — just run attention.
            crate::kv_cache::decode_attention_multihead(
                cache, device, &q_rope, kv, &mut attn_out,
                config.num_heads, config.num_kv_heads, d,
            )?;
        }
        AttentionMode::PagedAttention => {
            // Paged attention path: KV is stored in scattered blocks via PagedKVPool.
            // Cache already populated by fused kernel above.
            // The real paged path is: engine pre-appends to PagedKVPool, then calls
            //   crate::paged_attention::paged_attention(cache, device, &q_rope, pool,
            //       &block_tables, &seq_lens, &mut attn_out, batch, num_heads,
            //       num_kv_heads, head_dim, max_blocks_per_seq)
            // which reads K/V from scattered blocks instead of contiguous cache.
            crate::kv_cache::decode_attention(cache, device, &q_rope, kv, &mut attn_out, d)?;
        }
        AttentionMode::SlidingWindow { window_size } => {
            // Sliding window path: cache already populated by fused kernel.
            let cache_len = kv.len;
            crate::sliding_window::sliding_window_decode(
                cache, device,
                &q_rope,       // [head_dim]
                &kv.k,         // [cache_len, head_dim]
                &kv.v,         // [cache_len, head_dim]
                &mut attn_out, // [head_dim]
                cache_len,
                d,
                *window_size,
            )?;
        }
    }

    // 6. Output projection
    // attn_out is [batch, hidden_size] (all heads concatenated after multihead attention)
    // wo is [hidden, hidden] (transposed from HF's [hidden, hidden])
    // C[1, 896] = attn_out[1, 896] @ wo[896, 896]
    let mut attn_projected = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    ops::gemm(cache, device, &attn_out, &weights.wo, &mut attn_projected, bn, h, h)?;

    // 7+8. Fused residual + FFN norm
    let mut ffn_normed = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    let mut residual = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    ops::fused_residual_rmsnorm(cache, device, &attn_projected, x, &weights.ffn_norm,
        &mut ffn_normed, &mut residual, h, config.norm_eps)?;

    // 9-10. Gate + Up + Fused SwiGLU + Down
    let mut gate = GpuTensor::<f32>::zeros(device, shape_bf.clone(), warp_ir::DType::F32)?;
    let mut up = GpuTensor::<f32>::zeros(device, shape_bf.clone(), warp_ir::DType::F32)?;

    if let Some(ref w_gate_up) = weights.w_gate_up {
        // ── Fused gate+up path: ONE GEMM instead of TWO ──
        // w_gate_up is [hidden_size, 2*ffn_dim]
        let gu_dim = ffn * 2;
        let shape_gu = warp_ir::Shape::from_static(&[batch as usize, gu_dim as usize]);
        let mut gate_up = GpuTensor::<f32>::zeros(device, shape_gu, warp_ir::DType::F32)?;
        ops::gemm(cache, device, &ffn_normed, w_gate_up, &mut gate_up, bn, gu_dim, h)?;

        // Split gate+up on GPU
        ops::split_gate_up(cache, device, &gate_up, &mut gate, &mut up, ffn, bn)?;
    } else {
        // ── Unfused fallback: 2 separate GEMMs ──
        ops::gemm(cache, device, &ffn_normed, &weights.w_gate, &mut gate, bn, ffn, h)?;
        ops::gemm(cache, device, &ffn_normed, &weights.w_up, &mut up, bn, ffn, h)?;
    }

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
        bq: None,
        bk: None,
        bv: None,
        wqkv: None,
        bqkv: None,
        w_gate_up: None,
    })
}

/// Create random weights with fused QKV and gate+up for testing.
pub fn random_weights_fused(
    device: &WarpDevice,
    config: &TransformerConfig,
) -> Result<TransformerBlockWeights, DeviceError> {
    let mut weights = random_weights(device, config)?;
    weights.fuse_projections(device)?;
    Ok(weights)
}

impl TransformerBlockWeights {
    /// Fuse Q/K/V weight matrices into a single QKV matrix and gate/up into gate_up.
    /// This is called once at model load time — the fused weights are computed on CPU
    /// and uploaded to GPU. No runtime overhead.
    ///
    /// After fusion, the individual wq/wk/wv and w_gate/w_up are still present
    /// (for the prefill path or fallback), but the decode path uses the fused versions.
    pub fn fuse_projections(&mut self, device: &WarpDevice) -> Result<(), DeviceError> {
        // Fuse QKV: [K, Nq] + [K, Nk] + [K, Nv] → [K, Nq+Nk+Nv]
        self.wqkv = Some(ops::concat_weights_n(device, &[&self.wq, &self.wk, &self.wv])?);

        // Fuse QKV biases if all three exist
        if let (Some(ref bq), Some(ref bk), Some(ref bv)) = (&self.bq, &self.bk, &self.bv) {
            self.bqkv = Some(ops::concat_biases(device, &[bq, bk, bv])?);
        }

        // Fuse gate+up: [K, Ngate] + [K, Nup] → [K, Ngate+Nup]
        self.w_gate_up = Some(ops::concat_weights_n(device, &[&self.w_gate, &self.w_up])?);

        Ok(())
    }

    /// Returns total GPU memory used by this layer's weights (including fused).
    pub fn total_memory_bytes(&self) -> usize {
        let mut total = self.attn_norm.size_bytes()
            + self.wq.size_bytes()
            + self.wk.size_bytes()
            + self.wv.size_bytes()
            + self.wo.size_bytes()
            + self.ffn_norm.size_bytes()
            + self.w_gate.size_bytes()
            + self.w_up.size_bytes()
            + self.w_down.size_bytes();
        if let Some(ref wqkv) = self.wqkv { total += wqkv.size_bytes(); }
        if let Some(ref bqkv) = self.bqkv { total += bqkv.size_bytes(); }
        if let Some(ref w_gate_up) = self.w_gate_up { total += w_gate_up.size_bytes(); }
        if let Some(ref bq) = self.bq { total += bq.size_bytes(); }
        if let Some(ref bk) = self.bk { total += bk.size_bytes(); }
        if let Some(ref bv) = self.bv { total += bv.size_bytes(); }
        total
    }
}

// ═════════════════════════════════════════════════════════════════
// FP16 mixed-precision decode/prefill — cast-GEMM-cast pattern
// ═════════════════════════════════════════════════════════════════

/// Mixed-precision GEMM helper: cast F32 activations → F16, HGEMM with F16 weight, cast output → F32.
///
/// This is the core operation for bandwidth-bound decode: weight matrices stay in FP16
/// (half the bytes to read from VRAM) while activations enter and leave as F32.
pub fn gemm_f16_mixed(
    cache: &KernelCache,
    device: &WarpDevice,
    input_f32: &GpuTensor<f32>,
    weight_f16: &GpuTensor<half::f16>,
    output_f32: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
    // Pre-allocated F16 buffers (may be oversized — we only use the needed portion)
    input_f16_buf: &mut GpuTensor<half::f16>,
    output_f16_buf: &mut GpuTensor<half::f16>,
) -> Result<(), DeviceError> {
    // Create correctly-sized views by allocating small wrappers.
    // The underlying GPU memory comes from the pre-allocated buffers (no cudaMalloc).
    // We just need correctly-sized tensors for the cast kernels.
    let in_numel = (m * k) as usize;
    let out_numel = (m * n) as usize;

    // For the cast kernels, we need tensors with the right numel.
    // Since the pre-allocated buffers are oversized, we create temporary GpuTensor
    // wrappers pointing to the same data but with correct sizes.
    // SAFETY: We only read/write up to in_numel/out_numel elements, and the buffer
    // has at least that many elements allocated.

    // Cast input F32 → F16
    // Use input_f32.numel which is the correct count
    fp16::cast_f32_to_f16(cache, device, input_f32, input_f16_buf)?;

    // HGEMM: cuBLAS uses m/n/k params to determine how many elements to read/write,
    // not the tensor numel. So oversized buffers are fine for GEMM.
    cublas_gemm::gemm_cublas_f16(device, input_f16_buf, weight_f16, output_f16_buf, m, n, k)?;

    // Cast output F16 → F32
    // Problem: output_f16_buf.numel may be larger than output_f32.numel.
    // The cast kernel uses input.numel which would overflow output_f32.
    // Fix: temporarily set output_f16_buf.numel to match output_f32.numel.
    let saved_numel = output_f16_buf.numel;
    output_f16_buf.numel = out_numel;
    fp16::cast_f16_to_f32(cache, device, output_f16_buf, output_f32)?;
    output_f16_buf.numel = saved_numel;

    Ok(())
}

/// Prefill with FP16 mixed-precision weights — populates KV cache.
///
/// Same structure as `transformer_block_prefill` but all 7 projection GEMMs
/// use the cast-GEMM-cast pattern with FP16 weight matrices for 2x bandwidth savings.
/// Norms, RoPE, attention, residuals, and SwiGLU all stay F32.
pub fn transformer_block_prefill_f16_mixed(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    weights: &TransformerBlockWeightsF16,
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

    // Pre-allocate shared F16 buffers for cast-GEMM-cast calls
    let bn = bsz * n;
    let max_dim = h.max(ffn) as usize;
    let mut f16_in = GpuTensor::<half::f16>::zeros(device,
        warp_ir::Shape::from_static(&[bn as usize, max_dim]), warp_ir::DType::F16)?;
    let mut f16_out = GpuTensor::<half::f16>::zeros(device,
        warp_ir::Shape::from_static(&[bn as usize, max_dim]), warp_ir::DType::F16)?;

    // 1. Attention input norm (F32)
    let mut normed = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::rmsnorm(cache, device, x, &weights.attn_norm, &mut normed, h, config.norm_eps)?;

    // 2. Q, K, V projections — F16 mixed
    let mut q = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut k_proj = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;
    let mut v_proj = GpuTensor::<f32>::zeros(device, shape_bnk.clone(), warp_ir::DType::F32)?;

    gemm_f16_mixed(cache, device, &normed, &weights.wq, &mut q, bn, h, h, &mut f16_in, &mut f16_out)?;
    gemm_f16_mixed(cache, device, &normed, &weights.wk, &mut k_proj, bn, kv_dim, h, &mut f16_in, &mut f16_out)?;
    gemm_f16_mixed(cache, device, &normed, &weights.wv, &mut v_proj, bn, kv_dim, h, &mut f16_in, &mut f16_out)?;

    // 2b. Add biases if present (F32)
    if let Some(ref bq) = weights.bq {
        let mut qb = GpuTensor::<f32>::zeros(device, q.shape.clone(), warp_ir::DType::F32)?;
        ops::broadcast_add(cache, device, &q, bq, &mut qb)?;
        q = qb;
    }
    if let Some(ref bk) = weights.bk {
        let mut kb = GpuTensor::<f32>::zeros(device, k_proj.shape.clone(), warp_ir::DType::F32)?;
        ops::broadcast_add(cache, device, &k_proj, bk, &mut kb)?;
        k_proj = kb;
    }
    if let Some(ref bv) = weights.bv {
        let mut vb = GpuTensor::<f32>::zeros(device, v_proj.shape.clone(), warp_ir::DType::F32)?;
        ops::broadcast_add(cache, device, &v_proj, bv, &mut vb)?;
        v_proj = vb;
    }

    // 3. RoPE on Q and K
    let ns = n as usize;
    let nh = config.num_heads as usize;
    let nkv = config.num_kv_heads as usize;
    let hd = d as usize;

    let mut q_hf = GpuTensor::<f32>::zeros(device,
        warp_ir::Shape::from_static(&[nh, ns, hd]), warp_ir::DType::F32)?;
    ops::transpose_to_heads_first(cache, device, &q, &mut q_hf, config.num_heads, n, d)?;

    let mut k_hf = GpuTensor::<f32>::zeros(device,
        warp_ir::Shape::from_static(&[nkv, ns, hd]), warp_ir::DType::F32)?;
    ops::transpose_to_heads_first(cache, device, &k_proj, &mut k_hf, config.num_kv_heads, n, d)?;

    let mut q_rope = GpuTensor::<f32>::zeros(device,
        warp_ir::Shape::from_static(&[nh, ns, hd]), warp_ir::DType::F32)?;
    let mut k_rope = GpuTensor::<f32>::zeros(device,
        warp_ir::Shape::from_static(&[nkv, ns, hd]), warp_ir::DType::F32)?;
    crate::rope::rope(cache, device, &q_hf, &mut q_rope, bsz * config.num_heads, n, d, config.rope_base, pos_offset)?;
    crate::rope::rope(cache, device, &k_hf, &mut k_rope, bsz * config.num_kv_heads, n, d, config.rope_base, pos_offset)?;

    // 3.5. Store K/V in cache
    {
        let kv_d = nkv * hd;
        let mut k_for_cache = GpuTensor::<f32>::zeros(device,
            warp_ir::Shape::from_static(&[ns, kv_d]), warp_ir::DType::F32)?;
        ops::transpose_to_positions_first(cache, device, &k_rope, &mut k_for_cache, config.num_kv_heads, n, d)?;
        kv.prefill(cache, device, &k_for_cache, &v_proj, n)?;
    }

    // 4. Flash Attention (F32)
    let attn_batch = bsz * config.num_heads;
    let k_for_attn = repeat_kv(device, &k_rope, bsz * config.num_kv_heads, attn_batch, n, d)?;
    let mut v_hf = GpuTensor::<f32>::zeros(device,
        warp_ir::Shape::from_static(&[nkv, ns, hd]), warp_ir::DType::F32)?;
    ops::transpose_to_heads_first(cache, device, &v_proj, &mut v_hf, config.num_kv_heads, n, d)?;
    let v_for_attn = repeat_kv(device, &v_hf, bsz * config.num_kv_heads, attn_batch, n, d)?;

    let shape_attn = warp_ir::Shape::from_static(&[attn_batch as usize, n as usize, d as usize]);
    let mut attn_out = GpuTensor::<f32>::zeros(device, shape_attn, warp_ir::DType::F32)?;
    crate::attention_ext::attention_best(
        cache, device, &q_rope, &k_for_attn, &v_for_attn,
        &mut attn_out, attn_batch, n, d, true,
    )?;

    // 5. Output projection — F16 mixed
    let mut attn_out_pf = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::transpose_to_positions_first(cache, device, &attn_out, &mut attn_out_pf, config.num_heads, n, d)?;

    let mut attn_projected = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    gemm_f16_mixed(cache, device, &attn_out_pf, &weights.wo, &mut attn_projected, bn, h, h, &mut f16_in, &mut f16_out)?;

    // 6+7. Fused residual + FFN norm (F32)
    let mut ffn_normed = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    let mut residual1 = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    ops::fused_residual_rmsnorm(cache, device, &attn_projected, x, &weights.ffn_norm,
        &mut ffn_normed, &mut residual1, h, config.norm_eps)?;

    // 8. Gate + Up — F16 mixed
    let mut gate = GpuTensor::<f32>::zeros(device, shape_bnf.clone(), warp_ir::DType::F32)?;
    let mut up = GpuTensor::<f32>::zeros(device, shape_bnf.clone(), warp_ir::DType::F32)?;
    gemm_f16_mixed(cache, device, &ffn_normed, &weights.w_gate, &mut gate, bn, ffn, h, &mut f16_in, &mut f16_out)?;
    gemm_f16_mixed(cache, device, &ffn_normed, &weights.w_up, &mut up, bn, ffn, h, &mut f16_in, &mut f16_out)?;

    // 9. Fused SwiGLU (F32)
    let mut swiglu_out = GpuTensor::<f32>::zeros(device, shape_bnf, warp_ir::DType::F32)?;
    ops::fused_silu_mul(cache, device, &gate, &up, &mut swiglu_out)?;

    // 10. Down projection — F16 mixed
    let mut ffn_out = GpuTensor::<f32>::zeros(device, shape_bnh.clone(), warp_ir::DType::F32)?;
    gemm_f16_mixed(cache, device, &swiglu_out, &weights.w_down, &mut ffn_out, bn, h, ffn, &mut f16_in, &mut f16_out)?;

    // 11. Final residual (F32)
    let mut output = GpuTensor::<f32>::zeros(device, shape_bnh, warp_ir::DType::F32)?;
    ops::add(cache, device, &residual1, &ffn_out, &mut output)?;

    Ok(output)
}

/// Decode with FP16 mixed-precision weights — single token, uses KV cache.
///
/// This is the hot path for autoregressive generation. By storing weights in FP16,
/// each GEMM reads half the weight bytes from VRAM — giving ~2x decode speedup
/// on bandwidth-bound GPUs.
///
/// Pattern per GEMM: cast activation F32→F16, HGEMM with F16 weight, cast output F16→F32.
/// All non-GEMM operations (norms, RoPE, attention, residuals, SwiGLU) stay F32.
pub fn transformer_block_decode_f16_mixed(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    weights: &TransformerBlockWeightsF16,
    config: &TransformerConfig,
    kv: &mut LayerKVCache,
    batch: u32,
    pos: u32,
) -> Result<GpuTensor<f32>, DeviceError> {
    let max_dim = config.hidden_size.max(config.ffn_dim) as usize;
    let mut f16_in = GpuTensor::<half::f16>::zeros(device,
        warp_ir::Shape::from_static(&[batch as usize, max_dim]), warp_ir::DType::F16)?;
    let mut f16_out = GpuTensor::<half::f16>::zeros(device,
        warp_ir::Shape::from_static(&[batch as usize, max_dim]), warp_ir::DType::F16)?;
    transformer_block_decode_f16_mixed_prealloc(cache, device, x, weights, config, kv, batch, pos, &mut f16_in, &mut f16_out)
}

/// FP16 mixed-precision decode with pre-allocated F16 buffers (zero cudaMalloc).
pub fn transformer_block_decode_f16_mixed_prealloc(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    weights: &TransformerBlockWeightsF16,
    config: &TransformerConfig,
    kv: &mut LayerKVCache,
    batch: u32,
    pos: u32,
    mut f16_in: &mut GpuTensor<half::f16>,
    mut f16_out: &mut GpuTensor<half::f16>,
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

    // 1. RMSNorm (F32)
    let mut normed = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    ops::rmsnorm(cache, device, x, &weights.attn_norm, &mut normed, h, config.norm_eps)?;

    // 2. Q, K, V projections — F16 mixed (cast-HGEMM-cast) using shared F16 buffers
    let mut q = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    let mut new_k = GpuTensor::<f32>::zeros(device, shape_bk.clone(), warp_ir::DType::F32)?;
    let mut new_v = GpuTensor::<f32>::zeros(device, shape_bk.clone(), warp_ir::DType::F32)?;

    gemm_f16_mixed(cache, device, &normed, &weights.wq, &mut q, bn, h, h, &mut f16_in, &mut f16_out)?;
    gemm_f16_mixed(cache, device, &normed, &weights.wk, &mut new_k, bn, kv_dim, h, &mut f16_in, &mut f16_out)?;
    gemm_f16_mixed(cache, device, &normed, &weights.wv, &mut new_v, bn, kv_dim, h, &mut f16_in, &mut f16_out)?;

    // 2b. Add biases if present (F32)
    if let Some(ref bq) = weights.bq {
        let mut qb = GpuTensor::<f32>::zeros(device, q.shape.clone(), warp_ir::DType::F32)?;
        ops::broadcast_add(cache, device, &q, bq, &mut qb)?;
        q = qb;
    }
    if let Some(ref bk) = weights.bk {
        let mut kb = GpuTensor::<f32>::zeros(device, new_k.shape.clone(), warp_ir::DType::F32)?;
        ops::broadcast_add(cache, device, &new_k, bk, &mut kb)?;
        new_k = kb;
    }
    if let Some(ref bv) = weights.bv {
        let mut vb = GpuTensor::<f32>::zeros(device, new_v.shape.clone(), warp_ir::DType::F32)?;
        ops::broadcast_add(cache, device, &new_v, bv, &mut vb)?;
        new_v = vb;
    }

    // 3. RoPE (single position, F32)
    let mut q_rope = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    let mut k_rope = GpuTensor::<f32>::zeros(device, shape_bk.clone(), warp_ir::DType::F32)?;
    crate::rope::rope(cache, device, &q, &mut q_rope, batch * config.num_heads, 1, d, config.rope_base, pos)?;
    crate::rope::rope(cache, device, &new_k, &mut k_rope, batch * config.num_kv_heads, 1, d, config.rope_base, pos)?;

    // 4-5. Append to cache and run attention (F32), dispatching by attention mode.
    let mut attn_out = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;

    match &config.attention_mode {
        AttentionMode::Standard => {
            kv.append(cache, device, &k_rope, &new_v)?;
            crate::kv_cache::decode_attention_multihead(
                cache, device, &q_rope, kv, &mut attn_out,
                config.num_heads, config.num_kv_heads, d,
            )?;
        }
        AttentionMode::PagedAttention => {
            kv.append(cache, device, &k_rope, &new_v)?;
            crate::kv_cache::decode_attention(cache, device, &q_rope, kv, &mut attn_out, d)?;
        }
        AttentionMode::SlidingWindow { window_size } => {
            kv.append(cache, device, &k_rope, &new_v)?;
            let cache_len = kv.len;
            crate::sliding_window::sliding_window_decode(
                cache, device,
                &q_rope,
                &kv.k,
                &kv.v,
                &mut attn_out,
                cache_len,
                d,
                *window_size,
            )?;
        }
    }

    // 6. Output projection — F16 mixed
    let mut attn_projected = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    gemm_f16_mixed(cache, device, &attn_out, &weights.wo, &mut attn_projected, bn, h, h, &mut f16_in, &mut f16_out)?;

    // 7+8. Fused residual + FFN norm (F32)
    let mut ffn_normed = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    let mut residual = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    ops::fused_residual_rmsnorm(cache, device, &attn_projected, x, &weights.ffn_norm,
        &mut ffn_normed, &mut residual, h, config.norm_eps)?;

    // 9-10. Gate + Up + Fused SwiGLU + Down — F16 mixed
    let mut gate = GpuTensor::<f32>::zeros(device, shape_bf.clone(), warp_ir::DType::F32)?;
    let mut up = GpuTensor::<f32>::zeros(device, shape_bf.clone(), warp_ir::DType::F32)?;
    gemm_f16_mixed(cache, device, &ffn_normed, &weights.w_gate, &mut gate, bn, ffn, h, &mut f16_in, &mut f16_out)?;
    gemm_f16_mixed(cache, device, &ffn_normed, &weights.w_up, &mut up, bn, ffn, h, &mut f16_in, &mut f16_out)?;

    // Fused SwiGLU (F32)
    let mut swiglu = GpuTensor::<f32>::zeros(device, shape_bf, warp_ir::DType::F32)?;
    ops::fused_silu_mul(cache, device, &gate, &up, &mut swiglu)?;

    let mut ffn_out = GpuTensor::<f32>::zeros(device, shape_bh.clone(), warp_ir::DType::F32)?;
    gemm_f16_mixed(cache, device, &swiglu, &weights.w_down, &mut ffn_out, bn, h, ffn, &mut f16_in, &mut f16_out)?;

    // 11. Residual (F32)
    let mut output = GpuTensor::<f32>::zeros(device, shape_bh, warp_ir::DType::F32)?;
    ops::add(cache, device, &residual, &ffn_out, &mut output)?;

    Ok(output)
}

// ═══════════════════════════════════════════════════════════════
// Full FP16 decode — NO casting. Hidden state stays FP16 throughout.
// Only cast F32→F16 once (from embedding) and F16→F32 once (for LM head argmax).
// This eliminates 14 cast kernels per layer = 336 fewer launches for 24 layers.
// ═══════════════════════════════════════════════════════════════

/// Full FP16 transformer decode — hidden state stays FP16 across all operations.
/// Norms compute internally in F32 for stability but input/output FP16.
/// GEMMs use cuBLAS HGEMM (FP16 × FP16 → FP16 with F32 accumulate).
/// Attention uses F32 accumulation internally but FP16 I/O.
pub fn transformer_block_decode_full_f16(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<half::f16>,                // [1, H] FP16 input
    weights: &TransformerBlockWeightsF16,
    config: &TransformerConfig,
    kv: &mut crate::kv_cache::LayerKVCacheF16,
    batch: u32,
    pos: u32,
) -> Result<GpuTensor<half::f16>, DeviceError> {
    let h = config.hidden_size;
    let d = config.head_dim;
    let kv_dim = config.kv_dim();
    let ffn = config.ffn_dim;
    let bn = batch;

    let shape_bh = warp_ir::Shape::from_static(&[batch as usize, h as usize]);
    let shape_bk = warp_ir::Shape::from_static(&[batch as usize, kv_dim as usize]);
    let shape_bf = warp_ir::Shape::from_static(&[batch as usize, ffn as usize]);

    // 1. RMSNorm (FP16 in → F32 internal → FP16 out)
    let mut normed = GpuTensor::<half::f16>::zeros(device, shape_bh.clone(), warp_ir::DType::F16)?;
    fp16::f16_rmsnorm(cache, device, x, &weights.attn_norm_f16(), &mut normed, h, config.norm_eps)?;

    // 2. QKV GEMMs (FP16 × FP16 → FP16, tensor cores)
    let mut q = GpuTensor::<half::f16>::zeros(device, shape_bh.clone(), warp_ir::DType::F16)?;
    let mut new_k = GpuTensor::<half::f16>::zeros(device, shape_bk.clone(), warp_ir::DType::F16)?;
    let mut new_v = GpuTensor::<half::f16>::zeros(device, shape_bk.clone(), warp_ir::DType::F16)?;

    cublas_gemm::gemm_cublas_f16(device, &normed, &weights.wq, &mut q, bn, h, h)?;
    cublas_gemm::gemm_cublas_f16(device, &normed, &weights.wk, &mut new_k, bn, kv_dim, h)?;
    cublas_gemm::gemm_cublas_f16(device, &normed, &weights.wv, &mut new_v, bn, kv_dim, h)?;

    // 2b. Biases (F32 bias added after casting — small overhead, maintains precision)
    if let Some(ref bq) = weights.bq {
        // For now, cast bias to F16 and use f16_add. Could pre-store F16 biases.
        let mut bq_f16 = GpuTensor::<half::f16>::zeros(device, bq.shape.clone(), warp_ir::DType::F16)?;
        fp16::cast_f32_to_f16(cache, device, bq, &mut bq_f16)?;
        let mut qb = GpuTensor::<half::f16>::zeros(device, q.shape.clone(), warp_ir::DType::F16)?;
        fp16::f16_add(cache, device, &q, &bq_f16, &mut qb)?;
        q = qb;
    }
    if let Some(ref bk) = weights.bk {
        let mut bk_f16 = GpuTensor::<half::f16>::zeros(device, bk.shape.clone(), warp_ir::DType::F16)?;
        fp16::cast_f32_to_f16(cache, device, bk, &mut bk_f16)?;
        let mut kb = GpuTensor::<half::f16>::zeros(device, new_k.shape.clone(), warp_ir::DType::F16)?;
        fp16::f16_add(cache, device, &new_k, &bk_f16, &mut kb)?;
        new_k = kb;
    }
    if let Some(ref bv) = weights.bv {
        let mut bv_f16 = GpuTensor::<half::f16>::zeros(device, bv.shape.clone(), warp_ir::DType::F16)?;
        fp16::cast_f32_to_f16(cache, device, bv, &mut bv_f16)?;
        let mut vb = GpuTensor::<half::f16>::zeros(device, new_v.shape.clone(), warp_ir::DType::F16)?;
        fp16::f16_add(cache, device, &new_v, &bv_f16, &mut vb)?;
        new_v = vb;
    }

    // 3. RoPE (FP16)
    let mut q_rope = GpuTensor::<half::f16>::zeros(device, shape_bh.clone(), warp_ir::DType::F16)?;
    let mut k_rope = GpuTensor::<half::f16>::zeros(device, shape_bk.clone(), warp_ir::DType::F16)?;
    fp16::f16_rope(cache, device, &q, &mut q_rope, batch * config.num_heads, 1, d, config.rope_base, pos)?;
    fp16::f16_rope(cache, device, &new_k, &mut k_rope, batch * config.num_kv_heads, 1, d, config.rope_base, pos)?;

    // 4. KV cache append + attention (FP16)
    kv.append(cache, device, &k_rope, &new_v)?;
    let mut attn_out = GpuTensor::<half::f16>::zeros(device, shape_bh.clone(), warp_ir::DType::F16)?;
    crate::kv_cache::decode_attention_multihead_f16(
        cache, device, &q_rope, kv, &mut attn_out,
        config.num_heads, config.num_kv_heads, d,
    )?;

    // 5. Output projection (FP16 HGEMM)
    let mut attn_proj = GpuTensor::<half::f16>::zeros(device, shape_bh.clone(), warp_ir::DType::F16)?;
    cublas_gemm::gemm_cublas_f16(device, &attn_out, &weights.wo, &mut attn_proj, bn, h, h)?;

    // 6+7. Fused residual + FFN norm (FP16 with F32 internal)
    let mut ffn_normed = GpuTensor::<half::f16>::zeros(device, shape_bh.clone(), warp_ir::DType::F16)?;
    let mut residual = GpuTensor::<half::f16>::zeros(device, shape_bh.clone(), warp_ir::DType::F16)?;
    fp16::f16_fused_residual_rmsnorm(cache, device, &attn_proj, x, &weights.ffn_norm_f16(),
        &mut ffn_normed, &mut residual, h, config.norm_eps)?;

    // 8-9. Gate + Up + SwiGLU (FP16)
    let mut gate = GpuTensor::<half::f16>::zeros(device, shape_bf.clone(), warp_ir::DType::F16)?;
    let mut up = GpuTensor::<half::f16>::zeros(device, shape_bf.clone(), warp_ir::DType::F16)?;
    cublas_gemm::gemm_cublas_f16(device, &ffn_normed, &weights.w_gate, &mut gate, bn, ffn, h)?;
    cublas_gemm::gemm_cublas_f16(device, &ffn_normed, &weights.w_up, &mut up, bn, ffn, h)?;

    let mut swiglu = GpuTensor::<half::f16>::zeros(device, shape_bf, warp_ir::DType::F16)?;
    fp16::f16_fused_silu_mul(cache, device, &gate, &up, &mut swiglu)?;

    // 10. Down projection (FP16 HGEMM)
    let mut ffn_out = GpuTensor::<half::f16>::zeros(device, shape_bh.clone(), warp_ir::DType::F16)?;
    cublas_gemm::gemm_cublas_f16(device, &swiglu, &weights.w_down, &mut ffn_out, bn, h, ffn)?;

    // 11. Final residual (FP16)
    let mut output = GpuTensor::<half::f16>::zeros(device, shape_bh, warp_ir::DType::F16)?;
    fp16::f16_add(cache, device, &residual, &ffn_out, &mut output)?;

    Ok(output)
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

    #[test]
    fn fused_qkv_decode_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let config = TransformerConfig::tiny();
        let h = config.hidden_size;
        let kv_dim = config.kv_dim();

        // Build unfused weights
        let unfused = random_weights(&dev, &config).unwrap();

        // Build fused weights from the SAME data
        let mut fused = TransformerBlockWeights {
            attn_norm: GpuTensor::from_host(&dev, &unfused.attn_norm.to_host(&dev).unwrap(), unfused.attn_norm.shape.clone(), DType::F32).unwrap(),
            wq: GpuTensor::from_host(&dev, &unfused.wq.to_host(&dev).unwrap(), unfused.wq.shape.clone(), DType::F32).unwrap(),
            wk: GpuTensor::from_host(&dev, &unfused.wk.to_host(&dev).unwrap(), unfused.wk.shape.clone(), DType::F32).unwrap(),
            wv: GpuTensor::from_host(&dev, &unfused.wv.to_host(&dev).unwrap(), unfused.wv.shape.clone(), DType::F32).unwrap(),
            wo: GpuTensor::from_host(&dev, &unfused.wo.to_host(&dev).unwrap(), unfused.wo.shape.clone(), DType::F32).unwrap(),
            ffn_norm: GpuTensor::from_host(&dev, &unfused.ffn_norm.to_host(&dev).unwrap(), unfused.ffn_norm.shape.clone(), DType::F32).unwrap(),
            w_gate: GpuTensor::from_host(&dev, &unfused.w_gate.to_host(&dev).unwrap(), unfused.w_gate.shape.clone(), DType::F32).unwrap(),
            w_up: GpuTensor::from_host(&dev, &unfused.w_up.to_host(&dev).unwrap(), unfused.w_up.shape.clone(), DType::F32).unwrap(),
            w_down: GpuTensor::from_host(&dev, &unfused.w_down.to_host(&dev).unwrap(), unfused.w_down.shape.clone(), DType::F32).unwrap(),
            bq: None, bk: None, bv: None,
            wqkv: None, bqkv: None, w_gate_up: None,
        };
        fused.fuse_projections(&dev).unwrap();
        assert!(fused.wqkv.is_some(), "wqkv should be Some after fuse");
        assert!(fused.w_gate_up.is_some(), "w_gate_up should be Some after fuse");

        // Run decode with both and compare
        let batch = 1u32;
        let shape_bh = Shape::from_static(&[batch as usize, h as usize]);
        let x_data: Vec<f32> = (0..h as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let x1 = GpuTensor::from_host(&dev, &x_data, shape_bh.clone(), DType::F32).unwrap();
        let x2 = GpuTensor::from_host(&dev, &x_data, shape_bh, DType::F32).unwrap();

        let mut kv1 = crate::kv_cache::LayerKVCache::new(&dev, 64, kv_dim).unwrap();
        let mut kv2 = crate::kv_cache::LayerKVCache::new(&dev, 64, kv_dim).unwrap();

        let out_unfused = transformer_block_decode(&cache, &dev, &x1, &unfused, &config, &mut kv1, batch, 0).unwrap();
        let out_fused = transformer_block_decode(&cache, &dev, &x2, &fused, &config, &mut kv2, batch, 0).unwrap();
        dev.synchronize().unwrap();

        let r1 = out_unfused.to_host(&dev).unwrap();
        let r2 = out_fused.to_host(&dev).unwrap();
        assert_eq!(r1.len(), r2.len());

        let max_diff: f32 = r1.iter().zip(r2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("Fused vs unfused decode max diff: {max_diff:.2e}");
        assert!(max_diff < 1e-3, "Fused QKV decode diverged from unfused: max_diff={max_diff}");
        println!("Fused QKV + gate+up decode: PASS (max diff = {max_diff:.2e})");
    }

    #[test]
    fn fused_decode_perf() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // Use a Qwen-like config: H=896, FFN=4864, 14 heads, 2 KV heads
        let config = TransformerConfig {
            hidden_size: 896,
            num_heads: 14,
            num_kv_heads: 2,
            head_dim: 64,
            ffn_dim: 4864,
            rope_base: 1000000.0,
            norm_eps: 1e-6,
            attention_mode: AttentionMode::Standard,
        };
        let kv_dim = config.kv_dim();

        let unfused = random_weights(&dev, &config).unwrap();
        let mut fused_w = random_weights(&dev, &config).unwrap();
        fused_w.fuse_projections(&dev).unwrap();

        let batch = 1u32;
        let h = config.hidden_size;
        let shape_bh = Shape::from_static(&[batch as usize, h as usize]);
        let x_data: Vec<f32> = (0..h as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let x = GpuTensor::from_host(&dev, &x_data, shape_bh, DType::F32).unwrap();

        let iters = 100;

        // Warmup + bench unfused
        {
            let mut kv = crate::kv_cache::LayerKVCache::new(&dev, 256, kv_dim).unwrap();
            let _ = transformer_block_decode(&cache, &dev, &x, &unfused, &config, &mut kv, batch, 0).unwrap();
            dev.synchronize().unwrap();

            let mut kv = crate::kv_cache::LayerKVCache::new(&dev, 256, kv_dim).unwrap();
            let start = std::time::Instant::now();
            for i in 0..iters {
                let _ = transformer_block_decode(&cache, &dev, &x, &unfused, &config, &mut kv, batch, i as u32).unwrap();
            }
            dev.synchronize().unwrap();
            let unfused_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let mut kv = crate::kv_cache::LayerKVCache::new(&dev, 256, kv_dim).unwrap();
            let start = std::time::Instant::now();
            for i in 0..iters {
                let _ = transformer_block_decode(&cache, &dev, &x, &fused_w, &config, &mut kv, batch, i as u32).unwrap();
            }
            dev.synchronize().unwrap();
            let fused_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let speedup = unfused_ms / fused_ms;
            println!("\n=== Fused QKV + Gate+Up Decode Perf (Qwen-like H={h} FFN={}) ===", config.ffn_dim);
            println!("  Unfused: {unfused_ms:.3}ms/token");
            println!("  Fused:   {fused_ms:.3}ms/token");
            println!("  Speedup: {speedup:.2}x");
        }
    }
}

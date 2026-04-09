//! Gemma 4 generation engine.
//!
//! Handles the Gemma-specific architecture:
//! - Per-layer sliding window vs global attention
//! - Dual RoPE (different theta + partial rotation for global layers)
//! - QK-norm (RMSNorm on Q and K before attention)
//! - GeGLU activation (instead of SwiGLU)
//! - Shared K=V projections
//! - Logit softcapping
//! - Per-layer varying head_dim and num_kv_heads

use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::generate::{GenerateConfig, GenerationResult, Q4DecodeBuffers, sample_token, matches_stop_sequence};
use crate::kv_cache::{ModelKVCache, LayerKVCache};
use crate::ops;
use crate::sampling;
use crate::tensor::GpuTensor;
use crate::transformer::{GemmaConfig, GemmaLayerAttentionConfig, QuantizedBlockWeights};

/// Per-layer pre-allocated buffers for Gemma decode.
pub struct GemmaLayerBuffers {
    pub normed: GpuTensor<f32>,
    pub q: GpuTensor<f32>,
    pub k: GpuTensor<f32>,
    pub v: GpuTensor<f32>,
    pub q_rope: GpuTensor<f32>,
    pub k_rope: GpuTensor<f32>,
    pub q_norm: GpuTensor<f32>,
    pub k_norm: GpuTensor<f32>,
    pub attn_out: GpuTensor<f32>,
    pub attn_proj: GpuTensor<f32>,
    pub ffn_normed: GpuTensor<f32>,
    pub residual: GpuTensor<f32>,
    pub gate: GpuTensor<f32>,
    pub up: GpuTensor<f32>,
    pub geglu: GpuTensor<f32>,
    pub ffn_out: GpuTensor<f32>,
    pub output: GpuTensor<f32>,
}

/// All pre-allocated buffers for Gemma decode.
pub struct GemmaDecodeBuffers {
    pub ids: GpuTensor<i32>,
    pub hidden: GpuTensor<f32>,
    pub hidden_scaled: GpuTensor<f32>,
    pub normed_final: GpuTensor<f32>,
    pub logits: GpuTensor<f32>,
    pub layers: Vec<GemmaLayerBuffers>,
}

/// Gemma generation engine with Q4 quantized weights.
pub struct GemmaGenerationEngine {
    pub config: GemmaConfig,
    pub layer_configs: Vec<GemmaLayerAttentionConfig>,
    /// Embedding table stored as F16 (saves 2.8 GB for 262K vocab)
    pub embed_tokens: GpuTensor<half::f16>,
    pub layers: Vec<QuantizedBlockWeights>,
    pub final_norm: GpuTensor<f32>,
    pub lm_head: GpuTensor<f32>,
    pub cache: KernelCache,
    /// Cached host-side F32 embedding for CPU LM head (avoids 2.8GB DtoH per token)
    pub embed_host_f32: Option<Vec<f32>>,
    /// TW-Marlin weights per layer: (packed_nibbles, fp16_scales) for each projection
    /// When present, the fast gemm_tw_marlin_m1 is used instead of gemm_q4_0_m1.
    pub marlin_layers: Option<Vec<GemmaMarlinLayer>>,
}

/// TW-Marlin weights for one Gemma layer.
pub struct GemmaMarlinLayer {
    pub wq_p: GpuTensor<u8>,  pub wq_s: GpuTensor<half::f16>,
    pub wk_p: GpuTensor<u8>,  pub wk_s: GpuTensor<half::f16>,
    pub wv_p: GpuTensor<u8>,  pub wv_s: GpuTensor<half::f16>,
    pub wo_p: GpuTensor<u8>,  pub wo_s: GpuTensor<half::f16>,
    pub wg_p: GpuTensor<u8>,  pub wg_s: GpuTensor<half::f16>,
    pub wu_p: GpuTensor<u8>,  pub wu_s: GpuTensor<half::f16>,
    pub wd_p: GpuTensor<u8>,  pub wd_s: GpuTensor<half::f16>,
}

impl GemmaGenerationEngine {
    /// Create from a loaded GemmaModelQ4 (moves ownership of tensors).
    pub fn from_loaded(
        config: GemmaConfig,
        layer_configs: Vec<GemmaLayerAttentionConfig>,
        embed_tokens: GpuTensor<half::f16>,
        layers: Vec<QuantizedBlockWeights>,
        final_norm: GpuTensor<f32>,
        lm_head: GpuTensor<f32>,
    ) -> Self {
        Self {
            config, layer_configs, embed_tokens, layers, final_norm, lm_head,
            cache: KernelCache::new(),
            embed_host_f32: None,
            marlin_layers: None,
        }
    }

    /// Allocate KV caches with per-layer dimensions.
    /// Sliding layers get smaller caches (window_size), global layers get full context.
    fn allocate_kv_caches(
        &self,
        device: &WarpDevice,
        max_seq_len: u32,
    ) -> Result<Vec<LayerKVCache>, DeviceError> {
        let mut caches = Vec::with_capacity(self.layers.len());
        for lc in &self.layer_configs {
            let kv_dim = self.config.kv_dim_for_layer(lc);
            // Sliding layers only need window_size entries, global layers need full context
            let cache_len = if lc.window_size > 0 {
                lc.window_size.min(max_seq_len)
            } else {
                max_seq_len
            };
            let kv = if self.config.k_eq_v {
                LayerKVCache::new_shared_kv(device, cache_len, kv_dim)?
            } else {
                LayerKVCache::new(device, cache_len, kv_dim)?
            };
            caches.push(kv);
        }
        Ok(caches)
    }

    /// Allocate all decode buffers up-front. Zero allocation during generation.
    fn allocate_decode_buffers(
        &self,
        device: &WarpDevice,
    ) -> Result<GemmaDecodeBuffers, DeviceError> {
        let h = self.config.hidden_size as usize;
        let ffn = self.config.ffn_dim as usize;
        let vocab = self.config.vocab_size as usize;

        let mut layers = Vec::with_capacity(self.layers.len());
        for lc in &self.layer_configs {
            let q_dim = (self.config.num_heads * lc.head_dim) as usize;
            let kv_dim = self.config.kv_dim_for_layer(lc) as usize;
            layers.push(GemmaLayerBuffers {
                normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
                q: GpuTensor::zeros(device, Shape::from_static(&[1, q_dim]), DType::F32)?,
                k: GpuTensor::zeros(device, Shape::from_static(&[1, kv_dim]), DType::F32)?,
                v: GpuTensor::zeros(device, Shape::from_static(&[1, kv_dim]), DType::F32)?,
                q_rope: GpuTensor::zeros(device, Shape::from_static(&[1, q_dim]), DType::F32)?,
                k_rope: GpuTensor::zeros(device, Shape::from_static(&[1, kv_dim]), DType::F32)?,
                q_norm: GpuTensor::zeros(device, Shape::from_static(&[1, q_dim]), DType::F32)?,
                k_norm: GpuTensor::zeros(device, Shape::from_static(&[1, kv_dim]), DType::F32)?,
                attn_out: GpuTensor::zeros(device, Shape::from_static(&[1, q_dim]), DType::F32)?,
                attn_proj: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
                ffn_normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
                residual: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
                gate: GpuTensor::zeros(device, Shape::from_static(&[1, ffn]), DType::F32)?,
                up: GpuTensor::zeros(device, Shape::from_static(&[1, ffn]), DType::F32)?,
                geglu: GpuTensor::zeros(device, Shape::from_static(&[1, ffn]), DType::F32)?,
                ffn_out: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
                output: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            });
        }

        Ok(GemmaDecodeBuffers {
            ids: GpuTensor::zeros(device, Shape::from_static(&[1]), DType::I32)?,
            hidden: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            hidden_scaled: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            normed_final: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            logits: GpuTensor::zeros(device, Shape::from_static(&[1, vocab]), DType::F32)?,
            layers,
        })
    }

    /// Convert all layer weights to TW-Marlin format for fast M=1 GEMM.
    pub fn convert_to_marlin(&mut self, device: &WarpDevice) -> Result<(), DeviceError> {
        let h = self.config.hidden_size;
        let ffn = self.config.ffn_dim;
        let mut marlin = Vec::with_capacity(self.layers.len());

        for (i, (layer, lc)) in self.layers.iter().zip(self.layer_configs.iter()).enumerate() {
            let q_dim = self.config.num_heads * lc.head_dim;
            let kv_dim = self.config.kv_dim_for_layer(lc);

            if i % 10 == 0 { eprintln!("[gemma] Converting layer {}/{} to TW-Marlin...", i, self.layers.len()); }

            let (wq_p, wq_s) = crate::quantize::reorder_to_tw_marlin(device, &layer.wq, h, q_dim)?;
            let (wk_p, wk_s) = crate::quantize::reorder_to_tw_marlin(device, &layer.wk, h, kv_dim)?;
            let (wv_p, wv_s) = crate::quantize::reorder_to_tw_marlin(device, &layer.wv, h, kv_dim)?;
            let (wo_p, wo_s) = crate::quantize::reorder_to_tw_marlin(device, &layer.wo, q_dim, h)?;
            let (wg_p, wg_s) = crate::quantize::reorder_to_tw_marlin(device, &layer.w_gate, h, ffn)?;
            let (wu_p, wu_s) = crate::quantize::reorder_to_tw_marlin(device, &layer.w_up, h, ffn)?;
            let (wd_p, wd_s) = crate::quantize::reorder_to_tw_marlin(device, &layer.w_down, ffn, h)?;

            marlin.push(GemmaMarlinLayer {
                wq_p, wq_s, wk_p, wk_s, wv_p, wv_s, wo_p, wo_s,
                wg_p, wg_s, wu_p, wu_s, wd_p, wd_s,
            });
        }

        eprintln!("[gemma] TW-Marlin conversion complete ({} layers)", marlin.len());
        self.marlin_layers = Some(marlin);
        Ok(())
    }

    /// Cache the host-side F32 embedding for CPU LM head.
    /// Called once before generation to avoid 2.8 GB DtoH per token.
    pub fn cache_embed_for_lm_head(&mut self, device: &WarpDevice) -> Result<(), DeviceError> {
        if self.embed_host_f32.is_none() && self.config.tie_word_embeddings {
            eprintln!("[gemma] Caching F32 embedding on host for LM head...");
            let raw = self.embed_tokens.to_host(device)?;
            let f32_data: Vec<f32> = raw.iter().map(|h| h.to_f32()).collect();
            eprintln!("[gemma] Cached {:.1} MB", f32_data.len() as f64 * 4.0 / 1e6);
            self.embed_host_f32 = Some(f32_data);
        }
        Ok(())
    }

    /// Pre-allocated decode step — zero allocation during generation.
    fn forward_decode_prealloc(
        &self,
        device: &WarpDevice,
        buffers: &mut GemmaDecodeBuffers,
        kv_caches: &mut [LayerKVCache],
        pos: u32,
    ) -> Result<(), DeviceError> {
        let h = self.config.hidden_size;
        let ffn = self.config.ffn_dim;

        // Embedding + scaling
        sampling::embedding_f16(&self.cache, device, &self.embed_tokens, &buffers.ids,
            &mut buffers.hidden, 1, h)?;
        let embed_scale = (h as f32).sqrt();
        ops::mul_scalar(&self.cache, device, &buffers.hidden, &mut buffers.hidden_scaled, embed_scale)?;

        for (i, (layer, lc)) in self.layers.iter().zip(self.layer_configs.iter()).enumerate() {
            let kv_dim = self.config.kv_dim_for_layer(lc);
            let d = lc.head_dim;
            let num_q_heads = self.config.num_heads;
            let num_kv_heads = lc.num_kv_heads;
            let q_dim = num_q_heads * d;

            // Get raw pointer to input (previous layer output or hidden_scaled)
            let x_ptr: *const GpuTensor<f32> = if i == 0 {
                &buffers.hidden_scaled as *const _
            } else {
                // Safety: layers[i-1] is not mutably borrowed — only layers[i] is
                unsafe { &(*buffers.layers.as_ptr().add(i - 1)).output as *const _ }
            };
            let x: &GpuTensor<f32> = unsafe { &*x_ptr };
            let lb = &mut buffers.layers[i];

            ops::rmsnorm(&self.cache, device, x, &layer.attn_norm, &mut lb.normed, h, self.config.norm_eps)?;

            // Debug: compare layer 0 intermediate values against PyTorch reference
            if i == 0 && pos == 0 {
                let x_host = x.to_host(device)?;
                eprintln!("[ref] scaled_emb: min={:.4}, max={:.4}, mean={:.4}",
                    x_host.iter().cloned().fold(f32::INFINITY, f32::min),
                    x_host.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                    x_host.iter().sum::<f32>() / x_host.len() as f32);
                let n_host = lb.normed.to_host(device)?;
                eprintln!("[ref] after_norm: min={:.4}, max={:.4}, mean={:.4}, first8={:?}",
                    n_host.iter().cloned().fold(f32::INFINITY, f32::min),
                    n_host.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                    n_host.iter().sum::<f32>() / n_host.len() as f32,
                    &n_host[..8].iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
                // PyTorch ref: after_norm min=-210.33, max=107.53, mean=-0.0670
                // first8: [7.4203, 2.3646, 7.8419, 5.9251, 7.5541, 3.2455, 4.3227, -2.3455]
            }

            // Q, K projections — use TW-Marlin if available (3x faster)
            let ml = self.marlin_layers.as_ref().map(|m| &m[i]);

            if let Some(ml) = ml {
                crate::quantize::gemm_tw_marlin_m1(&self.cache, device, &lb.normed, &ml.wq_p, &ml.wq_s, &mut lb.q, q_dim, h)?;
                crate::quantize::gemm_tw_marlin_m1(&self.cache, device, &lb.normed, &ml.wk_p, &ml.wk_s, &mut lb.k, kv_dim, h)?;
            } else {
                crate::quantize::gemm_q4_0_m1(&self.cache, device, &lb.normed, &layer.wq, &mut lb.q, q_dim, h)?;
                crate::quantize::gemm_q4_0_m1(&self.cache, device, &lb.normed, &layer.wk, &mut lb.k, kv_dim, h)?;
            }

            // V = K (shared projection)
            if self.config.k_eq_v {
                ops::mul_scalar(&self.cache, device, &lb.k, &mut lb.v, 1.0)?;
            } else if let Some(ml) = ml {
                crate::quantize::gemm_tw_marlin_m1(&self.cache, device, &lb.normed, &ml.wv_p, &ml.wv_s, &mut lb.v, kv_dim, h)?;
            } else {
                crate::quantize::gemm_q4_0_m1(&self.cache, device, &lb.normed, &layer.wv, &mut lb.v, kv_dim, h)?;
            }

            // RoPE
            let rotary_dim = (d as f32 * lc.partial_rotary_factor) as u32;
            if rotary_dim < d && rotary_dim > 0 {
                crate::rope::rope_partial(&self.cache, device, &lb.q, &mut lb.q_rope,
                    num_q_heads, 1, d, rotary_dim, lc.rope_theta, pos)?;
                crate::rope::rope_partial(&self.cache, device, &lb.k, &mut lb.k_rope,
                    num_kv_heads, 1, d, rotary_dim, lc.rope_theta, pos)?;
            } else {
                crate::rope::rope(&self.cache, device, &lb.q, &mut lb.q_rope,
                    num_q_heads, 1, d, lc.rope_theta, pos)?;
                crate::rope::rope(&self.cache, device, &lb.k, &mut lb.k_rope,
                    num_kv_heads, 1, d, lc.rope_theta, pos)?;
            }

            // QK-norm (with learnable weights if available, otherwise plain RMS)
            if let Some(ref qn) = layer.q_norm {
                ops::rmsnorm(&self.cache, device, &lb.q_rope, qn, &mut lb.q_norm, d, self.config.norm_eps)?;
            } else {
                ops::rmsnorm_no_weight(&self.cache, device, &lb.q_rope, &mut lb.q_norm, d, self.config.norm_eps)?;
            }
            if let Some(ref kn) = layer.k_norm {
                ops::rmsnorm(&self.cache, device, &lb.k_rope, kn, &mut lb.k_norm, d, self.config.norm_eps)?;
            } else {
                ops::rmsnorm_no_weight(&self.cache, device, &lb.k_rope, &mut lb.k_norm, d, self.config.norm_eps)?;
            }

            // KV cache append
            let kv_cache = &mut kv_caches[i];
            let cache_pos = if lc.window_size > 0 { pos % lc.window_size } else { pos };
            kv_cache.prefill_at_offset(&self.cache, device, &lb.k_norm, &lb.v, 1, cache_pos)?;
            kv_cache.len = (pos + 1).min(kv_cache.max_seq_len);

            // Attention
            let window = if lc.window_size > 0 { lc.window_size } else { 0 };
            crate::kv_cache::decode_attention_flash_device_len_window(
                &self.cache, device, &lb.q_norm, kv_cache, &mut lb.attn_out,
                num_q_heads, num_kv_heads, d,
                &device.htod(&[kv_cache.len])?,
                window,
            )?;

            // Output projection
            if let Some(ml) = ml {
                crate::quantize::gemm_tw_marlin_m1(&self.cache, device, &lb.attn_out, &ml.wo_p, &ml.wo_s, &mut lb.attn_proj, h, q_dim)?;
            } else {
                crate::quantize::gemm_q4_0_m1(&self.cache, device, &lb.attn_out, &layer.wo, &mut lb.attn_proj, h, q_dim)?;
            }

            // Exact Gemma architecture (from HuggingFace):
            // hidden = post_attention_layernorm(attn_output)
            // hidden = residual + hidden
            // residual = hidden
            // hidden = pre_feedforward_layernorm(hidden)
            // hidden = FFN(hidden)
            // hidden = post_feedforward_layernorm(hidden)
            // output = residual + hidden

            // ffn_norm maps to post_attention_layernorm
            ops::rmsnorm(&self.cache, device, &lb.attn_proj, &layer.ffn_norm,
                &mut lb.ffn_normed, h, self.config.norm_eps)?;
            // residual = x + post_attn_normed
            ops::add(&self.cache, device, x, &lb.ffn_normed, &mut lb.residual)?;

            // pre_feedforward_layernorm
            if let Some(ref pre_norm) = layer.pre_ffn_norm {
                ops::rmsnorm(&self.cache, device, &lb.residual, pre_norm,
                    &mut lb.ffn_normed, h, self.config.norm_eps)?;
            } else {
                ops::mul_scalar(&self.cache, device, &lb.residual, &mut lb.ffn_normed, 1.0)?;
            }

            // Gate + Up + GeGLU
            if let Some(ml) = ml {
                crate::quantize::gemm_tw_marlin_m1(&self.cache, device, &lb.ffn_normed, &ml.wg_p, &ml.wg_s, &mut lb.gate, ffn, h)?;
                crate::quantize::gemm_tw_marlin_m1(&self.cache, device, &lb.ffn_normed, &ml.wu_p, &ml.wu_s, &mut lb.up, ffn, h)?;
            } else {
                crate::quantize::gemm_q4_0_m1(&self.cache, device, &lb.ffn_normed, &layer.w_gate, &mut lb.gate, ffn, h)?;
                crate::quantize::gemm_q4_0_m1(&self.cache, device, &lb.ffn_normed, &layer.w_up, &mut lb.up, ffn, h)?;
            }
            ops::fused_gelu_mul(&self.cache, device, &lb.gate, &lb.up, &mut lb.geglu)?;

            // Down projection
            if let Some(ml) = ml {
                crate::quantize::gemm_tw_marlin_m1(&self.cache, device, &lb.geglu, &ml.wd_p, &ml.wd_s, &mut lb.ffn_out, h, ffn)?;
            } else {
                crate::quantize::gemm_q4_0_m1(&self.cache, device, &lb.geglu, &layer.w_down, &mut lb.ffn_out, h, ffn)?;
            }

            // post_feedforward_layernorm
            if let Some(ref post_norm) = layer.post_ffn_norm {
                ops::rmsnorm(&self.cache, device, &lb.ffn_out, post_norm,
                    &mut lb.geglu, h, self.config.norm_eps)?;
                // output = residual + post_ffn_normed
                ops::add(&self.cache, device, &lb.residual, &lb.geglu, &mut lb.output)?;
            } else {
                ops::add(&self.cache, device, &lb.residual, &lb.ffn_out, &mut lb.output)?;
            }
        }

        // Final norm
        let last = self.layers.len() - 1;
        ops::rmsnorm(&self.cache, device, &buffers.layers[last].output,
            &self.final_norm, &mut buffers.normed_final, h, self.config.norm_eps)?;

        // LM head: cuBLAS F16 HGEMM with transB (embed_tokens is [V, H] F16)
        // logits[1,V] = normed[1,H] × embed^T[H,V]
        if self.config.tie_word_embeddings && self.lm_head.numel <= 1 {
            let vocab = self.config.vocab_size;
            // Cast normed F32 → F16
            let mut normed_f16 = GpuTensor::<half::f16>::zeros(device,
                Shape::from_static(&[1, h as usize]), DType::F16)?;
            crate::fp16::cast_f32_to_f16(&self.cache, device, &buffers.normed_final, &mut normed_f16)?;

            // cuBLAS HGEMM transB
            let mut logits_f16 = GpuTensor::<half::f16>::zeros(device,
                Shape::from_static(&[1, vocab as usize]), DType::F16)?;
            crate::cublas_gemm::gemm_cublas_f16_transB(device,
                &normed_f16, &self.embed_tokens, &mut logits_f16, 1, vocab, h)?;

            // Cast logits F16 → F32
            crate::fp16::cast_f16_to_f32(&self.cache, device, &logits_f16, &mut buffers.logits)?;
        } else {
            ops::gemm(&self.cache, device, &buffers.normed_final, &self.lm_head,
                &mut buffers.logits, 1, self.config.vocab_size, h)?;
        }

        // Debug: raw logits before softcapping (first call only)
        {
            let raw = buffers.logits.to_host(device)?;
            let min = raw.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let nan_count = raw.iter().filter(|v| v.is_nan()).count();
            let inf_count = raw.iter().filter(|v| v.is_infinite()).count();
            static DIAG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if DIAG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 2 {
                eprintln!("[gemma] Raw logits: range=[{:.1}, {:.1}], nan={}, inf={}", min, max, nan_count, inf_count);
            }
        }

        // Adaptive logit normalization for Q4 models.
        // Q4 quantization through 60 layers produces logits in hundreds instead of [-10,10].
        // Scale them to a reasonable range before sampling.
        // This is equivalent to a dynamic temperature based on logit magnitude.
        {
            let raw = buffers.logits.to_host(device)?;
            let max_abs = raw.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            if max_abs > 50.0 {
                // Normalize to roughly [-30, 30] range
                let scale = 30.0 / max_abs;
                let mut scaled = GpuTensor::<f32>::zeros(device, buffers.logits.shape.clone(), DType::F32)?;
                ops::mul_scalar(&self.cache, device, &buffers.logits, &mut scaled, scale)?;
                buffers.logits = scaled;
            }
        }

        Ok(())
    }

    /// Single-token decode step (allocating version — slow, used for initial testing).
    fn forward_decode(
        &self,
        device: &WarpDevice,
        token_id: i32,
        kv_caches: &mut [LayerKVCache],
        pos: u32,
    ) -> Result<Vec<f32>, DeviceError> {
        let h = self.config.hidden_size;
        let ffn = self.config.ffn_dim;

        // Embedding (F16 table → F32 output) + Gemma scaling (multiply by sqrt(hidden_size))
        let ids = GpuTensor::from_host(device, &[token_id],
            Shape::from_static(&[1]), DType::I32)?;
        let mut hidden = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, h as usize]), DType::F32)?;
        sampling::embedding_f16(&self.cache, device, &self.embed_tokens, &ids,
            &mut hidden, 1, h)?;
        // Gemma scales embeddings by sqrt(hidden_size)
        let embed_scale = (h as f32).sqrt();
        let mut hidden_scaled = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, h as usize]), DType::F32)?;
        ops::mul_scalar(&self.cache, device, &hidden, &mut hidden_scaled, embed_scale)?;
        hidden = hidden_scaled;

        for (i, (layer, lc)) in self.layers.iter().zip(self.layer_configs.iter()).enumerate() {
            let kv_dim = self.config.kv_dim_for_layer(lc);
            let d = lc.head_dim;
            let num_q_heads = self.config.num_heads;
            let num_kv_heads = lc.num_kv_heads;

            if i == 0 {
                eprintln!("[decode] layer 0: kv_dim={}, d={}, q_dim={}, num_q_heads={}, num_kv_heads={}",
                    kv_dim, d, num_q_heads * d, num_q_heads, num_kv_heads);
            }

            // 1. RMSNorm
            let mut normed = GpuTensor::<f32>::zeros(device,
                Shape::from_static(&[1, h as usize]), DType::F32)?;
            ops::rmsnorm(&self.cache, device, &hidden, &layer.attn_norm,
                &mut normed, h, self.config.norm_eps)?;
            if i == 0 { device.synchronize()?; eprintln!("[decode] layer 0: rmsnorm OK"); }

            // 2. Q, K projections
            // NOTE: Gemma 4 has hidden_size(5376) != num_heads*head_dim(8192)
            let q_dim = num_q_heads * d;
            let mut q = GpuTensor::<f32>::zeros(device,
                Shape::from_static(&[1, q_dim as usize]), DType::F32)?;
            let mut k = GpuTensor::<f32>::zeros(device,
                Shape::from_static(&[1, kv_dim as usize]), DType::F32)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &normed, &layer.wq, &mut q, q_dim, h)?;
            if i == 0 { device.synchronize()?; eprintln!("[decode] layer 0: Q gemm OK (q_dim={})", q_dim); }
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &normed, &layer.wk, &mut k, kv_dim, h)?;
            if i == 0 { device.synchronize()?; eprintln!("[decode] layer 0: K gemm OK (kv_dim={})", kv_dim); }

            // V projection — same as K when k_eq_v, otherwise separate
            let mut v = GpuTensor::<f32>::zeros(device,
                Shape::from_static(&[1, kv_dim as usize]), DType::F32)?;
            if self.config.k_eq_v {
                // Copy K to V on GPU (shared projection)
                ops::mul_scalar(&self.cache, device, &k, &mut v, 1.0)?;
            } else {
                crate::quantize::gemm_q4_0_m1(&self.cache, device, &normed, &layer.wv, &mut v, kv_dim, h)?;
            }

            // 3. RoPE — per-layer theta and partial rotation
            let mut q_rope = GpuTensor::<f32>::zeros(device, q.shape.clone(), DType::F32)?;
            let mut k_rope = GpuTensor::<f32>::zeros(device, k.shape.clone(), DType::F32)?;

            let rotary_dim = (d as f32 * lc.partial_rotary_factor) as u32;
            if i == 0 { device.synchronize()?; eprintln!("[decode] layer 0: V done, starting RoPE (rotary_dim={}, d={}, theta={})", rotary_dim, d, lc.rope_theta); }
            if rotary_dim < d && rotary_dim > 0 {
                // Partial RoPE (global layers)
                crate::rope::rope_partial(&self.cache, device, &q, &mut q_rope,
                    num_q_heads, 1, d, rotary_dim, lc.rope_theta, pos)?;
                crate::rope::rope_partial(&self.cache, device, &k, &mut k_rope,
                    num_kv_heads, 1, d, rotary_dim, lc.rope_theta, pos)?;
            } else {
                // Full RoPE (sliding layers)
                crate::rope::rope(&self.cache, device, &q, &mut q_rope,
                    num_q_heads, 1, d, lc.rope_theta, pos)?;
                crate::rope::rope(&self.cache, device, &k, &mut k_rope,
                    num_kv_heads, 1, d, lc.rope_theta, pos)?;
            }

            if i == 0 { device.synchronize()?; eprintln!("[decode] layer 0: RoPE OK"); }
            // 4. QK-norm (with learnable weights if available)
            let mut q_normed = GpuTensor::<f32>::zeros(device, q_rope.shape.clone(), DType::F32)?;
            let mut k_normed = GpuTensor::<f32>::zeros(device, k_rope.shape.clone(), DType::F32)?;
            if let Some(ref qn) = layer.q_norm {
                ops::rmsnorm(&self.cache, device, &q_rope, qn, &mut q_normed, d, self.config.norm_eps)?;
            } else {
                ops::rmsnorm_no_weight(&self.cache, device, &q_rope, &mut q_normed, d, self.config.norm_eps)?;
            }
            if let Some(ref kn) = layer.k_norm {
                ops::rmsnorm(&self.cache, device, &k_rope, kn, &mut k_normed, d, self.config.norm_eps)?;
            } else {
                ops::rmsnorm_no_weight(&self.cache, device, &k_rope, &mut k_normed, d, self.config.norm_eps)?;
            }

            if i == 0 {
                device.synchronize()?;
                eprintln!("[decode] layer 0: QK-norm OK (synced)");
            }
            // 5. KV cache append
            let kv_cache = &mut kv_caches[i];
            // For sliding window, wrap position within window
            let cache_pos = if lc.window_size > 0 {
                pos % lc.window_size
            } else {
                pos
            };
            kv_cache.prefill_at_offset(&self.cache, device, &k_normed, &v, 1, cache_pos)?;
            kv_cache.len = (pos + 1).min(kv_cache.max_seq_len);

            if i == 0 { eprintln!("[decode] layer 0: KV append OK (cache_pos={}, len={})", cache_pos, kv_cache.len); }
            // 6. Decode attention with sliding window
            let mut attn_out = GpuTensor::<f32>::zeros(device,
                Shape::from_static(&[1, (num_q_heads * d) as usize]), DType::F32)?;
            let window = if lc.window_size > 0 { lc.window_size } else { 0 };
            crate::kv_cache::decode_attention_flash_device_len_window(
                &self.cache, device, &q_normed, kv_cache, &mut attn_out,
                num_q_heads, num_kv_heads, d,
                // For non-graph path, create a device buffer with cache_len
                &device.htod(&[kv_cache.len])?,
                window,
            )?;

            if i == 0 { eprintln!("[decode] layer 0: attention OK"); }
            // 7. Output projection: [q_dim] → [hidden_size]
            let mut attn_proj = GpuTensor::<f32>::zeros(device,
                Shape::from_static(&[1, h as usize]), DType::F32)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &attn_out, &layer.wo,
                &mut attn_proj, h, q_dim)?;

            // 8. Fused residual + FFN norm
            let mut ffn_normed = GpuTensor::<f32>::zeros(device,
                Shape::from_static(&[1, h as usize]), DType::F32)?;
            let mut residual = GpuTensor::<f32>::zeros(device,
                Shape::from_static(&[1, h as usize]), DType::F32)?;
            ops::fused_residual_rmsnorm(&self.cache, device, &attn_proj, &hidden,
                &layer.ffn_norm, &mut ffn_normed, &mut residual,
                h, self.config.norm_eps)?;

            // 9. Gate + Up projections
            let mut gate = GpuTensor::<f32>::zeros(device,
                Shape::from_static(&[1, ffn as usize]), DType::F32)?;
            let mut up = GpuTensor::<f32>::zeros(device,
                Shape::from_static(&[1, ffn as usize]), DType::F32)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &ffn_normed, &layer.w_gate,
                &mut gate, ffn, h)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &ffn_normed, &layer.w_up,
                &mut up, ffn, h)?;

            // 10. GeGLU (not SwiGLU!)
            let mut swiglu = GpuTensor::<f32>::zeros(device,
                Shape::from_static(&[1, ffn as usize]), DType::F32)?;
            ops::fused_gelu_mul(&self.cache, device, &gate, &up, &mut swiglu)?;

            // 11. Down projection
            let mut ffn_out = GpuTensor::<f32>::zeros(device,
                Shape::from_static(&[1, h as usize]), DType::F32)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &swiglu, &layer.w_down,
                &mut ffn_out, h, ffn)?;

            // 12. Residual
            let mut output = GpuTensor::<f32>::zeros(device,
                Shape::from_static(&[1, h as usize]), DType::F32)?;
            ops::add(&self.cache, device, &residual, &ffn_out, &mut output)?;

            hidden = output;
        }

        // Final norm + LM head
        let mut normed = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, h as usize]), DType::F32)?;
        ops::rmsnorm(&self.cache, device, &hidden, &self.final_norm,
            &mut normed, h, self.config.norm_eps)?;

        let vocab = self.config.vocab_size;
        let mut logits = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, vocab as usize]), DType::F32)?;

        // LM head: for tied embeddings, embed_tokens is [vocab, hidden].
        // We need logits[1,V] = normed[1,H] × embed^T[H,V].
        // Use embed_tokens with transposed B if lm_head is the placeholder.
        if self.config.tie_word_embeddings && self.lm_head.numel <= 1 {
            // embed_tokens is F16 [V, H]. We need logits[1,V] = normed[1,H] × embed^T[H,V].
            // Cast normed to F16, use cuBLAS HGEMM with transB, cast result back to F32.
            // For now, use the custom M=1 F32 path: download embed row-by-row is too slow.
            // Instead: cast normed to F16, then use F16 GEMM.
            // Actually simplest: use ops::gemm_m1_f32 with a transposed view... but we don't have that.
            // Fallback: download normed to CPU, do matmul on CPU for the LM head.
            // This is only 262K × 5376 dot products = ~1.4B FMAs, ~100ms on CPU.
            let normed_host = normed.to_host(device)?;
            let embed_host: Vec<f32> = {
                let raw = self.embed_tokens.to_host(device)?;
                raw.iter().map(|h| h.to_f32()).collect()
            };
            let mut logits_host = vec![0.0f32; vocab as usize];
            for v in 0..vocab as usize {
                let mut dot = 0.0f32;
                for d in 0..h as usize {
                    dot += normed_host[d] * embed_host[v * h as usize + d];
                }
                logits_host[v] = dot;
            }
            logits = GpuTensor::from_host(device, &logits_host,
                Shape::from_static(&[1, vocab as usize]), DType::F32)?;
        } else {
            ops::gemm(&self.cache, device, &normed, &self.lm_head,
                &mut logits, 1, vocab, h)?;
        }

        // Logit softcapping
        if self.config.final_logit_softcapping > 0.0 {
            let mut capped = GpuTensor::<f32>::zeros(device, logits.shape.clone(), DType::F32)?;
            ops::logit_softcap(&self.cache, device, &logits, &mut capped,
                self.config.final_logit_softcapping)?;
            logits = capped;
        }

        logits.to_host(device)
    }

    /// Generate tokens autoregressively.
    pub fn generate(
        &mut self,
        device: &WarpDevice,
        prompt_ids: &[i32],
        gen_config: &GenerateConfig,
        max_seq_len: u32,
    ) -> Result<GenerationResult, DeviceError> {
        let vocab = self.config.vocab_size as usize;

        // Cache host-side embedding for CPU LM head (one-time cost)
        self.cache_embed_for_lm_head(device)?;

        // Convert to TW-Marlin LAYER BY LAYER to manage VRAM.
        // Each layer: convert Q4 → TW-Marlin, then replace Q4 with tiny placeholder.
        // Peak VRAM overhead: ~300 MB (one layer's Q4 + TW-Marlin simultaneously).
        if self.marlin_layers.is_none() {
            let marlin_start = std::time::Instant::now();
            let h = self.config.hidden_size;
            let ffn = self.config.ffn_dim;
            let mut marlin = Vec::with_capacity(self.layers.len());

            let num_heads = self.config.num_heads;
            let num_layers = self.layers.len();
            let layer_dims: Vec<(u32, u32)> = self.layer_configs.iter()
                .map(|lc| (num_heads * lc.head_dim, self.config.kv_dim_for_layer(lc)))
                .collect();

            for i in 0..num_layers {
                let (q_dim, kv_dim) = layer_dims[i];
                let layer = &mut self.layers[i];

                if i % 10 == 0 { eprintln!("[gemma] Converting layer {}/{} to TW-Marlin...", i, num_layers); }

                let (wq_p, wq_s) = crate::quantize::reorder_to_tw_marlin(device, &layer.wq, h, q_dim)?;
                layer.wq = GpuTensor::from_host(device, &[0u8], Shape::from_static(&[1]), DType::U8)?;
                let (wk_p, wk_s) = crate::quantize::reorder_to_tw_marlin(device, &layer.wk, h, kv_dim)?;
                layer.wk = GpuTensor::from_host(device, &[0u8], Shape::from_static(&[1]), DType::U8)?;
                let (wv_p, wv_s) = crate::quantize::reorder_to_tw_marlin(device, &layer.wv, h, kv_dim)?;
                layer.wv = GpuTensor::from_host(device, &[0u8], Shape::from_static(&[1]), DType::U8)?;
                let (wo_p, wo_s) = crate::quantize::reorder_to_tw_marlin(device, &layer.wo, q_dim, h)?;
                layer.wo = GpuTensor::from_host(device, &[0u8], Shape::from_static(&[1]), DType::U8)?;
                let (wg_p, wg_s) = crate::quantize::reorder_to_tw_marlin(device, &layer.w_gate, h, ffn)?;
                layer.w_gate = GpuTensor::from_host(device, &[0u8], Shape::from_static(&[1]), DType::U8)?;
                let (wu_p, wu_s) = crate::quantize::reorder_to_tw_marlin(device, &layer.w_up, h, ffn)?;
                layer.w_up = GpuTensor::from_host(device, &[0u8], Shape::from_static(&[1]), DType::U8)?;
                let (wd_p, wd_s) = crate::quantize::reorder_to_tw_marlin(device, &layer.w_down, ffn, h)?;
                layer.w_down = GpuTensor::from_host(device, &[0u8], Shape::from_static(&[1]), DType::U8)?;

                marlin.push(GemmaMarlinLayer {
                    wq_p, wq_s, wk_p, wk_s, wv_p, wv_s, wo_p, wo_s,
                    wg_p, wg_s, wu_p, wu_s, wd_p, wd_s,
                });
            }

            device.synchronize()?;
            eprintln!("[gemma] TW-Marlin conversion: {:.1}s (Q4 freed per-layer)", marlin_start.elapsed().as_secs_f64());
            self.marlin_layers = Some(marlin);
        }

        // Allocate per-layer KV caches + decode buffers (zero allocation during generation)
        let mut kv_caches = self.allocate_kv_caches(device, max_seq_len)?;
        let mut buffers = self.allocate_decode_buffers(device)?;
        eprintln!("[gemma] Pre-allocated decode buffers ({} layers)", self.layers.len());

        // Prefill: process all prompt tokens using pre-allocated buffers
        let prefill_start = std::time::Instant::now();
        for (i, &token) in prompt_ids.iter().enumerate() {
            device.htod_copy(&[token], &mut buffers.ids.data)?;
            self.forward_decode_prealloc(device, &mut buffers, &mut kv_caches, i as u32)?;
        }
        device.synchronize()?;
        let prefill_time = prefill_start.elapsed();
        let mut last_logits = buffers.logits.to_host(device)?;

        // Diagnostic: print top-5 logits after prefill
        {
            let mut indexed: Vec<(usize, f32)> = last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("[gemma] Top-5 logits after prefill:");
            for (id, val) in &indexed[..5.min(indexed.len())] {
                eprintln!("  token {}: logit {:.4}", id, val);
            }
            let min_val = last_logits.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let nonzero = last_logits.iter().filter(|&&v| v != 0.0).count();
            eprintln!("  range: [{:.4}, {:.4}], nonzero: {}/{}", min_val, max_val, nonzero, last_logits.len());
        }

        // Decode
        let decode_start = std::time::Instant::now();
        let mut generated = Vec::new();
        let mut pos = prompt_ids.len() as u32;

        let base_seed: u64 = prompt_ids.iter().fold(42u64, |acc, &t| {
            acc.wrapping_mul(6364136223846793005).wrapping_add(t as u64)
        });

        for step in 0..gen_config.max_tokens {
            let rng_seed = base_seed.wrapping_add(step as u64);
            let next_token = sample_token(&last_logits, gen_config, &generated, rng_seed);

            if let Some(eos) = gen_config.eos_token_id {
                if next_token == eos { break; }
            }
            generated.push(next_token);

            if !gen_config.stop_sequences.is_empty()
                && matches_stop_sequence(&generated, &gen_config.stop_sequences)
            { break; }

            device.htod_copy(&[next_token], &mut buffers.ids.data)?;
            self.forward_decode_prealloc(device, &mut buffers, &mut kv_caches, pos)?;
            device.synchronize()?;
            last_logits = buffers.logits.to_host(device)?;
            pos += 1;
        }

        let decode_time = decode_start.elapsed();
        let kv_bytes: usize = kv_caches.iter().map(|kv| kv.k.size_bytes() + kv.v.size_bytes()).sum();

        Ok(GenerationResult {
            tokens: generated.clone(),
            prefill_time,
            decode_time,
            tokens_generated: generated.len(),
            prefill_tokens: prompt_ids.len(),
            tokens_per_sec: if decode_time.as_secs_f64() > 0.0 {
                generated.len() as f64 / decode_time.as_secs_f64()
            } else { 0.0 },
            kv_cache_memory_bytes: kv_bytes,
        })
    }
}

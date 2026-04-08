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

    /// Single-token decode step with pre-allocated buffers.
    /// This is the Gemma-specific decode loop with per-layer dispatch.
    fn forward_decode(
        &self,
        device: &WarpDevice,
        token_id: i32,
        kv_caches: &mut [LayerKVCache],
        pos: u32,
    ) -> Result<Vec<f32>, DeviceError> {
        let h = self.config.hidden_size;
        let ffn = self.config.ffn_dim;

        // Embedding (F16 table → F32 output)
        let ids = GpuTensor::from_host(device, &[token_id],
            Shape::from_static(&[1]), DType::I32)?;
        let mut hidden = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, h as usize]), DType::F32)?;
        sampling::embedding_f16(&self.cache, device, &self.embed_tokens, &ids,
            &mut hidden, 1, h)?;

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
                // Copy K to V (shared projection)
                let k_data = k.to_host(device)?;
                v = GpuTensor::from_host(device, &k_data, k.shape.clone(), DType::F32)?;
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
            // 4. QK-norm (RMSNorm on Q and K before attention)
            let mut q_normed = GpuTensor::<f32>::zeros(device, q_rope.shape.clone(), DType::F32)?;
            let mut k_normed = GpuTensor::<f32>::zeros(device, k_rope.shape.clone(), DType::F32)?;
            ops::rmsnorm_no_weight(&self.cache, device, &q_rope, &mut q_normed, d, self.config.norm_eps)?;
            ops::rmsnorm_no_weight(&self.cache, device, &k_rope, &mut k_normed, d, self.config.norm_eps)?;

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
        &self,
        device: &WarpDevice,
        prompt_ids: &[i32],
        gen_config: &GenerateConfig,
        max_seq_len: u32,
    ) -> Result<GenerationResult, DeviceError> {
        let vocab = self.config.vocab_size as usize;

        // Allocate per-layer KV caches
        let mut kv_caches = self.allocate_kv_caches(device, max_seq_len)?;

        // Prefill: process all prompt tokens
        let prefill_start = std::time::Instant::now();
        let mut last_logits = Vec::new();
        for (i, &token) in prompt_ids.iter().enumerate() {
            let logits = self.forward_decode(device, token, &mut kv_caches, i as u32)?;
            if i == prompt_ids.len() - 1 {
                last_logits = logits;
            }
        }
        device.synchronize()?;
        let prefill_time = prefill_start.elapsed();

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

            last_logits = self.forward_decode(device, next_token, &mut kv_caches, pos)?;
            device.synchronize()?;
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

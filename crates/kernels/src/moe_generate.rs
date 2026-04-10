//! MoE generation engine for Gemma 4 26B-A4B.
//!
//! Expert weights live in RAM, streamed to GPU per-token.
//! Attention + dense MLP + norms stay in VRAM permanently.

use warp_ir::{DType, Shape};
use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::generate::{GenerateConfig, GenerationResult, sample_token, matches_stop_sequence};
use crate::kv_cache::LayerKVCache;
use crate::ops;
use crate::sampling;
use crate::tensor::GpuTensor;
use crate::transformer::{GemmaConfig, GemmaLayerAttentionConfig};

/// MoE generation engine.
pub struct MoEGenerationEngine {
    pub config: GemmaConfig,
    pub layer_configs: Vec<GemmaLayerAttentionConfig>,
    pub embed_tokens: GpuTensor<half::f16>,
    pub final_norm: GpuTensor<f32>,
    pub cache: KernelCache,

    // Per-layer permanent VRAM weights
    pub layers: Vec<MoELayerVRAM>,

    // Per-layer RAM-hosted expert weights (streamed per-token)
    pub expert_gate_up_host: Vec<Vec<half::f16>>,  // [layer][128 * 2*704 * 2816]
    pub expert_down_host: Vec<Vec<half::f16>>,      // [layer][128 * 2816 * 704]

    // Cached host embedding for LM head
    pub embed_host_f32: Option<Vec<f32>>,
}

/// Per-layer weights stored permanently in VRAM.
pub struct MoELayerVRAM {
    // Norms (F32 with Gemma +1)
    pub attn_norm: GpuTensor<f32>,
    pub post_attn_norm: GpuTensor<f32>,
    pub pre_ffn_norm: GpuTensor<f32>,
    pub post_ffn_norm: GpuTensor<f32>,
    pub post_ffn_norm_1: GpuTensor<f32>,
    pub pre_ffn_norm_2: GpuTensor<f32>,
    pub post_ffn_norm_2: GpuTensor<f32>,
    pub q_norm: GpuTensor<f32>,
    pub k_norm: GpuTensor<f32>,
    pub layer_scalar: f32,

    // Attention Q4
    pub wq: GpuTensor<u8>,
    pub wk: GpuTensor<u8>,
    pub wv: GpuTensor<u8>,
    pub wo: GpuTensor<u8>,

    // Dense MLP Q4
    pub w_gate: GpuTensor<u8>,
    pub w_up: GpuTensor<u8>,
    pub w_down: GpuTensor<u8>,

    // Router F32
    pub router_proj: GpuTensor<f32>,
    pub router_scale: GpuTensor<f32>,
    pub per_expert_scale: GpuTensor<f32>,
}

impl MoEGenerationEngine {
    pub fn generate(
        &mut self,
        device: &WarpDevice,
        prompt_ids: &[i32],
        gen_config: &GenerateConfig,
        max_seq_len: u32,
    ) -> Result<GenerationResult, DeviceError> {
        let h = self.config.hidden_size;
        let dense_ffn = self.config.ffn_dim;
        let moe_dim = 704u32; // moe_intermediate_size
        let num_experts = 128u32;
        let top_k = 8u32;

        // Cache host embedding for LM head
        if self.embed_host_f32.is_none() && self.config.tie_word_embeddings {
            let raw = self.embed_tokens.to_host(device)?;
            self.embed_host_f32 = Some(raw.iter().map(|h| h.to_f32()).collect());
        }

        // KV caches
        let mut kv_caches: Vec<LayerKVCache> = self.layer_configs.iter().map(|lc| {
            let kv_dim = self.config.kv_dim_for_layer(lc);
            let cache_len = if lc.window_size > 0 { lc.window_size.min(max_seq_len) } else { max_seq_len };
            LayerKVCache::new(device, cache_len, kv_dim).unwrap()
        }).collect();

        // Pre-allocated expert GPU buffer (reused per layer)
        let expert_gu_size = 2 * moe_dim as usize * h as usize;
        let expert_d_size = h as usize * moe_dim as usize;

        let prefill_start = std::time::Instant::now();

        // Process each prompt token
        let mut last_logits = Vec::new();
        for (t, &token) in prompt_ids.iter().enumerate() {
            last_logits = self.forward_one_token(device, token, t as u32, &mut kv_caches,
                h, dense_ffn, moe_dim, num_experts, top_k)?;
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
            let next_token = sample_token(&last_logits, gen_config, &generated,
                base_seed.wrapping_add(step as u64));
            if let Some(eos) = gen_config.eos_token_id {
                if next_token == eos { break; }
            }
            generated.push(next_token);
            if !gen_config.stop_sequences.is_empty()
                && matches_stop_sequence(&generated, &gen_config.stop_sequences)
            { break; }

            last_logits = self.forward_one_token(device, next_token, pos, &mut kv_caches,
                h, dense_ffn, moe_dim, num_experts, top_k)?;
            device.synchronize()?;
            pos += 1;
        }

        let decode_time = decode_start.elapsed();
        let kv_bytes: usize = kv_caches.iter().map(|kv| kv.k.size_bytes() + kv.v.size_bytes()).sum();

        Ok(GenerationResult {
            tokens: generated.clone(), prefill_time, decode_time,
            tokens_generated: generated.len(), prefill_tokens: prompt_ids.len(),
            tokens_per_sec: if decode_time.as_secs_f64() > 0.0 {
                generated.len() as f64 / decode_time.as_secs_f64()
            } else { 0.0 },
            kv_cache_memory_bytes: kv_bytes,
        })
    }

    /// Hidden state buffer — stored between calls
    fn forward_one_token(
        &self,
        device: &WarpDevice,
        token_id: i32,
        pos: u32,
        kv_caches: &mut [LayerKVCache],
        h: u32,
        dense_ffn: u32,
        moe_dim: u32,
        num_experts: u32,
        top_k: u32,
    ) -> Result<Vec<f32>, DeviceError> {
        let ids = GpuTensor::from_host(device, &[token_id],
            Shape::from_static(&[1]), DType::I32)?;
        let mut hidden = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, h as usize]), DType::F32)?;
        sampling::embedding_f16(&self.cache, device, &self.embed_tokens, &ids,
            &mut hidden, 1, h)?;

        // Scale by sqrt(hidden_size) — Gemma convention
        let mut scaled = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, h as usize]), DType::F32)?;
        ops::mul_scalar(&self.cache, device, &hidden, &mut scaled, (h as f32).sqrt())?;
        hidden = scaled;

        for (i, layer) in self.layers.iter().enumerate() {
            let lc = &self.layer_configs[i];
            let kv_dim = self.config.kv_dim_for_layer(lc);
            let d = lc.head_dim;
            let num_q_heads = self.config.num_heads;
            let num_kv_heads = lc.num_kv_heads;
            let q_dim = num_q_heads * d;

            // 1. Attention
            let mut normed = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F32)?;
            ops::rmsnorm(&self.cache, device, &hidden, &layer.attn_norm, &mut normed, h, self.config.norm_eps)?;

            let mut q = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, q_dim as usize]), DType::F32)?;
            let mut k = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, kv_dim as usize]), DType::F32)?;
            let mut v = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, kv_dim as usize]), DType::F32)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &normed, &layer.wq, &mut q, q_dim, h)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &normed, &layer.wk, &mut k, kv_dim, h)?;
            if self.config.k_eq_v {
                ops::mul_scalar(&self.cache, device, &k, &mut v, 1.0)?;
            } else {
                crate::quantize::gemm_q4_0_m1(&self.cache, device, &normed, &layer.wv, &mut v, kv_dim, h)?;
            }

            // RoPE
            let mut q_rope = GpuTensor::<f32>::zeros(device, q.shape.clone(), DType::F32)?;
            let mut k_rope = GpuTensor::<f32>::zeros(device, k.shape.clone(), DType::F32)?;
            crate::rope::rope(&self.cache, device, &q, &mut q_rope, num_q_heads, 1, d, lc.rope_theta, pos)?;
            crate::rope::rope(&self.cache, device, &k, &mut k_rope, num_kv_heads, 1, d, lc.rope_theta, pos)?;

            // QK-norm (learned weights)
            let mut q_n = GpuTensor::<f32>::zeros(device, q_rope.shape.clone(), DType::F32)?;
            let mut k_n = GpuTensor::<f32>::zeros(device, k_rope.shape.clone(), DType::F32)?;
            ops::rmsnorm(&self.cache, device, &q_rope, &layer.q_norm, &mut q_n, d, self.config.norm_eps)?;
            ops::rmsnorm(&self.cache, device, &k_rope, &layer.k_norm, &mut k_n, d, self.config.norm_eps)?;

            // KV append + attention
            let kv_cache = &mut kv_caches[i];
            let cache_pos = if lc.window_size > 0 { pos % lc.window_size } else { pos };
            kv_cache.prefill_at_offset(&self.cache, device, &k_n, &v, 1, cache_pos)?;
            kv_cache.len = (pos + 1).min(kv_cache.max_seq_len);

            let window = if lc.window_size > 0 { lc.window_size } else { 0 };
            let cache_len_buf = device.htod(&[kv_cache.len])?;
            let mut attn_out = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, q_dim as usize]), DType::F32)?;
            crate::kv_cache::decode_attention_flash_device_len_window(
                &self.cache, device, &q_n, kv_cache, &mut attn_out,
                num_q_heads, num_kv_heads, d, &cache_len_buf, window)?;

            // Output projection
            let mut attn_proj = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F32)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &attn_out, &layer.wo, &mut attn_proj, h, q_dim)?;

            // Post-attention norm + residual
            let mut post_attn = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F32)?;
            ops::rmsnorm(&self.cache, device, &attn_proj, &layer.post_attn_norm, &mut post_attn, h, self.config.norm_eps)?;
            let mut residual = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F32)?;
            ops::add(&self.cache, device, &hidden, &post_attn, &mut residual)?;

            // 2. Dense MLP
            let mut ffn_in = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F32)?;
            ops::rmsnorm(&self.cache, device, &residual, &layer.pre_ffn_norm, &mut ffn_in, h, self.config.norm_eps)?;

            let mut gate = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, dense_ffn as usize]), DType::F32)?;
            let mut up = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, dense_ffn as usize]), DType::F32)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &ffn_in, &layer.w_gate, &mut gate, dense_ffn, h)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &ffn_in, &layer.w_up, &mut up, dense_ffn, h)?;
            let mut geglu = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, dense_ffn as usize]), DType::F32)?;
            ops::fused_gelu_mul(&self.cache, device, &gate, &up, &mut geglu)?;
            let mut dense_out = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F32)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &geglu, &layer.w_down, &mut dense_out, h, dense_ffn)?;

            // 3. MoE path (parallel with dense MLP — uses residual, not ffn_in)
            // Router
            let (expert_ids, expert_weights) = crate::moe::route_experts(
                &self.cache, device, &residual, &layer.router_proj, &layer.router_scale,
                &layer.per_expert_scale, h, num_experts, top_k, self.config.norm_eps)?;

            // MoE input: pre_feedforward_layernorm_2(residual)
            let mut moe_in = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F32)?;
            ops::rmsnorm(&self.cache, device, &residual, &layer.pre_ffn_norm_2, &mut moe_in, h, self.config.norm_eps)?;

            // Run 8 active experts (stream from RAM)
            let gu_per_expert = 2 * moe_dim as usize * h as usize;
            let d_per_expert = h as usize * moe_dim as usize;
            let mut moe_accumulated = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F32)?;

            for (idx, (&eid, &weight)) in expert_ids.iter().zip(expert_weights.iter()).enumerate() {
                let eid = eid as usize;

                // Slice expert weights from host RAM
                let gu_offset = eid * gu_per_expert;
                let gu_slice = &self.expert_gate_up_host[i][gu_offset..gu_offset + gu_per_expert];

                let d_offset = eid * d_per_expert;
                let d_slice = &self.expert_down_host[i][d_offset..d_offset + d_per_expert];

                // Upload expert to GPU as F16, then run with cuBLAS
                let expert_gu = GpuTensor::from_host(device, gu_slice,
                    Shape::from_static(&[2 * moe_dim as usize, h as usize]), DType::F16)?;
                let expert_down = GpuTensor::from_host(device, d_slice,
                    Shape::from_static(&[h as usize, moe_dim as usize]), DType::F16)?;

                // Cast input to F16 for HGEMM
                let mut moe_in_f16 = GpuTensor::<half::f16>::zeros(device,
                    Shape::from_static(&[1, h as usize]), DType::F16)?;
                crate::fp16::cast_f32_to_f16(&self.cache, device, &moe_in, &mut moe_in_f16)?;

                // gate_up = moe_in @ expert_gu^T (F16 HGEMM transB)
                let mut gate_up_f16 = GpuTensor::<half::f16>::zeros(device,
                    Shape::from_static(&[1, (2 * moe_dim) as usize]), DType::F16)?;
                crate::cublas_gemm::gemm_cublas_f16_transB(device,
                    &moe_in_f16, &expert_gu, &mut gate_up_f16, 1, 2 * moe_dim, h)?;

                // Cast back to F32 for split + GeGLU
                let mut gate_up_f32 = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[1, (2 * moe_dim) as usize]), DType::F32)?;
                crate::fp16::cast_f16_to_f32(&self.cache, device, &gate_up_f16, &mut gate_up_f32)?;

                let mut g = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, moe_dim as usize]), DType::F32)?;
                let mut u = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, moe_dim as usize]), DType::F32)?;
                ops::split_gate_up(&self.cache, device, &gate_up_f32, &mut g, &mut u, moe_dim, 1)?;

                let mut gg = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, moe_dim as usize]), DType::F32)?;
                ops::fused_gelu_mul(&self.cache, device, &g, &u, &mut gg)?;

                // down = gg @ expert_down^T (F16)
                let mut gg_f16 = GpuTensor::<half::f16>::zeros(device,
                    Shape::from_static(&[1, moe_dim as usize]), DType::F16)?;
                crate::fp16::cast_f32_to_f16(&self.cache, device, &gg, &mut gg_f16)?;

                let mut expert_out_f16 = GpuTensor::<half::f16>::zeros(device,
                    Shape::from_static(&[1, h as usize]), DType::F16)?;
                crate::cublas_gemm::gemm_cublas_f16_transB(device,
                    &gg_f16, &expert_down, &mut expert_out_f16, 1, h, moe_dim)?;

                let mut expert_out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[1, h as usize]), DType::F32)?;
                crate::fp16::cast_f16_to_f32(&self.cache, device, &expert_out_f16, &mut expert_out)?;

                // Accumulate: moe_accumulated += weight * expert_out
                let mut weighted = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[1, h as usize]), DType::F32)?;
                ops::mul_scalar(&self.cache, device, &expert_out, &mut weighted, weight)?;
                let mut new_acc = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[1, h as usize]), DType::F32)?;
                ops::add(&self.cache, device, &moe_accumulated, &weighted, &mut new_acc)?;
                moe_accumulated = new_acc;
            }

            // 4. Combine dense + MoE
            let mut dense_normed = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F32)?;
            ops::rmsnorm(&self.cache, device, &dense_out, &layer.post_ffn_norm_1, &mut dense_normed, h, self.config.norm_eps)?;

            let mut moe_normed = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F32)?;
            ops::rmsnorm(&self.cache, device, &moe_accumulated, &layer.post_ffn_norm_2, &mut moe_normed, h, self.config.norm_eps)?;

            let mut combined = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F32)?;
            ops::add(&self.cache, device, &dense_normed, &moe_normed, &mut combined)?;

            // 5. Final norm + residual + layer_scalar
            let mut final_normed = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F32)?;
            ops::rmsnorm(&self.cache, device, &combined, &layer.post_ffn_norm, &mut final_normed, h, self.config.norm_eps)?;

            let mut output = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F32)?;
            ops::add(&self.cache, device, &residual, &final_normed, &mut output)?;

            // *= layer_scalar
            let mut output_scaled = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F32)?;
            ops::mul_scalar(&self.cache, device, &output, &mut output_scaled, layer.layer_scalar)?;

            hidden = output_scaled;
        }

        // Return final hidden state for LM head
        // Store in self.last_hidden via interior mutability... or just return it
        // Actually, return it — the caller manages it
        // We need to restructure: forward_one_token should return the hidden state
        // Let's use a simple Output struct

        // Final norm
        let mut normed = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, h as usize]), DType::F32)?;
        ops::rmsnorm(&self.cache, device, &hidden, &self.final_norm, &mut normed, h, self.config.norm_eps)?;

        // LM head: CPU matmul with cached F32 embedding
        let normed_host = normed.to_host(device)?;
        let embed_host = self.embed_host_f32.as_ref()
            .ok_or_else(|| DeviceError::Memory("embed_host_f32 not cached".into()))?;
        let vocab = self.config.vocab_size as usize;
        let mut logits = vec![0.0f32; vocab];
        for v in 0..vocab {
            let mut dot = 0.0f32;
            for d in 0..h as usize {
                dot += normed_host[d] * embed_host[v * h as usize + d];
            }
            logits[v] = dot;
        }

        Ok(logits)
    }
}

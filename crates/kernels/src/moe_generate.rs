//! MoE generation engine for Gemma 4 26B-A4B.
//!
//! Expert weights live in RAM, streamed to GPU per-token.
//! Attention + dense MLP + norms stay in VRAM permanently.
//! All intermediate tensors pre-allocated for zero-allocation decode.

use warp_ir::{DType, Shape};
use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::generate::{GenerateConfig, GenerationResult, sample_token, matches_stop_sequence};
use crate::kv_cache::LayerKVCache;
use crate::ops;
use crate::sampling;
use crate::tensor::GpuTensor;
use crate::transformer::{GemmaConfig, GemmaLayerAttentionConfig};

/// Per-layer weights stored permanently in VRAM.
pub struct MoELayerVRAM {
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
    pub wq: GpuTensor<u8>,
    pub wk: GpuTensor<u8>,
    pub wv: GpuTensor<u8>,
    pub wo: GpuTensor<u8>,
    pub w_gate: GpuTensor<u8>,
    pub w_up: GpuTensor<u8>,
    pub w_down: GpuTensor<u8>,
    pub router_proj: GpuTensor<f32>,
    pub router_scale: GpuTensor<f32>,
    pub per_expert_scale: GpuTensor<f32>,
}

/// Pre-allocated decode buffers — ZERO allocation during generation.
struct MoEDecodeBuffers {
    // Global
    hidden: GpuTensor<f32>,
    hidden_scaled: GpuTensor<f32>,

    // Per-layer attention
    normed: GpuTensor<f32>,
    q: GpuTensor<f32>,     // max q_dim across layers
    k: GpuTensor<f32>,     // max kv_dim
    v: GpuTensor<f32>,
    q_rope: GpuTensor<f32>,
    k_rope: GpuTensor<f32>,
    q_n: GpuTensor<f32>,
    k_n: GpuTensor<f32>,
    attn_out: GpuTensor<f32>,
    attn_proj: GpuTensor<f32>,
    post_attn: GpuTensor<f32>,
    residual: GpuTensor<f32>,
    cache_len_buf: cudarc::driver::CudaSlice<u32>,

    // Dense MLP
    ffn_in: GpuTensor<f32>,
    dense_gate: GpuTensor<f32>,
    dense_up: GpuTensor<f32>,
    dense_geglu: GpuTensor<f32>,
    dense_out: GpuTensor<f32>,

    // MoE
    moe_in: GpuTensor<f32>,
    moe_accumulated: GpuTensor<f32>,

    // Expert reusable buffers
    expert_gu_gpu: GpuTensor<half::f16>,    // [2*704, 2816]
    expert_down_gpu: GpuTensor<half::f16>,  // [2816, 704]
    moe_in_f16: GpuTensor<half::f16>,
    gate_up_f16: GpuTensor<half::f16>,
    gate_up_f32: GpuTensor<f32>,
    expert_gate: GpuTensor<f32>,
    expert_up: GpuTensor<f32>,
    expert_geglu: GpuTensor<f32>,
    expert_geglu_f16: GpuTensor<half::f16>,
    expert_out_f16: GpuTensor<half::f16>,
    expert_out: GpuTensor<f32>,
    expert_weighted: GpuTensor<f32>,
    expert_new_acc: GpuTensor<f32>,

    // Combine
    dense_normed: GpuTensor<f32>,
    moe_normed: GpuTensor<f32>,
    combined: GpuTensor<f32>,
    final_normed: GpuTensor<f32>,
    output: GpuTensor<f32>,
    output_scaled: GpuTensor<f32>,

    // LM head
    lm_normed: GpuTensor<f32>,
}

/// MoE generation engine.
pub struct MoEGenerationEngine {
    pub config: GemmaConfig,
    pub layer_configs: Vec<GemmaLayerAttentionConfig>,
    pub embed_tokens: GpuTensor<half::f16>,
    pub final_norm: GpuTensor<f32>,
    pub cache: KernelCache,
    pub layers: Vec<MoELayerVRAM>,
    pub expert_gate_up_host: Vec<Vec<half::f16>>,
    pub expert_down_host: Vec<Vec<half::f16>>,
    pub embed_host_f32: Option<Vec<f32>>,
}

impl MoEGenerationEngine {
    fn allocate_buffers(&self, device: &WarpDevice) -> Result<MoEDecodeBuffers, DeviceError> {
        let h = self.config.hidden_size as usize;
        let max_q_dim = self.layer_configs.iter().map(|lc| (self.config.num_heads * lc.head_dim) as usize).max().unwrap_or(h);
        let max_kv_dim = self.layer_configs.iter().map(|lc| self.config.kv_dim_for_layer(lc) as usize).max().unwrap_or(h);
        let dense_ffn = self.config.ffn_dim as usize;
        let moe_dim = 704usize;

        Ok(MoEDecodeBuffers {
            hidden: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            hidden_scaled: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            q: GpuTensor::zeros(device, Shape::from_static(&[1, max_q_dim]), DType::F32)?,
            k: GpuTensor::zeros(device, Shape::from_static(&[1, max_kv_dim]), DType::F32)?,
            v: GpuTensor::zeros(device, Shape::from_static(&[1, max_kv_dim]), DType::F32)?,
            q_rope: GpuTensor::zeros(device, Shape::from_static(&[1, max_q_dim]), DType::F32)?,
            k_rope: GpuTensor::zeros(device, Shape::from_static(&[1, max_kv_dim]), DType::F32)?,
            q_n: GpuTensor::zeros(device, Shape::from_static(&[1, max_q_dim]), DType::F32)?,
            k_n: GpuTensor::zeros(device, Shape::from_static(&[1, max_kv_dim]), DType::F32)?,
            attn_out: GpuTensor::zeros(device, Shape::from_static(&[1, max_q_dim]), DType::F32)?,
            attn_proj: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            post_attn: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            residual: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            cache_len_buf: device.alloc_zeros::<u32>(1)?,
            ffn_in: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            dense_gate: GpuTensor::zeros(device, Shape::from_static(&[1, dense_ffn]), DType::F32)?,
            dense_up: GpuTensor::zeros(device, Shape::from_static(&[1, dense_ffn]), DType::F32)?,
            dense_geglu: GpuTensor::zeros(device, Shape::from_static(&[1, dense_ffn]), DType::F32)?,
            dense_out: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            moe_in: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            moe_accumulated: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            expert_gu_gpu: GpuTensor::zeros(device, Shape::from_static(&[2*moe_dim, h]), DType::F16)?,
            expert_down_gpu: GpuTensor::zeros(device, Shape::from_static(&[h, moe_dim]), DType::F16)?,
            moe_in_f16: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F16)?,
            gate_up_f16: GpuTensor::zeros(device, Shape::from_static(&[1, 2*moe_dim]), DType::F16)?,
            gate_up_f32: GpuTensor::zeros(device, Shape::from_static(&[1, 2*moe_dim]), DType::F32)?,
            expert_gate: GpuTensor::zeros(device, Shape::from_static(&[1, moe_dim]), DType::F32)?,
            expert_up: GpuTensor::zeros(device, Shape::from_static(&[1, moe_dim]), DType::F32)?,
            expert_geglu: GpuTensor::zeros(device, Shape::from_static(&[1, moe_dim]), DType::F32)?,
            expert_geglu_f16: GpuTensor::zeros(device, Shape::from_static(&[1, moe_dim]), DType::F16)?,
            expert_out_f16: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F16)?,
            expert_out: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            expert_weighted: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            expert_new_acc: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            dense_normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            moe_normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            combined: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            final_normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            output: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            output_scaled: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            lm_normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
        })
    }

    fn forward_prealloc(
        &self, device: &WarpDevice, b: &mut MoEDecodeBuffers,
        kv_caches: &mut [LayerKVCache], pos: u32,
    ) -> Result<Vec<f32>, DeviceError> {
        let h = self.config.hidden_size;
        let dense_ffn = self.config.ffn_dim;
        let moe_dim = 704u32;
        let num_experts = 128u32;
        let top_k = 8u32;

        // Embedding + scale
        let embed_scale = (h as f32).sqrt();
        ops::mul_scalar(&self.cache, device, &b.hidden, &mut b.hidden_scaled, embed_scale)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let lc = &self.layer_configs[i];
            let kv_dim = self.config.kv_dim_for_layer(lc);
            let d = lc.head_dim;
            let num_q_heads = self.config.num_heads;
            let num_kv_heads = lc.num_kv_heads;
            let q_dim = num_q_heads * d;

            let x: &GpuTensor<f32> = if i == 0 {
                &b.hidden_scaled
            } else {
                &b.output_scaled
            };

            // Attention
            ops::rmsnorm(&self.cache, device, x,
                &layer.attn_norm, &mut b.normed, h, self.config.norm_eps)?;

            crate::quantize::gemm_q4_0_m1(&self.cache, device, &b.normed, &layer.wq, &mut b.q, q_dim, h)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &b.normed, &layer.wk, &mut b.k, kv_dim, h)?;
            if self.config.k_eq_v {
                ops::mul_scalar(&self.cache, device, &b.k, &mut b.v, 1.0)?;
            } else {
                crate::quantize::gemm_q4_0_m1(&self.cache, device, &b.normed, &layer.wv, &mut b.v, kv_dim, h)?;
            }

            crate::rope::rope(&self.cache, device, &b.q, &mut b.q_rope, num_q_heads, 1, d, lc.rope_theta, pos)?;
            crate::rope::rope(&self.cache, device, &b.k, &mut b.k_rope, num_kv_heads, 1, d, lc.rope_theta, pos)?;

            ops::rmsnorm(&self.cache, device, &b.q_rope, &layer.q_norm, &mut b.q_n, d, self.config.norm_eps)?;
            ops::rmsnorm(&self.cache, device, &b.k_rope, &layer.k_norm, &mut b.k_n, d, self.config.norm_eps)?;

            let kv = &mut kv_caches[i];
            let cache_pos = if lc.window_size > 0 { pos % lc.window_size } else { pos };
            kv.prefill_at_offset(&self.cache, device, &b.k_n, &b.v, 1, cache_pos)?;
            kv.len = (pos + 1).min(kv.max_seq_len);

            let window = if lc.window_size > 0 { lc.window_size } else { 0 };
            device.htod_copy(&[kv.len], &mut b.cache_len_buf)?;
            crate::kv_cache::decode_attention_flash_device_len_window(
                &self.cache, device, &b.q_n, kv, &mut b.attn_out,
                num_q_heads, num_kv_heads, d, &b.cache_len_buf, window)?;

            crate::quantize::gemm_q4_0_m1(&self.cache, device, &b.attn_out, &layer.wo, &mut b.attn_proj, h, q_dim)?;

            ops::rmsnorm(&self.cache, device, &b.attn_proj, &layer.post_attn_norm, &mut b.post_attn, h, self.config.norm_eps)?;
            ops::add(&self.cache, device, x, &b.post_attn, &mut b.residual)?;

            // Dense MLP
            ops::rmsnorm(&self.cache, device, &b.residual, &layer.pre_ffn_norm, &mut b.ffn_in, h, self.config.norm_eps)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &b.ffn_in, &layer.w_gate, &mut b.dense_gate, dense_ffn, h)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &b.ffn_in, &layer.w_up, &mut b.dense_up, dense_ffn, h)?;
            ops::fused_gelu_mul(&self.cache, device, &b.dense_gate, &b.dense_up, &mut b.dense_geglu)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &b.dense_geglu, &layer.w_down, &mut b.dense_out, h, dense_ffn)?;

            // MoE: route
            let (expert_ids, expert_weights) = crate::moe::route_experts(
                &self.cache, device, &b.residual, &layer.router_proj, &layer.router_scale,
                &layer.per_expert_scale, h, num_experts, top_k, self.config.norm_eps)?;

            // MoE input
            ops::rmsnorm(&self.cache, device, &b.residual, &layer.pre_ffn_norm_2, &mut b.moe_in, h, self.config.norm_eps)?;

            // Cast moe_in to F16 once (reused for all experts)
            crate::fp16::cast_f32_to_f16(&self.cache, device, &b.moe_in, &mut b.moe_in_f16)?;

            // Zero accumulator
            let zero_host = vec![0.0f32; h as usize];
            device.htod_copy(&zero_host, &mut b.moe_accumulated.data)?;

            let gu_per = 2 * moe_dim as usize * h as usize;
            let d_per = h as usize * moe_dim as usize;

            for (_, (&eid, &weight)) in expert_ids.iter().zip(expert_weights.iter()).enumerate() {
                let eid = eid as usize;
                let gu_off = eid * gu_per;
                let d_off = eid * d_per;

                // Upload expert weights to pre-allocated GPU buffers
                device.htod_copy(&self.expert_gate_up_host[i][gu_off..gu_off + gu_per], &mut b.expert_gu_gpu.data)?;
                device.htod_copy(&self.expert_down_host[i][d_off..d_off + d_per], &mut b.expert_down_gpu.data)?;

                // gate_up HGEMM transB
                crate::cublas_gemm::gemm_cublas_f16_transB(device,
                    &b.moe_in_f16, &b.expert_gu_gpu, &mut b.gate_up_f16, 1, 2*moe_dim, h)?;
                crate::fp16::cast_f16_to_f32(&self.cache, device, &b.gate_up_f16, &mut b.gate_up_f32)?;

                ops::split_gate_up(&self.cache, device, &b.gate_up_f32, &mut b.expert_gate, &mut b.expert_up, moe_dim, 1)?;
                ops::fused_gelu_mul(&self.cache, device, &b.expert_gate, &b.expert_up, &mut b.expert_geglu)?;

                // down HGEMM transB
                crate::fp16::cast_f32_to_f16(&self.cache, device, &b.expert_geglu, &mut b.expert_geglu_f16)?;
                crate::cublas_gemm::gemm_cublas_f16_transB(device,
                    &b.expert_geglu_f16, &b.expert_down_gpu, &mut b.expert_out_f16, 1, h, moe_dim)?;
                crate::fp16::cast_f16_to_f32(&self.cache, device, &b.expert_out_f16, &mut b.expert_out)?;

                // Accumulate
                ops::mul_scalar(&self.cache, device, &b.expert_out, &mut b.expert_weighted, weight)?;
                ops::add(&self.cache, device, &b.moe_accumulated, &b.expert_weighted, &mut b.expert_new_acc)?;
                // Swap: accumulated = new_acc (pointer swap would be ideal, but copy for now)
                ops::mul_scalar(&self.cache, device, &b.expert_new_acc, &mut b.moe_accumulated, 1.0)?;
            }

            // Combine dense + MoE
            ops::rmsnorm(&self.cache, device, &b.dense_out, &layer.post_ffn_norm_1, &mut b.dense_normed, h, self.config.norm_eps)?;
            ops::rmsnorm(&self.cache, device, &b.moe_accumulated, &layer.post_ffn_norm_2, &mut b.moe_normed, h, self.config.norm_eps)?;
            ops::add(&self.cache, device, &b.dense_normed, &b.moe_normed, &mut b.combined)?;

            ops::rmsnorm(&self.cache, device, &b.combined, &layer.post_ffn_norm, &mut b.final_normed, h, self.config.norm_eps)?;
            ops::add(&self.cache, device, &b.residual, &b.final_normed, &mut b.output)?;
            ops::mul_scalar(&self.cache, device, &b.output, &mut b.output_scaled, layer.layer_scalar)?;
        }

        // Final norm + LM head (CPU with cached embedding)
        ops::rmsnorm(&self.cache, device, &b.output_scaled, &self.final_norm, &mut b.lm_normed, h, self.config.norm_eps)?;
        let normed_host = b.lm_normed.to_host(device)?;
        let embed = self.embed_host_f32.as_ref().unwrap();
        let vocab = self.config.vocab_size as usize;
        let mut logits = vec![0.0f32; vocab];
        for v in 0..vocab {
            let mut dot = 0.0f32;
            for d in 0..h as usize {
                dot += normed_host[d] * embed[v * h as usize + d];
            }
            logits[v] = dot;
        }
        Ok(logits)
    }

    pub fn generate(
        &mut self, device: &WarpDevice, prompt_ids: &[i32],
        gen_config: &GenerateConfig, max_seq_len: u32,
    ) -> Result<GenerationResult, DeviceError> {
        // Cache embedding for LM head
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

        // Pre-allocate ALL decode buffers
        let mut bufs = self.allocate_buffers(device)?;
        eprintln!("[moe] Pre-allocated decode buffers");

        let h = self.config.hidden_size;
        let prefill_start = std::time::Instant::now();

        // Prefill
        let mut last_logits = Vec::new();
        for (t, &token) in prompt_ids.iter().enumerate() {
            let ids = GpuTensor::from_host(device, &[token], Shape::from_static(&[1]), DType::I32)?;
            sampling::embedding_f16(&self.cache, device, &self.embed_tokens, &ids, &mut bufs.hidden, 1, h)?;
            last_logits = self.forward_prealloc(device, &mut bufs, &mut kv_caches, t as u32)?;
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
            let next = sample_token(&last_logits, gen_config, &generated, base_seed.wrapping_add(step as u64));
            if let Some(eos) = gen_config.eos_token_id { if next == eos { break; } }
            generated.push(next);
            if !gen_config.stop_sequences.is_empty() && matches_stop_sequence(&generated, &gen_config.stop_sequences) { break; }

            let ids = GpuTensor::from_host(device, &[next], Shape::from_static(&[1]), DType::I32)?;
            sampling::embedding_f16(&self.cache, device, &self.embed_tokens, &ids, &mut bufs.hidden, 1, h)?;
            last_logits = self.forward_prealloc(device, &mut bufs, &mut kv_caches, pos)?;
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
}

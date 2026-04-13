//! All-VRAM Q4 MoE generation engine for Gemma 4 26B-A4B.
//!
//! ALL weights in VRAM at Q4 (13.4 GB). No RAM streaming.
//! Active per token: 1.39 GB → 150+ tok/sec target.
//! Uses TW-Marlin GEMM for all projections including experts.

use warp_ir::{DType, Shape};
use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::generate::{GenerateConfig, GenerationResult, sample_token, matches_stop_sequence};
use crate::kv_cache::LayerKVCache;
use crate::ops;
use crate::sampling;
use crate::tensor::GpuTensor;
use crate::transformer::{GemmaConfig, GemmaLayerAttentionConfig};

/// Per-layer Q4 weights — ALL in VRAM including experts.
pub struct MoEQ4Layer {
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
    pub wq: GpuTensor<u8>, pub wk: GpuTensor<u8>,
    pub wv: GpuTensor<u8>, pub wo: GpuTensor<u8>,

    // Dense MLP Q4
    pub w_gate: GpuTensor<u8>, pub w_up: GpuTensor<u8>, pub w_down: GpuTensor<u8>,

    // Router F32
    pub router_proj: GpuTensor<f32>,
    pub router_scale: GpuTensor<f32>,
    pub per_expert_scale: GpuTensor<f32>,

    // Expert Q4 — ALL 128 experts contiguous in VRAM
    pub experts_gu_q4: GpuTensor<u8>,    // Q4_0 column-major blocks, all experts concatenated
    pub experts_d_q4: GpuTensor<u8>,
    pub gu_q4_per_expert: usize,
    pub d_q4_per_expert: usize,
    // Dimensions for the expert FFN GEMMs
    pub expert_gu_k: u32,  // K dimension for gate_up (= hidden_size)
    pub expert_gu_n: u32,  // N dimension for gate_up (= 2 * moe_dim)
    pub expert_d_k: u32,   // K dimension for down (= moe_dim)
    pub expert_d_n: u32,   // N dimension for down (= hidden_size)
}

/// Pre-allocated decode buffers.
struct Q4MoEBuffers {
    hidden: GpuTensor<f32>,
    hidden_scaled: GpuTensor<f32>,
    normed: GpuTensor<f32>,
    q: GpuTensor<f32>, k: GpuTensor<f32>, v: GpuTensor<f32>,
    q_rope: GpuTensor<f32>, k_rope: GpuTensor<f32>,
    q_n: GpuTensor<f32>, k_n: GpuTensor<f32>,
    attn_out: GpuTensor<f32>, attn_proj: GpuTensor<f32>,
    post_attn: GpuTensor<f32>, residual: GpuTensor<f32>,
    ffn_in: GpuTensor<f32>,
    dense_gate: GpuTensor<f32>, dense_up: GpuTensor<f32>,
    dense_geglu: GpuTensor<f32>, dense_out: GpuTensor<f32>,
    moe_in: GpuTensor<f32>,
    moe_accumulated: GpuTensor<f32>,
    expert_out: GpuTensor<f32>,
    expert_weighted: GpuTensor<f32>,
    expert_gate_up: GpuTensor<f32>,  // [1, 2*moe_dim] for expert FFN
    expert_gate: GpuTensor<f32>,
    expert_up: GpuTensor<f32>,
    expert_geglu: GpuTensor<f32>,
    dense_normed: GpuTensor<f32>,
    moe_normed: GpuTensor<f32>,
    combined: GpuTensor<f32>,
    final_normed: GpuTensor<f32>,
    output: GpuTensor<f32>,
    output_scaled: GpuTensor<f32>,
    lm_normed: GpuTensor<f32>,
    cache_len_buf: cudarc::driver::CudaSlice<u32>,
}

pub struct MoEQ4Engine {
    pub config: GemmaConfig,
    pub layer_configs: Vec<GemmaLayerAttentionConfig>,
    pub embed_tokens: GpuTensor<half::f16>,
    pub final_norm: GpuTensor<f32>,
    pub cache: KernelCache,
    pub layers: Vec<MoEQ4Layer>,
}

impl MoEQ4Engine {
    fn allocate_buffers(&self, device: &WarpDevice) -> Result<Q4MoEBuffers, DeviceError> {
        let h = self.config.hidden_size as usize;
        let max_q = self.layer_configs.iter().map(|lc| (self.config.num_heads * lc.head_dim) as usize).max().unwrap_or(h);
        let max_kv = self.layer_configs.iter().map(|lc| self.config.kv_dim_for_layer(lc) as usize).max().unwrap_or(h);
        let ffn = self.config.ffn_dim as usize;
        let moe_dim = 704usize;

        Ok(Q4MoEBuffers {
            hidden: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            hidden_scaled: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            q: GpuTensor::zeros(device, Shape::from_static(&[1, max_q]), DType::F32)?,
            k: GpuTensor::zeros(device, Shape::from_static(&[1, max_kv]), DType::F32)?,
            v: GpuTensor::zeros(device, Shape::from_static(&[1, max_kv]), DType::F32)?,
            q_rope: GpuTensor::zeros(device, Shape::from_static(&[1, max_q]), DType::F32)?,
            k_rope: GpuTensor::zeros(device, Shape::from_static(&[1, max_kv]), DType::F32)?,
            q_n: GpuTensor::zeros(device, Shape::from_static(&[1, max_q]), DType::F32)?,
            k_n: GpuTensor::zeros(device, Shape::from_static(&[1, max_kv]), DType::F32)?,
            attn_out: GpuTensor::zeros(device, Shape::from_static(&[1, max_q]), DType::F32)?,
            attn_proj: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            post_attn: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            residual: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            ffn_in: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            dense_gate: GpuTensor::zeros(device, Shape::from_static(&[1, ffn]), DType::F32)?,
            dense_up: GpuTensor::zeros(device, Shape::from_static(&[1, ffn]), DType::F32)?,
            dense_geglu: GpuTensor::zeros(device, Shape::from_static(&[1, ffn]), DType::F32)?,
            dense_out: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            moe_in: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            moe_accumulated: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            expert_out: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            expert_weighted: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            expert_gate_up: GpuTensor::zeros(device, Shape::from_static(&[1, 2*moe_dim]), DType::F32)?,
            expert_gate: GpuTensor::zeros(device, Shape::from_static(&[1, moe_dim]), DType::F32)?,
            expert_up: GpuTensor::zeros(device, Shape::from_static(&[1, moe_dim]), DType::F32)?,
            expert_geglu: GpuTensor::zeros(device, Shape::from_static(&[1, moe_dim]), DType::F32)?,
            dense_normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            moe_normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            combined: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            final_normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            output: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            output_scaled: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            lm_normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            cache_len_buf: device.alloc_zeros::<u32>(1)?,
        })
    }

    fn forward_decode(
        &self, device: &WarpDevice, b: &mut Q4MoEBuffers,
        kv_caches: &mut [LayerKVCache], pos: u32,
    ) -> Result<Vec<f32>, DeviceError> {
        let h = self.config.hidden_size;
        let ffn = self.config.ffn_dim;
        let moe_dim = 704u32;
        let top_k = 8u32;
        let num_experts = 128u32;

        ops::mul_scalar(&self.cache, device, &b.hidden, &mut b.hidden_scaled, (h as f32).sqrt())?;

        for (i, layer) in self.layers.iter().enumerate() {
            let lc = &self.layer_configs[i];
            let d = lc.head_dim;
            let q_dim = self.config.num_heads * d;
            let kv_dim = self.config.kv_dim_for_layer(lc);
            let x = if i == 0 { &b.hidden_scaled } else { &b.output_scaled };

            // 1. Attention (Split-K for better occupancy on small N)
            ops::rmsnorm(&self.cache, device, x, &layer.attn_norm, &mut b.normed, h, self.config.norm_eps)?;
            crate::quantize::gemm_q4_0_m1_splitk(&self.cache, device, &b.normed, &layer.wq, &mut b.q, q_dim, h)?;
            crate::quantize::gemm_q4_0_m1_splitk(&self.cache, device, &b.normed, &layer.wk, &mut b.k, kv_dim, h)?;
            if self.config.k_eq_v {
                ops::mul_scalar(&self.cache, device, &b.k, &mut b.v, 1.0)?;
            } else {
                crate::quantize::gemm_q4_0_m1_splitk(&self.cache, device, &b.normed, &layer.wv, &mut b.v, kv_dim, h)?;
            }

            crate::rope::rope(&self.cache, device, &b.q, &mut b.q_rope, self.config.num_heads, 1, d, lc.rope_theta, pos)?;
            crate::rope::rope(&self.cache, device, &b.k, &mut b.k_rope, lc.num_kv_heads, 1, d, lc.rope_theta, pos)?;
            ops::rmsnorm(&self.cache, device, &b.q_rope, &layer.q_norm, &mut b.q_n, d, self.config.norm_eps)?;
            ops::rmsnorm(&self.cache, device, &b.k_rope, &layer.k_norm, &mut b.k_n, d, self.config.norm_eps)?;

            let kv = &mut kv_caches[i];
            let cp = if lc.window_size > 0 { pos % lc.window_size } else { pos };
            kv.prefill_at_offset(&self.cache, device, &b.k_n, &b.v, 1, cp)?;
            kv.len = (pos + 1).min(kv.max_seq_len);
            let win = if lc.window_size > 0 { lc.window_size } else { 0 };
            device.htod_copy(&[kv.len], &mut b.cache_len_buf)?;
            crate::kv_cache::decode_attention_flash_device_len_window(
                &self.cache, device, &b.q_n, kv, &mut b.attn_out,
                self.config.num_heads, lc.num_kv_heads, d, &b.cache_len_buf, win)?;

            crate::quantize::gemm_q4_0_m1_splitk(&self.cache, device, &b.attn_out, &layer.wo, &mut b.attn_proj, h, q_dim)?;
            ops::rmsnorm(&self.cache, device, &b.attn_proj, &layer.post_attn_norm, &mut b.post_attn, h, self.config.norm_eps)?;
            ops::add(&self.cache, device, x, &b.post_attn, &mut b.residual)?;

            // 2. Dense MLP
            ops::rmsnorm(&self.cache, device, &b.residual, &layer.pre_ffn_norm, &mut b.ffn_in, h, self.config.norm_eps)?;
            crate::quantize::gemm_q4_0_m1_splitk(&self.cache, device, &b.ffn_in, &layer.w_gate, &mut b.dense_gate, ffn, h)?;
            crate::quantize::gemm_q4_0_m1_splitk(&self.cache, device, &b.ffn_in, &layer.w_up, &mut b.dense_up, ffn, h)?;
            ops::fused_gelu_mul(&self.cache, device, &b.dense_gate, &b.dense_up, &mut b.dense_geglu)?;
            crate::quantize::gemm_q4_0_m1_splitk(&self.cache, device, &b.dense_geglu, &layer.w_down, &mut b.dense_out, h, ffn)?;

            // 3. MoE
            let (expert_ids, expert_weights) = crate::moe::route_experts(
                &self.cache, device, &b.residual, &layer.router_proj, &layer.router_scale,
                &layer.per_expert_scale, h, num_experts, top_k, self.config.norm_eps)?;

            ops::rmsnorm(&self.cache, device, &b.residual, &layer.pre_ffn_norm_2, &mut b.moe_in, h, self.config.norm_eps)?;

            // Zero accumulator
            let zeros = vec![0.0f32; h as usize];
            device.htod_copy(&zeros, &mut b.moe_accumulated.data)?;

            // Run 8 active experts — ALL in VRAM, GPU-side buffer offset (ZERO copies)
            for (_, (&eid, &weight)) in expert_ids.iter().zip(expert_weights.iter()).enumerate() {
                let eid = eid as usize;
                let gu_off = eid * layer.gu_q4_per_expert;
                let d_off = eid * layer.d_q4_per_expert;

                // Expert FFN using Split-K buffer offset GEMM (max occupancy, zero copies)
                crate::quantize::gemm_q4_0_m1_splitk_from_buffer(&self.cache, device, &b.moe_in,
                    &layer.experts_gu_q4, gu_off, &mut b.expert_gate_up,
                    layer.expert_gu_n, layer.expert_gu_k)?;
                ops::split_gate_up(&self.cache, device, &b.expert_gate_up,
                    &mut b.expert_gate, &mut b.expert_up, moe_dim, 1)?;
                ops::fused_gelu_mul(&self.cache, device, &b.expert_gate, &b.expert_up, &mut b.expert_geglu)?;
                crate::quantize::gemm_q4_0_m1_splitk_from_buffer(&self.cache, device, &b.expert_geglu,
                    &layer.experts_d_q4, d_off, &mut b.expert_out,
                    layer.expert_d_n, layer.expert_d_k)?;

                // Accumulate
                ops::mul_scalar(&self.cache, device, &b.expert_out, &mut b.expert_weighted, weight)?;
                ops::add(&self.cache, device, &b.moe_accumulated, &b.expert_weighted, &mut b.expert_out)?;
                ops::mul_scalar(&self.cache, device, &b.expert_out, &mut b.moe_accumulated, 1.0)?;
            }

            // 4. Combine dense + MoE
            ops::rmsnorm(&self.cache, device, &b.dense_out, &layer.post_ffn_norm_1, &mut b.dense_normed, h, self.config.norm_eps)?;
            ops::rmsnorm(&self.cache, device, &b.moe_accumulated, &layer.post_ffn_norm_2, &mut b.moe_normed, h, self.config.norm_eps)?;
            ops::add(&self.cache, device, &b.dense_normed, &b.moe_normed, &mut b.combined)?;

            ops::rmsnorm(&self.cache, device, &b.combined, &layer.post_ffn_norm, &mut b.final_normed, h, self.config.norm_eps)?;
            ops::add(&self.cache, device, &b.residual, &b.final_normed, &mut b.output)?;
            ops::mul_scalar(&self.cache, device, &b.output, &mut b.output_scaled, layer.layer_scalar)?;
        }

        // LM head
        ops::rmsnorm(&self.cache, device, &b.output_scaled, &self.final_norm, &mut b.lm_normed, h, self.config.norm_eps)?;
        let vocab = self.config.vocab_size;
        let mut nf16 = GpuTensor::<half::f16>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F16)?;
        crate::fp16::cast_f32_to_f16(&self.cache, device, &b.lm_normed, &mut nf16)?;
        let mut lf16 = GpuTensor::<half::f16>::zeros(device, Shape::from_static(&[1, vocab as usize]), DType::F16)?;
        crate::cublas_gemm::gemm_cublas_f16_transB(device, &nf16, &self.embed_tokens, &mut lf16, 1, vocab, h)?;
        let mut logits = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, vocab as usize]), DType::F32)?;
        crate::fp16::cast_f16_to_f32(&self.cache, device, &lf16, &mut logits)?;
        logits.to_host(device)
    }

    pub fn generate(
        &self, device: &WarpDevice, prompt_ids: &[i32],
        gen_config: &GenerateConfig, max_seq_len: u32,
    ) -> Result<GenerationResult, DeviceError> {
        let h = self.config.hidden_size;
        let mut kv_caches: Vec<LayerKVCache> = self.layer_configs.iter().map(|lc| {
            let kv_dim = self.config.kv_dim_for_layer(lc);
            let cl = if lc.window_size > 0 { lc.window_size.min(max_seq_len) } else { max_seq_len };
            LayerKVCache::new(device, cl, kv_dim).unwrap()
        }).collect();

        let mut bufs = self.allocate_buffers(device)?;
        eprintln!("[moe-q4] Pre-allocated decode buffers, all weights in VRAM");

        let ps = std::time::Instant::now();
        let mut last_logits = Vec::new();
        for (t, &tok) in prompt_ids.iter().enumerate() {
            let ids = GpuTensor::from_host(device, &[tok], Shape::from_static(&[1]), DType::I32)?;
            sampling::embedding_f16(&self.cache, device, &self.embed_tokens, &ids, &mut bufs.hidden, 1, h)?;
            last_logits = self.forward_decode(device, &mut bufs, &mut kv_caches, t as u32)?;
        }
        device.synchronize()?;
        let pt = ps.elapsed();

        let ds = std::time::Instant::now();
        let mut gen = Vec::new();
        let mut pos = prompt_ids.len() as u32;
        let seed: u64 = prompt_ids.iter().fold(42u64, |a, &t| a.wrapping_mul(6364136223846793005).wrapping_add(t as u64));

        for step in 0..gen_config.max_tokens {
            let next = sample_token(&last_logits, gen_config, &gen, seed.wrapping_add(step as u64));
            if let Some(eos) = gen_config.eos_token_id { if next == eos { break; } }
            gen.push(next);
            if !gen_config.stop_sequences.is_empty() && matches_stop_sequence(&gen, &gen_config.stop_sequences) { break; }
            let ids = GpuTensor::from_host(device, &[next], Shape::from_static(&[1]), DType::I32)?;
            sampling::embedding_f16(&self.cache, device, &self.embed_tokens, &ids, &mut bufs.hidden, 1, h)?;
            last_logits = self.forward_decode(device, &mut bufs, &mut kv_caches, pos)?;
            device.synchronize()?;
            pos += 1;
        }

        let dt = ds.elapsed();
        let kb: usize = kv_caches.iter().map(|kv| kv.k.size_bytes() + kv.v.size_bytes()).sum();
        Ok(GenerationResult {
            tokens: gen.clone(), prefill_time: pt, decode_time: dt,
            tokens_generated: gen.len(), prefill_tokens: prompt_ids.len(),
            tokens_per_sec: if dt.as_secs_f64() > 0.0 { gen.len() as f64 / dt.as_secs_f64() } else { 0.0 },
            kv_cache_memory_bytes: kb,
        })
    }
}

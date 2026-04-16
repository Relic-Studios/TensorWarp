//! All-VRAM MoE generation engine for Gemma 4 26B-A4B.
//!
//! F16 attention/MLP (cuBLAS HGEMM) + Q4 experts (TW-Marlin or native GGUF).
//! Active per token: ~4B params → target 50+ tok/sec on RTX 4090.

use warp_ir::{DType, Shape};
use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::generate::{GenerateConfig, GenerationResult, sample_token, matches_stop_sequence};
use crate::kv_cache::LayerKVCache;
use crate::ops;
use crate::sampling;
use crate::tensor::GpuTensor;
use crate::transformer::{GemmaConfig, GemmaLayerAttentionConfig};

/// Per-layer weights — F16 for attention/MLP, Q4 for experts.
pub struct MoEQ4Layer {
    // Norms (F32)
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

    // Attention — F16 (cuBLAS HGEMM)
    pub wq: GpuTensor<half::f16>, pub wk: GpuTensor<half::f16>,
    pub wv: GpuTensor<half::f16>, pub wo: GpuTensor<half::f16>,

    // Dense MLP — F16
    pub w_gate: GpuTensor<half::f16>, pub w_up: GpuTensor<half::f16>,
    pub w_down: GpuTensor<half::f16>,

    // Router F32
    pub router_proj: GpuTensor<f32>,
    pub router_scale: GpuTensor<f32>,
    pub per_expert_scale: GpuTensor<f32>,

    // Expert weights — native GGUF (Q4_K/Q8_0) or TW-Marlin (SafeTensors)
    pub experts_gu_raw: GpuTensor<u8>,
    pub gu_bytes_per_expert: usize,
    pub experts_d_raw: GpuTensor<u8>,
    pub d_bytes_per_expert: usize,
    pub d_block_bytes: u32,
    pub d_block_elems: u32,
    pub use_native_gguf_experts: bool,
    pub experts_gu_scales: Option<GpuTensor<half::f16>>,
    pub experts_d_scales: Option<GpuTensor<half::f16>>,
    pub gu_scales_per_expert: usize,
    pub d_scales_per_expert: usize,
    pub expert_gu_k: u32,
    pub expert_gu_n: u32,
    pub expert_d_k: u32,
    pub expert_d_n: u32,
}

/// Pre-allocated decode buffers — fused RMSNorm→F16 + GemmEx F16→F32 eliminates ~236 cast kernels/token.
struct Buffers {
    // F32 hidden state pipeline
    hidden: GpuTensor<f32>,
    hidden_scaled: GpuTensor<f32>,
    normed_f16: GpuTensor<half::f16>,  // fused rmsnorm_f16out (was: normed F32 + cast + normed_f16)
    // Attention intermediates — Q/K/V now come out F32 directly from GemmEx
    q: GpuTensor<f32>, k: GpuTensor<f32>, v: GpuTensor<f32>,
    v_normed: GpuTensor<f32>,
    q_n: GpuTensor<f32>, k_n: GpuTensor<f32>,
    q_rope: GpuTensor<f32>, k_rope: GpuTensor<f32>,
    attn_out: GpuTensor<f32>,
    attn_out_f16: GpuTensor<half::f16>,  // still needed for O projection input
    attn_proj: GpuTensor<f32>,           // GemmEx F32 output (no attn_proj_f16)
    post_attn: GpuTensor<f32>,
    residual: GpuTensor<f32>,
    // Dense MLP — fused: rmsnorm_f16out + GemmEx eliminates 4 casts/layer
    ffn_in_f16: GpuTensor<half::f16>,    // fused rmsnorm_f16out (no ffn_in F32)
    dense_gate: GpuTensor<f32>, dense_up: GpuTensor<f32>,  // GemmEx F32 out (no F16 versions)
    dense_geglu: GpuTensor<f32>,
    dense_geglu_f16: GpuTensor<half::f16>,  // still needed for down projection input
    dense_out: GpuTensor<f32>,              // GemmEx F32 out (no dense_out_f16)
    // MoE — batched: all 8 experts in parallel
    moe_in: GpuTensor<f32>,
    moe_accumulated: GpuTensor<f32>,
    expert_gate_up_all: GpuTensor<f32>,   // [8, 2*moe_dim] for batched gate+up
    expert_geglu_all: GpuTensor<f32>,     // [8, moe_dim] for batched split_geglu
    // Legacy single-expert buffers (for TW-Marlin path which isn't batched yet)
    expert_out: GpuTensor<f32>,
    expert_gate_up: GpuTensor<f32>,
    expert_geglu: GpuTensor<f32>,
    // Router pre-allocated
    router_normed: GpuTensor<f32>,
    router_scaled: GpuTensor<f32>,
    router_scaled2: GpuTensor<f32>,
    router_logits: GpuTensor<f32>,
    router_probs: GpuTensor<f32>,
    topk_ids: GpuTensor<f32>,
    topk_weights: GpuTensor<f32>,
    // Combine
    dense_normed: GpuTensor<f32>,
    moe_normed: GpuTensor<f32>,
    combined: GpuTensor<f32>,
    final_normed: GpuTensor<f32>,
    output: GpuTensor<f32>,
    output_scaled: GpuTensor<f32>,
    // LM head — fused: rmsnorm_f16out + GemmEx eliminates 2 casts
    lm_normed_f16: GpuTensor<half::f16>,  // fused (no lm_normed F32)
    lm_logits: GpuTensor<f32>,            // GemmEx F32 out (no lm_logits_f16)
    lm_capped: GpuTensor<f32>,
    pos_buf: cudarc::driver::CudaSlice<u32>,                // device-side position for graph capture
    cache_len_buf: cudarc::driver::CudaSlice<u32>,
    cache_len_bufs: Vec<cudarc::driver::CudaSlice<u32>>,  // per-layer for CUDA graph capture
    attn_scratch: crate::kv_cache::FlashDecodeScratch,
    // GPU-side expert dispatch (eliminates 26 DtoH syncs per token)
    expert_gu_offsets: cudarc::driver::CudaSlice<u32>,   // [top_k] gate+up byte offsets
    expert_d_offsets: cudarc::driver::CudaSlice<u32>,    // [top_k] down byte offsets
    expert_gu_scale_offsets: cudarc::driver::CudaSlice<u32>, // [top_k] TW-Marlin scale offsets
    expert_d_scale_offsets: cudarc::driver::CudaSlice<u32>,  // [top_k] TW-Marlin scale offsets
}

pub struct MoEQ4Engine {
    pub config: GemmaConfig,
    pub layer_configs: Vec<GemmaLayerAttentionConfig>,
    pub embed_tokens: GpuTensor<half::f16>,
    pub final_norm: GpuTensor<f32>,
    pub cache: KernelCache,
    pub layers: Vec<MoEQ4Layer>,
    pub weights_reordered: bool,
}

impl MoEQ4Engine {
    fn allocate_buffers(&self, device: &WarpDevice) -> Result<Buffers, DeviceError> {
        let h = self.config.hidden_size as usize;
        let max_q = self.layer_configs.iter().map(|lc| (self.config.num_heads * lc.head_dim) as usize).max().unwrap_or(h);
        let max_kv = self.layer_configs.iter().map(|lc| self.config.kv_dim_for_layer(lc) as usize).max().unwrap_or(h);
        let ffn = self.config.ffn_dim as usize;
        let moe_dim = 704usize;

        Ok(Buffers {
            hidden: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            hidden_scaled: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            normed_f16: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F16)?,
            q: GpuTensor::zeros(device, Shape::from_static(&[1, max_q]), DType::F32)?,
            k: GpuTensor::zeros(device, Shape::from_static(&[1, max_kv]), DType::F32)?,
            v: GpuTensor::zeros(device, Shape::from_static(&[1, max_kv]), DType::F32)?,
            v_normed: GpuTensor::zeros(device, Shape::from_static(&[1, max_kv]), DType::F32)?,
            q_n: GpuTensor::zeros(device, Shape::from_static(&[1, max_q]), DType::F32)?,
            k_n: GpuTensor::zeros(device, Shape::from_static(&[1, max_kv]), DType::F32)?,
            q_rope: GpuTensor::zeros(device, Shape::from_static(&[1, max_q]), DType::F32)?,
            k_rope: GpuTensor::zeros(device, Shape::from_static(&[1, max_kv]), DType::F32)?,
            attn_out: GpuTensor::zeros(device, Shape::from_static(&[1, max_q]), DType::F32)?,
            attn_out_f16: GpuTensor::zeros(device, Shape::from_static(&[1, max_q]), DType::F16)?,
            attn_proj: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            post_attn: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            residual: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            ffn_in_f16: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F16)?,
            dense_gate: GpuTensor::zeros(device, Shape::from_static(&[1, ffn]), DType::F32)?,
            dense_up: GpuTensor::zeros(device, Shape::from_static(&[1, ffn]), DType::F32)?,
            dense_geglu: GpuTensor::zeros(device, Shape::from_static(&[1, ffn]), DType::F32)?,
            dense_geglu_f16: GpuTensor::zeros(device, Shape::from_static(&[1, ffn]), DType::F16)?,
            dense_out: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            moe_in: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            moe_accumulated: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            expert_gate_up_all: GpuTensor::zeros(device, Shape::from_static(&[8, 2*moe_dim]), DType::F32)?,
            expert_geglu_all: GpuTensor::zeros(device, Shape::from_static(&[8, moe_dim]), DType::F32)?,
            expert_out: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            expert_gate_up: GpuTensor::zeros(device, Shape::from_static(&[1, 2*moe_dim]), DType::F32)?,
            expert_geglu: GpuTensor::zeros(device, Shape::from_static(&[1, moe_dim]), DType::F32)?,
            router_normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            router_scaled: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            router_scaled2: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            router_logits: GpuTensor::zeros(device, Shape::from_static(&[1, 128]), DType::F32)?,
            router_probs: GpuTensor::zeros(device, Shape::from_static(&[1, 128]), DType::F32)?,
            topk_ids: GpuTensor::zeros(device, Shape::from_static(&[1, 8]), DType::F32)?,
            topk_weights: GpuTensor::zeros(device, Shape::from_static(&[1, 8]), DType::F32)?,
            dense_normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            moe_normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            combined: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            final_normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            output: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            output_scaled: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            lm_normed_f16: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F16)?,
            lm_logits: GpuTensor::zeros(device, Shape::from_static(&[1, self.config.vocab_size as usize]), DType::F32)?,
            lm_capped: GpuTensor::zeros(device, Shape::from_static(&[1, self.config.vocab_size as usize]), DType::F32)?,
            pos_buf: device.alloc_zeros::<u32>(1)?,
            cache_len_buf: device.alloc_zeros::<u32>(1)?,
            cache_len_bufs: (0..self.layers.len()).map(|_| device.alloc_zeros::<u32>(1)).collect::<Result<Vec<_>,_>>()?,
            expert_gu_offsets: device.alloc_zeros::<u32>(8)?,
            expert_d_offsets: device.alloc_zeros::<u32>(8)?,
            expert_gu_scale_offsets: device.alloc_zeros::<u32>(8)?,
            expert_d_scale_offsets: device.alloc_zeros::<u32>(8)?,
            attn_scratch: crate::kv_cache::FlashDecodeScratch::new(
                device, self.config.num_heads,
                self.layer_configs.iter().map(|lc|
                    if lc.window_size > 0 { lc.window_size } else { 2048 }
                ).max().unwrap_or(2048),
                self.layer_configs.iter().map(|lc| lc.head_dim).max().unwrap_or(256),
            )?,
        })
    }

    fn forward_decode(
        &self, device: &WarpDevice, b: &mut Buffers,
        kv_caches: &mut [LayerKVCache], pos: u32,
    ) -> Result<Vec<f32>, DeviceError> {
        let h = self.config.hidden_size;
        let ffn = self.config.ffn_dim;
        let moe_dim = 704u32;
        let top_k = 8u32;
        let num_experts = 128u32;

        // Embedding scaling: x *= sqrt(hidden_size)
        ops::mul_scalar(&self.cache, device, &b.hidden, &mut b.hidden_scaled, (h as f32).sqrt())?;

        let mut last_cache_len = u32::MAX;
        for (i, layer) in self.layers.iter().enumerate() {
            let lc = &self.layer_configs[i];
            let d = lc.head_dim;
            let q_dim = self.config.num_heads * d;
            let kv_dim = self.config.kv_dim_for_layer(lc);
            let x = if i == 0 { &b.hidden_scaled } else { &b.output_scaled };

            // ── 1. Attention
            ops::rmsnorm_f16out(&self.cache, device, x, &layer.attn_norm, &mut b.normed_f16, h, self.config.norm_eps)?;

            crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.normed_f16, &layer.wq, &mut b.q, 1, q_dim, h)?;
            crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.normed_f16, &layer.wk, &mut b.k, 1, kv_dim, h)?;

            if self.config.k_eq_v && lc.is_global {
                ops::mul_scalar(&self.cache, device, &b.k, &mut b.v, 1.0)?;
            } else {
                crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.normed_f16, &layer.wv, &mut b.v, 1, kv_dim, h)?;
            }

            // QK-norm → V-norm → RoPE (Gemma 4: norm BEFORE RoPE)
            ops::rmsnorm(&self.cache, device, &b.q, &layer.q_norm, &mut b.q_n, d, self.config.norm_eps)?;
            ops::rmsnorm(&self.cache, device, &b.k, &layer.k_norm, &mut b.k_n, d, self.config.norm_eps)?;
            ops::rmsnorm_no_weight(&self.cache, device, &b.v, &mut b.v_normed, d, self.config.norm_eps)?;

            let rotary_dim = (d as f32 * lc.partial_rotary_factor) as u32;
            if rotary_dim < d {
                crate::rope::rope_partial(&self.cache, device, &b.q_n, &mut b.q_rope, self.config.num_heads, 1, d, rotary_dim, lc.rope_theta, pos)?;
                crate::rope::rope_partial(&self.cache, device, &b.k_n, &mut b.k_rope, lc.num_kv_heads, 1, d, rotary_dim, lc.rope_theta, pos)?;
            } else {
                crate::rope::rope(&self.cache, device, &b.q_n, &mut b.q_rope, self.config.num_heads, 1, d, lc.rope_theta, pos)?;
                crate::rope::rope(&self.cache, device, &b.k_n, &mut b.k_rope, lc.num_kv_heads, 1, d, lc.rope_theta, pos)?;
            }

            // KV cache + attention
            let kv = &mut kv_caches[i];
            let cp = if lc.window_size > 0 { pos % lc.window_size } else { pos };
            kv.prefill_at_offset(&self.cache, device, &b.k_rope, &b.v_normed, 1, cp)?;
            kv.len = (pos + 1).min(kv.max_seq_len);
            let win = if lc.window_size > 0 { lc.window_size } else { 0 };
            if kv.len != last_cache_len {
                device.htod_copy(&[kv.len], &mut b.cache_len_buf)?;
                last_cache_len = kv.len;
            }
            crate::kv_cache::decode_attention_flash_prealloc(
                &self.cache, device, &b.q_rope, kv, &mut b.attn_out,
                &mut b.attn_scratch,
                self.config.num_heads, lc.num_kv_heads, d, &b.cache_len_buf, win,
                0.0, 1.0)?;

            // O projection: cast + GemmEx F16→F32 (was: cast + GEMM_f16 + cast — saves 1 cast)
            crate::fp16::cast_f32_to_f16(&self.cache, device, &b.attn_out, &mut b.attn_out_f16)?;
            crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.attn_out_f16, &layer.wo, &mut b.attn_proj, 1, h, q_dim)?;
            // Fused: residual = x + rmsnorm(attn_proj, post_attn_norm)  (2 kernels → 1)
            ops::fused_rmsnorm_add(&self.cache, device, x, &b.attn_proj, &layer.post_attn_norm, &mut b.residual, h, self.config.norm_eps)?;

            // ── 2+3. Dense MLP + MoE
            let scalar_root = 1.0 / (h as f32).sqrt();
            ops::fused_triple_norm(&self.cache, device,
                &b.residual, &layer.pre_ffn_norm, &layer.pre_ffn_norm_2, &layer.router_scale,
                &mut b.ffn_in_f16, &mut b.router_scaled2, &mut b.moe_in,
                h, self.config.norm_eps, scalar_root)?;

            // Dense MLP GEMMs
            crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.ffn_in_f16, &layer.w_gate, &mut b.dense_gate, 1, ffn, h)?;
            crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.ffn_in_f16, &layer.w_up, &mut b.dense_up, 1, ffn, h)?;
            ops::fused_gelu_mul(&self.cache, device, &b.dense_gate, &b.dense_up, &mut b.dense_geglu)?;
            crate::fp16::cast_f32_to_f16(&self.cache, device, &b.dense_geglu, &mut b.dense_geglu_f16)?;
            crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.dense_geglu_f16, &layer.w_down, &mut b.dense_out, 1, h, ffn)?;

            // Router
            crate::cublas_gemm::gemm_cublas_f32_transB(device,
                &b.router_scaled2, &layer.router_proj, &mut b.router_logits, 1, num_experts, h)?;
            crate::sampling::softmax(&self.cache, device, &b.router_logits, &mut b.router_probs, 1, num_experts)?;

            ops::router_topk(&self.cache, device,
                &b.router_probs, &layer.per_expert_scale,
                &mut b.topk_ids, &mut b.topk_weights,
                num_experts, top_k)?;

            // MoE expert dispatch
            device.stream.memset_zeros(&mut b.moe_accumulated.data)
                .map_err(|e| DeviceError::Memory(format!("memset: {e}")))?;

            if layer.use_native_gguf_experts {
                // ── Batched GGUF path: group-major layout + scalar kernels ──
                // 1. All 8 gate+up GEMMs in parallel (group-major coalesced)
                crate::moe_batched::batched_q4k_gate_up(&self.cache, device,
                    &b.moe_in, &layer.experts_gu_raw, &b.topk_ids,
                    &mut b.expert_gate_up_all,
                    layer.gu_bytes_per_expert as u32,
                    layer.expert_gu_n, layer.expert_gu_k, top_k)?;

                // 2. All 8 split_geglus in parallel
                crate::moe_batched::batched_split_geglu(&self.cache, device,
                    &b.expert_gate_up_all, &mut b.expert_geglu_all,
                    moe_dim, top_k)?;

                // 3. All 8 down GEMMs + weighted axpy fused
                if layer.d_block_bytes == 34 {
                    crate::moe_batched::batched_q8_down_axpy(&self.cache, device,
                        &b.expert_geglu_all, &layer.experts_d_raw,
                        &b.topk_ids, &b.topk_weights, &mut b.moe_accumulated,
                        layer.d_bytes_per_expert as u32,
                        layer.expert_d_n, layer.expert_d_k, top_k)?;
                } else {
                    crate::moe_batched::batched_q5_down_axpy(&self.cache, device,
                        &b.expert_geglu_all, &layer.experts_d_raw,
                        &b.topk_ids, &b.topk_weights, &mut b.moe_accumulated,
                        layer.d_bytes_per_expert as u32,
                        layer.expert_d_n, layer.expert_d_k, top_k)?;
                }
            } else {
                // ── Batched TW-Marlin path: 3 launches instead of 32 + DtoH sync ──
                // 1. All 8 gate+up GEMMs in parallel
                crate::moe_batched_marlin::batched_marlin_gate_up(&self.cache, device,
                    &b.moe_in,
                    &layer.experts_gu_raw,
                    layer.experts_gu_scales.as_ref().unwrap(),
                    &b.topk_ids,
                    &mut b.expert_gate_up_all,
                    layer.gu_bytes_per_expert as u32,
                    layer.gu_scales_per_expert as u32,
                    layer.expert_gu_n, layer.expert_gu_k, top_k)?;

                // 2. All 8 split_geglus in parallel
                crate::moe_batched_marlin::batched_marlin_split_geglu(&self.cache, device,
                    &b.expert_gate_up_all, &mut b.expert_geglu_all,
                    moe_dim, top_k)?;

                // 3. All 8 down GEMMs + weighted axpy fused
                crate::moe_batched_marlin::batched_marlin_down_axpy(&self.cache, device,
                    &b.expert_geglu_all,
                    &layer.experts_d_raw,
                    layer.experts_d_scales.as_ref().unwrap(),
                    &b.topk_ids, &b.topk_weights,
                    &mut b.moe_accumulated,
                    layer.d_bytes_per_expert as u32,
                    layer.d_scales_per_expert as u32,
                    layer.expert_d_n, layer.expert_d_k, top_k)?;
            }

            // ── 4. Combine
            ops::fused_moe_combine(&self.cache, device,
                &mut b.output_scaled, &mut b.output,
                &b.dense_out, &b.moe_accumulated, &b.residual,
                &layer.post_ffn_norm_1, &layer.post_ffn_norm_2, &layer.post_ffn_norm,
                h, self.config.norm_eps, layer.layer_scalar)?;
        }

        // ── LM head (fused: rmsnorm_f16out + GemmEx — saves 2 casts) ──
        let vocab = self.config.vocab_size;
        ops::rmsnorm_f16out(&self.cache, device, &b.output_scaled, &self.final_norm, &mut b.lm_normed_f16, h, self.config.norm_eps)?;
        crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.lm_normed_f16, &self.embed_tokens, &mut b.lm_logits, 1, vocab, h)?;

        if self.config.final_logit_softcapping > 0.0 {
            ops::logit_softcap(&self.cache, device, &b.lm_logits, &mut b.lm_capped, self.config.final_logit_softcapping)?;
            return b.lm_capped.to_host(device);
        }
        b.lm_logits.to_host(device)
    }

    /// Graph-capturable forward pass: all position-dependent ops read from device buffers.
    /// Does NOT include embedding lookup or LM head (those run outside the graph).
    fn forward_graph_capturable(
        &self, device: &WarpDevice, b: &mut Buffers,
        kv_caches: &mut [LayerKVCache],
    ) -> Result<(), DeviceError> {
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

            // Attention
            ops::rmsnorm_f16out(&self.cache, device, x, &layer.attn_norm, &mut b.normed_f16, h, self.config.norm_eps)?;
            crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.normed_f16, &layer.wq, &mut b.q, 1, q_dim, h)?;
            crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.normed_f16, &layer.wk, &mut b.k, 1, kv_dim, h)?;
            if self.config.k_eq_v && lc.is_global {
                ops::mul_scalar(&self.cache, device, &b.k, &mut b.v, 1.0)?;
            } else {
                crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.normed_f16, &layer.wv, &mut b.v, 1, kv_dim, h)?;
            }

            ops::rmsnorm(&self.cache, device, &b.q, &layer.q_norm, &mut b.q_n, d, self.config.norm_eps)?;
            ops::rmsnorm(&self.cache, device, &b.k, &layer.k_norm, &mut b.k_n, d, self.config.norm_eps)?;
            ops::rmsnorm_no_weight(&self.cache, device, &b.v, &mut b.v_normed, d, self.config.norm_eps)?;

            // RoPE with device-pos (graph-safe)
            let rotary_dim = (d as f32 * lc.partial_rotary_factor) as u32;
            if rotary_dim < d {
                crate::rope::rope_partial_device_pos(&self.cache, device, &b.q_n, &mut b.q_rope,
                    self.config.num_heads, 1, d, rotary_dim, lc.rope_theta, &b.pos_buf)?;
                crate::rope::rope_partial_device_pos(&self.cache, device, &b.k_n, &mut b.k_rope,
                    lc.num_kv_heads, 1, d, rotary_dim, lc.rope_theta, &b.pos_buf)?;
            } else {
                crate::rope::rope_device_pos(&self.cache, device, &b.q_n, &mut b.q_rope,
                    self.config.num_heads, 1, d, lc.rope_theta, &b.pos_buf)?;
                crate::rope::rope_device_pos(&self.cache, device, &b.k_n, &mut b.k_rope,
                    lc.num_kv_heads, 1, d, lc.rope_theta, &b.pos_buf)?;
            }

            // KV cache with device-pos (graph-safe)
            let kv = &mut kv_caches[i];
            let win = if lc.window_size > 0 { lc.window_size } else { 0 };
            kv.append_device_pos(&self.cache, device, &b.k_rope, &b.v_normed, &b.pos_buf, win)?;

            crate::kv_cache::decode_attention_flash_prealloc(
                &self.cache, device, &b.q_rope, kv, &mut b.attn_out,
                &mut b.attn_scratch,
                self.config.num_heads, lc.num_kv_heads, d, &b.cache_len_bufs[i], win,
                0.0, 1.0)?;

            crate::fp16::cast_f32_to_f16(&self.cache, device, &b.attn_out, &mut b.attn_out_f16)?;
            crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.attn_out_f16, &layer.wo, &mut b.attn_proj, 1, h, q_dim)?;
            ops::fused_rmsnorm_add(&self.cache, device, x, &b.attn_proj, &layer.post_attn_norm, &mut b.residual, h, self.config.norm_eps)?;

            // Dense MLP + MoE (fused triple-norm)
            let scalar_root = 1.0 / (h as f32).sqrt();
            ops::fused_triple_norm(&self.cache, device,
                &b.residual, &layer.pre_ffn_norm, &layer.pre_ffn_norm_2, &layer.router_scale,
                &mut b.ffn_in_f16, &mut b.router_scaled2, &mut b.moe_in,
                h, self.config.norm_eps, scalar_root)?;

            crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.ffn_in_f16, &layer.w_gate, &mut b.dense_gate, 1, ffn, h)?;
            crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.ffn_in_f16, &layer.w_up, &mut b.dense_up, 1, ffn, h)?;
            ops::fused_gelu_mul(&self.cache, device, &b.dense_gate, &b.dense_up, &mut b.dense_geglu)?;
            crate::fp16::cast_f32_to_f16(&self.cache, device, &b.dense_geglu, &mut b.dense_geglu_f16)?;
            crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.dense_geglu_f16, &layer.w_down, &mut b.dense_out, 1, h, ffn)?;

            crate::cublas_gemm::gemm_cublas_f32_transB(device,
                &b.router_scaled2, &layer.router_proj, &mut b.router_logits, 1, num_experts, h)?;
            crate::sampling::softmax(&self.cache, device, &b.router_logits, &mut b.router_probs, 1, num_experts)?;
            ops::router_topk(&self.cache, device,
                &b.router_probs, &layer.per_expert_scale,
                &mut b.topk_ids, &mut b.topk_weights, num_experts, top_k)?;

            device.stream.memset_zeros(&mut b.moe_accumulated.data)
                .map_err(|e| DeviceError::Memory(format!("memset: {e}")))?;

            if layer.use_native_gguf_experts {
                crate::moe_batched::batched_q4k_gate_up(&self.cache, device,
                    &b.moe_in, &layer.experts_gu_raw, &b.topk_ids,
                    &mut b.expert_gate_up_all, layer.gu_bytes_per_expert as u32,
                    layer.expert_gu_n, layer.expert_gu_k, top_k)?;
                crate::moe_batched::batched_split_geglu(&self.cache, device,
                    &b.expert_gate_up_all, &mut b.expert_geglu_all, moe_dim, top_k)?;
                if layer.d_block_bytes == 34 {
                    crate::moe_batched::batched_q8_down_axpy(&self.cache, device,
                        &b.expert_geglu_all, &layer.experts_d_raw,
                        &b.topk_ids, &b.topk_weights, &mut b.moe_accumulated,
                        layer.d_bytes_per_expert as u32, layer.expert_d_n, layer.expert_d_k, top_k)?;
                } else {
                    crate::moe_batched::batched_q5_down_axpy(&self.cache, device,
                        &b.expert_geglu_all, &layer.experts_d_raw,
                        &b.topk_ids, &b.topk_weights, &mut b.moe_accumulated,
                        layer.d_bytes_per_expert as u32, layer.expert_d_n, layer.expert_d_k, top_k)?;
                }
            } else {
                crate::moe_batched_marlin::batched_marlin_gate_up(&self.cache, device,
                    &b.moe_in, &layer.experts_gu_raw,
                    layer.experts_gu_scales.as_ref().unwrap(), &b.topk_ids,
                    &mut b.expert_gate_up_all,
                    layer.gu_bytes_per_expert as u32, layer.gu_scales_per_expert as u32,
                    layer.expert_gu_n, layer.expert_gu_k, top_k)?;
                crate::moe_batched_marlin::batched_marlin_split_geglu(&self.cache, device,
                    &b.expert_gate_up_all, &mut b.expert_geglu_all, moe_dim, top_k)?;
                crate::moe_batched_marlin::batched_marlin_down_axpy(&self.cache, device,
                    &b.expert_geglu_all, &layer.experts_d_raw,
                    layer.experts_d_scales.as_ref().unwrap(),
                    &b.topk_ids, &b.topk_weights, &mut b.moe_accumulated,
                    layer.d_bytes_per_expert as u32, layer.d_scales_per_expert as u32,
                    layer.expert_d_n, layer.expert_d_k, top_k)?;
            }

            ops::fused_moe_combine(&self.cache, device,
                &mut b.output_scaled, &mut b.output,
                &b.dense_out, &b.moe_accumulated, &b.residual,
                &layer.post_ffn_norm_1, &layer.post_ffn_norm_2, &layer.post_ffn_norm,
                h, self.config.norm_eps, layer.layer_scalar)?;
        }
        Ok(())
    }

    /// Reorder GGUF expert weight blocks from column-major (n,kb) to group-major (kb,n).
    /// This gives 10-20x better memory coalescing for warp-cooperative GEMV kernels.
    /// Runs once, replaces expert weight buffers in-place via temp allocation.
    fn reorder_expert_weights(&mut self, device: &WarpDevice) -> Result<(), DeviceError> {
        if self.weights_reordered { return Ok(()); }
        eprintln!("[moe] Reordering expert weights to group-major layout...");
        let rs = std::time::Instant::now();
        for layer in self.layers.iter_mut() {
            if !layer.use_native_gguf_experts { continue; }
            let gu_nkb = layer.expert_gu_k / 256;
            let d_nkb = layer.expert_d_k / 32;

            // Reorder the entire gate+up buffer (all 128 experts concatenated)
            let gu_reordered = ops::reorder_expert_blocks(&self.cache, device,
                &layer.experts_gu_raw, layer.expert_gu_n, gu_nkb, 144)?;
            layer.experts_gu_raw = gu_reordered;

            // Reorder the entire down buffer
            let d_reordered = ops::reorder_expert_blocks(&self.cache, device,
                &layer.experts_d_raw, layer.expert_d_n, d_nkb, layer.d_block_bytes)?;
            layer.experts_d_raw = d_reordered;
        }
        device.synchronize()?;
        self.weights_reordered = true;
        eprintln!("[moe] Weight reorder complete ({:.1}s)", rs.elapsed().as_secs_f64());
        Ok(())
    }

    /// Run LM head eagerly (outside CUDA graph).
    fn run_lm_head(&self, device: &WarpDevice, b: &mut Buffers) -> Result<Vec<f32>, DeviceError> {
        let h = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        ops::rmsnorm_f16out(&self.cache, device, &b.output_scaled, &self.final_norm, &mut b.lm_normed_f16, h, self.config.norm_eps)?;
        crate::cublas_gemm::gemm_cublas_f16in_f32out_transB(device, &b.lm_normed_f16, &self.embed_tokens, &mut b.lm_logits, 1, vocab, h)?;
        if self.config.final_logit_softcapping > 0.0 {
            ops::logit_softcap(&self.cache, device, &b.lm_logits, &mut b.lm_capped, self.config.final_logit_softcapping)?;
            return b.lm_capped.to_host(device);
        }
        b.lm_logits.to_host(device)
    }

    pub fn generate(
        &mut self, device: &WarpDevice, prompt_ids: &[i32],
        gen_config: &GenerateConfig, max_seq_len: u32,
    ) -> Result<GenerationResult, DeviceError> {
        self.reorder_expert_weights(device)?;
        let h = self.config.hidden_size;
        let mut kv_caches: Vec<LayerKVCache> = self.layer_configs.iter().map(|lc| {
            let kv_dim = self.config.kv_dim_for_layer(lc);
            let cl = if lc.window_size > 0 { lc.window_size.min(max_seq_len) } else { max_seq_len };
            LayerKVCache::new(device, cl, kv_dim).unwrap()
        }).collect();

        let mut bufs = self.allocate_buffers(device)?;
        let mut ids_gpu = GpuTensor::from_host(device, &[0i32], Shape::from_static(&[1]), DType::I32)?;
        eprintln!("[moe] Decode buffers allocated");

        // ── Prefill (eager, no graph) ──
        let ps = std::time::Instant::now();
        let mut last_logits = Vec::new();
        for (t, &tok) in prompt_ids.iter().enumerate() {
            device.htod_copy(&[tok], &mut ids_gpu.data)?;
            sampling::embedding_f16(&self.cache, device, &self.embed_tokens, &ids_gpu, &mut bufs.hidden, 1, h)?;
            last_logits = self.forward_decode(device, &mut bufs, &mut kv_caches, t as u32)?;
        }
        device.synchronize()?;
        let pt = ps.elapsed();

        // ── Decode with CUDA graph ──
        let ds = std::time::Instant::now();
        let mut gen = Vec::new();
        let mut pos = prompt_ids.len() as u32;
        let seed: u64 = prompt_ids.iter().fold(42u64, |a, &t| a.wrapping_mul(6364136223846793005).wrapping_add(t as u64));

        // Step 0: first decode token (eager — pre-compiles all kernels + warms caches)
        let next0 = sample_token(&last_logits, gen_config, &gen, seed);
        if gen_config.eos_token_id.map_or(false, |eos| next0 == eos) {
            let dt = ds.elapsed();
            let kb: usize = kv_caches.iter().map(|kv| kv.k.size_bytes() + kv.v.size_bytes()).sum();
            return Ok(GenerationResult { tokens: gen.clone(), prefill_time: pt, decode_time: dt,
                tokens_generated: gen.len(), prefill_tokens: prompt_ids.len(),
                tokens_per_sec: 0.0, kv_cache_memory_bytes: kb });
        }
        gen.push(next0);
        // Run first decode eagerly to pre-compile all NVRTC kernels
        device.htod_copy(&[next0], &mut ids_gpu.data)?;
        sampling::embedding_f16(&self.cache, device, &self.embed_tokens, &ids_gpu, &mut bufs.hidden, 1, h)?;
        device.htod_copy(&[pos], &mut bufs.pos_buf)?;
        for (i, kv) in kv_caches.iter().enumerate() {
            device.htod_copy(&[kv.len], &mut bufs.cache_len_bufs[i])?;
        }
        self.forward_graph_capturable(device, &mut bufs, &mut kv_caches)?;
        last_logits = self.run_lm_head(device, &mut bufs)?;
        // Update KV cache lengths on host side
        for (i, kv) in kv_caches.iter_mut().enumerate() {
            let lc = &self.layer_configs[i];
            kv.len = (pos + 1).min(kv.max_seq_len);
        }
        pos += 1;

        // ── Capture CUDA graph ──
        let capture_stream = device.stream.context()
            .new_stream()
            .map_err(|e| DeviceError::Launch(format!("capture stream: {e}")))?;
        let graph_device = device.with_stream(capture_stream.clone());

        // Pre-set device buffers for capture
        device.htod_copy(&[pos], &mut bufs.pos_buf)?;
        for (i, kv) in kv_caches.iter().enumerate() {
            let new_len = (pos + 1).min(kv.max_seq_len);
            device.htod_copy(&[new_len], &mut bufs.cache_len_bufs[i])?;
        }
        graph_device.synchronize()?;

        // Run once eagerly on capture stream to establish buffer state
        sampling::embedding_f16(&self.cache, &graph_device, &self.embed_tokens, &ids_gpu, &mut bufs.hidden, 1, h)?;
        self.forward_graph_capturable(&graph_device, &mut bufs, &mut kv_caches)?;
        graph_device.synchronize()?;
        // Undo the KV cache side effects of the dry run
        for (i, kv) in kv_caches.iter_mut().enumerate() {
            let lc = &self.layer_configs[i];
            kv.len = (pos).min(kv.max_seq_len); // restore to pre-dry-run
        }

        // NOW capture
        let graph = crate::cuda_graph::GraphCapture::record_with_device(
            &graph_device, &capture_stream,
            || {
                sampling::embedding_f16(&self.cache, &graph_device, &self.embed_tokens, &ids_gpu, &mut bufs.hidden, 1, h)?;
                self.forward_graph_capturable(&graph_device, &mut bufs, &mut kv_caches)?;
                Ok(())
            },
        )?;
        eprintln!("[moe] CUDA graph captured!");

        // ── Decode loop with graph replay ──
        for step in 1..gen_config.max_tokens {
            let next = sample_token(&last_logits, gen_config, &gen, seed.wrapping_add(step as u64));
            if let Some(eos) = gen_config.eos_token_id { if next == eos { break; } }
            gen.push(next);
            if !gen_config.stop_sequences.is_empty() && matches_stop_sequence(&gen, &gen_config.stop_sequences) { break; }

            // Update device buffers (host → device copies BEFORE graph replay)
            graph_device.htod_copy(&[next], &mut ids_gpu.data)?;
            graph_device.htod_copy(&[pos], &mut bufs.pos_buf)?;
            // cache_len_bufs must reflect the length AFTER this step's KV append
            // (attention needs to read the newly-appended entry)
            for (i, kv) in kv_caches.iter().enumerate() {
                let new_len = (pos + 1).min(kv.max_seq_len);
                graph_device.htod_copy(&[new_len], &mut bufs.cache_len_bufs[i])?;
            }

            // Replay graph (embedding + all transformer layers)
            graph.replay()?;

            // LM head (outside graph)
            last_logits = self.run_lm_head(&graph_device, &mut bufs)?;

            // Update KV cache lengths on host side
            for kv in kv_caches.iter_mut() {
                kv.len = (pos + 1).min(kv.max_seq_len);
            }
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

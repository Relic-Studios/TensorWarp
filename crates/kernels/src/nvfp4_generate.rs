//! NVFP4 Gemma 4 31B generation engine.
//!
//! FP4 weights stay on VRAM (~14 GB). Per-GEMM: dequant FP4→F16, run HGEMM, discard F16.
//! Uses the same Gemma architecture as gemma_generate.rs but with FP4 weight format.

use warp_ir::{DType, Shape};
use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::generate::{GenerateConfig, GenerationResult, sample_token, matches_stop_sequence};
use crate::kv_cache::LayerKVCache;
use crate::ops;
use crate::sampling;
use crate::tensor::GpuTensor;
use crate::transformer::{GemmaConfig, GemmaLayerAttentionConfig};

/// Reexport FP4Weight from loader
pub use crate::nvfp4::dequant_fp4_to_f16;

/// Run one FP4 GEMM: dequant weight → F16, cast input → F16, HGEMM transB, cast output → F32
fn fp4_gemm(
    cache: &KernelCache,
    device: &WarpDevice,
    input_f32: &GpuTensor<f32>,
    packed: &GpuTensor<u8>,
    scales: &GpuTensor<u8>,
    global_scale: f32,
    output_f32: &mut GpuTensor<f32>,
    m: u32, n: u32, k: u32,
    // Pre-allocated temp buffers
    input_f16: &mut GpuTensor<half::f16>,
    weight_f16: &mut GpuTensor<half::f16>,
    output_f16: &mut GpuTensor<half::f16>,
) -> Result<(), DeviceError> {
    // 1. Dequant FP4 → F16 (weight is [n, k] stored as [n, k/2] packed)
    dequant_fp4_to_f16(cache, device, packed, scales, global_scale, weight_f16, n, k)?;

    // 2. Cast input F32 → F16
    crate::fp16::cast_f32_to_f16(cache, device, input_f32, input_f16)?;

    // 3. HGEMM transB: output[m,n] = input[m,k] × weight^T[k,n]
    // weight_f16 is [n, k], transB transposes to [k, n]
    crate::cublas_gemm::gemm_cublas_f16_transB(device, input_f16, weight_f16, output_f16, m, n, k)?;

    // 4. Cast output F16 → F32
    crate::fp16::cast_f16_to_f32(cache, device, output_f16, output_f32)?;

    Ok(())
}

/// Pre-allocated decode buffers.
struct FP4DecodeBuffers {
    hidden: GpuTensor<f32>,
    hidden_scaled: GpuTensor<f32>,
    normed: GpuTensor<f32>,
    q: GpuTensor<f32>,
    k: GpuTensor<f32>,
    v: GpuTensor<f32>,
    q_rope: GpuTensor<f32>,
    k_rope: GpuTensor<f32>,
    q_n: GpuTensor<f32>,
    k_n: GpuTensor<f32>,
    attn_out: GpuTensor<f32>,
    attn_proj: GpuTensor<f32>,
    post_attn: GpuTensor<f32>,
    residual: GpuTensor<f32>,
    ffn_in: GpuTensor<f32>,
    gate: GpuTensor<f32>,
    up: GpuTensor<f32>,
    geglu: GpuTensor<f32>,
    ffn_out: GpuTensor<f32>,
    combined: GpuTensor<f32>,
    output: GpuTensor<f32>,
    output_scaled: GpuTensor<f32>,
    lm_normed: GpuTensor<f32>,
    cache_len_buf: cudarc::driver::CudaSlice<u32>,

    // F16 temp buffers for FP4 GEMM (reused across all GEMMs)
    input_f16: GpuTensor<half::f16>,     // max([1,h], [1,q_dim], [1,ffn])
    weight_f16: GpuTensor<half::f16>,    // max weight size across all projections
    output_f16: GpuTensor<half::f16>,    // max output size
}

pub struct NVFP4GenerationEngine {
    pub config: GemmaConfig,
    pub layer_configs: Vec<GemmaLayerAttentionConfig>,
    pub embed_tokens: GpuTensor<half::f16>,
    pub final_norm: GpuTensor<f32>,
    pub cache: KernelCache,

    // Per-layer FP4 weights (on VRAM)
    pub layers: Vec<NVFP4LayerVRAM>,
}

/// Per-layer weights (references to loader's FP4 data)
pub struct NVFP4LayerVRAM {
    pub attn_norm: GpuTensor<f32>,
    pub post_attn_norm: GpuTensor<f32>,
    pub pre_ffn_norm: GpuTensor<f32>,
    pub post_ffn_norm: GpuTensor<f32>,
    pub q_norm: GpuTensor<f32>,
    pub k_norm: GpuTensor<f32>,
    pub layer_scalar: f32,

    // FP4 packed + scales for each projection
    pub wq_packed: GpuTensor<u8>, pub wq_scales: GpuTensor<u8>, pub wq_gs: f32, pub wq_rows: u32, pub wq_cols: u32,
    pub wk_packed: GpuTensor<u8>, pub wk_scales: GpuTensor<u8>, pub wk_gs: f32, pub wk_rows: u32, pub wk_cols: u32,
    pub wv_packed: Option<GpuTensor<u8>>, pub wv_scales: Option<GpuTensor<u8>>, pub wv_gs: f32, pub wv_rows: u32, pub wv_cols: u32,
    pub wo_packed: GpuTensor<u8>, pub wo_scales: GpuTensor<u8>, pub wo_gs: f32, pub wo_rows: u32, pub wo_cols: u32,
    pub wg_packed: GpuTensor<u8>, pub wg_scales: GpuTensor<u8>, pub wg_gs: f32, pub wg_rows: u32, pub wg_cols: u32,
    pub wu_packed: GpuTensor<u8>, pub wu_scales: GpuTensor<u8>, pub wu_gs: f32, pub wu_rows: u32, pub wu_cols: u32,
    pub wd_packed: GpuTensor<u8>, pub wd_scales: GpuTensor<u8>, pub wd_gs: f32, pub wd_rows: u32, pub wd_cols: u32,
}

impl NVFP4GenerationEngine {
    fn allocate_buffers(&self, device: &WarpDevice) -> Result<FP4DecodeBuffers, DeviceError> {
        let h = self.config.hidden_size as usize;
        let max_q = self.layer_configs.iter().map(|lc| (self.config.num_heads * lc.head_dim) as usize).max().unwrap_or(h);
        let max_kv = self.layer_configs.iter().map(|lc| self.config.kv_dim_for_layer(lc) as usize).max().unwrap_or(h);
        let ffn = self.config.ffn_dim as usize;
        let max_dim = h.max(max_q).max(ffn);
        // Max weight: FFN gate/up is [ffn, h] = largest projection
        let max_weight = ffn * h;
        let max_out = max_q.max(ffn).max(h);

        Ok(FP4DecodeBuffers {
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
            gate: GpuTensor::zeros(device, Shape::from_static(&[1, ffn]), DType::F32)?,
            up: GpuTensor::zeros(device, Shape::from_static(&[1, ffn]), DType::F32)?,
            geglu: GpuTensor::zeros(device, Shape::from_static(&[1, ffn]), DType::F32)?,
            ffn_out: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            combined: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            output: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            output_scaled: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            lm_normed: GpuTensor::zeros(device, Shape::from_static(&[1, h]), DType::F32)?,
            cache_len_buf: device.alloc_zeros::<u32>(1)?,
            input_f16: GpuTensor::zeros(device, Shape::from_static(&[1, max_dim]), DType::F16)?,
            weight_f16: GpuTensor::zeros(device, Shape::from_static(&[max_weight]), DType::F16)?,
            output_f16: GpuTensor::zeros(device, Shape::from_static(&[1, max_out]), DType::F16)?,
        })
    }

    fn forward_decode(
        &self, device: &WarpDevice, b: &mut FP4DecodeBuffers,
        kv_caches: &mut [LayerKVCache], pos: u32,
    ) -> Result<Vec<f32>, DeviceError> {
        let h = self.config.hidden_size;
        let ffn = self.config.ffn_dim;
        let embed_scale = (h as f32).sqrt();

        ops::mul_scalar(&self.cache, device, &b.hidden, &mut b.hidden_scaled, embed_scale)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let lc = &self.layer_configs[i];
            let d = lc.head_dim;
            let q_dim = self.config.num_heads * d;
            let kv_dim = self.config.kv_dim_for_layer(lc);

            let x = if i == 0 { &b.hidden_scaled } else { &b.output_scaled };

            // Attention norm
            ops::rmsnorm(&self.cache, device, x, &layer.attn_norm, &mut b.normed, h, self.config.norm_eps)?;

            // Q, K, V projections via FP4 GEMM
            fp4_gemm(&self.cache, device, &b.normed,
                &layer.wq_packed, &layer.wq_scales, layer.wq_gs,
                &mut b.q, 1, q_dim, h,
                &mut b.input_f16, &mut b.weight_f16, &mut b.output_f16)?;
            fp4_gemm(&self.cache, device, &b.normed,
                &layer.wk_packed, &layer.wk_scales, layer.wk_gs,
                &mut b.k, 1, kv_dim, h,
                &mut b.input_f16, &mut b.weight_f16, &mut b.output_f16)?;
            if let (Some(ref vp), Some(ref vs)) = (&layer.wv_packed, &layer.wv_scales) {
                fp4_gemm(&self.cache, device, &b.normed,
                    vp, vs, layer.wv_gs,
                    &mut b.v, 1, kv_dim, h,
                    &mut b.input_f16, &mut b.weight_f16, &mut b.output_f16)?;
            } else {
                ops::mul_scalar(&self.cache, device, &b.k, &mut b.v, 1.0)?;
            }

            // RoPE
            let rotary_dim = (d as f32 * lc.partial_rotary_factor) as u32;
            if rotary_dim < d && rotary_dim > 0 {
                crate::rope::rope_partial(&self.cache, device, &b.q, &mut b.q_rope,
                    self.config.num_heads, 1, d, rotary_dim, lc.rope_theta, pos)?;
                crate::rope::rope_partial(&self.cache, device, &b.k, &mut b.k_rope,
                    lc.num_kv_heads, 1, d, rotary_dim, lc.rope_theta, pos)?;
            } else {
                crate::rope::rope(&self.cache, device, &b.q, &mut b.q_rope,
                    self.config.num_heads, 1, d, lc.rope_theta, pos)?;
                crate::rope::rope(&self.cache, device, &b.k, &mut b.k_rope,
                    lc.num_kv_heads, 1, d, lc.rope_theta, pos)?;
            }

            // QK-norm
            ops::rmsnorm(&self.cache, device, &b.q_rope, &layer.q_norm, &mut b.q_n, d, self.config.norm_eps)?;
            ops::rmsnorm(&self.cache, device, &b.k_rope, &layer.k_norm, &mut b.k_n, d, self.config.norm_eps)?;

            // KV cache
            let kv = &mut kv_caches[i];
            let cache_pos = if lc.window_size > 0 { pos % lc.window_size } else { pos };
            kv.prefill_at_offset(&self.cache, device, &b.k_n, &b.v, 1, cache_pos)?;
            kv.len = (pos + 1).min(kv.max_seq_len);

            let window = if lc.window_size > 0 { lc.window_size } else { 0 };
            device.htod_copy(&[kv.len], &mut b.cache_len_buf)?;
            crate::kv_cache::decode_attention_flash_device_len_window(
                &self.cache, device, &b.q_n, kv, &mut b.attn_out,
                self.config.num_heads, lc.num_kv_heads, d, &b.cache_len_buf, window)?;

            // Output projection
            fp4_gemm(&self.cache, device, &b.attn_out,
                &layer.wo_packed, &layer.wo_scales, layer.wo_gs,
                &mut b.attn_proj, 1, h, q_dim,
                &mut b.input_f16, &mut b.weight_f16, &mut b.output_f16)?;

            // Post-attention norm + residual (Gemma architecture)
            ops::rmsnorm(&self.cache, device, &b.attn_proj, &layer.post_attn_norm, &mut b.post_attn, h, self.config.norm_eps)?;
            ops::add(&self.cache, device, x, &b.post_attn, &mut b.residual)?;

            // Dense MLP
            ops::rmsnorm(&self.cache, device, &b.residual, &layer.pre_ffn_norm, &mut b.ffn_in, h, self.config.norm_eps)?;
            fp4_gemm(&self.cache, device, &b.ffn_in,
                &layer.wg_packed, &layer.wg_scales, layer.wg_gs,
                &mut b.gate, 1, ffn, h,
                &mut b.input_f16, &mut b.weight_f16, &mut b.output_f16)?;
            fp4_gemm(&self.cache, device, &b.ffn_in,
                &layer.wu_packed, &layer.wu_scales, layer.wu_gs,
                &mut b.up, 1, ffn, h,
                &mut b.input_f16, &mut b.weight_f16, &mut b.output_f16)?;
            ops::fused_gelu_mul(&self.cache, device, &b.gate, &b.up, &mut b.geglu)?;
            fp4_gemm(&self.cache, device, &b.geglu,
                &layer.wd_packed, &layer.wd_scales, layer.wd_gs,
                &mut b.ffn_out, 1, h, ffn,
                &mut b.input_f16, &mut b.weight_f16, &mut b.output_f16)?;

            // Final norm + residual + layer_scalar
            ops::rmsnorm(&self.cache, device, &b.ffn_out, &layer.post_ffn_norm, &mut b.combined, h, self.config.norm_eps)?;
            ops::add(&self.cache, device, &b.residual, &b.combined, &mut b.output)?;
            ops::mul_scalar(&self.cache, device, &b.output, &mut b.output_scaled, layer.layer_scalar)?;
        }

        // Final norm + LM head (GPU F16 HGEMM transB with tied embedding)
        ops::rmsnorm(&self.cache, device, &b.output_scaled, &self.final_norm, &mut b.lm_normed, h, self.config.norm_eps)?;
        let vocab = self.config.vocab_size;
        let mut normed_f16 = GpuTensor::<half::f16>::zeros(device, Shape::from_static(&[1, h as usize]), DType::F16)?;
        crate::fp16::cast_f32_to_f16(&self.cache, device, &b.lm_normed, &mut normed_f16)?;
        let mut logits_f16 = GpuTensor::<half::f16>::zeros(device, Shape::from_static(&[1, vocab as usize]), DType::F16)?;
        crate::cublas_gemm::gemm_cublas_f16_transB(device, &normed_f16, &self.embed_tokens, &mut logits_f16, 1, vocab, h)?;
        let mut logits = GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, vocab as usize]), DType::F32)?;
        crate::fp16::cast_f16_to_f32(&self.cache, device, &logits_f16, &mut logits)?;
        logits.to_host(device)
    }

    pub fn generate(
        &self, device: &WarpDevice, prompt_ids: &[i32],
        gen_config: &GenerateConfig, max_seq_len: u32,
    ) -> Result<GenerationResult, DeviceError> {
        let h = self.config.hidden_size;
        let mut kv_caches: Vec<LayerKVCache> = self.layer_configs.iter().map(|lc| {
            let kv_dim = self.config.kv_dim_for_layer(lc);
            let cache_len = if lc.window_size > 0 { lc.window_size.min(max_seq_len) } else { max_seq_len };
            LayerKVCache::new(device, cache_len, kv_dim).unwrap()
        }).collect();

        let mut bufs = self.allocate_buffers(device)?;
        eprintln!("[nvfp4] Pre-allocated decode buffers");

        let prefill_start = std::time::Instant::now();
        let mut last_logits = Vec::new();
        for (t, &token) in prompt_ids.iter().enumerate() {
            let ids = GpuTensor::from_host(device, &[token], Shape::from_static(&[1]), DType::I32)?;
            sampling::embedding_f16(&self.cache, device, &self.embed_tokens, &ids, &mut bufs.hidden, 1, h)?;
            last_logits = self.forward_decode(device, &mut bufs, &mut kv_caches, t as u32)?;
        }
        device.synchronize()?;
        let prefill_time = prefill_start.elapsed();

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
            last_logits = self.forward_decode(device, &mut bufs, &mut kv_caches, pos)?;
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

//! All-VRAM Q4 MoE loader. ALL weights including experts quantized to Q4 on GPU.
//! Total VRAM: ~16.7 GB. No RAM streaming needed.

use warp_ir::{DType, Shape};
use warp_kernels::cache::KernelCache;
use warp_kernels::device::WarpDevice;
use warp_kernels::tensor::GpuTensor;
use warp_kernels::transformer::GemmaConfig;
use warp_kernels::moe_q4_generate::{MoEQ4Layer, MoEQ4Engine};

use crate::gemma::GemmaHFConfig;
use crate::safetensors_loader::{ShardedSafeTensorsLoader, LoaderError};

pub fn load_moe_q4(
    loader: &ShardedSafeTensorsLoader,
    hf_config: &GemmaHFConfig,
    device: &WarpDevice,
) -> Result<MoEQ4Engine, LoaderError> {
    let config = hf_config.to_gemma_config();
    let layer_configs = config.layer_configs();
    let h = config.hidden_size;
    let moe_dim = 704u32;

    let prefix = if loader.load_f32("model.language_model.embed_tokens.weight", device).is_ok() {
        "model.language_model"
    } else { "model" };
    eprintln!("[moe-q4] Prefix: {prefix}");

    // Embedding F16
    eprintln!("[moe-q4] Loading embedding F16...");
    let embed_f32 = loader.load_f32(&format!("{prefix}.embed_tokens.weight"), device)?;
    let kcache = KernelCache::new();
    let mut embed = GpuTensor::<half::f16>::zeros(device, embed_f32.shape.clone(), DType::F16)
        .map_err(|e| LoaderError::Device(e.to_string()))?;
    warp_kernels::fp16::cast_f32_to_f16(&kcache, device, &embed_f32, &mut embed)
        .map_err(|e| LoaderError::Device(e.to_string()))?;
    drop(embed_f32);

    let load_norm = |name: &str| -> Result<GpuTensor<f32>, LoaderError> {
        // Gemma 4 uses `x * weight` (NOT `x * (1+weight)` like Gemma 3)
        let t = loader.load_f32(name, device)?;
        Ok(t)
    };

    // Load weight as F16 for cuBLAS HGEMM (gemm_cublas_f16_transB expects [N, K])
    // SafeTensors stores as [N, K] row-major, same as what HGEMM needs — no transpose!
    let load_f16 = |name: &str, _k: u32, _n: u32| -> Result<GpuTensor<half::f16>, LoaderError> {
        let w_f32 = loader.load_f32(name, device)?;
        let kcache = KernelCache::new();
        let mut w_f16 = GpuTensor::<half::f16>::zeros(device, w_f32.shape.clone(), DType::F16)
            .map_err(|e| LoaderError::Device(e.to_string()))?;
        warp_kernels::fp16::cast_f32_to_f16(&kcache, device, &w_f32, &mut w_f16)
            .map_err(|e| LoaderError::Device(e.to_string()))?;
        Ok(w_f16)
    };

    let mut layers = Vec::with_capacity(config.num_layers as usize);
    for i in 0..config.num_layers {
        let lc = &layer_configs[i as usize];
        let q_dim = config.num_heads * lc.head_dim;
        let kv_dim = config.kv_dim_for_layer(lc);
        let lp = format!("{prefix}.layers.{i}");
        eprintln!("[moe-q4] Layer {i}/{} ({})...", config.num_layers,
            if lc.is_global { "global" } else { "sliding" });

        let attn_norm = load_norm(&format!("{lp}.input_layernorm.weight"))?;
        let post_attn_norm = load_norm(&format!("{lp}.post_attention_layernorm.weight"))?;
        let pre_ffn_norm = load_norm(&format!("{lp}.pre_feedforward_layernorm.weight"))?;
        let post_ffn_norm = load_norm(&format!("{lp}.post_feedforward_layernorm.weight"))?;
        let post_ffn_norm_1 = load_norm(&format!("{lp}.post_feedforward_layernorm_1.weight"))?;
        let pre_ffn_norm_2 = load_norm(&format!("{lp}.pre_feedforward_layernorm_2.weight"))?;
        let post_ffn_norm_2 = load_norm(&format!("{lp}.post_feedforward_layernorm_2.weight"))?;
        let q_norm = load_norm(&format!("{lp}.self_attn.q_norm.weight"))?;
        let k_norm = load_norm(&format!("{lp}.self_attn.k_norm.weight"))?;
        let layer_scalar = loader.load_f32(&format!("{lp}.layer_scalar"), device)
            .ok().and_then(|t| t.to_host(device).ok()).and_then(|v| v.first().copied()).unwrap_or(1.0);

        // Attention — F16 for cuBLAS HGEMM (zero quality loss)
        let wq = load_f16(&format!("{lp}.self_attn.q_proj.weight"), h, q_dim)?;
        let wk = load_f16(&format!("{lp}.self_attn.k_proj.weight"), h, kv_dim)?;
        let wv = match load_f16(&format!("{lp}.self_attn.v_proj.weight"), h, kv_dim) {
            Ok(v) => v,
            Err(_) => load_f16(&format!("{lp}.self_attn.k_proj.weight"), h, kv_dim)?,
        };
        let wo = load_f16(&format!("{lp}.self_attn.o_proj.weight"), q_dim, h)?;

        // Dense MLP — F16
        let dfn = config.ffn_dim;
        let w_gate = load_f16(&format!("{lp}.mlp.gate_proj.weight"), h, dfn)?;
        let w_up = load_f16(&format!("{lp}.mlp.up_proj.weight"), h, dfn)?;
        let w_down = load_f16(&format!("{lp}.mlp.down_proj.weight"), dfn, h)?;

        // Router F32
        let router_proj = loader.load_f32(&format!("{lp}.router.proj.weight"), device)?;
        let router_scale = loader.load_f32(&format!("{lp}.router.scale"), device)?;
        let per_expert_scale = loader.load_f32(&format!("{lp}.router.per_expert_scale"), device)?;

        // Expert Q4 — quantize ALL 128 experts and concatenate into one VRAM buffer
        eprintln!("[moe-q4]   Quantizing 128 experts to Q4...");
        let experts_gu_f32 = loader.load_f32(&format!("{lp}.experts.gate_up_proj"), device)?;
        let gu_all = experts_gu_f32.to_host(device).map_err(|e| LoaderError::Device(e.to_string()))?;
        drop(experts_gu_f32);

        let experts_d_f32 = loader.load_f32(&format!("{lp}.experts.down_proj"), device)?;
        let d_all = experts_d_f32.to_host(device).map_err(|e| LoaderError::Device(e.to_string()))?;
        drop(experts_d_f32);

        // TW-Marlin format per expert
        let gu_k = h; let gu_n = 2 * moe_dim;
        let gu_num_groups = gu_k / 32;
        let gu_packed_per = (gu_num_groups * gu_n * 16) as usize;
        let gu_scales_per = (gu_num_groups * gu_n) as usize;

        let d_k = moe_dim; let d_n = h;
        let d_num_groups = d_k / 32;
        let d_packed_per = (d_num_groups * d_n * 16) as usize;
        let d_scales_per = (d_num_groups * d_n) as usize;

        let gu_f32_per = (gu_n * gu_k) as usize;
        let d_f32_per = (d_n * d_k) as usize;

        let mut all_gu_packed = vec![0u8; 128 * gu_packed_per];
        let mut all_gu_scales = vec![half::f16::ZERO; 128 * gu_scales_per];
        let mut all_d_packed = vec![0u8; 128 * d_packed_per];
        let mut all_d_scales = vec![half::f16::ZERO; 128 * d_scales_per];

        for e in 0..128usize {
            let gu_start = e * gu_f32_per;
            let (gp, gs) = cpu_quantize_tw_marlin(&gu_all[gu_start..gu_start + gu_f32_per], gu_k, gu_n);
            all_gu_packed[e * gu_packed_per..(e + 1) * gu_packed_per].copy_from_slice(&gp);
            all_gu_scales[e * gu_scales_per..(e + 1) * gu_scales_per].copy_from_slice(&gs);

            let d_start = e * d_f32_per;
            let (dp, ds) = cpu_quantize_tw_marlin(&d_all[d_start..d_start + d_f32_per], d_k, d_n);
            all_d_packed[e * d_packed_per..(e + 1) * d_packed_per].copy_from_slice(&dp);
            all_d_scales[e * d_scales_per..(e + 1) * d_scales_per].copy_from_slice(&ds);
        }
        drop(gu_all); drop(d_all);

        let experts_gu_packed = GpuTensor::from_host(device, &all_gu_packed,
            Shape::from_static(&[all_gu_packed.len()]), DType::U8)
            .map_err(|e| LoaderError::Device(e.to_string()))?;
        let experts_gu_scales = GpuTensor::from_host(device, &all_gu_scales,
            Shape::from_static(&[all_gu_scales.len()]), DType::F16)
            .map_err(|e| LoaderError::Device(e.to_string()))?;
        let experts_d_packed = GpuTensor::from_host(device, &all_d_packed,
            Shape::from_static(&[all_d_packed.len()]), DType::U8)
            .map_err(|e| LoaderError::Device(e.to_string()))?;
        let experts_d_scales = GpuTensor::from_host(device, &all_d_scales,
            Shape::from_static(&[all_d_scales.len()]), DType::F16)
            .map_err(|e| LoaderError::Device(e.to_string()))?;

        device.synchronize().map_err(|e| LoaderError::Device(e.to_string()))?;

        // Upload TW-Marlin packed data as raw bytes for the native GEMM path
        // (SafeTensors path still uses TW-Marlin internally, wrapped as raw buffers)
        let experts_gu_raw = GpuTensor::from_host(device, &all_gu_packed,
            Shape::from_static(&[all_gu_packed.len()]), DType::U8)
            .map_err(|e| LoaderError::Device(e.to_string()))?;
        let experts_d_raw = GpuTensor::from_host(device, &all_d_packed,
            Shape::from_static(&[all_d_packed.len()]), DType::U8)
            .map_err(|e| LoaderError::Device(e.to_string()))?;

        layers.push(MoEQ4Layer {
            attn_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm,
            post_ffn_norm_1, pre_ffn_norm_2, post_ffn_norm_2,
            q_norm, k_norm, layer_scalar,
            wq, wk, wv, wo, w_gate, w_up, w_down,
            router_proj, router_scale, per_expert_scale,
            experts_gu_raw, gu_bytes_per_expert: gu_packed_per,
            experts_d_raw, d_bytes_per_expert: d_packed_per,
            d_block_bytes: 20, d_block_elems: 32,
            use_native_gguf_experts: false,
            experts_gu_scales: Some(experts_gu_scales),
            experts_d_scales: Some(experts_d_scales),
            gu_scales_per_expert: gu_scales_per,
            d_scales_per_expert: d_scales_per,
            expert_gu_k: gu_k, expert_gu_n: gu_n,
            expert_d_k: d_k, expert_d_n: d_n,
        });
    }

    let final_norm = load_norm(&format!("{prefix}.norm.weight"))?;

    eprintln!("[moe-q4] All weights in VRAM!");
    Ok(MoEQ4Engine {
        config, layer_configs, embed_tokens: embed, final_norm,
        cache: KernelCache::new(), layers, weights_reordered: false,
    })
}

/// CPU quantization to TW-Marlin separated format.
/// Input: row-major [N, K] float32.
/// Output: (packed_nibbles, scales_fp16) in TW-Marlin layout.
/// packed: [num_k_groups][N][16] contiguous per group
/// scales: [num_k_groups][N] as FP16
fn cpu_quantize_tw_marlin(data: &[f32], k: u32, n: u32) -> (Vec<u8>, Vec<half::f16>) {
    let num_k_groups = k / 32;
    let packed_size = (num_k_groups * n * 16) as usize;
    let scales_size = (num_k_groups * n) as usize;
    let mut packed = vec![0u8; packed_size];
    let mut scales = vec![half::f16::ZERO; scales_size];

    for col in 0..n as usize {
        for g in 0..num_k_groups as usize {
            let k_start = g * 32;
            // Gather 32 elements from column col, rows k_start..k_start+32
            // Input is [N, K] row-major: element(n_idx, k_idx) = data[n_idx * K + k_idx]
            let mut vals = [0.0f32; 32];
            for i in 0..32 {
                vals[i] = data[col * k as usize + k_start + i];
            }

            let amax = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = amax / 7.0;
            let inv_scale = if scale != 0.0 { 7.0 / amax } else { 0.0 };

            // TW-Marlin layout: scales[g * N + col], packed[(g * N + col) * 16 .. + 16]
            scales[g * n as usize + col] = half::f16::from_f32(scale);
            let p_off = (g * n as usize + col) * 16;
            for i in 0..16 {
                let v0 = vals[2 * i];
                let v1 = vals[2 * i + 1];
                let mut q0 = (v0 * inv_scale).round() as i32 + 8;
                let mut q1 = (v1 * inv_scale).round() as i32 + 8;
                q0 = q0.clamp(0, 15);
                q1 = q1.clamp(0, 15);
                packed[p_off + i] = (q0 as u8) | ((q1 as u8) << 4);
            }
        }
    }
    (packed, scales)
}

/// CPU Q4 quantization for column-major blocks (legacy Q4_0 format).
/// Input: row-major [N, K] float32. Output: Q4_0 column-major blocks.
/// Each column has K/32 blocks of 20 bytes (4B scale + 16B packed nibbles).
fn cpu_quantize_q4_col_major(data: &[f32], k: u32, n: u32) -> Vec<u8> {
    let num_k_blocks = k / 32;
    let total_blocks = num_k_blocks * n;
    let mut out = vec![0u8; (total_blocks * 20) as usize];

    for col in 0..n as usize {
        for b in 0..num_k_blocks as usize {
            let block_off = (col * num_k_blocks as usize + b) * 20;
            let k_start = b * 32;

            // Gather column values (row-major: element [row, col] = data[row * n + col]... wait
            // Actually the expert is stored as [N, K] contiguous, so element at (n_idx, k_idx) = data[n_idx * K + k_idx]
            // For Q4 column-major: we want to quantize along K dimension for each output column N
            let mut vals = [0.0f32; 32];
            for i in 0..32 {
                vals[i] = data[col * k as usize + k_start + i];
            }

            let amax = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = amax / 7.0;
            let inv_scale = if scale != 0.0 { 7.0 / amax } else { 0.0 };

            out[block_off..block_off + 4].copy_from_slice(&scale.to_le_bytes());
            for i in 0..16 {
                let v0 = vals[2 * i];
                let v1 = vals[2 * i + 1];
                let mut q0 = (v0 * inv_scale).round() as i32 + 8;
                let mut q1 = (v1 * inv_scale).round() as i32 + 8;
                q0 = q0.clamp(0, 15);
                q1 = q1.clamp(0, 15);
                out[block_off + 4 + i] = (q0 as u8) | ((q1 as u8) << 4);
            }
        }
    }
    out
}

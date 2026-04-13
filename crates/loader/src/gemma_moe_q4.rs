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
        let t = loader.load_f32(name, device)?;
        let mut h = t.to_host(device).map_err(|e| LoaderError::Device(e.to_string()))?;
        for v in &mut h { *v += 1.0; }
        GpuTensor::from_host(device, &h, t.shape.clone(), DType::F32)
            .map_err(|e| LoaderError::Device(e.to_string()))
    };

    let load_q4 = |name: &str, k: u32, n: u32| -> Result<GpuTensor<u8>, LoaderError> {
        let c = KernelCache::new();
        let w = loader.load_f32_transposed(name, device)?;
        warp_kernels::quantize::quantize_weights_q4_0(&c, device, &w, k, n)
            .map_err(|e| LoaderError::Device(e.to_string()))
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

        // Attention Q4
        let wq = load_q4(&format!("{lp}.self_attn.q_proj.weight"), h, q_dim)?;
        let wk = load_q4(&format!("{lp}.self_attn.k_proj.weight"), h, kv_dim)?;
        let wv = match load_q4(&format!("{lp}.self_attn.v_proj.weight"), h, kv_dim) {
            Ok(v) => v,
            Err(_) => load_q4(&format!("{lp}.self_attn.k_proj.weight"), h, kv_dim)?,
        };
        let wo = load_q4(&format!("{lp}.self_attn.o_proj.weight"), q_dim, h)?;

        // Dense MLP Q4
        let dfn = config.ffn_dim;
        let w_gate = load_q4(&format!("{lp}.mlp.gate_proj.weight"), h, dfn)?;
        let w_up = load_q4(&format!("{lp}.mlp.up_proj.weight"), h, dfn)?;
        let w_down = load_q4(&format!("{lp}.mlp.down_proj.weight"), dfn, h)?;

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

        // Q4 sizes per expert
        let gu_k = h; let gu_n = 2 * moe_dim;
        let gu_blocks = (gu_k / 32) * gu_n;
        let gu_q4_per = (gu_blocks * 20) as usize;

        let d_k = moe_dim; let d_n = h;
        let d_blocks = (d_k / 32) * d_n;
        let d_q4_per = (d_blocks * 20) as usize;

        let gu_f32_per = (gu_n * gu_k) as usize;  // elements per expert (row-major [gu_n, gu_k])
        let d_f32_per = (d_n * d_k) as usize;

        // Quantize each expert on CPU and concatenate
        let mut all_gu_q4 = vec![0u8; 128 * gu_q4_per];
        let mut all_d_q4 = vec![0u8; 128 * d_q4_per];

        for e in 0..128usize {
            // Expert gate_up: stored as [gu_n, gu_k] row-major (= [1408, 2816])
            // Q4 quantize column-major blocks: iterate cols, then K-blocks per col
            let gu_start = e * gu_f32_per;
            let gu_f32 = &gu_all[gu_start..gu_start + gu_f32_per];
            let gu_q4 = cpu_quantize_q4_col_major(gu_f32, gu_k, gu_n);
            all_gu_q4[e * gu_q4_per..(e + 1) * gu_q4_per].copy_from_slice(&gu_q4);

            let d_start = e * d_f32_per;
            let d_f32 = &d_all[d_start..d_start + d_f32_per];
            let d_q4 = cpu_quantize_q4_col_major(d_f32, d_k, d_n);
            all_d_q4[e * d_q4_per..(e + 1) * d_q4_per].copy_from_slice(&d_q4);
        }
        drop(gu_all); drop(d_all);

        let experts_gu_q4 = GpuTensor::from_host(device, &all_gu_q4,
            Shape::from_static(&[all_gu_q4.len()]), DType::U8)
            .map_err(|e| LoaderError::Device(e.to_string()))?;
        let experts_d_q4 = GpuTensor::from_host(device, &all_d_q4,
            Shape::from_static(&[all_d_q4.len()]), DType::U8)
            .map_err(|e| LoaderError::Device(e.to_string()))?;

        device.synchronize().map_err(|e| LoaderError::Device(e.to_string()))?;

        layers.push(MoEQ4Layer {
            attn_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm,
            post_ffn_norm_1, pre_ffn_norm_2, post_ffn_norm_2,
            q_norm, k_norm, layer_scalar,
            wq, wk, wv, wo, w_gate, w_up, w_down,
            router_proj, router_scale, per_expert_scale,
            experts_gu_q4, experts_d_q4,
            gu_q4_per_expert: gu_q4_per,
            d_q4_per_expert: d_q4_per,
        });
    }

    let final_norm = load_norm(&format!("{prefix}.norm.weight"))?;

    eprintln!("[moe-q4] All weights in VRAM!");
    Ok(MoEQ4Engine {
        config, layer_configs, embed_tokens: embed, final_norm,
        cache: KernelCache::new(), layers,
    })
}

/// CPU Q4 quantization for column-major blocks.
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

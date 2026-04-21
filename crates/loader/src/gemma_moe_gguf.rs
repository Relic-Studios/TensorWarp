//! GGUF-based MoE loader for Gemma 4 26B-A4B.
//!
//! Loads pre-quantized Q4_K_M weights from GGUF files (bartowski / llama.cpp format),
//! dequantizes to F32, re-quantizes to TW-Marlin for our GEMM kernels.
//! This preserves the quality advantage of calibrated k-quants while using our fast kernels.

use std::path::Path;
use warp_ir::{DType, Shape};
use warp_kernels::cache::KernelCache;
use warp_kernels::device::WarpDevice;
use warp_kernels::tensor::GpuTensor;
use warp_kernels::transformer::GemmaConfig;
use warp_kernels::moe_q4_generate::{MoEQ4Layer, MoEQ4Engine};

use crate::gguf::{GgufModel, GgufError};
use crate::safetensors_loader::LoaderError;

/// Build a GemmaConfig from GGUF metadata + tensor shapes.
fn config_from_gguf(model: &GgufModel) -> Result<GemmaConfig, LoaderError> {
    let arch = model.architecture().unwrap_or("gemma4");

    let get_u32 = |key: &str| -> Result<u32, LoaderError> {
        let prefixed = format!("{arch}.{key}");
        model.metadata.get(&prefixed)
            .or_else(|| model.metadata.get(key))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .ok_or_else(|| LoaderError::Config(format!("missing GGUF metadata: {key}")))
    };

    let get_f32 = |key: &str| -> Option<f32> {
        let prefixed = format!("{arch}.{key}");
        model.metadata.get(&prefixed)
            .or_else(|| model.metadata.get(key))
            .and_then(|v| match v {
                crate::gguf::GgufValue::Float32(f) => Some(*f),
                crate::gguf::GgufValue::Float64(f) => Some(*f as f32),
                _ => None,
            })
    };

    let hidden_size = get_u32("embedding_length")?;
    let num_layers = get_u32("block_count")?;
    let num_heads = get_u32("attention.head_count")?;
    let vocab_size = get_u32("vocab_size").unwrap_or(262144);
    let expert_count = get_u32("expert_count").unwrap_or(128);
    let expert_used = get_u32("expert_used_count").unwrap_or(8);

    let rope_theta = get_f32("rope.freq_base").unwrap_or(10000.0);
    let norm_eps = get_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);

    // Derive head_dim and kv_heads from ACTUAL tensor shapes (metadata can be misleading)
    // blk.0 is a sliding layer: attn_q [H, q_dim], attn_k [H, kv_dim]
    let q_dim_0 = model.tensors.iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .and_then(|t| t.dims.get(1).copied())
        .unwrap_or(4096) as u32;
    let kv_dim_0 = model.tensors.iter()
        .find(|t| t.name == "blk.0.attn_k.weight")
        .and_then(|t| t.dims.get(1).copied())
        .unwrap_or(2048) as u32;

    let head_dim = q_dim_0 / num_heads; // sliding head_dim
    let num_kv_heads = kv_dim_0 / head_dim;

    // Dense MLP ffn_dim from tensor shape
    let ffn_dim = model.tensors.iter()
        .find(|t| t.name == "blk.0.ffn_gate.weight")
        .and_then(|t| t.dims.get(1).copied())
        .unwrap_or(2112) as u32;

    // Find a global layer (e.g. layer 5) to get global head_dim
    let sliding_window = get_u32("attention.sliding_window").unwrap_or(1024);
    let sliding_window_pattern = get_u32("attention.sliding_window_pattern").unwrap_or(6);
    let first_global = if sliding_window_pattern > 0 { sliding_window_pattern - 1 } else { 5 };

    let global_q_dim = model.tensors.iter()
        .find(|t| t.name == format!("blk.{first_global}.attn_q.weight"))
        .and_then(|t| t.dims.get(1).copied())
        .unwrap_or((num_heads * head_dim * 2) as u64) as u32;
    let global_kv_dim = model.tensors.iter()
        .find(|t| t.name == format!("blk.{first_global}.attn_k.weight"))
        .and_then(|t| t.dims.get(1).copied())
        .unwrap_or(kv_dim_0 as u64) as u32;
    let global_head_dim = global_q_dim / num_heads;
    let num_global_kv_heads = global_kv_dim / global_head_dim;

    eprintln!("[gguf] Config: {num_layers} layers, H={hidden_size}, vocab={vocab_size}");
    eprintln!("[gguf]   Sliding: {num_heads} heads, head_dim={head_dim}, {num_kv_heads} KV heads, q_dim={q_dim_0}, kv_dim={kv_dim_0}");
    eprintln!("[gguf]   Global:  {num_heads} heads, head_dim={global_head_dim}, {num_global_kv_heads} KV heads, q_dim={global_q_dim}, kv_dim={global_kv_dim}");
    eprintln!("[gguf]   Dense FFN: {ffn_dim}, MoE: {expert_count} experts, top-{expert_used}");
    eprintln!("[gguf]   Sliding window: {sliding_window}, pattern: 1 global per {sliding_window_pattern}");

    Ok(GemmaConfig {
        hidden_size,
        num_heads,
        num_kv_heads,
        num_global_kv_heads,
        head_dim,
        global_head_dim,
        ffn_dim,
        vocab_size,
        num_layers,
        norm_eps,
        sliding_window,
        sliding_window_pattern,
        rope_theta,
        rope_theta_global: 1000000.0,
        partial_rotary_factor: 0.25,
        k_eq_v: true,
        final_logit_softcapping: 30.0,
        tie_word_embeddings: true,
    })
}

/// Load Gemma 4 MoE from a GGUF file, producing a MoEQ4Engine with TW-Marlin weights.
pub fn load_moe_gguf<P: AsRef<Path>>(
    path: P,
    device: &WarpDevice,
) -> Result<MoEQ4Engine, LoaderError> {
    let path = path.as_ref();
    eprintln!("[gguf] Loading: {}", path.display());

    let model = GgufModel::load(path)
        .map_err(|e| LoaderError::Config(format!("GGUF load failed: {e}")))?;

    eprintln!("[gguf] GGUF v{}, {} tensors, arch={}",
        model.version, model.tensors.len(),
        model.architecture().unwrap_or("unknown"));

    // Print dtype distribution and first tensors for debugging
    let mut dtype_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for t in &model.tensors {
        *dtype_counts.entry(format!("{:?}", t.dtype)).or_insert(0) += 1;
    }
    eprintln!("[gguf] Dtype distribution: {:?}", dtype_counts);

    for (i, t) in model.tensors.iter().take(30).enumerate() {
        eprintln!("[gguf]   [{i:3}] {:60} {:?} {:?}", t.name, t.dtype, t.dims);
    }
    if model.tensors.len() > 30 {
        eprintln!("[gguf]   ... and {} more", model.tensors.len() - 30);
    }


    let config = config_from_gguf(&model)?;
    let layer_configs = config.layer_configs();
    let h = config.hidden_size;

    // Detect expert intermediate size from tensor shapes
    // ffn_down_exps shape is [n_experts, hidden_size, moe_dim] in GGUF
    // ffn_gate_up_exps shape is [n_experts, 2*moe_dim, hidden_size]
    let moe_dim = {
        let down_info = model.tensors.iter().find(|t| t.name == "blk.0.ffn_down_exps.weight");
        if let Some(info) = down_info {
            // dims[2] or dims[1] depending on layout
            let dim_candidates: Vec<u64> = info.dims.iter().copied()
                .filter(|&d| d != 128 && d != h as u64)
                .collect();
            dim_candidates.first().copied().unwrap_or(704) as u32
        } else {
            704u32 // Gemma 4 26B default
        }
    };
    eprintln!("[gguf] Expert intermediate dim (moe_dim): {moe_dim}");

    let kcache = KernelCache::new();

    // ── Embedding ────────────────────────────────────────────────────────
    eprintln!("[gguf] Loading embedding...");
    let embed_f32 = model.get_tensor_f32("token_embd.weight")
        .ok_or_else(|| LoaderError::Config("missing token_embd.weight".into()))?;
    let embed_numel = embed_f32.len();
    let embed_rows = config.vocab_size as usize;
    let embed_cols = config.hidden_size as usize;
    eprintln!("[gguf]   Embedding: {} elements, expected {}x{} = {}",
        embed_numel, embed_rows, embed_cols, embed_rows * embed_cols);
    let embed_gpu_f32 = GpuTensor::from_host(device, &embed_f32,
        Shape::from_static(&[embed_numel]), DType::F32)
        .map_err(|e| LoaderError::Device(e.to_string()))?;
    let mut embed_f16 = GpuTensor::<half::f16>::zeros(device,
        Shape::from_static(&[embed_numel]), DType::F16)
        .map_err(|e| LoaderError::Device(e.to_string()))?;
    warp_kernels::fp16::cast_f32_to_f16(&kcache, device, &embed_gpu_f32, &mut embed_f16)
        .map_err(|e| LoaderError::Device(e.to_string()))?;
    drop(embed_f32);
    drop(embed_gpu_f32);

    // ── Helper closures ──────────────────────────────────────────────────
    let load_norm = |name: &str| -> Result<GpuTensor<f32>, LoaderError> {
        let mut data = model.get_tensor_f32(name)
            .ok_or_else(|| LoaderError::Config(format!("missing tensor: {name}")))?;
        // Gemma 4 RMSNorm uses `output = x/rms(x) * weight` (NOT `(1+weight)` like Gemma 3)
        // Weights are stored as the actual multiplier — do NOT add +1.
        GpuTensor::from_host(device, &data, Shape::from_static(&[data.len()]), DType::F32)
            .map_err(|e| LoaderError::Device(e.to_string()))
    };

    let load_norm_optional = |name: &str| -> Result<GpuTensor<f32>, LoaderError> {
        match model.get_tensor_f32(name) {
            Some(data) => {
                // Gemma 4: no +1, weights used directly as multiplier
                GpuTensor::from_host(device, &data, Shape::from_static(&[data.len()]), DType::F32)
                    .map_err(|e| LoaderError::Device(e.to_string()))
            }
            None => {
                // Return ones (identity for RMSNorm)
                let ones = vec![1.0f32; h as usize];
                GpuTensor::from_host(device, &ones, Shape::from_static(&[h as usize]), DType::F32)
                    .map_err(|e| LoaderError::Device(e.to_string()))
            }
        }
    };

    // Load weight as F16 for cuBLAS HGEMM (gemm_cublas_f16_transB expects [N, K] row-major).
    // GGUF stores weights as [N, K] row-major — no transposition needed for F16 HGEMM!
    let load_f16 = |name: &str, k: u32, n: u32| -> Result<GpuTensor<half::f16>, LoaderError> {
        let data = model.get_tensor_f32(name)
            .ok_or_else(|| LoaderError::Config(format!("missing tensor: {name}")))?;
        let expected = (k as usize) * (n as usize);
        if data.len() != expected {
            return Err(LoaderError::Config(format!(
                "{name}: data len {} != expected K*N {}*{}={}", data.len(), k, n, expected)));
        }
        // Convert F32 → F16 on CPU (weight loading is one-time cost)
        let f16_data: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
        GpuTensor::from_host(device, &f16_data, Shape::from_static(&[expected]), DType::F16)
            .map_err(|e| LoaderError::Device(e.to_string()))
    };

    // Load F32 tensor from GGUF, transpose [N,K] → [K,N], upload to GPU, quantize to TW-Marlin Q4.
    // GGUF stores weights as [N, K] row-major (same as PyTorch nn.Linear).
    // Our quantizer expects transposed [K, N] layout (same as SafeTensors load_f32_transposed).
    let load_q4 = |name: &str, k: u32, n: u32| -> Result<GpuTensor<u8>, LoaderError> {
        let dtype_info = model.tensors.iter().find(|t| t.name == name)
            .map(|t| format!("{:?}", t.dtype)).unwrap_or_else(|| "NOT FOUND".into());
        let data = model.get_tensor_f32(name)
            .ok_or_else(|| LoaderError::Config(format!("missing/unsupported tensor: {name} (dtype={dtype_info})")))?;

        // GGUF stores weights in ggml order: dims[0]=innermost.
        // For a 2D weight [dims[0], dims[1]], the memory layout is [dims[1]][dims[0]].
        // For attn_q with dims=[2816, 4096]: memory is [4096][2816] = [N, K].
        // quantize_weights_q4_0 expects [K, N] row-major, so we must transpose.
        let ku = k as usize;
        let nu = n as usize;
        let expected = ku * nu;
        if data.len() != expected {
            return Err(LoaderError::Config(format!(
                "{name}: data len {} != expected K*N {}*{}={}", data.len(), ku, nu, expected)));
        }
        let mut transposed = vec![0.0f32; expected];
        for row in 0..nu {
            for col in 0..ku {
                transposed[col * nu + row] = data[row * ku + col];
            }
        }

        let gpu_f32 = GpuTensor::from_host(device, &transposed,
            Shape::from_static(&[ku, nu]), DType::F32)
            .map_err(|e| LoaderError::Device(e.to_string()))?;
        let c = KernelCache::new();
        warp_kernels::quantize::quantize_weights_q4_0(&c, device, &gpu_f32, k, n)
            .map_err(|e| LoaderError::Device(e.to_string()))
    };

    let load_f32_gpu = |name: &str| -> Result<GpuTensor<f32>, LoaderError> {
        let data = model.get_tensor_f32(name)
            .ok_or_else(|| LoaderError::Config(format!("missing tensor: {name}")))?;
        GpuTensor::from_host(device, &data, Shape::from_static(&[data.len()]), DType::F32)
            .map_err(|e| LoaderError::Device(e.to_string()))
    };

    // ── Layers ───────────────────────────────────────────────────────────
    let mut layers = Vec::with_capacity(config.num_layers as usize);
    for i in 0..config.num_layers {
        let lc = &layer_configs[i as usize];
        let q_dim = config.num_heads * lc.head_dim;
        let kv_dim = config.kv_dim_for_layer(lc);
        eprintln!("[gguf] Layer {i}/{} ({})...", config.num_layers,
            if lc.is_global { "global" } else { "sliding" });

        // Norms — llama.cpp GGUF names for Gemma 4
        let attn_norm = load_norm(&format!("blk.{i}.attn_norm.weight"))?;
        // post_attention_norm is a separate tensor in this GGUF
        let post_attn_norm = load_norm_optional(&format!("blk.{i}.post_attention_norm.weight"))?;
        let pre_ffn_norm = load_norm_optional(&format!("blk.{i}.ffn_norm.weight"))?;
        let post_ffn_norm = load_norm_optional(&format!("blk.{i}.post_ffw_norm.weight"))?;
        let post_ffn_norm_1 = load_norm_optional(&format!("blk.{i}.post_ffw_norm_1.weight"))?;
        let pre_ffn_norm_2 = load_norm_optional(&format!("blk.{i}.pre_ffw_norm_2.weight"))?;
        let post_ffn_norm_2 = load_norm_optional(&format!("blk.{i}.post_ffw_norm_2.weight"))?;
        let q_norm = load_norm_optional(&format!("blk.{i}.attn_q_norm.weight"))?;
        let k_norm = load_norm_optional(&format!("blk.{i}.attn_k_norm.weight"))?;

        // Layer scalar (optional, defaults to 1.0)
        let layer_scalar = model.get_tensor_f32(&format!("blk.{i}.layer_output_scale.weight"))
            .and_then(|v| v.first().copied())
            .unwrap_or(1.0);

        // Attention projections → F16 (zero quality loss, cuBLAS HGEMM)
        if i == 0 {
            if let Some(vw) = model.get_tensor_f32(&format!("blk.{i}.attn_v.weight")) {
                eprintln!("[gguf] V_proj L0: len={}, row0[0:4]={:?}, row1[0:4]={:?}",
                    vw.len(), &vw[..4], &vw[2816..2820]);
            }
        }
        let wq = load_f16(&format!("blk.{i}.attn_q.weight"), h, q_dim)?;
        let wk = load_f16(&format!("blk.{i}.attn_k.weight"), h, kv_dim)?;
        let wv = match load_f16(&format!("blk.{i}.attn_v.weight"), h, kv_dim) {
            Ok(v) => v,
            Err(_) => load_f16(&format!("blk.{i}.attn_k.weight"), h, kv_dim)?, // K=V sharing
        };
        let wo = load_f16(&format!("blk.{i}.attn_output.weight"), q_dim, h)?;

        // Dense MLP → F16
        let dfn = config.ffn_dim;
        let w_gate = load_f16(&format!("blk.{i}.ffn_gate.weight"), h, dfn)?;
        let w_up = load_f16(&format!("blk.{i}.ffn_up.weight"), h, dfn)?;
        let w_down = load_f16(&format!("blk.{i}.ffn_down.weight"), dfn, h)?;

        // Router
        let router_proj = load_f32_gpu(&format!("blk.{i}.ffn_gate_inp.weight"))?;
        // Router scale — GGUF has blk.{i}.ffn_gate_inp.scale [H] and ffn_down_exps.scale [128]
        let router_scale = match load_f32_gpu(&format!("blk.{i}.ffn_gate_inp.scale")) {
            Ok(s) => s,
            Err(_) => {
                let ones = vec![1.0f32];
                GpuTensor::from_host(device, &ones, Shape::from_static(&[1]), DType::F32)
                    .map_err(|e| LoaderError::Device(e.to_string()))?
            }
        };
        let per_expert_scale = match load_f32_gpu(&format!("blk.{i}.ffn_down_exps.scale")) {
            Ok(s) => s,
            Err(_) => {
                let n_experts = 128usize;
                let ones = vec![1.0f32; n_experts];
                GpuTensor::from_host(device, &ones, Shape::from_static(&[n_experts]), DType::F32)
                    .map_err(|e| LoaderError::Device(e.to_string()))?
            }
        };

        // ── Expert weights — upload raw GGUF bytes (zero re-quantization) ──
        eprintln!("[gguf]   Loading experts (native Q4_K/Q8_0)...");

        let gu_k = h;
        let gu_n = 2 * moe_dim;
        let d_k = moe_dim;
        let d_n = h;
        let n_experts = 128usize;

        // gate_up: Q4_K raw bytes — upload directly to GPU
        let gu_raw = model.get_tensor_raw(&format!("blk.{i}.ffn_gate_up_exps.weight"))
            .ok_or_else(|| LoaderError::Config(format!("missing ffn_gate_up_exps for layer {i}")))?;
        let gu_bytes_per_expert = gu_raw.len() / n_experts;
        let gu_total = gu_raw.len();
        let experts_gu_raw = GpuTensor::from_host(device, &gu_raw[..gu_total],
            Shape::from_static(&[gu_total]), DType::U8)
            .map_err(|e| LoaderError::Device(e.to_string()))?;

        // down: detect dtype (Q8_0=34B/block or Q5_0=22B/block depending on layer)
        let d_tensor_name = format!("blk.{i}.ffn_down_exps.weight");
        let d_dtype = model.tensors.iter().find(|t| t.name == d_tensor_name)
            .map(|t| t.dtype).unwrap_or(crate::gguf::GgufDType::Q8_0);
        let d_block_bytes = match d_dtype {
            crate::gguf::GgufDType::Q8_0 => 34u32,
            crate::gguf::GgufDType::Q5_0 => 22u32,
            other => { eprintln!("[gguf]   WARNING: down_exps dtype {:?}, defaulting to Q8_0", other); 34u32 }
        };
        let d_raw = model.get_tensor_raw(&d_tensor_name)
            .ok_or_else(|| LoaderError::Config(format!("missing ffn_down_exps for layer {i}")))?;
        let d_bytes_per_expert = d_raw.len() / n_experts;
        let d_total = d_raw.len();
        let experts_d_raw = GpuTensor::from_host(device, &d_raw[..d_total],
            Shape::from_static(&[d_total]), DType::U8)
            .map_err(|e| LoaderError::Device(e.to_string()))?;

        device.synchronize().map_err(|e| LoaderError::Device(e.to_string()))?;

        layers.push(MoEQ4Layer {
            attn_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm,
            post_ffn_norm_1, pre_ffn_norm_2, post_ffn_norm_2,
            q_norm, k_norm, layer_scalar,
            wq, wk, wv, wo, w_gate, w_up, w_down,
            router_proj, router_scale, per_expert_scale,
            experts_gu_raw, gu_bytes_per_expert,
            experts_d_raw, d_bytes_per_expert,
            d_block_bytes, d_block_elems: 32,
            use_native_gguf_experts: true,
            experts_gu_scales: None, experts_d_scales: None,
            gu_scales_per_expert: 0, d_scales_per_expert: 0,
            expert_gu_k: gu_k, expert_gu_n: gu_n,
            expert_d_k: d_k, expert_d_n: d_n,
        });
    }

    let final_norm = load_norm("output_norm.weight")?;

    eprintln!("[gguf] All weights loaded and quantized to TW-Marlin!");
    Ok(MoEQ4Engine {
        config, layer_configs, embed_tokens: embed_f16, lm_head: None, final_norm,
        cache: KernelCache::new(), layers, weights_reordered: false,
    })
}

/// CPU quantization to TW-Marlin separated format (same as gemma_moe_q4.rs).
fn cpu_quantize_tw_marlin(data: &[f32], k: u32, n: u32) -> (Vec<u8>, Vec<half::f16>) {
    let num_k_groups = k / 32;
    let packed_size = (num_k_groups * n * 16) as usize;
    let scales_size = (num_k_groups * n) as usize;
    let mut packed = vec![0u8; packed_size];
    let mut scales = vec![half::f16::ZERO; scales_size];

    for col in 0..n as usize {
        for g in 0..num_k_groups as usize {
            let k_start = g * 32;
            let mut vals = [0.0f32; 32];
            for i in 0..32 {
                vals[i] = data[col * k as usize + k_start + i];
            }

            let amax = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = amax / 7.0;
            let inv_scale = if scale != 0.0 { 7.0 / amax } else { 0.0 };

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

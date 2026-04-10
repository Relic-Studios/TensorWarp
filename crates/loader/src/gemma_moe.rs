//! Gemma 4 26B-A4B MoE model loader.
//!
//! Loads the Mixture of Experts model with:
//! - Attention weights (Q4 for projections, F32 for norms)
//! - Dense MLP weights (Q4)
//! - Expert weights (stored as F32 3D tensors — too large for Q4 on 3D)
//! - Router weights (F32)
//!
//! Total VRAM: ~18-20 GB at mixed Q4/F32 — fits 24 GB.

use serde::Deserialize;
use warp_ir::{DType, Shape};
use warp_kernels::cache::KernelCache;
use warp_kernels::device::WarpDevice;
use warp_kernels::tensor::GpuTensor;
use warp_kernels::transformer::GemmaConfig;

use crate::gemma::GemmaHFConfig;
use crate::safetensors_loader::{ShardedSafeTensorsLoader, LoaderError};

/// Per-layer MoE weights.
pub struct GemmaMoELayerWeights {
    // Norms (F32, with Gemma +1 offset applied)
    pub attn_norm: GpuTensor<f32>,
    pub post_attn_norm: GpuTensor<f32>,
    pub pre_ffn_norm: GpuTensor<f32>,       // for dense MLP
    pub post_ffn_norm: GpuTensor<f32>,       // final post-FFN
    pub post_ffn_norm_1: GpuTensor<f32>,     // for dense MLP output
    pub pre_ffn_norm_2: GpuTensor<f32>,      // for MoE input
    pub post_ffn_norm_2: GpuTensor<f32>,     // for MoE output
    pub q_norm: GpuTensor<f32>,
    pub k_norm: GpuTensor<f32>,
    pub layer_scalar: f32,

    // Attention projections (Q4 quantized)
    pub wq: GpuTensor<u8>,
    pub wk: GpuTensor<u8>,
    pub wv: GpuTensor<u8>,  // may be same as wk for global layers
    pub wo: GpuTensor<u8>,

    // Dense MLP (Q4 quantized)
    pub w_gate: GpuTensor<u8>,
    pub w_up: GpuTensor<u8>,
    pub w_down: GpuTensor<u8>,

    // Router
    pub router_proj: GpuTensor<f32>,        // [hidden_size, num_experts]
    pub router_scale: GpuTensor<f32>,       // [hidden_size]
    pub per_expert_scale: GpuTensor<f32>,   // [num_experts]

    // Expert weights stored on HOST (RAM) as F16 — too large for VRAM (60 GB at F32)
    // Streamed to GPU per-token: only 8 active experts copied per layer
    pub experts_gate_up_host: Vec<half::f16>,  // [128, 2*704, 2816] flattened on CPU
    pub experts_down_host: Vec<half::f16>,     // [128, 2816, 704] flattened on CPU
}

/// Loaded MoE model — split into VRAM and RAM components.
pub struct GemmaMoEModel {
    pub config: GemmaConfig,
    pub embed_tokens: GpuTensor<half::f16>,
    pub layers_vram: Vec<GemmaMoELayerVRAM>,
    pub expert_gate_up_host: Vec<Vec<half::f16>>,  // [layer][128 * 2*704 * 2816]
    pub expert_down_host: Vec<Vec<half::f16>>,     // [layer][128 * 2816 * 704]
    pub final_norm: GpuTensor<f32>,
}

/// Per-layer VRAM-resident weights (no experts).
pub struct GemmaMoELayerVRAM {
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
    pub wq: GpuTensor<half::f16>,
    pub wk: GpuTensor<half::f16>,
    pub wv: GpuTensor<half::f16>,
    pub wo: GpuTensor<half::f16>,
    pub w_gate: GpuTensor<u8>,
    pub w_up: GpuTensor<u8>,
    pub w_down: GpuTensor<u8>,
    pub router_proj: GpuTensor<f32>,
    pub router_scale: GpuTensor<f32>,
    pub per_expert_scale: GpuTensor<f32>,
}

impl GemmaMoEModel {
    pub fn load(
        loader: &ShardedSafeTensorsLoader,
        hf_config: &GemmaHFConfig,
        device: &WarpDevice,
    ) -> Result<Self, LoaderError> {
        let config = hf_config.to_gemma_config();
        let h = config.hidden_size;
        let layer_configs = config.layer_configs();

        // Detect prefix
        let prefix = if loader.load_f32("model.language_model.embed_tokens.weight", device).is_ok() {
            "model.language_model"
        } else {
            "model"
        };
        eprintln!("[moe] Weight prefix: {prefix}");

        // Embedding F16
        eprintln!("[moe] Loading embedding as F16...");
        let embed_f32 = loader.load_f32(&format!("{prefix}.embed_tokens.weight"), device)?;
        let kcache = KernelCache::new();
        let mut embed_tokens = GpuTensor::<half::f16>::zeros(device, embed_f32.shape.clone(), DType::F16)
            .map_err(|e| LoaderError::Device(e.to_string()))?;
        warp_kernels::fp16::cast_f32_to_f16(&kcache, device, &embed_f32, &mut embed_tokens)
            .map_err(|e| LoaderError::Device(e.to_string()))?;
        drop(embed_f32);
        device.synchronize().map_err(|e| LoaderError::Device(e.to_string()))?;

        // Gemma norm +1 offset helper
        let load_norm = |name: &str| -> Result<GpuTensor<f32>, LoaderError> {
            let t = loader.load_f32(name, device)?;
            let mut host = t.to_host(device).map_err(|e| LoaderError::Device(e.to_string()))?;
            for v in &mut host { *v += 1.0; }
            GpuTensor::from_host(device, &host, t.shape.clone(), DType::F32)
                .map_err(|e| LoaderError::Device(e.to_string()))
        };

        let load_q4 = |name: &str, k: u32, n: u32| -> Result<GpuTensor<u8>, LoaderError> {
            let cache = KernelCache::new();
            let w = loader.load_f32_transposed(name, device)?;
            warp_kernels::quantize::quantize_weights_q4_0(&cache, device, &w, k, n)
                .map_err(|e| LoaderError::Device(e.to_string()))
        };

        // Layers
        let mut layers_vram = Vec::with_capacity(config.num_layers as usize);
        let mut all_expert_gu = Vec::with_capacity(config.num_layers as usize);
        let mut all_expert_d = Vec::with_capacity(config.num_layers as usize);
        for i in 0..config.num_layers {
            let lc = &layer_configs[i as usize];
            let q_dim = config.num_heads * lc.head_dim;
            let kv_dim = config.kv_dim_for_layer(lc);
            let lp = format!("{prefix}.layers.{i}");

            eprintln!("[moe] Loading layer {i}/{} ({})...",
                config.num_layers, if lc.is_global { "global" } else { "sliding" });

            // Norms
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
                .ok().and_then(|t| t.to_host(device).ok())
                .and_then(|v| v.first().copied()).unwrap_or(1.0);

            // Attention F16 (higher precision → correct router outputs)
            let wq = loader.load_f16_transposed(&format!("{lp}.self_attn.q_proj.weight"), device)?;
            let wk = loader.load_f16_transposed(&format!("{lp}.self_attn.k_proj.weight"), device)?;
            let wv = match loader.load_f16_transposed(&format!("{lp}.self_attn.v_proj.weight"), device) {
                Ok(v) => v,
                Err(_) => loader.load_f16_transposed(&format!("{lp}.self_attn.k_proj.weight"), device)?,
            };
            let wo = loader.load_f16_transposed(&format!("{lp}.self_attn.o_proj.weight"), device)?;

            // Dense MLP Q4
            let dense_ffn = config.ffn_dim;
            let w_gate = load_q4(&format!("{lp}.mlp.gate_proj.weight"), h, dense_ffn)?;
            let w_up = load_q4(&format!("{lp}.mlp.up_proj.weight"), h, dense_ffn)?;
            let w_down = load_q4(&format!("{lp}.mlp.down_proj.weight"), dense_ffn, h)?;

            // Router (F32 — small)
            let router_proj = loader.load_f32_transposed(&format!("{lp}.router.proj.weight"), device)?;
            let router_scale_raw = loader.load_f32(&format!("{lp}.router.scale"), device)?;
            // Router scale gets +1 offset too (it's RMSNorm-like)
            let mut rs_host = router_scale_raw.to_host(device).map_err(|e| LoaderError::Device(e.to_string()))?;
            // Actually router.scale is NOT a norm weight — it's a learned scale vector, no +1
            let router_scale = GpuTensor::from_host(device, &rs_host,
                router_scale_raw.shape.clone(), DType::F32)
                .map_err(|e| LoaderError::Device(e.to_string()))?;
            let per_expert_scale = loader.load_f32(&format!("{lp}.router.per_expert_scale"), device)?;

            // Expert weights — load to HOST as F16 (too large for VRAM: 2 GB/layer × 30 = 60 GB)
            // Will stream only 8 active experts to GPU per-token
            eprintln!("[moe]   Loading experts to RAM (128 × gate_up + down)...");
            let experts_gate_up_f32 = loader.load_f32(&format!("{lp}.experts.gate_up_proj"), device)?;
            let gu_host_f32 = experts_gate_up_f32.to_host(device).map_err(|e| LoaderError::Device(e.to_string()))?;
            drop(experts_gate_up_f32); // free VRAM immediately
            let experts_gate_up_host: Vec<half::f16> = gu_host_f32.iter().map(|v| half::f16::from_f32(*v)).collect();
            drop(gu_host_f32);

            let experts_down_f32 = loader.load_f32(&format!("{lp}.experts.down_proj"), device)?;
            let d_host_f32 = experts_down_f32.to_host(device).map_err(|e| LoaderError::Device(e.to_string()))?;
            drop(experts_down_f32);
            let experts_down_host: Vec<half::f16> = d_host_f32.iter().map(|v| half::f16::from_f32(*v)).collect();
            drop(d_host_f32);

            layers_vram.push(GemmaMoELayerVRAM {
                attn_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm,
                post_ffn_norm_1, pre_ffn_norm_2, post_ffn_norm_2,
                q_norm, k_norm, layer_scalar,
                wq, wk, wv, wo, w_gate, w_up, w_down,
                router_proj, router_scale, per_expert_scale,
            });
            all_expert_gu.push(experts_gate_up_host);
            all_expert_d.push(experts_down_host);

            device.synchronize().map_err(|e| LoaderError::Device(e.to_string()))?;
        }

        // Final norm
        let final_norm = load_norm(&format!("{prefix}.norm.weight"))?;

        eprintln!("[moe] Model loaded!");
        Ok(Self {
            config, embed_tokens,
            layers_vram,
            expert_gate_up_host: all_expert_gu,
            expert_down_host: all_expert_d,
            final_norm,
        })
    }
}

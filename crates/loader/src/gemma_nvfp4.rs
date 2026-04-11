//! Gemma 4 31B NVFP4 model loader.
//!
//! Loads NVIDIA FP4-quantized weights (18.5 GB VRAM total).
//! Weights stay in FP4 on VRAM. Per-GEMM dequantization to F16.
//! Uses the existing Gemma dense architecture (60 layers, sliding+global attention).

use warp_ir::{DType, Shape};
use warp_kernels::cache::KernelCache;
use warp_kernels::device::WarpDevice;
use warp_kernels::tensor::GpuTensor;

use crate::gemma::GemmaHFConfig;
use crate::safetensors_loader::{ShardedSafeTensorsLoader, LoaderError};
use warp_kernels::transformer::GemmaConfig;

/// FP4 packed weight + scales for one linear projection.
pub struct FP4Weight {
    pub packed: GpuTensor<u8>,      // [out, in/2] — 2 FP4 values per byte
    pub scales: GpuTensor<u8>,      // [out, in/16] — FP8 E4M3 per-group scales
    pub global_scale: f32,          // weight_scale_2
    pub rows: u32,                  // output dim
    pub cols: u32,                  // input dim (original, 2× packed cols)
}

/// Per-layer weights in NVFP4 format.
pub struct NVFP4LayerWeights {
    // Norms (BF16 → F32 with Gemma +1 offset)
    pub attn_norm: GpuTensor<f32>,
    pub post_attn_norm: GpuTensor<f32>,
    pub pre_ffn_norm: GpuTensor<f32>,
    pub post_ffn_norm: GpuTensor<f32>,
    pub q_norm: GpuTensor<f32>,
    pub k_norm: GpuTensor<f32>,
    pub layer_scalar: f32,

    // Attention projections (FP4)
    pub wq: FP4Weight,
    pub wk: FP4Weight,
    pub wv: Option<FP4Weight>,  // None for global layers (K=V)
    pub wo: FP4Weight,

    // Dense MLP (FP4)
    pub w_gate: FP4Weight,
    pub w_up: FP4Weight,
    pub w_down: FP4Weight,
}

/// Loaded NVFP4 Gemma model.
pub struct GemmaNVFP4Model {
    pub config: GemmaConfig,
    pub embed_tokens: GpuTensor<half::f16>,
    pub layers: Vec<NVFP4LayerWeights>,
    pub final_norm: GpuTensor<f32>,
}

impl GemmaNVFP4Model {
    pub fn load(
        loader: &ShardedSafeTensorsLoader,
        hf_config: &GemmaHFConfig,
        device: &WarpDevice,
    ) -> Result<Self, LoaderError> {
        let config = hf_config.to_gemma_config();
        let layer_configs = config.layer_configs();
        let h = config.hidden_size;

        let prefix = if loader.load_f32("model.language_model.embed_tokens.weight", device).is_ok() {
            "model.language_model"
        } else { "model" };
        eprintln!("[nvfp4] Weight prefix: {prefix}");

        // Embedding F16
        eprintln!("[nvfp4] Loading embedding as F16...");
        let embed_f32 = loader.load_f32(&format!("{prefix}.embed_tokens.weight"), device)?;
        let kcache = KernelCache::new();
        let mut embed_tokens = GpuTensor::<half::f16>::zeros(device, embed_f32.shape.clone(), DType::F16)
            .map_err(|e| LoaderError::Device(e.to_string()))?;
        warp_kernels::fp16::cast_f32_to_f16(&kcache, device, &embed_f32, &mut embed_tokens)
            .map_err(|e| LoaderError::Device(e.to_string()))?;
        drop(embed_f32);

        let load_norm = |name: &str| -> Result<GpuTensor<f32>, LoaderError> {
            let t = loader.load_f32(name, device)?;
            let mut host = t.to_host(device).map_err(|e| LoaderError::Device(e.to_string()))?;
            for v in &mut host { *v += 1.0; }
            GpuTensor::from_host(device, &host, t.shape.clone(), DType::F32)
                .map_err(|e| LoaderError::Device(e.to_string()))
        };

        // Load FP4 weight: packed bytes + FP8 scales + global scale
        let load_fp4 = |name: &str| -> Result<FP4Weight, LoaderError> {
            let packed = loader.load_raw(&format!("{name}.weight"), device)?;
            let scales = loader.load_raw(&format!("{name}.weight_scale"), device)?;
            let gs_tensor = loader.load_f32(&format!("{name}.weight_scale_2"), device)?;
            let gs_host = gs_tensor.to_host(device).map_err(|e| LoaderError::Device(e.to_string()))?;
            let global_scale = gs_host[0];

            // Infer dimensions from packed shape
            let packed_shape = &packed.shape;
            let dims = packed_shape.dims();
            let rows = dims[0].static_val().unwrap_or(0) as u32;
            let packed_cols = dims[1].static_val().unwrap_or(0) as u32;
            let cols = packed_cols * 2; // 2 FP4 values per byte

            Ok(FP4Weight { packed, scales, global_scale, rows, cols })
        };

        let mut layers = Vec::with_capacity(config.num_layers as usize);
        for i in 0..config.num_layers {
            let lc = &layer_configs[i as usize];
            let lp = format!("{prefix}.layers.{i}");
            eprintln!("[nvfp4] Loading layer {i}/{} ({})...",
                config.num_layers, if lc.is_global { "global" } else { "sliding" });

            let attn_norm = load_norm(&format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = load_norm(&format!("{lp}.post_attention_layernorm.weight"))?;
            let pre_ffn_norm = load_norm(&format!("{lp}.pre_feedforward_layernorm.weight"))?;
            let post_ffn_norm = load_norm(&format!("{lp}.post_feedforward_layernorm.weight"))?;
            let q_norm = load_norm(&format!("{lp}.self_attn.q_norm.weight"))?;
            let k_norm = load_norm(&format!("{lp}.self_attn.k_norm.weight"))?;

            let layer_scalar = loader.load_f32(&format!("{lp}.layer_scalar"), device)
                .ok().and_then(|t| t.to_host(device).ok())
                .and_then(|v| v.first().copied()).unwrap_or(1.0);

            let wq = load_fp4(&format!("{lp}.self_attn.q_proj"))?;
            let wk = load_fp4(&format!("{lp}.self_attn.k_proj"))?;
            let wv = load_fp4(&format!("{lp}.self_attn.v_proj")).ok();
            let wo = load_fp4(&format!("{lp}.self_attn.o_proj"))?;

            let w_gate = load_fp4(&format!("{lp}.mlp.gate_proj"))?;
            let w_up = load_fp4(&format!("{lp}.mlp.up_proj"))?;
            let w_down = load_fp4(&format!("{lp}.mlp.down_proj"))?;

            layers.push(NVFP4LayerWeights {
                attn_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm,
                q_norm, k_norm, layer_scalar,
                wq, wk, wv, wo, w_gate, w_up, w_down,
            });
        }

        let final_norm = load_norm(&format!("{prefix}.norm.weight"))?;

        eprintln!("[nvfp4] Model loaded!");
        Ok(Self { config, embed_tokens, layers, final_norm })
    }
}

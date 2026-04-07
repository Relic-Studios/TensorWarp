//! Gemma 4 model loader.
//!
//! Loads Gemma 4 31B (and Gemma 3 27B) from HuggingFace SafeTensors format.
//! Handles the Gemma-specific architecture: sliding/global attention pattern,
//! shared K=V projections, GeGLU activation, dual RoPE, QK-norm.

use serde::Deserialize;
use warp_ir::{DType, Shape};
use warp_kernels::cache::KernelCache;
use warp_kernels::device::WarpDevice;
use warp_kernels::tensor::GpuTensor;
use warp_kernels::transformer::{
    GemmaConfig, GemmaLayerAttentionConfig, QuantizedBlockWeights, TransformerConfig,
};

use crate::safetensors_loader::{ShardedSafeTensorsLoader, LoaderError};

/// Top-level HuggingFace config.json wrapper.
/// Gemma 4 nests model config under `text_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct GemmaHFConfigWrapper {
    #[serde(default)]
    pub text_config: Option<GemmaHFConfig>,
    // Top-level fields (Gemma 3 puts them here, Gemma 4 nests them)
    #[serde(default)]
    pub hidden_size: Option<u32>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub model_type: Option<String>,
}

/// HuggingFace config.json for Gemma 4 / Gemma 3 text model.
#[derive(Debug, Clone, Deserialize)]
pub struct GemmaHFConfig {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub num_hidden_layers: u32,
    pub vocab_size: u32,
    #[serde(default = "default_head_dim")]
    pub head_dim: u32,
    #[serde(default)]
    pub global_head_dim: Option<u32>,
    #[serde(default)]
    pub num_global_key_value_heads: Option<u32>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_local_base_freq: Option<f64>,
    #[serde(default = "default_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: u32,
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: u32,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub attention_k_eq_v: Option<bool>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub hidden_activation: Option<String>,
    #[serde(default)]
    pub max_position_embeddings: Option<u32>,
    /// Explicit per-layer attention types (Gemma 4).
    /// Values: "sliding_attention" or "full_attention"
    #[serde(default)]
    pub layer_types: Vec<String>,
}

fn default_head_dim() -> u32 { 256 }
fn default_rope_theta() -> f64 { 10000.0 }
fn default_norm_eps() -> f64 { 1e-6 }
fn default_sliding_window() -> u32 { 1024 }
fn default_sliding_window_pattern() -> u32 { 6 }

impl GemmaHFConfig {
    /// Load from a config.json file.
    /// Handles both Gemma 4 (nested under `text_config`) and Gemma 3 (top-level) formats.
    pub fn from_json(path: &str) -> Result<Self, LoaderError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| LoaderError::Io(e.to_string()))?;

        // Try parsing as wrapper first (Gemma 4 with nested text_config)
        if let Ok(wrapper) = serde_json::from_str::<GemmaHFConfigWrapper>(&content) {
            if let Some(mut tc) = wrapper.text_config {
                // Inherit tie_word_embeddings from top level if not set in text_config
                if !tc.tie_word_embeddings && wrapper.tie_word_embeddings {
                    tc.tie_word_embeddings = wrapper.tie_word_embeddings;
                }
                return Ok(tc);
            }
        }

        // Fall back to direct parse (Gemma 3 style, top-level fields)
        serde_json::from_str(&content)
            .map_err(|e| LoaderError::Config(e.to_string()))
    }

    /// Convert to our internal GemmaConfig.
    pub fn to_gemma_config(&self) -> GemmaConfig {
        let global_head_dim = self.global_head_dim.unwrap_or(self.head_dim);
        let num_global_kv_heads = self.num_global_key_value_heads.unwrap_or(self.num_key_value_heads);
        let rope_theta_sliding = self.rope_local_base_freq.unwrap_or(10000.0) as f32;
        let rope_theta_global = self.rope_theta as f32;

        // Derive sliding_window_pattern from layer_types if available
        let pattern = if !self.layer_types.is_empty() {
            // Count consecutive sliding layers before first global
            let first_global = self.layer_types.iter()
                .position(|t| t == "full_attention")
                .unwrap_or(self.layer_types.len());
            (first_global + 1) as u32
        } else {
            self.sliding_window_pattern
        };

        GemmaConfig {
            hidden_size: self.hidden_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            num_global_kv_heads,
            head_dim: self.head_dim,
            global_head_dim,
            ffn_dim: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_layers: self.num_hidden_layers,
            norm_eps: self.rms_norm_eps as f32,
            sliding_window: self.sliding_window,
            sliding_window_pattern: pattern,
            rope_theta: rope_theta_sliding,
            rope_theta_global,
            partial_rotary_factor: self.partial_rotary_factor.unwrap_or(1.0),
            k_eq_v: self.attention_k_eq_v.unwrap_or(false),
            final_logit_softcapping: self.final_logit_softcapping.unwrap_or(0.0),
            tie_word_embeddings: self.tie_word_embeddings,
        }
    }

    /// Convert to a basic TransformerConfig (for shared kernel infrastructure).
    /// Uses the sliding layer dimensions since those are the most common.
    pub fn to_transformer_config(&self) -> TransformerConfig {
        TransformerConfig {
            hidden_size: self.hidden_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            ffn_dim: self.intermediate_size,
            rope_base: self.rope_local_base_freq.unwrap_or(10000.0) as f32,
            norm_eps: self.rms_norm_eps as f32,
            attention_mode: warp_kernels::transformer::AttentionMode::SlidingWindow {
                window_size: self.sliding_window,
            },
        }
    }

    /// Check if this looks like a Gemma model (vs LLaMA/Qwen).
    pub fn is_gemma(config_json: &str) -> bool {
        config_json.contains("\"gemma") || config_json.contains("sliding_window_pattern")
            || config_json.contains("attention_k_eq_v")
    }
}

/// A loaded Gemma model with Q4 quantized weights.
pub struct GemmaModelQ4 {
    pub config: GemmaConfig,
    pub layer_configs: Vec<GemmaLayerAttentionConfig>,
    pub embed_tokens: GpuTensor<f32>,
    pub layers: Vec<QuantizedBlockWeights>,
    pub final_norm: GpuTensor<f32>,
    /// LM head — may be the transposed embedding table (tie_word_embeddings).
    pub lm_head: GpuTensor<f32>,
    pub cache: KernelCache,
}

impl GemmaModelQ4 {
    /// Load Gemma model from SafeTensors with Q4 quantization.
    pub fn load(
        loader: &ShardedSafeTensorsLoader,
        hf_config: &GemmaHFConfig,
        device: &WarpDevice,
    ) -> Result<Self, LoaderError> {
        let config = hf_config.to_gemma_config();
        let layer_configs = config.layer_configs();
        let h = config.hidden_size;
        let ffn = config.ffn_dim;

        eprintln!("Loading Gemma model: {} layers, H={}, FFN={}, vocab={}",
            config.num_layers, h, ffn, config.vocab_size);
        eprintln!("  Attention pattern: {} sliding (window={}) + {} global",
            config.num_layers - config.num_layers / config.sliding_window_pattern,
            config.sliding_window,
            config.num_layers / config.sliding_window_pattern);
        if config.k_eq_v { eprintln!("  K=V shared projections enabled"); }

        // Embedding
        let embed_tokens = loader.load_f32("model.embed_tokens.weight", device)?;

        // Layers
        let mut layers = Vec::with_capacity(config.num_layers as usize);
        for i in 0..config.num_layers {
            let lc = &layer_configs[i as usize];
            let kv_dim = config.kv_dim_for_layer(lc);
            let prefix = format!("model.layers.{i}");

            eprintln!("  Loading layer {i}/{} ({}, kv_heads={}, head_dim={})...",
                config.num_layers,
                if lc.is_global { "global" } else { "sliding" },
                lc.num_kv_heads, lc.head_dim);

            let layer = load_gemma_layer_q4(loader, device, &prefix, h, kv_dim, ffn, &config)?;
            layers.push(layer);
        }

        // Final norm
        let final_norm = loader.load_f32("model.norm.weight", device)?;

        // LM head (tied to embedding for Gemma)
        let lm_head = if config.tie_word_embeddings {
            eprintln!("  Tied embeddings: transposing embed_tokens for lm_head");
            loader.load_f32_transposed("model.embed_tokens.weight", device)?
        } else {
            loader.load_f32_transposed("lm_head.weight", device)?
        };

        let layer_bytes: usize = layers.iter().map(|l| {
            l.wq.size_bytes() + l.wk.size_bytes() + l.wv.size_bytes() + l.wo.size_bytes()
            + l.w_gate.size_bytes() + l.w_up.size_bytes() + l.w_down.size_bytes()
            + l.attn_norm.size_bytes() + l.ffn_norm.size_bytes()
        }).sum();
        eprintln!("Gemma model loaded: {:.1} GB",
            (embed_tokens.size_bytes() + final_norm.size_bytes() + lm_head.size_bytes()
             + layer_bytes) as f64 / 1e9);

        Ok(Self {
            config,
            layer_configs,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            cache: KernelCache::new(),
        })
    }
}

/// Load a single Gemma layer with Q4 quantization.
fn load_gemma_layer_q4(
    loader: &ShardedSafeTensorsLoader,
    device: &WarpDevice,
    prefix: &str,
    h: u32,
    kv_dim: u32,
    ffn: u32,
    config: &GemmaConfig,
) -> Result<QuantizedBlockWeights, LoaderError> {
    // Gemma weight names follow the same pattern as LLaMA
    // Biases: Gemma doesn't use QKV biases
    let bq = loader.load_f32(&format!("{prefix}.self_attn.q_proj.bias"), device).ok();
    let bk = loader.load_f32(&format!("{prefix}.self_attn.k_proj.bias"), device).ok();
    let bv = if config.k_eq_v {
        // K=V: no separate V bias
        bk.as_ref().map(|t| {
            let data = t.to_host(device).unwrap();
            GpuTensor::from_host(device, &data, t.shape.clone(), DType::F32).unwrap()
        })
    } else {
        loader.load_f32(&format!("{prefix}.self_attn.v_proj.bias"), device).ok()
    };

    let wq = load_and_quantize_q4(loader, device, &format!("{prefix}.self_attn.q_proj.weight"), h, h)?;
    let wk = load_and_quantize_q4(loader, device, &format!("{prefix}.self_attn.k_proj.weight"), h, kv_dim)?;

    // K=V sharing: V projection uses the same weights as K
    let wv = if config.k_eq_v {
        load_and_quantize_q4(loader, device, &format!("{prefix}.self_attn.k_proj.weight"), h, kv_dim)?
    } else {
        load_and_quantize_q4(loader, device, &format!("{prefix}.self_attn.v_proj.weight"), h, kv_dim)?
    };

    Ok(QuantizedBlockWeights {
        attn_norm: loader.load_f32(&format!("{prefix}.input_layernorm.weight"), device)?,
        wq, wk, wv,
        wo: load_and_quantize_q4(loader, device, &format!("{prefix}.self_attn.o_proj.weight"), h, h)?,
        ffn_norm: loader.load_f32(&format!("{prefix}.post_attention_layernorm.weight"), device)?,
        w_gate: load_and_quantize_q4(loader, device, &format!("{prefix}.mlp.gate_proj.weight"), h, ffn)?,
        w_up: load_and_quantize_q4(loader, device, &format!("{prefix}.mlp.up_proj.weight"), h, ffn)?,
        w_down: load_and_quantize_q4(loader, device, &format!("{prefix}.mlp.down_proj.weight"), ffn, h)?,
        bq, bk, bv,
        wq_bm: None, wk_bm: None, wv_bm: None, wo_bm: None,
        w_gate_bm: None, w_up_bm: None, w_down_bm: None,
    })
}

/// Load a weight tensor, transpose it, and quantize to Q4_0.
fn load_and_quantize_q4(
    loader: &ShardedSafeTensorsLoader,
    device: &WarpDevice,
    name: &str,
    k: u32,
    n: u32,
) -> Result<GpuTensor<u8>, LoaderError> {
    let cache = KernelCache::new();
    let w = loader.load_f32_transposed(name, device)?;
    let q = warp_kernels::quantize::quantize_weights_q4_0(&cache, device, &w, k, n)
        .map_err(|e| LoaderError::Device(e.to_string()))?;
    Ok(q)
}

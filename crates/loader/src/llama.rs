//! LLaMA model definition and weight loading.
//!
//! Maps HuggingFace LLaMA weight names to TensorWarp's transformer block
//! structure. Supports LLaMA 2/3, Mistral, and any model using the same
//! weight naming convention.

use crate::safetensors_loader::{LoaderError, SafeTensorsLoader};
use warp_kernels::cache::KernelCache;
use warp_kernels::device::{DeviceError, WarpDevice};
use warp_kernels::tensor::GpuTensor;
use warp_kernels::transformer::{TransformerBlockWeights, TransformerConfig};

/// LLaMA model configuration.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: u32,
    pub intermediate_size: u32, // FFN dim
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub num_hidden_layers: u32,
    pub vocab_size: u32,
    pub rope_theta: Option<f64>,
    pub rms_norm_eps: Option<f64>,
    pub max_position_embeddings: Option<u32>,
}

impl LlamaConfig {
    pub fn head_dim(&self) -> u32 {
        self.hidden_size / self.num_attention_heads
    }

    pub fn to_transformer_config(&self) -> TransformerConfig {
        TransformerConfig {
            hidden_size: self.hidden_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim(),
            ffn_dim: self.intermediate_size,
            rope_base: self.rope_theta.unwrap_or(10000.0) as f32,
            norm_eps: self.rms_norm_eps.unwrap_or(1e-6) as f32,
        }
    }

    /// Load config from a HuggingFace config.json file.
    pub fn from_json(path: &str) -> Result<Self, LoaderError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| LoaderError::Io(e.to_string()))?;
        serde_json::from_str(&content)
            .map_err(|e| LoaderError::Config(e.to_string()))
    }

    /// Common model presets.
    pub fn llama_7b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            num_hidden_layers: 32,
            vocab_size: 32000,
            rope_theta: Some(10000.0),
            rms_norm_eps: Some(1e-6),
            max_position_embeddings: Some(4096),
        }
    }

    pub fn llama_3_8b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 14336,
            num_attention_heads: 32,
            num_key_value_heads: 8, // GQA
            num_hidden_layers: 32,
            vocab_size: 128256,
            rope_theta: Some(500000.0),
            rms_norm_eps: Some(1e-5),
            max_position_embeddings: Some(8192),
        }
    }

    pub fn tiny_llama() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 5632,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            num_hidden_layers: 22,
            vocab_size: 32000,
            rope_theta: Some(10000.0),
            rms_norm_eps: Some(1e-5),
            max_position_embeddings: Some(2048),
        }
    }
}

/// A loaded LLaMA model.
pub struct LlamaModel {
    pub config: LlamaConfig,
    pub transformer_config: TransformerConfig,
    /// Token embedding table [vocab_size, hidden_size]
    pub embed_tokens: GpuTensor<f32>,
    /// Per-layer weights
    pub layers: Vec<TransformerBlockWeights>,
    /// Final RMSNorm weight [hidden_size]
    pub final_norm: GpuTensor<f32>,
    /// Output projection (lm_head) [vocab_size, hidden_size]
    pub lm_head: GpuTensor<f32>,
}

impl LlamaModel {
    /// Load a LLaMA model from SafeTensors files.
    ///
    /// `model_path` should be the directory containing the .safetensors
    /// and config.json files, OR a single .safetensors file path.
    pub fn load(
        loader: &SafeTensorsLoader,
        config: &LlamaConfig,
        device: &WarpDevice,
    ) -> Result<Self, LoaderError> {
        let tc = config.to_transformer_config();
        let h = config.hidden_size as usize;
        let kv_dim = (config.num_key_value_heads * config.head_dim()) as usize;
        let ffn = config.intermediate_size as usize;

        // Load embedding
        let embed_tokens = loader.load_f32("model.embed_tokens.weight", device)?;

        // Load layers
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let layer = load_layer(loader, device, &prefix, h, kv_dim, ffn)?;
            layers.push(layer);
        }

        // Final norm
        let final_norm = loader.load_f32("model.norm.weight", device)?;

        // LM head (output projection)
        let lm_head = loader.load_f32("lm_head.weight", device)?;

        Ok(Self {
            config: config.clone(),
            transformer_config: tc,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
        })
    }

    /// How much GPU memory this model uses (approximate).
    pub fn memory_bytes(&self) -> usize {
        let mut total = self.embed_tokens.size_bytes() + self.final_norm.size_bytes() + self.lm_head.size_bytes();
        for layer in &self.layers {
            total += layer.attn_norm.size_bytes()
                + layer.wq.size_bytes()
                + layer.wk.size_bytes()
                + layer.wv.size_bytes()
                + layer.wo.size_bytes()
                + layer.ffn_norm.size_bytes()
                + layer.w_gate.size_bytes()
                + layer.w_up.size_bytes()
                + layer.w_down.size_bytes();
        }
        total
    }

    pub fn summary(&self) -> String {
        format!(
            "LlamaModel: {} layers, H={}, FFN={}, vocab={}, {:.1} GB",
            self.layers.len(),
            self.config.hidden_size,
            self.config.intermediate_size,
            self.config.vocab_size,
            self.memory_bytes() as f64 / 1e9,
        )
    }
}

/// Load weights for a single transformer layer.
fn load_layer(
    loader: &SafeTensorsLoader,
    device: &WarpDevice,
    prefix: &str,
    h: usize,
    kv_dim: usize,
    ffn: usize,
) -> Result<TransformerBlockWeights, LoaderError> {
    Ok(TransformerBlockWeights {
        attn_norm: loader.load_f32(&format!("{prefix}.input_layernorm.weight"), device)?,
        wq: loader.load_f32(&format!("{prefix}.self_attn.q_proj.weight"), device)?,
        wk: loader.load_f32(&format!("{prefix}.self_attn.k_proj.weight"), device)?,
        wv: loader.load_f32(&format!("{prefix}.self_attn.v_proj.weight"), device)?,
        wo: loader.load_f32(&format!("{prefix}.self_attn.o_proj.weight"), device)?,
        ffn_norm: loader.load_f32(&format!("{prefix}.post_attention_layernorm.weight"), device)?,
        w_gate: loader.load_f32(&format!("{prefix}.mlp.gate_proj.weight"), device)?,
        w_up: loader.load_f32(&format!("{prefix}.mlp.up_proj.weight"), device)?,
        w_down: loader.load_f32(&format!("{prefix}.mlp.down_proj.weight"), device)?,
    })
}

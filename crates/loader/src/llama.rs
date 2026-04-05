//! LLaMA model definition and weight loading.
//!
//! Maps HuggingFace LLaMA weight names to TensorWarp's transformer block
//! structure. Supports LLaMA 2/3, Mistral, and any model using the same
//! weight naming convention.

use crate::safetensors_loader::{LoaderError, SafeTensorsLoader, ShardedSafeTensorsLoader};
use warp_kernels::cache::KernelCache;
use warp_kernels::device::{DeviceError, WarpDevice};
use warp_kernels::tensor::GpuTensor;
use warp_kernels::transformer::{TransformerBlockWeights, TransformerBlockWeightsF16, TransformerConfig};

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
    /// Whether lm_head shares weights with embed_tokens (Qwen, Gemma, etc.)
    #[serde(default)]
    pub tie_word_embeddings: bool,
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
            attention_mode: warp_kernels::transformer::AttentionMode::Standard,
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
            tie_word_embeddings: false,
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
            tie_word_embeddings: false,
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
            tie_word_embeddings: false,
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
        loader: &ShardedSafeTensorsLoader,
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
        // HF stores as [vocab, hidden]. Our GEMM needs [hidden, vocab]. Transpose on load.
        let lm_head = if config.tie_word_embeddings {
            match loader.load_f32_transposed("lm_head.weight", device) {
                Ok(t) => t,
                Err(_) => {
                    log::info!("tie_word_embeddings=true, transposing embed_tokens for lm_head");
                    loader.load_f32_transposed("model.embed_tokens.weight", device)?
                }
            }
        } else {
            match loader.load_f32_transposed("lm_head.weight", device) {
                Ok(t) => t,
                Err(_) => {
                    log::info!("lm_head.weight not found, transposing embed_tokens as fallback");
                    loader.load_f32_transposed("model.embed_tokens.weight", device)?
                }
            }
        };

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
            total += layer.total_memory_bytes();
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
    loader: &ShardedSafeTensorsLoader,
    device: &WarpDevice,
    prefix: &str,
    h: usize,
    kv_dim: usize,
    ffn: usize,
) -> Result<TransformerBlockWeights, LoaderError> {
    // Try loading QKV biases (Qwen/Phi have them, LLaMA/Mistral don't)
    let bq = loader.load_f32(&format!("{prefix}.self_attn.q_proj.bias"), device).ok();
    let bk = loader.load_f32(&format!("{prefix}.self_attn.k_proj.bias"), device).ok();
    let bv = loader.load_f32(&format!("{prefix}.self_attn.v_proj.bias"), device).ok();

    let wq = loader.load_f32_transposed(&format!("{prefix}.self_attn.q_proj.weight"), device)?;
    let wk = loader.load_f32_transposed(&format!("{prefix}.self_attn.k_proj.weight"), device)?;
    let wv = loader.load_f32_transposed(&format!("{prefix}.self_attn.v_proj.weight"), device)?;
    let w_gate = loader.load_f32_transposed(&format!("{prefix}.mlp.gate_proj.weight"), device)?;
    let w_up = loader.load_f32_transposed(&format!("{prefix}.mlp.up_proj.weight"), device)?;

    // Fused QKV and gate+up weights are available via fuse_projections()
    // but disabled by default because the extra memory (1GB+) causes regression
    // on bandwidth-bound decode. Enable when memory is not a constraint.

    Ok(TransformerBlockWeights {
        // Norm weights are 1D — no transposition needed
        attn_norm: loader.load_f32(&format!("{prefix}.input_layernorm.weight"), device)?,
        // Weight matrices are [out, in] in HF — transpose to [in, out] for our GEMM
        wq,
        wk,
        wv,
        wo: loader.load_f32_transposed(&format!("{prefix}.self_attn.o_proj.weight"), device)?,
        ffn_norm: loader.load_f32(&format!("{prefix}.post_attention_layernorm.weight"), device)?,
        w_gate,
        w_up,
        w_down: loader.load_f32_transposed(&format!("{prefix}.mlp.down_proj.weight"), device)?,
        bq,
        bk,
        bv,
        wqkv: None, // Disabled: fused weights add 1GB+ memory, causing regression
        bqkv: None,
        w_gate_up: None,
    })
}

/// Load weights for a single transformer layer in FP16 (norms/biases stay F32).
fn load_layer_f16(
    loader: &ShardedSafeTensorsLoader,
    device: &WarpDevice,
    prefix: &str,
    h: usize,
    kv_dim: usize,
    ffn: usize,
) -> Result<TransformerBlockWeightsF16, LoaderError> {
    // Biases stay F32
    let bq = loader.load_f32(&format!("{prefix}.self_attn.q_proj.bias"), device).ok();
    let bk = loader.load_f32(&format!("{prefix}.self_attn.k_proj.bias"), device).ok();
    let bv = loader.load_f32(&format!("{prefix}.self_attn.v_proj.bias"), device).ok();

    let attn_norm = loader.load_f32(&format!("{prefix}.input_layernorm.weight"), device)?;
    let ffn_norm = loader.load_f32(&format!("{prefix}.post_attention_layernorm.weight"), device)?;

    // Pre-compute FP16 norm weights for the full FP16 pipeline
    let cache = warp_kernels::cache::KernelCache::new();
    let mut attn_norm_f16 = warp_kernels::tensor::GpuTensor::<half::f16>::zeros(device, attn_norm.shape.clone(), warp_ir::DType::F16)
        .map_err(|e| LoaderError::Device(e.to_string()))?;
    warp_kernels::fp16::cast_f32_to_f16(&cache, device, &attn_norm, &mut attn_norm_f16)
        .map_err(|e| LoaderError::Device(e.to_string()))?;
    let mut ffn_norm_f16 = warp_kernels::tensor::GpuTensor::<half::f16>::zeros(device, ffn_norm.shape.clone(), warp_ir::DType::F16)
        .map_err(|e| LoaderError::Device(e.to_string()))?;
    warp_kernels::fp16::cast_f32_to_f16(&cache, device, &ffn_norm, &mut ffn_norm_f16)
        .map_err(|e| LoaderError::Device(e.to_string()))?;

    Ok(TransformerBlockWeightsF16 {
        attn_norm,
        ffn_norm,
        attn_norm_f16: Some(attn_norm_f16),
        ffn_norm_f16: Some(ffn_norm_f16),
        // Weight matrices loaded as FP16 — half the memory bandwidth
        wq: loader.load_f16_transposed(&format!("{prefix}.self_attn.q_proj.weight"), device)?,
        wk: loader.load_f16_transposed(&format!("{prefix}.self_attn.k_proj.weight"), device)?,
        wv: loader.load_f16_transposed(&format!("{prefix}.self_attn.v_proj.weight"), device)?,
        wo: loader.load_f16_transposed(&format!("{prefix}.self_attn.o_proj.weight"), device)?,
        w_gate: loader.load_f16_transposed(&format!("{prefix}.mlp.gate_proj.weight"), device)?,
        w_up: loader.load_f16_transposed(&format!("{prefix}.mlp.up_proj.weight"), device)?,
        w_down: loader.load_f16_transposed(&format!("{prefix}.mlp.down_proj.weight"), device)?,
        // Pre-compute FP16 biases before moving originals into struct
        bq_f16: if let Some(ref b) = bq {
            let mut bf = warp_kernels::tensor::GpuTensor::<half::f16>::zeros(device, b.shape.clone(), warp_ir::DType::F16)
                .map_err(|e| LoaderError::Device(e.to_string()))?;
            warp_kernels::fp16::cast_f32_to_f16(&cache, device, b, &mut bf)
                .map_err(|e| LoaderError::Device(e.to_string()))?;
            Some(bf)
        } else { None },
        bk_f16: if let Some(ref b) = bk {
            let mut bf = warp_kernels::tensor::GpuTensor::<half::f16>::zeros(device, b.shape.clone(), warp_ir::DType::F16)
                .map_err(|e| LoaderError::Device(e.to_string()))?;
            warp_kernels::fp16::cast_f32_to_f16(&cache, device, b, &mut bf)
                .map_err(|e| LoaderError::Device(e.to_string()))?;
            Some(bf)
        } else { None },
        bv_f16: if let Some(ref b) = bv {
            let mut bf = warp_kernels::tensor::GpuTensor::<half::f16>::zeros(device, b.shape.clone(), warp_ir::DType::F16)
                .map_err(|e| LoaderError::Device(e.to_string()))?;
            warp_kernels::fp16::cast_f32_to_f16(&cache, device, b, &mut bf)
                .map_err(|e| LoaderError::Device(e.to_string()))?;
            Some(bf)
        } else { None },
        bq,
        bk,
        bv,
    })
}

/// A loaded LLaMA model with FP16 weight matrices for mixed-precision inference.
///
/// Weight matrices (Q/K/V/O/gate/up/down) are stored in FP16 — halving memory
/// bandwidth for decode GEMMs. Norms, biases, embeddings, and lm_head stay F32
/// for numerical stability.
pub struct LlamaModelF16 {
    pub config: LlamaConfig,
    pub transformer_config: TransformerConfig,
    /// Token embedding table [vocab_size, hidden_size] — F32
    pub embed_tokens: GpuTensor<f32>,
    /// Per-layer weights with FP16 weight matrices
    pub layers: Vec<TransformerBlockWeightsF16>,
    /// Final RMSNorm weight [hidden_size] — F32
    pub final_norm: GpuTensor<f32>,
    /// Output projection (lm_head) [hidden_size, vocab_size] — F32
    pub lm_head: GpuTensor<f32>,
}

impl LlamaModelF16 {
    /// Load a LLaMA model with FP16 weight matrices for mixed-precision decode.
    pub fn load_f16(
        loader: &ShardedSafeTensorsLoader,
        config: &LlamaConfig,
        device: &WarpDevice,
    ) -> Result<Self, LoaderError> {
        let tc = config.to_transformer_config();
        let h = config.hidden_size as usize;
        let kv_dim = (config.num_key_value_heads * config.head_dim()) as usize;
        let ffn = config.intermediate_size as usize;

        // Embedding stays F32
        let embed_tokens = loader.load_f32("model.embed_tokens.weight", device)?;

        // Load layers with FP16 weight matrices
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let layer = load_layer_f16(loader, device, &prefix, h, kv_dim, ffn)?;
            layers.push(layer);
        }

        // Final norm stays F32
        let final_norm = loader.load_f32("model.norm.weight", device)?;

        // LM head stays F32 (small relative to layer weights, needs precision for logits)
        let lm_head = if config.tie_word_embeddings {
            match loader.load_f32_transposed("lm_head.weight", device) {
                Ok(t) => t,
                Err(_) => {
                    log::info!("tie_word_embeddings=true, transposing embed_tokens for lm_head");
                    loader.load_f32_transposed("model.embed_tokens.weight", device)?
                }
            }
        } else {
            match loader.load_f32_transposed("lm_head.weight", device) {
                Ok(t) => t,
                Err(_) => {
                    log::info!("lm_head.weight not found, transposing embed_tokens as fallback");
                    loader.load_f32_transposed("model.embed_tokens.weight", device)?
                }
            }
        };

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
        let mut total = self.embed_tokens.size_bytes()
            + self.final_norm.size_bytes()
            + self.lm_head.size_bytes();
        for layer in &self.layers {
            total += layer.total_memory_bytes();
        }
        total
    }

    pub fn summary(&self) -> String {
        format!(
            "LlamaModelF16: {} layers, H={}, FFN={}, vocab={}, {:.1} GB (FP16 weights)",
            self.layers.len(),
            self.config.hidden_size,
            self.config.intermediate_size,
            self.config.vocab_size,
            self.memory_bytes() as f64 / 1e9,
        )
    }
}

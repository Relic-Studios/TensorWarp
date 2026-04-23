//! Qwen3.5 hybrid loader (Gated DeltaNet + full attention).
//!
//! Tensor names verified against actual Huihui-Qwen3.5-9B-abliterated
//! checkpoint (`config.architectures = ["Qwen3_5ForCausalLM"]`):
//!
//!   model.language_model.embed_tokens.weight                 [V, D]
//!   model.language_model.layers.{i}.input_layernorm.weight   [D]
//!   model.language_model.layers.{i}.post_attention_layernorm.weight [D]
//!
//!   --- Linear-attn layers (3 of every 4) ---
//!   model.language_model.layers.{i}.linear_attn.A_log              [H_v]
//!   model.language_model.layers.{i}.linear_attn.dt_bias            [H_v]
//!   model.language_model.layers.{i}.linear_attn.conv1d.weight      [D_conv, 1, K]
//!   model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight [D_conv, D]
//!   model.language_model.layers.{i}.linear_attn.in_proj_z.weight   [V_dim, D]
//!   model.language_model.layers.{i}.linear_attn.in_proj_a.weight   [H_v, D]
//!   model.language_model.layers.{i}.linear_attn.in_proj_b.weight   [H_v, D]
//!   model.language_model.layers.{i}.linear_attn.norm.weight        [dv]
//!   model.language_model.layers.{i}.linear_attn.out_proj.weight    [D, V_dim]
//!
//!   --- Full-attn layers (1 of every 4) ---
//!   model.language_model.layers.{i}.self_attn.q_proj.weight        [H * head_dim, D]   = [8192, 4096]
//!   model.language_model.layers.{i}.self_attn.k_proj.weight        [H_kv * head_dim, D] = [1024, 4096]
//!   model.language_model.layers.{i}.self_attn.v_proj.weight        [H_kv * head_dim, D]
//!   model.language_model.layers.{i}.self_attn.o_proj.weight        [D, H * head_dim]
//!   model.language_model.layers.{i}.self_attn.q_norm.weight        [head_dim] = [256]
//!   model.language_model.layers.{i}.self_attn.k_norm.weight        [head_dim]
//!
//!   --- MLP (every layer) ---
//!   model.language_model.layers.{i}.mlp.gate_proj.weight           [D_ffn, D]
//!   model.language_model.layers.{i}.mlp.up_proj.weight             [D_ffn, D]
//!   model.language_model.layers.{i}.mlp.down_proj.weight           [D, D_ffn]
//!
//!   model.language_model.norm.weight                               [D]
//!   lm_head.weight                                                 [V, D]

use warp_ir::{DType, Shape};
use warp_kernels::cache::KernelCache;
use warp_kernels::device::WarpDevice;
use warp_kernels::tensor::GpuTensor;
use warp_kernels::gated_delta_net::{
    DeltaNetLayerState, GatedDeltaNetConfig, GatedDeltaNetWeights,
};

use crate::safetensors_loader::{ShardedSafeTensorsLoader, LoaderError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35LayerKind {
    Linear,
    Full,
}

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub hidden_size: usize,                  // 4096
    pub num_hidden_layers: usize,            // 32
    pub intermediate_size: usize,            // 12288
    pub head_dim: usize,                     // 256 (full-attn)
    pub num_attention_heads: usize,          // 32 (full-attn query heads)
    pub num_key_value_heads: usize,          // 4 (corrected: very aggressive GQA)
    pub vocab_size: usize,                   // 248077-ish
    pub max_position_embeddings: usize,      // 262144
    pub rope_theta: f32,                     // ~1e6
    pub rmsnorm_eps: f32,                    // 1e-6
    pub full_attention_interval: usize,      // 4
    pub gdn: GatedDeltaNetConfig,
}

impl Qwen35Config {
    pub fn qwen3_5_9b_abliterated() -> Self {
        Self {
            hidden_size: 4096,
            num_hidden_layers: 32,
            intermediate_size: 12288,
            head_dim: 256,
            num_attention_heads: 32,
            num_key_value_heads: 4,             // verified: not 16!
            vocab_size: 248078,                  // verified
            max_position_embeddings: 262144,
            rope_theta: 1_000_000.0,
            rmsnorm_eps: 1e-6,
            full_attention_interval: 4,
            gdn: GatedDeltaNetConfig::qwen3_5_9b(),
        }
    }

    pub fn layer_kind(&self, layer_idx: usize) -> Qwen35LayerKind {
        if (layer_idx + 1) % self.full_attention_interval == 0 {
            Qwen35LayerKind::Full
        } else {
            Qwen35LayerKind::Linear
        }
    }

    pub fn count_kinds(&self) -> (usize, usize) {
        let mut linear = 0;
        let mut full = 0;
        for i in 0..self.num_hidden_layers {
            match self.layer_kind(i) {
                Qwen35LayerKind::Linear => linear += 1,
                Qwen35LayerKind::Full => full += 1,
            }
        }
        (linear, full)
    }
}

pub struct Qwen35FullAttentionWeights {
    pub input_norm: GpuTensor<f32>,
    pub wq: GpuTensor<half::f16>,
    pub wk: GpuTensor<half::f16>,
    pub wv: GpuTensor<half::f16>,
    pub wo: GpuTensor<half::f16>,
    pub q_norm: GpuTensor<f32>,
    pub k_norm: GpuTensor<f32>,
    pub post_attn_norm: GpuTensor<f32>,
    pub w_gate: GpuTensor<half::f16>,
    pub w_up: GpuTensor<half::f16>,
    pub w_down: GpuTensor<half::f16>,
}

pub enum Qwen35LayerWeights {
    Linear {
        input_norm: GpuTensor<f32>,
        gdn: GatedDeltaNetWeights,
        post_attn_norm: GpuTensor<f32>,
        w_gate: GpuTensor<half::f16>,
        w_up: GpuTensor<half::f16>,
        w_down: GpuTensor<half::f16>,
    },
    Full(Qwen35FullAttentionWeights),
}

pub struct Qwen35Model {
    pub config: Qwen35Config,
    pub embed_tokens: GpuTensor<half::f16>,
    pub layers: Vec<Qwen35LayerWeights>,
    pub final_norm: GpuTensor<f32>,
    pub lm_head: GpuTensor<half::f16>,
}

// ─── Loader ─────────────────────────────────────────────────────────────

pub fn load_qwen3_5(
    loader: &ShardedSafeTensorsLoader,
    config: &Qwen35Config,
    device: &WarpDevice,
) -> Result<Qwen35Model, LoaderError> {
    let kcache = KernelCache::new();

    // Auto-detect prefix (multimodal models have model.language_model.* prefix)
    let prefix = if loader
        .load_f32("model.language_model.embed_tokens.weight", device)
        .is_ok()
    {
        "model.language_model"
    } else {
        "model"
    };
    eprintln!("[qwen3_5] using tensor prefix: {prefix}");

    let load_f16 = |name: &str| -> Result<GpuTensor<half::f16>, LoaderError> {
        let w_f32 = loader.load_f32(name, device)?;
        let mut w_f16 =
            GpuTensor::<half::f16>::zeros(device, w_f32.shape.clone(), DType::F16)
                .map_err(|e| LoaderError::Device(e.to_string()))?;
        warp_kernels::fp16::cast_f32_to_f16(&kcache, device, &w_f32, &mut w_f16)
            .map_err(|e| LoaderError::Device(e.to_string()))?;
        Ok(w_f16)
    };
    let load_norm =
        |name: &str| -> Result<GpuTensor<f32>, LoaderError> { loader.load_f32(name, device) };

    // ── Embedding ─────────────────────────────────────────────────────
    eprintln!("[qwen3_5] loading embedding (vocab={})", config.vocab_size);
    let embed_tokens = load_f16(&format!("{prefix}.embed_tokens.weight"))?;

    // ── Per-layer weights ─────────────────────────────────────────────
    let (n_linear, n_full) = config.count_kinds();
    eprintln!("[qwen3_5] {} layers ({} linear + {} full)",
              config.num_hidden_layers, n_linear, n_full);

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let p = format!("{prefix}.layers.{}", i);
        let kind = config.layer_kind(i);

        let post_attn_norm =
            load_norm(&format!("{p}.post_attention_layernorm.weight"))?;
        let w_gate = load_f16(&format!("{p}.mlp.gate_proj.weight"))?;
        let w_up = load_f16(&format!("{p}.mlp.up_proj.weight"))?;
        let w_down = load_f16(&format!("{p}.mlp.down_proj.weight"))?;
        let input_norm = load_norm(&format!("{p}.input_layernorm.weight"))?;

        let layer = match kind {
            Qwen35LayerKind::Linear => {
                let gdn = GatedDeltaNetWeights {
                    w_qkv:    load_f16(&format!("{p}.linear_attn.in_proj_qkv.weight"))?,
                    w_z:      load_f16(&format!("{p}.linear_attn.in_proj_z.weight"))?,
                    w_b:      load_f16(&format!("{p}.linear_attn.in_proj_b.weight"))?,
                    w_a:      load_f16(&format!("{p}.linear_attn.in_proj_a.weight"))?,
                    conv_w:   load_f16(&format!("{p}.linear_attn.conv1d.weight"))?,
                    a_log:    load_norm(&format!("{p}.linear_attn.A_log"))?,
                    dt_bias:  load_norm(&format!("{p}.linear_attn.dt_bias"))?,
                    norm_w:   load_norm(&format!("{p}.linear_attn.norm.weight"))?,
                    w_out:    load_f16(&format!("{p}.linear_attn.out_proj.weight"))?,
                };
                Qwen35LayerWeights::Linear {
                    input_norm,
                    gdn,
                    post_attn_norm,
                    w_gate, w_up, w_down,
                }
            }
            Qwen35LayerKind::Full => {
                let wq = load_f16(&format!("{p}.self_attn.q_proj.weight"))?;
                let wk = load_f16(&format!("{p}.self_attn.k_proj.weight"))?;
                let wv = load_f16(&format!("{p}.self_attn.v_proj.weight"))?;
                let wo = load_f16(&format!("{p}.self_attn.o_proj.weight"))?;
                let q_norm = load_norm(&format!("{p}.self_attn.q_norm.weight"))?;
                let k_norm = load_norm(&format!("{p}.self_attn.k_norm.weight"))?;
                Qwen35LayerWeights::Full(Qwen35FullAttentionWeights {
                    input_norm,
                    wq, wk, wv, wo,
                    q_norm, k_norm,
                    post_attn_norm,
                    w_gate, w_up, w_down,
                })
            }
        };
        layers.push(layer);
        if i % 4 == 0 || i == config.num_hidden_layers - 1 {
            eprintln!("[qwen3_5]   layer {:2}: {:?} ✓", i, kind);
        }
    }

    let final_norm = load_norm(&format!("{prefix}.norm.weight"))?;
    let lm_head = match load_f16("lm_head.weight") {
        Ok(t) => {
            eprintln!("[qwen3_5] using untied lm_head");
            t
        }
        Err(_) => {
            eprintln!("[qwen3_5] tied lm_head — re-loading embed as fp16");
            load_f16(&format!("{prefix}.embed_tokens.weight"))?
        }
    };

    eprintln!("[qwen3_5] load complete.");
    Ok(Qwen35Model {
        config: config.clone(),
        embed_tokens,
        layers,
        final_norm,
        lm_head,
    })
}

pub fn alloc_delta_state(
    device: &WarpDevice,
    model: &Qwen35Model,
    batch: usize,
) -> Result<Vec<Option<DeltaNetLayerState>>, LoaderError> {
    let mut states = Vec::with_capacity(model.config.num_hidden_layers);
    for layer in &model.layers {
        match layer {
            Qwen35LayerWeights::Linear { .. } => {
                let s = DeltaNetLayerState::new(device, &model.config.gdn, batch)
                    .map_err(|e| LoaderError::Device(e.to_string()))?;
                states.push(Some(s));
            }
            Qwen35LayerWeights::Full(_) => states.push(None),
        }
    }
    Ok(states)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_kind_pattern() {
        let cfg = Qwen35Config::qwen3_5_9b_abliterated();
        assert_eq!(cfg.layer_kind(0), Qwen35LayerKind::Linear);
        assert_eq!(cfg.layer_kind(2), Qwen35LayerKind::Linear);
        assert_eq!(cfg.layer_kind(3), Qwen35LayerKind::Full);
        assert_eq!(cfg.layer_kind(7), Qwen35LayerKind::Full);
        assert_eq!(cfg.layer_kind(31), Qwen35LayerKind::Full);
    }

    #[test]
    fn count_kinds_matches_3to1_ratio() {
        let cfg = Qwen35Config::qwen3_5_9b_abliterated();
        let (linear, full) = cfg.count_kinds();
        assert_eq!(full, 8);
        assert_eq!(linear, 24);
        assert_eq!(linear + full, cfg.num_hidden_layers);
    }
}

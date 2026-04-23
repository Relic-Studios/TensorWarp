//! Qwen3.5 hybrid generation engine.
//!
//! Per decode step:
//!   1. embed_tokens[token_id] → hidden                    [B, D]
//!   2. for each layer i:
//!        if linear:  hidden = gated_delta_net_step(hidden) + ffn(hidden)
//!        if full:    hidden = full_attn_with_gate(hidden) + ffn(hidden)
//!   3. final_norm(hidden) → x_final
//!   4. lm_head @ x_final → logits                         [B, V]
//!   5. sample(logits) → next_token
//!
//! Hybrid layer dispatch is what makes this engine Qwen3.5-specific.
//! All sub-kernels (attention, ffn, norm, embedding, sampling) reuse
//! existing TensorWarp infrastructure where possible — this file is the
//! ORCHESTRATOR, not the heavy compute.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;
use crate::gated_delta_net::{
    DeltaNetLayerState, GatedDeltaNetConfig, GatedDeltaNetStepBuffers,
    GatedDeltaNetWeights, gated_delta_net_step,
};

// ─── Re-exports of types for engine orchestration ────────────────────────
//
// `Qwen35Model` lives in warp-loader because that's where weight loading
// is owned. Engine takes a borrow of the model + per-step state.
//
// To avoid a circular dep, the engine accepts a borrow trait/types
// rather than importing from warp-loader directly. For now we duplicate
// a slim layer enum that mirrors the loader's Qwen35LayerWeights.

pub trait Qwen35Layer {
    fn is_linear(&self) -> bool;
    fn gdn_weights(&self) -> Option<&GatedDeltaNetWeights>;
    fn ffn_norm(&self) -> &GpuTensor<f32>;
    fn ffn_gate(&self) -> &GpuTensor<half::f16>;
    fn ffn_up(&self) -> &GpuTensor<half::f16>;
    fn ffn_down(&self) -> &GpuTensor<half::f16>;
    // Full-attention accessors (None for linear layers)
    fn attn_input_norm(&self) -> Option<&GpuTensor<f32>>;
    fn attn_wq(&self) -> Option<&GpuTensor<half::f16>>;
    fn attn_wk(&self) -> Option<&GpuTensor<half::f16>>;
    fn attn_wv(&self) -> Option<&GpuTensor<half::f16>>;
    fn attn_gate(&self) -> Option<&GpuTensor<half::f16>>;
    fn attn_wo(&self) -> Option<&GpuTensor<half::f16>>;
}

// ─── Top-level engine ────────────────────────────────────────────────────

/// Holds per-step buffers + persistent recurrent state for one Qwen3.5 model.
pub struct Qwen35Engine {
    pub gdn_cfg: GatedDeltaNetConfig,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub max_position_embeddings: usize,
    pub rmsnorm_eps: f32,

    /// Per-layer DeltaNet state (None for full-attn layers).
    pub delta_states: Vec<Option<DeltaNetLayerState>>,
    /// Per-layer step buffers (one for linear-attn layers; full layers use
    /// their own scratch via existing TensorWarp kernels).
    pub gdn_bufs: Vec<Option<GatedDeltaNetStepBuffers>>,

    // Top-level decode buffers
    pub hidden: GpuTensor<f32>,           // [B, D]
    pub residual: GpuTensor<f32>,         // [B, D]
    pub ffn_normed: GpuTensor<f32>,       // [B, D]
    pub ffn_gate_buf: GpuTensor<f32>,     // [B, D_ffn]
    pub ffn_up_buf: GpuTensor<f32>,       // [B, D_ffn]
    pub ffn_silu_buf: GpuTensor<f32>,     // [B, D_ffn]
    pub ffn_out_buf: GpuTensor<f32>,      // [B, D]
    pub final_normed: GpuTensor<f32>,     // [B, D]
    pub logits: GpuTensor<f32>,           // [B, V]
}

impl Qwen35Engine {
    /// Allocate engine state for batch=1 decode.
    pub fn new(
        device: &WarpDevice,
        gdn_cfg: GatedDeltaNetConfig,
        hidden_size: usize,
        vocab_size: usize,
        intermediate_size: usize,
        num_layers: usize,
        max_position_embeddings: usize,
        rmsnorm_eps: f32,
        layer_kinds: &[bool],   // true = linear, false = full
    ) -> Result<Self, DeviceError> {
        let batch = 1;
        let alloc1d = |dim: usize| -> Result<GpuTensor<f32>, DeviceError> {
            GpuTensor::<f32>::zeros(device, Shape::from_static(&[batch, dim]), DType::F32)
        };

        let mut delta_states = Vec::with_capacity(num_layers);
        let mut gdn_bufs = Vec::with_capacity(num_layers);
        for &is_linear in layer_kinds.iter() {
            if is_linear {
                delta_states.push(Some(DeltaNetLayerState::new(device, &gdn_cfg, batch)?));
                gdn_bufs.push(Some(alloc_gdn_bufs(device, &gdn_cfg, batch)?));
            } else {
                delta_states.push(None);
                gdn_bufs.push(None);
            }
        }

        Ok(Self {
            gdn_cfg,
            hidden_size,
            vocab_size,
            num_layers,
            max_position_embeddings,
            rmsnorm_eps,
            delta_states,
            gdn_bufs,
            hidden: alloc1d(hidden_size)?,
            residual: alloc1d(hidden_size)?,
            ffn_normed: alloc1d(hidden_size)?,
            ffn_gate_buf: alloc1d(intermediate_size)?,
            ffn_up_buf: alloc1d(intermediate_size)?,
            ffn_silu_buf: alloc1d(intermediate_size)?,
            ffn_out_buf: alloc1d(hidden_size)?,
            final_normed: alloc1d(hidden_size)?,
            logits: alloc1d(vocab_size)?,
        })
    }

    /// Reset all DeltaNet states + KV caches at session boundary.
    pub fn reset(&mut self, _device: &WarpDevice) -> Result<(), DeviceError> {
        // TODO: zero state matrices, reset KV cache cursor for full layers
        Ok(())
    }

    /// One forward decode step. Returns `&self.logits` (already populated).
    ///
    /// SKELETON — the meat of each layer call is TODO:
    ///   - Embedding lookup (gather kernel; existing in TensorWarp)
    ///   - Linear layer: gated_delta_net_step (DONE) + FFN + residual
    ///   - Full layer:   attention (existing) + output gate + FFN + residual
    ///   - Final norm + LM head matmul + sampling
    ///
    /// All compute kernels needed already exist in TensorWarp (rmsnorm,
    /// swiglu, gemm_q4, attention, sampling, gather). This function is
    /// the orchestration loop.
    pub fn forward_decode_step<L: Qwen35Layer>(
        &mut self,
        device: &WarpDevice,
        cache: &KernelCache,
        token_id: u32,
        position: usize,
        embed_tokens: &GpuTensor<half::f16>,
        layers: &[L],
        final_norm: &GpuTensor<f32>,
        lm_head: &GpuTensor<half::f16>,
    ) -> Result<&GpuTensor<f32>, DeviceError> {
        // ── 1. Embedding lookup ──────────────────────────────────────────
        // hidden = embed_tokens[token_id]    (gather row)
        // TODO: call existing gather kernel
        // crate::gather::gather_row_f16_to_f32(cache, device, embed_tokens, token_id, &mut self.hidden)?;

        // ── 2. Layer loop with hybrid dispatch ───────────────────────────
        for (i, layer) in layers.iter().enumerate() {
            // Save residual: residual = hidden
            // TODO: clone hidden into self.residual

            if layer.is_linear() {
                // Linear (Gated DeltaNet) layer
                let gdn = layer.gdn_weights()
                    .expect("linear layer must have gdn weights");
                let state = self.delta_states[i].as_mut()
                    .expect("linear layer must have delta state");
                let bufs = self.gdn_bufs[i].as_mut()
                    .expect("linear layer must have gdn bufs");
                // Output goes into ffn_normed (a free f32 buffer of [B, D]).
                // Then we swap into hidden for the next layer.
                gated_delta_net_step(
                    device, cache,
                    &self.hidden,         // post-input_layernorm
                    &self.residual,       // pre-input_layernorm (saved at top of layer)
                    gdn, state, bufs,
                    &self.gdn_cfg,
                    &mut self.ffn_normed, // tmp buffer for stage-C output
                )?;
                std::mem::swap(&mut self.hidden, &mut self.ffn_normed);
            } else {
                // Full attention layer
                // TODO: call existing TensorWarp attention kernels with
                // additional output-gate step. Specifically:
                //   1. RMSNorm(hidden) → normed                  [B, D]
                //   2. q = wq @ normed; k = wk @ normed; v = wv @ normed
                //   3. apply RoPE to q, k
                //   4. flash_attention(q, k, v, kv_cache) → attn_out
                //   5. og = sigmoid(W_attn_gate @ normed)        [B, D]
                //   6. attn_gated = og * (W_o @ attn_out)
                //   7. hidden += attn_gated  (residual)
                let _ = layer.attn_input_norm();
                let _ = position;
                // Stub: copy hidden → hidden (no-op)
            }

            // ── FFN block (SwiGLU) — same for both layer kinds ──────────
            // TODO:
            //   1. normed = RMSNorm(hidden, layer.ffn_norm())
            //   2. gate = w_gate @ normed
            //   3. up   = w_up @ normed
            //   4. silu_up = SiLU(gate) * up
            //   5. ffn_out = w_down @ silu_up
            //   6. hidden += ffn_out  (residual)
            let _ = layer.ffn_norm();
            let _ = layer.ffn_gate();
        }

        // ── 3. Final RMSNorm ─────────────────────────────────────────────
        // TODO: rmsnorm(hidden, final_norm) → final_normed
        let _ = final_norm;

        // ── 4. LM head matmul ────────────────────────────────────────────
        // TODO: logits = lm_head @ final_normed   [B, V]
        let _ = lm_head;

        Ok(&self.logits)
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────

fn alloc_gdn_bufs(
    device: &WarpDevice,
    cfg: &GatedDeltaNetConfig,
    batch: usize,
) -> Result<GatedDeltaNetStepBuffers, DeviceError> {
    let alloc = |dims: &[usize]| -> Result<GpuTensor<f32>, DeviceError> {
        GpuTensor::<f32>::zeros(device, Shape::from_static(dims), DType::F32)
    };
    let v_dim = cfg.value_dim();
    let d_conv = cfg.conv_dim();
    let h_v = cfg.num_value_heads;
    let dk = cfg.key_head_dim;
    let dv = cfg.value_head_dim;

    let alloc_f16 = |dims: &[usize]| -> Result<GpuTensor<half::f16>, DeviceError> {
        GpuTensor::<half::f16>::zeros(device, Shape::from_static(dims), DType::F16)
    };

    Ok(GatedDeltaNetStepBuffers {
        hidden_f16:   alloc_f16(&[batch, cfg.hidden_size])?,
        conv_in_pre:  alloc(&[batch, d_conv])?,
        beta_logit:   alloc(&[batch, h_v])?,
        a_logit:      alloc(&[batch, h_v])?,
        q:        alloc(&[batch, h_v, dk])?,
        k:        alloc(&[batch, h_v, dk])?,
        v:        alloc(&[batch, h_v, dv])?,
        z:        alloc(&[batch, v_dim])?,
        beta:     alloc(&[batch, h_v])?,
        g:        alloc(&[batch, h_v])?,
        y:        alloc(&[batch, h_v, dv])?,
        y_gated:  alloc(&[batch, v_dim])?,
    })
}

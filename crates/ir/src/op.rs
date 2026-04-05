//! Operations — the verbs of the Warp IR.
//!
//! Design philosophy: ops are *specific enough* to carry optimization info
//! (attention is a first-class op, not a subgraph) but *general enough*
//! to represent any transformer variant without per-model special cases.
//!
//! Fused ops are explicit — FusedMatMulAdd is a real op, not an annotation.
//! The optimizer *creates* fused ops by pattern-matching on unfused ones.

use serde::{Deserialize, Serialize};

/// Activation function variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Activation {
    Relu,
    Gelu,
    GeluTanh, // GELU approximation used by most LLMs
    Silu,     // SiLU/Swish — used by LLaMA, Mistral, etc.
    Tanh,
    Sigmoid,
}

/// How to handle attention masking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttentionMask {
    /// No mask (full attention).
    None,
    /// Causal (lower-triangular) — the common case for autoregressive.
    Causal,
    /// Explicit mask tensor provided as an input.
    Explicit,
}

/// Reduction operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
}

/// Elementwise binary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
}

/// Elementwise unary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Recip,
    Cast(crate::DType),
}

/// Pooling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PoolMode {
    Max,
    Avg,
    /// Global average pool — reduces spatial dims to 1×1.
    GlobalAvg,
    /// Global max pool.
    GlobalMax,
}

/// Resize / upsample interpolation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResizeMode {
    Nearest,
    Bilinear,
    Bicubic,
}

/// Grid sample interpolation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InterpolationMode {
    Nearest,
    Bilinear,
    Bicubic,
}

/// Grid sample padding mode (how to handle out-of-bound coordinates).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GridPaddingMode {
    Zeros,
    Border,
    Reflection,
}

/// Padding mode for Pad op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PadMode {
    Constant,
    Reflect,
    Replicate,
}

/// All operations in the Warp IR.
///
/// Organized by category. Each variant carries only the *attributes*
/// of the operation (constants, configuration). Tensor inputs/outputs
/// are tracked by the graph, not the op.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    // === Tensor creation ===
    /// External input to the graph (weights, activations, KV cache).
    /// `index` identifies which input this is.
    Input { index: u32 },

    /// Constant tensor baked into the graph.
    Constant { data: ConstantData },

    /// Identity (no-op, pass-through). Used by optimizer for dead code elimination.
    Identity,

    // === Core compute ===
    /// Matrix multiplication. Inputs: [A, B]. Output: A @ B.
    /// transpose_a/b happen before the multiply.
    MatMul {
        transpose_a: bool,
        transpose_b: bool,
    },

    /// Batched matrix multiplication.
    BatchMatMul {
        transpose_a: bool,
        transpose_b: bool,
    },

    /// Elementwise binary op. Inputs: [A, B]. Broadcasting applies.
    Binary { op: BinaryOp },

    /// Elementwise unary op. Input: [A].
    Unary { op: UnaryOp },

    /// Activation function. Input: [A].
    Activate { activation: Activation },

    // === Normalization ===
    /// Layer normalization. Inputs: [X, gamma, beta]. Axis: last dim.
    LayerNorm { eps: f32 },

    /// RMS normalization. Inputs: [X, gamma]. No bias, no mean subtraction.
    RmsNorm { eps: f32 },

    // === Attention ===
    /// Scaled dot-product attention (FlashAttention-compatible).
    /// Inputs: [Q, K, V] or [Q, K, V, mask] if mask is Explicit.
    /// This is a *fused* op — the optimizer should try to pattern-match
    /// Q@K^T -> scale -> mask -> softmax -> @V into this single op.
    Attention {
        num_heads: u32,
        num_kv_heads: u32, // for GQA/MQA
        head_dim: u32,
        mask: AttentionMask,
        scale: Option<f32>, // defaults to 1/sqrt(head_dim) if None
    },

    /// Paged attention for KV cache during autoregressive decoding.
    /// Inputs: [Q, K_cache, V_cache, block_table, seq_lens].
    PagedAttention {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        scale: Option<f32>,
    },

    /// Rotary positional embedding. Input: [X, cos_cache, sin_cache].
    RotaryEmbed {
        head_dim: u32,
        interleaved: bool, // GPT-NeoX style vs LLaMA style
    },

    // === Shape manipulation ===
    Reshape { target_shape: Vec<i64> }, // -1 for inferred dim
    Transpose { perm: Vec<usize> },
    Concat { axis: i32 },
    Split { axis: i32, sizes: Vec<usize> },
    Gather { axis: i32 },
    Slice { starts: Vec<i64>, ends: Vec<i64>, steps: Vec<i64> },

    // === Convolution ===
    /// N-D convolution (typically 2D). Inputs: [input, weight, (optional) bias].
    /// input: [N, C_in, H, W], weight: [C_out, C_in/groups, kH, kW]
    /// groups=1: normal conv. groups=C_in: depthwise conv.
    Conv {
        kernel_size: Vec<u32>,
        stride: Vec<u32>,
        padding: Vec<u32>,
        dilation: Vec<u32>,
        groups: u32,
    },

    /// Transposed convolution / deconvolution. Inputs: [input, weight, (optional) bias].
    ConvTranspose {
        kernel_size: Vec<u32>,
        stride: Vec<u32>,
        padding: Vec<u32>,
        output_padding: Vec<u32>,
        dilation: Vec<u32>,
        groups: u32,
    },

    // === Pooling ===
    /// Spatial pooling. Input: [X] with shape [N, C, H, W].
    Pool {
        mode: PoolMode,
        kernel_size: Vec<u32>,
        stride: Vec<u32>,
        padding: Vec<u32>,
    },

    // === Normalization (spatial) ===
    /// Batch normalization. Inputs: [X, scale, bias, running_mean, running_var].
    /// X: [N, C, H, W]. During inference, uses running stats.
    BatchNorm { eps: f32 },

    /// Instance normalization. Inputs: [X, scale, bias].
    InstanceNorm { eps: f32 },

    /// Group normalization. Inputs: [X, scale, bias].
    GroupNorm { num_groups: u32, eps: f32 },

    // === Spatial transforms ===
    /// Resize / upsample. Input: [X].
    /// Either scales or output sizes must be specified.
    Resize {
        mode: ResizeMode,
        scales: Option<Vec<f32>>,
        sizes: Option<Vec<i64>>,
    },

    /// Grid sample — sample from input at coordinates specified by grid.
    /// Inputs: [input, grid]. input: [N,C,H,W], grid: [N,H_out,W_out,2].
    GridSample {
        mode: InterpolationMode,
        padding_mode: GridPaddingMode,
        align_corners: bool,
    },

    // === Detection / post-processing ===
    /// Non-maximum suppression. Inputs: [boxes, scores].
    NonMaxSuppression {
        iou_threshold: f32,
        score_threshold: f32,
        max_output: u32,
    },

    /// Top-K selection. Input: [X]. Returns [values, indices].
    TopK { k: u32, axis: i32, largest: bool },

    // === Padding ===
    /// Explicit padding. Input: [X]. pads: [before_d0, after_d0, before_d1, after_d1, ...].
    Pad { mode: PadMode, pads: Vec<i64>, value: f32 },

    // === Reduction ===
    Reduce { op: ReduceOp, axes: Vec<i32>, keepdim: bool },

    /// Softmax along an axis. Separate from Reduce because it's
    /// a compound operation (exp, sum, div) that should stay fused.
    Softmax { axis: i32 },

    // === Embedding ===
    /// Token embedding lookup. Inputs: [weight_table, indices].
    Embedding,

    // === MoE (Mixture of Experts) ===
    /// Top-K expert routing. Inputs: [hidden, gate_weights].
    /// Outputs: [dispatched, expert_indices, expert_weights].
    MoeGate {
        num_experts: u32,
        top_k: u32,
    },

    /// Expert computation (runs selected experts).
    /// Inputs: [dispatched, expert_weights_list...].
    MoeExpert {
        num_experts: u32,
    },

    // === Quantization ===
    /// Quantize a tensor. Input: [X]. Output: quantized X.
    Quantize { target_dtype: crate::DType, block_size: u32 },

    /// Dequantize a tensor. Input: [X_quant]. Output: float X.
    Dequantize { source_dtype: crate::DType, block_size: u32 },

    /// Quantized matrix multiply (e.g., W4A16 — 4-bit weights, 16-bit activations).
    /// Inputs: [activations, quantized_weights, scales, (optional) zeros].
    QuantizedMatMul {
        weight_dtype: crate::DType,
        group_size: u32,
    },

    // === Fused operations ===
    // These are created by the optimizer, never by the user.

    /// MatMul + Bias Add. Inputs: [A, B, bias].
    FusedMatMulBias {
        transpose_a: bool,
        transpose_b: bool,
    },

    /// MatMul + Bias + Activation. Inputs: [A, B, bias].
    FusedMatMulBiasAct {
        transpose_a: bool,
        transpose_b: bool,
        activation: Activation,
    },

    /// Add + RMSNorm (residual connection + norm). Inputs: [residual, X, gamma].
    FusedResidualRmsNorm { eps: f32 },

    /// Complete SwiGLU block: gate_proj, up_proj, silu, mul, down_proj.
    /// Inputs: [X, gate_weight, up_weight, down_weight].
    FusedSwiGLU,

    /// Auto-fused elementwise chain. Created by the autofuse optimizer pass.
    /// Contains the generated CUDA kernel source and the chain name.
    /// Inputs: [external inputs to the chain, in order].
    AutoFused {
        kernel_name: String,
        kernel_src: String,
        num_inputs: usize,
    },

    // === Speculative decoding ===
    /// Verify draft tokens against target model logits.
    /// Inputs: [draft_tokens, draft_probs, target_probs].
    /// Output: [accepted_tokens, num_accepted].
    SpeculativeVerify {
        max_draft_tokens: u32,
    },
}

/// Constant data embedded in the graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstantData {
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    // For large weight tensors, store a reference instead of inline data.
    External { name: String, byte_offset: usize, byte_length: usize },
}

impl Op {
    /// How many tensor inputs this op expects.
    /// Returns None for variable-input ops (Concat, MoeExpert).
    pub fn num_inputs(&self) -> Option<usize> {
        match self {
            Op::Input { .. } | Op::Constant { .. } => Some(0),

            Op::Unary { .. } | Op::Activate { .. } | Op::Softmax { .. }
            | Op::Reshape { .. } | Op::Transpose { .. }
            | Op::Quantize { .. } | Op::Dequantize { .. }
            | Op::Embedding
            | Op::Pool { .. } | Op::Resize { .. }
            | Op::TopK { .. } | Op::Pad { .. } => Some(1),

            Op::Binary { .. } | Op::MatMul { .. } | Op::BatchMatMul { .. }
            | Op::Gather { .. }
            | Op::GridSample { .. }  // input, grid
            | Op::NonMaxSuppression { .. } => Some(2),  // boxes, scores

            Op::Conv { .. } | Op::ConvTranspose { .. } => Some(2), // input, weight (bias optional → handled as 2 or 3)

            Op::LayerNorm { .. } => Some(3), // X, gamma, beta
            Op::RmsNorm { .. } | Op::RotaryEmbed { .. } => Some(3),
            Op::BatchNorm { .. } => Some(5), // X, scale, bias, running_mean, running_var
            Op::InstanceNorm { .. } | Op::GroupNorm { .. } => Some(3), // X, scale, bias

            Op::FusedMatMulBias { .. } => Some(3), // A, B, bias
            Op::FusedMatMulBiasAct { .. } => Some(3),
            Op::FusedResidualRmsNorm { .. } => Some(3), // residual, X, gamma

            Op::Attention { mask, .. } => Some(if *mask == AttentionMask::Explicit { 4 } else { 3 }),
            Op::PagedAttention { .. } => Some(5),

            Op::QuantizedMatMul { .. } => Some(3), // activations, weights, scales (+ optional zeros)
            Op::FusedSwiGLU => Some(4), // X, gate_w, up_w, down_w
            Op::AutoFused { num_inputs, .. } => Some(*num_inputs),

            Op::MoeGate { .. } => Some(2),
            Op::SpeculativeVerify { .. } => Some(3),

            Op::Reduce { .. } => Some(1),
            Op::Slice { .. } => Some(1),

            Op::Identity => Some(0),

            // Variable inputs
            Op::Concat { .. } | Op::Split { .. } | Op::MoeExpert { .. } => None,
        }
    }

    /// Whether this op is a fused operation (created by optimizer, not user).
    pub fn is_fused(&self) -> bool {
        matches!(
            self,
            Op::FusedMatMulBias { .. }
                | Op::FusedMatMulBiasAct { .. }
                | Op::FusedResidualRmsNorm { .. }
                | Op::FusedSwiGLU
                | Op::AutoFused { .. }
        )
    }

    /// Whether this op is purely a data movement (no compute).
    pub fn is_data_movement(&self) -> bool {
        matches!(
            self,
            Op::Reshape { .. }
                | Op::Transpose { .. }
                | Op::Concat { .. }
                | Op::Split { .. }
                | Op::Gather { .. }
                | Op::Slice { .. }
                | Op::Pad { .. }
        )
    }

    /// Whether this op is elementwise (can be fused with neighbors trivially).
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            Op::Binary { .. } | Op::Unary { .. } | Op::Activate { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_input_counts() {
        assert_eq!(Op::Input { index: 0 }.num_inputs(), Some(0));
        assert_eq!(
            Op::MatMul {
                transpose_a: false,
                transpose_b: false,
            }
            .num_inputs(),
            Some(2)
        );
        assert_eq!(
            Op::Attention {
                num_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
                mask: AttentionMask::Causal,
                scale: None,
            }
            .num_inputs(),
            Some(3)
        );
        assert_eq!(
            Op::Attention {
                num_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
                mask: AttentionMask::Explicit,
                scale: None,
            }
            .num_inputs(),
            Some(4)
        );
    }

    #[test]
    fn fused_ops_marked() {
        assert!(Op::FusedSwiGLU.is_fused());
        assert!(!Op::MatMul {
            transpose_a: false,
            transpose_b: false,
        }
        .is_fused());
    }
}

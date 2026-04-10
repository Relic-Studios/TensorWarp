//! Mixture of Experts (MoE) kernels for Gemma 4 26B-A4B.
//!
//! Architecture (from HuggingFace source):
//!   Router: RMSNorm(x) * scale * (1/sqrt(H)) → linear → softmax → top-k
//!   Experts: gate_up_proj[E, 2*D, H] + down_proj[E, H, D]
//!   Per expert: linear(x, gate_up) → chunk → gelu(gate)*up → linear(down)
//!
//! Dense MLP and MoE run in PARALLEL:
//!   dense_out = MLP(pre_feedforward_layernorm(residual))
//!   moe_out = Experts(pre_feedforward_layernorm_2(residual))
//!   combined = post_ffn_norm_1(dense_out) + post_ffn_norm_2(moe_out)
//!   output = (residual + post_ffn_norm(combined)) * layer_scalar

use warp_ir::{DType, Shape};
use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;
use crate::ops;

/// Router: compute expert scores and select top-k.
/// Matches HF Gemma4TextRouter exactly:
///   1. RMSNorm(x) without learned weights
///   2. Multiply by scale_vec * (1/sqrt(hidden_size))
///   3. Linear projection → [num_experts] logits
///   4. Softmax → probabilities
///   5. Top-K selection
///   6. Normalize weights to sum=1
///   7. Multiply by per_expert_scale
pub fn route_experts(
    cache: &KernelCache,
    device: &WarpDevice,
    hidden: &GpuTensor<f32>,            // [1, hidden_size]
    router_proj: &GpuTensor<f32>,       // [hidden_size, num_experts] (transposed)
    router_scale: &GpuTensor<f32>,      // [hidden_size] — learned scale vector
    per_expert_scale: &GpuTensor<f32>,  // [num_experts]
    hidden_size: u32,
    num_experts: u32,
    top_k: u32,
    norm_eps: f32,
) -> Result<(Vec<u32>, Vec<f32>), DeviceError> {
    // 1. RMSNorm without weights
    let mut normed = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, hidden_size as usize]), DType::F32)?;
    ops::rmsnorm_no_weight(cache, device, hidden, &mut normed, hidden_size, norm_eps)?;

    // 2. Multiply by scale * (1/sqrt(hidden_size))
    let scalar_root = 1.0 / (hidden_size as f32).sqrt();
    let mut scaled = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, hidden_size as usize]), DType::F32)?;
    ops::mul(cache, device, &normed, router_scale, &mut scaled)?;
    let mut scaled2 = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, hidden_size as usize]), DType::F32)?;
    ops::mul_scalar(cache, device, &scaled, &mut scaled2, scalar_root)?;

    // 3. Linear projection → expert logits
    let mut logits = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, num_experts as usize]), DType::F32)?;
    ops::gemm(cache, device, &scaled2, router_proj, &mut logits, 1, num_experts, hidden_size)?;

    // 4. Softmax
    let mut probs = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, num_experts as usize]), DType::F32)?;
    crate::sampling::softmax(cache, device, &logits, &mut probs, 1, num_experts)?;

    // 5+6+7. Top-K + normalize + per_expert_scale (on CPU — E=128 is tiny)
    let probs_host = probs.to_host(device)?;
    let per_expert_host = per_expert_scale.to_host(device)?;

    let mut indexed: Vec<(u32, f32)> = probs_host.iter().enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_ids: Vec<u32> = indexed[..top_k as usize].iter().map(|(id, _)| *id).collect();
    let top_weights: Vec<f32> = indexed[..top_k as usize].iter().map(|(_, w)| *w).collect();

    // Normalize to sum=1
    let sum: f32 = top_weights.iter().sum();
    let mut normalized: Vec<f32> = if sum > 0.0 {
        top_weights.iter().map(|w| w / sum).collect()
    } else {
        vec![1.0 / top_k as f32; top_k as usize]
    };

    // Apply per_expert_scale
    for (i, &eid) in top_ids.iter().enumerate() {
        normalized[i] *= per_expert_host[eid as usize];
    }

    Ok((top_ids, normalized))
}

/// Run a single expert's FFN from fused weight tensors.
///
/// gate_up_all: [num_experts, 2*moe_dim, hidden_size] stored as contiguous F32
/// down_all: [num_experts, hidden_size, moe_dim] stored as contiguous F32
///
/// This extracts expert_id's slice and runs: gelu(gate(x)) * up(x) → down → output
pub fn run_expert_from_fused(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,               // [1, hidden_size]
    gate_up_all: &GpuTensor<f32>,         // [num_experts, 2*moe_dim, hidden_size]
    down_all: &GpuTensor<f32>,            // [num_experts, hidden_size, moe_dim]
    expert_id: u32,
    output: &mut GpuTensor<f32>,          // [1, hidden_size]
    hidden_size: u32,
    moe_dim: u32,
    num_experts: u32,
) -> Result<(), DeviceError> {
    // Expert's gate_up slice: offset = expert_id * 2 * moe_dim * hidden_size
    let gu_offset = expert_id as usize * 2 * moe_dim as usize * hidden_size as usize;
    let gu_size = 2 * moe_dim as usize * hidden_size as usize;

    // Expert's down slice: offset = expert_id * hidden_size * moe_dim
    let d_offset = expert_id as usize * hidden_size as usize * moe_dim as usize;
    let d_size = hidden_size as usize * moe_dim as usize;

    // Create views into the fused tensors
    // TODO: use GPU slice views instead of copying to avoid allocation
    let gu_host = gate_up_all.to_host(device)?;
    let gu_slice = &gu_host[gu_offset..gu_offset + gu_size];
    let expert_gu = GpuTensor::from_host(device, gu_slice,
        Shape::from_static(&[2 * moe_dim as usize, hidden_size as usize]), DType::F32)?;

    let d_host = down_all.to_host(device)?;
    let d_slice = &d_host[d_offset..d_offset + d_size];
    let expert_down = GpuTensor::from_host(device, d_slice,
        Shape::from_static(&[hidden_size as usize, moe_dim as usize]), DType::F32)?;

    // gate_up = input × expert_gu^T → [1, 2*moe_dim]
    // Note: HF uses nn.functional.linear(x, W) which is x @ W^T
    // Our GEMM is C = A × B, so we need B stored as [hidden_size, 2*moe_dim]
    // But expert_gu is [2*moe_dim, hidden_size], so we need transposed GEMM
    // For now use cuBLAS with transB
    let mut gate_up = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, (2 * moe_dim) as usize]), DType::F32)?;
    crate::cublas_gemm::gemm_cublas_f32_transB(device,
        input, &expert_gu, &mut gate_up, 1, 2 * moe_dim, hidden_size)?;

    // Split + GeGLU
    let mut gate = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, moe_dim as usize]), DType::F32)?;
    let mut up = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, moe_dim as usize]), DType::F32)?;
    ops::split_gate_up(cache, device, &gate_up, &mut gate, &mut up, moe_dim, 1)?;

    let mut geglu = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, moe_dim as usize]), DType::F32)?;
    ops::fused_gelu_mul(cache, device, &gate, &up, &mut geglu)?;

    // down = geglu × expert_down^T → [1, hidden_size]
    crate::cublas_gemm::gemm_cublas_f32_transB(device,
        &geglu, &expert_down, output, 1, hidden_size, moe_dim)?;

    Ok(())
}

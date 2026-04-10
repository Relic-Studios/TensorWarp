//! Mixture of Experts (MoE) kernels for Gemma 4 26B-A4B.
//!
//! Each MoE layer:
//! 1. Router: hidden[1,H] × router_proj[H,E] → expert_logits[1,E]
//! 2. Top-K: select top_k experts from E experts
//! 3. For each selected expert: run small FFN (gate_up + gelu + down)
//! 4. Weighted sum of expert outputs
//!
//! Expert weights are stored fused: gate_up_proj[E, 2*D, H], down_proj[E, H, D]
//! where E=128 experts, D=704 (moe_intermediate_size), H=2816 (hidden_size).

use warp_ir::{DType, Shape};
use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;
use crate::ops;
use cudarc::driver::{LaunchConfig, PushKernelArg};

/// Router: compute expert scores and select top-k.
/// Returns (selected_expert_ids, expert_weights) on the HOST.
pub fn route_experts(
    cache: &KernelCache,
    device: &WarpDevice,
    hidden: &GpuTensor<f32>,          // [1, hidden_size]
    router_proj: &GpuTensor<f32>,     // [hidden_size, num_experts]
    router_scale: f32,                 // global scale
    per_expert_scale: &GpuTensor<f32>, // [num_experts]
    num_experts: u32,
    top_k: u32,
) -> Result<(Vec<u32>, Vec<f32>), DeviceError> {
    let h = hidden.numel as u32;

    // Router GEMM: logits = hidden × router_proj → [1, num_experts]
    let mut logits = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, num_experts as usize]), DType::F32)?;
    ops::gemm(cache, device, hidden, router_proj, &mut logits, 1, num_experts, h)?;

    // Apply per-expert scale: logits[i] *= per_expert_scale[i] * router_scale
    let mut scaled = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, num_experts as usize]), DType::F32)?;
    ops::mul(cache, device, &logits, per_expert_scale, &mut scaled)?;
    ops::mul_scalar(cache, device, &scaled, &mut logits, router_scale)?;

    // Softmax
    let mut probs = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, num_experts as usize]), DType::F32)?;
    crate::sampling::softmax(cache, device, &logits, &mut probs, 1, num_experts)?;

    // Top-K on CPU (num_experts=128 is small enough)
    let probs_host = probs.to_host(device)?;
    let mut indexed: Vec<(u32, f32)> = probs_host.iter().enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let selected_ids: Vec<u32> = indexed[..top_k as usize].iter().map(|(id, _)| *id).collect();
    let selected_weights: Vec<f32> = indexed[..top_k as usize].iter().map(|(_, w)| *w).collect();

    // Normalize weights to sum to 1
    let weight_sum: f32 = selected_weights.iter().sum();
    let normalized: Vec<f32> = if weight_sum > 0.0 {
        selected_weights.iter().map(|w| w / weight_sum).collect()
    } else {
        vec![1.0 / top_k as f32; top_k as usize]
    };

    Ok((selected_ids, normalized))
}

/// Run a single expert's FFN: output = down(gelu(gate_up(x)))
/// gate_up is fused: [2*moe_dim, hidden_size], split after GEMM.
pub fn run_expert_ffn(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,            // [1, hidden_size]
    gate_up_weight: &GpuTensor<f32>,   // [2*moe_dim, hidden_size] — one expert's slice
    down_weight: &GpuTensor<f32>,      // [hidden_size, moe_dim] — one expert's slice
    output: &mut GpuTensor<f32>,       // [1, hidden_size]
    hidden_size: u32,
    moe_dim: u32,
) -> Result<(), DeviceError> {
    // gate_up = input × gate_up_weight^T → [1, 2*moe_dim]
    let mut gate_up = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, (2 * moe_dim) as usize]), DType::F32)?;
    ops::gemm(cache, device, input, gate_up_weight, &mut gate_up, 1, 2 * moe_dim, hidden_size)?;

    // Split into gate[moe_dim] and up[moe_dim]
    let mut gate = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, moe_dim as usize]), DType::F32)?;
    let mut up = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, moe_dim as usize]), DType::F32)?;
    ops::split_gate_up(cache, device, &gate_up, &mut gate, &mut up, moe_dim, 1)?;

    // GeGLU: gelu(gate) * up
    let mut geglu = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, moe_dim as usize]), DType::F32)?;
    ops::fused_gelu_mul(cache, device, &gate, &up, &mut geglu)?;

    // down = geglu × down_weight^T → [1, hidden_size]
    ops::gemm(cache, device, &geglu, down_weight, output, 1, hidden_size, moe_dim)?;

    Ok(())
}

/// Run the full MoE layer: route + dispatch to top-k experts + aggregate.
/// Expert weights are stored as fused tensors:
///   gate_up_all: [num_experts, 2*moe_dim, hidden_size] (flattened)
///   down_all: [num_experts, hidden_size, moe_dim] (flattened)
pub fn moe_forward(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,             // [1, hidden_size]
    router_proj: &GpuTensor<f32>,       // [hidden_size, num_experts]
    router_scale: f32,
    per_expert_scale: &GpuTensor<f32>,  // [num_experts]
    gate_up_all: &[u8],                 // raw bytes for all experts' gate_up (GPU)
    down_all: &[u8],                    // raw bytes for all experts' down (GPU)
    output: &mut GpuTensor<f32>,        // [1, hidden_size]
    hidden_size: u32,
    moe_dim: u32,
    num_experts: u32,
    top_k: u32,
) -> Result<(), DeviceError> {
    // 1. Route: select top-k experts
    let (expert_ids, weights) = route_experts(
        cache, device, input, router_proj, router_scale,
        per_expert_scale, num_experts, top_k)?;

    // 2. Run each expert and accumulate weighted output
    // Zero the output first
    let mut accumulated = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, hidden_size as usize]), DType::F32)?;

    for (idx, (&expert_id, &weight)) in expert_ids.iter().zip(weights.iter()).enumerate() {
        // Extract this expert's weights from the fused tensors
        // gate_up: expert_id * (2*moe_dim*hidden_size) bytes offset
        // down: expert_id * (hidden_size*moe_dim) bytes offset
        // TODO: implement expert weight slicing from GPU memory

        let mut expert_out = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, hidden_size as usize]), DType::F32)?;

        // For now, placeholder — need to slice expert weights from fused tensor
        // run_expert_ffn(cache, device, input, &expert_gate_up, &expert_down,
        //     &mut expert_out, hidden_size, moe_dim)?;

        // Accumulate: output += weight * expert_out
        let mut scaled_out = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, hidden_size as usize]), DType::F32)?;
        ops::mul_scalar(cache, device, &expert_out, &mut scaled_out, weight)?;
        let mut new_acc = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, hidden_size as usize]), DType::F32)?;
        ops::add(cache, device, &accumulated, &scaled_out, &mut new_acc)?;
        accumulated = new_acc;
    }

    // Copy accumulated to output
    ops::mul_scalar(cache, device, &accumulated, output, 1.0)?;

    Ok(())
}

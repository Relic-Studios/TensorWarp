//! Qwen3.5 FFN + full-attention block helpers.
//!
//! These compose existing TensorWarp kernels (RMSNorm, RoPE, flash attention,
//! cuBLAS HGEMM) for the Qwen3.5 hybrid architecture. Each block executes
//! one decode step and writes output back to the hidden state with residual.
//!
//! FFN block (used after every layer, both linear and full):
//!   normed = RMSNorm(hidden, post_attn_norm)
//!   gate_pre = w_gate @ normed
//!   up_pre   = w_up   @ normed
//!   silu_up = SiLU(gate_pre) * up_pre
//!   ffn_out = w_down @ silu_up
//!   hidden += ffn_out
//!
//! Full-attention block (used in 8 of 32 layers — every 4th):
//!   TODO next session — see header comment in qwen3_5_blocks_attn_TODO.md

use cudarc::driver::{LaunchConfig, PushKernelArg};
use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::cublas_gemm::gemm_cublas_f16in_f32out_transB as hgemm_f16f32;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

const FFN_KERNEL_SRC: &str = include_str!("cuda/qwen3_5_ffn.cu");
const SWIGLU_FN: &str = "qwen35_swiglu_elementwise_kernel";

// ─── FFN block buffers ──────────────────────────────────────────────────

pub struct FfnStepBuffers {
    pub normed: GpuTensor<f32>,           // [B, D]    RMSNorm output
    pub normed_f16: GpuTensor<half::f16>, // [B, D]    cast for cuBLAS input
    pub gate_pre: GpuTensor<f32>,         // [B, D_ffn]  gate matmul output
    pub up_pre: GpuTensor<f32>,           // [B, D_ffn]  up matmul output
    pub silu_up: GpuTensor<f32>,          // [B, D_ffn]  SiLU(gate) * up
    pub silu_up_f16: GpuTensor<half::f16>, // [B, D_ffn]  cast for down-proj cuBLAS
    pub ffn_out: GpuTensor<f32>,          // [B, D]    down-proj output
}

impl FfnStepBuffers {
    pub fn new(
        device: &WarpDevice,
        batch: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, DeviceError> {
        let alloc_f32 = |dim: usize| -> Result<GpuTensor<f32>, DeviceError> {
            GpuTensor::<f32>::zeros(device, Shape::from_static(&[batch, dim]), DType::F32)
        };
        let alloc_f16 = |dim: usize| -> Result<GpuTensor<half::f16>, DeviceError> {
            GpuTensor::<half::f16>::zeros(device, Shape::from_static(&[batch, dim]), DType::F16)
        };
        Ok(Self {
            normed:        alloc_f32(hidden_size)?,
            normed_f16:    alloc_f16(hidden_size)?,
            gate_pre:      alloc_f32(intermediate_size)?,
            up_pre:        alloc_f32(intermediate_size)?,
            silu_up:       alloc_f32(intermediate_size)?,
            silu_up_f16:   alloc_f16(intermediate_size)?,
            ffn_out:       alloc_f32(hidden_size)?,
        })
    }
}

// ─── FFN block step ─────────────────────────────────────────────────────

pub fn ffn_block_step(
    device: &WarpDevice,
    cache: &KernelCache,
    hidden: &mut GpuTensor<f32>,                 // [B, D] in/out
    post_attn_norm: &GpuTensor<f32>,             // [D]
    w_gate: &GpuTensor<half::f16>,               // [D_ffn, D]
    w_up: &GpuTensor<half::f16>,                 // [D_ffn, D]
    w_down: &GpuTensor<half::f16>,               // [D, D_ffn]
    bufs: &mut FfnStepBuffers,
    rmsnorm_eps: f32,
) -> Result<(), DeviceError> {
    let batch = hidden.shape.dims()[0].static_val().unwrap();
    let hidden_size = hidden.shape.dims()[1].static_val().unwrap();
    let intermediate_size = bufs.gate_pre.shape.dims()[1].static_val().unwrap();
    let m = batch as u32;
    let k = hidden_size as u32;
    let n = intermediate_size as u32;

    // 1. RMSNorm
    crate::ops::rmsnorm(cache, device, hidden, post_attn_norm,
                        &mut bufs.normed, k, rmsnorm_eps)?;

    // 2. Cast normed → f16 for cuBLAS input
    crate::fp16::cast_f32_to_f16(cache, device, &bufs.normed, &mut bufs.normed_f16)?;

    // 3. Gate + Up via cuBLAS HGEMM
    hgemm_f16f32(device, &bufs.normed_f16, w_gate, &mut bufs.gate_pre, m, n, k)?;
    hgemm_f16f32(device, &bufs.normed_f16, w_up, &mut bufs.up_pre, m, n, k)?;

    // 4. SwiGLU element-wise: silu_up = SiLU(gate) * up
    {
        let f = cache.get_or_compile_with_opts(
            device, FFN_KERNEL_SRC, SWIGLU_FN,
            &[WarpDevice::cuda_include_path()], None,
        )?;
        let total = (batch * intermediate_size) as i32;
        let cfg_launch = LaunchConfig::for_num_elems(total as u32);
        unsafe {
            device.stream.launch_builder(&f)
                .arg(&bufs.gate_pre.data)
                .arg(&bufs.up_pre.data)
                .arg(&mut bufs.silu_up.data)
                .arg(&total)
                .launch(cfg_launch)
                .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
        }
    }

    // 5. Cast silu_up → f16 for down-proj cuBLAS input
    crate::fp16::cast_f32_to_f16(cache, device, &bufs.silu_up, &mut bufs.silu_up_f16)?;

    // 6. Down projection: ffn_out = w_down @ silu_up
    hgemm_f16f32(device, &bufs.silu_up_f16, w_down, &mut bufs.ffn_out, m, k, n)?;

    // 7. Residual add: hidden += ffn_out  (reuse gated_delta_net's residual_add kernel)
    {
        let f = cache.get_or_compile_with_opts(
            device, crate::gated_delta_net::residual_add_kernel_src(), "gdn_residual_add_kernel",
            &[WarpDevice::cuda_include_path()], None,
        )?;
        let total = (batch * hidden_size) as i32;
        let batch_i = batch as i32;
        let d = hidden_size as i32;
        let cfg_launch = LaunchConfig::for_num_elems(total as u32);
        unsafe {
            device.stream.launch_builder(&f)
                .arg(&bufs.ffn_out.data)
                .arg(&mut hidden.data)
                .arg(&batch_i).arg(&d)
                .launch(cfg_launch)
                .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
        }
    }

    Ok(())
}

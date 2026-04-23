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
//!   normed = RMSNorm(hidden, input_norm)
//!   q_pre = w_q @ normed  → [B, H × dh]
//!   k_pre = w_k @ normed  → [B, H_kv × dh]
//!   v_pre = w_v @ normed  → [B, H_kv × dh]
//!   q = per_head_rmsnorm(q_pre, q_norm)   per (head, head_dim)
//!   k = per_head_rmsnorm(k_pre, k_norm)
//!   q = RoPE(q); k = RoPE(k)
//!   kv_cache.append(k, v)
//!   attn_out = flash_attention(q, kv_cache)
//!   out = w_o @ attn_out
//!   hidden += out  (residual)

use cudarc::driver::{LaunchConfig, PushKernelArg};
use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::cublas_gemm::gemm_cublas_f16in_f32out_transB as hgemm_f16f32;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

const FFN_KERNEL_SRC: &str = include_str!("cuda/qwen3_5_ffn.cu");
const SWIGLU_FN: &str = "qwen35_swiglu_elementwise_kernel";

const ATTN_KERNEL_SRC: &str = include_str!("cuda/qwen3_5_attn.cu");
const PER_HEAD_RMSNORM_FN: &str = "qwen35_per_head_rmsnorm_kernel";

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
    residual_add(device, cache, &bufs.ffn_out, hidden, batch as i32, hidden_size as i32)?;

    Ok(())
}

// ─── Full-attention block buffers ───────────────────────────────────────

pub struct FullAttnStepBuffers {
    pub normed: GpuTensor<f32>,             // [B, D]   RMSNorm output
    pub normed_f16: GpuTensor<half::f16>,   // [B, D]   cast for cuBLAS
    pub q_pre: GpuTensor<f32>,              // [B, H × dh]
    pub k_pre: GpuTensor<f32>,              // [B, H_kv × dh]
    pub v_pre: GpuTensor<f32>,              // [B, H_kv × dh]
    pub q_normed: GpuTensor<f32>,           // [B, H × dh]   post per-head RMSNorm
    pub k_normed: GpuTensor<f32>,           // [B, H_kv × dh]
    pub q_rope: GpuTensor<f32>,             // [B, H × dh]   post-RoPE
    pub k_rope: GpuTensor<f32>,             // [B, H_kv × dh]
    pub attn_out: GpuTensor<f32>,           // [B, H × dh]
    pub attn_out_f16: GpuTensor<half::f16>, // [B, H × dh]   cast for o-proj cuBLAS
    pub out: GpuTensor<f32>,                // [B, D]   o-proj output
}

impl FullAttnStepBuffers {
    pub fn new(
        device: &WarpDevice,
        batch: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, DeviceError> {
        let alloc_f32 = |dim: usize| -> Result<GpuTensor<f32>, DeviceError> {
            GpuTensor::<f32>::zeros(device, Shape::from_static(&[batch, dim]), DType::F32)
        };
        let alloc_f16 = |dim: usize| -> Result<GpuTensor<half::f16>, DeviceError> {
            GpuTensor::<half::f16>::zeros(device, Shape::from_static(&[batch, dim]), DType::F16)
        };
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        Ok(Self {
            normed:        alloc_f32(hidden_size)?,
            normed_f16:    alloc_f16(hidden_size)?,
            q_pre:         alloc_f32(q_dim)?,
            k_pre:         alloc_f32(kv_dim)?,
            v_pre:         alloc_f32(kv_dim)?,
            q_normed:      alloc_f32(q_dim)?,
            k_normed:      alloc_f32(kv_dim)?,
            q_rope:        alloc_f32(q_dim)?,
            k_rope:        alloc_f32(kv_dim)?,
            attn_out:      alloc_f32(q_dim)?,
            attn_out_f16:  alloc_f16(q_dim)?,
            out:           alloc_f32(hidden_size)?,
        })
    }
}

// ─── Full-attention block step ──────────────────────────────────────────

pub fn full_attn_block_step(
    device: &WarpDevice,
    cache: &KernelCache,
    hidden: &mut GpuTensor<f32>,                     // [B, D]   in/out (residual)
    input_norm: &GpuTensor<f32>,                     // [D]
    wq: &GpuTensor<half::f16>,                       // [H × dh, D]
    wk: &GpuTensor<half::f16>,                       // [H_kv × dh, D]
    wv: &GpuTensor<half::f16>,                       // [H_kv × dh, D]
    wo: &GpuTensor<half::f16>,                       // [D, H × dh]
    q_norm: &GpuTensor<f32>,                         // [dh]
    k_norm: &GpuTensor<f32>,                         // [dh]
    kv_cache: &mut crate::kv_cache::LayerKVCache,
    bufs: &mut FullAttnStepBuffers,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_base: f32,
    rmsnorm_eps: f32,
    cache_len_buf: &cudarc::driver::CudaSlice<u32>,  // for flash attn device-side cache len
) -> Result<(), DeviceError> {
    let batch = hidden.shape.dims()[0].static_val().unwrap();
    let hidden_size = hidden.shape.dims()[1].static_val().unwrap();
    let m = batch as u32;
    let k = hidden_size as u32;
    let q_dim = (num_heads * head_dim) as u32;
    let kv_dim = (num_kv_heads * head_dim) as u32;

    // 1. Pre-attention RMSNorm
    crate::ops::rmsnorm(cache, device, hidden, input_norm,
                        &mut bufs.normed, k, rmsnorm_eps)?;

    // 2. Cast normed → f16 for cuBLAS
    crate::fp16::cast_f32_to_f16(cache, device, &bufs.normed, &mut bufs.normed_f16)?;

    // 3. Q, K, V projections via cuBLAS HGEMM
    hgemm_f16f32(device, &bufs.normed_f16, wq, &mut bufs.q_pre, m, q_dim, k)?;
    hgemm_f16f32(device, &bufs.normed_f16, wk, &mut bufs.k_pre, m, kv_dim, k)?;
    hgemm_f16f32(device, &bufs.normed_f16, wv, &mut bufs.v_pre, m, kv_dim, k)?;

    // 4. Per-head RMSNorm on Q and K (Qwen3.5-specific, Gemma-style)
    per_head_rmsnorm(device, cache, &bufs.q_pre, q_norm, &mut bufs.q_normed,
                     batch, num_heads, head_dim, rmsnorm_eps)?;
    per_head_rmsnorm(device, cache, &bufs.k_pre, k_norm, &mut bufs.k_normed,
                     batch, num_kv_heads, head_dim, rmsnorm_eps)?;

    // 5. RoPE on Q and K (offset = current cache_len, i.e. position of new token)
    let seq_len = 1u32;
    let position = kv_cache.len;
    crate::rope::rope(cache, device, &bufs.q_normed, &mut bufs.q_rope,
                       num_heads as u32 * batch as u32, seq_len, head_dim as u32,
                       rope_base, position)?;
    crate::rope::rope(cache, device, &bufs.k_normed, &mut bufs.k_rope,
                       num_kv_heads as u32 * batch as u32, seq_len, head_dim as u32,
                       rope_base, position)?;

    // 6. Append K, V to KV cache
    kv_cache.append(cache, device, &bufs.k_rope, &bufs.v_pre)?;

    // 7. Flash attention (device-side cache len buffer for graph-friendly)
    crate::kv_cache::decode_attention_flash_device_len(
        cache, device, &bufs.q_rope, kv_cache, &mut bufs.attn_out,
        num_heads as u32, num_kv_heads as u32, head_dim as u32, cache_len_buf,
    )?;

    // 8. Output projection via cuBLAS
    crate::fp16::cast_f32_to_f16(cache, device, &bufs.attn_out, &mut bufs.attn_out_f16)?;
    hgemm_f16f32(device, &bufs.attn_out_f16, wo, &mut bufs.out, m, k, q_dim)?;

    // 9. Residual add: hidden += out
    residual_add(device, cache, &bufs.out, hidden, batch as i32, hidden_size as i32)?;

    Ok(())
}

// ─── Helpers ────────────────────────────────────────────────────────────

fn per_head_rmsnorm(
    device: &WarpDevice,
    cache: &KernelCache,
    x: &GpuTensor<f32>,
    weight: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    batch: usize,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile_with_opts(
        device, ATTN_KERNEL_SRC, PER_HEAD_RMSNORM_FN,
        &[WarpDevice::cuda_include_path()], None,
    )?;
    let block_dim = head_dim.next_power_of_two().min(1024) as u32;
    let cfg_launch = LaunchConfig {
        grid_dim: (batch as u32, num_heads as u32, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: (block_dim as usize * 4) as u32,
    };
    let b_i = batch as i32;
    let h_i = num_heads as i32;
    let dh = head_dim as i32;
    unsafe {
        device.stream.launch_builder(&f)
            .arg(&x.data)
            .arg(&weight.data)
            .arg(&mut out.data)
            .arg(&b_i).arg(&h_i).arg(&dh)
            .arg(&eps)
            .launch(cfg_launch)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

fn residual_add(
    device: &WarpDevice,
    cache: &KernelCache,
    src: &GpuTensor<f32>,
    dst: &mut GpuTensor<f32>,
    batch: i32,
    dim: i32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile_with_opts(
        device, crate::gated_delta_net::residual_add_kernel_src(), "gdn_residual_add_kernel",
        &[WarpDevice::cuda_include_path()], None,
    )?;
    let total = (batch * dim) as u32;
    let cfg_launch = LaunchConfig::for_num_elems(total);
    unsafe {
        device.stream.launch_builder(&f)
            .arg(&src.data)
            .arg(&mut dst.data)
            .arg(&batch).arg(&dim)
            .launch(cfg_launch)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

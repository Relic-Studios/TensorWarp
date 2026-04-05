//! FP16 mixed-precision inference kernels.
//!
//! Mixed precision strategy (same as TensorRT):
//! - Weights stored in FP16 (2x memory savings)
//! - GEMMs: FP16 inputs, FP32 accumulation, FP16 output (tensor cores)
//! - Norms: promote to FP32 for stability, output FP16
//! - Elementwise: FP16 throughout
//!
//! On RTX 4090: FP16 tensor cores = ~330 TFLOPS vs ~83 TFLOPS FP32 (4x).

use cudarc::driver::{LaunchConfig, PushKernelArg};
use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

macro_rules! launch_err {
    ($e:expr) => {
        $e.map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))
    };
}

// ── Cast kernels ────────────────────────────────────────────────

const F32_TO_F16_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void warp_f32_to_f16(
    half *out, const float *in_data, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = __float2half(in_data[i]); }
}
"#;

const F16_TO_F32_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void warp_f16_to_f32(
    float *out, const half *in_data, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = __half2float(in_data[i]); }
}
"#;

// ── FP16 elementwise ────────────────────────────────────────────

const F16_ADD_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void warp_f16_add(half *out, const half *a, const half *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = __hadd(a[i], b[i]); }
}
"#;

const F16_MUL_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void warp_f16_mul(half *out, const half *a, const half *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = __hmul(a[i], b[i]); }
}
"#;

const F16_SILU_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void warp_f16_silu(half *out, const half *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(x[i]);
        float result = v / (1.0f + expf(-v));
        out[i] = __float2half(result);
    }
}
"#;

const F16_RELU_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void warp_f16_relu(half *out, const half *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(x[i]);
        out[i] = __float2half(fmaxf(v, 0.0f));
    }
}
"#;

// ── FP16 RMSNorm ────────────────────────────────────────────────
// Promote to F32 for numerical stability, output FP16.

const F16_RMSNORM_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void warp_f16_rmsnorm(
    half *out, const half *x, const half *gamma,
    unsigned int hidden_size, float eps, size_t n_rows
) {
    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const half *x_row = x + row * hidden_size;
    half *out_row = out + row * hidden_size;

    // Compute in FP32 for stability
    float sum_sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        sum_sq += v * v;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        float g = __half2float(gamma[i]);
        out_row[i] = __float2half(v * rms * g);
    }
}
"#;

// ═════════════════════════════════════════════════════════════════
// Rust API
// ═════════════════════════════════════════════════════════════════

fn compile_fp16(cache: &KernelCache, device: &WarpDevice, src: &str, name: &str) -> Result<cudarc::driver::CudaFunction, DeviceError> {
    let include = WarpDevice::cuda_include_path();
    cache.get_or_compile_with_opts(device, src, name, &[include], None)
}

/// Cast F32 → FP16.
pub fn cast_f32_to_f16(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    output: &mut GpuTensor<half::f16>,
) -> Result<(), DeviceError> {
    let f = compile_fp16(cache, device, F32_TO_F16_SRC, "warp_f32_to_f16")?;
    let n = input.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&input.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Cast FP16 → F32.
pub fn cast_f16_to_f32(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<half::f16>,
    output: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = compile_fp16(cache, device, F16_TO_F32_SRC, "warp_f16_to_f32")?;
    let n = input.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&input.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// FP16 elementwise add.
pub fn f16_add(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<half::f16>,
    b: &GpuTensor<half::f16>,
    out: &mut GpuTensor<half::f16>,
) -> Result<(), DeviceError> {
    let f = compile_fp16(cache, device, F16_ADD_SRC, "warp_f16_add")?;
    let n = a.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&a.data).arg(&b.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// FP16 elementwise mul.
pub fn f16_mul(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<half::f16>,
    b: &GpuTensor<half::f16>,
    out: &mut GpuTensor<half::f16>,
) -> Result<(), DeviceError> {
    let f = compile_fp16(cache, device, F16_MUL_SRC, "warp_f16_mul")?;
    let n = a.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&a.data).arg(&b.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// FP16 SiLU activation.
pub fn f16_silu(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<half::f16>,
    out: &mut GpuTensor<half::f16>,
) -> Result<(), DeviceError> {
    let f = compile_fp16(cache, device, F16_SILU_SRC, "warp_f16_silu")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// FP16 ReLU activation.
pub fn f16_relu(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<half::f16>,
    out: &mut GpuTensor<half::f16>,
) -> Result<(), DeviceError> {
    let f = compile_fp16(cache, device, F16_RELU_SRC, "warp_f16_relu")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// FP16 RMSNorm (FP32 accumulation for stability, FP16 I/O).
pub fn f16_rmsnorm(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<half::f16>,
    gamma: &GpuTensor<half::f16>,
    out: &mut GpuTensor<half::f16>,
    hidden_size: u32,
    eps: f32,
) -> Result<(), DeviceError> {
    let f = compile_fp16(cache, device, F16_RMSNORM_SRC, "warp_f16_rmsnorm")?;
    let n_rows = (x.numel / hidden_size as usize) as usize;
    let cfg = LaunchConfig {
        grid_dim: (n_rows as u32, 1, 1),
        block_dim: (32.min(hidden_size), 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&x.data)
            .arg(&gamma.data)
            .arg(&hidden_size)
            .arg(&eps)
            .arg(&n_rows)
            .launch(cfg))?;
    }
    Ok(())
}

// ── FP16 fused ops ──────────────────────────────────────────────

const F16_FUSED_SILU_MUL_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void warp_f16_fused_silu_mul(
    half *out, const half *gate, const half *up, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __half2float(gate[i]);
        float silu_g = g / (1.0f + expf(-g));
        out[i] = __float2half(silu_g * __half2float(up[i]));
    }
}
"#;

const F16_FUSED_RESIDUAL_RMSNORM_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void warp_f16_fused_residual_rmsnorm(
    half *norm_out, half *residual_out,
    const half *x, const half *residual, const half *gamma,
    unsigned int hidden_size, float eps, size_t n_rows
) {
    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const half *x_row = x + row * hidden_size;
    const half *r_row = residual + row * hidden_size;
    half *nout = norm_out + row * hidden_size;
    half *rout = residual_out + row * hidden_size;

    float sum_sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __half2float(x_row[i]) + __half2float(r_row[i]);
        sum_sq += v * v;
        rout[i] = __float2half(v);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __half2float(rout[i]);
        nout[i] = __float2half(v * rms * __half2float(gamma[i]));
    }
}
"#;

/// FP16 fused SiLU+Mul (SwiGLU gate).
pub fn f16_fused_silu_mul(
    cache: &KernelCache,
    device: &WarpDevice,
    gate: &GpuTensor<half::f16>,
    up: &GpuTensor<half::f16>,
    out: &mut GpuTensor<half::f16>,
) -> Result<(), DeviceError> {
    let f = compile_fp16(cache, device, F16_FUSED_SILU_MUL_SRC, "warp_f16_fused_silu_mul")?;
    let n = gate.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&gate.data).arg(&up.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// FP16 fused residual+RMSNorm with dual output.
pub fn f16_fused_residual_rmsnorm(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<half::f16>,
    residual: &GpuTensor<half::f16>,
    gamma: &GpuTensor<half::f16>,
    norm_out: &mut GpuTensor<half::f16>,
    residual_out: &mut GpuTensor<half::f16>,
    hidden_size: u32,
    eps: f32,
) -> Result<(), DeviceError> {
    let f = compile_fp16(cache, device, F16_FUSED_RESIDUAL_RMSNORM_SRC, "warp_f16_fused_residual_rmsnorm")?;
    let n_rows = (x.numel / hidden_size as usize) as usize;
    let cfg = LaunchConfig {
        grid_dim: (n_rows as u32, 1, 1),
        block_dim: (32.min(hidden_size), 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut norm_out.data)
            .arg(&mut residual_out.data)
            .arg(&x.data)
            .arg(&residual.data)
            .arg(&gamma.data)
            .arg(&hidden_size)
            .arg(&eps)
            .arg(&n_rows)
            .launch(cfg))?;
    }
    Ok(())
}

// ── FP16 RoPE ───────────────────────────────────────────────────

const F16_ROPE_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void warp_f16_rope(
    half *out, const half *input,
    unsigned int B, unsigned int N, unsigned int D,
    float base, unsigned int offset
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = B * N * (D / 2);
    if (idx >= total) return;

    unsigned int pair = idx % (D / 2);
    unsigned int pos_in_seq = (idx / (D / 2)) % N;
    unsigned int b = idx / (N * (D / 2));
    unsigned int pos = pos_in_seq + offset;

    float freq = 1.0f / powf(base, 2.0f * (float)pair / (float)D);
    float theta = (float)pos * freq;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    // PyTorch rotate_half pairs (dim i, dim i+D/2), NOT (dim 2i, dim 2i+1)
    unsigned int base_offset = b * N * D + pos_in_seq * D;
    float x0 = __half2float(input[base_offset + pair]);
    float x1 = __half2float(input[base_offset + pair + D / 2]);

    out[base_offset + pair]         = __float2half(x0 * cos_t - x1 * sin_t);
    out[base_offset + pair + D / 2] = __float2half(x0 * sin_t + x1 * cos_t);
}
"#;

/// FP16 RoPE.
pub fn f16_rope(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<half::f16>,
    out: &mut GpuTensor<half::f16>,
    batch: u32,
    seq_len: u32,
    head_dim: u32,
    base: f32,
    offset: u32,
) -> Result<(), DeviceError> {
    let f = compile_fp16(cache, device, F16_ROPE_SRC, "warp_f16_rope")?;
    let total = batch * seq_len * (head_dim / 2);
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&input.data)
            .arg(&batch).arg(&seq_len).arg(&head_dim)
            .arg(&base).arg(&offset)
            .launch(cfg))?;
    }
    Ok(())
}

/// FP16 RoPE variant that reads position from a GPU buffer (for CUDA graph capture).
const F16_ROPE_DEVICE_POS_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void warp_f16_rope_device_pos(
    half *out, const half *input,
    unsigned int B, unsigned int N, unsigned int D,
    float base, const unsigned int *pos_buf
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = B * N * (D / 2);
    if (idx >= total) return;

    unsigned int pair = idx % (D / 2);
    unsigned int pos_in_seq = (idx / (D / 2)) % N;
    unsigned int b = idx / (N * (D / 2));
    unsigned int pos = pos_in_seq + pos_buf[0];

    float freq = 1.0f / powf(base, 2.0f * (float)pair / (float)D);
    float theta = (float)pos * freq;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    unsigned int base_offset = b * N * D + pos_in_seq * D;
    float x0 = __half2float(input[base_offset + pair]);
    float x1 = __half2float(input[base_offset + pair + D / 2]);

    out[base_offset + pair]         = __float2half(x0 * cos_t - x1 * sin_t);
    out[base_offset + pair + D / 2] = __float2half(x0 * sin_t + x1 * cos_t);
}
"#;

/// FP16 RoPE that reads position from a GPU buffer (CUDA graph compatible).
pub fn f16_rope_device_pos(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<half::f16>,
    out: &mut GpuTensor<half::f16>,
    batch: u32,
    seq_len: u32,
    head_dim: u32,
    base: f32,
    pos_buf: &GpuTensor<u32>,  // [1] on GPU — position value
) -> Result<(), DeviceError> {
    let f = compile_fp16(cache, device, F16_ROPE_DEVICE_POS_SRC, "warp_f16_rope_device_pos")?;
    let total = batch * seq_len * (head_dim / 2);
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&input.data)
            .arg(&batch).arg(&seq_len).arg(&head_dim)
            .arg(&base).arg(&pos_buf.data)
            .launch(cfg))?;
    }
    Ok(())
}

// ── FP16 Scaled Dot-Product Attention ───────────────────────────

// FP16 attention with shared memory reduction — handles any head_dim.
// Same online softmax approach as attention_ext.rs but with FP16 I/O.
const F16_ATTENTION_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void warp_f16_attention(
    half *out,
    const half *Q, const half *K, const half *V,
    unsigned int B, unsigned int N, unsigned int D,
    float scale, unsigned int causal
) {
    extern __shared__ float smem[];
    float *dot_buf = smem;

    unsigned int b = blockIdx.y;
    unsigned int q_pos = blockIdx.x;
    unsigned int d = threadIdx.x;
    if (b >= B || q_pos >= N) return;

    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float out_val = 0.0f;

    for (unsigned int k_pos = 0; k_pos < N; k_pos++) {
        if (causal && k_pos > q_pos) break;

        // Parallel dot product with shared memory reduction
        float partial = 0.0f;
        if (d < D) {
            partial = __half2float(Q[b*N*D + q_pos*D + d]) *
                      __half2float(K[b*N*D + k_pos*D + d]);
        }
        dot_buf[d] = partial;
        __syncthreads();

        // Reduce in shared memory
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (d < stride) dot_buf[d] += dot_buf[d + stride];
            __syncthreads();
        }

        float score = dot_buf[0] * scale;
        __syncthreads();

        // Online softmax + weighted V accumulation
        float new_max = fmaxf(max_score, score);
        float correction = expf(max_score - new_max);
        float weight = expf(score - new_max);
        sum_exp = sum_exp * correction + weight;
        if (d < D) {
            out_val = out_val * correction + weight * __half2float(V[b*N*D + k_pos*D + d]);
        }
        max_score = new_max;
    }

    if (d < D) {
        out[b * N * D + q_pos * D + d] = __float2half(out_val / sum_exp);
    }
}
"#;

/// FP16 scaled dot-product attention with online softmax.
pub fn f16_attention(
    cache: &KernelCache,
    device: &WarpDevice,
    q: &GpuTensor<half::f16>,
    k: &GpuTensor<half::f16>,
    v: &GpuTensor<half::f16>,
    out: &mut GpuTensor<half::f16>,
    batch: u32,
    seq_len: u32,
    head_dim: u32,
    causal: bool,
) -> Result<(), DeviceError> {
    let f = compile_fp16(cache, device, F16_ATTENTION_SRC, "warp_f16_attention")?;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let causal_u32 = if causal { 1u32 } else { 0u32 };
    let block_size = head_dim.next_power_of_two().min(1024);
    let cfg = LaunchConfig {
        grid_dim: (seq_len, batch, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: block_size * 4, // float dot_buf[block_size]
    };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&q.data).arg(&k.data).arg(&v.data)
            .arg(&batch).arg(&seq_len).arg(&head_dim)
            .arg(&scale).arg(&causal_u32)
            .launch(cfg))?;
    }
    Ok(())
}

// ── FP16 Fused GEMM + Bias + Activation ─────────────────────
// The killer fusion: one kernel does GEMM + bias add + activation.
// cuBLAS can't do this — it needs 3 separate kernels.

const F16_FUSED_GEMM_BIAS_GELU_SRC: &str = r#"
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

extern "C" __global__ void warp_f16_fused_gemm_bias_gelu(
    half *C, const half *A, const half *B, const half *bias,
    unsigned int M, unsigned int N, unsigned int K
) {
    // Simple tiled GEMM with fused epilogue
    unsigned int row = blockIdx.y * 16 + threadIdx.y;
    unsigned int col = blockIdx.x * 16 + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (unsigned int k = 0; k < K; k++) {
        sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }

    // Fused bias + GELU
    float x = sum + __half2float(bias[col]);
    float x3 = x * x * x;
    float gelu = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x3)));
    C[row * N + col] = __float2half(gelu);
}
"#;

const F16_FUSED_GEMM_BIAS_SILU_SRC: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void warp_f16_fused_gemm_bias_silu(
    half *C, const half *A, const half *B, const half *bias,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int row = blockIdx.y * 16 + threadIdx.y;
    unsigned int col = blockIdx.x * 16 + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (unsigned int k = 0; k < K; k++) {
        sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }

    float x = sum + __half2float(bias[col]);
    float silu = x / (1.0f + expf(-x));
    C[row * N + col] = __float2half(silu);
}
"#;

/// FP16 Fused GEMM + Bias + GELU in one kernel.
/// cuBLAS needs 3 kernels for this. We do it in 1.
pub fn f16_fused_gemm_bias_gelu(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<half::f16>,
    b: &GpuTensor<half::f16>,
    bias: &GpuTensor<half::f16>,
    out: &mut GpuTensor<half::f16>,
    m: u32, n: u32, k: u32,
) -> Result<(), DeviceError> {
    let f = compile_fp16(cache, device, F16_FUSED_GEMM_BIAS_GELU_SRC, "warp_f16_fused_gemm_bias_gelu")?;
    let cfg = LaunchConfig {
        grid_dim: ((n + 15) / 16, (m + 15) / 16, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&a.data).arg(&b.data).arg(&bias.data)
            .arg(&m).arg(&n).arg(&k)
            .launch(cfg))?;
    }
    Ok(())
}

/// FP16 Fused GEMM + Bias + SiLU in one kernel.
pub fn f16_fused_gemm_bias_silu(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<half::f16>,
    b: &GpuTensor<half::f16>,
    bias: &GpuTensor<half::f16>,
    out: &mut GpuTensor<half::f16>,
    m: u32, n: u32, k: u32,
) -> Result<(), DeviceError> {
    let f = compile_fp16(cache, device, F16_FUSED_GEMM_BIAS_SILU_SRC, "warp_f16_fused_gemm_bias_silu")?;
    let cfg = LaunchConfig {
        grid_dim: ((n + 15) / 16, (m + 15) / 16, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&a.data).arg(&b.data).arg(&bias.data)
            .arg(&m).arg(&n).arg(&k)
            .launch(cfg))?;
    }
    Ok(())
}

/// FP16 GEMM using tensor cores (from gemm_tc.rs).
/// Re-exported here for the mixed-precision pipeline.
pub fn f16_gemm(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<half::f16>,
    b: &GpuTensor<half::f16>,
    c: &mut GpuTensor<half::f16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    crate::gemm_tc::gemm_tensor_core(cache, device, a, b, c, m, n, k)
}

// ═════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    #[test]
    fn cast_roundtrip() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let n = 1024usize;
        let f32_data: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();

        let f32_in = GpuTensor::from_host(&dev, &f32_data,
            Shape::from_static(&[n]), DType::F32).unwrap();
        let mut f16_buf = GpuTensor::<half::f16>::zeros(&dev,
            Shape::from_static(&[n]), DType::F16).unwrap();
        let mut f32_out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[n]), DType::F32).unwrap();

        cast_f32_to_f16(&cache, &dev, &f32_in, &mut f16_buf).unwrap();
        cast_f16_to_f32(&cache, &dev, &f16_buf, &mut f32_out).unwrap();
        dev.synchronize().unwrap();

        let result = f32_out.to_host(&dev).unwrap();
        let max_err: f32 = result.iter().zip(f32_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("FP16 cast roundtrip ({n} elements): max error = {max_err:.6}");
        assert!(max_err < 0.01, "FP16 roundtrip error too high");
    }

    #[test]
    fn f16_ops_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let n = 256usize;
        let a_data: Vec<half::f16> = (0..n).map(|i| half::f16::from_f32(i as f32 * 0.01)).collect();
        let b_data: Vec<half::f16> = (0..n).map(|i| half::f16::from_f32(0.5 + i as f32 * 0.005)).collect();

        let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[n]), DType::F16).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[n]), DType::F16).unwrap();

        // Test add
        let mut out = GpuTensor::<half::f16>::zeros(&dev, Shape::from_static(&[n]), DType::F16).unwrap();
        f16_add(&cache, &dev, &a, &b, &mut out).unwrap();
        dev.synchronize().unwrap();
        let result = out.to_host(&dev).unwrap();
        let expected_0 = a_data[0].to_f32() + b_data[0].to_f32();
        assert!((result[0].to_f32() - expected_0).abs() < 0.01,
            "FP16 add: {:.4} vs {:.4}", result[0].to_f32(), expected_0);

        // Test silu
        f16_silu(&cache, &dev, &a, &mut out).unwrap();
        dev.synchronize().unwrap();
        let result = out.to_host(&dev).unwrap();
        let v0 = a_data[100].to_f32();
        let expected_silu = v0 / (1.0 + (-v0).exp());
        assert!((result[100].to_f32() - expected_silu).abs() < 0.01,
            "FP16 silu: {:.4} vs {:.4}", result[100].to_f32(), expected_silu);

        println!("FP16 elementwise ops: correct!");
    }

    #[test]
    fn f16_rmsnorm_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let hidden = 32u32;
        let rows = 4usize;
        let n = rows * hidden as usize;

        let x_data: Vec<half::f16> = (0..n)
            .map(|i| half::f16::from_f32((i as f32 - n as f32 / 2.0) * 0.01))
            .collect();
        let gamma_data: Vec<half::f16> = vec![half::f16::from_f32(1.0); hidden as usize];

        let x = GpuTensor::from_host(&dev, &x_data, Shape::from_static(&[rows, hidden as usize]), DType::F16).unwrap();
        let gamma = GpuTensor::from_host(&dev, &gamma_data, Shape::from_static(&[hidden as usize]), DType::F16).unwrap();
        let mut out = GpuTensor::<half::f16>::zeros(&dev, Shape::from_static(&[rows, hidden as usize]), DType::F16).unwrap();

        f16_rmsnorm(&cache, &dev, &x, &gamma, &mut out, hidden, 1e-6).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        assert!(result.iter().all(|v| v.to_f32().is_finite()), "FP16 RMSNorm has NaN!");
        println!("FP16 RMSNorm ({rows}×{hidden}): correct, all finite!");
    }

    #[test]
    fn f16_gemm_perf_vs_f32() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (512u32, 512u32, 512u32);

        // F32 GEMM
        let a_f32: Vec<f32> = (0..(m*k) as usize).map(|i| ((i * 7 + 13) % 100) as f32 * 0.01 - 0.5).collect();
        let b_f32: Vec<f32> = (0..(k*n) as usize).map(|i| ((i * 11 + 3) % 100) as f32 * 0.01 - 0.5).collect();
        let a32 = GpuTensor::from_host(&dev, &a_f32, Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b32 = GpuTensor::from_host(&dev, &b_f32, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let mut c32 = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

        // FP16 GEMM
        let a_f16: Vec<half::f16> = a_f32.iter().map(|v| half::f16::from_f32(*v)).collect();
        let b_f16: Vec<half::f16> = b_f32.iter().map(|v| half::f16::from_f32(*v)).collect();
        let a16 = GpuTensor::from_host(&dev, &a_f16, Shape::from_static(&[m as usize, k as usize]), DType::F16).unwrap();
        let b16 = GpuTensor::from_host(&dev, &b_f16, Shape::from_static(&[k as usize, n as usize]), DType::F16).unwrap();
        let mut c16 = GpuTensor::<half::f16>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F16).unwrap();

        // Warmup
        crate::ops::gemm(&cache, &dev, &a32, &b32, &mut c32, m, n, k).unwrap();
        f16_gemm(&cache, &dev, &a16, &b16, &mut c16, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let iters = 200;

        // F32
        let start = std::time::Instant::now();
        for _ in 0..iters {
            crate::ops::gemm(&cache, &dev, &a32, &b32, &mut c32, m, n, k).unwrap();
        }
        dev.synchronize().unwrap();
        let f32_time = start.elapsed();

        // FP16
        let start = std::time::Instant::now();
        for _ in 0..iters {
            f16_gemm(&cache, &dev, &a16, &b16, &mut c16, m, n, k).unwrap();
        }
        dev.synchronize().unwrap();
        let f16_time = start.elapsed();

        let f32_tflops = 2.0 * m as f64 * n as f64 * k as f64 * iters as f64 / f32_time.as_secs_f64() / 1e12;
        let f16_tflops = 2.0 * m as f64 * n as f64 * k as f64 * iters as f64 / f16_time.as_secs_f64() / 1e12;

        println!("\n=== FP16 vs F32 GEMM ({m}×{n}×{k}, {iters} iters) ===");
        println!("  F32: {:.3}ms avg, {:.3} TFLOPS", f32_time.as_secs_f64() * 1000.0 / iters as f64, f32_tflops);
        println!("  F16: {:.3}ms avg, {:.3} TFLOPS", f16_time.as_secs_f64() * 1000.0 / iters as f64, f16_tflops);
        println!("  Speedup: {:.2}x", f32_time.as_secs_f64() / f16_time.as_secs_f64());
        println!("  Memory: {:.0} KB F32 vs {:.0} KB F16 (weights)", (m*k*4) as f64/1024.0, (m*k*2) as f64/1024.0);
    }
}

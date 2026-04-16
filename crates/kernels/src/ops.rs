//! High-level cached kernel operations.
//!
//! These are the user-facing functions that combine the cache, device,
//! and kernel source into simple callable operations. Each op compiles
//! on first call and is instant on subsequent calls.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Helper to launch with proper error mapping.
macro_rules! launch_err {
    ($e:expr) => {
        $e.map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))
    };
}

// ── Elementwise ──────────────────────────────────────────────────

const ADD_SRC: &str = r#"
extern "C" __global__ void warp_add(float *out, const float *a, const float *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = a[i] + b[i]; }
}
"#;

const MUL_SRC: &str = r#"
extern "C" __global__ void warp_mul(float *out, const float *a, const float *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = a[i] * b[i]; }
}
"#;

const GELU_SRC: &str = r#"
extern "C" __global__ void warp_gelu(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float v3 = v * v * v;
        out[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v3)));
    }
}
"#;

const SILU_SRC: &str = r#"
extern "C" __global__ void warp_silu(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i] = v / (1.0f + expf(-v));
    }
}
"#;

const SUB_SRC: &str = r#"
extern "C" __global__ void warp_sub(float *out, const float *a, const float *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = a[i] - b[i]; }
}
"#;

const DIV_SRC: &str = r#"
extern "C" __global__ void warp_div(float *out, const float *a, const float *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = a[i] / b[i]; }
}
"#;

// ── Transpose ───────────────────────────────────────────────────
// 2D transpose: [M, N] → [N, M]

const TRANSPOSE_2D_SRC: &str = r#"
#define TILE 32
extern "C" __global__ void warp_transpose_2d(
    float *out, const float *in_data,
    unsigned int M, unsigned int N
) {
    __shared__ float tile[TILE][TILE + 1]; // +1 avoids bank conflicts

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;

    // Read from input [M, N]
    unsigned int row = by * TILE + ty;
    unsigned int col = bx * TILE + tx;
    if (row < M && col < N) {
        tile[ty][tx] = in_data[row * N + col];
    }
    __syncthreads();

    // Write to output [N, M] — transposed indices
    unsigned int out_row = bx * TILE + ty;
    unsigned int out_col = by * TILE + tx;
    if (out_row < N && out_col < M) {
        out[out_row * M + out_col] = tile[tx][ty];
    }
}
"#;

// ── Reduce ──────────────────────────────────────────────────────
// Reduce along last dimension: [rows, cols] → [rows]

const REDUCE_SUM_SRC: &str = r#"
extern "C" __global__ void warp_reduce_sum(
    float *out, const float *input,
    unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (unsigned int c = 0; c < cols; c++) {
        sum += input[row * cols + c];
    }
    out[row] = sum;
}
"#;

const REDUCE_MEAN_SRC: &str = r#"
extern "C" __global__ void warp_reduce_mean(
    float *out, const float *input,
    unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (unsigned int c = 0; c < cols; c++) {
        sum += input[row * cols + c];
    }
    out[row] = sum / (float)cols;
}
"#;

const REDUCE_MAX_SRC: &str = r#"
extern "C" __global__ void warp_reduce_max(
    float *out, const float *input,
    unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float mx = -1e30f;
    for (unsigned int c = 0; c < cols; c++) {
        float v = input[row * cols + c];
        if (v > mx) mx = v;
    }
    out[row] = mx;
}
"#;

const RELU_SRC: &str = r#"
extern "C" __global__ void warp_relu(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = fmaxf(x[i], 0.0f); }
}
"#;

const SIGMOID_SRC: &str = r#"
extern "C" __global__ void warp_sigmoid(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = 1.0f / (1.0f + expf(-x[i])); }
}
"#;

const TANH_SRC: &str = r#"
extern "C" __global__ void warp_tanh(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = tanhf(x[i]); }
}
"#;

const LEAKY_RELU_SRC: &str = r#"
extern "C" __global__ void warp_leaky_relu(
    float *out, const float *x, float alpha, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i] = v > 0.0f ? v : alpha * v;
    }
}
"#;

const CLIP_SRC: &str = r#"
extern "C" __global__ void warp_clip(
    float *out, const float *x, float lo, float hi, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = fminf(fmaxf(x[i], lo), hi); }
}
"#;

// ── GroupNorm ────────────────────────────────────────────────────
// Used heavily by Stable Diffusion, ViT variants.
// y = ((x - mean) / sqrt(var + eps)) * scale + bias
// Mean/var computed per group across spatial dims.

const GROUPNORM_SRC: &str = r#"
extern "C" __global__ void warp_groupnorm(
    float *out,
    const float *x,         // [C, spatial]
    const float *scale,     // [C]
    const float *bias,      // [C]
    unsigned int C,
    unsigned int spatial,    // H * W
    unsigned int num_groups,
    float eps
) {
    unsigned int group = blockIdx.x;
    if (group >= num_groups) return;

    unsigned int channels_per_group = C / num_groups;
    unsigned int group_start = group * channels_per_group;
    unsigned int group_size = channels_per_group * spatial;

    // Compute mean and variance for this group
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < group_size; i += blockDim.x) {
        unsigned int c = group_start + i / spatial;
        unsigned int s = i % spatial;
        sum += x[c * spatial + s];
    }

    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    sum = __shfl_sync(0xffffffff, sum, 0);
    float mean = sum / (float)group_size;

    // Variance
    float var_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < group_size; i += blockDim.x) {
        unsigned int c = group_start + i / spatial;
        unsigned int s = i % spatial;
        float diff = x[c * spatial + s] - mean;
        var_sum += diff * diff;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    var_sum = __shfl_sync(0xffffffff, var_sum, 0);
    float inv_std = rsqrtf(var_sum / (float)group_size + eps);

    // Normalize
    for (unsigned int i = threadIdx.x; i < group_size; i += blockDim.x) {
        unsigned int c = group_start + i / spatial;
        unsigned int s = i % spatial;
        unsigned int idx = c * spatial + s;
        out[idx] = (x[idx] - mean) * inv_std * scale[c] + bias[c];
    }
}
"#;

// ── Concat ──────────────────────────────────────────────────────
// Concatenate tensors along the channel dimension (axis=1 for NCHW).

const CONCAT_SRC: &str = r#"
extern "C" __global__ void warp_concat_channels(
    float *out,             // [total_C, spatial]
    const float *a,         // [C_a, spatial]
    const float *b,         // [C_b, spatial]
    unsigned int C_a,
    unsigned int C_b,
    unsigned int spatial
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = (C_a + C_b) * spatial;
    if (idx >= total) return;

    unsigned int c = idx / spatial;
    unsigned int s = idx % spatial;

    if (c < C_a) {
        out[idx] = a[c * spatial + s];
    } else {
        out[idx] = b[(c - C_a) * spatial + s];
    }
}
"#;

const FUSED_ADD_GELU_SRC: &str = r#"
extern "C" __global__ void warp_fused_add_gelu(float *out, const float *a, const float *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a[i] + b[i];
        float v3 = v * v * v;
        out[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v3)));
    }
}
"#;

const FUSED_ADD_SILU_SRC: &str = r#"
extern "C" __global__ void warp_fused_add_silu(float *out, const float *a, const float *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a[i] + b[i];
        out[i] = v / (1.0f + expf(-v));
    }
}
"#;

const RMSNORM_SRC: &str = r#"
extern "C" __global__ void warp_rmsnorm(
    float *out, const float *x, const float *gamma,
    unsigned int hidden_size, float eps, size_t n_rows
) {
    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const float *x_row = x + row * hidden_size;
    float *out_row = out + row * hidden_size;

    // Compute RMS
    float sum_sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = x_row[i];
        sum_sq += v * v;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    // Broadcast from lane 0
    sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    // Normalize: x * rms * gamma
    // NOTE: For Gemma models, gamma is a learned offset (initialized to 0),
    // so the caller should pass (1+gamma) if needed. Standard LLaMA uses gamma directly.
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = x_row[i] * rms * gamma[i];
    }
}
"#;

/// Fused RMSNorm with F16 output — eliminates a separate F32→F16 cast kernel.
/// out_f16[i] = __float2half(x[i] * rms * gamma[i])
const RMSNORM_F16OUT_SRC: &str = r#"
extern "C" __global__ void warp_rmsnorm_f16out(
    unsigned short *out, const float *x, const float *gamma,
    unsigned int hidden_size, float eps, size_t n_rows
) {
    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const float *x_row = x + row * hidden_size;
    unsigned short *out_row = out + row * hidden_size;

    float sum_sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = x_row[i];
        sum_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = x_row[i] * rms * gamma[i];
        unsigned short h;
        asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(val));
        out_row[i] = h;
    }
}
"#;

/// Fused RMSNorm (no weight) with F16 output.
const RMSNORM_NO_WEIGHT_F16OUT_SRC: &str = r#"
extern "C" __global__ void warp_rmsnorm_noweight_f16out(
    unsigned short *out, const float *x,
    unsigned int hidden_size, float eps, size_t n_rows
) {
    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const float *x_row = x + row * hidden_size;
    unsigned short *out_row = out + row * hidden_size;

    float sum_sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = x_row[i];
        sum_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = x_row[i] * rms;
        unsigned short h;
        asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(val));
        out_row[i] = h;
    }
}
"#;

/// RMSNorm without learnable weights — used for QK-norm in Gemma 3/4.
/// Normalizes each row: out[i] = x[i] * rsqrt(mean(x^2) + eps)
const RMSNORM_NO_WEIGHT_SRC: &str = r#"
extern "C" __global__ void warp_rmsnorm_noweight(
    float *out, const float *x,
    unsigned int hidden_size, float eps, size_t n_rows
) {
    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const float *x_row = x + row * hidden_size;
    float *out_row = out + row * hidden_size;

    float sum_sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = x_row[i];
        sum_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = x_row[i] * rms;
    }
}
"#;

/// QK-norm: RMSNorm without learnable weights.
/// Normalizes Q and K tensors before attention (Gemma 3/4).
/// Input shape: [num_heads, head_dim] flattened — each head is one row.
pub fn rmsnorm_no_weight(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    hidden_size: u32,
    eps: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, RMSNORM_NO_WEIGHT_SRC, "warp_rmsnorm_noweight")?;
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
            .arg(&hidden_size)
            .arg(&eps)
            .arg(&n_rows)
            .launch(cfg))?;
    }
    Ok(())
}

/// RMSNorm without weights, outputting F16 directly. Fuses rmsnorm_no_weight + cast_f32_to_f16.
pub fn rmsnorm_no_weight_f16out(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    out: &mut GpuTensor<half::f16>,
    hidden_size: u32,
    eps: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, RMSNORM_NO_WEIGHT_F16OUT_SRC, "warp_rmsnorm_noweight_f16out")?;
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
            .arg(&hidden_size)
            .arg(&eps)
            .arg(&n_rows)
            .launch(cfg))?;
    }
    Ok(())
}

/// Cached elementwise add.
pub fn add(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, ADD_SRC, "warp_add")?;
    let n = a.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&a.data).arg(&b.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Broadcast add: out[i] = a[i] + b[i % b_len].
/// Adds a 1D bias vector to each row of a 2D tensor.
/// a: [rows, cols], b: [cols], out: [rows, cols]
const BROADCAST_ADD_SRC: &str = r#"
extern "C" __global__ void warp_broadcast_add(
    float *out, const float *a, const float *b,
    size_t total, size_t b_len
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        out[i] = a[i] + b[i % b_len];
    }
}
"#;

pub fn broadcast_add(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,       // [rows, cols]
    bias: &GpuTensor<f32>,    // [cols]
    out: &mut GpuTensor<f32>, // [rows, cols]
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, BROADCAST_ADD_SRC, "warp_broadcast_add")?;
    let total = a.numel;
    let b_len = bias.numel;
    let cfg = LaunchConfig::for_num_elems(total as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&a.data).arg(&bias.data).arg(&total).arg(&b_len)
            .launch(cfg))?;
    }
    Ok(())
}

/// Cached elementwise mul.
pub fn mul(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, MUL_SRC, "warp_mul")?;
    let n = a.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&a.data).arg(&b.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Cached elementwise sub.
pub fn sub(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, SUB_SRC, "warp_sub")?;
    let n = a.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&a.data).arg(&b.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Cached elementwise div.
pub fn div(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, DIV_SRC, "warp_div")?;
    let n = a.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&a.data).arg(&b.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// 2D transpose: [M, N] → [N, M] with shared memory tiling.
pub fn transpose_2d(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, TRANSPOSE_2D_SRC, "warp_transpose_2d")?;
    let tile = 32u32;
    let cfg = LaunchConfig {
        grid_dim: ((n + tile - 1) / tile, (m + tile - 1) / tile, 1),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&m).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Reduce sum along last dimension: [rows, cols] → [rows].
pub fn reduce_sum(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    rows: u32,
    cols: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, REDUCE_SUM_SRC, "warp_reduce_sum")?;
    let cfg = LaunchConfig::for_num_elems(rows);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&rows).arg(&cols)
            .launch(cfg))?;
    }
    Ok(())
}

/// Reduce mean along last dimension: [rows, cols] → [rows].
pub fn reduce_mean(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    rows: u32,
    cols: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, REDUCE_MEAN_SRC, "warp_reduce_mean")?;
    let cfg = LaunchConfig::for_num_elems(rows);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&rows).arg(&cols)
            .launch(cfg))?;
    }
    Ok(())
}

/// Reduce max along last dimension: [rows, cols] → [rows].
pub fn reduce_max(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    rows: u32,
    cols: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, REDUCE_MAX_SRC, "warp_reduce_max")?;
    let cfg = LaunchConfig::for_num_elems(rows);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&rows).arg(&cols)
            .launch(cfg))?;
    }
    Ok(())
}

/// Cached GELU activation.
pub fn gelu(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, GELU_SRC, "warp_gelu")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Cached SiLU activation.
pub fn silu(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, SILU_SRC, "warp_silu")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Cached ReLU.
pub fn relu(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, RELU_SRC, "warp_relu")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Cached sigmoid.
pub fn sigmoid(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, SIGMOID_SRC, "warp_sigmoid")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Cached tanh.
pub fn tanh_act(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, TANH_SRC, "warp_tanh")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Cached leaky ReLU.
pub fn leaky_relu(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    alpha: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, LEAKY_RELU_SRC, "warp_leaky_relu")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&alpha).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Cached clip/clamp.
pub fn clip(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    lo: f32,
    hi: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, CLIP_SRC, "warp_clip")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&lo).arg(&hi).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Group normalization. Used by Stable Diffusion, modern ViTs.
pub fn groupnorm(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    scale: &GpuTensor<f32>,
    bias: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    channels: u32,
    spatial: u32,
    num_groups: u32,
    eps: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, GROUPNORM_SRC, "warp_groupnorm")?;
    let batch = x.numel as u32 / (channels * spatial);

    for n in 0..batch {
        let off = (n * channels) as usize * spatial as usize;
        let cfg = LaunchConfig {
            grid_dim: (num_groups, 1, 1),
            block_dim: (32, 1, 1), // one warp per group
            shared_mem_bytes: 0,
        };
        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut out.data.slice_mut(off..))
                .arg(&x.data.slice(off..))
                .arg(&scale.data)
                .arg(&bias.data)
                .arg(&channels)
                .arg(&spatial)
                .arg(&num_groups)
                .arg(&eps)
                .launch(cfg))?;
        }
    }
    Ok(())
}

/// Concat two tensors along channel dimension.
/// a: [C_a, spatial], b: [C_b, spatial] → out: [C_a + C_b, spatial]
pub fn concat_channels(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    c_a: u32,
    c_b: u32,
    spatial: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, CONCAT_SRC, "warp_concat_channels")?;
    let batch = a.numel as u32 / (c_a * spatial);
    for n in 0..batch {
        let a_off = (n * c_a) as usize * spatial as usize;
        let b_off = (n * c_b) as usize * spatial as usize;
        let out_off = (n * (c_a + c_b)) as usize * spatial as usize;
        let total = (c_a + c_b) * spatial;
        let cfg = LaunchConfig::for_num_elems(total);
        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut out.data.slice_mut(out_off..))
                .arg(&a.data.slice(a_off..))
                .arg(&b.data.slice(b_off..))
                .arg(&c_a).arg(&c_b).arg(&spatial)
                .launch(cfg))?;
        }
    }
    Ok(())
}

/// Cached fused Add + GELU (single kernel, 1 memory pass instead of 2).
pub fn fused_add_gelu(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, FUSED_ADD_GELU_SRC, "warp_fused_add_gelu")?;
    let n = a.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&a.data).arg(&b.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Cached fused Add + SiLU.
pub fn fused_add_silu(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, FUSED_ADD_SILU_SRC, "warp_fused_add_silu")?;
    let n = a.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&a.data).arg(&b.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

// ── Fused residual + RMSNorm ──────────────────────────────────
// out = rmsnorm(x + residual, gamma)
// Saves one kernel launch + one global memory pass.

// Fused residual + RMSNorm with DUAL output:
// residual_out = x + residual (for later use)
// norm_out = rmsnorm(x + residual, gamma)
// One kernel, one memory pass, two outputs. Saves 1 launch vs separate add+rmsnorm.
const FUSED_RESIDUAL_RMSNORM_SRC: &str = r#"
extern "C" __global__ void warp_fused_residual_rmsnorm(
    float *norm_out,        // rmsnorm result
    float *residual_out,    // x + residual (un-normalized sum)
    const float *x, const float *residual, const float *gamma,
    unsigned int hidden_size, float eps, size_t n_rows
) {
    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const float *x_row = x + row * hidden_size;
    const float *r_row = residual + row * hidden_size;
    float *nout = norm_out + row * hidden_size;
    float *rout = residual_out + row * hidden_size;

    float sum_sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = x_row[i] + r_row[i];
        sum_sq += v * v;
        rout[i] = v;     // store residual sum
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        nout[i] = rout[i] * rms * gamma[i];
    }
}
"#;

// ── Fused SwiGLU gate: silu(gate) * up ──────────────────────
// out = silu(gate) * up
// Saves one kernel launch.

const FUSED_SILU_MUL_SRC: &str = r#"
extern "C" __global__ void warp_fused_silu_mul(
    float *out, const float *gate, const float *up, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}
"#;

/// Fused triple-norm: reads residual once, produces 3 outputs:
/// 1. ffn_f16 = f16(rmsnorm(x, gamma1))  — dense MLP input
/// 2. router_scaled = rmsnorm_noweight(x) * router_scale * scalar_root — router input
/// 3. moe_in = rmsnorm(x, gamma2) — expert MoE input
/// Replaces 5 kernels (3 norms + mul + mul_scalar) with 1.
const FUSED_TRIPLE_NORM_SRC: &str = r#"
extern "C" __global__ void fused_triple_norm(
    unsigned short* __restrict__ ffn_f16,     // [H] F16 output
    float* __restrict__ router_scaled,        // [H] router output (normed * scale * root)
    float* __restrict__ moe_in,               // [H] MoE input
    const float* __restrict__ x,              // [H] input (residual)
    const float* __restrict__ gamma1,         // [H] pre_ffn_norm
    const float* __restrict__ gamma2,         // [H] pre_ffn_norm_2
    const float* __restrict__ router_scale,   // [H] router scale weights
    unsigned int H, float eps, float scalar_root
) {
    float sum_sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);
    sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);
    float rms = rsqrtf(sum_sq / (float)H + eps);

    for (unsigned int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = x[i];
        float normed = v * rms;
        // Output 1: F16 normed with gamma1
        float f = normed * gamma1[i];
        unsigned short h;
        asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
        ffn_f16[i] = h;
        // Output 2: router (no-weight norm * router_scale * scalar_root)
        router_scaled[i] = normed * router_scale[i] * scalar_root;
        // Output 3: normed with gamma2
        moe_in[i] = normed * gamma2[i];
    }
}
"#;

pub fn fused_triple_norm(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    gamma1: &GpuTensor<f32>,
    gamma2: &GpuTensor<f32>,
    router_scale: &GpuTensor<f32>,
    ffn_f16: &mut GpuTensor<half::f16>,
    router_scaled: &mut GpuTensor<f32>,
    moe_in: &mut GpuTensor<f32>,
    hidden_size: u32,
    eps: f32,
    scalar_root: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, FUSED_TRIPLE_NORM_SRC, "fused_triple_norm")?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32.min(hidden_size), 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut ffn_f16.data).arg(&mut router_scaled.data).arg(&mut moe_in.data)
            .arg(&x.data).arg(&gamma1.data).arg(&gamma2.data).arg(&router_scale.data)
            .arg(&hidden_size).arg(&eps).arg(&scalar_root)
            .launch(cfg))?;
    }
    Ok(())
}

/// Fused rmsnorm + add: out = a + rmsnorm(b, gamma).
/// Replaces rmsnorm(b) → post_attn + add(x, post_attn) → residual (2 kernels → 1).
const FUSED_RMSNORM_ADD_SRC: &str = r#"
extern "C" __global__ void warp_fused_rmsnorm_add(
    float *out, const float *a, const float *b, const float *gamma,
    unsigned int hidden_size, float eps
) {
    float sum_sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = b[i];
        sum_sq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);
    sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);
    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out[i] = a[i] + b[i] * rms * gamma[i];
    }
}
"#;

pub fn fused_rmsnorm_add(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    gamma: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    hidden_size: u32,
    eps: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, FUSED_RMSNORM_ADD_SRC, "warp_fused_rmsnorm_add")?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32.min(hidden_size), 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&a.data).arg(&b.data).arg(&gamma.data)
            .arg(&hidden_size).arg(&eps)
            .launch(cfg))?;
    }
    Ok(())
}

/// Fused residual add + RMSNorm with dual output:
/// - norm_out = rmsnorm(x + residual, gamma)
/// - residual_out = x + residual (un-normalized, for downstream residual)
///
/// Replaces separate add + rmsnorm (2 launches + 2 memory passes → 1 launch + 1 pass).
pub fn fused_residual_rmsnorm(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    residual: &GpuTensor<f32>,
    gamma: &GpuTensor<f32>,
    norm_out: &mut GpuTensor<f32>,
    residual_out: &mut GpuTensor<f32>,
    hidden_size: u32,
    eps: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, FUSED_RESIDUAL_RMSNORM_SRC, "warp_fused_residual_rmsnorm")?;
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

/// Fused SwiGLU gate: out = silu(gate) * up.
/// Replaces separate silu + mul (2 launches → 1).
pub fn fused_silu_mul(
    cache: &KernelCache,
    device: &WarpDevice,
    gate: &GpuTensor<f32>,
    up: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, FUSED_SILU_MUL_SRC, "warp_fused_silu_mul")?;
    let n = gate.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&gate.data).arg(&up.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Fused GeGLU gate: out = gelu(gate) * up.
/// Used by Gemma 2/3/4 instead of SwiGLU. Uses the tanh approximation of GELU
/// (gelu_pytorch_tanh): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
const FUSED_GELU_MUL_SRC: &str = r#"
extern "C" __global__ void warp_fused_gelu_mul(
    float *out, const float *gate, const float *up, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float gelu_g = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
        out[i] = gelu_g * up[i];
    }
}
"#;

/// Fused GeGLU gate: out = gelu_tanh(gate) * up.
/// Same interface as fused_silu_mul but with GELU activation (Gemma family).
pub fn fused_gelu_mul(
    cache: &KernelCache,
    device: &WarpDevice,
    gate: &GpuTensor<f32>,
    up: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, FUSED_GELU_MUL_SRC, "warp_fused_gelu_mul")?;
    let n = gate.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&gate.data).arg(&up.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Fused split + GeGLU: out[i] = gelu(gate_up[i]) * gate_up[i + dim]
/// Eliminates split_gate_up + fused_gelu_mul (2 kernels → 1).
const SPLIT_GEGLU_SRC: &str = r#"
extern "C" __global__ void warp_split_geglu(
    float *out, const float *gate_up, size_t dim
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        float g = gate_up[i];
        float u = gate_up[i + dim];
        float gelu_g = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
        out[i] = gelu_g * u;
    }
}
"#;

pub fn split_geglu(
    cache: &KernelCache,
    device: &WarpDevice,
    gate_up: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    dim: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, SPLIT_GEGLU_SRC, "warp_split_geglu")?;
    let cfg = LaunchConfig::for_num_elems(dim);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&gate_up.data).arg(&(dim as usize))
            .launch(cfg))?;
    }
    Ok(())
}

/// Logit softcapping: out = cap * tanh(x / cap).
/// Used by Gemma 2/4 to bound logit magnitudes before softmax.
/// cap=0.0 means disabled (no-op copy).
const LOGIT_SOFTCAP_SRC: &str = r#"
extern "C" __global__ void warp_logit_softcap(
    float *out, const float *x, float cap, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = cap * tanhf(x[i] / cap);
    }
}
"#;

/// Apply logit softcapping: out = cap * tanh(x / cap).
/// No-op if cap <= 0.0.
/// Multiply all elements by a scalar: out[i] = x[i] * s
const MUL_SCALAR_SRC: &str = r#"
extern "C" __global__ void warp_mul_scalar(float *out, const float *x, float s, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * s;
}
"#;

pub fn mul_scalar(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    scalar: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, MUL_SCALAR_SRC, "warp_mul_scalar")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&scalar).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Fused axpy: acc[i] += alpha * x[i]  (in-place accumulate)
/// Replaces mul_scalar + add + copy (3 kernels → 1).
const AXPY_SRC: &str = r#"
extern "C" __global__ void warp_axpy(float *acc, const float *x, float alpha, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += alpha * x[i];
}
"#;

pub fn axpy(
    cache: &KernelCache,
    device: &WarpDevice,
    acc: &mut GpuTensor<f32>,
    x: &GpuTensor<f32>,
    alpha: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, AXPY_SRC, "warp_axpy")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut acc.data).arg(&x.data).arg(&alpha).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// axpy with device-resident weight: acc[i] += (*weight_ptr) * x[i]
/// Reads the scalar weight from a device pointer — no DtoH needed.
const AXPY_INDIRECT_SRC: &str = r#"
extern "C" __global__ void warp_axpy_indirect(float *acc, const float *x, const float *weight_ptr, size_t n) {
    float alpha = *weight_ptr;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += alpha * x[i];
}
"#;

pub fn axpy_indirect(
    cache: &KernelCache,
    device: &WarpDevice,
    acc: &mut GpuTensor<f32>,
    x: &GpuTensor<f32>,
    weight_ptr: &cudarc::driver::CudaView<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, AXPY_INDIRECT_SRC, "warp_axpy_indirect")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut acc.data).arg(&x.data).arg(weight_ptr).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Compute expert byte offsets on GPU from router topK IDs. No DtoH needed.
/// Outputs 4 offset arrays (gate+up packed, gate+up scales, down packed, down scales).
const COMPUTE_EXPERT_OFFSETS_SRC: &str = r#"
extern "C" __global__ void compute_expert_offsets(
    unsigned int* gu_offsets,
    unsigned int* d_offsets,
    unsigned int* gu_scale_offsets,
    unsigned int* d_scale_offsets,
    const float* topk_ids,
    unsigned int gu_bytes_per_expert,
    unsigned int d_bytes_per_expert,
    unsigned int gu_scales_per_expert,
    unsigned int d_scales_per_expert,
    int top_k
) {
    int i = threadIdx.x;
    if (i >= top_k) return;
    unsigned int eid = (unsigned int)topk_ids[i];
    gu_offsets[i] = eid * gu_bytes_per_expert;
    d_offsets[i] = eid * d_bytes_per_expert;
    gu_scale_offsets[i] = eid * gu_scales_per_expert;
    d_scale_offsets[i] = eid * d_scales_per_expert;
}
"#;

pub fn compute_expert_offsets(
    cache: &KernelCache,
    device: &WarpDevice,
    topk_ids: &GpuTensor<f32>,
    gu_offsets: &mut cudarc::driver::CudaSlice<u32>,
    d_offsets: &mut cudarc::driver::CudaSlice<u32>,
    gu_scale_offsets: &mut cudarc::driver::CudaSlice<u32>,
    d_scale_offsets: &mut cudarc::driver::CudaSlice<u32>,
    gu_bytes_per_expert: u32,
    d_bytes_per_expert: u32,
    gu_scales_per_expert: u32,
    d_scales_per_expert: u32,
    top_k: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, COMPUTE_EXPERT_OFFSETS_SRC, "compute_expert_offsets")?;
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (top_k, 1, 1), shared_mem_bytes: 0 };
    let top_k_i = top_k as i32;
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(gu_offsets)
            .arg(d_offsets)
            .arg(gu_scale_offsets)
            .arg(d_scale_offsets)
            .arg(&topk_ids.data)
            .arg(&gu_bytes_per_expert)
            .arg(&d_bytes_per_expert)
            .arg(&gu_scales_per_expert)
            .arg(&d_scales_per_expert)
            .arg(&top_k_i)
            .launch(cfg))?;
    }
    Ok(())
}

/// GPU-side router TopK: finds top-K experts from softmax probs, normalizes weights.
/// Output: out_ids[0..K] = expert indices (as f32 for uniform buffer), out_weights[0..K] = normalized weights.
/// Single-block kernel for N≤1024 experts, K≤32.
const ROUTER_TOPK_SRC: &str = r#"
extern "C" __global__ void warp_router_topk(
    float *out_ids, float *out_weights,
    const float *probs, const float *per_expert_scale,
    int N, int K
) {
    // Single-block: each thread handles a subset of experts
    __shared__ float s_prob[1024];
    __shared__ int   s_idx[1024];

    int tid = threadIdx.x;
    // Load probs + indices into shared memory
    for (int i = tid; i < N; i += blockDim.x) {
        s_prob[i] = probs[i];
        s_idx[i] = i;
    }
    __syncthreads();

    // Simple selection sort for top-K (K is small, N is small)
    // Only thread 0 does the selection to avoid complexity
    if (tid == 0) {
        for (int k = 0; k < K; k++) {
            int best = k;
            float best_val = s_prob[k];
            for (int j = k + 1; j < N; j++) {
                if (s_prob[j] > best_val) {
                    best = j;
                    best_val = s_prob[j];
                }
            }
            // Swap
            if (best != k) {
                float tp = s_prob[k]; s_prob[k] = s_prob[best]; s_prob[best] = tp;
                int ti = s_idx[k]; s_idx[k] = s_idx[best]; s_idx[best] = ti;
            }
        }
        // Normalize and apply per-expert scale
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += s_prob[k];
        float inv_sum = (sum > 0.0f) ? (1.0f / sum) : (1.0f / (float)K);
        for (int k = 0; k < K; k++) {
            int eid = s_idx[k];
            out_ids[k] = (float)eid;
            out_weights[k] = s_prob[k] * inv_sum * per_expert_scale[eid];
        }
    }
}
"#;

/// GPU-side router TopK — avoids 2× DtoH of 128 floats per layer.
/// Returns (expert_ids, expert_weights) on host via single small DtoH.
pub fn router_topk(
    cache: &KernelCache,
    device: &WarpDevice,
    probs: &GpuTensor<f32>,
    per_expert_scale: &GpuTensor<f32>,
    out_ids: &mut GpuTensor<f32>,
    out_weights: &mut GpuTensor<f32>,
    num_experts: u32,
    top_k: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, ROUTER_TOPK_SRC, "warp_router_topk")?;
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out_ids.data).arg(&mut out_weights.data)
            .arg(&probs.data).arg(&per_expert_scale.data)
            .arg(&(num_experts as i32)).arg(&(top_k as i32))
            .launch(cfg))?;
    }
    Ok(())
}

pub fn logit_softcap(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    cap: f32,
) -> Result<(), DeviceError> {
    if cap <= 0.0 {
        // No-op — GPU-side copy (was DtoH+HtoD roundtrip, ~1MB for 256K vocab)
        let f = cache.get_or_compile(device, MUL_SCALAR_SRC, "warp_mul_scalar")?;
        let n = x.numel;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut out.data).arg(&x.data).arg(&1.0f32).arg(&n)
                .launch(cfg))?;
        }
        return Ok(());
    }
    let f = cache.get_or_compile(device, LOGIT_SOFTCAP_SRC, "warp_logit_softcap")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&cap).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

// ── Fused QKV projection ─────────────────────────────────────
// Computes Q, K, V in a single GEMM by concatenating weight matrices:
// [normed] × [Wq | Wk | Wv] → [Q | K | V]
// 3 kernel launches → 1.

const SPLIT3_SRC: &str = r#"
extern "C" __global__ void warp_split3(
    float *q, float *k, float *v,
    const float *qkv,
    unsigned int q_dim, unsigned int k_dim, unsigned int v_dim,
    unsigned int total_dim, unsigned int rows
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = rows * total_dim;
    if (idx >= total) return;

    unsigned int row = idx / total_dim;
    unsigned int col = idx % total_dim;

    float val = qkv[idx];

    if (col < q_dim) {
        q[row * q_dim + col] = val;
    } else if (col < q_dim + k_dim) {
        k[row * k_dim + (col - q_dim)] = val;
    } else {
        v[row * v_dim + (col - q_dim - k_dim)] = val;
    }
}
"#;

/// Fused QKV projection: compute Q, K, V in one GEMM + split.
/// input: [batch, hidden_size]
/// wq: [hidden_size, q_dim], wk: [hidden_size, k_dim], wv: [hidden_size, v_dim]
///
/// Instead of 3 separate GEMMs, concatenates weights and does 1 wider GEMM,
/// then splits the output. Saves 2 kernel launches per transformer block.
pub fn fused_qkv_projection(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    wq: &GpuTensor<f32>,
    wk: &GpuTensor<f32>,
    wv: &GpuTensor<f32>,
    q_out: &mut GpuTensor<f32>,
    k_out: &mut GpuTensor<f32>,
    v_out: &mut GpuTensor<f32>,
    m: u32,       // batch*seq (rows)
    q_dim: u32,   // typically hidden_size
    k_dim: u32,   // typically kv_dim
    v_dim: u32,   // typically kv_dim
    hidden: u32,  // input hidden size (K dimension)
) -> Result<(), DeviceError> {
    let total_out = q_dim + k_dim + v_dim;

    // Concatenate weights on CPU (one-time cost, should be cached)
    let wq_host = wq.to_host(device)?;
    let wk_host = wk.to_host(device)?;
    let wv_host = wv.to_host(device)?;

    // Build concatenated weight [hidden, q_dim+k_dim+v_dim]
    let mut concat_w = vec![0.0f32; (hidden * total_out) as usize];
    for row in 0..hidden as usize {
        // Q columns
        for c in 0..q_dim as usize {
            concat_w[row * total_out as usize + c] = wq_host[row * q_dim as usize + c];
        }
        // K columns
        for c in 0..k_dim as usize {
            concat_w[row * total_out as usize + q_dim as usize + c] = wk_host[row * k_dim as usize + c];
        }
        // V columns
        for c in 0..v_dim as usize {
            concat_w[row * total_out as usize + q_dim as usize + k_dim as usize + c] = wv_host[row * v_dim as usize + c];
        }
    }

    let w_concat = GpuTensor::from_host(device, &concat_w,
        warp_ir::Shape::from_static(&[hidden as usize, total_out as usize]), warp_ir::DType::F32)?;

    // Single wide GEMM: [m, hidden] × [hidden, total_out] → [m, total_out]
    let mut qkv = GpuTensor::<f32>::zeros(device,
        warp_ir::Shape::from_static(&[m as usize, total_out as usize]), warp_ir::DType::F32)?;
    gemm(cache, device, input, &w_concat, &mut qkv, m, total_out, hidden)?;

    // Split into Q, K, V
    let f = cache.get_or_compile(device, SPLIT3_SRC, "warp_split3")?;
    let total = m * total_out;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut q_out.data)
            .arg(&mut k_out.data)
            .arg(&mut v_out.data)
            .arg(&qkv.data)
            .arg(&q_dim)
            .arg(&k_dim)
            .arg(&v_dim)
            .arg(&total_out)
            .arg(&m)
            .launch(cfg))?;
    }
    Ok(())
}

// ── Fused Gate+Up projection ─────────────────────────────────
// For SwiGLU FFN: compute gate and up projections in one GEMM.
// [input] × [W_gate | W_up] → [gate | up], then split.

const SPLIT2_SRC: &str = r#"
extern "C" __global__ void warp_split2(
    float *a_out, float *b_out,
    const float *ab,
    unsigned int a_dim, unsigned int b_dim,
    unsigned int total_dim, unsigned int rows
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = rows * total_dim;
    if (idx >= total) return;
    unsigned int row = idx / total_dim;
    unsigned int col = idx % total_dim;
    float val = ab[idx];
    if (col < a_dim) a_out[row * a_dim + col] = val;
    else b_out[row * b_dim + (col - a_dim)] = val;
}
"#;

/// Fused Gate+Up projection: compute both in one GEMM + split.
/// Saves 1 kernel launch per transformer block FFN.
pub fn fused_gate_up_proj(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    w_gate: &GpuTensor<f32>,
    w_up: &GpuTensor<f32>,
    gate_out: &mut GpuTensor<f32>,
    up_out: &mut GpuTensor<f32>,
    m: u32,
    ffn_dim: u32,
    hidden: u32,
) -> Result<(), DeviceError> {
    let total_out = ffn_dim * 2;

    // Concatenate gate + up weights [hidden, 2*ffn_dim]
    let wg_host = w_gate.to_host(device)?;
    let wu_host = w_up.to_host(device)?;

    let mut concat_w = vec![0.0f32; (hidden * total_out) as usize];
    for row in 0..hidden as usize {
        for c in 0..ffn_dim as usize {
            concat_w[row * total_out as usize + c] = wg_host[row * ffn_dim as usize + c];
            concat_w[row * total_out as usize + ffn_dim as usize + c] = wu_host[row * ffn_dim as usize + c];
        }
    }

    let w_concat = GpuTensor::from_host(device, &concat_w,
        warp_ir::Shape::from_static(&[hidden as usize, total_out as usize]), warp_ir::DType::F32)?;

    // Single GEMM
    let mut gate_up = GpuTensor::<f32>::zeros(device,
        warp_ir::Shape::from_static(&[m as usize, total_out as usize]), warp_ir::DType::F32)?;
    gemm(cache, device, input, &w_concat, &mut gate_up, m, total_out, hidden)?;

    // Split
    let f = cache.get_or_compile(device, SPLIT2_SRC, "warp_split2")?;
    let total = m * total_out;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut gate_out.data)
            .arg(&mut up_out.data)
            .arg(&gate_up.data)
            .arg(&ffn_dim)
            .arg(&ffn_dim)
            .arg(&total_out)
            .arg(&m)
            .launch(cfg))?;
    }
    Ok(())
}

// ── LayerNorm (proper, with mean subtraction) ────────────────

const LAYERNORM_SRC: &str = r#"
extern "C" __global__ void warp_layernorm(
    float *out, const float *x, const float *gamma, const float *beta,
    unsigned int hidden_size, float eps, size_t n_rows
) {
    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const float *x_row = x + row * hidden_size;
    float *out_row = out + row * hidden_size;

    // Compute mean
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum += x_row[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    sum = __shfl_sync(0xffffffff, sum, 0);
    float mean = sum / (float)hidden_size;

    // Compute variance
    float var_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = x_row[i] - mean;
        var_sum += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    var_sum = __shfl_sync(0xffffffff, var_sum, 0);

    float inv_std = rsqrtf(var_sum / (float)hidden_size + eps);

    // Normalize: (x - mean) * inv_std * gamma + beta
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = (x_row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
"#;

/// Proper LayerNorm with mean subtraction (for BERT, GPT-2, ViT).
/// Unlike RMSNorm, this subtracts the mean and adds learnable bias.
pub fn layernorm(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    gamma: &GpuTensor<f32>,
    beta: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    hidden_size: u32,
    eps: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, LAYERNORM_SRC, "warp_layernorm")?;
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
            .arg(&beta.data)
            .arg(&hidden_size)
            .arg(&eps)
            .arg(&n_rows)
            .launch(cfg))?;
    }
    Ok(())
}

/// Fused RMSNorm → F16 output (eliminates separate cast kernel).
pub fn rmsnorm_f16out(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    gamma: &GpuTensor<f32>,
    out: &mut GpuTensor<half::f16>,
    hidden_size: u32,
    eps: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, RMSNORM_F16OUT_SRC, "warp_rmsnorm_f16out")?;
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

/// Cached RMSNorm.
pub fn rmsnorm(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    gamma: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    hidden_size: u32,
    eps: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, RMSNORM_SRC, "warp_rmsnorm")?;
    let n_rows = (x.numel / hidden_size as usize) as usize;
    let cfg = LaunchConfig {
        grid_dim: (n_rows as u32, 1, 1),
        block_dim: (32.min(hidden_size), 1, 1), // one warp per row
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

// ── GEMM ─────────────────────────────────────────────────────────

const GEMM_TILED_SRC: &str = r#"
#define TILE 32

extern "C" __global__ void warp_gemm_tiled(
    float *C, const float *A, const float *B,
    unsigned int M, unsigned int N, unsigned int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int row = by * TILE + ty;
    unsigned int col = bx * TILE + tx;

    float sum = 0.0f;

    for (unsigned int t = 0; t < (K + TILE - 1) / TILE; t++) {
        As[ty][tx] = (row < M && t * TILE + tx < K) ? A[row * K + t * TILE + tx] : 0.0f;
        Bs[ty][tx] = (t * TILE + ty < K && col < N) ? B[(t * TILE + ty) * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (unsigned int i = 0; i < TILE; i++)
            sum += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
"#;

/// Cached GEMM: C = A × B
/// Automatically selects the best kernel:
///   - Fast register-tiled (128×128 blocks) for sizes >= 128
///   - Simple tiled (32×32) for small sizes
pub fn gemm(
    _cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    c: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    // Always use cuBLAS — it's faster than our custom kernels at ALL sizes.
    // cuBLAS launch overhead (~5-10μs) is negligible vs our NVRTC kernels.
    match crate::cublas_gemm::gemm_cublas_f32(device, a, b, c, m, n, k) {
        Ok(()) => return Ok(()),
        Err(e) => {
            log::warn!("cuBLAS GEMM failed ({}x{}x{}): {}. Falling back.", m, n, k, e);
        }
    }

    // Custom NVRTC kernels — faster than cuBLAS for small matrices due to lower overhead
    let cache = _cache;
    if m >= 128 && n >= 128 {
        return crate::gemm_fast::gemm_fast(cache, device, a, b, c, m, n, k);
    }
    if m >= 64 && n >= 64 {
        return crate::gemm_v2::gemm_v2_med(cache, device, a, b, c, m, n, k);
    }

    // Fall back to simple tiled for small sizes
    let tile = 32u32;
    let f = cache.get_or_compile(device, GEMM_TILED_SRC, "warp_gemm_tiled")?;

    let cfg = LaunchConfig {
        grid_dim: ((n + tile - 1) / tile, (m + tile - 1) / tile, 1),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut c.data)
            .arg(&a.data)
            .arg(&b.data)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg))?;
    }
    Ok(())
}

// ── M=1 F32 GEMM with Split-K ──────────────────────────────────
//
// Specialized for M=1 (single-token decode / LM head projection).
// One thread per output column, full K dot product per thread.
// Split-K for SM occupancy on large N (vocab projection).
// Graph-capturable — no cuBLAS dependency.
//
// B is stored row-major [K, N]: element (k, n) at B[k * N + n].
// Adjacent threads read adjacent columns — coalesced row reads.

/// M=1 F32 GEMM — direct write (no atomicAdd, no memset needed). Graph-capturable.
const GEMM_M1_F32_SRC: &str = r#"
extern "C" __global__ void warp_gemm_m1_f32(
    float* __restrict__ C,
    const float* __restrict__ A,       // [1, K]
    const float* __restrict__ B,       // [K, N] row-major
    int K,
    int N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float dot = 0.0f;

    // Process 4 elements at a time for ILP
    int k = 0;
    for (; k + 3 < K; k += 4) {
        float a0 = __ldg(&A[k]);
        float a1 = __ldg(&A[k + 1]);
        float a2 = __ldg(&A[k + 2]);
        float a3 = __ldg(&A[k + 3]);
        dot += a0 * B[k * N + n];
        dot += a1 * B[(k + 1) * N + n];
        dot += a2 * B[(k + 2) * N + n];
        dot += a3 * B[(k + 3) * N + n];
    }
    for (; k < K; k++) {
        dot += __ldg(&A[k]) * B[k * N + n];
    }

    C[n] = dot;
}
"#;

/// M=1 F32 GEMM with Split-K (for small N needing more SM occupancy).
const GEMM_M1_F32_SPLITK_SRC: &str = r#"
extern "C" __global__ void warp_gemm_m1_f32_splitk(
    float* __restrict__ C,
    const float* __restrict__ A,       // [1, K]
    const float* __restrict__ B,       // [K, N] row-major
    int K,
    int N,
    int k_per_split
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    int k_split = blockIdx.y;
    int k_start = k_split * k_per_split;
    int k_end = k_start + k_per_split;
    if (k_end > K) k_end = K;

    float dot = 0.0f;

    int k = k_start;
    for (; k + 3 < k_end; k += 4) {
        float a0 = __ldg(&A[k]);
        float a1 = __ldg(&A[k + 1]);
        float a2 = __ldg(&A[k + 2]);
        float a3 = __ldg(&A[k + 3]);
        dot += a0 * B[k * N + n];
        dot += a1 * B[(k + 1) * N + n];
        dot += a2 * B[(k + 2) * N + n];
        dot += a3 * B[(k + 3) * N + n];
    }
    for (; k < k_end; k++) {
        dot += __ldg(&A[k]) * B[k * N + n];
    }

    atomicAdd(&C[n], dot);
}
"#;

/// M=1 F32 GEMM: C[1,N] = A[1,K] × B[K,N] — graph-capturable, no cuBLAS.
///
/// Specialized for the LM head vocab projection (M=1, N=vocab, K=hidden).
/// Uses Split-K for SM occupancy on large N.
pub fn gemm_m1_f32(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,       // [1, K]
    b: &GpuTensor<f32>,       // [K, N] row-major
    c: &mut GpuTensor<f32>,   // [1, N]
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    let threads = 256u32;
    let n_blocks = (n + threads - 1) / threads;

    // Adaptive Split-K (same logic as Q4 M=1)
    let splits = if k < 2048 || n_blocks >= 128 {
        1
    } else {
        let target = 256u32;
        let max_splits = (k / 256).max(1);
        ((target + n_blocks - 1) / n_blocks).max(1).min(max_splits)
    };

    let k_i = k as i32;
    let n_i = n as i32;

    if splits == 1 {
        // Direct write — no memset, no atomicAdd. Fully graph-capturable.
        let f = cache.get_or_compile(device, GEMM_M1_F32_SRC, "warp_gemm_m1_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut c.data)
                .arg(&a.data)
                .arg(&b.data)
                .arg(&k_i)
                .arg(&n_i)
                .launch(cfg))?;
        }
    } else {
        // Split-K with atomicAdd (needs memset)
        let k_per_split = ((k + splits - 1) / splits) as i32;
        let f = cache.get_or_compile(device, GEMM_M1_F32_SPLITK_SRC, "warp_gemm_m1_f32_splitk")?;
        device.stream.memset_zeros(&mut c.data)
            .map_err(|e| DeviceError::Memory(format!("memset zeros: {e}")))?;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, splits, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut c.data)
                .arg(&a.data)
                .arg(&b.data)
                .arg(&k_i)
                .arg(&n_i)
                .arg(&k_per_split)
                .launch(cfg))?;
        }
    }
    Ok(())
}

/// Simple tiled GEMM (32×32 tiles). Used by autotuner for comparison.
pub fn gemm_tiled32(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    c: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    let tile = 32u32;
    let f = cache.get_or_compile(device, GEMM_TILED_SRC, "warp_gemm_tiled")?;
    let cfg = LaunchConfig {
        grid_dim: ((n + tile - 1) / tile, (m + tile - 1) / tile, 1),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut c.data).arg(&a.data).arg(&b.data)
            .arg(&m).arg(&n).arg(&k)
            .launch(cfg))?;
    }
    Ok(())
}

/// Fused GEMM + Bias + GELU: out = GELU(A @ B + bias)
/// Single kernel — cuBLAS cannot do this.
pub fn fused_gemm_bias_gelu(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    bias: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    crate::gemm_tc::fused_gemm_bias_gelu(cache, device, a, b, bias, out, m, n, k)
}

/// Fused GEMM + Bias + SiLU: out = SiLU(A @ B + bias)
/// For SwiGLU gate projection.
pub fn fused_gemm_bias_silu(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    bias: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    crate::gemm_tc::fused_gemm_bias_silu(cache, device, a, b, bias, out, m, n, k)
}

// ── InstanceNorm ────────────────────────────────────────────────
// Like GroupNorm with num_groups = C (each channel normalized independently).
// y = ((x - mean) / sqrt(var + eps)) * scale + bias
// Mean/var computed per channel across spatial dims.

const INSTANCENORM_SRC: &str = r#"
extern "C" __global__ void warp_instancenorm(
    float *out,
    const float *x,         // [C, spatial]
    const float *scale,     // [C]
    const float *bias,      // [C]
    unsigned int C,
    unsigned int spatial,    // H * W
    float eps
) {
    unsigned int c = blockIdx.x;
    if (c >= C) return;

    const float *x_ch = x + c * spatial;
    float *out_ch = out + c * spatial;

    // Compute mean for this channel
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < spatial; i += blockDim.x) {
        sum += x_ch[i];
    }

    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    sum = __shfl_sync(0xffffffff, sum, 0);
    float mean = sum / (float)spatial;

    // Variance
    float var_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < spatial; i += blockDim.x) {
        float diff = x_ch[i] - mean;
        var_sum += diff * diff;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    var_sum = __shfl_sync(0xffffffff, var_sum, 0);
    float inv_std = rsqrtf(var_sum / (float)spatial + eps);

    // Normalize
    float s = scale[c];
    float b = bias[c];
    for (unsigned int i = threadIdx.x; i < spatial; i += blockDim.x) {
        out_ch[i] = (x_ch[i] - mean) * inv_std * s + b;
    }
}
"#;

/// Instance normalization. Each channel normalized independently (like GroupNorm with groups=C).
/// Used by style transfer networks, pix2pix, CycleGAN.
pub fn instancenorm(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    scale: &GpuTensor<f32>,
    bias: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    channels: u32,
    spatial: u32,
    eps: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, INSTANCENORM_SRC, "warp_instancenorm")?;
    let batch = x.numel as u32 / (channels * spatial);

    for n in 0..batch {
        let off = (n * channels) as usize * spatial as usize;
        let cfg = LaunchConfig {
            grid_dim: (channels, 1, 1),
            block_dim: (32, 1, 1), // one warp per channel
            shared_mem_bytes: 0,
        };
        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut out.data.slice_mut(off..))
                .arg(&x.data.slice(off..))
                .arg(&scale.data)
                .arg(&bias.data)
                .arg(&channels)
                .arg(&spatial)
                .arg(&eps)
                .launch(cfg))?;
        }
    }
    Ok(())
}

// ── Broadcast Copy ─────────────────────────────────────────────
// Fills a [rows, cols] matrix by broadcasting a [cols] bias vector across rows.

const BROADCAST_COPY_SRC: &str = r#"
extern "C" __global__ void warp_broadcast_copy(
    float *out, const float *bias, unsigned int rows, unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        out[idx] = bias[idx % cols];
    }
}
"#;

/// Broadcast a 1D bias [N] into a 2D tensor [M, N] by repeating per row.
pub fn broadcast_copy(
    cache: &KernelCache,
    device: &WarpDevice,
    bias: &GpuTensor<f32>,   // [N]
    out: &mut GpuTensor<f32>, // [M, N]
    m: u32, n: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, BROADCAST_COPY_SRC, "warp_broadcast_copy")?;
    let total = m * n;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&bias.data).arg(&m).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// GEMM + bias fusion: C = A @ B + bias (single cuBLAS launch + 1 cheap copy).
///
/// Pre-loads bias into C via broadcast_copy, then runs cuBLAS with beta=1.0.
/// Eliminates the separate broadcast_add kernel launch.
pub fn gemm_bias(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    bias: &GpuTensor<f32>,
    c: &mut GpuTensor<f32>,
    m: u32, n: u32, k: u32,
) -> Result<(), DeviceError> {
    // Broadcast bias into c: each row gets a copy of bias[0..N]
    broadcast_copy(cache, device, bias, c, m, n)?;
    // GEMM with beta=1.0: C = 1.0*A@B + 1.0*C = A@B + bias
    crate::cublas_gemm::gemm_cublas_f32_add(device, a, b, c, m, n, k)?;
    Ok(())
}

// ── Fused QKV / Gate+Up Split Kernels ─────────────────────────
// Split a fused [batch, total] output back into separate tensors on GPU.
// Avoids GPU→CPU→GPU roundtrip.

const SPLIT_QKV_SRC: &str = r#"
extern "C" __global__ void warp_split_qkv(
    float *q_out, float *k_out, float *v_out,
    const float *qkv, unsigned int h, unsigned int kv_dim, unsigned int batch
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_per_row = h + kv_dim + kv_dim;
    unsigned int total = batch * total_per_row;
    if (idx >= total) return;

    unsigned int b = idx / total_per_row;
    unsigned int d = idx % total_per_row;

    if (d < h) {
        q_out[b * h + d] = qkv[idx];
    } else if (d < h + kv_dim) {
        k_out[b * kv_dim + (d - h)] = qkv[idx];
    } else {
        v_out[b * kv_dim + (d - h - kv_dim)] = qkv[idx];
    }
}
"#;

const SPLIT_GATE_UP_SRC: &str = r#"
extern "C" __global__ void warp_split_gate_up(
    float *gate_out, float *up_out,
    const float *gate_up, unsigned int ffn_dim, unsigned int batch
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_per_row = ffn_dim + ffn_dim;
    unsigned int total = batch * total_per_row;
    if (idx >= total) return;

    unsigned int b = idx / total_per_row;
    unsigned int d = idx % total_per_row;

    if (d < ffn_dim) {
        gate_out[b * ffn_dim + d] = gate_up[idx];
    } else {
        up_out[b * ffn_dim + (d - ffn_dim)] = gate_up[idx];
    }
}
"#;

/// Split a fused QKV tensor [batch, h+kv_dim+kv_dim] into Q, K, V on GPU.
pub fn split_qkv(
    cache: &KernelCache,
    device: &WarpDevice,
    qkv: &GpuTensor<f32>,
    q: &mut GpuTensor<f32>,
    k: &mut GpuTensor<f32>,
    v: &mut GpuTensor<f32>,
    h: u32,
    kv_dim: u32,
    batch: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, SPLIT_QKV_SRC, "warp_split_qkv")?;
    let total = batch * (h + kv_dim + kv_dim);
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut q.data).arg(&mut k.data).arg(&mut v.data)
            .arg(&qkv.data).arg(&h).arg(&kv_dim).arg(&batch)
            .launch(cfg))?;
    }
    Ok(())
}

/// Split a fused gate+up tensor [batch, 2*ffn_dim] into gate and up on GPU.
pub fn split_gate_up(
    cache: &KernelCache,
    device: &WarpDevice,
    gate_up: &GpuTensor<f32>,
    gate: &mut GpuTensor<f32>,
    up: &mut GpuTensor<f32>,
    ffn_dim: u32,
    batch: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, SPLIT_GATE_UP_SRC, "warp_split_gate_up")?;
    let total = batch * ffn_dim * 2;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut gate.data).arg(&mut up.data)
            .arg(&gate_up.data).arg(&ffn_dim).arg(&batch)
            .launch(cfg))?;
    }
    Ok(())
}

/// Concatenate 2 or 3 weight matrices along the N (column) dimension on CPU,
/// then upload the fused matrix to GPU. Runs once at model load time.
///
/// All inputs must have shape [K, N_i]. Output has shape [K, N_total].
pub fn concat_weights_n(
    device: &WarpDevice,
    matrices: &[&GpuTensor<f32>],
) -> Result<GpuTensor<f32>, DeviceError> {
    assert!(matrices.len() >= 2, "Need at least 2 matrices to concatenate");

    let k = matrices[0].shape.dim(0).static_val().expect("dim 0 must be static");
    let ns: Vec<usize> = matrices.iter().map(|m| m.shape.dim(1).static_val().expect("dim 1 must be static")).collect();
    let n_total: usize = ns.iter().sum();

    // Download all matrices to host
    let host_data: Vec<Vec<f32>> = matrices
        .iter()
        .map(|m| m.to_host(device))
        .collect::<Result<Vec<_>, _>>()?;

    // Interleave rows: for each row, copy columns from each matrix
    let mut fused = vec![0.0f32; k * n_total];
    for row in 0..k {
        let mut col_offset = 0;
        for (mat_idx, n_i) in ns.iter().enumerate() {
            fused[row * n_total + col_offset..row * n_total + col_offset + n_i]
                .copy_from_slice(&host_data[mat_idx][row * n_i..(row + 1) * n_i]);
            col_offset += n_i;
        }
    }

    GpuTensor::from_host(
        device,
        &fused,
        warp_ir::Shape::from_static(&[k, n_total]),
        warp_ir::DType::F32,
    )
}

/// Concatenate bias vectors (1D) on CPU, upload to GPU. Runs once at load time.
pub fn concat_biases(
    device: &WarpDevice,
    biases: &[&GpuTensor<f32>],
) -> Result<GpuTensor<f32>, DeviceError> {
    let mut total_len = 0usize;
    let mut host_parts: Vec<Vec<f32>> = Vec::new();
    for b in biases {
        let data = b.to_host(device)?;
        total_len += data.len();
        host_parts.push(data);
    }
    let mut fused = Vec::with_capacity(total_len);
    for part in &host_parts {
        fused.extend_from_slice(part);
    }
    GpuTensor::from_host(
        device,
        &fused,
        warp_ir::Shape::from_static(&[total_len]),
        warp_ir::DType::F32,
    )
}

// ── Layout Transpose Kernels ───────────────────────────────────
// Transpose between positions-first [S, H*D] and heads-first [H, S, D]
// layouts entirely on GPU — eliminates catastrophic CPU roundtrips.

const TRANSPOSE_SHD_TO_HSD_SRC: &str = r#"
extern "C" __global__ void warp_transpose_shd_hsd(
    float *out,       // [H, S, D] (heads-first)
    const float *in_data,  // [S, H*D] (positions-first)
    unsigned int H, unsigned int S, unsigned int D
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = H * S * D;
    if (idx >= total) return;

    // Decompose flat index as positions-first [S, H*D]
    unsigned int d = idx % D;
    unsigned int h = (idx / D) % H;
    unsigned int s = idx / (H * D);

    // Write to heads-first [H, S, D]
    out[h * S * D + s * D + d] = in_data[idx];
}
"#;

const TRANSPOSE_HSD_TO_SHD_SRC: &str = r#"
extern "C" __global__ void warp_transpose_hsd_shd(
    float *out,       // [S, H*D] (positions-first)
    const float *in_data,  // [H, S, D] (heads-first)
    unsigned int H, unsigned int S, unsigned int D
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = H * S * D;
    if (idx >= total) return;

    // Decompose flat index as heads-first [H, S, D]
    unsigned int d = idx % D;
    unsigned int s = (idx / D) % S;
    unsigned int h = idx / (S * D);

    // Write to positions-first [S, H*D]
    out[s * H * D + h * D + d] = in_data[idx];
}
"#;

/// Transpose [S, H*D] (positions-first) → [H, S, D] (heads-first) on GPU.
pub fn transpose_to_heads_first(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,   // [S, H*D]
    output: &mut GpuTensor<f32>, // [H, S, D]
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, TRANSPOSE_SHD_TO_HSD_SRC, "warp_transpose_shd_hsd")?;
    let total = num_heads * seq_len * head_dim;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&input.data)
            .arg(&num_heads).arg(&seq_len).arg(&head_dim)
            .launch(cfg))?;
    }
    Ok(())
}

/// Transpose [H, S, D] (heads-first) → [S, H*D] (positions-first) on GPU.
pub fn transpose_to_positions_first(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,   // [H, S, D]
    output: &mut GpuTensor<f32>, // [S, H*D]
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, TRANSPOSE_HSD_TO_SHD_SRC, "warp_transpose_hsd_shd")?;
    let total = num_heads * seq_len * head_dim;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&input.data)
            .arg(&num_heads).arg(&seq_len).arg(&head_dim)
            .launch(cfg))?;
    }
    Ok(())
}

// ── Fused Bias + RoPE + KV Cache Append ──────────────────────
// Replaces 6 separate launches (3 bias_add + 2 RoPE + 1 KV append) with 1.
// For decode (single token), processes Q bias+RoPE, K bias+RoPE+cache_append,
// and V bias+cache_append in a single kernel.
//
// Uses rotate_half pairing: (dim i, dim i+D/2), matching the existing RoPE kernel.
const FUSED_BIAS_ROPE_APPEND_SRC: &str = r#"
extern "C" __global__ void warp_fused_bias_rope_append(
    float *q_out,            // [num_heads * head_dim] — Q with bias + RoPE applied
    float *k_out,            // [kv_dim] — K with bias + RoPE applied
    float *k_cache,          // [max_seq, kv_dim]
    float *v_cache,          // [max_seq, kv_dim]
    const float *q_in,       // [num_heads * head_dim] — raw Q from GEMM
    const float *k_in,       // [kv_dim] — raw K from GEMM
    const float *v_in,       // [kv_dim] — raw V from GEMM
    const float *bq,         // Q bias [num_heads * head_dim] (ignored if has_bias == 0)
    const float *bk,         // K bias [kv_dim]
    const float *bv,         // V bias [kv_dim]
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int kv_dim,
    unsigned int pos,
    unsigned int max_seq,
    float rope_base,
    unsigned int has_bias     // 1 = apply biases, 0 = skip bias pointers
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int q_total = num_heads * head_dim;
    unsigned int half_d = head_dim / 2;

    // ── Q: bias + RoPE (rotate_half pairing) ──
    if (idx < q_total) {
        unsigned int head = idx / head_dim;
        unsigned int d = idx % head_dim;
        unsigned int pair = d % half_d;  // dimension pair index

        // Compute RoPE angle
        float freq = 1.0f / powf(rope_base, 2.0f * (float)pair / (float)head_dim);
        float theta = (float)pos * freq;
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        unsigned int base_off = head * head_dim;
        // Load both halves of the pair, applying bias
        float x0 = q_in[base_off + pair];
        float x1 = q_in[base_off + pair + half_d];
        if (has_bias) {
            x0 += bq[base_off + pair];
            x1 += bq[base_off + pair + half_d];
        }

        if (d < half_d) {
            q_out[idx] = x0 * cos_t - x1 * sin_t;
        } else {
            q_out[idx] = x0 * sin_t + x1 * cos_t;
        }
    }

    // ── K: bias + RoPE + cache append | V: bias + cache append ──
    if (idx < kv_dim) {
        unsigned int kv_head = idx / head_dim;
        unsigned int d = idx % head_dim;
        unsigned int pair = d % half_d;

        // K: bias + RoPE
        float freq = 1.0f / powf(rope_base, 2.0f * (float)pair / (float)head_dim);
        float theta = (float)pos * freq;
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        unsigned int kv_base = kv_head * head_dim;
        float k0 = k_in[kv_base + pair];
        float k1 = k_in[kv_base + pair + half_d];
        if (has_bias) {
            k0 += bk[kv_base + pair];
            k1 += bk[kv_base + pair + half_d];
        }

        float k_roped;
        if (d < half_d) {
            k_roped = k0 * cos_t - k1 * sin_t;
        } else {
            k_roped = k0 * sin_t + k1 * cos_t;
        }
        k_out[idx] = k_roped;

        // Append K to cache
        if (pos < max_seq) {
            k_cache[pos * kv_dim + idx] = k_roped;
        }

        // V: bias + cache append (no RoPE needed for V)
        float vval = v_in[idx];
        if (has_bias) {
            vval += bv[idx];
        }
        if (pos < max_seq) {
            v_cache[pos * kv_dim + idx] = vval;
        }
    }
}
"#;

/// Device-pos variant: reads pos from a device pointer for CUDA graph compatibility.
const FUSED_BIAS_ROPE_APPEND_DEVICE_POS_SRC: &str = r#"
extern "C" __global__ void warp_fused_bias_rope_append_dp(
    float *q_out, float *k_out,
    float *k_cache, float *v_cache,
    const float *q_in, const float *k_in, const float *v_in,
    const float *bq, const float *bk, const float *bv,
    unsigned int num_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int kv_dim,
    const unsigned int *pos_buf,  // device pointer to pos
    unsigned int max_seq, float rope_base, unsigned int has_bias
) {
    unsigned int pos = pos_buf[0];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int q_total = num_heads * head_dim;
    unsigned int half_d = head_dim / 2;

    if (idx < q_total) {
        unsigned int head = idx / head_dim;
        unsigned int d = idx % head_dim;
        unsigned int pair = d % half_d;
        float freq = 1.0f / powf(rope_base, 2.0f * (float)pair / (float)head_dim);
        float theta = (float)pos * freq;
        float cos_t = cosf(theta); float sin_t = sinf(theta);
        unsigned int base_off = head * head_dim;
        float x0 = q_in[base_off + pair]; float x1 = q_in[base_off + pair + half_d];
        if (has_bias) { x0 += bq[base_off + pair]; x1 += bq[base_off + pair + half_d]; }
        if (d < half_d) q_out[idx] = x0 * cos_t - x1 * sin_t;
        else q_out[idx] = x0 * sin_t + x1 * cos_t;
    }

    if (idx < kv_dim) {
        unsigned int kv_head = idx / head_dim;
        unsigned int d = idx % head_dim;
        unsigned int pair = d % half_d;
        float freq = 1.0f / powf(rope_base, 2.0f * (float)pair / (float)head_dim);
        float theta = (float)pos * freq;
        float cos_t = cosf(theta); float sin_t = sinf(theta);
        unsigned int kv_base = kv_head * head_dim;
        float k0 = k_in[kv_base + pair]; float k1 = k_in[kv_base + pair + half_d];
        if (has_bias) { k0 += bk[kv_base + pair]; k1 += bk[kv_base + pair + half_d]; }
        float k_roped;
        if (d < half_d) k_roped = k0 * cos_t - k1 * sin_t;
        else k_roped = k0 * sin_t + k1 * cos_t;
        k_out[idx] = k_roped;
        if (pos < max_seq) k_cache[pos * kv_dim + idx] = k_roped;
        float vval = v_in[idx];
        if (has_bias) vval += bv[idx];
        if (pos < max_seq) v_cache[pos * kv_dim + idx] = vval;
    }
}
"#;

/// Fused bias + RoPE + KV cache append with device-side pos pointer.
/// For CUDA graph capture: pos is read from device memory, not kernel arg.
pub fn fused_bias_rope_append_device_pos(
    cache: &KernelCache,
    device: &WarpDevice,
    q_in: &GpuTensor<f32>, k_in: &GpuTensor<f32>, v_in: &GpuTensor<f32>,
    q_out: &mut GpuTensor<f32>, k_out: &mut GpuTensor<f32>,
    bq: &GpuTensor<f32>, bk: &GpuTensor<f32>, bv: &GpuTensor<f32>,
    k_cache: &mut GpuTensor<f32>, v_cache: &mut GpuTensor<f32>,
    num_heads: u32, num_kv_heads: u32, head_dim: u32, kv_dim: u32,
    pos_buf: &cudarc::driver::CudaSlice<u32>,
    max_seq: u32, rope_base: f32, has_bias: bool,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, FUSED_BIAS_ROPE_APPEND_DEVICE_POS_SRC,
        "warp_fused_bias_rope_append_dp")?;
    let q_total = num_heads * head_dim;
    let total = q_total.max(kv_dim);
    let cfg = LaunchConfig::for_num_elems(total);
    let has_bias_flag: u32 = if has_bias { 1 } else { 0 };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut q_out.data).arg(&mut k_out.data)
            .arg(&mut k_cache.data).arg(&mut v_cache.data)
            .arg(&q_in.data).arg(&k_in.data).arg(&v_in.data)
            .arg(&bq.data).arg(&bk.data).arg(&bv.data)
            .arg(&num_heads).arg(&num_kv_heads)
            .arg(&head_dim).arg(&kv_dim)
            .arg(pos_buf)
            .arg(&max_seq).arg(&rope_base).arg(&has_bias_flag)
            .launch(cfg))?;
    }
    Ok(())
}

/// Fused bias + RoPE + KV cache append.
///
/// Replaces 6 separate kernel launches with 1:
///   3x broadcast_add (Q/K/V biases) + 2x RoPE (Q/K) + 1x KV append
///
/// For models without biases (e.g. LLaMA), pass `has_bias=false` and the
/// bias pointers are ignored (pass any valid buffer — they won't be read).
///
/// The KV cache position is updated externally (caller must increment
/// `LayerKVCache::len` after this call).
pub fn fused_bias_rope_append(
    cache: &KernelCache,
    device: &WarpDevice,
    q_in: &GpuTensor<f32>,        // [num_heads * head_dim]
    k_in: &GpuTensor<f32>,        // [kv_dim]
    v_in: &GpuTensor<f32>,        // [kv_dim]
    q_out: &mut GpuTensor<f32>,   // [num_heads * head_dim]
    k_out: &mut GpuTensor<f32>,   // [kv_dim]
    bq: &GpuTensor<f32>,          // Q bias (ignored if !has_bias)
    bk: &GpuTensor<f32>,          // K bias
    bv: &GpuTensor<f32>,          // V bias
    k_cache: &mut GpuTensor<f32>, // [max_seq, kv_dim]
    v_cache: &mut GpuTensor<f32>, // [max_seq, kv_dim]
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    pos: u32,
    max_seq: u32,
    rope_base: f32,
    has_bias: bool,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, FUSED_BIAS_ROPE_APPEND_SRC, "warp_fused_bias_rope_append")?;
    // Thread count = max(q_total, kv_dim) — kernel handles both ranges
    let q_total = num_heads * head_dim;
    let total = q_total.max(kv_dim);
    let cfg = LaunchConfig::for_num_elems(total);
    let has_bias_flag: u32 = if has_bias { 1 } else { 0 };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut q_out.data)
            .arg(&mut k_out.data)
            .arg(&mut k_cache.data)
            .arg(&mut v_cache.data)
            .arg(&q_in.data)
            .arg(&k_in.data)
            .arg(&v_in.data)
            .arg(&bq.data)
            .arg(&bk.data)
            .arg(&bv.data)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&head_dim)
            .arg(&kv_dim)
            .arg(&pos)
            .arg(&max_seq)
            .arg(&rope_base)
            .arg(&has_bias_flag)
            .launch(cfg))?;
    }
    Ok(())
}

// ── Weight layout reorder: column-major-blocks → group-major-blocks ��────────
// ── Fused MoE combine: 6 kernels → 1 ─────────────────────────────────────────
// Computes: output_scaled = (residual + rmsnorm(rmsnorm(dense,g1) + rmsnorm(moe,g2), g3)) * scalar

const FUSED_MOE_COMBINE_SRC: &str = r#"
extern "C" __global__ void fused_moe_combine(
    float* __restrict__ output_scaled,
    float* __restrict__ output,
    const float* __restrict__ dense_out,
    const float* __restrict__ moe_acc,
    const float* __restrict__ residual,
    const float* __restrict__ gamma1,
    const float* __restrict__ gamma2,
    const float* __restrict__ gamma3,
    unsigned int H, float eps, float layer_scalar
) {
    // Phase 1: sum_sq for both dense and moe RMSNorms simultaneously
    float sum_sq_d = 0.0f, sum_sq_m = 0.0f;
    for (unsigned int i = threadIdx.x; i < H; i += blockDim.x) {
        float d = dense_out[i], m = moe_acc[i];
        sum_sq_d += d * d;
        sum_sq_m += m * m;
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum_sq_d += __shfl_down_sync(0xFFFFFFFF, sum_sq_d, off);
        sum_sq_m += __shfl_down_sync(0xFFFFFFFF, sum_sq_m, off);
    }
    sum_sq_d = __shfl_sync(0xFFFFFFFF, sum_sq_d, 0);
    sum_sq_m = __shfl_sync(0xFFFFFFFF, sum_sq_m, 0);
    float rms_d = rsqrtf(sum_sq_d / (float)H + eps);
    float rms_m = rsqrtf(sum_sq_m / (float)H + eps);

    // Phase 2: combined = rmsnorm(dense)*g1 + rmsnorm(moe)*g2, accumulate sum_sq
    float sum_sq_c = 0.0f;
    for (unsigned int i = threadIdx.x; i < H; i += blockDim.x) {
        float c = dense_out[i] * rms_d * gamma1[i] + moe_acc[i] * rms_m * gamma2[i];
        output[i] = c;  // temp store
        sum_sq_c += c * c;
    }
    for (int off = 16; off > 0; off >>= 1)
        sum_sq_c += __shfl_down_sync(0xFFFFFFFF, sum_sq_c, off);
    sum_sq_c = __shfl_sync(0xFFFFFFFF, sum_sq_c, 0);
    float rms_c = rsqrtf(sum_sq_c / (float)H + eps);

    // Phase 3: output = residual + rmsnorm(combined)*g3, scaled
    for (unsigned int i = threadIdx.x; i < H; i += blockDim.x) {
        float out = residual[i] + output[i] * rms_c * gamma3[i];
        output[i] = out;
        output_scaled[i] = out * layer_scalar;
    }
}
"#;

pub fn fused_moe_combine(
    cache: &KernelCache,
    device: &WarpDevice,
    output_scaled: &mut GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
    dense_out: &GpuTensor<f32>,
    moe_acc: &GpuTensor<f32>,
    residual: &GpuTensor<f32>,
    gamma1: &GpuTensor<f32>,
    gamma2: &GpuTensor<f32>,
    gamma3: &GpuTensor<f32>,
    hidden_size: u32,
    eps: f32,
    layer_scalar: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, FUSED_MOE_COMBINE_SRC, "fused_moe_combine")?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32.min(hidden_size), 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output_scaled.data).arg(&mut output.data)
            .arg(&dense_out.data).arg(&moe_acc.data).arg(&residual.data)
            .arg(&gamma1.data).arg(&gamma2.data).arg(&gamma3.data)
            .arg(&hidden_size).arg(&eps).arg(&layer_scalar)
            .launch(cfg))?;
    }
    Ok(())
}

// Converts Q4_K/Q8_0 expert weights from (n, kb) to (kb, n) layout for
// coalesced memory access. Consecutive warp threads then hit adjacent blocks
// instead of blocks N*block_bytes apart.

const REORDER_BLOCKS_SRC: &str = r#"
extern "C" __global__ void reorder_blocks(
    unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    int N, int num_k_blocks, int block_bytes,
    int num_experts, int expert_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blocks_per_expert = N * num_k_blocks;
    int total = num_experts * blocks_per_expert;
    if (idx >= total) return;

    int expert = idx / blocks_per_expert;
    int local = idx % blocks_per_expert;
    int n = local / num_k_blocks;
    int kb = local % num_k_blocks;

    long long base = (long long)expert * expert_bytes;
    long long old_off = base + (long long)(n * num_k_blocks + kb) * block_bytes;
    long long new_off = base + (long long)(kb * N + n) * block_bytes;

    // Copy one block
    for (int b = 0; b < block_bytes; b++) {
        dst[new_off + b] = src[old_off + b];
    }
}
"#;

/// Reorder expert weight blocks from column-major (n, kb) to group-major (kb, n).
/// Handles multiple experts concatenated in the buffer.
/// Runs once at engine startup. Returns the reordered buffer.
pub fn reorder_expert_blocks(
    cache: &KernelCache,
    device: &WarpDevice,
    src: &GpuTensor<u8>,
    n: u32,            // output columns per expert
    num_k_blocks: u32, // K / block_elements per expert
    block_bytes: u32,  // 144 for Q4_K, 34 for Q8_0, 22 for Q5_0
) -> Result<GpuTensor<u8>, DeviceError> {
    let expert_blocks = n * num_k_blocks;
    let expert_bytes = (expert_blocks * block_bytes) as usize;
    let num_experts = src.numel / expert_bytes;
    let total_blocks = num_experts as u32 * expert_blocks;

    let dst_data = device.alloc_zeros::<u8>(src.numel)?;
    let mut dst = GpuTensor {
        data: dst_data,
        shape: src.shape.clone(),
        dtype: src.dtype,
        numel: src.numel,
    };

    let f = cache.get_or_compile(device, REORDER_BLOCKS_SRC, "reorder_blocks")?;
    let threads = 256u32;
    let grid = (total_blocks + threads - 1) / threads;
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut dst.data).arg(&src.data)
            .arg(&(n as i32)).arg(&(num_k_blocks as i32)).arg(&(block_bytes as i32))
            .arg(&(num_experts as i32)).arg(&(expert_bytes as i32))
            .launch(cfg))?;
    }
    device.synchronize()?;
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::{DType, Shape};

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        let dev = WarpDevice::new(0).ok()?;
        let cache = KernelCache::new();
        Some((dev, cache))
    }

    #[test]
    fn cached_add_perf() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let n = 4 * 1024 * 1024;
        let shape = Shape::from_static(&[n]);
        let a_data: Vec<f32> = (0..n).map(|i| (i % 1000) as f32 * 0.001).collect();
        let b_data: Vec<f32> = (0..n).map(|i| ((i + 500) % 1000) as f32 * 0.001).collect();

        let a = GpuTensor::from_host(&dev, &a_data, shape.clone(), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        // Warmup (compiles kernel)
        add(&cache, &dev, &a, &b, &mut out).unwrap();
        dev.synchronize().unwrap();

        // Timed (uses cache)
        let iters = 1000;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            add(&cache, &dev, &a, &b, &mut out).unwrap();
        }
        dev.synchronize().unwrap();
        let elapsed = start.elapsed();

        let bytes_per_iter = n as f64 * 4.0 * 3.0;
        let bandwidth_gb_s = bytes_per_iter * iters as f64 / elapsed.as_secs_f64() / 1e9;

        println!(
            "Cached add: {}M elems × {} iters = {:.2}ms ({:.1} GB/s)",
            n / (1024 * 1024), iters,
            elapsed.as_secs_f64() * 1000.0, bandwidth_gb_s,
        );
        println!("{}", cache.stats());

        // Verify
        let result = out.to_host(&dev).unwrap();
        assert!((result[0] - (a_data[0] + b_data[0])).abs() < 1e-5);
    }

    #[test]
    fn cached_gemm_perf() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (512u32, 512u32, 512u32);
        let a_data: Vec<f32> = (0..(m*k) as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let b_data: Vec<f32> = (0..(k*n) as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();

        let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let mut c = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

        // Warmup
        gemm(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let iters = 100;
        let flops_per = 2.0 * m as f64 * n as f64 * k as f64;

        let start = std::time::Instant::now();
        for _ in 0..iters {
            gemm(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
        }
        dev.synchronize().unwrap();
        let elapsed = start.elapsed();

        let tflops = flops_per * iters as f64 / elapsed.as_secs_f64() / 1e12;
        println!(
            "Cached GEMM {m}x{n}x{k}: {:.3}ms avg, {:.3} TFLOPS",
            elapsed.as_secs_f64() * 1000.0 / iters as f64, tflops,
        );
        println!("{}", cache.stats());
    }

    #[test]
    fn cached_rmsnorm() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let hidden = 32u32;
        let rows = 4usize;
        let n = rows * hidden as usize;

        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.01).collect();
        let gamma_data: Vec<f32> = vec![1.0; hidden as usize];
        let shape = Shape::from_static(&[rows, hidden as usize]);
        let gamma_shape = Shape::from_static(&[hidden as usize]);

        let x = GpuTensor::from_host(&dev, &x_data, shape.clone(), DType::F32).unwrap();
        let gamma = GpuTensor::from_host(&dev, &gamma_data, gamma_shape, DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        rmsnorm(&cache, &dev, &x, &gamma, &mut out, hidden, 1e-6).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();

        // Verify against CPU reference
        for row in 0..rows {
            let start = row * hidden as usize;
            let end = start + hidden as usize;
            let row_data = &x_data[start..end];

            let sum_sq: f32 = row_data.iter().map(|v| v * v).sum();
            let rms = (sum_sq / hidden as f32 + 1e-6).sqrt().recip();

            for (i, &v) in row_data.iter().enumerate() {
                let expected = v * rms * gamma_data[i];
                let actual = result[start + i];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "RMSNorm mismatch at row {row} col {i}: got {actual}, expected {expected}"
                );
            }
        }
        println!("RMSNorm: {rows}x{hidden} correct!");
    }

    #[test]
    fn fusion_speedup() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let n = 4 * 1024 * 1024;
        let shape = Shape::from_static(&[n]);
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) / n as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| ((i + 1000) as f32 - n as f32 / 2.0) / n as f32).collect();

        let a = GpuTensor::from_host(&dev, &a_data, shape.clone(), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, shape.clone(), DType::F32).unwrap();
        let mut tmp = GpuTensor::<f32>::zeros(&dev, shape.clone(), DType::F32).unwrap();
        let mut out1 = GpuTensor::<f32>::zeros(&dev, shape.clone(), DType::F32).unwrap();
        let mut out2 = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        // Warmup both paths
        add(&cache, &dev, &a, &b, &mut tmp).unwrap();
        gelu(&cache, &dev, &tmp, &mut out1).unwrap();
        fused_add_gelu(&cache, &dev, &a, &b, &mut out2).unwrap();
        dev.synchronize().unwrap();

        let iters = 500;

        // Unfused: add then gelu (2 kernels, 2 memory passes)
        let start = std::time::Instant::now();
        for _ in 0..iters {
            add(&cache, &dev, &a, &b, &mut tmp).unwrap();
            gelu(&cache, &dev, &tmp, &mut out1).unwrap();
        }
        dev.synchronize().unwrap();
        let unfused = start.elapsed();

        // Fused: add+gelu (1 kernel, 1 memory pass)
        let start = std::time::Instant::now();
        for _ in 0..iters {
            fused_add_gelu(&cache, &dev, &a, &b, &mut out2).unwrap();
        }
        dev.synchronize().unwrap();
        let fused = start.elapsed();

        let speedup = unfused.as_secs_f64() / fused.as_secs_f64();
        println!(
            "Fusion speedup ({n} elements × {iters} iters):");
        println!("  Unfused (add + gelu): {:.2}ms", unfused.as_secs_f64() * 1000.0);
        println!("  Fused (add_gelu):     {:.2}ms", fused.as_secs_f64() * 1000.0);
        println!("  Speedup:              {:.2}x", speedup);

        // Verify both give same result
        let r1 = out1.to_host(&dev).unwrap();
        let r2 = out2.to_host(&dev).unwrap();
        let max_diff: f32 = r1.iter().zip(r2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "Fused vs unfused mismatch: {max_diff}");
        println!("  Numerical match:      max diff = {max_diff:.2e}");
    }
}

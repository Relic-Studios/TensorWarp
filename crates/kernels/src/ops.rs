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

    // Normalize
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = x_row[i] * rms * gamma[i];
    }
}
"#;

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
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    c: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    // Smart dispatch: best kernel per size range
    if m >= 128 && n >= 128 {
        // Large: register-tiled v1 (beats double-buffered v2 at these sizes)
        return crate::gemm_fast::gemm_fast(cache, device, a, b, c, m, n, k);
    }
    if m >= 64 && n >= 64 {
        // Medium: v2 medium-block variant
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

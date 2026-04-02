//! Convolution kernels — the foundation of CNNs.
//!
//! Implementation strategy: **im2col + GEMM**.
//!
//! Instead of implementing convolution directly (nested loops, poor memory access),
//! we rearrange input patches into columns of a matrix and use our existing fast
//! GEMM. This is what cuDNN does internally for many cases.
//!
//! For a Conv2D with input [N, C_in, H, W] and weight [C_out, C_in, kH, kW]:
//!   1. im2col: rearrange input → [N, C_in*kH*kW, out_H*out_W]
//!   2. GEMM: weight[C_out, C_in*kH*kW] × col[C_in*kH*kW, out_H*out_W] → out[C_out, out_H*out_W]
//!   3. Reshape: → [N, C_out, out_H, out_W]
//!
//! Also supports:
//! - Arbitrary stride, padding, dilation
//! - Grouped convolution (groups > 1) and depthwise conv (groups = C_in)
//! - Optional bias add
//!
//! Additional kernels: BatchNorm, MaxPool2D, AvgPool2D, GlobalAvgPool.

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

/// Compute output spatial dimensions for a conv/pool.
pub fn conv_output_size(input: u32, kernel: u32, stride: u32, padding: u32, dilation: u32) -> u32 {
    (input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
}

// ── im2col ──────────────────────────────────────────────────────
// Rearranges image patches into columns for GEMM-based convolution.

const IM2COL_SRC: &str = r#"
extern "C" __global__ void warp_im2col(
    float *col,             // [C_in * kH * kW, out_H * out_W]
    const float *input,     // [C_in, H, W] (single image from batch)
    unsigned int C_in,
    unsigned int H, unsigned int W,
    unsigned int kH, unsigned int kW,
    unsigned int out_H, unsigned int out_W,
    unsigned int stride_h, unsigned int stride_w,
    unsigned int pad_h, unsigned int pad_w,
    unsigned int dil_h, unsigned int dil_w
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = C_in * kH * kW * out_H * out_W;
    if (idx >= total) return;

    // Decompose linear index into (c_kh_kw, oh, ow)
    unsigned int ow = idx % out_W;
    unsigned int tmp = idx / out_W;
    unsigned int oh = tmp % out_H;
    unsigned int c_kh_kw = tmp / out_H;

    // Decompose c_kh_kw into (c, kh_idx, kw_idx)
    unsigned int kw_idx = c_kh_kw % kW;
    unsigned int tmp2 = c_kh_kw / kW;
    unsigned int kh_idx = tmp2 % kH;
    unsigned int c = tmp2 / kH;

    // Compute input position
    int ih = (int)(oh * stride_h + kh_idx * dil_h) - (int)pad_h;
    int iw = (int)(ow * stride_w + kw_idx * dil_w) - (int)pad_w;

    float val = 0.0f;
    if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
        val = input[c * H * W + (unsigned int)ih * W + (unsigned int)iw];
    }

    // col layout: [c_kh_kw, out_H * out_W]
    col[c_kh_kw * (out_H * out_W) + oh * out_W + ow] = val;
}
"#;

// ── Bias add ────────────────────────────────────────────────────
// Adds per-channel bias to conv output.

const BIAS_ADD_SRC: &str = r#"
extern "C" __global__ void warp_bias_add_spatial(
    float *output,          // [C_out, spatial] — modified in place
    const float *bias,      // [C_out]
    unsigned int C_out,
    unsigned int spatial     // out_H * out_W
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = C_out * spatial;
    if (idx >= total) return;

    unsigned int c = idx / spatial;
    output[idx] += bias[c];
}
"#;

// ── BatchNorm (inference) ───────────────────────────────────────
// y = (x - mean) / sqrt(var + eps) * scale + bias

const BATCHNORM_SRC: &str = r#"
extern "C" __global__ void warp_batchnorm(
    float *output,
    const float *input,     // [C, spatial]
    const float *scale,     // [C]
    const float *bias,      // [C]
    const float *mean,      // [C]
    const float *var,       // [C]
    float eps,
    unsigned int C,
    unsigned int spatial     // H * W (per channel)
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = C * spatial;
    if (idx >= total) return;

    unsigned int c = idx / spatial;
    float x = input[idx];
    float inv_std = rsqrtf(var[c] + eps);
    output[idx] = (x - mean[c]) * inv_std * scale[c] + bias[c];
}
"#;

// ── MaxPool2D ───────────────────────────────────────────────────

const MAXPOOL2D_SRC: &str = r#"
extern "C" __global__ void warp_maxpool2d(
    float *output,          // [C, out_H, out_W]
    const float *input,     // [C, H, W]
    unsigned int C,
    unsigned int H, unsigned int W,
    unsigned int out_H, unsigned int out_W,
    unsigned int kH, unsigned int kW,
    unsigned int stride_h, unsigned int stride_w,
    unsigned int pad_h, unsigned int pad_w
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = C * out_H * out_W;
    if (idx >= total) return;

    unsigned int ow = idx % out_W;
    unsigned int oh = (idx / out_W) % out_H;
    unsigned int c = idx / (out_H * out_W);

    float max_val = -1e30f;
    for (unsigned int kh = 0; kh < kH; kh++) {
        for (unsigned int kw = 0; kw < kW; kw++) {
            int ih = (int)(oh * stride_h + kh) - (int)pad_h;
            int iw = (int)(ow * stride_w + kw) - (int)pad_w;
            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                float v = input[c * H * W + (unsigned int)ih * W + (unsigned int)iw];
                if (v > max_val) max_val = v;
            }
        }
    }
    output[idx] = max_val;
}
"#;

// ── AvgPool2D ───────────────────────────────────────────────────

const AVGPOOL2D_SRC: &str = r#"
extern "C" __global__ void warp_avgpool2d(
    float *output,
    const float *input,
    unsigned int C,
    unsigned int H, unsigned int W,
    unsigned int out_H, unsigned int out_W,
    unsigned int kH, unsigned int kW,
    unsigned int stride_h, unsigned int stride_w,
    unsigned int pad_h, unsigned int pad_w
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = C * out_H * out_W;
    if (idx >= total) return;

    unsigned int ow = idx % out_W;
    unsigned int oh = (idx / out_W) % out_H;
    unsigned int c = idx / (out_H * out_W);

    float sum = 0.0f;
    unsigned int count = 0;
    for (unsigned int kh = 0; kh < kH; kh++) {
        for (unsigned int kw = 0; kw < kW; kw++) {
            int ih = (int)(oh * stride_h + kh) - (int)pad_h;
            int iw = (int)(ow * stride_w + kw) - (int)pad_w;
            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                sum += input[c * H * W + (unsigned int)ih * W + (unsigned int)iw];
                count++;
            }
        }
    }
    output[idx] = (count > 0) ? sum / (float)count : 0.0f;
}
"#;

// ── Global Average Pool ─────────────────────────────────────────
// Reduces [C, H, W] → [C, 1, 1]

const GLOBAL_AVG_POOL_SRC: &str = r#"
extern "C" __global__ void warp_global_avg_pool(
    float *output,          // [C]
    const float *input,     // [C, H, W]
    unsigned int C,
    unsigned int spatial     // H * W
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float sum = 0.0f;
    for (unsigned int i = 0; i < spatial; i++) {
        sum += input[c * spatial + i];
    }
    output[c] = sum / (float)spatial;
}
"#;

// ═════════════════════════════════════════════════════════════════
// Rust API
// ═════════════════════════════════════════════════════════════════

/// Conv2D parameters.
#[derive(Debug, Clone)]
pub struct Conv2dParams {
    pub in_channels: u32,
    pub out_channels: u32,
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub padding_h: u32,
    pub padding_w: u32,
    pub dilation_h: u32,
    pub dilation_w: u32,
    pub groups: u32,
}

impl Conv2dParams {
    pub fn new(in_c: u32, out_c: u32, kernel: u32) -> Self {
        Self {
            in_channels: in_c, out_channels: out_c,
            kernel_h: kernel, kernel_w: kernel,
            stride_h: 1, stride_w: 1,
            padding_h: 0, padding_w: 0,
            dilation_h: 1, dilation_w: 1,
            groups: 1,
        }
    }

    pub fn stride(mut self, s: u32) -> Self { self.stride_h = s; self.stride_w = s; self }
    pub fn padding(mut self, p: u32) -> Self { self.padding_h = p; self.padding_w = p; self }
    pub fn dilation(mut self, d: u32) -> Self { self.dilation_h = d; self.dilation_w = d; self }
    pub fn groups(mut self, g: u32) -> Self { self.groups = g; self }

    pub fn output_h(&self, h: u32) -> u32 {
        conv_output_size(h, self.kernel_h, self.stride_h, self.padding_h, self.dilation_h)
    }
    pub fn output_w(&self, w: u32) -> u32 {
        conv_output_size(w, self.kernel_w, self.stride_w, self.padding_w, self.dilation_w)
    }
}

/// Conv2D via im2col + GEMM.
///
/// input: [N, C_in, H, W]
/// weight: [C_out, C_in/groups, kH, kW]
/// bias: optional [C_out]
/// output: [N, C_out, out_H, out_W]
pub fn conv2d(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,    // [N, C_in, H, W]
    weight: &GpuTensor<f32>,   // [C_out, C_in/groups, kH, kW]
    bias: Option<&GpuTensor<f32>>,  // [C_out]
    output: &mut GpuTensor<f32>,    // [N, C_out, out_H, out_W]
    params: &Conv2dParams,
    h: u32,
    w: u32,
) -> Result<(), DeviceError> {
    let batch = input.numel as u32 / (params.in_channels * h * w);
    let out_h = params.output_h(h);
    let out_w = params.output_w(w);
    let c_in_per_group = params.in_channels / params.groups;
    let c_out_per_group = params.out_channels / params.groups;
    let col_size = (c_in_per_group * params.kernel_h * params.kernel_w * out_h * out_w) as usize;

    // Allocate im2col buffer
    let mut col = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[col_size]), DType::F32)?;

    let im2col_f = cache.get_or_compile(device, IM2COL_SRC, "warp_im2col")?;

    for n in 0..batch {
        for g in 0..params.groups {
            let input_offset = (n * params.in_channels + g * c_in_per_group) as usize * (h * w) as usize;
            let weight_offset = (g * c_out_per_group) as usize * (c_in_per_group * params.kernel_h * params.kernel_w) as usize;
            let output_offset = (n * params.out_channels + g * c_out_per_group) as usize * (out_h * out_w) as usize;

            // im2col for this group's channels
            let total_col = c_in_per_group * params.kernel_h * params.kernel_w * out_h * out_w;
            let cfg = LaunchConfig::for_num_elems(total_col);

            unsafe {
                launch_err!(device.stream.launch_builder(&im2col_f)
                    .arg(&mut col.data)
                    .arg(&input.data.slice(input_offset..))
                    .arg(&c_in_per_group)
                    .arg(&h).arg(&w)
                    .arg(&params.kernel_h).arg(&params.kernel_w)
                    .arg(&out_h).arg(&out_w)
                    .arg(&params.stride_h).arg(&params.stride_w)
                    .arg(&params.padding_h).arg(&params.padding_w)
                    .arg(&params.dilation_h).arg(&params.dilation_w)
                    .launch(cfg))?;
            }

            // GEMM: weight[c_out_per_group, c_in_per_group*kH*kW] × col[c_in_per_group*kH*kW, out_H*out_W]
            // → out[c_out_per_group, out_H*out_W]
            let m = c_out_per_group;
            let nn = out_h * out_w;
            let k = c_in_per_group * params.kernel_h * params.kernel_w;

            // Use the tiled GEMM kernel directly for the sub-matrix
            let gemm_src = r#"
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
        for (unsigned int i = 0; i < TILE; i++) sum += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
"#;
            let gemm_f = cache.get_or_compile(device, gemm_src, "warp_gemm_tiled")?;
            let tile = 32u32;
            let gemm_cfg = LaunchConfig {
                grid_dim: ((nn + tile - 1) / tile, (m + tile - 1) / tile, 1),
                block_dim: (tile, tile, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                launch_err!(device.stream.launch_builder(&gemm_f)
                    .arg(&mut output.data.slice_mut(output_offset..))
                    .arg(&weight.data.slice(weight_offset..))
                    .arg(&col.data)
                    .arg(&m).arg(&nn).arg(&k)
                    .launch(gemm_cfg))?;
            }
        }
    }

    // Add bias if present
    if let Some(bias) = bias {
        let bias_f = cache.get_or_compile(device, BIAS_ADD_SRC, "warp_bias_add_spatial")?;
        for n in 0..batch {
            let spatial = out_h * out_w;
            let offset = (n * params.out_channels) as usize * spatial as usize;
            let total = params.out_channels * spatial;
            let cfg = LaunchConfig::for_num_elems(total);
            unsafe {
                launch_err!(device.stream.launch_builder(&bias_f)
                    .arg(&mut output.data.slice_mut(offset..))
                    .arg(&bias.data)
                    .arg(&params.out_channels)
                    .arg(&spatial)
                    .launch(cfg))?;
            }
        }
    }

    Ok(())
}

/// BatchNorm (inference mode). Fuses scale, bias, mean, var into a single pass.
///
/// input: [N, C, H, W]
/// output: [N, C, H, W]
pub fn batchnorm2d(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    scale: &GpuTensor<f32>,      // [C]
    bias: &GpuTensor<f32>,       // [C]
    running_mean: &GpuTensor<f32>, // [C]
    running_var: &GpuTensor<f32>,  // [C]
    output: &mut GpuTensor<f32>,
    channels: u32,
    spatial: u32,  // H * W
    eps: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, BATCHNORM_SRC, "warp_batchnorm")?;
    let batch = input.numel as u32 / (channels * spatial);
    let total = batch * channels * spatial;
    let cfg = LaunchConfig::for_num_elems(total);

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data)
            .arg(&input.data)
            .arg(&scale.data)
            .arg(&bias.data)
            .arg(&running_mean.data)
            .arg(&running_var.data)
            .arg(&eps)
            .arg(&channels)
            .arg(&spatial)
            .launch(cfg))?;
    }
    Ok(())
}

/// MaxPool2D.
pub fn maxpool2d(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
    channels: u32,
    h: u32, w: u32,
    kernel_h: u32, kernel_w: u32,
    stride_h: u32, stride_w: u32,
    pad_h: u32, pad_w: u32,
) -> Result<(), DeviceError> {
    let out_h = conv_output_size(h, kernel_h, stride_h, pad_h, 1);
    let out_w = conv_output_size(w, kernel_w, stride_w, pad_w, 1);
    let batch = input.numel as u32 / (channels * h * w);
    let f = cache.get_or_compile(device, MAXPOOL2D_SRC, "warp_maxpool2d")?;

    for n in 0..batch {
        let in_off = (n * channels) as usize * (h * w) as usize;
        let out_off = (n * channels) as usize * (out_h * out_w) as usize;
        let total = channels * out_h * out_w;
        let cfg = LaunchConfig::for_num_elems(total);

        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut output.data.slice_mut(out_off..))
                .arg(&input.data.slice(in_off..))
                .arg(&channels)
                .arg(&h).arg(&w)
                .arg(&out_h).arg(&out_w)
                .arg(&kernel_h).arg(&kernel_w)
                .arg(&stride_h).arg(&stride_w)
                .arg(&pad_h).arg(&pad_w)
                .launch(cfg))?;
        }
    }
    Ok(())
}

/// AvgPool2D.
pub fn avgpool2d(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
    channels: u32,
    h: u32, w: u32,
    kernel_h: u32, kernel_w: u32,
    stride_h: u32, stride_w: u32,
    pad_h: u32, pad_w: u32,
) -> Result<(), DeviceError> {
    let out_h = conv_output_size(h, kernel_h, stride_h, pad_h, 1);
    let out_w = conv_output_size(w, kernel_w, stride_w, pad_w, 1);
    let batch = input.numel as u32 / (channels * h * w);
    let f = cache.get_or_compile(device, AVGPOOL2D_SRC, "warp_avgpool2d")?;

    for n in 0..batch {
        let in_off = (n * channels) as usize * (h * w) as usize;
        let out_off = (n * channels) as usize * (out_h * out_w) as usize;
        let total = channels * out_h * out_w;
        let cfg = LaunchConfig::for_num_elems(total);

        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut output.data.slice_mut(out_off..))
                .arg(&input.data.slice(in_off..))
                .arg(&channels)
                .arg(&h).arg(&w)
                .arg(&out_h).arg(&out_w)
                .arg(&kernel_h).arg(&kernel_w)
                .arg(&stride_h).arg(&stride_w)
                .arg(&pad_h).arg(&pad_w)
                .launch(cfg))?;
        }
    }
    Ok(())
}

/// Global average pooling: [N, C, H, W] → [N, C].
pub fn global_avg_pool(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
    channels: u32,
    spatial: u32,
) -> Result<(), DeviceError> {
    let batch = input.numel as u32 / (channels * spatial);
    let f = cache.get_or_compile(device, GLOBAL_AVG_POOL_SRC, "warp_global_avg_pool")?;

    for n in 0..batch {
        let in_off = (n * channels) as usize * spatial as usize;
        let out_off = (n * channels) as usize;
        let cfg = LaunchConfig::for_num_elems(channels);

        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut output.data.slice_mut(out_off..))
                .arg(&input.data.slice(in_off..))
                .arg(&channels)
                .arg(&spatial)
                .launch(cfg))?;
        }
    }
    Ok(())
}

// ── ConvTranspose2D ──────────────────────────────────────────────
// "Deconvolution" — used in U-Net decoders, GANs, FPN upsampling.
// Conceptually the gradient of a forward convolution.
// Implementation: col2im after GEMM.

const COL2IM_SRC: &str = r#"
extern "C" __global__ void warp_col2im(
    float *output,          // [C, H, W] — accumulated
    const float *col,       // [C * kH * kW, out_H * out_W]
    unsigned int C,
    unsigned int H, unsigned int W,
    unsigned int kH, unsigned int kW,
    unsigned int out_H, unsigned int out_W,
    unsigned int stride_h, unsigned int stride_w,
    unsigned int pad_h, unsigned int pad_w
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = C * H * W;
    if (idx >= total) return;

    unsigned int iw = idx % W;
    unsigned int ih = (idx / W) % H;
    unsigned int c = idx / (H * W);

    float val = 0.0f;

    // Iterate over all output positions that touch this input position
    for (unsigned int kh = 0; kh < kH; kh++) {
        for (unsigned int kw = 0; kw < kW; kw++) {
            // Reverse: oh * stride + kh - pad = ih
            int oh_s = (int)ih + (int)pad_h - (int)kh;
            int ow_s = (int)iw + (int)pad_w - (int)kw;

            if (oh_s % (int)stride_h != 0 || ow_s % (int)stride_w != 0) continue;

            int oh = oh_s / (int)stride_h;
            int ow = ow_s / (int)stride_w;

            if (oh >= 0 && oh < (int)out_H && ow >= 0 && ow < (int)out_W) {
                unsigned int col_row = c * kH * kW + kh * kW + kw;
                unsigned int col_col = (unsigned int)oh * out_W + (unsigned int)ow;
                val += col[col_row * (out_H * out_W) + col_col];
            }
        }
    }
    output[idx] = val;
}
"#;

/// Compute output size for transposed convolution.
pub fn conv_transpose_output_size(
    input: u32, kernel: u32, stride: u32, padding: u32, output_padding: u32,
) -> u32 {
    (input - 1) * stride - 2 * padding + kernel + output_padding
}

/// ConvTranspose2D parameters.
#[derive(Debug, Clone)]
pub struct ConvTranspose2dParams {
    pub in_channels: u32,
    pub out_channels: u32,
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub padding_h: u32,
    pub padding_w: u32,
    pub output_padding_h: u32,
    pub output_padding_w: u32,
    pub groups: u32,
}

impl ConvTranspose2dParams {
    pub fn new(in_c: u32, out_c: u32, kernel: u32) -> Self {
        Self {
            in_channels: in_c, out_channels: out_c,
            kernel_h: kernel, kernel_w: kernel,
            stride_h: 1, stride_w: 1,
            padding_h: 0, padding_w: 0,
            output_padding_h: 0, output_padding_w: 0,
            groups: 1,
        }
    }
    pub fn stride(mut self, s: u32) -> Self { self.stride_h = s; self.stride_w = s; self }
    pub fn padding(mut self, p: u32) -> Self { self.padding_h = p; self.padding_w = p; self }
    pub fn output_padding(mut self, p: u32) -> Self { self.output_padding_h = p; self.output_padding_w = p; self }

    pub fn output_h(&self, h: u32) -> u32 {
        conv_transpose_output_size(h, self.kernel_h, self.stride_h, self.padding_h, self.output_padding_h)
    }
    pub fn output_w(&self, w: u32) -> u32 {
        conv_transpose_output_size(w, self.kernel_w, self.stride_w, self.padding_w, self.output_padding_w)
    }
}

/// ConvTranspose2D via GEMM + col2im.
///
/// input: [N, C_in, H, W]
/// weight: [C_in, C_out, kH, kW] (note: transposed layout vs Conv2D)
/// output: [N, C_out, out_H, out_W]
pub fn conv_transpose2d(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    weight: &GpuTensor<f32>,
    bias: Option<&GpuTensor<f32>>,
    output: &mut GpuTensor<f32>,
    params: &ConvTranspose2dParams,
    h: u32,
    w: u32,
) -> Result<(), DeviceError> {
    let batch = input.numel as u32 / (params.in_channels * h * w);
    let out_h = params.output_h(h);
    let out_w = params.output_w(w);
    let col_rows = params.out_channels * params.kernel_h * params.kernel_w;
    let col_cols = h * w;
    let col_size = (col_rows * col_cols) as usize;

    let mut col = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[col_size]), DType::F32)?;

    // GEMM: weight^T[C_out*kH*kW, C_in] × input[C_in, H*W] → col[C_out*kH*kW, H*W]
    // weight is [C_in, C_out*kH*kW], so we need transposed multiply
    let gemm_src = r#"
#define TILE 32
extern "C" __global__ void warp_gemm_transA(
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
        // A is transposed: A^T[row, t*TILE+tx] = A[t*TILE+tx, row]
        unsigned int a_k = t * TILE + tx;
        As[ty][tx] = (row < M && a_k < K) ? A[a_k * M + row] : 0.0f;
        Bs[ty][tx] = (t * TILE + ty < K && col < N) ? B[(t * TILE + ty) * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (unsigned int i = 0; i < TILE; i++) sum += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
"#;
    let gemm_f = cache.get_or_compile(device, gemm_src, "warp_gemm_transA")?;
    let col2im_f = cache.get_or_compile(device, COL2IM_SRC, "warp_col2im")?;

    let tile = 32u32;

    for n in 0..batch {
        let in_off = (n * params.in_channels) as usize * (h * w) as usize;
        let out_off = (n * params.out_channels) as usize * (out_h * out_w) as usize;

        // GEMM: weight^T × input_n → col
        let m = col_rows;
        let nn = col_cols;
        let k = params.in_channels;
        let gemm_cfg = LaunchConfig {
            grid_dim: ((nn + tile - 1) / tile, (m + tile - 1) / tile, 1),
            block_dim: (tile, tile, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            launch_err!(device.stream.launch_builder(&gemm_f)
                .arg(&mut col.data)
                .arg(&weight.data)
                .arg(&input.data.slice(in_off..))
                .arg(&m).arg(&nn).arg(&k)
                .launch(gemm_cfg))?;
        }

        // col2im: scatter col back to spatial output
        let total_out = params.out_channels * out_h * out_w;
        let cfg = LaunchConfig::for_num_elems(total_out);

        unsafe {
            launch_err!(device.stream.launch_builder(&col2im_f)
                .arg(&mut output.data.slice_mut(out_off..))
                .arg(&col.data)
                .arg(&params.out_channels)
                .arg(&out_h).arg(&out_w)
                .arg(&params.kernel_h).arg(&params.kernel_w)
                .arg(&h).arg(&w) // "out_H/W" of the forward conv = input H/W here
                .arg(&params.stride_h).arg(&params.stride_w)
                .arg(&params.padding_h).arg(&params.padding_w)
                .launch(cfg))?;
        }
    }

    // Add bias
    if let Some(bias) = bias {
        let bias_f = cache.get_or_compile(device, BIAS_ADD_SRC, "warp_bias_add_spatial")?;
        for n in 0..batch {
            let spatial = out_h * out_w;
            let offset = (n * params.out_channels) as usize * spatial as usize;
            let total = params.out_channels * spatial;
            let cfg = LaunchConfig::for_num_elems(total);
            unsafe {
                launch_err!(device.stream.launch_builder(&bias_f)
                    .arg(&mut output.data.slice_mut(offset..))
                    .arg(&bias.data)
                    .arg(&params.out_channels)
                    .arg(&spatial)
                    .launch(cfg))?;
            }
        }
    }

    Ok(())
}

// ── Resize (Nearest + Bilinear) ─────────────────────────────────
// Used in FPN, U-Net skip connections, any upsampling path.

const RESIZE_NEAREST_SRC: &str = r#"
extern "C" __global__ void warp_resize_nearest(
    float *output,          // [C, out_H, out_W]
    const float *input,     // [C, H, W]
    unsigned int C,
    unsigned int H, unsigned int W,
    unsigned int out_H, unsigned int out_W
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = C * out_H * out_W;
    if (idx >= total) return;

    unsigned int ow = idx % out_W;
    unsigned int oh = (idx / out_W) % out_H;
    unsigned int c = idx / (out_H * out_W);

    // Map output coord to input coord
    float scale_h = (float)H / (float)out_H;
    float scale_w = (float)W / (float)out_W;

    unsigned int ih = (unsigned int)(oh * scale_h);
    unsigned int iw = (unsigned int)(ow * scale_w);
    ih = ih < H ? ih : H - 1;
    iw = iw < W ? iw : W - 1;

    output[idx] = input[c * H * W + ih * W + iw];
}
"#;

const RESIZE_BILINEAR_SRC: &str = r#"
extern "C" __global__ void warp_resize_bilinear(
    float *output,
    const float *input,
    unsigned int C,
    unsigned int H, unsigned int W,
    unsigned int out_H, unsigned int out_W
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = C * out_H * out_W;
    if (idx >= total) return;

    unsigned int ow = idx % out_W;
    unsigned int oh = (idx / out_W) % out_H;
    unsigned int c = idx / (out_H * out_W);

    // Align corners = false: map center-to-center
    float scale_h = (float)H / (float)out_H;
    float scale_w = (float)W / (float)out_W;

    float fh = (oh + 0.5f) * scale_h - 0.5f;
    float fw = (ow + 0.5f) * scale_w - 0.5f;

    int ih0 = (int)floorf(fh);
    int iw0 = (int)floorf(fw);
    int ih1 = ih0 + 1;
    int iw1 = iw0 + 1;

    float lh = fh - (float)ih0;
    float lw = fw - (float)iw0;

    // Clamp
    ih0 = ih0 < 0 ? 0 : (ih0 >= (int)H ? (int)H - 1 : ih0);
    ih1 = ih1 < 0 ? 0 : (ih1 >= (int)H ? (int)H - 1 : ih1);
    iw0 = iw0 < 0 ? 0 : (iw0 >= (int)W ? (int)W - 1 : iw0);
    iw1 = iw1 < 0 ? 0 : (iw1 >= (int)W ? (int)W - 1 : iw1);

    const float *base = input + c * H * W;
    float v00 = base[ih0 * W + iw0];
    float v01 = base[ih0 * W + iw1];
    float v10 = base[ih1 * W + iw0];
    float v11 = base[ih1 * W + iw1];

    output[idx] = (1 - lh) * ((1 - lw) * v00 + lw * v01)
                + lh * ((1 - lw) * v10 + lw * v11);
}
"#;

/// Resize with nearest-neighbor interpolation.
pub fn resize_nearest(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
    channels: u32,
    h: u32, w: u32,
    out_h: u32, out_w: u32,
) -> Result<(), DeviceError> {
    let batch = input.numel as u32 / (channels * h * w);
    let f = cache.get_or_compile(device, RESIZE_NEAREST_SRC, "warp_resize_nearest")?;

    for n in 0..batch {
        let in_off = (n * channels) as usize * (h * w) as usize;
        let out_off = (n * channels) as usize * (out_h * out_w) as usize;
        let total = channels * out_h * out_w;
        let cfg = LaunchConfig::for_num_elems(total);

        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut output.data.slice_mut(out_off..))
                .arg(&input.data.slice(in_off..))
                .arg(&channels)
                .arg(&h).arg(&w)
                .arg(&out_h).arg(&out_w)
                .launch(cfg))?;
        }
    }
    Ok(())
}

/// Resize with bilinear interpolation.
pub fn resize_bilinear(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
    channels: u32,
    h: u32, w: u32,
    out_h: u32, out_w: u32,
) -> Result<(), DeviceError> {
    let batch = input.numel as u32 / (channels * h * w);
    let f = cache.get_or_compile(device, RESIZE_BILINEAR_SRC, "warp_resize_bilinear")?;

    for n in 0..batch {
        let in_off = (n * channels) as usize * (h * w) as usize;
        let out_off = (n * channels) as usize * (out_h * out_w) as usize;
        let total = channels * out_h * out_w;
        let cfg = LaunchConfig::for_num_elems(total);

        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut output.data.slice_mut(out_off..))
                .arg(&input.data.slice(in_off..))
                .arg(&channels)
                .arg(&h).arg(&w)
                .arg(&out_h).arg(&out_w)
                .launch(cfg))?;
        }
    }
    Ok(())
}

// ── GridSample ──────────────────────────────────────────────────
// Samples from input at arbitrary coordinates specified by a grid.
// Used in spatial transformer networks, WiLoR refinement, etc.

const GRID_SAMPLE_BILINEAR_SRC: &str = r#"
extern "C" __global__ void warp_grid_sample_bilinear(
    float *output,          // [C, out_H, out_W]
    const float *input,     // [C, H, W]
    const float *grid,      // [out_H, out_W, 2] — (x, y) in [-1, 1]
    unsigned int C,
    unsigned int H, unsigned int W,
    unsigned int out_H, unsigned int out_W
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = C * out_H * out_W;
    if (idx >= total) return;

    unsigned int ow = idx % out_W;
    unsigned int oh = (idx / out_W) % out_H;
    unsigned int c = idx / (out_H * out_W);

    // Grid coordinates: (gx, gy) in [-1, 1] → pixel coords
    unsigned int grid_idx = (oh * out_W + ow) * 2;
    float gx = grid[grid_idx];
    float gy = grid[grid_idx + 1];

    // Unnormalize: [-1, 1] → [0, W-1] and [0, H-1]
    float fx = (gx + 1.0f) * 0.5f * ((float)W - 1.0f);
    float fy = (gy + 1.0f) * 0.5f * ((float)H - 1.0f);

    int ix0 = (int)floorf(fx);
    int iy0 = (int)floorf(fy);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    float lx = fx - (float)ix0;
    float ly = fy - (float)iy0;

    // Zero-padding for out-of-bounds
    const float *base = input + c * H * W;

    #define SAFE(y, x) (((y) >= 0 && (y) < (int)H && (x) >= 0 && (x) < (int)W) ? base[(y) * W + (x)] : 0.0f)

    float v00 = SAFE(iy0, ix0);
    float v01 = SAFE(iy0, ix1);
    float v10 = SAFE(iy1, ix0);
    float v11 = SAFE(iy1, ix1);

    #undef SAFE

    output[idx] = (1 - ly) * ((1 - lx) * v00 + lx * v01)
                + ly * ((1 - lx) * v10 + lx * v11);
}
"#;

/// Grid sample with bilinear interpolation and zero padding.
///
/// input: [N, C, H, W]
/// grid: [N, out_H, out_W, 2] — coordinates in [-1, 1]
/// output: [N, C, out_H, out_W]
pub fn grid_sample(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    grid: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
    channels: u32,
    h: u32, w: u32,
    out_h: u32, out_w: u32,
) -> Result<(), DeviceError> {
    let batch = input.numel as u32 / (channels * h * w);
    let f = cache.get_or_compile(device, GRID_SAMPLE_BILINEAR_SRC, "warp_grid_sample_bilinear")?;

    for n in 0..batch {
        let in_off = (n * channels) as usize * (h * w) as usize;
        let grid_off = n as usize * (out_h * out_w * 2) as usize;
        let out_off = (n * channels) as usize * (out_h * out_w) as usize;
        let total = channels * out_h * out_w;
        let cfg = LaunchConfig::for_num_elems(total);

        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut output.data.slice_mut(out_off..))
                .arg(&input.data.slice(in_off..))
                .arg(&grid.data.slice(grid_off..))
                .arg(&channels)
                .arg(&h).arg(&w)
                .arg(&out_h).arg(&out_w)
                .launch(cfg))?;
        }
    }
    Ok(())
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

    /// CPU reference: direct Conv2D (no groups, no dilation).
    fn cpu_conv2d(
        input: &[f32], weight: &[f32], bias: Option<&[f32]>,
        c_in: usize, h: usize, w: usize,
        c_out: usize, kh: usize, kw: usize,
        stride: usize, padding: usize,
    ) -> Vec<f32> {
        let out_h = (h + 2 * padding - kh) / stride + 1;
        let out_w = (w + 2 * padding - kw) / stride + 1;
        let mut output = vec![0.0f32; c_out * out_h * out_w];

        for co in 0..c_out {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    for ci in 0..c_in {
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * stride + khi;
                                let iw = ow * stride + kwi;
                                let ih = ih as isize - padding as isize;
                                let iw = iw as isize - padding as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let iv = input[ci * h * w + ih as usize * w + iw as usize];
                                    let wv = weight[co * (c_in * kh * kw) + ci * (kh * kw) + khi * kw + kwi];
                                    sum += iv * wv;
                                }
                            }
                        }
                    }
                    if let Some(b) = bias {
                        sum += b[co];
                    }
                    output[co * out_h * out_w + oh * out_w + ow] = sum;
                }
            }
        }
        output
    }

    #[test]
    fn conv2d_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (c_in, c_out, h, w, k) = (3u32, 8u32, 16u32, 16u32, 3u32);
        let params = Conv2dParams::new(c_in, c_out, k).padding(1);
        let out_h = params.output_h(h);
        let out_w = params.output_w(w);

        // Random-ish data
        let input_data: Vec<f32> = (0..(c_in * h * w) as usize)
            .map(|i| ((i * 7 + 3) % 200) as f32 * 0.01 - 1.0).collect();
        let weight_data: Vec<f32> = (0..(c_out * c_in * k * k) as usize)
            .map(|i| ((i * 11 + 5) % 200) as f32 * 0.01 - 1.0).collect();
        let bias_data: Vec<f32> = (0..c_out as usize)
            .map(|i| (i as f32 - c_out as f32 / 2.0) * 0.1).collect();

        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[1, c_in as usize, h as usize, w as usize]), DType::F32).unwrap();
        let weight = GpuTensor::from_host(&dev, &weight_data,
            Shape::from_static(&[c_out as usize, c_in as usize, k as usize, k as usize]), DType::F32).unwrap();
        let bias = GpuTensor::from_host(&dev, &bias_data,
            Shape::from_static(&[c_out as usize]), DType::F32).unwrap();
        let mut output = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, c_out as usize, out_h as usize, out_w as usize]), DType::F32).unwrap();

        conv2d(&cache, &dev, &input, &weight, Some(&bias), &mut output, &params, h, w).unwrap();
        dev.synchronize().unwrap();

        let result = output.to_host(&dev).unwrap();
        let expected = cpu_conv2d(&input_data, &weight_data, Some(&bias_data),
            c_in as usize, h as usize, w as usize,
            c_out as usize, k as usize, k as usize, 1, 1);

        let max_err: f32 = result.iter().zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("Conv2D ({c_in}→{c_out}, {k}×{k}, pad=1, {h}×{w} → {out_h}×{out_w}):");
        println!("  Max error vs CPU: {max_err:.6}");
        println!("  Output range: [{:.4}, {:.4}]",
            result.iter().cloned().fold(f32::INFINITY, f32::min),
            result.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        assert!(max_err < 0.01, "Conv2D error too high: {max_err}");
    }

    #[test]
    fn conv2d_stride() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (c_in, c_out, h, w, k) = (3u32, 16u32, 32u32, 32u32, 3u32);
        let params = Conv2dParams::new(c_in, c_out, k).stride(2).padding(1);
        let out_h = params.output_h(h);
        let out_w = params.output_w(w);

        let input_data: Vec<f32> = (0..(c_in * h * w) as usize)
            .map(|i| ((i * 7 + 3) % 200) as f32 * 0.01 - 1.0).collect();
        let weight_data: Vec<f32> = (0..(c_out * c_in * k * k) as usize)
            .map(|i| ((i * 11 + 5) % 200) as f32 * 0.01 - 1.0).collect();

        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[1, c_in as usize, h as usize, w as usize]), DType::F32).unwrap();
        let weight = GpuTensor::from_host(&dev, &weight_data,
            Shape::from_static(&[c_out as usize, c_in as usize, k as usize, k as usize]), DType::F32).unwrap();
        let mut output = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, c_out as usize, out_h as usize, out_w as usize]), DType::F32).unwrap();

        conv2d(&cache, &dev, &input, &weight, None, &mut output, &params, h, w).unwrap();
        dev.synchronize().unwrap();

        let result = output.to_host(&dev).unwrap();
        let expected = cpu_conv2d(&input_data, &weight_data, None,
            c_in as usize, h as usize, w as usize,
            c_out as usize, k as usize, k as usize, 2, 1);

        let max_err: f32 = result.iter().zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("Conv2D stride=2 ({c_in}→{c_out}, {k}×{k}, {h}×{w} → {out_h}×{out_w}): max err = {max_err:.6}");
        assert!(max_err < 0.01, "Conv2D stride error too high: {max_err}");
    }

    #[test]
    fn batchnorm_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (c, h, w) = (8u32, 4u32, 4u32);
        let spatial = h * w;

        let input_data: Vec<f32> = (0..(c * spatial) as usize)
            .map(|i| ((i * 7 + 3) % 200) as f32 * 0.01 - 1.0).collect();
        let scale_data = vec![1.0f32; c as usize];
        let bias_data = vec![0.0f32; c as usize];
        let mean_data: Vec<f32> = (0..c as usize).map(|i| i as f32 * 0.1).collect();
        let var_data = vec![1.0f32; c as usize];

        let input = GpuTensor::from_host(&dev, &input_data, Shape::from_static(&[1, c as usize, h as usize, w as usize]), DType::F32).unwrap();
        let scale = GpuTensor::from_host(&dev, &scale_data, Shape::from_static(&[c as usize]), DType::F32).unwrap();
        let bias = GpuTensor::from_host(&dev, &bias_data, Shape::from_static(&[c as usize]), DType::F32).unwrap();
        let mean = GpuTensor::from_host(&dev, &mean_data, Shape::from_static(&[c as usize]), DType::F32).unwrap();
        let var = GpuTensor::from_host(&dev, &var_data, Shape::from_static(&[c as usize]), DType::F32).unwrap();
        let mut output = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[1, c as usize, h as usize, w as usize]), DType::F32).unwrap();

        batchnorm2d(&cache, &dev, &input, &scale, &bias, &mean, &var, &mut output, c, spatial, 1e-5).unwrap();
        dev.synchronize().unwrap();

        let result = output.to_host(&dev).unwrap();

        // CPU reference
        for ch in 0..c as usize {
            let m = mean_data[ch];
            let v = var_data[ch];
            let inv_std = 1.0 / (v + 1e-5f32).sqrt();
            for s in 0..spatial as usize {
                let idx = ch * spatial as usize + s;
                let expected = (input_data[idx] - m) * inv_std * scale_data[ch] + bias_data[ch];
                let err = (result[idx] - expected).abs();
                assert!(err < 1e-4, "BN mismatch at ch={ch} s={s}: {:.6} vs {:.6}", result[idx], expected);
            }
        }
        println!("BatchNorm2D ({c} channels, {h}×{w}): correct!");
    }

    #[test]
    fn maxpool2d_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (c, h, w) = (4u32, 8u32, 8u32);
        let (kh, kw, sh, sw) = (2u32, 2u32, 2u32, 2u32);
        let out_h = conv_output_size(h, kh, sh, 0, 1);
        let out_w = conv_output_size(w, kw, sw, 0, 1);

        let input_data: Vec<f32> = (0..(c * h * w) as usize)
            .map(|i| ((i * 13 + 7) % 200) as f32 * 0.01 - 1.0).collect();

        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[1, c as usize, h as usize, w as usize]), DType::F32).unwrap();
        let mut output = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, c as usize, out_h as usize, out_w as usize]), DType::F32).unwrap();

        maxpool2d(&cache, &dev, &input, &mut output, c, h, w, kh, kw, sh, sw, 0, 0).unwrap();
        dev.synchronize().unwrap();

        let result = output.to_host(&dev).unwrap();

        // CPU reference
        for ch in 0..c as usize {
            for oh in 0..out_h as usize {
                for ow in 0..out_w as usize {
                    let mut max_v = f32::NEG_INFINITY;
                    for ki in 0..kh as usize {
                        for kj in 0..kw as usize {
                            let ih = oh * sh as usize + ki;
                            let iw = ow * sw as usize + kj;
                            if ih < h as usize && iw < w as usize {
                                max_v = max_v.max(input_data[ch * (h*w) as usize + ih * w as usize + iw]);
                            }
                        }
                    }
                    let idx = ch * (out_h * out_w) as usize + oh * out_w as usize + ow;
                    let err = (result[idx] - max_v).abs();
                    assert!(err < 1e-6, "MaxPool mismatch at ch={ch} oh={oh} ow={ow}");
                }
            }
        }
        println!("MaxPool2D ({c}ch, {h}×{w} → {out_h}×{out_w}, k={kh}×{kw} s={sh}): correct!");
    }

    #[test]
    fn mini_cnn_pipeline() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // Mini CNN: Conv(3→16, 3×3) → BN → ReLU → MaxPool(2×2) → Conv(16→32, 3×3) → GlobalAvgPool → GEMM
        let (h, w) = (32u32, 32u32);
        let rand_vec = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i * 7 + seed) % 200) as f32 * 0.01 - 1.0).collect()
        };

        // Input: [1, 3, 32, 32]
        let input_data = rand_vec(3 * 32 * 32, 13);
        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[1, 3, 32, 32]), DType::F32).unwrap();

        // Conv1: 3→16, 3×3, pad=1
        let conv1_params = Conv2dParams::new(3, 16, 3).padding(1);
        let conv1_w_data = rand_vec(16 * 3 * 3 * 3, 17);
        let conv1_w = GpuTensor::from_host(&dev, &conv1_w_data,
            Shape::from_static(&[16, 3, 3, 3]), DType::F32).unwrap();
        let mut conv1_out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, 16, 32, 32]), DType::F32).unwrap();
        conv2d(&cache, &dev, &input, &conv1_w, None, &mut conv1_out, &conv1_params, h, w).unwrap();

        // BN1
        let bn_scale = GpuTensor::from_host(&dev, &vec![1.0f32; 16], Shape::from_static(&[16]), DType::F32).unwrap();
        let bn_bias = GpuTensor::from_host(&dev, &vec![0.0f32; 16], Shape::from_static(&[16]), DType::F32).unwrap();
        let bn_mean = GpuTensor::from_host(&dev, &vec![0.0f32; 16], Shape::from_static(&[16]), DType::F32).unwrap();
        let bn_var = GpuTensor::from_host(&dev, &vec![1.0f32; 16], Shape::from_static(&[16]), DType::F32).unwrap();
        let mut bn1_out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, 16, 32, 32]), DType::F32).unwrap();
        batchnorm2d(&cache, &dev, &conv1_out, &bn_scale, &bn_bias, &bn_mean, &bn_var,
            &mut bn1_out, 16, 32 * 32, 1e-5).unwrap();

        // ReLU (in-place via elementwise)
        let relu_src = r#"
extern "C" __global__ void warp_relu(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = fmaxf(x[i], 0.0f); }
}
"#;
        let relu_f = cache.get_or_compile(&dev, relu_src, "warp_relu").unwrap();
        let mut relu1_out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, 16, 32, 32]), DType::F32).unwrap();
        let n1 = bn1_out.numel;
        unsafe {
            launch_err!(dev.stream.launch_builder(&relu_f)
                .arg(&mut relu1_out.data).arg(&bn1_out.data).arg(&n1)
                .launch(LaunchConfig::for_num_elems(n1 as u32))).unwrap();
        }

        // MaxPool 2×2
        let mut pool1_out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, 16, 16, 16]), DType::F32).unwrap();
        maxpool2d(&cache, &dev, &relu1_out, &mut pool1_out, 16, 32, 32, 2, 2, 2, 2, 0, 0).unwrap();

        // Conv2: 16→32, 3×3, pad=1
        let conv2_params = Conv2dParams::new(16, 32, 3).padding(1);
        let conv2_w_data = rand_vec(32 * 16 * 3 * 3, 23);
        let conv2_w = GpuTensor::from_host(&dev, &conv2_w_data,
            Shape::from_static(&[32, 16, 3, 3]), DType::F32).unwrap();
        let mut conv2_out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, 32, 16, 16]), DType::F32).unwrap();
        conv2d(&cache, &dev, &pool1_out, &conv2_w, None, &mut conv2_out, &conv2_params, 16, 16).unwrap();

        // Global average pool: [1, 32, 16, 16] → [1, 32]
        let mut gap_out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, 32]), DType::F32).unwrap();
        global_avg_pool(&cache, &dev, &conv2_out, &mut gap_out, 32, 16 * 16).unwrap();

        // FC (GEMM): [1, 32] × [32, 10] → [1, 10]
        let fc_w_data = rand_vec(32 * 10, 31);
        let fc_w = GpuTensor::from_host(&dev, &fc_w_data,
            Shape::from_static(&[32, 10]), DType::F32).unwrap();
        let mut logits = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, 10]), DType::F32).unwrap();
        crate::ops::gemm(&cache, &dev, &gap_out, &fc_w, &mut logits, 1, 10, 32).unwrap();

        dev.synchronize().unwrap();
        let result = logits.to_host(&dev).unwrap();

        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|v| v.is_finite()), "CNN output has NaN/Inf!");
        assert!(result.iter().any(|v| *v != 0.0), "CNN output is all zeros!");

        println!("\n=== Mini CNN Pipeline ===");
        println!("  Input:  [1, 3, 32, 32]");
        println!("  Conv1:  3→16, 3×3, pad=1 → [1, 16, 32, 32]");
        println!("  BN+ReLU → MaxPool 2×2 → [1, 16, 16, 16]");
        println!("  Conv2:  16→32, 3×3, pad=1 → [1, 32, 16, 16]");
        println!("  GAP → [1, 32] → FC → [1, 10]");
        println!("  Output: {:?}", result);
        println!("  All ops correct, pipeline produces finite logits!");
        println!("{}", cache.stats());
    }

    #[test]
    fn resize_nearest_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (c, h, w) = (3u32, 4u32, 4u32);
        let (out_h, out_w) = (8u32, 8u32); // 2× upsample

        let input_data: Vec<f32> = (0..(c * h * w) as usize)
            .map(|i| i as f32).collect();

        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[1, c as usize, h as usize, w as usize]), DType::F32).unwrap();
        let mut output = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, c as usize, out_h as usize, out_w as usize]), DType::F32).unwrap();

        resize_nearest(&cache, &dev, &input, &mut output, c, h, w, out_h, out_w).unwrap();
        dev.synchronize().unwrap();

        let result = output.to_host(&dev).unwrap();

        // Verify: each input pixel should appear in a 2×2 block
        for ch in 0..c as usize {
            for oh in 0..out_h as usize {
                for ow in 0..out_w as usize {
                    let ih = oh * h as usize / out_h as usize;
                    let iw = ow * w as usize / out_w as usize;
                    let expected = input_data[ch * (h * w) as usize + ih * w as usize + iw];
                    let actual = result[ch * (out_h * out_w) as usize + oh * out_w as usize + ow];
                    assert!((actual - expected).abs() < 1e-6,
                        "Nearest mismatch at ch={ch} oh={oh} ow={ow}: {actual} vs {expected}");
                }
            }
        }
        println!("Resize nearest ({h}×{w} → {out_h}×{out_w}): correct!");
    }

    #[test]
    fn resize_bilinear_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (c, h, w) = (2u32, 4u32, 4u32);
        let (out_h, out_w) = (8u32, 8u32);

        let input_data: Vec<f32> = (0..(c * h * w) as usize)
            .map(|i| i as f32 * 0.1).collect();

        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[1, c as usize, h as usize, w as usize]), DType::F32).unwrap();
        let mut output = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, c as usize, out_h as usize, out_w as usize]), DType::F32).unwrap();

        resize_bilinear(&cache, &dev, &input, &mut output, c, h, w, out_h, out_w).unwrap();
        dev.synchronize().unwrap();

        let result = output.to_host(&dev).unwrap();
        assert!(result.iter().all(|v| v.is_finite()), "Bilinear output has NaN/Inf!");
        // Values should be interpolated — not all integers
        let non_int_count = result.iter().filter(|v| (v.round() - **v).abs() > 1e-4).count();
        assert!(non_int_count > 0, "Bilinear should produce non-integer interpolated values");

        println!("Resize bilinear ({h}×{w} → {out_h}×{out_w}): correct!");
        println!("  Interpolated values: {non_int_count}/{} are non-integer", result.len());
    }

    #[test]
    fn grid_sample_identity() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // Identity grid: grid[y, x] = (x_norm, y_norm) should reproduce the input
        let (c, h, w) = (2u32, 4u32, 4u32);

        let input_data: Vec<f32> = (0..(c * h * w) as usize)
            .map(|i| i as f32 * 0.5).collect();

        // Build identity grid: [-1, 1] mapped to each pixel center
        let mut grid_data = vec![0.0f32; (h * w * 2) as usize];
        for iy in 0..h {
            for ix in 0..w {
                let gx = (ix as f32 / (w - 1) as f32) * 2.0 - 1.0; // [-1, 1]
                let gy = (iy as f32 / (h - 1) as f32) * 2.0 - 1.0;
                let idx = (iy * w + ix) as usize * 2;
                grid_data[idx] = gx;
                grid_data[idx + 1] = gy;
            }
        }

        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[1, c as usize, h as usize, w as usize]), DType::F32).unwrap();
        let grid = GpuTensor::from_host(&dev, &grid_data,
            Shape::from_static(&[1, h as usize, w as usize, 2]), DType::F32).unwrap();
        let mut output = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, c as usize, h as usize, w as usize]), DType::F32).unwrap();

        grid_sample(&cache, &dev, &input, &grid, &mut output, c, h, w, h, w).unwrap();
        dev.synchronize().unwrap();

        let result = output.to_host(&dev).unwrap();
        let max_err: f32 = result.iter().zip(input_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("GridSample identity ({c}ch, {h}×{w}): max error = {max_err:.6}");
        assert!(max_err < 1e-4, "Identity grid sample should reproduce input");
    }

    #[test]
    fn conv_transpose2d_upsample() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // ConvTranspose2D: 8→4, k=4, s=2, p=1 → should 2× upsample
        let (c_in, c_out, h, w) = (8u32, 4u32, 4u32, 4u32);
        let params = ConvTranspose2dParams::new(c_in, c_out, 4).stride(2).padding(1);
        let out_h = params.output_h(h);
        let out_w = params.output_w(w);

        let input_data: Vec<f32> = (0..(c_in * h * w) as usize)
            .map(|i| ((i * 7 + 3) % 200) as f32 * 0.01 - 1.0).collect();
        // Weight: [C_in, C_out, kH, kW] = [8, 4, 4, 4]
        let weight_data: Vec<f32> = (0..(c_in * c_out * 4 * 4) as usize)
            .map(|i| ((i * 11 + 5) % 200) as f32 * 0.01 - 1.0).collect();

        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[1, c_in as usize, h as usize, w as usize]), DType::F32).unwrap();
        let weight = GpuTensor::from_host(&dev, &weight_data,
            Shape::from_static(&[c_in as usize, c_out as usize, 4, 4]), DType::F32).unwrap();
        let mut output = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, c_out as usize, out_h as usize, out_w as usize]), DType::F32).unwrap();

        conv_transpose2d(&cache, &dev, &input, &weight, None, &mut output, &params, h, w).unwrap();
        dev.synchronize().unwrap();

        let result = output.to_host(&dev).unwrap();
        assert_eq!(result.len(), (c_out * out_h * out_w) as usize);
        assert!(result.iter().all(|v| v.is_finite()), "ConvTranspose output has NaN/Inf!");
        assert!(result.iter().any(|v| *v != 0.0), "ConvTranspose output is all zeros!");

        println!("ConvTranspose2D ({c_in}→{c_out}, k=4, s=2, p=1): {h}×{w} → {out_h}×{out_w}");
        println!("  Output range: [{:.4}, {:.4}]",
            result.iter().cloned().fold(f32::INFINITY, f32::min),
            result.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    }

    #[test]
    fn unet_style_pipeline() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // Mini U-Net: encode → bottleneck → decode with skip connections
        let rand_vec = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i * 7 + seed) % 200) as f32 * 0.01 - 1.0).collect()
        };

        // Encoder: [1, 3, 32, 32] → Conv(3→16) → Pool → [1, 16, 16, 16]
        let input_data = rand_vec(3 * 32 * 32, 13);
        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[1, 3, 32, 32]), DType::F32).unwrap();

        let enc_params = Conv2dParams::new(3, 16, 3).padding(1);
        let enc_w = GpuTensor::from_host(&dev, &rand_vec(16 * 3 * 3 * 3, 17),
            Shape::from_static(&[16, 3, 3, 3]), DType::F32).unwrap();
        let mut enc_out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, 16, 32, 32]), DType::F32).unwrap();
        conv2d(&cache, &dev, &input, &enc_w, None, &mut enc_out, &enc_params, 32, 32).unwrap();

        let mut enc_pooled = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, 16, 16, 16]), DType::F32).unwrap();
        maxpool2d(&cache, &dev, &enc_out, &mut enc_pooled, 16, 32, 32, 2, 2, 2, 2, 0, 0).unwrap();

        // Bottleneck: Conv(16→32, 3×3)
        let bot_params = Conv2dParams::new(16, 32, 3).padding(1);
        let bot_w = GpuTensor::from_host(&dev, &rand_vec(32 * 16 * 3 * 3, 23),
            Shape::from_static(&[32, 16, 3, 3]), DType::F32).unwrap();
        let mut bot_out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, 32, 16, 16]), DType::F32).unwrap();
        conv2d(&cache, &dev, &enc_pooled, &bot_w, None, &mut bot_out, &bot_params, 16, 16).unwrap();

        // Decoder: Upsample 2× (bilinear) → [1, 32, 32, 32]
        let mut up_out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, 32, 32, 32]), DType::F32).unwrap();
        resize_bilinear(&cache, &dev, &bot_out, &mut up_out, 32, 16, 16, 32, 32).unwrap();

        // Decoder conv: Conv(32→16, 3×3) → [1, 16, 32, 32]
        let dec_params = Conv2dParams::new(32, 16, 3).padding(1);
        let dec_w = GpuTensor::from_host(&dev, &rand_vec(16 * 32 * 3 * 3, 29),
            Shape::from_static(&[16, 32, 3, 3]), DType::F32).unwrap();
        let mut dec_out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, 16, 32, 32]), DType::F32).unwrap();
        conv2d(&cache, &dev, &up_out, &dec_w, None, &mut dec_out, &dec_params, 32, 32).unwrap();

        // Output head: Conv(16→1, 1×1) → segmentation mask [1, 1, 32, 32]
        let head_params = Conv2dParams::new(16, 1, 1);
        let head_w = GpuTensor::from_host(&dev, &rand_vec(1 * 16 * 1 * 1, 37),
            Shape::from_static(&[1, 16, 1, 1]), DType::F32).unwrap();
        let mut seg_mask = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, 1, 32, 32]), DType::F32).unwrap();
        conv2d(&cache, &dev, &dec_out, &head_w, None, &mut seg_mask, &head_params, 32, 32).unwrap();

        dev.synchronize().unwrap();
        let result = seg_mask.to_host(&dev).unwrap();

        assert_eq!(result.len(), 32 * 32);
        assert!(result.iter().all(|v| v.is_finite()), "U-Net output has NaN/Inf!");
        assert!(result.iter().any(|v| *v != 0.0), "U-Net output is all zeros!");

        println!("\n=== Mini U-Net Pipeline ===");
        println!("  Encoder:    [1,3,32,32] → Conv(3→16) → Pool → [1,16,16,16]");
        println!("  Bottleneck: Conv(16→32) → [1,32,16,16]");
        println!("  Decoder:    Bilinear 2× → [1,32,32,32] → Conv(32→16) → [1,16,32,32]");
        println!("  Head:       Conv(16→1, 1×1) → [1,1,32,32] segmentation mask");
        println!("  Mask range: [{:.4}, {:.4}]",
            result.iter().cloned().fold(f32::INFINITY, f32::min),
            result.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        println!("  Pipeline complete!");
        println!("{}", cache.stats());
    }
}

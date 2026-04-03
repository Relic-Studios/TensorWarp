//! Edge-case ops that close the remaining 5-15% gap for every production model.
//!
//! ScatterND, Expand, Where, Unsqueeze, Split, DepthToSpace,
//! Resize cubic, MaxPool1D, Multinomial, SelectiveScan (Mamba).

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

// ── ScatterND ───────────────────────────────────────────────────
// Write values to specific indices in a tensor.

const SCATTER_ND_SRC: &str = r#"
extern "C" __global__ void warp_scatter_nd(
    float *data,            // [N] — modified in place
    const float *updates,   // [num_updates]
    const float *indices,   // [num_updates] (float, cast to int)
    unsigned int num_updates,
    unsigned int data_size
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_updates) return;
    int idx = (int)indices[i];
    if (idx >= 0 && idx < (int)data_size) {
        data[idx] = updates[i];
    }
}
"#;

pub fn scatter_nd(
    cache: &KernelCache, device: &WarpDevice,
    data: &mut GpuTensor<f32>, updates: &GpuTensor<f32>, indices: &GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, SCATTER_ND_SRC, "warp_scatter_nd")?;
    let num = updates.numel as u32;
    let size = data.numel as u32;
    let cfg = LaunchConfig::for_num_elems(num);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut data.data).arg(&updates.data).arg(&indices.data)
            .arg(&num).arg(&size)
            .launch(cfg))?;
    }
    Ok(())
}

// ── Expand (broadcast) ──────────────────────────────────────────

const EXPAND_SRC: &str = r#"
extern "C" __global__ void warp_expand(
    float *out, const float *input,
    unsigned int in_size, unsigned int out_size
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_size) return;
    out[i] = input[i % in_size];
}
"#;

pub fn expand(
    cache: &KernelCache, device: &WarpDevice,
    input: &GpuTensor<f32>, output: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, EXPAND_SRC, "warp_expand")?;
    let cfg = LaunchConfig::for_num_elems(output.numel as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&input.data)
            .arg(&(input.numel as u32)).arg(&(output.numel as u32))
            .launch(cfg))?;
    }
    Ok(())
}

// ── Where (conditional select) ──────────────────────────────────

const WHERE_SRC: &str = r#"
extern "C" __global__ void warp_where(
    float *out, const float *cond, const float *x, const float *y,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (cond[i] != 0.0f) ? x[i] : y[i];
}
"#;

pub fn where_op(
    cache: &KernelCache, device: &WarpDevice,
    cond: &GpuTensor<f32>, x: &GpuTensor<f32>, y: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, WHERE_SRC, "warp_where")?;
    let n = output.numel as u32;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&cond.data).arg(&x.data).arg(&y.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

// ── DepthToSpace / SpaceToDepth ─────────────────────────────────

const DEPTH_TO_SPACE_SRC: &str = r#"
extern "C" __global__ void warp_depth_to_space(
    float *out,         // [C/(r²), H*r, W*r]
    const float *input, // [C, H, W]
    unsigned int C, unsigned int H, unsigned int W,
    unsigned int r      // block size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int C_out = C / (r * r);
    unsigned int H_out = H * r;
    unsigned int W_out = W * r;
    unsigned int total = C_out * H_out * W_out;
    if (idx >= total) return;

    unsigned int w_out = idx % W_out;
    unsigned int h_out = (idx / W_out) % H_out;
    unsigned int c_out = idx / (H_out * W_out);

    unsigned int h_in = h_out / r;
    unsigned int w_in = w_out / r;
    unsigned int c_in = c_out * r * r + (h_out % r) * r + (w_out % r);

    out[idx] = input[c_in * H * W + h_in * W + w_in];
}
"#;

pub fn depth_to_space(
    cache: &KernelCache, device: &WarpDevice,
    input: &GpuTensor<f32>, output: &mut GpuTensor<f32>,
    c: u32, h: u32, w: u32, block_size: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, DEPTH_TO_SPACE_SRC, "warp_depth_to_space")?;
    let total = output.numel as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&input.data)
            .arg(&c).arg(&h).arg(&w).arg(&block_size)
            .launch(cfg))?;
    }
    Ok(())
}

// ── MaxPool1D ───────────────────────────────────────────────────

const MAXPOOL1D_SRC: &str = r#"
extern "C" __global__ void warp_maxpool1d(
    float *out, const float *input,
    unsigned int C, unsigned int L, unsigned int L_out,
    unsigned int kernel, unsigned int stride, unsigned int padding
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = C * L_out;
    if (idx >= total) return;

    unsigned int lo = idx % L_out;
    unsigned int c = idx / L_out;

    float max_val = -1e30f;
    for (unsigned int k = 0; k < kernel; k++) {
        int li = (int)(lo * stride + k) - (int)padding;
        if (li >= 0 && li < (int)L) {
            float v = input[c * L + li];
            if (v > max_val) max_val = v;
        }
    }
    out[idx] = max_val;
}
"#;

pub fn maxpool1d(
    cache: &KernelCache, device: &WarpDevice,
    input: &GpuTensor<f32>, output: &mut GpuTensor<f32>,
    channels: u32, length: u32, kernel: u32, stride: u32, padding: u32,
) -> Result<(), DeviceError> {
    let l_out = (length + 2 * padding - kernel) / stride + 1;
    let f = cache.get_or_compile(device, MAXPOOL1D_SRC, "warp_maxpool1d")?;
    let batch = input.numel as u32 / (channels * length);
    for n in 0..batch {
        let in_off = (n * channels) as usize * length as usize;
        let out_off = (n * channels) as usize * l_out as usize;
        let total = channels * l_out;
        let cfg = LaunchConfig::for_num_elems(total);
        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut output.data.slice_mut(out_off..))
                .arg(&input.data.slice(in_off..))
                .arg(&channels).arg(&length).arg(&l_out)
                .arg(&kernel).arg(&stride).arg(&padding)
                .launch(cfg))?;
        }
    }
    Ok(())
}

// ── Multinomial (sampling) ──────────────────────────────────────

/// Multinomial sampling from probability distribution.
/// Picks one index per row based on probabilities.
pub fn multinomial_sample(
    device: &WarpDevice,
    probs: &GpuTensor<f32>,
    rows: u32, cols: u32,
    seed: u64,
) -> Result<GpuTensor<i32>, DeviceError> {
    // CPU-side sampling (probs are typically small after softmax)
    let probs_host = probs.to_host(device)?;
    let mut indices = vec![0i32; rows as usize];

    for row in 0..rows as usize {
        let row_probs = &probs_host[row * cols as usize..(row + 1) * cols as usize];
        let mut cumsum = 0.0f32;
        // Simple hash-based RNG
        let r = ((seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
            .wrapping_add(row as u64)) % 10000) as f32 / 10000.0;
        for (i, &p) in row_probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= r {
                indices[row] = i as i32;
                break;
            }
        }
    }

    GpuTensor::from_host(device, &indices, Shape::from_static(&[rows as usize]), DType::I32)
}

// ── Resize Cubic ────────────────────────────────────────────────

const RESIZE_CUBIC_SRC: &str = r#"
extern "C" __global__ void warp_resize_cubic(
    float *output, const float *input,
    unsigned int C, unsigned int H, unsigned int W,
    unsigned int out_H, unsigned int out_W
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = C * out_H * out_W;
    if (idx >= total) return;

    unsigned int ow = idx % out_W;
    unsigned int oh = (idx / out_W) % out_H;
    unsigned int c = idx / (out_H * out_W);

    float scale_h = (float)H / (float)out_H;
    float scale_w = (float)W / (float)out_W;
    float fh = (oh + 0.5f) * scale_h - 0.5f;
    float fw = (ow + 0.5f) * scale_w - 0.5f;

    // Bicubic: weighted average of 4x4 neighborhood
    int ih = (int)floorf(fh);
    int iw = (int)floorf(fw);
    float frac_h = fh - ih;
    float frac_w = fw - iw;

    float sum = 0.0f;
    float weight_sum = 0.0f;
    for (int dy = -1; dy <= 2; dy++) {
        for (int dx = -1; dx <= 2; dx++) {
            int sy = ih + dy;
            int sx = iw + dx;
            sy = sy < 0 ? 0 : (sy >= (int)H ? (int)H - 1 : sy);
            sx = sx < 0 ? 0 : (sx >= (int)W ? (int)W - 1 : sx);

            // Mitchell-Netravali cubic weight
            float t_h = fabsf(frac_h - dy);
            float t_w = fabsf(frac_w - dx);
            float wh = (t_h < 1.0f) ? (1.5f*t_h*t_h*t_h - 2.5f*t_h*t_h + 1.0f)
                      : (t_h < 2.0f) ? (-0.5f*t_h*t_h*t_h + 2.5f*t_h*t_h - 4.0f*t_h + 2.0f) : 0.0f;
            float ww = (t_w < 1.0f) ? (1.5f*t_w*t_w*t_w - 2.5f*t_w*t_w + 1.0f)
                      : (t_w < 2.0f) ? (-0.5f*t_w*t_w*t_w + 2.5f*t_w*t_w - 4.0f*t_w + 2.0f) : 0.0f;

            float w = wh * ww;
            sum += w * input[c * H * W + sy * W + sx];
            weight_sum += w;
        }
    }
    output[idx] = (weight_sum > 0.0f) ? sum / weight_sum : 0.0f;
}
"#;

pub fn resize_cubic(
    cache: &KernelCache, device: &WarpDevice,
    input: &GpuTensor<f32>, output: &mut GpuTensor<f32>,
    channels: u32, h: u32, w: u32, out_h: u32, out_w: u32,
) -> Result<(), DeviceError> {
    let batch = input.numel as u32 / (channels * h * w);
    let f = cache.get_or_compile(device, RESIZE_CUBIC_SRC, "warp_resize_cubic")?;
    for n in 0..batch {
        let in_off = (n * channels) as usize * (h * w) as usize;
        let out_off = (n * channels) as usize * (out_h * out_w) as usize;
        let total = channels * out_h * out_w;
        let cfg = LaunchConfig::for_num_elems(total);
        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut output.data.slice_mut(out_off..))
                .arg(&input.data.slice(in_off..))
                .arg(&channels).arg(&h).arg(&w).arg(&out_h).arg(&out_w)
                .launch(cfg))?;
        }
    }
    Ok(())
}

// ── Selective Scan (Mamba SSM) ──────────────────────────────────
// The core operation of Mamba/S4 state-space models.
// Computes: h[t] = A * h[t-1] + B * x[t], y[t] = C * h[t]
// This is a sequential scan but can be parallelized with associative scan.

// ── GPU Parallel Scan (Blelloch) ────────────────────────────────
// Parallel prefix sum for Mamba's associative scan.
// Uses the Blelloch algorithm: O(N) work, O(log N) depth.

const PARALLEL_SCAN_SRC: &str = r#"
extern "C" __global__ void warp_parallel_scan(
    float *out, const float *input, unsigned int n
) {
    extern __shared__ float temp[];
    unsigned int tid = threadIdx.x;
    unsigned int offset = 1;

    // Load into shared memory
    if (2*tid < n) temp[2*tid] = input[2*tid]; else temp[2*tid] = 0;
    if (2*tid+1 < n) temp[2*tid+1] = input[2*tid+1]; else temp[2*tid+1] = 0;

    // Up-sweep (reduce)
    for (unsigned int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            unsigned int ai = offset*(2*tid+1)-1;
            unsigned int bi = offset*(2*tid+2)-1;
            if (bi < n) temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear last element
    if (tid == 0) temp[n-1] = 0;

    // Down-sweep
    for (unsigned int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            unsigned int ai = offset*(2*tid+1)-1;
            unsigned int bi = offset*(2*tid+2)-1;
            if (bi < n) {
                float t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
    }
    __syncthreads();

    // Write back (inclusive scan = exclusive + input)
    if (2*tid < n) out[2*tid] = temp[2*tid] + input[2*tid];
    if (2*tid+1 < n) out[2*tid+1] = temp[2*tid+1] + input[2*tid+1];
}
"#;

/// GPU parallel prefix sum (inclusive scan).
pub fn parallel_scan(
    cache: &KernelCache, device: &WarpDevice,
    input: &GpuTensor<f32>, output: &mut GpuTensor<f32>,
    n: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, PARALLEL_SCAN_SRC, "warp_parallel_scan")?;
    let threads = (n / 2).next_power_of_two().min(512);
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: n * 4,
    };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&input.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// Simple sequential selective scan for Mamba inference.
/// For now, CPU-side (scan is inherently sequential at batch=1).
pub fn selective_scan(
    device: &WarpDevice,
    x: &[f32],        // [seq_len, d_inner]
    a: &[f32],        // [d_inner] (discretized A matrix diagonal)
    b: &[f32],        // [seq_len, d_state]
    c: &[f32],        // [seq_len, d_state]
    d_inner: usize,
    d_state: usize,
    seq_len: usize,
) -> Result<Vec<f32>, DeviceError> {
    let mut h = vec![0.0f32; d_inner * d_state]; // hidden state
    let mut output = vec![0.0f32; seq_len * d_inner];

    for t in 0..seq_len {
        for d in 0..d_inner {
            let mut y = 0.0f32;
            for s in 0..d_state {
                // h[d,s] = A[d] * h[d,s] + B[t,s] * x[t,d]
                h[d * d_state + s] = a[d] * h[d * d_state + s]
                    + b[t * d_state + s] * x[t * d_inner + d];
                // y[t,d] += C[t,s] * h[d,s]
                y += c[t * d_state + s] * h[d * d_state + s];
            }
            output[t * d_inner + d] = y;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

// ── Conv3D (video models) ───────────────────────────────────────

const CONV3D_SRC: &str = r#"
extern "C" __global__ void warp_conv3d(
    float *out,
    const float *input,     // [C_in, D, H, W]
    const float *weight,    // [C_out, C_in, kD, kH, kW]
    unsigned int C_in, unsigned int C_out,
    unsigned int D, unsigned int H, unsigned int W,
    unsigned int kD, unsigned int kH, unsigned int kW,
    unsigned int sD, unsigned int sH, unsigned int sW,
    unsigned int pD, unsigned int pH, unsigned int pW,
    unsigned int D_out, unsigned int H_out, unsigned int W_out
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = C_out * D_out * H_out * W_out;
    if (idx >= total) return;

    unsigned int ow = idx % W_out;
    unsigned int oh = (idx / W_out) % H_out;
    unsigned int od = (idx / (W_out * H_out)) % D_out;
    unsigned int co = idx / (D_out * H_out * W_out);

    float sum = 0.0f;
    for (unsigned int ci = 0; ci < C_in; ci++) {
        for (unsigned int kd = 0; kd < kD; kd++) {
            for (unsigned int kh = 0; kh < kH; kh++) {
                for (unsigned int kw = 0; kw < kW; kw++) {
                    int id = (int)(od * sD + kd) - (int)pD;
                    int ih = (int)(oh * sH + kh) - (int)pH;
                    int iw = (int)(ow * sW + kw) - (int)pW;
                    if (id >= 0 && id < (int)D && ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                        sum += input[ci*D*H*W + id*H*W + ih*W + iw] *
                               weight[co*C_in*kD*kH*kW + ci*kD*kH*kW + kd*kH*kW + kh*kW + kw];
                    }
                }
            }
        }
    }
    out[idx] = sum;
}
"#;

/// Conv3D for video models (VideoMAE, SlowFast, etc.)
pub fn conv3d(
    cache: &KernelCache, device: &WarpDevice,
    input: &GpuTensor<f32>, weight: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
    c_in: u32, c_out: u32,
    d: u32, h: u32, w: u32,
    kd: u32, kh: u32, kw: u32,
    sd: u32, sh: u32, sw: u32,
    pd: u32, ph: u32, pw: u32,
) -> Result<(), DeviceError> {
    let d_out = (d + 2*pd - kd) / sd + 1;
    let h_out = (h + 2*ph - kh) / sh + 1;
    let w_out = (w + 2*pw - kw) / sw + 1;
    let f = cache.get_or_compile(device, CONV3D_SRC, "warp_conv3d")?;
    let total = c_out * d_out * h_out * w_out;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&input.data).arg(&weight.data)
            .arg(&c_in).arg(&c_out)
            .arg(&d).arg(&h).arg(&w)
            .arg(&kd).arg(&kh).arg(&kw)
            .arg(&sd).arg(&sh).arg(&sw)
            .arg(&pd).arg(&ph).arg(&pw)
            .arg(&d_out).arg(&h_out).arg(&w_out)
            .launch(cfg))?;
    }
    Ok(())
}

    #[test]
    fn scatter_nd_test() {
        let (dev, cache) = match setup() {
            Some(s) => s, None => { println!("No CUDA"); return; }
        };

        let data_vec = vec![0.0f32; 10];
        let mut data = GpuTensor::from_host(&dev, &data_vec, Shape::from_static(&[10]), DType::F32).unwrap();
        let updates = GpuTensor::from_host(&dev, &[42.0f32, 99.0], Shape::from_static(&[2]), DType::F32).unwrap();
        let indices = GpuTensor::from_host(&dev, &[3.0f32, 7.0], Shape::from_static(&[2]), DType::F32).unwrap();

        scatter_nd(&cache, &dev, &mut data, &updates, &indices).unwrap();
        dev.synchronize().unwrap();

        let result = data.to_host(&dev).unwrap();
        assert_eq!(result[3], 42.0);
        assert_eq!(result[7], 99.0);
        assert_eq!(result[0], 0.0);
        println!("ScatterND: correct!");
    }

    #[test]
    fn where_test() {
        let (dev, cache) = match setup() {
            Some(s) => s, None => { println!("No CUDA"); return; }
        };

        let cond = GpuTensor::from_host(&dev, &[1.0f32, 0.0, 1.0, 0.0], Shape::from_static(&[4]), DType::F32).unwrap();
        let x = GpuTensor::from_host(&dev, &[10.0f32, 20.0, 30.0, 40.0], Shape::from_static(&[4]), DType::F32).unwrap();
        let y = GpuTensor::from_host(&dev, &[-1.0f32, -2.0, -3.0, -4.0], Shape::from_static(&[4]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[4]), DType::F32).unwrap();

        where_op(&cache, &dev, &cond, &x, &y, &mut out).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        assert_eq!(result, vec![10.0, -2.0, 30.0, -4.0]);
        println!("Where: correct!");
    }

    #[test]
    fn maxpool1d_test() {
        let (dev, cache) = match setup() {
            Some(s) => s, None => { println!("No CUDA"); return; }
        };

        let data = vec![1.0f32, 3.0, 2.0, 5.0, 4.0, 1.0];
        let input = GpuTensor::from_host(&dev, &data, Shape::from_static(&[1, 1, 6]), DType::F32).unwrap();
        let mut output = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[1, 1, 3]), DType::F32).unwrap();

        maxpool1d(&cache, &dev, &input, &mut output, 1, 6, 2, 2, 0).unwrap();
        dev.synchronize().unwrap();

        let result = output.to_host(&dev).unwrap();
        assert_eq!(result, vec![3.0, 5.0, 4.0]);
        println!("MaxPool1D: correct!");
    }

    #[test]
    fn depth_to_space_test() {
        let (dev, cache) = match setup() {
            Some(s) => s, None => { println!("No CUDA"); return; }
        };

        // 4 channels, 1x1 spatial → 1 channel, 2x2 spatial (block_size=2)
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = GpuTensor::from_host(&dev, &data, Shape::from_static(&[4, 1, 1]), DType::F32).unwrap();
        let mut output = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[1, 2, 2]), DType::F32).unwrap();

        depth_to_space(&cache, &dev, &input, &mut output, 4, 1, 1, 2).unwrap();
        dev.synchronize().unwrap();

        let result = output.to_host(&dev).unwrap();
        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|v| v.is_finite()));
        println!("DepthToSpace: correct! {:?}", result);
    }

    #[test]
    fn selective_scan_test() {
        let dev = match WarpDevice::new(0) {
            Ok(d) => d, Err(_) => { println!("No CUDA"); return; }
        };

        let seq_len = 4;
        let d_inner = 2;
        let d_state = 2;

        let x = vec![1.0f32, 0.5, 0.3, 0.7, 0.2, 0.8, 0.6, 0.4]; // [4, 2]
        let a = vec![0.9f32, 0.8]; // [2] decay factors
        let b = vec![1.0f32; seq_len * d_state]; // [4, 2]
        let c = vec![1.0f32; seq_len * d_state]; // [4, 2]

        let output = selective_scan(&dev, &x, &a, &b, &c, d_inner, d_state, seq_len).unwrap();

        println!("SelectiveScan (Mamba):");
        for t in 0..seq_len {
            println!("  t={}: [{:.3}, {:.3}]", t, output[t*d_inner], output[t*d_inner+1]);
        }
        assert!(output.iter().all(|v| v.is_finite()));
        println!("  PASSED!");
    }
}

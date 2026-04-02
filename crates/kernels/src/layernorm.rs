//! LayerNorm GPU kernel.
//!
//! LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta
//!
//! Unlike RMSNorm (which only divides by the root-mean-square), LayerNorm
//! subtracts the mean first and applies both scale (gamma) and shift (beta).
//! Used by GPT-2, GPT-J, BERT, Phi, and many non-LLaMA architectures.
//!
//! We use a two-pass approach per row:
//! 1. Compute mean and variance via Welford's online algorithm
//! 2. Normalize and apply affine transform
//!
//! Single block per row, warp reduction for statistics.

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

/// LayerNorm with gamma (scale) and beta (bias).
/// x: [n_rows, hidden_size], gamma: [hidden_size], beta: [hidden_size]
/// out: [n_rows, hidden_size]
const LAYERNORM_SRC: &str = r#"
extern "C" __global__ void warp_layernorm(
    float *out,
    const float *x,
    const float *gamma,
    const float *beta,
    unsigned int hidden_size,
    float eps,
    unsigned int n_rows
) {
    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const float *x_row = x + row * hidden_size;
    float *out_row = out + row * hidden_size;

    // Pass 1: compute mean
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum += x_row[i];
    }
    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    sum = __shfl_sync(0xffffffff, sum, 0);
    float mean = sum / (float)hidden_size;

    // Pass 2: compute variance
    float var_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = x_row[i] - mean;
        var_sum += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    var_sum = __shfl_sync(0xffffffff, var_sum, 0);
    float inv_std = rsqrtf(var_sum / (float)hidden_size + eps);

    // Pass 3: normalize and apply affine transform
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = (x_row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
"#;

/// LayerNorm without bias (gamma only, no beta).
/// Some architectures (e.g., GPT-NeoX variants) use this.
const LAYERNORM_NO_BIAS_SRC: &str = r#"
extern "C" __global__ void warp_layernorm_no_bias(
    float *out,
    const float *x,
    const float *gamma,
    unsigned int hidden_size,
    float eps,
    unsigned int n_rows
) {
    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const float *x_row = x + row * hidden_size;
    float *out_row = out + row * hidden_size;

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum += x_row[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    sum = __shfl_sync(0xffffffff, sum, 0);
    float mean = sum / (float)hidden_size;

    float var_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = x_row[i] - mean;
        var_sum += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    var_sum = __shfl_sync(0xffffffff, var_sum, 0);
    float inv_std = rsqrtf(var_sum / (float)hidden_size + eps);

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = (x_row[i] - mean) * inv_std * gamma[i];
    }
}
"#;

/// Fused Residual + LayerNorm: out = LayerNorm(residual + x, gamma, beta)
/// Avoids an extra add kernel launch + memory round-trip.
const FUSED_RESIDUAL_LAYERNORM_SRC: &str = r#"
extern "C" __global__ void warp_fused_residual_layernorm(
    float *out,
    float *residual_out,
    const float *residual,
    const float *x,
    const float *gamma,
    const float *beta,
    unsigned int hidden_size,
    float eps,
    unsigned int n_rows
) {
    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const float *res_row = residual + row * hidden_size;
    const float *x_row = x + row * hidden_size;
    float *out_row = out + row * hidden_size;
    float *res_out_row = residual_out + row * hidden_size;

    // Compute residual + x and mean in one pass
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = res_row[i] + x_row[i];
        res_out_row[i] = v;  // store the residual sum for skip connection
        sum += v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    sum = __shfl_sync(0xffffffff, sum, 0);
    float mean = sum / (float)hidden_size;

    float var_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = res_out_row[i] - mean;
        var_sum += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    var_sum = __shfl_sync(0xffffffff, var_sum, 0);
    float inv_std = rsqrtf(var_sum / (float)hidden_size + eps);

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = (res_out_row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
"#;

/// LayerNorm with gamma and beta.
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
    let n_rows = (x.numel / hidden_size as usize) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_rows, 1, 1),
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

/// LayerNorm without bias (gamma only).
pub fn layernorm_no_bias(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    gamma: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    hidden_size: u32,
    eps: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, LAYERNORM_NO_BIAS_SRC, "warp_layernorm_no_bias")?;
    let n_rows = (x.numel / hidden_size as usize) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_rows, 1, 1),
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

/// Fused residual add + LayerNorm.
/// Writes both the residual sum (for skip connections) and the normalized output.
pub fn fused_residual_layernorm(
    cache: &KernelCache,
    device: &WarpDevice,
    residual: &GpuTensor<f32>,
    x: &GpuTensor<f32>,
    gamma: &GpuTensor<f32>,
    beta: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    residual_out: &mut GpuTensor<f32>,
    hidden_size: u32,
    eps: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, FUSED_RESIDUAL_LAYERNORM_SRC, "warp_fused_residual_layernorm")?;
    let n_rows = (x.numel / hidden_size as usize) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_rows, 1, 1),
        block_dim: (32.min(hidden_size), 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&mut residual_out.data)
            .arg(&residual.data)
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

// ── CPU references ──────────────────────────────────────────────

/// CPU reference LayerNorm for testing.
pub fn cpu_layernorm(
    x: &[f32], gamma: &[f32], beta: &[f32],
    out: &mut [f32], hidden_size: usize, eps: f32,
) {
    let n_rows = x.len() / hidden_size;
    for row in 0..n_rows {
        let start = row * hidden_size;
        let end = start + hidden_size;
        let x_row = &x[start..end];

        let mean: f32 = x_row.iter().sum::<f32>() / hidden_size as f32;
        let var: f32 = x_row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden_size as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for i in 0..hidden_size {
            out[start + i] = (x_row[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

/// CPU reference LayerNorm without bias.
pub fn cpu_layernorm_no_bias(
    x: &[f32], gamma: &[f32],
    out: &mut [f32], hidden_size: usize, eps: f32,
) {
    let n_rows = x.len() / hidden_size;
    for row in 0..n_rows {
        let start = row * hidden_size;
        let end = start + hidden_size;
        let x_row = &x[start..end];

        let mean: f32 = x_row.iter().sum::<f32>() / hidden_size as f32;
        let var: f32 = x_row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden_size as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for i in 0..hidden_size {
            out[start + i] = (x_row[i] - mean) * inv_std * gamma[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::{DType, Shape};

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    #[test]
    fn layernorm_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let hidden = 32u32;
        let rows = 8usize;
        let n = rows * hidden as usize;
        let eps = 1e-5f32;

        let x_data: Vec<f32> = (0..n).map(|i| ((i % 37) as f32 - 18.0) * 0.1).collect();
        let gamma_data: Vec<f32> = (0..hidden as usize).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
        let beta_data: Vec<f32> = (0..hidden as usize).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

        // CPU reference
        let mut cpu_out = vec![0.0f32; n];
        cpu_layernorm(&x_data, &gamma_data, &beta_data, &mut cpu_out, hidden as usize, eps);

        // GPU
        let shape = Shape::from_static(&[rows, hidden as usize]);
        let x = GpuTensor::from_host(&dev, &x_data, shape.clone(), DType::F32).unwrap();
        let gamma = GpuTensor::from_host(&dev, &gamma_data, Shape::from_static(&[hidden as usize]), DType::F32).unwrap();
        let beta = GpuTensor::from_host(&dev, &beta_data, Shape::from_static(&[hidden as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        layernorm(&cache, &dev, &x, &gamma, &beta, &mut out, hidden, eps).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();
        let max_err: f32 = gpu_out.iter().zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("LayerNorm {rows}x{hidden}: max error = {max_err:.6}");
        assert!(max_err < 1e-3, "Max error {max_err} too high");
    }

    #[test]
    fn layernorm_no_bias_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let hidden = 64u32;
        let rows = 4usize;
        let n = rows * hidden as usize;
        let eps = 1e-5f32;

        let x_data: Vec<f32> = (0..n).map(|i| ((i % 41) as f32 - 20.0) * 0.08).collect();
        let gamma_data: Vec<f32> = (0..hidden as usize).map(|i| 0.8 + (i % 3) as f32 * 0.1).collect();

        let mut cpu_out = vec![0.0f32; n];
        cpu_layernorm_no_bias(&x_data, &gamma_data, &mut cpu_out, hidden as usize, eps);

        let shape = Shape::from_static(&[rows, hidden as usize]);
        let x = GpuTensor::from_host(&dev, &x_data, shape.clone(), DType::F32).unwrap();
        let gamma = GpuTensor::from_host(&dev, &gamma_data, Shape::from_static(&[hidden as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        layernorm_no_bias(&cache, &dev, &x, &gamma, &mut out, hidden, eps).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();
        let max_err: f32 = gpu_out.iter().zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("LayerNorm (no bias) {rows}x{hidden}: max error = {max_err:.6}");
        assert!(max_err < 1e-3, "Max error {max_err} too high");
    }

    #[test]
    fn fused_residual_layernorm_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let hidden = 32u32;
        let rows = 4usize;
        let n = rows * hidden as usize;
        let eps = 1e-5f32;

        let res_data: Vec<f32> = (0..n).map(|i| ((i % 29) as f32 - 14.0) * 0.05).collect();
        let x_data: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.07).collect();
        let gamma_data: Vec<f32> = (0..hidden as usize).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
        let beta_data: Vec<f32> = (0..hidden as usize).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

        // CPU reference: residual + x, then layernorm
        let combined: Vec<f32> = res_data.iter().zip(x_data.iter()).map(|(a, b)| a + b).collect();
        let mut cpu_out = vec![0.0f32; n];
        cpu_layernorm(&combined, &gamma_data, &beta_data, &mut cpu_out, hidden as usize, eps);

        let shape = Shape::from_static(&[rows, hidden as usize]);
        let res = GpuTensor::from_host(&dev, &res_data, shape.clone(), DType::F32).unwrap();
        let x = GpuTensor::from_host(&dev, &x_data, shape.clone(), DType::F32).unwrap();
        let gamma = GpuTensor::from_host(&dev, &gamma_data, Shape::from_static(&[hidden as usize]), DType::F32).unwrap();
        let beta = GpuTensor::from_host(&dev, &beta_data, Shape::from_static(&[hidden as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape.clone(), DType::F32).unwrap();
        let mut res_out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        fused_residual_layernorm(&cache, &dev, &res, &x, &gamma, &beta, &mut out, &mut res_out, hidden, eps).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();
        let gpu_res_out = res_out.to_host(&dev).unwrap();

        // Check residual sum is correct
        let res_err: f32 = gpu_res_out.iter().zip(combined.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(res_err < 1e-5, "Residual sum error {res_err}");

        // Check normalized output
        let max_err: f32 = gpu_out.iter().zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Fused Residual+LayerNorm {rows}x{hidden}: max error = {max_err:.6}");
        assert!(max_err < 1e-3, "Max error {max_err} too high");
    }

    #[test]
    fn layernorm_perf() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // Realistic: 4096 hidden, 512 tokens
        let hidden = 4096u32;
        let rows = 512usize;
        let n = rows * hidden as usize;
        let eps = 1e-5f32;

        let x_data: Vec<f32> = (0..n).map(|i| ((i % 100) as f32 - 50.0) * 0.01).collect();
        let gamma_data: Vec<f32> = vec![1.0; hidden as usize];
        let beta_data: Vec<f32> = vec![0.0; hidden as usize];

        let shape = Shape::from_static(&[rows, hidden as usize]);
        let x = GpuTensor::from_host(&dev, &x_data, shape.clone(), DType::F32).unwrap();
        let gamma = GpuTensor::from_host(&dev, &gamma_data, Shape::from_static(&[hidden as usize]), DType::F32).unwrap();
        let beta = GpuTensor::from_host(&dev, &beta_data, Shape::from_static(&[hidden as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        // Warmup
        layernorm(&cache, &dev, &x, &gamma, &beta, &mut out, hidden, eps).unwrap();
        dev.synchronize().unwrap();

        let iters = 500;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            layernorm(&cache, &dev, &x, &gamma, &beta, &mut out, hidden, eps).unwrap();
        }
        dev.synchronize().unwrap();
        let elapsed = start.elapsed();

        let bytes = n as f64 * 4.0 * 3.0; // read x + gamma, write out
        let bandwidth = bytes * iters as f64 / elapsed.as_secs_f64() / 1e9;

        println!("\nLayerNorm perf ({rows}x{hidden}):");
        println!("  {:.3}μs avg ({iters} iters)", elapsed.as_secs_f64() * 1e6 / iters as f64);
        println!("  {:.1} GB/s effective bandwidth", bandwidth);
    }
}

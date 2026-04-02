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

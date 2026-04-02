//! cuBLAS GEMM benchmark — the performance target.
//!
//! cuBLAS is NVIDIA's hand-tuned BLAS library. It's the gold standard
//! for GEMM performance on NVIDIA GPUs. Our goal is to get within
//! 85% of cuBLAS on FP32, then beat it on fused operations.

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::CudaSlice;
use std::time::Instant;

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Run a cuBLAS SGEMM and return TFLOPS.
pub fn cublas_sgemm_bench(
    device: &WarpDevice,
    m: i32,
    n: i32,
    k: i32,
    iters: usize,
) -> Result<BenchResult, DeviceError> {
    let blas = CudaBlas::new(device.stream.clone())
        .map_err(|e| DeviceError::Launch(e.to_string()))?;

    let a_data: Vec<f32> = (0..(m * k) as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let b_data: Vec<f32> = (0..(k * n) as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();

    let a_dev = device.htod(&a_data)?;
    let b_dev = device.htod(&b_data)?;
    let mut c_dev: CudaSlice<f32> = device.alloc_zeros((m * n) as usize)?;

    // cuBLAS uses column-major, but we want row-major C = A × B.
    // Trick: compute C^T = B^T × A^T in column-major, which gives us C in row-major.
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n,  // columns of C (row-major → rows of C^T)
        n: m,  // rows of C
        k,
        alpha: 1.0f32,
        lda: n,  // leading dim of B (treated as A in col-major)
        ldb: k,  // leading dim of A
        beta: 0.0f32,
        ldc: n,  // leading dim of C
    };

    // Warmup
    unsafe { blas.gemm(cfg, &b_dev, &a_dev, &mut c_dev) }
        .map_err(|e| DeviceError::Launch(e.to_string()))?;
    device.synchronize()?;

    // Timed
    let start = Instant::now();
    for _ in 0..iters {
        unsafe { blas.gemm(cfg, &b_dev, &a_dev, &mut c_dev) }
            .map_err(|e| DeviceError::Launch(e.to_string()))?;
    }
    device.synchronize()?;
    let elapsed = start.elapsed();

    let flops_per = 2.0 * m as f64 * n as f64 * k as f64;
    let total_flops = flops_per * iters as f64;
    let tflops = total_flops / elapsed.as_secs_f64() / 1e12;
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;

    Ok(BenchResult { tflops, avg_ms, iters })
}

/// Run a cuBLAS FP16 GEMM (uses Tensor Cores) and return TFLOPS.
pub fn cublas_hgemm_bench(
    device: &WarpDevice,
    m: i32,
    n: i32,
    k: i32,
    iters: usize,
) -> Result<BenchResult, DeviceError> {
    let blas = CudaBlas::new(device.stream.clone())
        .map_err(|e| DeviceError::Launch(e.to_string()))?;

    let a_data: Vec<half::f16> = (0..(m * k) as usize)
        .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.01))
        .collect();
    let b_data: Vec<half::f16> = (0..(k * n) as usize)
        .map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.01))
        .collect();

    let a_dev = device.htod(&a_data)?;
    let b_dev = device.htod(&b_data)?;
    let mut c_dev: CudaSlice<half::f16> = device.alloc_zeros((m * n) as usize)?;

    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n,
        n: m,
        k,
        alpha: half::f16::from_f32(1.0),
        lda: n,
        ldb: k,
        beta: half::f16::from_f32(0.0),
        ldc: n,
    };

    // Warmup
    unsafe { blas.gemm(cfg, &b_dev, &a_dev, &mut c_dev) }
        .map_err(|e| DeviceError::Launch(e.to_string()))?;
    device.synchronize()?;

    let start = Instant::now();
    for _ in 0..iters {
        unsafe { blas.gemm(cfg, &b_dev, &a_dev, &mut c_dev) }
            .map_err(|e| DeviceError::Launch(e.to_string()))?;
    }
    device.synchronize()?;
    let elapsed = start.elapsed();

    let flops_per = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = flops_per * iters as f64 / elapsed.as_secs_f64() / 1e12;
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;

    Ok(BenchResult { tflops, avg_ms, iters })
}

/// Compare our tiled GEMM against cuBLAS.
pub fn compare_gemm(
    device: &WarpDevice,
    cache: &KernelCache,
    m: u32,
    n: u32,
    k: u32,
    iters: usize,
) -> Result<ComparisonResult, DeviceError> {
    // Our GEMM
    let a_data: Vec<f32> = (0..(m * k) as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let b_data: Vec<f32> = (0..(k * n) as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();

    let a = GpuTensor::from_host(device, &a_data, warp_ir::Shape::from_static(&[m as usize, k as usize]), warp_ir::DType::F32)?;
    let b = GpuTensor::from_host(device, &b_data, warp_ir::Shape::from_static(&[k as usize, n as usize]), warp_ir::DType::F32)?;
    let mut c = GpuTensor::<f32>::zeros(device, warp_ir::Shape::from_static(&[m as usize, n as usize]), warp_ir::DType::F32)?;

    // Warmup
    crate::ops::gemm(cache, device, &a, &b, &mut c, m, n, k)?;
    device.synchronize()?;

    let start = Instant::now();
    for _ in 0..iters {
        crate::ops::gemm(cache, device, &a, &b, &mut c, m, n, k)?;
    }
    device.synchronize()?;
    let ours_elapsed = start.elapsed();

    let flops_per = 2.0 * m as f64 * n as f64 * k as f64;
    let ours_tflops = flops_per * iters as f64 / ours_elapsed.as_secs_f64() / 1e12;

    // cuBLAS
    let cublas = cublas_sgemm_bench(device, m as i32, n as i32, k as i32, iters)?;

    let ratio = ours_tflops / cublas.tflops.max(1e-9);
    Ok(ComparisonResult {
        ours: BenchResult {
            tflops: ours_tflops,
            avg_ms: ours_elapsed.as_secs_f64() * 1000.0 / iters as f64,
            iters,
        },
        cublas,
        ratio,
    })
}

#[derive(Debug, Clone)]
pub struct BenchResult {
    pub tflops: f64,
    pub avg_ms: f64,
    pub iters: usize,
}

#[derive(Debug)]
pub struct ComparisonResult {
    pub ours: BenchResult,
    pub cublas: BenchResult,
    /// Our TFLOPS / cuBLAS TFLOPS (1.0 = parity)
    pub ratio: f64,
}

impl std::fmt::Display for ComparisonResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  TensorWarp: {:.3} TFLOPS ({:.3}ms avg)", self.ours.tflops, self.ours.avg_ms)?;
        writeln!(f, "  cuBLAS:     {:.3} TFLOPS ({:.3}ms avg)", self.cublas.tflops, self.cublas.avg_ms)?;
        write!(f, "  Ratio:      {:.1}% of cuBLAS", self.ratio * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        let dev = WarpDevice::new(0).ok()?;
        Some((dev, KernelCache::new()))
    }

    #[test]
    fn cublas_sgemm_perf() {
        let (dev, _) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        println!("\n=== cuBLAS SGEMM (FP32) Benchmarks ===");
        for &(m, n, k) in &[
            (128, 128, 128),
            (512, 512, 512),
            (1024, 1024, 1024),
            (4096, 4096, 4096),
        ] {
            let result = cublas_sgemm_bench(&dev, m, n, k, 50).unwrap();
            println!("  {m:4}x{n:4}x{k:4}: {:.3} TFLOPS ({:.3}ms)", result.tflops, result.avg_ms);
        }
    }

    #[test]
    fn cublas_hgemm_perf() {
        let (dev, _) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        println!("\n=== cuBLAS HGEMM (FP16 Tensor Cores) Benchmarks ===");
        for &(m, n, k) in &[
            (128, 128, 128),
            (512, 512, 512),
            (1024, 1024, 1024),
            (4096, 4096, 4096),
        ] {
            let result = cublas_hgemm_bench(&dev, m, n, k, 50).unwrap();
            println!("  {m:4}x{n:4}x{k:4}: {:.3} TFLOPS ({:.3}ms)", result.tflops, result.avg_ms);
        }
    }

    #[test]
    fn gemm_vs_cublas() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        println!("\n=== TensorWarp vs cuBLAS GEMM (FP32) ===");
        for &(m, n, k) in &[
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
        ] {
            let result = compare_gemm(&dev, &cache, m, n, k, 50).unwrap();
            println!("GEMM {m}x{n}x{k}:");
            println!("{result}");
        }
    }
}

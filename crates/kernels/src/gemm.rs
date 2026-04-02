//! GEMM (General Matrix Multiply) kernel.
//!
//! The most performance-critical kernel in any inference engine.
//! Every linear layer, every attention projection, every MLP is a GEMM.
//!
//! Current: naive GEMM via CUDA C for correctness, then we optimize.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Launch C = A × B on GPU.
/// A: [M, K], B: [K, N], C: [M, N]
pub fn launch_gemm(
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    c: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    // Naive kernel: one thread per output element
    // This is the correctness baseline — optimization comes in Phase 4+
    let cuda_src = r#"
extern "C" __global__ void warp_gemm(
    float *C, const float *A, const float *B,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
"#;

    let (_module, func) = device.load_cuda_source(cuda_src, "warp_gemm")?;

    let block_dim = 16u32;
    let grid_x = (n + block_dim - 1) / block_dim;
    let grid_y = (m + block_dim - 1) / block_dim;

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (block_dim, block_dim, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device.stream
            .launch_builder(&func)
            .arg(&mut c.data)
            .arg(&a.data)
            .arg(&b.data)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }

    Ok(())
}

/// Tiled GEMM with shared memory — the first real optimization.
/// Each thread block loads tiles of A and B into shared memory,
/// reducing global memory accesses by factor of TILE_SIZE.
pub fn launch_gemm_tiled(
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    c: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    let tile = 32u32;

    let cuda_src = format!(r#"
#define TILE {tile}

extern "C" __global__ void warp_gemm_tiled(
    float *C, const float *A, const float *B,
    unsigned int M, unsigned int N, unsigned int K
) {{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;

    unsigned int row = by * TILE + ty;
    unsigned int col = bx * TILE + tx;

    float sum = 0.0f;

    for (unsigned int t = 0; t < (K + TILE - 1) / TILE; t++) {{
        // Load tile of A into shared memory
        if (row < M && t * TILE + tx < K)
            As[ty][tx] = A[row * K + t * TILE + tx];
        else
            As[ty][tx] = 0.0f;

        // Load tile of B into shared memory
        if (t * TILE + ty < K && col < N)
            Bs[ty][tx] = B[(t * TILE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (unsigned int i = 0; i < TILE; i++) {{
            sum += As[ty][i] * Bs[i][tx];
        }}

        __syncthreads();
    }}

    if (row < M && col < N) {{
        C[row * N + col] = sum;
    }}
}}
"#);

    let (_module, func) = device.load_cuda_source(&cuda_src, "warp_gemm_tiled")?;

    let grid_x = (n + tile - 1) / tile;
    let grid_y = (m + tile - 1) / tile;

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0, // statically allocated in kernel
    };

    unsafe {
        device.stream
            .launch_builder(&func)
            .arg(&mut c.data)
            .arg(&a.data)
            .arg(&b.data)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }

    Ok(())
}

/// Naive CPU GEMM for reference/validation.
pub fn cpu_gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::{DType, Shape};

    fn get_device() -> Option<WarpDevice> {
        WarpDevice::new(0).ok()
    }

    #[test]
    fn cpu_gemm_reference() {
        let (m, n, k) = (4, 3, 2);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0f32; m * n];
        cpu_gemm(&a, &b, &mut c, m, n, k);
        assert_eq!(c, vec![9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 29.0, 40.0, 51.0, 39.0, 54.0, 69.0]);
    }

    #[test]
    fn gpu_gemm_naive_small() {
        let dev = match get_device() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (64u32, 128u32, 48u32);
        let a_data: Vec<f32> = (0..(m*k) as usize).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let b_data: Vec<f32> = (0..(k*n) as usize).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let mut c_ref = vec![0.0f32; (m*n) as usize];
        cpu_gemm(&a_data, &b_data, &mut c_ref, m as usize, n as usize, k as usize);

        let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let mut c = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

        launch_gemm(&dev, &a, &b, &mut c, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let c_gpu = c.to_host(&dev).unwrap();
        let max_err: f32 = c_gpu.iter().zip(c_ref.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Naive GEMM {m}x{n}x{k}: max error = {max_err:.6}");
        assert!(max_err < 1e-3, "Max error {max_err} too high");
    }

    #[test]
    fn gpu_gemm_tiled_small() {
        let dev = match get_device() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (64u32, 128u32, 48u32);
        let a_data: Vec<f32> = (0..(m*k) as usize).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let b_data: Vec<f32> = (0..(k*n) as usize).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let mut c_ref = vec![0.0f32; (m*n) as usize];
        cpu_gemm(&a_data, &b_data, &mut c_ref, m as usize, n as usize, k as usize);

        let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let mut c = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

        launch_gemm_tiled(&dev, &a, &b, &mut c, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let c_gpu = c.to_host(&dev).unwrap();
        let max_err: f32 = c_gpu.iter().zip(c_ref.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Tiled GEMM {m}x{n}x{k}: max error = {max_err:.6}");
        assert!(max_err < 1e-2, "Max error {max_err} too high");
    }

    #[test]
    fn gpu_gemm_perf_comparison() {
        let dev = match get_device() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (512u32, 512u32, 512u32);
        let a_data: Vec<f32> = (0..(m*k) as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let b_data: Vec<f32> = (0..(k*n) as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();

        let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let mut c = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

        let iters = 20;
        let flops_per_iter = 2.0 * m as f64 * n as f64 * k as f64;

        // Warmup + benchmark naive
        launch_gemm(&dev, &a, &b, &mut c, m, n, k).unwrap();
        dev.synchronize().unwrap();
        let start = std::time::Instant::now();
        for _ in 0..iters {
            launch_gemm(&dev, &a, &b, &mut c, m, n, k).unwrap();
        }
        dev.synchronize().unwrap();
        let naive_elapsed = start.elapsed();
        let naive_tflops = flops_per_iter * iters as f64 / naive_elapsed.as_secs_f64() / 1e12;

        // Warmup + benchmark tiled
        launch_gemm_tiled(&dev, &a, &b, &mut c, m, n, k).unwrap();
        dev.synchronize().unwrap();
        let start = std::time::Instant::now();
        for _ in 0..iters {
            launch_gemm_tiled(&dev, &a, &b, &mut c, m, n, k).unwrap();
        }
        dev.synchronize().unwrap();
        let tiled_elapsed = start.elapsed();
        let tiled_tflops = flops_per_iter * iters as f64 / tiled_elapsed.as_secs_f64() / 1e12;

        println!("GEMM {m}x{n}x{k} comparison:");
        println!("  Naive:  {:.2}ms avg, {:.3} TFLOPS", naive_elapsed.as_secs_f64() * 1000.0 / iters as f64, naive_tflops);
        println!("  Tiled:  {:.2}ms avg, {:.3} TFLOPS", tiled_elapsed.as_secs_f64() * 1000.0 / iters as f64, tiled_tflops);
        println!("  Speedup: {:.2}x", tiled_tflops / naive_tflops.max(0.001));
    }
}

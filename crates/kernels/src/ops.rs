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

/// Cached tiled GEMM: C = A × B
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

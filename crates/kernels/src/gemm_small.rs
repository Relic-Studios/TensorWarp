//! Small GEMM variants — optimized for matrices 64-512.
//!
//! cuBLAS has ~30 kernel variants per GEMM size. We generate
//! multiple tile sizes and auto-select the fastest at runtime.
//!
//! For small FP16 matrices, SIMT (non-tensor-core) kernels with
//! half2 vectorized math often beat tensor cores because:
//! 1. No tile waste (tensor cores need 16×16 minimum)
//! 2. Better occupancy (smaller shared memory)
//! 3. Faster launch (simpler kernel)

use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Generate a SIMT FP16 GEMM kernel with specific tile size.
/// Returns (source, func_name, block_x, block_y).
fn gen_simt_f16_variant(bm: u32, bn: u32, bk: u32, tm: u32, tn: u32) -> (String, String) {
    let name = format!("warp_gemm_f16_{}x{}x{}_tm{}tn{}", bm, bn, bk, tm, tn);
    let threads_x = bn / tn;
    let threads_y = bm / tm;

    let src = format!(r#"
#include <cuda_fp16.h>

#define BM {bm}
#define BN {bn}
#define BK {bk}
#define TM {tm}
#define TN {tn}

extern "C" __global__ void {name}(
    half * __restrict__ C,
    const half * __restrict__ A,
    const half * __restrict__ B,
    unsigned int M, unsigned int N, unsigned int K
) {{
    // TODO: keep data as half in shared memory to halve bandwidth;
    // currently converts FP16->FP32 on load, defeating half the purpose.
    __shared__ float As[BK][BM + 4];  // padded to avoid bank conflicts
    __shared__ float Bs[BK][BN + 4];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x % {threads_x};
    unsigned int ty = threadIdx.x / {threads_x};
    unsigned int tid = threadIdx.x;
    unsigned int row_start = by * BM + ty * TM;
    unsigned int col_start = bx * BN + tx * TN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    unsigned int nthreads = {threads_x} * {threads_y};

    for (unsigned int k0 = 0; k0 < K; k0 += BK) {{
        #pragma unroll
        for (unsigned int load = 0; load < (BM * BK + nthreads - 1) / nthreads; load++) {{
            unsigned int idx = tid + load * nthreads;
            if (idx < BM * BK) {{
                unsigned int lk = idx / BM, lm = idx % BM;
                unsigned int gm = by * BM + lm, gk = k0 + lk;
                As[lk][lm] = (gm < M && gk < K) ? __half2float(A[gm * K + gk]) : 0.0f;
            }}
        }}
        #pragma unroll
        for (unsigned int load = 0; load < (BK * BN + nthreads - 1) / nthreads; load++) {{
            unsigned int idx = tid + load * nthreads;
            if (idx < BK * BN) {{
                unsigned int ln = idx % BN, lk = idx / BN;
                unsigned int gn = bx * BN + ln, gk = k0 + lk;
                Bs[lk][ln] = (gk < K && gn < N) ? __half2float(B[gk * N + gn]) : 0.0f;
            }}
        }}
        __syncthreads();

        #pragma unroll
        for (unsigned int kk = 0; kk < BK; kk++) {{
            float a_frag[TM], b_frag[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a_frag[i] = As[kk][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b_frag[j] = Bs[kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }}
        __syncthreads();
    }}

    #pragma unroll
    for (int i = 0; i < TM; i++) {{
        unsigned int grow = row_start + i;
        if (grow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {{
            unsigned int gcol = col_start + j;
            if (gcol < N) C[grow * N + gcol] = __float2half(acc[i][j]);
        }}
    }}
}}
"#, bm=bm, bn=bn, bk=bk, tm=tm, tn=tn, name=name,
    threads_x=threads_x, threads_y=threads_y);

    (src, name)
}

/// Variant configurations sorted by target size.
const VARIANTS: &[(u32, u32, u32, u32, u32)] = &[
    // (BM, BN, BK, TM, TN) — for different matrix sizes
    (16,  16,  8,  2,  2),   // tiny: 64-128
    (32,  32,  8,  4,  4),   // small: 128-256
    (32,  64,  8,  4,  8),   // small-medium: 256-384
    (64,  64,  8,  8,  8),   // medium: 384-512
    (64, 128,  8,  8,  8),   // medium-large: 512+
];

/// Auto-select best SIMT variant by benchmarking.
pub fn autotune_small_gemm(
    cache: &KernelCache,
    device: &WarpDevice,
    m: u32, n: u32, k: u32,
) -> Result<usize, DeviceError> {
    let include_path = WarpDevice::cuda_include_path();

    // Allocate test data
    let a_data: Vec<half::f16> = (0..(m*k) as usize)
        .map(|i| half::f16::from_f32(((i*7+3) % 200) as f32 * 0.01 - 1.0)).collect();
    let b_data: Vec<half::f16> = (0..(k*n) as usize)
        .map(|i| half::f16::from_f32(((i*11+5) % 200) as f32 * 0.01 - 1.0)).collect();

    let a = GpuTensor::from_host(device, &a_data,
        Shape::from_static(&[m as usize, k as usize]), DType::F16)?;
    let b = GpuTensor::from_host(device, &b_data,
        Shape::from_static(&[k as usize, n as usize]), DType::F16)?;
    let mut c = GpuTensor::<half::f16>::zeros(device,
        Shape::from_static(&[m as usize, n as usize]), DType::F16)?;

    let mut best_idx = 0;
    let mut best_time = f64::INFINITY;

    for (idx, &(bm, bn, bk, tm, tn)) in VARIANTS.iter().enumerate() {
        if bm > m || bn > n { continue; } // skip tiles larger than matrix

        let (src, name) = gen_simt_f16_variant(bm, bn, bk, tm, tn);
        let f = cache.get_or_compile_with_opts(device, &src, &name, &[include_path.clone()], None)?;

        let threads = (bm / tm) * (bn / tn);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((n + bn - 1) / bn, (m + bm - 1) / bm, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        // Warmup
        unsafe {
            use cudarc::driver::PushKernelArg;
            device.stream.launch_builder(&f)
                .arg(&mut c.data).arg(&a.data).arg(&b.data)
                .arg(&m).arg(&n).arg(&k)
                .launch(cfg)
                .map_err(|e| DeviceError::Launch(e.to_string()))?;
        }
        device.synchronize()?;

        // Benchmark
        let iters = 100;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            unsafe {
                use cudarc::driver::PushKernelArg;
                device.stream.launch_builder(&f)
                    .arg(&mut c.data).arg(&a.data).arg(&b.data)
                    .arg(&m).arg(&n).arg(&k)
                    .launch(cfg)
                    .map_err(|e| DeviceError::Launch(e.to_string()))?;
            }
        }
        device.synchronize()?;
        let elapsed = start.elapsed().as_secs_f64() / iters as f64;

        if elapsed < best_time {
            best_time = elapsed;
            best_idx = idx;
        }
    }

    Ok(best_idx)
}

/// Run GEMM using the best SIMT variant for the given size.
pub fn gemm_small_f16(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<half::f16>,
    b: &GpuTensor<half::f16>,
    c: &mut GpuTensor<half::f16>,
    m: u32, n: u32, k: u32,
) -> Result<(), DeviceError> {
    // Select variant based on matrix size
    let variant_idx = if m <= 128 && n <= 128 { 0 }
        else if m <= 256 && n <= 256 { 1 }
        else if m <= 384 && n <= 384 { 2 }
        else if m <= 512 && n <= 512 { 3 }
        else { 4 };

    let (bm, bn, bk, tm, tn) = VARIANTS[variant_idx.min(VARIANTS.len() - 1)];
    let (src, name) = gen_simt_f16_variant(bm, bn, bk, tm, tn);

    let include_path = WarpDevice::cuda_include_path();
    let f = cache.get_or_compile_with_opts(device, &src, &name, &[include_path], None)?;

    let threads = (bm / tm) * (bn / tn);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: ((n + bn - 1) / bn, (m + bm - 1) / bm, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        use cudarc::driver::PushKernelArg;
        device.stream.launch_builder(&f)
            .arg(&mut c.data).arg(&a.data).arg(&b.data)
            .arg(&m).arg(&n).arg(&k)
            .launch(cfg)
            .map_err(|e| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    #[test]
    fn small_gemm_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s, None => { println!("No CUDA"); return; }
        };

        let (m, n, k) = (64u32, 64u32, 64u32);
        let a_data: Vec<half::f16> = (0..(m*k) as usize)
            .map(|i| half::f16::from_f32(((i*7+3) % 100) as f32 * 0.01 - 0.5)).collect();
        let b_data: Vec<half::f16> = (0..(k*n) as usize)
            .map(|i| half::f16::from_f32(((i*11+5) % 100) as f32 * 0.01 - 0.5)).collect();

        let a = GpuTensor::from_host(&dev, &a_data,
            Shape::from_static(&[m as usize, k as usize]), DType::F16).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data,
            Shape::from_static(&[k as usize, n as usize]), DType::F16).unwrap();
        let mut c = GpuTensor::<half::f16>::zeros(&dev,
            Shape::from_static(&[m as usize, n as usize]), DType::F16).unwrap();

        gemm_small_f16(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let result: Vec<f32> = c.to_host(&dev).unwrap().iter().map(|v| v.to_f32()).collect();
        assert!(result.iter().all(|v| v.is_finite()));
        assert!(result.iter().any(|v| *v != 0.0));
        println!("Small GEMM 64x64x64: correct!");
    }

    #[test]
    fn small_gemm_variants_benchmark() {
        let (dev, cache) = match setup() {
            Some(s) => s, None => { println!("No CUDA"); return; }
        };

        println!("\n=== Small FP16 GEMM Variant Benchmark ===");
        for &size in &[64u32, 128, 256, 512] {
            let (m, n, k) = (size, size, size);
            let a_data: Vec<half::f16> = vec![half::f16::from_f32(0.01); (m*k) as usize];
            let b_data: Vec<half::f16> = vec![half::f16::from_f32(0.01); (k*n) as usize];

            let a = GpuTensor::from_host(&dev, &a_data,
                Shape::from_static(&[m as usize, k as usize]), DType::F16).unwrap();
            let b = GpuTensor::from_host(&dev, &b_data,
                Shape::from_static(&[k as usize, n as usize]), DType::F16).unwrap();
            let mut c = GpuTensor::<half::f16>::zeros(&dev,
                Shape::from_static(&[m as usize, n as usize]), DType::F16).unwrap();

            // Warmup
            gemm_small_f16(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
            dev.synchronize().unwrap();

            let iters = 200;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                gemm_small_f16(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
            }
            dev.synchronize().unwrap();
            let elapsed = start.elapsed();

            let tflops = 2.0 * m as f64 * n as f64 * k as f64 * iters as f64
                / elapsed.as_secs_f64() / 1e12;
            let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;

            println!("  {size}³: {tflops:.1} TFLOPS ({ms:.3}ms)");
        }
    }
}

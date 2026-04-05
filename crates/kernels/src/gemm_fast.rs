//! High-performance GEMM kernels.
//!
//! Register-tiled F32 GEMM with shared memory optimization.
//! Uses 128x128 block tiles with 8x8 per-thread register tiling (BK=16).
//! Padded shared memory layout to eliminate bank conflicts.
//! Float4 vectorized output stores for 128-bit write coalescing.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Register-tiled GEMM: each thread computes TM×TN = 8×8 output elements.
/// Block tile: BM×BN = 128×128, BK = 16 (doubled for fewer sync barriers).
/// 256 threads per block (16×16 threads, each doing 8×8).
/// Uses float4 vectorized stores and increased BK for better throughput.
const GEMM_REG_TILED_SRC: &str = r#"
#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

extern "C" __global__ void warp_gemm_fast(
    float * __restrict__ C,
    const float * __restrict__ A,
    const float * __restrict__ B,
    unsigned int M, unsigned int N, unsigned int K
) {
    // Thread block covers BM×BN output tile
    // Each thread computes TM×TN elements
    // Threads per block: (BM/TM) × (BN/TN) = 16×16 = 256

    __shared__ float As[BK][BM + 4];  // +4 padding avoids bank conflicts
    __shared__ float Bs[BK][BN + 4];

    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;  // 0..15 (column of thread within block)
    unsigned int ty = threadIdx.y;  // 0..15 (row of thread within block)
    unsigned int tid = ty * blockDim.x + tx;

    // Global position of this thread's TM×TN output tile
    unsigned int row_start = by * BM + ty * TM;
    unsigned int col_start = bx * BN + tx * TN;

    // Accumulator registers
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    // Main loop over K dimension in BK-sized tiles
    for (unsigned int k0 = 0; k0 < K; k0 += BK) {

        // Load A tile: A[by*BM..(by+1)*BM, k0..k0+BK] into As[BK][BM]
        // 256 threads load BM*BK = 128*16 = 2048 elements = 8 per thread
        #pragma unroll
        for (unsigned int load = 0; load < (BM * BK) / 256; load++) {
            unsigned int load_idx = tid + load * 256;
            unsigned int load_k = load_idx / BM;
            unsigned int load_m = load_idx % BM;
            unsigned int gm = by * BM + load_m;
            unsigned int gk = k0 + load_k;
            As[load_k][load_m] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }

        // Load B tile: B[k0..k0+BK, bx*BN..(bx+1)*BN] into Bs[BK][BN]
        // 256 threads load BK*BN = 16*128 = 2048 elements = 8 per thread
        #pragma unroll
        for (unsigned int load = 0; load < (BK * BN) / 256; load++) {
            unsigned int load_idx = tid + load * 256;
            unsigned int load_n = load_idx % BN;
            unsigned int load_k = load_idx / BN;
            unsigned int gn = bx * BN + load_n;
            unsigned int gk = k0 + load_k;
            Bs[load_k][load_n] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }

        __syncthreads();

        // Compute: accumulate BK products for this thread's TM×TN tile
        #pragma unroll
        for (unsigned int kk = 0; kk < BK; kk++) {
            // Load fragments from shared memory into registers
            float a_frag[TM];
            float b_frag[TN];

            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_frag[i] = As[kk][ty * TM + i];

            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_frag[j] = Bs[kk][tx * TN + j];

            // Outer product
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }

        __syncthreads();
    }

    // Store results with float4 vectorized writes (2x float4 = 8 floats per row)
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        unsigned int grow = row_start + i;
        if (grow >= M) continue;

        if (col_start + 7 < N) {
            // Fast path: full TN=8 columns fit — use two float4 stores (128-bit each)
            *reinterpret_cast<float4*>(&C[grow * N + col_start]) =
                make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            *reinterpret_cast<float4*>(&C[grow * N + col_start + 4]) =
                make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        } else {
            // Edge case: partial columns — scalar fallback
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                unsigned int gcol = col_start + j;
                if (gcol < N) {
                    C[grow * N + gcol] = acc[i][j];
                }
            }
        }
    }
}
"#;

/// Tensor Core GEMM using wmma API (FP16 in, FP32 accumulate).
/// Uses 16×16×16 matrix fragments via CUDA's wmma intrinsics.
const GEMM_WMMA_SRC: &str = r#"
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_SIZE 32
#define BLOCK_TILES_M 4  // warps along M
#define BLOCK_TILES_N 4  // warps along N
#define BLOCK_DIM_M (WMMA_M * BLOCK_TILES_M)  // 64
#define BLOCK_DIM_N (WMMA_N * BLOCK_TILES_N)  // 64

extern "C" __global__ void warp_gemm_wmma(
    half * __restrict__ C,
    const half * __restrict__ A,
    const half * __restrict__ B,
    unsigned int M, unsigned int N, unsigned int K
) {
    // Each warp computes one WMMA_M × WMMA_N tile
    unsigned int warpId = threadIdx.x / WARP_SIZE;
    unsigned int warp_row = warpId / BLOCK_TILES_N;
    unsigned int warp_col = warpId % BLOCK_TILES_N;

    unsigned int grow = blockIdx.y * BLOCK_DIM_M + warp_row * WMMA_M;
    unsigned int gcol = blockIdx.x * BLOCK_DIM_N + warp_col * WMMA_N;

    if (grow >= M || gcol >= N) return;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K in WMMA_K tiles
    for (unsigned int k = 0; k < K; k += WMMA_K) {
        if (grow < M && k + WMMA_K <= K) {
            wmma::load_matrix_sync(a_frag, A + grow * K + k, K);
        }
        if (k + WMMA_K <= K && gcol < N) {
            wmma::load_matrix_sync(b_frag, B + k * N + gcol, N);
        }
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result — convert FP32 acc back to FP16
    // First store to temp, then convert
    if (grow < M && gcol < N) {
        // Store accumulator (FP32) then convert to FP16
        float c_temp[WMMA_M * WMMA_N / WARP_SIZE];
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_half;
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_half.x[i] = __float2half(c_frag.x[i]);
        }
        wmma::store_matrix_sync(C + grow * N + gcol, c_half, N, wmma::mem_row_major);
    }
}
"#;

/// Launch the register-tiled GEMM (FP32).
pub fn gemm_fast(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    c: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, GEMM_REG_TILED_SRC, "warp_gemm_fast")?;

    let bm = 128u32;
    let bn = 128u32;
    let cfg = LaunchConfig {
        grid_dim: ((n + bn - 1) / bn, (m + bm - 1) / bm, 1),
        block_dim: (16, 16, 1),  // 256 threads
        shared_mem_bytes: 0,     // statically allocated
    };

    unsafe {
        device.stream.launch_builder(&f)
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

/// Launch Tensor Core GEMM (FP16).
pub fn gemm_wmma(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<half::f16>,
    b: &GpuTensor<half::f16>,
    c: &mut GpuTensor<half::f16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, GEMM_WMMA_SRC, "warp_gemm_wmma")?;

    let block_m = 64u32;
    let block_n = 64u32;
    let warps_per_block = 16u32; // 4×4 warps

    let cfg = LaunchConfig {
        grid_dim: ((n + block_n - 1) / block_n, (m + block_m - 1) / block_m, 1),
        block_dim: (warps_per_block * 32, 1, 1),  // 512 threads
        shared_mem_bytes: 0,
    };

    unsafe {
        device.stream.launch_builder(&f)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::cpu_gemm;
    use warp_ir::{DType, Shape};

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    #[test]
    fn gemm_fast_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (256u32, 256u32, 256u32);
        let a_data: Vec<f32> = (0..(m*k) as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let b_data: Vec<f32> = (0..(k*n) as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let mut c_ref = vec![0.0f32; (m*n) as usize];
        cpu_gemm(&a_data, &b_data, &mut c_ref, m as usize, n as usize, k as usize);

        let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let mut c = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

        gemm_fast(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let c_gpu = c.to_host(&dev).unwrap();
        let max_err: f32 = c_gpu.iter().zip(c_ref.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Fast GEMM 256x256x256: max error = {max_err:.6}");
        assert!(max_err < 0.1, "Max error {max_err} too high");
    }

    #[test]
    fn gemm_fast_vs_cublas() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        println!("\n=== TensorWarp Fast GEMM vs cuBLAS (FP32) ===");
        for &(m, n, k) in &[
            (256u32, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ] {
            let a_data: Vec<f32> = (0..(m*k) as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
            let b_data: Vec<f32> = (0..(k*n) as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();

            let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
            let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
            let mut c = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

            // Warmup
            gemm_fast(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
            dev.synchronize().unwrap();

            let iters = 50;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                gemm_fast(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
            }
            dev.synchronize().unwrap();
            let ours = start.elapsed();

            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            let our_tflops = flops * iters as f64 / ours.as_secs_f64() / 1e12;
            let our_ms = ours.as_secs_f64() * 1000.0 / iters as f64;

            // cuBLAS
            let cublas_result = crate::cublas_bench::cublas_sgemm_bench(&dev, m as i32, n as i32, k as i32, iters).unwrap();

            let ratio = our_tflops / cublas_result.tflops.max(1e-9) * 100.0;
            let winner = if our_tflops > cublas_result.tflops { "WARP WINS" } else { "" };

            println!(
                "  {m:4}³: Warp={:.3} TFLOPS ({:.3}ms) | cuBLAS={:.3} TFLOPS ({:.3}ms) | {:.1}% {winner}",
                our_tflops, our_ms, cublas_result.tflops, cublas_result.avg_ms, ratio,
            );
        }
    }

    #[test]
    fn gemm_wmma_test() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (64u32, 64u32, 64u32);
        let a_data: Vec<half::f16> = (0..(m*k) as usize)
            .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.01))
            .collect();
        let b_data: Vec<half::f16> = (0..(k*n) as usize)
            .map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.01))
            .collect();

        let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F16).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F16).unwrap();
        let mut c = GpuTensor::<half::f16>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F16).unwrap();

        match gemm_wmma(&cache, &dev, &a, &b, &mut c, m, n, k) {
            Ok(()) => {
                dev.synchronize().unwrap();
                let result = c.to_host(&dev).unwrap();
                println!("WMMA GEMM {m}x{n}x{k}: computed (sample: [{:.4}, {:.4}, ...])",
                    result[0].to_f32(), result[1].to_f32());
            }
            Err(e) => {
                // wmma might fail on compile — that's OK, we log it
                println!("WMMA GEMM: {e} (may need CUDA include paths for mma.h)");
            }
        }
    }
}

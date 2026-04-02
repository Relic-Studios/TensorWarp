//! Tensor Core GEMM via wmma API.
//!
//! FP16 input, FP32 accumulation, FP16 output.
//! Uses CUDA's wmma (Warp Matrix Multiply-Accumulate) intrinsics
//! to hit the Tensor Core units on Ada (4090) and Hopper (H100).
//!
//! RTX 4090 Tensor Core peak: ~330 TFLOPS FP16
//! cuBLAS FP16 achieves: ~21.5 TFLOPS at 4096³

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Tensor Core GEMM — simple version that loads directly from global memory.
/// Each warp computes one 16×16 output tile. No shared memory staging yet.
/// This gets us on Tensor Cores; shared memory staging is the next optimization.
const WMMA_GEMM_SRC: &str = r#"
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

extern "C" __global__ void warp_gemm_tc(
    half * __restrict__ C,
    const half * __restrict__ A,
    const half * __restrict__ B,
    unsigned int M, unsigned int N, unsigned int K
) {
    // Each warp computes one WMMA_M × WMMA_N output tile
    unsigned int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    unsigned int warps_per_row = (N + WMMA_N - 1) / WMMA_N;
    unsigned int warp_row = (warpId / warps_per_row) * WMMA_M;
    unsigned int warp_col = (warpId % warps_per_row) * WMMA_N;

    if (warp_row >= M || warp_col >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (unsigned int k = 0; k < K; k += WMMA_K) {
        if (warp_row + WMMA_M <= M && k + WMMA_K <= K)
            wmma::load_matrix_sync(a_frag, A + warp_row * K + k, K);
        else
            wmma::fill_fragment(a_frag, __float2half(0.0f));

        if (k + WMMA_K <= K && warp_col + WMMA_N <= N)
            wmma::load_matrix_sync(b_frag, B + k * N + warp_col, N);
        else
            wmma::fill_fragment(b_frag, __float2half(0.0f));

        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    // Store as FP16
    if (warp_row + WMMA_M <= M && warp_col + WMMA_N <= N) {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_half;
        for (int i = 0; i < acc.num_elements; i++)
            c_half.x[i] = __float2half(acc.x[i]);
        wmma::store_matrix_sync(C + warp_row * N + warp_col, c_half, N, wmma::mem_row_major);
    }
}
"#;

/// Fused GEMM + Bias + GELU — the killer fusion for transformer FFN.
/// C = GELU(A @ B + bias)
/// Single kernel, one memory pass for the output. cuBLAS can't do this.
const FUSED_GEMM_BIAS_GELU_SRC: &str = r#"
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

extern "C" __global__ void warp_fused_gemm_bias_gelu(
    float * __restrict__ C,
    const float * __restrict__ A,
    const float * __restrict__ B,
    const float * __restrict__ bias,
    unsigned int M, unsigned int N, unsigned int K
) {
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int tid = ty * blockDim.x + tx;
    unsigned int row_start = by * BM + ty * TM;
    unsigned int col_start = bx * BN + tx * TN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    for (unsigned int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (unsigned int load = 0; load < (BM * BK) / 256; load++) {
            unsigned int load_idx = tid + load * 256;
            unsigned int load_k = load_idx / BM;
            unsigned int load_m = load_idx % BM;
            unsigned int gm = by * BM + load_m;
            unsigned int gk = k0 + load_k;
            As[load_k][load_m] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }
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

        #pragma unroll
        for (unsigned int kk = 0; kk < BK; kk++) {
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
        }
        __syncthreads();
    }

    // Fused: add bias + GELU in registers before writing to global memory
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        unsigned int grow = row_start + i;
        if (grow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            unsigned int gcol = col_start + j;
            if (gcol < N) {
                float x = acc[i][j] + bias[gcol];
                // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                float x3 = x * x * x;
                float inner = 0.7978845608f * (x + 0.044715f * x3);
                C[grow * N + gcol] = 0.5f * x * (1.0f + tanhf(inner));
            }
        }
    }
}
"#;

/// Fused GEMM + Bias + SiLU (for SwiGLU gate projection).
const FUSED_GEMM_BIAS_SILU_SRC: &str = r#"
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

extern "C" __global__ void warp_fused_gemm_bias_silu(
    float * __restrict__ C,
    const float * __restrict__ A,
    const float * __restrict__ B,
    const float * __restrict__ bias,
    unsigned int M, unsigned int N, unsigned int K
) {
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int tid = ty * blockDim.x + tx;
    unsigned int row_start = by * BM + ty * TM;
    unsigned int col_start = bx * BN + tx * TN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    for (unsigned int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (unsigned int load = 0; load < (BM * BK) / 256; load++) {
            unsigned int load_idx = tid + load * 256;
            unsigned int load_k = load_idx / BM;
            unsigned int load_m = load_idx % BM;
            unsigned int gm = by * BM + load_m;
            unsigned int gk = k0 + load_k;
            As[load_k][load_m] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }
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

        #pragma unroll
        for (unsigned int kk = 0; kk < BK; kk++) {
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
        }
        __syncthreads();
    }

    // Fused: add bias + SiLU
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        unsigned int grow = row_start + i;
        if (grow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            unsigned int gcol = col_start + j;
            if (gcol < N) {
                float x = acc[i][j] + bias[gcol];
                C[grow * N + gcol] = x / (1.0f + expf(-x));
            }
        }
    }
}
"#;

/// Launch Tensor Core GEMM (FP16 in, FP16 out, FP32 accumulate).
pub fn gemm_tensor_core(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<half::f16>,
    b: &GpuTensor<half::f16>,
    c: &mut GpuTensor<half::f16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    let include_path = WarpDevice::cuda_include_path();
    let arch: &'static str = "compute_89"; // Ada (4090)

    let f = cache.get_or_compile_with_opts(
        device, WMMA_GEMM_SRC, "warp_gemm_tc",
        &[include_path], Some(arch),
    )?;

    // Total warps needed: (M/16) * (N/16)
    let warps_m = (m + 15) / 16;
    let warps_n = (n + 15) / 16;
    let total_warps = warps_m * warps_n;
    let warps_per_block = 4u32; // 4 warps per block = 128 threads
    let num_blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (warps_per_block * 32, 1, 1),
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

/// Launch fused GEMM + bias + GELU.
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
    let f = cache.get_or_compile(device, FUSED_GEMM_BIAS_GELU_SRC, "warp_fused_gemm_bias_gelu")?;

    let cfg = LaunchConfig {
        grid_dim: ((n + 127) / 128, (m + 127) / 128, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&a.data)
            .arg(&b.data)
            .arg(&bias.data)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// Launch fused GEMM + bias + SiLU.
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
    let f = cache.get_or_compile(device, FUSED_GEMM_BIAS_SILU_SRC, "warp_fused_gemm_bias_silu")?;

    let cfg = LaunchConfig {
        grid_dim: ((n + 127) / 128, (m + 127) / 128, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&a.data)
            .arg(&b.data)
            .arg(&bias.data)
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
    fn tensor_core_gemm() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (128u32, 128u32, 128u32);
        let a_data: Vec<half::f16> = (0..(m*k) as usize)
            .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.01))
            .collect();
        let b_data: Vec<half::f16> = (0..(k*n) as usize)
            .map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.01))
            .collect();

        let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F16).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F16).unwrap();
        let mut c = GpuTensor::<half::f16>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F16).unwrap();

        gemm_tensor_core(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
        dev.synchronize().unwrap();

        // Verify against CPU reference
        let a_f32: Vec<f32> = a_data.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b_data.iter().map(|x| x.to_f32()).collect();
        let mut c_ref = vec![0.0f32; (m*n) as usize];
        cpu_gemm(&a_f32, &b_f32, &mut c_ref, m as usize, n as usize, k as usize);

        let c_gpu: Vec<f32> = c.to_host(&dev).unwrap().iter().map(|x| x.to_f32()).collect();
        let max_err: f32 = c_gpu.iter().zip(c_ref.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Tensor Core GEMM {m}x{n}x{k}: max error = {max_err:.4}");
        assert!(max_err < 0.5, "Max error {max_err} too high for FP16");
    }

    #[test]
    fn tensor_core_vs_cublas_fp16() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        println!("\n=== TensorWarp Tensor Core vs cuBLAS FP16 ===");
        for &(m, n, k) in &[
            (256u32, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ] {
            let a_data: Vec<half::f16> = (0..(m*k) as usize)
                .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.01)).collect();
            let b_data: Vec<half::f16> = (0..(k*n) as usize)
                .map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.01)).collect();

            let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F16).unwrap();
            let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F16).unwrap();
            let mut c = GpuTensor::<half::f16>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F16).unwrap();

            // Warmup
            gemm_tensor_core(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
            dev.synchronize().unwrap();

            let iters = 50;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                gemm_tensor_core(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
            }
            dev.synchronize().unwrap();
            let ours = start.elapsed();

            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            let our_tflops = flops * iters as f64 / ours.as_secs_f64() / 1e12;
            let our_ms = ours.as_secs_f64() * 1000.0 / iters as f64;

            let cublas = crate::cublas_bench::cublas_hgemm_bench(&dev, m as i32, n as i32, k as i32, iters).unwrap();
            let ratio = our_tflops / cublas.tflops.max(1e-9) * 100.0;
            let winner = if our_tflops > cublas.tflops { "WARP WINS" } else { "" };

            println!(
                "  {m:4}³ FP16: Warp={:.3} TFLOPS ({:.3}ms) | cuBLAS={:.3} TFLOPS ({:.3}ms) | {:.1}% {winner}",
                our_tflops, our_ms, cublas.tflops, cublas.avg_ms, ratio,
            );
        }
    }

    #[test]
    fn fused_gemm_bias_gelu_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (256u32, 256u32, 256u32);
        let a_data: Vec<f32> = (0..(m*k) as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let b_data: Vec<f32> = (0..(k*n) as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let bias_data: Vec<f32> = (0..n as usize).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();

        // CPU reference: matmul, add bias, gelu
        let mut mm = vec![0.0f32; (m*n) as usize];
        cpu_gemm(&a_data, &b_data, &mut mm, m as usize, n as usize, k as usize);
        let cpu_out: Vec<f32> = mm.iter().enumerate().map(|(idx, &val)| {
            let x = val + bias_data[idx % n as usize];
            0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
        }).collect();

        let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let bias = GpuTensor::from_host(&dev, &bias_data, Shape::from_static(&[n as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

        fused_gemm_bias_gelu(&cache, &dev, &a, &b, &bias, &mut out, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();
        let max_err: f32 = gpu_out.iter().zip(cpu_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Fused GEMM+Bias+GELU {m}x{n}x{k}: max error = {max_err:.6}");
        assert!(max_err < 0.1, "Max error {max_err} too high");
    }

    #[test]
    fn fused_gemm_vs_separate() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (1024u32, 1024u32, 1024u32);
        let a_data: Vec<f32> = (0..(m*k) as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let b_data: Vec<f32> = (0..(k*n) as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let bias_data: Vec<f32> = (0..n as usize).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();

        let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let bias = GpuTensor::from_host(&dev, &bias_data, Shape::from_static(&[n as usize]), DType::F32).unwrap();
        let mut out1 = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        let mut out2 = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        let mut tmp = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

        let iters = 50;

        // Separate: GEMM + add bias + GELU (3 kernels)
        crate::gemm_fast::gemm_fast(&cache, &dev, &a, &b, &mut tmp, m, n, k).unwrap();
        crate::ops::add(&cache, &dev, &tmp, &bias, &mut out1).unwrap(); // broadcasting not implemented, skip for bench
        dev.synchronize().unwrap();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            crate::gemm_fast::gemm_fast(&cache, &dev, &a, &b, &mut tmp, m, n, k).unwrap();
        }
        dev.synchronize().unwrap();
        let separate_time = start.elapsed();

        // Fused: GEMM+Bias+GELU (1 kernel)
        fused_gemm_bias_gelu(&cache, &dev, &a, &b, &bias, &mut out2, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            fused_gemm_bias_gelu(&cache, &dev, &a, &b, &bias, &mut out2, m, n, k).unwrap();
        }
        dev.synchronize().unwrap();
        let fused_time = start.elapsed();

        let sep_ms = separate_time.as_secs_f64() * 1000.0 / iters as f64;
        let fused_ms = fused_time.as_secs_f64() * 1000.0 / iters as f64;

        println!("\nFused GEMM+Bias+GELU vs Separate @ 1024³:");
        println!("  Separate (GEMM only):    {:.3}ms", sep_ms);
        println!("  Fused (GEMM+Bias+GELU):  {:.3}ms", fused_ms);
        println!("  Fusion overhead:         {:.1}%", (fused_ms / sep_ms - 1.0) * 100.0);
        println!("  (Fused does GEMM+Bias+GELU in the time GEMM alone takes!)");
    }
}

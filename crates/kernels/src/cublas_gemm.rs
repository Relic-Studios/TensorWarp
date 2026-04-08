//! cuBLAS-backed GEMM — actual peak-performance matrix multiplication.
//!
//! Uses NVIDIA's hand-tuned SASS assembly kernels via the cuBLAS library.
//! This is the fastest possible GEMM on NVIDIA GPUs. Instead of competing
//! with cuBLAS using NVRTC-compiled kernels, we USE cuBLAS for raw GEMM
//! throughput and add our value on top (fusion, autotuning, speculative decode).
//!
//! Row-major to column-major mapping:
//!   Row-major: C[M,N] = A[M,K] * B[K,N]
//!   Column-major trick: C^T[N,M] = B^T[N,K] * A^T[K,M]
//!   cuBLAS call: gemm(OP_N, OP_N, N, M, K, B, N, A, K, C, N)
//!
//! Since B^T in column-major is just B in row-major (and same for A),
//! we pass B first, then A, and swap m/n.

use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;

use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// F32 GEMM via cuBLAS: C = A @ B (row-major)
///
/// Uses cuBLAS SGEMM with the column-major trick for row-major tensors.
/// This calls NVIDIA's hand-tuned SASS kernels — the fastest possible
/// F32 GEMM on the hardware.
pub fn gemm_cublas_f32(
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    c: &mut GpuTensor<f32>,
    m: u32, n: u32, k: u32,
) -> Result<(), DeviceError> {
    let blas = device.cublas()?;

    // Row-major C[M,N] = A[M,K] * B[K,N]
    // Column-major: C^T[N,M] = B^T[N,K] * A^T[K,M]
    // cuBLAS args: m=N, n=M, k=K, A=B(lda=N), B=A(ldb=K), C(ldc=N)
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,   // columns of C (rows of C^T in col-major)
        n: m as i32,   // rows of C
        k: k as i32,
        alpha: 1.0f32,
        lda: n as i32,  // leading dim of B (first matrix in col-major)
        ldb: k as i32,  // leading dim of A (second matrix in col-major)
        beta: 0.0f32,
        ldc: n as i32,  // leading dim of C
    };

    unsafe {
        blas.gemm(cfg, &b.data, &a.data, &mut c.data)
            .map_err(|e| DeviceError::Launch(format!("cuBLAS sgemm: {e}")))?;
    }
    Ok(())
}

/// F32 GEMM with transposed B: C[M,N] = A[M,K] × B^T[K,N] where B is stored as [N,K].
/// Used for tied embeddings: logits = hidden × embed_tokens^T.
pub fn gemm_cublas_f32_transB(
    device: &WarpDevice,
    a: &GpuTensor<f32>,   // [M, K] row-major
    b: &GpuTensor<f32>,   // [N, K] row-major (will be transposed)
    c: &mut GpuTensor<f32>, // [M, N] row-major
    m: u32, n: u32, k: u32,
) -> Result<(), DeviceError> {
    let blas = device.cublas()?;

    // Row-major C[M,N] = A[M,K] × B^T[K,N] where B stored as [N,K]
    // Column-major: C^T[N,M] = (B^T)^T[N,K] × A^T[K,M] = B[N,K] × A^T[K,M]
    // cuBLAS: transa=CUBLAS_OP_T (transpose B in col-major = no-transpose of B row-major... wait)
    //
    // Actually: in column-major land:
    //   C^T[N,M] = B_colmaj[N,K] × A^T_colmaj[K,M]
    //   B row-major [N,K] = B^T column-major [K,N], so we need transA=T to get [N,K]
    //   A row-major [M,K] = A^T column-major [K,M], no transpose needed
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_T,  // transpose B (stored as [N,K] row-major)
        transb: cublasOperation_t::CUBLAS_OP_N,  // A as-is
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0f32,
        lda: k as i32,   // leading dim of B in col-major (K, since B is [N,K] row-major = [K,N] col-major)
        ldb: k as i32,   // leading dim of A in col-major (K, since A is [M,K] row-major = [K,M] col-major)
        beta: 0.0f32,
        ldc: n as i32,
    };

    unsafe {
        blas.gemm(cfg, &b.data, &a.data, &mut c.data)
            .map_err(|e| DeviceError::Launch(format!("cuBLAS sgemm transB: {e}")))?;
    }
    Ok(())
}

/// FP16 GEMM via cuBLAS: C = A @ B (row-major, half precision)
///
/// Uses cuBLAS HGEMM with automatic tensor core dispatch.
/// On Ampere+ GPUs this will use tensor cores for peak FP16 throughput.
pub fn gemm_cublas_f16(
    device: &WarpDevice,
    a: &GpuTensor<half::f16>,
    b: &GpuTensor<half::f16>,
    c: &mut GpuTensor<half::f16>,
    m: u32, n: u32, k: u32,
) -> Result<(), DeviceError> {
    let blas = device.cublas()?;

    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: half::f16::from_f32(1.0),
        lda: n as i32,
        ldb: k as i32,
        beta: half::f16::from_f32(0.0),
        ldc: n as i32,
    };

    unsafe {
        blas.gemm(cfg, &b.data, &a.data, &mut c.data)
            .map_err(|e| DeviceError::Launch(format!("cuBLAS hgemm: {e}")))?;
    }
    Ok(())
}

/// F32 GEMM with additive accumulation: C = A @ B + C (row-major)
///
/// Like gemm_cublas_f32 but uses beta=1.0 so the existing contents of C
/// are added to the result. This enables GEMM+bias fusion: pre-load bias
/// into C, then call this to get C = A@B + bias in a single cuBLAS launch.
pub fn gemm_cublas_f32_add(
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    c: &mut GpuTensor<f32>,
    m: u32, n: u32, k: u32,
) -> Result<(), DeviceError> {
    let blas = device.cublas()?;

    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0f32,
        lda: n as i32,
        ldb: k as i32,
        beta: 1.0f32,   // <-- accumulate into C instead of overwriting
        ldc: n as i32,
    };

    unsafe {
        blas.gemm(cfg, &b.data, &a.data, &mut c.data)
            .map_err(|e| DeviceError::Launch(format!("cuBLAS sgemm (add): {e}")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::{DType, Shape};

    fn setup() -> Option<WarpDevice> {
        WarpDevice::new(0).ok()
    }

    /// CPU reference GEMM for correctness checking.
    fn cpu_gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    #[test]
    fn cublas_gemm_f32_correctness() {
        let device = match setup() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let m = 256usize;
        let n = 256usize;
        let k = 512usize;

        // Generate deterministic test data
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.01)
            .collect();

        // CPU reference
        let c_ref = cpu_gemm_f32(&a_data, &b_data, m, n, k);

        // GPU cuBLAS
        let a_gpu = GpuTensor::from_host(&device, &a_data,
            Shape::from_static(&[m, k]), DType::F32).unwrap();
        let b_gpu = GpuTensor::from_host(&device, &b_data,
            Shape::from_static(&[k, n]), DType::F32).unwrap();
        let mut c_gpu = GpuTensor::<f32>::zeros(&device,
            Shape::from_static(&[m, n]), DType::F32).unwrap();

        gemm_cublas_f32(&device, &a_gpu, &b_gpu, &mut c_gpu,
            m as u32, n as u32, k as u32).unwrap();
        device.synchronize().unwrap();

        let c_result = device.dtoh(&c_gpu.data).unwrap();

        // Check max error
        let max_err = c_ref.iter().zip(c_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("cuBLAS F32 GEMM {}x{}x{}: max error = {:.6}", m, n, k, max_err);
        assert!(max_err < 0.01, "cuBLAS F32 GEMM error too large: {}", max_err);
    }

    #[test]
    fn cublas_gemm_f16_correctness() {
        let device = match setup() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let m = 256usize;
        let n = 256usize;
        let k = 512usize;

        // Generate test data
        let a_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        let b_f32: Vec<f32> = (0..k * n)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.01)
            .collect();

        // CPU reference (in f32 for precision)
        let c_ref = cpu_gemm_f32(&a_f32, &b_f32, m, n, k);

        // Convert to f16 for GPU
        let a_f16: Vec<half::f16> = a_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
        let b_f16: Vec<half::f16> = b_f32.iter().map(|&x| half::f16::from_f32(x)).collect();

        let a_gpu = GpuTensor::from_host(&device, &a_f16,
            Shape::from_static(&[m, k]), DType::F16).unwrap();
        let b_gpu = GpuTensor::from_host(&device, &b_f16,
            Shape::from_static(&[k, n]), DType::F16).unwrap();
        let mut c_gpu = GpuTensor::<half::f16>::zeros(&device,
            Shape::from_static(&[m, n]), DType::F16).unwrap();

        gemm_cublas_f16(&device, &a_gpu, &b_gpu, &mut c_gpu,
            m as u32, n as u32, k as u32).unwrap();
        device.synchronize().unwrap();

        let c_result_f16 = device.dtoh(&c_gpu.data).unwrap();
        let c_result: Vec<f32> = c_result_f16.iter().map(|x| x.to_f32()).collect();

        // FP16 has less precision, so allow larger error
        let max_err = c_ref.iter().zip(c_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("cuBLAS F16 GEMM {}x{}x{}: max error = {:.6}", m, n, k, max_err);
        assert!(max_err < 0.1, "cuBLAS F16 GEMM error too large: {}", max_err);
    }
}

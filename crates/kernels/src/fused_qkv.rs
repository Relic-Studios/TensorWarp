//! Fused QKV projection: replaces 3 cuBLAS HGEMM calls with 1 custom kernel.
//!
//! For M=1 decode, cuBLAS has significant API overhead per call (~5-10µs).
//! This kernel loads the F16 input vector into shared memory once and
//! computes Q, K, V projections in a single launch.
//!
//! Weights are F16 stored as [N, K] (transposed). Output is F32.

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;
use cudarc::driver::{LaunchConfig, PushKernelArg};

macro_rules! launch_err {
    ($e:expr) => {
        $e.map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))
    };
}

const FUSED_QKV_GEMV_SRC: &str = r#"
extern "C" __global__ void fused_qkv_gemv(
    float* __restrict__ Q,            // [q_dim] output
    float* __restrict__ K,            // [kv_dim] output
    float* __restrict__ V,            // [kv_dim] output (only written if !k_eq_v)
    const unsigned short* __restrict__ A,   // [K_in] input F16
    const unsigned short* __restrict__ Wq,  // [q_dim, K_in] F16 row-major
    const unsigned short* __restrict__ Wk,  // [kv_dim, K_in] F16 row-major
    const unsigned short* __restrict__ Wv,  // [kv_dim, K_in] F16 row-major
    int K_in, int q_dim, int kv_dim, int k_eq_v
) {
    // Load input A into shared memory (F16 → F16, stays as half for reads)
    extern __shared__ unsigned short A_smem[];
    for (int i = threadIdx.x; i < K_in; i += blockDim.x) {
        A_smem[i] = A[i];
    }
    __syncthreads();

    int total_out = q_dim + kv_dim + (k_eq_v ? 0 : kv_dim);
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= total_out) return;

    // Determine which projection and local index
    const unsigned short* W;
    float* out;
    int local_n;
    if (n < q_dim) {
        W = Wq; out = Q; local_n = n;
    } else if (n < q_dim + kv_dim) {
        W = Wk; out = K; local_n = n - q_dim;
    } else {
        W = Wv; out = V; local_n = n - q_dim - kv_dim;
    }

    // Dot product: W[local_n, :] · A[:] in F16→F32
    const unsigned short* w_row = W + (long long)local_n * K_in;
    float dot = 0.0f;

    // Vectorized: read 4 F16 values (8 bytes) at a time
    int k = 0;
    for (; k + 3 < K_in; k += 4) {
        // Read 4 F16 weights
        unsigned short w0 = __ldg(w_row + k);
        unsigned short w1 = __ldg(w_row + k + 1);
        unsigned short w2 = __ldg(w_row + k + 2);
        unsigned short w3 = __ldg(w_row + k + 3);
        // Read 4 F16 activations from shared memory
        unsigned short a0 = A_smem[k];
        unsigned short a1 = A_smem[k + 1];
        unsigned short a2 = A_smem[k + 2];
        unsigned short a3 = A_smem[k + 3];
        // Convert and FMA
        float fw0, fw1, fw2, fw3, fa0, fa1, fa2, fa3;
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(fw0) : "h"(w0));
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(fw1) : "h"(w1));
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(fw2) : "h"(w2));
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(fw3) : "h"(w3));
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(fa0) : "h"(a0));
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(fa1) : "h"(a1));
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(fa2) : "h"(a2));
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(fa3) : "h"(a3));
        dot = fmaf(fw0, fa0, dot);
        dot = fmaf(fw1, fa1, dot);
        dot = fmaf(fw2, fa2, dot);
        dot = fmaf(fw3, fa3, dot);
    }
    // Remainder
    for (; k < K_in; k++) {
        float fw, fa;
        unsigned short w = __ldg(w_row + k);
        unsigned short a = A_smem[k];
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(fw) : "h"(w));
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(fa) : "h"(a));
        dot = fmaf(fw, fa, dot);
    }

    out[local_n] = dot;
}
"#;

/// Fused QKV projection: single kernel for Q, K, V GEMV with F16 weights → F32 output.
/// If k_eq_v is true, V is not computed (caller should copy K → V).
pub fn fused_qkv_gemv(
    cache: &KernelCache,
    device: &WarpDevice,
    input_f16: &GpuTensor<half::f16>,  // [1, K]
    wq: &GpuTensor<half::f16>,        // [q_dim, K]
    wk: &GpuTensor<half::f16>,        // [kv_dim, K]
    wv: &GpuTensor<half::f16>,        // [kv_dim, K]
    q_out: &mut GpuTensor<f32>,        // [1, q_dim]
    k_out: &mut GpuTensor<f32>,        // [1, kv_dim]
    v_out: &mut GpuTensor<f32>,        // [1, kv_dim]
    k_in: u32, q_dim: u32, kv_dim: u32,
    k_eq_v: bool,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, FUSED_QKV_GEMV_SRC, "fused_qkv_gemv")?;
    let total_out = q_dim + kv_dim + if k_eq_v { 0 } else { kv_dim };
    let threads = 256u32;
    let blocks = (total_out + threads - 1) / threads;
    let shared_bytes = k_in * 2; // F16 input vector

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    let k_eq_v_flag = if k_eq_v { 1i32 } else { 0i32 };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut q_out.data).arg(&mut k_out.data).arg(&mut v_out.data)
            .arg(&input_f16.data)
            .arg(&wq.data).arg(&wk.data).arg(&wv.data)
            .arg(&(k_in as i32)).arg(&(q_dim as i32)).arg(&(kv_dim as i32)).arg(&k_eq_v_flag)
            .launch(cfg))?;
    }

    // If k_eq_v, copy K output to V
    if k_eq_v {
        device.stream.memcpy_dtod(&k_out.data, &mut v_out.data)
            .map_err(|e| DeviceError::Memory(format!("K→V copy: {e}")))?;
    }

    Ok(())
}

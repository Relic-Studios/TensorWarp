//! Fused SwiGLU kernel.
//!
//! SwiGLU(x, W_gate, W_up) = SiLU(x @ W_gate) * (x @ W_up)
//!
//! Standard approach: 2 GEMMs + SiLU + elementwise multiply = 4 kernel launches.
//! Our fused kernel: compute both GEMMs, apply SiLU to gate, multiply with up,
//! all without writing intermediate results to global memory.
//!
//! This is the FFN activation used by LLaMA, Mistral, Qwen, and most
//! modern transformers. Fusing it is one of TensorWarp's biggest wins.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Fused SwiGLU: out = SiLU(X @ W_gate) * (X @ W_up)
/// Two GEMMs + SiLU + mul in a single kernel.
/// X: [M, K], W_gate: [K, N], W_up: [K, N], out: [M, N]
const FUSED_SWIGLU_SRC: &str = r#"
#define BM 128
#define BN 64
#define BK 8
#define TM 8
#define TN 4

extern "C" __global__ void warp_fused_swiglu(
    float * __restrict__ out,
    const float * __restrict__ X,
    const float * __restrict__ W_gate,
    const float * __restrict__ W_up,
    unsigned int M, unsigned int N, unsigned int K
) {
    __shared__ float Xs[BK][BM];
    __shared__ float Wgs[BK][BN];
    __shared__ float Wus[BK][BN];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x % 16;
    unsigned int ty = threadIdx.x / 16;
    unsigned int tid = threadIdx.x;
    unsigned int row_start = by * BM + ty * TM;
    unsigned int col_start = bx * BN + tx * TN;

    // Two accumulators: one for gate, one for up
    float gate_acc[TM][TN];
    float up_acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            gate_acc[i][j] = 0.0f;
            up_acc[i][j] = 0.0f;
        }

    for (unsigned int k0 = 0; k0 < K; k0 += BK) {
        // Load X tile
        #pragma unroll
        for (unsigned int load = 0; load < (BM * BK + 255) / 256; load++) {
            unsigned int li = tid + load * 256;
            if (li < BM * BK) {
                unsigned int lk = li / BM, lm = li % BM;
                unsigned int gm = by * BM + lm, gk = k0 + lk;
                Xs[lk][lm] = (gm < M && gk < K) ? X[gm * K + gk] : 0.0f;
            }
        }
        // Load W_gate tile
        #pragma unroll
        for (unsigned int load = 0; load < (BK * BN + 255) / 256; load++) {
            unsigned int li = tid + load * 256;
            if (li < BK * BN) {
                unsigned int ln = li % BN, lk = li / BN;
                unsigned int gn = bx * BN + ln, gk = k0 + lk;
                Wgs[lk][ln] = (gk < K && gn < N) ? W_gate[gk * N + gn] : 0.0f;
            }
        }
        // Load W_up tile
        #pragma unroll
        for (unsigned int load = 0; load < (BK * BN + 255) / 256; load++) {
            unsigned int li = tid + load * 256;
            if (li < BK * BN) {
                unsigned int ln = li % BN, lk = li / BN;
                unsigned int gn = bx * BN + ln, gk = k0 + lk;
                Wus[lk][ln] = (gk < K && gn < N) ? W_up[gk * N + gn] : 0.0f;
            }
        }
        __syncthreads();

        // Compute both GEMMs simultaneously
        #pragma unroll
        for (unsigned int kk = 0; kk < BK; kk++) {
            float x_frag[TM];
            float g_frag[TN], u_frag[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) x_frag[i] = Xs[kk][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                g_frag[j] = Wgs[kk][tx * TN + j];
                u_frag[j] = Wus[kk][tx * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    gate_acc[i][j] += x_frag[i] * g_frag[j];
                    up_acc[i][j] += x_frag[i] * u_frag[j];
                }
        }
        __syncthreads();
    }

    // Apply SiLU to gate and multiply with up — all in registers
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        unsigned int grow = row_start + i;
        if (grow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            unsigned int gcol = col_start + j;
            if (gcol < N) {
                float g = gate_acc[i][j];
                float silu_g = g / (1.0f + expf(-g));
                out[grow * N + gcol] = silu_g * up_acc[i][j];
            }
        }
    }
}
"#;

/// Launch fused SwiGLU: out = SiLU(X @ W_gate) * (X @ W_up)
pub fn fused_swiglu(
    cache: &KernelCache,
    device: &WarpDevice,
    x: &GpuTensor<f32>,
    w_gate: &GpuTensor<f32>,
    w_up: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, FUSED_SWIGLU_SRC, "warp_fused_swiglu")?;

    let cfg = LaunchConfig {
        grid_dim: ((n + 63) / 64, (m + 127) / 128, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&x.data)
            .arg(&w_gate.data)
            .arg(&w_up.data)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// CPU reference for SwiGLU.
pub fn cpu_swiglu(
    x: &[f32], w_gate: &[f32], w_up: &[f32], out: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut gate_sum = 0.0f32;
            let mut up_sum = 0.0f32;
            for p in 0..k {
                gate_sum += x[i * k + p] * w_gate[p * n + j];
                up_sum += x[i * k + p] * w_up[p * n + j];
            }
            let silu_gate = gate_sum / (1.0 + (-gate_sum).exp());
            out[i * n + j] = silu_gate * up_sum;
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
    fn fused_swiglu_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (256u32, 256u32, 256u32);
        let x_data: Vec<f32> = (0..(m*k) as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let wg_data: Vec<f32> = (0..(k*n) as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let wu_data: Vec<f32> = (0..(k*n) as usize).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();

        let mut cpu_out = vec![0.0f32; (m*n) as usize];
        cpu_swiglu(&x_data, &wg_data, &wu_data, &mut cpu_out, m as usize, n as usize, k as usize);

        let x = GpuTensor::from_host(&dev, &x_data, Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let wg = GpuTensor::from_host(&dev, &wg_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let wu = GpuTensor::from_host(&dev, &wu_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

        fused_swiglu(&cache, &dev, &x, &wg, &wu, &mut out, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();
        let max_err: f32 = gpu_out.iter().zip(cpu_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Fused SwiGLU {m}x{n}x{k}: max error = {max_err:.6}");
        assert!(max_err < 0.1, "Max error {max_err} too high");
    }

    #[test]
    fn fused_swiglu_vs_separate() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (512u32, 512u32, 512u32);
        let x_data: Vec<f32> = (0..(m*k) as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let wg_data: Vec<f32> = (0..(k*n) as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let wu_data: Vec<f32> = (0..(k*n) as usize).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();

        let x = GpuTensor::from_host(&dev, &x_data, Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let wg = GpuTensor::from_host(&dev, &wg_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let wu = GpuTensor::from_host(&dev, &wu_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let mut out_fused = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

        // Separate path buffers
        let mut gate = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        let mut up = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        let mut gate_act = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        let mut out_sep = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

        let iters = 50;

        // Warmup both paths
        fused_swiglu(&cache, &dev, &x, &wg, &wu, &mut out_fused, m, n, k).unwrap();
        crate::ops::gemm(&cache, &dev, &x, &wg, &mut gate, m, n, k).unwrap();
        crate::ops::gemm(&cache, &dev, &x, &wu, &mut up, m, n, k).unwrap();
        crate::ops::silu(&cache, &dev, &gate, &mut gate_act).unwrap();
        crate::ops::mul(&cache, &dev, &gate_act, &up, &mut out_sep).unwrap();
        dev.synchronize().unwrap();

        // Benchmark separate: 2 GEMMs + SiLU + mul = 4 kernels
        let start = std::time::Instant::now();
        for _ in 0..iters {
            crate::ops::gemm(&cache, &dev, &x, &wg, &mut gate, m, n, k).unwrap();
            crate::ops::gemm(&cache, &dev, &x, &wu, &mut up, m, n, k).unwrap();
            crate::ops::silu(&cache, &dev, &gate, &mut gate_act).unwrap();
            crate::ops::mul(&cache, &dev, &gate_act, &up, &mut out_sep).unwrap();
        }
        dev.synchronize().unwrap();
        let sep_time = start.elapsed();

        // Benchmark fused: 1 kernel
        let start = std::time::Instant::now();
        for _ in 0..iters {
            fused_swiglu(&cache, &dev, &x, &wg, &wu, &mut out_fused, m, n, k).unwrap();
        }
        dev.synchronize().unwrap();
        let fused_time = start.elapsed();

        let sep_ms = sep_time.as_secs_f64() * 1000.0 / iters as f64;
        let fused_ms = fused_time.as_secs_f64() * 1000.0 / iters as f64;
        let speedup = sep_time.as_secs_f64() / fused_time.as_secs_f64();

        println!("\nSwiGLU Fusion @ {m}x{n}x{k} ({iters} iters):");
        println!("  Separate (2 GEMM + SiLU + mul): {sep_ms:.3}ms");
        println!("  Fused (1 kernel):               {fused_ms:.3}ms");
        println!("  Speedup:                        {speedup:.2}x");
    }
}

//! Split-K GEMM kernel optimized for M=1 decode (autoregressive inference).
//!
//! Standard GEMM launches ceil(M/BM) x ceil(N/BN) blocks. For M=1, this means
//! only ceil(N/BN) blocks — far too few to saturate modern GPUs with thousands
//! of SMs. Split-K partitions the K dimension into `splits` slices, launching
//! `splits` times more blocks that each compute partial dot products and
//! atomicAdd into the output.
//!
//! For LLaMA-7B decode (M=1, K=4096, N=4096): standard GEMM launches ~16 blocks,
//! Split-K with 16 splits launches ~256 blocks — full GPU utilization.

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

const SPLITK_SRC: &str = r#"
extern "C" __global__ void warp_gemm_splitk(
    float *out,        // [M, N] — output (atomicAdd into this)
    const float *A,    // [M, K]
    const float *B,    // [K, N]
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int splits  // number of K-partitions
) {
    // blockIdx.x = column block (n_start = blockIdx.x * blockDim.x)
    // blockIdx.y = which K-split
    // threadIdx.x = thread within block

    unsigned int split_id = blockIdx.y;
    unsigned int k_per_split = (K + splits - 1) / splits;
    unsigned int k_start = split_id * k_per_split;
    unsigned int k_end = k_start + k_per_split;
    if (k_end > K) k_end = K;

    // Each thread handles one output column n
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // For each row m (M is small, so loop over all rows)
    for (unsigned int m = 0; m < M; m++) {
        float sum = 0.0f;
        // Vectorized K-reduction (4-wide unroll)
        unsigned int k = k_start;
        for (; k + 3 < k_end; k += 4) {
            sum += A[m * K + k]     * B[k * N + n]
                 + A[m * K + k + 1] * B[(k+1) * N + n]
                 + A[m * K + k + 2] * B[(k+2) * N + n]
                 + A[m * K + k + 3] * B[(k+3) * N + n];
        }
        for (; k < k_end; k++) {
            sum += A[m * K + k] * B[k * N + n];
        }
        atomicAdd(&out[m * N + n], sum);
    }
}
"#;

/// Split-K GEMM: partitions K dimension across `splits` thread blocks.
///
/// The output buffer `c` is zeroed before launch (required by atomicAdd accumulation).
/// Best for small M (1-8) with large K where standard GEMM under-utilizes the GPU.
pub fn gemm_splitk(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    c: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
    splits: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, SPLITK_SRC, "warp_gemm_splitk")?;

    // Zero the output buffer — atomicAdd accumulates into it
    device
        .stream
        .memset_zeros(&mut c.data)
        .map_err(|e| DeviceError::Memory(e.to_string()))?;

    let block_x = 256u32;
    let grid_x = (n + block_x - 1) / block_x;
    let grid_y = splits;

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (block_x, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_err!(device
            .stream
            .launch_builder(&f)
            .arg(&mut c.data)
            .arg(&a.data)
            .arg(&b.data)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .arg(&splits)
            .launch(cfg))?;
    }
    Ok(())
}

/// Auto-selecting Split-K GEMM: chooses split factor based on K size.
///
/// Heuristic: splits = clamp(K / 256, 2, 32).
///   K=4096  -> 16 splits
///   K=11008 -> 32 splits (clamped)
///   K=512   -> 2 splits
pub fn gemm_splitk_auto(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    c: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    let splits = auto_splits(k);
    gemm_splitk(cache, device, a, b, c, m, n, k, splits)
}

/// Compute the auto-selected split factor for a given K.
pub fn auto_splits(k: u32) -> u32 {
    (k / 256).clamp(2, 32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::{DType, Shape};

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    /// Reference CPU GEMM for correctness checking.
    fn cpu_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for i in 0..k {
                    sum += a[row * k + i] * b[i * n + col];
                }
                c[row * n + col] = sum;
            }
        }
        c
    }

    #[test]
    fn splitk_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => {
                println!("No CUDA, skipping");
                return;
            }
        };

        // Test 1: M=1, K=4096, N=4096 (decode-step QKV projection)
        {
            let (m, n, k) = (1u32, 4096u32, 4096u32);
            let a_data: Vec<f32> = (0..(m * k) as usize)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
                .collect();
            let b_data: Vec<f32> = (0..(k * n) as usize)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.01)
                .collect();

            let expected = cpu_gemm(&a_data, &b_data, m as usize, n as usize, k as usize);

            let a = GpuTensor::from_host(
                &dev,
                &a_data,
                Shape::from_static(&[m as usize, k as usize]),
                DType::F32,
            )
            .unwrap();
            let b = GpuTensor::from_host(
                &dev,
                &b_data,
                Shape::from_static(&[k as usize, n as usize]),
                DType::F32,
            )
            .unwrap();
            let mut c = GpuTensor::<f32>::zeros(
                &dev,
                Shape::from_static(&[m as usize, n as usize]),
                DType::F32,
            )
            .unwrap();

            gemm_splitk_auto(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
            dev.synchronize().unwrap();

            let result = c.to_host(&dev).unwrap();
            let max_err = result
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            println!(
                "Split-K M=1 K=4096 N=4096: max_err={:.6}, splits={}",
                max_err,
                auto_splits(k)
            );
            assert!(
                max_err < 0.001,
                "Split-K M=1 error too high: {max_err}"
            );
        }

        // Test 2: M=4, K=11008, N=4096 (LLaMA FFN down-projection, batched decode)
        {
            let (m, n, k) = (4u32, 4096u32, 11008u32);
            let a_data: Vec<f32> = (0..(m * k) as usize)
                .map(|i| ((i % 19) as f32 - 9.0) * 0.005)
                .collect();
            let b_data: Vec<f32> = (0..(k * n) as usize)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.005)
                .collect();

            let expected = cpu_gemm(&a_data, &b_data, m as usize, n as usize, k as usize);

            let a = GpuTensor::from_host(
                &dev,
                &a_data,
                Shape::from_static(&[m as usize, k as usize]),
                DType::F32,
            )
            .unwrap();
            let b = GpuTensor::from_host(
                &dev,
                &b_data,
                Shape::from_static(&[k as usize, n as usize]),
                DType::F32,
            )
            .unwrap();
            let mut c = GpuTensor::<f32>::zeros(
                &dev,
                Shape::from_static(&[m as usize, n as usize]),
                DType::F32,
            )
            .unwrap();

            gemm_splitk_auto(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
            dev.synchronize().unwrap();

            let result = c.to_host(&dev).unwrap();
            let max_err = result
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            println!(
                "Split-K M=4 K=11008 N=4096: max_err={:.6}, splits={}",
                max_err,
                auto_splits(k)
            );
            assert!(
                max_err < 0.001,
                "Split-K M=4 error too high: {max_err}"
            );
        }

        // Test 3: Compare Split-K vs regular GEMM for identical results
        {
            let (m, n, k) = (1u32, 4096u32, 4096u32);
            let a_data: Vec<f32> = (0..(m * k) as usize)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.01)
                .collect();
            let b_data: Vec<f32> = (0..(k * n) as usize)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
                .collect();

            let a = GpuTensor::from_host(
                &dev,
                &a_data,
                Shape::from_static(&[m as usize, k as usize]),
                DType::F32,
            )
            .unwrap();
            let b = GpuTensor::from_host(
                &dev,
                &b_data,
                Shape::from_static(&[k as usize, n as usize]),
                DType::F32,
            )
            .unwrap();

            // Regular GEMM
            let mut c_ref = GpuTensor::<f32>::zeros(
                &dev,
                Shape::from_static(&[m as usize, n as usize]),
                DType::F32,
            )
            .unwrap();
            crate::ops::gemm(&cache, &dev, &a, &b, &mut c_ref, m, n, k).unwrap();

            // Split-K GEMM
            let mut c_splitk = GpuTensor::<f32>::zeros(
                &dev,
                Shape::from_static(&[m as usize, n as usize]),
                DType::F32,
            )
            .unwrap();
            gemm_splitk_auto(&cache, &dev, &a, &b, &mut c_splitk, m, n, k).unwrap();

            dev.synchronize().unwrap();

            let ref_result = c_ref.to_host(&dev).unwrap();
            let splitk_result = c_splitk.to_host(&dev).unwrap();
            let max_err = ref_result
                .iter()
                .zip(splitk_result.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            println!("Split-K vs regular GEMM: max_err={:.6}", max_err);
            assert!(
                max_err < 0.001,
                "Split-K vs regular GEMM error too high: {max_err}"
            );
        }
    }

    #[test]
    fn splitk_auto_splits_heuristic() {
        assert_eq!(auto_splits(512), 2);
        assert_eq!(auto_splits(1024), 4);
        assert_eq!(auto_splits(4096), 16);
        assert_eq!(auto_splits(11008), 32); // clamped
        assert_eq!(auto_splits(16384), 32); // clamped
        assert_eq!(auto_splits(256), 2);    // minimum
    }
}

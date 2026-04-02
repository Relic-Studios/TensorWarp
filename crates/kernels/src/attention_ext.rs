//! Extended attention — handles head_dim > 32 via multi-warp reduction.
//!
//! The original flash attention uses one warp (32 threads) for the dot product,
//! limiting head_dim to 32. Real models use head_dim=64-128.
//!
//! This version uses shared memory for the dot product reduction,
//! supporting arbitrary head_dim up to block size.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Extended attention using shared memory reduction.
/// Supports head_dim up to 1024.
const EXTENDED_ATTENTION_SRC: &str = r#"
extern "C" __global__ void warp_attention_ext(
    float *out,           // [B, N, D]
    const float *Q,       // [B, N, D]
    const float *K,       // [B, N, D]
    const float *V,       // [B, N, D]
    unsigned int B,
    unsigned int N,
    unsigned int D,
    float scale,
    unsigned int causal
) {
    extern __shared__ float smem[];
    float *dot_buf = smem; // [blockDim.x] for reduction

    unsigned int b = blockIdx.y;
    unsigned int i = blockIdx.x;
    unsigned int d = threadIdx.x;

    if (b >= B || i >= N) return;

    const float *q_row = Q + b * N * D + i * D;
    float *out_row = out + b * N * D + i * D;

    float running_max = -1e30f;
    float running_sum = 0.0f;
    float running_out = (d < D) ? 0.0f : 0.0f;

    unsigned int end_j = causal ? (i + 1) : N;

    for (unsigned int j = 0; j < end_j; j++) {
        const float *k_row = K + b * N * D + j * D;

        // Compute dot product Q[i]·K[j] via shared memory reduction
        float partial = 0.0f;
        if (d < D) {
            partial = q_row[d] * k_row[d];
        }
        dot_buf[threadIdx.x] = partial;
        __syncthreads();

        // Tree reduction
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                dot_buf[threadIdx.x] += dot_buf[threadIdx.x + stride];
            }
            __syncthreads();
        }

        float score = dot_buf[0] * scale;

        // Online softmax
        float new_max = fmaxf(running_max, score);
        float correction = expf(running_max - new_max);
        float weight = expf(score - new_max);

        running_sum = running_sum * correction + weight;
        if (d < D) {
            running_out = running_out * correction + weight * V[b * N * D + j * D + d];
        }
        running_max = new_max;

        __syncthreads();
    }

    if (d < D) {
        out_row[d] = running_out / running_sum;
    }
}
"#;

/// Launch extended attention supporting any head_dim.
pub fn attention_extended(
    cache: &KernelCache,
    device: &WarpDevice,
    q: &GpuTensor<f32>,
    k: &GpuTensor<f32>,
    v: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    batch: u32,
    seq_len: u32,
    head_dim: u32,
    causal: bool,
) -> Result<(), DeviceError> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let causal_u32 = if causal { 1u32 } else { 0u32 };

    let f = cache.get_or_compile(device, EXTENDED_ATTENTION_SRC, "warp_attention_ext")?;

    // Round block size up to next power of 2 for reduction
    let block_size = head_dim.next_power_of_two().max(32);
    let shared_mem = block_size * 4; // float per thread

    let cfg = LaunchConfig {
        grid_dim: (seq_len, batch, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_mem,
    };

    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&q.data)
            .arg(&k.data)
            .arg(&v.data)
            .arg(&batch)
            .arg(&seq_len)
            .arg(&head_dim)
            .arg(&scale)
            .arg(&causal_u32)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// Smart attention dispatch: picks best kernel for the head_dim.
pub fn attention_best(
    cache: &KernelCache,
    device: &WarpDevice,
    q: &GpuTensor<f32>,
    k: &GpuTensor<f32>,
    v: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    batch: u32,
    seq_len: u32,
    head_dim: u32,
    causal: bool,
) -> Result<(), DeviceError> {
    if head_dim <= 32 {
        // Use warp-level flash attention (fastest for small head_dim)
        crate::attention::attention_flash(cache, device, q, k, v, out, batch, seq_len, head_dim, causal)
    } else {
        // Use extended attention with shared memory reduction
        attention_extended(cache, device, q, k, v, out, batch, seq_len, head_dim, causal)
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
    fn extended_attention_64d() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (b, n, d) = (1u32, 16u32, 64u32);
        let total = (b * n * d) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, d as usize]);

        let q_data: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.08).collect();
        let k_data: Vec<f32> = (0..total).map(|i| ((i % 11) as f32 - 5.0) * 0.08).collect();
        let v_data: Vec<f32> = (0..total).map(|i| ((i % 7) as f32 - 3.0) * 0.08).collect();

        // CPU reference
        let mut cpu_out = vec![0.0f32; total];
        crate::attention::cpu_attention(&q_data, &k_data, &v_data, &mut cpu_out, b as usize, n as usize, d as usize, false);

        let q = GpuTensor::from_host(&dev, &q_data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &k_data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &v_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        attention_extended(&cache, &dev, &q, &k, &v, &mut out, b, n, d, false).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();
        let max_err: f32 = gpu_out.iter().zip(cpu_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Extended attention D=64: max error = {max_err:.6}");
        assert!(max_err < 1e-3, "Max error {max_err} too high");
    }

    #[test]
    fn extended_attention_128d_causal() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // LLaMA head_dim = 128
        let (b, n, d) = (1u32, 32u32, 128u32);
        let total = (b * n * d) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, d as usize]);

        let q_data: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let k_data: Vec<f32> = (0..total).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
        let v_data: Vec<f32> = (0..total).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();

        let mut cpu_out = vec![0.0f32; total];
        crate::attention::cpu_attention(&q_data, &k_data, &v_data, &mut cpu_out, b as usize, n as usize, d as usize, true);

        let q = GpuTensor::from_host(&dev, &q_data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &k_data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &v_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        attention_extended(&cache, &dev, &q, &k, &v, &mut out, b, n, d, true).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();
        let max_err: f32 = gpu_out.iter().zip(cpu_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Extended causal attention D=128 (LLaMA head_dim): max error = {max_err:.6}");
        assert!(max_err < 1e-2, "Max error {max_err} too high");
    }

    #[test]
    fn attention_dispatch() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // Should use flash for D=32
        let (b, n, d) = (1u32, 8u32, 32u32);
        let total = (b * n * d) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, d as usize]);

        let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
        let q = GpuTensor::from_host(&dev, &data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        attention_best(&cache, &dev, &q, &k, &v, &mut out, b, n, d, true).unwrap();
        dev.synchronize().unwrap();
        let result = out.to_host(&dev).unwrap();
        assert!(result.iter().all(|v| v.is_finite()));
        println!("Attention dispatch D=32: uses flash (warp-level)");

        // Should use extended for D=128
        let d = 128u32;
        let total = (b * n * d) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, d as usize]);
        let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001).collect();
        let q = GpuTensor::from_host(&dev, &data, shape.clone(), DType::F32).unwrap();
        let k = GpuTensor::from_host(&dev, &data, shape.clone(), DType::F32).unwrap();
        let v = GpuTensor::from_host(&dev, &data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        attention_best(&cache, &dev, &q, &k, &v, &mut out, b, n, d, true).unwrap();
        dev.synchronize().unwrap();
        let result = out.to_host(&dev).unwrap();
        assert!(result.iter().all(|v| v.is_finite()));
        println!("Attention dispatch D=128: uses extended (shared mem reduction)");
    }
}

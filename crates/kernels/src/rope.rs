//! Rotary Position Embedding (RoPE).
//!
//! RoPE encodes position information by rotating pairs of dimensions
//! in the query/key vectors. Used by LLaMA, Mistral, Qwen, and most
//! modern transformers.
//!
//! For each pair (x_2i, x_{2i+1}) at position p:
//!   x_2i'   = x_2i * cos(θ_p_i) - x_{2i+1} * sin(θ_p_i)
//!   x_{2i+1}' = x_2i * sin(θ_p_i) + x_{2i+1} * cos(θ_p_i)
//!
//! where θ_p_i = p * base^(-2i/d), base typically 10000.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

const ROPE_SRC: &str = r#"
extern "C" __global__ void warp_rope(
    float *out,          // [B, N, D]
    const float *input,  // [B, N, D]
    unsigned int B,
    unsigned int N,      // seq_len
    unsigned int D,      // head_dim (must be even)
    float base,          // typically 10000.0
    unsigned int offset  // position offset (for KV cache continuation)
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = B * N * (D / 2);
    if (idx >= total) return;

    unsigned int pair = idx % (D / 2);         // which dimension pair
    unsigned int pos_in_seq = (idx / (D / 2)) % N;  // position in sequence
    unsigned int b = idx / (N * (D / 2));      // batch

    unsigned int pos = pos_in_seq + offset;

    // θ = pos * base^(-2*pair/D)
    float freq = 1.0f / powf(base, 2.0f * (float)pair / (float)D);
    float theta = (float)pos * freq;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    unsigned int base_idx = b * N * D + pos_in_seq * D + 2 * pair;
    float x0 = input[base_idx];
    float x1 = input[base_idx + 1];

    out[base_idx]     = x0 * cos_t - x1 * sin_t;
    out[base_idx + 1] = x0 * sin_t + x1 * cos_t;
}
"#;

/// Apply RoPE to a tensor.
/// input/out: [batch, seq_len, head_dim] where head_dim is even.
pub fn rope(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    batch: u32,
    seq_len: u32,
    head_dim: u32,
    base: f32,
    offset: u32,
) -> Result<(), DeviceError> {
    assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");

    let f = cache.get_or_compile(device, ROPE_SRC, "warp_rope")?;
    let total = batch * seq_len * (head_dim / 2);
    let cfg = LaunchConfig::for_num_elems(total);

    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&input.data)
            .arg(&batch)
            .arg(&seq_len)
            .arg(&head_dim)
            .arg(&base)
            .arg(&offset)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// CPU reference RoPE.
pub fn cpu_rope(
    input: &[f32], out: &mut [f32],
    batch: usize, seq_len: usize, head_dim: usize,
    base: f32, offset: usize,
) {
    for b in 0..batch {
        for pos_in_seq in 0..seq_len {
            let pos = pos_in_seq + offset;
            for pair in 0..(head_dim / 2) {
                let freq = 1.0 / base.powf(2.0 * pair as f32 / head_dim as f32);
                let theta = pos as f32 * freq;
                let (sin_t, cos_t) = theta.sin_cos();

                let idx = b * seq_len * head_dim + pos_in_seq * head_dim + 2 * pair;
                let x0 = input[idx];
                let x1 = input[idx + 1];

                out[idx] = x0 * cos_t - x1 * sin_t;
                out[idx + 1] = x0 * sin_t + x1 * cos_t;
            }
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
    fn gpu_rope_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (b, n, d) = (2u32, 16u32, 32u32);
        let total = (b * n * d) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, d as usize]);

        let input_data: Vec<f32> = (0..total).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();

        let mut cpu_out = vec![0.0f32; total];
        cpu_rope(&input_data, &mut cpu_out, b as usize, n as usize, d as usize, 10000.0, 0);

        let input = GpuTensor::from_host(&dev, &input_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        rope(&cache, &dev, &input, &mut out, b, n, d, 10000.0, 0).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();
        let max_err: f32 = gpu_out.iter().zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("RoPE B={b} N={n} D={d}: max error = {max_err:.6}");
        assert!(max_err < 1e-4, "Max error {max_err} too high");
    }

    #[test]
    fn gpu_rope_with_offset() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (b, n, d) = (1u32, 8u32, 16u32);
        let total = (b * n * d) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, d as usize]);

        let input_data: Vec<f32> = (0..total).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let offset = 42u32; // simulating KV cache continuation

        let mut cpu_out = vec![0.0f32; total];
        cpu_rope(&input_data, &mut cpu_out, b as usize, n as usize, d as usize, 10000.0, offset as usize);

        let input = GpuTensor::from_host(&dev, &input_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        rope(&cache, &dev, &input, &mut out, b, n, d, 10000.0, offset).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();
        let max_err: f32 = gpu_out.iter().zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("RoPE with offset={offset}: max error = {max_err:.6}");
        assert!(max_err < 1e-4);
    }

    #[test]
    fn rope_preserves_norm() {
        // RoPE is a rotation — it should preserve vector norms
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (b, n, d) = (1u32, 4u32, 8u32);
        let total = (b * n * d) as usize;
        let shape = Shape::from_static(&[b as usize, n as usize, d as usize]);

        let input_data: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let input = GpuTensor::from_host(&dev, &input_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        rope(&cache, &dev, &input, &mut out, b, n, d, 10000.0, 0).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();

        for pos in 0..n as usize {
            let start = pos * d as usize;
            let end = start + d as usize;
            let norm_before: f32 = input_data[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_after: f32 = gpu_out[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm_before - norm_after).abs() < 1e-4,
                "Norm not preserved at pos {pos}: before={norm_before}, after={norm_after}"
            );
        }
        println!("RoPE preserves vector norms: verified");
    }
}

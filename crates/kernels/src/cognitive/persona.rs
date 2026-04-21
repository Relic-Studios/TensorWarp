//! # K2 — Persona Coherence
//!
//! Single fused reduction kernel computing
//!
//! ```text
//! coherence = dot(h, v_thomas) / (||h|| + eps)
//! ```
//!
//! where `v_thomas` is pre-normalised to unit length (so the `||v||` term
//! drops out). Reads `h` once from global memory, accumulates both the
//! dot product and the squared-norm in FP32 registers per-thread, then
//! reduces across the block via warp-shuffle + shared-memory tree.
//!
//! ## Numerical stability
//!
//! Even though `h` may be stored in FP16 in production, accumulation is
//! **always FP32** — FP16 max representable is 65,504, and ‖h‖² can easily
//! exceed that for `d_model = 2048` with typical residual magnitudes.
//! This scaffold uses FP32 inputs; an FP16-input variant is a future pass.
//!
//! ## Output
//!
//! A single `f32` scalar into a device buffer. The caller performs a
//! 4-byte DtoH copy (via `buffer.to_host(device)`) to retrieve it.
//! Target latency: <20 µs per call, bounded by kernel-launch overhead.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

const PERSONA_COHERENCE_SRC: &str = r#"
#define BLOCK_SIZE 256

extern "C" __global__ void cognitive_persona_coherence(
    const float* __restrict__ h,          // [d_model]
    const float* __restrict__ v_thomas,   // [d_model], unit-normalized
    float* __restrict__ out_coherence,    // [1]
    unsigned int d_model,
    float epsilon)
{
    const int tid = threadIdx.x;

    // Per-thread partial sums.
    float dot = 0.0f;
    float norm_sq = 0.0f;

    // Grid-stride over d_model. Single block assumed (d_model ≤ 4096 for Gemma 4).
    for (unsigned int i = (unsigned int)tid; i < d_model; i += BLOCK_SIZE) {
        float hv = h[i];
        float tv = v_thomas[i];
        dot    += hv * tv;       // FMA on hardware that supports it.
        norm_sq += hv * hv;
    }

    // Warp-level reduction via shuffle.
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot     += __shfl_down_sync(0xFFFFFFFF, dot,     offset);
        norm_sq += __shfl_down_sync(0xFFFFFFFF, norm_sq, offset);
    }

    // Block-level reduction via shared memory.
    __shared__ float smem_dot[BLOCK_SIZE / 32];
    __shared__ float smem_norm[BLOCK_SIZE / 32];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        smem_dot[warp_id]  = dot;
        smem_norm[warp_id] = norm_sq;
    }
    __syncthreads();

    // Final reduction in the first warp.
    if (warp_id == 0) {
        int num_warps = BLOCK_SIZE / 32;
        float d = (lane_id < num_warps) ? smem_dot[lane_id]  : 0.0f;
        float n = (lane_id < num_warps) ? smem_norm[lane_id] : 0.0f;

        for (int offset = 16; offset > 0; offset >>= 1) {
            d += __shfl_down_sync(0xFFFFFFFF, d, offset);
            n += __shfl_down_sync(0xFFFFFFFF, n, offset);
        }

        if (lane_id == 0) {
            // Assumes v_thomas is unit-normalized, so ||v|| == 1 → omitted.
            float norm = sqrtf(n) + epsilon;
            out_coherence[0] = d / norm;
        }
    }
}
"#;

/// Launch the fused persona-coherence reduction.
///
/// ## Arguments
/// - `h`: residual stream, shape `[d_model]`, FP32.
/// - `v_thomas`: persona direction, shape `[d_model]`, FP32, **unit-normalised**.
/// - `out_coherence`: pre-allocated `f32` buffer of length 1.
/// - `d_model`: dimension — must match tensor shapes.
/// - `epsilon`: numerical floor on `‖h‖`, typically `1e-8`.
///
/// ## Returns
/// A scalar in `[-1, 1]` written to `out_coherence[0]`:
/// - `1.0` → perfect alignment with persona direction
/// - `0.0` → orthogonal (generic, no persona present)
/// - negative → anti-aligned (adversarial or corrupted)
pub fn launch_persona_coherence(
    device: &WarpDevice,
    h: &GpuTensor<f32>,
    v_thomas: &GpuTensor<f32>,
    out_coherence: &mut GpuTensor<f32>,
    d_model: u32,
    epsilon: f32,
) -> Result<(), DeviceError> {
    if (h.numel as u32) < d_model {
        return Err(DeviceError::Launch(format!(
            "h too small: {} < {}",
            h.numel, d_model
        )));
    }
    if (v_thomas.numel as u32) < d_model {
        return Err(DeviceError::Launch(format!(
            "v_thomas too small: {} < {}",
            v_thomas.numel, d_model
        )));
    }
    if out_coherence.numel < 1 {
        return Err(DeviceError::Launch(
            "out_coherence buffer empty".into(),
        ));
    }

    let (_module, func) = device.load_cuda_source(
        PERSONA_COHERENCE_SRC,
        "cognitive_persona_coherence",
    )?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device
            .stream
            .launch_builder(&func)
            .arg(&h.data)
            .arg(&v_thomas.data)
            .arg(&mut out_coherence.data)
            .arg(&d_model)
            .arg(&epsilon)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| {
                DeviceError::Launch(e.to_string())
            })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::{DType, Shape};

    fn get_device() -> Option<WarpDevice> {
        WarpDevice::new(0).ok()
    }

    fn run_coherence(dev: &WarpDevice, h_data: &[f32], v_data: &[f32]) -> f32 {
        let d_model = h_data.len() as u32;
        let h = GpuTensor::from_host(
            dev,
            h_data,
            Shape::from_static(&[d_model as usize]),
            DType::F32,
        )
        .unwrap();
        let v = GpuTensor::from_host(
            dev,
            v_data,
            Shape::from_static(&[d_model as usize]),
            DType::F32,
        )
        .unwrap();
        let mut out =
            GpuTensor::<f32>::zeros(dev, Shape::from_static(&[1]), DType::F32).unwrap();

        launch_persona_coherence(dev, &h, &v, &mut out, d_model, 1e-8).unwrap();
        dev.synchronize().unwrap();
        out.to_host(dev).unwrap()[0]
    }

    #[test]
    fn coherence_parallel_vectors_is_one() {
        let dev = match get_device() {
            Some(d) => d,
            None => {
                println!("No CUDA device, skipping");
                return;
            }
        };
        // h and v both point along the first basis direction → cosine = 1
        let mut h = vec![0.0f32; 2048];
        let mut v = vec![0.0f32; 2048];
        h[0] = 1.0;
        v[0] = 1.0;
        let c = run_coherence(&dev, &h, &v);
        println!("parallel coherence = {c}");
        assert!((c - 1.0).abs() < 1e-4, "expected 1.0, got {c}");
    }

    #[test]
    fn coherence_orthogonal_vectors_is_zero() {
        let dev = match get_device() {
            Some(d) => d,
            None => {
                println!("No CUDA device, skipping");
                return;
            }
        };
        let mut h = vec![0.0f32; 2048];
        let mut v = vec![0.0f32; 2048];
        h[0] = 1.0;
        v[1] = 1.0;
        let c = run_coherence(&dev, &h, &v);
        println!("orthogonal coherence = {c}");
        assert!(c.abs() < 1e-4, "expected 0.0, got {c}");
    }

    #[test]
    fn coherence_45_degree_is_half_sqrt2() {
        let dev = match get_device() {
            Some(d) => d,
            None => {
                println!("No CUDA device, skipping");
                return;
            }
        };
        // h = (1, 1, 0, …), v = (1, 0, 0, …) → cos(45°) = 1/√2
        let mut h = vec![0.0f32; 2048];
        let mut v = vec![0.0f32; 2048];
        h[0] = 1.0;
        h[1] = 1.0;
        v[0] = 1.0;
        let c = run_coherence(&dev, &h, &v);
        let expected = 1.0f32 / 2.0f32.sqrt();
        println!("45° coherence = {c}, expected {expected}");
        assert!((c - expected).abs() < 1e-4, "expected {expected}, got {c}");
    }

    #[test]
    fn coherence_anti_parallel_is_negative_one() {
        let dev = match get_device() {
            Some(d) => d,
            None => {
                println!("No CUDA device, skipping");
                return;
            }
        };
        let mut h = vec![0.0f32; 2048];
        let mut v = vec![0.0f32; 2048];
        h[0] = 1.0;
        v[0] = -1.0;
        let c = run_coherence(&dev, &h, &v);
        println!("anti-parallel coherence = {c}");
        assert!((c + 1.0).abs() < 1e-4, "expected -1.0, got {c}");
    }
}

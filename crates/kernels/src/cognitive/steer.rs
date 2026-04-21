//! # K3 — Steering Injection (SAXPY on the residual stream)
//!
//! In-place vector update:
//!
//! ```text
//! h[i] ← h[i] + α · v[i]   ∀ i ∈ [0, d_model)
//! ```
//!
//! Implements Contrastive Activation Addition (CAA) for persona steering:
//! the Thomas Vector `v` is added to the residual stream at a chosen layer,
//! scaled by `α` from the PID controller that's watching coherence.
//!
//! ## CUDA Graph compatibility (the critical design point)
//!
//! `α` changes per token (PID output depends on current coherence). If we
//! passed `α` by value, CUDA graph capture would bake in whichever value
//! happened to be in scope at capture time — the graph would become frozen
//! at the wrong alpha and replay it forever.
//!
//! The fix per Gemini's research: the kernel takes `alpha_ptr: *const f32`
//! instead of `alpha: f32`. During graph capture, the pointer's *address*
//! is recorded. Between graph replays, the host does a tiny async HtoD copy
//! to overwrite the value at that address. The graph stays immutable; the
//! behaviour stays dynamic. Mirrors how TensorWarp already handles the
//! per-token position scalar (`pos_buf`).
//!
//! ## Performance
//!
//! Pure memory-bound SAXPY. For `d_model = 2048` (8 KB each for `h` and
//! `v`), the kernel reads 16 KB and writes 8 KB — target <10 µs per call.

use cudarc::driver::{LaunchConfig, PushKernelArg, CudaSlice};

use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

const STEER_INJECT_SRC: &str = r#"
extern "C" __global__ void cognitive_steer_inject(
    float* __restrict__ h,              // [d_model], modified in place
    const float* __restrict__ v,        // [d_model], Thomas Vector
    const float* __restrict__ alpha_ptr, // [1], device-side scalar
    unsigned int d_model)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_model) return;

    float a = *alpha_ptr;  // broadcast across all threads
    h[i] = h[i] + a * v[i];
}
"#;

/// Launch the in-place steering injection.
///
/// ## Arguments
/// - `h`: residual stream, shape `[d_model]`, FP32 — **mutated in place**.
/// - `v`: Thomas Vector, shape `[d_model]`, FP32.
/// - `alpha_ptr`: device-side buffer of exactly 1 `f32` holding the current
///   steering strength. Allocate once at setup (`device.alloc_zeros::<f32>(1)?`),
///   update between forward passes with `device.htod_copy(&[alpha], alpha_ptr)?`.
/// - `d_model`: dimension.
///
/// ## Graph-capture pattern
///
/// ```ignore
/// // One-time setup
/// let alpha_buf: CudaSlice<f32> = device.alloc_zeros::<f32>(1)?;
///
/// // Per-token (outside or inside a CUDA graph)
/// device.htod_copy(&[new_alpha], &mut alpha_buf)?;    // update scalar
/// launch_steer_inject(device, &mut h, &v, &alpha_buf, d_model)?;
/// ```
pub fn launch_steer_inject(
    device: &WarpDevice,
    h: &mut GpuTensor<f32>,
    v: &GpuTensor<f32>,
    alpha_ptr: &CudaSlice<f32>,
    d_model: u32,
) -> Result<(), DeviceError> {
    if (h.numel as u32) < d_model {
        return Err(DeviceError::Launch(format!(
            "h too small: {} < {}",
            h.numel, d_model
        )));
    }
    if (v.numel as u32) < d_model {
        return Err(DeviceError::Launch(format!(
            "v too small: {} < {}",
            v.numel, d_model
        )));
    }
    if alpha_ptr.len() < 1 {
        return Err(DeviceError::Launch("alpha_ptr buffer empty".into()));
    }

    let (_module, func) =
        device.load_cuda_source(STEER_INJECT_SRC, "cognitive_steer_inject")?;

    let block_size = 256u32;
    let grid = (d_model + block_size - 1) / block_size;

    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device
            .stream
            .launch_builder(&func)
            .arg(&mut h.data)
            .arg(&v.data)
            .arg(alpha_ptr)
            .arg(&d_model)
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

    #[test]
    fn steer_scaled_add_is_saxpy() {
        let dev = match get_device() {
            Some(d) => d,
            None => {
                println!("No CUDA device, skipping");
                return;
            }
        };

        let d_model = 2048u32;
        let h_init = vec![0.0f32; d_model as usize];
        let v_data = vec![1.0f32; d_model as usize];
        let alpha = 2.0f32;

        let mut h = GpuTensor::from_host(
            &dev,
            &h_init,
            Shape::from_static(&[d_model as usize]),
            DType::F32,
        )
        .unwrap();
        let v = GpuTensor::from_host(
            &dev,
            &v_data,
            Shape::from_static(&[d_model as usize]),
            DType::F32,
        )
        .unwrap();
        let alpha_ptr = dev.htod(&[alpha]).unwrap();

        launch_steer_inject(&dev, &mut h, &v, &alpha_ptr, d_model).unwrap();
        dev.synchronize().unwrap();

        let h_out = h.to_host(&dev).unwrap();
        // h_init = 0, v = 1, α = 2 → h_out[i] = 0 + 2·1 = 2.0 for all i
        let max_err = h_out
            .iter()
            .map(|&x| (x - 2.0).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "steer SAXPY error {max_err}");
        println!("steer SAXPY: max error = {max_err:.2e}");
    }

    #[test]
    fn steer_is_additive_with_nonzero_start() {
        let dev = match get_device() {
            Some(d) => d,
            None => {
                println!("No CUDA device, skipping");
                return;
            }
        };

        let d_model = 512u32;
        let h_init: Vec<f32> = (0..d_model).map(|i| i as f32 * 0.01).collect();
        let v_data: Vec<f32> = (0..d_model).map(|i| (i as f32).sin()).collect();
        let alpha = 0.5f32;

        let mut h = GpuTensor::from_host(
            &dev,
            &h_init,
            Shape::from_static(&[d_model as usize]),
            DType::F32,
        )
        .unwrap();
        let v = GpuTensor::from_host(
            &dev,
            &v_data,
            Shape::from_static(&[d_model as usize]),
            DType::F32,
        )
        .unwrap();
        let alpha_ptr = dev.htod(&[alpha]).unwrap();

        launch_steer_inject(&dev, &mut h, &v, &alpha_ptr, d_model).unwrap();
        dev.synchronize().unwrap();

        let h_out = h.to_host(&dev).unwrap();
        // Expected: h_out[i] = h_init[i] + α·v[i]
        let max_err = h_out
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let expected = h_init[i] + alpha * v_data[i];
                (x - expected).abs()
            })
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "steer additive error {max_err}");
        println!("steer additive: max error = {max_err:.2e}");
    }
}

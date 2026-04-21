//! # K1 — Sparse Autoencoder projection + Top-K extraction
//!
//! Given residual `h ∈ R^{d_model}` and encoder weights `W_enc ∈ R^{d_model × d_sae}`
//! (column-major) with bias `b_enc ∈ R^{d_sae}`, this kernel computes:
//!
//! ```text
//! latent = ReLU(h · W_enc + b_enc)   ∈ R^{d_sae}
//! (indices, values) = top_k(latent, k)
//! ```
//!
//! The full `latent` vector is never materialised in global memory — top-K
//! extraction is fused directly into the GEMV epilogue. Only the ~400 B
//! of sparse output (k pairs of `(u32, f32)`) is written back.
//!
//! ## Phase 1 implementation notes
//!
//! - FP32 throughout for correctness-first scaffolding. The research
//!   recommended INT4-quantised W_enc + FP16 h for the production target;
//!   that's a future pass once the FP32 path is verified.
//! - Column-major W_enc is required: column `f` is the encoder direction
//!   for feature `f`, and threads processing adjacent features read
//!   adjacent elements of the same row of W_enc — coalesced loads.
//! - One CUDA block per launch. Each thread processes `d_sae / block_size`
//!   features via grid-stride, maintains its own local top-K in registers,
//!   then thread 0 merges the per-thread lists into the final top-K.
//! - Top-K size is compile-time (TOP_K=50). To use a different k, change
//!   the constant and the kernel JIT-compiles a new variant (nvrtc caches).
//!
//! ## Known tradeoff vs. research's GridSelect recommendation
//!
//! The research proposed a warp-cooperative GridSelect for maximum
//! occupancy. This scaffold uses a simpler per-thread top-K + thread-0
//! merge — correctness-first. Gemini's sketch is linked in
//! `synderesis/docs/research_answers/gpu-kernels-activation-capture.md`
//! as the optimisation target.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Number of top SAE features extracted per capture. Compile-time in the
/// CUDA kernel — change both here and in `SAE_TOPK_SRC`.
pub const TOP_K: usize = 50;

/// One (index, value) pair from the sparse top-K output.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SaeTopK {
    pub index: u32,
    pub value: f32,
}

const SAE_TOPK_SRC: &str = r#"
// Compile-time constants — keep in sync with TOP_K in src/cognitive/sae.rs
#define TOP_K 50
#define MAX_BLOCK_SIZE 64

struct Pair { float val; unsigned int idx; };

__device__ __forceinline__ void insert_topk(
    float v, unsigned int i,
    float* vals, unsigned int* idxs)
{
    // Branch predictor strongly favours this reject path — sparse activations
    // mean most features never exceed the TOP_K-th threshold.
    if (v <= vals[TOP_K - 1]) return;

    int pos = TOP_K - 2;
    while (pos >= 0 && vals[pos] < v) {
        vals[pos + 1] = vals[pos];
        idxs[pos + 1] = idxs[pos];
        pos--;
    }
    vals[pos + 1] = v;
    idxs[pos + 1] = i;
}

extern "C" __global__ void cognitive_sae_topk(
    const float* __restrict__ W_enc,      // column-major [d_model, d_sae]
    const float* __restrict__ b_enc,      // [d_sae]
    const float* __restrict__ h,          // [d_model]
    unsigned int* __restrict__ out_idxs,  // [TOP_K]
    float* __restrict__ out_vals,         // [TOP_K]
    unsigned int d_model,
    unsigned int d_sae)
{
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Per-thread top-K in registers.
    float local_vals[TOP_K];
    unsigned int local_idxs[TOP_K];
    #pragma unroll
    for (int i = 0; i < TOP_K; i++) {
        local_vals[i] = -1e30f;
        local_idxs[i] = 0xFFFFFFFFu;
    }

    // Grid-stride over SAE features. Column-major W_enc means the weights
    // for feature `f` are contiguous: W_enc[0..d_model-1 + f*d_model].
    for (unsigned int f = (unsigned int)tid; f < d_sae; f += (unsigned int)block_size) {
        float dp = 0.0f;
        const float* col = W_enc + (size_t)f * (size_t)d_model;
        #pragma unroll 4
        for (unsigned int i = 0; i < d_model; i++) {
            dp += h[i] * col[i];
        }
        float act = fmaxf(0.0f, dp + b_enc[f]);
        insert_topk(act, f, local_vals, local_idxs);
    }

    // Shared-memory buffers for merging per-thread top-K lists.
    __shared__ float smem_vals[MAX_BLOCK_SIZE * TOP_K];
    __shared__ unsigned int smem_idxs[MAX_BLOCK_SIZE * TOP_K];

    #pragma unroll
    for (int i = 0; i < TOP_K; i++) {
        smem_vals[tid * TOP_K + i] = local_vals[i];
        smem_idxs[tid * TOP_K + i] = local_idxs[i];
    }
    __syncthreads();

    // Thread 0 merges all per-thread lists into the final top-K.
    if (tid == 0) {
        float final_vals[TOP_K];
        unsigned int final_idxs[TOP_K];
        #pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            final_vals[i] = -1e30f;
            final_idxs[i] = 0xFFFFFFFFu;
        }

        for (int t = 0; t < block_size; t++) {
            for (int i = 0; i < TOP_K; i++) {
                insert_topk(
                    smem_vals[t * TOP_K + i],
                    smem_idxs[t * TOP_K + i],
                    final_vals, final_idxs);
            }
        }

        for (int i = 0; i < TOP_K; i++) {
            out_vals[i]  = final_vals[i];
            out_idxs[i]  = final_idxs[i];
        }
    }
}
"#;

/// Launch the fused SAE projection + Top-K kernel.
///
/// ## Arguments
/// - `w_enc`: encoder weights, **column-major**, shape `[d_model, d_sae]`, FP32.
/// - `b_enc`: encoder bias, shape `[d_sae]`, FP32.
/// - `h`: residual stream, shape `[d_model]`, FP32.
/// - `out_idxs`: pre-allocated output buffer of `u32`, length `TOP_K`.
/// - `out_vals`: pre-allocated output buffer of `f32`, length `TOP_K`.
/// - `d_model`, `d_sae`: dimensions — enforced to match tensor shapes.
///
/// ## Constraints
/// - `d_sae` must be > 0. No upper bound enforced — the grid-stride loop scales.
/// - `out_idxs` and `out_vals` must have `numel >= TOP_K`.
///
/// ## Determinism
/// Floating-point nondeterminism: parallel reductions have non-associative
/// summation, so the exact `latent` values (and possibly the tie-broken
/// top-K tail) may differ across launches. For validation, check **exact
/// index match** of the top-K, not bitwise-equal values.
pub fn launch_sae_topk(
    device: &WarpDevice,
    w_enc: &GpuTensor<f32>,
    b_enc: &GpuTensor<f32>,
    h: &GpuTensor<f32>,
    out_idxs: &mut GpuTensor<u32>,
    out_vals: &mut GpuTensor<f32>,
    d_model: u32,
    d_sae: u32,
) -> Result<(), DeviceError> {
    // Sanity checks.
    if (w_enc.numel as u32) < d_model.saturating_mul(d_sae) {
        return Err(DeviceError::Launch(format!(
            "w_enc too small: {} elements, need {} × {} = {}",
            w_enc.numel,
            d_model,
            d_sae,
            d_model as usize * d_sae as usize,
        )));
    }
    if (b_enc.numel as u32) < d_sae {
        return Err(DeviceError::Launch(format!(
            "b_enc too small: {} elements, need {}",
            b_enc.numel, d_sae,
        )));
    }
    if (h.numel as u32) < d_model {
        return Err(DeviceError::Launch(format!(
            "h too small: {} elements, need {}",
            h.numel, d_model,
        )));
    }
    if out_idxs.numel < TOP_K || out_vals.numel < TOP_K {
        return Err(DeviceError::Launch(format!(
            "output buffers too small: idxs={}, vals={}, need {}",
            out_idxs.numel, out_vals.numel, TOP_K,
        )));
    }

    let (_module, func) = device.load_cuda_source(SAE_TOPK_SRC, "cognitive_sae_topk")?;

    // Block size must be ≤ MAX_BLOCK_SIZE (64) per the kernel's SMEM layout.
    let block_size = 64u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0, // all SMEM is statically declared in the kernel
    };

    unsafe {
        device
            .stream
            .launch_builder(&func)
            .arg(&w_enc.data)
            .arg(&b_enc.data)
            .arg(&h.data)
            .arg(&mut out_idxs.data)
            .arg(&mut out_vals.data)
            .arg(&d_model)
            .arg(&d_sae)
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
    fn sae_topk_returns_largest_features_in_order() {
        let dev = match get_device() {
            Some(d) => d,
            None => {
                println!("No CUDA device, skipping");
                return;
            }
        };

        let d_model = 4u32;
        let d_sae = 64u32;

        // W_enc column-major: column i has value (i+1) at row 0, zeros elsewhere.
        // With h = [1, 0, 0, 0], dot(h, col_i) = i+1, so latent[i] = ReLU(i+1) = i+1.
        // Top-50 should be indices 63, 62, 61, ..., 14 with values 64, 63, ..., 15.
        let mut w_data = vec![0.0f32; (d_model * d_sae) as usize];
        for i in 0..d_sae as usize {
            w_data[i * d_model as usize] = (i + 1) as f32;
        }
        let b_data = vec![0.0f32; d_sae as usize];
        let h_data = vec![1.0f32, 0.0, 0.0, 0.0];

        let w_enc = GpuTensor::from_host(
            &dev,
            &w_data,
            Shape::from_static(&[d_model as usize, d_sae as usize]),
            DType::F32,
        )
        .unwrap();
        let b_enc = GpuTensor::from_host(
            &dev,
            &b_data,
            Shape::from_static(&[d_sae as usize]),
            DType::F32,
        )
        .unwrap();
        let h = GpuTensor::from_host(
            &dev,
            &h_data,
            Shape::from_static(&[d_model as usize]),
            DType::F32,
        )
        .unwrap();
        let mut out_idxs =
            GpuTensor::<u32>::zeros(&dev, Shape::from_static(&[TOP_K]), DType::U32).unwrap();
        let mut out_vals =
            GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[TOP_K]), DType::F32).unwrap();

        launch_sae_topk(
            &dev,
            &w_enc,
            &b_enc,
            &h,
            &mut out_idxs,
            &mut out_vals,
            d_model,
            d_sae,
        )
        .unwrap();
        dev.synchronize().unwrap();

        let idxs = out_idxs.to_host(&dev).unwrap();
        let vals = out_vals.to_host(&dev).unwrap();

        println!("top-K indices: {:?}", &idxs[..10]);
        println!("top-K values:  {:?}", &vals[..10]);

        // Highest should be index 63 with value 64.0.
        assert_eq!(idxs[0], 63, "top-1 index should be 63");
        assert!((vals[0] - 64.0).abs() < 1e-4, "top-1 value {}", vals[0]);

        // Top-K should be strictly decreasing.
        for i in 1..10 {
            assert!(
                vals[i] < vals[i - 1],
                "top-K not sorted descending at index {i}: {} >= {}",
                vals[i],
                vals[i - 1]
            );
        }

        // Index[i] should be (63 - i) for the first 50 entries.
        for i in 0..10 {
            let expected_idx = 63 - i as u32;
            let expected_val = (64 - i) as f32;
            assert_eq!(idxs[i], expected_idx, "idx[{i}]");
            assert!(
                (vals[i] - expected_val).abs() < 1e-4,
                "val[{i}] = {}, expected {expected_val}",
                vals[i]
            );
        }
    }

    #[test]
    fn sae_topk_respects_relu_zeros_out_negatives() {
        let dev = match get_device() {
            Some(d) => d,
            None => {
                println!("No CUDA device, skipping");
                return;
            }
        };

        let d_model = 4u32;
        let d_sae = 64u32;

        // W_enc column-major: column i has value -(i+1) at row 0.
        // With h = [1, 0, 0, 0] and b = 0, pre-ReLU latent[i] = -(i+1), all negative.
        // Post-ReLU latent = 0 for all features. Top-K values should all be 0.
        let mut w_data = vec![0.0f32; (d_model * d_sae) as usize];
        for i in 0..d_sae as usize {
            w_data[i * d_model as usize] = -((i + 1) as f32);
        }
        let b_data = vec![0.0f32; d_sae as usize];
        let h_data = vec![1.0f32, 0.0, 0.0, 0.0];

        let w_enc = GpuTensor::from_host(
            &dev,
            &w_data,
            Shape::from_static(&[d_model as usize, d_sae as usize]),
            DType::F32,
        )
        .unwrap();
        let b_enc = GpuTensor::from_host(
            &dev,
            &b_data,
            Shape::from_static(&[d_sae as usize]),
            DType::F32,
        )
        .unwrap();
        let h = GpuTensor::from_host(
            &dev,
            &h_data,
            Shape::from_static(&[d_model as usize]),
            DType::F32,
        )
        .unwrap();
        let mut out_idxs =
            GpuTensor::<u32>::zeros(&dev, Shape::from_static(&[TOP_K]), DType::U32).unwrap();
        let mut out_vals =
            GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[TOP_K]), DType::F32).unwrap();

        launch_sae_topk(
            &dev,
            &w_enc,
            &b_enc,
            &h,
            &mut out_idxs,
            &mut out_vals,
            d_model,
            d_sae,
        )
        .unwrap();
        dev.synchronize().unwrap();

        let vals = out_vals.to_host(&dev).unwrap();
        // Every value should be clamped to 0 by ReLU.
        let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("all-negative ReLU test: max post-ReLU = {max_val}");
        assert!(max_val <= 0.0, "ReLU failed; got positive {max_val}");
    }
}

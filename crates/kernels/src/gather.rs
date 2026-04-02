//! Gather, Slice, Split, and Pad kernels — the missing ops that block real models.
//!
//! Gather: index into a tensor along an axis (embedding lookup, advanced indexing)
//! Slice: extract a sub-tensor along axes
//! Split: divide a tensor into chunks along an axis
//! Pad: add padding to tensor edges

use cudarc::driver::{LaunchConfig, PushKernelArg};
use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

macro_rules! launch_err {
    ($e:expr) => {
        $e.map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))
    };
}

// ── Gather ──────────────────────────────────────────────────────
// out[i] = input[indices[i]] (along axis 0)
// This is how embedding lookups work in ONNX.

const GATHER_SRC: &str = r#"
extern "C" __global__ void warp_gather(
    float *out,             // [num_indices, inner_size]
    const float *input,     // [vocab_size, inner_size]
    const float *indices,   // [num_indices] (float because our tensors are f32)
    unsigned int inner_size,
    unsigned int num_indices,
    unsigned int vocab_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_indices * inner_size;
    if (idx >= total) return;

    unsigned int out_row = idx / inner_size;
    unsigned int col = idx % inner_size;

    int gather_idx = (int)indices[out_row];
    if (gather_idx < 0) gather_idx += (int)vocab_size; // negative indexing
    if (gather_idx >= 0 && gather_idx < (int)vocab_size) {
        out[idx] = input[gather_idx * inner_size + col];
    } else {
        out[idx] = 0.0f;
    }
}
"#;

/// Gather along axis 0: out[i] = input[indices[i]].
pub fn gather(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,    // [vocab_size, inner_size]
    indices: &GpuTensor<f32>,  // [num_indices] (stored as f32, cast to int)
    output: &mut GpuTensor<f32>, // [num_indices, inner_size]
    vocab_size: u32,
    inner_size: u32,
) -> Result<(), DeviceError> {
    let num_indices = indices.numel as u32;
    let f = cache.get_or_compile(device, GATHER_SRC, "warp_gather")?;
    let total = num_indices * inner_size;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&input.data).arg(&indices.data)
            .arg(&inner_size).arg(&num_indices).arg(&vocab_size)
            .launch(cfg))?;
    }
    Ok(())
}

// ── Slice ───────────────────────────────────────────────────────
// Extract a contiguous sub-tensor along the last axis.

const SLICE_LAST_SRC: &str = r#"
extern "C" __global__ void warp_slice_last(
    float *out,
    const float *input,
    unsigned int rows,
    unsigned int in_cols,
    unsigned int start,
    unsigned int out_cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = rows * out_cols;
    if (idx >= total) return;

    unsigned int row = idx / out_cols;
    unsigned int col = idx % out_cols;
    out[idx] = input[row * in_cols + start + col];
}
"#;

/// Slice along the last dimension: out = input[:, start:start+length].
pub fn slice_last(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
    rows: u32,
    in_cols: u32,
    start: u32,
    out_cols: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, SLICE_LAST_SRC, "warp_slice_last")?;
    let total = rows * out_cols;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&input.data)
            .arg(&rows).arg(&in_cols).arg(&start).arg(&out_cols)
            .launch(cfg))?;
    }
    Ok(())
}

// ── Split ───────────────────────────────────────────────────────
// Split a tensor into N chunks along the last axis.

/// Split along last dimension into chunks.
pub fn split_last(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    outputs: &mut [GpuTensor<f32>],
    rows: u32,
    in_cols: u32,
    split_sizes: &[u32],
) -> Result<(), DeviceError> {
    let mut offset = 0u32;
    for (i, &size) in split_sizes.iter().enumerate() {
        slice_last(cache, device, input, &mut outputs[i], rows, in_cols, offset, size)?;
        offset += size;
    }
    Ok(())
}

// ── Pad ─────────────────────────────────────────────────────────
// Pad a 2D tensor with a constant value.

const PAD_2D_SRC: &str = r#"
extern "C" __global__ void warp_pad_2d(
    float *out,
    const float *input,
    unsigned int in_rows, unsigned int in_cols,
    unsigned int out_rows, unsigned int out_cols,
    unsigned int pad_top, unsigned int pad_left,
    float pad_value
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = out_rows * out_cols;
    if (idx >= total) return;

    unsigned int out_row = idx / out_cols;
    unsigned int out_col = idx % out_cols;

    int in_row = (int)out_row - (int)pad_top;
    int in_col = (int)out_col - (int)pad_left;

    if (in_row >= 0 && in_row < (int)in_rows && in_col >= 0 && in_col < (int)in_cols) {
        out[idx] = input[in_row * in_cols + in_col];
    } else {
        out[idx] = pad_value;
    }
}
"#;

/// Pad a 2D tensor with a constant value.
pub fn pad_2d(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
    in_rows: u32, in_cols: u32,
    out_rows: u32, out_cols: u32,
    pad_top: u32, pad_left: u32,
    pad_value: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, PAD_2D_SRC, "warp_pad_2d")?;
    let total = out_rows * out_cols;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&input.data)
            .arg(&in_rows).arg(&in_cols)
            .arg(&out_rows).arg(&out_cols)
            .arg(&pad_top).arg(&pad_left)
            .arg(&pad_value)
            .launch(cfg))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    #[test]
    fn gather_embedding() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // Embedding table: 10 tokens, dim=4
        let vocab: Vec<f32> = (0..40).map(|i| i as f32 * 0.1).collect();
        let indices: Vec<f32> = vec![0.0, 3.0, 7.0, 1.0]; // look up tokens 0, 3, 7, 1

        let input = GpuTensor::from_host(&dev, &vocab, Shape::from_static(&[10, 4]), DType::F32).unwrap();
        let idx = GpuTensor::from_host(&dev, &indices, Shape::from_static(&[4]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[4, 4]), DType::F32).unwrap();

        gather(&cache, &dev, &input, &idx, &mut out, 10, 4).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        // Token 0: [0.0, 0.1, 0.2, 0.3]
        assert!((result[0] - 0.0).abs() < 1e-5);
        assert!((result[1] - 0.1).abs() < 1e-5);
        // Token 3: [1.2, 1.3, 1.4, 1.5]
        assert!((result[4] - 1.2).abs() < 1e-5);
        // Token 7: [2.8, 2.9, 3.0, 3.1]
        assert!((result[8] - 2.8).abs() < 1e-5);
        println!("Gather (embedding lookup): correct!");
    }

    #[test]
    fn slice_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let input = GpuTensor::from_host(&dev, &data, Shape::from_static(&[4, 5]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[4, 2]), DType::F32).unwrap();

        // Slice columns 1..3
        slice_last(&cache, &dev, &input, &mut out, 4, 5, 1, 2).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        assert_eq!(result[0], 1.0); // row 0, col 1
        assert_eq!(result[1], 2.0); // row 0, col 2
        assert_eq!(result[2], 6.0); // row 1, col 1
        println!("Slice: correct!");
    }

    #[test]
    fn pad_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input = GpuTensor::from_host(&dev, &data, Shape::from_static(&[2, 2]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[4, 4]), DType::F32).unwrap();

        // Pad: 1 row top, 1 row bottom, 1 col left, 1 col right
        pad_2d(&cache, &dev, &input, &mut out, 2, 2, 4, 4, 1, 1, 0.0).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        // Row 0: all zeros (padding)
        assert_eq!(result[0], 0.0);
        // Row 1, col 1: first element of input
        assert_eq!(result[5], 1.0);
        assert_eq!(result[6], 2.0);
        // Row 2, col 1-2: second row of input
        assert_eq!(result[9], 3.0);
        assert_eq!(result[10], 4.0);
        println!("Pad: correct!");
    }
}

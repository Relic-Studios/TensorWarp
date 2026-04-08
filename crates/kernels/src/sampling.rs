//! Embedding, softmax, and sampling kernels.
//!
//! These are the "missing link" between the transformer block and
//! actual text generation. Without these, we can compute hidden states
//! but can't produce tokens.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

// ── Embedding ─────────────────────────────────────────────────────

const EMBEDDING_SRC: &str = r#"
extern "C" __global__ void warp_embedding(
    float *out,              // [seq_len, hidden_size]
    const float *table,      // [vocab_size, hidden_size]
    const int *indices,      // [seq_len]
    unsigned int seq_len,
    unsigned int hidden_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = seq_len * hidden_size;
    if (idx >= total) return;

    unsigned int pos = idx / hidden_size;
    unsigned int dim = idx % hidden_size;
    int token_id = indices[pos];

    out[idx] = table[token_id * hidden_size + dim];
}
"#;

/// Token embedding lookup: out[i, :] = table[indices[i], :]
pub fn embedding(
    cache: &KernelCache,
    device: &WarpDevice,
    table: &GpuTensor<f32>,     // [vocab_size, hidden_size]
    indices: &GpuTensor<i32>,    // [seq_len]
    out: &mut GpuTensor<f32>,    // [seq_len, hidden_size]
    seq_len: u32,
    hidden_size: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, EMBEDDING_SRC, "warp_embedding")?;
    let total = seq_len * hidden_size;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&table.data)
            .arg(&indices.data)
            .arg(&seq_len)
            .arg(&hidden_size)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// F16 embedding lookup with F32 output: out[i,:] = (float)table_f16[indices[i], :]
const EMBEDDING_F16_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void warp_embedding_f16(
    float *out,
    const __half *table,
    const int *indices,
    unsigned int seq_len,
    unsigned int hidden_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = seq_len * hidden_size;
    if (idx >= total) return;

    unsigned int pos = idx / hidden_size;
    unsigned int dim = idx % hidden_size;
    int token_id = indices[pos];

    out[idx] = __half2float(table[token_id * hidden_size + dim]);
}
"#;

/// F16 embedding lookup → F32 output. Saves 50% VRAM on embedding table.
pub fn embedding_f16(
    cache: &KernelCache,
    device: &WarpDevice,
    table: &GpuTensor<half::f16>,
    indices: &GpuTensor<i32>,
    out: &mut GpuTensor<f32>,
    seq_len: u32,
    hidden_size: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, EMBEDDING_F16_SRC, "warp_embedding_f16")?;
    let total = seq_len * hidden_size;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&table.data)
            .arg(&indices.data)
            .arg(&seq_len)
            .arg(&hidden_size)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

// ── Softmax ───────────────────────────────────────────────────────

const SOFTMAX_SRC: &str = r#"
extern "C" __global__ void warp_softmax(
    float *out,           // [rows, cols]
    const float *input,   // [rows, cols]
    unsigned int rows,
    unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    const float *in_row = input + row * cols;
    float *out_row = out + row * cols;

    // Step 1: find max (for numerical stability)
    float max_val = -1e30f;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = in_row[i];
        if (v > max_val) max_val = v;
    }
    // Warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    // Step 2: compute exp(x - max) and sum
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(in_row[i] - max_val);
        out_row[i] = e;
        sum += e;
    }
    // Warp reduce sum
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    sum = __shfl_sync(0xffffffff, sum, 0);

    // Step 3: normalize
    float inv_sum = 1.0f / sum;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] *= inv_sum;
    }
}
"#;

/// Softmax along the last dimension: out[i] = softmax(input[i])
pub fn softmax(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    rows: u32,
    cols: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, SOFTMAX_SRC, "warp_softmax")?;
    let cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (32.min(cols), 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&input.data)
            .arg(&rows)
            .arg(&cols)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

// ── Argmax ────────────────────────────────────────────────────────

const ARGMAX_SRC: &str = r#"
extern "C" __global__ void warp_argmax(
    int *out,             // [rows]
    const float *input,   // [rows, cols]
    unsigned int rows,
    unsigned int cols
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float *in_row = input + row * cols;
    float max_val = in_row[0];
    int max_idx = 0;

    for (unsigned int i = 1; i < cols; i++) {
        float v = in_row[i];
        if (v > max_val) {
            max_val = v;
            max_idx = (int)i;
        }
    }
    out[row] = max_idx;
}
"#;

/// Argmax along the last dimension: out[i] = argmax(input[i, :])
pub fn argmax(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    out: &mut GpuTensor<i32>,
    rows: u32,
    cols: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, ARGMAX_SRC, "warp_argmax")?;
    let cfg = LaunchConfig::for_num_elems(rows);
    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&input.data)
            .arg(&rows)
            .arg(&cols)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

// ── Top-p (nucleus) sampling ──────────────────────────────────────

const TOP_P_SAMPLE_SRC: &str = r#"
extern "C" __global__ void warp_top_p_sample(
    int *out_token,        // [1] output token id
    const float *probs,    // [vocab_size] sorted probabilities
    const int *indices,    // [vocab_size] original indices (sorted by prob desc)
    float top_p,
    float random_val,      // uniform random in [0, 1)
    unsigned int vocab_size
) {
    // Single-thread kernel (sampling is not parallelizable)
    if (threadIdx.x != 0) return;

    float cumsum = 0.0f;
    for (unsigned int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (cumsum >= top_p || cumsum >= random_val * cumsum / top_p) {
            // Scale random_val to [0, cumsum] and find the token
            float threshold = random_val * cumsum;
            float running = 0.0f;
            for (unsigned int j = 0; j <= i; j++) {
                running += probs[j];
                if (running >= threshold) {
                    out_token[0] = indices[j];
                    return;
                }
            }
            out_token[0] = indices[i];
            return;
        }
    }
    out_token[0] = indices[0]; // fallback
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::{DType, Shape};

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    #[test]
    fn gpu_embedding() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let vocab = 100;
        let hidden = 32;
        let seq = 8;

        // Embedding table: each row is [token_id * 0.01, token_id * 0.01, ...]
        let table_data: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i / hidden) as f32 * 0.01)
            .collect();
        let indices_data: Vec<i32> = vec![0, 5, 10, 15, 20, 50, 99, 1];

        let table = GpuTensor::from_host(&dev, &table_data,
            Shape::from_static(&[vocab, hidden]), DType::F32).unwrap();
        let indices = GpuTensor::from_host(&dev, &indices_data,
            Shape::from_static(&[seq]), DType::I32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[seq, hidden]), DType::F32).unwrap();

        embedding(&cache, &dev, &table, &indices, &mut out, seq as u32, hidden as u32).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        // Check: out[0, :] should be table[0, :] = 0.0
        assert!((result[0] - 0.0).abs() < 1e-5);
        // out[1, :] should be table[5, :] = 0.05
        assert!((result[hidden] - 0.05).abs() < 1e-5);
        // out[6, :] should be table[99, :] = 0.99
        assert!((result[6 * hidden] - 0.99).abs() < 1e-4);
        println!("Embedding: {seq} tokens × {hidden} dims correct!");
    }

    #[test]
    fn gpu_softmax() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let rows = 4u32;
        let cols = 16u32;
        let input_data: Vec<f32> = (0..(rows * cols) as usize)
            .map(|i| (i as f32 - 32.0) * 0.1)
            .collect();

        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[rows as usize, cols as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[rows as usize, cols as usize]), DType::F32).unwrap();

        softmax(&cache, &dev, &input, &mut out, rows, cols).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();

        // Each row should sum to ~1.0
        for r in 0..rows as usize {
            let row_sum: f32 = result[r * cols as usize..(r + 1) * cols as usize].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "Row {r} sums to {row_sum}, expected 1.0"
            );
            // All values should be positive
            assert!(result[r * cols as usize..(r + 1) * cols as usize].iter().all(|&v| v >= 0.0));
        }
        println!("Softmax: {rows}x{cols} correct (all rows sum to 1.0)!");
    }

    #[test]
    fn gpu_argmax() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let rows = 4u32;
        let cols = 32u32;
        // Each row has a clear maximum at a known position
        let mut input_data = vec![0.0f32; (rows * cols) as usize];
        input_data[0 * cols as usize + 7] = 10.0;   // row 0 max at col 7
        input_data[1 * cols as usize + 0] = 10.0;   // row 1 max at col 0
        input_data[2 * cols as usize + 31] = 10.0;  // row 2 max at col 31
        input_data[3 * cols as usize + 15] = 10.0;  // row 3 max at col 15

        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[rows as usize, cols as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<i32>::zeros(&dev,
            Shape::from_static(&[rows as usize]), DType::I32).unwrap();

        argmax(&cache, &dev, &input, &mut out, rows, cols).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        assert_eq!(result, vec![7, 0, 31, 15]);
        println!("Argmax: {rows} rows correct! (found max indices: {:?})", result);
    }

    #[test]
    fn softmax_numerical_stability() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // Large values that would overflow without max-subtraction
        let cols = 32u32;
        let input_data: Vec<f32> = (0..cols as usize)
            .map(|i| 100.0 + i as f32 * 0.1)
            .collect();

        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[1, cols as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, cols as usize]), DType::F32).unwrap();

        softmax(&cache, &dev, &input, &mut out, 1, cols).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "Sum = {sum}, expected 1.0");
        assert!(result.iter().all(|v| v.is_finite()), "Has NaN/Inf!");
        println!("Softmax numerical stability: passed (input range 100-103)");
    }
}

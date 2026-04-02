//! Quantization kernels — block-scaled INT8 and INT4 for weight compression.
//!
//! Supported formats (GGUF-compatible block structure):
//!
//! **Q8_0**: 8-bit per-block quantization
//!   Block = [f32 scale (4B)] + [i8 × 32 (32B)] = 36 bytes per 32 elements
//!   Compression: 3.56× vs f32 (36 vs 128 bytes per 32 elements)
//!
//! **Q4_0**: 4-bit per-block quantization
//!   Block = [f32 scale (4B)] + [u8 × 16 (16B)] = 20 bytes per 32 elements
//!   Each u8 packs two 4-bit values: low nibble = even index, high nibble = odd
//!   Compression: 6.4× vs f32 (20 vs 128 bytes per 32 elements)
//!
//! Key kernel: **W4A16 Quantized GEMM** — dequantizes weights on-the-fly during
//! tiled matrix multiply. Reads 4-bit weights from global memory (4× less bandwidth),
//! dequantizes in shared memory, then does standard tile multiply in f32.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Block size for quantization (elements per block).
pub const BLOCK_SIZE: u32 = 32;
/// Q8_0 block size in bytes: 4 (f32 scale) + 32 (i8 values).
pub const Q8_0_BLOCK_BYTES: u32 = 4 + BLOCK_SIZE;
/// Q4_0 block size in bytes: 4 (f32 scale) + 16 (packed u8 pairs).
pub const Q4_0_BLOCK_BYTES: u32 = 4 + BLOCK_SIZE / 2;

macro_rules! launch_err {
    ($e:expr) => {
        $e.map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))
    };
}

// ── Q8_0 Quantize ───────────────────────────────────────────────

const QUANTIZE_Q8_0_SRC: &str = r#"
extern "C" __global__ void warp_quantize_q8_0(
    unsigned char *out,     // [num_blocks * 36] packed output
    const float *input,     // [n] float input
    unsigned int n,         // total elements (must be multiple of 32)
    unsigned int num_blocks // n / 32
) {
    unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const float *src = input + block_idx * 32;
    unsigned char *dst = out + block_idx * 36;  // 4 + 32

    // Find absmax in block
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }

    float scale = amax / 127.0f;
    float inv_scale = (scale != 0.0f) ? 127.0f / amax : 0.0f;

    // Write scale as f32 (first 4 bytes of block)
    *((float *)dst) = scale;

    // Quantize values
    signed char *qvals = (signed char *)(dst + 4);
    for (int i = 0; i < 32; i++) {
        float v = src[i] * inv_scale;
        // Round to nearest, clamp to [-128, 127]
        int q = (int)roundf(v);
        q = q < -128 ? -128 : (q > 127 ? 127 : q);
        qvals[i] = (signed char)q;
    }
}
"#;

// ── Q8_0 Dequantize ─────────────────────────────────────────────

const DEQUANTIZE_Q8_0_SRC: &str = r#"
extern "C" __global__ void warp_dequantize_q8_0(
    float *out,                  // [n] float output
    const unsigned char *input,  // [num_blocks * 36] packed blocks
    unsigned int n,
    unsigned int num_blocks
) {
    unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const unsigned char *src = input + block_idx * 36;
    float *dst = out + block_idx * 32;

    float scale = *((const float *)src);
    const signed char *qvals = (const signed char *)(src + 4);

    for (int i = 0; i < 32; i++) {
        dst[i] = scale * (float)qvals[i];
    }
}
"#;

// ── Q4_0 Quantize ───────────────────────────────────────────────

const QUANTIZE_Q4_0_SRC: &str = r#"
extern "C" __global__ void warp_quantize_q4_0(
    unsigned char *out,     // [num_blocks * 20] packed output
    const float *input,     // [n] float input
    unsigned int n,
    unsigned int num_blocks
) {
    unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const float *src = input + block_idx * 32;
    unsigned char *dst = out + block_idx * 20;  // 4 + 16

    // Find absmax
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }

    float scale = amax / 7.0f;
    float inv_scale = (scale != 0.0f) ? 7.0f / amax : 0.0f;

    // Write scale
    *((float *)dst) = scale;

    // Quantize and pack pairs into bytes
    // Each value maps to [0, 15]: q = round(v / scale) + 8
    // Low nibble = even index, high nibble = odd index
    unsigned char *packed = dst + 4;
    for (int i = 0; i < 16; i++) {
        float v0 = src[2 * i];
        float v1 = src[2 * i + 1];

        int q0 = (int)roundf(v0 * inv_scale) + 8;
        int q1 = (int)roundf(v1 * inv_scale) + 8;

        q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
        q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);

        packed[i] = (unsigned char)(q0 | (q1 << 4));
    }
}
"#;

// ── Q4_0 Dequantize ─────────────────────────────────────────────

const DEQUANTIZE_Q4_0_SRC: &str = r#"
extern "C" __global__ void warp_dequantize_q4_0(
    float *out,
    const unsigned char *input,
    unsigned int n,
    unsigned int num_blocks
) {
    unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const unsigned char *src = input + block_idx * 20;
    float *dst = out + block_idx * 32;

    float scale = *((const float *)src);
    const unsigned char *packed = src + 4;

    for (int i = 0; i < 16; i++) {
        unsigned char byte = packed[i];
        int q0 = (byte & 0x0F) - 8;       // low nibble
        int q1 = ((byte >> 4) & 0x0F) - 8; // high nibble

        dst[2 * i]     = scale * (float)q0;
        dst[2 * i + 1] = scale * (float)q1;
    }
}
"#;

// ── W4A16 Quantized GEMM ────────────────────────────────────────
// C[M,N] = A[M,K] × dequant(B_q4[K,N])
//
// Weights B are quantized along the K dimension in blocks of 32.
// Layout: for column j, block b → offset (j * num_k_blocks + b) * 20
//
// TILE = 32 = BLOCK_SIZE so each K-tile load dequantizes exactly one
// block per column, keeping the logic clean.

const GEMM_Q4_0_SRC: &str = r#"
#define TILE 32

extern "C" __global__ void warp_gemm_q4_0(
    float *C,                       // [M, N] output
    const float *A,                 // [M, K] activations (f32)
    const unsigned char *B_quant,   // quantized weights
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_k_blocks       // K / 32
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int row = by * TILE + ty;
    unsigned int col = bx * TILE + tx;

    float sum = 0.0f;

    for (unsigned int t = 0; t < num_k_blocks; t++) {
        // Load A tile (standard)
        unsigned int a_col = t * TILE + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // Load B tile by dequantizing Q4_0 blocks
        // Each column of B has its blocks at: (col * num_k_blocks + t) * 20
        unsigned int b_col = bx * TILE + tx;
        if (b_col < N) {
            const unsigned char *block_ptr = B_quant + (b_col * num_k_blocks + t) * 20;
            float scale = *((const float *)block_ptr);
            const unsigned char *packed = block_ptr + 4;

            // Thread ty reads element ty from this block
            unsigned int byte_idx = ty / 2;
            unsigned char byte = packed[byte_idx];
            int q;
            if (ty % 2 == 0) {
                q = (byte & 0x0F) - 8;
            } else {
                q = ((byte >> 4) & 0x0F) - 8;
            }
            Bs[ty][tx] = scale * (float)q;
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (unsigned int i = 0; i < TILE; i++)
            sum += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
"#;

// ── W8A16 Quantized GEMM ────────────────────────────────────────
// Same structure but for Q8_0 weights.

const GEMM_Q8_0_SRC: &str = r#"
#define TILE 32

extern "C" __global__ void warp_gemm_q8_0(
    float *C,
    const float *A,
    const unsigned char *B_quant,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_k_blocks
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int row = by * TILE + ty;
    unsigned int col = bx * TILE + tx;

    float sum = 0.0f;

    for (unsigned int t = 0; t < num_k_blocks; t++) {
        unsigned int a_col = t * TILE + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        unsigned int b_col = bx * TILE + tx;
        if (b_col < N) {
            // Q8_0 block: [f32 scale (4B)] + [i8 × 32 (32B)] = 36B
            const unsigned char *block_ptr = B_quant + (b_col * num_k_blocks + t) * 36;
            float scale = *((const float *)block_ptr);
            const signed char *qvals = (const signed char *)(block_ptr + 4);

            Bs[ty][tx] = scale * (float)qvals[ty];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (unsigned int i = 0; i < TILE; i++)
            sum += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
"#;

// ═════════════════════════════════════════════════════════════════
// Rust API
// ═════════════════════════════════════════════════════════════════

/// Quantize f32 tensor to Q8_0 format.
/// Input: [n] f32 (n must be divisible by BLOCK_SIZE)
/// Output: [num_blocks * Q8_0_BLOCK_BYTES] u8
pub fn quantize_q8_0(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    output: &mut GpuTensor<u8>,
) -> Result<(), DeviceError> {
    let n = input.numel as u32;
    assert!(n % BLOCK_SIZE == 0, "n ({n}) must be divisible by BLOCK_SIZE ({BLOCK_SIZE})");
    let num_blocks = n / BLOCK_SIZE;

    let f = cache.get_or_compile(device, QUANTIZE_Q8_0_SRC, "warp_quantize_q8_0")?;
    let cfg = LaunchConfig::for_num_elems(num_blocks);

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data)
            .arg(&input.data)
            .arg(&n)
            .arg(&num_blocks)
            .launch(cfg))?;
    }
    Ok(())
}

/// Dequantize Q8_0 back to f32.
pub fn dequantize_q8_0(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<u8>,
    output: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let n = output.numel as u32;
    let num_blocks = n / BLOCK_SIZE;

    let f = cache.get_or_compile(device, DEQUANTIZE_Q8_0_SRC, "warp_dequantize_q8_0")?;
    let cfg = LaunchConfig::for_num_elems(num_blocks);

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data)
            .arg(&input.data)
            .arg(&n)
            .arg(&num_blocks)
            .launch(cfg))?;
    }
    Ok(())
}

/// Quantize f32 tensor to Q4_0 format.
/// Input: [n] f32 (n must be divisible by BLOCK_SIZE)
/// Output: [num_blocks * Q4_0_BLOCK_BYTES] u8
pub fn quantize_q4_0(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    output: &mut GpuTensor<u8>,
) -> Result<(), DeviceError> {
    let n = input.numel as u32;
    assert!(n % BLOCK_SIZE == 0, "n ({n}) must be divisible by BLOCK_SIZE ({BLOCK_SIZE})");
    let num_blocks = n / BLOCK_SIZE;

    let f = cache.get_or_compile(device, QUANTIZE_Q4_0_SRC, "warp_quantize_q4_0")?;
    let cfg = LaunchConfig::for_num_elems(num_blocks);

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data)
            .arg(&input.data)
            .arg(&n)
            .arg(&num_blocks)
            .launch(cfg))?;
    }
    Ok(())
}

/// Dequantize Q4_0 back to f32.
pub fn dequantize_q4_0(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<u8>,
    output: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let n = output.numel as u32;
    let num_blocks = n / BLOCK_SIZE;

    let f = cache.get_or_compile(device, DEQUANTIZE_Q4_0_SRC, "warp_dequantize_q4_0")?;
    let cfg = LaunchConfig::for_num_elems(num_blocks);

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data)
            .arg(&input.data)
            .arg(&n)
            .arg(&num_blocks)
            .launch(cfg))?;
    }
    Ok(())
}

/// Quantize a weight matrix [K, N] to Q4_0 column-block format.
///
/// Quantizes along K in blocks of 32. Output layout:
/// for column j, block b → offset (j * num_k_blocks + b) * Q4_0_BLOCK_BYTES
///
/// This is the format expected by `gemm_q4_0`.
pub fn quantize_weights_q4_0(
    _cache: &KernelCache,
    device: &WarpDevice,
    weights: &GpuTensor<f32>,  // [K, N] row-major
    k: u32,
    n: u32,
) -> Result<GpuTensor<u8>, DeviceError> {
    assert!(k % BLOCK_SIZE == 0, "K ({k}) must be divisible by BLOCK_SIZE ({BLOCK_SIZE})");
    let num_k_blocks = k / BLOCK_SIZE;
    let total_blocks = num_k_blocks * n;
    let out_bytes = (total_blocks * Q4_0_BLOCK_BYTES) as usize;

    // We need to transpose the quantization: input is row-major [K, N]
    // but we need column-wise blocks. Do it on CPU for now (weight quantization
    // is a one-time offline cost).
    let w_host = weights.to_host(device)?;
    let mut q_host = vec![0u8; out_bytes];

    for j in 0..n as usize {
        for b in 0..num_k_blocks as usize {
            let block_offset = (j * num_k_blocks as usize + b) * Q4_0_BLOCK_BYTES as usize;
            let k_start = b * BLOCK_SIZE as usize;

            // Gather column j, rows k_start..k_start+32
            let mut vals = [0.0f32; 32];
            for i in 0..32 {
                vals[i] = w_host[(k_start + i) * n as usize + j];
            }

            // Find absmax
            let amax = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = amax / 7.0;
            let inv_scale = if scale != 0.0 { 7.0 / amax } else { 0.0 };

            // Write scale
            let scale_bytes = scale.to_le_bytes();
            q_host[block_offset..block_offset + 4].copy_from_slice(&scale_bytes);

            // Pack 4-bit values
            for i in 0..16 {
                let v0 = vals[2 * i];
                let v1 = vals[2 * i + 1];

                let mut q0 = (v0 * inv_scale).round() as i32 + 8;
                let mut q1 = (v1 * inv_scale).round() as i32 + 8;
                q0 = q0.clamp(0, 15);
                q1 = q1.clamp(0, 15);

                q_host[block_offset + 4 + i] = (q0 as u8) | ((q1 as u8) << 4);
            }
        }
    }

    GpuTensor::from_host(
        device,
        &q_host,
        warp_ir::Shape::from_static(&[out_bytes]),
        warp_ir::DType::U8,
    )
}

/// Quantize a weight matrix [K, N] to Q8_0 column-block format.
pub fn quantize_weights_q8_0(
    _cache: &KernelCache,
    device: &WarpDevice,
    weights: &GpuTensor<f32>,
    k: u32,
    n: u32,
) -> Result<GpuTensor<u8>, DeviceError> {
    assert!(k % BLOCK_SIZE == 0, "K ({k}) must be divisible by BLOCK_SIZE ({BLOCK_SIZE})");
    let num_k_blocks = k / BLOCK_SIZE;
    let total_blocks = num_k_blocks * n;
    let out_bytes = (total_blocks * Q8_0_BLOCK_BYTES) as usize;

    let w_host = weights.to_host(device)?;
    let mut q_host = vec![0u8; out_bytes];

    for j in 0..n as usize {
        for b in 0..num_k_blocks as usize {
            let block_offset = (j * num_k_blocks as usize + b) * Q8_0_BLOCK_BYTES as usize;
            let k_start = b * BLOCK_SIZE as usize;

            let mut vals = [0.0f32; 32];
            for i in 0..32 {
                vals[i] = w_host[(k_start + i) * n as usize + j];
            }

            let amax = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = amax / 127.0;
            let inv_scale = if scale != 0.0 { 127.0 / amax } else { 0.0 };

            let scale_bytes = scale.to_le_bytes();
            q_host[block_offset..block_offset + 4].copy_from_slice(&scale_bytes);

            for i in 0..32 {
                let q = (vals[i] * inv_scale).round() as i32;
                let q = q.clamp(-128, 127);
                q_host[block_offset + 4 + i] = q as u8; // i8 as u8
            }
        }
    }

    GpuTensor::from_host(
        device,
        &q_host,
        warp_ir::Shape::from_static(&[out_bytes]),
        warp_ir::DType::U8,
    )
}

/// W4A16 Quantized GEMM: C = A × dequant(B_q4)
///
/// A: [M, K] f32 activations
/// B_quant: Q4_0 column-block quantized weights (from `quantize_weights_q4_0`)
/// C: [M, N] f32 output
///
/// K must be divisible by 32 (BLOCK_SIZE).
pub fn gemm_q4_0(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,       // [M, K]
    b_quant: &GpuTensor<u8>,  // quantized [K, N]
    c: &mut GpuTensor<f32>,   // [M, N]
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    assert!(k % BLOCK_SIZE == 0, "K must be divisible by {BLOCK_SIZE}");
    let num_k_blocks = k / BLOCK_SIZE;
    let tile = 32u32;

    let f = cache.get_or_compile(device, GEMM_Q4_0_SRC, "warp_gemm_q4_0")?;
    let cfg = LaunchConfig {
        grid_dim: ((n + tile - 1) / tile, (m + tile - 1) / tile, 1),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut c.data)
            .arg(&a.data)
            .arg(&b_quant.data)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .arg(&num_k_blocks)
            .launch(cfg))?;
    }
    Ok(())
}

/// W8A16 Quantized GEMM: C = A × dequant(B_q8)
pub fn gemm_q8_0(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b_quant: &GpuTensor<u8>,
    c: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    assert!(k % BLOCK_SIZE == 0, "K must be divisible by {BLOCK_SIZE}");
    let num_k_blocks = k / BLOCK_SIZE;
    let tile = 32u32;

    let f = cache.get_or_compile(device, GEMM_Q8_0_SRC, "warp_gemm_q8_0")?;
    let cfg = LaunchConfig {
        grid_dim: ((n + tile - 1) / tile, (m + tile - 1) / tile, 1),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut c.data)
            .arg(&a.data)
            .arg(&b_quant.data)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .arg(&num_k_blocks)
            .launch(cfg))?;
    }
    Ok(())
}

// ═════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::{DType, Shape};

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    /// CPU reference: quantize to Q8_0 and dequantize back
    fn cpu_q8_0_roundtrip(input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; input.len()];
        for block in 0..(input.len() / 32) {
            let src = &input[block * 32..(block + 1) * 32];
            let dst = &mut output[block * 32..(block + 1) * 32];

            let amax = src.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = amax / 127.0;
            let inv_scale = if scale != 0.0 { 127.0 / amax } else { 0.0 };

            for i in 0..32 {
                let q = (src[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
                dst[i] = scale * q as f32;
            }
        }
        output
    }

    /// CPU reference: quantize to Q4_0 and dequantize back
    fn cpu_q4_0_roundtrip(input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; input.len()];
        for block in 0..(input.len() / 32) {
            let src = &input[block * 32..(block + 1) * 32];
            let dst = &mut output[block * 32..(block + 1) * 32];

            let amax = src.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = amax / 7.0;
            let inv_scale = if scale != 0.0 { 7.0 / amax } else { 0.0 };

            for i in 0..32 {
                let q = ((src[i] * inv_scale).round() as i32 + 8).clamp(0, 15) - 8;
                dst[i] = scale * q as f32;
            }
        }
        output
    }

    #[test]
    fn q8_0_roundtrip_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let n = 1024usize;
        let input_data: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 200) as f32 * 0.01 - 1.0).collect();

        let input = GpuTensor::from_host(&dev, &input_data, Shape::from_static(&[n]), DType::F32).unwrap();
        let num_blocks = n / BLOCK_SIZE as usize;
        let q_bytes = num_blocks * Q8_0_BLOCK_BYTES as usize;

        let mut quantized = GpuTensor::<u8>::zeros(&dev,
            Shape::from_static(&[q_bytes]), DType::U8).unwrap();
        let mut dequantized = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[n]), DType::F32).unwrap();

        quantize_q8_0(&cache, &dev, &input, &mut quantized).unwrap();
        dequantize_q8_0(&cache, &dev, &quantized, &mut dequantized).unwrap();
        dev.synchronize().unwrap();

        let result = dequantized.to_host(&dev).unwrap();
        let expected = cpu_q8_0_roundtrip(&input_data);

        let max_err: f32 = result.iter().zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Q8_0 should be very close to CPU reference
        assert!(max_err < 1e-5, "Q8_0 GPU vs CPU reference: max error {max_err}");

        // Check quantization error vs original
        let quant_err: f32 = result.iter().zip(input_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("Q8_0 roundtrip ({n} elements):");
        println!("  GPU vs CPU ref: max error = {max_err:.2e}");
        println!("  Quantization error: max = {quant_err:.4} (expected ≤ scale/127)");
        println!("  Compression: {:.1}x ({} → {} bytes)",
            (n * 4) as f32 / q_bytes as f32, n * 4, q_bytes);
    }

    #[test]
    fn q4_0_roundtrip_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let n = 1024usize;
        let input_data: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 200) as f32 * 0.01 - 1.0).collect();

        let input = GpuTensor::from_host(&dev, &input_data, Shape::from_static(&[n]), DType::F32).unwrap();
        let num_blocks = n / BLOCK_SIZE as usize;
        let q_bytes = num_blocks * Q4_0_BLOCK_BYTES as usize;

        let mut quantized = GpuTensor::<u8>::zeros(&dev,
            Shape::from_static(&[q_bytes]), DType::U8).unwrap();
        let mut dequantized = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[n]), DType::F32).unwrap();

        quantize_q4_0(&cache, &dev, &input, &mut quantized).unwrap();
        dequantize_q4_0(&cache, &dev, &quantized, &mut dequantized).unwrap();
        dev.synchronize().unwrap();

        let result = dequantized.to_host(&dev).unwrap();
        let expected = cpu_q4_0_roundtrip(&input_data);

        let max_err: f32 = result.iter().zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_err < 1e-5, "Q4_0 GPU vs CPU reference: max error {max_err}");

        let quant_err: f32 = result.iter().zip(input_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("Q4_0 roundtrip ({n} elements):");
        println!("  GPU vs CPU ref: max error = {max_err:.2e}");
        println!("  Quantization error: max = {quant_err:.4} (expected ≤ scale/7)");
        println!("  Compression: {:.1}x ({} → {} bytes)",
            (n * 4) as f32 / q_bytes as f32, n * 4, q_bytes);
    }

    #[test]
    fn gemm_q4_0_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (64u32, 64u32, 128u32);

        // Random-ish test data
        let a_data: Vec<f32> = (0..(m * k) as usize)
            .map(|i| ((i * 7 + 13) % 100) as f32 * 0.01 - 0.5)
            .collect();
        let b_data: Vec<f32> = (0..(k * n) as usize)
            .map(|i| ((i * 11 + 3) % 100) as f32 * 0.01 - 0.5)
            .collect();

        let a = GpuTensor::from_host(&dev, &a_data,
            Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data,
            Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();

        // Quantize weights
        let b_quant = quantize_weights_q4_0(&cache, &dev, &b, k, n).unwrap();

        // Quantized GEMM
        let mut c_quant = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        gemm_q4_0(&cache, &dev, &a, &b_quant, &mut c_quant, m, n, k).unwrap();
        dev.synchronize().unwrap();

        // Full-precision GEMM for reference
        let mut c_ref = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        crate::ops::gemm(&cache, &dev, &a, &b, &mut c_ref, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let quant_result = c_quant.to_host(&dev).unwrap();
        let ref_result = c_ref.to_host(&dev).unwrap();

        // Compute relative error
        let mut max_abs_err = 0.0f32;
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        for (q, r) in quant_result.iter().zip(ref_result.iter()) {
            let err = (q - r).abs();
            if err > max_abs_err { max_abs_err = err; }
            sum_sq_err += (err as f64).powi(2);
            sum_sq_ref += (*r as f64).powi(2);
        }
        let rmse = (sum_sq_err / quant_result.len() as f64).sqrt();
        let nrmse = (sum_sq_err / sum_sq_ref.max(1e-10)).sqrt();

        println!("W4A16 GEMM Q4_0 ({m}×{n}×{k}):");
        println!("  Max abs error: {max_abs_err:.4}");
        println!("  RMSE:          {rmse:.6}");
        println!("  NRMSE:         {nrmse:.6} ({:.2}%)", nrmse * 100.0);
        println!("  Weight memory: {} → {} bytes ({:.1}x compression)",
            (k * n) * 4, b_quant.numel, ((k * n) * 4) as f32 / b_quant.numel as f32);

        // Q4_0 with group_size=32 should have NRMSE < 10% for these value ranges
        assert!(nrmse < 0.15, "NRMSE {nrmse:.4} too high for Q4_0");
    }

    #[test]
    fn gemm_q8_0_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (64u32, 64u32, 128u32);

        let a_data: Vec<f32> = (0..(m * k) as usize)
            .map(|i| ((i * 7 + 13) % 100) as f32 * 0.01 - 0.5)
            .collect();
        let b_data: Vec<f32> = (0..(k * n) as usize)
            .map(|i| ((i * 11 + 3) % 100) as f32 * 0.01 - 0.5)
            .collect();

        let a = GpuTensor::from_host(&dev, &a_data,
            Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data,
            Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();

        let b_quant = quantize_weights_q8_0(&cache, &dev, &b, k, n).unwrap();

        let mut c_quant = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        gemm_q8_0(&cache, &dev, &a, &b_quant, &mut c_quant, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let mut c_ref = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        crate::ops::gemm(&cache, &dev, &a, &b, &mut c_ref, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let quant_result = c_quant.to_host(&dev).unwrap();
        let ref_result = c_ref.to_host(&dev).unwrap();

        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        let mut max_abs_err = 0.0f32;
        for (q, r) in quant_result.iter().zip(ref_result.iter()) {
            let err = (q - r).abs();
            if err > max_abs_err { max_abs_err = err; }
            sum_sq_err += (err as f64).powi(2);
            sum_sq_ref += (*r as f64).powi(2);
        }
        let nrmse = (sum_sq_err / sum_sq_ref.max(1e-10)).sqrt();

        println!("W8A16 GEMM Q8_0 ({m}×{n}×{k}):");
        println!("  Max abs error: {max_abs_err:.6}");
        println!("  NRMSE:         {nrmse:.6} ({:.2}%)", nrmse * 100.0);
        println!("  Weight memory: {} → {} bytes ({:.1}x compression)",
            (k * n) * 4, b_quant.numel, ((k * n) * 4) as f32 / b_quant.numel as f32);

        // Q8_0 should be much more accurate than Q4_0
        assert!(nrmse < 0.02, "NRMSE {nrmse:.4} too high for Q8_0");
    }

    #[test]
    fn gemm_q4_0_vs_f32_perf() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (256u32, 256u32, 256u32);

        let a_data: Vec<f32> = (0..(m * k) as usize)
            .map(|i| ((i * 7 + 13) % 100) as f32 * 0.01 - 0.5)
            .collect();
        let b_data: Vec<f32> = (0..(k * n) as usize)
            .map(|i| ((i * 11 + 3) % 100) as f32 * 0.01 - 0.5)
            .collect();

        let a = GpuTensor::from_host(&dev, &a_data,
            Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data,
            Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let b_q4 = quantize_weights_q4_0(&cache, &dev, &b, k, n).unwrap();

        let mut c = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

        // Warmup
        crate::ops::gemm(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
        gemm_q4_0(&cache, &dev, &a, &b_q4, &mut c, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let iters = 200;

        // F32 GEMM
        let start = std::time::Instant::now();
        for _ in 0..iters {
            crate::ops::gemm(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
        }
        dev.synchronize().unwrap();
        let f32_time = start.elapsed();

        // Q4_0 GEMM
        let start = std::time::Instant::now();
        for _ in 0..iters {
            gemm_q4_0(&cache, &dev, &a, &b_q4, &mut c, m, n, k).unwrap();
        }
        dev.synchronize().unwrap();
        let q4_time = start.elapsed();

        let f32_ms = f32_time.as_secs_f64() * 1000.0 / iters as f64;
        let q4_ms = q4_time.as_secs_f64() * 1000.0 / iters as f64;
        let weight_bytes_f32 = (k * n * 4) as f64;
        let weight_bytes_q4 = b_q4.numel as f64;

        println!("\nW4A16 vs F32 GEMM ({m}×{n}×{k}, {iters} iters):");
        println!("  F32:  {f32_ms:.3}ms avg | weights = {:.0} KB", weight_bytes_f32 / 1024.0);
        println!("  Q4_0: {q4_ms:.3}ms avg | weights = {:.0} KB ({:.1}x smaller)",
            weight_bytes_q4 / 1024.0, weight_bytes_f32 / weight_bytes_q4);
        println!("  Speed ratio: {:.2}x", f32_ms / q4_ms);
    }
}

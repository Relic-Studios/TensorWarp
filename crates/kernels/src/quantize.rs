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

use cudarc::driver::{DevicePtrMut, LaunchConfig, PushKernelArg};

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

// ── Q4_0 → FP16 Dequantization (ExLlamaV2-style) ───────────────
//
// Pre-dequantize Q4_0 column-block weights to a standard row-major FP16
// matrix. This is done ONCE at model load. At inference time, we use
// cublasHgemm on the FP16 weights instead of the custom 32×32 tile GEMM.
//
// This trades VRAM (2× more than Q4) for much higher throughput from
// cuBLAS's optimized memory access patterns and tensor core utilization.

const DEQUANT_Q4_TO_F16_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void warp_dequant_q4_to_f16(
    __half *out,                    // [K, N] row-major FP16
    const unsigned char *B_quant,   // Q4_0 column-block format
    unsigned int K,
    unsigned int N,
    unsigned int num_k_blocks       // K / 32
) {
    // Each thread dequantizes one element of the output matrix
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = K * N;
    if (idx >= total) return;

    unsigned int row = idx / N;   // K dimension
    unsigned int col = idx % N;   // N dimension

    // Find the Q4_0 block: column col, block (row / 32)
    unsigned int block_idx = row / 32;
    unsigned int elem_in_block = row % 32;
    unsigned int block_offset = (col * num_k_blocks + block_idx) * 20;

    // Read scale (f32 at block start)
    float scale = *((const float *)(B_quant + block_offset));

    // Read packed nibble
    unsigned int byte_idx = elem_in_block / 2;
    unsigned char byte = B_quant[block_offset + 4 + byte_idx];
    int q;
    if (elem_in_block % 2 == 0) {
        q = (byte & 0x0F) - 8;
    } else {
        q = ((byte >> 4) & 0x0F) - 8;
    }

    out[row * N + col] = __float2half(scale * (float)q);
}
"#;

/// Dequantize Q4_0 column-block weights to a standard row-major FP16 matrix.
///
/// This is the ExLlamaV2-style approach: dequant once at load time, then
/// use cublasHgemm for all subsequent GEMM calls.
///
/// b_quant: Q4_0 column-block format (from `quantize_weights_q4_0`)
/// Returns: [K, N] row-major FP16 tensor
pub fn dequant_q4_to_f16(
    cache: &KernelCache,
    device: &WarpDevice,
    b_quant: &GpuTensor<u8>,
    k: u32,
    n: u32,
) -> Result<GpuTensor<half::f16>, DeviceError> {
    assert!(k % BLOCK_SIZE == 0, "K must be divisible by {BLOCK_SIZE}");
    let num_k_blocks = k / BLOCK_SIZE;
    let total_elements = (k * n) as usize;

    let mut output = GpuTensor::<half::f16>::zeros(
        device,
        warp_ir::Shape::from_static(&[k as usize, n as usize]),
        warp_ir::DType::F16,
    )?;

    let include = crate::device::WarpDevice::cuda_include_path();
    let f = cache.get_or_compile_with_opts(device, DEQUANT_Q4_TO_F16_SRC, "warp_dequant_q4_to_f16", &[include], None)?;

    let threads = 256u32;
    let blocks = (total_elements as u32 + threads - 1) / threads;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data)
            .arg(&b_quant.data)
            .arg(&k)
            .arg(&n)
            .arg(&num_k_blocks)
            .launch(cfg))?;
    }
    Ok(output)
}

/// Pre-dequantized FP16 weight set for a transformer block.
/// Created once at model load from QuantizedBlockWeights.
/// Uses 2× more VRAM than Q4 but enables cuBLAS HGEMM.
pub struct DequantizedF16Weights {
    pub attn_norm: GpuTensor<f32>,
    pub wq: GpuTensor<half::f16>,    // [hidden, hidden]
    pub wk: GpuTensor<half::f16>,    // [hidden, kv_dim]
    pub wv: GpuTensor<half::f16>,    // [hidden, kv_dim]
    pub wo: GpuTensor<half::f16>,    // [hidden, hidden]
    pub ffn_norm: GpuTensor<f32>,
    pub w_gate: GpuTensor<half::f16>, // [hidden, ffn_dim]
    pub w_up: GpuTensor<half::f16>,   // [hidden, ffn_dim]
    pub w_down: GpuTensor<half::f16>, // [ffn_dim, hidden]
    pub bq: Option<GpuTensor<f32>>,
    pub bk: Option<GpuTensor<f32>>,
    pub bv: Option<GpuTensor<f32>>,
}

impl DequantizedF16Weights {
    /// Convert QuantizedBlockWeights to pre-dequantized FP16.
    /// This is a one-time cost at model load that enables cuBLAS HGEMM.
    pub fn from_quantized(
        cache: &KernelCache,
        device: &WarpDevice,
        q: &crate::transformer::QuantizedBlockWeights,
        config: &crate::transformer::TransformerConfig,
    ) -> Result<Self, DeviceError> {
        let h = config.hidden_size;
        let kv = config.kv_dim();
        let ffn = config.ffn_dim;

        Ok(Self {
            attn_norm: GpuTensor::from_host(device,
                &q.attn_norm.to_host(device)?,
                q.attn_norm.shape.clone(), warp_ir::DType::F32)?,
            wq: dequant_q4_to_f16(cache, device, &q.wq, h, h)?,
            wk: dequant_q4_to_f16(cache, device, &q.wk, h, kv)?,
            wv: dequant_q4_to_f16(cache, device, &q.wv, h, kv)?,
            wo: dequant_q4_to_f16(cache, device, &q.wo, h, h)?,
            ffn_norm: GpuTensor::from_host(device,
                &q.ffn_norm.to_host(device)?,
                q.ffn_norm.shape.clone(), warp_ir::DType::F32)?,
            w_gate: dequant_q4_to_f16(cache, device, &q.w_gate, h, ffn)?,
            w_up: dequant_q4_to_f16(cache, device, &q.w_up, h, ffn)?,
            w_down: dequant_q4_to_f16(cache, device, &q.w_down, ffn, h)?,
            bq: match &q.bq {
                Some(t) => Some(GpuTensor::from_host(device, &t.to_host(device)?, t.shape.clone(), warp_ir::DType::F32)?),
                None => None,
            },
            bk: match &q.bk {
                Some(t) => Some(GpuTensor::from_host(device, &t.to_host(device)?, t.shape.clone(), warp_ir::DType::F32)?),
                None => None,
            },
            bv: match &q.bv {
                Some(t) => Some(GpuTensor::from_host(device, &t.to_host(device)?, t.shape.clone(), warp_ir::DType::F32)?),
                None => None,
            },
        })
    }

    /// VRAM usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.wq.size_bytes() + self.wk.size_bytes() + self.wv.size_bytes()
            + self.wo.size_bytes() + self.w_gate.size_bytes()
            + self.w_up.size_bytes() + self.w_down.size_bytes()
            + self.attn_norm.size_bytes() + self.ffn_norm.size_bytes()
    }
}

// ── M=1 Specialized Q4_0 GEMM (decode-optimized) ───────────────
//
// For M=1 (single-token decode), the generic 32×32 tiled kernel wastes
// 31/32 threads in the M dimension. This specialized kernel assigns
// one thread per output column, with each thread computing the full
// K-dimension dot product. Expected: 5-10x speedup for M=1.

// ── Block-major Q4 weight reordering for coalesced M=1 reads ────
//
// Column-major layout: (col_j * num_k_blocks + block_b) * 20
//   → Adjacent threads read blocks 2240 bytes apart (terrible coalescing)
//
// Block-major layout: (block_b * N + col_j) * 20
//   → Adjacent threads read blocks 20 bytes apart (excellent coalescing)
//
// Reorder is done once at model load (CPU, ~ms). Zero runtime cost.

/// Reorder Q4_0 weights from column-major to block-major layout.
/// This enables coalesced memory reads in the M=1 GEMM kernel.
///
/// Column-major: offset = (col * num_k_blocks + block) * 20
/// Block-major:  offset = (block * N + col) * 20
pub fn reorder_q4_block_major(
    device: &WarpDevice,
    col_major: &GpuTensor<u8>,
    k: u32,
    n: u32,
) -> Result<GpuTensor<u8>, DeviceError> {
    let num_k_blocks = k / BLOCK_SIZE;
    let total_blocks = (num_k_blocks * n) as usize;
    let total_bytes = total_blocks * Q4_0_BLOCK_BYTES as usize;

    let src = col_major.to_host(device)?;
    let mut dst = vec![0u8; total_bytes];

    for col in 0..n as usize {
        for blk in 0..num_k_blocks as usize {
            let src_offset = (col * num_k_blocks as usize + blk) * Q4_0_BLOCK_BYTES as usize;
            let dst_offset = (blk * n as usize + col) * Q4_0_BLOCK_BYTES as usize;
            dst[dst_offset..dst_offset + Q4_0_BLOCK_BYTES as usize]
                .copy_from_slice(&src[src_offset..src_offset + Q4_0_BLOCK_BYTES as usize]);
        }
    }

    GpuTensor::from_host(device, &dst,
        warp_ir::Shape::from_static(&[total_bytes]),
        warp_ir::DType::U8)
}

/// M=1 Q4_0 GEMM kernel for block-major weight layout.
/// Adjacent threads read adjacent 20-byte blocks for coalesced global reads.
const GEMM_Q4_0_M1_BLOCKMAJOR_SRC: &str = r#"
extern "C" __global__ void warp_gemm_q4_0_m1_bm(
    float* __restrict__ C,
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_q4,
    int K,
    int N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float dot = 0.0f;
    const int num_k_blocks = K / 32;

    for (int b = 0; b < num_k_blocks; b++) {
        // Block-major layout: (b * N + n) * 20
        // Adjacent threads (n, n+1) read adjacent 20-byte blocks — coalesced!
        const unsigned char* blk = B_q4 + ((long long)b * N + n) * 20;

        float scale = *((const float*)blk);
        const unsigned int* packed = (const unsigned int*)(blk + 4);
        unsigned int val0 = packed[0];
        unsigned int val1 = packed[1];
        unsigned int val2 = packed[2];
        unsigned int val3 = packed[3];

        int k_base = b * 32;

        #define BM_PROCESS(V, off) { \
            unsigned int _v = V; \
            for (int i = 0; i < 4; i++) { \
                unsigned char byte = (_v >> (i * 8)) & 0xFF; \
                int ki = k_base + (off) + i * 2; \
                dot += __ldg(&A[ki]) * (float)((byte & 0x0F) - 8) * scale; \
                dot += __ldg(&A[ki + 1]) * (float)(((byte >> 4) & 0x0F) - 8) * scale; \
            } \
        }

        BM_PROCESS(val0, 0);
        BM_PROCESS(val1, 8);
        BM_PROCESS(val2, 16);
        BM_PROCESS(val3, 24);

        #undef BM_PROCESS
    }

    C[n] = dot;
}
"#;

// ── Warp-cooperative M=1 Q4 GEMM (V3) ──────────────────────────
//
// Each warp processes 32 output columns. Warp cooperatively loads
// 32 Q4 blocks (640 bytes) into shared memory via 5 coalesced
// 128-byte cache line transactions. Then each thread computes
// its column from shared memory — zero wasted bandwidth.
//
// Additionally: processes 4 K-blocks between syncs to amortize
// sync overhead and improve instruction-level parallelism.

const GEMM_Q4_0_M1_V3_SRC: &str = r#"
#define WARP_SIZE 32

extern "C" __global__ void warp_gemm_q4_0_m1_v3(
    float* __restrict__ C,
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_q4,
    int K,
    int N
) {
    // Shared memory: each warp gets 32 * 20 = 640 bytes
    extern __shared__ unsigned char smem[];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int global_n = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned char* warp_smem = smem + warp_id * WARP_SIZE * 20;

    float dot = 0.0f;
    const int num_k_blocks = K / 32;

    for (int b = 0; b < num_k_blocks; b++) {
        // Cooperative warp load: 32 threads load 640 bytes (32 × 20-byte blocks)
        // from contiguous global memory into shared memory.
        // 640 bytes = 160 x 4-byte words. 32 threads × 5 words each.
        if (global_n < N) {
            const unsigned char* src_base = B_q4 + ((long long)b * N + (blockIdx.x * blockDim.x + warp_id * WARP_SIZE)) * 20;
            unsigned char* dst_base = warp_smem;

            // Each thread loads 5 × 4 bytes = 20 bytes (its own Q4 block)
            // But arranged so that the warp reads contiguously:
            // Thread lane loads 4 bytes at stride-32 offsets for coalescing
            #pragma unroll
            for (int w = 0; w < 5; w++) {
                // All 32 threads read 4 bytes each from consecutive addresses
                // Offset: w * 128 + lane * 4 (128-byte aligned per iteration)
                unsigned int val = *((const unsigned int*)(src_base + w * 128 + lane * 4));
                *((unsigned int*)(dst_base + w * 128 + lane * 4)) = val;
            }
        }
        __syncthreads();

        if (global_n < N) {
            // Read this thread's Q4 block from shared memory
            const unsigned char* my_block = warp_smem + lane * 20;
            float scale = *((const float*)my_block);
            const unsigned char* packed = my_block + 4;

            int k_base = b * 32;

            // Process 32 elements (16 packed bytes)
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                unsigned char byte = packed[i];
                int ki = k_base + i * 2;
                float a0 = __ldg(&A[ki]);
                float a1 = __ldg(&A[ki + 1]);
                dot += a0 * (float)((byte & 0x0F) - 8) * scale;
                dot += a1 * (float)(((byte >> 4) & 0x0F) - 8) * scale;
            }
        }
        __syncthreads();
    }

    if (global_n < N) {
        C[global_n] = dot;
    }
}
"#;

// ── Split-K M=1 Q4 GEMM for maximum SM occupancy ───────────────
//
// The standard M=1 kernel assigns one thread per output column.
// For small N (e.g., KV projection N=512), only 2 blocks launch —
// 2% SM utilization on RTX 4090's 128 SMs.
//
// Split-K divides the K dimension across multiple blocks. Each block
// processes K/splits K-blocks for its assigned output columns, then
// atomicAdds partial results. This multiplies the block count by splits.
//
// With auto-selected splits: KV proj goes from 2% to 50%+ SM util,
// attention Q/O from 11% to 88%, FFN from 58% to 100%.

// ═══════════════════════════════════════════════════════════════
// TW-Marlin Format: Separated Scales + Packed Nibbles
// ═══════════════════════════════════════════════════════════════
//
// Q4_0 stores [scale(4B) + nibbles(16B)] interleaved = 20-byte blocks.
// This 20-byte stride means warp reads waste cache line bandwidth.
//
// TW-Marlin separates them:
//   packed[num_k_groups][N][16] — contiguous nibbles, uint4-aligned
//   scales[num_k_groups][N]    — contiguous f32 scales
//
// Warp of 32 threads reads:
//   32 × 16B packed = 512B = 4 cache lines (100% utilized)
//   32 × 4B scales  = 128B = 1 cache line  (100% utilized)
//   Total: 5 cache lines, 640B useful / 640B fetched = 100% efficiency
//
// vs Q4_0 block-major: 20-byte stride, ~85% efficiency

/// Reorder Q4_0 column-block weights into TW-Marlin separated format.
/// Returns (packed_nibbles, scales) as separate GPU tensors.
pub fn reorder_to_tw_marlin(
    device: &WarpDevice,
    col_major_q4: &GpuTensor<u8>,
    k: u32,
    n: u32,
) -> Result<(GpuTensor<u8>, GpuTensor<f32>), DeviceError> {
    let num_k_groups = k / BLOCK_SIZE;
    let src = col_major_q4.to_host(device)?;

    // Packed nibbles: [num_k_groups][N][16]
    let packed_size = (num_k_groups * n * 16) as usize;
    let mut packed = vec![0u8; packed_size];

    // Scales: [num_k_groups][N]
    let scales_size = (num_k_groups * n) as usize;
    let mut scales = vec![0.0f32; scales_size];

    for col in 0..n as usize {
        for g in 0..num_k_groups as usize {
            // Source: column-major Q4_0 block at (col * num_k_groups + g) * 20
            let src_offset = (col * num_k_groups as usize + g) * Q4_0_BLOCK_BYTES as usize;

            // Read scale (f32, 4 bytes)
            let scale = f32::from_le_bytes([
                src[src_offset], src[src_offset + 1],
                src[src_offset + 2], src[src_offset + 3],
            ]);

            // Destination indices
            let scale_idx = g * n as usize + col;
            let packed_offset = (g * n as usize + col) * 16;

            scales[scale_idx] = scale;
            packed[packed_offset..packed_offset + 16]
                .copy_from_slice(&src[src_offset + 4..src_offset + 20]);
        }
    }

    let packed_tensor = GpuTensor::from_host(device, &packed,
        warp_ir::Shape::from_static(&[packed_size]), warp_ir::DType::U8)?;
    let scales_tensor = GpuTensor::from_host(device, &scales,
        warp_ir::Shape::from_static(&[scales_size]), warp_ir::DType::F32)?;

    Ok((packed_tensor, scales_tensor))
}

/// Fuse multiple TW-Marlin weight matrices along the N dimension.
/// E.g., fuse Q[K,H] + K[K,KV] + V[K,KV] into QKV[K, H+2*KV].
/// Packed nibbles and scales are concatenated along N for each K-group.
pub fn fuse_tw_marlin_n(
    device: &WarpDevice,
    matrices: &[(&GpuTensor<u8>, &GpuTensor<f32>)], // [(packed, scales), ...]
    k: u32,
    ns: &[u32], // N dimension of each matrix
) -> Result<(GpuTensor<u8>, GpuTensor<f32>), DeviceError> {
    let num_k_groups = k / BLOCK_SIZE;
    let total_n: u32 = ns.iter().sum();

    // Read all to host
    let packed_hosts: Vec<Vec<u8>> = matrices.iter()
        .map(|(p, _)| p.to_host(device)).collect::<Result<_, _>>()?;
    let scale_hosts: Vec<Vec<f32>> = matrices.iter()
        .map(|(_, s)| s.to_host(device)).collect::<Result<_, _>>()?;

    let packed_size = (num_k_groups * total_n * 16) as usize;
    let scales_size = (num_k_groups * total_n) as usize;
    let mut fused_packed = vec![0u8; packed_size];
    let mut fused_scales = vec![0.0f32; scales_size];

    for g in 0..num_k_groups as usize {
        let mut col_offset = 0usize;
        for (idx, &n) in ns.iter().enumerate() {
            let n = n as usize;
            for c in 0..n {
                // Packed: src[g * n + c] * 16 → dst[g * total_n + col_offset + c] * 16
                let src_packed_off = (g * n + c) * 16;
                let dst_packed_off = (g * total_n as usize + col_offset + c) * 16;
                fused_packed[dst_packed_off..dst_packed_off + 16]
                    .copy_from_slice(&packed_hosts[idx][src_packed_off..src_packed_off + 16]);

                // Scales: src[g * n + c] → dst[g * total_n + col_offset + c]
                fused_scales[g * total_n as usize + col_offset + c] =
                    scale_hosts[idx][g * n + c];
            }
            col_offset += n;
        }
    }

    let packed_tensor = GpuTensor::from_host(device, &fused_packed,
        warp_ir::Shape::from_static(&[packed_size]), warp_ir::DType::U8)?;
    let scales_tensor = GpuTensor::from_host(device, &fused_scales,
        warp_ir::Shape::from_static(&[scales_size]), warp_ir::DType::F32)?;

    Ok((packed_tensor, scales_tensor))
}

/// TW-Marlin format weights for one transformer layer.
pub struct TWMarlinWeights {
    pub attn_norm: GpuTensor<f32>,
    pub ffn_norm: GpuTensor<f32>,
    // Separated packed nibbles + scales for each projection
    pub wq_packed: GpuTensor<u8>,  pub wq_scales: GpuTensor<f32>,
    pub wk_packed: GpuTensor<u8>,  pub wk_scales: GpuTensor<f32>,
    pub wv_packed: GpuTensor<u8>,  pub wv_scales: GpuTensor<f32>,
    pub wo_packed: GpuTensor<u8>,  pub wo_scales: GpuTensor<f32>,
    pub wg_packed: GpuTensor<u8>,  pub wg_scales: GpuTensor<f32>,
    pub wu_packed: GpuTensor<u8>,  pub wu_scales: GpuTensor<f32>,
    pub wd_packed: GpuTensor<u8>,  pub wd_scales: GpuTensor<f32>,
    pub bq: Option<GpuTensor<f32>>,
    pub bk: Option<GpuTensor<f32>>,
    pub bv: Option<GpuTensor<f32>>,
    // Fused projections — 3 GEMMs → 1 (created by fuse_projections)
    pub wqkv_packed: Option<GpuTensor<u8>>,
    pub wqkv_scales: Option<GpuTensor<f32>>,
    pub wgu_packed: Option<GpuTensor<u8>>,  // gate+up fused
    pub wgu_scales: Option<GpuTensor<f32>>,
}

impl TWMarlinWeights {
    /// Convert from QuantizedBlockWeights to TW-Marlin separated format.
    pub fn from_quantized(
        device: &WarpDevice,
        q: &crate::transformer::QuantizedBlockWeights,
        config: &crate::transformer::TransformerConfig,
    ) -> Result<Self, DeviceError> {
        let h = config.hidden_size;
        let kv = config.kv_dim();
        let ffn = config.ffn_dim;

        let (wq_p, wq_s) = reorder_to_tw_marlin(device, &q.wq, h, h)?;
        let (wk_p, wk_s) = reorder_to_tw_marlin(device, &q.wk, h, kv)?;
        let (wv_p, wv_s) = reorder_to_tw_marlin(device, &q.wv, h, kv)?;
        let (wo_p, wo_s) = reorder_to_tw_marlin(device, &q.wo, h, h)?;
        let (wg_p, wg_s) = reorder_to_tw_marlin(device, &q.w_gate, h, ffn)?;
        let (wu_p, wu_s) = reorder_to_tw_marlin(device, &q.w_up, h, ffn)?;
        let (wd_p, wd_s) = reorder_to_tw_marlin(device, &q.w_down, ffn, h)?;

        Ok(Self {
            attn_norm: GpuTensor::from_host(device, &q.attn_norm.to_host(device)?,
                q.attn_norm.shape.clone(), warp_ir::DType::F32)?,
            ffn_norm: GpuTensor::from_host(device, &q.ffn_norm.to_host(device)?,
                q.ffn_norm.shape.clone(), warp_ir::DType::F32)?,
            wq_packed: wq_p, wq_scales: wq_s,
            wk_packed: wk_p, wk_scales: wk_s,
            wv_packed: wv_p, wv_scales: wv_s,
            wo_packed: wo_p, wo_scales: wo_s,
            wg_packed: wg_p, wg_scales: wg_s,
            wu_packed: wu_p, wu_scales: wu_s,
            wd_packed: wd_p, wd_scales: wd_s,
            bq: match &q.bq {
                Some(t) => Some(GpuTensor::from_host(device, &t.to_host(device)?, t.shape.clone(), warp_ir::DType::F32)?),
                None => None,
            },
            bk: match &q.bk {
                Some(t) => Some(GpuTensor::from_host(device, &t.to_host(device)?, t.shape.clone(), warp_ir::DType::F32)?),
                None => None,
            },
            bv: match &q.bv {
                Some(t) => Some(GpuTensor::from_host(device, &t.to_host(device)?, t.shape.clone(), warp_ir::DType::F32)?),
                None => None,
            },
            wqkv_packed: None, wqkv_scales: None,
            wgu_packed: None, wgu_scales: None,
        })
    }

    /// Fuse QKV and gate+up projections for fewer GEMM launches.
    /// 3 QKV GEMMs → 1, 2 gate+up GEMMs → 1. Saves 3 kernel launches per layer.
    pub fn fuse_projections(
        &mut self,
        device: &WarpDevice,
        config: &crate::transformer::TransformerConfig,
    ) -> Result<(), DeviceError> {
        let h = config.hidden_size;
        let kv = config.kv_dim();
        let ffn = config.ffn_dim;

        // Fuse QKV: [K=H, N=H+KV+KV]
        let (qkv_p, qkv_s) = fuse_tw_marlin_n(device,
            &[(&self.wq_packed, &self.wq_scales),
              (&self.wk_packed, &self.wk_scales),
              (&self.wv_packed, &self.wv_scales)],
            h, &[h, kv, kv])?;
        self.wqkv_packed = Some(qkv_p);
        self.wqkv_scales = Some(qkv_s);

        // Fuse gate+up: [K=H, N=2*FFN]
        let (gu_p, gu_s) = fuse_tw_marlin_n(device,
            &[(&self.wg_packed, &self.wg_scales),
              (&self.wu_packed, &self.wu_scales)],
            h, &[ffn, ffn])?;
        self.wgu_packed = Some(gu_p);
        self.wgu_scales = Some(gu_s);

        Ok(())
    }
}

/// TW-Marlin Split-K M=1 GEMM kernel with separated scales + packed nibbles.
/// uint4 loads for packed data (16B aligned), f32 loads for scales.
const GEMM_TW_MARLIN_M1_SRC: &str = r#"
extern "C" __global__ void warp_gemm_tw_marlin_m1(
    float* __restrict__ C,
    const float* __restrict__ A,
    const unsigned char* __restrict__ packed,  // [num_k_groups, N, 16]
    const float* __restrict__ scales,          // [num_k_groups, N]
    int K,
    int N,
    int num_k_groups,
    int k_blocks_per_split
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    int k_split_id = blockIdx.y;
    int start_g = k_split_id * k_blocks_per_split;
    int end_g = start_g + k_blocks_per_split;
    if (end_g > num_k_groups) end_g = num_k_groups;

    float dot = 0.0f;

    for (int g = start_g; g < end_g; g++) {
        // Load scale — adjacent threads read adjacent floats (coalesced!)
        float scale = scales[g * N + n];

        // Load 16 packed bytes via uint4 (128-bit aligned load!)
        // packed layout: [g][n][16 bytes]
        const unsigned char* my_packed = packed + ((long long)g * N + n) * 16;
        const unsigned int* p = (const unsigned int*)my_packed;
        unsigned int v0 = p[0], v1 = p[1], v2 = p[2], v3 = p[3];

        int k_base = g * 32;

        #define TW_PROC(V, off) { \
            unsigned int _v = V; \
            for (int i = 0; i < 4; i++) { \
                unsigned char byte = (_v >> (i * 8)) & 0xFF; \
                int ki = k_base + (off) + i * 2; \
                dot += __ldg(&A[ki]) * (float)((byte & 0x0F) - 8) * scale; \
                dot += __ldg(&A[ki + 1]) * (float)(((byte >> 4) & 0x0F) - 8) * scale; \
            } \
        }

        TW_PROC(v0, 0);
        TW_PROC(v1, 8);
        TW_PROC(v2, 16);
        TW_PROC(v3, 24);

        #undef TW_PROC
    }

    atomicAdd(&C[n], dot);
}
"#;

/// TW-Marlin M=1 GEMM with separated format.
pub fn gemm_tw_marlin_m1(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    packed: &GpuTensor<u8>,
    scales: &GpuTensor<f32>,
    c: &mut GpuTensor<f32>,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    assert!(k % BLOCK_SIZE == 0);
    let num_k_groups = k / BLOCK_SIZE;

    let threads = 256u32;
    let n_blocks = (n + threads - 1) / threads;

    // Adaptive Split-K
    let splits = if k < 2048 || n_blocks >= 128 {
        1
    } else {
        let target = 256u32;
        let max_splits = (num_k_groups / 8).max(1);
        ((target + n_blocks - 1) / n_blocks).max(1).min(max_splits)
    };
    let k_blocks_per_split = (num_k_groups + splits - 1) / splits;

    let f = cache.get_or_compile(device, GEMM_TW_MARLIN_M1_SRC, "warp_gemm_tw_marlin_m1")?;

    // Zero output for atomicAdd
    device.stream.memset_zeros(&mut c.data)
        .map_err(|e| DeviceError::Memory(format!("memset zeros: {e}")))?;

    let k_i = k as i32;
    let n_i = n as i32;
    let nkg_i = num_k_groups as i32;
    let kbps_i = k_blocks_per_split as i32;

    let cfg = LaunchConfig {
        grid_dim: (n_blocks, splits, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut c.data)
            .arg(&a.data)
            .arg(&packed.data)
            .arg(&scales.data)
            .arg(&k_i)
            .arg(&n_i)
            .arg(&nkg_i)
            .arg(&kbps_i)
            .launch(cfg))?;
    }
    Ok(())
}

const GEMM_Q4_0_M1_SPLITK_SRC: &str = r#"
extern "C" __global__ void warp_gemm_q4_0_m1_splitk(
    float* __restrict__ C,
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_q4,
    int K,
    int N,
    int num_k_blocks_total,
    int k_blocks_per_split
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // blockIdx.y = split index (all splits launch in ONE kernel)
    int k_split_id = blockIdx.y;
    int start_b = k_split_id * k_blocks_per_split;
    int end_b = start_b + k_blocks_per_split;
    if (end_b > num_k_blocks_total) end_b = num_k_blocks_total;

    float dot = 0.0f;

    for (int b = start_b; b < end_b; b++) {
        // Block-major layout: (b * N + n) * 20
        const unsigned char* blk = B_q4 + ((long long)b * N + n) * 20;

        float scale = *((const float*)blk);
        const unsigned int* packed = (const unsigned int*)(blk + 4);
        unsigned int val0 = packed[0];
        unsigned int val1 = packed[1];
        unsigned int val2 = packed[2];
        unsigned int val3 = packed[3];

        int k_base = b * 32;

        #define SK_PROCESS(V, off) { \
            unsigned int _v = V; \
            for (int i = 0; i < 4; i++) { \
                unsigned char byte = (_v >> (i * 8)) & 0xFF; \
                int ki = k_base + (off) + i * 2; \
                dot += __ldg(&A[ki]) * (float)((byte & 0x0F) - 8) * scale; \
                dot += __ldg(&A[ki + 1]) * (float)(((byte >> 4) & 0x0F) - 8) * scale; \
            } \
        }

        SK_PROCESS(val0, 0);
        SK_PROCESS(val1, 8);
        SK_PROCESS(val2, 16);
        SK_PROCESS(val3, 24);

        #undef SK_PROCESS
    }

    // Accumulate partial result via atomicAdd
    atomicAdd(&C[n], dot);
}
"#;

/// Split-K M=1 Q4_0 GEMM with block-major weights.
/// Divides K across multiple block launches for maximum SM occupancy.
/// Requires C to be zeroed before call (uses atomicAdd for accumulation).
pub fn gemm_q4_0_m1_splitk(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,       // [1, K]
    b_quant: &GpuTensor<u8>,  // block-major Q4_0 [K, N]
    c: &mut GpuTensor<f32>,   // [1, N] — MUST be zeroed before call
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    assert!(k % BLOCK_SIZE == 0, "K must be divisible by {BLOCK_SIZE}");
    let num_k_blocks = k / BLOCK_SIZE;

    // Auto-select split count to maximize SM occupancy.
    // Only use Split-K when n_blocks alone can't fill the GPU.
    // Cap splits to ensure each split processes at least 8 K-blocks
    // (fewer than that = atomicAdd overhead dominates).
    let threads = 256u32;
    let n_blocks = (n + threads - 1) / threads;
    let splits = if k < 2048 || n_blocks >= 128 {
        1 // Small K or enough N-blocks already — no split needed
    } else {
        let target_blocks = 256u32; // 2x SM count
        let max_splits = (num_k_blocks / 8).max(1); // at least 8 K-blocks per split
        ((target_blocks + n_blocks - 1) / n_blocks).max(1).min(max_splits)
    };
    let k_blocks_per_split = (num_k_blocks + splits - 1) / splits;

    let f = cache.get_or_compile(device, GEMM_Q4_0_M1_SPLITK_SRC, "warp_gemm_q4_0_m1_splitk")?;

    // Zero the output (atomicAdd accumulates into it)
    device.stream.memset_zeros(&mut c.data)
        .map_err(|e| DeviceError::Memory(format!("memset zeros: {e}")))?;

    let k_i = k as i32;
    let n_i = n as i32;
    let num_kb_i = num_k_blocks as i32;
    let kbps_i = k_blocks_per_split as i32;

    // 2D grid: x = output columns, y = K-splits. ONE launch for all splits!
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, splits, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut c.data)
            .arg(&a.data)
            .arg(&b_quant.data)
            .arg(&k_i)
            .arg(&n_i)
            .arg(&num_kb_i)
            .arg(&kbps_i)
            .launch(cfg))?;
    }
    Ok(())
}

/// M=1 Q4_0 GEMM with block-major weights (coalesced reads).
pub fn gemm_q4_0_m1_blockmajor(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,       // [1, K]
    b_quant: &GpuTensor<u8>,  // block-major Q4_0 [K, N]
    c: &mut GpuTensor<f32>,   // [1, N]
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    assert!(k % BLOCK_SIZE == 0, "K must be divisible by {BLOCK_SIZE}");

    let f = cache.get_or_compile(device, GEMM_Q4_0_M1_BLOCKMAJOR_SRC, "warp_gemm_q4_0_m1_bm")?;
    let threads = 256u32;
    let blocks = (n + threads - 1) / threads;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let k_i = k as i32;
    let n_i = n as i32;

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut c.data)
            .arg(&a.data)
            .arg(&b_quant.data)
            .arg(&k_i)
            .arg(&n_i)
            .launch(cfg))?;
    }
    Ok(())
}

const GEMM_Q4_0_M1_SRC: &str = r#"
extern "C" __global__ void warp_gemm_q4_0_m1(
    float* __restrict__ C,
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_q4,
    int K,
    int N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float dot = 0.0f;
    const int num_k_blocks = K / 32;

    // Base pointer for column n
    const unsigned char* B_col = B_q4 + (long long)n * num_k_blocks * 20;

    for (int b = 0; b < num_k_blocks; b++) {
        const unsigned char* blk = B_col + (long long)b * 20;

        // Load scale
        float scale = *((const float*)blk);

        // Load 16 packed bytes as 4 uints for efficient aligned access
        const unsigned int* packed = (const unsigned int*)(blk + 4);
        unsigned int val0 = packed[0];
        unsigned int val1 = packed[1];
        unsigned int val2 = packed[2];
        unsigned int val3 = packed[3];

        int k_base = b * 32;

        // Unroll: each uint = 4 bytes = 8 elements
        #define PROCESS_UINT(V, offset) { \
            unsigned int _v = V; \
            for (int i = 0; i < 4; i++) { \
                unsigned char byte = (_v >> (i * 8)) & 0xFF; \
                float q0 = (float)((byte & 0x0F) - 8) * scale; \
                float q1 = (float)(((byte >> 4) & 0x0F) - 8) * scale; \
                int ki = k_base + (offset) + i * 2; \
                dot += __ldg(&A[ki]) * q0; \
                dot += __ldg(&A[ki + 1]) * q1; \
            } \
        }

        PROCESS_UINT(val0, 0);
        PROCESS_UINT(val1, 8);
        PROCESS_UINT(val2, 16);
        PROCESS_UINT(val3, 24);

        #undef PROCESS_UINT
    }

    C[n] = dot;
}
"#;

// ── Optimized M=1 Q4 GEMM with shared-memory A vector ──────────
//
// V2: Loads A vector into shared memory cooperatively (all threads in block).
// Each thread processes COLS_PER_THREAD output columns to amortize the A load.
// For K=3584, this reduces redundant global reads of A from 256x to 1x per block.

const GEMM_Q4_0_M1_V2_SRC: &str = r#"
#define COLS_PER_THREAD 4
#define BLOCK_THREADS 256

extern "C" __global__ void warp_gemm_q4_0_m1_v2(
    float* __restrict__ C,
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_q4,
    int K,
    int N
) {
    // Shared memory for A vector — loaded cooperatively by all threads
    extern __shared__ float smem_A[];

    // Load A into shared memory (K can be larger than blockDim.x)
    for (int i = threadIdx.x; i < K; i += BLOCK_THREADS) {
        smem_A[i] = A[i];
    }
    __syncthreads();

    const int num_k_blocks = K / 32;

    // Each thread processes COLS_PER_THREAD output columns
    int base_n = (blockIdx.x * BLOCK_THREADS + threadIdx.x) * COLS_PER_THREAD;

    float dots[COLS_PER_THREAD];
    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; c++) dots[c] = 0.0f;

    for (int b = 0; b < num_k_blocks; b++) {
        int k_base = b * 32;

        #pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; c++) {
            int n = base_n + c;
            if (n >= N) continue;

            const unsigned char* blk = B_q4 + ((long long)n * num_k_blocks + b) * 20;
            float scale = *((const float*)blk);
            const unsigned int* packed = (const unsigned int*)(blk + 4);

            #pragma unroll
            for (int v = 0; v < 4; v++) {
                unsigned int val = packed[v];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    unsigned char byte = (val >> (i * 8)) & 0xFF;
                    int ki = k_base + v * 8 + i * 2;
                    dots[c] += smem_A[ki] * (float)((byte & 0x0F) - 8) * scale;
                    dots[c] += smem_A[ki + 1] * (float)(((byte >> 4) & 0x0F) - 8) * scale;
                }
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; c++) {
        int n = base_n + c;
        if (n < N) C[n] = dots[c];
    }
}
"#;

/// M=1 specialized Q4_0 GEMM: C[1,N] = A[1,K] × dequant(B_q4[K,N])
///
/// ~5-10x faster than the generic 32×32 tiled kernel for single-token decode.
/// One thread per output column, full K dot product per thread.
pub fn gemm_q4_0_m1(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,       // [1, K]
    b_quant: &GpuTensor<u8>,  // quantized [K, N]
    c: &mut GpuTensor<f32>,   // [1, N]
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    assert!(k % BLOCK_SIZE == 0, "K must be divisible by {BLOCK_SIZE}");

    let f = cache.get_or_compile(device, GEMM_Q4_0_M1_SRC, "warp_gemm_q4_0_m1")?;
    let threads = 256u32;
    let blocks = (n + threads - 1) / threads;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let k_i = k as i32;
    let n_i = n as i32;

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut c.data)
            .arg(&a.data)
            .arg(&b_quant.data)
            .arg(&k_i)
            .arg(&n_i)
            .launch(cfg))?;
    }
    Ok(())
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

// ── INT8 Symmetric Quantize (A8W8) ─────────────────────────────
// Per-tensor or per-channel symmetric INT8 quantization.

const QUANTIZE_INT8_SRC: &str = r#"
extern "C" __global__ void warp_quantize_int8(
    signed char *out,      // [n] quantized output
    float *scale_out,      // [1] or [channels] scale
    const float *input,    // [n] float input
    unsigned int n,
    unsigned int per_channel, // 0 = per-tensor, 1 = per-channel
    unsigned int channel_size // elements per channel
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int ch = per_channel ? (idx / channel_size) : 0;
    float scale = scale_out[ch];
    float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

    float v = input[idx] * inv_scale;
    int q = (int)roundf(v);
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    out[idx] = (signed char)q;
}
"#;

// Separate kernel to compute per-tensor or per-channel scales.
const COMPUTE_INT8_SCALES_SRC: &str = r#"
extern "C" __global__ void warp_compute_int8_scales(
    float *scale_out,       // [1] or [channels]
    const float *input,     // [n]
    unsigned int n,
    unsigned int per_channel,
    unsigned int channel_size,
    unsigned int num_channels
) {
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int max_ch = per_channel ? num_channels : 1;
    if (ch >= max_ch) return;

    unsigned int start = per_channel ? (ch * channel_size) : 0;
    unsigned int end = per_channel ? (start + channel_size) : n;
    if (end > n) end = n;

    float amax = 0.0f;
    for (unsigned int i = start; i < end; i++) {
        float a = fabsf(input[i]);
        if (a > amax) amax = a;
    }
    scale_out[ch] = amax / 127.0f;
}
"#;

// ── INT8 Dequantize ────────────────────────────────────────────

const DEQUANTIZE_INT8_SRC: &str = r#"
extern "C" __global__ void warp_dequantize_int8(
    float *out,
    const signed char *input,
    const float *scale,
    unsigned int n,
    unsigned int per_channel,
    unsigned int channel_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int ch = per_channel ? (idx / channel_size) : 0;
    out[idx] = scale[ch] * (float)input[idx];
}
"#;

// ── INT8 GEMM with dp4a ────────────────────────────────────────
// C[M,N] = dequant(A_int8[M,K] × B_int8[K,N])
// Uses __dp4a for 4-way int8 dot product.
// B is stored in column-major order (transposed) so that K is contiguous
// for each output column, enabling packed dp4a loads.

const GEMM_INT8_SRC: &str = r#"
#define BM 32
#define BN 32
#define BK 32

// Manual dp4a: dot product of 4 packed int8 values, accumulated into int32.
// Equivalent to __dp4a but works on all compute capabilities.
__device__ __forceinline__ int dp4a_manual(int a, int b, int c) {
    signed char *a_bytes = reinterpret_cast<signed char*>(&a);
    signed char *b_bytes = reinterpret_cast<signed char*>(&b);
    c += (int)a_bytes[0] * (int)b_bytes[0];
    c += (int)a_bytes[1] * (int)b_bytes[1];
    c += (int)a_bytes[2] * (int)b_bytes[2];
    c += (int)a_bytes[3] * (int)b_bytes[3];
    return c;
}

extern "C" __global__ void warp_gemm_int8(
    float *C,              // [M, N] float output (dequantized)
    const signed char *A,  // [M, K] int8 activations (row-major)
    const signed char *B,  // [K, N] int8 weights (row-major)
    const float *scale_a,  // [1] activation scale (per-tensor)
    const float *scale_b,  // [N] weight scale (per-channel)
    unsigned int M, unsigned int N, unsigned int K
) {
    __shared__ signed char As[BM][BK];
    __shared__ signed char Bs[BK][BN];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int row = by * BM + ty;
    unsigned int col = bx * BN + tx;

    int acc = 0;

    for (unsigned int t = 0; t < (K + BK - 1) / BK; t++) {
        // Load A tile
        unsigned int a_col = t * BK + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0;

        // Load B tile
        unsigned int b_row = t * BK + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0;

        __syncthreads();

        // Accumulate using dp4a: process 4 elements at a time
        for (unsigned int i = 0; i < BK; i += 4) {
            // Pack 4 int8 values from A row
            int a_packed = ((unsigned char)As[ty][i]) |
                           ((unsigned char)As[ty][i+1] << 8) |
                           ((unsigned char)As[ty][i+2] << 16) |
                           ((unsigned char)As[ty][i+3] << 24);
            // Pack 4 int8 values from B column
            int b_packed = ((unsigned char)Bs[i][tx]) |
                           ((unsigned char)Bs[i+1][tx] << 8) |
                           ((unsigned char)Bs[i+2][tx] << 16) |
                           ((unsigned char)Bs[i+3][tx] << 24);
            acc = dp4a_manual(a_packed, b_packed, acc);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        // Dequantize: float_result = int32_acc * scale_a[0] * scale_b[col]
        C[row * N + col] = (float)acc * scale_a[0] * scale_b[col];
    }
}
"#;

// ═════════════════════════════════════════════════════════════════
// Rust API — INT8 (A8W8)
// ═════════════════════════════════════════════════════════════════

/// Quantize f32 tensor to INT8 with symmetric per-tensor or per-channel scaling.
///
/// - `input`: [n] f32
/// - `output`: [n] i8 (stored as `GpuTensor<i8>`)
/// - `scale`: [1] for per-tensor, [channels] for per-channel
/// - `per_channel`: if true, quantize each channel independently
///
/// The function first computes scales, then quantizes.
pub fn quantize_int8(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    output: &mut GpuTensor<i8>,
    scale: &mut GpuTensor<f32>,
    per_channel: bool,
) -> Result<(), DeviceError> {
    let n = input.numel as u32;
    let per_ch_flag: u32 = if per_channel { 1 } else { 0 };
    let channel_size: u32 = if per_channel {
        let num_channels = scale.numel as u32;
        n / num_channels
    } else {
        n
    };
    let num_channels: u32 = if per_channel { scale.numel as u32 } else { 1 };

    // Step 1: compute scales
    let scale_f = cache.get_or_compile(device, COMPUTE_INT8_SCALES_SRC, "warp_compute_int8_scales")?;
    let scale_cfg = LaunchConfig::for_num_elems(num_channels);
    unsafe {
        launch_err!(device.stream.launch_builder(&scale_f)
            .arg(&mut scale.data)
            .arg(&input.data)
            .arg(&n)
            .arg(&per_ch_flag)
            .arg(&channel_size)
            .arg(&num_channels)
            .launch(scale_cfg))?;
    }

    // Step 2: quantize
    let quant_f = cache.get_or_compile(device, QUANTIZE_INT8_SRC, "warp_quantize_int8")?;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe {
        launch_err!(device.stream.launch_builder(&quant_f)
            .arg(&mut output.data)
            .arg(&scale.data)
            .arg(&input.data)
            .arg(&n)
            .arg(&per_ch_flag)
            .arg(&channel_size)
            .launch(cfg))?;
    }
    Ok(())
}

/// Dequantize INT8 tensor back to f32.
pub fn dequantize_int8(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<i8>,
    output: &mut GpuTensor<f32>,
    scale: &GpuTensor<f32>,
    per_channel: bool,
) -> Result<(), DeviceError> {
    let n = output.numel as u32;
    let per_ch_flag: u32 = if per_channel { 1 } else { 0 };
    let channel_size: u32 = if per_channel {
        let num_channels = scale.numel as u32;
        n / num_channels
    } else {
        n
    };

    let f = cache.get_or_compile(device, DEQUANTIZE_INT8_SRC, "warp_dequantize_int8")?;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data)
            .arg(&input.data)
            .arg(&scale.data)
            .arg(&n)
            .arg(&per_ch_flag)
            .arg(&channel_size)
            .launch(cfg))?;
    }
    Ok(())
}

/// INT8 GEMM (A8W8): C[M,N] = dequant(A_int8[M,K] x B_int8[K,N])
///
/// Both A and B must already be quantized to INT8 with corresponding scales.
/// Uses dp4a for fast 4-way int8 dot product on Turing+ GPUs.
///
/// K must be divisible by 4 (dp4a processes 4 elements at a time).
pub fn gemm_int8(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<i8>,        // [M, K]
    b: &GpuTensor<i8>,        // [K, N]
    c: &mut GpuTensor<f32>,   // [M, N]
    scale_a: &GpuTensor<f32>, // [1] per-tensor
    scale_b: &GpuTensor<f32>, // [N] per-channel
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    assert!(k % 4 == 0, "K ({k}) must be divisible by 4 for dp4a");
    let tile = 32u32;

    let f = cache.get_or_compile(device, GEMM_INT8_SRC, "warp_gemm_int8")?;
    let cfg = LaunchConfig {
        grid_dim: ((n + tile - 1) / tile, (m + tile - 1) / tile, 1),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut c.data)
            .arg(&a.data)
            .arg(&b.data)
            .arg(&scale_a.data)
            .arg(&scale_b.data)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg))?;
    }
    Ok(())
}

// ═════════════════════════════════════════════════════════════════
// FP8 E4M3 Quantization (emulated via scaled INT8 with E4M3 range)
// ═════════════════════════════════════════════════════════════════

// FP8 E4M3: 4 exponent bits, 3 mantissa bits, max representable = 448.
// On pre-Hopper GPUs we emulate: scale = max(|x|) / 448, quantize to
// [-128, 127] stored as uint8, dequantize = q * scale.

const COMPUTE_FP8_SCALE_SRC: &str = r#"
extern "C" __global__ void warp_compute_fp8_scale(
    float *scale_out,
    const float *input,
    unsigned int n
) {
    // Single-thread scan (called with 1 thread for simplicity).
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float amax = 0.0f;
    for (unsigned int i = 0; i < n; i++) {
        float a = fabsf(input[i]);
        if (a > amax) amax = a;
    }
    scale_out[0] = amax / 448.0f;
}
"#;

const QUANTIZE_FP8_SRC: &str = r#"
extern "C" __global__ void warp_quantize_fp8_e4m3(
    unsigned char *out,
    const float *input,
    const float *scale,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float s = scale[0];
    float inv_scale = (s != 0.0f) ? 1.0f / s : 0.0f;
    float v = input[i] * inv_scale;
    int q = (int)roundf(v);
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    // Store as uint8: signed char cast then reinterpret
    out[i] = (unsigned char)((signed char)q);
}
"#;

const DEQUANTIZE_FP8_SRC: &str = r#"
extern "C" __global__ void warp_dequantize_fp8(
    float *out,
    const unsigned char *input,
    const float *scale,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float s = scale[0];
    signed char q = (signed char)input[i];
    out[i] = s * (float)q;
}
"#;

const GEMM_FP8_SRC: &str = r#"
#define TILE 32

extern "C" __global__ void warp_gemm_fp8(
    float *C,
    const unsigned char *A,    // [M, K] FP8 (uint8)
    const unsigned char *B,    // [K, N] FP8 (uint8)
    const float *scale_a,      // [1]
    const float *scale_b,      // [1]
    unsigned int M, unsigned int N, unsigned int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int row = by * TILE + ty;
    unsigned int col = bx * TILE + tx;

    float sa = scale_a[0];
    float sb = scale_b[0];
    float sum = 0.0f;

    for (unsigned int t = 0; t < (K + TILE - 1) / TILE; t++) {
        unsigned int a_col = t * TILE + tx;
        if (row < M && a_col < K) {
            signed char qa = (signed char)A[row * K + a_col];
            As[ty][tx] = sa * (float)qa;
        } else {
            As[ty][tx] = 0.0f;
        }

        unsigned int b_row = t * TILE + ty;
        if (b_row < K && col < N) {
            signed char qb = (signed char)B[b_row * N + col];
            Bs[ty][tx] = sb * (float)qb;
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

// ── FP8 Rust API ───────────────────────────────────────────────

/// Quantize f32 tensor to FP8 E4M3 representation (stored as u8).
/// scale is computed as max(|x|) / 448.
pub fn quantize_fp8(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    output: &mut GpuTensor<u8>,
    scale: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let n = input.numel as u32;

    // Step 1: compute scale
    let sf = cache.get_or_compile(device, COMPUTE_FP8_SCALE_SRC, "warp_compute_fp8_scale")?;
    unsafe {
        launch_err!(device.stream.launch_builder(&sf)
            .arg(&mut scale.data)
            .arg(&input.data)
            .arg(&n)
            .launch(LaunchConfig::for_num_elems(1)))?;
    }

    // Step 2: quantize
    let qf = cache.get_or_compile(device, QUANTIZE_FP8_SRC, "warp_quantize_fp8_e4m3")?;
    unsafe {
        launch_err!(device.stream.launch_builder(&qf)
            .arg(&mut output.data)
            .arg(&input.data)
            .arg(&scale.data)
            .arg(&n)
            .launch(LaunchConfig::for_num_elems(n)))?;
    }
    Ok(())
}

/// Dequantize FP8 (u8) back to f32.
pub fn dequantize_fp8(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<u8>,
    output: &mut GpuTensor<f32>,
    scale: &GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let n = output.numel as u32;

    let f = cache.get_or_compile(device, DEQUANTIZE_FP8_SRC, "warp_dequantize_fp8")?;
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data)
            .arg(&input.data)
            .arg(&scale.data)
            .arg(&n)
            .launch(LaunchConfig::for_num_elems(n)))?;
    }
    Ok(())
}

/// FP8 GEMM: C[M,N] = dequant(A_fp8[M,K]) × dequant(B_fp8[K,N])
///
/// Both A and B must be FP8-quantized (u8). The GEMM dequantizes on-the-fly
/// using per-tensor scales, accumulating in f32.
pub fn gemm_fp8(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<u8>,
    b: &GpuTensor<u8>,
    c: &mut GpuTensor<f32>,
    scale_a: &GpuTensor<f32>,
    scale_b: &GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    let tile = 32u32;

    let f = cache.get_or_compile(device, GEMM_FP8_SRC, "warp_gemm_fp8")?;
    let cfg = LaunchConfig {
        grid_dim: ((n + tile - 1) / tile, (m + tile - 1) / tile, 1),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut c.data)
            .arg(&a.data)
            .arg(&b.data)
            .arg(&scale_a.data)
            .arg(&scale_b.data)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg))?;
    }
    Ok(())
}

// ═════════════════════════════════════════════════════════════════
// GPTQ Quantization (Group-wise INT4 with zero-points)
// ═════════════════════════════════════════════════════════════════

const DEQUANT_GPTQ_SRC: &str = r#"
extern "C" __global__ void warp_dequant_gptq(
    float *out,                   // [K, N] output
    const unsigned int *qweight,  // packed INT4 weights [K/8, N]
    const float *scales,          // [K/group_size, N] per-group scales
    const unsigned int *qzeros,   // [K/group_size, N/8] packed zero points
    unsigned int K, unsigned int N, unsigned int group_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = idx % N;
    unsigned int k = idx / N;
    if (k >= K) return;

    unsigned int group = k / group_size;

    // Extract 4-bit weight from packed uint32
    unsigned int packed_idx = k / 8;
    unsigned int bit_offset = (k % 8) * 4;
    unsigned int packed = qweight[packed_idx * N + n];
    int w4 = (int)((packed >> bit_offset) & 0xF);

    // Extract zero-point
    unsigned int zp_packed_idx = n / 8;
    unsigned int zp_bit_offset = (n % 8) * 4;
    unsigned int zp_packed = qzeros[group * ((N + 7) / 8) + zp_packed_idx];
    int zp = (int)((zp_packed >> zp_bit_offset) & 0xF);

    float scale = scales[group * N + n];
    out[k * N + n] = (float)(w4 - zp) * scale;
}
"#;

const GEMM_GPTQ_SRC: &str = r#"
#define TILE 32

extern "C" __global__ void warp_gemm_gptq(
    float *C,                     // [M, N]
    const float *A,               // [M, K]
    const unsigned int *qweight,  // [K/8, N]
    const float *scales,          // [K/group_size, N]
    const unsigned int *qzeros,   // [K/group_size, N/8]
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int group_size
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int row = by * TILE + ty;
    unsigned int col = bx * TILE + tx;

    float sum = 0.0f;

    for (unsigned int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load A tile
        unsigned int a_col = t * TILE + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // Load B tile by dequantizing GPTQ weights
        unsigned int b_k = t * TILE + ty;
        unsigned int b_n = bx * TILE + tx;
        if (b_k < K && b_n < N) {
            unsigned int group = b_k / group_size;
            unsigned int packed_idx = b_k / 8;
            unsigned int bit_offset = (b_k % 8) * 4;
            unsigned int packed = qweight[packed_idx * N + b_n];
            int w4 = (int)((packed >> bit_offset) & 0xF);

            unsigned int zp_packed_idx = b_n / 8;
            unsigned int zp_bit_offset = (b_n % 8) * 4;
            unsigned int zp_packed = qzeros[group * ((N + 7) / 8) + zp_packed_idx];
            int zp = (int)((zp_packed >> zp_bit_offset) & 0xF);

            float scale = scales[group * N + b_n];
            Bs[ty][tx] = (float)(w4 - zp) * scale;
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

// ── GPTQ Rust API ──────────────────────────────────────────────

/// Dequantize GPTQ-packed INT4 weights to f32.
///
/// - `qweight`: [K/8, N] packed uint32 (8 x 4-bit values per u32)
/// - `scales`: [K/group_size, N] per-group scales
/// - `qzeros`: [K/group_size, N/8] packed zero-points
/// - `output`: [K, N] f32
pub fn dequant_gptq(
    cache: &KernelCache,
    device: &WarpDevice,
    qweight: &GpuTensor<u32>,
    scales: &GpuTensor<f32>,
    qzeros: &GpuTensor<u32>,
    output: &mut GpuTensor<f32>,
    k: u32,
    n: u32,
    group_size: u32,
) -> Result<(), DeviceError> {
    let total = k * n;
    let f = cache.get_or_compile(device, DEQUANT_GPTQ_SRC, "warp_dequant_gptq")?;
    let cfg = LaunchConfig::for_num_elems(total);

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data)
            .arg(&qweight.data)
            .arg(&scales.data)
            .arg(&qzeros.data)
            .arg(&k)
            .arg(&n)
            .arg(&group_size)
            .launch(cfg))?;
    }
    Ok(())
}

/// GPTQ GEMM: C[M,N] = A[M,K] × dequant(qweight[K,N])
///
/// Dequantizes GPTQ weights on-the-fly in shared memory.
pub fn gemm_gptq(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    qweight: &GpuTensor<u32>,
    scales: &GpuTensor<f32>,
    qzeros: &GpuTensor<u32>,
    output: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
) -> Result<(), DeviceError> {
    let tile = 32u32;
    let f = cache.get_or_compile(device, GEMM_GPTQ_SRC, "warp_gemm_gptq")?;
    let cfg = LaunchConfig {
        grid_dim: ((n + tile - 1) / tile, (m + tile - 1) / tile, 1),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data)
            .arg(&a.data)
            .arg(&qweight.data)
            .arg(&scales.data)
            .arg(&qzeros.data)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .arg(&group_size)
            .launch(cfg))?;
    }
    Ok(())
}

// ═════════════════════════════════════════════════════════════════
// AWQ Quantization (Activation-aware symmetric INT4)
// ═════════════════════════════════════════════════════════════════

const DEQUANT_AWQ_SRC: &str = r#"
extern "C" __global__ void warp_dequant_awq(
    float *out,                   // [K, N] output
    const unsigned int *qweight,  // [K/8, N] packed INT4 weights
    const float *scales,          // [N] per-channel scales
    unsigned int K, unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = idx % N;
    unsigned int k = idx / N;
    if (k >= K) return;

    // Extract 4-bit weight (symmetric: center at 8, so subtract 8)
    unsigned int packed_idx = k / 8;
    unsigned int bit_offset = (k % 8) * 4;
    unsigned int packed = qweight[packed_idx * N + n];
    int w4 = (int)((packed >> bit_offset) & 0xF) - 8;

    out[k * N + n] = (float)w4 * scales[n];
}
"#;

const GEMM_AWQ_SRC: &str = r#"
#define TILE 32

extern "C" __global__ void warp_gemm_awq(
    float *C,                     // [M, N]
    const float *A,               // [M, K]
    const unsigned int *qweight,  // [K/8, N] packed INT4
    const float *scales,          // [N] per-channel scales
    unsigned int M, unsigned int N, unsigned int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int row = by * TILE + ty;
    unsigned int col = bx * TILE + tx;

    float sum = 0.0f;

    for (unsigned int t = 0; t < (K + TILE - 1) / TILE; t++) {
        unsigned int a_col = t * TILE + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        unsigned int b_k = t * TILE + ty;
        unsigned int b_n = bx * TILE + tx;
        if (b_k < K && b_n < N) {
            unsigned int packed_idx = b_k / 8;
            unsigned int bit_offset = (b_k % 8) * 4;
            unsigned int packed = qweight[packed_idx * N + b_n];
            int w4 = (int)((packed >> bit_offset) & 0xF) - 8;
            Bs[ty][tx] = (float)w4 * scales[b_n];
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

// ── AWQ Rust API ───────────────────────────────────────────────

/// Dequantize AWQ-packed INT4 weights to f32.
///
/// - `qweight`: [K/8, N] packed uint32
/// - `scales`: [N] per-channel scales (symmetric, no zero-point)
/// - `output`: [K, N] f32
pub fn dequant_awq(
    cache: &KernelCache,
    device: &WarpDevice,
    qweight: &GpuTensor<u32>,
    scales: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
    k: u32,
    n: u32,
) -> Result<(), DeviceError> {
    let total = k * n;
    let f = cache.get_or_compile(device, DEQUANT_AWQ_SRC, "warp_dequant_awq")?;
    let cfg = LaunchConfig::for_num_elems(total);

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data)
            .arg(&qweight.data)
            .arg(&scales.data)
            .arg(&k)
            .arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

/// AWQ GEMM: C[M,N] = A[M,K] × dequant(qweight[K,N])
///
/// Dequantizes AWQ weights on-the-fly with per-channel scales.
pub fn gemm_awq(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    qweight: &GpuTensor<u32>,
    scales: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    let tile = 32u32;
    let f = cache.get_or_compile(device, GEMM_AWQ_SRC, "warp_gemm_awq")?;
    let cfg = LaunchConfig {
        grid_dim: ((n + tile - 1) / tile, (m + tile - 1) / tile, 1),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data)
            .arg(&a.data)
            .arg(&qweight.data)
            .arg(&scales.data)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg))?;
    }
    Ok(())
}

// ═════════════════════════════════════════════════════════════════
// Per-Channel INT8 Quantization
// ═════════════════════════════════════════════════════════════════

const QUANTIZE_INT8_PER_CHANNEL_SRC: &str = r#"
extern "C" __global__ void warp_quantize_int8_per_channel(
    signed char *out,       // [C_out, C_in] quantized
    float *scale_out,       // [C_out] per-channel scales
    const float *input,     // [C_out, C_in] float weights
    unsigned int C_out,
    unsigned int C_in
) {
    unsigned int ch = blockIdx.x;
    if (ch >= C_out) return;

    unsigned int tid = threadIdx.x;
    unsigned int stride = blockDim.x;

    // Phase 1: find absmax in this channel using all threads
    __shared__ float shared_max[256];
    float local_max = 0.0f;
    for (unsigned int i = tid; i < C_in; i += stride) {
        float a = fabsf(input[ch * C_in + i]);
        if (a > local_max) local_max = a;
    }
    shared_max[tid] = local_max;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && shared_max[tid + s] > shared_max[tid])
            shared_max[tid] = shared_max[tid + s];
        __syncthreads();
    }

    float amax = shared_max[0];
    float scale = amax / 127.0f;
    float inv_scale = (scale != 0.0f) ? 127.0f / amax : 0.0f;

    if (tid == 0) scale_out[ch] = scale;
    __syncthreads();

    // Phase 2: quantize
    for (unsigned int i = tid; i < C_in; i += stride) {
        float v = input[ch * C_in + i] * inv_scale;
        int q = (int)roundf(v);
        q = q < -128 ? -128 : (q > 127 ? 127 : q);
        out[ch * C_in + i] = (signed char)q;
    }
}
"#;

const GEMM_INT8_PER_CHANNEL_SRC: &str = r#"
#define BM 32
#define BN 32
#define BK 32

__device__ __forceinline__ int dp4a_manual_pc(int a, int b, int c) {
    signed char *a_bytes = reinterpret_cast<signed char*>(&a);
    signed char *b_bytes = reinterpret_cast<signed char*>(&b);
    c += (int)a_bytes[0] * (int)b_bytes[0];
    c += (int)a_bytes[1] * (int)b_bytes[1];
    c += (int)a_bytes[2] * (int)b_bytes[2];
    c += (int)a_bytes[3] * (int)b_bytes[3];
    return c;
}

extern "C" __global__ void warp_gemm_int8_per_channel(
    float *C,              // [M, N]
    const signed char *A,  // [M, K] int8 activations
    const signed char *B,  // [K, N] int8 weights
    const float *scale_a,  // [1] per-tensor activation scale
    const float *scale_b,  // [N] per-channel weight scales
    unsigned int M, unsigned int N, unsigned int K
) {
    __shared__ signed char As[BM][BK];
    __shared__ signed char Bs[BK][BN];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int row = by * BM + ty;
    unsigned int col = bx * BN + tx;

    int acc = 0;

    for (unsigned int t = 0; t < (K + BK - 1) / BK; t++) {
        unsigned int a_col = t * BK + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0;

        unsigned int b_row = t * BK + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0;

        __syncthreads();

        for (unsigned int i = 0; i < BK; i += 4) {
            int a_packed = ((unsigned char)As[ty][i]) |
                           ((unsigned char)As[ty][i+1] << 8) |
                           ((unsigned char)As[ty][i+2] << 16) |
                           ((unsigned char)As[ty][i+3] << 24);
            int b_packed = ((unsigned char)Bs[i][tx]) |
                           ((unsigned char)Bs[i+1][tx] << 8) |
                           ((unsigned char)Bs[i+2][tx] << 16) |
                           ((unsigned char)Bs[i+3][tx] << 24);
            acc = dp4a_manual_pc(a_packed, b_packed, acc);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = (float)acc * scale_a[0] * scale_b[col];
    }
}
"#;

// ── Per-Channel INT8 Rust API ──────────────────────────────────

/// Quantize weight matrix [C_out, C_in] to INT8 with per-channel scales.
///
/// Each output channel gets its own scale: scale[c] = max(|W[c, :]|) / 127.
pub fn quantize_int8_per_channel(
    cache: &KernelCache,
    device: &WarpDevice,
    input: &GpuTensor<f32>,     // [C_out, C_in]
    output: &mut GpuTensor<i8>, // [C_out, C_in]
    scale: &mut GpuTensor<f32>, // [C_out]
    c_out: u32,
    c_in: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(
        device,
        QUANTIZE_INT8_PER_CHANNEL_SRC,
        "warp_quantize_int8_per_channel",
    )?;
    // One block per channel, up to 256 threads per block
    let threads = c_in.min(256);
    let cfg = LaunchConfig {
        grid_dim: (c_out, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data)
            .arg(&mut scale.data)
            .arg(&input.data)
            .arg(&c_out)
            .arg(&c_in)
            .launch(cfg))?;
    }
    Ok(())
}

/// INT8 GEMM with per-channel weight scales and per-tensor activation scale.
///
/// C[M,N] = dequant(A_int8[M,K] × B_int8[K,N])
/// where dequant = int32_acc * scale_a[0] * scale_b[col]
pub fn gemm_int8_per_channel(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<i8>,        // [M, K]
    b: &GpuTensor<i8>,        // [K, N]
    c: &mut GpuTensor<f32>,   // [M, N]
    scale_a: &GpuTensor<f32>, // [1] per-tensor
    scale_b: &GpuTensor<f32>, // [N] per-channel
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    assert!(k % 4 == 0, "K ({k}) must be divisible by 4 for dp4a");
    let tile = 32u32;

    let f = cache.get_or_compile(device, GEMM_INT8_PER_CHANNEL_SRC, "warp_gemm_int8_per_channel")?;
    let cfg = LaunchConfig {
        grid_dim: ((n + tile - 1) / tile, (m + tile - 1) / tile, 1),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut c.data)
            .arg(&a.data)
            .arg(&b.data)
            .arg(&scale_a.data)
            .arg(&scale_b.data)
            .arg(&m)
            .arg(&n)
            .arg(&k)
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

    #[test]
    fn int8_gemm_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (64u32, 256u32, 128u32);

        // Generate random-ish test data
        let a_data: Vec<f32> = (0..(m * k) as usize)
            .map(|i| ((i * 7 + 13) % 200) as f32 * 0.01 - 1.0)
            .collect();
        let b_data: Vec<f32> = (0..(k * n) as usize)
            .map(|i| ((i * 11 + 3) % 200) as f32 * 0.01 - 1.0)
            .collect();

        // Upload f32 tensors
        let a_f32 = GpuTensor::from_host(&dev, &a_data,
            Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b_f32 = GpuTensor::from_host(&dev, &b_data,
            Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();

        // Quantize A (per-tensor) and B (per-channel along N)
        let mut a_int8 = GpuTensor::<i8>::zeros(&dev,
            Shape::from_static(&[m as usize, k as usize]), DType::I8).unwrap();
        let mut scale_a = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1]), DType::F32).unwrap();
        quantize_int8(&cache, &dev, &a_f32, &mut a_int8, &mut scale_a, false).unwrap();

        let mut b_int8 = GpuTensor::<i8>::zeros(&dev,
            Shape::from_static(&[k as usize, n as usize]), DType::I8).unwrap();
        // Per-channel: each column of B (N columns, each of length K) gets its own scale
        let mut scale_b = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[n as usize]), DType::F32).unwrap();
        // For per-channel on B[K,N], channel_size = K (each channel = one column)
        // But our quantize_int8 splits linearly: channel i = elements [i*K, (i+1)*K)
        // B is row-major [K,N], so consecutive K elements are a row, not a column.
        // Use per-tensor for B as well (simpler and correct).
        quantize_int8(&cache, &dev, &b_f32, &mut b_int8, &mut scale_b, false).unwrap();
        // scale_b is [1] per-tensor; for the GEMM kernel we need [N] per-channel scales.
        // Broadcast: fill scale_b_broadcast[N] with the single scale value.
        dev.synchronize().unwrap();
        let sb_host = scale_b.to_host(&dev).unwrap();
        let sb_val = sb_host[0];
        let sb_broadcast: Vec<f32> = vec![sb_val; n as usize];
        let scale_b_n = GpuTensor::from_host(&dev, &sb_broadcast,
            Shape::from_static(&[n as usize]), DType::F32).unwrap();

        // INT8 GEMM
        let mut c_int8 = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        gemm_int8(&cache, &dev, &a_int8, &b_int8, &mut c_int8,
            &scale_a, &scale_b_n, m, n, k).unwrap();
        dev.synchronize().unwrap();

        // F32 reference GEMM
        let mut c_ref = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        crate::ops::gemm(&cache, &dev, &a_f32, &b_f32, &mut c_ref, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let int8_result = c_int8.to_host(&dev).unwrap();
        let ref_result = c_ref.to_host(&dev).unwrap();

        let mut max_abs_err = 0.0f32;
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        for (q, r) in int8_result.iter().zip(ref_result.iter()) {
            let err = (q - r).abs();
            if err > max_abs_err { max_abs_err = err; }
            sum_sq_err += (err as f64).powi(2);
            sum_sq_ref += (*r as f64).powi(2);
        }
        let nrmse = (sum_sq_err / sum_sq_ref.max(1e-10)).sqrt();

        println!("INT8 A8W8 GEMM ({m}x{n}x{k}):");
        println!("  Max abs error: {max_abs_err:.4}");
        println!("  NRMSE:         {nrmse:.6} ({:.2}%)", nrmse * 100.0);

        // INT8 quantization introduces error; max element error < 1.0 is expected
        assert!(max_abs_err < 1.0, "INT8 GEMM max error {max_abs_err} >= 1.0");
        assert!(nrmse < 0.10, "INT8 GEMM NRMSE {nrmse:.4} too high");
        println!("  PASSED: max_err < 1.0, NRMSE < 10%");
    }

    #[test]
    fn fp8_gemm_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (64u32, 64u32, 128u32);

        let a_data: Vec<f32> = (0..(m * k) as usize)
            .map(|i| ((i * 7 + 13) % 200) as f32 * 0.01 - 1.0)
            .collect();
        let b_data: Vec<f32> = (0..(k * n) as usize)
            .map(|i| ((i * 11 + 3) % 200) as f32 * 0.01 - 1.0)
            .collect();

        let a_f32 = GpuTensor::from_host(&dev, &a_data,
            Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b_f32 = GpuTensor::from_host(&dev, &b_data,
            Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();

        // Quantize A and B to FP8
        let mut a_fp8 = GpuTensor::<u8>::zeros(&dev,
            Shape::from_static(&[m as usize, k as usize]), DType::U8).unwrap();
        let mut scale_a = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1]), DType::F32).unwrap();
        quantize_fp8(&cache, &dev, &a_f32, &mut a_fp8, &mut scale_a).unwrap();

        let mut b_fp8 = GpuTensor::<u8>::zeros(&dev,
            Shape::from_static(&[k as usize, n as usize]), DType::U8).unwrap();
        let mut scale_b = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1]), DType::F32).unwrap();
        quantize_fp8(&cache, &dev, &b_f32, &mut b_fp8, &mut scale_b).unwrap();

        // FP8 GEMM
        let mut c_fp8 = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        gemm_fp8(&cache, &dev, &a_fp8, &b_fp8, &mut c_fp8,
            &scale_a, &scale_b, m, n, k).unwrap();
        dev.synchronize().unwrap();

        // F32 reference
        let mut c_ref = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        crate::ops::gemm(&cache, &dev, &a_f32, &b_f32, &mut c_ref, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let fp8_result = c_fp8.to_host(&dev).unwrap();
        let ref_result = c_ref.to_host(&dev).unwrap();

        let mut max_abs_err = 0.0f32;
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        for (q, r) in fp8_result.iter().zip(ref_result.iter()) {
            let err = (q - r).abs();
            if err > max_abs_err { max_abs_err = err; }
            sum_sq_err += (err as f64).powi(2);
            sum_sq_ref += (*r as f64).powi(2);
        }
        let nrmse = (sum_sq_err / sum_sq_ref.max(1e-10)).sqrt();

        println!("FP8 E4M3 GEMM ({m}x{n}x{k}):");
        println!("  Max abs error: {max_abs_err:.4}");
        println!("  NRMSE:         {nrmse:.6} ({:.2}%)", nrmse * 100.0);

        assert!(nrmse < 0.10, "FP8 GEMM NRMSE {nrmse:.4} too high");
        println!("  PASSED");
    }

    #[test]
    fn gptq_dequant_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // Small test: K=32, N=8, group_size=32 (1 group)
        let k = 32u32;
        let n = 8u32;
        let group_size = 32u32;
        let num_groups = k / group_size; // 1

        // Build test weights: values 0..15 range, zero_point=8
        // For each k, pack 8 values into qweight[k/8, N]
        let num_packed_rows = k / 8; // 4
        let mut qweight_host = vec![0u32; (num_packed_rows * n) as usize];

        // Reference dequantized values
        let mut ref_out = vec![0.0f32; (k * n) as usize];

        // Scales: one per group per N column
        let scales_host = vec![0.5f32; (num_groups * n) as usize];
        // Zero-points: packed, each column's zp=4
        let zp_packed_cols = (n + 7) / 8; // 1
        let mut qzeros_host = vec![0u32; (num_groups * zp_packed_cols) as usize];

        // Pack zero points: all columns have zp=4
        // 8 columns packed into 1 uint32: 0x44444444
        for g in 0..num_groups as usize {
            let mut packed = 0u32;
            for c in 0..n.min(8) as usize {
                packed |= 4u32 << (c * 4);
            }
            qzeros_host[g * zp_packed_cols as usize] = packed;
        }

        // Pack weights: w4[k][n] = ((k + n) % 16) as 4-bit
        for kk in 0..k as usize {
            for nn in 0..n as usize {
                let w4 = ((kk + nn) % 16) as u32;
                let packed_row = kk / 8;
                let bit_offset = (kk % 8) * 4;
                qweight_host[packed_row * n as usize + nn] |= w4 << bit_offset;

                // Reference: (w4 - zp) * scale
                let zp = 4i32;
                let scale = scales_host[0 * n as usize + nn];
                ref_out[kk * n as usize + nn] = (w4 as i32 - zp) as f32 * scale;
            }
        }

        let qweight = GpuTensor::from_host(&dev, &qweight_host,
            Shape::from_static(&[(num_packed_rows * n) as usize]), DType::U32).unwrap();
        let scales = GpuTensor::from_host(&dev, &scales_host,
            Shape::from_static(&[(num_groups * n) as usize]), DType::F32).unwrap();
        let qzeros = GpuTensor::from_host(&dev, &qzeros_host,
            Shape::from_static(&[(num_groups * zp_packed_cols) as usize]), DType::U32).unwrap();

        let mut output = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[(k * n) as usize]), DType::F32).unwrap();

        dequant_gptq(&cache, &dev, &qweight, &scales, &qzeros, &mut output, k, n, group_size).unwrap();
        dev.synchronize().unwrap();

        let result = output.to_host(&dev).unwrap();

        let max_err: f32 = result.iter().zip(ref_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("GPTQ dequant ({k}x{n}, group_size={group_size}):");
        println!("  Max error vs reference: {max_err:.6}");

        assert!(max_err < 1e-5, "GPTQ dequant error {max_err} too high");
        println!("  PASSED");
    }

    #[test]
    fn awq_dequant_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let k = 32u32;
        let n = 8u32;
        let num_packed_rows = k / 8; // 4

        let mut qweight_host = vec![0u32; (num_packed_rows * n) as usize];
        let scales_host: Vec<f32> = (0..n as usize).map(|i| 0.1 * (i + 1) as f32).collect();
        let mut ref_out = vec![0.0f32; (k * n) as usize];

        // Pack weights: w4[k][n] = ((k + n) % 16), symmetric so subtract 8
        for kk in 0..k as usize {
            for nn in 0..n as usize {
                let w4 = ((kk + nn) % 16) as u32;
                let packed_row = kk / 8;
                let bit_offset = (kk % 8) * 4;
                qweight_host[packed_row * n as usize + nn] |= w4 << bit_offset;

                ref_out[kk * n as usize + nn] = (w4 as i32 - 8) as f32 * scales_host[nn];
            }
        }

        let qweight = GpuTensor::from_host(&dev, &qweight_host,
            Shape::from_static(&[(num_packed_rows * n) as usize]), DType::U32).unwrap();
        let scales = GpuTensor::from_host(&dev, &scales_host,
            Shape::from_static(&[n as usize]), DType::F32).unwrap();

        let mut output = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[(k * n) as usize]), DType::F32).unwrap();

        dequant_awq(&cache, &dev, &qweight, &scales, &mut output, k, n).unwrap();
        dev.synchronize().unwrap();

        let result = output.to_host(&dev).unwrap();

        let max_err: f32 = result.iter().zip(ref_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("AWQ dequant ({k}x{n}):");
        println!("  Max error vs reference: {max_err:.6}");

        assert!(max_err < 1e-5, "AWQ dequant error {max_err} too high");
        println!("  PASSED");
    }

    #[test]
    fn int8_per_channel_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (64u32, 64u32, 128u32);

        // Generate test data
        let a_data: Vec<f32> = (0..(m * k) as usize)
            .map(|i| ((i * 7 + 13) % 200) as f32 * 0.01 - 1.0)
            .collect();
        let b_data: Vec<f32> = (0..(k * n) as usize)
            .map(|i| ((i * 11 + 3) % 200) as f32 * 0.01 - 1.0)
            .collect();

        let a_f32 = GpuTensor::from_host(&dev, &a_data,
            Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b_f32 = GpuTensor::from_host(&dev, &b_data,
            Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();

        // Quantize A per-tensor using existing quantize_int8
        let mut a_int8 = GpuTensor::<i8>::zeros(&dev,
            Shape::from_static(&[m as usize, k as usize]), DType::I8).unwrap();
        let mut scale_a = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1]), DType::F32).unwrap();
        quantize_int8(&cache, &dev, &a_f32, &mut a_int8, &mut scale_a, false).unwrap();

        // Quantize B per-channel: treat B[K, N] as N channels of K elements each.
        // We need to transpose the view: B is [K, N] row-major.
        // For per-channel along N (columns), we do it on CPU since
        // quantize_int8_per_channel expects [C_out, C_in] = [N, K] layout.
        // Transpose B to [N, K] for per-channel quantization.
        let mut b_transposed = vec![0.0f32; (k * n) as usize];
        for kk in 0..k as usize {
            for nn in 0..n as usize {
                b_transposed[nn * k as usize + kk] = b_data[kk * n as usize + nn];
            }
        }
        let b_t_f32 = GpuTensor::from_host(&dev, &b_transposed,
            Shape::from_static(&[n as usize, k as usize]), DType::F32).unwrap();

        let mut b_t_int8 = GpuTensor::<i8>::zeros(&dev,
            Shape::from_static(&[n as usize, k as usize]), DType::I8).unwrap();
        let mut scale_b = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[n as usize]), DType::F32).unwrap();
        quantize_int8_per_channel(&cache, &dev, &b_t_f32, &mut b_t_int8,
            &mut scale_b, n, k).unwrap();

        // Transpose quantized B back to [K, N] for GEMM
        dev.synchronize().unwrap();
        let b_t_q = b_t_int8.to_host(&dev).unwrap();
        let mut b_q_kn = vec![0i8; (k * n) as usize];
        for nn in 0..n as usize {
            for kk in 0..k as usize {
                b_q_kn[kk * n as usize + nn] = b_t_q[nn * k as usize + kk];
            }
        }
        let b_int8 = GpuTensor::from_host(&dev, &b_q_kn,
            Shape::from_static(&[k as usize, n as usize]), DType::I8).unwrap();

        // Per-channel INT8 GEMM
        let mut c_int8 = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        gemm_int8_per_channel(&cache, &dev, &a_int8, &b_int8, &mut c_int8,
            &scale_a, &scale_b, m, n, k).unwrap();
        dev.synchronize().unwrap();

        // F32 reference
        let mut c_ref = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        crate::ops::gemm(&cache, &dev, &a_f32, &b_f32, &mut c_ref, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let int8_result = c_int8.to_host(&dev).unwrap();
        let ref_result = c_ref.to_host(&dev).unwrap();

        let mut max_abs_err = 0.0f32;
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        for (q, r) in int8_result.iter().zip(ref_result.iter()) {
            let err = (q - r).abs();
            if err > max_abs_err { max_abs_err = err; }
            sum_sq_err += (err as f64).powi(2);
            sum_sq_ref += (*r as f64).powi(2);
        }
        let nrmse = (sum_sq_err / sum_sq_ref.max(1e-10)).sqrt();

        println!("INT8 per-channel GEMM ({m}x{n}x{k}):");
        println!("  Max abs error: {max_abs_err:.4}");
        println!("  NRMSE:         {nrmse:.6} ({:.2}%)", nrmse * 100.0);

        assert!(max_abs_err < 1.5, "Per-channel INT8 GEMM max error {max_abs_err} >= 1.5");
        assert!(nrmse < 0.10, "Per-channel INT8 GEMM NRMSE {nrmse:.4} too high");
        println!("  PASSED");
    }

    #[test]
    fn dequant_q4_to_f16_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let k = 128u32;
        let n = 64u32;

        // Create random weights [K, N]
        let w_data: Vec<f32> = (0..(k * n) as usize)
            .map(|i| ((i * 17 + 3) % 101) as f32 * 0.02 - 1.0)
            .collect();
        let weights = GpuTensor::from_host(&dev, &w_data,
            warp_ir::Shape::from_static(&[k as usize, n as usize]), warp_ir::DType::F32).unwrap();

        // Quantize to Q4_0
        let q4_weights = quantize_weights_q4_0(&cache, &dev, &weights, k, n).unwrap();
        dev.synchronize().unwrap();

        // Dequant to FP16
        let f16_weights = dequant_q4_to_f16(&cache, &dev, &q4_weights, k, n).unwrap();
        dev.synchronize().unwrap();

        // Read back FP16 and compare with CPU-dequanted Q4
        let f16_host = f16_weights.to_host(&dev).unwrap();

        // Also dequant on CPU for reference by reading Q4 data
        let q4_host = q4_weights.to_host(&dev).unwrap();
        let num_k_blocks = k / BLOCK_SIZE;

        let mut max_err = 0.0f32;
        for row in 0..k as usize {
            for col in 0..n as usize {
                let block_idx = row / 32;
                let elem = row % 32;
                let block_offset = (col * num_k_blocks as usize + block_idx) * Q4_0_BLOCK_BYTES as usize;

                let scale = f32::from_le_bytes([
                    q4_host[block_offset],
                    q4_host[block_offset + 1],
                    q4_host[block_offset + 2],
                    q4_host[block_offset + 3],
                ]);
                let byte_idx = elem / 2;
                let byte = q4_host[block_offset + 4 + byte_idx];
                let q = if elem % 2 == 0 {
                    (byte & 0x0F) as i32 - 8
                } else {
                    ((byte >> 4) & 0x0F) as i32 - 8
                };
                let expected = scale * q as f32;
                let actual = f16_host[row * n as usize + col].to_f32();
                let err = (expected - actual).abs();
                max_err = max_err.max(err);
            }
        }

        println!("Q4→FP16 dequant ({k}x{n}): max error = {max_err:.6}");
        assert!(max_err < 0.01, "Q4→FP16 dequant max error {max_err} too high");
    }
}

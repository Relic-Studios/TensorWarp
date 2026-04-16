//! Batched MoE expert kernels: run all 8 experts in parallel.
//!
//! Instead of 32 sequential kernel launches per layer (8 experts × 4 ops each),
//! we launch 3 batched kernels that process all experts simultaneously:
//!   1. Batched gate+up GEMM (Q4_K): Grid (ceil(N/256), 8)
//!   2. Batched split_geglu: Grid (ceil(moe_dim/256), 8)
//!   3. Batched down GEMM (Q8_0) + weighted axpy: Grid (ceil(N/256), 8)
//!
//! This increases SM occupancy from 4.7% (6 blocks) to 37%+ (48 blocks).

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;
use crate::quantize::{Q4_K_BLOCK_ELEMS, Q4_K_BLOCK_BYTES, Q8_0_GGUF_BLOCK_ELEMS, Q8_0_GGUF_BLOCK_BYTES};
use cudarc::driver::{LaunchConfig, PushKernelArg};

macro_rules! launch_err {
    ($e:expr) => {
        $e.map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))
    };
}

// ── Batched Q4_K gate+up GEMM ──────────────────────────────────────────────
const BATCHED_Q4K_GEMM_SRC: &str = r#"
__device__ __forceinline__ void get_sm_k4(int j, const unsigned char* s, float* sc, float* mn) {
    if (j < 4) { *sc = (float)(s[j] & 63); *mn = (float)(s[j+4] & 63); }
    else { *sc = (float)((s[j+4]&0xF)|((s[j-4]>>6)<<4)); *mn = (float)((s[j+4]>>4)|((s[j]>>6)<<4)); }
}

extern "C" __global__ void batched_gemm_q4k_m1(
    float* __restrict__ C,           // [top_k, N] output
    const float* __restrict__ A,     // [1, K] input (shared)
    const unsigned char* __restrict__ B_base,
    const float* __restrict__ topk_ids,
    unsigned int bytes_per_expert,
    int K, int N, int num_k_blocks
) {
    int expert_idx = blockIdx.y;
    int eid = (int)topk_ids[expert_idx];
    const unsigned char* B_q4k = B_base + (unsigned long long)eid * bytes_per_expert;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float dot = 0.0f;
    for (int kb = 0; kb < num_k_blocks; kb++) {
        const unsigned char* blk = B_q4k + ((long long)kb * N + n) * 144;
        unsigned short d_bits = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        unsigned short dmin_bits = (unsigned short)blk[2] | ((unsigned short)blk[3] << 8);
        float d, dmin;
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(d) : "h"(d_bits));
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(dmin) : "h"(dmin_bits));
        const unsigned char* scales = blk + 4;
        const unsigned char* qs = blk + 16;
        int k_base = kb * 256; int is = 0, q_off = 0;
        for (int chunk = 0; chunk < 4; chunk++) {
            float sc0, m0, sc1, m1;
            get_sm_k4(is, scales, &sc0, &m0);
            get_sm_k4(is + 1, scales, &sc1, &m1);
            float d1 = d*sc0, dm1 = dmin*m0, d2 = d*sc1, dm2 = dmin*m1;
            const float* a_ptr = A + k_base + chunk * 64;
            for (int l = 0; l < 32; l++) dot += a_ptr[l] * (d1*(float)(qs[q_off+l]&0xF) - dm1);
            for (int l = 0; l < 32; l++) dot += a_ptr[32+l] * (d2*(float)(qs[q_off+l]>>4) - dm2);
            q_off += 32; is += 2;
        }
    }
    C[expert_idx * N + n] = dot;
}
"#;

// ── Batched split_geglu ────────────────────────────────────────────────────
const BATCHED_SPLIT_GEGLU_SRC: &str = r#"
extern "C" __global__ void batched_split_geglu(
    float* __restrict__ out,         // [top_k, moe_dim]
    const float* __restrict__ gate_up, // [top_k, 2*moe_dim]
    int moe_dim
) {
    int expert_idx = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= moe_dim) return;

    const float* gu = gate_up + expert_idx * 2 * moe_dim;
    float g = gu[i];
    float u = gu[moe_dim + i];
    float gelu_g = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
    out[expert_idx * moe_dim + i] = gelu_g * u;
}
"#;

// ── Batched Q8_0 down GEMM + weighted accumulate (fused) ───────────────────
const BATCHED_Q8_DOWN_AXPY_SRC: &str = r#"
extern "C" __global__ void batched_gemm_q8_down_axpy(
    float* __restrict__ acc,         // [1, N] output (atomicAdd from all experts)
    const float* __restrict__ A_all, // [top_k, K] input (each expert has its own input)
    const unsigned char* __restrict__ B_base,
    const float* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    unsigned int bytes_per_expert,
    int K, int N, int num_k_blocks, int A_stride
) {
    int expert_idx = blockIdx.y;
    int eid = (int)topk_ids[expert_idx];
    float weight = topk_weights[expert_idx];
    const unsigned char* B_q8 = B_base + (unsigned long long)eid * bytes_per_expert;
    const float* A = A_all + expert_idx * A_stride;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float dot = 0.0f;
    for (int kb = 0; kb < num_k_blocks; kb++) {
        const unsigned char* blk = B_q8 + ((long long)kb * N + n) * 34;
        unsigned short scale_bits = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        float scale;
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(scale) : "h"(scale_bits));
        const signed char* q = (const signed char*)(blk + 2);
        const float* a_ptr = A + kb * 32;
        float gdot = 0.0f;
        for (int j = 0; j < 32; j++) gdot += a_ptr[j] * (float)q[j];
        dot += gdot * scale;
    }
    atomicAdd(&acc[n], weight * dot);
}
"#;

// ── Batched Q5_0 down GEMM + weighted accumulate (fused) ───────────────────
const BATCHED_Q5_DOWN_AXPY_SRC: &str = r#"
extern "C" __global__ void batched_gemm_q5_down_axpy(
    float* __restrict__ acc,
    const float* __restrict__ A_all,
    const unsigned char* __restrict__ B_base,
    const float* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    unsigned int bytes_per_expert,
    int K, int N, int num_k_blocks, int A_stride
) {
    int expert_idx = blockIdx.y;
    int eid = (int)topk_ids[expert_idx];
    float weight = topk_weights[expert_idx];
    const unsigned char* B_q5 = B_base + (unsigned long long)eid * bytes_per_expert;
    const float* A = A_all + expert_idx * A_stride;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float dot = 0.0f;
    for (int kb = 0; kb < num_k_blocks; kb++) {
        const unsigned char* blk = B_q5 + ((long long)kb * N + n) * 22;
        unsigned short scale_bits = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        float scale;
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(scale) : "h"(scale_bits));
        unsigned int qh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8) |
                          ((unsigned int)blk[4] << 16) | ((unsigned int)blk[5] << 24);
        const unsigned char* qs = blk + 6;
        const float* a_ptr = A + kb * 32;
        float gdot = 0.0f;
        for (int j = 0; j < 16; j++) {
            unsigned int lo = qs[j] & 0xF, hi = qs[j] >> 4;
            unsigned int hb0 = (qh >> (2*j)) & 1, hb1 = (qh >> (2*j+1)) & 1;
            gdot += a_ptr[2*j] * (float)((int)(lo|(hb0<<4)) - 16);
            gdot += a_ptr[2*j+1] * (float)((int)(hi|(hb1<<4)) - 16);
        }
        dot += gdot * scale;
    }
    atomicAdd(&acc[n], weight * dot);
}
"#;

/// Batched Q4_K gate+up GEMM: all `top_k` experts in one launch.
pub fn batched_q4k_gate_up(
    cache: &KernelCache, device: &WarpDevice,
    input: &GpuTensor<f32>,           // [1, K]
    experts_raw: &GpuTensor<u8>,      // all expert weights
    topk_ids: &GpuTensor<f32>,        // [top_k] expert IDs
    output: &mut GpuTensor<f32>,      // [top_k, N]
    bytes_per_expert: u32,
    n: u32, k: u32, top_k: u32,
) -> Result<(), DeviceError> {
    assert!(k % Q4_K_BLOCK_ELEMS == 0);
    let num_k_blocks = k / Q4_K_BLOCK_ELEMS;
    let f = cache.get_or_compile(device, BATCHED_Q4K_GEMM_SRC, "batched_gemm_q4k_m1")?;
    let threads = 256u32;
    let x_blocks = (n + threads - 1) / threads;
    let cfg = LaunchConfig { grid_dim: (x_blocks, top_k, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&input.data).arg(&experts_raw.data)
            .arg(&topk_ids.data).arg(&bytes_per_expert)
            .arg(&(k as i32)).arg(&(n as i32)).arg(&(num_k_blocks as i32))
            .launch(cfg))?;
    }
    Ok(())
}

/// Batched split_geglu: all `top_k` experts in one launch.
pub fn batched_split_geglu(
    cache: &KernelCache, device: &WarpDevice,
    gate_up: &GpuTensor<f32>,    // [top_k, 2*moe_dim]
    output: &mut GpuTensor<f32>, // [top_k, moe_dim]
    moe_dim: u32, top_k: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, BATCHED_SPLIT_GEGLU_SRC, "batched_split_geglu")?;
    let threads = 256u32;
    let x_blocks = (moe_dim + threads - 1) / threads;
    let cfg = LaunchConfig { grid_dim: (x_blocks, top_k, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
    let md = moe_dim as i32;
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&gate_up.data).arg(&md)
            .launch(cfg))?;
    }
    Ok(())
}

/// Batched Q8_0 down GEMM + weighted axpy: all experts fused into one launch.
/// Uses atomicAdd to accumulate weighted outputs from all experts.
pub fn batched_q8_down_axpy(
    cache: &KernelCache, device: &WarpDevice,
    geglu_all: &GpuTensor<f32>,       // [top_k, K_down] input per expert
    experts_raw: &GpuTensor<u8>,      // all expert down weights
    topk_ids: &GpuTensor<f32>,
    topk_weights: &GpuTensor<f32>,
    acc: &mut GpuTensor<f32>,          // [1, N] accumulated output (must be zero-initialized)
    bytes_per_expert: u32,
    n: u32, k: u32, top_k: u32,
) -> Result<(), DeviceError> {
    assert!(k % Q8_0_GGUF_BLOCK_ELEMS == 0);
    let num_k_blocks = k / Q8_0_GGUF_BLOCK_ELEMS;
    let f = cache.get_or_compile(device, BATCHED_Q8_DOWN_AXPY_SRC, "batched_gemm_q8_down_axpy")?;
    let threads = 256u32;
    let x_blocks = (n + threads - 1) / threads;
    let cfg = LaunchConfig { grid_dim: (x_blocks, top_k, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
    let a_stride = k as i32; // stride between expert inputs in geglu_all
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut acc.data).arg(&geglu_all.data).arg(&experts_raw.data)
            .arg(&topk_ids.data).arg(&topk_weights.data).arg(&bytes_per_expert)
            .arg(&(k as i32)).arg(&(n as i32)).arg(&(num_k_blocks as i32)).arg(&a_stride)
            .launch(cfg))?;
    }
    Ok(())
}

/// Batched Q5_0 down GEMM + weighted axpy (same pattern as Q8_0).
pub fn batched_q5_down_axpy(
    cache: &KernelCache, device: &WarpDevice,
    geglu_all: &GpuTensor<f32>,
    experts_raw: &GpuTensor<u8>,
    topk_ids: &GpuTensor<f32>,
    topk_weights: &GpuTensor<f32>,
    acc: &mut GpuTensor<f32>,
    bytes_per_expert: u32,
    n: u32, k: u32, top_k: u32,
) -> Result<(), DeviceError> {
    assert!(k % crate::quantize::Q5_0_BLOCK_ELEMS == 0);
    let num_k_blocks = k / crate::quantize::Q5_0_BLOCK_ELEMS;
    let f = cache.get_or_compile(device, BATCHED_Q5_DOWN_AXPY_SRC, "batched_gemm_q5_down_axpy")?;
    let threads = 256u32;
    let x_blocks = (n + threads - 1) / threads;
    let cfg = LaunchConfig { grid_dim: (x_blocks, top_k, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
    let a_stride = k as i32;
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut acc.data).arg(&geglu_all.data).arg(&experts_raw.data)
            .arg(&topk_ids.data).arg(&topk_weights.data).arg(&bytes_per_expert)
            .arg(&(k as i32)).arg(&(n as i32)).arg(&(num_k_blocks as i32)).arg(&a_stride)
            .launch(cfg))?;
    }
    Ok(())
}

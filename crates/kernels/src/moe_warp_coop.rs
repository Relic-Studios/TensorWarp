//! Warp-cooperative MoE expert GEMV kernels.
//!
//! Each output column is computed by a full warp (32 threads) instead of 1 thread.
//! The activation vector A lives in shared memory to prevent L1 cache thrashing.
//! For batched dispatch, blockIdx.y selects the expert (reads topk_ids from device).
//!
//! Q4_K: 32 threads per column, each handles 4 bytes of qs (8 quantized values) per super-block.
//!        9 super-blocks × 8 values/thread = 72 FMAs per thread (was 2304).
//! Q8_0: 32 threads per column, each handles 1 element per block.
//!        22 blocks × 1 FMA/thread = 22 FMAs per thread (was 704).

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;
use cudarc::driver::{LaunchConfig, PushKernelArg};

macro_rules! launch_err {
    ($e:expr) => {
        $e.map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))
    };
}

// ── Warp-cooperative batched Q4_K gate+up GEMV ─────────────────────────────
const WARP_COOP_Q4K_SRC: &str = r#"
extern "C" __global__ void warp_coop_q4k_batched(
    float* __restrict__ C,                // [top_k, N] output
    const float* __restrict__ A,          // [1, K] input (via __ldg L1 cache)
    const unsigned char* __restrict__ B_base,
    const float* __restrict__ topk_ids,
    unsigned int bytes_per_expert,
    int K, int N, int num_k_blocks
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid & 31;
    int warps_per_block = blockDim.x / 32;
    int n = blockIdx.x * warps_per_block + warp_id;
    if (n >= N) return;

    int expert_idx = blockIdx.y;
    int eid = (int)topk_ids[expert_idx];
    const unsigned char* B_expert = B_base + (unsigned long long)eid * bytes_per_expert;

    float dot = 0.0f;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        const unsigned char* blk = B_expert + ((long long)kb * N + n) * 144;

        // Block header broadcast
        unsigned short d_bits = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        unsigned short dmin_bits = (unsigned short)blk[2] | ((unsigned short)blk[3] << 8);
        float d, dmin;
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(d) : "h"(d_bits));
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(dmin) : "h"(dmin_bits));

        const unsigned char* sc_ptr = blk + 4;
        const unsigned char* qs = blk + 16;

        int chunk = lane >> 3;
        int local_off = (lane & 7) * 4;

        int is_lo = chunk * 2;
        int is_hi = is_lo + 1;
        float sc_lo, m_lo, sc_hi, m_hi;

        if (is_lo < 4) {
            sc_lo = (float)(sc_ptr[is_lo] & 63);
            m_lo  = (float)(sc_ptr[is_lo + 4] & 63);
        } else {
            sc_lo = (float)((sc_ptr[is_lo+4] & 0xF) | ((sc_ptr[is_lo-4] >> 6) << 4));
            m_lo  = (float)((sc_ptr[is_lo+4] >> 4) | ((sc_ptr[is_lo] >> 6) << 4));
        }
        if (is_hi < 4) {
            sc_hi = (float)(sc_ptr[is_hi] & 63);
            m_hi  = (float)(sc_ptr[is_hi + 4] & 63);
        } else {
            sc_hi = (float)((sc_ptr[is_hi+4] & 0xF) | ((sc_ptr[is_hi-4] >> 6) << 4));
            m_hi  = (float)((sc_ptr[is_hi+4] >> 4) | ((sc_ptr[is_hi] >> 6) << 4));
        }

        float d_lo = d * sc_lo, dm_lo = dmin * m_lo;
        float d_hi = d * sc_hi, dm_hi = dmin * m_hi;

        unsigned int packed = *(const unsigned int*)(qs + chunk * 32 + local_off);
        int k_base = kb * 256 + chunk * 64;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned char qb = (packed >> (i * 8)) & 0xFF;
            float w_lo = d_lo * (float)(qb & 0xF) - dm_lo;
            float w_hi = d_hi * (float)(qb >> 4) - dm_hi;
            dot = fmaf(w_lo, __ldg(&A[k_base + local_off + i]), dot);
            dot = fmaf(w_hi, __ldg(&A[k_base + 32 + local_off + i]), dot);
        }
    }

    // Warp-level butterfly reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
    }

    if (lane == 0) {
        C[expert_idx * N + n] = dot;
    }
}
"#;

// ── Warp-cooperative batched Q8_0 down GEMV + weighted axpy ────────────────
const WARP_COOP_Q8_DOWN_AXPY_SRC: &str = r#"
extern "C" __global__ void warp_coop_q8_down_axpy(
    float* __restrict__ acc,              // [1, N] accumulated output (atomicAdd)
    const float* __restrict__ A_all,      // [top_k, K] per-expert inputs
    const unsigned char* __restrict__ B_base,
    const float* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    unsigned int bytes_per_expert,
    int K, int N, int num_k_blocks, int A_stride
) {
    int tid = threadIdx.x;
    int expert_idx = blockIdx.y;
    const float* A = A_all + expert_idx * A_stride;

    int warp_id = tid / 32;
    int lane = tid & 31;
    int warps_per_block = blockDim.x / 32;
    int n = blockIdx.x * warps_per_block + warp_id;
    if (n >= N) return;

    int eid = (int)topk_ids[expert_idx];
    float weight = topk_weights[expert_idx];
    const unsigned char* B_q8 = B_base + (unsigned long long)eid * bytes_per_expert;

    float dot = 0.0f;

    // Each lane processes one element per Q8_0 block, iterating over all blocks
    for (int block_idx = 0; block_idx < num_k_blocks; block_idx++) {
        const unsigned char* blk = B_q8 + ((long long)block_idx * N + n) * 34;
        unsigned short scale_bits = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        float scale;
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(scale) : "h"(scale_bits));

        signed char q = ((const signed char*)(blk + 2))[lane];
        dot = fmaf(scale * (float)q, __ldg(&A[block_idx * 32 + lane]), dot);
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
    }

    if (lane == 0) {
        atomicAdd(&acc[n], weight * dot);
    }
}
"#;

// ── Batched split_geglu (reuse from moe_batched.rs) ────────────────────────
const BATCHED_SPLIT_GEGLU_SRC: &str = r#"
extern "C" __global__ void warp_coop_split_geglu(
    float* __restrict__ out,
    const float* __restrict__ gate_up,
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

/// Warp-cooperative batched Q4_K gate+up GEMV.
/// All top_k experts in parallel, 32 threads per output column, A in shared memory.
pub fn warp_coop_q4k_gate_up(
    cache: &KernelCache, device: &WarpDevice,
    input: &GpuTensor<f32>,           // [1, K]
    experts_raw: &GpuTensor<u8>,
    topk_ids: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,      // [top_k, N]
    bytes_per_expert: u32,
    n: u32, k: u32, top_k: u32,
) -> Result<(), DeviceError> {
    let num_k_blocks = k / 256;  // Q4_K: 256 elements per super-block
    let f = cache.get_or_compile(device, WARP_COOP_Q4K_SRC, "warp_coop_q4k_batched")?;

    // 8 warps per block = 256 threads, 8 output columns per block
    let threads_per_block = 256u32;
    let warps_per_block = threads_per_block / 32;
    let x_blocks = (n + warps_per_block - 1) / warps_per_block;

    let cfg = LaunchConfig {
        grid_dim: (x_blocks, top_k, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&input.data).arg(&experts_raw.data)
            .arg(&topk_ids.data).arg(&bytes_per_expert)
            .arg(&(k as i32)).arg(&(n as i32)).arg(&(num_k_blocks as i32))
            .launch(cfg))?;
    }
    Ok(())
}

/// Warp-cooperative batched Q8_0 down GEMV + weighted axpy (fused).
pub fn warp_coop_q8_down_axpy(
    cache: &KernelCache, device: &WarpDevice,
    geglu_all: &GpuTensor<f32>,       // [top_k, K_down]
    experts_raw: &GpuTensor<u8>,
    topk_ids: &GpuTensor<f32>,
    topk_weights: &GpuTensor<f32>,
    acc: &mut GpuTensor<f32>,          // [1, N] must be zero-initialized
    bytes_per_expert: u32,
    n: u32, k: u32, top_k: u32,
) -> Result<(), DeviceError> {
    let num_k_blocks = k / 32;  // Q8_0: 32 elements per block
    let f = cache.get_or_compile(device, WARP_COOP_Q8_DOWN_AXPY_SRC, "warp_coop_q8_down_axpy")?;

    let threads_per_block = 256u32;
    let warps_per_block = threads_per_block / 32;
    let x_blocks = (n + warps_per_block - 1) / warps_per_block;

    let cfg = LaunchConfig {
        grid_dim: (x_blocks, top_k, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

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

/// Batched split_geglu for warp-cooperative pipeline.
pub fn warp_coop_split_geglu(
    cache: &KernelCache, device: &WarpDevice,
    gate_up: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
    moe_dim: u32, top_k: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, BATCHED_SPLIT_GEGLU_SRC, "warp_coop_split_geglu")?;
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

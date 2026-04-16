//! Batched TW-Marlin expert GEMV kernels for MoE.
//!
//! Processes all 8 experts in parallel using blockIdx.y for expert selection.
//! Reads topk_ids from device memory — no DtoH sync needed.
//! Reduces 32 sequential kernel launches to 3 batched launches per layer.

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;
use crate::quantize::BLOCK_SIZE;
use cudarc::driver::{LaunchConfig, PushKernelArg};

macro_rules! launch_err {
    ($e:expr) => {
        $e.map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))
    };
}

// ── Batched TW-Marlin gate+up GEMV ─────────────────────────────────────────
const BATCHED_MARLIN_GATE_UP_SRC: &str = r#"
extern "C" __global__ void batched_marlin_gate_up(
    float* __restrict__ C,                // [top_k, N] output
    const float* __restrict__ A,          // [1, K] input (shared across experts)
    const unsigned char* __restrict__ packed_base,
    const unsigned short* __restrict__ scales_base,
    const float* __restrict__ topk_ids,
    unsigned int packed_bytes_per_expert,
    unsigned int scales_per_expert,
    int K, int N, int num_k_groups
) {
    int expert_idx = blockIdx.y;
    int eid = (int)topk_ids[expert_idx];
    const unsigned char* packed = packed_base + (unsigned long long)eid * packed_bytes_per_expert;
    const unsigned short* scales = scales_base + (unsigned long long)eid * scales_per_expert;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float dot = 0.0f;

    #define TW_NIB8(V, off) { \
        unsigned int _v = V; \
        gdot += __ldg(a_base + (off))     * (float)((int)(_v & 0xF) - 8); \
        gdot += __ldg(a_base + (off) + 1) * (float)((int)((_v >> 4) & 0xF) - 8); \
        gdot += __ldg(a_base + (off) + 2) * (float)((int)((_v >> 8) & 0xF) - 8); \
        gdot += __ldg(a_base + (off) + 3) * (float)((int)((_v >> 12) & 0xF) - 8); \
        gdot += __ldg(a_base + (off) + 4) * (float)((int)((_v >> 16) & 0xF) - 8); \
        gdot += __ldg(a_base + (off) + 5) * (float)((int)((_v >> 20) & 0xF) - 8); \
        gdot += __ldg(a_base + (off) + 6) * (float)((int)((_v >> 24) & 0xF) - 8); \
        gdot += __ldg(a_base + (off) + 7) * (float)((int)(_v >> 28) - 8); \
    }

    for (int g = 0; g < num_k_groups; g++) {
        unsigned short scale_bits = scales[g * N + n];
        float scale;
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(scale) : "h"(scale_bits));

        const unsigned char* my_packed = packed + ((long long)g * N + n) * 16;
        const unsigned int* p = (const unsigned int*)my_packed;
        unsigned int v0 = p[0], v1 = p[1], v2 = p[2], v3 = p[3];

        const float* a_base = A + g * 32;
        float gdot = 0.0f;
        TW_NIB8(v0, 0);
        TW_NIB8(v1, 8);
        TW_NIB8(v2, 16);
        TW_NIB8(v3, 24);
        dot += gdot * scale;
    }
    #undef TW_NIB8

    C[expert_idx * N + n] = dot;
}
"#;

// ── Batched TW-Marlin down GEMV + weighted axpy (fused) ────────────────────
const BATCHED_MARLIN_DOWN_AXPY_SRC: &str = r#"
extern "C" __global__ void batched_marlin_down_axpy(
    float* __restrict__ acc,              // [1, N] output (atomicAdd)
    const float* __restrict__ A_all,      // [top_k, K] per-expert inputs
    const unsigned char* __restrict__ packed_base,
    const unsigned short* __restrict__ scales_base,
    const float* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    unsigned int packed_bytes_per_expert,
    unsigned int scales_per_expert,
    int K, int N, int num_k_groups, int A_stride
) {
    int expert_idx = blockIdx.y;
    int eid = (int)topk_ids[expert_idx];
    float weight = topk_weights[expert_idx];
    const unsigned char* packed = packed_base + (unsigned long long)eid * packed_bytes_per_expert;
    const unsigned short* scales = scales_base + (unsigned long long)eid * scales_per_expert;
    const float* A = A_all + expert_idx * A_stride;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float dot = 0.0f;

    #define TW_NIB8(V, off) { \
        unsigned int _v = V; \
        gdot += __ldg(a_base + (off))     * (float)((int)(_v & 0xF) - 8); \
        gdot += __ldg(a_base + (off) + 1) * (float)((int)((_v >> 4) & 0xF) - 8); \
        gdot += __ldg(a_base + (off) + 2) * (float)((int)((_v >> 8) & 0xF) - 8); \
        gdot += __ldg(a_base + (off) + 3) * (float)((int)((_v >> 12) & 0xF) - 8); \
        gdot += __ldg(a_base + (off) + 4) * (float)((int)((_v >> 16) & 0xF) - 8); \
        gdot += __ldg(a_base + (off) + 5) * (float)((int)((_v >> 20) & 0xF) - 8); \
        gdot += __ldg(a_base + (off) + 6) * (float)((int)((_v >> 24) & 0xF) - 8); \
        gdot += __ldg(a_base + (off) + 7) * (float)((int)(_v >> 28) - 8); \
    }

    for (int g = 0; g < num_k_groups; g++) {
        unsigned short scale_bits = scales[g * N + n];
        float scale;
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(scale) : "h"(scale_bits));

        const unsigned char* my_packed = packed + ((long long)g * N + n) * 16;
        const unsigned int* p = (const unsigned int*)my_packed;
        unsigned int v0 = p[0], v1 = p[1], v2 = p[2], v3 = p[3];

        const float* a_base = A + g * 32;
        float gdot = 0.0f;
        TW_NIB8(v0, 0);
        TW_NIB8(v1, 8);
        TW_NIB8(v2, 16);
        TW_NIB8(v3, 24);
        dot += gdot * scale;
    }
    #undef TW_NIB8

    atomicAdd(&acc[n], weight * dot);
}
"#;

// ── Batched split_geglu ────────────────────────────────────────────────────
const BATCHED_SPLIT_GEGLU_SRC: &str = r#"
extern "C" __global__ void batched_marlin_split_geglu(
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

/// Batched TW-Marlin gate+up GEMV: all top_k experts in one launch.
pub fn batched_marlin_gate_up(
    cache: &KernelCache, device: &WarpDevice,
    input: &GpuTensor<f32>,
    packed_buf: &GpuTensor<u8>,
    scales_buf: &GpuTensor<half::f16>,
    topk_ids: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
    packed_bytes_per_expert: u32,
    scales_per_expert: u32,
    n: u32, k: u32, top_k: u32,
) -> Result<(), DeviceError> {
    assert!(k % BLOCK_SIZE == 0);
    let num_k_groups = k / BLOCK_SIZE;
    let f = cache.get_or_compile(device, BATCHED_MARLIN_GATE_UP_SRC, "batched_marlin_gate_up")?;
    let threads = 256u32;
    let x_blocks = (n + threads - 1) / threads;
    let cfg = LaunchConfig { grid_dim: (x_blocks, top_k, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&input.data)
            .arg(&packed_buf.data).arg(&scales_buf.data)
            .arg(&topk_ids.data)
            .arg(&packed_bytes_per_expert).arg(&scales_per_expert)
            .arg(&(k as i32)).arg(&(n as i32)).arg(&(num_k_groups as i32))
            .launch(cfg))?;
    }
    Ok(())
}

/// Batched TW-Marlin down GEMV + weighted axpy fused.
pub fn batched_marlin_down_axpy(
    cache: &KernelCache, device: &WarpDevice,
    geglu_all: &GpuTensor<f32>,
    packed_buf: &GpuTensor<u8>,
    scales_buf: &GpuTensor<half::f16>,
    topk_ids: &GpuTensor<f32>,
    topk_weights: &GpuTensor<f32>,
    acc: &mut GpuTensor<f32>,
    packed_bytes_per_expert: u32,
    scales_per_expert: u32,
    n: u32, k: u32, top_k: u32,
) -> Result<(), DeviceError> {
    assert!(k % BLOCK_SIZE == 0);
    let num_k_groups = k / BLOCK_SIZE;
    let f = cache.get_or_compile(device, BATCHED_MARLIN_DOWN_AXPY_SRC, "batched_marlin_down_axpy")?;
    let threads = 256u32;
    let x_blocks = (n + threads - 1) / threads;
    let cfg = LaunchConfig { grid_dim: (x_blocks, top_k, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
    let a_stride = k as i32;
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut acc.data).arg(&geglu_all.data)
            .arg(&packed_buf.data).arg(&scales_buf.data)
            .arg(&topk_ids.data).arg(&topk_weights.data)
            .arg(&packed_bytes_per_expert).arg(&scales_per_expert)
            .arg(&(k as i32)).arg(&(n as i32)).arg(&(num_k_groups as i32)).arg(&a_stride)
            .launch(cfg))?;
    }
    Ok(())
}

/// Batched split_geglu for TW-Marlin pipeline.
pub fn batched_marlin_split_geglu(
    cache: &KernelCache, device: &WarpDevice,
    gate_up: &GpuTensor<f32>,
    output: &mut GpuTensor<f32>,
    moe_dim: u32, top_k: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, BATCHED_SPLIT_GEGLU_SRC, "batched_marlin_split_geglu")?;
    let threads = 256u32;
    let x_blocks = (moe_dim + threads - 1) / threads;
    let cfg = LaunchConfig { grid_dim: (x_blocks, top_k, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut output.data).arg(&gate_up.data).arg(&(moe_dim as i32))
            .launch(cfg))?;
    }
    Ok(())
}

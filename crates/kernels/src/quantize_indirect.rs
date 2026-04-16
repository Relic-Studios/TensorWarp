// Indirect GEMM variants: read byte offset from device memory.
// Eliminates the DtoH sync for router topK expert IDs.
// Each kernel reads its offset from offsets[expert_idx] on the GPU.

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;
use crate::quantize::{Q4_K_BLOCK_ELEMS, Q8_0_GGUF_BLOCK_ELEMS, Q5_0_BLOCK_ELEMS, BLOCK_SIZE};
use cudarc::driver::{LaunchConfig, PushKernelArg};

macro_rules! launch_err {
    ($e:expr) => {
        $e.map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))
    };
}

const GEMM_Q4_K_M1_INDIRECT_SRC: &str = r#"
__device__ __forceinline__ void get_scale_min_k4_i(int j, const unsigned char* scales, float* sc, float* mn) {
    if (j < 4) { *sc = (float)(scales[j] & 63); *mn = (float)(scales[j+4] & 63); }
    else { *sc = (float)((scales[j+4]&0xF)|((scales[j-4]>>6)<<4)); *mn = (float)((scales[j+4]>>4)|((scales[j]>>6)<<4)); }
}
extern "C" __global__ void warp_gemm_q4_k_m1_indirect(
    float* __restrict__ C, const float* __restrict__ A,
    const unsigned char* __restrict__ B_base,
    const unsigned int* __restrict__ offsets,
    int expert_idx,
    int K, int N, int num_k_blocks
) {
    const unsigned char* B_q4k = B_base + offsets[expert_idx];
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float dot = 0.0f;
    for (int kb = 0; kb < num_k_blocks; kb++) {
        const unsigned char* blk = B_q4k + ((long long)n * num_k_blocks + kb) * 144;
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
            get_scale_min_k4_i(is, scales, &sc0, &m0);
            get_scale_min_k4_i(is + 1, scales, &sc1, &m1);
            float d1 = d*sc0, dm1 = dmin*m0, d2 = d*sc1, dm2 = dmin*m1;
            const float* a_ptr = A + k_base + chunk * 64;
            for (int l = 0; l < 32; l++) dot += a_ptr[l] * (d1*(float)(qs[q_off+l]&0xF) - dm1);
            for (int l = 0; l < 32; l++) dot += a_ptr[32+l] * (d2*(float)(qs[q_off+l]>>4) - dm2);
            q_off += 32; is += 2;
        }
    }
    C[n] = dot;
}
"#;

pub fn gemm_q4_k_m1_indirect(
    cache: &KernelCache, device: &WarpDevice,
    a: &GpuTensor<f32>, buffer: &GpuTensor<u8>,
    offsets: &cudarc::driver::CudaSlice<u32>, expert_idx: i32,
    c: &mut GpuTensor<f32>, n: u32, k: u32,
) -> Result<(), DeviceError> {
    assert!(k % Q4_K_BLOCK_ELEMS == 0);
    let num_k_blocks = k / Q4_K_BLOCK_ELEMS;
    let f = cache.get_or_compile(device, GEMM_Q4_K_M1_INDIRECT_SRC, "warp_gemm_q4_k_m1_indirect")?;
    let threads = 256u32;
    let blocks = (n + threads - 1) / threads;
    let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut c.data).arg(&a.data).arg(&buffer.data)
            .arg(offsets).arg(&expert_idx)
            .arg(&(k as i32)).arg(&(n as i32)).arg(&(num_k_blocks as i32))
            .launch(cfg))?;
    }
    Ok(())
}

const GEMM_Q8_0_GGUF_M1_INDIRECT_SRC: &str = r#"
extern "C" __global__ void warp_gemm_q8_0_gguf_m1_indirect(
    float* __restrict__ C, const float* __restrict__ A,
    const unsigned char* __restrict__ B_base,
    const unsigned int* __restrict__ offsets,
    int expert_idx,
    int K, int N, int num_k_blocks
) {
    const unsigned char* B_q8 = B_base + offsets[expert_idx];
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float dot = 0.0f;
    for (int kb = 0; kb < num_k_blocks; kb++) {
        const unsigned char* blk = B_q8 + ((long long)n * num_k_blocks + kb) * 34;
        unsigned short scale_bits = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        float scale;
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(scale) : "h"(scale_bits));
        const signed char* q = (const signed char*)(blk + 2);
        const float* a_ptr = A + kb * 32;
        float gdot = 0.0f;
        for (int j = 0; j < 32; j++) gdot += a_ptr[j] * (float)q[j];
        dot += gdot * scale;
    }
    C[n] = dot;
}
"#;

pub fn gemm_q8_0_gguf_m1_indirect(
    cache: &KernelCache, device: &WarpDevice,
    a: &GpuTensor<f32>, buffer: &GpuTensor<u8>,
    offsets: &cudarc::driver::CudaSlice<u32>, expert_idx: i32,
    c: &mut GpuTensor<f32>, n: u32, k: u32,
) -> Result<(), DeviceError> {
    assert!(k % Q8_0_GGUF_BLOCK_ELEMS == 0);
    let num_k_blocks = k / Q8_0_GGUF_BLOCK_ELEMS;
    let f = cache.get_or_compile(device, GEMM_Q8_0_GGUF_M1_INDIRECT_SRC, "warp_gemm_q8_0_gguf_m1_indirect")?;
    let threads = 256u32;
    let blocks = (n + threads - 1) / threads;
    let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut c.data).arg(&a.data).arg(&buffer.data)
            .arg(offsets).arg(&expert_idx)
            .arg(&(k as i32)).arg(&(n as i32)).arg(&(num_k_blocks as i32))
            .launch(cfg))?;
    }
    Ok(())
}

const GEMM_Q5_0_M1_INDIRECT_SRC: &str = r#"
extern "C" __global__ void warp_gemm_q5_0_m1_indirect(
    float* __restrict__ C, const float* __restrict__ A,
    const unsigned char* __restrict__ B_base,
    const unsigned int* __restrict__ offsets,
    int expert_idx,
    int K, int N, int num_k_blocks
) {
    const unsigned char* B_q5 = B_base + offsets[expert_idx];
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float dot = 0.0f;
    for (int kb = 0; kb < num_k_blocks; kb++) {
        const unsigned char* blk = B_q5 + ((long long)n * num_k_blocks + kb) * 22;
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
    C[n] = dot;
}
"#;

pub fn gemm_q5_0_m1_indirect(
    cache: &KernelCache, device: &WarpDevice,
    a: &GpuTensor<f32>, buffer: &GpuTensor<u8>,
    offsets: &cudarc::driver::CudaSlice<u32>, expert_idx: i32,
    c: &mut GpuTensor<f32>, n: u32, k: u32,
) -> Result<(), DeviceError> {
    assert!(k % Q5_0_BLOCK_ELEMS == 0);
    let num_k_blocks = k / Q5_0_BLOCK_ELEMS;
    let f = cache.get_or_compile(device, GEMM_Q5_0_M1_INDIRECT_SRC, "warp_gemm_q5_0_m1_indirect")?;
    let threads = 256u32;
    let blocks = (n + threads - 1) / threads;
    let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut c.data).arg(&a.data).arg(&buffer.data)
            .arg(offsets).arg(&expert_idx)
            .arg(&(k as i32)).arg(&(n as i32)).arg(&(num_k_blocks as i32))
            .launch(cfg))?;
    }
    Ok(())
}

const GEMM_TW_MARLIN_M1_INDIRECT_SRC: &str = r#"
extern "C" __global__ void warp_gemm_tw_marlin_m1_direct_indirect(
    float* __restrict__ C, const float* __restrict__ A,
    const unsigned char* __restrict__ packed_base,
    const unsigned short* __restrict__ scales_base,
    const unsigned int* __restrict__ packed_offsets,
    const unsigned int* __restrict__ scales_offsets,
    int expert_idx,
    int K, int N, int num_k_groups, int k_blocks_per_split
) {
    const unsigned char* packed = packed_base + packed_offsets[expert_idx];
    const unsigned short* scales = scales_base + scales_offsets[expert_idx];
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
        TW_NIB8(v0, 0); TW_NIB8(v1, 8); TW_NIB8(v2, 16); TW_NIB8(v3, 24);
        dot += gdot * scale;
    }
    #undef TW_NIB8
    C[n] = dot;
}
"#;

pub fn gemm_tw_marlin_m1_indirect(
    cache: &KernelCache, device: &WarpDevice,
    a: &GpuTensor<f32>,
    packed_buf: &GpuTensor<u8>,
    packed_offsets: &cudarc::driver::CudaSlice<u32>,
    scales_buf: &GpuTensor<half::f16>,
    scales_offsets: &cudarc::driver::CudaSlice<u32>,
    expert_idx: i32,
    c: &mut GpuTensor<f32>, n: u32, k: u32,
) -> Result<(), DeviceError> {
    assert!(k % BLOCK_SIZE == 0);
    let num_k_groups = k / BLOCK_SIZE;
    let threads = 256u32;
    let n_blocks = (n + threads - 1) / threads;
    let f = cache.get_or_compile(device, GEMM_TW_MARLIN_M1_INDIRECT_SRC, "warp_gemm_tw_marlin_m1_direct_indirect")?;
    let cfg = LaunchConfig { grid_dim: (n_blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut c.data).arg(&a.data)
            .arg(&packed_buf.data).arg(&scales_buf.data)
            .arg(packed_offsets).arg(scales_offsets).arg(&expert_idx)
            .arg(&(k as i32)).arg(&(n as i32)).arg(&(num_k_groups as i32)).arg(&(num_k_groups as i32))
            .launch(cfg))?;
    }
    Ok(())
}

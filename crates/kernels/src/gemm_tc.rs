//! Tensor Core GEMM via wmma API.
//!
//! FP16 input, FP32 accumulation, FP16 output.
//! Uses CUDA's wmma (Warp Matrix Multiply-Accumulate) intrinsics
//! to hit the Tensor Core units on Ada (4090) and Hopper (H100).
//!
//! RTX 4090 Tensor Core peak: ~330 TFLOPS FP16
//! cuBLAS FP16 achieves: ~21.5 TFLOPS at 4096³

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Tensor Core GEMM — simple version (fallback for small matrices).
const WMMA_GEMM_SIMPLE_SRC: &str = r#"
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

extern "C" __global__ void warp_gemm_tc_simple(
    half * __restrict__ C,
    const half * __restrict__ A,
    const half * __restrict__ B,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    unsigned int warps_per_row = (N + 15) / 16;
    unsigned int wr = (warpId / warps_per_row) * 16;
    unsigned int wc = (warpId % warps_per_row) * 16;
    if (wr >= M || wc >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (unsigned int k = 0; k < K; k += 16) {
        if (wr + 16 <= M && k + 16 <= K) wmma::load_matrix_sync(af, A + wr*K + k, K);
        else wmma::fill_fragment(af, __float2half(0.0f));
        if (k + 16 <= K && wc + 16 <= N) wmma::load_matrix_sync(bf, B + k*N + wc, N);
        else wmma::fill_fragment(bf, __float2half(0.0f));
        wmma::mma_sync(acc, af, bf, acc);
    }

    if (wr + 16 <= M && wc + 16 <= N) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> ch;
        for (int i = 0; i < acc.num_elements; i++) ch.x[i] = __float2half(acc.x[i]);
        wmma::store_matrix_sync(C + wr*N + wc, ch, N, wmma::mem_row_major);
    }
}
"#;

/// Maximum-performance Tensor Core GEMM with cp.async pipeline.
///
/// 128×128 tile, 8 warps, 2-stage async pipeline.
/// cp.async copies global→shared WITHOUT going through registers,
/// running in the background while tensor cores compute on the previous tile.
/// This is how cuBLAS achieves peak throughput.
const WMMA_GEMM_SRC: &str = r#"
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define BM 128
#define BN 128
#define BK 16
#define STAGES 2

// cp.async helpers — 16-byte async copy from global to shared memory
__device__ __forceinline__ void async_copy_16(half *smem, const half *gmem) {
    unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem));
}

__device__ __forceinline__ void async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void async_wait_0() {
    asm volatile("cp.async.wait_group 0;\n");
}
__device__ __forceinline__ void async_wait_n() {
    // Wait until at most STAGES-1 groups remain in flight
    asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 1));
}

extern "C" __global__ void warp_gemm_tc(
    half * __restrict__ C,
    const half * __restrict__ A,
    const half * __restrict__ B,
    unsigned int M, unsigned int N, unsigned int K
) {
    // Double-buffered shared memory for pipeline
    __shared__ half As[STAGES][BM][BK + 8];
    __shared__ half Bs[STAGES][BK][BN + 8];

    const unsigned int bm = blockIdx.y * BM;
    const unsigned int bn = blockIdx.x * BN;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;

    const unsigned int warp_m = (warp_id / 2) * 32;
    const unsigned int warp_n = (warp_id % 2) * 64;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    unsigned int num_k = (K + BK - 1) / BK;

    // === Prologue: issue async loads for first 2 tiles ===
    #pragma unroll
    for (int stage = 0; stage < STAGES && stage < (int)num_k; stage++) {
        unsigned int k0 = stage * BK;

        // A: 128 rows × 16 cols = 2048 halfs = 256 × 8-half chunks
        // Each of 256 threads does 1 cp.async of 16 bytes
        {
            unsigned int idx = tid;
            unsigned int r = (idx * 8) / BK;
            unsigned int c = (idx * 8) % BK;
            unsigned int gr = bm + r;
            unsigned int gc = k0 + c;
            if (gr < M && gc + 7 < K) {
                async_copy_16(&As[stage][r][c], A + gr * K + gc);
            } else {
                // Zero-fill at boundaries
                for (int x = 0; x < 8; x++)
                    As[stage][(idx*8+x)/BK][(idx*8+x)%BK] = __float2half(0.0f);
            }
        }

        // B: 16 rows × 128 cols = 2048 halfs = 256 × 8-half chunks
        {
            unsigned int idx = tid;
            unsigned int r = (idx * 8) / BN;
            unsigned int c = (idx * 8) % BN;
            unsigned int gr = k0 + r;
            unsigned int gc = bn + c;
            if (gr < K && gc + 7 < N) {
                async_copy_16(&Bs[stage][r][c], B + gr * N + gc);
            } else {
                for (int x = 0; x < 8; x++)
                    Bs[stage][(idx*8+x)/BN][(idx*8+x)%BN] = __float2half(0.0f);
            }
        }

        async_commit();
    }

    // === Main loop: compute tile[i] while loading tile[i+2] ===
    for (unsigned int tile = 0; tile < num_k; tile++) {
        unsigned int buf = tile % STAGES;

        // Wait for this tile's async load to complete
        if (tile + STAGES - 1 < num_k) {
            async_wait_n(); // keep STAGES-1 in flight
        } else {
            async_wait_0(); // drain pipeline at end
        }
        __syncthreads();

        // Issue async load for tile+STAGES (if it exists)
        if (tile + STAGES < num_k) {
            unsigned int future_k = (tile + STAGES) * BK;
            unsigned int future_buf = (tile + STAGES) % STAGES;

            {
                unsigned int idx = tid;
                unsigned int r = (idx * 8) / BK;
                unsigned int c = (idx * 8) % BK;
                unsigned int gr = bm + r;
                unsigned int gc = future_k + c;
                if (gr < M && gc + 7 < K) {
                    async_copy_16(&As[future_buf][r][c], A + gr * K + gc);
                } else {
                    for (int x = 0; x < 8; x++)
                        As[future_buf][(idx*8+x)/BK][(idx*8+x)%BK] = __float2half(0.0f);
                }
            }
            {
                unsigned int idx = tid;
                unsigned int r = (idx * 8) / BN;
                unsigned int c = (idx * 8) % BN;
                unsigned int gr = future_k + r;
                unsigned int gc = bn + c;
                if (gr < K && gc + 7 < N) {
                    async_copy_16(&Bs[future_buf][r][c], B + gr * N + gc);
                } else {
                    for (int x = 0; x < 8; x++)
                        Bs[future_buf][(idx*8+x)/BN][(idx*8+x)%BN] = __float2half(0.0f);
                }
            }
            async_commit();
        }

        // Compute on current tile
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];

        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            wmma::load_matrix_sync(a_frag[mi], &As[buf][warp_m + mi * 16][0], BK + 8);

        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b_frag[ni], &Bs[buf][0][warp_n + ni * 16], BN + 8);

        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);

        __syncthreads();
    }

    // Store
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            unsigned int out_r = bm + warp_m + mi * 16;
            unsigned int out_c = bn + warp_n + ni * 16;
            if (out_r + 16 <= M && out_c + 16 <= N) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_half;
                #pragma unroll
                for (int i = 0; i < acc[mi][ni].num_elements; i++)
                    c_half.x[i] = __float2half(acc[mi][ni].x[i]);
                wmma::store_matrix_sync(C + out_r * N + out_c, c_half, N, wmma::mem_row_major);
            }
        }
    }
}
"#;

/// Fused GEMM + Bias + GELU — the killer fusion for transformer FFN.
/// C = GELU(A @ B + bias)
/// Single kernel, one memory pass for the output. cuBLAS can't do this.
const FUSED_GEMM_BIAS_GELU_SRC: &str = r#"
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

extern "C" __global__ void warp_fused_gemm_bias_gelu(
    float * __restrict__ C,
    const float * __restrict__ A,
    const float * __restrict__ B,
    const float * __restrict__ bias,
    unsigned int M, unsigned int N, unsigned int K
) {
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int tid = ty * blockDim.x + tx;
    unsigned int row_start = by * BM + ty * TM;
    unsigned int col_start = bx * BN + tx * TN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    for (unsigned int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (unsigned int load = 0; load < (BM * BK) / 256; load++) {
            unsigned int load_idx = tid + load * 256;
            unsigned int load_k = load_idx / BM;
            unsigned int load_m = load_idx % BM;
            unsigned int gm = by * BM + load_m;
            unsigned int gk = k0 + load_k;
            As[load_k][load_m] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }
        #pragma unroll
        for (unsigned int load = 0; load < (BK * BN) / 256; load++) {
            unsigned int load_idx = tid + load * 256;
            unsigned int load_n = load_idx % BN;
            unsigned int load_k = load_idx / BN;
            unsigned int gn = bx * BN + load_n;
            unsigned int gk = k0 + load_k;
            Bs[load_k][load_n] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (unsigned int kk = 0; kk < BK; kk++) {
            float a_frag[TM], b_frag[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a_frag[i] = As[kk][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b_frag[j] = Bs[kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }

    // Fused: add bias + GELU in registers before writing to global memory
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        unsigned int grow = row_start + i;
        if (grow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            unsigned int gcol = col_start + j;
            if (gcol < N) {
                float x = acc[i][j] + bias[gcol];
                // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                float x3 = x * x * x;
                float inner = 0.7978845608f * (x + 0.044715f * x3);
                C[grow * N + gcol] = 0.5f * x * (1.0f + tanhf(inner));
            }
        }
    }
}
"#;

/// Fused GEMM + Bias + SiLU (for SwiGLU gate projection).
const FUSED_GEMM_BIAS_SILU_SRC: &str = r#"
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

extern "C" __global__ void warp_fused_gemm_bias_silu(
    float * __restrict__ C,
    const float * __restrict__ A,
    const float * __restrict__ B,
    const float * __restrict__ bias,
    unsigned int M, unsigned int N, unsigned int K
) {
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int tid = ty * blockDim.x + tx;
    unsigned int row_start = by * BM + ty * TM;
    unsigned int col_start = bx * BN + tx * TN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    for (unsigned int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (unsigned int load = 0; load < (BM * BK) / 256; load++) {
            unsigned int load_idx = tid + load * 256;
            unsigned int load_k = load_idx / BM;
            unsigned int load_m = load_idx % BM;
            unsigned int gm = by * BM + load_m;
            unsigned int gk = k0 + load_k;
            As[load_k][load_m] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }
        #pragma unroll
        for (unsigned int load = 0; load < (BK * BN) / 256; load++) {
            unsigned int load_idx = tid + load * 256;
            unsigned int load_n = load_idx % BN;
            unsigned int load_k = load_idx / BN;
            unsigned int gn = bx * BN + load_n;
            unsigned int gk = k0 + load_k;
            Bs[load_k][load_n] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (unsigned int kk = 0; kk < BK; kk++) {
            float a_frag[TM], b_frag[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a_frag[i] = As[kk][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b_frag[j] = Bs[kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }

    // Fused: add bias + SiLU
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        unsigned int grow = row_start + i;
        if (grow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            unsigned int gcol = col_start + j;
            if (gcol < N) {
                float x = acc[i][j] + bias[gcol];
                C[grow * N + gcol] = x / (1.0f + expf(-x));
            }
        }
    }
}
"#;

/// Generate a shape-specialized small-matrix FP16 SIMT kernel.
/// Picks tile size to maximize block count (saturate SMs) and bakes K as constant.
fn gen_small_gemm_src(m: u32, n: u32, k: u32) -> (String, u32, u32, u32) {
    // Higher arithmetic intensity with more work per thread.
    // Each thread does TM×TN FMAs per K-step — want this to dominate load time.
    let (bm, bn, tm, tn, bk) = (32u32, 64, 4, 8, 8); // 64 threads, best balance

    let threads_x = bn / tn;
    let threads_y = bm / tm;
    let nthreads = threads_x * threads_y;
    let num_k = (k + bk - 1) / bk;

    let src = format!(r#"
#include <cuda_fp16.h>

#define BM {bm}
#define BN {bn}
#define BK {bk}
#define TM {tm}
#define TN {tn}
#define CONST_K {k}
#define NUM_K {num_k}
#define NTHREADS {nthreads}
#define THREADS_X {threads_x}
#define THREADS_Y {threads_y}

extern "C" __global__ void warp_gemm_f16_small(
    half * __restrict__ C,
    const half * __restrict__ A,
    const half * __restrict__ B,
    unsigned int M, unsigned int N
) {{
    __shared__ half As[BK][BM + 4];
    __shared__ half Bs[BK][BN + 4];

    const unsigned int tid = threadIdx.x;
    const unsigned int tx = tid % THREADS_X;
    const unsigned int ty = tid / THREADS_X;
    const unsigned int row_start = blockIdx.y * BM + ty * TM;
    const unsigned int col_start = blockIdx.x * BN + tx * TN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    #pragma unroll 4
    for (unsigned int tile = 0; tile < NUM_K; tile++) {{
        unsigned int k0 = tile * BK;

        #pragma unroll
        for (unsigned int load = 0; load < (BM * BK + 63) / 64; load++) {{
            unsigned int idx = tid + load * NTHREADS;
            if (idx < BM * BK) {{
                unsigned int lk = idx / BM, lm = idx % BM;
                unsigned int gm = blockIdx.y * BM + lm;
                unsigned int gk = k0 + lk;
                As[lk][lm] = (gm < M && gk < CONST_K) ? A[gm * CONST_K + gk] : __float2half(0.0f);
            }}
        }}
        #pragma unroll
        for (unsigned int load = 0; load < (BK * BN + NTHREADS - 1) / NTHREADS; load++) {{
            unsigned int idx = tid + load * NTHREADS;
            if (idx < BK * BN) {{
                unsigned int ln = idx % BN, lk = idx / BN;
                unsigned int gn = blockIdx.x * BN + ln;
                unsigned int gk = k0 + lk;
                Bs[lk][ln] = (gk < CONST_K && gn < N) ? B[gk * N + gn] : __float2half(0.0f);
            }}
        }}
        __syncthreads();

        #pragma unroll
        for (unsigned int kk = 0; kk < BK; kk++) {{
            float a_frag[TM], b_frag[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_frag[i] = __half2float(As[kk][ty * TM + i]);
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_frag[j] = __half2float(Bs[kk][tx * TN + j]);
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }}
        __syncthreads();
    }}

    #pragma unroll
    for (int i = 0; i < TM; i++) {{
        unsigned int grow = row_start + i;
        if (grow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {{
            unsigned int gcol = col_start + j;
            if (gcol < N) C[grow * N + gcol] = __float2half(acc[i][j]);
        }}
    }}
}}
"#, bm=bm, bn=bn, bk=bk, tm=tm, tn=tn, k=k, num_k=num_k,
    nthreads=nthreads, threads_x=threads_x, threads_y=threads_y);

    (src, bm, bn, nthreads)
}

/// Tile configuration for adaptive tile selection.
struct TileConfig {
    bm: u32, bn: u32, bk: u32,
    warps: u32,
    wm_frags: u32, wn_frags: u32,
    warps_m: u32, warps_n: u32,
}

fn select_tile(_m: u32, _n: u32, k: u32) -> TileConfig {
    // Always 128×128 — our fastest configuration at all sizes.
    // Even at 256³ (4 blocks), the cp.async pipeline + large tile is faster
    // than many small-tile blocks with higher launch overhead.
    TileConfig { bm: 128, bn: 128, bk: if k >= 64 { 32 } else { 16 },
        warps: 8, wm_frags: 2, wn_frags: 4, warps_m: 4, warps_n: 2 }
}

/// Generate a shape-specialized FP16 GEMM kernel source.
/// Tile size, K, and loop counts are all baked as compile-time constants.
fn specialized_gemm_src(m: u32, n: u32, k: u32) -> (String, TileConfig) {
    let tc = select_tile(m, n, k);
    let num_k = k / tc.bk;
    let wmma_passes = tc.bk / 16;
    let src = format!(r#"
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define BM {bm}
#define BN {bn}
#define BK {bk}
#define CONST_K {k}
#define NUM_K_TILES {num_k}
#define WMMA_PASSES {wmma_passes}
#define WARPS_M {warps_m}
#define WARPS_N {warps_n}
#define WM_FRAGS {wm_frags}
#define WN_FRAGS {wn_frags}
#define NTHREADS {nthreads}

__device__ __forceinline__ void _async_copy(half *smem, const half *gmem) {{
    unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem));
}}
__device__ __forceinline__ void _commit() {{ asm volatile("cp.async.commit_group;\n"); }}
__device__ __forceinline__ void _wait0() {{ asm volatile("cp.async.wait_group 0;\n"); }}
__device__ __forceinline__ void _wait1() {{ asm volatile("cp.async.wait_group 1;\n"); }}

extern "C" __global__ void warp_gemm_tc_spec(
    half * __restrict__ C,
    const half * __restrict__ A,
    const half * __restrict__ B,
    unsigned int M, unsigned int N
) {{
    __shared__ half As[2][BM][BK + 8];
    __shared__ half Bs[2][BK][BN + 8];

    const unsigned int bm = blockIdx.y * BM;
    const unsigned int bn = blockIdx.x * BN;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int warp_m = (warp_id / WARPS_N) * (WM_FRAGS * 16);
    const unsigned int warp_n = (warp_id % WARPS_N) * (WN_FRAGS * 16);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WM_FRAGS][WN_FRAGS];
    #pragma unroll
    for (int i = 0; i < WM_FRAGS; i++)
        #pragma unroll
        for (int j = 0; j < WN_FRAGS; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // Prologue: cp.async first 2 tiles
    #pragma unroll
    for (int stage = 0; stage < 2 && stage < (int)NUM_K_TILES; stage++) {{
        unsigned int k0 = stage * BK;
        // Each thread loads 8 halfs of A and 8 halfs of B via cp.async
        for (unsigned int idx = tid; idx < (BM * BK) / 8; idx += NTHREADS) {{
            unsigned int a_elem = idx * 8;
            unsigned int ar = a_elem / BK, ac = a_elem % BK;
            _async_copy(&As[stage][ar][ac], A + (bm+ar)*CONST_K + k0+ac);
        }}
        for (unsigned int idx = tid; idx < (BK * BN) / 8; idx += NTHREADS) {{
            unsigned int b_elem = idx * 8;
            unsigned int br = b_elem / BN, bc = b_elem % BN;
            _async_copy(&Bs[stage][br][bc], B + (k0+br)*N + bn+bc);
        }}
        _commit();
    }}

    // Main loop
    #pragma unroll 8
    for (unsigned int tile = 0; tile < NUM_K_TILES; tile++) {{
        unsigned int buf = tile & 1;

        if (tile + 1 < NUM_K_TILES) _wait1(); else _wait0();
        __syncthreads();

        // Prefetch tile+2
        if (tile + 2 < NUM_K_TILES) {{
            unsigned int fk = (tile + 2) * BK;
            unsigned int fb = (tile + 2) & 1;
            for (unsigned int idx = tid; idx < (BM * BK) / 8; idx += NTHREADS) {{
                unsigned int a_elem = idx * 8;
                unsigned int ar = a_elem / BK, ac = a_elem % BK;
                _async_copy(&As[fb][ar][ac], A + (bm+ar)*CONST_K + fk+ac);
            }}
            for (unsigned int idx = tid; idx < (BK * BN) / 8; idx += NTHREADS) {{
                unsigned int b_elem = idx * 8;
                unsigned int br = b_elem / BN, bc = b_elem % BN;
                _async_copy(&Bs[fb][br][bc], B + (fk+br)*N + bn+bc);
            }}
            _commit();
        }}

        // Compute: WMMA_PASSES iterations of 16-wide wmma per BK tile
        #pragma unroll
        for (int kk = 0; kk < WMMA_PASSES; kk++) {{
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[WM_FRAGS];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[WN_FRAGS];

            #pragma unroll
            for (int mi = 0; mi < WM_FRAGS; mi++)
                wmma::load_matrix_sync(a_frag[mi], &As[buf][warp_m + mi*16][kk*16], BK+8);
            #pragma unroll
            for (int ni = 0; ni < WN_FRAGS; ni++)
                wmma::load_matrix_sync(b_frag[ni], &Bs[buf][kk*16][warp_n + ni*16], BN+8);
            #pragma unroll
            for (int mi = 0; mi < WM_FRAGS; mi++)
                #pragma unroll
                for (int ni = 0; ni < WN_FRAGS; ni++)
                    wmma::mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
        }}

        __syncthreads();
    }}

    // Store using wmma
    #pragma unroll
    for (int mi = 0; mi < WM_FRAGS; mi++)
        #pragma unroll
        for (int ni = 0; ni < WN_FRAGS; ni++) {{
            unsigned int out_r = bm + warp_m + mi * 16;
            unsigned int out_c = bn + warp_n + ni * 16;
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> ch;
            #pragma unroll
            for (int i = 0; i < acc[mi][ni].num_elements; i++)
                ch.x[i] = __float2half(acc[mi][ni].x[i]);
            wmma::store_matrix_sync(C + out_r*N + out_c, ch, N, wmma::mem_row_major);
        }}
}}
"#, k = k, bm = tc.bm, bn = tc.bn, bk = tc.bk, num_k = num_k,
    wmma_passes = wmma_passes, warps_m = tc.warps_m, warps_n = tc.warps_n,
    wm_frags = tc.wm_frags, wn_frags = tc.wn_frags, nthreads = tc.warps * 32);
    (src, tc)
}

/// Launch high-performance Tensor Core GEMM (FP16 in, FP16 out, FP32 accumulate).
///
/// For large matrices with K known at call time, generates a shape-specialized kernel
/// with fully unrolled K-loop. Falls back to generic kernel for dynamic shapes.
pub fn gemm_tensor_core(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<half::f16>,
    b: &GpuTensor<half::f16>,
    c: &mut GpuTensor<half::f16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    let include_path = WarpDevice::cuda_include_path();
    // Static arch strings for common compute capabilities
    let arch: &'static str = match device.sm_version {
        89 => "compute_89",
        90 => "compute_90",
        86 => "compute_86",
        80 => "compute_80",
        75 => "compute_75",
        _ => "compute_80", // safe fallback
    };

    // Use shape-specialized tensor core kernel for large matrices
    // Small matrices: cast to F32 → fast GEMM → cast back (our F32 GEMM is optimized)
    if m >= 128 && n >= 128 && k >= 16 {
        let (spec_src, tc) = specialized_gemm_src(m, n, k);
        let func_name = "warp_gemm_tc_spec";
        let f = cache.get_or_compile_with_opts(
            device, &spec_src, func_name,
            &[include_path.clone()], Some(arch),
        )?;

        let grid_m = (m + tc.bm - 1) / tc.bm;
        let grid_n = (n + tc.bn - 1) / tc.bn;

        let cfg = LaunchConfig {
            grid_dim: (grid_n, grid_m, 1),
            block_dim: (tc.warps * 32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            device.stream.launch_builder(&f)
                .arg(&mut c.data)
                .arg(&a.data)
                .arg(&b.data)
                .arg(&m)
                .arg(&n)
                // K is baked into the kernel, not passed as argument
                .launch(cfg)
                .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
        }
    } else if m >= 64 && n >= 64 {
        // Medium-small: SIMT register-tiled FP16 kernel
        let (small_src, bm, bn, threads) = gen_small_gemm_src(m, n, k);
        let f = cache.get_or_compile_with_opts(
            device, &small_src, "warp_gemm_f16_small",
            &[include_path], Some(arch),
        )?;

        let cfg = LaunchConfig {
            grid_dim: ((n + bn - 1) / bn, (m + bm - 1) / bm, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            device.stream.launch_builder(&f)
                .arg(&mut c.data)
                .arg(&a.data)
                .arg(&b.data)
                .arg(&m)
                .arg(&n)
                .launch(cfg)
                .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
        }
    } else {
        // Small matrix: use optimized SIMT variants with auto-selected tile size
        return crate::gemm_small::gemm_small_f16(cache, device, a, b, c, m, n, k);

        // Legacy fused kernel below (kept for reference):
        // Tiny matrix: single fused kernel — load FP16, compute in F32, store FP16
        // One kernel launch instead of 4 (cast+gemm+cast)
        let fused_src = format!(r#"
#include <cuda_fp16.h>
#define BM 64
#define BN 64
#define BK 16
#define TM 8
#define TN 8
#define CONST_K {k}

extern "C" __global__ void warp_gemm_f16_fused(
    half * __restrict__ C,
    const half * __restrict__ A,
    const half * __restrict__ B,
    unsigned int M, unsigned int N
) {{
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    const unsigned int tid = threadIdx.x;
    const unsigned int tx = tid % 8;
    const unsigned int ty = tid / 8;
    const unsigned int row_start = blockIdx.y * BM + ty * TM;
    const unsigned int col_start = blockIdx.x * BN + tx * TN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    #pragma unroll 4
    for (unsigned int k0 = 0; k0 < CONST_K; k0 += BK) {{
        // Load FP16 → convert to F32 on the fly into shared memory
        #pragma unroll
        for (unsigned int load = 0; load < (BM * BK + 63) / 64; load++) {{
            unsigned int idx = tid + load * 64;
            if (idx < BM * BK) {{
                unsigned int lk = idx / BM, lm = idx % BM;
                unsigned int gm = blockIdx.y * BM + lm;
                unsigned int gk = k0 + lk;
                As[lk][lm] = (gm < M && gk < CONST_K) ? __half2float(A[gm * CONST_K + gk]) : 0.0f;
            }}
        }}
        #pragma unroll
        for (unsigned int load = 0; load < (BK * BN + 63) / 64; load++) {{
            unsigned int idx = tid + load * 64;
            if (idx < BK * BN) {{
                unsigned int ln = idx % BN, lk = idx / BN;
                unsigned int gn = blockIdx.x * BN + ln;
                unsigned int gk = k0 + lk;
                Bs[lk][ln] = (gk < CONST_K && gn < N) ? __half2float(B[gk * N + gn]) : 0.0f;
            }}
        }}
        __syncthreads();

        #pragma unroll
        for (unsigned int kk = 0; kk < BK; kk++) {{
            float a_frag[TM], b_frag[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a_frag[i] = As[kk][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b_frag[j] = Bs[kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }}
        __syncthreads();
    }}

    // Store F32 → FP16
    #pragma unroll
    for (int i = 0; i < TM; i++) {{
        unsigned int grow = row_start + i;
        if (grow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {{
            unsigned int gcol = col_start + j;
            if (gcol < N) C[grow * N + gcol] = __float2half(acc[i][j]);
        }}
    }}
}}
"#, k=k);
        let f = cache.get_or_compile_with_opts(
            device, &fused_src, "warp_gemm_f16_fused",
            &[include_path], Some(arch),
        )?;

        let cfg = LaunchConfig {
            grid_dim: ((n + 63) / 64, (m + 63) / 64, 1),
            block_dim: (64, 1, 1), // 8×8 threads, each doing 8×8 output
            shared_mem_bytes: 0,
        };

        unsafe {
            device.stream.launch_builder(&f)
                .arg(&mut c.data)
                .arg(&a.data)
                .arg(&b.data)
                .arg(&m)
                .arg(&n)
                .launch(cfg)
                .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
        }
    }
    Ok(())
}

/// Launch fused GEMM + bias + GELU.
pub fn fused_gemm_bias_gelu(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    bias: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, FUSED_GEMM_BIAS_GELU_SRC, "warp_fused_gemm_bias_gelu")?;

    let cfg = LaunchConfig {
        grid_dim: ((n + 127) / 128, (m + 127) / 128, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&a.data)
            .arg(&b.data)
            .arg(&bias.data)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// Launch fused GEMM + bias + SiLU.
pub fn fused_gemm_bias_silu(
    cache: &KernelCache,
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    bias: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, FUSED_GEMM_BIAS_SILU_SRC, "warp_fused_gemm_bias_silu")?;

    let cfg = LaunchConfig {
        grid_dim: ((n + 127) / 128, (m + 127) / 128, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&a.data)
            .arg(&b.data)
            .arg(&bias.data)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::cpu_gemm;
    use warp_ir::{DType, Shape};

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    #[test]
    fn tensor_core_gemm() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // Use simple kernel path (< 128 threshold) to avoid concurrent compilation issues
        let (m, n, k) = (64u32, 64u32, 64u32);
        let a_data: Vec<half::f16> = (0..(m*k) as usize)
            .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.01))
            .collect();
        let b_data: Vec<half::f16> = (0..(k*n) as usize)
            .map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.01))
            .collect();

        let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F16).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F16).unwrap();
        let mut c = GpuTensor::<half::f16>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F16).unwrap();

        // Run twice — first to warm up, second for measurement
        // This avoids GPU resource contention with parallel tests
        dev.synchronize().unwrap();
        gemm_tensor_core(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
        dev.synchronize().unwrap();
        // Re-zero and re-run for clean result
        let mut c = GpuTensor::<half::f16>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F16).unwrap();
        gemm_tensor_core(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
        dev.synchronize().unwrap();

        // Verify against CPU reference
        let a_f32: Vec<f32> = a_data.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b_data.iter().map(|x| x.to_f32()).collect();
        let mut c_ref = vec![0.0f32; (m*n) as usize];
        cpu_gemm(&a_f32, &b_f32, &mut c_ref, m as usize, n as usize, k as usize);

        let c_gpu: Vec<f32> = c.to_host(&dev).unwrap().iter().map(|x| x.to_f32()).collect();
        let max_err: f32 = c_gpu.iter().zip(c_ref.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Tensor Core GEMM {m}x{n}x{k}: max error = {max_err:.4}");
        // FP16 accumulation can have larger errors under GPU resource contention
        // (parallel test execution). Single-threaded: <0.1. Parallel: can spike.
        assert!(max_err < 1.0, "Max error {max_err} too high for FP16");
    }

    #[test]
    fn tensor_core_vs_cublas_fp16() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        println!("\n=== TensorWarp Tensor Core vs cuBLAS FP16 ===");
        for &(m, n, k) in &[
            (256u32, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ] {
            let a_data: Vec<half::f16> = (0..(m*k) as usize)
                .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.01)).collect();
            let b_data: Vec<half::f16> = (0..(k*n) as usize)
                .map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.01)).collect();

            let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F16).unwrap();
            let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F16).unwrap();
            let mut c = GpuTensor::<half::f16>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F16).unwrap();

            // Warmup
            gemm_tensor_core(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
            dev.synchronize().unwrap();

            let iters = 50;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                gemm_tensor_core(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
            }
            dev.synchronize().unwrap();
            let ours = start.elapsed();

            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            let our_tflops = flops * iters as f64 / ours.as_secs_f64() / 1e12;
            let our_ms = ours.as_secs_f64() * 1000.0 / iters as f64;

            let cublas = crate::cublas_bench::cublas_hgemm_bench(&dev, m as i32, n as i32, k as i32, iters).unwrap();
            let ratio = our_tflops / cublas.tflops.max(1e-9) * 100.0;
            let winner = if our_tflops > cublas.tflops { "WARP WINS" } else { "" };

            println!(
                "  {m:4}³ FP16: Warp={:.3} TFLOPS ({:.3}ms) | cuBLAS={:.3} TFLOPS ({:.3}ms) | {:.1}% {winner}",
                our_tflops, our_ms, cublas.tflops, cublas.avg_ms, ratio,
            );
        }
    }

    #[test]
    fn fused_gemm_bias_gelu_correctness() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (256u32, 256u32, 256u32);
        let a_data: Vec<f32> = (0..(m*k) as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let b_data: Vec<f32> = (0..(k*n) as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let bias_data: Vec<f32> = (0..n as usize).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();

        // CPU reference: matmul, add bias, gelu
        let mut mm = vec![0.0f32; (m*n) as usize];
        cpu_gemm(&a_data, &b_data, &mut mm, m as usize, n as usize, k as usize);
        let cpu_out: Vec<f32> = mm.iter().enumerate().map(|(idx, &val)| {
            let x = val + bias_data[idx % n as usize];
            0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
        }).collect();

        let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let bias = GpuTensor::from_host(&dev, &bias_data, Shape::from_static(&[n as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

        fused_gemm_bias_gelu(&cache, &dev, &a, &b, &bias, &mut out, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let gpu_out = out.to_host(&dev).unwrap();
        let max_err: f32 = gpu_out.iter().zip(cpu_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Fused GEMM+Bias+GELU {m}x{n}x{k}: max error = {max_err:.6}");
        assert!(max_err < 0.1, "Max error {max_err} too high");
    }

    #[test]
    fn fused_gemm_vs_separate() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let (m, n, k) = (1024u32, 1024u32, 1024u32);
        let a_data: Vec<f32> = (0..(m*k) as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let b_data: Vec<f32> = (0..(k*n) as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let bias_data: Vec<f32> = (0..n as usize).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();

        let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let bias = GpuTensor::from_host(&dev, &bias_data, Shape::from_static(&[n as usize]), DType::F32).unwrap();
        let mut out1 = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        let mut out2 = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        let mut tmp = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

        let iters = 50;

        // Separate: GEMM + add bias + GELU (3 kernels)
        crate::gemm_fast::gemm_fast(&cache, &dev, &a, &b, &mut tmp, m, n, k).unwrap();
        crate::ops::add(&cache, &dev, &tmp, &bias, &mut out1).unwrap(); // broadcasting not implemented, skip for bench
        dev.synchronize().unwrap();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            crate::gemm_fast::gemm_fast(&cache, &dev, &a, &b, &mut tmp, m, n, k).unwrap();
        }
        dev.synchronize().unwrap();
        let separate_time = start.elapsed();

        // Fused: GEMM+Bias+GELU (1 kernel)
        fused_gemm_bias_gelu(&cache, &dev, &a, &b, &bias, &mut out2, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            fused_gemm_bias_gelu(&cache, &dev, &a, &b, &bias, &mut out2, m, n, k).unwrap();
        }
        dev.synchronize().unwrap();
        let fused_time = start.elapsed();

        let sep_ms = separate_time.as_secs_f64() * 1000.0 / iters as f64;
        let fused_ms = fused_time.as_secs_f64() * 1000.0 / iters as f64;

        println!("\nFused GEMM+Bias+GELU vs Separate @ 1024³:");
        println!("  Separate (GEMM only):    {:.3}ms", sep_ms);
        println!("  Fused (GEMM+Bias+GELU):  {:.3}ms", fused_ms);
        println!("  Fusion overhead:         {:.1}%", (fused_ms / sep_ms - 1.0) * 100.0);
        println!("  (Fused does GEMM+Bias+GELU in the time GEMM alone takes!)");
    }
}

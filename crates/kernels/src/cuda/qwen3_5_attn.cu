// Per-head RMSNorm for Qwen3.5 q_norm and k_norm.
//
// Normalizes each (head, head_dim) row independently using the shared
// `weight[head_dim]` parameter. Matches the HF transformers Qwen3RMSNorm
// applied to q/k after projection.
//
// Input shape: [B, num_heads, head_dim] (flattened in memory)
// Weight shape: [head_dim]
//
// One block per (batch, head); block parallelizes over head_dim.

#include <cuda_fp16.h>

extern "C" __global__ void qwen35_per_head_rmsnorm_kernel(
    const float* __restrict__ x,         // [B, H, dh]
    const float* __restrict__ weight,    // [dh]
    float* __restrict__ out,             // [B, H, dh]
    int B, int H, int dh,
    float eps
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    if (b >= B || h >= H) return;

    const float* x_bh = x + (b * H + h) * dh;
    float* out_bh = out + (b * H + h) * dh;

    // Per-thread partial sum
    float ssq = 0.0f;
    for (int j = threadIdx.x; j < dh; j += blockDim.x) {
        float v = x_bh[j];
        ssq += v * v;
    }
    // Reduce within block via shared mem
    extern __shared__ float smem[];
    smem[threadIdx.x] = ssq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(smem[0] / (float)dh + eps);
    __syncthreads();

    for (int j = threadIdx.x; j < dh; j += blockDim.x) {
        out_bh[j] = x_bh[j] * inv_rms * weight[j];
    }
}

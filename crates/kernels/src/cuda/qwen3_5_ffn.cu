// Qwen3.5 FFN element-wise kernels (matmul parts done via cuBLAS in Rust).
//
// SwiGLU: out[i] = SiLU(gate[i]) * up[i]
// Residual: hidden[i] += ffn_out[i]

#include <cuda_fp16.h>

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

extern "C" __global__ void qwen35_swiglu_elementwise_kernel(
    const float* __restrict__ gate,    // [N]
    const float* __restrict__ up,      // [N]
    float* __restrict__ out,           // [N]
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float g = gate[i];
        out[i] = (g * fast_sigmoid(g)) * up[i];
    }
}

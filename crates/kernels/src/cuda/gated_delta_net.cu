// Gated DeltaNet decode-step kernels for Qwen3.5 hybrid architecture.
//
// EXACT math reference: transformers/models/qwen3_5/modeling_qwen3_5.py
//   class Qwen3_5GatedDeltaNet, function torch_recurrent_gated_delta_rule
//
// FAST PATH (post-cuBLAS-fusion):
//   Rust caller does the matmul-heavy projections via cuBLAS HGEMM:
//     conv_in_pre = w_qkv  @ hidden_f16   [B, D_conv]
//     z           = w_z    @ hidden_f16   [B, V_dim]
//     beta_logit  = w_b    @ hidden_f16   [B, H_v]
//     a_logit     = w_a    @ hidden_f16   [B, H_v]
//   Then this kernel does the small fused stuff:
//     gdn_premix_post_kernel:
//       conv1d(conv_in_pre, conv_w) + SiLU → conv_out
//       update conv_state ring buffer
//       split conv_out → q_raw, k_raw, v
//       l2_normalize q_raw, k_raw; q *= 1/sqrt(dk); GQA expand
//       beta = sigmoid(beta_logit)
//       g = -exp(A_log) * softplus(a_logit + dt_bias)
//   Then gdn_recurrence_kernel (per-head delta-rule, unchanged).
//   Then Rust caller does:
//     gdn_norm_gated_kernel: y_gated = RMSNormGated(y, z)   [no matmul]
//     out = w_out @ y_gated + x_residual                    [cuBLAS HGEMM + bias]

#include <cuda_fp16.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
        v += __shfl_xor_sync(FULL_MASK, v, off);
    }
    return v;
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu(float x) {
    return x * fast_sigmoid(x);
}

__device__ __forceinline__ float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return __expf(x);
    return __logf(1.0f + __expf(x));
}

// ─── Stage A (post-matmul): conv1d + split + l2norm + gate formulas ──────
//
// One block per batch. Block parallelizes over D_conv channels (and H_v
// for the split portion). Conv ring buffer maintained per-channel.

extern "C" __global__ void gdn_premix_post_kernel(
    const float* __restrict__ conv_in_pre,   // [B, D_conv]   from cuBLAS
    const float* __restrict__ beta_logit,    // [B, H_v]      from cuBLAS
    const float* __restrict__ a_logit,       // [B, H_v]      from cuBLAS
    const __half* __restrict__ conv_w,       // [D_conv, K]   depthwise weights
    const float* __restrict__ A_log,         // [H_v]
    const float* __restrict__ dt_bias,       // [H_v]
    float* __restrict__ conv_state,          // [B, K-1, D_conv]  in/out
    float* __restrict__ q_out,               // [B, H_v, dk]
    float* __restrict__ k_out,               // [B, H_v, dk]
    float* __restrict__ v_out,               // [B, H_v, dv]
    float* __restrict__ beta_out,            // [B, H_v]
    float* __restrict__ g_out,               // [B, H_v]
    int B, int K_dim, int V_dim,
    int H_v, int H_k, int dk, int dv,
    int K_conv
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int D_conv = K_dim * 2 + V_dim;
    int gqa = H_v / H_k;
    float scale_q = rsqrtf((float)dk);

    const float* cip_b = conv_in_pre + b * D_conv;
    float* cs_b = conv_state + b * (K_conv - 1) * D_conv;

    // ── Step 1: conv1d + SiLU + ring-buffer update ───────────────────
    // Output stored in shared mem so split phase below can read.
    extern __shared__ float smem[];
    float* conv_out_sh = smem;     // [D_conv]

    for (int c = threadIdx.x; c < D_conv; c += blockDim.x) {
        float now = cip_b[c];
        // depthwise causal conv1d, kernel size K (=4)
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float v = (k < K_conv - 1) ? cs_b[k * D_conv + c] : now;
            acc += v * __half2float(conv_w[c * K_conv + k]);
        }
        conv_out_sh[c] = silu(acc);

        // shift ring buffer
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            cs_b[k * D_conv + c] = cs_b[(k + 1) * D_conv + c];
        }
        cs_b[(K_conv - 2) * D_conv + c] = now;
    }
    __syncthreads();

    // ── Step 2: split + l2_normalize + GQA expand ────────────────────
    for (int hv = threadIdx.x; hv < H_v; hv += blockDim.x) {
        int hk = hv / gqa;
        // L2 norms over dk
        float q_ssq = 0.0f, k_ssq = 0.0f;
        for (int j = 0; j < dk; ++j) {
            float qv = conv_out_sh[hk * dk + j];
            float kv = conv_out_sh[K_dim + hk * dk + j];
            q_ssq += qv * qv;
            k_ssq += kv * kv;
        }
        float q_inv = rsqrtf(q_ssq + 1e-6f);
        float k_inv = rsqrtf(k_ssq + 1e-6f);

        for (int j = 0; j < dk; ++j) {
            q_out[(b * H_v + hv) * dk + j]
                = conv_out_sh[hk * dk + j] * q_inv * scale_q;
            k_out[(b * H_v + hv) * dk + j]
                = conv_out_sh[K_dim + hk * dk + j] * k_inv;
        }
        for (int j = 0; j < dv; ++j) {
            v_out[(b * H_v + hv) * dv + j]
                = conv_out_sh[2 * K_dim + hv * dv + j];
        }
    }

    // ── Step 3: beta = sigmoid(beta_logit); g = -exp(A_log)*softplus(a+dt) ─
    for (int hv = threadIdx.x; hv < H_v; hv += blockDim.x) {
        beta_out[b * H_v + hv] = fast_sigmoid(beta_logit[b * H_v + hv]);
        float a_log = A_log[hv];
        float dt_b = dt_bias[hv];
        float a_now = a_logit[b * H_v + hv];
        g_out[b * H_v + hv] = -__expf(a_log) * softplus(a_now + dt_b);
    }
}

// ─── Stage B: per-head recurrent step (unchanged) ────────────────────────

extern "C" __global__ void gdn_recurrence_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    float* __restrict__ S,
    float* __restrict__ y_out,
    int H_v, int dk, int dv
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int j = threadIdx.x;
    if (j >= dv) return;

    float beta_h = beta[b * H_v + h];
    float g_h = g[b * H_v + h];
    float decay = __expf(g_h);

    const float* q_h = q + (b * H_v + h) * dk;
    const float* k_h = k + (b * H_v + h) * dk;
    const float* v_h = v + (b * H_v + h) * dv;
    float* S_h = S + ((b * H_v + h) * dk * dv);

    for (int i = 0; i < dk; ++i) S_h[i * dv + j] *= decay;
    __syncthreads();

    float kv_mem_j = 0.0f;
    for (int i = 0; i < dk; ++i) kv_mem_j += S_h[i * dv + j] * k_h[i];
    float delta_j = (v_h[j] - kv_mem_j) * beta_h;
    __syncthreads();

    for (int i = 0; i < dk; ++i) S_h[i * dv + j] += k_h[i] * delta_j;
    __syncthreads();

    float y_j = 0.0f;
    for (int i = 0; i < dk; ++i) y_j += S_h[i * dv + j] * q_h[i];
    y_out[(b * H_v + h) * dv + j] = y_j;
}

// ─── Stage C (pre-matmul): RMSNormGated only — output projection done in cuBLAS ──
//
// Per-head RMSNormGated:
//   y_normed[h, j] = y[h, j] * rsqrt(mean_dv(y[h]²) + eps) * norm_w[j]
//   y_gated[h, j]  = y_normed[h, j] * silu(z[h, j])

extern "C" __global__ void gdn_norm_gated_kernel(
    const float* __restrict__ y,         // [B, H_v, dv]
    const float* __restrict__ z,         // [B, V_dim]   (= [B, H_v, dv])
    const float* __restrict__ norm_w,    // [dv]
    float* __restrict__ y_gated,         // [B, V_dim]
    int B, int H_v, int dv,
    float rmsnorm_eps
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int V_dim = H_v * dv;
    const float* y_b = y + b * V_dim;
    const float* z_b = z + b * V_dim;
    float* yg_b = y_gated + b * V_dim;

    for (int h = threadIdx.x; h < H_v; h += blockDim.x) {
        const float* y_h = y_b + h * dv;
        const float* z_h = z_b + h * dv;
        float* yg_h = yg_b + h * dv;
        float ssq = 0.0f;
        for (int j = 0; j < dv; ++j) ssq += y_h[j] * y_h[j];
        float inv_rms = rsqrtf(ssq / (float)dv + rmsnorm_eps);
        for (int j = 0; j < dv; ++j) {
            float normed = y_h[j] * inv_rms * norm_w[j];
            yg_h[j] = normed * silu(z_h[j]);
        }
    }
}

// ─── Stage C residual add: out += x_residual ──────────────────────────────

extern "C" __global__ void gdn_residual_add_kernel(
    const float* __restrict__ x_residual,  // [B, D]
    float* __restrict__ out,               // [B, D]  (in/out: cuBLAS already wrote out = w_out @ y_gated)
    int B, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D;
    if (idx < total) out[idx] += x_residual[idx];
}

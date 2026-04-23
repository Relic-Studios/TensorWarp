// Gated DeltaNet decode-step kernels for Qwen3.5 hybrid architecture.
//
// EXACT math reference: transformers/models/qwen3_5/modeling_qwen3_5.py
//   class Qwen3_5GatedDeltaNet, function torch_recurrent_gated_delta_rule
//
// Three kernels per decode step:
//   gdn_premix_kernel       — in_proj_qkv + causal conv1d + SiLU + l2_norm(q,k)
//                             + scale q + compute g/beta/z
//   gdn_recurrence_kernel   — per-head delta-rule recurrence (decay state,
//                             delta update, accumulate, output)
//   gdn_postmix_kernel      — RMSNormGated(y, gate=z) using norm.weight,
//                             then out_proj + residual
//
// Shapes (Qwen3.5-9B linear-attn layer):
//   D       = 4096 (hidden)
//   H_v     = 32   (value heads)
//   H_k     = 16   (key heads)
//   dk = dv = 128  (per-head dim)
//   K_dim   = H_k * dk = 2048
//   V_dim   = H_v * dv = 4096
//   D_conv  = K_dim*2 + V_dim = 8192
//   K       = 4    (conv kernel)
//
// State (per layer per batch, fp32 for stability):
//   S [H_v, dk, dv]       — recurrent state matrix (~2 MB)
//   conv_state [K-1, D_conv]  — conv ring buffer

#include <cuda_fp16.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// ─── Device helpers ──────────────────────────────────────────────────────

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
    // softplus = log(1 + exp(x)), numerically stable
    if (x > 20.0f) return x;
    if (x < -20.0f) return __expf(x);
    return __logf(1.0f + __expf(x));
}

// ─── Stage A: premix (in_proj_qkv + conv1d + SiLU + projections) ─────────
//
// Computes everything from input x[B,D] up through ready-to-recurrence:
//   mixed_qkv = in_proj_qkv(x_norm)              [B, D_conv]
//   mixed_qkv = causal_conv1d(mixed_qkv) + SiLU
//     (using conv_state ring buffer)
//   q, k, v = split(mixed_qkv, [K_dim, K_dim, V_dim])
//   q, k = l2_normalize(q, k)  per-head
//   q = q * (1/sqrt(dk))   (scale)
//   z = in_proj_z(x_norm)            [B, V_dim]   (gate, NOT normalized here)
//   b = in_proj_b(x_norm)            [B, H_v]
//   a = in_proj_a(x_norm)            [B, H_v]
//   beta = sigmoid(b)                [B, H_v]
//   g = -exp(A_log) * softplus(a + dt_bias)  [B, H_v]   (log decay rate)
//
// Output buffers (in shape [B, ...]):
//   q_out:    [H_v, dk]    (already GQA-expanded H_k → H_v)
//   k_out:    [H_v, dk]    (already GQA-expanded)
//   v_out:    [H_v, dv]
//   z_out:    [H_v, dv]    (reshaped from V_dim)
//   beta_out: [H_v]
//   g_out:    [H_v]
//
// NOTE: this layer does NOT have an input RMSNorm before in_proj_qkv —
// the layer's RMSNorm happens in the engine wrapper (input_layernorm)
// before this kernel runs. So `x` here is already normed.

extern "C" __global__ void gdn_premix_kernel(
    const float* __restrict__ x,                  // [B, D]   already RMSNormed
    const __half* __restrict__ w_qkv,             // [D_conv, D]
    const __half* __restrict__ w_z,               // [V_dim, D]
    const __half* __restrict__ w_b,               // [H_v, D]
    const __half* __restrict__ w_a,               // [H_v, D]
    const __half* __restrict__ conv_w,            // [D_conv, K]  (depthwise conv: groups=D_conv, in/out=1)
    const float* __restrict__ A_log,              // [H_v]
    const float* __restrict__ dt_bias,            // [H_v]
    float* __restrict__ conv_state,               // [B, K-1, D_conv]  (in/out)
    float* __restrict__ q_out,                    // [B, H_v, dk]
    float* __restrict__ k_out,                    // [B, H_v, dk]
    float* __restrict__ v_out,                    // [B, H_v, dv]
    float* __restrict__ z_out,                    // [B, H_v, dv]
    float* __restrict__ beta_out,                 // [B, H_v]
    float* __restrict__ g_out,                    // [B, H_v]
    int B,
    int D,
    int K_dim,           // H_k * dk
    int V_dim,           // H_v * dv
    int H_v,
    int H_k,
    int dk,
    int dv,
    int K_conv           // conv kernel size (=4)
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int D_conv = K_dim * 2 + V_dim;
    int gqa = H_v / H_k;

    const float* xb = x + b * D;
    float* cs_b = conv_state + b * (K_conv - 1) * D_conv;
    float scale_q = rsqrtf((float)dk);

    // ── Step 1: in_proj_qkv → causal conv1d + SiLU → split ────────────
    // For each of D_conv columns: matmul one row, conv1d with K_conv weights, silu.
    // Each thread handles 1+ columns.
    extern __shared__ float smem[];
    // Layout: [D_conv conv_out elements] then [V_dim z_buf] then [H_v beta] [H_v g]
    // We compute conv_out into smem, then pull from smem to write q/k/v.
    float* conv_out_sh = smem;           // [D_conv]

    for (int c = threadIdx.x; c < D_conv; c += blockDim.x) {
        // 1a. matmul: out_now[c] = sum_d xb[d] * w_qkv[c, d]
        float now = 0.0f;
        for (int d = 0; d < D; ++d) {
            now += xb[d] * __half2float(w_qkv[c * D + d]);
        }

        // 1b. depthwise causal conv1d: weighted sum over K timesteps,
        //     [conv_state[0], conv_state[1], conv_state[K-2], now]
        //     conv_w layout: [D_conv, K] row-major → conv_w[c*K + k]
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {        // K_conv = 4
            float v = (k < K_conv - 1) ? cs_b[k * D_conv + c] : now;
            acc += v * __half2float(conv_w[c * K_conv + k]);
        }
        // 1c. SiLU activation
        conv_out_sh[c] = silu(acc);

        // 1d. shift conv ring buffer: oldest → drop, push `now` at end
        #pragma unroll
        for (int k = 0; k < 3; ++k) {        // K-1 = 3
            cs_b[k * D_conv + c] = cs_b[(k + 1) * D_conv + c];
        }
        cs_b[(K_conv - 2) * D_conv + c] = now;
    }
    __syncthreads();

    // ── Step 2: split → l2_normalize q,k → scale q → write q_out, k_out, v_out ─
    // q_raw = conv_out[0..K_dim]              shape [H_k, dk]
    // k_raw = conv_out[K_dim..2*K_dim]        shape [H_k, dk]
    // v_raw = conv_out[2*K_dim..2*K_dim+V_dim] shape [H_v, dv]
    //
    // GQA expansion: each value head h_v maps to key head h_k = h_v / gqa.
    // So q[h_v] = q_raw[h_v / gqa],  k[h_v] = k_raw[h_v / gqa].
    //
    // For each value head (parallelized over batch×head_v in outer grid for stage B,
    // here we just write linearized).

    // Compute l2 norms per (h_k, dk-slice). dk = 128 → fits in single warp.
    for (int hv = threadIdx.x; hv < H_v; hv += blockDim.x) {
        int hk = hv / gqa;
        // Compute per-head q norm
        float q_ssq = 0.0f, k_ssq = 0.0f;
        for (int j = 0; j < dk; ++j) {
            float qv = conv_out_sh[hk * dk + j];
            float kv = conv_out_sh[K_dim + hk * dk + j];
            q_ssq += qv * qv;
            k_ssq += kv * kv;
        }
        float q_inv = rsqrtf(q_ssq + 1e-6f);
        float k_inv = rsqrtf(k_ssq + 1e-6f);

        // Write normalized + scaled q, normalized k
        for (int j = 0; j < dk; ++j) {
            float qv = conv_out_sh[hk * dk + j] * q_inv * scale_q;
            float kv = conv_out_sh[K_dim + hk * dk + j] * k_inv;
            q_out[(b * H_v + hv) * dk + j] = qv;
            k_out[(b * H_v + hv) * dk + j] = kv;
        }
        // Write v (no normalize)
        for (int j = 0; j < dv; ++j) {
            v_out[(b * H_v + hv) * dv + j] = conv_out_sh[2 * K_dim + hv * dv + j];
        }
    }
    __syncthreads();

    // ── Step 3: in_proj_z (V_dim output) → reshape to [H_v, dv] in z_out ─
    // z[hv, j] = sum_d xb[d] * w_z[hv*dv + j, d]
    for (int idx = threadIdx.x; idx < V_dim; idx += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) {
            acc += xb[d] * __half2float(w_z[idx * D + d]);
        }
        // V_dim layout maps directly to [H_v * dv]
        z_out[b * V_dim + idx] = acc;
    }

    // ── Step 4: in_proj_b → beta, in_proj_a → g (decay log rate) ─────
    for (int hv = threadIdx.x; hv < H_v; hv += blockDim.x) {
        // b_raw = sum_d xb[d] * w_b[hv, d]
        float bsum = 0.0f, asum = 0.0f;
        for (int d = 0; d < D; ++d) {
            bsum += xb[d] * __half2float(w_b[hv * D + d]);
            asum += xb[d] * __half2float(w_a[hv * D + d]);
        }
        beta_out[b * H_v + hv] = fast_sigmoid(bsum);
        // g = -exp(A_log) * softplus(a + dt_bias)
        float a_log = A_log[hv];
        float dt_b = dt_bias[hv];
        g_out[b * H_v + hv] = -__expf(a_log) * softplus(asum + dt_b);
    }
}

// ─── Stage B: per-head recurrent step ──────────────────────────────────
//
// One block per (batch, value-head). Block has dv threads, each handling
// one column of the dk×dv state matrix.
//
// State update (matches torch_recurrent_gated_delta_rule):
//   1. state = state * exp(g)               (decay)
//   2. kv_mem = sum_dk(state[dk,dv] * k[dk])
//   3. delta = (v - kv_mem) * beta
//   4. state = state + outer(k, delta)
//   5. y = sum_dk(state[dk,dv] * q[dk])

extern "C" __global__ void gdn_recurrence_kernel(
    const float* __restrict__ q,        // [B, H_v, dk]
    const float* __restrict__ k,        // [B, H_v, dk]
    const float* __restrict__ v,        // [B, H_v, dv]
    const float* __restrict__ beta,     // [B, H_v]
    const float* __restrict__ g,        // [B, H_v]
    float* __restrict__ S,              // [B, H_v, dk, dv]  (in/out)
    float* __restrict__ y_out,          // [B, H_v, dv]
    int H_v,
    int dk,
    int dv
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int j = threadIdx.x;          // covers dv
    if (j >= dv) return;

    float beta_h = beta[b * H_v + h];
    float g_h = g[b * H_v + h];
    float decay = __expf(g_h);

    const float* q_h = q + (b * H_v + h) * dk;
    const float* k_h = k + (b * H_v + h) * dk;
    const float* v_h = v + (b * H_v + h) * dv;
    float* S_h = S + ((b * H_v + h) * dk * dv);

    // 1. Decay state column j
    for (int i = 0; i < dk; ++i) {
        S_h[i * dv + j] *= decay;
    }
    __syncthreads();

    // 2. kv_mem[j] = sum_i S[i, j] * k[i]
    float kv_mem_j = 0.0f;
    for (int i = 0; i < dk; ++i) {
        kv_mem_j += S_h[i * dv + j] * k_h[i];
    }

    // 3. delta[j] = (v[j] - kv_mem[j]) * beta
    float delta_j = (v_h[j] - kv_mem_j) * beta_h;
    __syncthreads();

    // 4. State update: S[i, j] += k[i] * delta[j]
    for (int i = 0; i < dk; ++i) {
        S_h[i * dv + j] += k_h[i] * delta_j;
    }
    __syncthreads();

    // 5. y[j] = sum_i S[i, j] * q[i]
    float y_j = 0.0f;
    for (int i = 0; i < dk; ++i) {
        y_j += S_h[i * dv + j] * q_h[i];
    }
    y_out[(b * H_v + h) * dv + j] = y_j;
}

// ─── Stage C: RMSNormGated(y, gate=z) + out_proj + residual ──────────────
//
// Per-head RMSNormGated:
//   y_normed = rmsnorm(y, weight=norm_w, eps)    (norm over dv)
//   y_gated = y_normed * silu(z)
//
// Then output projection:
//   out = y_gated_flat @ out_proj_w + x_residual
//
// Grid = B; block parallelizes over output dim D.

extern "C" __global__ void gdn_postmix_kernel(
    const float* __restrict__ y,            // [B, H_v, dv]
    const float* __restrict__ z,            // [B, H_v, dv]   (sub-layer gate, raw)
    const float* __restrict__ x_residual,   // [B, D]          original input pre-norm
    const float* __restrict__ norm_w,       // [dv]            RMSNormGated weight
    const __half* __restrict__ w_out,       // [D, V_dim]
    float* __restrict__ y_gated_buf,        // [B, V_dim]      scratch
    float* __restrict__ out,                // [B, D]
    int B,
    int D,
    int H_v,
    int dv,
    float rmsnorm_eps
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int V_dim = H_v * dv;
    const float* y_b = y + b * V_dim;
    const float* z_b = z + b * V_dim;
    float* yg_b = y_gated_buf + b * V_dim;

    // Per-head RMSNormGated: normalize each (h, dv) row, multiply by silu(z)
    for (int h = threadIdx.x; h < H_v; h += blockDim.x) {
        float ssq = 0.0f;
        const float* y_h = y_b + h * dv;
        const float* z_h = z_b + h * dv;
        float* yg_h = yg_b + h * dv;
        for (int j = 0; j < dv; ++j) {
            ssq += y_h[j] * y_h[j];
        }
        float inv_rms = rsqrtf(ssq / float(dv) + rmsnorm_eps);
        for (int j = 0; j < dv; ++j) {
            float normed = y_h[j] * inv_rms * norm_w[j];
            yg_h[j] = normed * silu(z_h[j]);
        }
    }
    __syncthreads();

    // Output projection: out = yg @ w_out + x_residual
    float* out_b = out + b * D;
    const float* xr_b = x_residual + b * D;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float acc = xr_b[j];
        for (int i = 0; i < V_dim; ++i) {
            acc += yg_b[b * V_dim + i] * __half2float(w_out[j * V_dim + i]);
        }
        out_b[j] = acc;
    }
}

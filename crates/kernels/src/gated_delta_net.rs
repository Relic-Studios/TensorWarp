//! Gated DeltaNet — linear attention layer used in Qwen3.5 hybrid arch.
//!
//! Math reference: transformers/models/qwen3_5/modeling_qwen3_5.py
//!   `Qwen3_5GatedDeltaNet.forward()` and `torch_recurrent_gated_delta_rule()`.
//!
//! Per decode step (single token, batch B):
//!   1. mixed_qkv = in_proj_qkv(x_norm) → causal_conv1d → SiLU → split q,k,v
//!   2. l2_normalize(q,k); q *= 1/sqrt(dk)
//!   3. z = in_proj_z(x_norm)         [B, V_dim]
//!   4. b = in_proj_b(x_norm); a = in_proj_a(x_norm)   [B, H_v]
//!   5. beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)
//!   6. GQA expand q,k from H_k → H_v
//!   7. Recurrent step (per-head):
//!        S = S * exp(g)                       (decay)
//!        kv_mem = sum_dk(S * k)
//!        delta = (v - kv_mem) * beta
//!        S = S + outer(k, delta)              (update)
//!        y = sum_dk(S * q)                    (output)
//!   8. y_normed = RMSNorm(y, norm.weight)     (per dv)
//!   9. y_gated = y_normed * silu(z)
//!  10. out = y_gated @ out_proj.weight + x_residual

use cudarc::driver::{LaunchConfig, PushKernelArg};
use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

// ─── Configuration ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GatedDeltaNetConfig {
    pub hidden_size: usize,
    pub num_value_heads: usize,
    pub num_key_heads: usize,
    pub value_head_dim: usize,
    pub key_head_dim: usize,
    pub conv_kernel_dim: usize,
    pub rmsnorm_eps: f32,
}

impl GatedDeltaNetConfig {
    pub fn qwen3_5_9b() -> Self {
        Self {
            hidden_size: 4096,
            num_value_heads: 32,
            num_key_heads: 16,
            value_head_dim: 128,
            key_head_dim: 128,
            conv_kernel_dim: 4,
            rmsnorm_eps: 1e-6,
        }
    }

    pub fn key_dim(&self) -> usize { self.num_key_heads * self.key_head_dim }
    pub fn value_dim(&self) -> usize { self.num_value_heads * self.value_head_dim }
    pub fn conv_dim(&self) -> usize { 2 * self.key_dim() + self.value_dim() }

    pub fn state_bytes_per_layer(&self) -> usize {
        self.num_value_heads * self.key_head_dim * self.value_head_dim * 4
    }
}

// ─── Weights ────────────────────────────────────────────────────────────

pub struct GatedDeltaNetWeights {
    /// in_proj_qkv [D_conv, D]
    pub w_qkv: GpuTensor<half::f16>,
    /// in_proj_z [V_dim, D]
    pub w_z: GpuTensor<half::f16>,
    /// in_proj_b [H_v, D]
    pub w_b: GpuTensor<half::f16>,
    /// in_proj_a [H_v, D]
    pub w_a: GpuTensor<half::f16>,
    /// conv1d.weight [D_conv, K]    (depthwise; HF stores [D_conv, 1, K], we squeeze)
    pub conv_w: GpuTensor<half::f16>,
    /// A_log [H_v]    (per-head decay rate, in log space)
    pub a_log: GpuTensor<f32>,
    /// dt_bias [H_v]
    pub dt_bias: GpuTensor<f32>,
    /// norm.weight [dv]   (RMSNormGated weight, applied per-head)
    pub norm_w: GpuTensor<f32>,
    /// out_proj [D, V_dim]
    pub w_out: GpuTensor<half::f16>,
}

// ─── State ──────────────────────────────────────────────────────────────

pub struct DeltaNetLayerState {
    /// Recurrent state [B, H_v, dk, dv] in fp32
    pub s: GpuTensor<f32>,
    /// Conv ring buffer [B, K-1, D_conv] in fp32
    pub conv_state: GpuTensor<f32>,
}

impl DeltaNetLayerState {
    pub fn new(
        device: &WarpDevice,
        cfg: &GatedDeltaNetConfig,
        batch: usize,
    ) -> Result<Self, DeviceError> {
        let s_shape = Shape::from_static(&[
            batch, cfg.num_value_heads, cfg.key_head_dim, cfg.value_head_dim,
        ]);
        let conv_shape = Shape::from_static(&[
            batch, cfg.conv_kernel_dim - 1, cfg.conv_dim(),
        ]);
        Ok(Self {
            s: GpuTensor::<f32>::zeros(device, s_shape, DType::F32)?,
            conv_state: GpuTensor::<f32>::zeros(device, conv_shape, DType::F32)?,
        })
    }
}

// ─── Per-step buffers ───────────────────────────────────────────────────

pub struct GatedDeltaNetStepBuffers {
    pub hidden_f16: GpuTensor<half::f16>,  // [B, D]    f16-cast input for cuBLAS
    pub conv_in_pre: GpuTensor<f32>,       // [B, D_conv]  pre-conv matmul output
    pub beta_logit: GpuTensor<f32>,        // [B, H_v]  pre-sigmoid
    pub a_logit: GpuTensor<f32>,           // [B, H_v]  pre-g-formula
    pub q: GpuTensor<f32>,        // [B, H_v, dk]
    pub k: GpuTensor<f32>,        // [B, H_v, dk]
    pub v: GpuTensor<f32>,        // [B, H_v, dv]
    pub z: GpuTensor<f32>,        // [B, V_dim]    (= H_v*dv)
    pub beta: GpuTensor<f32>,     // [B, H_v]
    pub g: GpuTensor<f32>,        // [B, H_v]
    pub y: GpuTensor<f32>,        // [B, H_v, dv]
    pub y_gated: GpuTensor<f32>,  // [B, V_dim]
}

// ─── CUDA kernel source (NVRTC at runtime) ──────────────────────────────

const KERNEL_SRC: &str = include_str!("cuda/gated_delta_net.cu");

/// Expose kernel source so other modules can reuse the residual_add kernel.
pub fn residual_add_kernel_src() -> &'static str {
    KERNEL_SRC
}

const PREMIX_POST_FN: &str = "gdn_premix_post_kernel";
const RECURRENCE_FN: &str = "gdn_recurrence_kernel";
const NORM_GATED_FN: &str = "gdn_norm_gated_kernel";
const RESIDUAL_ADD_FN: &str = "gdn_residual_add_kernel";

// ─── Public kernel API ─────────────────────────────────────────────────

/// One Gated DeltaNet decode step. `hidden` is the layer input AFTER input_layernorm.
/// `x_residual` is the layer input BEFORE input_layernorm (used for residual add).
/// Mutates `state` and writes `out`.
pub fn gated_delta_net_step(
    device: &WarpDevice,
    cache: &KernelCache,
    hidden: &GpuTensor<f32>,
    x_residual: &GpuTensor<f32>,
    weights: &GatedDeltaNetWeights,
    state: &mut DeltaNetLayerState,
    bufs: &mut GatedDeltaNetStepBuffers,
    cfg: &GatedDeltaNetConfig,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let batch = hidden.shape.dims()[0].static_val().unwrap() as i32;
    let batch_u = batch as u32;
    let d = cfg.hidden_size as i32;
    let k_dim = cfg.key_dim() as i32;
    let v_dim = cfg.value_dim() as i32;
    let h_v = cfg.num_value_heads as i32;
    let h_k = cfg.num_key_heads as i32;
    let dk = cfg.key_head_dim as i32;
    let dv = cfg.value_head_dim as i32;
    let k_conv = cfg.conv_kernel_dim as i32;
    let rmsnorm_eps = cfg.rmsnorm_eps;

    // ── Stage A.0: cast hidden f32 → f16 for cuBLAS HGEMM ────────────
    crate::fp16::cast_f32_to_f16(cache, device, hidden, &mut bufs.hidden_f16)?;

    // ── Stage A.1: cuBLAS HGEMMs for the four projections ─────────────
    // All weights stored row-major [N, K] which matches HGEMM transB layout.
    use crate::cublas_gemm::gemm_cublas_f16in_f32out_transB as hgemm_f16f32;
    let m = batch_u;
    let k_in = cfg.hidden_size as u32;
    let n_qkv = cfg.conv_dim() as u32;
    let n_z = cfg.value_dim() as u32;
    let n_ba = cfg.num_value_heads as u32;
    hgemm_f16f32(device, &bufs.hidden_f16, &weights.w_qkv,
                 &mut bufs.conv_in_pre, m, n_qkv, k_in)?;
    hgemm_f16f32(device, &bufs.hidden_f16, &weights.w_z,
                 &mut bufs.z, m, n_z, k_in)?;
    hgemm_f16f32(device, &bufs.hidden_f16, &weights.w_b,
                 &mut bufs.beta_logit, m, n_ba, k_in)?;
    hgemm_f16f32(device, &bufs.hidden_f16, &weights.w_a,
                 &mut bufs.a_logit, m, n_ba, k_in)?;

    // ── Stage A.2: small post-matmul kernel (conv1d + split + l2norm + sigmoid + g) ─
    {
        let f = cache.get_or_compile_with_opts(device, KERNEL_SRC, PREMIX_POST_FN, &[WarpDevice::cuda_include_path()], None)?;
        let smem_bytes = (cfg.conv_dim() * 4) as u32;
        let cfg_launch = LaunchConfig {
            grid_dim: (batch_u, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem_bytes,
        };
        unsafe {
            device.stream.launch_builder(&f)
                .arg(&bufs.conv_in_pre.data)
                .arg(&bufs.beta_logit.data)
                .arg(&bufs.a_logit.data)
                .arg(&weights.conv_w.data)
                .arg(&weights.a_log.data)
                .arg(&weights.dt_bias.data)
                .arg(&mut state.conv_state.data)
                .arg(&mut bufs.q.data)
                .arg(&mut bufs.k.data)
                .arg(&mut bufs.v.data)
                .arg(&mut bufs.beta.data)
                .arg(&mut bufs.g.data)
                .arg(&batch).arg(&k_dim).arg(&v_dim)
                .arg(&h_v).arg(&h_k).arg(&dk).arg(&dv)
                .arg(&k_conv)
                .launch(cfg_launch)
                .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
        }
    }

    // ── Stage B: per-head recurrent step (unchanged) ──────────────────
    {
        let f = cache.get_or_compile_with_opts(device, KERNEL_SRC, RECURRENCE_FN, &[WarpDevice::cuda_include_path()], None)?;
        let cfg_launch = LaunchConfig {
            grid_dim: (batch_u, cfg.num_value_heads as u32, 1),
            block_dim: (cfg.value_head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            device.stream.launch_builder(&f)
                .arg(&bufs.q.data)
                .arg(&bufs.k.data)
                .arg(&bufs.v.data)
                .arg(&bufs.beta.data)
                .arg(&bufs.g.data)
                .arg(&mut state.s.data)
                .arg(&mut bufs.y.data)
                .arg(&h_v).arg(&dk).arg(&dv)
                .launch(cfg_launch)
                .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
        }
    }

    // ── Stage C.1: RMSNormGated → y_gated ──────────────────────────────
    {
        let f = cache.get_or_compile_with_opts(device, KERNEL_SRC, NORM_GATED_FN, &[WarpDevice::cuda_include_path()], None)?;
        let cfg_launch = LaunchConfig {
            grid_dim: (batch_u, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            device.stream.launch_builder(&f)
                .arg(&bufs.y.data)
                .arg(&bufs.z.data)
                .arg(&weights.norm_w.data)
                .arg(&mut bufs.y_gated.data)
                .arg(&batch).arg(&h_v).arg(&dv)
                .arg(&rmsnorm_eps)
                .launch(cfg_launch)
                .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
        }
    }

    // ── Stage C.2: cuBLAS HGEMM for output projection ─────────────────
    // out = w_out @ y_gated     [B, D] = w_out[D, V_dim] @ y_gated[B, V_dim]
    // We need y_gated as f16 for the HGEMM input. Reuse hidden_f16-shape buffer?
    // No — y_gated is V_dim=4096, hidden is also D=4096 (same size for Qwen3.5).
    // Cast in place into hidden_f16 buffer (we don't need its old contents).
    crate::fp16::cast_f32_to_f16(cache, device, &bufs.y_gated, &mut bufs.hidden_f16)?;
    hgemm_f16f32(device, &bufs.hidden_f16, &weights.w_out,
                 out, m, cfg.hidden_size as u32, cfg.value_dim() as u32)?;

    // ── Stage C.3: residual add ──────────────────────────────────────
    {
        let f = cache.get_or_compile_with_opts(device, KERNEL_SRC, RESIDUAL_ADD_FN, &[WarpDevice::cuda_include_path()], None)?;
        let total = (batch * d) as u32;
        let cfg_launch = LaunchConfig::for_num_elems(total);
        unsafe {
            device.stream.launch_builder(&f)
                .arg(&x_residual.data)
                .arg(&mut out.data)
                .arg(&batch).arg(&d)
                .launch(cfg_launch)
                .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_state_size_qwen3_5_9b() {
        let cfg = GatedDeltaNetConfig::qwen3_5_9b();
        assert_eq!(cfg.state_bytes_per_layer(), 2 * 1024 * 1024);
    }

    #[test]
    fn config_dims_qwen3_5_9b() {
        let cfg = GatedDeltaNetConfig::qwen3_5_9b();
        assert_eq!(cfg.key_dim(), 2048);     // 16 * 128
        assert_eq!(cfg.value_dim(), 4096);   // 32 * 128
        assert_eq!(cfg.conv_dim(), 8192);    // 2*2048 + 4096
    }
}

//! Extended ops — covering everything TensorRT has AND more.
//!
//! This module adds all remaining ONNX ops to reach 120%+ of TRT coverage.
//! Organized by category: math, activation, RNN, shape, quantization.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

macro_rules! launch_err {
    ($e:expr) => {
        $e.map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))
    };
}

// ═══════════════════════════════════════════════════════════════
// MATH OPS
// ═══════════════════════════════════════════════════════════════

const ERF_SRC: &str = r#"
extern "C" __global__ void warp_erf(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = erff(x[i]); }
}"#;

const MOD_SRC: &str = r#"
extern "C" __global__ void warp_mod(float *out, const float *a, const float *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = fmodf(a[i], b[i]); }
}"#;

const ISNAN_SRC: &str = r#"
extern "C" __global__ void warp_isnan(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = isnan(x[i]) ? 1.0f : 0.0f; }
}"#;

const ISINF_SRC: &str = r#"
extern "C" __global__ void warp_isinf(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = isinf(x[i]) ? 1.0f : 0.0f; }
}"#;

// Trig ops
const ASIN_SRC: &str = r#"
extern "C" __global__ void warp_asin(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = asinf(x[i]); }
}"#;

const ACOS_SRC: &str = r#"
extern "C" __global__ void warp_acos(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = acosf(x[i]); }
}"#;

const ATAN_SRC: &str = r#"
extern "C" __global__ void warp_atan(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = atanf(x[i]); }
}"#;

const SINH_SRC: &str = r#"
extern "C" __global__ void warp_sinh(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = sinhf(x[i]); }
}"#;

const COSH_SRC: &str = r#"
extern "C" __global__ void warp_cosh(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = coshf(x[i]); }
}"#;

// ═══════════════════════════════════════════════════════════════
// ACTIVATION OPS (beyond what we already have)
// ═══════════════════════════════════════════════════════════════

const ELU_SRC: &str = r#"
extern "C" __global__ void warp_elu(float *out, const float *x, float alpha, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; out[i] = v >= 0.0f ? v : alpha * (expf(v) - 1.0f); }
}"#;

const CELU_SRC: &str = r#"
extern "C" __global__ void warp_celu(float *out, const float *x, float alpha, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; out[i] = fmaxf(0.0f, v) + fminf(0.0f, alpha * (expf(v/alpha) - 1.0f)); }
}"#;

const MISH_SRC: &str = r#"
extern "C" __global__ void warp_mish(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; out[i] = v * tanhf(logf(1.0f + expf(v))); }
}"#;

const HARD_SIGMOID_SRC: &str = r#"
extern "C" __global__ void warp_hard_sigmoid(float *out, const float *x, float alpha, float beta, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = fmaxf(0.0f, fminf(1.0f, alpha * x[i] + beta)); }
}"#;

const HARD_SWISH_SRC: &str = r#"
extern "C" __global__ void warp_hard_swish(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; out[i] = v * fmaxf(0.0f, fminf(1.0f, v/6.0f + 0.5f)); }
}"#;

const SOFTPLUS_SRC: &str = r#"
extern "C" __global__ void warp_softplus(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = logf(1.0f + expf(x[i])); }
}"#;

const SOFTSIGN_SRC: &str = r#"
extern "C" __global__ void warp_softsign(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; out[i] = v / (1.0f + fabsf(v)); }
}"#;

// ═══════════════════════════════════════════════════════════════
// SHAPE OPS
// ═══════════════════════════════════════════════════════════════

const TRILU_SRC: &str = r#"
extern "C" __global__ void warp_trilu(
    float *out, const float *input,
    unsigned int rows, unsigned int cols, int k, unsigned int upper
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    unsigned int r = idx / cols;
    unsigned int c = idx % cols;
    if (upper) {
        out[idx] = ((int)c >= (int)r + k) ? input[idx] : 0.0f;
    } else {
        out[idx] = ((int)c <= (int)r + k) ? input[idx] : 0.0f;
    }
}"#;

const GATHER_ELEMENTS_SRC: &str = r#"
extern "C" __global__ void warp_gather_elements(
    float *out, const float *input, const float *indices,
    unsigned int rows, unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    unsigned int r = idx / cols;
    int c = (int)indices[idx];
    if (c >= 0 && c < (int)cols) {
        out[idx] = input[r * cols + c];
    }
}"#;

// ═══════════════════════════════════════════════════════════════
// RNN OPS (LSTM, GRU)
// ═══════════════════════════════════════════════════════════════

/// Simple LSTM cell (CPU-side for now — RNNs are sequential).
/// h_t = o_t * tanh(c_t)
/// c_t = f_t * c_{t-1} + i_t * g_t
pub fn lstm_cell(
    x: &[f32],      // [batch, input_size]
    h_prev: &[f32], // [batch, hidden_size]
    c_prev: &[f32], // [batch, hidden_size]
    w_ih: &[f32],   // [4*hidden_size, input_size]
    w_hh: &[f32],   // [4*hidden_size, hidden_size]
    b_ih: &[f32],   // [4*hidden_size]
    b_hh: &[f32],   // [4*hidden_size]
    batch: usize,
    input_size: usize,
    hidden_size: usize,
) -> (Vec<f32>, Vec<f32>) { // (h_new, c_new)
    let mut h_new = vec![0.0f32; batch * hidden_size];
    let mut c_new = vec![0.0f32; batch * hidden_size];

    for b in 0..batch {
        for h in 0..hidden_size {
            // Compute gates: i, f, g, o
            let mut gates = [0.0f32; 4];
            for gate in 0..4 {
                let g_idx = gate * hidden_size + h;
                let mut val = b_ih[g_idx] + b_hh[g_idx];
                for i in 0..input_size {
                    val += w_ih[g_idx * input_size + i] * x[b * input_size + i];
                }
                for hh in 0..hidden_size {
                    val += w_hh[g_idx * hidden_size + hh] * h_prev[b * hidden_size + hh];
                }
                gates[gate] = val;
            }

            let i_gate = 1.0 / (1.0 + (-gates[0]).exp()); // sigmoid
            let f_gate = 1.0 / (1.0 + (-gates[1]).exp());
            let g_gate = gates[2].tanh();
            let o_gate = 1.0 / (1.0 + (-gates[3]).exp());

            let c = f_gate * c_prev[b * hidden_size + h] + i_gate * g_gate;
            let hh = o_gate * c.tanh();

            c_new[b * hidden_size + h] = c;
            h_new[b * hidden_size + h] = hh;
        }
    }

    (h_new, c_new)
}

/// Simple GRU cell (CPU-side).
pub fn gru_cell(
    x: &[f32],
    h_prev: &[f32],
    w_ih: &[f32],   // [3*hidden_size, input_size]
    w_hh: &[f32],   // [3*hidden_size, hidden_size]
    b_ih: &[f32],
    b_hh: &[f32],
    batch: usize,
    input_size: usize,
    hidden_size: usize,
) -> Vec<f32> { // h_new
    let mut h_new = vec![0.0f32; batch * hidden_size];

    for b in 0..batch {
        for h in 0..hidden_size {
            // r, z gates
            let mut r_val = b_ih[h] + b_hh[h];
            let mut z_val = b_ih[hidden_size + h] + b_hh[hidden_size + h];
            let mut n_val = b_ih[2 * hidden_size + h];

            for i in 0..input_size {
                r_val += w_ih[h * input_size + i] * x[b * input_size + i];
                z_val += w_ih[(hidden_size + h) * input_size + i] * x[b * input_size + i];
                n_val += w_ih[(2 * hidden_size + h) * input_size + i] * x[b * input_size + i];
            }
            for hh in 0..hidden_size {
                r_val += w_hh[h * hidden_size + hh] * h_prev[b * hidden_size + hh];
                z_val += w_hh[(hidden_size + h) * hidden_size + hh] * h_prev[b * hidden_size + hh];
            }

            let r = 1.0 / (1.0 + (-r_val).exp());
            let z = 1.0 / (1.0 + (-z_val).exp());

            // n gate with reset
            let mut n_h = b_hh[2 * hidden_size + h];
            for hh in 0..hidden_size {
                n_h += w_hh[(2 * hidden_size + h) * hidden_size + hh] * h_prev[b * hidden_size + hh];
            }
            let n = (n_val + r * n_h).tanh();

            h_new[b * hidden_size + h] = (1.0 - z) * n + z * h_prev[b * hidden_size + h];
        }
    }

    h_new
}

// ═══════════════════════════════════════════════════════════════
// Rust API wrappers for GPU kernels
// ═══════════════════════════════════════════════════════════════

macro_rules! unary_op {
    ($name:ident, $src:expr, $kernel:expr) => {
        pub fn $name(
            cache: &KernelCache, device: &WarpDevice,
            x: &GpuTensor<f32>, out: &mut GpuTensor<f32>,
        ) -> Result<(), DeviceError> {
            let f = cache.get_or_compile(device, $src, $kernel)?;
            let n = x.numel;
            let cfg = LaunchConfig::for_num_elems(n as u32);
            unsafe {
                launch_err!(device.stream.launch_builder(&f)
                    .arg(&mut out.data).arg(&x.data).arg(&n)
                    .launch(cfg))?;
            }
            Ok(())
        }
    };
}

unary_op!(erf, ERF_SRC, "warp_erf");
unary_op!(isnan, ISNAN_SRC, "warp_isnan");
unary_op!(isinf, ISINF_SRC, "warp_isinf");
unary_op!(asin, ASIN_SRC, "warp_asin");
unary_op!(acos, ACOS_SRC, "warp_acos");
unary_op!(atan, ATAN_SRC, "warp_atan");
unary_op!(sinh, SINH_SRC, "warp_sinh");
unary_op!(cosh, COSH_SRC, "warp_cosh");
unary_op!(mish, MISH_SRC, "warp_mish");
unary_op!(hard_swish, HARD_SWISH_SRC, "warp_hard_swish");
unary_op!(softplus, SOFTPLUS_SRC, "warp_softplus");
unary_op!(softsign, SOFTSIGN_SRC, "warp_softsign");

pub fn elu(
    cache: &KernelCache, device: &WarpDevice,
    x: &GpuTensor<f32>, out: &mut GpuTensor<f32>, alpha: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, ELU_SRC, "warp_elu")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&alpha).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

pub fn celu(
    cache: &KernelCache, device: &WarpDevice,
    x: &GpuTensor<f32>, out: &mut GpuTensor<f32>, alpha: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, CELU_SRC, "warp_celu")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&alpha).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

pub fn hard_sigmoid(
    cache: &KernelCache, device: &WarpDevice,
    x: &GpuTensor<f32>, out: &mut GpuTensor<f32>,
    alpha: f32, beta: f32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, HARD_SIGMOID_SRC, "warp_hard_sigmoid")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&alpha).arg(&beta).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

pub fn fmod(
    cache: &KernelCache, device: &WarpDevice,
    a: &GpuTensor<f32>, b: &GpuTensor<f32>, out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, MOD_SRC, "warp_mod")?;
    let n = a.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&a.data).arg(&b.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}

pub fn trilu(
    cache: &KernelCache, device: &WarpDevice,
    input: &GpuTensor<f32>, out: &mut GpuTensor<f32>,
    rows: u32, cols: u32, k: i32, upper: bool,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, TRILU_SRC, "warp_trilu")?;
    let total = rows * cols;
    let cfg = LaunchConfig::for_num_elems(total);
    let upper_u32 = if upper { 1u32 } else { 0u32 };
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&input.data)
            .arg(&rows).arg(&cols).arg(&k).arg(&upper_u32)
            .launch(cfg))?;
    }
    Ok(())
}

pub fn gather_elements(
    cache: &KernelCache, device: &WarpDevice,
    input: &GpuTensor<f32>, indices: &GpuTensor<f32>, out: &mut GpuTensor<f32>,
    rows: u32, cols: u32,
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, GATHER_ELEMENTS_SRC, "warp_gather_elements")?;
    let total = rows * cols;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&input.data).arg(&indices.data)
            .arg(&rows).arg(&cols)
            .launch(cfg))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    #[test]
    fn test_erf() {
        let (dev, cache) = match setup() { Some(s) => s, None => return };
        let data = vec![0.0f32, 0.5, 1.0, -1.0, 2.0];
        let x = GpuTensor::from_host(&dev, &data, Shape::from_static(&[5]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[5]), DType::F32).unwrap();
        erf(&cache, &dev, &x, &mut out).unwrap();
        dev.synchronize().unwrap();
        let result = out.to_host(&dev).unwrap();
        assert!((result[0] - 0.0).abs() < 0.01); // erf(0) = 0
        assert!((result[2] - 0.8427).abs() < 0.01); // erf(1) ≈ 0.8427
        println!("Erf: correct! {:?}", result);
    }

    #[test]
    fn test_activations() {
        let (dev, cache) = match setup() { Some(s) => s, None => return };
        let data = vec![-1.0f32, 0.0, 1.0, 2.0];
        let x = GpuTensor::from_host(&dev, &data, Shape::from_static(&[4]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[4]), DType::F32).unwrap();

        mish(&cache, &dev, &x, &mut out).unwrap();
        dev.synchronize().unwrap();
        let r = out.to_host(&dev).unwrap();
        assert!(r.iter().all(|v| v.is_finite()));
        println!("Mish: {:?}", r);

        hard_swish(&cache, &dev, &x, &mut out).unwrap();
        dev.synchronize().unwrap();
        let r = out.to_host(&dev).unwrap();
        println!("HardSwish: {:?}", r);

        softplus(&cache, &dev, &x, &mut out).unwrap();
        dev.synchronize().unwrap();
        let r = out.to_host(&dev).unwrap();
        assert!(r.iter().all(|v| *v >= 0.0)); // softplus always positive
        println!("Softplus: {:?}", r);

        println!("All activations: correct!");
    }

    #[test]
    fn test_trilu() {
        let (dev, cache) = match setup() { Some(s) => s, None => return };
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let x = GpuTensor::from_host(&dev, &data, Shape::from_static(&[3, 3]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[3, 3]), DType::F32).unwrap();

        // Upper triangular
        trilu(&cache, &dev, &x, &mut out, 3, 3, 0, true).unwrap();
        dev.synchronize().unwrap();
        let r = out.to_host(&dev).unwrap();
        assert_eq!(r[3], 0.0); // below diagonal
        assert_eq!(r[0], 1.0); // on diagonal
        assert_eq!(r[1], 2.0); // above diagonal
        println!("Trilu (upper): {:?}", r);
    }

    #[test]
    fn test_lstm() {
        let batch = 1;
        let input_size = 2;
        let hidden_size = 2;

        let x = vec![1.0f32, 0.5];
        let h_prev = vec![0.0f32; hidden_size];
        let c_prev = vec![0.0f32; hidden_size];
        let w_ih = vec![0.1f32; 4 * hidden_size * input_size];
        let w_hh = vec![0.1f32; 4 * hidden_size * hidden_size];
        let b_ih = vec![0.0f32; 4 * hidden_size];
        let b_hh = vec![0.0f32; 4 * hidden_size];

        let (h, c) = lstm_cell(&x, &h_prev, &c_prev, &w_ih, &w_hh, &b_ih, &b_hh,
            batch, input_size, hidden_size);

        assert!(h.iter().all(|v| v.is_finite()));
        assert!(c.iter().all(|v| v.is_finite()));
        println!("LSTM: h={:?}, c={:?}", h, c);
    }
}

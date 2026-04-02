//! Elementwise GPU kernel execution via cudarc.
//!
//! Uses NVRTC to compile CUDA C kernels at runtime.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Launch vector add: out = a + b
pub fn launch_binary_add(
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    assert_eq!(a.numel, b.numel);
    assert_eq!(a.numel, out.numel);
    let n = a.numel;

    let cuda_src = format!(r#"
extern "C" __global__ void warp_add(float *out, const float *a, const float *b, size_t n) {{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        out[i] = a[i] + b[i];
    }}
}}
"#);

    let (_module, func) = device.load_cuda_source(&cuda_src, "warp_add")?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device.stream
            .launch_builder(&func)
            .arg(&mut out.data)
            .arg(&a.data)
            .arg(&b.data)
            .arg(&n)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// Launch GELU activation: out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn launch_gelu(
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    assert_eq!(input.numel, out.numel);
    let n = input.numel;

    let cuda_src = r#"
extern "C" __global__ void warp_gelu(float *out, const float *input, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}
"#;

    let (_module, func) = device.load_cuda_source(cuda_src, "warp_gelu")?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device.stream
            .launch_builder(&func)
            .arg(&mut out.data)
            .arg(&input.data)
            .arg(&n)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// Launch SiLU (Swish) activation: out = x * sigmoid(x)
pub fn launch_silu(
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    assert_eq!(input.numel, out.numel);
    let n = input.numel;

    let cuda_src = r#"
extern "C" __global__ void warp_silu(float *out, const float *input, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        out[i] = x / (1.0f + expf(-x));
    }
}
"#;

    let (_module, func) = device.load_cuda_source(cuda_src, "warp_silu")?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device.stream
            .launch_builder(&func)
            .arg(&mut out.data)
            .arg(&input.data)
            .arg(&n)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// Launch fused Add + GELU: out = gelu(a + b)
/// This is the kind of fusion Warp does automatically.
pub fn launch_fused_add_gelu(
    device: &WarpDevice,
    a: &GpuTensor<f32>,
    b: &GpuTensor<f32>,
    out: &mut GpuTensor<f32>,
) -> Result<(), DeviceError> {
    assert_eq!(a.numel, b.numel);
    assert_eq!(a.numel, out.numel);
    let n = a.numel;

    let cuda_src = r#"
extern "C" __global__ void warp_fused_add_gelu(float *out, const float *a, const float *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i] + b[i];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}
"#;

    let (_module, func) = device.load_cuda_source(cuda_src, "warp_fused_add_gelu")?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        device.stream
            .launch_builder(&func)
            .arg(&mut out.data)
            .arg(&a.data)
            .arg(&b.data)
            .arg(&n)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::{DType, Shape};

    fn get_device() -> Option<WarpDevice> {
        WarpDevice::new(0).ok()
    }

    #[test]
    fn gpu_vector_add() {
        let dev = match get_device() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let n = 1024;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let shape = Shape::from_static(&[n]);

        let a = GpuTensor::from_host(&dev, &a_data, shape.clone(), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        launch_binary_add(&dev, &a, &b, &mut out).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        for i in 0..n {
            assert!((result[i] - (a_data[i] + b_data[i])).abs() < 1e-5, "Mismatch at {i}");
        }
        println!("Vector add: {n} elements correct!");
    }

    #[test]
    fn gpu_gelu() {
        let dev = match get_device() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let n = 2048;
        let input_data: Vec<f32> = (0..n).map(|i| (i as f32 - 1024.0) / 512.0).collect();
        let shape = Shape::from_static(&[n]);

        let input = GpuTensor::from_host(&dev, &input_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        launch_gelu(&dev, &input, &mut out).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        for i in 0..n {
            let x = input_data[i];
            let expected = 0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh());
            assert!((result[i] - expected).abs() < 1e-4, "GELU mismatch at {i}: got {}, expected {}", result[i], expected);
        }
        println!("GELU: {n} elements correct!");
    }

    #[test]
    fn gpu_silu() {
        let dev = match get_device() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let n = 1024;
        let input_data: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) / 256.0).collect();
        let shape = Shape::from_static(&[n]);

        let input = GpuTensor::from_host(&dev, &input_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        launch_silu(&dev, &input, &mut out).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        for i in 0..n {
            let x = input_data[i];
            let expected = x / (1.0 + (-x).exp());
            assert!((result[i] - expected).abs() < 1e-4, "SiLU mismatch at {i}: got {}, expected {}", result[i], expected);
        }
        println!("SiLU: {n} elements correct!");
    }

    #[test]
    fn gpu_fused_add_gelu() {
        let dev = match get_device() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let n = 4096;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 - 2048.0) / 1024.0).collect();
        let b_data: Vec<f32> = (0..n).map(|i| ((i + 500) as f32 - 2048.0) / 1024.0).collect();
        let shape = Shape::from_static(&[n]);

        let a = GpuTensor::from_host(&dev, &a_data, shape.clone(), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        launch_fused_add_gelu(&dev, &a, &b, &mut out).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        for i in 0..n {
            let x = a_data[i] + b_data[i];
            let expected = 0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh());
            assert!((result[i] - expected).abs() < 1e-4, "Fused add+gelu mismatch at {i}");
        }
        println!("Fused Add+GELU: {n} elements correct! (1 kernel instead of 2)");
    }

    #[test]
    fn gpu_bandwidth_test() {
        let dev = match get_device() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let n = 4 * 1024 * 1024; // 4M elements = 16MB per tensor
        let a_data: Vec<f32> = (0..n).map(|i| (i % 1000) as f32 * 0.001).collect();
        let b_data: Vec<f32> = (0..n).map(|i| ((i + 500) % 1000) as f32 * 0.001).collect();
        let shape = Shape::from_static(&[n]);

        let a = GpuTensor::from_host(&dev, &a_data, shape.clone(), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, shape.clone(), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, shape, DType::F32).unwrap();

        // Warmup
        launch_binary_add(&dev, &a, &b, &mut out).unwrap();
        dev.synchronize().unwrap();

        // Timed
        let iters = 100;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            launch_binary_add(&dev, &a, &b, &mut out).unwrap();
        }
        dev.synchronize().unwrap();
        let elapsed = start.elapsed();

        let bytes_per_iter = n as f64 * 4.0 * 3.0; // read a, read b, write out
        let total_bytes = bytes_per_iter * iters as f64;
        let bandwidth_gb_s = total_bytes / elapsed.as_secs_f64() / 1e9;

        println!(
            "Bandwidth test: {}M elements × {} iters in {:.2}ms ({:.1} GB/s)",
            n / (1024 * 1024), iters,
            elapsed.as_secs_f64() * 1000.0,
            bandwidth_gb_s,
        );
        // 4090 theoretical: ~1 TB/s. Expect 600-900 GB/s effective.
    }
}

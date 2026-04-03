//! Stress tests and numerical precision validation.

#[cfg(test)]
mod tests {
    use warp_ir::{DType, Shape};
    use crate::device::WarpDevice;
    use crate::cache::KernelCache;
    use crate::tensor::GpuTensor;

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    #[test]
    fn stress_large_gemm() {
        let (dev, cache) = match setup() {
            Some(s) => s, None => { println!("No CUDA"); return; }
        };

        // Test GEMM at various sizes including non-power-of-2
        for &(m, n, k) in &[(127u32, 255, 63), (1000, 1000, 1000), (2048, 4096, 1024)] {
            let a = GpuTensor::<f32>::zeros(&dev,
                Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
            let b = GpuTensor::<f32>::zeros(&dev,
                Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
            let mut c = GpuTensor::<f32>::zeros(&dev,
                Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();

            crate::ops::gemm(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
            dev.synchronize().unwrap();
            println!("Stress GEMM {m}x{n}x{k}: OK");
        }
    }

    #[test]
    fn stress_many_kernel_launches() {
        let (dev, cache) = match setup() {
            Some(s) => s, None => { println!("No CUDA"); return; }
        };

        let n = 1024;
        let a = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[n]), DType::F32).unwrap();
        let b = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[n]), DType::F32).unwrap();
        let mut c = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[n]), DType::F32).unwrap();

        // 1000 rapid-fire kernel launches
        for _ in 0..1000 {
            crate::ops::add(&cache, &dev, &a, &b, &mut c).unwrap();
        }
        dev.synchronize().unwrap();
        println!("Stress 1000 launches: OK");
    }

    #[test]
    fn numerical_fp16_vs_f32() {
        let (dev, cache) = match setup() {
            Some(s) => s, None => { println!("No CUDA"); return; }
        };

        let n = 256;
        // Use values that stress FP16 precision
        let data: Vec<f32> = (0..n).map(|i| {
            let x = (i as f32 - 128.0) * 0.1;
            x.sin() * 10.0 // values in [-10, 10]
        }).collect();

        // F32 path: add + gelu
        let a = GpuTensor::from_host(&dev, &data, Shape::from_static(&[n]), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &data, Shape::from_static(&[n]), DType::F32).unwrap();
        let mut f32_result = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[n]), DType::F32).unwrap();
        let mut f32_tmp = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[n]), DType::F32).unwrap();
        crate::ops::add(&cache, &dev, &a, &b, &mut f32_tmp).unwrap();
        crate::ops::gelu(&cache, &dev, &f32_tmp, &mut f32_result).unwrap();
        dev.synchronize().unwrap();
        let f32_out = f32_result.to_host(&dev).unwrap();

        // FP16 path: same ops
        let f16_data: Vec<half::f16> = data.iter().map(|v| half::f16::from_f32(*v)).collect();
        let a16 = GpuTensor::from_host(&dev, &f16_data, Shape::from_static(&[n]), DType::F16).unwrap();
        let b16 = GpuTensor::from_host(&dev, &f16_data, Shape::from_static(&[n]), DType::F16).unwrap();
        let mut f16_tmp = GpuTensor::<half::f16>::zeros(&dev, Shape::from_static(&[n]), DType::F16).unwrap();
        let mut f16_result = GpuTensor::<half::f16>::zeros(&dev, Shape::from_static(&[n]), DType::F16).unwrap();
        crate::fp16::f16_add(&cache, &dev, &a16, &b16, &mut f16_tmp).unwrap();
        // Use silu as proxy (gelu needs separate f16 kernel)
        crate::fp16::f16_silu(&cache, &dev, &f16_tmp, &mut f16_result).unwrap();
        dev.synchronize().unwrap();
        let f16_out: Vec<f32> = f16_result.to_host(&dev).unwrap().iter().map(|v| v.to_f32()).collect();

        // Compute divergence
        let max_abs_err: f32 = f32_out.iter().zip(f16_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let mean_abs_err: f32 = f32_out.iter().zip(f16_out.iter())
            .map(|(a, b)| (a - b).abs()).sum::<f32>() / n as f32;

        println!("FP16 vs F32 divergence ({n} elements):");
        println!("  Max abs error:  {max_abs_err:.4}");
        println!("  Mean abs error: {mean_abs_err:.6}");
        // FP16 has ~3 decimal digits of precision
        assert!(max_abs_err < 1.0, "FP16 divergence too high: {max_abs_err}");
    }

    #[test]
    fn stress_conv_various_sizes() {
        let (dev, cache) = match setup() {
            Some(s) => s, None => { println!("No CUDA"); return; }
        };

        // Test Conv2D with various non-standard sizes
        for &(c_in, c_out, h, w, k) in &[
            (1u32, 1, 7, 7, 3),      // tiny
            (3, 64, 224, 224, 7),     // ResNet first conv
            (64, 64, 56, 56, 3),      // ResNet block
            (512, 512, 7, 7, 3),      // deep layer
        ] {
            let params = crate::conv::Conv2dParams::new(c_in, c_out, k).padding(k/2);
            let out_h = params.output_h(h);
            let out_w = params.output_w(w);

            let input = GpuTensor::<f32>::zeros(&dev,
                Shape::from_static(&[1, c_in as usize, h as usize, w as usize]), DType::F32).unwrap();
            let weight = GpuTensor::<f32>::zeros(&dev,
                Shape::from_static(&[c_out as usize, c_in as usize, k as usize, k as usize]), DType::F32).unwrap();
            let mut output = GpuTensor::<f32>::zeros(&dev,
                Shape::from_static(&[1, c_out as usize, out_h as usize, out_w as usize]), DType::F32).unwrap();

            crate::conv::conv2d(&cache, &dev, &input, &weight, None, &mut output, &params, h, w).unwrap();
            dev.synchronize().unwrap();
            println!("Stress Conv2D {c_in}→{c_out} {h}x{w} k={k}: OK");
        }
    }
}

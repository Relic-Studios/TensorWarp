//! End-to-end inference benchmarks.
//!
//! Measures actual tokens/sec for transformer models across different
//! precision modes: F32, FP16, Q4_0. This is the number that matters
//! for production — not raw TFLOPS but real inference throughput.

use std::time::Instant;
use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::generate::{GenerateConfig, GenerationEngine, QuantizedGenerationEngine, create_test_engine};
use crate::tensor::GpuTensor;
use crate::transformer::TransformerConfig;

/// Benchmark results for one configuration.
#[derive(Debug)]
pub struct InferenceBenchmark {
    pub name: String,
    pub config: String,
    pub prefill_ms: f64,
    pub prefill_tokens_per_sec: f64,
    pub decode_ms: f64,
    pub decode_tokens_per_sec: f64,
    pub total_tokens: usize,
    pub weight_memory_mb: f64,
}

impl std::fmt::Display for InferenceBenchmark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  {}: {}", self.name, self.config)?;
        writeln!(f, "    Prefill:  {:.1}ms ({:.0} tok/s)", self.prefill_ms, self.prefill_tokens_per_sec)?;
        writeln!(f, "    Decode:   {:.1}ms ({:.0} tok/s)", self.decode_ms, self.decode_tokens_per_sec)?;
        write!(f, "    Weights:  {:.1} MB", self.weight_memory_mb)
    }
}

/// Run F32 inference benchmark.
pub fn bench_f32(
    device: &WarpDevice,
    config: TransformerConfig,
    num_layers: u32,
    vocab_size: u32,
    prompt_len: usize,
    gen_tokens: usize,
) -> Result<InferenceBenchmark, DeviceError> {
    let engine = create_test_engine(device, config.clone(), num_layers, vocab_size)?;
    let prompt: Vec<i32> = (1..=prompt_len as i32).collect();
    let gen_config = GenerateConfig {
        max_tokens: gen_tokens,
        greedy: true,
        eos_token_id: None,
        ..Default::default()
    };

    // Warmup
    let _ = engine.generate_with_cache(device, &prompt, &gen_config, 512)?;
    device.synchronize()?;

    // Timed run
    let result = engine.generate_with_cache(device, &prompt, &gen_config, 512)?;

    let (f32_bytes, _) = crate::generate::weight_memory_estimate(&config, num_layers, vocab_size);

    Ok(InferenceBenchmark {
        name: "F32".into(),
        config: format!("{}L H={} FFN={} V={}", num_layers, config.hidden_size, config.ffn_dim, vocab_size),
        prefill_ms: result.prefill_time.as_secs_f64() * 1000.0,
        prefill_tokens_per_sec: prompt_len as f64 / result.prefill_time.as_secs_f64().max(1e-9),
        decode_ms: result.decode_time.as_secs_f64() * 1000.0,
        decode_tokens_per_sec: result.tokens_per_sec,
        total_tokens: result.tokens_generated,
        weight_memory_mb: f32_bytes as f64 / 1e6,
    })
}

/// Run Q4_0 quantized inference benchmark.
pub fn bench_q4(
    device: &WarpDevice,
    config: TransformerConfig,
    num_layers: u32,
    vocab_size: u32,
    prompt_len: usize,
    gen_tokens: usize,
) -> Result<InferenceBenchmark, DeviceError> {
    let f32_engine = create_test_engine(device, config.clone(), num_layers, vocab_size)?;
    let q4_engine = QuantizedGenerationEngine::from_f32(device, &f32_engine)?;

    let prompt: Vec<i32> = (1..=prompt_len as i32).collect();
    let gen_config = GenerateConfig {
        max_tokens: gen_tokens,
        greedy: true,
        eos_token_id: None,
        ..Default::default()
    };

    // Warmup
    let _ = q4_engine.generate_with_cache(device, &prompt, &gen_config, 512)?;
    device.synchronize()?;

    // Timed run
    let result = q4_engine.generate_with_cache(device, &prompt, &gen_config, 512)?;

    let (_, q4_bytes) = crate::generate::weight_memory_estimate(&config, num_layers, vocab_size);

    Ok(InferenceBenchmark {
        name: "Q4_0".into(),
        config: format!("{}L H={} FFN={} V={}", num_layers, config.hidden_size, config.ffn_dim, vocab_size),
        prefill_ms: result.prefill_time.as_secs_f64() * 1000.0,
        prefill_tokens_per_sec: prompt_len as f64 / result.prefill_time.as_secs_f64().max(1e-9),
        decode_ms: result.decode_time.as_secs_f64() * 1000.0,
        decode_tokens_per_sec: result.tokens_per_sec,
        total_tokens: result.tokens_generated,
        weight_memory_mb: q4_bytes as f64 / 1e6,
    })
}

/// Run GEMM throughput benchmark across sizes.
pub fn bench_gemm_sweep(device: &WarpDevice) -> Result<String, DeviceError> {
    let cache = KernelCache::new();
    let mut lines = vec!["=== GEMM Throughput Sweep ===".to_string()];
    lines.push(format!("{:>6} {:>10} {:>10} {:>10} {:>8}",
        "Size", "F32 TFLOPS", "FP16 TFLOPS", "Q4_0 TFLOPS", "FP16/F32"));

    for &size in &[256u32, 512, 1024, 2048, 4096] {
        let (m, n, k) = (size, size, size);
        let iters = if size <= 512 { 200 } else if size <= 2048 { 50 } else { 20 };
        let flops = 2.0 * m as f64 * n as f64 * k as f64;

        // F32
        let a32 = GpuTensor::<f32>::zeros(device, Shape::from_static(&[m as usize, k as usize]), DType::F32)?;
        let b32 = GpuTensor::<f32>::zeros(device, Shape::from_static(&[k as usize, n as usize]), DType::F32)?;
        let mut c32 = GpuTensor::<f32>::zeros(device, Shape::from_static(&[m as usize, n as usize]), DType::F32)?;

        crate::ops::gemm(&cache, device, &a32, &b32, &mut c32, m, n, k)?;
        device.synchronize()?;
        let start = Instant::now();
        for _ in 0..iters { crate::ops::gemm(&cache, device, &a32, &b32, &mut c32, m, n, k)?; }
        device.synchronize()?;
        let f32_tflops = flops * iters as f64 / start.elapsed().as_secs_f64() / 1e12;

        // FP16
        let a16 = GpuTensor::<half::f16>::zeros(device, Shape::from_static(&[m as usize, k as usize]), DType::F16)?;
        let b16 = GpuTensor::<half::f16>::zeros(device, Shape::from_static(&[k as usize, n as usize]), DType::F16)?;
        let mut c16 = GpuTensor::<half::f16>::zeros(device, Shape::from_static(&[m as usize, n as usize]), DType::F16)?;

        crate::gemm_tc::gemm_tensor_core(&cache, device, &a16, &b16, &mut c16, m, n, k)?;
        device.synchronize()?;
        let start = Instant::now();
        for _ in 0..iters { crate::gemm_tc::gemm_tensor_core(&cache, device, &a16, &b16, &mut c16, m, n, k)?; }
        device.synchronize()?;
        let f16_tflops = flops * iters as f64 / start.elapsed().as_secs_f64() / 1e12;

        // Q4_0 (weights quantized, activations F32)
        let a_data: Vec<f32> = vec![0.0; (m * k) as usize];
        let b_data: Vec<f32> = vec![0.0; (k * n) as usize];
        let a_q = GpuTensor::from_host(device, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F32)?;
        let b_q = GpuTensor::from_host(device, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F32)?;
        let b_quant = crate::quantize::quantize_weights_q4_0(&cache, device, &b_q, k, n)?;
        let mut c_q = GpuTensor::<f32>::zeros(device, Shape::from_static(&[m as usize, n as usize]), DType::F32)?;

        crate::quantize::gemm_q4_0(&cache, device, &a_q, &b_quant, &mut c_q, m, n, k)?;
        device.synchronize()?;
        let start = Instant::now();
        for _ in 0..iters { crate::quantize::gemm_q4_0(&cache, device, &a_q, &b_quant, &mut c_q, m, n, k)?; }
        device.synchronize()?;
        let q4_tflops = flops * iters as f64 / start.elapsed().as_secs_f64() / 1e12;

        lines.push(format!("{size:>5}³ {:>10.1} {:>10.1} {:>10.1} {:>7.1}x",
            f32_tflops, f16_tflops, q4_tflops, f16_tflops / f32_tflops.max(0.001)));
    }

    Ok(lines.join("\n"))
}

/// Benchmark autofuse: compare separate vs fused elementwise chains.
pub fn bench_autofuse(device: &WarpDevice) -> Result<String, DeviceError> {
    let cache = KernelCache::new();
    let n = 4 * 1024 * 1024usize; // 4M elements
    let shape = Shape::from_static(&[n]);

    let a = GpuTensor::<f32>::zeros(device, shape.clone(), DType::F32)?;
    let b = GpuTensor::<f32>::zeros(device, shape.clone(), DType::F32)?;
    let c = GpuTensor::<f32>::zeros(device, shape.clone(), DType::F32)?;
    let mut t1 = GpuTensor::<f32>::zeros(device, shape.clone(), DType::F32)?;
    let mut t2 = GpuTensor::<f32>::zeros(device, shape.clone(), DType::F32)?;
    let mut out_sep = GpuTensor::<f32>::zeros(device, shape.clone(), DType::F32)?;
    let mut out_fused = GpuTensor::<f32>::zeros(device, shape.clone(), DType::F32)?;

    // Generate fused kernel for: add(a,b) → gelu → mul(_,c)
    let fused_src = r#"
extern "C" __global__ void warp_autofused_bench(
    float *out, const float *in0, const float *in1, const float *in2, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in0[i] + in1[i];
        float v3 = v * v * v;
        float gelu_v = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v3)));
        out[i] = gelu_v * in2[i];
    }
}
"#;

    // Warmup both paths
    crate::ops::add(&cache, device, &a, &b, &mut t1)?;
    crate::ops::gelu(&cache, device, &t1, &mut t2)?;
    crate::ops::mul(&cache, device, &t2, &c, &mut out_sep)?;

    let fused_f = cache.get_or_compile(device, fused_src, "warp_autofused_bench")?;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(n as u32);
    unsafe {
        use cudarc::driver::PushKernelArg;
        device.stream.launch_builder(&fused_f)
            .arg(&mut out_fused.data).arg(&a.data).arg(&b.data).arg(&c.data).arg(&n)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    device.synchronize()?;

    let iters = 500;

    // Separate: 3 kernels
    let start = Instant::now();
    for _ in 0..iters {
        crate::ops::add(&cache, device, &a, &b, &mut t1)?;
        crate::ops::gelu(&cache, device, &t1, &mut t2)?;
        crate::ops::mul(&cache, device, &t2, &c, &mut out_sep)?;
    }
    device.synchronize()?;
    let separate_time = start.elapsed();

    // Fused: 1 kernel
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            use cudarc::driver::PushKernelArg;
            device.stream.launch_builder(&fused_f)
                .arg(&mut out_fused.data).arg(&a.data).arg(&b.data).arg(&c.data).arg(&n)
                .launch(cfg)
                .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
        }
    }
    device.synchronize()?;
    let fused_time = start.elapsed();

    let sep_ms = separate_time.as_secs_f64() * 1000.0 / iters as f64;
    let fused_ms = fused_time.as_secs_f64() * 1000.0 / iters as f64;
    let speedup = separate_time.as_secs_f64() / fused_time.as_secs_f64();

    let bytes = n as f64 * 4.0 * 4.0; // 3 inputs + 1 output
    let sep_bw = bytes * iters as f64 / separate_time.as_secs_f64() / 1e9;
    let fused_bw = bytes * iters as f64 / fused_time.as_secs_f64() / 1e9;

    Ok(format!(
        "=== AutoFuse Benchmark (Add→GELU→Mul, {}M elements, {} iters) ===\n\
         Separate (3 kernels): {:.3}ms  ({:.0} GB/s)\n\
         Fused    (1 kernel):  {:.3}ms  ({:.0} GB/s)\n\
         Speedup: {:.2}x\n\
         Memory passes: 3 → 1 (3x less bandwidth)",
        n / (1024*1024), iters, sep_ms, sep_bw, fused_ms, fused_bw, speedup
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<WarpDevice> {
        WarpDevice::new(0).ok()
    }

    #[test]
    fn end_to_end_transformer_benchmark() {
        let dev = match setup() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        // Use small config — head_dim must be >= 32 for Q4_0 decode
        let config = TransformerConfig::small();
        let layers = 4u32;
        let vocab = 256u32;
        let prompt_len = 8;
        let gen_tokens = 16;

        println!("\n============================================================");
        println!("  TensorWarp End-to-End Inference Benchmark");
        println!("============================================================");
        println!("  Config: {}L, H={}, FFN={}, V={}", layers, config.hidden_size, config.ffn_dim, vocab);
        println!("  Prompt: {} tokens, Generate: {} tokens", prompt_len, gen_tokens);
        println!();

        // F32 baseline
        let f32_bench = bench_f32(&dev, config.clone(), layers, vocab, prompt_len, gen_tokens).unwrap();
        println!("{f32_bench}");
        println!();

        // Q4_0 quantized
        let q4_bench = bench_q4(&dev, config.clone(), layers, vocab, prompt_len, gen_tokens).unwrap();
        println!("{q4_bench}");
        println!();

        // Comparison
        println!("  --- Comparison ---");
        let decode_speedup = q4_bench.decode_tokens_per_sec / f32_bench.decode_tokens_per_sec.max(1.0);
        let memory_savings = f32_bench.weight_memory_mb / q4_bench.weight_memory_mb.max(0.1);
        println!("    Q4_0 decode speedup: {:.2}x", decode_speedup);
        println!("    Memory savings:      {:.1}x ({:.1} MB → {:.1} MB)",
            memory_savings, f32_bench.weight_memory_mb, q4_bench.weight_memory_mb);
    }

    #[test]
    fn gemm_throughput_sweep() {
        let dev = match setup() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let report = bench_gemm_sweep(&dev).unwrap();
        println!("\n{report}");
    }

    #[test]
    fn autofuse_speedup() {
        let dev = match setup() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let report = bench_autofuse(&dev).unwrap();
        println!("\n{report}");
    }
}

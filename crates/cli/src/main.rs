//! TensorWarp CLI — the inference engine entry point.
//!
//! Usage:
//!   tensorwarp bench              Run GPU kernel benchmarks
//!   tensorwarp generate           Generate tokens with test model
//!   tensorwarp info               Show GPU device info
//!   tensorwarp load <path>        Load and inspect a SafeTensors file
//!
//! To run a real model:
//!   tensorwarp run <model_dir>    Load model and generate text

use std::time::Instant;

use warp_kernels::cache::KernelCache;
use warp_kernels::device::WarpDevice;
use warp_kernels::generate::{create_test_engine, GenerateConfig};
use warp_kernels::transformer::TransformerConfig;

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("info");

    match cmd {
        "info" => cmd_info(),
        "bench" => cmd_bench(),
        "generate" | "gen" => cmd_generate(),
        "load" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("");
            if path.is_empty() {
                eprintln!("Usage: tensorwarp load <path.safetensors>");
                std::process::exit(1);
            }
            cmd_load(path);
        }
        "onnx" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("");
            if path.is_empty() {
                eprintln!("Usage: tensorwarp onnx <model.onnx>");
                std::process::exit(1);
            }
            cmd_onnx(path);
        }
        "compile" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("");
            if path.is_empty() {
                eprintln!("Usage: tensorwarp compile <model.onnx> [--opt O2]");
                std::process::exit(1);
            }
            cmd_compile(path);
        }
        "profile" => {
            cmd_profile();
        }
        "help" | "--help" | "-h" => cmd_help(),
        _ => {
            eprintln!("Unknown command: {cmd}");
            cmd_help();
            std::process::exit(1);
        }
    }
}

fn cmd_help() {
    println!(r#"
TensorWarp — A Self-Optimizing GPU Inference Engine

USAGE:
    tensorwarp <command> [args]

COMMANDS:
    info              Show GPU device information
    bench             Run GPU kernel benchmarks (F32/FP16/Q4_0/AutoFuse)
    generate          Generate tokens with a test model
    load <path>       Load and inspect a SafeTensors model file
    onnx <path>       Load and inspect an ONNX model file
    compile <path>    Compile ONNX model through optimization pipeline
    profile           Profile GPU performance characteristics
    help              Show this help message

EXAMPLES:
    tensorwarp info
    tensorwarp bench
    tensorwarp generate
    tensorwarp load path/to/model.safetensors
    tensorwarp onnx path/to/model.onnx
"#);
}

fn cmd_info() {
    println!("=== TensorWarp Device Info ===\n");

    match WarpDevice::new(0) {
        Ok(dev) => {
            println!("  {}", dev.summary());
            println!("  CUDA include: {}", WarpDevice::cuda_include_path());

            // Quick kernel compile test
            let cache = KernelCache::new();
            let start = Instant::now();
            let src = r#"extern "C" __global__ void test(float *x) { x[0] = 1.0f; }"#;
            match cache.get_or_compile(&dev, src, "test") {
                Ok(_) => {
                    println!("  NVRTC compile: {:.1}ms", start.elapsed().as_secs_f64() * 1000.0);
                    println!("  Status: Ready");
                }
                Err(e) => println!("  NVRTC: FAILED ({e})"),
            }
        }
        Err(e) => {
            println!("  No CUDA device found: {e}");
            println!("  TensorWarp requires an NVIDIA GPU with CUDA support.");
        }
    }
}

fn cmd_bench() {
    let dev = WarpDevice::new(0).expect("No CUDA device");
    let cache = KernelCache::new();

    println!("=== TensorWarp Kernel Benchmarks ===\n");
    println!("Device: {}\n", dev.summary());

    // GEMM benchmarks
    println!("--- GEMM (FP32) ---");
    for &(m, n, k) in &[(256u32, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)] {
        let a_data: Vec<f32> = (0..(m*k) as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let b_data: Vec<f32> = (0..(k*n) as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();

        let a = warp_kernels::GpuTensor::from_host(&dev, &a_data,
            warp_ir::Shape::from_static(&[m as usize, k as usize]), warp_ir::DType::F32).unwrap();
        let b = warp_kernels::GpuTensor::from_host(&dev, &b_data,
            warp_ir::Shape::from_static(&[k as usize, n as usize]), warp_ir::DType::F32).unwrap();
        let mut c = warp_kernels::GpuTensor::<f32>::zeros(&dev,
            warp_ir::Shape::from_static(&[m as usize, n as usize]), warp_ir::DType::F32).unwrap();

        // Warmup
        warp_kernels::ops::gemm(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
        dev.synchronize().unwrap();

        let iters = 50;
        let start = Instant::now();
        for _ in 0..iters {
            warp_kernels::ops::gemm(&cache, &dev, &a, &b, &mut c, m, n, k).unwrap();
        }
        dev.synchronize().unwrap();
        let elapsed = start.elapsed();

        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let tflops = flops * iters as f64 / elapsed.as_secs_f64() / 1e12;
        let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;

        let cublas = warp_kernels::cublas_bench::cublas_sgemm_bench(&dev, m as i32, n as i32, k as i32, iters).unwrap();
        let ratio = tflops / cublas.tflops.max(1e-9) * 100.0;
        let marker = if tflops > cublas.tflops { " <<<" } else { "" };

        println!("  {m:4}³: {tflops:.2} TFLOPS ({ms:.3}ms) | cuBLAS: {:.2} TFLOPS | {ratio:.0}%{marker}", cublas.tflops);
    }

    // Fusion benchmark
    println!("\n--- Fusion Speedup ---");
    let n = 4 * 1024 * 1024;
    let shape = warp_ir::Shape::from_static(&[n]);
    let a_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 2.0).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 1.5).collect();
    let a = warp_kernels::GpuTensor::from_host(&dev, &a_data, shape.clone(), warp_ir::DType::F32).unwrap();
    let b = warp_kernels::GpuTensor::from_host(&dev, &b_data, shape.clone(), warp_ir::DType::F32).unwrap();
    let mut tmp = warp_kernels::GpuTensor::<f32>::zeros(&dev, shape.clone(), warp_ir::DType::F32).unwrap();
    let mut out1 = warp_kernels::GpuTensor::<f32>::zeros(&dev, shape.clone(), warp_ir::DType::F32).unwrap();
    let mut out2 = warp_kernels::GpuTensor::<f32>::zeros(&dev, shape, warp_ir::DType::F32).unwrap();

    // Warmup
    warp_kernels::ops::add(&cache, &dev, &a, &b, &mut tmp).unwrap();
    warp_kernels::ops::gelu(&cache, &dev, &tmp, &mut out1).unwrap();
    warp_kernels::ops::fused_add_gelu(&cache, &dev, &a, &b, &mut out2).unwrap();
    dev.synchronize().unwrap();

    let iters = 500;
    let start = Instant::now();
    for _ in 0..iters {
        warp_kernels::ops::add(&cache, &dev, &a, &b, &mut tmp).unwrap();
        warp_kernels::ops::gelu(&cache, &dev, &tmp, &mut out1).unwrap();
    }
    dev.synchronize().unwrap();
    let unfused = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        warp_kernels::ops::fused_add_gelu(&cache, &dev, &a, &b, &mut out2).unwrap();
    }
    dev.synchronize().unwrap();
    let fused = start.elapsed();

    println!("  Unfused (add + gelu): {:.2}ms", unfused.as_secs_f64() * 1000.0 / iters as f64);
    println!("  Fused (add_gelu):     {:.2}ms", fused.as_secs_f64() * 1000.0 / iters as f64);
    println!("  Speedup:              {:.2}x", unfused.as_secs_f64() / fused.as_secs_f64());

    // GEMM sweep (F32 vs FP16 vs Q4_0)
    println!("\n--- GEMM Throughput Sweep ---");
    match warp_kernels::bench::bench_gemm_sweep(&dev) {
        Ok(report) => println!("{report}"),
        Err(e) => println!("  Error: {e}"),
    }

    // AutoFuse benchmark
    println!();
    match warp_kernels::bench::bench_autofuse(&dev) {
        Ok(report) => println!("{report}"),
        Err(e) => println!("  AutoFuse error: {e}"),
    }

    println!("\n{}", cache.stats());
}

fn cmd_generate() {
    let dev = WarpDevice::new(0).expect("No CUDA device");

    println!("=== TensorWarp Text Generation ===\n");
    println!("Device: {}\n", dev.summary());

    let config = TransformerConfig::tiny();
    println!("Creating test model: {} layers, H={}, FFN={}, vocab=256",
        4, config.hidden_size, config.ffn_dim);

    let engine = create_test_engine(&dev, config, 4, 256).unwrap();

    let prompt = vec![1i32, 2, 3, 4, 5, 6, 7, 8];
    println!("Prompt tokens: {:?}\n", prompt);

    let gen_config = GenerateConfig {
        max_tokens: 32,
        greedy: true,
        eos_token_id: None,
        ..Default::default()
    };

    println!("Generating with KV cache...");
    let result = engine.generate_with_cache(&dev, &prompt, &gen_config, 256).unwrap();
    println!("\n{result}");
    println!("\n{}", engine.cache.stats());
}

fn cmd_load(path: &str) {
    println!("=== Loading SafeTensors: {path} ===\n");

    match warp_loader::SafeTensorsLoader::open(path) {
        Ok(loader) => {
            match loader.summary() {
                Ok(summary) => println!("{summary}"),
                Err(e) => eprintln!("Error reading tensors: {e}"),
            }
        }
        Err(e) => {
            eprintln!("Failed to open {path}: {e}");
            std::process::exit(1);
        }
    }
}

fn cmd_onnx(path: &str) {
    println!("=== TensorWarp ONNX Inspector ===\n");
    println!("Loading: {path}");

    let start = Instant::now();
    match warp_loader::OnnxModel::load(path) {
        Ok(model) => {
            let load_time = start.elapsed();
            println!("Loaded in {:.1}ms\n", load_time.as_secs_f64() * 1000.0);
            println!("{}", model.summary());

            // Check op compatibility
            let mut unsupported = Vec::new();
            for node in &model.nodes {
                if let Err(_e) = warp_loader::onnx::OnnxModel::map_op(node) {
                    unsupported.push(format!("  {} ({})", node.op_type, node.name));
                }
            }
            if !unsupported.is_empty() {
                println!("\nUnsupported ops ({}):", unsupported.len());
                for op in &unsupported {
                    println!("{op}");
                }
            } else {
                println!("\nAll ops supported — model is ready to run!");
            }
        }
        Err(e) => {
            eprintln!("Failed to load ONNX model: {e}");
            std::process::exit(1);
        }
    }
}

fn cmd_compile(path: &str) {
    println!("=== TensorWarp Compiler ===\n");
    println!("Loading: {path}");

    let start = Instant::now();
    match warp_loader::OnnxModel::load(path) {
        Ok(model) => {
            println!("Parsed in {:.1}ms", start.elapsed().as_secs_f64() * 1000.0);
            println!("{}", model.summary());

            // Compile through optimization pipeline
            println!("\nCompiling with O2 optimization...");
            let compile_start = Instant::now();
            let compiled = warp_loader::onnx_compile::compile_onnx(
                &model, warp_optimizer::OptimizationLevel::O2);
            let compile_time = compile_start.elapsed();

            println!("Compiled in {:.1}ms", compile_time.as_secs_f64() * 1000.0);
            println!("{}", compiled.summary());
            println!("\nModel ready for execution!");
        }
        Err(e) => {
            eprintln!("Failed: {e}");
            std::process::exit(1);
        }
    }
}

fn cmd_profile() {
    let dev = WarpDevice::new(0).expect("No CUDA device");
    println!("=== TensorWarp GPU Profiler ===\n");
    println!("Device: {}\n", dev.summary());

    let cache = KernelCache::new();

    // Profile kernel launch overhead
    let n = 1024usize;
    let a = warp_kernels::GpuTensor::<f32>::zeros(&dev,
        warp_ir::Shape::from_static(&[n]), warp_ir::DType::F32).unwrap();
    let b = warp_kernels::GpuTensor::<f32>::zeros(&dev,
        warp_ir::Shape::from_static(&[n]), warp_ir::DType::F32).unwrap();
    let mut c = warp_kernels::GpuTensor::<f32>::zeros(&dev,
        warp_ir::Shape::from_static(&[n]), warp_ir::DType::F32).unwrap();

    // Warmup
    warp_kernels::ops::add(&cache, &dev, &a, &b, &mut c).unwrap();
    dev.synchronize().unwrap();

    // Measure launch overhead
    let iters = 10000;
    let start = Instant::now();
    for _ in 0..iters {
        warp_kernels::ops::add(&cache, &dev, &a, &b, &mut c).unwrap();
    }
    dev.synchronize().unwrap();
    let elapsed = start.elapsed();
    let us_per_launch = elapsed.as_secs_f64() * 1e6 / iters as f64;

    println!("Kernel launch overhead: {:.1}μs per launch ({iters} iters)", us_per_launch);

    // Profile memory bandwidth
    let big_n = 16 * 1024 * 1024; // 16M elements
    let big_a = warp_kernels::GpuTensor::<f32>::zeros(&dev,
        warp_ir::Shape::from_static(&[big_n]), warp_ir::DType::F32).unwrap();
    let big_b = warp_kernels::GpuTensor::<f32>::zeros(&dev,
        warp_ir::Shape::from_static(&[big_n]), warp_ir::DType::F32).unwrap();
    let mut big_c = warp_kernels::GpuTensor::<f32>::zeros(&dev,
        warp_ir::Shape::from_static(&[big_n]), warp_ir::DType::F32).unwrap();

    warp_kernels::ops::add(&cache, &dev, &big_a, &big_b, &mut big_c).unwrap();
    dev.synchronize().unwrap();

    let bw_iters = 100;
    let start = Instant::now();
    for _ in 0..bw_iters {
        warp_kernels::ops::add(&cache, &dev, &big_a, &big_b, &mut big_c).unwrap();
    }
    dev.synchronize().unwrap();
    let bw_elapsed = start.elapsed();
    let bytes = big_n as f64 * 4.0 * 3.0 * bw_iters as f64; // 3 arrays (a, b, c)
    let bandwidth = bytes / bw_elapsed.as_secs_f64() / 1e9;

    println!("Memory bandwidth: {:.0} GB/s (elementwise add, {big_n} elements)", bandwidth);
    println!("\n{}", cache.stats());
}

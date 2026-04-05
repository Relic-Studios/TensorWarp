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
use warp_loader::hub::{self, ModelFormat};

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
        "run" => {
            let model_id = args.get(2).map(|s| s.as_str()).unwrap_or("");
            if model_id.is_empty() {
                eprintln!("Usage: tensorwarp run <model_path_or_id> [--prompt \"Hello\"] [--max-tokens 128] [--temperature 0.7] [--top-k 50] [--top-p 0.9] [--repetition-penalty 1.1] [--greedy] [--chat] [--fp16] [--system \"...\"]");
                std::process::exit(1);
            }
            let prompt = parse_str_arg(&args, "--prompt").unwrap_or_else(|| "Hello".to_string());
            let max_tokens = parse_u32_arg(&args, "--max-tokens").unwrap_or(128);
            let temperature = parse_f32_arg(&args, "--temperature").unwrap_or(0.7);
            let fp16 = args.iter().any(|a| a == "--fp16");
            let q4 = args.iter().any(|a| a == "--q4");
            let q4_f16 = args.iter().any(|a| a == "--q4-f16");
            cmd_run(model_id, &prompt, max_tokens, temperature, fp16, q4, q4_f16);
        }
        "compile" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("");
            if path.is_empty() {
                eprintln!("Usage: tensorwarp compile <model.onnx> [--opt O2]");
                std::process::exit(1);
            }
            cmd_compile(path);
        }
        "pipeline" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("");
            if path.is_empty() {
                eprintln!("Usage: tensorwarp pipeline <model.onnx> [--opt-level O0|O1|O2|O3]");
                std::process::exit(1);
            }
            let opt_level = parse_opt_level(&args);
            cmd_pipeline(path, opt_level);
        }
        "profile" => {
            cmd_profile();
        }
        "serve" => cmd_serve(),
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
    run <model>       Load model and generate text (SafeTensors/GGUF/ONNX)
    load <path>       Load and inspect a SafeTensors model file
    onnx <path>       Load and inspect an ONNX model file
    compile <path>    Compile ONNX model through optimization pipeline
    pipeline <path>   Full ONNX pipeline: load -> optimize -> execute with dummy input
    profile           Profile GPU performance characteristics
    serve             Start OpenAI-compatible HTTP inference server
    help              Show this help message

EXAMPLES:
    tensorwarp info
    tensorwarp bench
    tensorwarp generate
    tensorwarp run ./models/llama-7b --prompt "Once upon a time"
    tensorwarp run ./models/llama-3-8b --prompt "Hello" --max-tokens 256 --temperature 0.8
    tensorwarp run ./models/mistral --prompt "Explain AI" --top-k 50 --top-p 0.9 --chat
    tensorwarp run ./models/qwen-0.5b --prompt "Hello" --fp16  # FP16 mixed-precision (~2x decode speedup)
    tensorwarp load path/to/model.safetensors
    tensorwarp onnx path/to/model.onnx
    tensorwarp pipeline path/to/model.onnx --opt-level O2
    tensorwarp serve
"#);
}

fn cmd_serve() {
    println!("=== TensorWarp Inference Server ===\n");
    println!("Start the OpenAI-compatible HTTP server with:\n");
    println!("  cargo run -p tensorwarp-server\n");
    println!("Environment variables:");
    println!("  HOST   Bind address (default: 0.0.0.0)");
    println!("  PORT   Listen port  (default: 8000)");
    println!("  MODEL  Model name   (default: tensorwarp-test)\n");
    println!("Example:");
    println!("  MODEL=llama-7b PORT=8000 cargo run -p tensorwarp-server\n");
    println!("Endpoints:");
    println!("  POST /v1/completions");
    println!("  POST /v1/chat/completions");
    println!("  GET  /v1/models");
    println!("  GET  /health");
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

fn cmd_run(model_id: &str, prompt: &str, max_tokens: u32, temperature: f32, fp16: bool, q4: bool, q4_f16: bool) {
    println!("=== TensorWarp Run ===\n");

    let total_start = Instant::now();

    // Parse additional CLI args for sampling
    let args: Vec<String> = std::env::args().collect();
    let top_k = parse_u32_arg(&args, "--top-k").map(|v| v as usize);
    let top_p = parse_f32_arg(&args, "--top-p");
    let rep_penalty = parse_f32_arg(&args, "--repetition-penalty").unwrap_or(1.0);
    let greedy = args.iter().any(|a| a == "--greedy");
    let chat = args.iter().any(|a| a == "--chat");
    let system_prompt = parse_str_arg(&args, "--system");

    // Step 1: Resolve model path
    let model_path = match hub::resolve_model_path(model_id) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    println!("Model path: {}", model_path.display());

    // Step 2: Detect format
    let format = hub::detect_model_format(&model_path);
    println!("Format:     {format}");
    println!("Prompt:     \"{prompt}\"");
    println!("Max tokens: {max_tokens}");
    println!("Temperature: {temperature}");
    if let Some(k) = top_k { println!("Top-K:      {k}"); }
    if let Some(p) = top_p { println!("Top-P:      {p}"); }
    if rep_penalty != 1.0 { println!("Rep penalty: {rep_penalty}"); }
    println!();

    match format {
        ModelFormat::SafeTensors => {
            run_safetensors(
                &model_path, prompt, max_tokens, temperature,
                top_k, top_p, rep_penalty, greedy, chat, system_prompt.as_deref(), fp16, q4, q4_f16,
            );
        }

        ModelFormat::Gguf => {
            let gguf_path = if model_path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                model_path.clone()
            } else {
                let gguf_files: Vec<_> = std::fs::read_dir(&model_path)
                    .into_iter()
                    .flatten()
                    .flatten()
                    .filter(|e| e.file_name().to_str().map(|n| n.ends_with(".gguf")).unwrap_or(false))
                    .collect();

                if gguf_files.is_empty() {
                    eprintln!("No .gguf files found in {}", model_path.display());
                    std::process::exit(1);
                }
                gguf_files[0].path()
            };

            println!("Loading GGUF: {}", gguf_path.display());
            match warp_loader::GgufModel::load(&gguf_path) {
                Ok(model) => {
                    println!("{}", model.summary());
                    println!("GGUF model loaded. Inference with quantized weights coming soon.");
                }
                Err(e) => {
                    eprintln!("Failed to load GGUF: {e}");
                    std::process::exit(1);
                }
            }
        }

        ModelFormat::Onnx => {
            let onnx_path = if model_path.extension().and_then(|e| e.to_str()) == Some("onnx") {
                model_path.clone()
            } else {
                let onnx_files: Vec<_> = std::fs::read_dir(&model_path)
                    .into_iter()
                    .flatten()
                    .flatten()
                    .filter(|e| e.file_name().to_str().map(|n| n.ends_with(".onnx")).unwrap_or(false))
                    .collect();

                if onnx_files.is_empty() {
                    eprintln!("No .onnx files found in {}", model_path.display());
                    std::process::exit(1);
                }
                onnx_files[0].path()
            };

            println!("Loading ONNX: {}", onnx_path.display());
            match warp_loader::OnnxModel::load(onnx_path.to_str().unwrap_or("")) {
                Ok(model) => {
                    println!("{}", model.summary());
                    println!("\nUse `tensorwarp pipeline {}` for full inference.", onnx_path.display());
                }
                Err(e) => {
                    eprintln!("Failed to load ONNX: {e}");
                    std::process::exit(1);
                }
            }
        }

        ModelFormat::Unknown => {
            eprintln!("Could not detect model format in {}", model_path.display());
            eprintln!("Supported formats: .safetensors, .gguf, .onnx");
            std::process::exit(1);
        }
    }

    println!("\nTotal wall time: {:.1}ms", total_start.elapsed().as_secs_f64() * 1000.0);
}

/// Full SafeTensors inference pipeline: load model + tokenizer, tokenize, generate, decode.
fn run_safetensors(
    model_path: &std::path::Path,
    prompt: &str,
    max_tokens: u32,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    rep_penalty: f32,
    greedy: bool,
    chat: bool,
    system_prompt: Option<&str>,
    fp16: bool,
    q4: bool,
    q4_f16: bool,
) {
    use warp_kernels::generate::GenerateConfig;
    use warp_kernels::mem_pool::GpuMemPool;

    // 1. Find SafeTensors files
    let st_files: Vec<_> = std::fs::read_dir(model_path)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_name().to_str().map(|n| n.ends_with(".safetensors")).unwrap_or(false))
        .collect();

    if st_files.is_empty() {
        eprintln!("No .safetensors files found in {}", model_path.display());
        std::process::exit(1);
    }

    println!("Found {} safetensors file(s)", st_files.len());

    // 2. Load config.json
    let config_path = model_path.join("config.json");
    let llama_config = if config_path.exists() {
        match warp_loader::LlamaConfig::from_json(config_path.to_str().unwrap_or("")) {
            Ok(cfg) => {
                println!("Config: {} layers, H={}, vocab={}, GQA={}/{}",
                    cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size,
                    cfg.num_attention_heads, cfg.num_key_value_heads);
                cfg
            }
            Err(e) => {
                eprintln!("Failed to load config.json: {e}");
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("No config.json found in {} — cannot determine model architecture.", model_path.display());
        std::process::exit(1);
    };

    // 3. Initialize GPU
    let device = match WarpDevice::new(0) {
        Ok(d) => {
            println!("GPU: {}", d.summary());
            d
        }
        Err(e) => {
            eprintln!("No CUDA device: {e}");
            std::process::exit(1);
        }
    };

    // 4. Load tokenizer
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        match warp_loader::Tokenizer::from_file(&tokenizer_path) {
            Ok(tok) => {
                println!("Tokenizer: {} tokens", tok.vocab_size());
                Some(tok)
            }
            Err(e) => {
                println!("Tokenizer load failed: {e} — will output raw token IDs");
                None
            }
        }
    } else {
        println!("No tokenizer.json found — will output raw token IDs");
        None
    };

    // 5. Format prompt with chat template if requested
    let formatted_prompt = if chat {
        let template = if let Some(ref tok) = tokenizer {
            warp_loader::ChatTemplate::detect(tok)
        } else {
            warp_loader::ChatTemplate::raw()
        };
        println!("Chat template: {:?}", template.bos_token);
        template.format_prompt(system_prompt, prompt)
    } else {
        prompt.to_string()
    };

    // 6. Tokenize prompt
    let prompt_ids: Vec<i32> = if let Some(ref tok) = tokenizer {
        let ids = tok.encode(&formatted_prompt);
        println!("Prompt tokens: {} tokens", ids.len());
        if ids.len() > 10 {
            println!("  first 10: {:?}", &ids[..10]);
        } else {
            println!("  {:?}", ids);
        }
        ids.iter().map(|&id| id as i32).collect()
    } else {
        // No tokenizer — use raw bytes as token IDs (for testing)
        formatted_prompt.bytes().map(|b| b as i32).collect()
    };

    if prompt_ids.is_empty() {
        eprintln!("Empty prompt after tokenization");
        std::process::exit(1);
    }

    // 7. Load model weights
    let prec_label = if q4_f16 { " (Q4→FP16 dequantized)" } else if q4 { " (Q4_0 quantized)" } else if fp16 { " (FP16 mixed-precision)" } else { "" };
    println!("\nLoading model weights{}...", prec_label);
    let load_start = Instant::now();

    // Use sharded loader for multi-file models (7B+), single loader otherwise
    let st_paths: Vec<std::path::PathBuf> = st_files.iter().map(|e| e.path()).collect();
    let sharded = warp_loader::safetensors_loader::ShardedSafeTensorsLoader::open(&st_paths);
    let loader = match &sharded {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Failed to open safetensors: {e}");
            std::process::exit(1);
        }
    };
    let loader = sharded.unwrap();

    // 8. Configure generation (shared between F32 and F16 paths)
    let eos_token_id = if let Some(ref tok) = tokenizer {
        tok.eos_token_id.map(|id| id as i32)
    } else {
        Some(2) // default LLaMA EOS
    };

    let gen_config = GenerateConfig {
        max_tokens: max_tokens as usize,
        temperature,
        eos_token_id,
        greedy: greedy || temperature <= 0.01,
        top_k,
        top_p,
        repetition_penalty: rep_penalty,
        stop_sequences: Vec::new(),
    };

    // Cap KV cache to a sensible default — full context window (131K for Qwen 7B)
    // would require 15+ GB of VRAM for cache alone.
    let model_max = llama_config.max_position_embeddings.unwrap_or(4096);
    let cli_args: Vec<String> = std::env::args().collect();
    let max_seq_len = parse_u32_arg(&cli_args, "--max-seq-len")
        .unwrap_or(2048.min(model_max))
        .min(model_max);

    // Branch: Q4→FP16 dequantized, Q4 quantized, FP16 mixed-precision, or F32
    let (result, kernel_stats) = if q4_f16 {
        // Q4→FP16 path: load Q4, dequant to FP16, use cuBLAS HGEMM for decode
        let model = match warp_loader::LlamaModelQ4::load_q4(&loader, &llama_config, &device) {
            Ok(m) => {
                let load_time = load_start.elapsed();
                println!("Q4 model loaded in {:.1}ms", load_time.as_secs_f64() * 1000.0);
                println!("{}", m.summary());
                m
            }
            Err(e) => {
                eprintln!("Failed to load model: {e}");
                std::process::exit(1);
            }
        };

        let q_engine = warp_kernels::generate::QuantizedGenerationEngine {
            config: model.transformer_config.clone(),
            vocab_size: llama_config.vocab_size,
            embed_tokens: model.embed_tokens,
            layers: model.layers,
            final_norm: model.final_norm,
            lm_head: model.lm_head,
            cache: warp_kernels::cache::KernelCache::new(),
        };

        println!("Dequantizing Q4→FP16...");
        let dequant_start = std::time::Instant::now();
        let f16_engine = match warp_kernels::generate::DequantF16GenerationEngine::from_quantized(&device, &q_engine) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Failed to dequantize: {e}");
                std::process::exit(1);
            }
        };
        println!("Dequantized in {:.1}ms (FP16 weights: {:.1} MB)",
            dequant_start.elapsed().as_secs_f64() * 1000.0,
            f16_engine.weight_memory_bytes() as f64 / 1e6);

        println!("\nGenerating (max_seq_len={}, Q4→FP16 cuBLAS HGEMM)...\n", max_seq_len);
        println!("--- output ---");

        let result = match f16_engine.generate_with_cache(
            &device, &prompt_ids, &gen_config, max_seq_len, &q_engine.layers,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("\nGeneration failed: {e}");
                std::process::exit(1);
            }
        };

        if let Some(ref tok) = tokenizer {
            let out_ids: Vec<u32> = result.tokens.iter().map(|&t| t as u32).collect();
            let output = tok.decode(&out_ids);
            println!("{}", output);
        }
        println!("--- end ---");

        (result, f16_engine.cache.stats())
    } else if q4 {
        // Q4_0 quantized path — 6.4x memory savings
        let model = match warp_loader::LlamaModelQ4::load_q4(&loader, &llama_config, &device) {
            Ok(m) => {
                let load_time = load_start.elapsed();
                println!("Model loaded in {:.1}ms", load_time.as_secs_f64() * 1000.0);
                println!("{}", m.summary());
                m
            }
            Err(e) => {
                eprintln!("Failed to load model: {e}");
                std::process::exit(1);
            }
        };

        let engine = warp_kernels::generate::QuantizedGenerationEngine {
            config: model.transformer_config.clone(),
            vocab_size: llama_config.vocab_size,
            embed_tokens: model.embed_tokens,
            layers: model.layers,
            final_norm: model.final_norm,
            lm_head: model.lm_head,
            cache: warp_kernels::cache::KernelCache::new(),
        };

        println!("\nGenerating (max_seq_len={}, Q4_0 quantized)...\n", max_seq_len);
        println!("--- output ---");

        let result = match engine.generate_with_cache(&device, &prompt_ids, &gen_config, max_seq_len) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("\nGeneration failed: {e}");
                std::process::exit(1);
            }
        };

        if let Some(ref tok) = tokenizer {
            let out_ids: Vec<u32> = result.tokens.iter().map(|&t| t as u32).collect();
            let output = tok.decode(&out_ids);
            println!("{}", output);
        }
        println!("--- end ---");

        (result, engine.cache.stats())
    } else if fp16 {
        // FP16 mixed-precision path
        let model = match warp_loader::LlamaModelF16::load_f16(&loader, &llama_config, &device) {
            Ok(m) => {
                let load_time = load_start.elapsed();
                println!("Model loaded in {:.1}ms", load_time.as_secs_f64() * 1000.0);
                println!("{}", m.summary());
                m
            }
            Err(e) => {
                eprintln!("Failed to load model weights: {e}");
                std::process::exit(1);
            }
        };

        let engine = warp_kernels::generate::GenerationEngineF16 {
            config: model.transformer_config.clone(),
            vocab_size: llama_config.vocab_size,
            embed_tokens: model.embed_tokens,
            layers: model.layers,
            final_norm: model.final_norm,
            lm_head: model.lm_head,
            cache: KernelCache::new(),
            pool: GpuMemPool::new(),
        };

        println!("\nGenerating (max_seq_len={}, FP16 mixed-precision)...\n", max_seq_len);
        println!("--- output ---");

        let result = match engine.generate_with_cache(&device, &prompt_ids, &gen_config, max_seq_len) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("\nGeneration failed: {e}");
                std::process::exit(1);
            }
        };
        let stats = engine.cache.stats();
        (result, stats)
    } else {
        // F32 path (original)
        let model = match warp_loader::LlamaModel::load(&loader, &llama_config, &device) {
            Ok(m) => {
                let load_time = load_start.elapsed();
                println!("Model loaded in {:.1}ms", load_time.as_secs_f64() * 1000.0);
                println!("{}", m.summary());
                m
            }
            Err(e) => {
                eprintln!("Failed to load model weights: {e}");
                std::process::exit(1);
            }
        };

        let engine = warp_kernels::generate::GenerationEngine {
            config: model.transformer_config.clone(),
            vocab_size: llama_config.vocab_size,
            embed_tokens: model.embed_tokens,
            layers: model.layers,
            final_norm: model.final_norm,
            lm_head: model.lm_head,
            cache: KernelCache::new(),
            pool: GpuMemPool::new(),
        };

        println!("\nGenerating (max_seq_len={})...\n", max_seq_len);
        println!("--- output ---");

        let result = match engine.generate_with_cache(&device, &prompt_ids, &gen_config, max_seq_len) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("\nGeneration failed: {e}");
                std::process::exit(1);
            }
        };
        let stats = engine.cache.stats();
        (result, stats)
    };

    // 9. Decode output
    let output_text = if let Some(ref tok) = tokenizer {
        let output_ids: Vec<u32> = result.tokens.iter().map(|&t| t as u32).collect();
        tok.decode(&output_ids)
    } else {
        format!("{:?}", result.tokens)
    };

    println!("{output_text}");
    println!("--- end ---\n");

    // 10. Print timing stats
    println!("{result}");
    println!("\nKernel cache: {}", kernel_stats);
}

/// Parse a --key value string argument from CLI args.
fn parse_str_arg(args: &[String], key: &str) -> Option<String> {
    for (i, arg) in args.iter().enumerate() {
        if arg == key {
            return args.get(i + 1).cloned();
        }
    }
    None
}

/// Parse a --key value u32 argument from CLI args.
fn parse_u32_arg(args: &[String], key: &str) -> Option<u32> {
    parse_str_arg(args, key).and_then(|v| v.parse().ok())
}

/// Parse a --key value f32 argument from CLI args.
fn parse_f32_arg(args: &[String], key: &str) -> Option<f32> {
    parse_str_arg(args, key).and_then(|v| v.parse().ok())
}

/// Parse --opt-level from CLI args. Defaults to O2.
fn parse_opt_level(args: &[String]) -> warp_optimizer::OptimizationLevel {
    for (i, arg) in args.iter().enumerate() {
        if arg == "--opt-level" || arg == "--opt" {
            if let Some(val) = args.get(i + 1) {
                return match val.as_str() {
                    "O0" | "o0" | "0" => warp_optimizer::OptimizationLevel::O0,
                    "O1" | "o1" | "1" => warp_optimizer::OptimizationLevel::O1,
                    "O2" | "o2" | "2" => warp_optimizer::OptimizationLevel::O2,
                    "O3" | "o3" | "3" => warp_optimizer::OptimizationLevel::O3,
                    _ => {
                        eprintln!("Unknown optimization level '{val}', using O2");
                        warp_optimizer::OptimizationLevel::O2
                    }
                };
            }
        }
    }
    warp_optimizer::OptimizationLevel::O2
}

fn cmd_pipeline(path: &str, opt_level: warp_optimizer::OptimizationLevel) {
    println!("=== TensorWarp Inference Pipeline ===\n");
    println!("Loading: {path}");
    println!("Optimization: O{}\n", opt_level as u8);

    let total_start = Instant::now();

    // Step 1: Load and compile through the pipeline
    let pipe = match warp_loader::InferencePipeline::load(path, 0, opt_level) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Pipeline failed: {e}");
            std::process::exit(1);
        }
    };

    println!("{}\n", pipe.summary());

    // Step 2: Create dummy inputs based on model input specs
    let dummy_inputs: Vec<(&str, Vec<f32>, Vec<usize>)> = pipe.model.inputs.iter()
        .map(|io| {
            let shape: Vec<usize> = io.shape.iter()
                .map(|&d| if d <= 0 { 1usize } else { d as usize })
                .collect();
            let numel: usize = shape.iter().product();
            let data: Vec<f32> = (0..numel).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
            (io.name.as_str(), data, shape)
        })
        .collect();

    println!("Running inference with dummy inputs...");
    for (name, _, shape) in &dummy_inputs {
        println!("  {name}: {:?}", shape);
    }

    let infer_start = Instant::now();
    match pipe.infer(&dummy_inputs) {
        Ok(outputs) => {
            let infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
            let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

            println!("\nOutputs:");
            for (i, (output_spec, data)) in pipe.model.outputs.iter().zip(outputs.iter()).enumerate() {
                let preview: Vec<String> = data.iter().take(8)
                    .map(|v| format!("{v:.4}"))
                    .collect();
                let suffix = if data.len() > 8 { " ..." } else { "" };
                println!("  [{}] {}: {} values [{}{}]",
                    i, output_spec.name, data.len(),
                    preview.join(", "), suffix);
                if !data.is_empty() {
                    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
                    println!("       range: [{min:.4}, {max:.4}], mean: {mean:.4}");
                }
            }

            println!("\nTiming:");
            println!("  Inference:  {infer_ms:.1}ms");
            println!("  Total:      {total_ms:.1}ms");
            println!("\nPipeline complete!");
        }
        Err(e) => {
            eprintln!("Inference failed: {e}");
            std::process::exit(1);
        }
    }
}

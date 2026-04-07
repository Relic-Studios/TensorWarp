# TensorWarp

GPU inference engine written in Rust and CUDA. Compiles and runs ONNX models on NVIDIA GPUs with JIT kernel compilation, automatic operator fusion, and quantized inference.

## Requirements

- Rust 1.75+
- CUDA Toolkit 12.x
- NVIDIA GPU (RTX 30/40 series tested)
- Windows or Linux

## Build

```bash
cargo build --release
cargo test -p warp-kernels    # 184 tests
```

## Usage

### Run an ONNX model

```bash
# Inspect model ops and compatibility
cargo run --release -- onnx model.onnx

# Run inference with dummy inputs
cargo run --release -- pipeline model.onnx
```

The ONNX pipeline loads the model, compiles it through the optimization passes, uploads weights to GPU, and executes. Any ONNX model whose ops are supported will run without model-specific code.

### Run LLM inference

```bash
# Download a model (Qwen, LLaMA, Mistral — any LLaMA-family)
# Then run with quantization + CUDA graph acceleration:

cargo run --release -p tensorwarp -- run <model_path> \
  --q4 --marlin --graph --greedy --max-tokens 64

# Other flags:
#   --prompt "Your prompt"     Custom prompt
#   --temperature 0.7          Sampling temperature
#   --top-k 50                 Top-K filtering
#   --top-p 0.9                Nucleus sampling
#   --max-tokens 128           Max tokens to generate
#   --chat                     Chat mode
#   --system "..."             System prompt
#   --profile                  Per-operation timing breakdown
```

### Other commands

```bash
cargo run --release -- info       # GPU device info
cargo run --release -- bench      # GEMM and fusion benchmarks
cargo run --release -- generate   # Test generation with tiny model
cargo run --release -- compile model.onnx   # Compile without running
```

## Performance

Measured on RTX 4090, April 2026.

| Model | Config | Decode speed |
|-------|--------|-------------|
| Qwen 7B (Q4) | `--q4 --marlin --graph --greedy` | 67-71 tok/sec |
| Qwen 0.5B (Q4) | `--q4` | 115 tok/sec |

| Kernel | Result |
|--------|--------|
| FP16 Tensor Core GEMM | 155 TFLOPS (97% of cuBLAS) |
| Auto-fused elementwise chain | 6.43x speedup over separate launches |
| Q4 weight compression | 6.4x smaller, 3.5x faster than F32 |
| CUDA graph decode | +16% vs eager execution |

### ONNX model correctness

| Model | Type | Nodes | Result |
|-------|------|-------|--------|
| SqueezeNet 1.0 | CNN | 66 | Exact match with ONNX Runtime |
| Tiny Transformer | Attention+FFN | 62 | Runs end-to-end, Slice indexing bug in progress |

## Architecture

```
Input (ONNX / SafeTensors / GGUF)
  |
  v
warp-loader         Parse model, extract weights and graph
  |
  v
warp-ir             Intermediate representation (130+ ops, SSA graph)
  |
  v
warp-optimizer      Fusion passes:
                      O1: MatMul+Bias+Activation, Residual+RMSNorm
                      O2: AutoFuse arbitrary elementwise chains
                      Dead code elimination, constant folding
  |
  v
warp-kernels        160+ CUDA kernels, JIT compiled via NVRTC
                    Cached to disk (99.9% hit rate, <10ms reload)
  |
  +---> ONNX Interpreter    Graph walker, dispatches per-op to GPU.
  |                          Model-agnostic. Works for any supported model.
  |
  +---> LLM Decode Engine   Optimized autoregressive path.
                             CUDA graph capture, KV cache, pre-allocated
                             buffers, TW-Marlin Q4 quantization.
```

Both paths use the same kernel library. The ONNX interpreter handles arbitrary models. The LLM engine adds decode-specific optimizations (CUDA graphs, KV cache management, quantized GEMM).

## Crates

| Crate | Purpose |
|-------|---------|
| `warp-ir` | Graph IR with 130+ operation types, shape inference, SSA form |
| `warp-optimizer` | Pattern fusion, autofuse, constant folding, dead code elimination |
| `warp-kernels` | CUDA kernels: GEMM (F32/FP16/Q4/Q8), attention, convolution, normalization, quantization, generation engine |
| `warp-loader` | ONNX protobuf parser, ONNX executor, SafeTensors/GGUF loaders, HuggingFace model loader, tokenizer |
| `warp-codegen` | PTX and Metal code generation |
| `warp-runtime` | Runtime types and scheduling |
| `warp-cli` | Command-line interface |
| `warp-python` | PyO3 Python bindings |
| `warp-server` | OpenAI-compatible HTTP server |

## Supported ONNX operations

**Compute:** MatMul, Gemm, Conv (1D/2D/3D), ConvTranspose, DepthwiseConv

**Normalization:** RMSNorm, LayerNorm, BatchNorm, GroupNorm, InstanceNorm

**Activation:** ReLU, GELU, SiLU, Sigmoid, Tanh, LeakyReLU, Clip, Softmax

**Pooling:** MaxPool (1D/2D), AvgPool, GlobalAveragePool

**Elementwise:** Add, Sub, Mul, Div, Pow, Sqrt, Abs, Neg, Exp, Log, Sin, Cos, Ceil, Floor

**Reduction:** ReduceMean, ReduceSum, ReduceMax

**Shape:** Reshape, Transpose, Concat, Slice, Gather, Split, Flatten, Squeeze, Unsqueeze, Expand, Pad

**Other:** Softmax, ArgMax, TopK, Dropout, Identity, Constant, Shape, Cast, Resize, NonMaxSuppression

**Quantization:** Q4_0, Q8_0, W4A16 GEMM, W8A16 GEMM, TW-Marlin format

**Fused (created by optimizer):** MatMul+Bias, MatMul+Bias+Activation, Residual+RMSNorm, SiLU*Mul (SwiGLU), AutoFused elementwise chains

## LLM-specific features

**KV Cache:** GPU-resident per-layer cache with bulk prefill and fused K+V append.

**TW-Marlin Q4:** Custom quantized weight format with separated packed nibbles and FP16 scales. Adaptive Split-K, fused QKV projections (3 GEMMs into 1), fused gate+up projections (2 GEMMs into 1).

**CUDA Graphs:** Full transformer decode step captured as a single GPU command. Device-side position and cache length buffers allow graph replay without re-recording.

**Sampling:** Greedy (argmax), temperature scaling, top-K, top-P (nucleus), repetition penalty, stop sequences.

## Current limitations

- LLM support is limited to LLaMA-family architectures (Qwen, LLaMA, Mistral)
- No cross-attention (blocks Whisper, Stable Diffusion)
- No sliding window attention (blocks Gemma 4)
- Multi-GPU requires Linux + NCCL
- Metal backend has codegen but is untested on hardware
- ONNX Slice op has an indexing bug affecting some transformer exports

## Programmatic usage

```rust
use warp_kernels::device::WarpDevice;
use warp_kernels::tensor::GpuTensor;
use warp_ir::{DType, Shape};

// Initialize GPU
let device = WarpDevice::new(0)?;

// Load and run ONNX model
use warp_loader::{OnnxModel, OnnxExecutor};
let model = OnnxModel::load("model.onnx")?;
let exec = OnnxExecutor::new(&device, &model)?;
let outputs = exec.run(&device, &[("input", &input_tensor)])?;

// Or use the full pipeline (load + optimize + execute)
use warp_loader::InferencePipeline;
let pipe = InferencePipeline::load("model.onnx", 0,
    warp_optimizer::OptimizationLevel::O2)?;
let results = pipe.infer(&[("input", data_vec, shape_vec)])?;
```

## License

MIT OR Apache-2.0

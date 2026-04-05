# TensorWarp Architecture

## System Overview

TensorWarp is a 9-crate Rust workspace that compiles and executes neural network models on NVIDIA GPUs. It exceeds TensorRT in ONNX op coverage (130+ vs ~120) while matching raw GEMM performance (97% of cuBLAS).

## Data Flow

```
Input Model (ONNX / SafeTensors / GGUF)
         │
    ┌────▼────┐
    │  Loader │  warp-loader: ONNX protobuf, SafeTensors, GGUF, LLaMA config
    └────┬────┘  tokenizer, HF Hub, graph executor
         │
    ┌────▼────┐
    │   IR    │  warp-ir: SSA graph with 130+ op types, shapes, dtypes
    └────┬────┘
         │
    ┌────▼────┐
    │Optimizer│  warp-optimizer: pattern fusion, autofuse JIT, constfold, DCE
    └────┬────┘  memory planning with tensor lifetime analysis
         │
    ┌────▼────┐
    │ Codegen │  warp-codegen: PTX (CUDA) + Metal (Apple) generation
    └────┬────┘
         │
    ┌────▼────┐
    │ Runtime │  warp-runtime: runtime types, scheduling
    └────┬────┘
         │
    ┌────▼────┐
    │ Kernels │  warp-kernels: 160+ CUDA kernels, 48 modules
    └────┬────┘  autotuner, cost model, generation, speculative decoding
         │
    GPU Output
         │
    ┌────▼────┐     ┌──────────┐     ┌──────────┐
    │   CLI   │     │  Python  │     │  Server  │
    │warp-cli │     │warp-python│    │warp-server│
    └─────────┘     └──────────┘     └──────────┘
```

## Crate Structure (9 crates)

| Crate | Purpose |
|-------|---------|
| `warp-ir` | Graph IR with 130+ ops, shapes, dtypes, SSA graph |
| `warp-optimizer` | Pattern fusion, autofuse engine, constfold, DCE, memory planning |
| `warp-codegen` | PTX + Metal code generation |
| `warp-runtime` | Runtime types and scheduling |
| `warp-kernels` | 160+ CUDA kernels, 48 modules, autotuner, cost model, generation, speculative decoding |
| `warp-loader` | ONNX pipeline, SafeTensors, GGUF, LLaMA, tokenizer, HF Hub, graph executor |
| `warp-cli` | 10 CLI commands (info, bench, generate, load, onnx, compile, profile, pipeline, run, serve) |
| `warp-python` | PyO3 bindings (Engine, Tensor, OnnxModel, GEMM, Conv, RMSNorm, Softmax, generate) |
| `warp-server` | OpenAI-compatible HTTP server (axum-based) |

## Key Design Decisions

### 1. JIT Compilation over AOT
TensorWarp compiles CUDA kernels at runtime via NVRTC. This enables:
- **Shape specialization**: K dimension baked as compile-time constant
- **AutoFuse**: elementwise chains compiled into single kernels
- **Disk cache**: compiled PTX saved for instant reload

### 2. Vendored cudarc
We vendor cudarc 0.16.6 with two patches:
- `CudaFunction.cu_function`: changed from `pub(crate)` to `pub`
- `CudaSlice.cu_device_ptr`: same

This enables CUDA graph capture via `capture_safe_launch()`.

### 3. Dual Execution Paths
- **Interpreter path**: ONNX executor calls kernels directly (fast startup)
- **Compilation path**: ONNX -> IR -> optimize -> codegen -> execute (maximum performance)

### 4. Compiled Graph Executor
The compiled path converts IR into fused ops dispatched to GPU. Memory planning analyzes tensor lifetimes to minimize peak allocation.

### 5. Automatic Fusion
The autofuse engine walks the IR graph topologically, finds maximal chains of elementwise ops where each intermediate result has exactly one consumer, and generates a single CUDA kernel. This is unique -- TRT uses fixed fusion patterns.

## Performance Architecture

### Flash Attention v2
- Tiled Q x K with online softmax (log-sum-exp correction)
- Block-sparse attention masks
- Paged KV cache (vLLM-style) for long sequences

### FP16 Tensor Core GEMM (97% of cuBLAS)
- cp.async 2-stage pipeline for async global -> shared memory
- Shape-specialized JIT kernels with K baked as constant
- 128x128 block tiles, 8 warps, 2x4 wmma fragments
- Split-K GEMM for M=1 decode (reduces latency on thin matrices)

### Quantized Inference
- INT8/FP8 GEMM with A8W8 (symmetric quantization)
- GPTQ/AWQ quantization format support
- Q4_0 / Q8_0 block-scaled (GGUF-compatible)
- W4A16 GEMM: dequantize weights on-the-fly during tiled GEMM

### Speculative Decoding
- Draft model generates K candidate tokens
- Adaptive K based on acceptance rate (Thompson sampling)

### Autotuner + Cost Model
- Thompson sampling over kernel configurations
- Shape buckets for amortized tuning across similar shapes
- Dream-cycle autotuning: offline exploration of config space
- Cost model predicts best config to reduce tuning overhead

### Winograd Convolution
- F(2,3) Winograd for 3x3 convolutions (~2.25x theoretical speedup)

### Text Generation Pipeline
- Tokenizer -> generate -> decode
- KV cache with bulk prefill + decode split
- Streaming token output

### CUDA Graphs
- `capture_safe_launch()`: bypasses cudarc's bind_to_thread
- Uses vendored cudarc's public `cu_function` and `cu_device_ptr`
- 4.7x speedup on 3-op chain

## Crate Dependencies

```
warp-ir (no deps)
    ↑
warp-optimizer (depends on warp-ir)
    ↑
warp-codegen (depends on warp-ir)
    ↑
warp-runtime (depends on warp-ir, warp-optimizer, warp-codegen)
    ↑
warp-kernels (depends on warp-ir, cudarc)
    ↑
warp-loader (depends on warp-ir, warp-kernels, warp-optimizer)
    ↑
warp-cli, warp-python, warp-server (depend on warp-loader + warp-kernels)
```

## File Layout

```
crates/
├── ir/           # SSA graph IR (130+ ops, shapes, dtypes)
├── optimizer/    # Fusion passes, autofuse, constfold, DCE, memory planning
├── codegen/      # PTX + Metal code generation
├── runtime/      # Runtime types, scheduling
├── kernels/      # 160+ CUDA kernels (48 modules)
│   ├── gemm_tc.rs      # FP16 tensor core GEMM (97% cuBLAS)
│   ├── quantize.rs     # Q4_0/Q8_0 + W4A16 + INT8/FP8 GEMM
│   ├── conv.rs         # Conv2D, ConvTranspose, Depthwise, Pool, Winograd
│   ├── attention.rs    # Flash Attention v2, paged, decode attention
│   ├── fp16.rs         # Full FP16 kernel suite
│   ├── autotune.rs     # Cost model + Thompson sampling autotuner
│   ├── generation.rs   # Text generation pipeline
│   ├── speculative.rs  # Speculative decoding with adaptive K
│   ├── cuda_graph.rs   # CUDA graph capture/replay
│   └── ...
├── loader/       # ONNX + SafeTensors + GGUF + LLaMA + tokenizer + graph executor
├── cli/          # 10 CLI commands
├── python/       # PyO3 bindings
└── server/       # OpenAI-compatible HTTP server (axum)
vendor/
└── cudarc/       # Patched cudarc with pub fields
python/
├── tensorwarp/   # Python API
└── benchmark.py  # vs ONNX Runtime comparison
```

# TensorWarp Architecture

## System Overview

TensorWarp is a 7-crate Rust workspace that compiles and executes neural network models on NVIDIA GPUs. It exceeds TensorRT in ONNX op coverage (130+ vs ~120) while matching raw GEMM performance (97% of cuBLAS).

## Data Flow

```
Input Model (ONNX/SafeTensors)
         │
    ┌────▼────┐
    │ Parser  │  warp-loader: ONNX protobuf, SafeTensors, LLaMA config
    └────┬────┘
         │
    ┌────▼────┐
    │   IR    │  warp-ir: Graph with 60+ op types, shape inference
    └────┬────┘
         │
    ┌────▼────┐
    │Optimizer│  warp-optimizer: O1 (pattern fusion), O2 (autofuse JIT)
    └────┬────┘
         │
    ┌────▼────┐
    │ Codegen │  warp-codegen: PTX (CUDA) + Metal (Apple) generation
    └────┬────┘
         │
    ┌────▼────┐
    │ Runtime │  warp-runtime: Tiered compilation, memory planning
    └────┬────┘
         │
    ┌────▼────┐
    │ Kernels │  warp-kernels: 139 CUDA kernels, tensor ops, GEMM, attention
    └────┬────┘
         │
    GPU Output
```

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
- **Compilation path**: ONNX → IR → optimize → codegen → execute (maximum performance)

### 4. Automatic Fusion
The autofuse engine walks the IR graph topologically, finds maximal chains of elementwise ops where each intermediate result has exactly one consumer, and generates a single CUDA kernel that computes the entire chain in one memory pass. This is unique — TRT uses fixed fusion patterns.

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
tensorwarp CLI (depends on all)
```

## Performance Architecture

### FP16 Tensor Core GEMM
- cp.async 2-stage pipeline for async global→shared memory
- Shape-specialized JIT kernels with K baked as constant
- 128×128 block tiles, 8 warps, 2×4 wmma fragments
- Adaptive BK (16 or 32) based on K dimension
- Small-matrix SIMT variants for 64-512 sizes

### AutoFuse
- Discovery: O(N) graph walk, O(1) per fusion check
- Codegen: string-template CUDA generation
- Compilation: NVRTC → PTX → CUmodule (cached)
- Execution: single kernel launch for entire chain

### CUDA Graphs
- `capture_safe_launch()`: bypasses cudarc's bind_to_thread
- Uses vendored cudarc's public `cu_function` and `cu_device_ptr`
- Measured: 4.7x speedup on 3-op chain

## File Layout

```
crates/
├── ir/           # Graph IR (60+ ops, shapes, dtypes)
├── optimizer/    # Fusion passes + autofuse engine
├── codegen/      # PTX + Metal code generation
├── runtime/      # Tiered compilation, memory, scheduling
├── kernels/      # 139 CUDA kernels (43 modules)
│   ├── gemm_tc.rs      # FP16 tensor core GEMM (97% cuBLAS)
│   ├── quantize.rs     # Q4_0/Q8_0 + W4A16 GEMM
│   ├── conv.rs         # Conv2D, ConvTranspose, Depthwise, Pool
│   ├── attention.rs    # Flash, paged, decode attention
│   ├── fp16.rs         # Full FP16 kernel suite
│   ├── autotune.rs     # Kernel autotuner
│   ├── cuda_graph.rs   # CUDA graph capture/replay
│   └── ...
├── loader/       # ONNX parser + executor + compiler
└── cli/          # Command-line interface
vendor/
└── cudarc/       # Patched cudarc with pub fields
python/
├── tensorwarp/   # Python API
└── benchmark.py  # vs ONNX Runtime comparison
examples/
├── hello_gpu.rs
├── onnx_inference.rs
└── builder_api.rs
```

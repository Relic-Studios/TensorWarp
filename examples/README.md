# TensorWarp Examples

Standalone examples demonstrating key TensorWarp features. Each example is a
self-contained binary that initializes a CUDA device, runs GPU compute, and
prints results to stdout.

**Prerequisite:** A CUDA-capable GPU and the CUDA toolkit must be installed.

## Running

```bash
# Minimal GPU matrix multiply
cargo run --example hello_gpu

# ONNX model construction and inference
cargo run --example onnx_inference

# Builder SDK: programmatic model construction
cargo run --example builder_api
```

## Examples

### hello_gpu

Minimal example that creates a CUDA device, uploads two matrices to the GPU,
runs a GEMM (General Matrix Multiply) kernel, and reads the result back. Good
starting point for understanding the `WarpDevice`, `GpuTensor`, and `ops::gemm`
APIs.

### onnx_inference

Constructs a small ONNX-style model in memory (Conv -> ReLU -> GlobalAvgPool ->
FC), loads it into the `OnnxExecutor`, and runs inference. Shows how to work
with `OnnxModel`, `OnnxExecutor`, and named input/output tensors. In production,
replace the synthetic model with `OnnxModel::load("model.onnx")`.

### builder_api

Uses the `ModelBuilder` SDK to define a neural network graph layer by layer
(Input -> ReLU -> Linear -> Softmax), compiles it into a `CompiledModel`, and
runs inference. Demonstrates the TensorRT-style builder pattern for
programmatic model construction.

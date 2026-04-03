# Contributing to TensorWarp

Thanks for your interest in contributing to TensorWarp!

## Getting Started

```bash
# Clone
git clone https://github.com/Relic-Studios/TensorWarp.git
cd TensorWarp

# Build (requires CUDA 12.x + Rust 1.75+)
cargo build

# Run tests (requires NVIDIA GPU)
cargo test --workspace --exclude cudarc

# Run benchmarks
cargo run -- bench
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system design.

The codebase is organized as a 7-crate Rust workspace:
- `warp-ir`: Graph IR (start here to understand the data model)
- `warp-optimizer`: Fusion passes (add new optimizations here)
- `warp-codegen`: PTX/Metal generation
- `warp-runtime`: Execution scheduling
- `warp-kernels`: GPU kernels (most development happens here)
- `warp-loader`: Model loading (ONNX, SafeTensors)
- `tensorwarp` (CLI): User-facing commands

## Adding a New CUDA Kernel

1. Add the CUDA source as a `const &str` in the appropriate module
2. Add a Rust wrapper function following the existing pattern
3. Add a `#[test]` that validates correctness against CPU reference
4. Wire into the ONNX executor if it maps to an ONNX op

Example pattern:
```rust
const MY_KERNEL_SRC: &str = r#"
extern "C" __global__ void warp_my_op(float *out, const float *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = x[i] * 2.0f; }
}
"#;

pub fn my_op(cache: &KernelCache, device: &WarpDevice,
             x: &GpuTensor<f32>, out: &mut GpuTensor<f32>) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, MY_KERNEL_SRC, "warp_my_op")?;
    let n = x.numel;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut out.data).arg(&x.data).arg(&n)
            .launch(cfg))?;
    }
    Ok(())
}
```

## Adding a New ONNX Op

1. Add mapping in `crates/loader/src/onnx.rs` → `map_op()`
2. Add execution in `crates/loader/src/onnx_exec.rs` → `execute_node()`
3. Add the op name to `supported_ops()` list
4. Add a validation test in `crates/loader/src/onnx_validate.rs`

## Code Style

- Follow existing patterns (the codebase is self-consistent)
- Use `launch_err!` macro for kernel launches
- Add `#[test]` for every new function
- Prefer `Result<T, DeviceError>` over unwrap/panic

## Performance Testing

```bash
# GEMM throughput
cargo test -p warp-kernels gemm_throughput_sweep -- --nocapture

# AutoFuse speedup
cargo test -p warp-kernels autofuse_speedup -- --nocapture

# FP16 vs cuBLAS
cargo test -p warp-kernels tensor_core_vs_cublas -- --nocapture
```

## License

MIT OR Apache-2.0

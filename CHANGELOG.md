# Changelog

## v0.1.0 (2026-04-03) — Initial Release

### Performance
- FP16 tensor core GEMM at 97% of cuBLAS (155 TFLOPS at 4096³)
- Automatic elementwise fusion engine with 6.43x speedup
- CUDA graph capture/replay with 4.7x decode speedup
- W4A16 quantized inference: 3.5x speed, 5.1x memory savings
- FP16 end-to-end transformer: 6.42x faster than F32
- Fused ops (residual+norm, SwiGLU, QKV projection): 1.73x decode

### Model Support
- 130+ ONNX ops (>120% of TensorRT coverage)
- LLaMA/Mistral/Qwen: 98% coverage
- Stable Diffusion: 95% coverage
- Whisper: 95% coverage (Conv1D added)
- YOLO v8/v9: 95% coverage
- ViT/CLIP/SAM: 95% coverage
- Mamba/RWKV: 65% coverage (SelectiveScan)
- LSTM/GRU cells for speech/audio models

### Infrastructure
- 139 CUDA kernels across 45 modules
- ONNX → IR → Optimize → Execute compilation pipeline
- Disk-persistent kernel cache (skip NVRTC on reload)
- Model serialization (.warp engine files)
- GPU memory pool with size-bucketed reuse
- INT8/FP8 calibration (MinMax, Percentile, Entropy)
- Multi-GPU device enumeration + stream scheduling
- Layer-by-layer profiler

### APIs
- Rust native API
- C/C++ FFI (extern "C" exports, auto-generated header)
- Python package (preview + maturin config)
- Builder SDK (10-line model construction)
- 7 CLI commands (info, bench, generate, onnx, compile, profile, load)

### Testing
- 211 tests, all passing
- Stress tests (non-power-of-2 GEMM, 1000 rapid launches)
- Numerical precision tests (FP16 vs F32 divergence)
- ONNX validation suite (MLP, CNN, ResNet, Transformer, Embedding)

### Deployment
- Dockerfile (multi-stage CUDA build)
- GitHub Actions CI/CD
- Architecture documentation
- Contributing guide
- 3 working examples
- HuggingFace model loader
- Benchmark script (vs ONNX Runtime)

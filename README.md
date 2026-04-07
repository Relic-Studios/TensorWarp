# TensorWarp 🌀

**A from-scratch GPU inference engine in Rust + CUDA.**

> *"We don't use TensorRT. We are the TensorRT."*
> — us, at 3am, probably wrong but vibing

TensorWarp JIT-compiles CUDA kernels via NVRTC, fuses operations automatically, and runs real models through a modular ONNX pipeline. It's not finished. It's not always faster than the big frameworks. But it's ours, it's open source, and every line was written from zero.

```
┌─────────────────────────────────────────────────────┐
│  "How hard can GPU inference be?"                   │
│                                                     │
│  The GPU inference:                                 │
│  ┌───────────────────────────────────────────────┐  │
│  │ ONNX Model                                    │  │
│  │   ↓ parse protobuf by hand (no codegen)       │  │
│  │ IR Graph (130+ ops, SSA, arena-allocated)      │  │
│  │   ↓ pattern fusion + autofuse + DCE            │  │
│  │ Optimized Graph                                │  │
│  │   ↓ JIT compile CUDA C → PTX → cubin           │  │
│  │ GPU Kernels (cached to disk, 99.9% hit rate)   │  │
│  │   ↓ CUDA graph capture (single launch)         │  │
│  │ Tokens go brr                                  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## What Actually Works (no cap)

### ✅ LLM Inference — Real, Measured, Reproducible
```
Model          Config                         tok/sec    GPU
─────────────  ─────────────────────────────  ─────────  ────────
Qwen 7B Q4     --q4 --marlin --graph --greedy  67-71     RTX 4090
Qwen 0.5B Q4   --q4                            115       RTX 4090
```

Is that faster than TensorRT? **No.** TRT would probably do 100-130+ on the same model. But we wrote every kernel from scratch in ~6 weeks and we're at 55-60% of theoretical bandwidth. We're not done.

### ✅ ONNX Model Execution — CNN Verified Correct
```
Model          Nodes   Params    Result
─────────────  ──────  ────────  ──────────────────────────────
SqueezeNet     66      1.2M      ✅ Exact match with ONNX Runtime
Tiny Transformer 62    ~80K      ✅ Runs end-to-end (output shape bug in Slice)
```

These run through the **modular ONNX interpreter path** — not hard-coded model support. If the ops have kernels, the model runs. No per-model engineering needed.

### ✅ Kernel Performance
```
FP16 Tensor Core GEMM:     155 TFLOPS (97% of cuBLAS)
Elementwise auto-fusion:   6.43x speedup (3 kernels → 1)
Q4 weight compression:     6.4x smaller, 3.5x faster than F32
CUDA graph decode:         +16% vs eager (single GPU launch per step)
```

### 🚧 What's In Progress
- More ONNX model coverage (ResNet, MobileNet, ViT next)
- Transformer Slice/indexing correctness
- Cross-attention (unlocks Whisper, Stable Diffusion)
- Sliding window attention (unlocks Gemma 4)
- Pushing toward 100 tok/sec on 7B

### ❌ What Doesn't Work Yet
- Stable Diffusion / SDXL (need cross-attention + ConvTranspose2D validation)
- Whisper (need encoder-decoder architecture)
- Multi-GPU (infrastructure exists, NCCL needs Linux)
- Metal backend (codegen exists, never tested on real hardware)

## Architecture

```
ONNX / SafeTensors / GGUF
        │
        ▼
   ┌─────────┐     ┌───────────┐
   │ warp-ir  │────▶│ optimizer  │  O1: pattern fusion (MatMul+Bias+Act)
   │ 130+ ops │     │           │  O2: autofuse (arbitrary elementwise chains)
   │ SSA graph│     │           │  DCE, const fold
   └─────────┘     └─────┬─────┘
                         │
                    ┌────▼──────┐
                    │ warp-     │  160+ CUDA kernels
                    │ kernels   │  NVRTC JIT compilation
                    │           │  Disk-cached (< 10ms reload)
                    └─────┬─────┘
                          │
                 ┌────────┴────────┐
                 │                 │
          ┌──────▼──────┐  ┌──────▼──────┐
          │ ONNX interp │  │ LLM decode  │
          │ (any model) │  │ (optimized) │
          │ walk graph,  │  │ CUDA graph, │
          │ dispatch ops │  │ KV cache,   │
          └─────────────┘  │ TW-Marlin   │
                           └─────────────┘
```

Two execution paths:
- **ONNX Interpreter**: Model-agnostic. Walks the graph, dispatches ops to GPU kernels. Works for CNNs, transformers, anything with supported ops.
- **LLM Decode Engine**: Hand-optimized for autoregressive generation. CUDA graphs, pre-allocated buffers, TW-Marlin Q4 format, fused kernels.

Both use the same underlying kernel library.

## Quick Start

```bash
# Build (requires Rust 1.75+, CUDA 12.x, NVIDIA GPU)
cargo build --release

# GPU info
cargo run --release -- info

# Run an ONNX model
cargo run --release -- pipeline model.onnx

# Inspect ONNX model ops
cargo run --release -- onnx model.onnx

# Run LLM inference (Qwen/LLaMA from HuggingFace)
cargo run --release -p tensorwarp -- run <model_path> --q4 --marlin --graph --greedy --max-tokens 64

# Benchmarks
cargo run --release -- bench

# Tests (184 passing)
cargo test -p warp-kernels
```

## Crate Structure

| Crate | What it does | Lines |
|-------|-------------|-------|
| `warp-ir` | SSA graph IR, 130+ ops, shapes, dtypes | ~2K |
| `warp-optimizer` | Pattern fusion, autofuse, constfold, DCE | ~1.5K |
| `warp-kernels` | 160+ CUDA kernels, GEMM, attention, quantization, generation | ~33K |
| `warp-loader` | ONNX parser + executor, SafeTensors, GGUF, LLaMA loader, tokenizer | ~7K |
| `warp-cli` | CLI: info, bench, generate, run, onnx, compile, pipeline, profile | ~1K |

**Total: ~45K lines of Rust, 160+ CUDA kernels, 184 tests, 9 crates**

## Key Technical Details

### TW-Marlin Q4 Format
Custom quantized weight layout for M=1 decode GEMMs:
- Separated packed nibbles + FP16 scales (halved scale bandwidth)
- Adaptive Split-K for SM occupancy on small N
- Fused QKV projections (3 GEMMs → 1 GEMM + split)
- Fused gate+up projections (2 GEMMs → 1 GEMM + split)
- Factored scale multiply (1 FMUL per group instead of 32)

### CUDA Graph Acceleration
Entire transformer decode captured as a single GPU command:
- 28 layers × ~10 ops = ~280 kernel launches → 1 graph replay
- Device-side pos/cache_len buffers (no re-record per step)
- Patched vendored cudarc for graph capture safety

### AutoFuse Engine
Discovers and fuses arbitrary elementwise chains at compile time:
```
Before: Add → GELU → Mul → Sigmoid  (4 kernel launches, 4 memory passes)
After:  warp_autofused_0              (1 kernel launch, 1 memory pass)
```
Walks IR graph topologically, finds maximal single-user chains, generates per-chain CUDA kernels via NVRTC.

### LLM Decode Profiling (7B Q4, RTX 4090)
```
Component                    Time/step    % of total
─────────────────────────    ─────────    ──────────
Graph replay (28 layers)     8-10 ms      ~65%
LM head (cuBLAS F32)         2.5 ms       ~20%
Logits readback (608 KB)     0.13 ms       ~1%
CPU sampling (greedy)         0.1 ms       ~1%
```
Bottleneck is GPU compute at ~42% of theoretical bandwidth. Time is spread evenly across operations — no single kernel dominates.

## What We Learned

Things that worked:
- Pre-allocated decode buffers → 5x speedup (biggest single win)
- cuBLAS for LM head → 87% bandwidth utilization
- CUDA graph capture → 16% speedup
- Adaptive Split-K → 71% speedup on 7B Q4

Things that didn't work:
- Q4 LM head (stride-16 access kills coalescing vs cuBLAS F32)
- GPU argmax (sync overhead > CPU argmax time — CPU work is hidden behind GPU)
- cp.async double-buffer for M=1 (L1 cache already optimal)
- Self-speculative decode (25% acceptance without fine-tuning)

## Roadmap

**Now:**
- [ ] Fix ONNX Slice/indexing for transformer correctness
- [ ] Test more models (ResNet, MobileNet, ViT)
- [ ] Cross-attention kernel (unlocks Whisper + diffusion)
- [ ] Sliding window attention (unlocks Gemma 4)

**Soon:**
- [ ] FP16 LM head (halve the 2.5ms bottleneck)
- [ ] ONNX compiled graph path (IR → fused → CUDA graph)
- [ ] Non-causal attention mode

**Later:**
- [ ] Tensor parallelism (multi-GPU, NCCL)
- [ ] Persistent kernel (fuse all non-GEMM ops between GEMMs)
- [ ] Metal backend (codegen exists, needs Apple hardware)

## License

MIT OR Apache-2.0

---

*Built from scratch by [Relic Studios](https://github.com/Relic-Studios). Not affiliated with NVIDIA. TensorRT is NVIDIA's trademark — we just respect the hustle.*

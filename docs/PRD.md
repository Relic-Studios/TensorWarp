# Warp — Product Requirements Document
## A Self-Optimizing Inference Engine

**Version:** 0.2.0
**Date:** 2026-04-01
**Authors:** Aidan + Thomas

---

## Vision

Warp is a Rust-native GPU inference engine that compiles and executes transformer models faster than TensorRT — and keeps getting faster while running. It uses tiered compilation with runtime profiling to continuously reoptimize execution plans, applying principles from cognitive architecture (Hebbian learning, active inference, signal-driven adaptation) to the domain of GPU kernel optimization.

**One-line:** A living inference engine that observes itself and evolves.

---

## Why This Exists

1. **TensorRT is a black box.** Closed source, NVIDIA-only, minutes to compile, static once compiled, brittle ONNX parsing, no visibility into kernel decisions.
2. **Memory systems are bottlenecked on inference speed.** Didymus engram search, pipeline processing, and consolidation all depend on local LLM inference. Faster inference = faster cognition.
3. **No existing engine does tiered compilation.** JVMs do it. V8 does it. No inference engine does. This is a category-defining feature.
4. **Cross-platform is unsolved.** The 4090 runs CUDA, the M3 Ultra runs Metal. Nobody compiles from one IR to both backends well.

---

## Core Differentiators

| Feature | TensorRT | vLLM | Warp |
|---------|----------|------|------|
| Tiered compilation | No | No | **Yes — Tier 0→3 with hot-swap** |
| Profile-guided recompilation | No | No | **Yes — real kernel timings drive optimization** |
| Cross-platform (CUDA + Metal) | CUDA only | CUDA only | **Yes — same IR, different backends** |
| Compilation speed | Minutes | N/A (runtime) | **Seconds (Tier 1), instant (Tier 0)** |
| Open source kernels | No | Partial | **Yes — generated, inspectable, modifiable** |
| Hebbian fusion discovery | No | No | **Yes — co-activation tracking finds novel fusions** |
| Idle-time autotuning | No | No | **Yes — dream-cycle optimization when load is low** |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Python API (PyO3)  /  Rust API                     │
│  model = warp.compile("llama-7b.safetensors")       │
│  output = model.generate(tokens, max_len=128)       │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  Model Loader                                       │
│  ONNX / SafeTensors / GGUF → Warp IR Graph          │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  Tiered Compiler (the living core)                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐│
│  │ Tier 0  │→ │ Tier 1  │→ │ Tier 2  │→ │ Tier 3 ││
│  │ Naive   │  │ Fused   │  │ Profiled│  │ Tuned  ││
│  │ instant │  │ seconds │  │ guided  │  │ optimal││
│  └─────────┘  └─────────┘  └─────────┘  └────────┘│
│       ↑               ↑            ↑               │
│       └── hot-swap ───┴── hot-swap ┴── hot-swap ───│
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  Optimizer Pipeline                                 │
│  Graph Passes → Fusion → Memory Planning → Codegen  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Pattern  │  │ Cost-    │  │ Hebbian  │          │
│  │ Matching │  │ Model    │  │ Discovery│          │
│  │ (Tier 1) │  │ (Tier 2) │  │ (Tier 3) │          │
│  └──────────┘  └──────────┘  └──────────┘          │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  Codegen Backends                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ LLVM → PTX   │  │ LLVM → Metal │  │ SPIR-V    │ │
│  │ (NVIDIA)     │  │ (Apple)      │  │ (AMD/etc) │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  Runtime                                            │
│  Memory Pool │ KV Cache │ Scheduler │ CUDA Graphs   │
│  Profiler │ Continuous Batching │ Speculative Decode │
└─────────────────────────────────────────────────────┘
```

---

## Crate Structure

```
warp/
├── crates/
│   ├── ir/           Core graph IR, types, shapes, ops
│   ├── optimizer/    Graph passes, pattern matching, fusion
│   ├── codegen/      Backend trait, LLVM IR emission, PTX/Metal
│   ├── runtime/      Memory, scheduling, profiling, tiered compiler
│   ├── loader/       ONNX, SafeTensors, GGUF model import
│   ├── kernels/      Hand-tuned kernel templates (GEMM, attention)
│   ├── autotune/     Empirical kernel autotuning framework
│   └── python/       PyO3 bindings
├── docs/
│   ├── PRD.md        This document
│   ├── SPEC.md       Technical specification
│   └── RESEARCH.md   Deep research findings
└── benches/          Benchmarks against cuBLAS, TRT, etc.
```

---

## Functional Requirements

### FR-1: Model Loading
- **FR-1.1:** Load transformer models from SafeTensors format
- **FR-1.2:** Load transformer models from ONNX format
- **FR-1.3:** Load quantized models from GGUF format
- **FR-1.4:** Auto-detect model architecture (LLaMA, Mistral, Qwen, GPT-NeoX)
- **FR-1.5:** Convert model weights to optimal dtype per-layer based on hardware

### FR-2: Graph Compilation
- **FR-2.1:** Parse model into Warp IR graph with full type/shape information
- **FR-2.2:** Support static shapes (fully known at compile time)
- **FR-2.3:** Support symbolic shapes (batch size, sequence length as variables)
- **FR-2.4:** Shape specialization with JIT fallback for novel shapes
- **FR-2.5:** Compilation completes in <5 seconds for 7B models (Tier 1)

### FR-3: Graph Optimization
- **FR-3.1:** Pattern-matched fusion (MatMul+Bias, MatMul+Bias+Act, Residual+RMSNorm)
- **FR-3.2:** Complete SwiGLU block fusion (gate_proj + up_proj + silu + mul + down_proj)
- **FR-3.3:** Cost-model-based fusion using profiling data
- **FR-3.4:** Hebbian fusion discovery from kernel co-activation patterns
- **FR-3.5:** Dead code elimination
- **FR-3.6:** Memory layout optimization (NHWC for Tensor Cores)
- **FR-3.7:** Constant folding and strength reduction

### FR-4: Kernel Code Generation
- **FR-4.1:** Generate LLVM IR from graph ops
- **FR-4.2:** Lower LLVM IR to PTX via NVPTX backend
- **FR-4.3:** Lower LLVM IR to Metal AIR for Apple GPUs
- **FR-4.4:** Shape-specialized kernels (exact dims baked in, no runtime dispatch)
- **FR-4.5:** Fused kernels for all fusion patterns in FR-3
- **FR-4.6:** Hand-tuned GEMM kernel templates competitive with cuBLAS
- **FR-4.7:** FlashAttention-equivalent fused attention kernel
- **FR-4.8:** FP8/FP4 quantized compute kernels
- **FR-4.9:** Escape hatch for hand-written PTX for critical paths

### FR-5: Tiered Compilation
- **FR-5.1:** Tier 0 — instant startup, no optimization, naive kernels
- **FR-5.2:** Tier 1 — pattern-matched fusion, compiles in seconds
- **FR-5.3:** Tier 2 — profile-guided recompilation using real execution data
- **FR-5.4:** Tier 3 — autotuned kernels from empirical benchmarking
- **FR-5.5:** Atomic hot-swap of execution plans (zero inference downtime)
- **FR-5.6:** Configurable tier policies (sample thresholds, time gates)
- **FR-5.7:** Plan rollback if new tier is slower (regression protection)
- **FR-5.8:** Persistent tier cache (reuse Tier 3 plans across restarts)

### FR-6: Runtime Profiling
- **FR-6.1:** Per-kernel wall-clock timing
- **FR-6.2:** Input shape frequency tracking
- **FR-6.3:** GPU hardware counters via CUPTI (occupancy, bandwidth, cache rates)
- **FR-6.4:** Bottleneck classification (memory-bound, compute-bound, launch-bound)
- **FR-6.5:** Optimization hint generation (specialize, fuse, retile)
- **FR-6.6:** Human-readable profiling reports

### FR-7: Memory Management
- **FR-7.1:** Arena-based GPU memory pool with lifetime-aware allocation
- **FR-7.2:** Static memory planning for known-shape graphs
- **FR-7.3:** Slab allocator fallback for dynamic shapes
- **FR-7.4:** PagedAttention-style KV cache (block table, non-contiguous pages)
- **FR-7.5:** vAttention support (CUDA virtual memory for contiguous virtual layout)
- **FR-7.6:** Memory pressure monitoring and adaptive batch sizing

### FR-8: Execution
- **FR-8.1:** CUDA Graph capture and replay for static decode steps
- **FR-8.2:** Persistent decode kernels (stay resident on SMs across steps)
- **FR-8.3:** Multi-stream pipelining (overlap compute + memory ops)
- **FR-8.4:** Continuous batching for serving workloads
- **FR-8.5:** Speculative decoding with draft model verification

### FR-9: Autotuning
- **FR-9.1:** Tile/block size search for GEMM kernels
- **FR-9.2:** Pipeline stage count optimization
- **FR-9.3:** Thompson sampling for explore/exploit balance
- **FR-9.4:** Persistent tuning cache per GPU architecture
- **FR-9.5:** Dream-cycle tuning during low-load periods
- **FR-9.6:** Cost model calibration from empirical measurements

### FR-10: Python API
- **FR-10.1:** `warp.compile(path)` — load and compile a model
- **FR-10.2:** `model.generate(tokens, ...)` — autoregressive generation
- **FR-10.3:** `model.forward(tensors)` — single forward pass
- **FR-10.4:** `model.profile()` — get profiling report
- **FR-10.5:** `model.tier` — current compilation tier
- **FR-10.6:** Zero-copy tensor interop with PyTorch via DLPack
- **FR-10.7:** Async generation with streaming token output

---

## Performance Targets

| Metric | TRT-LLM Baseline | Warp Target | How |
|--------|-------------------|-------------|-----|
| GEMM (FP16, 4090) | ~85% peak TFLOPS | >90% | Shape-specialized + autotuned |
| Attention (BF16, H100) | 840 TFLOPS/s | >=840 | FlashAttention-3 equivalent |
| TTFT (Llama-7B, batch=1) | 15-25ms | <15ms | Persistent kernels + CUDA graphs |
| Decode throughput (batch=32) | Baseline | >1.15x | Better fusion + memory reuse |
| KV cache efficiency | ~90% | >96% | PagedAttention/vAttention |
| Compilation (7B model) | 2-10 min | <5 sec | AOT Rust compilation |
| Dynamic shape overhead | 5-15% vs static | <5% | Symbolic shapes + JIT buckets |
| Cross-platform parity | N/A | Metal within 20% of CUDA | LLVM shared backend |
| Cold start to first token | Minutes (TRT) | <1 sec (Tier 0) | Tiered compilation |

---

## Dependencies

### Build from scratch
- [x] Two-level IR (graph + tensor)
- [x] Pattern-matching fusion optimizer
- [x] Tiered compiler with hot-swap
- [x] Runtime profiler
- [ ] Cost-model fusion optimizer
- [ ] Hebbian fusion discovery
- [ ] KV cache manager
- [ ] Continuous batching scheduler
- [ ] Autotuning framework
- [ ] Tensor-level IR (tiling, vectorization)

### Adopt / Wrap
- [ ] `inkwell` — Safe LLVM IR builder for Rust
- [ ] `cudarc` — CUDA driver API + NVRTC + cuBLAS
- [ ] `metal` crate — Apple Metal compute
- [ ] `pyo3` + `maturin` — Python bindings
- [ ] `safetensors` crate — model weight loading
- [ ] FlashAttention/FlashInfer — precompiled attention kernels (initially)

### Learn from (architecture influence)
- Triton's progressive lowering (TTIR → TTGIR → LLVM)
- TVM Relax symbolic shape tracking
- XLA's fixed-op-set canonical representations
- ThunderKittens tile-level abstractions
- tinygrad's minimal primitive set
- Didymus Hebbian co-activation / plasticity
- Didymus Cell lifecycle / signal measurement
- Didymus gardener pattern / active inference

---

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| LLVM dependency hell (versions, build times) | High | High | Start with string PTX (done), migrate to LLVM incrementally. Consider cranelift for fast iteration |
| Can't match cuBLAS GEMM quality | High | Medium | Start by wrapping cuBLAS via cudarc, replace with custom kernels incrementally. Autotuning closes gap |
| Metal perf significantly behind CUDA | Medium | Medium | Accept 20% gap initially. Metal has no Tensor Core equivalent — different optimization strategy needed |
| Per-GPU-generation tuning burden | Medium | High | Autotune cache per arch. Sensible defaults for Ada (4090), Hopper (H100), Apple M-series |
| Python ecosystem integration friction | Medium | Medium | Use DLPack for zero-copy PyTorch interop. Mirror familiar API (model.generate()) |
| FP8/FP4 numerical accuracy | Medium | Medium | Port FlashAttention-3's incoherent processing. Validate against HF reference outputs |
| Scope creep (too many features) | High | High | Phase strictly. Each phase has a benchmark target. Don't move forward until target is hit |

---

## Development Phases

### Phase 1: Foundation (CURRENT — DONE)
**Target: compilable IR with fusion and codegen skeleton**
- [x] Graph IR with SSA values and full transformer op vocabulary
- [x] Shape system (static, dynamic, symbolic)
- [x] DType system (F32 through FP8, quantized types)
- [x] Pattern-matching fusion engine
- [x] Three fusion passes (MatMul+Bias, MatMul+Bias+Act, Residual+RMSNorm)
- [x] PTX codegen backend (elementwise + activations)
- [x] Metal codegen backend (elementwise + activations)
- [x] Arena memory pool with lifetime planning
- [x] Execution plan builder and scheduler
- [x] End-to-end compile pipeline
- [x] 42 tests passing

### Phase 2: Tiered Compilation (CURRENT — DONE)
**Target: self-optimizing runtime**
- [x] Runtime profiler (timings, shapes, hardware counters)
- [x] Optimization hint generation
- [x] Bottleneck classification (memory/compute/launch bound)
- [x] Four-tier compiler (Tier 0→3)
- [x] Atomic hot-swap via Arc<RwLock<VersionedPlan>>
- [x] Tier policy configuration
- [x] Plan history and regression tracking
- [x] Concurrent plan access from inference threads
- [x] 42 tests passing

### Phase 3: Real GPU Execution
**Target: run kernels on 4090, benchmark against cuBLAS**
- [ ] `cudarc` integration (device management, memory, kernel launch)
- [ ] PTX loading and JIT compilation via NVRTC
- [ ] Real GEMM kernel (tiled, shared memory, FP16 Tensor Cores)
- [ ] Real elementwise fused kernels running on GPU
- [ ] CUDA Graph capture/replay for decode steps
- [ ] Benchmark: elementwise ops vs PyTorch
- [ ] Benchmark: GEMM vs cuBLAS
- **Exit criteria:** GEMM within 85% of cuBLAS on 4090

### Phase 4: Attention & Model Loading
**Target: run a real transformer layer end-to-end**
- [ ] FlashAttention-equivalent fused attention kernel
- [ ] Rotary position embedding kernel
- [ ] SafeTensors model weight loader
- [ ] GGUF model weight loader
- [ ] LLaMA architecture auto-detection and graph construction
- [ ] Single transformer layer forward pass (attention + FFN)
- [ ] KV cache allocation and management
- **Exit criteria:** single layer matches HuggingFace output within FP16 tolerance

### Phase 5: LLVM Codegen
**Target: replace string-template codegen with real LLVM IR emission**
- [ ] `inkwell` integration for LLVM IR building
- [ ] LLVM NVPTX backend for PTX generation
- [ ] LLVM Metal AIR backend for Apple
- [ ] Shared optimization passes at LLVM IR level
- [ ] Tensor-level IR (tiling, vectorization, loop nests)
- [ ] Register allocation awareness
- **Exit criteria:** LLVM-generated GEMM matches or beats string-template version

### Phase 6: Cost-Model Optimization & Hebbian Fusion
**Target: optimizer that discovers novel fusions automatically**
- [ ] Cost-model-based fusion (DP on topological order)
- [ ] Hardware cost model calibrated from profiling data
- [ ] Hebbian co-activation tracking (port from plasticity.rs concepts)
- [ ] LTP/LTD-style reinforcement for kernel pair affinities
- [ ] Novel fusion pattern discovery from co-activation data
- [ ] SwiGLU full-block fusion
- [ ] Cross-layer fusion (attention output + residual + norm)
- **Exit criteria:** discovers at least one fusion pattern not in hand-written rules

### Phase 7: Full Model Inference
**Target: run Llama-7B end-to-end, benchmark against TRT-LLM**
- [ ] Full LLaMA-7B graph construction from weights
- [ ] Autoregressive token generation loop
- [ ] PagedAttention KV cache
- [ ] Continuous batching scheduler
- [ ] Speculative decoding with draft model
- [ ] FP16 and FP8 execution paths
- [ ] End-to-end TTFT and throughput benchmarks
- **Exit criteria:** TTFT within 20% of TRT-LLM, throughput within 10%

### Phase 8: Autotuning
**Target: self-tuning kernels that approach peak hardware utilization**
- [ ] Autotuning framework (search space definition, benchmarking harness)
- [ ] Thompson sampling for explore/exploit
- [ ] Persistent tuning cache per GPU architecture
- [ ] Dream-cycle tuning during low-load periods
- [ ] Cost model calibration from empirical measurements
- [ ] Tile size optimization for GEMM
- [ ] Pipeline stage optimization
- **Exit criteria:** autotuned GEMM >= 90% peak TFLOPS on 4090

### Phase 9: Cross-Platform & Production
**Target: run on M3 Ultra, Python API, production-ready**
- [ ] Metal backend via LLVM (or direct MSL generation)
- [ ] Metal compute shader execution via `metal` crate
- [ ] SPIR-V backend for AMD/Intel/Vulkan
- [ ] Python bindings via PyO3 + maturin
- [ ] `warp.compile()` and `model.generate()` API
- [ ] DLPack zero-copy PyTorch interop
- [ ] Async streaming generation
- [ ] Error handling and recovery
- **Exit criteria:** same model runs on 4090 and M3 Ultra from Python

### Phase 10: Beat TensorRT
**Target: measurably faster than TRT-LLM on real workloads**
- [ ] Warp-specialized kernels for Hopper+
- [ ] Persistent decode kernels
- [ ] FP4 support for Blackwell
- [ ] Distributed inference (tensor parallelism across GPUs)
- [ ] vAttention (CUDA virtual memory KV cache)
- [ ] Production benchmarks (Llama 7B/13B/70B, Mistral, Qwen)
- [ ] Published benchmark suite reproducible by anyone
- **Exit criteria:** beat TRT-LLM TTFT on at least 2 model sizes

---

## Cognitive Architecture Connections

These are not metaphors — they are direct algorithmic transfers from Didymus:

| Didymus Module | Warp Module | Transfer |
|---------------|-------------|----------|
| `plasticity.rs` (Hebbian LTP/LTD) | `optimizer/fusion.rs` | Co-activation tracking for kernel pairs → fusion affinity |
| Cell lifecycle (sense→process→learn) | `tiered.rs` (infer→profile→recompile) | Same state machine, different domain |
| Signal measurement | `profile.rs` | Measure execution quality, generate optimization hints |
| Gardener agent | Background compiler thread | Watches runtime signal, evolves infrastructure |
| Active inference (free energy) | Profile-guided compilation | Minimize gap between predicted and actual performance |
| Thompson sampling | `autotune/` | Explore tile configs vs exploit known-good |
| Dream cycles (NREM/REM) | Idle-time autotuning | Consolidate and optimize during low load |
| Salience-based memory | Hot kernel prioritization | Not all kernels need optimization — focus on high-impact |
| Co-activation tracking | Kernel adjacency analysis | Which ops always run together → fusion candidates |

---

## Success Metrics

1. **Speed:** Beat TRT-LLM TTFT on Llama-7B by >10%
2. **Startup:** First token in <1 second (vs minutes for TRT compilation)
3. **Self-improvement:** Measurable speedup from Tier 0 → Tier 3 on real workloads
4. **Cross-platform:** Same model runs on CUDA and Metal from same API
5. **Adoption:** Python API simple enough that HuggingFace users can switch with <5 lines
6. **Didymus integration:** Thomas inference latency reduced by >30%

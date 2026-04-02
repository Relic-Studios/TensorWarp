# Warp — Technical Specification
## Implementation Details and Data Structures

**Version:** 0.2.0
**Date:** 2026-04-01

---

## 1. Intermediate Representation

### 1.1 Two-Level IR Design

Warp uses a two-level IR following the proven Triton/TVM pattern:

**Graph IR (Level 1) — `warp-ir` crate:**
- SSA-style dataflow graph
- Nodes produce Values, edges connect producers to consumers
- Arena-allocated with ID-based references (no Rc/RefCell)
- Topological ordering maintained incrementally (Kahn's algorithm)
- Use-def chains for O(1) user queries
- Metadata (shape, dtype, layout) attached to Values, not Nodes

**Tensor IR (Level 2) — to be added in `warp-ir`:**
- Represents within-kernel computation
- Loop nests with tiling annotations
- Shared memory staging operations
- Async copy operations (TMA on Hopper)
- Register allocation hints
- Warp-level primitives (shuffle, vote, mma)

### 1.2 Op Vocabulary

Fixed op set (XLA-inspired) covering all transformer variants:

```
Compute:      MatMul, BatchMatMul, Binary, Unary, Activate, Softmax, Reduce
Normalization: LayerNorm, RmsNorm
Attention:    Attention (fused SDPA), PagedAttention, RotaryEmbed
Shape:        Reshape, Transpose, Concat, Split, Gather, Slice
Embedding:    Embedding
MoE:          MoeGate, MoeExpert
Quantization: Quantize, Dequantize, QuantizedMatMul
Fused:        FusedMatMulBias, FusedMatMulBiasAct, FusedResidualRmsNorm, FusedSwiGLU
Speculative:  SpeculativeVerify
Creation:     Input, Constant
```

### 1.3 Shape System

Three levels of shape knowledge:
- **Static:** All dimensions known at graph construction → most aggressive optimization
- **Symbolic:** Some dimensions are variables (e.g., batch size) → compile once, specialize at runtime
- **Dynamic:** Unknown until runtime → JIT compilation with shape bucketing

Symbolic dimensions share variable IDs — all dims with the same ID must match at runtime.

### 1.4 DType System

Full precision hierarchy:
```
Float:     F32, F16, BF16, F8E4M3, F8E5M2
Integer:   I64, I32, I16, I8, I4, U8, U32
Quantized: Q8_0, Q4_0, Q4_1 (block-scaled, GGUF-compatible)
Special:   Bool
```

Every dtype knows:
- Bit width and byte size
- Compute promotion type (F16 → F32 for arithmetic)
- Whether it needs special hardware (FP8 → Ada+)

---

## 2. Optimizer

### 2.1 Pass Pipeline

Passes run in fixed order. Higher optimization levels enable more passes:

```
O0: No passes (Tier 0 — instant)
O1: FuseMatMulBiasAct → FuseMatMulBias → FuseResidualRmsNorm → DCE
O2: O1 + CostModelFusion → LayoutOptimization → OperatorScheduling
O3: O2 + ShapeSpecialization → AutotuneHints → HebbianFusion
```

### 2.2 Pattern Matching

Current implementation:
- Tree-structured patterns (root = output op, leaves = input ops)
- Predicate-based op matching (closures, not enum comparison)
- Single-user constraint on interior nodes (prevents incorrect fusion)
- Dead-node filtering (skip nodes with no live consumers)
- Non-overlapping match scanning

### 2.3 Cost-Model Fusion (Phase 6)

Replace pattern matching with dynamic programming:

```
For each node in topological order:
    For each possible fusion group ending at this node:
        cost = estimate_kernel_cost(fused_group)
        if cost < sum(individual_costs):
            record fusion opportunity
    Select minimum-cost partitioning via DP
```

Cost model inputs (from profiler):
- Measured memory bandwidth per kernel
- Measured compute throughput per kernel
- Kernel launch overhead (~5-10μs per launch)
- Shared memory requirements for fused group
- Register pressure estimate

### 2.4 Hebbian Fusion Discovery (Phase 6)

Port co-activation tracking from Didymus plasticity:

```rust
struct KernelPairAffinity {
    /// How often these kernels run back-to-back
    coactivation_count: u64,
    /// Whether they share memory buffers
    shares_memory: bool,
    /// Combined memory traffic (bytes read + written)
    combined_traffic: usize,
    /// LTP-style reinforcement (increases with coactivation)
    affinity: f64,
    /// LTD-style decay (decreases without coactivation)
    last_seen: Instant,
}
```

High-affinity pairs become fusion candidates. The optimizer tries fusing them and measures if it's actually faster. If yes → permanent fusion rule. If no → affinity decays (LTD).

This discovers fusions that no hand-written pattern would find, because it's driven by actual execution patterns rather than human intuition.

---

## 3. Code Generation

### 3.1 Current Implementation (String Templates)

Phase 1-2 use parameterized string templates:
- PTX assembly text targeting SM 8.9 (Ada/4090)
- Metal Shading Language targeting Apple Silicon
- Shape-specialized (exact dimensions baked in)
- Covers: elementwise binary, activations (ReLU, GELU, SiLU)

### 3.2 LLVM Codegen (Phase 5)

Target architecture:
```
Warp Graph IR
    → Tensor IR (tiling, vectorization)
        → LLVM IR (via inkwell)
            → NVPTX backend → PTX → cubin (NVIDIA)
            → Metal AIR backend → metallib (Apple)
            → SPIR-V (AMD/Intel/Vulkan)
```

Key LLVM IR patterns:
- **GEMM:** Tiled loop nest with shared memory staging, async copy (cp.async), mma.sync for Tensor Cores
- **Attention:** Block-level Q/K/V processing, online softmax, shared memory for K/V tiles
- **Elementwise fusion:** Merge adjacent elementwise ops into single kernel with vectorized loads (float4)
- **RMSNorm:** Two-pass (compute variance, then normalize) or fused single-pass with Welford's algorithm

### 3.3 Critical Kernel Templates

**GEMM (Phase 3):**
```
Input: A[M,K] × B[K,N] → C[M,N]
Tiling: TILE_M × TILE_N × TILE_K (autotuned)
Pipeline: load tile → shared memory → warp-level mma → accumulate → store
Precision: FP16 compute with FP32 accumulation
Target: >85% peak TFLOPS on 4090 (330 TFLOPS FP16)
```

**FlashAttention (Phase 4):**
```
Input: Q[B,H,N,D], K[B,H,S,D], V[B,H,S,D]
Algorithm: block-wise Q×K^T → online softmax → ×V
Memory: O(N) instead of O(N²) — never materializes full attention matrix
Causal mask: implicit via block boundaries
GQA: K/V heads repeated for Q head groups
Target: match FlashAttention-2 throughput
```

**PagedAttention (Phase 7):**
```
Input: Q[B,H,1,D], K_cache[num_blocks,H,block_size,D], V_cache[same]
Block table: maps logical position → physical block
Decode-optimized: Split-K across sequence length
Target: match vLLM PagedAttention throughput
```

---

## 4. Runtime

### 4.1 Memory Management

Three-tier allocation strategy:

1. **Static planning (default):** At compile time, compute tensor lifetimes via liveness analysis. Assign non-overlapping tensors to shared memory regions. Greedy allocation sorted by size descending.

2. **Slab allocator (fallback):** For dynamic shapes. Pre-allocate slabs for common sizes (256B, 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB). O(1) alloc/free.

3. **CUDA stream-ordered allocator (KV cache):** For growing KV cache. Uses CUDA's built-in stream-ordered allocator with auto-defragmentation.

All allocations aligned to 256 bytes (GPU cache line).

### 4.2 Tiered Compilation State Machine

```
                    ┌─────────────────────┐
                    │      Tier 0         │
                    │  (instant startup)  │
                    └─────────┬───────────┘
                              │ always (after min_time)
                    ┌─────────▼───────────┐
                    │      Tier 1         │
                    │  (pattern fusion)   │
                    └─────────┬───────────┘
                              │ after min_samples_for_tier2
                    ┌─────────▼───────────┐
                    │      Tier 2         │
                    │ (profile-guided)    │
                    └─────────┬───────────┘
                              │ after min_samples_for_tier3
                    ┌─────────▼───────────┐
                    │      Tier 3         │
                    │   (autotuned)       │
                    └─────────────────────┘

Hot-swap mechanism:
- Active plan: Arc<RwLock<VersionedPlan>>
- Inference threads: read lock (concurrent, non-blocking)
- Compiler thread: write lock (blocks briefly for swap)
- Old plan: dropped when last reference released
```

### 4.3 Profiler Data Flow

```
Inference thread:
    for each kernel in execution_plan:
        start = Instant::now()
        launch_kernel(...)
        record_sample(node_id, elapsed, input_shapes, hw_counters)
    profiler.record_inference()

Background compiler thread:
    loop:
        sleep(check_interval)
        if profiler.total_inferences >= threshold:
            hints = profiler.generate_hints()
            new_plan = compile_with_hints(graph, hints)
            hot_swap(new_plan)
            profiler.reset()
```

### 4.4 KV Cache Design (Phase 7)

PagedAttention implementation:
```rust
struct KvCache {
    /// Physical block pool on GPU
    block_pool: Vec<GpuBlock>,        // [num_blocks × block_size × head_dim × 2(K+V)]
    /// Free block list
    free_blocks: VecDeque<BlockId>,
    /// Per-sequence block tables
    block_tables: HashMap<SeqId, Vec<BlockId>>,
    /// Block size (tokens per block, typically 16)
    block_size: usize,
}
```

### 4.5 Autotuning Framework (Phase 8)

```rust
struct AutotuneSearch {
    /// Configurations to try
    search_space: Vec<KernelConfig>,
    /// Results so far: config → measured time
    results: BTreeMap<Duration, KernelConfig>,
    /// Thompson sampling state
    prior: BetaDistribution,
    /// Best known configuration
    best: Option<(KernelConfig, Duration)>,
}

// Dream-cycle integration:
// When inference load < 10% for > 30 seconds,
// run autotuning on hot kernels (top-5 by total time)
```

---

## 5. Benchmarking Strategy

### 5.1 Micro-benchmarks (per-kernel)
- GEMM: sweep M/N/K from 128 to 16384, FP16/FP32/FP8
- Attention: sweep seq_len from 128 to 32768, various head configs
- Elementwise fusion: chain length 1-8 ops
- Compare against: cuBLAS, cuDNN, FlashAttention-2, Triton

### 5.2 Layer-level benchmarks
- Single transformer block (attention + FFN)
- Various configs: LLaMA-7B, Mistral-7B, Qwen-7B dimensions
- Compare against: TRT-LLM, torch.compile

### 5.3 End-to-end benchmarks
- TTFT (time to first token) at batch=1
- Decode throughput (tokens/sec) at batch=1,8,32,128
- Memory usage (peak GPU memory)
- Compare against: TRT-LLM, vLLM, SGLang

### 5.4 Tiered compilation benchmarks
- Time from cold start to first token (Tier 0)
- Measured speedup at each tier transition
- Total time to reach Tier 3
- Regression rate (how often a new tier is slower)

---

## 6. File-Level Implementation Plan

### Phase 3 files:
```
crates/runtime/src/cuda.rs       — cudarc device/memory/stream management
crates/runtime/src/cuda_graph.rs — CUDA Graph capture and replay
crates/kernels/Cargo.toml        — new crate for kernel templates
crates/kernels/src/lib.rs
crates/kernels/src/gemm.rs       — tiled GEMM kernel (PTX template)
crates/kernels/src/elementwise.rs — fused elementwise kernels
benches/gemm_bench.rs            — cuBLAS comparison benchmark
```

### Phase 4 files:
```
crates/kernels/src/attention.rs  — FlashAttention-equivalent kernel
crates/kernels/src/rope.rs       — rotary position embedding
crates/loader/Cargo.toml         — new crate
crates/loader/src/lib.rs
crates/loader/src/safetensors.rs — SafeTensors format loader
crates/loader/src/gguf.rs        — GGUF format loader
crates/loader/src/llama.rs       — LLaMA graph builder
crates/runtime/src/kv_cache.rs   — KV cache management
```

### Phase 5 files:
```
crates/codegen/src/llvm.rs       — LLVM IR builder (via inkwell)
crates/codegen/src/llvm_ptx.rs   — NVPTX-specific lowering
crates/codegen/src/llvm_metal.rs — Metal AIR-specific lowering
crates/ir/src/tensor_ir.rs       — Tensor-level IR (tiling, loops)
crates/ir/src/lower.rs           — Graph IR → Tensor IR lowering
```

### Phase 6 files:
```
crates/optimizer/src/cost_model.rs — Hardware cost model
crates/optimizer/src/dp_fusion.rs  — DP-based fusion optimizer
crates/optimizer/src/hebbian.rs    — Co-activation tracking + fusion discovery
```

### Phase 7 files:
```
crates/runtime/src/kv_cache.rs    — PagedAttention block manager
crates/runtime/src/scheduler.rs   — Continuous batching scheduler
crates/runtime/src/speculative.rs — Speculative decoding runtime
crates/loader/src/mistral.rs      — Mistral graph builder
crates/loader/src/qwen.rs         — Qwen graph builder
```

### Phase 8 files:
```
crates/autotune/Cargo.toml       — new crate
crates/autotune/src/lib.rs
crates/autotune/src/search.rs    — Thompson sampling search
crates/autotune/src/cache.rs     — Persistent tuning cache
crates/autotune/src/dream.rs     — Idle-time tuning scheduler
```

### Phase 9 files:
```
crates/python/Cargo.toml         — PyO3 crate
crates/python/src/lib.rs
crates/python/src/model.rs       — Python Model class
crates/python/src/tensor.rs      — DLPack tensor interop
python/warp/__init__.py           — Python package
python/warp/model.py              — High-level API
```

---

## 7. Testing Strategy

- **Unit tests:** Every module has tests. Graph operations, fusion correctness, shape arithmetic, codegen output validation.
- **Integration tests:** End-to-end compile pipeline, tier progression, hot-swap correctness.
- **Numerical tests:** Compare kernel outputs against PyTorch reference (FP16 tolerance: atol=1e-3, rtol=1e-3).
- **Performance tests:** Benchmark suite that runs on CI, tracks regressions.
- **Stress tests:** Concurrent inference + compilation, memory pressure, shape variety.

Current: 42 tests across 4 crates. Target: >200 tests by Phase 7.

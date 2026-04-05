# Research Prompt: Marlin-Style Adaptive Quantized GEMM Architecture for Universal GPU Inference

## Objective

TensorWarp is a from-scratch Rust+CUDA inference engine that currently achieves 28.6 tok/sec on Qwen 7B Q4_0 on an RTX 4090 — approximately 22% of the theoretical bandwidth floor (120 tok/sec). We have exhausted kernel-level optimizations (M=1 specialization, block-major weight reordering, warp-cooperative loading experiments) and determined that the remaining 78% gap is fundamentally caused by the Q4_0 block format's incompatibility with GPU memory transaction patterns.

**The core question: How do production systems like Marlin, GPTQ-Marlin, ExLlamaV2, and AWQ achieve 50-80% bandwidth utilization for INT4 GEMM at M=1, and how can TensorWarp implement an adaptive quantized GEMM system that automatically selects optimal weight layouts, dequantization strategies, and kernel configurations across all NVIDIA GPU architectures (Turing through Blackwell) and all model families?**

This research must produce implementation-ready specifications, not conceptual overviews.

## Context

### Current State / Pain Point

TensorWarp's Q4_0 implementation stores weights in 20-byte blocks (4-byte f32 scale + 16-byte packed nibbles, 32 elements per block). The M=1 GEMM kernel assigns one thread per output column, iterating over K-blocks. Despite block-major reordering for improved coalescing, each 20-byte block occupies a non-power-of-2 footprint that wastes cache line bandwidth:

- 20-byte blocks on 128-byte cache lines: ~85% wasted fetches for column-major layout
- Block-major layout reduces stride to 20 bytes between adjacent threads — the GPU hardware coalescer handles this, achieving ~22% bandwidth utilization
- Shared-memory staging experiments added sync overhead that negated coalescing gains (22,000 __syncthreads per token)
- The kernel is provably memory-bandwidth-bound (4.8 FLOP/byte vs 81.9 FLOP/byte roofline), yet achieves only 22% of peak bandwidth

The engine supports RTX 4090 (SM 8.9, Ada Lovelace, 128 SMs, 1008 GB/s, 24 GB VRAM, 48KB/96KB shared memory per SM, 64K registers per SM, 72 MB L2 cache). The codebase is 42K+ lines of Rust with inline CUDA C kernels compiled via NVRTC at runtime. Kernels are cached by source hash. The system uses cudarc (vendored, Rust CUDA bindings) and cuBLAS for FP16/F32 GEMM.

Verified performance baselines:
- Qwen 7B F32 (TF32 tensor cores): 107 tok/sec on 0.5B, 14-23 tok/sec on 7B
- Qwen 7B Q4_0: 28.6 tok/sec on 7B (from garbage output to correct in this session)
- cuBLAS FP16 HGEMM: 145 TFLOPS on 4090
- Q4→FP16 full dequant + cuBLAS: 0.3 tok/sec on 7B (17.4 GB weights don't fit alongside KV cache)

### What We Want

- INT4 GEMM at M=1 achieving 60-80% of peak memory bandwidth on RTX 4090 (60-96 tok/sec for 7B Q4)
- A weight layout and kernel system that adapts to:
  - Different GPU architectures (Turing SM 7.5, Ampere SM 8.0/8.6, Ada SM 8.9, Hopper SM 9.0, Blackwell SM 10.0)
  - Different quantization formats (Q4_0, Q4_K, GPTQ group-128, AWQ, GGUF variants)
  - Different model shapes (hidden_size from 896 to 8192+, FFN dims, GQA ratios)
  - Batch size 1 (decode) through batch size 32+ (continuous batching)
- The system should be a compiler/runtime that generates optimized kernels — not a library of hand-tuned kernels for specific shapes
- Must work within 24 GB VRAM constraint for 7B models (no full FP16 weight expansion)
- Quantization format conversion should be a one-time offline/load-time cost

## Key Questions

### 0. Marlin Weight Layout and Dequantization Pipeline

0a. **What is Marlin's exact memory layout for INT4 weights, and why does it achieve near-ideal bandwidth?**
    - How are 4-bit nibbles packed relative to warp-level MMA instruction consumption patterns?
    - What is the "striped partitioning scheme" — how are tiles distributed across SMs?
    - How does the offline weight pre-processing ("pre-shuffling") transform standard row-major INT4 weights into Marlin format?
    - What are the exact byte-level transformations? Provide the reorder algorithm.
    - How does Marlin handle the f16 group scales (group size 128)? Are scales interleaved with quantized data or stored separately?

0b. **How does Marlin pipeline async global loads with Tensor Core math?**
    - What is the `cp.async` pipeline depth (number of stages)?
    - How are the `lop3` bitwise PTX tricks used for INT4→FP16 dequantization? What are the exact PTX instructions?
    - How does Marlin ensure both CUDA core ALUs (for dequant) and Tensor Cores (for MMA) remain saturated simultaneously?
    - What is the register budget per thread, and how does Marlin avoid spilling?

0c. **What is Marlin's cache eviction strategy?**
    - The research mentions "strict cache eviction policy to prevent L2 pollution." How is this implemented — `cp.async` with cache hints? `ld.global.cg`? Explicit cache control PTX?
    - How does this interact with the A-vector (activation) which DOES benefit from L2 caching?

0d. **How does Marlin's performance profile change with batch size?**
    - At M=1 (pure decode): what percentage of peak bandwidth does Marlin achieve?
    - At what batch size does Marlin lose its advantage over cuBLAS operating on pre-dequantized FP16?
    - How does ExLlamaV2's simpler "dequant to temp buffer + cuBLAS HGEMM" approach compare at M=1 through M=32?

### 1. Adaptive Kernel Generation Across GPU Architectures

1a. **What are the architectural differences between Turing/Ampere/Ada/Hopper/Blackwell that affect INT4 GEMM kernel design?**
    - Tensor Core instruction sets: `wmma` vs `mma.sync` vs `wgmma` — which INT4/INT8 modes are available on which SMs?
    - Shared memory sizes (48KB vs 100KB vs 228KB) and how they constrain tile sizes
    - Register file sizes and occupancy trade-offs per architecture
    - L2 cache sizes (6MB Turing → 72MB Ada → 50MB Hopper) and their impact on A-vector caching
    - Async copy capabilities (`cp.async` on Ampere+, TMA on Hopper+)
    - Does Ada SM 8.9 support native INT4 Tensor Core instructions, or must we use the INT8 double-packing trick?

1b. **How do production systems like TensorRT-LLM and vLLM handle multi-architecture kernel dispatch?**
    - Do they ship pre-compiled PTX/SASS for each architecture, or generate at runtime?
    - How do they select tile sizes, pipeline depths, and instruction sequences per SM version?
    - What is the CUTLASS approach to architecture-adaptive GEMM? How does CUTLASS 3.x differ from 2.x in handling different SM versions?

1c. **What is the minimum viable "kernel template" system for generating architecture-specific INT4 GEMM kernels?**
    - Can we use NVRTC (runtime PTX compilation) to generate architecture-specific kernels from parameterized templates?
    - What parameters need to be architecture-specific vs shape-specific vs format-specific?
    - How does TensorWarp's existing autotuner (Thompson sampling over shape buckets) integrate with architecture-specific kernel selection?

### 2. Quantization Format Universality

2a. **What are the exact byte-level differences between Q4_0, Q4_K, GPTQ (group-128), AWQ, and GGUF Q4 variants?**
    - Scale storage: per-block vs per-group vs per-channel. Float16 vs float32 scales.
    - Zero-point handling: symmetric (Q4_0) vs asymmetric (GPTQ with zero-points). How does zero-point affect the dequantization ALU cost?
    - Two-tier scaling (Q4_K superblocks): what is the exact structure and how does it affect kernel design?
    - Bit packing order: which formats use low-nibble-first vs high-nibble-first? Does this matter for vectorized unpacking?

2b. **Can a single kernel template handle multiple quantization formats through parameterization, or must each format have its own kernel?**
    - What are the performance costs of format abstraction (function pointers, template branching, etc.)?
    - How does llama.cpp handle format dispatch in its GGML backend?
    - Is there a "canonical internal format" that all external formats should be converted to at load time?

2c. **How should TensorWarp's offline weight processing pipeline be designed?**
    - Load from any format (SafeTensors, GGUF, GPTQ checkpoint) → convert to TensorWarp internal format → store cached
    - Should the internal format be architecture-specific (different layouts for Ada vs Hopper)?
    - How does Marlin's offline pre-processing cost scale with model size? Is it feasible to do at every cold start?

### 3. Model Architecture Adaptation

3a. **What architectural features across LLaMA, Qwen, Gemma, Mistral, Phi, and GPT-NeoX families affect the quantized GEMM kernel requirements?**
    - Different hidden sizes: how do non-power-of-2 dimensions (e.g., Qwen's 896, 3584) interact with tile sizes designed for power-of-2 (e.g., 128×128)?
    - GQA ratios: how does the KV projection dimension (often much smaller than hidden) affect GEMM efficiency? Is there a minimum N below which custom kernels lose to cuBLAS?
    - Biases vs no biases: does adding bias into the GEMM kernel (beta=1) significantly affect the quantized GEMM design?

3b. **How should the system handle Mixture-of-Experts (MoE) architectures?**
    - Sparse GEMM: when only 2 of 8 experts are active, does the quantized GEMM approach change?
    - Expert parallelism: how do persistent kernels interact with quantized weight loading?

3c. **How do different activation functions (SiLU/SwiGLU, GELU, GeGLU) affect kernel fusion opportunities with quantized GEMMs?**
    - Can the dequant + GEMM + activation be fused into a single kernel for M=1?
    - What is the register pressure cost of fusing activation into the GEMM epilogue?

### 4. Compiler and Runtime Design

4a. **How should TensorWarp's tiered compilation pipeline be extended for quantized GEMM?**
    - Tier 0 (instant): use cuBLAS with per-layer dequant to temp buffer — works everywhere, moderate speed
    - Tier 1 (fast): select from pre-compiled kernel templates based on shape + architecture + format
    - Tier 2 (tuned): autotuned tile sizes and pipeline parameters via Thompson sampling
    - Tier 3 (optimal): Marlin-style fully optimized kernel for the specific model on the specific GPU
    - How long should each tier take to compile? What is acceptable cold-start latency?

4b. **What is the right abstraction boundary between the compiler and the runtime?**
    - Which decisions are made at compile time (weight layout, tile sizes) vs runtime (batch size routing, KV cache length)?
    - How does vLLM's model runner handle the compile/runtime split for quantized models?
    - Should TensorWarp cache compiled kernels to disk (keyed by model hash + GPU architecture)?

4c. **How should the system profile and select between kernel variants at runtime?**
    - For a given (M, N, K, format, architecture) tuple, how many kernel variants should exist?
    - What is the profiling cost for autotuning, and how does it amortize across the decode loop?
    - How does TensorRT-LLM's algorithm selection for quantized GEMMs work in practice?

### 5. Memory Management for Quantized Inference

5a. **What is the optimal VRAM budget allocation for a 7B model on 24 GB?**
    - Weights (Q4): ~4.2 GB. Dequant temp buffer: ? KV cache: ? Activations: ? CUDA context: ?
    - How large should the FP16 dequant temp buffer be if using per-GEMM dequant? Per-layer? Per-projection?
    - Can the temp buffer be shared across layers (double-buffered with the GEMM execution)?

5b. **How do production systems manage memory for quantized weights that are stored in a different layout than their "active" format?**
    - Do they keep both the original and the pre-shuffled weights, or discard the original after conversion?
    - How does this interact with model-parallel / tensor-parallel sharding across GPUs?

5c. **What is the memory overhead of Marlin's pre-shuffled format compared to the raw INT4 data?**
    - Does the pre-shuffling add padding or alignment bytes?
    - How does group scale storage overhead compare across formats?

### 6. PTX-Level Implementation Details

6a. **What are the exact PTX instructions used by Marlin for INT4 dequantization?**
    - The `lop3` (ternary logic operation) trick for nibble extraction — what are the truth tables and operand layouts?
    - How is the unsigned 4-bit value (0-15) converted to signed (-8 to +7) using bitwise operations vs subtraction?
    - What is the full PTX instruction sequence from packed INT4 bytes to FP16 values ready for MMA?

6b. **How does NVRTC handle PTX-level intrinsics from CUDA C source?**
    - Can we use inline PTX (`asm volatile`) in NVRTC-compiled kernels?
    - What are the compile flags needed for specific SM targets?
    - How does TensorWarp's existing NVRTC pipeline (compile PTX string → load module → get function) interact with architecture-specific code?

6c. **What is the performance difference between CUDA C dequantization (bit shifts, masks) and PTX-level dequantization (lop3, prmt)?**
    - How many ALU cycles does a naive CUDA C nibble extraction take vs optimized PTX?
    - At M=1, is the dequantization ALU cost actually on the critical path, or is it hidden behind memory latency?

### 7. Benchmarking and Validation

7a. **What benchmark methodology should be used to isolate GEMM kernel performance from end-to-end inference overhead?**
    - How to measure effective bandwidth utilization of a single kernel launch vs the full decode loop?
    - What is the standard way to account for L2 cache warming effects (first vs steady-state calls)?
    - How do ExLlamaV2 and llama.cpp benchmark their quantized GEMM performance?

7b. **What are the published bandwidth utilization numbers for production INT4 GEMM kernels on RTX 4090?**
    - Marlin: claimed ~4x over FP16 at M=1 — what is the absolute GB/s?
    - ExLlamaV2: ~32.5 tok/sec on 7B — what bandwidth utilization does this imply?
    - llama.cpp Q4_K_M on RTX 4090: what are the actual tok/sec numbers?
    - TensorRT-LLM W4A16: published benchmarks on consumer GPUs?

## Desired Output

The research should produce:

1. **Marlin weight layout specification**: Exact byte-level format description with worked example for a [3584, 3584] weight matrix. Include the offline reorder algorithm in pseudocode.

2. **PTX dequantization recipe**: Complete PTX instruction sequence for INT4→FP16 dequant on Ada (SM 8.9), with cycle counts and register usage.

3. **Architecture feature matrix**: Table mapping SM versions to available instructions, shared memory sizes, async copy capabilities, and recommended kernel configurations.

4. **Quantization format comparison**: Table comparing Q4_0, Q4_K, GPTQ-128, AWQ byte layouts with dequant cost per element.

5. **Tiered kernel dispatch design**: Specification for TensorWarp's adaptive kernel selection system — how to route from (M, N, K, format, SM) to the optimal kernel variant.

6. **VRAM budget calculator**: Formula for computing optimal memory allocation across weights, KV cache, temp buffers, and activations for a given model size and GPU VRAM.

7. **Implementation roadmap**: Phased plan from current Q4_0 at 22% bandwidth → Marlin-style at 60%+ bandwidth, with estimated engineering effort per phase.

8. **Benchmark targets**: Specific tok/sec numbers to hit at each phase, with methodology for measuring bandwidth utilization.

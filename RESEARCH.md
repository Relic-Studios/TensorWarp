# GPU Inference Engine Research Report
## Building a Rust-Based Inference Compiler/Runtime to Outperform TensorRT
### Comprehensive Deep Research - April 2026

---

## 1. State of the Art in Inference Optimization (2024-2026)

### FlashAttention 3
- **Performance**: On H100, achieves 840 TFLOPS/s in BF16 (85% utilization) and 1.3 PFLOPS/s in FP8
- **Key Techniques**: Exploits Hopper's asynchronous Tensor Cores and TMA (Tensor Memory Accelerator) via warp-specialization; interleaves block-wise matmul and softmax operations; block quantization with incoherent processing for FP8
- **Accuracy**: FP8 FlashAttention-3 achieves 2.6x lower numerical error than baseline FP8 attention
- **Significance**: Outperforms cuDNN on sequences >= 1k tokens. Enables 2x longer context on same hardware
- **References**: [FlashAttention-3 Paper](https://arxiv.org/abs/2407.08608), [GTC 2025 Session](https://www.nvidia.com/en-us/on-demand/session/gtc25-S71368/)

### FlashDecoding & PagedAttention (vLLM)
- **PagedAttention**: Divides KV cache into non-contiguous blocks (pages), reducing memory waste from 60-80% to under 4%. Block table maps logical to physical blocks, analogous to OS virtual memory
- **FlashDecoding**: Applies Split-K optimization to decode attention kernels, parallelizing across KV length dimension
- **vAttention** (2025): Newer approach using CUDA virtual memory APIs for contiguous virtual memory while managing physical fragmentation — eliminates PagedAttention's indirection overhead
- **FlexAttention** (PyTorch 2.5+): Compile-time flexible attention supporting GQA + PagedAttention with FlashDecoding-level performance
- **References**: [vLLM PagedAttention Design](https://docs.vllm.ai/en/stable/design/paged_attention/), [FlashInfer Paper](https://arxiv.org/pdf/2501.01005), [FlexAttention Blog](https://pytorch.org/blog/flexattention-for-inference/)

### Persistent & Warp-Specialized Kernels
- **Persistent kernels**: Launch only as many CTAs as SMs, keep them resident throughout execution. Critical for Hopper where Tensor Core throughput is so high that deeper pipelines are needed
- **Warp specialization**: Different warps take on specialized roles (data movement vs compute) instead of classical homogeneous execution. CUTLASS ping-pong GEMM kernel is canonical example
- **Tawa** (2025): Automatic warp specialization compiler achieving up to 1.1x over cuBLAS GEMM and 1.2x over Triton for attention
- **Key insight**: On Hopper+, the asynchronous TMA + Tensor Core pipeline makes warp specialization essentially mandatory for peak performance
- **References**: [Tawa Paper](https://arxiv.org/abs/2510.14719), [CUTLASS Ping-Pong Blog](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/), [Warp Spec Performance Model](https://arxiv.org/abs/2506.11209)

### Speculative Decoding
- **State of art speedups**: 2-5x over autoregressive decoding depending on technique
  - Speculative Speculative Decoding (ICLR 2026): up to 2x over optimized speculative baselines, 5x over autoregressive
  - SpecBundle: up to 4x end-to-end over standard decoding
  - EAGLE3 at production scale: 1.4-2.0x at large batch sizes, 2-3x at moderate batches
  - Intel universal method: 2.8x without quality loss
- **Acceptance rates**: 0.6-0.8 typical with EAGLE3 drafts; code completion reaches 0.75-0.85
- **Production status**: Now built into vLLM, SGLang, TensorRT-LLM. No longer experimental
- **References**: [ICLR 2026 Paper](https://openreview.net/forum?id=aL1Wnml9Ef), [SpecBundle](https://www.lmsys.org/blog/2025-12-23-spec-bundle-phase-1/), [Meta at Scale](https://ai.meta.com/research/publications/efficient-speculative-decoding-for-llama-at-scale-challenges-and-solutions/)

### CUDA Graphs: Capabilities and Limitations
- **Core limitation**: Kernel parameters are hardcoded during capture; same values replayed. This conflicts with dynamic shapes
- **Workarounds**: 
  - Padding to fixed sizes + bucketing (multiple graphs for different shape ranges)
  - PyGraph (2025): Inserts prelude kernel that updates vendor kernel parameters on-the-fly during replay
  - vLLM: Fine-grained CUDA Graphs aware of prefill vs decode batches, runtime dispatch between full and piecewise graphs
  - TensorRT-RTX: Single-line API for intelligent capture with dynamic shapes
- **Best practice**: For autoregressive decode (fixed seq_len=1), CUDA graphs work well. For prefill with variable lengths, use bucketing or fall back to eager
- **References**: [PyGraph Paper](https://arxiv.org/html/2503.19779v2), [CUDA Graph in torch.compile](https://fkong.tech/posts/2025-12-23-cuda-graph-in-torch-compile/), [vLLM CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/)

### Kernel Fusion Strategies Beyond TRT
- **FlashFuser** (2025): Exploits Hopper's Distributed Shared Memory (DSM) for inter-SM fusion. Achieves 3.3x over TensorRT, 3.9x over PyTorch for convolution chains
- **Deep Kernel Fusion for Transformers**: Fuses entire FFN blocks (GEMMs + pointwise) into single DeepFusionKernel
- **Fused Triton Kernels**: Paged-attention achieving >90% peak TFLOPS, quantized dequant+GEMM with 64-124% average speedup on H100
- **FusionStitching**: Approximate dynamic programming + beam search for optimal fusion patterns, generates CUDA/LLVM IR
- **Key insight**: TRT's fusion is conservative and pattern-matching based. The frontier is DAG-based analysis + cost models that can discover novel fusions automatically
- **References**: [FlashFuser](https://arxiv.org/html/2512.12949v1), [Deep Kernel Fusion](https://arxiv.org/html/2602.11808), [Triton Kernels for LLM Inference](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/)

### FP8/FP4/INT4 Quantization — State of Art
- **FP8**: Mature. Preserves 99-100% of full-precision quality even at 405B+ scale. More robust to outliers than INT8. Supported on Hopper (H100) and all major frameworks
- **FP4 (NVFP4)**: Blackwell-native. Micro-tensor scaling for fine-grained calibration. MR-MXFP4 achieves ~15,000 tok/sec: 2.2x over BF16, 1.3x over FP8. No native INT4 on Blackwell — only INT8, FP8, FP4
- **MXFP4 (OCP format)**: Open standard. MR-GPTQ bridges gap, nearly matching NVFP4 quality
- **SGLang + NVFP4**: Up to 4x throughput improvement over Hopper for DeepSeek-R1 with MoE kernels
- **Key architectural decision**: Target FP8 as primary quantization, with FP4 path for Blackwell+. INT4 is legacy — FP4 preserves more information at same bit width
- **References**: [ICLR 2026 MXFP4 Paper](https://openreview.net/pdf?id=zCBGe9AqJZ), [FP4 on Blackwell](https://www.spheron.network/blog/fp4-quantization-blackwell-gpu-cost/), [TensorRT Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)

---

## 2. Compiler Approaches

### TVM Relax/TIR
- **Architecture**: Two-level IR — TensorIR (tensor-level) + Relax (graph-level). Python-first transformations
- **Dynamic shapes**: Relax uses symbolic variables for shape dimensions with symbolic deduction across operators, subgraph calls, and foreign function calls. Major improvement over Relay
- **Fusion**: Automatic pattern detection rather than human-defined fusion rules
- **Performance**: Reduces decode latency by up to 27% on some models
- **Limitations**: TensorRT still consistently faster for raw inference on NVIDIA hardware. TVM's strength is flexibility and cross-platform
- **What to adopt**: The symbolic shape tracking approach is excellent. The two-level IR (graph + tensor) is a proven pattern worth emulating
- **References**: [Relax Paper](https://arxiv.org/pdf/2311.02103), [TVM Architecture](https://tvm.apache.org/docs/arch/index.html)

### Triton (OpenAI) Compilation Pipeline
- **Pipeline**: Python AST → Triton IR (TTIR, MLIR-based SSA) → TritonGPU IR (TTGIR, hardware-specific) → LLVM IR → PTX → cubin via ptxas
- **Key innovation**: Block-level programming model. Automatic shared memory allocation via liveness analysis of block operands to `tl.dot`
- **CUDA Tile IR backend** (2025): NVIDIA contributed backend that maps Triton to CUDA Tile IR, preserving tile semantics. Alternative to PTX path for newer hardware
- **Warp specialization in Triton**: Emerging support for warp-specialized kernels via compiler passes
- **What to learn**: The progressive MLIR dialect lowering (TTIR → TTGIR → LLVM IR) is the right architecture. But Triton is Python-only and JIT — a compiled Rust approach can eliminate JIT overhead and enable AOT optimization
- **References**: [Triton Compilation Stages](https://pytorch.org/blog/triton-kernel-compilation-stages/), [Triton Internals Deep Dive](http://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/), [CUDA Tile IR Backend](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton)

### MLIR
- **Performance results**: MLIR-generated GEMM code achieves 95-119% of cuBLAS FP32 and 80-160% of cuBLAS FP16 on Ampere
- **Key advantage**: Drastically reduces entry cost for defining new abstraction levels. Composable, modular dialect system
- **mlirSynth**: Automatically raises low-level IR to high-level dialects, enabling up to 21.6x speedup on TPUs
- **Industry adoption**: Now backbone of OpenXLA, Triton, parts of NVIDIA CUDA ecosystem
- **Verdict for our project**: MLIR is worth building on IF you need the full multi-dialect lowering stack and don't mind the C++ dependency. For a Rust-native project, the better approach is to learn from MLIR's design (progressive lowering, composable passes, dialect boundaries) but implement a simpler, Rust-native IR
- **Risk**: MLIR is heavyweight (LLVM dependency), C++ ecosystem, long compile times. Not easily embeddable in a Rust project
- **References**: [MLIR Linalg Rationale](https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/), [Composable Codegen Paper](https://arxiv.org/pdf/2202.03293)

### XLA's HLO
- **Design**: Carefully curated, mostly orthogonal operation set. Multi-stage: target-independent optimization → target-dependent optimization → LLVM IR → machine code
- **What to learn**: 
  - Fixed operation set forces canonical representations, making pattern matching reliable
  - Target-independent passes first, then target-specific — clean separation
  - DHLO extends HLO with dynamic shapes by replacing compile-time constants with runtime tensor dataflow
- **Limitation**: LLVM has design mistakes preventing multithreaded compilation. MLIR fixes this with limited SSA scope
- **References**: [XLA Architecture](https://openxla.org/xla/architecture), [Operator Fusion in XLA](https://arxiv.org/pdf/2301.13062)

### torch.compile / Inductor
- **Fusion discovery**: Pattern matcher utility (`torch._inductor.pattern_matcher`), automatic op fusion, custom pass framework (e.g., `ActivationFusionPass`)
- **Hardware-aware**: Bases decisions on observed tensor shapes/dtypes, microarchitecture-aware tuning
- **Graph capture**: Modules inlined into graph, enabling pattern matching across module boundaries
- **Limitation**: Dynamic shapes + distributed support still developing. CUDA graph integration improving but complex
- **What to learn**: The pattern-matching-based fusion discovery with hardware-specific cost models is practical and effective
- **References**: [torch.compile with vLLM](https://blog.vllm.ai/2025/08/20/torch-compile.html), [State of torch.compile Aug 2025](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)

### Mojo/MAX (Modular)
- **Performance**: Blackwell conv2d at 130.7 TFLOPS matching CUTLASS in 770 lines of Mojo vs ~3k CUTLASS. FLUX.2 models 4x speedup. 50% faster custom Mamba kernels
- **Architecture**: PyTorch-like eager mode + `model.compile()` for production. Built-in autotuning. Direct hardware access without black-box library
- **Multi-backend**: Works across NVIDIA and AMD hardware, expanding Apple Silicon GPU support
- **Status**: MAX Python API stabilized. Mojo 1.0 planned for later 2026
- **What to learn**: Their "no black-box library" philosophy — compile from high-level down to hardware-specific code. Built-in autotuning is essential
- **Risk assessment**: Mojo is a proprietary language. Useful as competitive benchmark but not to build on
- **References**: [Modular GTC 2026](https://www.modular.com/blog/modular-at-nvidia-gtc-2026-max-on-blackwell-mojo-kernel-porting-and-deepseek-v3-on-b200), [MAX Inference](https://www.modular.com/max)

### New Compiler Frameworks (2025-2026)
- **SySTeC** (MIT): Exploits symmetry and sparsity for up to 30x computation reduction
- **Scorch**: Sparse tensor compiler for PyTorch, 1.05-5.80x over PyTorch Sparse on GNNs and sparse transformers
- **Hexagon-MLIR** (Qualcomm): MLIR-based stack for Hexagon NPU, compiles Triton & PyTorch
- **ML-Triton**: Hierarchical layers (workgroup, warp, instruction) aligning codegen with physical GPU architecture. >95% geometric mean of expert kernel performance
- **Triton-distributed**: Multi-node/GPU kernel automation with communication-compute autotuning
- **References**: [C4ML 2026](https://c4ml.org/), [awesome-tensor-compilers](https://github.com/merrymercy/awesome-tensor-compilers)

---

## 3. IR Design

### What a Good Tensor Computation IR Looks Like
A good IR for GPU tensor computation needs:
1. **Immutable tensor values with SSA semantics**: Enables classical compiler optimizations (CSE, DCE, LICM) to apply seamlessly to tensor operations
2. **Multi-level representation**: Graph level (operator DAG) + Tensor level (loop nests, tiling) + Hardware level (registers, shared memory, warp assignments)
3. **Explicit data movement**: Not just compute — the IR must represent memory transfers, shared memory staging, and async copies as first-class operations
4. **Fusion boundaries as first-class concepts**: The IR needs to express which operations can/should be fused

### SSA vs Sea-of-Nodes vs Dataflow Graph
- **SSA (used by MLIR, Triton, LLVM)**: Standard choice. Tensor values are immutable, def-use chains enable optimization. Challenge: polyhedral abstractions don't compose well with SSA
- **Dataflow graph (used by TVM, XLA, torch.fx)**: Natural for representing tensor computation DAGs. Fusion = graph partitioning. Less suitable for within-kernel optimization
- **Sea-of-nodes**: Not commonly used in tensor compilers. Overhead of the representation may not pay off for the relatively structured nature of tensor computation
- **Recommendation**: Use a dataflow graph at the graph level (operator fusion, scheduling) and SSA at the kernel level (register allocation, instruction scheduling). This is the Triton/TVM two-level pattern

### Representing Fusion Boundaries
- **Explicit fusion groups**: XLA groups ops into "fusion computations" — subgraphs that become single kernels
- **FuseFlow's fusion tables**: Name and memoize intermediate streams, allowing reference to subgraphs before materialization
- **Op boundary breaking**: Compound ops create boundaries that constrain fusion exploration. Breaking them up exposes more fusion opportunities
- **Cost-model-driven partitioning**: Interval-covering optimization selecting non-overlapping subgraphs minimizing estimated cost, solved via DP on topological order
- **Recommendation**: Model fusion as graph partitioning with cost model. Allow IR to represent "tentative" fusion groups that the optimizer can merge/split

### Dynamic vs Static Shapes
- **Static shapes**: Complete information → exact memory planning, fixed offsets, maximum optimization. TensorRT historically static-only
- **Dynamic shapes via symbolic dimensions**: Relax uses SymPy-style symbolic variables propagated through operators. torch.compile uses FX IR + SymPy expressions
- **DHLO approach**: Replace compile-time constant attributes with runtime tensor arguments
- **JIT specialization**: Compile specialized kernels for encountered shapes, cache them. New shapes trigger recompilation
- **Recommendation**: Primary path is symbolic shapes with JIT specialization. Represent shape variables in the IR as first-class symbols. Generate specialized code for common shape buckets at compile time, with JIT fallback for novel shapes

### Memory Layout Annotations
- **Standard formats**: NCHW, NHWC, plus tiled variants like nChw16c (channel blocked by 16 for AVX-512)
- **Stride-based representation**: PyTorch uses strides per dimension, mapping to formats via stride patterns. Flexible but implicit
- **Blocked/tiled layouts**: Split dimensions into blocks of fixed size. Critical for Tensor Core alignment requirements (e.g., 16x16 tiles for fp16 mma)
- **Layout propagation pass**: Assign preferred layouts to operators, propagate through layout-agnostic nodes, minimize transposes at boundaries
- **Recommendation**: Represent layouts explicitly in the IR as a layout descriptor (dimension order + tile factors + alignment). Implement layout propagation as a pass. NHWC should be default for GPU (Tensor Core preference), with automatic tiling for GEMM operands

---

## 4. Kernel Generation / Codegen

### Emitting PTX from High-Level Description
- **LLVM NVPTX backend**: The standard path. Emit LLVM IR → NVPTX backend → PTX → ptxas → cubin. Used by Triton, TVM, CUDA
- **Direct PTX emission**: DeepSeek showed 10x efficiency gains by hand-writing PTX for communication kernels on H800. Gives control over register usage, memory layout, pipeline scheduling
- **NVVM IR**: LLVM IR dialect designed for GPU compute kernels. NVVM compiler (LLVM-based) generates PTX from NVVM IR
- **Recommendation**: Use LLVM NVPTX backend as primary path (proven, maintained), with escape hatch for hand-tuned PTX for critical kernels (attention, GEMM). Don't try to emit PTX directly — the complexity isn't worth it when LLVM handles it well
- **References**: [LLVM NVPTX Docs](https://llvm.org/docs/NVPTXUsage.html), [DeepSeek PTX Innovation](https://medium.com/@noahbean3396/the-future-of-ai-compute-deepseeks-ptx-innovation-and-what-it-means-for-nvidia-f501b7a0f58e)

### Triton's Lowering Pipeline (Details)
1. **Python AST → Triton IR (TTIR)**: SSA construction from Python. CSE, broadcast management
2. **TTIR → TritonGPU IR (TTGIR)**: Backend-specific passes. NVIDIA TTGIR configures for Nvidia targets. Adds encoding attributes for data layout on GPU (blocked, MMA, shared)
3. **TTGIR → LLVM IR**: Lower GPU-specific constructs to LLVM intrinsics. Insert shared memory operations, barrier synchronization
4. **LLVM IR → PTX → cubin**: Standard LLVM backend + ptxas JIT

### Autotuning Strategies
- **tritonBLAS approach**: Cost models incorporating hardware-calibrated instruction latencies, bandwidths, cache behaviors. Search over hundreds (not millions) of configurations with negligible JIT overhead
- **Search dimensions**: Block sizes (BLOCK_M x BLOCK_N x BLOCK_K), warp/thread groupings, vector widths, memory layouts, shared memory staging, number of pipeline stages
- **LLM-specific**: Tile sizes on KV side constrained by page size of KV cache. Q-block sizes determine how many query tokens processed per kernel instance
- **ML-guided autotuning**: PEAK (2025) uses LLMs to suggest kernel configurations, achieving >95% of expert performance
- **Key principle**: Tile size = balance of enough threads for warp-group MMA, small enough to avoid padding waste, enough SMEM for double-buffering, grid fills whole waves or uses persistent kernel
- **Recommendation**: Implement cost-model-based search as primary strategy, with empirical autotuning as validation/fallback. Pre-tune for common shapes (powers of 2, transformer dimensions). Cache tuning results per GPU architecture
- **References**: [Autotuning with CUTLASS 4.2](https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/), [Triton Attention Anatomy](https://www.arxiv.org/pdf/2511.11581)

### Targeting Metal Compute Shaders from Same IR
- **Metal IR = LLVM Bitcode**: Metal shaders compile to LLVM Bitcode in .metallib files. This is significant — same LLVM foundation as CUDA's path
- **Metal Shading Language (MSL)**: C++14-based, implemented using Clang/LLVM
- **Strategy**: From our high-level IR, lower to LLVM IR (same as CUDA path), then use different backends: NVPTX for CUDA, Metal Bitcode for Apple
- **Key differences from CUDA**: No shared memory → use threadgroup memory. No warp shuffle → use SIMD group operations. Different memory model (unified memory on Apple Silicon). Max threadgroup sizes differ
- **Recommendation**: Share IR and optimization passes down to LLVM IR level. Fork at backend: NVPTX vs Metal AIR (Apple IR). Write a Metal-specific lowering pass handling threadgroup memory, SIMD groups, buffer bindings
- **References**: [Metal Shader Converter](https://developer.apple.com/metal/shader-converter/)

### SPIR-V as Portable Intermediate
- **2025 status**: SPIR-V backend for LLVM is now official goal. Microsoft adopting SPIR-V for Direct3D Shader Model 7+. Zig SPIR-V backend maturing
- **Capability**: SPIR-V can express 99% of PTX functionality. PTX can only do ~50% of SPIR-V functionality
- **Viable for**: Vulkan compute (cross-platform NVIDIA/AMD/Intel), WebGPU (via WGSL compilation), Intel GPUs via OpenCL
- **Not viable for**: Peak performance on NVIDIA (PTX gives more control) or Apple (Metal is native)
- **Recommendation**: Use SPIR-V as tertiary target for broad hardware support (AMD, Intel, future GPUs). Not as primary path for NVIDIA or Apple. Useful for the "works everywhere" tier vs "maximum performance" tier

### Persistent Kernel Techniques for Autoregressive Decoding
- **Problem**: Each decode step is a tiny kernel launch (batch=1-64, seq=1). Kernel launch overhead dominates
- **Solution**: Launch persistent kernel that stays resident, processes multiple decode steps without returning to CPU
- **SGLang's approach**: Zero-overhead overlap scheduler — while GPU runs current verify step, CPU prepares next draft/extend step
- **CUDA Graphs alternative**: Capture decode step as graph, replay with minimal launch overhead. Works well for fixed decode shapes
- **Recommendation**: Implement both persistent decode kernel (for single-stream latency optimization) and CUDA graph capture (for throughput batched scenarios)

---

## 5. Runtime Design

### Memory Pool Strategies
- **CUDA Stream-Ordered Allocator**: Built-in since CUDA 11.2. Auto-defragments via virtual address remapping. Use as baseline
- **BFC (Best-Fit with Coalescing)**: TensorFlow's approach. Slab-based for common sizes, buddy system for large allocations. ~80% lower fragmentation in recent implementations
- **Arena allocation**: Pre-allocate large pool, bump-allocate tensors with known lifetimes. Ideal for inference with static graphs where all tensor sizes known at compile time
- **SlabAlloc**: Warp-synchronous protocol using per-warp register-local bitmaps. For dynamic within-kernel allocation
- **Recommendation for inference**: 
  1. Static memory planning at compile time (compute tensor lifetimes, assign fixed offsets in pre-allocated arena)
  2. Fall back to slab allocator for dynamic shapes
  3. Use CUDA stream-ordered allocator for KV cache growth
  4. Implement memory pool per stream to avoid synchronization overhead

### CUDA Graph Capture and Replay
- **Best practices**: 
  - Use non-default streams for capture
  - Pin host memory for any H2D/D2H transfers within graph
  - Bucket inputs by shape, capture one graph per bucket
  - PyGraph technique: prelude kernel that patches parameters at replay time
  - Avoid conditional logic within captured graphs
- **Pitfalls**: 
  - Graphs capture ALL operations including allocations — use pool allocators
  - Can't mix eager ops and graph replay
  - Memory overhead: captured graphs hold references to all intermediate buffers
  - Debugging is painful — errors manifest as incorrect results, not crashes
- **References**: [NVIDIA CUDA Graph Best Practices](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html)

### Multi-Stream Execution and Pipelining
- **Core technique**: Overlap compute (model execution) with memory operations (KV cache management, input preprocessing) on separate streams
- **Best practices**: 
  - Use pinned host memory for async H2D transfers
  - Limit to 2-4 concurrent streams (more causes contention)
  - For MoE models: parallelize expert execution across streams (active research in PyTorch)
  - At large batch sizes, benefit diminishes — GPU is already saturated
- **Caution**: TensorRT's kernel selection assumes exclusive GPU access — concurrent streams can cause suboptimal kernel choices

### KV-Cache Management (PagedAttention-Style)
- **Block structure**: Each block stores data for BLOCK_SIZE tokens (typically 16) at one head. Block size 16, head size 128 → 2048 elements per block
- **Block table**: Maps virtual (logical) block indices to physical GPU memory blocks. Lightweight indirection layer
- **Memory savings**: 60-80% waste → under 4% waste
- **vAttention (2025 evolution)**: Uses CUDA virtual memory APIs for contiguous virtual layout without physical contiguity. Eliminates custom attention kernel requirement — standard FlashAttention kernels work directly
- **Recommendation**: Implement PagedAttention-style block management. Consider vAttention approach for simpler kernel compatibility. Block size should be tunable (16 is common default)
- **References**: [vLLM PagedAttention](https://docs.vllm.ai/en/stable/design/paged_attention/), [vAttention Issue](https://github.com/vllm-project/vllm/issues/17612)

### Scheduling: vLLM, TGI, SGLang
- **vLLM**: Distributed scheduler with dynamic GPU routing, preemption support, PagedAttention memory management. Continuous batching
- **SGLang**: Emphasis on aggressive concurrency, context management, multi-GPU scaling. Optimized prompt/generation pipeline structure. Often outperforms vLLM in TTFT benchmarks as of 2025
- **TGI**: Rust HTTP router + Python/gRPC model server. Entered maintenance mode Dec 2025 — Hugging Face now recommends vLLM or SGLang
- **Key insight**: "KV cache is the real bottleneck resource" — all engines converge on this. Scheduling must be KV-cache-aware
- **Recommendation**: Build scheduler that treats KV cache as primary resource to optimize. Implement continuous batching with preemption. Study SGLang's architecture — it's the newest and most performance-focused
- **References**: [LLM Inference Servers Compared](https://blog.premai.io/llm-inference-servers-compared-vllm-vs-tgi-vs-sglang-vs-triton-2026/)

---

## 6. Specific Speed Wins Over TRT

### Where People Have Beat TRT
- **FlashFuser**: 3.3x over TensorRT on convolution chains via DSM-based inter-SM fusion
- **vLLM**: Better TTFT (50-80ms) at high concurrency (100+ users) vs TRT degrading under high load
- **DeepSeek**: 10x training efficiency via hand-tuned PTX communication kernels
- **FlashAttention-3**: Outperforms cuDNN (which TRT uses) for attention on sequences >= 1k tokens
- **Fused Triton kernels**: Quantized dequant+GEMM 64-124% average speedup over unfused, with peaks of 295% on H100

### TRT Architectural Limitations
1. **ONNX parser reliability**: Valid ONNX models produce bad outputs or fail to parse. Forces manual graph surgery with ONNX-GraphSurgeon
2. **Hardware coupling**: Serialized engines don't work across GPU architectures. Must rebuild per GPU
3. **Kernel selection under concurrent load**: Optimizes assuming exclusive GPU access. Suboptimal when sharing GPU with other workloads
4. **Conservative fusion**: Pattern-matching based. Misses novel fusion opportunities that cost-model approaches discover
5. **Static shape legacy**: Dynamic shape support added later and is less optimized than static path
6. **Closed-source kernels**: Can't inspect, modify, or optimize individual kernels. Black box
7. **C++ plugin API**: Custom operations require C++ plugins — high friction for development iteration

### FlashInfer
- **What it is**: Kernel library + JIT generator for LLM serving. MLSys 2025 Best Paper
- **Architecture**: CUDA/CUTLASS template system + Jinja-based JIT compilation. Inspector-executor API that first analyzes request shapes, then dispatches tuned kernels
- **JIT compilation**: All kernels for Llama JIT-compile within 15 seconds via split compilation and minimized header dependencies
- **Key innovation**: Composable attention variants via functors (LogitsTransform, QueryTransform). Users define custom attention without writing CUDA
- **Performance**: State-of-art attention and MoE kernels integrated into vLLM, SGLang, and MLC-Engine
- **References**: [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer), [NVIDIA Blog](https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/)

### ThunderKittens
- **What it is**: Tile-primitive GPU kernel framework from Stanford Hazy Research. Used by Together AI, Jump Trading, Cursor in production
- **Performance**: BF16/FP8 GEMMs at or near cuBLAS speed, up to 2x faster than cuBLAS on H100. Attention near cuDNN speed on B200
- **ThunderKittens 2.0** (Jan 2026): Full Blackwell support, MXFP8 and NVFP4 precision
- **Multi-GPU**: ParallelKittens achieves up to 2.6x over NCCL in under 100 lines of code
- **Cross-platform**: ThunderMittens (Apple Silicon), HipKittens (AMD)
- **Megakernel interpreter**: Plugs arbitrary simple kernels together into megakernels with automatic pipelining and async fetching
- **Key insight for our project**: ThunderKittens proves that tile-level abstractions can match hand-tuned libraries. Their interpreter template is a novel approach to fusion
- **References**: [ThunderKittens GitHub](https://github.com/HazyResearch/ThunderKittens), [TK 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2), [ParallelKittens](https://hazyresearch.stanford.edu/blog/2025-11-17-pk)

### Liger Kernel
- **What it is**: Efficient Triton kernels for LLM training/inference from LinkedIn
- **Performance**: 20% training throughput increase, 60% GPU memory reduction vs HuggingFace implementations
- **Batch-size sensitivity**: torch.compile handles layernorms better at small batches; Liger better at high batches. Kernel selection should be adaptive
- **References**: [Liger Kernel GitHub](https://github.com/linkedin/Liger-Kernel)

---

## 7. Cross-Platform Targeting

### CUDA PTX vs Metal IR vs SPIR-V
| Feature | CUDA PTX | Metal IR | SPIR-V |
|---------|----------|----------|--------|
| Foundation | NVIDIA proprietary | LLVM Bitcode | Khronos standard |
| Shared memory | Yes (explicit) | Threadgroup memory | Workgroup memory |
| Warp/SIMD | 32 threads (warp) | 32 threads (SIMD group) | Implementation-defined |
| Tensor Cores | Full access via mma.sync | Via MPS, limited direct access | No direct TC access |
| JIT | ptxas at load time | Metal compiler at load time | Vendor driver at load time |
| Async copy | TMA on Hopper+ | Limited | Not standard |
| Best for | NVIDIA peak perf | Apple Silicon | Broad compatibility |

### "Compile Once, Target Multiple GPUs"
- **hetGPU** (2025): Unified GPU IR capturing parallelism + memory semantics architecture-neutrally. JIT to native at load time. Research-stage
- **Portability layers**: alpaka, Kokkos, RAJA — single-source C++ targeting NVIDIA/AMD/Intel. Not ML-specific
- **Practical approach**: Share IR and optimization passes through a common mid-level IR. Fork at code generation:
  - NVIDIA: LLVM NVPTX → PTX → cubin
  - Apple: LLVM → Metal AIR → metallib
  - Broad: LLVM → SPIR-V → Vulkan compute
- **ThunderKittens model**: Same tile-level abstraction, different backends (TK for NVIDIA, ThunderMittens for Apple, HipKittens for AMD). Proves the concept works in practice
- **References**: [hetGPU Paper](https://arxiv.org/pdf/2506.15993), [One Kernel for All GPUs](https://hazyresearch.stanford.edu/blog/2025-09-22-pgl)

### tinygrad Multi-Backend Architecture
- **Philosophy**: Only 12 primitive ops (add, mul, reduce, reshape, permute, pad, etc.). All complex operations compose from these
- **Lazy evaluation**: Nothing materialized until `.realize()`. Compiler has full visibility for fusion
- **Backend as weekend project**: If you can schedule 12 basic ops on an accelerator, you can run LLaMA on it
- **Backends**: CPU, OpenCL, Metal, CUDA, AMD, Qualcomm, WebGPU, Mesa NIR (NVK/NAK)
- **Key lesson**: Minimal primitive set + aggressive lazy fusion = easy multi-backend. But performance ceiling is lower than hand-tuned kernels
- **What to adopt**: The "minimal primitive set" philosophy for multi-backend support. But supplement with hand-tuned kernel templates for critical paths (attention, GEMM)
- **References**: [tinygrad Docs](https://docs.tinygrad.org/), [tinygrad Backends](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/backends.md)

### Burn (Rust ML Framework) Backend Architecture
- **Design**: Generic `Backend` trait — write model once, run on any backend without code changes. Composable backends augmented with autodiff and kernel fusion
- **Backends**: LibTorch, Candle, CUDA, Metal, WGPU/SPIR-V. All with Tensor Core support
- **Memory management**: Infrastructure for pluggable memory strategies per backend
- **Thread safety**: Leverages Rust ownership system for tensor tracking
- **Key insight**: Burn proves the Rust `Backend` trait pattern works for multi-backend ML. We should study and potentially build on or parallel their design
- **Limitation**: Not focused on inference compilation/optimization. More framework than compiler
- **References**: [Burn GitHub](https://github.com/tracel-ai/burn), [Burn Docs](https://burn.dev/docs/burn/)

---

## 8. Rust Ecosystem

### cudarc (Rust CUDA Bindings)
- **Capabilities**: 
  - Safe wrappers around CUDA driver API + NVRTC
  - `CudaSlice<T>` for device memory (like `Vec<T>` on GPU)
  - NVRTC compilation: `compile_ptx()` → `Ptx` → load into device
  - cublas, cublaslt, curand, nccl wrappers
  - Dynamic loading (default), dynamic linking, or static linking options
- **Limitations**:
  - Not all CUDA features covered
  - Advanced memory allocation strategies limited
  - Ecosystem still in flux
- **Status**: Active development, consolidation efforts with rust-cuda underway
- **Verdict**: Solid foundation for our CUDA backend. Use cudarc for driver API + NVRTC, extend as needed for custom allocators and advanced features
- **References**: [cudarc Docs](https://docs.rs/cudarc), [Rust CUDA Update May 2025](https://rust-gpu.github.io/blog/2025/05/27/rust-cuda-update/)

### metal-rs
- **Status**: Deprecated name, now just `metal` crate. Active with recent updates Dec 2025
- **Capabilities**: Unsafe Rust bindings for Metal 3D Graphics API including compute
- **Ecosystem direction**: Moving toward wgpu for cross-platform compute, with metal-rs for Apple-specific features
- **References**: [metal crate](https://docs.rs/metal/latest/metal/), [LambdaClass Metal+Rust FFT](https://blog.lambdaclass.com/using-metal-and-rust-to-make-fft-even-faster/)

### ash (Vulkan from Rust)
- **Capabilities**: Direct Vulkan bindings with SPIR-V shader support
- **Rust GPU project**: Compiles Rust code to SPIR-V. Ready for production use as of 2025
- **Limitation**: Developer experience rough — specific nightly Rust versions required, spirv_builder hides complexity but still complex
- **Viable for compute**: Yes, but Vulkan compute overhead is higher than direct CUDA/Metal. Best for portability tier, not peak performance tier
- **References**: [Ash on Best of Web](https://best-of-web.builder.io/library/ash-rs/ash), [Rust on Every GPU](https://rust-gpu.github.io/blog/2025/07/25/rust-on-every-gpu/)

### wgpu (WebGPU from Rust)
- **What it is**: Cross-platform graphics/compute library on Vulkan, Metal, DX12, WebGL2, WebGPU
- **For ML inference**: Viable for lightweight models. Burn uses wgpu backend. Workgroup size 64 recommended for 1D compute
- **Limitation**: Performance ceiling lower than native APIs. No Tensor Core access. Shader language (WGSL) less expressive than CUDA
- **Best use**: Broadest compatibility tier. Web deployment. Not for peak performance
- **References**: [wgpu.rs](https://wgpu.rs/), [Burn wgpu Backend](https://burn.dev/)

### PyO3 for Python Bindings
- **Best practices**:
  - Use `maturin` for building/packaging
  - Minimize Python/Rust type conversion — keep tensors in Rust as long as possible
  - Use `Bound<'py, T>` smart pointers for GIL-bound Python objects
  - `pyo3-tch` crate for PyTorch tensor interop
  - Release GIL (`Python::allow_threads`) during long computations
- **Key concern**: Type conversion overhead is the bottleneck. Design API to accept and return opaque tensor handles, not copying data
- **References**: [PyO3 Docs](https://docs.rs/pyo3), [PyO3 Design Discussion](https://github.com/PyO3/pyo3/discussions/4780)

### Existing Rust Inference Engines/Compilers
1. **Infire** (Cloudflare): Rust LLM inference engine, 7% faster than vLLM 0.10.0 on H100 NVL. Maximizes memory, network I/O, and GPU utilization
2. **Crane**: Pure Rust LLM/VLM/TTS engine using Candle. Compiles to binary like llama.cpp but cleaner
3. **Candle** (Hugging Face): Rust-native ML framework for inference. Designed for serverless/edge — low latency, small binaries
4. **ort**: Rust bindings for ONNX Runtime. Used by Bloop (code search), SurrealDB, Google Magika
5. **Burn**: Framework with multi-backend support (covered above)
6. **tract**: Fast inference for ONNX and TensorFlow models
- **References**: [Infire/Cloudflare Blog](https://blog.cloudflare.com/cloudflares-most-efficient-ai-inference-engine/), [Crane GitHub](https://github.com/lucasjinreal/Crane), [ort Docs](https://ort.pyke.io/)

---

## Architectural Recommendations

### What to Build vs Adopt

**Build from scratch:**
- Custom IR with two levels: graph-level (fusion, scheduling) + tensor-level (tiling, vectorization, memory staging)
- Rust-native compilation pipeline (no MLIR/C++ dependency)
- Memory manager with static planning + slab fallback
- KV cache manager (PagedAttention-style or vAttention-style)
- Kernel fusion optimizer with cost-model-based partitioning
- Autotuning framework integrated into compilation

**Adopt/wrap:**
- LLVM backend for PTX generation (via llvm-sys or inkwell crate)
- cudarc for CUDA runtime/driver interaction
- metal-rs for Metal backend
- FlashAttention/FlashInfer kernels as precompiled libraries initially (replace with custom-generated later)
- PyO3 + maturin for Python bindings

**Learn from but don't depend on:**
- Triton's progressive lowering architecture (TTIR → TTGIR → LLVM IR)
- TVM Relax's symbolic shape tracking
- XLA's fixed-op-set approach for canonical representations
- ThunderKittens' tile-level abstraction and megakernel interpreter
- tinygrad's minimal primitive set for multi-backend support

### Critical Risks and Gotchas

1. **LLVM dependency management**: LLVM versions, build times, cross-platform builds are painful. Consider using cranelift for faster iteration, LLVM for production
2. **Hopper/Blackwell specialization**: Peak performance requires warp specialization, TMA, async copy — these are NVIDIA-specific and change each generation. Budget for per-generation tuning
3. **Quantization kernel correctness**: FP8/FP4 have subtle numerical accuracy issues. FlashAttention-3's incoherent processing for FP8 is non-trivial to reproduce
4. **CUDA version compatibility**: Different CUDA toolkit versions have different PTX ISA support. Must manage compatibility matrix
5. **Metal limitations**: No equivalent to warp-level primitives until M3/M4 SIMD groups. Limited Tensor Core access vs CUDA. Much smaller community for compute workloads
6. **Python ecosystem integration**: Must interop with PyTorch tensors (zero-copy if possible). Users expect `model.compile()` style API
7. **Autotuning time**: Kernel autotuning can take hours for large models. Need persistent cache, good defaults, and incremental tuning
8. **Benchmark fairness**: TRT has years of hand-tuning for common models. Beating it on Llama/GPT requires matching their kernel quality, not just compiler quality

### Suggested Development Phases

**Phase 1**: Core IR + CUDA GEMM/Attention codegen
- Build two-level IR in Rust
- Use cudarc for CUDA interaction, LLVM for PTX gen
- Target FlashAttention-equivalent attention kernel
- Benchmark against cuBLAS for GEMM, FlashAttention for attention

**Phase 2**: Graph optimization + fusion
- Implement cost-model-based fusion optimizer
- Add symbolic shape support
- Build CUDA Graph capture/replay
- Autotuning framework
- Target: match TRT on single-layer benchmarks

**Phase 3**: Full model compilation + runtime
- KV cache management (PagedAttention)
- Continuous batching scheduler
- Speculative decoding support
- Full Llama/Mistral model compilation
- Target: match or beat TRT-LLM on end-to-end serving benchmarks

**Phase 4**: Multi-backend + production
- Metal backend via LLVM
- SPIR-V/Vulkan for broad compatibility
- FP8/FP4 quantization support
- Python bindings (PyO3)
- Distributed inference (tensor parallelism)

---

## Key Performance Targets (What "Beating TRT" Means)

| Metric | TRT-LLM Baseline | Target |
|--------|-------------------|--------|
| GEMM throughput (H100 FP16) | ~85% peak TFLOPS | >90% peak TFLOPS |
| Attention (FA3 equivalent) | 840 TFLOPS/s BF16 | >=840 TFLOPS/s |
| TTFT (Llama-70B, batch=1) | 35-50ms | <35ms |
| Decode throughput (batch=128) | Baseline | >1.1x via better fusion |
| KV cache memory efficiency | ~90% | >96% (PagedAttention) |
| FP8 inference quality | 99% of FP16 | 99%+ (match FA3 FP8 accuracy) |
| Compilation time | Minutes (TRT builder) | Seconds (AOT compilation) |
| Dynamic shape overhead | 5-15% vs static | <5% via symbolic + JIT |

---

## Sources

### FlashAttention & Attention Optimization
- [FlashAttention-3 Paper](https://arxiv.org/abs/2407.08608)
- [FlashInfer Paper](https://arxiv.org/pdf/2501.01005)
- [FlexAttention for Inference](https://pytorch.org/blog/flexattention-for-inference/)
- [vLLM PagedAttention](https://docs.vllm.ai/en/stable/design/paged_attention/)

### Warp Specialization & Persistent Kernels
- [Tawa: Automatic Warp Specialization](https://arxiv.org/abs/2510.14719)
- [CUTLASS Ping-Pong GEMM](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)
- [Warp Spec Performance Model](https://arxiv.org/abs/2506.11209)
- [Optimal Software Pipelining](https://arxiv.org/abs/2512.18134)

### Speculative Decoding
- [Speculative Speculative Decoding (ICLR 2026)](https://openreview.net/forum?id=aL1Wnml9Ef)
- [SpecBundle & SpecForge](https://www.lmsys.org/blog/2025-12-23-spec-bundle-phase-1/)
- [Meta EAGLE at Scale](https://ai.meta.com/research/publications/efficient-speculative-decoding-for-llama-at-scale-challenges-and-solutions/)

### CUDA Graphs
- [PyGraph Paper](https://arxiv.org/html/2503.19779v2)
- [CUDA Graph in torch.compile](https://fkong.tech/posts/2025-12-23-cuda-graph-in-torch-compile/)
- [NVIDIA CUDA Graph Best Practices](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html)

### Kernel Fusion
- [FlashFuser](https://arxiv.org/html/2512.12949v1)
- [Deep Kernel Fusion for Transformers](https://arxiv.org/html/2602.11808)
- [Triton Kernels for LLM Inference](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/)

### Quantization
- [MXFP4 Bridge Paper (ICLR 2026)](https://openreview.net/pdf?id=zCBGe9AqJZ)
- [FP4 on Blackwell](https://www.spheron.network/blog/fp4-quantization-blackwell-gpu-cost/)
- [TensorRT Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)

### Compiler Frameworks
- [TVM Relax Paper](https://arxiv.org/pdf/2311.02103)
- [Triton Compilation Stages](https://pytorch.org/blog/triton-kernel-compilation-stages/)
- [MLIR Linalg Rationale](https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/)
- [XLA Architecture](https://openxla.org/xla/architecture)
- [torch.compile with vLLM](https://blog.vllm.ai/2025/08/20/torch-compile.html)
- [Modular GTC 2026](https://www.modular.com/blog/modular-at-nvidia-gtc-2026-max-on-blackwell-mojo-kernel-porting-and-deepseek-v3-on-b200)

### IR Design
- [Writing an Optimizing Tensor Compiler from Scratch](https://michaelmoroz.github.io/WritingAnOptimizingTensorCompilerFromScratch/)
- [PowerFusion: Instruction-level Graph IR](https://ar5iv.labs.arxiv.org/html/2307.04995)
- [DISC: Dynamic Shape Compiler](https://arxiv.org/pdf/2103.05288)
- [Layout Transformation in ML Compilers](https://apxml.com/courses/compiler-optimizations-machine-learning/chapter-2-graph-level-transformations/layout-transformation)

### Codegen & Autotuning
- [LLVM NVPTX Docs](https://llvm.org/docs/NVPTXUsage.html)
- [DeepSeek PTX Innovation](https://medium.com/@noahbean3396/the-future-of-ai-compute-deepseeks-ptx-innovation-and-what-it-means-for-nvidia-f501b7a0f58e)
- [CUTLASS 4.2 Autotuning](https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/)
- [Triton Attention Anatomy](https://www.arxiv.org/pdf/2511.11581)
- [Tiling Guide](https://ianbarber.blog/2025/05/30/keeping-a-gpu-busy-is-a-lot-about-tiling/)

### GPU Targeting
- [SPIR-V Specification](https://www.khronos.org/spirv/)
- [Metal Shader Converter](https://developer.apple.com/metal/shader-converter/)
- [hetGPU](https://arxiv.org/pdf/2506.15993)
- [All GPU Stacks Overview](https://lukemiles.org/e/all-the-gpu-stacks.html)
- [One Kernel for All GPUs](https://hazyresearch.stanford.edu/blog/2025-09-22-pgl)

### Kernel Libraries
- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [ThunderKittens GitHub](https://github.com/HazyResearch/ThunderKittens)
- [ThunderKittens 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)
- [ParallelKittens](https://hazyresearch.stanford.edu/blog/2025-11-17-pk)
- [Liger Kernel GitHub](https://github.com/linkedin/Liger-Kernel)

### Runtime & Serving
- [LLM Inference Servers Compared](https://blog.premai.io/llm-inference-servers-compared-vllm-vs-tgi-vs-sglang-vs-triton-2026/)
- [NVIDIA CUDA Stream-Ordered Allocator](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/)
- [vLLM Triton Backend Deep Dive](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html)

### Rust Ecosystem
- [cudarc Docs](https://docs.rs/cudarc)
- [Rust CUDA Update](https://rust-gpu.github.io/blog/2025/05/27/rust-cuda-update/)
- [Rust on Every GPU](https://rust-gpu.github.io/blog/2025/07/25/rust-on-every-gpu/)
- [Burn Framework](https://github.com/tracel-ai/burn)
- [Cloudflare Infire](https://blog.cloudflare.com/cloudflares-most-efficient-ai-inference-engine/)
- [ort (ONNX Runtime for Rust)](https://ort.pyke.io/)
- [wgpu](https://wgpu.rs/)
- [PyO3](https://docs.rs/pyo3)

### Benchmarks & Comparisons
- [Inference Framework Comparison (MDPI)](https://www.mdpi.com/2079-9292/14/15/2977)
- [vLLM vs TRT-LLM](https://www.yottalabs.ai/post/vllm-vs-tensorrt-llm-architecture-performance-and-production-tradeoffs)
- [vLLM vs SGLang vs MAX](https://www.ersteiger.com/posts/vllm-vs-max/)

### tinygrad
- [tinygrad Docs](https://docs.tinygrad.org/)
- [tinygrad Backends Notes](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/backends.md)
- [Mesa NIR Backend](https://www.phoronix.com/news/Tinygrad-Mesa-NIR-Backend)

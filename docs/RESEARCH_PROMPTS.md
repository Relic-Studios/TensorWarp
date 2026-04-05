# Research Prompts for TensorWarp Advancement

These are structured research questions for delivery to Gemini Deep Research (or similar) to find cutting-edge techniques for pushing TensorWarp beyond TensorRT performance.

---

## 1. CUDA Graph Capture with cuBLAS

**Question:** How do production inference engines (TensorRT-LLM, vLLM, SGLang) capture cuBLAS GEMM calls inside CUDA graphs for LLM decode loops? Specifically:
- How is the cuBLAS handle configured for graph capture (stream binding, workspace allocation)?
- How do they handle dynamic parameters (position IDs, KV cache lengths) that change between graph replays — do they use `cudaGraphExecKernelNodeSetParams`, device-side buffers, or re-capture?
- What is the measured speedup from CUDA graph replay vs individual kernel launches for M=1 decode GEMMs on RTX 4090?
- Are there known issues with cuBLAS kernel selection inside captured graphs (e.g., cuBLAS selecting different kernels during capture vs replay)?
- Sample code or architecture diagrams showing the capture/replay flow for a transformer decode step.

---

## 2. Persistent Decode Kernels for Transformer Inference

**Question:** What are the current state-of-the-art approaches for persistent decode kernels in transformer inference? Specifically:
- How does FasterTransformer/TensorRT-LLM implement persistent thread blocks that stay resident across the entire decode step?
- How is the coordination between persistent kernels and cuBLAS GEMM calls handled — atomic flags, stream events, or cooperative groups?
- What is the measured latency reduction from persistent kernels vs individual launches for a 7B model decode step on RTX 4090?
- Are there open-source implementations of persistent decode kernels (SGLang, FlashInfer, vLLM) that we can study?
- What are the register pressure and shared memory constraints for a persistent kernel that handles RMSNorm + RoPE + attention + SwiGLU for a model with H=3584, head_dim=128?

---

## 3. Optimal INT4 Quantization for Inference Speed

**Question:** What is the fastest INT4 GEMM implementation for M=1 decode on NVIDIA Ada/Ampere GPUs? Specifically:
- How does Marlin (GPTQ kernel) achieve near-FP16 throughput for INT4 GEMM? What tiling, dequantization, and memory access patterns does it use?
- How does ExLlamaV2's INT4 GEMM compare to Marlin in throughput for M=1, K=3584, N=18944?
- What is the optimal block format for INT4 weights on RTX 4090 — Q4_0 (per-block absmax), Q4_K (per-superblock with shared scale), or GPTQ (per-group with zero-points)?
- Can INT4 GEMM use tensor cores on RTX 4090 (SM89)? If so, how (via INT8 tensor cores with double-packed INT4)?
- What is the theoretical bandwidth-limited throughput for INT4 GEMM at M=1, K=3584, N=18944 on RTX 4090 (1008 GB/s)?

---

## 4. FlashDecoding and Split-K Attention for Long Contexts

**Question:** How does FlashDecoding (from Tri Dao / Flash Attention team) optimize decode attention for long sequences? Specifically:
- What is the Split-K approach for decode attention — how does it parallelize across the sequence length dimension?
- How does it compare to our current multihead decode attention kernel (one thread block per head, sequential iteration over cache positions)?
- What is the measured speedup for decode attention at cache_len=4096 vs cache_len=128?
- How does FlashInfer (from the CMU team) differ from FlashDecoding?
- Are there CUDA kernel implementations we can reference for Split-K decode attention with GQA (grouped query attention)?

---

## 5. Gemma Architecture Differences from LLaMA

**Question:** What are the architectural differences between Google's Gemma models (Gemma 2, Gemma 4) and LLaMA/Qwen that would affect an inference engine implementation? Specifically:
- Does Gemma use RMSNorm or LayerNorm? Pre-norm or post-norm?
- Does Gemma use RoPE? If so, what pairing convention (adjacent or half-split)?
- Does Gemma use GQA? What is the Q/KV head ratio for Gemma 4 31B?
- Does Gemma use SwiGLU for the FFN, or a different activation?
- Does Gemma use attention biases, position biases (ALiBi), or learned position embeddings?
- What is the exact `config.json` schema for Gemma 4 31B — what fields map to hidden_size, num_heads, num_kv_heads, intermediate_size?
- Are there weight naming differences from HuggingFace's LLaMA format?
- Does Gemma use tie_word_embeddings?

---

## 6. ONNX Runtime for Diffusion Models (Stable Diffusion, SDXL)

**Question:** How do production systems run Stable Diffusion / SDXL inference through ONNX? Specifically:
- What ONNX ops are required for the U-Net (Conv2D, GroupNorm, attention, Resize)?
- What is the typical ONNX export process from PyTorch for SDXL (`torch.onnx.export` settings, opset version)?
- What are the critical performance optimizations for U-Net inference — which ops dominate runtime?
- How does TensorRT optimize the U-Net graph (which fusions does it apply)?
- What are the memory requirements for SDXL U-Net at FP16 on RTX 4090?
- Are there public ONNX exports of SDXL that we can test with?

---

## 7. YOLOv10 Inference Optimization

**Question:** What are the performance-critical aspects of running YOLOv8/v10 inference through a custom engine? Specifically:
- What ONNX ops does YOLOv10 use (Conv2D, SiLU, Concat, Resize, NMS)?
- What is the typical input resolution and batch size for real-time detection?
- Where is the compute bottleneck — backbone convolutions or detection head?
- How does TensorRT optimize YOLO models (Conv+BN+SiLU fusion, INT8 quantization)?
- What throughput (FPS) does TensorRT achieve on YOLOv10 at 640x640 on RTX 4090?
- Are there public ONNX exports of YOLOv10 we can test with?

---

## 8. Memory-Efficient KV Cache for Large Models

**Question:** What are the current approaches for running 30B+ models in 24GB VRAM with efficient KV cache management? Specifically:
- How does vLLM's PagedAttention manage KV cache memory — block tables, dynamic allocation, eviction?
- How does SGLang's RadixAttention differ from PagedAttention?
- What is the memory overhead of paged vs contiguous KV cache?
- For a 31B model at INT4 (~15.5 GB weights), how much KV cache fits in 24 GB at different context lengths?
- How do systems handle KV cache overflow — offloading to CPU, quantizing KV to INT8, or truncating?
- What is the measured throughput impact of paged vs contiguous KV cache on RTX 4090?

---

## 9. Beating cuBLAS for M=1 GEMM

**Question:** Are there techniques to outperform cuBLAS for M=1 (single-token decode) GEMM on RTX 4090? Specifically:
- cuBLAS has per-call overhead (~50-100μs) that dominates for M=1. Can this be amortized?
- Does cublasLt with pre-compiled matmul plans reduce this overhead? If so, by how much?
- Can custom CUDA kernels outperform cuBLAS for M=1 by using persistent threads that keep weight tiles in registers/L2?
- What is the theoretical minimum latency for M=1, K=3584, N=3584 GEMM at FP16 on RTX 4090?
- How do Triton (triton-lang) M=1 GEMM kernels compare to cuBLAS?

---

## 10. Rust + CUDA Ecosystem for Inference

**Question:** What is the current state of the Rust + CUDA ecosystem for building inference engines? Specifically:
- What are the alternatives to cudarc for CUDA bindings in Rust (cuda-sys, rustacuda, cust)?
- How does burn-rs approach GPU inference? What can TensorWarp learn from their architecture?
- How does candle (HuggingFace's Rust ML framework) handle CUDA kernels and cuBLAS?
- Are there Rust bindings for cuDNN (for optimized Conv2D, BatchNorm)?
- What are the performance implications of calling CUDA through Rust FFI vs native C++?
- How do other Rust inference projects handle NVRTC (runtime compilation) vs pre-compiled kernels?

---

## How to Use These Prompts

Feed each prompt to Gemini Deep Research (or equivalent) individually. Collect the responses into `docs/RESEARCH_RESULTS.md`. Focus on:
1. **Code examples** — actual CUDA kernels, launch configs, architecture diagrams
2. **Measured numbers** — latency, throughput, TFLOPS on specific hardware
3. **Open-source references** — GitHub repos, papers with code
4. **Architecture decisions** — what tradeoffs did production systems make and why

The answers will directly inform TensorWarp's next development phases.

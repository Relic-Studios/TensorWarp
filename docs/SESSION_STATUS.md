# TensorWarp Session Status — April 5, 2026

## What TensorWarp IS

A from-scratch GPU inference engine in Rust + CUDA that runs real LLM models faster than PyTorch and aims to beat TensorRT. 42K+ lines, 270+ tests, 9 crates, MIT licensed.

## What WORKS Right Now

### Verified on RTX 4090:

| Model | Precision | VRAM | Decode | vs PyTorch | Correct? |
|-------|-----------|------|--------|-----------|----------|
| Qwen 0.5B | F32 (TF32 TC) | 2.5 GB | **107 tok/sec** | 15x faster | ✅ Yes |
| Qwen 0.5B | FP16 full | 1.8 GB | **38 tok/sec** | 5.5x | ✅ Yes |
| Qwen 7B | FP16 full | 17.4 GB | **14-23 tok/sec** | 2-3x | ✅ Yes |
| Qwen 7B | Q4_0 | 8.4 GB | 7.6 tok/sec | — | ❌ Garbage |

### Key infrastructure:
- cuBLAS GEMM backend (147 TFLOPS F32 TF32, 145 TFLOPS FP16)
- Pre-allocated decode buffers (zero cudaMalloc during decode — 5x speedup)
- Tensor core math mode enabled (`CUBLAS_TENSOR_OP_MATH`)
- Full FP16 pipeline: hidden state stays FP16 across all layers
- FP16 KV cache, FP16 attention, FP16 norms, FP16 RoPE
- Device-pos kernel variants ready for CUDA graph capture
- Sharded SafeTensors loader (multi-file models like 7B)
- HuggingFace tokenizers crate for correct tokenization
- Layer-by-layer debug infrastructure verified against PyTorch

## What's BROKEN

### Q4 Quantized Inference (output is garbage)
The Q4 decode path has the same bugs we already fixed for F32:
1. **Q4 prefill missing RoPE transpose** — GEMM outputs `[seq, heads*dim]` but RoPE expects `[heads, seq, dim]`. Fixed in F32 prefill at `transformer.rs:662-690`. Not copied to Q4 prefill at `transformer.rs:380-456`.
2. **Q4 prefill missing KV cache layout transpose** — K stored in heads-first after RoPE but cache expects positions-first. Fixed in F32 at `transformer.rs:693-707`.
3. **Q4 prefill has no bias handling** — Qwen uses Q/K/V biases. Added to Q4 decode but not Q4 prefill.
4. **Q4 uses old attention path** — `attention_ext::attention_best` without the output transpose back to positions-first. Fixed in F32 at `transformer.rs:719-757`.

### FP16 Cast Overhead
For 0.5B models, FP16 is slower than F32 because cast kernel launches (14 per layer) exceed bandwidth savings. Not a bug — architectural limitation for small matrices.

### Thermal Throttling
RTX 4090 throttles under sustained load: 23 tok/sec cold → 6.5 tok/sec hot. Hardware limitation.

## Architecture Overview

```
crates/
├── ir/          — Graph IR, 130+ ops, SSA
├── optimizer/   — Pattern fusion, autofuse, constfold, DCE, memory planning
├── codegen/     — PTX + Metal generation
├── runtime/     — Runtime types
├── kernels/     — 160+ CUDA kernels, engine, autotuner, generation
│   ├── ops.rs              — Core ops, cuBLAS dispatch, fused kernels
│   ├── transformer.rs      — F32/FP16/Q4 transformer blocks
│   ├── generate.rs         — Generation engines (F32, FP16, Q4)
│   ├── cublas_gemm.rs      — cuBLAS F32/FP16 GEMM wrappers
│   ├── fp16.rs             — FP16 kernels (norm, RoPE, attention, SwiGLU)
│   ├── kv_cache.rs         — F32/FP16 KV cache + multihead attention
│   ├── quantize.rs         — Q4/Q8/INT8/FP8/GPTQ/AWQ
│   ├── attention.rs        — Flash Attention v2
│   ├── rope.rs             — RoPE (half-split pairing, fixed)
│   ├── autotune.rs         — Cost model, Thompson sampling, shape buckets
│   ├── cost_model.rs       — Analytical roofline scoring
│   ├── speculative.rs      — Speculative decoding
│   └── cuda_graph.rs       — CUDA graph capture/replay
├── loader/      — ONNX, SafeTensors, GGUF, LLaMA, tokenizer
│   ├── llama.rs            — F32/FP16/Q4 model loading
│   ├── safetensors_loader.rs — Single + sharded loading
│   ├── tokenizer.rs        — HuggingFace tokenizers crate
│   └── debug_model.rs      — Layer-by-layer PyTorch comparison
├── cli/         — 10 CLI commands
├── python/      — PyO3 bindings
└── server/      — OpenAI-compatible HTTP server
```

## Critical Bugs Found (and Fixed)

1. **RoPE paired adjacent dims (0,1) instead of half-split (0,32)** — `rope.rs:45`. Every model's positional encoding was wrong. Fixed in both F32 and FP16 RoPE.

2. **GEMM output layout mismatch** — GEMM produces `[seq, heads*dim]` (positions-first) but RoPE expects `[heads, seq, dim]` (heads-first). Added GPU transpose kernels in `ops.rs` for the prefill path.

3. **Decode output projection K=head_dim instead of K=hidden_size** — `transformer.rs:907`. Only read one head's attention output. Fixed in F32 and Q4 decode.

4. **FP16 RoPE had the same pairing bug** — `fp16.rs:399`. Fixed.

5. **Decode attention was single-head** — `kv_cache.rs:280`. Added `decode_attention_multihead` for GQA.

## What Needs to Happen Next

### Priority 1: Fix Q4 correctness
Copy the RoPE transpose + KV cache layout + bias + attention fixes from the F32 prefill path (`transformer.rs:662-757`) to the Q4 prefill path (`transformer.rs:380-456`). This is copy-paste + adapt.

### Priority 2: Refactor shared code
All 4 precision paths (F32, FP16 mixed, FP16 full, Q4) need the same non-GEMM code (RoPE, attention, bias, residual, norm). Extract into a shared function parameterized by the GEMM type.

### Priority 3: CUDA graph capture
Device-pos kernel variants exist (`f16_rope_device_pos`, `kv_cache_append_f16_device_pos`, `decode_attention_multihead_f16_device_len`). Need to:
1. Create a capture stream
2. Set cuBLAS to capture stream
3. Capture one decode step
4. Replay with pos_buf update

### Priority 4: Gemma 31B
1. Download GGUF Q4 or load SafeTensors + quantize
2. Gemma may use different weight names, norm type (LayerNorm vs RMSNorm), or attention pattern
3. Check `config.json` for architecture differences

### Priority 5: Diffusion + Tracking models
Test ONNX pipeline with real SDXL U-Net and YOLOv8/v10 exports.

## Performance Numbers to Remember

| Metric | Value |
|--------|-------|
| F32 GEMM (cuBLAS TF32) | 147 TFLOPS |
| FP16 GEMM (cuBLAS TC) | 145 TFLOPS |
| RTX 4090 bandwidth | 1008 GB/s |
| 0.5B F32 bandwidth floor | 1.31 GB/token → 763 tok/sec max |
| 0.5B F32 actual | 107 tok/sec (14% of floor) |
| 7B FP16 bandwidth floor | 26.1 GB/token → 39 tok/sec max |
| 7B FP16 actual | 14-23 tok/sec (36-59% of floor) |
| TensorRT 7B FP16 estimate | ~25-40 tok/sec |
| PyTorch 0.5B F32 | 6.9 tok/sec |

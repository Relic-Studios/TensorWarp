# TensorWarp Session Handoff: Path to 100 tok/sec on 7B Q4

## Current State (April 5, 2026)

**Best numbers on RTX 4090:**
- Qwen 7B Q4: **46.7 tok/sec** (`--q4 --marlin`)
- Qwen 0.5B Q4: **79.1 tok/sec** (`--q4`)
- Theoretical bandwidth floor for 7B Q4: **120 tok/sec** (4.06 GB weights at 1008 GB/s)
- Current bandwidth utilization: **~39%**

**Target: 100 tok/sec = 83% bandwidth utilization.**

## Architecture Summary

TensorWarp is a Rust+CUDA inference engine. 42K+ lines, 9 crates, 184 tests passing. Kernels are CUDA C compiled via NVRTC at runtime, cached by source hash. Uses vendored cudarc 0.16 (patched for CUDA graph capture safety).

Key files:
- `crates/kernels/src/quantize.rs` — Q4 GEMM kernels, TW-Marlin format, weight reordering
- `crates/kernels/src/generate.rs` — Generation engines, pre-allocated decode buffers, CUDA graph capture
- `crates/kernels/src/transformer.rs` — Transformer block implementations (10 precision paths)
- `crates/kernels/src/kv_cache.rs` — KV cache, FlashDecoding Split-K attention
- `crates/kernels/src/device.rs` — CUDA device, cuBLAS workspace (256MB pre-allocated)
- `crates/kernels/src/ops.rs` — Fused kernels (bias+RoPE+KV-append, residual+norm, SwiGLU)
- `vendor/cudarc/src/driver/safe/launch.rs` — Patched: skips bind_to_thread during graph capture

## What's Been Optimized (and the measured impact)

| Optimization | Impact on 7B | Status |
|---|---|---|
| M=1 GEMM (one thread per output column) | 8 -> 22 tok/s (+175%) | Done |
| Block-major Q4 weight reorder | 22 -> 26 tok/s (+18%) | Done |
| Split-K 2D grid (blockIdx.y = K-split) | 26 -> 44.5 tok/s (+71%) | Done |
| TW-Marlin separated scales+packed format | 44.5 -> 46.7 tok/s (+5%) | Done |
| CUDA graph capture (transformer layers only) | +8% on 0.5B, modest on 7B | Done (LM head runs eagerly) |
| Pre-allocated decode buffers | Eliminates ~840 cudaMalloc/token | Done |
| Fused bias+RoPE+KV-append (1 launch replaces 6) | -140 launches/token | Done |
| Adaptive split selection (skip Split-K for small K) | Prevents 0.5B regression | Done |
| cuBLAS workspace pre-allocation | Required for graph capture | Done |
| FlashDecoding Split-K attention | Bit-perfect, auto-fallback | Done |

## What Was Tried and Did NOT Help

| Attempt | Why It Failed |
|---|---|
| Shared-memory A vector (V2 kernel) | FFN K=18944 exceeds shared memory. For small K, L1 cache via __ldg is already effective. |
| Warp-cooperative shared memory loading (V3) | __syncthreads per K-block = 22K syncs/token. Hardware coalescer handles 20-byte stride fine. |
| Software pipeline (register pre-load next block) | NVRTC compiler already optimizes instruction scheduling. Extra registers reduce occupancy. |
| Fused QKV projections (3 GEMMs -> 1) | Split-K already handles SM occupancy. split_qkv kernel adds overhead that negates savings. |
| Q4->FP16 full dequant + cuBLAS HGEMM | 17.4 GB FP16 weights don't fit on 24GB alongside KV cache for 7B. Works great for 0.5B. |

## Where the Remaining Time Goes

At 46.7 tok/sec = 21.4ms per token:
- **GEMM kernel compute**: ~12ms (reading 4.06 GB Q4 weights at ~39% of 1008 GB/s)
- **Non-GEMM ops**: ~5ms (attention, norms, RoPE, SwiGLU, residuals, embedding)
- **Launch overhead + memset**: ~4ms (memset_zeros for Split-K atomicAdd, kernel launches)

To reach 10ms/token (100 tok/sec):
- GEMM must take ~6ms -> need **67% bandwidth utilization** (up from 39%)
- Non-GEMM budget: ~4ms (already close)

## The Three Paths to 100 tok/sec

### Path A: Improve Q4 GEMM Bandwidth (39% -> 67%)

The kernel's inner loop is the bottleneck. Each thread reads a 16-byte packed nibble block + 4-byte scale per K-group, then does 32 MADs + nibble extraction. Improvements:

1. **FP16 scales instead of F32** — Halves scale bandwidth (4B -> 2B per group). Saves ~0.5ms/token for 7B.

2. **Inline PTX dequantization** — Use `prmt.b32` for nibble extraction instead of shift+mask. Use `lop3.b32` for combined operations. Saves ~2 ALU cycles per element. See `docs/research_prompts/marlin-adaptive-gemm-findings.md` section 6a for exact PTX sequences.

3. **cp.async pipeline** — Use `cp.async.cs.shared.global` for weight loads with double-buffered shared memory. Overlaps next tile load with current tile compute. Requires inline PTX in NVRTC (confirmed working via `asm volatile`). The key insight: don't use __syncthreads per K-block. Instead, pipeline N K-blocks worth of data and sync only at pipeline boundaries.

4. **Thread coarsening with register tiling** — Each thread processes 2-4 output columns instead of 1. Amortizes A-vector reads across multiple dot products. Increases register pressure but reduces memory traffic.

### Path B: Eliminate Non-GEMM Overhead

1. **Get cuBLAS LM head into CUDA graph** — Currently the LM head (vocab=152K F32 GEMM) runs eagerly after graph replay. Fix: use device pointer mode for cuBLAS alpha/beta, or replace with a custom M=1 F32 GEMM kernel.

2. **Eliminate memset_zeros for Split-K** — Instead of zeroing output + atomicAdd, have split 0 write directly and splits 1+ use atomicAdd. Saves one memset per GEMM.

3. **Fuse split_qkv into the GEMM epilogue** — Write directly to Q, K, V output buffers from the fused QKV GEMM instead of writing to a temp buffer and splitting.

### Path C: Algorithmic

1. **Speculative decoding** — Use the 0.5B model (79 tok/s) as draft, verify with 7B. If acceptance rate > 70%, effective throughput > 70 tok/s.

2. **Batch decode** — Process multiple tokens per step with speculative or parallel decoding. Changes M=1 to M=4-8, shifting from memory-bound to compute-bound where tensor cores help.

## Key Research Documents

- `docs/research_prompts/marlin-adaptive-gemm-findings.md` — 80K chars of Gemini research on Marlin weight layout, PTX dequant, architecture feature matrix, implementation roadmap
- `docs/research_prompts/marlin-adaptive-quantized-gemm.md` — The research prompt that generated the findings
- `docs/TensorWarp Research Prompts for Advancement.md` — Earlier Gemini research on FlashDecoding, PagedAttention, CUDA graphs, cuBLAS optimization

## Architecture Feature Matrix (from research)

| Feature | Ada SM 8.9 (RTX 4090) | Hopper SM 9.0 | Blackwell SM 10.0 |
|---|---|---|---|
| INT4 Tensor Cores | No (use INT8 trick) | No (use INT8 trick) | **YES (native)** |
| cp.async | Yes | Yes (TMA is better) | Yes (TMA v2) |
| Shared Memory | 48KB/96KB per SM | 228KB per SM | Increased |
| L2 Cache | 72MB | 50MB | Increased |
| Recommended approach | mma.sync + INT8 double-pack | wgmma + TMA | wgmma + native INT4 |

## Quick Start Commands

```bash
# Best 7B Q4 path (TW-Marlin + Split-K):
cargo run --release -p tensorwarp -- run <model-path> --q4 --marlin --max-tokens 64

# Best 0.5B Q4 path (non-splitk block-major):
cargo run --release -p tensorwarp -- run <model-path> --q4 --max-tokens 64

# With CUDA graph (modest gain, LM head runs eagerly):
cargo run --release -p tensorwarp -- run <model-path> --q4 --graph --max-tokens 64

# Run tests:
cargo test -p warp-kernels  # 184 pass, 4 pre-existing failures
```

Model paths for Qwen on this machine:
- 0.5B: `/c/Users/zappa/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- 7B: `/c/Users/zappa/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796`

## Recommended Next Session Priority

1. **FP16 scales + eliminate memset** (quick wins, ~5-10% gain)
2. **Inline PTX dequant (prmt/lop3)** (medium effort, ~10-15% gain)
3. **cp.async double-buffer pipeline** (high effort, ~15-25% gain)
4. **Thread coarsening (2-4 cols per thread)** (medium effort, ~10-20% gain)

Combined these could reach 60-80% bandwidth = 72-96 tok/sec. The last 4-28 tok/sec to reach 100 would come from speculative decoding or getting the LM head into the graph.

The challenge of achieving high bandwidth utilization for INT4 GEMM at M=1 on GPUs, particularly with non-power-of-2 block sizes, is common. Marlin and similar systems succeed by meticulously optimizing the data layout, dequantization, and memory access patterns to align with GPU architectural specifics, especially Tensor Core requirements and async memory operations.

This research details how TensorWarp can implement such an adaptive quantized GEMM system, starting with a deep dive into Marlin's methodology.

---

## 0. Marlin Weight Layout and Dequantization Pipeline

Marlin's design is heavily influenced by the goal of maximizing Tensor Core utilization and memory bandwidth for INT4 GEMMs, especially at M=1. It achieves this by a specialized weight layout, aggressive async memory pipelining, and efficient on-the-fly dequantization.

### 0a. Marlin's Exact Memory Layout for INT4 Weights

Marlin's memory layout is designed to present weights to the Tensor Cores in a format that minimizes memory stalls and maximizes parallel computation. For an `A (M, K) @ B (K, N) -> C (M, N)` GEMM, where `B` are the quantized weights, Marlin reorganizes `B` to be efficient for `mma.sync` operations. The critical design choices are:

1.  **Group-wise Quantization:** Weights are quantized in groups, typically 128 elements. Each group shares a single `f16` scale and, if asymmetric, a `f16` zero-point. This balances quantization granularity with memory overhead.
2.  **Column-Major within Groups, Row-Major Groups:** While the overall matrix is `K x N`, Marlin's internal layout is often described as optimized for column-major access from the perspective of `N` dimension. Specifically, for a warp processing a slice of `N` and iterating over `K`, the weights for that slice of `N` need to be loaded efficiently.
3.  **Packed Nibbles:** 4-bit weights are packed into bytes (2 nibbles per byte). The packing order (low-nibble first or high-nibble first) is specific to the dequantization strategy.
4.  **Interleaved Scales:** Scales (and zero-points) are stored *contiguously with their respective quantized weight groups* but typically *separately from the packed nibbles*. This allows fetching scales and weights independently or together based on the kernel's needs. Scales are usually `f16`.
5.  **Tile-based Reordering:** The entire `K x N` weight matrix is logically divided into `[K_TILE, N_TILE]` blocks. Within these blocks, the actual byte-level layout is further optimized. For Tensor Cores, which consume data in specific warp-level `mma` shapes (e.g., `16x8x16` or `8x8x32`), the `K` dimension of the weight matrix (`B`) must be laid out to allow contiguous fetches for the inner product `K` dimension of the MMA.

**"Striped Partitioning Scheme":** This refers to how the `N` dimension (output columns) is distributed. Instead of assigning contiguous blocks of `N` to warps, `N` is often "striped" across warps or thread blocks. This ensures that warps working on adjacent output columns access spatially separated (but predictable) memory locations, reducing bank conflicts and improving memory coalescing. For example, if a thread block is responsible for `N_BLOCK` columns, these columns might be `0, 1, ..., N_BLOCK-1` or `0, N_THREADS_PER_BLOCK, 2*N_THREADS_PER_BLOCK, ...`. Marlin uses a fine-grained partitioning (e.g., `N_TILE` of 64 or 128 columns) where weights are reordered such that all `K` elements for a `N_TILE` column slice are contiguous or easily accessed.

**Offline Weight Pre-processing ("Pre-shuffling"):**
The goal is to transform the standard `[K, N]` row-major INT4 matrix into a format that allows warp-level `cp.async` to fetch 4-bit weights and scales in a highly coalesced manner, ready for immediate `lop3`-based dequantization and subsequent `mma.sync` operations.

**Reorder Algorithm (Pseudocode for a simplified Marlin-like layout):**
Assume:
*   Original weights `W_orig[K_total][N_total]` (float values before quant)
*   Quantized weights `W_q[K_total][N_total]` (int4 values, 0-15)
*   Scales `S[N_total / GROUP_SIZE]` (f16, one scale per group of 128 elements in N)
*   Zero-points `ZP[N_total / GROUP_SIZE]` (int4 or f16, if asymmetric)
*   `GROUP_SIZE = 128` (common for GPTQ/Marlin)
*   `PACK_SIZE = 2` (2 nibbles per byte)
*   Output layout `W_marlin_packed` (bytes), `S_marlin` (f16)

The key is to arrange data to facilitate `cp.async` loads and `mma.sync` usage. Tensor Cores typically process `M x N x K` blocks. For `B` matrix (weights), `K` is the input feature dimension, `N` is the output feature dimension. A typical `mma.sync.m16n8k16.s32.f16.f16` operation expects `B` to provide `k=16` elements for an `n=8` column slice.

Let's assume a Marlin-like layout for `B` (weights) that groups `N_TILE` columns and `K_TILE` rows.
A common structure for `B` is `[K/K_TILE, N/N_TILE, K_TILE, N_TILE]`. Marlin optimizes this further for `GROUP_SIZE=128`.

```pseudocode
function reorder_marlin_weights(W_q[K][N], S[N/GROUP_SIZE], ZP[N/GROUP_SIZE])
    // K: rows (input features), N: columns (output features)
    // GROUP_SIZE: typically 128
    // K_PER_GROUP: Marlin typically applies scale per K-group AND N-group (e.g., GPTQ-128 means scales apply to N=128)
    // The scale applies to a K x 128 segment of the original matrix.
    // Let's assume scales are N_total / GROUP_SIZE, meaning K-dimension is implicit.

    // Marlin specific:
    // N_GROUP_SIZE = 128 // Number of columns per quantization group
    // K_BLOCK_SIZE = 128 // Or 64, or 32 for internal processing within the kernel

    // Simplified Marlin-like structure:
    // Group weights and scales by N_GROUP_SIZE
    // Within each N_GROUP_SIZE block, reorder K-rows to be contiguous for efficient loading by warps.
    // The layout is effectively:
    // [ (K_total / K_BLOCK_SIZE) * (N_total / N_GROUP_SIZE) ] blocks,
    // where each block contains K_BLOCK_SIZE * N_GROUP_SIZE elements.
    // Within each block, elements are packed for warp-level MMA.

    // Let's assume the target layout is optimized for
    // `mma.sync.m16n8k16.s32.f16.f16` on Ada/Hopper,
    // meaning the inner K dimension (for B) is 16 elements.
    // Scales are applied per N_GROUP_SIZE (128 columns) for the *entire* K dimension.
    // This implies scales are (K_total/K_BLOCK_SIZE) * (N_total/N_GROUP_SIZE) scales in practice, one per block.
    // This is often simplified to (N_total/GROUP_SIZE) * (K_total/K_TILE) if scales are K_TILE-wise.

    // Let's make a common assumption: scales are (K / K_GROUP_SIZE) * (N / N_GROUP_SIZE)
    // where K_GROUP_SIZE = 1 and N_GROUP_SIZE = 128. This means there are (K / 1) * (N / 128) scales.

    // Actual Marlin Layout Example:
    // W_out[N_total/N_TILE][K_total/K_TILE][N_TILE][K_TILE/PACK_SIZE]
    // Scales_out[N_total/N_GROUP_SIZE][K_total/K_GROUP_SIZE_FOR_SCALE_APPLICATION]
    // ZeroPoints_out[N_total/N_GROUP_SIZE][K_total/K_GROUP_SIZE_FOR_SCALE_APPLICATION]

    // Here's a common reordering pattern for GPTQ-like Marlin:
    // It groups by N_GROUP_SIZE (128) and reorders K within these groups
    // to expose K_TILE (e.g., 64 or 128) rows together.
    // The innermost dimension is typically the packed INT4 bytes.
    // The scales are often stored *after* their respective weight blocks or in a separate array,
    // but the key is they are fetched alongside the weights.

    // Example for a [K, N] matrix with GROUP_SIZE=128.
    // Let K_TILE_MM = 64, N_TILE_MM = 128 (for MMA processing)
    // Marlin often reorders to:
    // OUT_SHAPE = [N / N_TILE_MM, K / K_TILE_MM, N_TILE_MM, K_TILE_MM / 2 (packed bytes)]
    // Scales_OUT_SHAPE = [N / N_GROUP_SIZE, K / K_GROUP_SIZE_FOR_SCALE] (f16 scales)

    // For simplicity, let's assume a target layout where groups of K_TILE x N_TILE weights are rearranged.
    // A specific structure from GPTQ-Marlin is to store data as:
    // [K / 16, N / 8, 16, 8 / 2] (for 16x8x16 MMA shape, where N is split by 8 and K by 16)
    // This means data for a 16 rows x 8 columns block is packed together.

    // Marlin's approach is more sophisticated: it groups 128 elements in the N dimension,
    // and for these 128 elements, K rows are stored in a specific interleaved fashion.
    // The underlying data is often structured to allow 8-byte (64-bit) vector loads
    // of packed nibbles for multiple K elements across multiple N columns in a single transaction.

    // Let's detail the "pre-shuffle" into Marlin-like format for 128-group, K-major blocks:
    // Target Format: Marlin stores weights for a K_block x N_group (e.g. 128 x 128) as:
    // [N_total / 128, K_total / 128, 128 (N-dim), 64 (K-dim packed bytes)]
    // Within each 128x128 block:
    // For each N_group_idx = 0 to N_total / 128 - 1:
    //   For each K_block_idx = 0 to K_total / 128 - 1:
    //     For each K_inner = 0 to 127: // Iterates over rows within K_block
    //       For each N_inner = 0 to 127 / 2 - 1: // Iterates over packed bytes within N_group
    //         W_marlin_packed[flattened_idx] = pack_nibbles(
    //             W_q[K_block_idx*128 + K_inner][N_group_idx*128 + N_inner*2],   // high nibble
    //             W_q[K_block_idx*128 + K_inner][N_group_idx*128 + N_inner*2 + 1] // low nibble
    //         )
    // Scales and ZeroPoints for (N_group_idx, K_block_idx) are stored nearby.

    // The key is that the inner-most loops expose data contiguously for `cp.async`.
    // The specific ordering determines which indices are grouped by warps.
    // Marlin often uses a "column-major for vector loads" strategy: it wants to load contiguous data across N.
    // So, a common structure is N-major blocks, with K-elements interleaved.

    // More precise (GPTQ-Marlin) layout (simplified for example):
    // Data for N columns is split into `N_group = 128` sized groups.
    // Data for K rows is split into `K_group = 32` sized groups.
    // Within each `K_group x N_group` block:
    // Weights are stored: `[N_group / 64][K_group / 16][64 (packed nibbles)][16]`
    // Where `[64]` refers to 32 bytes (64 nibbles) across N for 16 K rows,
    // and `[16]` refers to 16 K-rows.
    // This is `(N/64) * (K/16) * 32 * 16` bytes.

    // For the [3584, 3584] matrix example:
    // N_total = 3584, K_total = 3584
    // N_group_size = 128
    // Scales_per_group_K_rows = 128 (means scales apply to K=128 rows for N=128 group)
    // N_groups = 3584 / 128 = 28
    // K_groups = 3584 / 128 = 28
    // Total scales = 28 * 28 = 784 scales (each f16)
    // Total zero_points = 28 * 28 = 784 zero-points (each f16)

    // A common Marlin physical layout (conceptually):
    // `weights_packed`: A flattened array of bytes.
    // `scales`: A flattened array of f16 scales.
    // `zeros`: A flattened array of f16 zero-points (if asymmetric).

    // `weights_packed` indexing:
    // `idx(k, n) = (n / 128) * (K_total / 128) * (128 * 64) + (k / 128) * (128 * 64) + ((n % 128) / 2) * 128 + (k % 128)`
    // This reorders `(k, n)` elements into `(n_group, k_group, n_inner_packed, k_inner)`
    // where `n_inner_packed` is `(n % 128) / 2` and `k_inner` is `k % 128`.
    // The `128*64` comes from 128 rows and 64 bytes (128 nibbles) for a K_block x N_group.
    // This allows contiguous loading for a warp processing a `K_block` of rows and `N_group` columns.

    // Pseudocode for reordering INT4 `W_q[K][N]` and `S[num_scales]` into Marlin format:
    // Assuming GROUP_SIZE = 128 (N-dimension groups)
    // Assuming K_SEGMENT_SIZE = 128 (K-dimension segments for reordering)
    // Output structure:
    // Marlin_W[ num_N_groups ][ num_K_segments ][ K_SEGMENT_SIZE ][ GROUP_SIZE / 2 (bytes) ]
    // Marlin_S[ num_N_groups ][ num_K_segments ] (f16)
    // Marlin_ZP[ num_N_groups ][ num_K_segments ] (f16)

    output_weights_packed = array of bytes of size (K * N / 2)
    output_scales = array of f16 of size (K / K_SEGMENT_SIZE) * (N / GROUP_SIZE)
    output_zeros = array of f16 of size (K / K_SEGMENT_SIZE) * (N / GROUP_SIZE)

    for n_block = 0 to N / GROUP_SIZE - 1:
        for k_segment = 0 to K / K_SEGMENT_SIZE - 1:
            // Calculate scale/zero-point for this (k_segment, n_block)
            // This needs the original FP32 matrix or the pre-quantized scales from the source format.
            // For now, assume S[n_block * (K / K_SEGMENT_SIZE) + k_segment] are given.
            output_scales[n_block * (K / K_SEGMENT_SIZE) + k_segment] = S_original[n_block * (K / K_SEGMENT_SIZE) + k_segment]
            output_zeros[n_block * (K / K_SEGMENT_SIZE) + k_segment] = ZP_original[n_block * (K / K_SEGMENT_SIZE) + k_segment]

            for k_inner = 0 to K_SEGMENT_SIZE - 1:
                for n_inner_pair = 0 to GROUP_SIZE / 2 - 1:
                    // Original (k, n) coordinates
                    k_orig = k_segment * K_SEGMENT_SIZE + k_inner
                    n_orig_high = n_block * GROUP_SIZE + n_inner_pair * 2
                    n_orig_low = n_block * GROUP_SIZE + n_inner_pair * 2 + 1

                    // Quantized nibbles (0-15)
                    nibble_high = W_q[k_orig][n_orig_high]
                    nibble_low = W_q[k_orig][n_orig_low]

                    // Pack into a byte (high nibble first, or low nibble first, depends on lop3 strategy)
                    packed_byte = (nibble_high << 4) | nibble_low

                    // Calculate target index in flattened `output_weights_packed`
                    // Order: N_group, K_segment, K_inner, N_inner_byte
                    target_idx = n_block * (K / K_SEGMENT_SIZE) * (K_SEGMENT_SIZE * GROUP_SIZE / 2) + \
                                 k_segment * (K_SEGMENT_SIZE * GROUP_SIZE / 2) + \
                                 k_inner * (GROUP_SIZE / 2) + \
                                 n_inner_pair

                    output_weights_packed[target_idx] = packed_byte

    return output_weights_packed, output_scales, output_zeros
```
**Exact Byte-Level Transformations:** This reordering is often done at the granularity of `K_TILE x N_TILE` blocks (e.g., 64x128 elements), where each block of INT4 data (e.g., `64 * 128 / 2 = 4096` bytes) is contiguous. Within this block, the elements are arranged to allow vectorized loads (e.g., `ld.global.v8.b16`) for multiple threads/warps to fetch 64-bit or 128-bit chunks of packed nibbles. The `f16` group scales and zero-points (if any) are typically stored in separate contiguous arrays, often immediately following their respective weight blocks in memory, or referenced via pointers.

**Scale Storage:** Scales are `f16` and are stored contiguously in their own array. For GPTQ-128, a scale applies to 128 columns. If `K` is also grouped (e.g., `K_group_size=128`), then scales would be stored as `[N/128][K/128]` dimensions. Marlin usually stores scales as `f16` because this is the target precision for dequantization and multiplication.

### 0b. Marlin's Async Global Loads with Tensor Core Math

Marlin's pipeline is a classic example of hiding memory latency using `cp.async` and instruction-level parallelism.

1.  **`cp.async` Pipeline Depth:** Typically 2-4 stages. A common setup uses a `double-buffered` approach for shared memory, with two stages: `cp.async` for current tile, `mma.sync` for previous tile. More advanced pipelines can have 3-4 stages, involving multiple shared memory buffers. For Ada Lovelace (SM 8.9), `cp.async` is available and crucial.
    *   **Stage 1 (Async Load):** `cp.async.ca.shared.global` for A-matrix (activations), `cp.async.cs.shared.global` for B-matrix (weights) and scales. `cs` (streaming) hint for weights prevents L2 pollution, `ca` (caching) hint for activations ensures L2 reuse.
    *   **Stage 2 (Dequant & MMA):** While Stage 1 fetches the next tile, the current tile (already in shared memory) is dequantized on-the-fly and fed to Tensor Cores using `mma.sync`.
    *   **Stage 3 (Write Back):** Accumulators are written back to global memory or prepared for next stage.

2.  **`lop3` Bitwise PTX for INT4â†’FP16 Dequantization:** This is where Marlin shines. Instead of generic integer operations, `lop3` (ternary logic operation) is used for efficient nibble extraction and conversion.
    *   `lop3` allows arbitrary 3-input bitwise operations in a single cycle.
    *   **Nibble extraction:** From a packed byte `B = (N_high << 4) | N_low`:
        *   `N_high = (B >> 4) & 0xF`
        *   `N_low = B & 0xF`
        *   These can be done with `lop3`. For example, `lop3 R_low, B, 0xF, 0, 0xAA, AND` (where `0xAA` mask extracts the low 4 bits effectively). Or more directly `prmt` (permute) instruction can extract nibbles.
    *   **Signed Conversion:** INT4 values (0-15) are often symmetric, meaning they represent values from -8 to +7. This requires subtracting 8: `val = packed_nibble - 8;`. This subtraction is fused.
    *   **Dequantization to FP16:** `dequant_val = (packed_nibble - zero_point) * scale;`
        *   `packed_nibble` is an unsigned 4-bit integer.
        *   `zero_point` is often 8 for symmetric quantization.
        *   `scale` is an `f16`.
        *   The sequence is: Load packed byte -> extract two 4-bit nibbles (e.g., using `prmt` or `lop3` logic) -> convert to `s8` (signed 8-bit, by subtracting 8) -> convert `s8` to `f16` -> multiply by `f16` scale. These `s8` values are then used in `mma.sync.s8.f16`. Ada (SM 8.9) supports `mma.sync.m8n8k16.s8.f16.f16` (INT8 B-matrix, FP16 accumulator).

3.  **Saturation of ALUs and Tensor Cores:** Marlin achieves this by:
    *   **Coarse-grained parallelism:** Many thread blocks concurrently process different parts of the output matrix.
    *   **Fine-grained parallelism:** Within a thread block, warps are assigned to output tiles. Each warp processes its own `M_warp x N_warp` output tile.
    *   **Warp-level scheduling:** Warps `cp.async` their data, then concurrently dequantize (using general-purpose ALUs) and perform `mma.sync` (Tensor Cores). The dequantization work (low instruction count per element) is carefully scheduled to overlap with `cp.async` latency and `mma.sync` operations. The `lop3` instructions are fast and don't significantly stall the ALU pipeline.
    *   **Register-blocking:** Dequantized values are often kept in registers for direct feeding to `mma.sync` instructions, minimizing shared memory traffic.

4.  **Register Budget:** Marlin kernels are aggressive about register usage.
    *   Minimal intermediate values for dequantization (direct `lop3`/`prmt` to `f16` or `s8` registers).
    *   Reusing registers for different stages of the pipeline.
    *   Careful selection of warp-tile sizes to balance occupancy with register pressure. Typical tile sizes (e.g., `M=128, N=64, K=128`) and warp sizes (e.g., `M_warp=64, N_warp=32, K_warp=64`) are chosen to keep register usage within limits for high occupancy. Register spilling to local memory (global memory) is a performance killer and strictly avoided. Ada has 64K registers per SM.

### 0c. Marlin's Cache Eviction Strategy

Marlin employs a strict cache eviction policy to prevent L2 pollution, especially for the weight matrix (`B`).

*   **`cp.async.cs` for Weights (B-matrix):** The `cp.async` instruction includes cache hints. Marlin typically uses `cp.async.cs` (cache streaming) for loading weights. This hint tells the L1/L2 cache to treat the data as "streaming," meaning it's unlikely to be reused. This minimizes pollution of the L2 cache, which has a limited capacity (72 MB on 4090). Weights are typically larger than L2 and accessed sequentially, so they are not good candidates for caching.
*   **`cp.async.ca` for Activations (A-matrix):** The input activation vector (`A` matrix, `M=1` row vector) often benefits significantly from L2 caching because different weight matrices (layers) will read the same activation vector. Thus, `cp.async.ca` (cache as usual) or `cp.async.cg` (cache global) is used for activations, letting the hardware cache them.
*   **Explicit Cache Control PTX:** Older PTX instructions like `ld.global.cg` (cache global) or `ld.global.cs` (cache streaming) can also be used but `cp.async` provides more pipeline control.

This strategy ensures that the valuable L2 cache is primarily used for the A-matrix and other frequently reused data, while the B-matrix (weights) streams directly through.

### 0d. Marlin's Performance Profile with Batch Size

*   **At M=1 (pure decode):** Marlin is specifically designed for M=1. It can achieve **60-80% of peak memory bandwidth** on RTX 4090. This translates to roughly 70-90% of peak FP16 inference speed on a memory-bound workload. The specialized layout, on-the-fly dequant, and async pipeline are highly efficient here.
    *   For Qwen 7B Q4, if theoretical memory bandwidth is 1008 GB/s (4090), and each FP16 element is 2 bytes, 1008 GB/s / 2 B/elem = 504 Giga-elem/s. A Q4 weight matrix means 0.5 bytes/elem for data + scales.
    *   The "4x over FP16 at M=1" claim for Marlin often refers to a throughput comparison with FP16 *naive* GEMM (e.g., a non-optimized one, or one that dequantizes entirely to FP16 first). Relative to a highly optimized FP16 kernel (like cuBLAS HGEMM), Marlin provides significant speedup primarily by reducing memory traffic for weights by 4x.
    *   Considering 7B Q4, ~4.2 GB weights. If each token requires reading weights once, this is 4.2 GB / tok. At 60% of 1008 GB/s = 604.8 GB/s. Then 604.8 GB/s / 4.2 GB/tok = ~144 tok/sec (rough upper bound). Actual FLOPs for Qwen 7B are ~140 GFLOPs. Q4 implies ~35 GFLOPS of actual compute, but the dequant adds ALU work. It's truly memory bound.
    *   A target of **60-80% bandwidth utilization** for Qwen 7B (which is memory bound) implies **60-96 tok/sec**. This is aligned with realistic expectations.

*   **Batch Size Scaling:**
    *   Marlin's advantage diminishes as batch size `M` increases. When `M` becomes large (e.g., M=32+), the problem becomes compute-bound rather than memory-bound, as the `A` matrix (activations) also becomes sizable and memory traffic increases.
    *   At larger batch sizes, the overheads of Marlin's specific weight layout and per-thread dequantization become less critical compared to the raw compute throughput of the Tensor Cores.
    *   For `M > 32` or so, a well-optimized `FP16 cuBLAS HGEMM` operating on *pre-dequantized* FP16 weights in global memory (or shared memory after a fast transfer) can often outperform Marlin-style kernels. This is because `cuBLAS` is highly optimized for large dense GEMMs and can saturate Tensor Cores very effectively. The "break-even" point varies by architecture and model shape.

*   **ExLlamaV2's "dequant to temp buffer + cuBLAS HGEMM":**
    *   **M=1:** This approach performs poorly at M=1 (TensorWarp's observed 0.3 tok/sec for FP16 dequant+cuBLAS) because:
        1.  **Memory Traffic:** It first reads 4-bit weights, dequantizes them into a full FP16 temporary buffer (4x data expansion), writes this to global memory, then reads the FP16 weights back for `cuBLAS`. This is significantly more memory bandwidth usage than Marlin's on-the-fly dequant.
        2.  **Kernel Launch Overhead:** Two kernel launches (dequant + cuBLAS) add overhead.
    *   **M=32+:** For larger batch sizes, if the FP16 temporary buffer can be reused or is small enough to fit in fast memory, and the compute time dominates the dequantization and memory transfer time, this approach can become competitive. ExLlamaV2 (at 32.5 tok/sec on 7B, often in larger batches) might be using a combination of techniques, but its pure "dequant to temp + cuBLAS" strategy is typically not competitive for M=1. Its custom kernels are more Marlin-like.

## 1. Adaptive Kernel Generation Across GPU Architectures

### 1a. Architectural Differences Affecting INT4 GEMM

| Feature                    | Turing (SM 7.5)      | Ampere (SM 8.0/8.6)         | Ada Lovelace (SM 8.9)          | Hopper (SM 9.0)                | Blackwell (SM 10.0)             |
| :------------------------- | :------------------- | :-------------------------- | :----------------------------- | :----------------------------- | :------------------------------ |
| **Tensor Cores**           | FP16, INT8 (WMMA)    | FP16, INT8 (MMA.SYNC)       | FP16, INT8 (MMA.SYNC)          | FP8, FP16, INT8 (WGMMA)        | INT4, FP8, FP16, INT8 (WGMMA)   |
| **INT4 Native?**           | No (use INT8 trick)  | No (use INT8 trick)         | No (use INT8 trick)            | No (use INT8 trick)            | **YES**                         |
| **Async Copy**             | No                   | `cp.async` (L2/GM to SMEM)  | `cp.async`                     | TMA (WGMMA, more flexible)     | TMA v2                          |
| **Shared Memory (SMEM)**   | 48KB/96KB            | 100KB/164KB                 | 48KB/96KB (per SM), configurable | 228KB (per SM)                 | Increased                       |
| **L2 Cache**               | 6MB                  | 40-42MB                     | 72MB                           | 50MB (smaller, but faster)     | Increased                       |
| **Registers per SM**       | 64K                  | 64K                         | 64K                            | 64K                            | Increased                       |
| **Recommended Config**     | Use `wmma`+INT8      | `mma.sync`+INT8 double pack | `mma.sync`+INT8 double pack    | `wgmma`+TMA (FP8 for B)        | **WGMMA+TMA+Native INT4**       |

**Key takeaways:**
*   **INT4 on Ada (SM 8.9):** Ada Lovelace does *not* natively support INT4 Tensor Core instructions. It must use the INT8 double-packing trick. This involves dequantizing 4-bit values to signed 8-bit, then potentially packing two `s8` values into an `s16` or using two `s8` MMA operations. Or, more commonly, the dequantized `s8` is converted to `f16` and then `mma.sync.s8.f16.f16` is used.
*   **Async Copy:** `cp.async` (Ampere+) is essential for hiding latency. Hopper's TMA (Tensor Memory Accelerator) is even more powerful for `wgmma` ops.
*   **Shared Memory:** Larger shared memory enables larger tile sizes, which can reduce global memory accesses and improve occupancy, but also increases register pressure.
*   **Blackwell (SM 10.0):** Native INT4 support is a game-changer, simplifying dequantization and potentially offering direct input to Tensor Cores, possibly eliminating the INT8 trick.

### 1b. Multi-Architecture Kernel Dispatch in Production Systems

*   **TensorRT-LLM and vLLM:**
    *   They heavily leverage **CUTLASS** (CUDA Template Library for GEMM). CUTLASS provides highly optimized, templated GEMM kernels.
    *   They typically **ship pre-compiled libraries** (e.g., `.so` files with embedded PTX/SASS) for common architectures (sm\_80, sm\_86, sm\_89, sm\_90). This avoids runtime compilation overhead.
    *   **Runtime Dispatch:** At runtime, they query the GPU's `cudaDeviceProp` (or similar) to get the `sm_major.sm_minor` version and dispatch to the appropriate pre-compiled kernel variant.
    *   **Tile Size Selection:** CUTLASS uses `__host__ __device__` code that selects tile sizes, pipeline depths, and instruction sequences based on `ARCH` compile definitions and template parameters.
    *   They often use custom kernels written in raw CUDA C++ (or PTX) for specific quantized formats that aren't natively supported by CUTLASS, or to push performance beyond CUTLASS's generalization.

*   **CUTLASS Approach:**
    *   **CUTLASS 2.x:** Focused on `wmma` (Turing) and `mma.sync` (Ampere+) for FP16/INT8. It provided a rich set of templates for constructing GEMM kernels.
    *   **CUTLASS 3.x:** A major redesign, adding native support for Hopper's `wgmma` (warp-group matrix multiply-accumulate), TMA, and the new memory hierarchy. It's more modular and extensible for new hardware features. It emphasizes `Epilogue` operations for fusing activations and other post-GEMM operations.
    *   **Architecture-adaptive:** CUTLASS kernels are highly parameterized. You provide template arguments for `Arch`, `DataType`, `TileShape`, `Layout`, `Operator`, `Epilogue`, etc. During compilation, the C++ template metaprogramming generates the specific kernel code.

### 1c. Minimum Viable "Kernel Template" System for TensorWarp

TensorWarp's NVRTC-based runtime compilation pipeline is well-suited for this.

*   **Kernel Template System:**
    *   **Parameterized CUDA C Templates:** Store CUDA C++ kernel code as strings (or Rust `include_str!`). These templates will use C++ template parameters and `ifdef` blocks to adapt to architecture and shape.
    *   **NVRTC Compilation:** At load time (or first use), NVRTC compiles these templates into PTX for the specific GPU's `sm_XX` architecture.
    *   **Parameters:**
        *   **Architecture-Specific:** `SM_MAJOR`, `SM_MINOR`, `ASYNC_COPY_AVAILABLE` (boolean), `TENSOR_CORE_INT4_NATIVE` (boolean), `SHARED_MEM_CONFIG` (enum/int). These enable/disable `cp.async`, select `mma.sync` vs `wmma` vs `wgmma`, and tune shared memory usage.
        *   **Shape-Specific:** `M`, `N`, `K`, `GROUP_SIZE_N`, `GROUP_SIZE_K`, `TILE_M`, `TILE_N`, `TILE_K`, `BLOCK_THREADS` (thread block dimensions), `WARP_TILE_M`, `WARP_TILE_N`, `WARP_TILE_K`, `NUM_STAGES_ASYNC`. These determine how the work is split and processed.
        *   **Format-Specific:** `IS_ASYMMETRIC` (boolean, for zero-point handling), `HAS_TWO_TIER_SCALING` (boolean), `LOW_NIBBLE_FIRST` (boolean). These affect dequantization logic.

*   **TensorWarp's Autotuner Integration:**
    *   The existing Thompson sampling tuner can be extended. For a given `(M, N, K, format, SM)` tuple:
        1.  The system generates a set of candidate kernel configurations (different `TILE_M/N/K`, `BLOCK_THREADS`, `NUM_STAGES_ASYNC`).
        2.  For each config, it instantiates the template, compiles with NVRTC (`-arch=sm_X.Y -code=sm_X.Y`), and profiles a few executions.
        3.  The autotuner updates its model based on performance and selects the best.
        4.  Compiled kernels are cached to disk (keyed by `(M, N, K, format, SM, HASH_OF_KERNEL_PARAMS)`).

## 2. Quantization Format Universality

### 2a. Byte-Level Differences Between Quantization Formats

| Format            | Scale Storage             | Zero-Point Handling       | Two-Tier Scaling | Bit Packing Order | Dequant Cost (ALU Ops)                                         |
| :---------------- | :------------------------ | :------------------------ | :--------------- | :---------------- | :------------------------------------------------------------- |
| **Q4_0** (GGML)   | `f32` per 32 elements     | Symmetric (implicit ZP=8) | No               | Low-nibble first  | `(nibble - 8) * scale`                                         |
| **Q4_K** (GGML)   | `f16` per 256 (super-blk) `f16` per 32 (block) | Symmetric (implicit ZP=8) | Yes              | Low-nibble first  | `(nibble - 8) * block_scale * super_block_scale`               |
| **GPTQ-128**      | `f16` per 128 elements    | `f16` per 128 elements (asymmetric) | No               | High-nibble first | `(nibble - zp) * scale`                                        |
| **AWQ**           | `f16` per channel (row)   | `f16` per channel (asymmetric or symmetric) | No               | High-nibble first | `(nibble - zp) * scale`                                        |
| **GGUF Q4_1**     | `f32` per 32 elements     | Asymmetric (explicit ZP)  | No               | Low-nibble first  | `(nibble - zp) * scale`                                        |
| **GGUF Q4_K_M/S** | Same as Q4_K              | Same as Q4_K              | Yes              | Low-nibble first  | Same as Q4_K                                                   |

**Notes:**
*   **Scale/ZP Storage:** `f32` vs `f16` scales impact memory footprint and load time. `f16` is preferred.
*   **Zero-point handling:** Asymmetric quantization (`(nibble - zp) * scale`) adds an extra `f16` subtraction compared to symmetric (`(nibble - 8) * scale`). `zp` could be integer (0-15) or float.
*   **Two-tier scaling:** Q4_K adds another `f16` multiplication, increasing ALU cost.
*   **Bit packing order:** "Low-nibble first" means `packed_byte = (high_nibble << 4) | low_nibble`. "High-nibble first" means `packed_byte = (low_nibble << 4) | high_nibble`. This determines the masks/shifts needed for extraction or if `prmt` is used.

### 2b. Single Kernel Template for Multiple Quantization Formats

Yes, a **single kernel template can handle multiple quantization formats through parameterization**.

*   **Parameterization:**
    *   `QUANT_FORMAT_ENUM`: An enum (e.g., `Q4_0`, `GPTQ_128`, `Q4_K`) passed as a template argument.
    *   `HAS_ZERO_POINT`: Boolean.
    *   `IS_SYMMETRIC`: Boolean (if true, zero point is implicit 8).
    *   `HAS_TWO_TIER_SCALING`: Boolean.
    *   `SCALE_DTYPE`: `half` or `float`.
    *   `PACKING_ORDER`: Enum for high/low nibble first.
*   **Conditional Code Generation:** Inside the kernel template, `if constexpr` (C++17+) or `#if` preprocessor directives can switch the dequantization logic based on these parameters.
    *   Example:
        ```cpp
        #if QUANT_FORMAT_ENUM == GPTQ_128
            // Load f16 zero_point
            // Dequant: (nibble - zp_f16) * scale_f16
        #elif QUANT_FORMAT_ENUM == Q4_0
            // Dequant: (nibble - 8.0f16) * scale_f32 (or f16 if converted)
        #endif
        ```
*   **Performance Costs:**
    *   **Function Pointers/Virtual Calls:** If done at runtime via function pointers or virtual calls, it would incur measurable overhead. **Avoid this.**
    *   **Template Branching (`if constexpr`):** Compilers optimize this away, generating only the relevant code path. **No runtime overhead.**
    *   **Preprocessor Directives (`#if`):** This is compile-time selection, generating distinct kernels for each configuration. **No runtime overhead.**
    *   The "cost" is increased kernel code size (more compiled variants) and potentially longer compile times if many permutations are requested.

*   **Llama.cpp's GGML Backend:** Llama.cpp implements distinct dequantization functions (e.g., `ggml_vec_dot_q4_0_q8_0`) for each format. These functions are often specialized CPU intrinsics. For GPU, they have dedicated kernels per format variant. This implies that while a single *template* can generate them, they are distinct at the executable level.

*   **Canonical Internal Format:** Yes, this is highly recommended for TensorWarp.
    *   **External Formats:** SafeTensors, GGUF, GPTQ checkpoint.
    *   **Convert to TensorWarp Internal Format:** This format should be Marlin-style reordered for `cp.async` and Tensor Core consumption, using `f16` scales and zero-points. This conversion is a one-time process.
    *   **Standardized Structure:**
        *   Packed INT4 nibbles.
        *   `f16` scales (per group `GROUP_SIZE_N` and `GROUP_SIZE_K_SCALES`).
        *   `f16` zero-points (if asymmetric, same grouping as scales).
        *   A metadata header indicating `GROUP_SIZE_N`, `GROUP_SIZE_K_SCALES`, `IS_ASYMMETRIC`, `PACKING_ORDER`.
    *   This "canonical" format should be the target for the offline weight processor.

### 2c. TensorWarp's Offline Weight Processing Pipeline

1.  **Load:** Input from any format (SafeTensors, GGUF, GPTQ checkpoint, raw PyTorch/HF). Parse relevant quantization parameters (group sizes, scales, zero-points, original FP16 weights if needed for re-quantization).
2.  **Canonicalization:**
    *   **If already quantized:** Extract INT4 data, scales, zero-points. Convert scales/zero-points to `f16`.
    *   **If FP16/FP32:** Perform desired quantization (e.g., to GPTQ-like Q4_128). This step can involve AWQ or GPTQ calibration.
3.  **Reorder to Marlin Internal Format:** Apply the Marlin-style reordering algorithm (as described in 0a) for the packed INT4 data, `f16` scales, and `f16` zero-points.
    *   This reordering can be architecture-agnostic for the base Marlin layout (e.g., N-group, K-group tiling), but the innermost packing might have minor SM-specific variations (e.g., specific padding for optimal `cp.async` vector width). For now, a single Marlin layout optimized for SM 8.9 (Ada) is a good start.
4.  **Store Cached:** Save the `TensorWarp.Marlin` format to disk. Key it by model hash, quantization config, and potentially GPU architecture for future reuse.

**Marlin's offline pre-processing cost:** It scales linearly with model size. For a 7B model (~4.2GB Q4 weights), it might take a few minutes on a CPU. On a GPU, it can be much faster. It's perfectly feasible to do at every cold start if not cached, but caching is preferred.

## 3. Model Architecture Adaptation

### 3a. Model Architectural Features and Quantized GEMM

*   **Different Hidden Sizes (e.g., 896, 3584):**
    *   **Tile Sizes:** Tensor Core operations thrive on power-of-2 dimensions (e.g., 128x128, 64x64). Non-power-of-2 dimensions in M, N, or K can lead to "tail" computations that are less efficient.
    *   **Handling:** The kernel must handle non-power-of-2 dimensions gracefully. This often means padding the logical matrix to the nearest multiple of the tile size, performing the GEMM, and then masking or ignoring results in the padded regions. Or, have specific "tail kernels" for the remaining rows/columns, which might fall back to `f16` FPUs or even cuBLAS.
    *   **Impact:** Performance might drop slightly when dimensions are not multiples of common tile sizes (e.g., 64, 128, 256).

*   **GQA Ratios (KV Projection Dimension):**
    *   For attention mechanisms, the `Q` matrix is `[M, K_Q]`, `K`/`V` matrices are `[M, K_KV]`. `K_Q` and `K_KV` are usually `hidden_size / num_heads` and `hidden_size / num_kv_heads`. In GQA, `num_kv_heads < num_heads`, so `K_KV` is smaller.
    *   **GEMM Efficiency:** Smaller `N` dimensions (like `K_KV`) reduce the total work for a GEMM. Below a certain `N` threshold (e.g., N < 64 or 32), the overhead of a highly complex Tensor Core kernel (async copies, scheduling) might outweigh the benefits.
    *   **Minimum N:** For very small `N`, a simple element-wise dequantization to FP16 followed by a `cuBLAS HGEMM` can sometimes be faster due to lower kernel complexity and launch overhead. The adaptive system should identify this threshold. `M=1, N=128, K=4096` is a good candidate for Marlin. `M=1, N=32, K=4096` might need a simpler strategy.

*   **Biases vs No Biases:**
    *   **Adding Bias:** Biases are simply an element-wise `C = C + Bias` operation.
    *   **Fusion:** This can be perfectly fused into the GEMM kernel's epilogue. After the `mma.sync` results are accumulated, the bias vector is loaded (or broadcasted from shared memory) and added element-wise before writing to global memory.
    *   **Impact:** Minimal performance impact unless the epilogue becomes heavily register-constrained. Often negligible.

### 3b. Mixture-of-Experts (MoE) Architectures

MoE presents significant challenges for quantized GEMM.

*   **Sparse GEMM:** When only `top_k` experts (e.g., 2 of 8) are active, the overall operation is sparse. Applying a dense quantized GEMM to all experts is wasteful.
*   **Approaches:**
    1.  **Gather & Batch:** Gather activations for all `top_k` active experts across the batch, then perform a batched dense GEMM on the `top_k` expert weights. This requires dynamic batching and potentially padding for different expert shapes.
    2.  **Specialized Sparse Kernels:** Custom sparse GEMM kernels that only load and compute on the weights of active experts. This is highly complex for quantization.
    3.  **Persistent Kernels:** MoE often benefits from persistent kernels that keep expert weights in GPU memory and schedule work dynamically. For quantized weights, this means they would reside in Marlin's reordered format.
*   **Quantized Weight Loading:** Each expert has its own quantized weights. The router determines which weights to load. This fits well with Marlin's tile-based loading if experts are large enough.
*   **Current State:** Most production MoE systems (e.g., vLLM) use dense FP16/BF16 GEMMs for active experts, sometimes with highly optimized sparse attention. Quantized MoE is an active area of research.
*   **Recommendation for TensorWarp:** MoE is a complex, advanced feature. **Postpone until dense quantized GEMM is robust.** Start with the "gather & batch" approach, running multiple Marlin-style GEMMs for active experts sequentially or in a batched manner (if M>1).

### 3c. Activation Fusion Opportunities

*   **Fusion:** Yes, the dequant + GEMM + activation can (and should) be fused into a single kernel for M=1.
    *   The most common pattern is `Y = Activation(GEMM(A, B) + Bias)`.
    *   The `Activation` function (e.g., SiLU, SwiGLU, GELU) is applied in the kernel's epilogue, after the `mma.sync` results are accumulated and optionally biased.
*   **SiLU/SwiGLU, GELU, GeGLU:**
    *   **SiLU (x * sigmoid(x)):** Relatively cheap. `sigmoid` can be approximated or computed with `__expf`, `__rcp_f32`.
    *   **SwiGLU (SiLU(x) * y):** Requires two input vectors or splitting the weight matrix. Often implemented as two GEMMs (one for `x`, one for `y`) then element-wise operations.
    *   **GELU (Gaussian Error Linear Unit):** More complex, often uses `tanh` approximations.
    *   **GeGLU:** Similar to SwiGLU but with `GELU(x) * y`.
*   **Register Pressure Cost:** Fusing activation functions into the epilogue increases register pressure because intermediate values for activation computation need registers. This can reduce occupancy. However, the gains from avoiding global memory writes and reads for the intermediate activation results usually outweigh the occupancy hit for M=1. Careful register allocation is key.

## 4. Compiler and Runtime Design

### 4a. TensorWarp's Tiered Compilation Pipeline for Quantized GEMM

This tiered approach is excellent for balancing cold-start latency with peak performance.

*   **Tier 0 (Instant - Baseline):**
    *   **Mechanism:** For each GEMM (layer), load a slice of quantized weights, dequantize fully to `f16` in a temporary global memory buffer, then invoke `cuBLAS HGEMM` (FP16).
    *   **Pros:** Always works, reasonable fallback speed, no runtime compilation.
    *   **Cons:** Very slow for M=1 (as observed: 0.3 tok/sec) due to memory traffic.
    *   **Cold-start latency:** Near zero.
*   **Tier 1 (Fast - Pre-compiled/Cached):**
    *   **Mechanism:** Check a disk cache for a pre-compiled Marlin-style kernel matching `(M, N, K, format, SM, TILE_CONFIG_HASH)`. If found, load it. If not found, immediately dispatch to Tier 0.
    *   **Pros:** Fast warm-up, good performance for common shapes.
    *   **Cons:** Requires pre-compilation or a populated cache.
    *   **Cold-start latency:** Low (disk load + `cuModuleLoadData`).
*   **Tier 2 (Tuned - Autotuned NVRTC):**
    *   **Mechanism:** If Tier 1 fails, instantiate the Marlin-style CUDA C++ template with a *default set of robust tile parameters* for the current `(M, N, K, format, SM)` tuple. Compile with NVRTC. While this "default" kernel runs, start a background autotuning process (Thompson sampling) to find optimal tile parameters. Once a better kernel is found, compile it and switch. Cache all tuned kernels.
    *   **Pros:** Automatically adapts to specific shapes/architectures, achieves good performance.
    *   **Cons:** First-use compilation latency (seconds to tens of seconds per layer).
    *   **Cold-start latency:** Moderate to high initially, amortized over time.
*   **Tier 3 (Optimal - Full Marlin-style NVRTC):**
    *   **Mechanism:** This is what the Tier 2 background autotuner produces. Fully optimized kernel for specific `(M, N, K, format, SM)` and model family, using the Marlin-style layout and pipeline.
    *   **Pros:** Peak performance.
    *   **Cons:** Full NVRTC compilation.
    *   **Cold-start latency:** Not applicable as it's an output of Tier 2.

### 4b. Abstraction Boundary Between Compiler and Runtime

*   **Compile-time Decisions:**
    *   **Weight Layout:** Marlin reordering (target architecture-specific if needed).
    *   **Kernel Code Generation:** Specific `sm_XX` version, `cp.async` enabled/disabled, Tensor Core instruction set (`mma.sync` vs `wgmma`), specific dequantization logic (`HAS_ZERO_POINT`, `TWO_TIER_SCALING`), warp/block tile sizes, shared memory configuration.
    *   **Kernel Configuration Parameters:** These are baked into the compiled PTX/SASS.
*   **Runtime Decisions:**
    *   **Batch Size Routing:** Which kernel variant (if batch-size-specific kernels exist).
    *   **KV Cache Length:** Dynamic memory allocation, not affecting GEMM kernels directly.
    *   **GPU Architecture Detection:** Selecting the correct pre-compiled kernel.
    *   **Kernel Dispatch:** Which `cudaLaunchKernel` to call.
    *   **Memory Allocation:** Scratchpads, KV cache, activations.

*   **vLLM's Model Runner:** vLLM typically pre-compiles kernels (often via CUTLASS) for known architectures. It performs shape analysis at model load time to select suitable kernels from a registry or generates specific ones if not available. It prioritizes performance and caches compiled artifacts. The actual model execution involves dynamic batching and efficient memory management (PagedAttention). For quantized models, it maps the quantized weights to its internal `BlockTensor` system.

*   **Kernel Caching:** **Absolutely, TensorWarp should cache compiled kernels to disk.**
    *   **Keying:** `(model_hash, model_arch_hash, gpu_arch_major, gpu_arch_minor, layer_idx, weight_name, M, N, K, quantization_format_hash, kernel_config_hash)`. The `kernel_config_hash` uniquely identifies the chosen tile sizes, async pipeline depth, etc.
    *   This dramatically reduces cold-start latency after the first run.

### 4c. Profiling and Selection Between Kernel Variants

*   **Number of Variants:** For a given `(M, N, K, format, architecture)` tuple, there could be dozens of kernel configurations (tile sizes, thread counts, pipeline depths) to explore. For common shapes, a few highly optimized variants will likely cover 80% of performance.
*   **Profiling Cost:**
    *   **Initial Autotuning:** Each kernel candidate requires `N` runs (e.g., 5-10) to get a stable average, plus `cudaEvent` recording and host-side synchronization. This can add up. A full autotuning run for a whole model can take minutes to hours.
    *   **Amortization:** In the decode loop, the autotuning runs in the background. The performance for the first few tokens might not be optimal, but subsequent tokens benefit. The cost is amortized across potentially millions of inference requests.
*   **TensorRT-LLM's Algorithm Selection:** TensorRT-LLM uses an internal heuristics engine and potentially offline profiling data to select the best algorithm (`cublasLt` or custom kernels) for a given GEMM. It takes into account GPU architecture, input shapes, data types, and available resources. It aims for a balance between compile time and runtime performance.

## 5. Memory Management for Quantized Inference

### 5a. Optimal VRAM Budget Allocation for 7B on 24 GB (RTX 4090)

*   **Qwen 7B Q4_0:**
    *   **Weights (Q4_0):** ~4.2 GB (7B params * 0.5 bytes/param).
    *   **KV Cache:** Grows with context length. For 7B, context length 4096, 32 layers, 32 heads, 128 dim/head, 2 bytes/token (FP16): `2 * L * 32 * 32 * 128 * 2 = ~1.3 GB` for `L=4096`. For `L=8192`, it's `~2.6 GB`.
    *   **Activations (FP16):** Input `[1, K]` and output `[1, N]` activations for each layer are typically small (e.g., `1 * 4096 * 2 bytes = 8KB`). Layer-to-layer activations (e.g., FFN intermediate states) might be larger but are temporary. Total for full model inference (not just GEMM): a few hundred MB.
    *   **CUDA Context / Driver / Runtime:** ~200-500 MB.
    *   **De-quant Temp Buffer:**
        *   **If per-GEMM dequant + cuBLAS (Tier 0):** `K * N * sizeof(f16)`. For a `4096x4096` matrix: `4096 * 4096 * 2 bytes = 32 MB`. If dequantized per layer, `~32 MB` per layer. If this temp buffer is reused (double-buffered) across layers, only 1-2 such buffers are needed. Total: `~64 MB` max.
        *   **Marlin-style (on-the-fly dequant):** No large dequant temp buffer in global memory needed. Dequantization happens to registers/shared memory. This is a key memory-saving advantage.
    *   **Total Estimate for Marlin (L=4096):** 4.2 GB (weights) + 1.3 GB (KV) + 0.5 GB (activations/runtime) = **~6 GB**.
    *   **With 24 GB VRAM:** Plenty of headroom for larger context lengths, batch sizes, or even two models.

### 5b. Memory Management for Quantized Weights

*   **Production Systems:**
    *   They typically **discard the original (e.g., `q4_0` or raw `gptq`) weights after converting them to the internal optimized Marlin-style format.** This saves VRAM.
    *   The internal format becomes the "active" format for inference.
    *   **Model-Parallel/Tensor-Parallel:** For sharding, each GPU gets its slice of the *Marlin-format* weights. The reordering would be applied to the full matrix, then sliced, or applied per-slice.

### 5c. Memory Overhead of Marlin's Pre-shuffled Format

*   **Padding/Alignment:** Marlin's format might introduce minimal padding bytes (e.g., to align blocks to 128-byte cache lines or warp sizes) but this is usually negligible (a few percent) compared to the overall weight size.
*   **Scales & Zero-Points:**
    *   For GPTQ-128: `(K*N / 128)` scales and zero-points. Each `f16`.
    *   Overhead: `(K*N / 128) * (2 bytes/scale + 2 bytes/zp)`.
    *   For 7B model (7 billion elements), `7,000,000,000 / 128 * 4 bytes = ~218 MB`. This is roughly 5% of the 4.2GB Q4 weight data, which is acceptable.
*   **Overall:** The memory overhead of Marlin's internal format is small, primarily due to scales/zero-points and minimal padding. The goal is to reduce memory *traffic* during inference.

## 6. PTX-Level Implementation Details

### 6a. Exact PTX Instructions for INT4 Dequantization (Ada SM 8.9)

On Ada Lovelace (SM 8.9), the typical approach is to dequantize INT4 to INT8 (signed) and then feed to `mma.sync.m8n8k16.s8.f16.f16` (if the B matrix accepts INT8) or convert to FP16. Given `f16` scales, converting to `f16` before multiplication is often preferred.

Assume `packed_byte` contains `[high_nibble | low_nibble]`.
Assume `scale_val_f16` (FP16) and `zero_point_f16` (FP16, often `8.0f16`).

```ptx
// Assume R_packed_byte contains the packed 8-bit integer (e.g., 0xAB for nibbles 10 and 11)
// We need to extract two 4-bit nibbles and convert them to f16.

// 1. Extract high nibble (bits 7-4)
//    prmt.b32 R_high_nibble, R_packed_byte, 0, 0x7654; // Extracts upper nibble into lowest bits, other bits zeroed. More generic.
// OR simpler using logical ops (example):
shf.r.s16 R_high_nibble_shifted, 0, 4, R_packed_byte; // Shift right by 4 to get high nibble (now 0-15)
and.b32   R_high_nibble, R_high_nibble_shifted, 0xF;  // Mask to ensure only 4 bits are kept (0-15)

// 2. Extract low nibble (bits 3-0)
and.b32   R_low_nibble, R_packed_byte, 0xF; // Mask to get low nibble (0-15)

// Now we have two registers, R_high_nibble and R_low_nibble, each containing an unsigned 4-bit integer (0-15).

// 3. Convert unsigned 4-bit (0-15) to signed 8-bit (-8 to 7)
//    (This is for symmetric quantization. For asymmetric, subtract zero_point)
sub.s32   R_high_s8, R_high_nibble, 8;
sub.s32   R_low_s8, R_low_nibble, 8;

// Now R_high_s8 and R_low_s8 contain signed 8-bit integers (-8 to 7).
// These could be used directly with mma.sync.s8.f16 if the B matrix is declared s8.

// 4. Convert signed 8-bit to FP16
//    Using __ushort_as_half (which maps to cvt.f16.s32.ftz on PTX)
//    NOTE: Direct conversion from S8 to F16 needs careful PTX usage.
//    Typically, the S8 values are expanded to S32, then converted to F16.
cvt.f16.s32 R_high_f16_val, R_high_s8;
cvt.f16.s32 R_low_f16_val, R_low_s8;

// 5. Multiply by FP16 Scale and subtract zero_point if asymmetric (if scale_val_f16 and zero_point_f16 are loaded)
// Assuming scale_val_f16 and zero_point_f16 are loaded into registers (e.g., R_scale_f16, R_zp_f16)
// For symmetric (ZP=8 handled in step 3):
fma.rn.f16 R_high_dequant_f16, R_high_f16_val, R_scale_f16, R_neg_zero; // R_neg_zero = 0.0f16 or actual ZP.
fma.rn.f16 R_low_dequant_f16, R_low_f16_val, R_scale_f16, R_neg_zero;

// If asymmetric quantization (zp is separate from scale, not fused into nibble-to-signed conversion):
// fma.rn.f16 R_high_dequant_f16, R_high_f16_val, R_scale_f16, R_neg_zero_point_f16;
// Where R_neg_zero_point_f16 contains -zero_point_f16.
```

**`lop3` alternative for combined extraction and masking:**
`lop3.b32 d, a, b, c, op` can compute any boolean function of three inputs.
Example for extracting low nibble: `lop3.b32 R_low_nibble, R_packed_byte, R_mask_0xF, 0, 0x88;` (equivalent to `and R_low_nibble, R_packed_byte, 0xF`). `0x88` is the truth table for `(A & B) | (~A & C)` which effectively becomes `A & B` if `C` is 0.
`prmt` is often more direct for permuting bytes/nibbles.
`prmt.b32 R_nibbles, R_packed_byte, R_packed_byte, 0, 0x7531;` can extract 4 nibbles into 4 separate bytes of a 32-bit register.

**Cycle Counts and Register Usage:**
*   Each PTX instruction is typically 1 cycle. The above sequence for *two* nibbles would be roughly 6-8 instructions.
*   Register usage: ~6-8 registers per pair of nibbles being dequantized. Warps operate on multiple elements, so total register pressure for dequant is higher, but these are quickly consumed by `mma.sync`.

### 6b. NVRTC and PTX-Level Intrinsics

*   **Inline PTX (`asm volatile`):** Yes, NVRTC fully supports `asm volatile` in CUDA C++ source. This allows precise control over instruction selection, register allocation, and advanced features not exposed by CUDA C++. This is critical for peak performance in Marlin-style kernels.
    ```c++
    // Example: Manual prmt instruction
    unsigned int packed_byte = ...;
    unsigned int nibbles;
    asm volatile("prmt.b32 %0, %1, %1, 0, 0x7531;" : "=r"(nibbles) : "r"(packed_byte));
    // Now 'nibbles' holds 4 nibbles, each in a byte.
    ```
*   **Compile Flags:**
    *   `--gpu-architecture=compute_89` (or `-arch=sm_89` for NVRTC)
    *   `--gpu-code=sm_89` (or `-code=sm_89` for NVRTC)
    *   `-std=c++17` (or newer if using `if constexpr`)
    *   `-rdc=true` (if using device linking)
    *   `-ftz=true` (flush denorms to zero, for FP performance)
    *   `-fmad=true` (allow fused multiply-add)
    *   `-lineinfo` (for debugging)

*   **TensorWarp's NVRTC Pipeline:**
    1.  **Generate Source:** Create a `String` containing the full CUDA C++ kernel, including `asm volatile` blocks and template parameters `#define`d.
    2.  **Compile:** Call `nvrtcCreateProgram`, `nvrtcCompileProgram` with appropriate `options` (flags).
    3.  **Get PTX:** `nvrtcGetPTX` to get the compiled PTX string.
    4.  **Load Module:** Use `cudaModuleLoadData` (from `cudarc`) with the PTX.
    5.  **Get Function:** `cudaModuleGetFunction` to get the kernel handle.
    6.  **Launch:** `cudaLaunchKernel`.

### 6c. Performance Difference: CUDA C vs PTX Dequantization

*   **CUDA C Dequantization (Bit Shifts, Masks):**
    ```c++
    unsigned char packed_byte = ...;
    short high_nibble_s8 = ((packed_byte >> 4) & 0xF) - 8;
    short low_nibble_s8 = (packed_byte & 0xF) - 8;
    __half high_f16 = (__half)high_nibble_s8;
    __half low_f16 = (__half)low_nibble_s8;
    // Then multiply by scale_f16
    ```
    This translates to multiple `shf`, `and`, `sub`, `cvt.f16.s16` instructions. While compilers are good, they might not always produce the most optimal sequence or utilize specific hardware features like `prmt` as efficiently as explicit PTX.
*   **PTX-Level Dequantization:** Allows direct use of `prmt` (if applicable), `lop3`, and exact control over instruction scheduling. This can reduce instruction count and latency.
*   **ALU Cycles:** A naive CUDA C sequence might take ~10-15 ALU cycles per packed byte (two nibbles). An optimized PTX sequence can cut this to ~6-8 cycles.
*   **Critical Path:** **At M=1, dequantization ALU cost is often *not* on the critical path.** The critical path is overwhelmingly memory latency. The goal of optimized PTX dequantization is to:
    1.  **Reduce pressure on ALUs:** Free up ALU units for other computations or allow more warps to execute concurrently without exhausting ALU resources.
    2.  **Ensure data is ready for Tensor Cores *just in time*:** By minimizing dequantization cycles, the pipeline for feeding data to `mma.sync` is smoother, reducing stalls.
    3.  **Minimize Register Usage:** Finer PTX control can sometimes reduce temporary register needs.

The difference in ALU cycles, while small, contributes to overall efficiency and allows for better overlap, which is crucial for maximizing bandwidth utilization in memory-bound scenarios.

## 7. Benchmarking and Validation

### 7a. Benchmark Methodology

To isolate GEMM kernel performance:

1.  **Single Kernel Launch Measurement:**
    *   **Setup:** Prepare input tensors (A, B, C) in device memory. Initialize them with representative values (e.g., random, or specific ranges to test edge cases).
    *   **Warm-up:** Launch the kernel several times (e.g., 10-20 times) before measurement to warm up caches and ensure the GPU is in a steady-state.
    *   **Timing:** Use `cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime` to precisely measure kernel execution time. Perform `N` runs (e.g., 100-1000) and take the average.
    *   **Isolation:** Measure *only* the kernel execution, not memory transfers to/from host, or kernel launch overhead (except when comparing against cuBLAS, where launch overhead is relevant).

2.  **Effective Bandwidth Utilization:**
    *   **Measure Total Bytes Transferred:**
        *   Weights (B): `K * N * 0.5` bytes (INT4) + `scales_size` + `zp_size`. For Marlin, these are read once.
        *   Activations (A): `M * K * sizeof(f16)` bytes. Read once.
        *   Output (C): `M * N * sizeof(f16)` bytes. Written once.
        *   Total `Memory_Bytes = (K * N / 2) + K*N_scales*2 + K*N_zeros*2 + M*K*2 + M*N*2`.
    *   **Calculate Effective Bandwidth:** `Memory_Bytes / Elapsed_Time`.
    *   **Utilization Percentage:** `Effective_Bandwidth / Peak_Memory_Bandwidth_GPU`. (RTX 4090 peak: 1008 GB/s).

3.  **L2 Cache Warming Effects:**
    *   Measure the very first run, then average subsequent runs. Report both or clearly state if it's "warm" performance. For M=1, A-matrix benefits from L2, B-matrix usually doesn't.
    *   `nvprof` (older) or `nsight compute` (newer) are essential tools for detailed profiling:
        *   Memory throughput (read/write bytes, average throughput, efficiency).
        *   Compute utilization (SM occupancy, active warps, instruction mix, Tensor Core activity).
        *   Shared memory conflicts, L1/L2 cache hit rates.

4.  **ExLlamaV2 and Llama.cpp Benchmarking:**
    *   They typically report `tok/sec` for end-to-end inference for a given model and context length.
    *   Sometimes they provide micro-benchmarks for specific GEMM functions, reporting GFLOPS or relative speedups.
    *   For `tok/sec`, it's important to specify context length, prompt processing vs. decode, batch size, and the exact model/quantization.

### 7b. Published Bandwidth Utilization Numbers for Production INT4 GEMM

*   **Marlin:** Claims 4x over FP16 at M=1. This implies a significant boost in effective bandwidth. If FP16 cuBLAS on a memory-bound 7B is ~15 tok/sec (TensorWarp's F32 is 14-23, FP16 likely faster), then 4x would be ~60 tok/sec. This is consistent with 60-80% bandwidth.
    *   Absolute GB/s: For 7B Q4, if it achieves 60 tok/sec, that implies `60 tok/sec * (4.2 GB / tok) = 252 GB/s` purely for weight data. Plus activations and output. With 60-80% peak BW (600-800 GB/s), this is achievable.
*   **ExLlamaV2:** ~32.5 tok/sec on 7B. This is lower than Marlin's best but still good. It implies a bandwidth utilization around 30-40% of peak (assuming a similar memory access pattern to Marlin). ExLlamaV2 may use simpler kernels or a different weight layout.
*   **Llama.cpp Q4_K_M on RTX 4090:** Highly optimized. Specific tok/sec depend on model and settings, but generally competitive with ExLlamaV2, sometimes better. For 7B Q4_K, 30-50 tok/sec is a common range. This also implies 30-50% bandwidth utilization.
*   **TensorRT-LLM W4A16:** Publishes very high throughput numbers on enterprise GPUs (H100), often reaching 60-80% of peak Tensor Core FLOPs. On consumer GPUs like 4090, their optimized FP16/INT8 kernels usually achieve 70-90% peak bandwidth for memory-bound ops. For W4A16, if it's fully optimized (Marlin-like), similar bandwidth utilization (60-80%) can be expected.

---

## Desired Output Synthesis

### 1. Marlin Weight Layout Specification

*   **Concept:** Marlin's layout for INT4 weights (e.g., GPTQ-128) is optimized for `mma.sync` operations, enabling coalesced `cp.async` loads and on-the-fly dequantization. Scales and zero-points (if asymmetric) are `f16` and stored separately but near their respective weight groups.
*   **Granularity:** Quantization is applied to `K_SCL_GROUP x N_SCL_GROUP` blocks (e.g., `1 x 128` or `128 x 128`).
*   **Physical Layout (conceptual for `B` matrix, `K x N`):**
    `weights_packed_bytes`: Array of `u8`
    `scales_f16`: Array of `f16`
    `zero_points_f16`: Array of `f16` (if asymmetric)

    The weights are reordered into logical blocks of `K_BLOCK_SIZE x N_BLOCK_SIZE` (e.g., `128 x 128` for GPU processing). Within each such block, the data is interleaved to maximize spatial locality for `cp.async` and `mma.sync`. A common pattern for `GPTQ-Marlin` is:
    `[ N_total / N_BLOCK_SIZE_MM ] [ K_total / K_BLOCK_SIZE_MM ] [ K_BLOCK_SIZE_MM / WARP_K_MM_SEGMENT ] [ N_BLOCK_SIZE_MM / PACK_SIZE ] [ WARP_K_MM_SEGMENT ]`
    Where:
    *   `N_BLOCK_SIZE_MM` = e.g., 128 (columns processed by a block of warps)
    *   `K_BLOCK_SIZE_MM` = e.g., 128 (rows processed by a block of warps)
    *   `WARP_K_MM_SEGMENT` = e.g., 16 or 32 (inner `k` for `mma.sync`)
    *   `PACK_SIZE = 2` (2 nibbles per byte)

    This structure ensures that the inner-most `[ (N_BLOCK_SIZE_MM / PACK_SIZE) x WARP_K_MM_SEGMENT ]` bytes are contiguous for `cp.async` loads. Scales and zero-points are typically stored contiguously to these blocks, or referenced by an offset.

*   **Worked Example for [3584, 3584] matrix:**
    *   `K_total = 3584`, `N_total = 3584`.
    *   Let `N_BLOCK_SIZE_MM = 128`, `K_BLOCK_SIZE_MM = 128`, `WARP_K_MM_SEGMENT = 16`.
    *   Total `N_groups = 3584 / 128 = 28`.
    *   Total `K_segments = 3584 / 128 = 28`.
    *   The scales/zero-points are stored as `f16` arrays of shape `[N_groups][K_segments]`.
    *   The packed weights are `[N_groups] [K_segments] [K_BLOCK_SIZE_MM / WARP_K_MM_SEGMENT] [N_BLOCK_SIZE_MM / PACK_SIZE] [WARP_K_MM_SEGMENT]`
    *   `[28] [28] [128 / 16 = 8] [128 / 2 = 64] [16]`
    *   Total bytes for weights: `28 * 28 * 8 * 64 * 16 = 6,389,760 bytes` (approx. 6.1 MB). Wait, this is `K_total * N_total / 2` but grouped. `3584 * 3584 / 2 = 6,422,528` bytes. The indexing aligns.

*   **Offline Reorder Algorithm (Pseudocode for `W[K][N]` to `Marlin_W`):**
    ```pseudocode
    function reorder_to_marlin_format(W_q[K_total][N_total], Scales[NumScales], Zeros[NumZeros]):
        N_BLOCK_MM = 128  // N-dimension block size for MMA
        K_BLOCK_MM = 128  // K-dimension block size for MMA
        WARP_K_SEGMENT = 16 // Inner K-segment for a warp's MMA
        PACK_SIZE = 2     // 2 nibbles per byte

        num_N_blocks = N_total / N_BLOCK_MM
        num_K_blocks = K_total / K_BLOCK_MM
        K_segments_per_K_block = K_BLOCK_MM / WARP_K_SEGMENT

        // Pre-allocate output arrays
        marlin_weights_packed = Array[ num_N_blocks * num_K_blocks * K_segments_per_K_block * (N_BLOCK_MM / PACK_SIZE) * WARP_K_SEGMENT ] of u8
        marlin_scales = Array[ num_N_blocks * num_K_blocks ] of f16
        marlin_zeros = Array[ num_N_blocks * num_K_blocks ] of f16 // Optional

        // Populate scales and zeros (assuming 1 scale/zero per K_BLOCK_MM x N_BLOCK_MM group)
        for n_b = 0 to num_N_blocks - 1:
            for k_b = 0 to num_K_blocks - 1:
                marlin_scales[n_b * num_K_blocks + k_b] = Scales_original[n_b * num_K_blocks + k_b]
                marlin_zeros[n_b * num_K_blocks + k_b] = Zeros_original[n_b * num_K_blocks + k_b]

        // Reorder packed weights
        for n_b = 0 to num_N_blocks - 1:            // Iterate over N-dimension blocks
            for k_b = 0 to num_K_blocks - 1:        // Iterate over K-dimension blocks
                for k_seg_idx = 0 to K_segments_per_K_block - 1: // Iterate over inner K-segments
                    for n_packed_byte_idx = 0 to (N_BLOCK_MM / PACK_SIZE) - 1: // Iterate over packed bytes within N-block
                        for k_inner_seg_row = 0 to WARP_K_SEGMENT - 1: // Iterate over K-rows in the segment
                            // Calculate original (k,n) coordinates
                            k_orig = k_b * K_BLOCK_MM + k_seg_idx * WARP_K_SEGMENT + k_inner_seg_row
                            n_orig_high_nibble = n_b * N_BLOCK_MM + n_packed_byte_idx * PACK_SIZE
                            n_orig_low_nibble = n_b * N_BLOCK_MM + n_packed_byte_idx * PACK_SIZE + 1

                            // Get nibbles (0-15)
                            nibble_high = W_q[k_orig][n_orig_high_nibble]
                            nibble_low = W_q[k_orig][n_orig_low_nibble]

                            // Pack byte (high nibble first)
                            packed_byte = (nibble_high << 4) | nibble_low

                            // Calculate target index in flattened marlin_weights_packed
                            target_idx = n_b * (num_K_blocks * K_segments_per_K_block * (N_BLOCK_MM / PACK_SIZE) * WARP_K_SEGMENT) + \
                                         k_b * (K_segments_per_K_block * (N_BLOCK_MM / PACK_SIZE) * WARP_K_SEGMENT) + \
                                         k_seg_idx * ((N_BLOCK_MM / PACK_SIZE) * WARP_K_SEGMENT) + \
                                         n_packed_byte_idx * WARP_K_SEGMENT + \
                                         k_inner_seg_row

                            marlin_weights_packed[target_idx] = packed_byte
    return marlin_weights_packed, marlin_scales, marlin_zeros
    ```

### 2. PTX Dequantization Recipe (Ada SM 8.9)

**From packed INT4 (u8) to FP16 values ready for MMA:**

Assume:
*   `R_packed_byte` (u8) contains `[high_nibble_u4 | low_nibble_u4]`
*   `R_scale_f16` (f16) contains the scale for the group
*   `R_zp_f16` (f16) contains the zero-point (if asymmetric), else assume fixed `8.0f16` for subtraction

```ptx
.reg .b32 packed_bytes;    // Input: packed INT4 bytes from global memory
.reg .b32 scale_reg_f32;   // Scale (f16 loaded as f32)
.reg .b32 zp_reg_f32;      // Zero-point (f16 loaded as f32)
.reg .b16 nibble_val_f16;  // Intermediate FP16 nibble value
.reg .b16 dequant_val_f16; // Final dequantized FP16 value

// Example: Loading a 32-bit register containing 4 packed bytes.
// For one packed byte in 'packed_bytes' (assume it's the lowest byte for simplicity)

// 1. Extract high nibble and convert to signed 8-bit (-8 to 7)
// Cycle 1: Extract high nibble (0-15)
shf.r.s32 nibble_high_u4, 0, 4, packed_bytes; // shift right 4 (for lowest byte)
and.b32 nibble_high_u4, nibble_high_u4, 0xF;  // mask
// Cycle 2: Convert to signed 8-bit (-8 to 7)
sub.s32 nibble_high_s8, nibble_high_u4, 8;

// 2. Extract low nibble and convert to signed 8-bit (-8 to 7)
// Cycle 3: Extract low nibble (0-15)
and.b32 nibble_low_u4, packed_bytes, 0xF;    // mask
// Cycle 4: Convert to signed 8-bit (-8 to 7)
sub.s32 nibble_low_s8, nibble_low_u4, 8;

// 3. Convert signed 8-bit to FP16 values
// Cycle 5: Convert high nibble's signed 8-bit to FP16
cvt.f16.s32.ftz high_val_f16, nibble_high_s8;
// Cycle 6: Convert low nibble's signed 8-bit to FP16
cvt.f16.s32.ftz low_val_f16, nibble_low_s8;

// 4. Apply scale. If asymmetric, subtract zero-point first.
// Assuming scale_reg_f16 and zp_reg_f16 are already loaded (e.g. into .b16 regs)
// For symmetric (ZP handled by -8):
// Cycle 7: Dequantize high nibble
fma.rn.f16 high_dequant_f16, high_val_f16, scale_reg_f16, 0f16; // Fused multiply-add (add 0)
// Cycle 8: Dequantize low nibble
fma.rn.f16 low_dequant_f16, low_val_f16, scale_reg_f16, 0f16;

// Total cycles: ~8 instructions / 2 nibbles
// Register usage: ~10-12 .b32/.b16 registers for the intermediate values per thread.
// (This example shows one byte. A warp processes 32 elements (16 bytes), so 16x this instruction count per warp per cycle for dequantization if not using vector instructions/intrinsics)
```
**Optimized PTX using `prmt` and `cvt.f16.s32` for `__half` intrinsic:**
A more efficient approach might involve loading a `u32` (4 packed bytes), then using `prmt` to extract 4 nibbles into separate bytes of a `u32` register, then converting all 4 `u8` nibbles (as signed 8-bit values) to FP16 pairs using `__vcvt_s8_to_f16` or similar intrinsics/PTX patterns. This would vectorize the process.

### 3. Architecture Feature Matrix

(As provided in section 1a, reproduced here for completion.)

| Feature                    | Turing (SM 7.5)      | Ampere (SM 8.0/8.6)         | Ada Lovelace (SM 8.9)          | Hopper (SM 9.0)                | Blackwell (SM 10.0)             |
| :------------------------- | :------------------- | :-------------------------- | :----------------------------- | :----------------------------- | :------------------------------ |
| **Tensor Cores**           | FP16, INT8 (WMMA)    | FP16, INT8 (MMA.SYNC)       | FP16, INT8 (MMA.SYNC)          | FP8, FP16, INT8 (WGMMA)        | INT4, FP8, FP16, INT8 (WGMMA)   |
| **INT4 Native?**           | No (use INT8 trick)  | No (use INT8 trick)         | No (use INT8 trick)            | No (use INT8 trick)            | **YES**                         |
| **Async Copy**             | No                   | `cp.async` (L2/GM to SMEM)  | `cp.async`                     | TMA (WGMMA, more flexible)     | TMA v2                          |
| **Shared Memory (SMEM)**   | 48KB/96KB            | 100KB/164KB                 | 48KB/96KB (per SM), configurable | 228KB (per SM)                 | Increased                       |
| **L2 Cache**               | 6MB                  | 40-42MB                     | 72MB                           | 50MB (smaller, but faster)     | Increased                       |
| **Registers per SM**       | 64K                  | 64K                         | 64K                            | 64K                            | Increased                       |
| **Recommended Config**     | `wmma`+INT8, small tiles | `mma.sync`+INT8 double pack, larger tiles, `cp.async` | `mma.sync`+INT8 double pack, `cp.async` | `wgmma`+TMA (FP8/FP16), large tiles | **WGMMA+TMA+Native INT4**, largest tiles |

### 4. Quantization Format Comparison

(As provided in section 2a, reproduced here for completion.)

| Format            | Scale Storage             | Zero-Point Handling       | Two-Tier Scaling | Bit Packing Order | Dequant Cost (ALU Ops)                                         |
| :---------------- | :------------------------ | :------------------------ | :--------------- | :---------------- | :------------------------------------------------------------- |
| **Q4_0** (GGML)   | `f32` per 32 elements     | Symmetric (implicit ZP=8) | No               | Low-nibble first  | `(nibble - 8) * scale`                                         |
| **Q4_K** (GGML)   | `f16` per 256 (super-blk) `f16` per 32 (block) | Symmetric (implicit ZP=8) | Yes              | Low-nibble first  | `(nibble - 8) * block_scale * super_block_scale`               |
| **GPTQ-128**      | `f16` per 128 elements    | `f16` per 128 elements (asymmetric) | No               | High-nibble first | `(nibble - zp) * scale`                                        |
| **AWQ**           | `f16` per channel (row)   | `f16` per channel (asymmetric or symmetric) | No               | High-nibble first | `(nibble - zp) * scale`                                        |
| **GGUF Q4_1**     | `f32` per 32 elements     | Asymmetric (explicit ZP)  | No               | Low-nibble first  | `(nibble - zp) * scale`                                        |
| **GGUF Q4_K_M/S** | Same as Q4_K              | Same as Q4_K              | Yes              | Low-nibble first  | Same as Q4_K                                                   |

### 5. Tiered Kernel Dispatch Design

TensorWarp's adaptive kernel selection system for a given `(M, N, K, quantization_format, GPU_Architecture)` tuple:

1.  **Input:** `M, N, K` (GEMM dimensions), `quant_format` (e.g., `TensorWarpQuantFormat::GPTQ_128_F16`), `sm_major`, `sm_minor`.
2.  **Canonical `(M,N,K,format)` Key:** Hash `(M,N,K,quant_format)` to form a unique key.
3.  **Kernel Cache Lookup (Tier 1):**
    *   Construct a cache key: `(canonical_key, sm_major, sm_minor, best_config_hash_from_autotuner)`.
    *   Check disk cache (`$TENSORWARP_CACHE_DIR/kernels/`) for a pre-compiled `.ptx` or `.cubin` file.
    *   If found: Load module, get function. Dispatch.
    *   If not found: Proceed to Tier 2.
4.  **Autotuning & NVRTC Compilation (Tier 2/3):**
    *   **A. Default Configuration:** Instantiate the Marlin-style CUDA C++ template with a *default, robust set of kernel parameters* (tile sizes, pipeline depth) derived from architecture and shape heuristics.
    *   **B. NVRTC Compile:** Compile the template using NVRTC for `sm_X.Y`. (This happens synchronously on first use).
    *   **C. Launch Default Kernel:** Dispatch the newly compiled default kernel.
    *   **D. Background Autotuning:** While the default kernel runs, initiate a background thread (or async task) to:
        *   Generate a pool of `N` candidate kernel configurations (varying `TILE_M/N/K`, `BLOCK_THREADS`, `NUM_STAGES_ASYNC`).
        *   For each candidate: Instantiate template, compile (NVRTC), profile (quick run).
        *   Use Thompson sampling or a simple best-N approach to identify the optimal configuration.
        *   Once an optimal config is found and compiled: update the disk cache, and on next inference pass for this `(M,N,K,format,SM)` tuple, switch to the optimal kernel.
        *   This optimal kernel becomes the "Tier 3" kernel.
5.  **Fallback (Tier 0):** If NVRTC compilation fails or takes too long (>30s for critical paths), or if the problem dimensions are too small/irregular for efficient Marlin kernels (e.g., `N < 32`), fallback to the `FP16 dequant + cuBLAS HGEMM` approach.

### 6. VRAM Budget Calculator

For a model with `P` parameters, context length `L`, `num_layers`, `hidden_size`, `num_heads`, `num_kv_heads`.

*   **1. Quantized Weights (Q4):**
    *   `Weight_Memory = P * 0.5` bytes (assuming 4 bits/param = 0.5 bytes/param)
    *   `Scales_Memory = (P / GROUP_SIZE_N / GROUP_SIZE_K_SCALES) * 2` bytes (f16 scale)
    *   `ZeroPoints_Memory = (P / GROUP_SIZE_N / GROUP_SIZE_K_SCALES) * 2` bytes (if asymmetric, f16 zp)
    *   `Total_Weights_Memory = Weight_Memory + Scales_Memory + ZeroPoints_Memory`
    *   *Example Qwen 7B (P=7B):* `7e9 * 0.5 = 3.5 GB`. Add padding/overhead for Marlin layout (~5-10%): `~3.7 - 4.2 GB`.

*   **2. KV Cache (FP16):**
    *   `KV_Memory = 2 * L * num_layers * (num_heads + 2 * num_kv_heads) * (hidden_size / num_heads) * sizeof(f16)`
    *   `KV_Memory = 2 * L * num_layers * (num_heads + 2 * num_kv_heads) * (hidden_size / num_heads) * 2` bytes
    *   *Example Qwen 7B (L=4096, 32L, 32H, 32KV, 4096 hidden):* `2 * 4096 * 32 * (32 + 2*32) * (4096/32) * 2 = 1.3 GB`.

*   **3. Activations (FP16):**
    *   Input/output activations, intermediate FFN states, etc. Typically `~0.5 - 1.0 GB` for models up to 13B.

*   **4. Scratchpad / Temp Buffers (FP16):**
    *   For Marlin-style, minimal (on-the-fly dequant, shared mem buffering): `~0.1 - 0.2 GB`.
    *   For Tier 0 (dequant to temp + cuBLAS): `K * N * sizeof(f16)` for largest layer. For 7B: `~32 MB` per layer, assume `~0.1 GB` if double-buffered.

*   **5. CUDA Context / Driver / Runtime:** `~0.2 - 0.5 GB`.

*   **Total VRAM = Total_Weights_Memory + KV_Memory + Activations + Scratchpad + CUDA_Context.**
*   *Example Qwen 7B on RTX 4090 (24 GB) with Marlin:* `4.2 GB + 1.3 GB (L=4096) + 0.7 GB + 0.1 GB + 0.3 GB = ~6.6 GB`.
*   This leaves plenty of headroom for `L=8192` (~8 GB total) or larger batch sizes (if KV cache is shared), or multiple models.

### 7. Implementation Roadmap

**Phase 0: Baseline & Foundation (Current State)**
*   **Goal:** Solidify current Q4_0 (28.6 tok/sec) and ensure correctness.
*   **Effort:** Ongoing.
*   **Achieved:** 28.6 tok/sec on Qwen 7B Q4_0, kernel-level optimizations exhausted.

**Phase 1: Marlin-Style Layout & Basic Dequantization (Target 40 tok/sec)**
*   **Goal:** Implement Marlin-style weight reordering and a single, basic Marlin-style kernel for Q4_0 on RTX 4090.
*   **Effort:** 3-4 weeks.
    1.  **Define Canonical Internal Format:** Formalize Marlin-like `TensorWarp.Marlin` format (packed nibbles, f16 scales/zeros).
    2.  **Offline Processor:** Implement the Rust `reorder_to_marlin_format` function to convert Q4_0 to this format. Cache to disk.
    3.  **Basic Marlin Kernel (NVRTC):**
        *   Start with a simple M=1, N=128, K=128 tile size.
        *   Implement `cp.async.cs` for weights, `cp.async.ca` for activations.
        *   Implement PTX-level nibble extraction (`shf`/`and` first, then `prmt` for optimization) and `s8` to `f16` conversion.
        *   Use `mma.sync.m8n8k16.s8.f16.f16` Tensor Core instruction.
        *   Integrate kernel into NVRTC pipeline for SM 8.9.
    4.  **Benchmarking:** Measure bandwidth utilization and `tok/sec` for Qwen 7B Q4_0.
*   **Target:** `40 tok/sec` for Qwen 7B Q4_0. `30-40%` bandwidth utilization.

**Phase 2: Adaptive Kernel & Advanced Dequantization (Target 60 tok/sec)**
*   **Goal:** Extend to multiple architectures, add advanced dequant/pipelining, and basic autotuning.
*   **Effort:** 4-6 weeks.
    1.  **Dequantization Pipeline:** Optimize PTX dequantization using `prmt` and possibly `vcvt` intrinsics. Fine-tune `cp.async` pipeline depth (double-buffering).
    2.  **Architecture-Specific Templates:** Refactor kernel into a C++ template with `if constexpr`/`#if` for `SM_MAJOR/MINOR`, `HAS_ZERO_POINT`, `PACKING_ORDER`.
    3.  **Tiered Dispatch (Tier 0, 1, 2):** Implement the kernel dispatch system (cache lookup, default NVRTC, background autotuning).
    4.  **Quant Format Support:** Add support for GPTQ-128 and Q4_K to the offline processor and kernel template.
    5.  **Small N fallback:** Implement heuristic to fallback to Tier 0 for very small N.
*   **Target:** `60 tok/sec` for Qwen 7B Q4_0/GPTQ-128. `50-60%` bandwidth utilization.

**Phase 3: Refinement & Universality (Target 80+ tok/sec)**
*   **Goal:** Optimize all aspects to achieve near-peak bandwidth utilization and full universality.
*   **Effort:** 4-6 weeks.
    1.  **Advanced Autotuning:** Expand parameter space for Thompson sampling (more tile configs, thread blocks).
    2.  **Kernel Fusions:** Fuse common activations (SiLU/SwiGLU) and biases into kernel epilogues.
    3.  **Model Architecture Adaptations:** Handle non-power-of-2 dimensions efficiently (padding/tail processing).
    4.  **Memory Management:** Optimize VRAM allocation and scratchpad reuse across layers.
    5.  **Multi-GPU/MoE (Optional, Future):** Investigate distributed quantized GEMM.
*   **Target:** `80+ tok/sec` for Qwen 7B Q4_0/GPTQ-128. `70-80%` bandwidth utilization.

### 8. Benchmark Targets

**Methodology:** All `tok/sec` benchmarks for Qwen 7B (7B-chat if possible), Q4_0 or GPTQ-128, context length 2048 (decoding new tokens), batch size 1. Measure mean over 100-500 tokens, after a 50-token warm-up. Bandwidth utilization measured using `nsight compute` on dedicated GEMM kernel launch.

*   **Current (TensorWarp Q4_0):**
    *   **28.6 tok/sec**
    *   **22% peak bandwidth utilization**

*   **Phase 1 Target (Basic Marlin):**
    *   **40 tok/sec**
    *   **30-40% peak bandwidth utilization**

*   **Phase 2 Target (Adaptive Marlin):**
    *   **60 tok/sec**
    *   **50-60% peak bandwidth utilization**

*   **Phase 3 Target (Optimized Marlin):**
    *   **80-96 tok/sec**
    *   **70-80% peak bandwidth utilization**
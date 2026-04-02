//! Paged Attention — the key to efficient LLM serving.
//!
//! Standard KV cache allocates max_seq_len * num_layers * 2 * kv_dim contiguously.
//! For a batch of 64 sequences at 4K context: that's gigabytes of reserved memory,
//! most of it empty.
//!
//! Paged attention (from vLLM) treats the KV cache like virtual memory:
//! - KV data is stored in fixed-size BLOCKS (e.g., 16 tokens per block)
//! - Each sequence has a BLOCK TABLE mapping logical position → physical block
//! - Blocks are allocated on demand from a free pool
//! - No memory waste from pre-allocation
//!
//! This enables:
//! - Continuous batching (sequences of different lengths share the same pool)
//! - Memory sharing (beam search can share prefix blocks via copy-on-write)
//! - Near-zero fragmentation
//!
//! The kernel reads Q for the current token, looks up K/V from scattered blocks
//! via the block table, and computes attention with online softmax.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Helper to launch with proper error mapping.
macro_rules! launch_err {
    ($e:expr) => {
        $e.map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))
    };
}

/// Block size for paged KV cache (tokens per block).
/// 16 is the standard from vLLM — good balance of granularity vs overhead.
pub const BLOCK_SIZE: u32 = 16;

/// Paged attention kernel.
///
/// For each query in the batch:
/// 1. Read Q vector for this head
/// 2. Walk the block table to find physical K/V blocks
/// 3. Compute attention scores with online softmax
/// 4. Write output
///
/// Grid: (num_heads, batch_size)
/// Block: (head_dim) — one thread per dimension
const PAGED_ATTENTION_SRC: &str = r#"
extern "C" __global__ void warp_paged_attention(
    float *out,                  // [batch, num_heads, head_dim]
    const float *q,              // [batch, num_heads, head_dim]
    const float *k_cache,        // [num_blocks, block_size, num_kv_heads, head_dim]
    const float *v_cache,        // [num_blocks, block_size, num_kv_heads, head_dim]
    const int *block_tables,     // [batch, max_blocks_per_seq]
    const int *seq_lens,         // [batch] — actual sequence length per request
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int block_size,
    unsigned int max_blocks_per_seq,
    float scale
) {
    unsigned int head = blockIdx.x;
    unsigned int batch_idx = blockIdx.y;
    unsigned int d = threadIdx.x;

    if (d >= head_dim) return;

    int seq_len = seq_lens[batch_idx];
    if (seq_len <= 0) return;

    // GQA: map query head to kv head
    unsigned int kv_head = head / (num_heads / num_kv_heads);

    // Load Q for this head
    float q_val = q[batch_idx * num_heads * head_dim + head * head_dim + d];

    // Block table for this sequence
    const int *block_table = block_tables + batch_idx * max_blocks_per_seq;

    // Online softmax state
    float running_max = -1e30f;
    float running_sum = 0.0f;
    float running_out = 0.0f;

    // Iterate over all cached positions via block table
    for (int pos = 0; pos < seq_len; pos++) {
        int block_idx_logical = pos / block_size;
        int block_offset = pos % block_size;
        int physical_block = block_table[block_idx_logical];

        // K[physical_block][block_offset][kv_head][d]
        unsigned int k_idx = physical_block * block_size * num_kv_heads * head_dim
                           + block_offset * num_kv_heads * head_dim
                           + kv_head * head_dim + d;
        float k_val = k_cache[k_idx];

        // Dot product Q·K — need reduction across head_dim
        float partial = q_val * k_val;

        // Warp reduction for dot product
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        float score = __shfl_sync(0xffffffff, partial, 0) * scale;

        // For head_dim > 32, use shared mem reduction
        // (This kernel handles head_dim <= 32 via warp shuffle;
        //  for larger head_dim, see paged_attention_extended below)

        // V[physical_block][block_offset][kv_head][d]
        unsigned int v_idx = physical_block * block_size * num_kv_heads * head_dim
                           + block_offset * num_kv_heads * head_dim
                           + kv_head * head_dim + d;
        float v_val = v_cache[v_idx];

        // Online softmax update
        float new_max = fmaxf(running_max, score);
        float correction = expf(running_max - new_max);
        float weight = expf(score - new_max);

        running_sum = running_sum * correction + weight;
        running_out = running_out * correction + weight * v_val;
        running_max = new_max;
    }

    // Write output
    if (running_sum > 0.0f) {
        out[batch_idx * num_heads * head_dim + head * head_dim + d] = running_out / running_sum;
    }
}
"#;

/// Extended paged attention for head_dim > 32.
/// Uses shared memory reduction for the dot product.
const PAGED_ATTENTION_EXT_SRC: &str = r#"
extern "C" __global__ void warp_paged_attention_ext(
    float *out,
    const float *q,
    const float *k_cache,
    const float *v_cache,
    const int *block_tables,
    const int *seq_lens,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int block_size,
    unsigned int max_blocks_per_seq,
    float scale
) {
    extern __shared__ float smem[];
    float *dot_buf = smem;

    unsigned int head = blockIdx.x;
    unsigned int batch_idx = blockIdx.y;
    unsigned int d = threadIdx.x;

    int seq_len = seq_lens[batch_idx];
    if (seq_len <= 0) return;

    unsigned int kv_head = head / (num_heads / num_kv_heads);

    float q_val = (d < head_dim) ? q[batch_idx * num_heads * head_dim + head * head_dim + d] : 0.0f;

    const int *block_table = block_tables + batch_idx * max_blocks_per_seq;

    float running_max = -1e30f;
    float running_sum = 0.0f;
    float running_out = 0.0f;

    for (int pos = 0; pos < seq_len; pos++) {
        int block_idx_logical = pos / block_size;
        int block_offset = pos % block_size;
        int physical_block = block_table[block_idx_logical];

        unsigned int kv_idx = physical_block * block_size * num_kv_heads * head_dim
                            + block_offset * num_kv_heads * head_dim
                            + kv_head * head_dim;

        // Compute dot product via shared memory reduction
        float partial = 0.0f;
        if (d < head_dim) {
            partial = q_val * k_cache[kv_idx + d];
        }
        dot_buf[threadIdx.x] = partial;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                dot_buf[threadIdx.x] += dot_buf[threadIdx.x + stride];
            __syncthreads();
        }

        float score = dot_buf[0] * scale;

        float v_val = (d < head_dim) ? v_cache[kv_idx + d] : 0.0f;

        float new_max = fmaxf(running_max, score);
        float correction = expf(running_max - new_max);
        float weight = expf(score - new_max);

        running_sum = running_sum * correction + weight;
        running_out = running_out * correction + weight * v_val;
        running_max = new_max;

        __syncthreads();
    }

    if (d < head_dim && running_sum > 0.0f) {
        out[batch_idx * num_heads * head_dim + head * head_dim + d] = running_out / running_sum;
    }
}
"#;

/// Paged KV cache block pool.
/// Manages physical blocks of KV data on the GPU.
pub struct PagedKVPool {
    /// K cache blocks: [num_blocks, block_size, num_kv_heads, head_dim]
    pub k_blocks: GpuTensor<f32>,
    /// V cache blocks: same layout
    pub v_blocks: GpuTensor<f32>,
    /// Free block indices
    free_blocks: Vec<u32>,
    /// Config
    pub num_blocks: u32,
    pub block_size: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
}

impl PagedKVPool {
    /// Allocate a pool of KV blocks on GPU.
    pub fn new(
        device: &WarpDevice,
        num_blocks: u32,
        block_size: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> Result<Self, DeviceError> {
        let block_elems = (num_blocks * block_size * num_kv_heads * head_dim) as usize;
        let shape = Shape::from_static(&[
            num_blocks as usize,
            block_size as usize,
            num_kv_heads as usize,
            head_dim as usize,
        ]);
        let k_blocks = GpuTensor::<f32>::zeros(device, shape.clone(), DType::F32)?;
        let v_blocks = GpuTensor::<f32>::zeros(device, shape, DType::F32)?;

        let free_blocks = (0..num_blocks).rev().collect();

        Ok(Self {
            k_blocks,
            v_blocks,
            free_blocks,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        })
    }

    /// Allocate a block, returns physical block index.
    pub fn alloc_block(&mut self) -> Option<u32> {
        self.free_blocks.pop()
    }

    /// Free a block back to the pool.
    pub fn free_block(&mut self, block_id: u32) {
        self.free_blocks.push(block_id);
    }

    /// Number of free blocks remaining.
    pub fn num_free(&self) -> usize {
        self.free_blocks.len()
    }

    /// Total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.k_blocks.size_bytes() + self.v_blocks.size_bytes()
    }

    pub fn summary(&self) -> String {
        format!(
            "PagedKVPool: {} blocks ({}×{}×{}×{}), {}/{} free, {:.2} MB",
            self.num_blocks, self.block_size, self.num_kv_heads, self.head_dim,
            self.block_size, self.num_free(), self.num_blocks,
            self.memory_bytes() as f64 / 1e6,
        )
    }
}

/// Per-sequence state for paged attention.
pub struct PagedSequence {
    /// Physical block indices for this sequence
    pub block_table: Vec<i32>,
    /// Current sequence length
    pub seq_len: u32,
    /// Max blocks this sequence can use
    pub max_blocks: u32,
}

impl PagedSequence {
    pub fn new(max_blocks: u32) -> Self {
        Self {
            block_table: Vec::with_capacity(max_blocks as usize),
            seq_len: 0,
            max_blocks,
        }
    }

    /// Ensure we have enough blocks allocated for the current position.
    /// Returns true if a new block was needed.
    pub fn ensure_blocks(&mut self, pool: &mut PagedKVPool) -> bool {
        let blocks_needed = (self.seq_len + pool.block_size - 1) / pool.block_size;
        if blocks_needed as usize > self.block_table.len() {
            if let Some(block) = pool.alloc_block() {
                self.block_table.push(block as i32);
                return true;
            }
        }
        false
    }

    /// Free all blocks back to the pool.
    pub fn free_all(&mut self, pool: &mut PagedKVPool) {
        for &block in &self.block_table {
            pool.free_block(block as u32);
        }
        self.block_table.clear();
        self.seq_len = 0;
    }
}

/// Append KV for a new token into the paged cache.
const PAGED_KV_APPEND_SRC: &str = r#"
extern "C" __global__ void warp_paged_kv_append(
    float *k_cache,              // [num_blocks, block_size, num_kv_heads, head_dim]
    float *v_cache,
    const float *new_k,          // [num_kv_heads, head_dim]
    const float *new_v,
    int physical_block,
    int block_offset,
    unsigned int num_kv_heads,
    unsigned int head_dim
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_kv_heads * head_dim;
    if (idx >= total) return;

    unsigned int cache_idx = physical_block * gridDim.y * num_kv_heads * head_dim
                           + block_offset * num_kv_heads * head_dim + idx;
    // Note: gridDim.y is being misused here. Let's use the direct offset.
    unsigned int base = physical_block * BLOCK_SIZE * num_kv_heads * head_dim
                      + block_offset * num_kv_heads * head_dim;
    k_cache[base + idx] = new_k[idx];
    v_cache[base + idx] = new_v[idx];
}
"#;

/// Generate the paged KV append kernel with correct block size baked in.
fn paged_kv_append_src(block_size: u32) -> String {
    format!(r#"
#define BLOCK_SIZE {block_size}

extern "C" __global__ void warp_paged_kv_append(
    float *k_cache,
    float *v_cache,
    const float *new_k,
    const float *new_v,
    int physical_block,
    int block_offset,
    unsigned int num_kv_heads,
    unsigned int head_dim
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_kv_heads * head_dim;
    if (idx >= total) return;

    unsigned int base = physical_block * BLOCK_SIZE * num_kv_heads * head_dim
                      + block_offset * num_kv_heads * head_dim;
    k_cache[base + idx] = new_k[idx];
    v_cache[base + idx] = new_v[idx];
}}
"#)
}

/// Append a token's KV into the paged cache at the right position.
pub fn paged_kv_append(
    cache: &KernelCache,
    device: &WarpDevice,
    pool: &mut PagedKVPool,
    seq: &mut PagedSequence,
    new_k: &GpuTensor<f32>,  // [num_kv_heads, head_dim]
    new_v: &GpuTensor<f32>,
) -> Result<(), DeviceError> {
    // Ensure we have a block for this position
    seq.seq_len += 1;
    seq.ensure_blocks(pool);

    let block_idx = ((seq.seq_len - 1) / pool.block_size) as usize;
    let block_offset = ((seq.seq_len - 1) % pool.block_size) as i32;
    let physical_block = seq.block_table[block_idx];

    let src = paged_kv_append_src(pool.block_size);
    let f = cache.get_or_compile(device, &src, "warp_paged_kv_append")?;

    let total = pool.num_kv_heads * pool.head_dim;
    let cfg = LaunchConfig::for_num_elems(total);

    unsafe {
        launch_err!(device.stream.launch_builder(&f)
            .arg(&mut pool.k_blocks.data)
            .arg(&mut pool.v_blocks.data)
            .arg(&new_k.data)
            .arg(&new_v.data)
            .arg(&physical_block)
            .arg(&block_offset)
            .arg(&pool.num_kv_heads)
            .arg(&pool.head_dim)
            .launch(cfg))?;
    }
    Ok(())
}

/// Run paged attention for a batch of sequences.
/// Automatically selects warp-level (head_dim <= 32) or extended (head_dim > 32).
pub fn paged_attention(
    cache: &KernelCache,
    device: &WarpDevice,
    q: &GpuTensor<f32>,              // [batch, num_heads, head_dim]
    pool: &PagedKVPool,
    block_tables: &GpuTensor<i32>,   // [batch, max_blocks_per_seq]
    seq_lens: &GpuTensor<i32>,       // [batch]
    out: &mut GpuTensor<f32>,        // [batch, num_heads, head_dim]
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_blocks_per_seq: u32,
) -> Result<(), DeviceError> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    if head_dim <= 32 {
        let f = cache.get_or_compile(device, PAGED_ATTENTION_SRC, "warp_paged_attention")?;
        let cfg = LaunchConfig {
            grid_dim: (num_heads, batch, 1),
            block_dim: (head_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut out.data)
                .arg(&q.data)
                .arg(&pool.k_blocks.data)
                .arg(&pool.v_blocks.data)
                .arg(&block_tables.data)
                .arg(&seq_lens.data)
                .arg(&num_heads)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&pool.block_size)
                .arg(&max_blocks_per_seq)
                .arg(&scale)
                .launch(cfg))?;
        }
    } else {
        let f = cache.get_or_compile(device, PAGED_ATTENTION_EXT_SRC, "warp_paged_attention_ext")?;
        let block_size_threads = head_dim.next_power_of_two().max(32);
        let shared_mem = block_size_threads * 4;
        let cfg = LaunchConfig {
            grid_dim: (num_heads, batch, 1),
            block_dim: (block_size_threads, 1, 1),
            shared_mem_bytes: shared_mem,
        };
        unsafe {
            launch_err!(device.stream.launch_builder(&f)
                .arg(&mut out.data)
                .arg(&q.data)
                .arg(&pool.k_blocks.data)
                .arg(&pool.v_blocks.data)
                .arg(&block_tables.data)
                .arg(&seq_lens.data)
                .arg(&num_heads)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&pool.block_size)
                .arg(&max_blocks_per_seq)
                .arg(&scale)
                .launch(cfg))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    #[test]
    fn paged_pool_alloc_free() {
        let (dev, _cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let mut pool = PagedKVPool::new(&dev, 64, BLOCK_SIZE, 4, 32).unwrap();
        println!("{}", pool.summary());
        assert_eq!(pool.num_free(), 64);

        let b0 = pool.alloc_block().unwrap();
        let b1 = pool.alloc_block().unwrap();
        assert_eq!(pool.num_free(), 62);

        pool.free_block(b0);
        pool.free_block(b1);
        assert_eq!(pool.num_free(), 64);
    }

    #[test]
    fn paged_sequence_block_allocation() {
        let (dev, _cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let mut pool = PagedKVPool::new(&dev, 64, BLOCK_SIZE, 4, 32).unwrap();
        let mut seq = PagedSequence::new(128);

        // Simulate 48 tokens — should need 3 blocks (16 per block)
        for _ in 0..48 {
            seq.seq_len += 1;
            seq.ensure_blocks(&mut pool);
        }

        assert_eq!(seq.block_table.len(), 3);
        assert_eq!(pool.num_free(), 61);

        seq.free_all(&mut pool);
        assert_eq!(pool.num_free(), 64);
        assert_eq!(seq.seq_len, 0);
    }

    #[test]
    fn paged_kv_append_and_attend() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let num_kv_heads = 1u32;
        let head_dim = 16u32;
        let num_heads = 1u32;
        let kv_dim = (num_kv_heads * head_dim) as usize;
        let max_blocks_per_seq = 8u32;

        let mut pool = PagedKVPool::new(&dev, 32, BLOCK_SIZE, num_kv_heads, head_dim).unwrap();
        let mut seq = PagedSequence::new(max_blocks_per_seq);

        // Append 10 tokens
        for pos in 0..10u32 {
            let k_data: Vec<f32> = (0..kv_dim).map(|d| pos as f32 + d as f32 * 0.01).collect();
            let v_data: Vec<f32> = (0..kv_dim).map(|d| -(pos as f32) + d as f32 * 0.01).collect();

            let k = GpuTensor::from_host(&dev, &k_data, Shape::from_static(&[kv_dim]), DType::F32).unwrap();
            let v = GpuTensor::from_host(&dev, &v_data, Shape::from_static(&[kv_dim]), DType::F32).unwrap();

            paged_kv_append(&cache, &dev, &mut pool, &mut seq, &k, &v).unwrap();
        }
        dev.synchronize().unwrap();

        assert_eq!(seq.seq_len, 10);
        assert_eq!(seq.block_table.len(), 1); // 10 tokens fits in 1 block (16 per block)

        // Now run paged attention
        let q_data: Vec<f32> = (0..head_dim as usize).map(|d| (d % 7) as f32 * 0.1 - 0.3).collect();
        let batch = 1u32;

        // Pad block table to max_blocks_per_seq
        let mut bt = seq.block_table.clone();
        bt.resize(max_blocks_per_seq as usize, 0);
        let block_table_gpu = GpuTensor::from_host(&dev, &bt,
            Shape::from_static(&[batch as usize, max_blocks_per_seq as usize]), DType::I32).unwrap();
        let seq_lens_data = vec![seq.seq_len as i32];
        let seq_lens_gpu = GpuTensor::from_host(&dev, &seq_lens_data,
            Shape::from_static(&[batch as usize]), DType::I32).unwrap();

        let q = GpuTensor::from_host(&dev, &q_data,
            Shape::from_static(&[batch as usize, num_heads as usize, head_dim as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[batch as usize, num_heads as usize, head_dim as usize]), DType::F32).unwrap();

        paged_attention(
            &cache, &dev, &q, &pool, &block_table_gpu, &seq_lens_gpu,
            &mut out, batch, num_heads, num_kv_heads, head_dim, max_blocks_per_seq,
        ).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        assert!(result.iter().all(|v| v.is_finite()), "Paged attention has NaN/Inf!");
        assert!(result.iter().any(|v| *v != 0.0), "Paged attention all zeros!");

        println!("Paged attention: 10 tokens, 1 head, D={head_dim}");
        println!("  Output: [{:.4}, {:.4}, {:.4}, ...]", result[0], result[1], result[2]);
        println!("  Blocks used: {} ({} free)", seq.block_table.len(), pool.num_free());

        seq.free_all(&mut pool);
    }

    #[test]
    fn paged_attention_extended_128d() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let num_kv_heads = 1u32;
        let head_dim = 128u32;  // LLaMA head_dim
        let num_heads = 1u32;
        let kv_dim = (num_kv_heads * head_dim) as usize;
        let max_blocks_per_seq = 16u32;

        let mut pool = PagedKVPool::new(&dev, 64, BLOCK_SIZE, num_kv_heads, head_dim).unwrap();
        let mut seq = PagedSequence::new(max_blocks_per_seq);

        // Append 40 tokens (spans 3 blocks)
        for pos in 0..40u32 {
            let k_data: Vec<f32> = (0..kv_dim).map(|d| ((pos as usize * kv_dim + d) % 101) as f32 * 0.01 - 0.5).collect();
            let v_data: Vec<f32> = (0..kv_dim).map(|d| ((pos as usize * kv_dim + d) % 83) as f32 * 0.01 - 0.4).collect();

            let k = GpuTensor::from_host(&dev, &k_data, Shape::from_static(&[kv_dim]), DType::F32).unwrap();
            let v = GpuTensor::from_host(&dev, &v_data, Shape::from_static(&[kv_dim]), DType::F32).unwrap();
            paged_kv_append(&cache, &dev, &mut pool, &mut seq, &k, &v).unwrap();
        }
        dev.synchronize().unwrap();

        assert_eq!(seq.seq_len, 40);
        assert_eq!(seq.block_table.len(), 3); // ceil(40/16) = 3 blocks

        let batch = 1u32;
        let q_data: Vec<f32> = (0..head_dim as usize).map(|d| (d % 11) as f32 * 0.05 - 0.25).collect();

        let mut bt = seq.block_table.clone();
        bt.resize(max_blocks_per_seq as usize, 0);
        let block_table_gpu = GpuTensor::from_host(&dev, &bt,
            Shape::from_static(&[batch as usize, max_blocks_per_seq as usize]), DType::I32).unwrap();
        let seq_lens_gpu = GpuTensor::from_host(&dev, &[seq.seq_len as i32],
            Shape::from_static(&[batch as usize]), DType::I32).unwrap();

        let q = GpuTensor::from_host(&dev, &q_data,
            Shape::from_static(&[batch as usize, num_heads as usize, head_dim as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[batch as usize, num_heads as usize, head_dim as usize]), DType::F32).unwrap();

        paged_attention(
            &cache, &dev, &q, &pool, &block_table_gpu, &seq_lens_gpu,
            &mut out, batch, num_heads, num_kv_heads, head_dim, max_blocks_per_seq,
        ).unwrap();
        dev.synchronize().unwrap();

        let result = out.to_host(&dev).unwrap();
        assert!(result.iter().all(|v| v.is_finite()), "Paged attention ext has NaN/Inf!");
        assert!(result.iter().any(|v| *v != 0.0), "Paged attention ext all zeros!");

        println!("Paged attention ext: 40 tokens, D={head_dim}, 3 blocks");
        println!("  Output: [{:.4}, {:.4}, {:.4}, ...]", result[0], result[1], result[2]);

        seq.free_all(&mut pool);
    }

    #[test]
    fn paged_attention_perf() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let num_kv_heads = 8u32;
        let head_dim = 128u32;
        let num_heads = 32u32;
        let kv_dim = (num_kv_heads * head_dim) as usize;
        let max_blocks_per_seq = 128u32;

        let mut pool = PagedKVPool::new(&dev, 1024, BLOCK_SIZE, num_kv_heads, head_dim).unwrap();
        let mut seq = PagedSequence::new(max_blocks_per_seq);

        // Fill 512 tokens
        for pos in 0..512u32 {
            let k_data: Vec<f32> = (0..kv_dim).map(|d| ((pos as usize + d) % 100) as f32 * 0.01).collect();
            let v_data = k_data.clone();
            let k = GpuTensor::from_host(&dev, &k_data, Shape::from_static(&[kv_dim]), DType::F32).unwrap();
            let v = GpuTensor::from_host(&dev, &v_data, Shape::from_static(&[kv_dim]), DType::F32).unwrap();
            paged_kv_append(&cache, &dev, &mut pool, &mut seq, &k, &v).unwrap();
        }
        dev.synchronize().unwrap();

        let batch = 1u32;
        let q_data: Vec<f32> = (0..(num_heads * head_dim) as usize).map(|d| d as f32 * 0.001).collect();
        let mut bt = seq.block_table.clone();
        bt.resize(max_blocks_per_seq as usize, 0);
        let block_table_gpu = GpuTensor::from_host(&dev, &bt,
            Shape::from_static(&[1, max_blocks_per_seq as usize]), DType::I32).unwrap();
        let seq_lens_gpu = GpuTensor::from_host(&dev, &[512i32],
            Shape::from_static(&[1]), DType::I32).unwrap();
        let q = GpuTensor::from_host(&dev, &q_data,
            Shape::from_static(&[1, num_heads as usize, head_dim as usize]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev,
            Shape::from_static(&[1, num_heads as usize, head_dim as usize]), DType::F32).unwrap();

        // Warmup
        paged_attention(
            &cache, &dev, &q, &pool, &block_table_gpu, &seq_lens_gpu,
            &mut out, batch, num_heads, num_kv_heads, head_dim, max_blocks_per_seq,
        ).unwrap();
        dev.synchronize().unwrap();

        let iters = 200;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            paged_attention(
                &cache, &dev, &q, &pool, &block_table_gpu, &seq_lens_gpu,
                &mut out, batch, num_heads, num_kv_heads, head_dim, max_blocks_per_seq,
            ).unwrap();
        }
        dev.synchronize().unwrap();
        let elapsed = start.elapsed();

        println!("\nPaged attention perf (512 tokens, {num_heads}H, {num_kv_heads}KVH, D={head_dim}):");
        println!("  {:.3}μs avg ({iters} iters)", elapsed.as_secs_f64() * 1e6 / iters as f64);
        println!("  Pool: {}", pool.summary());

        seq.free_all(&mut pool);
    }
}

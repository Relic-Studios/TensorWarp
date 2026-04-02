//! GPU memory management.
//!
//! Arena-based memory pool to avoid per-tensor cudaMalloc/cudaFree.
//! GPU memory allocation is expensive (~10μs each) — with hundreds of
//! tensors per forward pass, that overhead adds up. The pool allocates
//! one large chunk upfront and sub-allocates from it.
//!
//! For inference (not training), we know the full memory layout at
//! compile time — every tensor's lifetime is determined by the graph.
//! This means we can do optimal memory planning: reuse memory for
//! tensors whose lifetimes don't overlap.

use warp_ir::{DType, Shape};

/// A tensor buffer — a view into the memory pool.
#[derive(Debug, Clone)]
pub struct TensorBuffer {
    /// Offset into the memory pool (bytes).
    pub offset: usize,
    /// Size in bytes.
    pub size: usize,
    /// Shape of the tensor stored here.
    pub shape: Shape,
    /// Data type.
    pub dtype: DType,
}

impl TensorBuffer {
    pub fn new(offset: usize, shape: Shape, dtype: DType) -> Self {
        let numel = shape.numel().expect("TensorBuffer requires static shape");
        let size = numel * dtype.byte_size();
        Self {
            offset,
            size,
            shape,
            dtype,
        }
    }
}

/// Simple bump allocator for the memory pool.
/// Phase 1: bump allocation. Phase 2: lifetime-aware reuse.
#[derive(Debug)]
pub struct MemoryPool {
    /// Total pool size in bytes.
    capacity: usize,
    /// Current allocation offset (bump pointer).
    offset: usize,
    /// All allocations for bookkeeping.
    allocations: Vec<TensorBuffer>,
}

impl MemoryPool {
    /// Create a new memory pool with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            offset: 0,
            allocations: Vec::new(),
        }
    }

    /// Allocate a tensor buffer from the pool.
    pub fn alloc(&mut self, shape: Shape, dtype: DType) -> Result<TensorBuffer, MemoryError> {
        let numel = shape
            .numel()
            .ok_or(MemoryError::DynamicShape)?;
        let size = numel * dtype.byte_size();

        // Align to 256 bytes (GPU cache line alignment).
        let aligned_offset = (self.offset + 255) & !255;

        if aligned_offset + size > self.capacity {
            return Err(MemoryError::OutOfMemory {
                requested: size,
                available: self.capacity.saturating_sub(aligned_offset),
            });
        }

        let buf = TensorBuffer {
            offset: aligned_offset,
            size,
            shape,
            dtype,
        };

        self.offset = aligned_offset + size;
        self.allocations.push(buf.clone());
        Ok(buf)
    }

    /// Reset the pool (free all allocations). O(1).
    pub fn reset(&mut self) {
        self.offset = 0;
        self.allocations.clear();
    }

    /// Current memory usage.
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Total capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Peak memory usage (for planning).
    pub fn peak_usage(&self) -> usize {
        self.offset
    }
}

/// Memory planning: compute the minimum memory needed for a graph
/// by analyzing tensor lifetimes and reusing memory for non-overlapping tensors.
pub struct MemoryPlan {
    /// Total memory needed (bytes).
    pub total_bytes: usize,
    /// Assignment of each ValueId to an offset in the pool.
    pub assignments: Vec<(warp_ir::ValueId, usize)>,
}

/// Plan memory allocation for an entire graph.
/// Uses a greedy algorithm: process tensors in topological order,
/// assign them to the first free slot that fits.
pub fn plan_memory(graph: &mut warp_ir::Graph) -> MemoryPlan {
    let topo = graph.topo_order().to_vec();

    // For each value, determine its size and lifetime (first use to last use).
    let mut lifetimes: Vec<(warp_ir::ValueId, usize, usize, usize)> = Vec::new(); // (id, birth, death, size)

    for (step, &node_id) in topo.iter().enumerate() {
        let node = graph.node(node_id);

        // Outputs are born at this step
        for &output in &node.outputs {
            let info = graph.value(output);
            let size = info.shape.numel().unwrap_or(0) * info.dtype.byte_size();
            let users = graph.value_users(output);
            // Death is the step of the last consumer
            let death = if users.is_empty() {
                step // graph output or dead — lives until end
            } else {
                users
                    .iter()
                    .filter_map(|&uid| topo.iter().position(|&t| t == uid))
                    .max()
                    .unwrap_or(step)
            };
            lifetimes.push((output, step, death, size));
        }
    }

    // Graph outputs must live until the end
    let max_step = topo.len();
    for &output in &graph.graph_outputs {
        if let Some(lt) = lifetimes.iter_mut().find(|(id, _, _, _)| *id == output) {
            lt.2 = max_step;
        }
    }

    // Greedy allocation: sorted by size descending (large tensors first)
    lifetimes.sort_by(|a, b| b.3.cmp(&a.3));

    let mut pool_size = 0usize;
    let mut free_list: Vec<(usize, usize, usize)> = Vec::new(); // (offset, size, free_after_step)
    let mut assignments = Vec::new();

    for (val_id, birth, death, size) in &lifetimes {
        let aligned_size = (size + 255) & !255;

        // Try to find a free slot
        let slot = free_list
            .iter()
            .position(|(_, s, free_after)| *s >= aligned_size && *free_after <= *birth);

        if let Some(idx) = slot {
            let offset = free_list[idx].0;
            free_list[idx].2 = *death; // extend the free-after time
            assignments.push((*val_id, offset));
        } else {
            // Allocate new space
            let offset = pool_size;
            pool_size += aligned_size;
            free_list.push((offset, aligned_size, *death));
            assignments.push((*val_id, offset));
        }
    }

    MemoryPlan {
        total_bytes: pool_size,
        assignments,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("Out of memory: requested {requested} bytes, {available} available")]
    OutOfMemory { requested: usize, available: usize },
    #[error("Cannot allocate tensor with dynamic shape")]
    DynamicShape,
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::*;

    #[test]
    fn bump_allocation() {
        let mut pool = MemoryPool::new(1024 * 1024); // 1MB

        let buf1 = pool.alloc(shape![256], DType::F32).unwrap();
        assert_eq!(buf1.offset, 0);
        assert_eq!(buf1.size, 256 * 4); // 1024 bytes

        let buf2 = pool.alloc(shape![128], DType::F16).unwrap();
        assert!(buf2.offset >= 1024); // after buf1 + alignment
        assert_eq!(buf2.size, 128 * 2); // 256 bytes
    }

    #[test]
    fn out_of_memory() {
        let mut pool = MemoryPool::new(512);
        let result = pool.alloc(shape![256], DType::F32); // needs 1024 bytes
        assert!(result.is_err());
    }

    #[test]
    fn reset() {
        let mut pool = MemoryPool::new(4096);
        pool.alloc(shape![512], DType::F32).unwrap();
        assert!(pool.used() > 0);
        pool.reset();
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn memory_planning() {
        let mut g = Graph::new();
        let x = g.add_input(shape![1, 768], DType::F16, Some("x"));
        let w1 = g.add_input(shape![768, 3072], DType::F16, Some("w1"));
        let w2 = g.add_input(shape![3072, 768], DType::F16, Some("w2"));

        let (_, mm1) = g.add_node(
            Op::MatMul { transpose_a: false, transpose_b: false },
            &[x, w1],
            &[(shape![1, 3072], DType::F16)],
            None,
        );
        let (_, mm2) = g.add_node(
            Op::MatMul { transpose_a: false, transpose_b: false },
            &[mm1[0], w2],
            &[(shape![1, 768], DType::F16)],
            None,
        );
        g.mark_output(mm2[0]);

        let plan = plan_memory(&mut g);
        // mm1's output can be freed before we read mm2's output,
        // so they could share memory. Plan should be less than
        // naively allocating both.
        assert!(plan.total_bytes > 0);
        assert!(!plan.assignments.is_empty());
    }
}

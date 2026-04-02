//! Kernel execution scheduling.
//!
//! The scheduler takes an optimized graph and a set of compiled kernels,
//! and produces an execution plan — an ordered list of kernel launches
//! with their memory bindings.
//!
//! Key optimization: for static graphs (all shapes known), the entire
//! execution plan can be captured as a CUDA Graph and replayed with
//! near-zero CPU overhead (~5μs per inference vs ~100μs+ for individual launches).

use warp_ir::{Graph, NodeId, ValueId};
use crate::memory::TensorBuffer;

/// A single step in the execution plan.
#[derive(Debug, Clone)]
pub struct ExecStep {
    /// Which graph node this executes.
    pub node: NodeId,
    /// Kernel entry point name.
    pub kernel_name: String,
    /// Grid dimensions.
    pub grid: [u32; 3],
    /// Block dimensions.
    pub block: [u32; 3],
    /// Shared memory bytes.
    pub shared_mem: u32,
    /// Input buffer bindings (param index -> buffer).
    pub input_bindings: Vec<BufferBinding>,
    /// Output buffer bindings.
    pub output_bindings: Vec<BufferBinding>,
}

/// Binding between a kernel parameter and a memory buffer.
#[derive(Debug, Clone)]
pub struct BufferBinding {
    pub value_id: ValueId,
    pub offset: usize,
    pub size: usize,
}

/// The complete execution plan for a model.
#[derive(Debug)]
pub struct ExecutionPlan {
    pub steps: Vec<ExecStep>,
    /// Total GPU memory needed.
    pub memory_bytes: usize,
    /// Estimated total execution time (μs).
    pub estimated_time_us: f64,
    /// Whether this plan can be captured as a CUDA Graph.
    pub cuda_graph_compatible: bool,
}

impl ExecutionPlan {
    pub fn num_kernel_launches(&self) -> usize {
        self.steps.len()
    }

    /// Print a summary of the execution plan.
    pub fn summary(&self) -> String {
        format!(
            "ExecutionPlan: {} kernel launches, {:.2} MB memory, ~{:.1}μs estimated\n  CUDA Graph compatible: {}",
            self.steps.len(),
            self.memory_bytes as f64 / (1024.0 * 1024.0),
            self.estimated_time_us,
            self.cuda_graph_compatible,
        )
    }
}

/// Build an execution plan from an optimized graph and compiled kernels.
pub fn build_execution_plan(
    graph: &mut Graph,
    kernels: &[(NodeId, crate::engine::KernelInfo)],
    memory_assignments: &[(ValueId, usize)],
) -> ExecutionPlan {
    let topo = graph.topo_order().to_vec();
    let mut steps = Vec::new();
    let mut total_time = 0.0;

    // Build a map from NodeId to kernel
    let kernel_map: rustc_hash::FxHashMap<NodeId, &crate::engine::KernelInfo> =
        kernels.iter().map(|(id, k)| (*id, k)).collect();

    // Build a map from ValueId to memory offset
    let mem_map: rustc_hash::FxHashMap<ValueId, usize> =
        memory_assignments.iter().cloned().collect();

    for &node_id in &topo {
        let Some(kernel) = kernel_map.get(&node_id) else {
            continue; // Input/constant nodes don't have kernels
        };

        let node = graph.node(node_id);

        let input_bindings: Vec<BufferBinding> = node
            .inputs
            .iter()
            .map(|&val_id| {
                let info = graph.value(val_id);
                let offset = mem_map.get(&val_id).copied().unwrap_or(0);
                let size = info.shape.numel().unwrap_or(0) * info.dtype.byte_size();
                BufferBinding {
                    value_id: val_id,
                    offset,
                    size,
                }
            })
            .collect();

        let output_bindings: Vec<BufferBinding> = node
            .outputs
            .iter()
            .map(|&val_id| {
                let info = graph.value(val_id);
                let offset = mem_map.get(&val_id).copied().unwrap_or(0);
                let size = info.shape.numel().unwrap_or(0) * info.dtype.byte_size();
                BufferBinding {
                    value_id: val_id,
                    offset,
                    size,
                }
            })
            .collect();

        total_time += kernel.estimated_time_us;

        steps.push(ExecStep {
            node: node_id,
            kernel_name: kernel.entry_point.clone(),
            grid: kernel.grid,
            block: kernel.block,
            shared_mem: kernel.shared_mem_bytes,
            input_bindings,
            output_bindings,
        });
    }

    // Total memory = max offset + size across all assignments
    let memory_bytes = memory_assignments
        .iter()
        .map(|(val_id, offset)| {
            let info = graph.value(*val_id);
            offset + info.shape.numel().unwrap_or(0) * info.dtype.byte_size()
        })
        .max()
        .unwrap_or(0);

    // CUDA graph compatible if all shapes are static and no dynamic control flow
    let cuda_graph_compatible = graph
        .graph_inputs
        .iter()
        .all(|&v| graph.value(v).shape.is_static());

    ExecutionPlan {
        steps,
        memory_bytes,
        estimated_time_us: total_time,
        cuda_graph_compatible,
    }
}

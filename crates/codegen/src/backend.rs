//! Backend trait — the interface every codegen target implements.

use warp_ir::{DType, Graph, NodeId, Shape};

/// A compiled kernel ready for execution.
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    /// The kernel source/binary (PTX text, Metal shader, SPIR-V bytes).
    pub code: Vec<u8>,
    /// Entry point function name.
    pub entry_point: String,
    /// Grid dimensions (blocks).
    pub grid: [u32; 3],
    /// Block dimensions (threads per block).
    pub block: [u32; 3],
    /// Shared memory in bytes.
    pub shared_mem_bytes: u32,
    /// Human-readable description for debugging.
    pub description: String,
}

/// Configuration for kernel generation.
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Target compute capability (e.g., 89 for Ada, 90 for Hopper).
    pub compute_capability: Option<(u32, u32)>,
    /// Maximum threads per block.
    pub max_threads_per_block: u32,
    /// Maximum shared memory per block in bytes.
    pub max_shared_mem: u32,
    /// Whether to generate debug info.
    pub debug: bool,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            compute_capability: Some((8, 9)), // Ada Lovelace (4090)
            max_threads_per_block: 1024,
            max_shared_mem: 100 * 1024, // 100KB
            debug: false,
        }
    }
}

/// Errors during code generation.
#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    #[error("Unsupported operation: {0}")]
    UnsupportedOp(String),
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("Backend error: {0}")]
    BackendError(String),
}

/// The backend trait. Each GPU target implements this.
pub trait Backend: Send + Sync {
    /// Name of this backend (e.g., "cuda-ptx", "metal", "spirv").
    fn name(&self) -> &str;

    /// Generate a kernel for a single fused operation.
    fn generate_kernel(
        &self,
        graph: &Graph,
        node: NodeId,
        config: &KernelConfig,
    ) -> Result<CompiledKernel, CodegenError>;

    /// Generate kernels for an entire graph. Returns one kernel per
    /// "fusion group" (contiguous set of fused ops).
    fn generate_all(
        &self,
        graph: &Graph,
        config: &KernelConfig,
    ) -> Result<Vec<(NodeId, CompiledKernel)>, CodegenError> {
        let mut kernels = Vec::new();
        for (node_id, _node) in graph.nodes() {
            match self.generate_kernel(graph, node_id, config) {
                Ok(kernel) => kernels.push((node_id, kernel)),
                Err(CodegenError::UnsupportedOp(_)) => continue, // Skip data movement ops etc.
                Err(e) => return Err(e),
            }
        }
        Ok(kernels)
    }

    /// Estimate execution time in microseconds (for scheduling).
    fn estimate_cost(
        &self,
        graph: &Graph,
        node: NodeId,
        input_shapes: &[Shape],
        dtype: DType,
    ) -> f64;
}

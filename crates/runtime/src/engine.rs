//! The Warp inference engine — ties together IR, optimizer, codegen, and runtime.

use warp_codegen::{Backend, CompiledKernel, KernelConfig};
use warp_ir::Graph;

use crate::memory::plan_memory;
use crate::schedule::{build_execution_plan, ExecutionPlan};

/// Info about a compiled kernel (stored alongside the raw kernel).
#[derive(Debug, Clone)]
pub struct KernelInfo {
    pub entry_point: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_mem_bytes: u32,
    pub estimated_time_us: f64,
}

impl From<&CompiledKernel> for KernelInfo {
    fn from(k: &CompiledKernel) -> Self {
        Self {
            entry_point: k.entry_point.clone(),
            grid: k.grid,
            block: k.block,
            shared_mem_bytes: k.shared_mem_bytes,
            estimated_time_us: 0.0, // filled in by engine
        }
    }
}

/// The main inference engine.
pub struct Engine {
    config: KernelConfig,
}

impl Engine {
    pub fn new(config: KernelConfig) -> Self {
        Self { config }
    }

    /// Compile a graph: optimize → codegen → memory plan → execution plan.
    pub fn compile(
        &self,
        graph: &mut Graph,
        backend: &dyn Backend,
        opt_level: warp_optimizer::OptimizationLevel,
    ) -> Result<CompilationResult, EngineError> {
        // Step 1: Optimize
        let pipeline = warp_optimizer::PassPipeline::new(opt_level);
        let opt_stats = pipeline.run(graph);

        // Step 2: Generate kernels
        let compiled = backend
            .generate_all(graph, &self.config)
            .map_err(|e| EngineError::Codegen(e.to_string()))?;

        let kernels: Vec<_> = compiled
            .iter()
            .map(|(node_id, kernel)| {
                let mut info = KernelInfo::from(kernel);
                // Estimate cost
                let node = graph.node(*node_id);
                let output = node.outputs.first().copied();
                let dtype = output
                    .map(|o| graph.value(o).dtype)
                    .unwrap_or(warp_ir::DType::F32);
                info.estimated_time_us =
                    backend.estimate_cost(graph, *node_id, &[], dtype);
                (*node_id, info)
            })
            .collect();

        // Step 3: Memory planning
        let mem_plan = plan_memory(graph);

        // Step 4: Build execution plan
        let exec_plan =
            build_execution_plan(graph, &kernels, &mem_plan.assignments);

        Ok(CompilationResult {
            opt_stats,
            execution_plan: exec_plan,
            compiled_kernels: compiled
                .into_iter()
                .map(|(id, k)| (id, k))
                .collect(),
            memory_bytes: mem_plan.total_bytes,
        })
    }
}

/// Result of compiling a graph.
pub struct CompilationResult {
    pub opt_stats: warp_optimizer::pass::OptStats,
    pub execution_plan: ExecutionPlan,
    pub compiled_kernels: Vec<(warp_ir::NodeId, CompiledKernel)>,
    pub memory_bytes: usize,
}

impl CompilationResult {
    pub fn summary(&self) -> String {
        format!(
            "=== Warp Compilation Summary ===\n\
             Optimization:\n\
             \x20 MatMul+Bias fused:     {}\n\
             \x20 MatMul+Bias+Act fused: {}\n\
             \x20 Residual+Norm fused:   {}\n\
             \x20 Dead nodes:            {}\n\
             \x20 Passes run:            {}\n\
             \n\
             Codegen:\n\
             \x20 Kernels generated:     {}\n\
             \n\
             Memory:\n\
             \x20 Total GPU memory:      {:.2} MB\n\
             \n\
             Execution:\n\
             \x20 {}",
            self.opt_stats.matmul_bias_fused,
            self.opt_stats.matmul_bias_act_fused,
            self.opt_stats.residual_norm_fused,
            self.opt_stats.dead_nodes_found,
            self.opt_stats.passes_run,
            self.compiled_kernels.len(),
            self.memory_bytes as f64 / (1024.0 * 1024.0),
            self.execution_plan.summary(),
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Codegen error: {0}")]
    Codegen(String),
    #[error("Memory error: {0}")]
    Memory(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_codegen::ptx::PtxBackend;
    use warp_ir::*;
    use warp_optimizer::OptimizationLevel;

    #[test]
    fn end_to_end_compile() {
        // Build a simple FFN: x -> matmul -> add bias -> gelu
        let mut g = Graph::new();
        let x = g.add_input(shape![1, 768], DType::F32, Some("x"));
        let w = g.add_input(shape![768, 3072], DType::F32, Some("w"));
        let bias = g.add_input(shape![3072], DType::F32, Some("bias"));

        let (_, mm) = g.add_node(
            Op::MatMul { transpose_a: false, transpose_b: false },
            &[x, w],
            &[(shape![1, 3072], DType::F32)],
            None,
        );
        let (_, add) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[mm[0], bias],
            &[(shape![1, 3072], DType::F32)],
            None,
        );
        let (_, act) = g.add_node(
            Op::Activate { activation: Activation::GeluTanh },
            &[add[0]],
            &[(shape![1, 3072], DType::F32)],
            None,
        );
        g.mark_output(act[0]);

        let engine = Engine::new(KernelConfig::default());
        let backend = PtxBackend::new(89);
        let result = engine
            .compile(&mut g, &backend, OptimizationLevel::O1)
            .unwrap();

        let summary = result.summary();
        assert!(summary.contains("Warp Compilation Summary"));
        // After fusion: matmul+add+gelu should be fused into one op
        assert_eq!(result.opt_stats.matmul_bias_act_fused, 1);
    }
}

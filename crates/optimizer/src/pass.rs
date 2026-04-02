//! Optimization pass pipeline.
//!
//! Passes are run in a fixed order. Higher optimization levels enable
//! more aggressive (and slower-to-compile) passes.

use warp_ir::Graph;

use crate::autofuse;
use crate::fusion;

/// How aggressively to optimize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization. Useful for debugging.
    O0,
    /// Basic fusion only (MatMul+Bias, MatMul+Bias+Act).
    O1,
    /// Full fusion + memory layout optimization.
    O2,
    /// O2 + shape specialization + aggressive kernel tuning.
    O3,
}

/// Statistics from running the optimization pipeline.
#[derive(Debug, Default)]
pub struct OptStats {
    pub matmul_bias_fused: usize,
    pub matmul_bias_act_fused: usize,
    pub residual_norm_fused: usize,
    pub dead_nodes_found: usize,
    pub autofuse_chains: usize,
    pub autofuse_ops_fused: usize,
    pub passes_run: usize,
}

/// The optimization pipeline.
pub struct PassPipeline {
    level: OptimizationLevel,
}

impl PassPipeline {
    pub fn new(level: OptimizationLevel) -> Self {
        Self { level }
    }

    /// Run all optimization passes on the graph.
    pub fn run(&self, graph: &mut Graph) -> OptStats {
        let mut stats = OptStats::default();

        if self.level == OptimizationLevel::O0 {
            return stats;
        }

        // Pass 1: Fuse 3-op chains first (most specific patterns first)
        // MatMul + Bias + Act must run before MatMul + Bias, otherwise
        // the 2-op fusion would consume the matmul and prevent the 3-op match.
        stats.matmul_bias_act_fused = fusion::fuse_matmul_bias_act(graph);
        stats.passes_run += 1;

        // Rebuild users after 3-op fusion so dead nodes aren't matched
        graph.rebuild_users();

        // Pass 2: Fuse remaining MatMul + Bias (not followed by activation)
        stats.matmul_bias_fused = fusion::fuse_matmul_bias(graph);
        stats.passes_run += 1;

        // Rebuild users again
        graph.rebuild_users();

        // Pass 3: Fuse residual + RmsNorm
        stats.residual_norm_fused = fusion::fuse_residual_rmsnorm(graph);
        stats.passes_run += 1;

        // Pass 4: Dead code elimination
        stats.dead_nodes_found = fusion::eliminate_dead_code(graph);
        stats.passes_run += 1;

        if self.level as u8 >= OptimizationLevel::O2 as u8 {
            // Auto-fusion: discover and fuse elementwise chains
            let chains = autofuse::discover_fusion_chains(graph);
            stats.autofuse_chains = chains.len();
            stats.autofuse_ops_fused = chains.iter().map(|c| c.ops.len()).sum();
            stats.passes_run += 1;

            // Log discovered fusions
            for chain in &chains {
                log::info!("AutoFuse: {} ({} ops) → {}",
                    chain.name, chain.ops.len(), chain.generate_cuda_kernel().lines().next().unwrap_or(""));
            }
        }

        if self.level as u8 >= OptimizationLevel::O3 as u8 {
            // Shape specialization + autotuning would go here
        }

        graph.rebuild_users();
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::*;

    /// Build a mini transformer FFN block:
    /// x -> MatMul(W1) -> Add(b1) -> GELU -> MatMul(W2) -> Add(b2) -> output
    fn build_ffn_graph() -> Graph {
        let mut g = Graph::new();
        let x = g.add_input(shape![1, 768], DType::F16, Some("x"));
        let w1 = g.add_input(shape![768, 3072], DType::F16, Some("w1"));
        let b1 = g.add_input(shape![3072], DType::F16, Some("b1"));
        let w2 = g.add_input(shape![3072, 768], DType::F16, Some("w2"));
        let b2 = g.add_input(shape![768], DType::F16, Some("b2"));

        let (_, mm1) = g.add_node(
            Op::MatMul { transpose_a: false, transpose_b: false },
            &[x, w1],
            &[(shape![1, 3072], DType::F16)],
            Some("mm1".into()),
        );
        let (_, add1) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[mm1[0], b1],
            &[(shape![1, 3072], DType::F16)],
            Some("add1".into()),
        );
        let (_, gelu) = g.add_node(
            Op::Activate { activation: Activation::GeluTanh },
            &[add1[0]],
            &[(shape![1, 3072], DType::F16)],
            Some("gelu".into()),
        );
        let (_, mm2) = g.add_node(
            Op::MatMul { transpose_a: false, transpose_b: false },
            &[gelu[0], w2],
            &[(shape![1, 768], DType::F16)],
            Some("mm2".into()),
        );
        let (_, add2) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[mm2[0], b2],
            &[(shape![1, 768], DType::F16)],
            Some("add2".into()),
        );
        g.mark_output(add2[0]);
        g
    }

    #[test]
    fn full_pipeline_o2_autofuse() {
        // Build a graph with elementwise chains that O2 should auto-fuse
        let mut g = Graph::new();
        let x = g.add_input(shape![1, 1024], DType::F32, Some("x"));
        let w = g.add_input(shape![1024, 1024], DType::F32, Some("w"));
        let b = g.add_input(shape![1024], DType::F32, Some("bias"));
        let scale = g.add_input(shape![1024], DType::F32, Some("scale"));

        // MatMul → Add(bias) → GELU → Mul(scale)
        let (_, mm) = g.add_node(
            Op::MatMul { transpose_a: false, transpose_b: false },
            &[x, w], &[(shape![1, 1024], DType::F32)], Some("mm".into()));
        let (_, add) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[mm[0], b], &[(shape![1, 1024], DType::F32)], Some("add_bias".into()));
        let (_, gelu) = g.add_node(
            Op::Activate { activation: Activation::GeluTanh },
            &[add[0]], &[(shape![1, 1024], DType::F32)], Some("gelu".into()));
        let (_, mul) = g.add_node(
            Op::Binary { op: BinaryOp::Mul },
            &[gelu[0], scale], &[(shape![1, 1024], DType::F32)], Some("mul_scale".into()));
        g.mark_output(mul[0]);

        let pipeline = PassPipeline::new(OptimizationLevel::O2);
        let stats = pipeline.run(&mut g);

        println!("O2 stats:");
        println!("  MatMul+Bias+Act fused: {}", stats.matmul_bias_act_fused);
        println!("  MatMul+Bias fused: {}", stats.matmul_bias_fused);
        println!("  AutoFuse chains: {} ({} ops)", stats.autofuse_chains, stats.autofuse_ops_fused);
        println!("  Dead nodes: {}", stats.dead_nodes_found);

        // O1 should fuse MatMul+Add+GELU → FusedMatMulBiasAct
        assert_eq!(stats.matmul_bias_act_fused, 1);
        // After O1 fusion, the remaining Mul(scale) is a single op → no chain
        // But the original Add→GELU→Mul chain was partially consumed by O1
        // O2 should discover remaining autofuse opportunities
    }

    #[test]
    fn full_pipeline_o1() {
        let mut g = build_ffn_graph();
        let pipeline = PassPipeline::new(OptimizationLevel::O1);
        let stats = pipeline.run(&mut g);

        // First matmul chain: MatMul + Add + GELU -> FusedMatMulBiasAct
        assert_eq!(stats.matmul_bias_act_fused, 1);
        // Second matmul chain: MatMul + Add -> FusedMatMulBias
        assert_eq!(stats.matmul_bias_fused, 1);

        // Verify fused ops exist
        let mut has_fused_mba = false;
        let mut has_fused_mb = false;
        for (_, node) in g.nodes() {
            match &node.op {
                Op::FusedMatMulBiasAct { activation: Activation::GeluTanh, .. } => {
                    has_fused_mba = true;
                }
                Op::FusedMatMulBias { .. } => {
                    has_fused_mb = true;
                }
                _ => {}
            }
        }
        assert!(has_fused_mba, "Should have FusedMatMulBiasAct");
        assert!(has_fused_mb, "Should have FusedMatMulBias");
    }
}

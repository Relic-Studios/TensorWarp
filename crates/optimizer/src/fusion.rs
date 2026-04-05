//! Fusion passes — the primary optimization in Warp.
//!
//! Each fusion pass matches a pattern of unfused ops and replaces them
//! with a single fused op that maps to one GPU kernel launch.
//!
//! Fusion is where Warp beats TRT: we fuse more aggressively because
//! we target transformer-family models specifically.

use warp_ir::*;

use crate::pattern::{find_all_matches, PatternNode};

/// Fuse MatMul + Add (bias) into FusedMatMulBias.
///
/// Pattern: `Add(MatMul(A, B), bias)` → `FusedMatMulBias(A, B, bias)`
///
/// This is the most common fusion in transformer inference —
/// every linear layer is a matmul followed by a bias add.
pub fn fuse_matmul_bias(graph: &mut Graph) -> usize {
    let pattern = PatternNode::op(|op| matches!(op, Op::Binary { op: BinaryOp::Add }))
        .input(PatternNode::op(|op| matches!(op, Op::MatMul { .. })));

    let matches = find_all_matches(graph, &pattern);
    let count = matches.len();

    for m in matches {
        let add_node_id = m.matched_nodes[0];
        let mm_node_id = m.matched_nodes[1];

        let mm = graph.node(mm_node_id);
        let (transpose_a, transpose_b) = match &mm.op {
            Op::MatMul { transpose_a, transpose_b } => (*transpose_a, *transpose_b),
            _ => unreachable!(),
        };

        // Inputs: A (mm input 0), B (mm input 1), bias (add input 1)
        let a = graph.node(mm_node_id).inputs[0];
        let b = graph.node(mm_node_id).inputs[1];
        let bias = graph.node(add_node_id).inputs[1];

        // Replace the add node with the fused op, rewiring inputs
        graph.replace_op(
            add_node_id,
            Op::FusedMatMulBias {
                transpose_a,
                transpose_b,
            },
        );
        graph.replace_inputs(add_node_id, &[a, b, bias]);

        // The matmul node is now dead (no users) — will be cleaned up by DCE
    }

    count
}

/// Fuse MatMul + Add + Activation into FusedMatMulBiasAct.
///
/// Pattern: `Activate(Add(MatMul(A, B), bias))` → `FusedMatMulBiasAct(A, B, bias)`
///
/// Covers: Linear + GELU, Linear + SiLU, etc.
pub fn fuse_matmul_bias_act(graph: &mut Graph) -> usize {
    let pattern =
        PatternNode::op(|op| matches!(op, Op::Activate { .. })).input(
            PatternNode::op(|op| matches!(op, Op::Binary { op: BinaryOp::Add }))
                .input(PatternNode::op(|op| matches!(op, Op::MatMul { .. }))),
        );

    let matches = find_all_matches(graph, &pattern);
    let count = matches.len();

    for m in matches {
        let act_node_id = m.matched_nodes[0];
        let add_node_id = m.matched_nodes[1];
        let mm_node_id = m.matched_nodes[2];

        let activation = match &graph.node(act_node_id).op {
            Op::Activate { activation } => *activation,
            _ => unreachable!(),
        };

        let (transpose_a, transpose_b) = match &graph.node(mm_node_id).op {
            Op::MatMul { transpose_a, transpose_b } => (*transpose_a, *transpose_b),
            _ => unreachable!(),
        };

        let a = graph.node(mm_node_id).inputs[0];
        let b = graph.node(mm_node_id).inputs[1];
        let bias = graph.node(add_node_id).inputs[1];

        graph.replace_op(
            act_node_id,
            Op::FusedMatMulBiasAct {
                transpose_a,
                transpose_b,
                activation,
            },
        );
        graph.replace_inputs(act_node_id, &[a, b, bias]);
    }

    count
}

/// Fuse Add (residual) + RmsNorm into FusedResidualRmsNorm.
///
/// Pattern: `RmsNorm(Add(residual, X), gamma, ?)` → `FusedResidualRmsNorm(residual, X, gamma)`
///
/// This is the #1 memory-bandwidth bottleneck in transformer inference.
/// The residual add and norm both read the same tensor — fusing them
/// means one read instead of two. At 7B scale this saves ~100μs/layer.
pub fn fuse_residual_rmsnorm(graph: &mut Graph) -> usize {
    let pattern = PatternNode::op(|op| matches!(op, Op::RmsNorm { .. }))
        .input(PatternNode::op(|op| matches!(op, Op::Binary { op: BinaryOp::Add })));

    let matches = find_all_matches(graph, &pattern);
    let count = matches.len();

    for m in matches {
        let norm_node_id = m.matched_nodes[0];
        let add_node_id = m.matched_nodes[1];

        let eps = match &graph.node(norm_node_id).op {
            Op::RmsNorm { eps } => *eps,
            _ => unreachable!(),
        };

        let residual = graph.node(add_node_id).inputs[0];
        let x = graph.node(add_node_id).inputs[1];
        let gamma = graph.node(norm_node_id).inputs[1];

        graph.replace_op(norm_node_id, Op::FusedResidualRmsNorm { eps });
        graph.replace_inputs(norm_node_id, &[residual, x, gamma]);
    }

    count
}

/// Dead code elimination — remove nodes whose outputs are never used.
pub fn eliminate_dead_code(graph: &mut Graph) -> usize {
    let mut live = rustc_hash::FxHashSet::default();

    // Start from graph outputs, walk backwards to find all live nodes
    let mut worklist: Vec<NodeId> = graph
        .graph_outputs
        .iter()
        .map(|&v| graph.value_producer(v))
        .collect();

    while let Some(node_id) = worklist.pop() {
        if !live.insert(node_id) {
            continue;
        }
        for &input in &graph.node(node_id).inputs {
            worklist.push(graph.value_producer(input));
        }
    }

    // Replace dead nodes with Identity (no-op) and clear their inputs
    // This effectively removes them from the graph without compaction
    let total = graph.num_nodes();
    let mut dead = 0;
    for i in 0..total {
        let node_id = NodeId(i as u32);
        if !live.contains(&node_id) {
            // Check if this node is a graph input producer (don't kill those)
            let is_input = graph.node(node_id).inputs.is_empty();
            if !is_input {
                graph.replace_op(node_id, Op::Identity);
                graph.replace_inputs(node_id, &[]);
                dead += 1;
            }
        }
    }
    dead
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_bias_fusion() {
        let mut g = Graph::new();
        let x = g.add_input(shape![2, 768], DType::F16, Some("x"));
        let w = g.add_input(shape![768, 3072], DType::F16, Some("w"));
        let bias = g.add_input(shape![3072], DType::F16, Some("bias"));

        let (_, mm) = g.add_node(
            Op::MatMul { transpose_a: false, transpose_b: false },
            &[x, w],
            &[(shape![2, 3072], DType::F16)],
            None,
        );
        let (_, add) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[mm[0], bias],
            &[(shape![2, 3072], DType::F16)],
            None,
        );
        g.mark_output(add[0]);

        let fused = fuse_matmul_bias(&mut g);
        assert_eq!(fused, 1);

        // The add node should now be a FusedMatMulBias
        let output_producer = g.value_producer(g.graph_outputs[0]);
        let node = g.node(output_producer);
        assert!(matches!(node.op, Op::FusedMatMulBias { .. }));
        assert_eq!(node.inputs.len(), 3); // A, B, bias
    }

    #[test]
    fn test_matmul_bias_act_fusion() {
        let mut g = Graph::new();
        let x = g.add_input(shape![2, 768], DType::F16, Some("x"));
        let w = g.add_input(shape![768, 3072], DType::F16, Some("w"));
        let bias = g.add_input(shape![3072], DType::F16, Some("bias"));

        let (_, mm) = g.add_node(
            Op::MatMul { transpose_a: false, transpose_b: false },
            &[x, w],
            &[(shape![2, 3072], DType::F16)],
            None,
        );
        let (_, add) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[mm[0], bias],
            &[(shape![2, 3072], DType::F16)],
            None,
        );
        let (_, act) = g.add_node(
            Op::Activate { activation: Activation::GeluTanh },
            &[add[0]],
            &[(shape![2, 3072], DType::F16)],
            None,
        );
        g.mark_output(act[0]);

        let fused = fuse_matmul_bias_act(&mut g);
        assert_eq!(fused, 1);

        let output_producer = g.value_producer(g.graph_outputs[0]);
        let node = g.node(output_producer);
        assert!(matches!(
            node.op,
            Op::FusedMatMulBiasAct {
                activation: Activation::GeluTanh,
                ..
            }
        ));
    }

    #[test]
    fn test_residual_rmsnorm_fusion() {
        let mut g = Graph::new();
        let residual = g.add_input(shape![2, 768], DType::F16, Some("residual"));
        let x = g.add_input(shape![2, 768], DType::F16, Some("x"));
        let gamma = g.add_input(shape![768], DType::F16, Some("gamma"));
        // RmsNorm takes 3 inputs in our IR: [X, gamma, ?]
        // For this pattern, the first input to RmsNorm is the Add output
        let dummy = g.add_input(shape![768], DType::F16, Some("dummy"));

        let (_, add) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[residual, x],
            &[(shape![2, 768], DType::F16)],
            None,
        );
        let (_, norm) = g.add_node(
            Op::RmsNorm { eps: 1e-6 },
            &[add[0], gamma, dummy],
            &[(shape![2, 768], DType::F16)],
            None,
        );
        g.mark_output(norm[0]);

        let fused = fuse_residual_rmsnorm(&mut g);
        assert_eq!(fused, 1);

        let output_producer = g.value_producer(g.graph_outputs[0]);
        let node = g.node(output_producer);
        assert!(matches!(node.op, Op::FusedResidualRmsNorm { eps } if (eps - 1e-6).abs() < 1e-9));
        assert_eq!(node.inputs.len(), 3); // residual, x, gamma
    }
}

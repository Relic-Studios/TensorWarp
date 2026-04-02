//! Pattern matching engine for subgraph detection.
//!
//! Patterns describe subgraphs to match in the IR. When a pattern matches,
//! the matched nodes can be replaced with a fused equivalent.
//!
//! This is the core mechanism that turns `MatMul -> Add -> GELU` into
//! `FusedMatMulBiasAct` — which maps to a single GPU kernel launch.

use warp_ir::{Graph, NodeId, Op};

/// A predicate that matches a single node's operation.
pub type OpMatcher = Box<dyn Fn(&Op) -> bool + Send + Sync>;

/// A node in a pattern graph. Patterns are trees (not DAGs) rooted
/// at the "output" end — we match backwards from a candidate node.
pub struct PatternNode {
    /// Predicate to match the operation.
    pub matcher: OpMatcher,
    /// Pattern nodes that must match this node's inputs.
    /// Empty means "any inputs" (leaf of the pattern).
    pub input_patterns: Vec<PatternNode>,
    /// If true, the matched node's output must have exactly one consumer
    /// (otherwise fusing it would change semantics for other consumers).
    pub single_user: bool,
}

impl PatternNode {
    /// Create a pattern node matching a specific op predicate.
    pub fn op(matcher: impl Fn(&Op) -> bool + Send + Sync + 'static) -> Self {
        Self {
            matcher: Box::new(matcher),
            input_patterns: Vec::new(),
            single_user: true,
        }
    }

    /// Add an input pattern (matched against this node's inputs in order).
    pub fn input(mut self, pattern: PatternNode) -> Self {
        self.input_patterns.push(pattern);
        self
    }

    /// Don't require single-user for this node.
    pub fn allow_multi_user(mut self) -> Self {
        self.single_user = false;
        self
    }
}

/// Result of a successful pattern match.
pub struct PatternMatch {
    /// Matched nodes, ordered from root (output) to leaves (inputs).
    pub matched_nodes: Vec<NodeId>,
}

/// Try to match a pattern starting at `root_node` in the graph.
/// Returns the match if successful.
pub fn match_pattern(graph: &Graph, root_node: NodeId, pattern: &PatternNode) -> Option<PatternMatch> {
    let mut matched = Vec::new();
    // Root node: skip single_user check (it's the node being replaced).
    if match_recursive(graph, root_node, pattern, &mut matched, true) {
        Some(PatternMatch {
            matched_nodes: matched,
        })
    } else {
        None
    }
}

fn match_recursive(
    graph: &Graph,
    node_id: NodeId,
    pattern: &PatternNode,
    matched: &mut Vec<NodeId>,
    is_root: bool,
) -> bool {
    let node = graph.node(node_id);

    // Check op predicate
    if !(pattern.matcher)(&node.op) {
        return false;
    }

    // Check single-user constraint (all outputs must have single user).
    // Skip for root — the root's output is the fusion output, not an interior edge.
    if !is_root && pattern.single_user {
        for &output in &node.outputs {
            if !graph.has_single_user(output) {
                return false;
            }
        }
    }

    matched.push(node_id);

    // Match input patterns
    if pattern.input_patterns.is_empty() {
        return true; // Leaf — any inputs OK
    }

    if node.inputs.len() < pattern.input_patterns.len() {
        return false;
    }

    for (i, input_pattern) in pattern.input_patterns.iter().enumerate() {
        let input_val = node.inputs[i];
        let producer = graph.value_producer(input_val);
        if !match_recursive(graph, producer, input_pattern, matched, false) {
            return false;
        }
    }

    true
}

/// Scan the entire graph for all non-overlapping matches of a pattern.
pub fn find_all_matches(graph: &Graph, pattern: &PatternNode) -> Vec<PatternMatch> {
    let mut matches = Vec::new();
    let mut used_nodes = rustc_hash::FxHashSet::default();

    for (node_id, node) in graph.nodes() {
        if used_nodes.contains(&node_id) {
            continue;
        }
        // Skip dead nodes — outputs with no users and not graph outputs.
        let is_live = node.outputs.iter().any(|&out| {
            !graph.value_users(out).is_empty() || graph.graph_outputs.contains(&out)
        });
        if !is_live && !node.outputs.is_empty() {
            continue;
        }
        if let Some(m) = match_pattern(graph, node_id, pattern) {
            for &n in &m.matched_nodes {
                used_nodes.insert(n);
            }
            matches.push(m);
        }
    }

    matches
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::*;

    #[test]
    fn match_matmul_add() {
        // Build: Input -> MatMul -> Add (bias)
        let mut g = Graph::new();
        let x = g.add_input(shape![2, 4], DType::F16, Some("x"));
        let w = g.add_input(shape![4, 8], DType::F16, Some("w"));
        let bias = g.add_input(shape![8], DType::F16, Some("bias"));

        let (mm_id, mm_outs) = g.add_node(
            Op::MatMul { transpose_a: false, transpose_b: false },
            &[x, w],
            &[(shape![2, 8], DType::F16)],
            Some("mm".into()),
        );
        let (add_id, add_outs) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[mm_outs[0], bias],
            &[(shape![2, 8], DType::F16)],
            Some("add".into()),
        );
        g.mark_output(add_outs[0]);

        // Pattern: Add(MatMul(*), *)
        let pattern = PatternNode::op(|op| matches!(op, Op::Binary { op: BinaryOp::Add }))
            .input(PatternNode::op(|op| matches!(op, Op::MatMul { .. })));

        let m = match_pattern(&g, add_id, &pattern);
        assert!(m.is_some());
        let m = m.unwrap();
        assert_eq!(m.matched_nodes[0], add_id);
        assert_eq!(m.matched_nodes[1], mm_id);
    }

    #[test]
    fn no_match_when_multi_user() {
        // MatMul output used by both Add and another node
        let mut g = Graph::new();
        let x = g.add_input(shape![2, 4], DType::F16, Some("x"));
        let w = g.add_input(shape![4, 8], DType::F16, Some("w"));
        let bias = g.add_input(shape![8], DType::F16, Some("bias"));

        let (_, mm_outs) = g.add_node(
            Op::MatMul { transpose_a: false, transpose_b: false },
            &[x, w],
            &[(shape![2, 8], DType::F16)],
            None,
        );
        let (add_id, add_outs) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[mm_outs[0], bias],
            &[(shape![2, 8], DType::F16)],
            None,
        );
        // Second consumer of mm output
        let (_, _) = g.add_node(
            Op::Activate { activation: Activation::Relu },
            &[mm_outs[0]],
            &[(shape![2, 8], DType::F16)],
            None,
        );
        g.mark_output(add_outs[0]);

        // Pattern requires single-user on MatMul output
        let pattern = PatternNode::op(|op| matches!(op, Op::Binary { op: BinaryOp::Add }))
            .input(PatternNode::op(|op| matches!(op, Op::MatMul { .. })));

        let m = match_pattern(&g, add_id, &pattern);
        assert!(m.is_none()); // Should NOT match because mm has 2 users
    }
}

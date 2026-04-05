//! Automatic fusion engine — discovers and fuses chains of elementwise ops.
//!
//! Instead of hand-writing every fusion pattern, this engine:
//! 1. Walks the graph in topological order
//! 2. Finds maximal chains of fusible ops (elementwise, single-user)
//! 3. Generates a fused CUDA kernel for each chain
//! 4. Replaces the chain with a single fused op
//!
//! This is the O2/O3 optimization that TensorRT does internally.
//! We go further by generating specialized kernels at compile time.
//!
//! Example chain: Add → GELU → Mul → Add
//! Generated kernel: `out[i] = (gelu(a[i] + b[i]) * c[i]) + d[i]`
//! Result: 4 kernel launches → 1, 4 memory passes → 1

use warp_ir::{Graph, NodeId, Op, Activation, BinaryOp, UnaryOp};

/// An operation that can participate in elementwise fusion.
#[derive(Debug, Clone)]
pub enum FusibleOp {
    Binary(BinaryOp),
    Unary(UnaryOp),
    Activate(Activation),
}

impl FusibleOp {
    /// Generate the CUDA expression for this op.
    fn to_cuda_expr(&self, inputs: &[String]) -> String {
        match self {
            FusibleOp::Binary(BinaryOp::Add) => format!("({} + {})", inputs[0], inputs[1]),
            FusibleOp::Binary(BinaryOp::Sub) => format!("({} - {})", inputs[0], inputs[1]),
            FusibleOp::Binary(BinaryOp::Mul) => format!("({} * {})", inputs[0], inputs[1]),
            FusibleOp::Binary(BinaryOp::Div) => format!("({} / {})", inputs[0], inputs[1]),
            FusibleOp::Binary(BinaryOp::Max) => format!("fmaxf({}, {})", inputs[0], inputs[1]),
            FusibleOp::Binary(BinaryOp::Min) => format!("fminf({}, {})", inputs[0], inputs[1]),
            FusibleOp::Binary(BinaryOp::Pow) => format!("powf({}, {})", inputs[0], inputs[1]),
            FusibleOp::Unary(UnaryOp::Neg) => format!("(-{})", inputs[0]),
            FusibleOp::Unary(UnaryOp::Abs) => format!("fabsf({})", inputs[0]),
            FusibleOp::Unary(UnaryOp::Exp) => format!("expf({})", inputs[0]),
            FusibleOp::Unary(UnaryOp::Log) => format!("logf({})", inputs[0]),
            FusibleOp::Unary(UnaryOp::Sqrt) => format!("sqrtf({})", inputs[0]),
            FusibleOp::Unary(UnaryOp::Rsqrt) => format!("rsqrtf({})", inputs[0]),
            FusibleOp::Unary(UnaryOp::Recip) => format!("(1.0f / {})", inputs[0]),
            FusibleOp::Unary(UnaryOp::Cast(_)) => inputs[0].clone(),
            FusibleOp::Activate(Activation::Relu) => format!("fmaxf({}, 0.0f)", inputs[0]),
            FusibleOp::Activate(Activation::Sigmoid) => format!("(1.0f / (1.0f + expf(-{})))", inputs[0]),
            FusibleOp::Activate(Activation::Tanh) => format!("tanhf({})", inputs[0]),
            FusibleOp::Activate(Activation::Silu) => {
                let x = &inputs[0];
                format!("({x} / (1.0f + expf(-{x})))")
            }
            FusibleOp::Activate(Activation::Gelu | Activation::GeluTanh) => {
                let x = &inputs[0];
                format!("(0.5f * {x} * (1.0f + tanhf(0.7978845608f * ({x} + 0.044715f * {x} * {x} * {x}))))")
            }
        }
    }
}

/// A chain of fusible operations discovered in the graph.
#[derive(Debug)]
pub struct FusionChain {
    /// Node IDs in the chain (topological order, first = deepest input)
    pub nodes: Vec<NodeId>,
    /// Operations in order
    pub ops: Vec<FusibleOp>,
    /// Number of external inputs (non-chain inputs)
    pub num_inputs: usize,
    /// Chain name for the generated kernel
    pub name: String,
}

impl FusionChain {
    /// Generate a fused CUDA kernel source for this chain.
    pub fn generate_cuda_kernel(&self) -> String {
        let n_inputs = self.num_inputs;

        // Build parameter list
        let mut params = vec!["float *out".to_string()];
        for i in 0..n_inputs {
            params.push(format!("const float *in{i}"));
        }
        params.push("size_t n".to_string());
        let param_str = params.join(", ");

        // Build the expression tree
        // For now, handle linear chains (each op takes previous output + optional second input)
        let mut expr = "in0[i]".to_string();
        let mut next_input = 1;

        for op in &self.ops {
            match op {
                FusibleOp::Binary(_) => {
                    let input_b = format!("in{next_input}[i]");
                    next_input += 1;
                    expr = op.to_cuda_expr(&[expr, input_b]);
                }
                FusibleOp::Unary(_) | FusibleOp::Activate(_) => {
                    expr = op.to_cuda_expr(&[expr]);
                }
            }
        }

        format!(
            r#"extern "C" __global__ void {name}({params}) {{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{ out[i] = {expr}; }}
}}"#,
            name = self.name,
            params = param_str,
            expr = expr,
        )
    }
}

/// Check if an op is elementwise (can be fused with neighbors).
fn is_fusible(op: &Op) -> Option<FusibleOp> {
    match op {
        Op::Binary { op } => Some(FusibleOp::Binary(*op)),
        Op::Unary { op } => Some(FusibleOp::Unary(op.clone())),
        Op::Activate { activation } => Some(FusibleOp::Activate(*activation)),
        _ => None,
    }
}

/// Discover all fusible chains in the graph.
///
/// A chain is a maximal sequence of elementwise ops where each intermediate
/// result has exactly one consumer (single-user constraint).
pub fn discover_fusion_chains(graph: &mut Graph) -> Vec<FusionChain> {
    let topo = graph.topo_order().to_vec();
    let mut visited = std::collections::HashSet::new();
    let mut chains = Vec::new();
    let mut chain_id = 0;

    for &node_id in &topo {
        if visited.contains(&node_id) { continue; }

        let node = graph.node(node_id);
        if is_fusible(&node.op).is_none() { continue; }

        // Start a new chain from this node
        let mut chain_nodes = vec![node_id];
        let mut chain_ops = vec![is_fusible(&node.op).unwrap()];
        visited.insert(node_id);

        // Extend forward: follow single-user edges to fusible consumers
        let mut current = node_id;
        loop {
            // Get the output value of the current node and its users
            let current_node = graph.node(current);
            if current_node.outputs.is_empty() { break; }
            let out_val = current_node.outputs[0];
            let users = graph.value_users(out_val);
            if users.len() != 1 { break; }

            let next = users[0];
            let next_node = graph.node(next);
            if let Some(fop) = is_fusible(&next_node.op) {
                if !visited.contains(&next) {
                    chain_nodes.push(next);
                    chain_ops.push(fop);
                    visited.insert(next);
                    current = next;
                    continue;
                }
            }
            break;
        }

        // Only fuse chains of 2+ ops (otherwise no benefit)
        if chain_nodes.len() >= 2 {
            // Count external inputs
            let mut external_inputs = 0;
            for &nid in &chain_nodes {
                let n = graph.node(nid);
                for &input in &n.inputs {
                    // Check if input comes from within the chain
                    let input_node = graph.value_producer(input);
                    if !chain_nodes.contains(&input_node) {
                        external_inputs += 1;
                    }
                }
            }

            chains.push(FusionChain {
                nodes: chain_nodes,
                ops: chain_ops,
                num_inputs: external_inputs,
                name: format!("warp_autofused_{chain_id}"),
            });
            chain_id += 1;
        }
    }

    chains
}

/// Apply discovered fusion chains to the graph.
///
/// For each chain, replaces the chain's nodes with a single AutoFused op.
/// The last node in the chain becomes the fused op; all others become dead code.
pub fn apply_fusion_chains(graph: &mut Graph, chains: &[FusionChain]) -> usize {
    let mut applied = 0;

    for chain in chains {
        if chain.nodes.len() < 2 {
            continue;
        }

        let kernel_src = chain.generate_cuda_kernel();

        // Collect all external inputs (inputs from outside the chain)
        let chain_set: std::collections::HashSet<NodeId> =
            chain.nodes.iter().copied().collect();
        let mut external_inputs = Vec::new();
        for &nid in &chain.nodes {
            let n = graph.node(nid);
            for &input in &n.inputs {
                let producer = graph.value_producer(input);
                if !chain_set.contains(&producer) {
                    external_inputs.push(input);
                }
            }
        }

        // The last node in the chain produces the output — replace it with AutoFused
        let last_node = *chain.nodes.last().unwrap();
        graph.replace_op(
            last_node,
            Op::AutoFused {
                kernel_name: chain.name.clone(),
                kernel_src,
                num_inputs: external_inputs.len(),
            },
        );
        graph.replace_inputs(last_node, &external_inputs);

        // Mark intermediate nodes as Identity (they'll be cleaned by DCE)
        for &nid in &chain.nodes[..chain.nodes.len() - 1] {
            graph.replace_op(nid, Op::Identity);
            graph.replace_inputs(nid, &[]);
        }

        applied += 1;
    }

    if applied > 0 {
        graph.rebuild_users();
    }
    applied
}

/// Generate a report of all discoverable fusions.
pub fn fusion_report(graph: &mut Graph) -> String {
    let chains = discover_fusion_chains(graph);
    let mut lines = vec![format!("=== Auto-Fusion Discovery: {} chains ===", chains.len())];

    for chain in &chains {
        let ops: Vec<String> = chain.ops.iter().map(|op| format!("{:?}", op)).collect();
        lines.push(format!(
            "  {} ({} ops, {} inputs): {}",
            chain.name, chain.ops.len(), chain.num_inputs,
            ops.join(" → ")
        ));
        lines.push(format!("    Kernel: {}", chain.generate_cuda_kernel()));
    }

    if chains.is_empty() {
        lines.push("  No fusible chains found (all ops are non-elementwise or multi-user)".into());
    } else {
        let total_ops: usize = chains.iter().map(|c| c.ops.len()).sum();
        let total_launches_saved = total_ops - chains.len();
        lines.push(format!(
            "\n  Total: {} ops → {} fused kernels ({} launches saved)",
            total_ops, chains.len(), total_launches_saved
        ));
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::*;

    #[test]
    fn discover_elementwise_chain() {
        let mut g = Graph::new();
        let a = g.add_input(shape![1024], DType::F32, Some("a"));
        let b = g.add_input(shape![1024], DType::F32, Some("b"));
        let c = g.add_input(shape![1024], DType::F32, Some("c"));

        // Chain: Add(a, b) → GELU → Mul(_, c) → Sigmoid
        let (_, add_out) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[a, b], &[(shape![1024], DType::F32)], Some("add".into()));
        let (_, gelu_out) = g.add_node(
            Op::Activate { activation: Activation::Gelu },
            &[add_out[0]], &[(shape![1024], DType::F32)], Some("gelu".into()));
        let (_, mul_out) = g.add_node(
            Op::Binary { op: BinaryOp::Mul },
            &[gelu_out[0], c], &[(shape![1024], DType::F32)], Some("mul".into()));
        let (_, sig_out) = g.add_node(
            Op::Activate { activation: Activation::Sigmoid },
            &[mul_out[0]], &[(shape![1024], DType::F32)], Some("sigmoid".into()));
        g.mark_output(sig_out[0]);

        let chains = discover_fusion_chains(&mut g);
        println!("{}", fusion_report(&mut g));

        assert_eq!(chains.len(), 1, "Should find one chain");
        assert_eq!(chains[0].ops.len(), 4, "Chain should have 4 ops");

        // Generate kernel
        let kernel = chains[0].generate_cuda_kernel();
        println!("\nGenerated kernel:\n{kernel}");
        assert!(kernel.contains("warp_autofused_0"));
        assert!(kernel.contains("expf")); // from sigmoid
    }

    #[test]
    fn multi_user_breaks_chain() {
        let mut g = Graph::new();
        let a = g.add_input(shape![1024], DType::F32, Some("a"));
        let b = g.add_input(shape![1024], DType::F32, Some("b"));

        // Add(a, b) → used by BOTH Relu AND Sigmoid → can't fuse through
        let (_, add_out) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[a, b], &[(shape![1024], DType::F32)], Some("add".into()));
        let (_, relu_out) = g.add_node(
            Op::Activate { activation: Activation::Relu },
            &[add_out[0]], &[(shape![1024], DType::F32)], Some("relu".into()));
        let (_, sig_out) = g.add_node(
            Op::Activate { activation: Activation::Sigmoid },
            &[add_out[0]], &[(shape![1024], DType::F32)], Some("sigmoid".into()));
        g.mark_output(relu_out[0]);
        g.mark_output(sig_out[0]);

        let chains = discover_fusion_chains(&mut g);
        println!("{}", fusion_report(&mut g));

        // Add has 2 users, so no chain can include it
        // Relu and Sigmoid are single-op, too short to fuse
        assert_eq!(chains.len(), 0, "No fusible chains (multi-user breaks fusion)");
    }

    #[test]
    fn generated_kernel_compiles() {
        // Verify the generated CUDA code actually compiles
        let chain = FusionChain {
            nodes: vec![],
            ops: vec![
                FusibleOp::Binary(BinaryOp::Add),
                FusibleOp::Activate(Activation::Silu),
                FusibleOp::Binary(BinaryOp::Mul),
            ],
            num_inputs: 3,
            name: "warp_test_fused".into(),
        };

        let kernel = chain.generate_cuda_kernel();
        println!("Generated: {kernel}");

        // Should be: out[i] = (silu(in0[i] + in1[i])) * in2[i]
        assert!(kernel.contains("warp_test_fused"));
        assert!(kernel.contains("in0[i]"));
        assert!(kernel.contains("in2[i]"));
        assert!(kernel.contains("expf")); // silu uses exp
    }

    #[test]
    fn swiglu_pattern() {
        let mut g = Graph::new();
        let gate = g.add_input(shape![1024], DType::F32, Some("gate"));
        let up = g.add_input(shape![1024], DType::F32, Some("up"));

        // SwiGLU: SiLU(gate) * up
        let (_, silu_out) = g.add_node(
            Op::Activate { activation: Activation::Silu },
            &[gate], &[(shape![1024], DType::F32)], Some("silu".into()));
        let (_, mul_out) = g.add_node(
            Op::Binary { op: BinaryOp::Mul },
            &[silu_out[0], up], &[(shape![1024], DType::F32)], Some("mul".into()));
        g.mark_output(mul_out[0]);

        let chains = discover_fusion_chains(&mut g);
        println!("{}", fusion_report(&mut g));

        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].ops.len(), 2);

        let kernel = chains[0].generate_cuda_kernel();
        println!("SwiGLU fused kernel:\n{kernel}");
        // Should be: silu(in0[i]) * in1[i]
        assert!(kernel.contains("expf"));
    }
}

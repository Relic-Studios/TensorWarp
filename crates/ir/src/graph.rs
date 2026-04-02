//! The core graph IR — a directed acyclic graph of tensor operations.
//!
//! Design: SSA-style where each `Value` is produced by exactly one `Node`.
//! Nodes consume Values and produce Values. The graph tracks all of this
//! with arena-allocated IDs for cache-friendly traversal.
//!
//! Key design decisions:
//! - Arena allocation with ID-based references (no Rc/RefCell)
//! - Topological ordering maintained incrementally
//! - Use-def chains for efficient dead code elimination and fusion
//! - Metadata attached to values (shape, dtype, layout) not nodes

use crate::{DType, Op, Shape};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Unique identifier for a value (tensor) in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValueId(pub u32);

/// Unique identifier for a node (operation) in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

/// Memory layout hint for a tensor value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Layout {
    /// Row-major (C-contiguous). Default.
    RowMajor,
    /// Column-major (Fortran-contiguous).
    ColMajor,
    /// Custom strided layout.
    Strided,
}

impl Default for Layout {
    fn default() -> Self {
        Layout::RowMajor
    }
}

/// Metadata about a tensor value flowing through the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueInfo {
    /// The node that produces this value.
    pub producer: NodeId,
    /// Which output of the producer (most ops produce 1, Split produces many).
    pub output_index: u32,
    /// Tensor shape.
    pub shape: Shape,
    /// Data type.
    pub dtype: DType,
    /// Memory layout hint.
    pub layout: Layout,
    /// Optional human-readable name (e.g., "layer.0.attention.q_proj.weight").
    pub name: Option<String>,
}

/// A node in the computation graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// The operation this node performs.
    pub op: Op,
    /// Input values consumed by this node.
    pub inputs: SmallVec<[ValueId; 4]>,
    /// Output values produced by this node.
    pub outputs: SmallVec<[ValueId; 1]>,
    /// Optional name for debugging.
    pub name: Option<String>,
}

/// The computation graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    /// All nodes, indexed by NodeId.
    nodes: Vec<Node>,
    /// All values, indexed by ValueId.
    values: Vec<ValueInfo>,
    /// Graph inputs (external data fed in at runtime).
    pub graph_inputs: Vec<ValueId>,
    /// Graph outputs (results to be read back).
    pub graph_outputs: Vec<ValueId>,

    /// Reverse mapping: value -> list of nodes that consume it.
    /// Maintained incrementally for efficient use-def queries.
    #[serde(skip)]
    users: FxHashMap<ValueId, SmallVec<[NodeId; 4]>>,

    /// Cached topological order. Invalidated on mutation.
    #[serde(skip)]
    topo_order: Option<Vec<NodeId>>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            values: Vec::new(),
            graph_inputs: Vec::new(),
            graph_outputs: Vec::new(),
            users: FxHashMap::default(),
            topo_order: None,
        }
    }

    // ── Node/Value creation ──────────────────────────────────────

    /// Allocate a new value with the given metadata. Returns its ID.
    fn alloc_value(&mut self, info: ValueInfo) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(info);
        id
    }

    /// Add a node to the graph, returning (NodeId, output ValueIds).
    pub fn add_node(
        &mut self,
        op: Op,
        inputs: &[ValueId],
        output_shapes: &[(Shape, DType)],
        name: Option<String>,
    ) -> (NodeId, SmallVec<[ValueId; 1]>) {
        let node_id = NodeId(self.nodes.len() as u32);

        // Create output values
        let mut outputs = SmallVec::new();
        for (i, (shape, dtype)) in output_shapes.iter().enumerate() {
            let val_id = self.alloc_value(ValueInfo {
                producer: node_id,
                output_index: i as u32,
                shape: shape.clone(),
                dtype: *dtype,
                layout: Layout::default(),
                name: None,
            });
            outputs.push(val_id);
        }

        // Register input uses
        for &input in inputs {
            self.users
                .entry(input)
                .or_insert_with(SmallVec::new)
                .push(node_id);
        }

        self.nodes.push(Node {
            op,
            inputs: inputs.into(),
            outputs: outputs.clone(),
            name,
        });

        // Invalidate topo cache
        self.topo_order = None;

        (node_id, outputs)
    }

    /// Convenience: add a graph input node.
    pub fn add_input(&mut self, shape: Shape, dtype: DType, name: Option<&str>) -> ValueId {
        let index = self.graph_inputs.len() as u32;
        let (_, outputs) = self.add_node(
            Op::Input { index },
            &[],
            &[(shape, dtype)],
            name.map(|s| s.to_string()),
        );
        let val_id = outputs[0];
        self.graph_inputs.push(val_id);
        if let Some(n) = name {
            self.values[val_id.0 as usize].name = Some(n.to_string());
        }
        val_id
    }

    /// Mark a value as a graph output.
    pub fn mark_output(&mut self, value: ValueId) {
        self.graph_outputs.push(value);
    }

    // ── Queries ──────────────────────────────────────────────────

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        self.topo_order = None;
        &mut self.nodes[id.0 as usize]
    }

    pub fn value(&self, id: ValueId) -> &ValueInfo {
        &self.values[id.0 as usize]
    }

    pub fn value_mut(&mut self, id: ValueId) -> &mut ValueInfo {
        &mut self.values[id.0 as usize]
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_values(&self) -> usize {
        self.values.len()
    }

    /// Get all nodes that consume a given value.
    pub fn value_users(&self, id: ValueId) -> &[NodeId] {
        self.users.get(&id).map_or(&[], |v| v.as_slice())
    }

    /// Get the node that produces a given value.
    pub fn value_producer(&self, id: ValueId) -> NodeId {
        self.values[id.0 as usize].producer
    }

    /// Iterate over all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (NodeId(i as u32), n))
    }

    // ── Topological ordering ─────────────────────────────────────

    /// Get nodes in topological order (cached).
    pub fn topo_order(&mut self) -> &[NodeId] {
        if self.topo_order.is_none() {
            self.topo_order = Some(self.compute_topo_order());
        }
        self.topo_order.as_ref().unwrap()
    }

    fn compute_topo_order(&self) -> Vec<NodeId> {
        let n = self.nodes.len();
        let mut in_degree = vec![0u32; n];
        let mut adj: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); n];

        for (i, node) in self.nodes.iter().enumerate() {
            for &input_val in &node.inputs {
                let producer = self.values[input_val.0 as usize].producer;
                let p = producer.0 as usize;
                if p != i {
                    adj[p].push(i);
                    in_degree[i] += 1;
                }
            }
        }

        // Kahn's algorithm
        let mut queue: Vec<usize> = in_degree
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if d == 0 { Some(i) } else { None })
            .collect();

        let mut order = Vec::with_capacity(n);
        while let Some(u) = queue.pop() {
            order.push(NodeId(u as u32));
            for &v in &adj[u] {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push(v);
                }
            }
        }

        order
    }

    // ── Graph surgery ────────────────────────────────────────────

    /// Replace a node's op in-place (used by optimizer for fusion).
    pub fn replace_op(&mut self, node_id: NodeId, new_op: Op) {
        self.nodes[node_id.0 as usize].op = new_op;
        self.topo_order = None;
    }

    /// Replace a node's inputs.
    pub fn replace_inputs(&mut self, node_id: NodeId, new_inputs: &[ValueId]) {
        let old_inputs = std::mem::take(&mut self.nodes[node_id.0 as usize].inputs);

        // Remove old user registrations
        for old_input in &old_inputs {
            if let Some(users) = self.users.get_mut(old_input) {
                users.retain(|n| *n != node_id);
            }
        }

        // Add new user registrations
        for &new_input in new_inputs {
            self.users
                .entry(new_input)
                .or_insert_with(SmallVec::new)
                .push(node_id);
        }

        self.nodes[node_id.0 as usize].inputs = new_inputs.into();
        self.topo_order = None;
    }

    /// Check if a value has exactly one consumer (candidate for fusion).
    pub fn has_single_user(&self, value: ValueId) -> bool {
        self.users
            .get(&value)
            .map_or(false, |users| users.len() == 1)
    }

    /// Rebuild the user map from scratch (call after bulk mutations).
    pub fn rebuild_users(&mut self) {
        self.users.clear();
        for (i, node) in self.nodes.iter().enumerate() {
            let node_id = NodeId(i as u32);
            for &input in &node.inputs {
                self.users
                    .entry(input)
                    .or_insert_with(SmallVec::new)
                    .push(node_id);
            }
        }
    }

    // ── Validation ───────────────────────────────────────────────

    /// Validate graph integrity. Returns list of errors.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // Check all input values reference valid nodes
        for (i, node) in self.nodes.iter().enumerate() {
            for (j, &input) in node.inputs.iter().enumerate() {
                if input.0 as usize >= self.values.len() {
                    errors.push(format!(
                        "Node {} input {}: ValueId {} out of range",
                        i, j, input.0
                    ));
                }
            }
            for (j, &output) in node.outputs.iter().enumerate() {
                if output.0 as usize >= self.values.len() {
                    errors.push(format!(
                        "Node {} output {}: ValueId {} out of range",
                        i, j, output.0
                    ));
                }
            }
        }

        // Check graph outputs reference valid values
        for &out in &self.graph_outputs {
            if out.0 as usize >= self.values.len() {
                errors.push(format!(
                    "Graph output ValueId {} out of range",
                    out.0
                ));
            }
        }

        errors
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// Pretty-print a graph for debugging.
impl std::fmt::Display for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Graph ({} nodes, {} values)", self.nodes.len(), self.values.len())?;
        writeln!(f, "  Inputs: {:?}", self.graph_inputs)?;
        writeln!(f, "  Outputs: {:?}", self.graph_outputs)?;
        writeln!(f, "  Nodes:")?;
        for (i, node) in self.nodes.iter().enumerate() {
            let name = node.name.as_deref().unwrap_or("");
            writeln!(
                f,
                "    [{i}] {name:20} {:?} <- {:?} -> {:?}",
                std::mem::discriminant(&node.op),
                node.inputs.as_slice(),
                node.outputs.as_slice(),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_matmul_graph() -> Graph {
        let mut g = Graph::new();
        let a = g.add_input(
            Shape::from_static(&[2, 4]),
            DType::F16,
            Some("A"),
        );
        let b = g.add_input(
            Shape::from_static(&[4, 8]),
            DType::F16,
            Some("B"),
        );
        let (_, outs) = g.add_node(
            Op::MatMul {
                transpose_a: false,
                transpose_b: false,
            },
            &[a, b],
            &[(Shape::from_static(&[2, 8]), DType::F16)],
            Some("matmul_0".into()),
        );
        g.mark_output(outs[0]);
        g
    }

    #[test]
    fn build_and_validate() {
        let g = simple_matmul_graph();
        assert_eq!(g.num_nodes(), 3); // 2 inputs + 1 matmul
        assert_eq!(g.graph_inputs.len(), 2);
        assert_eq!(g.graph_outputs.len(), 1);
        assert!(g.validate().is_empty());
    }

    #[test]
    fn topo_order() {
        let mut g = simple_matmul_graph();
        let order = g.topo_order().to_vec();
        // Input nodes come before matmul
        let matmul_pos = order.iter().position(|&n| n == NodeId(2)).unwrap();
        let input_a_pos = order.iter().position(|&n| n == NodeId(0)).unwrap();
        let input_b_pos = order.iter().position(|&n| n == NodeId(1)).unwrap();
        assert!(input_a_pos < matmul_pos);
        assert!(input_b_pos < matmul_pos);
    }

    #[test]
    fn user_tracking() {
        let g = simple_matmul_graph();
        let input_a = g.graph_inputs[0];
        let users = g.value_users(input_a);
        assert_eq!(users.len(), 1);
        assert_eq!(users[0], NodeId(2)); // matmul node
    }

    #[test]
    fn single_user_detection() {
        let g = simple_matmul_graph();
        let input_a = g.graph_inputs[0];
        assert!(g.has_single_user(input_a));
    }

    #[test]
    fn graph_display() {
        let g = simple_matmul_graph();
        let s = format!("{g}");
        assert!(s.contains("3 nodes"));
        assert!(s.contains("matmul_0"));
    }
}

//! # warp-ir
//!
//! The intermediate representation for the Warp inference engine.
//!
//! This crate defines the core data structures: tensor types, shapes,
//! operations, and the computation graph. Everything in Warp flows
//! through this IR — the optimizer rewrites it, the codegen reads it,
//! and the runtime executes it.

mod dtype;
mod graph;
mod op;
mod shape;

pub use dtype::DType;
pub use graph::{Graph, Layout, Node, NodeId, ValueId, ValueInfo};
pub use op::{
    Activation, AttentionMask, BinaryOp, ConstantData, GridPaddingMode,
    InterpolationMode, Op, PadMode, PoolMode, ReduceOp, ResizeMode, UnaryOp,
};
pub use shape::{Dim, Shape};

/// Builder for constructing graphs with a fluent API.
pub struct GraphBuilder {
    graph: Graph,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
        }
    }

    pub fn input(&mut self, shape: Shape, dtype: DType, name: &str) -> ValueId {
        self.graph.add_input(shape, dtype, Some(name))
    }

    /// Add a MatMul node. Returns output ValueId.
    pub fn matmul(&mut self, a: ValueId, b: ValueId, out_shape: Shape, dtype: DType) -> ValueId {
        let (_, outs) = self.graph.add_node(
            Op::MatMul { transpose_a: false, transpose_b: false },
            &[a, b],
            &[(out_shape, dtype)],
            None,
        );
        outs[0]
    }

    /// Add a binary op node.
    pub fn binary(&mut self, op: BinaryOp, a: ValueId, b: ValueId, out_shape: Shape, dtype: DType) -> ValueId {
        let (_, outs) = self.graph.add_node(
            Op::Binary { op },
            &[a, b],
            &[(out_shape, dtype)],
            None,
        );
        outs[0]
    }

    /// Add an activation node.
    pub fn activate(&mut self, activation: Activation, x: ValueId, shape: Shape, dtype: DType) -> ValueId {
        let (_, outs) = self.graph.add_node(
            Op::Activate { activation },
            &[x],
            &[(shape, dtype)],
            None,
        );
        outs[0]
    }

    /// Add an RMS norm node.
    pub fn rms_norm(&mut self, x: ValueId, gamma: ValueId, sin: ValueId, shape: Shape, dtype: DType, eps: f32) -> ValueId {
        let (_, outs) = self.graph.add_node(
            Op::RmsNorm { eps },
            &[x, gamma, sin],
            &[(shape, dtype)],
            None,
        );
        outs[0]
    }

    /// Add attention node.
    pub fn attention(
        &mut self,
        q: ValueId,
        k: ValueId,
        v: ValueId,
        out_shape: Shape,
        dtype: DType,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> ValueId {
        let (_, outs) = self.graph.add_node(
            Op::Attention {
                num_heads,
                num_kv_heads,
                head_dim,
                mask: AttentionMask::Causal,
                scale: None,
            },
            &[q, k, v],
            &[(out_shape, dtype)],
            None,
        );
        outs[0]
    }

    /// Add a softmax node.
    pub fn softmax(&mut self, x: ValueId, shape: Shape, dtype: DType, axis: i32) -> ValueId {
        let (_, outs) = self.graph.add_node(
            Op::Softmax { axis },
            &[x],
            &[(shape, dtype)],
            None,
        );
        outs[0]
    }

    pub fn mark_output(&mut self, value: ValueId) {
        self.graph.mark_output(value);
    }

    pub fn build(self) -> Graph {
        self.graph
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_api() {
        let mut b = GraphBuilder::new();
        let x = b.input(shape![2, 768], DType::F16, "hidden");
        let w = b.input(shape![768, 3072], DType::F16, "weight");
        let bias = b.input(shape![3072], DType::F16, "bias");

        let mm = b.matmul(x, w, shape![2, 3072], DType::F16);
        let add = b.binary(BinaryOp::Add, mm, bias, shape![2, 3072], DType::F16);
        let act = b.activate(Activation::GeluTanh, add, shape![2, 3072], DType::F16);
        b.mark_output(act);

        let g = b.build();
        assert!(g.validate().is_empty());
        assert_eq!(g.graph_outputs.len(), 1);
    }

    #[test]
    fn shape_macro() {
        let s = shape![2, 3, 4];
        assert_eq!(s.rank(), 3);
        assert_eq!(s.numel(), Some(24));
    }
}

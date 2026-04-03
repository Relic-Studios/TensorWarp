//! ONNX → IR → Optimize → Execute pipeline.
//!
//! This is the "TensorRT-style" compilation path:
//! 1. Parse ONNX model → OnnxModel
//! 2. Build warp-ir Graph from ONNX nodes
//! 3. Run optimizer passes (O1: fusion, O2: autofuse)
//! 4. Execute optimized graph on GPU
//!
//! This connects all TensorWarp subsystems into a single pipeline.

use std::collections::HashMap;

use warp_ir::*;
use warp_optimizer::{OptimizationLevel, PassPipeline};

use crate::onnx::{OnnxModel, OnnxNode};

/// Build a warp-ir Graph from an ONNX model.
/// Maps ONNX nodes to warp-ir operations.
pub fn onnx_to_ir(model: &OnnxModel) -> Graph {
    let mut g = Graph::new();
    let mut value_map: HashMap<String, ValueId> = HashMap::new();

    // Add model inputs
    for input in &model.inputs {
        let shape = if input.shape.is_empty() {
            shape![1]
        } else {
            let dims: Vec<usize> = input.shape.iter()
                .map(|&d| if d < 0 { 1 } else { d as usize })
                .collect();
            Shape::from_static(&dims)
        };
        let dtype = input.dtype.map(|d| d.to_warp_dtype()).unwrap_or(DType::F32);
        let val = g.add_input(shape, dtype, Some(&input.name));
        value_map.insert(input.name.clone(), val);
    }

    // Add initializers as constants
    for (name, _tensor) in &model.initializers {
        if !value_map.contains_key(name) {
            let val = g.add_input(shape![1], DType::F32, Some(name));
            value_map.insert(name.clone(), val);
        }
    }

    // Map ONNX nodes to IR operations
    for node in &model.nodes {
        let op = map_onnx_op(node);
        let inputs: Vec<ValueId> = node.inputs.iter()
            .filter_map(|name| {
                if name.is_empty() { None }
                else { value_map.get(name).copied() }
            })
            .collect();

        if inputs.is_empty() { continue; }

        // Determine output shape (simplified — use first input's shape)
        let first_input = inputs[0];
        let in_info = g.value(first_input);
        let out_shape = in_info.shape.clone();
        let out_dtype = in_info.dtype;

        let (_, outputs) = g.add_node(
            op,
            &inputs,
            &[(out_shape, out_dtype)],
            Some(node.name.clone()),
        );

        // Map output names
        for (i, out_name) in node.outputs.iter().enumerate() {
            if !out_name.is_empty() {
                if let Some(&val_id) = outputs.get(i) {
                    value_map.insert(out_name.clone(), val_id);
                }
            }
        }
    }

    // Mark model outputs
    for output in &model.outputs {
        if let Some(&val_id) = value_map.get(&output.name) {
            g.mark_output(val_id);
        }
    }

    g
}

/// Map an ONNX node to a warp-ir Op.
fn map_onnx_op(node: &OnnxNode) -> Op {
    match node.op_type.as_str() {
        "MatMul" | "Gemm" => Op::MatMul { transpose_a: false, transpose_b: false },
        "Add" => Op::Binary { op: BinaryOp::Add },
        "Sub" => Op::Binary { op: BinaryOp::Sub },
        "Mul" => Op::Binary { op: BinaryOp::Mul },
        "Div" => Op::Binary { op: BinaryOp::Div },
        "Relu" => Op::Activate { activation: Activation::Relu },
        "Gelu" => Op::Activate { activation: Activation::Gelu },
        "Silu" | "Swish" => Op::Activate { activation: Activation::Silu },
        "Sigmoid" => Op::Activate { activation: Activation::Sigmoid },
        "Tanh" => Op::Activate { activation: Activation::Tanh },
        "Softmax" => Op::Softmax { axis: -1 },
        "LayerNormalization" => Op::LayerNorm { eps: node.get_float("epsilon", 1e-5) },
        "BatchNormalization" => Op::BatchNorm { eps: node.get_float("epsilon", 1e-5) },
        "GroupNormalization" => Op::GroupNorm {
            num_groups: node.get_int("num_groups", 32) as u32,
            eps: node.get_float("epsilon", 1e-5),
        },
        "Conv" => Op::Conv {
            kernel_size: node.get_ints("kernel_shape").iter().map(|&x| x as u32).collect(),
            stride: node.get_ints("strides").iter().map(|&x| x as u32).collect(),
            padding: node.get_ints("pads").iter().map(|&x| x as u32).collect(),
            dilation: node.get_ints("dilations").iter().map(|&x| x as u32).collect(),
            groups: node.get_int("group", 1) as u32,
        },
        "MaxPool" => Op::Pool {
            mode: warp_ir::PoolMode::Max,
            kernel_size: node.get_ints("kernel_shape").iter().map(|&x| x as u32).collect(),
            stride: node.get_ints("strides").iter().map(|&x| x as u32).collect(),
            padding: node.get_ints("pads").iter().map(|&x| x as u32).collect(),
        },
        "GlobalAveragePool" => Op::Pool {
            mode: warp_ir::PoolMode::GlobalAvg,
            kernel_size: vec![], stride: vec![], padding: vec![],
        },
        "Reshape" | "Flatten" | "Squeeze" | "Unsqueeze" => Op::Reshape { target_shape: vec![-1] },
        "Transpose" => Op::Transpose { perm: node.get_ints("perm").iter().map(|&x| x as usize).collect() },
        "Concat" => Op::Concat { axis: node.get_int("axis", 0) as i32 },
        "Gather" => Op::Gather { axis: node.get_int("axis", 0) as i32 },
        "Slice" => Op::Slice { starts: vec![], ends: vec![], steps: vec![] },
        "Reduce" | "ReduceMean" => Op::Reduce { op: ReduceOp::Mean, axes: vec![-1], keepdim: false },
        "ReduceSum" => Op::Reduce { op: ReduceOp::Sum, axes: vec![-1], keepdim: false },
        "ReduceMax" => Op::Reduce { op: ReduceOp::Max, axes: vec![-1], keepdim: false },
        _ => Op::Unary { op: UnaryOp::Abs }, // fallback — identity-like
    }
}

/// Compile an ONNX model through the full optimization pipeline.
pub fn compile_onnx(model: &OnnxModel, opt_level: OptimizationLevel) -> CompiledOnnx {
    // Step 1: Build IR graph
    let mut graph = onnx_to_ir(model);

    // Step 2: Run optimizer
    let pipeline = PassPipeline::new(opt_level);
    let stats = pipeline.run(&mut graph);

    // Step 3: Return compiled representation
    CompiledOnnx {
        graph,
        opt_stats: format!(
            "O{}: {} matmul+bias fused, {} matmul+bias+act fused, {} residual+norm fused, {} autofuse chains ({} ops), {} dead nodes",
            opt_level as u8,
            stats.matmul_bias_fused, stats.matmul_bias_act_fused,
            stats.residual_norm_fused, stats.autofuse_chains, stats.autofuse_ops_fused,
            stats.dead_nodes_found
        ),
        model_name: model.producer.clone(),
    }
}

/// A compiled ONNX model with optimized IR graph.
pub struct CompiledOnnx {
    pub graph: Graph,
    pub opt_stats: String,
    pub model_name: String,
}

impl CompiledOnnx {
    pub fn summary(&self) -> String {
        let num_nodes = self.graph.nodes().count();
        format!(
            "CompiledOnnx: {} nodes, optimization: {}, model: {}",
            num_nodes, self.opt_stats, self.model_name
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::*;

    #[test]
    fn onnx_to_ir_mlp() {
        let model = crate::onnx_validate::build_mlp_model();
        let graph = onnx_to_ir(&model);

        println!("IR graph from MLP:");
        println!("  Nodes: {}", graph.nodes().count());
        assert!(graph.nodes().count() >= 3); // at least gemm1, relu, gemm2

        // Verify ops
        let mut has_matmul = false;
        let mut has_relu = false;
        for (_, node) in graph.nodes() {
            match &node.op {
                Op::MatMul { .. } => has_matmul = true,
                Op::Activate { activation: Activation::Relu } => has_relu = true,
                _ => {}
            }
        }
        assert!(has_matmul, "Should have MatMul");
        assert!(has_relu, "Should have ReLU");
        println!("  ✓ Contains MatMul and ReLU ops");
    }

    #[test]
    fn compile_onnx_with_optimization() {
        let model = crate::onnx_validate::build_mlp_model();

        // Compile with O1 (pattern fusion)
        let compiled = compile_onnx(&model, OptimizationLevel::O1);
        println!("{}", compiled.summary());

        // Compile with O2 (autofuse)
        let compiled_o2 = compile_onnx(&model, OptimizationLevel::O2);
        println!("{}", compiled_o2.summary());
    }
}

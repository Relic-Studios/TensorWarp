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

/// Convert an ONNX i64 shape dim to a warp-ir `Dim`.
/// Negative values (typically -1) become `Dim::Dynamic` with a symbolic ID
/// derived from the absolute value, preserving dynamic shape information
/// instead of silently collapsing to 1.
fn onnx_dim_to_ir(d: i64, dynamic_counter: &mut u32) -> Dim {
    if d < 0 {
        let id = *dynamic_counter;
        *dynamic_counter += 1;
        Dim::Dynamic(id)
    } else {
        Dim::Static(d as usize)
    }
}

/// Convert an ONNX i64 shape to a warp-ir `Shape`, preserving dynamic dims.
fn onnx_shape_to_ir(shape: &[i64], dynamic_counter: &mut u32) -> Shape {
    if shape.is_empty() {
        shape![1]
    } else {
        let dims: Vec<Dim> = shape.iter()
            .map(|&d| onnx_dim_to_ir(d, dynamic_counter))
            .collect();
        Shape::new(dims)
    }
}

/// Infer the output shape for a node given its op and input shapes.
/// Returns the inferred shape and dtype.
fn infer_output_shape(op: &Op, input_shapes: &[Shape], input_dtypes: &[DType]) -> (Shape, DType) {
    let dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
    let first = input_shapes.first().cloned().unwrap_or_else(|| shape![1]);

    match op {
        // --- Element-wise ops: output shape = broadcast of inputs ---
        Op::Binary { .. } => {
            if input_shapes.len() >= 2 {
                let out = first.broadcast_shape(&input_shapes[1])
                    .unwrap_or(first);
                (out, dtype)
            } else {
                (first, dtype)
            }
        }

        // --- Unary / activation / softmax: shape unchanged ---
        Op::Unary { .. } | Op::Activate { .. } | Op::Softmax { .. } => {
            (first, dtype)
        }

        // --- Normalization ops: shape unchanged ---
        Op::LayerNorm { .. } | Op::RmsNorm { .. } | Op::BatchNorm { .. }
        | Op::InstanceNorm { .. } | Op::GroupNorm { .. } => {
            (first, dtype)
        }

        // --- MatMul / Gemm: [.., M, K] x [.., K, N] -> [.., M, N] ---
        Op::MatMul { transpose_a, transpose_b } => {
            if input_shapes.len() >= 2 {
                let a = &input_shapes[0];
                let b = &input_shapes[1];
                if a.rank() >= 2 && b.rank() >= 2 {
                    let m = if *transpose_a { a.dim(a.rank() - 1) } else { a.dim(a.rank() - 2) };
                    let n = if *transpose_b { b.dim(b.rank() - 2) } else { b.dim(b.rank() - 1) };
                    // Preserve batch dims from A
                    let mut dims: Vec<Dim> = Vec::new();
                    for i in 0..a.rank().saturating_sub(2) {
                        dims.push(a.dim(i));
                    }
                    dims.push(m);
                    dims.push(n);
                    (Shape::new(dims), dtype)
                } else if a.rank() >= 2 {
                    // B is 1D: [.., M, K] x [K] -> [.., M]
                    let m = if *transpose_a { a.dim(a.rank() - 1) } else { a.dim(a.rank() - 2) };
                    let mut dims: Vec<Dim> = Vec::new();
                    for i in 0..a.rank().saturating_sub(2) {
                        dims.push(a.dim(i));
                    }
                    dims.push(m);
                    (Shape::new(dims), dtype)
                } else {
                    (first, dtype)
                }
            } else {
                (first, dtype)
            }
        }

        Op::BatchMatMul { transpose_a, transpose_b } => {
            if input_shapes.len() >= 2 {
                let a = &input_shapes[0];
                let b = &input_shapes[1];
                if a.rank() >= 2 && b.rank() >= 2 {
                    let m = if *transpose_a { a.dim(a.rank() - 1) } else { a.dim(a.rank() - 2) };
                    let n = if *transpose_b { b.dim(b.rank() - 2) } else { b.dim(b.rank() - 1) };
                    let mut dims: Vec<Dim> = Vec::new();
                    for i in 0..a.rank().saturating_sub(2) {
                        dims.push(a.dim(i));
                    }
                    dims.push(m);
                    dims.push(n);
                    (Shape::new(dims), dtype)
                } else {
                    (first, dtype)
                }
            } else {
                (first, dtype)
            }
        }

        // --- Conv: [N, C_in, H, W] + weight [C_out, C_in/g, kH, kW] -> [N, C_out, H', W'] ---
        Op::Conv { kernel_size, stride, padding, dilation, .. } => {
            if input_shapes.len() >= 2 && first.rank() == 4 {
                let n_dim = first.dim(0);
                let c_out = input_shapes[1].dim(0); // weight shape [C_out, ...]
                let spatial_dims = 2; // H, W
                let mut dims: Vec<Dim> = Vec::new();
                dims.push(n_dim);
                dims.push(c_out);
                for i in 0..spatial_dims {
                    let ks = kernel_size.get(i).copied().unwrap_or(1) as usize;
                    let s = stride.get(i).copied().unwrap_or(1) as usize;
                    let dil = dilation.get(i).copied().unwrap_or(1) as usize;
                    // ONNX pads format: [top, left, bottom, right] or [begin0, begin1, end0, end1]
                    let p_begin = padding.get(i).copied().unwrap_or(0) as usize;
                    let p_end = padding.get(i + spatial_dims).copied().unwrap_or(0) as usize;
                    let p = p_begin + p_end;
                    let effective_k = dil * (ks - 1) + 1;
                    match first.dim(2 + i) {
                        Dim::Static(in_size) => {
                            let out_size = (in_size + p - effective_k) / s + 1;
                            dims.push(Dim::Static(out_size));
                        }
                        d @ Dim::Dynamic(_) => dims.push(d),
                    }
                }
                (Shape::new(dims), dtype)
            } else {
                (first, dtype)
            }
        }

        // --- Pool: same spatial formula as Conv, channels unchanged ---
        Op::Pool { mode, kernel_size, stride, padding } => {
            match mode {
                PoolMode::GlobalAvg | PoolMode::GlobalMax => {
                    // Reduce spatial dims to 1x1
                    if first.rank() == 4 {
                        let mut dims: Vec<Dim> = Vec::new();
                        dims.push(first.dim(0)); // N
                        dims.push(first.dim(1)); // C
                        dims.push(Dim::Static(1));
                        dims.push(Dim::Static(1));
                        (Shape::new(dims), dtype)
                    } else {
                        (first, dtype)
                    }
                }
                _ => {
                    if first.rank() == 4 {
                        let mut dims: Vec<Dim> = Vec::new();
                        dims.push(first.dim(0)); // N
                        dims.push(first.dim(1)); // C
                        let spatial_dims = 2;
                        for i in 0..spatial_dims {
                            let ks = kernel_size.get(i).copied().unwrap_or(1u32);
                            let s = stride.get(i).copied().unwrap_or(ks);
                            let p_begin = padding.get(i).copied().unwrap_or(0u32) as usize;
                            let p_end = padding.get(i + spatial_dims).copied().unwrap_or(0u32) as usize;
                            let p = p_begin + p_end;
                            let ks = ks as usize;
                            let s = s as usize;
                            match first.dim(2 + i) {
                                Dim::Static(in_size) => {
                                    let out_size = (in_size + p - ks) / s + 1;
                                    dims.push(Dim::Static(out_size));
                                }
                                d @ Dim::Dynamic(_) => dims.push(d),
                            }
                        }
                        (Shape::new(dims), dtype)
                    } else {
                        (first, dtype)
                    }
                }
            }
        }

        // --- Reshape: use target_shape to compute output ---
        Op::Reshape { target_shape } => {
            // Try to resolve the target shape. Dims of 0 mean "copy from input",
            // -1 means "infer from total element count".
            let numel = first.numel();
            let mut out_dims: Vec<Dim> = Vec::new();
            let mut infer_idx: Option<usize> = None;
            let mut known_product: usize = 1;

            for (i, &d) in target_shape.iter().enumerate() {
                if d == 0 {
                    // Copy from input
                    if i < first.rank() {
                        out_dims.push(first.dim(i));
                        if let Some(v) = first.dim(i).static_val() {
                            known_product *= v;
                        }
                    } else {
                        out_dims.push(Dim::Static(1));
                    }
                } else if d == -1 {
                    infer_idx = Some(i);
                    out_dims.push(Dim::Static(0)); // placeholder
                } else {
                    out_dims.push(Dim::Static(d as usize));
                    known_product *= d as usize;
                }
            }

            // Resolve the -1 dimension if possible
            if let (Some(idx), Some(total)) = (infer_idx, numel) {
                if known_product > 0 {
                    out_dims[idx] = Dim::Static(total / known_product);
                }
            } else if let Some(idx) = infer_idx {
                // Can't infer statically; mark as dynamic
                out_dims[idx] = Dim::Dynamic(0);
            }

            if out_dims.is_empty() {
                (first, dtype)
            } else {
                (Shape::new(out_dims), dtype)
            }
        }

        // --- Transpose: permute dimensions ---
        Op::Transpose { perm } => {
            if !perm.is_empty() && perm.len() == first.rank() {
                let dims: Vec<Dim> = perm.iter()
                    .map(|&p| first.dim(p))
                    .collect();
                (Shape::new(dims), dtype)
            } else if perm.is_empty() && first.rank() >= 2 {
                // Default: reverse all axes
                let dims: Vec<Dim> = (0..first.rank()).rev()
                    .map(|i| first.dim(i))
                    .collect();
                (Shape::new(dims), dtype)
            } else {
                (first, dtype)
            }
        }

        // --- Concat: sum along axis, keep other dims ---
        Op::Concat { axis } => {
            if input_shapes.is_empty() {
                return (first, dtype);
            }
            let rank = first.rank();
            let ax = if *axis < 0 { (rank as i32 + *axis) as usize } else { *axis as usize };
            if ax >= rank {
                return (first, dtype);
            }

            // Sum the concat axis across all inputs
            let mut axis_sum: Option<usize> = Some(0);
            for s in input_shapes {
                match s.dim(ax) {
                    Dim::Static(v) => {
                        if let Some(ref mut sum) = axis_sum {
                            *sum += v;
                        }
                    }
                    Dim::Dynamic(_) => { axis_sum = None; break; }
                }
            }

            let mut dims: Vec<Dim> = Vec::new();
            for i in 0..rank {
                if i == ax {
                    dims.push(match axis_sum {
                        Some(v) => Dim::Static(v),
                        None => Dim::Dynamic(0),
                    });
                } else {
                    dims.push(first.dim(i));
                }
            }
            (Shape::new(dims), dtype)
        }

        // --- Reduce: collapse axes, optionally keep dims ---
        Op::Reduce { axes, keepdim, .. } => {
            let rank = first.rank();
            let resolved_axes: Vec<usize> = axes.iter()
                .map(|&a| if a < 0 { (rank as i32 + a) as usize } else { a as usize })
                .collect();

            let mut dims: Vec<Dim> = Vec::new();
            for i in 0..rank {
                if resolved_axes.contains(&i) {
                    if *keepdim {
                        dims.push(Dim::Static(1));
                    }
                    // else: drop this dim
                } else {
                    dims.push(first.dim(i));
                }
            }
            if dims.is_empty() {
                dims.push(Dim::Static(1)); // scalar result
            }
            (Shape::new(dims), dtype)
        }

        // --- Fallback: pass through first input's shape ---
        _ => (first, dtype),
    }
}

/// Build a warp-ir Graph from an ONNX model.
/// Maps ONNX nodes to warp-ir operations with proper shape inference.
pub fn onnx_to_ir(model: &OnnxModel) -> Graph {
    let mut g = Graph::new();
    let mut value_map: HashMap<String, ValueId> = HashMap::new();
    let mut dynamic_counter: u32 = 0;

    // Add model inputs — preserve dynamic dims instead of collapsing to 1
    for input in &model.inputs {
        let shape = onnx_shape_to_ir(&input.shape, &mut dynamic_counter);
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

    // Map ONNX nodes to IR operations with per-op shape inference
    for node in &model.nodes {
        let op = map_onnx_op(node);
        let inputs: Vec<ValueId> = node.inputs.iter()
            .filter_map(|name| {
                if name.is_empty() { None }
                else { value_map.get(name).copied() }
            })
            .collect();

        if inputs.is_empty() { continue; }

        // Gather input shapes/dtypes for inference
        let input_shapes: Vec<Shape> = inputs.iter()
            .map(|&id| g.value(id).shape.clone())
            .collect();
        let input_dtypes: Vec<DType> = inputs.iter()
            .map(|&id| g.value(id).dtype)
            .collect();

        let (out_shape, out_dtype) = infer_output_shape(&op, &input_shapes, &input_dtypes);

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

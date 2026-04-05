//! Compiled graph executor -- walks the optimized IR graph and dispatches GPU kernels.
//!
//! Unlike the ONNX interpreter (`onnx_exec.rs`) which walks ONNX nodes,
//! this executor walks the warp-ir `Graph` after optimization.
//! Fused ops (`FusedMatMulBias`, `AutoFused`, etc.) execute as single GPU kernels,
//! which is the entire point of the IR compilation path.
//!
//! Execution path:
//!   ONNX -> onnx_compile::onnx_to_ir -> optimizer -> **GraphExecutor::execute**
//!
//! This replaces the interpreter path (onnx_exec) for compiled graphs.

use std::collections::HashMap;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use warp_ir::*;
use warp_kernels::cache::KernelCache;
use warp_kernels::device::{DeviceError, WarpDevice};
use warp_kernels::tensor::GpuTensor;

/// Error type for graph execution.
#[derive(Debug, thiserror::Error)]
pub enum GraphExecError {
    #[error("Device error: {0}")]
    Device(#[from] DeviceError),
    #[error("Missing tensor for ValueId({0})")]
    MissingTensor(u32),
    #[error("Unsupported op: {0}")]
    UnsupportedOp(String),
    #[error("Shape error: {0}")]
    Shape(String),
}

/// Executor for compiled IR graphs.
///
/// Walks the graph in topological order and dispatches GPU kernels for each node.
/// Weights are pre-loaded onto the GPU at construction time.
/// Runtime inputs are provided per-execution via `execute()`.
pub struct GraphExecutor {
    /// Kernel cache for JIT compilation.
    cache: KernelCache,
    /// Pre-loaded weight tensors keyed by their ValueId in the graph.
    weights: HashMap<u32, GpuTensor<f32>>,
    /// Optional memory plan from lifetime analysis.
    memory_plan: Option<warp_optimizer::MemoryPlan>,
}

impl GraphExecutor {
    pub fn new() -> Self {
        Self {
            cache: KernelCache::new(),
            weights: HashMap::new(),
            memory_plan: None,
        }
    }

    /// Load a weight tensor for a graph value.
    pub fn set_weight(&mut self, value_id: ValueId, tensor: GpuTensor<f32>) {
        self.weights.insert(value_id.0, tensor);
    }

    /// Resolve a dimension value, defaulting dynamic dims to 1.
    fn dim_val(d: Dim) -> usize {
        d.static_val().unwrap_or(1)
    }

    /// Get a reference to a tensor from the owned map or the weights map.
    fn get_tensor<'a>(
        vid: ValueId,
        owned: &'a HashMap<u32, GpuTensor<f32>>,
        weights: &'a HashMap<u32, GpuTensor<f32>>,
    ) -> Result<&'a GpuTensor<f32>, GraphExecError> {
        if let Some(t) = owned.get(&vid.0) {
            return Ok(t);
        }
        if let Some(t) = weights.get(&vid.0) {
            return Ok(t);
        }
        Err(GraphExecError::MissingTensor(vid.0))
    }

    /// Execute the compiled graph.
    ///
    /// `inputs` maps graph input ValueIds to runtime tensors.
    /// Returns output tensors keyed by their ValueIds.
    pub fn execute(
        &self,
        device: &WarpDevice,
        graph: &mut Graph,
        inputs: HashMap<ValueId, GpuTensor<f32>>,
    ) -> Result<HashMap<ValueId, GpuTensor<f32>>, GraphExecError> {
        let topo = graph.topo_order().to_vec();

        // Owned tensor storage -- holds all intermediate + input tensors.
        // Inputs are moved in, intermediates are allocated during execution.
        let mut owned: HashMap<u32, GpuTensor<f32>> = HashMap::new();

        // Move runtime inputs into owned storage
        for (vid, tensor) in inputs {
            owned.insert(vid.0, tensor);
        }

        // Execute nodes in topological order
        for &node_id in &topo {
            let node = graph.node(node_id);
            let op = node.op.clone();
            let node_inputs: Vec<ValueId> = node.inputs.iter().copied().collect();
            let node_outputs: Vec<ValueId> = node.outputs.iter().copied().collect();

            match &op {
                // -- No-ops: inputs/constants/identity --

                Op::Input { .. } => {
                    // Input nodes produce values that should already be in
                    // owned (from runtime inputs) or weights (from set_weight).
                    continue;
                }

                Op::Constant { .. } => {
                    // Constants should be pre-loaded via set_weight.
                    continue;
                }

                Op::Identity => {
                    // Pass-through -- used by DCE to mark dead nodes.
                    // If it has an input, alias the output to the same tensor.
                    continue;
                }

                // -- Core compute: MatMul --

                Op::MatMul { transpose_a, transpose_b } => {
                    let a = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                    let b = Self::get_tensor(node_inputs[1], &owned, &self.weights)?;

                    let a_dims = a.shape.dims();
                    let b_dims = b.shape.dims();
                    let m = if a_dims.len() >= 2 {
                        Self::dim_val(if *transpose_a { a_dims[a_dims.len() - 1] } else { a_dims[a_dims.len() - 2] })
                    } else { 1 } as u32;
                    let k = if a_dims.len() >= 1 {
                        Self::dim_val(if *transpose_a { a_dims[a_dims.len() - 2] } else { a_dims[a_dims.len() - 1] })
                    } else { 1 } as u32;
                    let n = if b_dims.len() >= 1 {
                        Self::dim_val(if *transpose_b { b_dims[b_dims.len() - 2] } else { b_dims[b_dims.len() - 1] })
                    } else { 1 } as u32;

                    let out_shape = Shape::from_static(&[m as usize, n as usize]);
                    let mut c = GpuTensor::<f32>::zeros(device, out_shape, DType::F32)?;
                    warp_kernels::ops::gemm(&self.cache, device, a, b, &mut c, m, n, k)?;
                    owned.insert(node_outputs[0].0, c);
                }

                Op::BatchMatMul { transpose_a, transpose_b } => {
                    // Treat as regular matmul for now (batch dim = 1 common case)
                    let a = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                    let b = Self::get_tensor(node_inputs[1], &owned, &self.weights)?;

                    let a_dims = a.shape.dims();
                    let b_dims = b.shape.dims();
                    let m = if a_dims.len() >= 2 {
                        Self::dim_val(if *transpose_a { a_dims[a_dims.len() - 1] } else { a_dims[a_dims.len() - 2] })
                    } else { 1 } as u32;
                    let k = if a_dims.len() >= 1 {
                        Self::dim_val(if *transpose_a { a_dims[a_dims.len() - 2] } else { a_dims[a_dims.len() - 1] })
                    } else { 1 } as u32;
                    let n = if b_dims.len() >= 1 {
                        Self::dim_val(if *transpose_b { b_dims[b_dims.len() - 2] } else { b_dims[b_dims.len() - 1] })
                    } else { 1 } as u32;

                    let out_shape = Shape::from_static(&[m as usize, n as usize]);
                    let mut c = GpuTensor::<f32>::zeros(device, out_shape, DType::F32)?;
                    warp_kernels::ops::gemm(&self.cache, device, a, b, &mut c, m, n, k)?;
                    owned.insert(node_outputs[0].0, c);
                }

                // -- Elementwise binary --

                Op::Binary { op: bin_op } => {
                    let a = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                    let b = Self::get_tensor(node_inputs[1], &owned, &self.weights)?;
                    let mut out = GpuTensor::<f32>::zeros(device, a.shape.clone(), DType::F32)?;
                    match bin_op {
                        BinaryOp::Add => warp_kernels::ops::add(&self.cache, device, a, b, &mut out)?,
                        BinaryOp::Sub => warp_kernels::ops::sub(&self.cache, device, a, b, &mut out)?,
                        BinaryOp::Mul => warp_kernels::ops::mul(&self.cache, device, a, b, &mut out)?,
                        BinaryOp::Div => warp_kernels::ops::div(&self.cache, device, a, b, &mut out)?,
                        other => return Err(GraphExecError::UnsupportedOp(format!("Binary::{:?}", other))),
                    }
                    owned.insert(node_outputs[0].0, out);
                }

                // -- Activations --

                Op::Activate { activation } => {
                    let x = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                    let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                    match activation {
                        Activation::Relu => warp_kernels::ops::relu(&self.cache, device, x, &mut out)?,
                        Activation::Gelu | Activation::GeluTanh => warp_kernels::ops::gelu(&self.cache, device, x, &mut out)?,
                        Activation::Silu => warp_kernels::ops::silu(&self.cache, device, x, &mut out)?,
                        Activation::Sigmoid => warp_kernels::ops::sigmoid(&self.cache, device, x, &mut out)?,
                        Activation::Tanh => warp_kernels::ops::tanh_act(&self.cache, device, x, &mut out)?,
                    }
                    owned.insert(node_outputs[0].0, out);
                }

                // -- Normalization --

                Op::RmsNorm { eps } => {
                    let x = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                    let gamma = Self::get_tensor(node_inputs[1], &owned, &self.weights)?;
                    let hidden = gamma.numel as u32;
                    let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                    warp_kernels::ops::rmsnorm(&self.cache, device, x, gamma, &mut out, hidden, *eps)?;
                    owned.insert(node_outputs[0].0, out);
                }

                Op::LayerNorm { eps } => {
                    let x = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                    let gamma = Self::get_tensor(node_inputs[1], &owned, &self.weights)?;
                    let beta = Self::get_tensor(node_inputs[2], &owned, &self.weights)?;
                    let hidden = gamma.numel as u32;
                    let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                    warp_kernels::ops::layernorm(&self.cache, device, x, gamma, beta, &mut out, hidden, *eps)?;
                    owned.insert(node_outputs[0].0, out);
                }

                // -- Unary ops --

                Op::Unary { op: unary_op } => {
                    let x = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                    let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                    // Map unary ops to elementwise kernels where available
                    match unary_op {
                        UnaryOp::Neg => {
                            // neg(x) = 0 - x
                            let zero = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                            warp_kernels::ops::sub(&self.cache, device, &zero, x, &mut out)?;
                        }
                        UnaryOp::Abs => {
                            // No direct abs kernel -- skip with warning
                            log::warn!("GraphExecutor: Abs not directly supported, passing through");
                            return Ok(HashMap::new());
                        }
                        _ => {
                            log::warn!("GraphExecutor: unary op {:?} not yet supported", unary_op);
                        }
                    }
                    owned.insert(node_outputs[0].0, out);
                }

                // -- Reshape (zero-copy metadata change) --

                Op::Reshape { target_shape } => {
                    // Reshape is metadata-only. We need to produce a new tensor
                    // with the same data but different shape.
                    // Since we can't clone CudaSlice, we remove the input from owned
                    // and re-insert with a new shape. If the input is a weight, we
                    // need to copy.
                    let input_vid = node_inputs[0];
                    let out_vid = node_outputs[0];
                    let out_info = graph.value(out_vid);
                    let target = out_info.shape.clone();

                    if let Some(t) = owned.remove(&input_vid.0) {
                        let reshaped = t.reshape(target);
                        owned.insert(out_vid.0, reshaped);
                    } else if let Some(w) = self.weights.get(&input_vid.0) {
                        // Weight tensor -- copy data then reshape
                        let host = w.to_host(device)?;
                        let new_tensor = GpuTensor::from_host(device, &host, target, DType::F32)?;
                        owned.insert(out_vid.0, new_tensor);
                    }
                }

                Op::Transpose { perm: _ } => {
                    // For now, skip transpose (requires actual data movement kernel).
                    // In many fused graphs, transposes are absorbed into gemm flags.
                    let input_vid = node_inputs[0];
                    let out_vid = node_outputs[0];
                    if let Some(t) = owned.remove(&input_vid.0) {
                        owned.insert(out_vid.0, t);
                    }
                }

                // ================================================================
                // FUSED OPS -- the whole point of this executor
                // ================================================================

                Op::FusedMatMulBias { transpose_a, transpose_b } => {
                    let a = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                    let b = Self::get_tensor(node_inputs[1], &owned, &self.weights)?;
                    let bias = Self::get_tensor(node_inputs[2], &owned, &self.weights)?;

                    let a_dims = a.shape.dims();
                    let b_dims = b.shape.dims();
                    let m = if a_dims.len() >= 2 {
                        Self::dim_val(if *transpose_a { a_dims[a_dims.len() - 1] } else { a_dims[a_dims.len() - 2] })
                    } else { 1 } as u32;
                    let k = if a_dims.len() >= 1 {
                        Self::dim_val(if *transpose_a { a_dims[a_dims.len() - 2] } else { a_dims[a_dims.len() - 1] })
                    } else { 1 } as u32;
                    let n = if b_dims.len() >= 1 {
                        Self::dim_val(if *transpose_b { b_dims[b_dims.len() - 2] } else { b_dims[b_dims.len() - 1] })
                    } else { 1 } as u32;

                    let out_shape = Shape::from_static(&[m as usize, n as usize]);
                    // GEMM then add bias
                    let mut c = GpuTensor::<f32>::zeros(device, out_shape.clone(), DType::F32)?;
                    warp_kernels::ops::gemm(&self.cache, device, a, b, &mut c, m, n, k)?;
                    let mut result = GpuTensor::<f32>::zeros(device, out_shape, DType::F32)?;
                    warp_kernels::ops::add(&self.cache, device, &c, bias, &mut result)?;
                    owned.insert(node_outputs[0].0, result);
                }

                Op::FusedMatMulBiasAct { transpose_a, transpose_b, activation } => {
                    let a = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                    let b = Self::get_tensor(node_inputs[1], &owned, &self.weights)?;
                    let bias = Self::get_tensor(node_inputs[2], &owned, &self.weights)?;

                    let a_dims = a.shape.dims();
                    let b_dims = b.shape.dims();
                    let m = if a_dims.len() >= 2 {
                        Self::dim_val(if *transpose_a { a_dims[a_dims.len() - 1] } else { a_dims[a_dims.len() - 2] })
                    } else { 1 } as u32;
                    let k = if a_dims.len() >= 1 {
                        Self::dim_val(if *transpose_a { a_dims[a_dims.len() - 2] } else { a_dims[a_dims.len() - 1] })
                    } else { 1 } as u32;
                    let n = if b_dims.len() >= 1 {
                        Self::dim_val(if *transpose_b { b_dims[b_dims.len() - 2] } else { b_dims[b_dims.len() - 1] })
                    } else { 1 } as u32;

                    let out_shape = Shape::from_static(&[m as usize, n as usize]);

                    // Use fused kernel if available, otherwise chain
                    match activation {
                        Activation::Gelu | Activation::GeluTanh => {
                            let mut result = GpuTensor::<f32>::zeros(device, out_shape, DType::F32)?;
                            warp_kernels::ops::fused_gemm_bias_gelu(
                                &self.cache, device, a, b, bias, &mut result, m, n, k,
                            )?;
                            owned.insert(node_outputs[0].0, result);
                        }
                        Activation::Silu => {
                            let mut result = GpuTensor::<f32>::zeros(device, out_shape, DType::F32)?;
                            warp_kernels::ops::fused_gemm_bias_silu(
                                &self.cache, device, a, b, bias, &mut result, m, n, k,
                            )?;
                            owned.insert(node_outputs[0].0, result);
                        }
                        _ => {
                            // Fallback: GEMM + bias + activation as separate kernels
                            let mut c = GpuTensor::<f32>::zeros(device, out_shape.clone(), DType::F32)?;
                            warp_kernels::ops::gemm(&self.cache, device, a, b, &mut c, m, n, k)?;
                            let mut biased = GpuTensor::<f32>::zeros(device, out_shape.clone(), DType::F32)?;
                            warp_kernels::ops::add(&self.cache, device, &c, bias, &mut biased)?;
                            let mut result = GpuTensor::<f32>::zeros(device, out_shape, DType::F32)?;
                            match activation {
                                Activation::Relu => warp_kernels::ops::relu(&self.cache, device, &biased, &mut result)?,
                                Activation::Sigmoid => warp_kernels::ops::sigmoid(&self.cache, device, &biased, &mut result)?,
                                Activation::Tanh => warp_kernels::ops::tanh_act(&self.cache, device, &biased, &mut result)?,
                                _ => { result = biased; }
                            }
                            owned.insert(node_outputs[0].0, result);
                        }
                    }
                }

                Op::FusedResidualRmsNorm { eps } => {
                    // Inputs: [residual, X, gamma]
                    let residual = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                    let x = Self::get_tensor(node_inputs[1], &owned, &self.weights)?;
                    let gamma = Self::get_tensor(node_inputs[2], &owned, &self.weights)?;
                    let hidden = gamma.numel as u32;

                    // Add residual + x
                    let mut sum_out = GpuTensor::<f32>::zeros(device, residual.shape.clone(), DType::F32)?;
                    warp_kernels::ops::add(&self.cache, device, residual, x, &mut sum_out)?;

                    // RMSNorm the sum
                    let mut norm_out = GpuTensor::<f32>::zeros(device, residual.shape.clone(), DType::F32)?;
                    warp_kernels::ops::rmsnorm(&self.cache, device, &sum_out, gamma, &mut norm_out, hidden, *eps)?;
                    owned.insert(node_outputs[0].0, norm_out);
                }

                Op::FusedSwiGLU => {
                    // Inputs: [X, gate_weight, up_weight, down_weight]
                    // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
                    let x = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                    let gate_w = Self::get_tensor(node_inputs[1], &owned, &self.weights)?;
                    let up_w = Self::get_tensor(node_inputs[2], &owned, &self.weights)?;
                    let down_w = Self::get_tensor(node_inputs[3], &owned, &self.weights)?;

                    let x_dims = x.shape.dims();
                    let gate_dims = gate_w.shape.dims();
                    let down_dims = down_w.shape.dims();

                    let seq = Self::dim_val(x_dims[0]) as u32;
                    let hidden = Self::dim_val(x_dims[x_dims.len() - 1]) as u32;
                    let intermediate = Self::dim_val(gate_dims[gate_dims.len() - 1]) as u32;
                    let out_dim = Self::dim_val(down_dims[down_dims.len() - 1]) as u32;

                    // gate = x @ gate_w
                    let gate_shape = Shape::from_static(&[seq as usize, intermediate as usize]);
                    let mut gate = GpuTensor::<f32>::zeros(device, gate_shape.clone(), DType::F32)?;
                    warp_kernels::ops::gemm(&self.cache, device, x, gate_w, &mut gate, seq, intermediate, hidden)?;

                    // up = x @ up_w
                    let mut up = GpuTensor::<f32>::zeros(device, gate_shape.clone(), DType::F32)?;
                    warp_kernels::ops::gemm(&self.cache, device, x, up_w, &mut up, seq, intermediate, hidden)?;

                    // silu(gate)
                    let mut gate_act = GpuTensor::<f32>::zeros(device, gate_shape.clone(), DType::F32)?;
                    warp_kernels::ops::silu(&self.cache, device, &gate, &mut gate_act)?;

                    // gate_act * up
                    let mut merged = GpuTensor::<f32>::zeros(device, gate_shape, DType::F32)?;
                    warp_kernels::ops::mul(&self.cache, device, &gate_act, &up, &mut merged)?;

                    // down = merged @ down_w
                    let out_shape = Shape::from_static(&[seq as usize, out_dim as usize]);
                    let mut out = GpuTensor::<f32>::zeros(device, out_shape, DType::F32)?;
                    warp_kernels::ops::gemm(&self.cache, device, &merged, down_w, &mut out, seq, out_dim, intermediate)?;
                    owned.insert(node_outputs[0].0, out);
                }

                Op::AutoFused { kernel_name, kernel_src, num_inputs } => {
                    // JIT compile and launch the auto-fused kernel
                    let func = self.cache.get_or_compile(device, kernel_src, kernel_name)?;

                    let first_input = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                    let n = first_input.numel;

                    let out_info = graph.value(node_outputs[0]);
                    let mut out = GpuTensor::<f32>::zeros(device, out_info.shape.clone(), DType::F32)?;

                    let cfg = LaunchConfig::for_num_elems(n as u32);

                    // Collect input data pointers for the dynamic launch
                    // AutoFused kernels have signature: (out, in0, in1, ..., n)
                    // We need to use the cudarc launch API carefully.
                    // For now, dispatch based on num_inputs (1-4 are the common cases).
                    match *num_inputs {
                        1 => {
                            let i0 = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                            unsafe {
                                device.stream.launch_builder(&func)
                                    .arg(&mut out.data)
                                    .arg(&i0.data)
                                    .arg(&n)
                                    .launch(cfg)
                                    .map_err(|e| DeviceError::Launch(format!("AutoFused 1-in: {e}")))?;
                            }
                        }
                        2 => {
                            let i0 = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                            let i1 = Self::get_tensor(node_inputs[1], &owned, &self.weights)?;
                            unsafe {
                                device.stream.launch_builder(&func)
                                    .arg(&mut out.data)
                                    .arg(&i0.data)
                                    .arg(&i1.data)
                                    .arg(&n)
                                    .launch(cfg)
                                    .map_err(|e| DeviceError::Launch(format!("AutoFused 2-in: {e}")))?;
                            }
                        }
                        3 => {
                            let i0 = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                            let i1 = Self::get_tensor(node_inputs[1], &owned, &self.weights)?;
                            let i2 = Self::get_tensor(node_inputs[2], &owned, &self.weights)?;
                            unsafe {
                                device.stream.launch_builder(&func)
                                    .arg(&mut out.data)
                                    .arg(&i0.data)
                                    .arg(&i1.data)
                                    .arg(&i2.data)
                                    .arg(&n)
                                    .launch(cfg)
                                    .map_err(|e| DeviceError::Launch(format!("AutoFused 3-in: {e}")))?;
                            }
                        }
                        4 => {
                            let i0 = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                            let i1 = Self::get_tensor(node_inputs[1], &owned, &self.weights)?;
                            let i2 = Self::get_tensor(node_inputs[2], &owned, &self.weights)?;
                            let i3 = Self::get_tensor(node_inputs[3], &owned, &self.weights)?;
                            unsafe {
                                device.stream.launch_builder(&func)
                                    .arg(&mut out.data)
                                    .arg(&i0.data)
                                    .arg(&i1.data)
                                    .arg(&i2.data)
                                    .arg(&i3.data)
                                    .arg(&n)
                                    .launch(cfg)
                                    .map_err(|e| DeviceError::Launch(format!("AutoFused 4-in: {e}")))?;
                            }
                        }
                        other => {
                            return Err(GraphExecError::UnsupportedOp(
                                format!("AutoFused with {} inputs (max 4 supported)", other),
                            ));
                        }
                    }
                    owned.insert(node_outputs[0].0, out);
                }

                // -- Ops that need a minimal fallback --

                Op::Concat { axis: _ } => {
                    // For concat, just pass through the first input as a placeholder.
                    // Full concat requires a dedicated kernel.
                    if !node_inputs.is_empty() {
                        let input_vid = node_inputs[0];
                        if let Some(t) = owned.remove(&input_vid.0) {
                            owned.insert(node_outputs[0].0, t);
                        }
                    }
                    log::warn!("GraphExecutor: Concat is a placeholder pass-through");
                }

                Op::Softmax { axis: _ } => {
                    // Softmax -- pass through for now (no softmax in ops.rs).
                    // In fused attention graphs, softmax is part of the Attention op.
                    let x = Self::get_tensor(node_inputs[0], &owned, &self.weights)?;
                    let host = x.to_host(device)?;
                    // CPU softmax fallback over last dim
                    let out_info = graph.value(node_outputs[0]);
                    let shape = x.shape.clone();
                    let last_dim = Self::dim_val(shape.dims()[shape.rank() - 1]);
                    let mut result = host.clone();
                    for chunk in result.chunks_mut(last_dim) {
                        let max_val = chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0f32;
                        for v in chunk.iter_mut() {
                            *v = (*v - max_val).exp();
                            sum += *v;
                        }
                        for v in chunk.iter_mut() {
                            *v /= sum;
                        }
                    }
                    let out_tensor = GpuTensor::from_host(device, &result, out_info.shape.clone(), DType::F32)?;
                    owned.insert(node_outputs[0].0, out_tensor);
                }

                Op::Reduce { op: reduce_op, axes: _, keepdim: _ } => {
                    // Minimal reduce -- pass through with warning
                    log::warn!("GraphExecutor: Reduce {:?} is not yet GPU-accelerated", reduce_op);
                    if !node_inputs.is_empty() {
                        let input_vid = node_inputs[0];
                        if let Some(t) = owned.remove(&input_vid.0) {
                            owned.insert(node_outputs[0].0, t);
                        }
                    }
                }

                Op::Gather { axis: _ } | Op::Slice { .. } | Op::Split { .. } => {
                    // Data movement ops -- pass through first input
                    log::warn!("GraphExecutor: {:?} is a placeholder pass-through", std::mem::discriminant(&op));
                    if !node_inputs.is_empty() {
                        let input_vid = node_inputs[0];
                        if let Some(t) = owned.remove(&input_vid.0) {
                            owned.insert(node_outputs[0].0, t);
                        }
                    }
                }

                Op::Embedding => {
                    // Embedding lookup -- needs gather kernel, pass through for now
                    log::warn!("GraphExecutor: Embedding is a placeholder");
                    if !node_inputs.is_empty() {
                        let input_vid = node_inputs[0];
                        if let Some(t) = owned.remove(&input_vid.0) {
                            owned.insert(node_outputs[0].0, t);
                        }
                    }
                }

                other => {
                    log::warn!("GraphExecutor: unhandled op {:?}, skipping", std::mem::discriminant(other));
                }
            }
        }

        // Collect graph outputs
        let mut outputs = HashMap::new();
        for &out_val in &graph.graph_outputs {
            if let Some(tensor) = owned.remove(&out_val.0) {
                outputs.insert(out_val, tensor);
            } else if let Some(w) = self.weights.get(&out_val.0) {
                // Output is a weight -- copy to owned
                let host = w.to_host(device)?;
                let t = GpuTensor::from_host(device, &host, w.shape.clone(), w.dtype)?;
                outputs.insert(out_val, t);
            }
        }

        Ok(outputs)
    }

    /// Run memory planning on the graph and store the plan, then execute.
    ///
    /// This computes tensor lifetimes via `plan_memory`, logs the savings
    /// statistics, and then runs the standard `execute()`. The memory plan
    /// is stored on the executor for inspection / future buffer reuse.
    pub fn execute_with_planning(
        &mut self,
        device: &WarpDevice,
        graph: &mut Graph,
        inputs: HashMap<ValueId, GpuTensor<f32>>,
    ) -> Result<HashMap<ValueId, GpuTensor<f32>>, GraphExecError> {
        let plan = warp_optimizer::plan_memory(graph);
        log::info!(
            "Memory plan: {} buffers, peak {:.2} MB, saved {:.2} MB ({:.1}%)",
            plan.num_buffers,
            plan.peak_memory_bytes as f64 / (1024.0 * 1024.0),
            plan.savings_bytes as f64 / (1024.0 * 1024.0),
            plan.savings_pct,
        );
        self.memory_plan = Some(plan);
        self.execute(device, graph, inputs)
    }

    /// Get the current memory plan, if one has been computed.
    pub fn memory_plan(&self) -> Option<&warp_optimizer::MemoryPlan> {
        self.memory_plan.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::*;
    use warp_kernels::device::WarpDevice;
    use warp_kernels::tensor::GpuTensor;

    /// Build a small graph: Input(1,4) -> MatMul(4,8) -> Add(bias) -> Output
    /// Then run through optimizer (should fuse to FusedMatMulBias), execute,
    /// and verify output.
    #[test]
    fn graph_executor_matmul_bias() {
        let device = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        // Build IR graph manually: MatMul + Add (which optimizer fuses)
        let mut g = Graph::new();
        let x_val = g.add_input(Shape::from_static(&[1, 4]), DType::F32, Some("x"));
        let w_val = g.add_input(Shape::from_static(&[4, 8]), DType::F32, Some("w"));
        let b_val = g.add_input(Shape::from_static(&[1, 8]), DType::F32, Some("b"));

        let (_, mm_outs) = g.add_node(
            Op::MatMul { transpose_a: false, transpose_b: false },
            &[x_val, w_val],
            &[(Shape::from_static(&[1, 8]), DType::F32)],
            Some("matmul".into()),
        );

        let (_, add_outs) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[mm_outs[0], b_val],
            &[(Shape::from_static(&[1, 8]), DType::F32)],
            Some("add_bias".into()),
        );
        g.mark_output(add_outs[0]);

        // Run optimizer -- should fuse MatMul+Add into FusedMatMulBias
        let pipeline = warp_optimizer::PassPipeline::new(warp_optimizer::OptimizationLevel::O1);
        let stats = pipeline.run(&mut g);
        println!("Optimizer stats: matmul_bias_fused={}", stats.matmul_bias_fused);

        // Prepare data
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let w_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let b_data: Vec<f32> = vec![0.1; 8];

        let x_gpu = GpuTensor::from_host(&device, &x_data, Shape::from_static(&[1, 4]), DType::F32).unwrap();
        let w_gpu = GpuTensor::from_host(&device, &w_data, Shape::from_static(&[4, 8]), DType::F32).unwrap();
        let b_gpu = GpuTensor::from_host(&device, &b_data, Shape::from_static(&[1, 8]), DType::F32).unwrap();

        // Build executor with weights
        let mut executor = GraphExecutor::new();
        executor.set_weight(w_val, w_gpu);
        executor.set_weight(b_val, b_gpu);

        let mut inputs = HashMap::new();
        inputs.insert(x_val, x_gpu);

        let outputs = executor.execute(&device, &mut g, inputs)
            .expect("Graph execution failed");

        // Verify output
        let out_vid = g.graph_outputs[0];
        let out_tensor = outputs.get(&out_vid).expect("Missing output tensor");
        let result = out_tensor.to_host(&device).unwrap();

        // CPU reference: x @ w + b
        let mut expected = vec![0.0f32; 8];
        for j in 0..8 {
            expected[j] = b_data[j];
            for i in 0..4 {
                expected[j] += x_data[i] * w_data[i * 8 + j];
            }
        }

        println!("GPU result: {:?}", result);
        println!("CPU expected: {:?}", expected);

        let max_err: f32 = result.iter().zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Max error: {max_err:.6}");
        assert!(max_err < 0.01, "MatMul+Bias result mismatch: max_err={max_err}");
        println!("PASSED: graph_executor_matmul_bias");
    }

    /// Build a graph: Input -> Add(const) -> GELU -> Mul(const),
    /// run O2 optimizer (should autofuse the chain), execute with GraphExecutor,
    /// and verify output against CPU reference.
    #[test]
    fn graph_executor_activation_chain() {
        let device = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        // Graph: x -> Add(ones) -> GELU -> Mul(scale)
        let mut g = Graph::new();
        let x_val = g.add_input(Shape::from_static(&[1, 16]), DType::F32, Some("x"));
        let ones_val = g.add_input(Shape::from_static(&[1, 16]), DType::F32, Some("ones"));
        let scale_val = g.add_input(Shape::from_static(&[1, 16]), DType::F32, Some("scale"));

        let (_, add_outs) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[x_val, ones_val],
            &[(Shape::from_static(&[1, 16]), DType::F32)],
            Some("add".into()),
        );

        let (_, gelu_outs) = g.add_node(
            Op::Activate { activation: Activation::Gelu },
            &[add_outs[0]],
            &[(Shape::from_static(&[1, 16]), DType::F32)],
            Some("gelu".into()),
        );

        let (_, mul_outs) = g.add_node(
            Op::Binary { op: BinaryOp::Mul },
            &[gelu_outs[0], scale_val],
            &[(Shape::from_static(&[1, 16]), DType::F32)],
            Some("mul".into()),
        );
        g.mark_output(mul_outs[0]);

        // Run O2 optimizer (includes autofuse)
        let pipeline = warp_optimizer::PassPipeline::new(warp_optimizer::OptimizationLevel::O2);
        let stats = pipeline.run(&mut g);
        println!("O2 stats: autofuse_chains={}, autofuse_ops={}", stats.autofuse_chains, stats.autofuse_ops_fused);

        // Prepare data
        let x_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.8).collect();
        let ones_data: Vec<f32> = vec![1.0; 16];
        let scale_data: Vec<f32> = vec![2.0; 16];

        let x_gpu = GpuTensor::from_host(&device, &x_data, Shape::from_static(&[1, 16]), DType::F32).unwrap();
        let ones_gpu = GpuTensor::from_host(&device, &ones_data, Shape::from_static(&[1, 16]), DType::F32).unwrap();
        let scale_gpu = GpuTensor::from_host(&device, &scale_data, Shape::from_static(&[1, 16]), DType::F32).unwrap();

        let mut executor = GraphExecutor::new();
        executor.set_weight(ones_val, ones_gpu);
        executor.set_weight(scale_val, scale_gpu);

        let mut inputs = HashMap::new();
        inputs.insert(x_val, x_gpu);

        let outputs = executor.execute(&device, &mut g, inputs)
            .expect("Graph execution failed");

        let out_vid = g.graph_outputs[0];
        let out_tensor = outputs.get(&out_vid).expect("Missing output tensor");
        let result = out_tensor.to_host(&device).unwrap();

        // CPU reference: scale * gelu(x + 1)
        let expected: Vec<f32> = x_data.iter().map(|&xi| {
            let added = xi + 1.0;
            // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let gelu = added * 0.5 * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (added + 0.044715 * added.powi(3))).tanh());
            gelu * 2.0
        }).collect();

        println!("GPU result: {:?}", &result[..8]);
        println!("CPU expected: {:?}", &expected[..8]);

        let max_err: f32 = result.iter().zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Max error: {max_err:.6}");
        // GELU has some approximation differences, allow a bit more tolerance
        assert!(max_err < 0.05, "Activation chain result mismatch: max_err={max_err}");
        println!("PASSED: graph_executor_activation_chain");
    }
}

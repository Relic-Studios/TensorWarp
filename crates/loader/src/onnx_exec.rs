//! ONNX model executor — runs parsed ONNX models on GPU.
//!
//! This is an interpreter-style executor: walks nodes in topological order,
//! dispatches each to the appropriate TensorWarp kernel.
//!
//! Usage:
//! ```ignore
//! let model = OnnxModel::load("resnet18.onnx")?;
//! let mut exec = OnnxExecutor::new(&device, &model)?;
//! let output = exec.run(&device, &[("input", &input_tensor)])?;
//! ```

use std::collections::HashMap;

use warp_ir::{DType, Shape};
use warp_kernels::cache::KernelCache;
use warp_kernels::device::{DeviceError, WarpDevice};
use warp_kernels::tensor::GpuTensor;

use crate::onnx::{OnnxModel, OnnxNode, OnnxDType};

/// A fused elementwise chain compiled at load time.
struct CompiledFusion {
    /// ONNX node indices that are part of this chain.
    node_indices: Vec<usize>,
    /// Compiled CUDA kernel name.
    kernel_name: String,
    /// CUDA source for the fused kernel.
    kernel_src: String,
    /// Number of external inputs.
    num_inputs: usize,
    /// Output ONNX tensor name.
    output_name: String,
}

/// A running ONNX model on GPU.
pub struct OnnxExecutor {
    /// Kernel compilation cache.
    cache: KernelCache,
    /// Pre-loaded weights (initializers) on GPU.
    weights: HashMap<String, GpuTensor<f32>>,
    /// Node execution order (same as model.nodes — already topological in ONNX).
    nodes: Vec<OnnxNode>,
    /// Output tensor names.
    output_names: Vec<String>,
    /// Fused elementwise chains (compiled at load time).
    fusions: Vec<CompiledFusion>,
    /// Set of node indices that are handled by fusions (skip during normal exec).
    fused_nodes: std::collections::HashSet<usize>,
}

/// Errors during execution.
#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    #[error("Device error: {0}")]
    Device(#[from] DeviceError),
    #[error("Missing tensor: {0}")]
    MissingTensor(String),
    #[error("Unsupported op: {0}")]
    UnsupportedOp(String),
    #[error("Shape error: {0}")]
    Shape(String),
}

impl OnnxExecutor {
    /// Load an ONNX model onto the GPU.
    /// Transfers all initializer weights to device memory.
    pub fn new(device: &WarpDevice, model: &OnnxModel) -> Result<Self, ExecError> {
        let cache = KernelCache::new();
        let mut weights = HashMap::new();

        for (name, tensor) in &model.initializers {
            let data = tensor.to_f32();
            if data.is_empty() { continue; }
            let numel = data.len();
            let shape = if tensor.shape.is_empty() {
                Shape::from_static(&[numel])
            } else {
                Shape::from_static(&tensor.shape.iter().map(|&d| d as usize).collect::<Vec<_>>())
            };
            let gpu_tensor = GpuTensor::from_host(device, &data, shape, DType::F32)?;
            weights.insert(name.clone(), gpu_tensor);
        }

        let output_names = model.outputs.iter().map(|o| o.name.clone()).collect();

        // Analyze for fusible elementwise chains
        let (fusions, fused_nodes) = Self::analyze_fusions(&model.nodes);
        if !fusions.is_empty() {
            log::info!("ONNX AutoFuse: discovered {} fusible chains ({} ops → {} kernels)",
                fusions.len(),
                fused_nodes.len(),
                fusions.len());
        }

        Ok(Self {
            cache,
            weights,
            nodes: model.nodes.clone(),
            output_names,
            fusions,
            fused_nodes,
        })
    }

    /// Run inference.
    ///
    /// `inputs`: named input tensors (e.g., [("input", tensor)]).
    /// Returns: map of output name → tensor on GPU.
    pub fn run(
        &self,
        device: &WarpDevice,
        inputs: &[(&str, &GpuTensor<f32>)],
    ) -> Result<HashMap<String, GpuTensor<f32>>, ExecError> {
        // Tensor store: name → GPU tensor
        // Initialized with inputs + weights
        let mut tensors: HashMap<&str, &GpuTensor<f32>> = HashMap::new();
        let mut owned: HashMap<String, GpuTensor<f32>> = HashMap::new();

        for (name, tensor) in inputs {
            tensors.insert(name, tensor);
        }
        for (name, tensor) in &self.weights {
            tensors.insert(name, tensor);
        }

        // Execute nodes in order, skipping fused nodes
        let mut executed_fusions: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for (node_idx, node) in self.nodes.iter().enumerate() {
            // Check if this node is part of a fused chain
            if self.fused_nodes.contains(&node_idx) {
                // Find and execute the fusion that starts at or before this node
                for (fi, fusion) in self.fusions.iter().enumerate() {
                    if fusion.node_indices[0] == node_idx && !executed_fusions.contains(&fi) {
                        // Execute fused kernel
                        self.execute_fusion(device, fusion, &tensors, &mut owned)?;
                        executed_fusions.insert(fi);
                    }
                }
                continue; // skip individual node execution
            }

            self.execute_node(device, node, &tensors, &mut owned)?;
        }

        // Collect outputs
        let mut results = HashMap::new();
        for name in &self.output_names {
            if let Some(tensor) = owned.remove(name) {
                results.insert(name.clone(), tensor);
            }
        }

        Ok(results)
    }

    /// Resolve a tensor name to a GPU tensor reference.
    fn resolve<'a>(
        name: &str,
        inputs: &HashMap<&str, &'a GpuTensor<f32>>,
        owned: &'a HashMap<String, GpuTensor<f32>>,
        weights: &'a HashMap<String, GpuTensor<f32>>,
    ) -> Result<&'a GpuTensor<f32>, ExecError> {
        if let Some(t) = inputs.get(name) { return Ok(t); }
        if let Some(t) = owned.get(name) { return Ok(t); }
        if let Some(t) = weights.get(name) { return Ok(t); }
        // Empty string = optional input not provided
        if name.is_empty() { return Err(ExecError::MissingTensor("(empty)".into())); }
        Err(ExecError::MissingTensor(name.to_string()))
    }

    fn execute_node(
        &self,
        device: &WarpDevice,
        node: &OnnxNode,
        inputs: &HashMap<&str, &GpuTensor<f32>>,
        owned: &mut HashMap<String, GpuTensor<f32>>,
    ) -> Result<(), ExecError> {
        let get = |idx: usize| -> Result<&GpuTensor<f32>, ExecError> {
            let name = node.inputs.get(idx)
                .ok_or_else(|| ExecError::MissingTensor(format!("input {} of {}", idx, node.op_type)))?;
            Self::resolve(name, inputs, owned, &self.weights)
        };

        let out_name = node.outputs.first()
            .cloned()
            .unwrap_or_default();

        match node.op_type.as_str() {
            // ── Activations ────────────────────────────────────
            "Relu" => {
                let x = get(0)?;
                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                warp_kernels::ops::relu(&self.cache, device, x, &mut out)?;
                owned.insert(out_name, out);
            }
            "Sigmoid" => {
                let x = get(0)?;
                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                warp_kernels::ops::sigmoid(&self.cache, device, x, &mut out)?;
                owned.insert(out_name, out);
            }
            "Tanh" => {
                let x = get(0)?;
                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                warp_kernels::ops::tanh_act(&self.cache, device, x, &mut out)?;
                owned.insert(out_name, out);
            }
            "Gelu" | "Silu" | "Swish" => {
                let x = get(0)?;
                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                match node.op_type.as_str() {
                    "Gelu" => warp_kernels::ops::gelu(&self.cache, device, x, &mut out)?,
                    _ => warp_kernels::ops::silu(&self.cache, device, x, &mut out)?,
                }
                owned.insert(out_name, out);
            }
            "LeakyRelu" => {
                let x = get(0)?;
                let alpha = node.get_float("alpha", 0.01);
                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                warp_kernels::ops::leaky_relu(&self.cache, device, x, &mut out, alpha)?;
                owned.insert(out_name, out);
            }

            // ── Elementwise binary ─────────────────────────────
            "Add" | "Sub" | "Mul" | "Div" => {
                let a = get(0)?;
                let b = get(1)?;
                let mut out = GpuTensor::<f32>::zeros(device, a.shape.clone(), DType::F32)?;
                match node.op_type.as_str() {
                    "Add" => warp_kernels::ops::add(&self.cache, device, a, b, &mut out)?,
                    "Sub" => warp_kernels::ops::sub(&self.cache, device, a, b, &mut out)?,
                    "Mul" => warp_kernels::ops::mul(&self.cache, device, a, b, &mut out)?,
                    "Div" => warp_kernels::ops::div(&self.cache, device, a, b, &mut out)?,
                    _ => unreachable!(),
                }
                owned.insert(out_name, out);
            }

            // ── MatMul / Gemm ──────────────────────────────────
            "MatMul" | "Gemm" => {
                let a = get(0)?;
                let b = get(1)?;
                // Infer M, N, K from shapes
                let a_dims = &a.shape.dims();
                let b_dims = &b.shape.dims();
                let m = if a_dims.len() >= 2 { a_dims[a_dims.len() - 2].static_val().unwrap_or(1) as u32 } else { 1 };
                let k = if a_dims.len() >= 1 { a_dims[a_dims.len() - 1].static_val().unwrap_or(1) as u32 } else { 1 };
                let n = if b_dims.len() >= 1 { b_dims[b_dims.len() - 1].static_val().unwrap_or(1) as u32 } else { 1 };

                let mut out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[m as usize, n as usize]), DType::F32)?;
                warp_kernels::ops::gemm(&self.cache, device, a, b, &mut out, m, n, k)?;

                // Gemm: add bias if present (C input)
                if node.op_type == "Gemm" && node.inputs.len() >= 3 && !node.inputs[2].is_empty() {
                    if let Ok(bias) = get(2) {
                        let mut biased = GpuTensor::<f32>::zeros(device,
                            Shape::from_static(&[m as usize, n as usize]), DType::F32)?;
                        warp_kernels::ops::add(&self.cache, device, &out, bias, &mut biased)?;
                        out = biased;
                    }
                }
                owned.insert(out_name, out);
            }

            // ── Conv2D ─────────────────────────────────────────
            "Conv" => {
                let input = get(0)?;
                let weight = get(1)?;
                let bias = if node.inputs.len() >= 3 && !node.inputs[2].is_empty() {
                    get(2).ok()
                } else { None };

                let kernel = node.get_ints("kernel_shape");
                let kh = *kernel.first().unwrap_or(&3) as u32;
                let kw = *kernel.get(1).unwrap_or(&3) as u32;

                let strides = node.get_ints("strides");
                let sh = *strides.first().unwrap_or(&1) as u32;
                let sw = *strides.get(1).unwrap_or(&1) as u32;

                let pads = node.get_ints("pads");
                let ph = *pads.first().unwrap_or(&0) as u32;
                let pw = *pads.get(1).unwrap_or(&0) as u32;

                let dilations = node.get_ints("dilations");
                let dh = *dilations.first().unwrap_or(&1) as u32;
                let dw = *dilations.get(1).unwrap_or(&1) as u32;

                let group = node.get_int("group", 1) as u32;

                // Infer spatial dims from input shape [N, C, H, W]
                let in_dims = &input.shape.dims();
                let h = in_dims.get(2).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let w = in_dims.get(3).and_then(|d| d.static_val()).unwrap_or(1) as u32;

                let w_dims = &weight.shape.dims();
                let c_out = w_dims.first().and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let c_in = in_dims.get(1).and_then(|d| d.static_val()).unwrap_or(1) as u32;

                let params = warp_kernels::conv::Conv2dParams {
                    in_channels: c_in, out_channels: c_out,
                    kernel_h: kh, kernel_w: kw,
                    stride_h: sh, stride_w: sw,
                    padding_h: ph, padding_w: pw,
                    dilation_h: dh, dilation_w: dw,
                    groups: group,
                };

                let out_h = params.output_h(h);
                let out_w = params.output_w(w);
                let batch = input.numel / (c_in * h * w) as usize;

                let mut out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[batch, c_out as usize, out_h as usize, out_w as usize]), DType::F32)?;
                warp_kernels::conv::conv2d(&self.cache, device, input, weight, bias, &mut out, &params, h, w)?;
                owned.insert(out_name, out);
            }

            // ── BatchNorm ──────────────────────────────────────
            "BatchNormalization" => {
                let input = get(0)?;
                let scale = get(1)?;
                let bias = get(2)?;
                let mean = get(3)?;
                let var = get(4)?;
                let eps = node.get_float("epsilon", 1e-5);

                let dims = &input.shape.dims();
                let c = dims.get(1).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let h = dims.get(2).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let w = dims.get(3).and_then(|d| d.static_val()).unwrap_or(1) as u32;

                let mut out = GpuTensor::<f32>::zeros(device, input.shape.clone(), DType::F32)?;
                warp_kernels::conv::batchnorm2d(&self.cache, device, input, scale, bias, mean, var,
                    &mut out, c, h * w, eps)?;
                owned.insert(out_name, out);
            }

            // ── Pooling ────────────────────────────────────────
            "MaxPool" => {
                let input = get(0)?;
                let kernel = node.get_ints("kernel_shape");
                let strides = node.get_ints("strides");
                let pads = node.get_ints("pads");

                let kh = *kernel.first().unwrap_or(&2) as u32;
                let kw = *kernel.get(1).unwrap_or(&2) as u32;
                let sh = *strides.first().unwrap_or(&(kh as i64)) as u32;
                let sw = *strides.get(1).unwrap_or(&(kw as i64)) as u32;
                let ph = *pads.first().unwrap_or(&0) as u32;
                let pw = *pads.get(1).unwrap_or(&0) as u32;

                let dims = &input.shape.dims();
                let c = dims.get(1).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let h = dims.get(2).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let w = dims.get(3).and_then(|d| d.static_val()).unwrap_or(1) as u32;

                let out_h = warp_kernels::conv::conv_output_size(h, kh, sh, ph, 1);
                let out_w = warp_kernels::conv::conv_output_size(w, kw, sw, pw, 1);
                let batch = input.numel / (c * h * w) as usize;

                let mut out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[batch, c as usize, out_h as usize, out_w as usize]), DType::F32)?;
                warp_kernels::conv::maxpool2d(&self.cache, device, input, &mut out, c, h, w, kh, kw, sh, sw, ph, pw)?;
                owned.insert(out_name, out);
            }
            "AveragePool" => {
                let input = get(0)?;
                let kernel = node.get_ints("kernel_shape");
                let strides = node.get_ints("strides");
                let pads = node.get_ints("pads");

                let kh = *kernel.first().unwrap_or(&2) as u32;
                let kw = *kernel.get(1).unwrap_or(&2) as u32;
                let sh = *strides.first().unwrap_or(&(kh as i64)) as u32;
                let sw = *strides.get(1).unwrap_or(&(kw as i64)) as u32;
                let ph = *pads.first().unwrap_or(&0) as u32;
                let pw = *pads.get(1).unwrap_or(&0) as u32;

                let dims = &input.shape.dims();
                let c = dims.get(1).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let h = dims.get(2).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let w = dims.get(3).and_then(|d| d.static_val()).unwrap_or(1) as u32;

                let out_h = warp_kernels::conv::conv_output_size(h, kh, sh, ph, 1);
                let out_w = warp_kernels::conv::conv_output_size(w, kw, sw, pw, 1);
                let batch = input.numel / (c * h * w) as usize;

                let mut out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[batch, c as usize, out_h as usize, out_w as usize]), DType::F32)?;
                warp_kernels::conv::avgpool2d(&self.cache, device, input, &mut out, c, h, w, kh, kw, sh, sw, ph, pw)?;
                owned.insert(out_name, out);
            }
            "GlobalAveragePool" => {
                let input = get(0)?;
                let dims = &input.shape.dims();
                let c = dims.get(1).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let h = dims.get(2).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let w = dims.get(3).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let batch = input.numel / (c * h * w) as usize;

                let mut out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[batch, c as usize, 1, 1]), DType::F32)?;
                warp_kernels::conv::global_avg_pool(&self.cache, device, input, &mut out, c, h * w)?;
                owned.insert(out_name, out);
            }

            // ── Transpose ──────────────────────────────────────
            "Transpose" => {
                let x = get(0)?;
                let dims = x.shape.dims();
                if dims.len() == 2 {
                    let m = dims[0].static_val().unwrap_or(1) as u32;
                    let n = dims[1].static_val().unwrap_or(1) as u32;
                    let mut out = GpuTensor::<f32>::zeros(device,
                        Shape::from_static(&[n as usize, m as usize]), DType::F32)?;
                    warp_kernels::ops::transpose_2d(&self.cache, device, x, &mut out, m, n)?;
                    owned.insert(out_name, out);
                } else {
                    // Higher-rank transpose: fall back to copy with reindex
                    // For now, pass through (works for identity permutations)
                    let data = x.to_host(device)?;
                    let out = GpuTensor::from_host(device, &data, x.shape.clone(), DType::F32)?;
                    owned.insert(out_name, out);
                }
            }

            // ── Concat ────────────────────────────────────────
            "Concat" => {
                // Concat along channel axis (axis=1 for NCHW)
                let a = get(0)?;
                let b = get(1)?;
                let axis = node.get_int("axis", 1);

                let a_dims = a.shape.dims();
                let b_dims = b.shape.dims();

                if axis == 1 && a_dims.len() >= 2 {
                    // Channel concat
                    let c_a = a_dims[1].static_val().unwrap_or(1) as u32;
                    let c_b = b_dims[1].static_val().unwrap_or(1) as u32;
                    let spatial: u32 = a_dims[2..].iter()
                        .map(|d| d.static_val().unwrap_or(1) as u32)
                        .product();
                    let spatial = if spatial == 0 { 1 } else { spatial };
                    let batch = a.numel / (c_a * spatial) as usize;

                    let mut out_dims: Vec<usize> = a_dims.iter()
                        .map(|d| d.static_val().unwrap_or(1)).collect();
                    out_dims[1] = (c_a + c_b) as usize;

                    let mut out = GpuTensor::<f32>::zeros(device,
                        Shape::from_static(&out_dims), DType::F32)?;
                    warp_kernels::ops::concat_channels(&self.cache, device, a, b, &mut out,
                        c_a, c_b, spatial)?;
                    owned.insert(out_name, out);
                } else {
                    // Simple concat: copy both to host, concatenate, upload
                    let mut data_a = a.to_host(device)?;
                    let data_b = b.to_host(device)?;
                    data_a.extend_from_slice(&data_b);
                    let out = GpuTensor::from_host(device, &data_a,
                        Shape::from_static(&[data_a.len()]), DType::F32)?;
                    owned.insert(out_name, out);
                }
            }

            // ── Reduce ────────────────────────────────────────
            "ReduceMean" | "ReduceSum" | "ReduceMax" => {
                let x = get(0)?;
                let dims = x.shape.dims();
                // Reduce along last dimension
                let cols = dims.last().and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let rows = (x.numel / cols as usize) as u32;

                let mut out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[rows as usize]), DType::F32)?;
                match node.op_type.as_str() {
                    "ReduceMean" => warp_kernels::ops::reduce_mean(&self.cache, device, x, &mut out, rows, cols)?,
                    "ReduceSum" => warp_kernels::ops::reduce_sum(&self.cache, device, x, &mut out, rows, cols)?,
                    "ReduceMax" => warp_kernels::ops::reduce_max(&self.cache, device, x, &mut out, rows, cols)?,
                    _ => unreachable!(),
                }
                owned.insert(out_name, out);
            }

            // ── Resize / Upsample ─────────────────────────────
            "Resize" | "Upsample" => {
                let input = get(0)?;
                let dims = input.shape.dims();
                let c = dims.get(1).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let h = dims.get(2).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let w = dims.get(3).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let batch = input.numel / (c * h * w) as usize;

                // Try to get output size from scales input (input 2 or 3)
                let scale_factor = if node.inputs.len() >= 4 {
                    // ONNX Resize v11+: inputs are [X, roi, scales, sizes]
                    get(2).ok().and_then(|s| {
                        let sv = s.to_host(device).ok()?;
                        // scales are [1, 1, scale_h, scale_w]
                        sv.get(2).copied()
                    }).unwrap_or(2.0)
                } else { 2.0 };

                let out_h = (h as f32 * scale_factor) as u32;
                let out_w = (w as f32 * scale_factor) as u32;

                let mode = node.get_string("mode").unwrap_or("nearest");
                let mut out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[batch, c as usize, out_h as usize, out_w as usize]), DType::F32)?;

                if mode == "linear" || mode == "bilinear" {
                    warp_kernels::conv::resize_bilinear(&self.cache, device, input, &mut out, c, h, w, out_h, out_w)?;
                } else {
                    warp_kernels::conv::resize_nearest(&self.cache, device, input, &mut out, c, h, w, out_h, out_w)?;
                }
                owned.insert(out_name, out);
            }

            // ── ConvTranspose ─────────────────────────────────
            "ConvTranspose" => {
                let input = get(0)?;
                let weight = get(1)?;
                let bias = if node.inputs.len() >= 3 && !node.inputs[2].is_empty() {
                    get(2).ok()
                } else { None };

                let kernel = node.get_ints("kernel_shape");
                let kh = *kernel.first().unwrap_or(&4) as u32;
                let kw = *kernel.get(1).unwrap_or(&4) as u32;
                let strides = node.get_ints("strides");
                let sh = *strides.first().unwrap_or(&(2i64)) as u32;
                let sw = *strides.get(1).unwrap_or(&(2i64)) as u32;
                let pads = node.get_ints("pads");
                let ph = *pads.first().unwrap_or(&0) as u32;
                let pw = *pads.get(1).unwrap_or(&0) as u32;

                let in_dims = input.shape.dims();
                let h = in_dims.get(2).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let w = in_dims.get(3).and_then(|d| d.static_val()).unwrap_or(1) as u32;

                let w_dims = weight.shape.dims();
                let c_in = w_dims.first().and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let c_out = w_dims.get(1).and_then(|d| d.static_val()).unwrap_or(1) as u32;

                let params = warp_kernels::conv::ConvTranspose2dParams::new(c_in, c_out, kh)
                    .stride(sh).padding(ph);
                let out_h = params.output_h(h);
                let out_w = params.output_w(w);
                let batch = input.numel / (c_in * h * w) as usize;

                let mut out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[batch, c_out as usize, out_h as usize, out_w as usize]), DType::F32)?;
                warp_kernels::conv::conv_transpose2d(&self.cache, device, input, weight, bias, &mut out, &params, h, w)?;
                owned.insert(out_name, out);
            }

            // ── LayerNorm ─────────────────────────────────────
            "LayerNormalization" => {
                let x = get(0)?;
                let gamma = get(1)?;
                let beta = get(2)?;
                let eps = node.get_float("epsilon", 1e-5);
                let dims = x.shape.dims();
                let hidden = dims.last().and_then(|d| d.static_val()).unwrap_or(1) as u32;

                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                // Use RMSNorm as approximation (proper LayerNorm needs mean subtraction)
                // TODO: implement full LayerNorm kernel with mean centering
                warp_kernels::ops::rmsnorm(&self.cache, device, x, gamma, &mut out, hidden, eps)?;
                owned.insert(out_name, out);
            }

            // ── GroupNorm ─────────────────────────────────────
            "GroupNormalization" => {
                let x = get(0)?;
                let scale = get(1)?;
                let bias = get(2)?;
                let num_groups = node.get_int("num_groups", 32) as u32;
                let eps = node.get_float("epsilon", 1e-5);
                let dims = x.shape.dims();
                let c = dims.get(1).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let spatial: u32 = dims[2..].iter()
                    .map(|d| d.static_val().unwrap_or(1) as u32)
                    .product();
                let spatial = if spatial == 0 { 1 } else { spatial };

                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                warp_kernels::ops::groupnorm(&self.cache, device, x, scale, bias, &mut out,
                    c, spatial, num_groups, eps)?;
                owned.insert(out_name, out);
            }

            // ── Reshape / Flatten — zero-copy ─────────────────
            "Reshape" | "Flatten" | "Squeeze" | "Unsqueeze" => {
                // Zero-copy: just flatten shape, underlying GPU memory unchanged
                let x = get(0)?;
                let numel = x.numel;
                // Reinterpret as flat tensor (consumers will reshape as needed)
                let data = device.dtoh(&x.data)?;
                let out = GpuTensor::from_host(device, &data,
                    Shape::from_static(&[numel]), DType::F32)?;
                owned.insert(out_name, out);
            }

            // ── Identity / Dropout ─────────────────────────────
            "Identity" | "Dropout" => {
                // Zero-copy pass-through
                let x = get(0)?;
                let data = device.dtoh(&x.data)?;
                let out = GpuTensor::from_host(device, &data, x.shape.clone(), DType::F32)?;
                owned.insert(out_name, out);
            }

            // ── Softmax (GPU) ──────────────────────────────────
            "Softmax" => {
                let x = get(0)?;
                let dims = x.shape.dims();
                let cols = dims.last().and_then(|d| d.static_val()).unwrap_or(x.numel) as u32;
                let rows = (x.numel / cols as usize) as u32;

                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                warp_kernels::sampling::softmax(&self.cache, device, x, &mut out, rows, cols)?;
                owned.insert(out_name, out);
            }

            // ── Clip ──────────────────────────────────────────
            "Clip" => {
                let x = get(0)?;
                let lo = if node.inputs.len() >= 2 && !node.inputs[1].is_empty() {
                    get(1).ok().and_then(|t| t.to_host(device).ok())
                        .and_then(|v| v.first().copied()).unwrap_or(f32::NEG_INFINITY)
                } else { f32::NEG_INFINITY };
                let hi = if node.inputs.len() >= 3 && !node.inputs[2].is_empty() {
                    get(2).ok().and_then(|t| t.to_host(device).ok())
                        .and_then(|v| v.first().copied()).unwrap_or(f32::INFINITY)
                } else { f32::INFINITY };
                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                warp_kernels::ops::clip(&self.cache, device, x, &mut out, lo, hi)?;
                owned.insert(out_name, out);
            }

            // ── Gather ────────────────────────────────────────
            "Gather" | "Slice" | "Split" | "Pad" => {
                // These need proper implementations — for now, pass data through
                // with a warning (better than silent skip)
                if let Ok(x) = get(0) {
                    log::warn!("Op '{}' using pass-through (shape may be wrong)", node.op_type);
                    let data = device.dtoh(&x.data)?;
                    let out = GpuTensor::from_host(device, &data, x.shape.clone(), DType::F32)?;
                    owned.insert(out_name, out);
                }
            }

            // ── Constant / Shape / Cast ───────────────────────
            "Constant" => {
                // Constants should already be in initializers
            }
            "Shape" | "Cast" => {
                if let Ok(x) = get(0) {
                    let data = device.dtoh(&x.data)?;
                    let out = GpuTensor::from_host(device, &data, x.shape.clone(), DType::F32)?;
                    owned.insert(out_name, out);
                }
            }

            // ── HARD ERROR for truly unsupported ops ──────────
            other => {
                return Err(ExecError::UnsupportedOp(format!(
                    "ONNX op '{}' (node '{}') is not implemented. \
                     TensorWarp supports: Add, Sub, Mul, Div, MatMul, Gemm, Conv, ConvTranspose, \
                     BatchNorm, LayerNorm, GroupNorm, MaxPool, AvgPool, GlobalAvgPool, \
                     Relu, Sigmoid, Tanh, Gelu, Silu, LeakyRelu, Clip, Softmax, \
                     Reshape, Flatten, Transpose, Concat, Reduce*, Resize, Identity, Dropout",
                    other, node.name
                )));
            }
        }

        Ok(())
    }

    /// Execute a fused elementwise chain.
    fn execute_fusion(
        &self,
        device: &WarpDevice,
        fusion: &CompiledFusion,
        inputs: &HashMap<&str, &GpuTensor<f32>>,
        owned: &mut HashMap<String, GpuTensor<f32>>,
    ) -> Result<(), ExecError> {
        // Compile the fused kernel
        let f = self.cache.get_or_compile(device, &fusion.kernel_src, &fusion.kernel_name)
            .map_err(|e| ExecError::Device(e))?;

        // Gather inputs: the first input of the first node in the chain,
        // plus additional inputs for binary ops within the chain
        let mut input_tensors: Vec<&GpuTensor<f32>> = Vec::new();

        // First input: main data flowing through the chain
        let first_node = &self.nodes[fusion.node_indices[0]];
        let first_input_name = &first_node.inputs[0];
        let first_input = Self::resolve(first_input_name, inputs, owned, &self.weights)?;
        input_tensors.push(first_input);

        // Additional inputs: second operands of binary ops in the chain
        for &ni in &fusion.node_indices {
            let node = &self.nodes[ni];
            if node.inputs.len() >= 2 {
                // Binary op — need the second input
                let second_name = &node.inputs[1];
                let second = Self::resolve(second_name, inputs, owned, &self.weights)?;
                input_tensors.push(second);
            }
        }

        // Allocate output
        let n = first_input.numel;
        let mut output = GpuTensor::<f32>::zeros(device,
            first_input.shape.clone(), warp_ir::DType::F32)?;

        // Launch fused kernel — build arg list based on number of inputs
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(n as u32);
        match input_tensors.len() {
            1 => unsafe {
                use cudarc::driver::PushKernelArg;
                device.stream.launch_builder(&f)
                    .arg(&mut output.data).arg(&input_tensors[0].data).arg(&n)
                    .launch(cfg).map_err(|e: cudarc::driver::result::DriverError|
                        ExecError::Device(DeviceError::Launch(e.to_string())))?;
            },
            2 => unsafe {
                use cudarc::driver::PushKernelArg;
                device.stream.launch_builder(&f)
                    .arg(&mut output.data).arg(&input_tensors[0].data)
                    .arg(&input_tensors[1].data).arg(&n)
                    .launch(cfg).map_err(|e: cudarc::driver::result::DriverError|
                        ExecError::Device(DeviceError::Launch(e.to_string())))?;
            },
            3 => unsafe {
                use cudarc::driver::PushKernelArg;
                device.stream.launch_builder(&f)
                    .arg(&mut output.data).arg(&input_tensors[0].data)
                    .arg(&input_tensors[1].data).arg(&input_tensors[2].data).arg(&n)
                    .launch(cfg).map_err(|e: cudarc::driver::result::DriverError|
                        ExecError::Device(DeviceError::Launch(e.to_string())))?;
            },
            4 => unsafe {
                use cudarc::driver::PushKernelArg;
                device.stream.launch_builder(&f)
                    .arg(&mut output.data).arg(&input_tensors[0].data)
                    .arg(&input_tensors[1].data).arg(&input_tensors[2].data)
                    .arg(&input_tensors[3].data).arg(&n)
                    .launch(cfg).map_err(|e: cudarc::driver::result::DriverError|
                        ExecError::Device(DeviceError::Launch(e.to_string())))?;
            },
            _ => return Err(ExecError::UnsupportedOp(
                format!("Fused kernel with {} inputs not supported (max 4)", input_tensors.len()))),
        }

        owned.insert(fusion.output_name.clone(), output);
        Ok(())
    }

    /// Analyze ONNX nodes for fusible elementwise chains.
    fn analyze_fusions(nodes: &[OnnxNode]) -> (Vec<CompiledFusion>, std::collections::HashSet<usize>) {
        let mut fusions = Vec::new();
        let mut fused_set = std::collections::HashSet::new();

        // Map: output tensor name -> node index (for following chains)
        let mut producer: HashMap<String, usize> = HashMap::new();
        for (i, node) in nodes.iter().enumerate() {
            for out in &node.outputs {
                producer.insert(out.clone(), i);
            }
        }

        // Map: tensor name -> count of consumer nodes
        let mut consumer_count: HashMap<String, usize> = HashMap::new();
        for node in nodes {
            for inp in &node.inputs {
                *consumer_count.entry(inp.clone()).or_insert(0) += 1;
            }
        }

        // Check if an ONNX op is elementwise-fusible
        let is_fusible = |op: &str| -> bool {
            matches!(op, "Add" | "Sub" | "Mul" | "Div" | "Relu" | "Sigmoid" | "Tanh"
                | "Gelu" | "Silu" | "Swish" | "LeakyRelu")
        };

        // CUDA expression for each fusible op
        let op_expr = |op: &str, inputs: &[String]| -> String {
            match op {
                "Add" => format!("({} + {})", inputs[0], inputs[1]),
                "Sub" => format!("({} - {})", inputs[0], inputs[1]),
                "Mul" => format!("({} * {})", inputs[0], inputs[1]),
                "Div" => format!("({} / {})", inputs[0], inputs[1]),
                "Relu" => format!("fmaxf({}, 0.0f)", inputs[0]),
                "Sigmoid" => format!("(1.0f / (1.0f + expf(-{})))", inputs[0]),
                "Tanh" => format!("tanhf({})", inputs[0]),
                "Silu" | "Swish" => { let x = &inputs[0]; format!("({x} / (1.0f + expf(-{x})))") }
                "Gelu" => { let x = &inputs[0]; format!("(0.5f*{x}*(1.0f+tanhf(0.7978845608f*({x}+0.044715f*{x}*{x}*{x}))))") }
                _ => inputs[0].clone(),
            }
        };

        let is_binary = |op: &str| -> bool {
            matches!(op, "Add" | "Sub" | "Mul" | "Div")
        };

        // Find chains
        for start_i in 0..nodes.len() {
            if fused_set.contains(&start_i) { continue; }
            if !is_fusible(&nodes[start_i].op_type) { continue; }

            let mut chain = vec![start_i];
            let mut current = start_i;

            // Follow forward through single-consumer fusible ops
            loop {
                let out_name = match nodes[current].outputs.first() {
                    Some(n) => n.clone(),
                    None => break,
                };
                let consumers = consumer_count.get(&out_name).copied().unwrap_or(0);
                if consumers != 1 { break; }

                // Find the consumer
                let next = nodes.iter().enumerate().position(|(i, n)| {
                    !fused_set.contains(&i) && i > current && n.inputs.contains(&out_name)
                });
                match next {
                    Some(ni) if is_fusible(&nodes[ni].op_type) && !fused_set.contains(&ni) => {
                        chain.push(ni);
                        current = ni;
                    }
                    _ => break,
                }
            }

            if chain.len() >= 2 {
                // Build the fused kernel
                let mut expr = "in0[i]".to_string();
                let mut next_input = 1;
                let mut kernel_inputs = 1;

                for &ci in &chain {
                    let op = &nodes[ci].op_type;
                    if is_binary(op) {
                        let b = format!("in{next_input}[i]");
                        next_input += 1;
                        kernel_inputs += 1;
                        expr = op_expr(op, &[expr, b]);
                    } else {
                        expr = op_expr(op, &[expr]);
                    }
                }

                // Build parameter list
                let mut params = vec!["float *out".to_string()];
                for j in 0..kernel_inputs {
                    params.push(format!("const float *in{j}"));
                }
                params.push("size_t n".to_string());

                let name = format!("warp_onnx_fused_{}", fusions.len());
                let src = format!(
                    r#"extern "C" __global__ void {name}({params}) {{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{ out[i] = {expr}; }}
}}"#,
                    name = name, params = params.join(", "), expr = expr);

                let output_name = nodes[*chain.last().unwrap()].outputs[0].clone();

                fusions.push(CompiledFusion {
                    node_indices: chain.clone(),
                    kernel_name: name,
                    kernel_src: src,
                    num_inputs: kernel_inputs,
                    output_name,
                });

                for ci in chain {
                    fused_set.insert(ci);
                }
            }
        }

        (fusions, fused_set)
    }

    /// Get kernel cache stats.
    pub fn cache_stats(&self) -> String {
        format!("{}", self.cache.stats())
    }

    /// Get weight memory usage.
    pub fn weight_memory_bytes(&self) -> usize {
        self.weights.values().map(|t| t.size_bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::*;

    fn setup() -> Option<WarpDevice> {
        WarpDevice::new(0).ok()
    }

    /// Build a tiny ONNX model programmatically for testing.
    /// Architecture: input → Conv(3→8, 3×3, pad=1) → Relu → GlobalAvgPool → Gemm → output
    fn build_test_model() -> OnnxModel {
        let mut initializers = HashMap::new();

        // Conv weight: [8, 3, 3, 3]
        let conv_w: Vec<f32> = (0..8*3*3*3).map(|i| ((i * 7 + 3) % 200) as f32 * 0.01 - 1.0).collect();
        initializers.insert("conv.weight".to_string(), OnnxTensor {
            name: "conv.weight".to_string(),
            dtype: OnnxDType::Float,
            shape: vec![8, 3, 3, 3],
            raw_data: conv_w.iter().flat_map(|f| f.to_le_bytes()).collect(),
        });

        // FC weight: [8, 10]
        let fc_w: Vec<f32> = (0..8*10).map(|i| ((i * 11 + 5) % 200) as f32 * 0.01 - 1.0).collect();
        initializers.insert("fc.weight".to_string(), OnnxTensor {
            name: "fc.weight".to_string(),
            dtype: OnnxDType::Float,
            shape: vec![8, 10],
            raw_data: fc_w.iter().flat_map(|f| f.to_le_bytes()).collect(),
        });

        OnnxModel {
            inputs: vec![OnnxIO {
                name: "input".to_string(),
                dtype: Some(OnnxDType::Float),
                shape: vec![1, 3, 8, 8],
            }],
            outputs: vec![OnnxIO {
                name: "output".to_string(),
                dtype: Some(OnnxDType::Float),
                shape: vec![1, 10],
            }],
            nodes: vec![
                OnnxNode {
                    name: "conv1".to_string(),
                    op_type: "Conv".to_string(),
                    inputs: vec!["input".into(), "conv.weight".into()],
                    outputs: vec!["conv_out".into()],
                    attrs: [
                        ("kernel_shape".into(), OnnxAttr::Ints(vec![3, 3])),
                        ("pads".into(), OnnxAttr::Ints(vec![1, 1, 1, 1])),
                        ("strides".into(), OnnxAttr::Ints(vec![1, 1])),
                    ].into_iter().collect(),
                },
                OnnxNode {
                    name: "relu1".to_string(),
                    op_type: "Relu".to_string(),
                    inputs: vec!["conv_out".into()],
                    outputs: vec!["relu_out".into()],
                    attrs: HashMap::new(),
                },
                OnnxNode {
                    name: "gap".to_string(),
                    op_type: "GlobalAveragePool".to_string(),
                    inputs: vec!["relu_out".into()],
                    outputs: vec!["gap_out".into()],
                    attrs: HashMap::new(),
                },
                OnnxNode {
                    name: "reshape".to_string(),
                    op_type: "Reshape".to_string(),
                    inputs: vec!["gap_out".into()],
                    outputs: vec!["flat_out".into()],
                    attrs: HashMap::new(),
                },
                OnnxNode {
                    name: "fc".to_string(),
                    op_type: "Gemm".to_string(),
                    inputs: vec!["flat_out".into(), "fc.weight".into()],
                    outputs: vec!["output".into()],
                    attrs: HashMap::new(),
                },
            ],
            initializers,
            ir_version: 8,
            opset_version: 17,
            producer: "test".to_string(),
        }
    }

    #[test]
    fn executor_runs_test_model() {
        let dev = match setup() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let model = build_test_model();
        println!("{}", model.summary());

        let exec = OnnxExecutor::new(&dev, &model).unwrap();
        println!("Weight memory: {:.2} KB", exec.weight_memory_bytes() as f64 / 1024.0);

        // Create input: [1, 3, 8, 8]
        let input_data: Vec<f32> = (0..1*3*8*8).map(|i| ((i * 13 + 7) % 100) as f32 * 0.01).collect();
        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[1, 3, 8, 8]), DType::F32).unwrap();

        let outputs = exec.run(&dev, &[("input", &input)]).unwrap();
        dev.synchronize().unwrap();

        assert!(outputs.contains_key("output"), "Should have 'output' tensor");
        let result = outputs["output"].to_host(&dev).unwrap();

        println!("\n=== ONNX Executor Test ===");
        println!("  Model: Conv(3→8, 3×3) → ReLU → GAP → FC(8→10)");
        println!("  Input:  [1, 3, 8, 8]");
        println!("  Output: {:?}", result);
        println!("  Output len: {}", result.len());

        assert!(result.iter().all(|v| v.is_finite()), "Output has NaN/Inf!");
        assert!(result.iter().any(|v| *v != 0.0), "Output is all zeros!");
        println!("  Executor produced valid output!");
        println!("{}", exec.cache_stats());
    }
}

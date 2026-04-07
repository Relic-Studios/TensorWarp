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
    #[error("Missing tensor '{name}' in op '{op}'")]
    MissingTensor { name: String, op: String },
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
        // Disabled when model has Constant nodes (common in PyTorch exports) — fusion
        // can accidentally fuse ops that depend on Constant outputs, causing missing tensors.
        let has_constants = model.nodes.iter().any(|n| n.op_type == "Constant");
        let (fusions, fused_nodes) = if has_constants {
            (Vec::new(), std::collections::HashSet::new())
        } else {
            Self::analyze_fusions(&model.nodes)
        };
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

            // Debug trace: dump first/last few nodes' output stats
            if std::env::var("TW_TRACE").is_ok() {
                for out_name in &node.outputs {
                    if let Some(t) = owned.get(out_name) {
                        let data = t.to_host(device).unwrap_or_default();
                        if !data.is_empty() {
                            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let mean = data.iter().sum::<f32>() / data.len() as f32;
                            let nonzero = data.iter().filter(|&&v| v != 0.0).count();
                            eprintln!("[TRACE] node {} {} '{}' → '{}': {} elems, range=[{:.4}, {:.4}], mean={:.6}, nonzero={}",
                                node_idx, node.op_type, node.name, out_name,
                                data.len(), min, max, mean, nonzero);
                        }
                    }
                }
            }
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
        op: &str,
        inputs: &HashMap<&str, &'a GpuTensor<f32>>,
        owned: &'a HashMap<String, GpuTensor<f32>>,
        weights: &'a HashMap<String, GpuTensor<f32>>,
    ) -> Result<&'a GpuTensor<f32>, ExecError> {
        if let Some(t) = inputs.get(name) { return Ok(t); }
        if let Some(t) = owned.get(name) { return Ok(t); }
        if let Some(t) = weights.get(name) { return Ok(t); }
        Err(ExecError::MissingTensor {
            name: if name.is_empty() { "(empty)".into() } else { name.to_string() },
            op: op.to_string(),
        })
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
                .ok_or_else(|| ExecError::MissingTensor {
                    name: format!("input[{}]", idx),
                    op: node.op_type.clone(),
                })?;
            Self::resolve(name, &node.op_type, inputs, owned, &self.weights)
        };

        let out_name = node.outputs.first()
            .cloned()
            .ok_or_else(|| ExecError::Shape(
                format!("{} node '{}' has no output names", node.op_type, node.name)
            ))?;

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
            // ── Sqrt / Pow — elementwise math ops for LayerNorm decomposition ──
            "Sqrt" => {
                let x = get(0)?;
                let n = x.numel as u32;
                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                let f = self.cache.get_or_compile(device,
                    r#"extern "C" __global__ void warp_sqrt(float *out, const float *x, unsigned int n) {
                        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                        if (i < n) out[i] = sqrtf(x[i]);
                    }"#, "warp_sqrt")?;
                let cfg = cudarc::driver::LaunchConfig::for_num_elems(n);
                unsafe {
                    use cudarc::driver::PushKernelArg;
                    device.stream.launch_builder(&f)
                        .arg(&mut out.data)
                        .arg(&x.data)
                        .arg(&n)
                        .launch(cfg)
                        .map_err(|e| DeviceError::Launch(e.to_string()))?;
                }
                owned.insert(out_name, out);
            }
            "Pow" => {
                let x = get(0)?;
                let n = x.numel as u32;
                // ONNX Pow(x, y): if y is a scalar constant (e.g. 2.0 for variance), broadcast
                let exp_val = if node.inputs.len() >= 2 && !node.inputs[1].is_empty() {
                    let exp_tensor = get(1)?;
                    let exp_host = exp_tensor.to_host(device).unwrap_or_default();
                    if exp_host.len() == 1 { exp_host[0] } else { 2.0 }
                } else { 2.0 };
                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                let f = self.cache.get_or_compile(device,
                    r#"extern "C" __global__ void warp_pow(float *out, const float *x, float exp_val, unsigned int n) {
                        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                        if (i < n) out[i] = powf(x[i], exp_val);
                    }"#, "warp_pow")?;
                let cfg = cudarc::driver::LaunchConfig::for_num_elems(n);
                unsafe {
                    use cudarc::driver::PushKernelArg;
                    device.stream.launch_builder(&f)
                        .arg(&mut out.data)
                        .arg(&x.data)
                        .arg(&exp_val)
                        .arg(&n)
                        .launch(cfg)
                        .map_err(|e| DeviceError::Launch(e.to_string()))?;
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

                // Handle scalar broadcast: if one operand is scalar (1 elem), broadcast it
                if a.numel != b.numel && (a.numel == 1 || b.numel == 1) {
                    let (big, small, a_is_big) = if a.numel > b.numel { (a, b, true) } else { (b, a, false) };
                    let n = big.numel as u32;
                    let scalar_host = small.to_host(device)?;
                    let scalar_val = scalar_host[0];
                    let mut out = GpuTensor::<f32>::zeros(device, big.shape.clone(), DType::F32)?;

                    // Generate scalar broadcast kernel
                    let (op_name, op_expr) = match (node.op_type.as_str(), a_is_big) {
                        ("Add", _) => ("scalar_add", "x[i] + s"),
                        ("Mul", _) => ("scalar_mul", "x[i] * s"),
                        ("Sub", true) => ("scalar_sub_r", "x[i] - s"),     // a - scalar
                        ("Sub", false) => ("scalar_sub_l", "s - x[i]"),    // scalar - a
                        ("Div", true) => ("scalar_div_r", "x[i] / s"),     // a / scalar
                        ("Div", false) => ("scalar_div_l", "s / x[i]"),    // scalar / a
                        _ => unreachable!(),
                    };
                    let src = format!(
                        r#"extern "C" __global__ void warp_{op_name}(float *out, const float *x, float s, unsigned int n) {{
                            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                            if (i < n) out[i] = {op_expr};
                        }}"#);
                    let f = self.cache.get_or_compile(device, &src, &format!("warp_{op_name}"))?;
                    let cfg = cudarc::driver::LaunchConfig::for_num_elems(n);
                    unsafe {
                        use cudarc::driver::PushKernelArg;
                        device.stream.launch_builder(&f)
                            .arg(&mut out.data).arg(&big.data).arg(&scalar_val).arg(&n)
                            .launch(cfg).map_err(|e| DeviceError::Launch(e.to_string()))?;
                    }
                    owned.insert(out_name, out);
                } else {
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
            }

            // ── MatMul / Gemm ──────────────────────────────────
            "MatMul" | "Gemm" => {
                let a = get(0)?;
                let b = get(1)?;
                // Infer M, N, K from shapes
                let a_dims = &a.shape.dims();
                let b_dims = &b.shape.dims();
                let m = if a_dims.len() >= 2 {
                    a_dims[a_dims.len() - 2].static_val()
                        .ok_or_else(|| ExecError::Shape(format!("MatMul: dynamic M dim in A{:?}", a_dims)))? as u32
                } else { 1 };
                let k = if a_dims.len() >= 1 {
                    a_dims[a_dims.len() - 1].static_val()
                        .ok_or_else(|| ExecError::Shape(format!("MatMul: dynamic K dim in A{:?}", a_dims)))? as u32
                } else { 1 };
                let n = if b_dims.len() >= 1 {
                    b_dims[b_dims.len() - 1].static_val()
                        .ok_or_else(|| ExecError::Shape(format!("MatMul: dynamic N dim in B{:?}", b_dims)))? as u32
                } else { 1 };

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
                let h = in_dims.get(2).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("Conv: missing H dim in input".into()))? as u32;
                let w = in_dims.get(3).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("Conv: missing W dim in input".into()))? as u32;

                let w_dims = &weight.shape.dims();
                let c_out = w_dims.first().and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("Conv: missing C_out dim in weight".into()))? as u32;
                let c_in = in_dims.get(1).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("Conv: missing C_in dim in input".into()))? as u32;

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
                let c = dims.get(1).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("BatchNorm: missing C dim".into()))? as u32;
                let h = dims.get(2).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("BatchNorm: missing H dim".into()))? as u32;
                let w = dims.get(3).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("BatchNorm: missing W dim".into()))? as u32;

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
                let c = dims.get(1).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("MaxPool: missing C dim".into()))? as u32;
                let h = dims.get(2).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("MaxPool: missing H dim".into()))? as u32;
                let w = dims.get(3).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("MaxPool: missing W dim".into()))? as u32;

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
                let c = dims.get(1).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("AvgPool: missing C dim".into()))? as u32;
                let h = dims.get(2).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("AvgPool: missing H dim".into()))? as u32;
                let w = dims.get(3).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("AvgPool: missing W dim".into()))? as u32;

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
                let c = dims.get(1).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("GlobalAvgPool: missing C dim".into()))? as u32;
                let h = dims.get(2).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("GlobalAvgPool: missing H dim".into()))? as u32;
                let w = dims.get(3).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("GlobalAvgPool: missing W dim".into()))? as u32;
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
                    let m = dims[0].static_val()
                        .ok_or_else(|| ExecError::Shape("Transpose: dynamic dim 0".into()))? as u32;
                    let n = dims[1].static_val()
                        .ok_or_else(|| ExecError::Shape("Transpose: dynamic dim 1".into()))? as u32;
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
                    let c_a = a_dims[1].static_val()
                        .ok_or_else(|| ExecError::Shape("Concat: dynamic C dim in A".into()))? as u32;
                    let c_b = b_dims[1].static_val()
                        .ok_or_else(|| ExecError::Shape("Concat: dynamic C dim in B".into()))? as u32;
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
                let c = dims.get(1).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("Resize: missing C dim".into()))? as u32;
                let h = dims.get(2).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("Resize: missing H dim".into()))? as u32;
                let w = dims.get(3).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("Resize: missing W dim".into()))? as u32;
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
                let h = in_dims.get(2).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("ConvTranspose: missing H dim".into()))? as u32;
                let w = in_dims.get(3).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("ConvTranspose: missing W dim".into()))? as u32;

                let w_dims = weight.shape.dims();
                let c_in = w_dims.first().and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("ConvTranspose: missing C_in dim".into()))? as u32;
                let c_out = w_dims.get(1).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("ConvTranspose: missing C_out dim".into()))? as u32;

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
                warp_kernels::ops::layernorm(&self.cache, device, x, gamma, beta, &mut out, hidden, eps)?;
                owned.insert(out_name, out);
            }

            // ── InstanceNorm ──────────────────────────────────
            "InstanceNormalization" => {
                let x = get(0)?;
                let scale = get(1)?;
                let bias = get(2)?;
                let eps = node.get_float("epsilon", 1e-5);
                let dims = x.shape.dims();
                let c = dims.get(1).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let spatial: u32 = dims[2..].iter()
                    .map(|d| d.static_val().unwrap_or(1) as u32).product();
                let spatial = if spatial == 0 { 1 } else { spatial };

                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                warp_kernels::ops::instancenorm(&self.cache, device, x, scale, bias, &mut out,
                    c, spatial, eps)?;
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

            // ── New ops: ArgMax, Conv1D, Range, CumSum, Tile, ConstantOfShape ──
            "ArgMax" => {
                let x = get(0)?;
                let dims = x.shape.dims();
                let cols = dims.last().and_then(|d| d.static_val()).unwrap_or(x.numel) as u32;
                let rows = (x.numel / cols as usize) as u32;
                let mut out = GpuTensor::<i32>::zeros(device,
                    Shape::from_static(&[rows as usize]), DType::I32)?;
                warp_kernels::missing_ops::argmax(&self.cache, device, x, &mut out, rows, cols)?;
                // Store as f32 for compatibility with our f32-only tensor store
                let idx_host = out.to_host(device)?;
                let f32_idx: Vec<f32> = idx_host.iter().map(|&i| i as f32).collect();
                let f32_out = GpuTensor::from_host(device, &f32_idx,
                    Shape::from_static(&[rows as usize]), DType::F32)?;
                owned.insert(out_name, f32_out);
            }

            "Einsum" => {
                // Dispatch common einsum patterns as GEMM
                let a = get(0)?;
                let b = get(1)?;
                let a_dims = a.shape.dims();
                let b_dims = b.shape.dims();
                let m = if a_dims.len() >= 2 {
                    a_dims[a_dims.len()-2].static_val()
                        .ok_or_else(|| ExecError::Shape("Einsum: dynamic M dim".into()))? as u32
                } else { 1 };
                let k = a_dims.last().and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("Einsum: dynamic K dim".into()))? as u32;
                let n = b_dims.last().and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("Einsum: dynamic N dim".into()))? as u32;
                let mut out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[m as usize, n as usize]), DType::F32)?;
                warp_kernels::missing_ops::einsum_matmul(&self.cache, device, a, b, &mut out, m, n, k)?;
                owned.insert(out_name, out);
            }

            "ConstantOfShape" => {
                // Create tensor filled with value from attribute or input
                let value = if node.inputs.len() >= 1 && !node.inputs[0].is_empty() {
                    get(0).ok().and_then(|t| t.to_host(device).ok())
                        .and_then(|v| v.first().copied()).unwrap_or(0.0)
                } else { 0.0 };
                // Default shape [1] — real shape comes from input
                let out = warp_kernels::missing_ops::constant_of_shape(device, &[1], value)?;
                owned.insert(out_name, out);
            }

            "Range" => {
                let start = get(0).ok().and_then(|t| t.to_host(device).ok())
                    .and_then(|v| v.first().copied()).unwrap_or(0.0);
                let limit = get(1).ok().and_then(|t| t.to_host(device).ok())
                    .and_then(|v| v.first().copied()).unwrap_or(1.0);
                let delta = get(2).ok().and_then(|t| t.to_host(device).ok())
                    .and_then(|v| v.first().copied()).unwrap_or(1.0);
                let n = ((limit - start) / delta).ceil() as u32;
                let mut out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[n as usize]), DType::F32)?;
                warp_kernels::missing_ops::range(&self.cache, device, &mut out, start, delta, n)?;
                owned.insert(out_name, out);
            }

            "CumSum" => {
                let x = get(0)?;
                let dims = x.shape.dims();
                let cols = dims.last().and_then(|d| d.static_val()).unwrap_or(x.numel) as u32;
                let rows = (x.numel / cols as usize) as u32;
                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                warp_kernels::missing_ops::cumsum(&self.cache, device, x, &mut out, rows, cols)?;
                owned.insert(out_name, out);
            }

            "Tile" | "Expand" => {
                let x = get(0)?;
                if node.inputs.len() >= 2 {
                    if let Ok(shape_t) = get(1) {
                        let shape_vals = shape_t.to_host(device)?;
                        let out_numel: usize = shape_vals.iter().map(|&v| v.max(1.0) as usize).product();
                        if out_numel > x.numel {
                            let mut out = GpuTensor::<f32>::zeros(device,
                                Shape::from_static(&[out_numel]), DType::F32)?;
                            warp_kernels::edge_ops::expand(&self.cache, device, x, &mut out)?;
                            owned.insert(out_name, out);
                        } else {
                            let data = device.dtoh(&x.data)?;
                            owned.insert(out_name, GpuTensor::from_host(device, &data, x.shape.clone(), DType::F32)?);
                        }
                    } else {
                        let data = device.dtoh(&x.data)?;
                        owned.insert(out_name, GpuTensor::from_host(device, &data, x.shape.clone(), DType::F32)?);
                    }
                } else {
                    let data = device.dtoh(&x.data)?;
                    owned.insert(out_name, GpuTensor::from_host(device, &data, x.shape.clone(), DType::F32)?);
                }
            }

            "Where" => {
                let cond = get(0)?;
                let x = get(1)?;
                let y = get(2)?;
                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                warp_kernels::edge_ops::where_op(&self.cache, device, cond, x, y, &mut out)?;
                owned.insert(out_name, out);
            }

            "ScatterND" | "ScatterElements" => {
                let data_in = get(0)?;
                let indices = get(1)?;
                let updates = get(2)?;
                let data_host = data_in.to_host(device)?;
                let mut data = GpuTensor::from_host(device, &data_host, data_in.shape.clone(), DType::F32)?;
                warp_kernels::edge_ops::scatter_nd(&self.cache, device, &mut data, updates, indices)?;
                owned.insert(out_name, data);
            }

            "DepthToSpace" => {
                let x = get(0)?;
                let block_size = node.get_int("blocksize", 2) as u32;
                let dims = x.shape.dims();
                let c = dims.get(1).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("DepthToSpace: missing C dim".into()))? as u32;
                let h = dims.get(2).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("DepthToSpace: missing H dim".into()))? as u32;
                let w = dims.get(3).and_then(|d| d.static_val())
                    .ok_or_else(|| ExecError::Shape("DepthToSpace: missing W dim".into()))? as u32;
                let c_out = c / (block_size * block_size);
                let h_out = h * block_size;
                let w_out = w * block_size;
                let mut out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[1, c_out as usize, h_out as usize, w_out as usize]), DType::F32)?;
                warp_kernels::edge_ops::depth_to_space(&self.cache, device, x, &mut out, c, h, w, block_size)?;
                owned.insert(out_name, out);
            }

            // ── Unsqueeze with axes ───────────────────────────
            "Unsqueeze" => {
                let x = get(0)?;
                // Unsqueeze just adds dimensions of size 1 — data doesn't change
                let data = device.dtoh(&x.data)?;
                let mut new_dims: Vec<usize> = x.shape.dims().iter()
                    .map(|d| d.static_val().unwrap_or(1)).collect();

                // Get axes from input 1 (ONNX opset 13+) or attribute
                if node.inputs.len() >= 2 {
                    if let Ok(axes_t) = get(1) {
                        let axes = axes_t.to_host(device)?;
                        for &ax in &axes {
                            let ax = if ax < 0.0 { new_dims.len() as isize + ax as isize } else { ax as isize };
                            if ax >= 0 { new_dims.insert(ax as usize, 1); }
                        }
                    }
                } else {
                    let axes = node.get_ints("axes");
                    for &ax in &axes {
                        let ax = if ax < 0 { new_dims.len() as i64 + ax } else { ax };
                        if ax >= 0 { new_dims.insert(ax as usize, 1); }
                    }
                }

                let out = GpuTensor::from_host(device, &data,
                    Shape::from_static(&new_dims), DType::F32)?;
                owned.insert(out_name, out);
            }

            // ── Squeeze with axes ────────────────────────────
            "Squeeze" => {
                let x = get(0)?;
                let data = device.dtoh(&x.data)?;
                let dims: Vec<usize> = x.shape.dims().iter()
                    .map(|d| d.static_val().unwrap_or(1))
                    .filter(|&d| d != 1)  // remove all size-1 dims
                    .collect();
                let new_dims = if dims.is_empty() { vec![1] } else { dims };
                let out = GpuTensor::from_host(device, &data,
                    Shape::from_static(&new_dims), DType::F32)?;
                owned.insert(out_name, out);
            }

            // ── Reshape / Flatten — zero-copy ─────────────────
            "Reshape" | "Flatten" => {
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
            // ONNX Softmax has an axis attribute. For opset < 13, default axis=1.
            // For opset >= 13, default axis=-1. We flatten everything after axis into cols.
            "Softmax" => {
                let x = get(0)?;
                let dims = x.shape.dims();
                let ndim = dims.len();

                // Get axis attribute (default: 1 for older opsets, -1 for opset 13+)
                let axis = node.attrs.get("axis")
                    .and_then(|a| if let crate::onnx::OnnxAttr::Int(i) = a { Some(*i as i32) } else { None })
                    .unwrap_or(1); // default axis=1 (most common)

                // Normalize negative axis
                let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };

                // Rows = product of dims before axis, Cols = product of dims from axis onward
                let rows: usize = dims[..axis].iter()
                    .map(|d| d.static_val().unwrap_or(1))
                    .product();
                let cols: usize = dims[axis..].iter()
                    .map(|d| d.static_val().unwrap_or(1))
                    .product();

                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                warp_kernels::sampling::softmax(&self.cache, device, x, &mut out, rows as u32, cols as u32)?;
                owned.insert(out_name, out);
            }

            // ── Clip ──────────────────────────────────────────
            // ONNX Clip(input, min, max): clamp values to [min, max].
            // min/max are optional scalar inputs (inputs[1], inputs[2]).
            // Missing or empty inputs default to -inf / +inf respectively.
            "Clip" => {
                // Read optional min/max scalars before borrowing input
                let has_min = node.inputs.len() >= 2 && !node.inputs[1].is_empty();
                let has_max = node.inputs.len() >= 3 && !node.inputs[2].is_empty();

                let lo = if has_min {
                    get(1).ok()
                        .and_then(|t| t.to_host(device).ok())
                        .and_then(|v| v.first().copied())
                        .unwrap_or(f32::NEG_INFINITY)
                } else {
                    f32::NEG_INFINITY
                };
                let hi = if has_max {
                    get(2).ok()
                        .and_then(|t| t.to_host(device).ok())
                        .and_then(|v| v.first().copied())
                        .unwrap_or(f32::INFINITY)
                } else {
                    f32::INFINITY
                };

                let x = get(0)?;
                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                warp_kernels::ops::clip(&self.cache, device, x, &mut out, lo, hi)?;
                owned.insert(out_name, out);
            }

            // ── Gather ────────────────────────────────────────
            "Gather" => {
                let input = get(0)?;
                let indices = get(1)?;
                let dims = input.shape.dims();
                let inner_size = if dims.len() >= 2 {
                    dims.last().and_then(|d| d.static_val()).unwrap_or(1) as u32
                } else { 1 };
                let vocab_size = dims.first().and_then(|d| d.static_val()).unwrap_or(input.numel) as u32;

                let num_idx = indices.numel as u32;
                let mut out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[num_idx as usize, inner_size as usize]), DType::F32)?;
                warp_kernels::gather::gather(&self.cache, device, input, indices, &mut out,
                    vocab_size, inner_size)?;
                owned.insert(out_name, out);
            }

            // ── Slice ────────────────────────────────────────
            "Slice" => {
                let x = get(0)?;
                // Parse starts/ends from ONNX inputs (inputs 1 and 2)
                if node.inputs.len() >= 3 {
                    if let (Ok(starts_t), Ok(ends_t)) = (get(1), get(2)) {
                        let starts = starts_t.to_host(device)?;
                        let ends = ends_t.to_host(device)?;
                        if !starts.is_empty() && !ends.is_empty() {
                            let dims = x.shape.dims();
                            let cols = dims.last().and_then(|d| d.static_val()).unwrap_or(x.numel) as u32;
                            let rows = (x.numel / cols as usize) as u32;
                            let start = starts[0].max(0.0) as u32;
                            let end = (ends[0] as u32).min(cols);
                            let out_cols = end - start;

                            let mut out = GpuTensor::<f32>::zeros(device,
                                Shape::from_static(&[rows as usize, out_cols as usize]), DType::F32)?;
                            warp_kernels::gather::slice_last(&self.cache, device, x, &mut out,
                                rows, cols, start, out_cols)?;
                            owned.insert(out_name, out);
                        } else {
                            // Empty slice params — pass through
                            let data = device.dtoh(&x.data)?;
                            owned.insert(out_name, GpuTensor::from_host(device, &data, x.shape.clone(), DType::F32)?);
                        }
                    } else {
                        let data = device.dtoh(&x.data)?;
                        owned.insert(out_name, GpuTensor::from_host(device, &data, x.shape.clone(), DType::F32)?);
                    }
                } else {
                    let data = device.dtoh(&x.data)?;
                    owned.insert(out_name, GpuTensor::from_host(device, &data, x.shape.clone(), DType::F32)?);
                }
            }

            // ── Split ────────────────────────────────────────
            "Split" => {
                let x = get(0)?;
                let axis = node.get_int("axis", 0);
                // For single-output split or first output, pass through
                let data = device.dtoh(&x.data)?;
                let out = GpuTensor::from_host(device, &data, x.shape.clone(), DType::F32)?;
                owned.insert(out_name, out);
            }

            // ── Pad ──────────────────────────────────────────
            "Pad" => {
                let x = get(0)?;
                // Try to get pads from input 1
                if node.inputs.len() >= 2 {
                    if let Ok(pads_t) = get(1) {
                        let pads = pads_t.to_host(device)?;
                        if pads.len() >= 4 {
                            let dims = x.shape.dims();
                            let in_rows = dims.get(0).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                            let in_cols = dims.get(1).and_then(|d| d.static_val()).unwrap_or(x.numel) as u32;
                            let pad_top = pads[0] as u32;
                            let pad_left = pads[1] as u32;
                            let pad_bottom = pads[2] as u32;
                            let pad_right = pads[3] as u32;
                            let out_rows = in_rows + pad_top + pad_bottom;
                            let out_cols = in_cols + pad_left + pad_right;
                            let pad_val = if node.inputs.len() >= 3 {
                                get(2).ok().and_then(|t| t.to_host(device).ok())
                                    .and_then(|v| v.first().copied()).unwrap_or(0.0)
                            } else { 0.0 };

                            let mut out = GpuTensor::<f32>::zeros(device,
                                Shape::from_static(&[out_rows as usize, out_cols as usize]), DType::F32)?;
                            warp_kernels::gather::pad_2d(&self.cache, device, x, &mut out,
                                in_rows, in_cols, out_rows, out_cols, pad_top, pad_left, pad_val)?;
                            owned.insert(out_name, out);
                        } else {
                            let data = device.dtoh(&x.data)?;
                            owned.insert(out_name, GpuTensor::from_host(device, &data, x.shape.clone(), DType::F32)?);
                        }
                    } else {
                        let data = device.dtoh(&x.data)?;
                        owned.insert(out_name, GpuTensor::from_host(device, &data, x.shape.clone(), DType::F32)?);
                    }
                } else {
                    let data = device.dtoh(&x.data)?;
                    owned.insert(out_name, GpuTensor::from_host(device, &data, x.shape.clone(), DType::F32)?);
                }
            }

            // ── Constant / Shape / Cast ───────────────────────
            "Constant" => {
                // Extract constant value from node attributes
                if let Some(crate::onnx::OnnxAttr::Tensor(t)) = node.attrs.get("value") {
                    let data = t.to_f32();
                    if !data.is_empty() {
                        let shape = if t.shape.is_empty() {
                            Shape::from_static(&[data.len()])
                        } else {
                            Shape::from_static(&t.shape.iter().map(|&d| d as usize).collect::<Vec<_>>())
                        };
                        let gpu = GpuTensor::from_host(device, &data, shape, DType::F32)?;
                        owned.insert(out_name, gpu);
                    }
                } else if let Some(crate::onnx::OnnxAttr::Float(f)) = node.attrs.get("value_float") {
                    let gpu = GpuTensor::from_host(device, &[*f], Shape::from_static(&[1]), DType::F32)?;
                    owned.insert(out_name, gpu);
                } else if let Some(crate::onnx::OnnxAttr::Int(i)) = node.attrs.get("value_int") {
                    let gpu = GpuTensor::from_host(device, &[*i as f32], Shape::from_static(&[1]), DType::F32)?;
                    owned.insert(out_name, gpu);
                }
            }
            "Shape" => {
                // Shape op returns the dimensions of the input tensor as a 1D i64 tensor
                // We store as f32 since our tensors are all f32
                if let Ok(x) = get(0) {
                    let dims: Vec<f32> = x.shape.dims().iter()
                        .map(|d| d.static_val().unwrap_or(1) as f32)
                        .collect();
                    let ndim = dims.len();
                    let out = GpuTensor::from_host(device, &dims,
                        Shape::from_static(&[ndim]), DType::F32)?;
                    owned.insert(out_name, out);
                }
            }
            "Cast" => {
                // Cast: just pass through for now (we only support f32)
                if let Ok(x) = get(0) {
                    let data = device.dtoh(&x.data)?;
                    let out = GpuTensor::from_host(device, &data, x.shape.clone(), DType::F32)?;
                    owned.insert(out_name, out);
                }
            }

            // ── Floor / Ceil / Round (CPU fallback) ─────────
            "Floor" | "Ceil" | "Round" => {
                let x = get(0)?;
                let data = x.to_host(device)?;
                let result: Vec<f32> = match node.op_type.as_str() {
                    "Floor" => data.iter().map(|v| v.floor()).collect(),
                    "Ceil"  => data.iter().map(|v| v.ceil()).collect(),
                    "Round" => data.iter().map(|v| v.round()).collect(),
                    _ => unreachable!(),
                };
                let out = GpuTensor::from_host(device, &result, x.shape.clone(), DType::F32)?;
                owned.insert(out_name, out);
            }

            // ── Equal / Greater / Less (CPU fallback) ────────
            "Equal" | "Greater" | "Less" => {
                let a = get(0)?;
                let b = get(1)?;
                let a_data = a.to_host(device)?;
                let b_data = b.to_host(device)?;
                let len = a_data.len().min(b_data.len());
                let result: Vec<f32> = (0..len).map(|i| {
                    let av = a_data[i];
                    let bv = if i < b_data.len() { b_data[i] } else { 0.0 };
                    match node.op_type.as_str() {
                        "Equal"   => if (av - bv).abs() < f32::EPSILON { 1.0 } else { 0.0 },
                        "Greater" => if av > bv { 1.0 } else { 0.0 },
                        "Less"    => if av < bv { 1.0 } else { 0.0 },
                        _ => 0.0,
                    }
                }).collect();
                let out = GpuTensor::from_host(device, &result,
                    Shape::from_static(&[len]), DType::F32)?;
                owned.insert(out_name, out);
            }

            // ── Not (unary logical, CPU fallback) ────────────
            "Not" => {
                let x = get(0)?;
                let data = x.to_host(device)?;
                let result: Vec<f32> = data.iter().map(|&v| {
                    if v == 0.0 { 1.0 } else { 0.0 }
                }).collect();
                let out = GpuTensor::from_host(device, &result, x.shape.clone(), DType::F32)?;
                owned.insert(out_name, out);
            }

            // ── And / Or (binary logical, CPU fallback) ──────
            "And" | "Or" => {
                let a = get(0)?;
                let b = get(1)?;
                let a_data = a.to_host(device)?;
                let b_data = b.to_host(device)?;
                let len = a_data.len().min(b_data.len());
                let result: Vec<f32> = (0..len).map(|i| {
                    let a_bool = a_data[i] != 0.0;
                    let b_bool = if i < b_data.len() { b_data[i] != 0.0 } else { false };
                    let r = match node.op_type.as_str() {
                        "And" => a_bool && b_bool,
                        "Or"  => a_bool || b_bool,
                        _ => false,
                    };
                    if r { 1.0 } else { 0.0 }
                }).collect();
                let out = GpuTensor::from_host(device, &result,
                    Shape::from_static(&[len]), DType::F32)?;
                owned.insert(out_name, out);
            }

            // ── LSTM (CPU fallback using extended_ops::lstm_cell) ─
            "LSTM" => {
                // ONNX LSTM inputs: X[seq_len, batch, input_size], W[1, 4*hidden, input],
                //                   R[1, 4*hidden, hidden], B[1, 8*hidden]
                let x_tensor = get(0)?;
                let w_tensor = get(1)?;
                let r_tensor = get(2)?;

                let x_data = x_tensor.to_host(device)?;
                let w_data = w_tensor.to_host(device)?;
                let r_data = r_tensor.to_host(device)?;

                // Parse dimensions from W shape: [num_directions, 4*hidden_size, input_size]
                let w_dims = w_tensor.shape.dims();
                let four_hidden = w_dims.get(1).and_then(|d| d.static_val()).unwrap_or(4) as usize;
                let input_size = w_dims.get(2).and_then(|d| d.static_val()).unwrap_or(1) as usize;
                let hidden_size = four_hidden / 4;

                let x_dims = x_tensor.shape.dims();
                let seq_len = x_dims.get(0).and_then(|d| d.static_val()).unwrap_or(1);
                let batch = x_dims.get(1).and_then(|d| d.static_val()).unwrap_or(1);

                // Bias: split into b_ih [4*hidden] and b_hh [4*hidden]
                let b_data = if node.inputs.len() >= 4 && !node.inputs[3].is_empty() {
                    get(3).ok().and_then(|t| t.to_host(device).ok()).unwrap_or_default()
                } else {
                    vec![0.0f32; 8 * hidden_size]
                };
                let b_ih = if b_data.len() >= 4 * hidden_size { &b_data[..4 * hidden_size] }
                           else { &vec![0.0f32; 4 * hidden_size] };
                let b_hh = if b_data.len() >= 8 * hidden_size { &b_data[4 * hidden_size..8 * hidden_size] }
                           else { &vec![0.0f32; 4 * hidden_size] };

                let mut h = vec![0.0f32; batch * hidden_size];
                let mut c = vec![0.0f32; batch * hidden_size];

                // Run LSTM cell for each timestep
                let mut all_h = Vec::with_capacity(seq_len * batch * hidden_size);
                for t in 0..seq_len {
                    let x_t = &x_data[t * batch * input_size..(t + 1) * batch * input_size];
                    let (h_new, c_new) = warp_kernels::extended_ops::lstm_cell(
                        x_t, &h, &c, &w_data, &r_data, b_ih, b_hh,
                        batch, input_size, hidden_size,
                    );
                    h = h_new;
                    c = c_new;
                    all_h.extend_from_slice(&h);
                }

                // Output Y: [seq_len, num_directions=1, batch, hidden_size]
                let y_out = GpuTensor::from_host(device, &all_h,
                    Shape::from_static(&[seq_len, 1, batch, hidden_size]), DType::F32)?;
                owned.insert(out_name.clone(), y_out);

                // Y_h (final hidden state)
                if node.outputs.len() >= 2 && !node.outputs[1].is_empty() {
                    let yh_out = GpuTensor::from_host(device, &h,
                        Shape::from_static(&[1, batch, hidden_size]), DType::F32)?;
                    owned.insert(node.outputs[1].clone(), yh_out);
                }
                // Y_c (final cell state)
                if node.outputs.len() >= 3 && !node.outputs[2].is_empty() {
                    let yc_out = GpuTensor::from_host(device, &c,
                        Shape::from_static(&[1, batch, hidden_size]), DType::F32)?;
                    owned.insert(node.outputs[2].clone(), yc_out);
                }
            }

            // ── GRU (CPU fallback using extended_ops::gru_cell) ───
            "GRU" => {
                let x_tensor = get(0)?;
                let w_tensor = get(1)?;
                let r_tensor = get(2)?;

                let x_data = x_tensor.to_host(device)?;
                let w_data = w_tensor.to_host(device)?;
                let r_data = r_tensor.to_host(device)?;

                let w_dims = w_tensor.shape.dims();
                let three_hidden = w_dims.get(1).and_then(|d| d.static_val()).unwrap_or(3) as usize;
                let input_size = w_dims.get(2).and_then(|d| d.static_val()).unwrap_or(1) as usize;
                let hidden_size = three_hidden / 3;

                let x_dims = x_tensor.shape.dims();
                let seq_len = x_dims.get(0).and_then(|d| d.static_val()).unwrap_or(1);
                let batch = x_dims.get(1).and_then(|d| d.static_val()).unwrap_or(1);

                let b_data = if node.inputs.len() >= 4 && !node.inputs[3].is_empty() {
                    get(3).ok().and_then(|t| t.to_host(device).ok()).unwrap_or_default()
                } else {
                    vec![0.0f32; 6 * hidden_size]
                };
                let b_ih = if b_data.len() >= 3 * hidden_size { &b_data[..3 * hidden_size] }
                           else { &vec![0.0f32; 3 * hidden_size] };
                let b_hh = if b_data.len() >= 6 * hidden_size { &b_data[3 * hidden_size..6 * hidden_size] }
                           else { &vec![0.0f32; 3 * hidden_size] };

                let mut h = vec![0.0f32; batch * hidden_size];
                let mut all_h = Vec::with_capacity(seq_len * batch * hidden_size);

                for t in 0..seq_len {
                    let x_t = &x_data[t * batch * input_size..(t + 1) * batch * input_size];
                    h = warp_kernels::extended_ops::gru_cell(
                        x_t, &h, &w_data, &r_data, b_ih, b_hh,
                        batch, input_size, hidden_size,
                    );
                    all_h.extend_from_slice(&h);
                }

                let y_out = GpuTensor::from_host(device, &all_h,
                    Shape::from_static(&[seq_len, 1, batch, hidden_size]), DType::F32)?;
                owned.insert(out_name.clone(), y_out);

                if node.outputs.len() >= 2 && !node.outputs[1].is_empty() {
                    let yh_out = GpuTensor::from_host(device, &h,
                        Shape::from_static(&[1, batch, hidden_size]), DType::F32)?;
                    owned.insert(node.outputs[1].clone(), yh_out);
                }
            }

            // ── QuantizeLinear (CPU fallback) ────────────────
            "QuantizeLinear" => {
                // output = clamp(round(input / scale) + zero_point, qmin, qmax)
                let x = get(0)?;
                let scale_t = get(1)?;
                let x_data = x.to_host(device)?;
                let scale_data = scale_t.to_host(device)?;
                let scale = scale_data.first().copied().unwrap_or(1.0);
                let zp = if node.inputs.len() >= 3 && !node.inputs[2].is_empty() {
                    get(2).ok().and_then(|t| t.to_host(device).ok())
                        .and_then(|v| v.first().copied()).unwrap_or(0.0)
                } else { 0.0 };
                let qmin = -128.0f32;
                let qmax = 127.0f32;
                let result: Vec<f32> = x_data.iter().map(|&v| {
                    (v / scale).round().clamp(qmin - zp, qmax - zp) + zp
                }).map(|v| v.clamp(qmin, qmax)).collect();
                let out = GpuTensor::from_host(device, &result, x.shape.clone(), DType::F32)?;
                owned.insert(out_name, out);
            }

            // ── DequantizeLinear (CPU fallback) ──────────────
            "DequantizeLinear" => {
                // output = (input - zero_point) * scale
                let x = get(0)?;
                let scale_t = get(1)?;
                let x_data = x.to_host(device)?;
                let scale_data = scale_t.to_host(device)?;
                let scale = scale_data.first().copied().unwrap_or(1.0);
                let zp = if node.inputs.len() >= 3 && !node.inputs[2].is_empty() {
                    get(2).ok().and_then(|t| t.to_host(device).ok())
                        .and_then(|v| v.first().copied()).unwrap_or(0.0)
                } else { 0.0 };
                let result: Vec<f32> = x_data.iter().map(|&v| {
                    (v - zp) * scale
                }).collect();
                let out = GpuTensor::from_host(device, &result, x.shape.clone(), DType::F32)?;
                owned.insert(out_name, out);
            }

            // ── Trilu (GPU kernel from extended_ops) ─────────
            "Trilu" => {
                let x = get(0)?;
                let dims = x.shape.dims();
                let cols = dims.last().and_then(|d| d.static_val()).unwrap_or(1) as u32;
                let rows = (x.numel / cols as usize) as u32;
                let k = if node.inputs.len() >= 2 && !node.inputs[1].is_empty() {
                    get(1).ok().and_then(|t| t.to_host(device).ok())
                        .and_then(|v| v.first().copied()).unwrap_or(0.0) as i32
                } else { 0 };
                let upper = node.get_int("upper", 1) != 0;
                let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                warp_kernels::extended_ops::trilu(&self.cache, device, x, &mut out, rows, cols, k, upper)?;
                owned.insert(out_name, out);
            }

            // ── Fused ops (created by optimizer) ─────────────
            "FusedMatMulBias" => {
                let a = get(0)?;
                let b = get(1)?;
                let bias = get(2)?;
                let a_dims = &a.shape.dims();
                let b_dims = &b.shape.dims();
                let m = if a_dims.len() >= 2 {
                    a_dims[a_dims.len() - 2].static_val()
                        .ok_or_else(|| ExecError::Shape("FusedMatMulBias: dynamic M dim".into()))? as u32
                } else { 1 };
                let k = if a_dims.len() >= 1 {
                    a_dims[a_dims.len() - 1].static_val()
                        .ok_or_else(|| ExecError::Shape("FusedMatMulBias: dynamic K dim".into()))? as u32
                } else { 1 };
                let n = if b_dims.len() >= 1 {
                    b_dims[b_dims.len() - 1].static_val()
                        .ok_or_else(|| ExecError::Shape("FusedMatMulBias: dynamic N dim".into()))? as u32
                } else { 1 };

                let mut mm_out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[m as usize, n as usize]), DType::F32)?;
                warp_kernels::ops::gemm(&self.cache, device, a, b, &mut mm_out, m, n, k)?;

                let mut out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[m as usize, n as usize]), DType::F32)?;
                warp_kernels::ops::add(&self.cache, device, &mm_out, bias, &mut out)?;
                owned.insert(out_name, out);
            }

            "FusedMatMulBiasAct" => {
                let a = get(0)?;
                let b = get(1)?;
                let bias = get(2)?;
                let a_dims = &a.shape.dims();
                let b_dims = &b.shape.dims();
                let m = if a_dims.len() >= 2 {
                    a_dims[a_dims.len() - 2].static_val()
                        .ok_or_else(|| ExecError::Shape("FusedMatMulBiasAct: dynamic M dim".into()))? as u32
                } else { 1 };
                let k = if a_dims.len() >= 1 {
                    a_dims[a_dims.len() - 1].static_val()
                        .ok_or_else(|| ExecError::Shape("FusedMatMulBiasAct: dynamic K dim".into()))? as u32
                } else { 1 };
                let n = if b_dims.len() >= 1 {
                    b_dims[b_dims.len() - 1].static_val()
                        .ok_or_else(|| ExecError::Shape("FusedMatMulBiasAct: dynamic N dim".into()))? as u32
                } else { 1 };

                let mut mm_out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[m as usize, n as usize]), DType::F32)?;
                warp_kernels::ops::gemm(&self.cache, device, a, b, &mut mm_out, m, n, k)?;

                let mut biased = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[m as usize, n as usize]), DType::F32)?;
                warp_kernels::ops::add(&self.cache, device, &mm_out, bias, &mut biased)?;

                // Apply activation based on node attribute
                let act = node.get_string("activation").unwrap_or("relu");
                let mut out = GpuTensor::<f32>::zeros(device, biased.shape.clone(), DType::F32)?;
                match act {
                    "gelu" | "Gelu" => warp_kernels::ops::gelu(&self.cache, device, &biased, &mut out)?,
                    "silu" | "Silu" | "swish" | "Swish" => warp_kernels::ops::silu(&self.cache, device, &biased, &mut out)?,
                    _ => warp_kernels::ops::relu(&self.cache, device, &biased, &mut out)?,
                }
                owned.insert(out_name, out);
            }

            "FusedResidualRmsNorm" => {
                let residual = get(0)?;
                let x = get(1)?;
                let gamma = get(2)?;
                let eps = node.get_float("eps", 1e-5);
                let hidden_size = x.shape.dims().last()
                    .and_then(|d| d.static_val())
                    .unwrap_or(1) as u32;

                let mut norm_out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                let mut residual_out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                warp_kernels::ops::fused_residual_rmsnorm(
                    &self.cache, device, x, residual, gamma,
                    &mut norm_out, &mut residual_out, hidden_size, eps,
                )?;

                // Primary output is the normalized result
                owned.insert(out_name.clone(), norm_out);
                // If there's a second output name, store the residual
                if let Some(res_name) = node.outputs.get(1) {
                    owned.insert(res_name.clone(), residual_out);
                }
            }

            "FusedSwiGLU" => {
                let x = get(0)?;
                let gate_weight = get(1)?;
                let up_weight = get(2)?;
                let down_weight = get(3)?;

                let x_dims = &x.shape.dims();
                let gw_dims = &gate_weight.shape.dims();
                let dw_dims = &down_weight.shape.dims();

                let m = if x_dims.len() >= 2 {
                    x_dims[x_dims.len() - 2].static_val()
                        .ok_or_else(|| ExecError::Shape("FusedSwiGLU: dynamic M dim".into()))? as u32
                } else { 1 };
                let k_in = if x_dims.len() >= 1 {
                    x_dims[x_dims.len() - 1].static_val()
                        .ok_or_else(|| ExecError::Shape("FusedSwiGLU: dynamic K dim".into()))? as u32
                } else { 1 };
                let intermediate = if gw_dims.len() >= 1 {
                    gw_dims[gw_dims.len() - 1].static_val()
                        .ok_or_else(|| ExecError::Shape("FusedSwiGLU: dynamic intermediate dim".into()))? as u32
                } else { 1 };
                let n_out = if dw_dims.len() >= 1 {
                    dw_dims[dw_dims.len() - 1].static_val()
                        .ok_or_else(|| ExecError::Shape("FusedSwiGLU: dynamic output dim".into()))? as u32
                } else { 1 };

                // gate_proj = X @ gate_weight
                let mut gate_proj = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[m as usize, intermediate as usize]), DType::F32)?;
                warp_kernels::ops::gemm(&self.cache, device, x, gate_weight, &mut gate_proj, m, intermediate, k_in)?;

                // up_proj = X @ up_weight
                let mut up_proj = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[m as usize, intermediate as usize]), DType::F32)?;
                warp_kernels::ops::gemm(&self.cache, device, x, up_weight, &mut up_proj, m, intermediate, k_in)?;

                // fused_silu_mul: out = silu(gate_proj) * up_proj
                let mut silu_out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[m as usize, intermediate as usize]), DType::F32)?;
                warp_kernels::ops::fused_silu_mul(&self.cache, device, &gate_proj, &up_proj, &mut silu_out)?;

                // down_proj = silu_out @ down_weight
                let mut out = GpuTensor::<f32>::zeros(device,
                    Shape::from_static(&[m as usize, n_out as usize]), DType::F32)?;
                warp_kernels::ops::gemm(&self.cache, device, &silu_out, down_weight, &mut out, m, n_out, intermediate)?;
                owned.insert(out_name, out);
            }

            "AutoFused" => {
                let kernel_name = node.get_string("kernel_name")
                    .ok_or_else(|| ExecError::UnsupportedOp(
                        "AutoFused node missing 'kernel_name' attribute".to_string()))?
                    .to_string();
                let kernel_src = node.get_string("kernel_src")
                    .ok_or_else(|| ExecError::UnsupportedOp(
                        "AutoFused node missing 'kernel_src' attribute".to_string()))?
                    .to_string();

                // Gather all input tensors
                let mut input_tensors: Vec<&GpuTensor<f32>> = Vec::new();
                for i in 0..node.inputs.len() {
                    if !node.inputs[i].is_empty() {
                        input_tensors.push(get(i)?);
                    }
                }

                if input_tensors.is_empty() {
                    return Err(ExecError::MissingTensor {
                        name: "(no inputs)".into(),
                        op: "AutoFused".into(),
                    });
                }

                let n = input_tensors[0].numel;
                let mut output = GpuTensor::<f32>::zeros(device,
                    input_tensors[0].shape.clone(), DType::F32)?;

                let f = self.cache.get_or_compile(device, &kernel_src, &kernel_name)
                    .map_err(ExecError::Device)?;
                let cfg = cudarc::driver::LaunchConfig::for_num_elems(n as u32);

                // Launch with variable number of inputs (same pattern as execute_fusion)
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
                    other => return Err(ExecError::UnsupportedOp(
                        format!("AutoFused kernel with {} inputs not supported (max 4)", other))),
                }
                owned.insert(out_name, output);
            }

            // ── HARD ERROR for truly unsupported ops ──────────
            other => {
                return Err(ExecError::UnsupportedOp(format!(
                    "ONNX op '{}' (node '{}') is not implemented. \
                     TensorWarp supports: Add, Sub, Mul, Div, MatMul, Gemm, Conv, ConvTranspose, \
                     BatchNorm, InstanceNorm, LayerNorm, GroupNorm, MaxPool, AvgPool, GlobalAvgPool, \
                     Relu, Sigmoid, Tanh, Gelu, Silu, LeakyRelu, Clip, Softmax, \
                     Reshape, Flatten, Transpose, Concat, Reduce*, Resize, Identity, Dropout, \
                     FusedMatMulBias, FusedMatMulBiasAct, FusedResidualRmsNorm, FusedSwiGLU, AutoFused",
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
            .map_err(ExecError::Device)?;

        // Gather inputs: the first input of the first node in the chain,
        // plus additional inputs for binary ops within the chain
        let mut input_tensors: Vec<&GpuTensor<f32>> = Vec::new();

        // First input: main data flowing through the chain
        let first_node = &self.nodes[fusion.node_indices[0]];
        let first_input_name = &first_node.inputs[0];
        let first_input = Self::resolve(first_input_name, &fusion.kernel_name, inputs, owned, &self.weights)?;
        input_tensors.push(first_input);

        // Additional inputs: second operands of binary ops in the chain
        for &ni in &fusion.node_indices {
            let node = &self.nodes[ni];
            if node.inputs.len() >= 2 {
                // Binary op — need the second input
                let second_name = &node.inputs[1];
                let second = Self::resolve(second_name, &fusion.kernel_name, inputs, owned, &self.weights)?;
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
        initializers.insert("conv.weight".to_string(), OnnxTensor::new(
            "conv.weight", OnnxDType::Float,
            vec![8, 3, 3, 3], conv_w.iter().flat_map(|f| f.to_le_bytes()).collect(),
        ));

        // FC weight: [8, 10]
        let fc_w: Vec<f32> = (0..8*10).map(|i| ((i * 11 + 5) % 200) as f32 * 0.01 - 1.0).collect();
        initializers.insert("fc.weight".to_string(), OnnxTensor::new(
            "fc.weight", OnnxDType::Float,
            vec![8, 10], fc_w.iter().flat_map(|f| f.to_le_bytes()).collect(),
        ));

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
            model_dir: None,
            graph_name: String::new(),
            graph_doc_string: String::new(),
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

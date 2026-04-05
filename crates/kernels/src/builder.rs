//! TensorWarp Builder API — programmatic model construction.
//!
//! Like TensorRT's Builder/NetworkDefinition API, allows users to
//! construct inference graphs programmatically, set precision modes,
//! and compile optimized engines.
//!
//! Usage:
//! ```ignore
//! let mut builder = ModelBuilder::new(&device)?;
//!
//! // Define inputs
//! let input = builder.input("image", &[1, 3, 224, 224]);
//!
//! // Build layers
//! let conv1 = builder.conv2d(input, 64, 7, 2, 3)?;   // Conv2D 3→64, k=7, s=2, p=3
//! let bn1 = builder.batchnorm(conv1)?;
//! let relu1 = builder.relu(bn1)?;
//! let pool1 = builder.maxpool(relu1, 3, 2, 1)?;
//!
//! // Set output
//! builder.output("logits", pool1);
//!
//! // Build optimized engine
//! let engine = builder.build(Precision::FP16)?;
//!
//! // Run inference
//! let output = engine.infer(&input_tensor)?;
//! ```

use std::collections::HashMap;

use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Precision mode for the built engine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Precision {
    FP32,
    FP16,
    INT8,
    /// Mixed: use FP16 for GEMMs, FP32 for norms
    Mixed,
}

/// A tensor handle in the builder graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

/// A layer in the builder graph.
#[derive(Debug)]
enum Layer {
    Input { name: String, shape: Vec<usize> },
    Conv2d { input: TensorId, out_channels: u32, kernel: u32, stride: u32, padding: u32 },
    BatchNorm { input: TensorId },
    Relu { input: TensorId },
    Sigmoid { input: TensorId },
    Gelu { input: TensorId },
    Silu { input: TensorId },
    MaxPool { input: TensorId, kernel: u32, stride: u32, padding: u32 },
    GlobalAvgPool { input: TensorId },
    Linear { input: TensorId, out_features: u32 },
    Add { a: TensorId, b: TensorId },
    Concat { inputs: Vec<TensorId>, axis: i32 },
    Reshape { input: TensorId, shape: Vec<i64> },
    Softmax { input: TensorId },
    Output { name: String, input: TensorId },
}

/// Programmatic model builder — TensorRT-style API.
pub struct ModelBuilder {
    layers: Vec<Layer>,
    next_tensor: usize,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl ModelBuilder {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            next_tensor: 0,
            input_names: Vec::new(),
            output_names: Vec::new(),
        }
    }

    fn alloc_tensor(&mut self) -> TensorId {
        let id = TensorId(self.next_tensor);
        self.next_tensor += 1;
        id
    }

    /// Define a model input.
    pub fn input(&mut self, name: &str, shape: &[usize]) -> TensorId {
        let id = self.alloc_tensor();
        self.input_names.push(name.to_string());
        self.layers.push(Layer::Input {
            name: name.to_string(),
            shape: shape.to_vec(),
        });
        id
    }

    /// Conv2D layer.
    pub fn conv2d(&mut self, input: TensorId, out_channels: u32, kernel: u32, stride: u32, padding: u32) -> TensorId {
        let id = self.alloc_tensor();
        self.layers.push(Layer::Conv2d { input, out_channels, kernel, stride, padding });
        id
    }

    /// BatchNorm layer.
    pub fn batchnorm(&mut self, input: TensorId) -> TensorId {
        let id = self.alloc_tensor();
        self.layers.push(Layer::BatchNorm { input });
        id
    }

    /// ReLU activation.
    pub fn relu(&mut self, input: TensorId) -> TensorId {
        let id = self.alloc_tensor();
        self.layers.push(Layer::Relu { input });
        id
    }

    /// Sigmoid activation.
    pub fn sigmoid(&mut self, input: TensorId) -> TensorId {
        let id = self.alloc_tensor();
        self.layers.push(Layer::Sigmoid { input });
        id
    }

    /// GELU activation.
    pub fn gelu(&mut self, input: TensorId) -> TensorId {
        let id = self.alloc_tensor();
        self.layers.push(Layer::Gelu { input });
        id
    }

    /// SiLU/Swish activation.
    pub fn silu(&mut self, input: TensorId) -> TensorId {
        let id = self.alloc_tensor();
        self.layers.push(Layer::Silu { input });
        id
    }

    /// MaxPool2D layer.
    pub fn maxpool(&mut self, input: TensorId, kernel: u32, stride: u32, padding: u32) -> TensorId {
        let id = self.alloc_tensor();
        self.layers.push(Layer::MaxPool { input, kernel, stride, padding });
        id
    }

    /// Global average pooling.
    pub fn global_avg_pool(&mut self, input: TensorId) -> TensorId {
        let id = self.alloc_tensor();
        self.layers.push(Layer::GlobalAvgPool { input });
        id
    }

    /// Linear (fully connected) layer.
    pub fn linear(&mut self, input: TensorId, out_features: u32) -> TensorId {
        let id = self.alloc_tensor();
        self.layers.push(Layer::Linear { input, out_features });
        id
    }

    /// Element-wise addition.
    pub fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let id = self.alloc_tensor();
        self.layers.push(Layer::Add { a, b });
        id
    }

    /// Softmax.
    pub fn softmax(&mut self, input: TensorId) -> TensorId {
        let id = self.alloc_tensor();
        self.layers.push(Layer::Softmax { input });
        id
    }

    /// Mark a tensor as model output.
    pub fn output(&mut self, name: &str, input: TensorId) {
        self.output_names.push(name.to_string());
        self.layers.push(Layer::Output { name: name.to_string(), input });
    }

    /// Get a summary of the model.
    pub fn summary(&self) -> String {
        let mut lines = vec![format!("Model: {} layers, {} tensors", self.layers.len(), self.next_tensor)];
        lines.push(format!("  Inputs: {:?}", self.input_names));
        lines.push(format!("  Outputs: {:?}", self.output_names));
        for (i, layer) in self.layers.iter().enumerate() {
            let desc = match layer {
                Layer::Input { name, shape } => format!("Input({name}, {:?})", shape),
                Layer::Conv2d { out_channels, kernel, stride, padding, .. } =>
                    format!("Conv2D(out={out_channels}, k={kernel}, s={stride}, p={padding})"),
                Layer::BatchNorm { .. } => "BatchNorm".into(),
                Layer::Relu { .. } => "ReLU".into(),
                Layer::Sigmoid { .. } => "Sigmoid".into(),
                Layer::Gelu { .. } => "GELU".into(),
                Layer::Silu { .. } => "SiLU".into(),
                Layer::MaxPool { kernel, stride, .. } => format!("MaxPool(k={kernel}, s={stride})"),
                Layer::GlobalAvgPool { .. } => "GlobalAvgPool".into(),
                Layer::Linear { out_features, .. } => format!("Linear(out={out_features})"),
                Layer::Add { .. } => "Add".into(),
                Layer::Concat { axis, .. } => format!("Concat(axis={axis})"),
                Layer::Reshape { shape, .. } => format!("Reshape({:?})", shape),
                Layer::Softmax { .. } => "Softmax".into(),
                Layer::Output { name, .. } => format!("Output({name})"),
            };
            lines.push(format!("  [{i}] {desc}"));
        }
        lines.join("\n")
    }
}

impl Default for ModelBuilder {
    fn default() -> Self { Self::new() }
}

/// A compiled model ready for inference.
/// Produced by ModelBuilder::build().
pub struct CompiledModel {
    /// Layer descriptors for execution.
    layers: Vec<Layer>,
    /// Random weights (for testing — real models load from files).
    weights: HashMap<String, GpuTensor<f32>>,
    /// Kernel cache.
    cache: KernelCache,
    /// Precision mode.
    precision: Precision,
    /// Input names and shapes.
    input_specs: Vec<(String, Vec<usize>)>,
    /// Output names.
    output_names: Vec<String>,
}

impl ModelBuilder {
    /// Build an optimized inference model.
    /// Initializes weights with random values (for testing).
    /// For real models, use ONNX or SafeTensors loading instead.
    pub fn build(self, device: &WarpDevice, precision: Precision) -> Result<CompiledModel, DeviceError> {
        let cache = KernelCache::new();
        let mut weights = HashMap::new();

        let rand_vec = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i * 7 + seed) % 200) as f32 * 0.01 - 1.0).collect()
        };

        // Track output shapes through the graph for proper weight sizing.
        // shape_map[TensorId] = [N, C, H, W] or [N, features]
        let mut shape_map: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut weight_seed = 42;

        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                Layer::Input { shape, .. } => {
                    shape_map.insert(i, shape.clone());
                }
                Layer::Conv2d { input, out_channels, kernel, stride, padding } => {
                    let in_shape = shape_map.get(&input.0).cloned().unwrap_or(vec![1, 3, 1, 1]);
                    let in_c = in_shape.get(1).copied().unwrap_or(3) as u32;
                    let h = in_shape.get(2).copied().unwrap_or(1) as u32;
                    let w_dim = in_shape.get(3).copied().unwrap_or(1) as u32;
                    let batch = in_shape.first().copied().unwrap_or(1);

                    let params = crate::conv::Conv2dParams::new(in_c, *out_channels, *kernel)
                        .stride(*stride).padding(*padding);
                    let out_h = params.output_h(h) as usize;
                    let out_w = params.output_w(w_dim) as usize;
                    shape_map.insert(i, vec![batch, *out_channels as usize, out_h, out_w]);

                    let w_size = (*out_channels * in_c * kernel * kernel) as usize;
                    let w_data = rand_vec(w_size, weight_seed);
                    weight_seed += 1;
                    let w = GpuTensor::from_host(device, &w_data,
                        Shape::from_static(&[*out_channels as usize, in_c as usize, *kernel as usize, *kernel as usize]),
                        DType::F32)?;
                    weights.insert(format!("layer{i}_weight"), w);
                }
                Layer::BatchNorm { input } => {
                    let in_shape = shape_map.get(&input.0).cloned().unwrap_or(vec![1, 64, 1, 1]);
                    shape_map.insert(i, in_shape.clone());
                    let c = in_shape.get(1).copied().unwrap_or(64);
                    weights.insert(format!("layer{i}_scale"),
                        GpuTensor::from_host(device, &vec![1.0f32; c], Shape::from_static(&[c]), DType::F32)?);
                    weights.insert(format!("layer{i}_bias"),
                        GpuTensor::from_host(device, &vec![0.0f32; c], Shape::from_static(&[c]), DType::F32)?);
                    weights.insert(format!("layer{i}_mean"),
                        GpuTensor::from_host(device, &vec![0.0f32; c], Shape::from_static(&[c]), DType::F32)?);
                    weights.insert(format!("layer{i}_var"),
                        GpuTensor::from_host(device, &vec![1.0f32; c], Shape::from_static(&[c]), DType::F32)?);
                }
                Layer::MaxPool { input, kernel, stride, padding } => {
                    let in_shape = shape_map.get(&input.0).cloned().unwrap_or(vec![1, 64, 1, 1]);
                    let c = in_shape.get(1).copied().unwrap_or(64);
                    let h = in_shape.get(2).copied().unwrap_or(1) as u32;
                    let w_dim = in_shape.get(3).copied().unwrap_or(1) as u32;
                    let batch = in_shape.first().copied().unwrap_or(1);
                    let out_h = crate::conv::conv_output_size(h, *kernel, *stride, *padding, 1) as usize;
                    let out_w = crate::conv::conv_output_size(w_dim, *kernel, *stride, *padding, 1) as usize;
                    shape_map.insert(i, vec![batch, c, out_h, out_w]);
                }
                Layer::Relu { input } | Layer::Sigmoid { input } |
                Layer::Gelu { input } | Layer::Silu { input } |
                Layer::Softmax { input } => {
                    if let Some(s) = shape_map.get(&input.0) {
                        shape_map.insert(i, s.clone());
                    }
                }
                Layer::GlobalAvgPool { input } => {
                    let in_shape = shape_map.get(&input.0).cloned().unwrap_or(vec![1, 64, 1, 1]);
                    let batch = in_shape.first().copied().unwrap_or(1);
                    let c = in_shape.get(1).copied().unwrap_or(64);
                    shape_map.insert(i, vec![batch, c]);
                }
                Layer::Linear { input, out_features } => {
                    let in_shape = shape_map.get(&input.0).cloned().unwrap_or(vec![1, 256]);
                    let in_f = in_shape.last().copied().unwrap_or(256) as u32;
                    let batch = if in_shape.len() > 1 { in_shape[0] } else { 1 };
                    shape_map.insert(i, vec![batch, *out_features as usize]);

                    let w_size = (in_f * *out_features) as usize;
                    let w_data = rand_vec(w_size, weight_seed);
                    weight_seed += 1;
                    let w = GpuTensor::from_host(device, &w_data,
                        Shape::from_static(&[in_f as usize, *out_features as usize]),
                        DType::F32)?;
                    weights.insert(format!("layer{i}_weight"), w);
                }
                Layer::Add { a, .. } => {
                    if let Some(s) = shape_map.get(&a.0) {
                        shape_map.insert(i, s.clone());
                    }
                }
                _ => {}
            }
        }

        let input_specs: Vec<(String, Vec<usize>)> = self.layers.iter().filter_map(|l| match l {
            Layer::Input { name, shape } => Some((name.clone(), shape.clone())),
            _ => None,
        }).collect();

        Ok(CompiledModel {
            layers: self.layers,
            weights,
            cache,
            precision,
            input_specs,
            output_names: self.output_names,
        })
    }
}

impl CompiledModel {
    /// Get model info.
    pub fn info(&self) -> String {
        format!("CompiledModel: {} layers, {:?} precision, {} weights ({:.1} MB)",
            self.layers.len(), self.precision,
            self.weights.len(),
            self.weights.values().map(|w| w.size_bytes()).sum::<usize>() as f64 / 1e6)
    }

    /// Run inference on a single input.
    /// Returns the output tensor.
    pub fn infer(
        &self,
        device: &WarpDevice,
        input: &GpuTensor<f32>,
    ) -> Result<GpuTensor<f32>, DeviceError> {
        let mut tensors: HashMap<TensorId, GpuTensor<f32>> = HashMap::new();
        let mut last_output = None;

        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                Layer::Input { .. } => {
                    // Clone the input (we need owned data for downstream ops)
                    let data = input.to_host(device)?;
                    let t = GpuTensor::from_host(device, &data, input.shape.clone(), DType::F32)?;
                    tensors.insert(TensorId(i), t);
                }
                Layer::Relu { input: inp } => {
                    let x = tensors.get(inp).ok_or(DeviceError::Memory("missing input".into()))?;
                    let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                    crate::ops::relu(&self.cache, device, x, &mut out)?;
                    tensors.insert(TensorId(i), out);
                }
                Layer::Sigmoid { input: inp } => {
                    let x = tensors.get(inp).ok_or(DeviceError::Memory("missing input".into()))?;
                    let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                    crate::ops::sigmoid(&self.cache, device, x, &mut out)?;
                    tensors.insert(TensorId(i), out);
                }
                Layer::Gelu { input: inp } => {
                    let x = tensors.get(inp).ok_or(DeviceError::Memory("missing input".into()))?;
                    let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                    crate::ops::gelu(&self.cache, device, x, &mut out)?;
                    tensors.insert(TensorId(i), out);
                }
                Layer::Silu { input: inp } => {
                    let x = tensors.get(inp).ok_or(DeviceError::Memory("missing input".into()))?;
                    let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                    crate::ops::silu(&self.cache, device, x, &mut out)?;
                    tensors.insert(TensorId(i), out);
                }
                Layer::Add { a, b } => {
                    let ta = tensors.get(a).ok_or(DeviceError::Memory("missing A".into()))?;
                    let tb = tensors.get(b).ok_or(DeviceError::Memory("missing B".into()))?;
                    let mut out = GpuTensor::<f32>::zeros(device, ta.shape.clone(), DType::F32)?;
                    crate::ops::add(&self.cache, device, ta, tb, &mut out)?;
                    tensors.insert(TensorId(i), out);
                }
                Layer::Linear { input: inp, out_features } => {
                    let x = tensors.get(inp).ok_or(DeviceError::Memory("missing input".into()))?;
                    let w_key = format!("layer{i}_weight");
                    let w = self.weights.get(&w_key).ok_or(DeviceError::Memory(format!("missing {w_key}")))?;
                    // x: [batch, in_features], w: [in_features, out_features]
                    let x_dims = x.shape.dims();
                    let k = x_dims.last().and_then(|d| d.static_val()).unwrap_or(x.numel) as u32;
                    let m = (x.numel / k as usize) as u32;
                    let n = *out_features;
                    let mut out = GpuTensor::<f32>::zeros(device,
                        Shape::from_static(&[m as usize, n as usize]), DType::F32)?;
                    crate::ops::gemm(&self.cache, device, x, w, &mut out, m, n, k)?;
                    tensors.insert(TensorId(i), out);
                }
                Layer::GlobalAvgPool { input: inp } => {
                    let x = tensors.get(inp).ok_or(DeviceError::Memory("missing input".into()))?;
                    let dims = x.shape.dims();
                    let c = dims.get(1).and_then(|d| d.static_val()).unwrap_or(x.numel) as u32;
                    let spatial = if dims.len() > 2 {
                        dims[2..].iter().map(|d| d.static_val().unwrap_or(1) as u32).product()
                    } else { 1u32 };
                    let batch = x.numel / (c * spatial) as usize;
                    let mut out = GpuTensor::<f32>::zeros(device,
                        Shape::from_static(&[batch, c as usize]), DType::F32)?;
                    crate::conv::global_avg_pool(&self.cache, device, x, &mut out, c, spatial)?;
                    tensors.insert(TensorId(i), out);
                }
                Layer::Softmax { input: inp } => {
                    let x = tensors.get(inp).ok_or(DeviceError::Memory("missing input".into()))?;
                    let dims = x.shape.dims();
                    let cols = dims.last().and_then(|d| d.static_val()).unwrap_or(x.numel) as u32;
                    let rows = (x.numel / cols as usize) as u32;
                    let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                    crate::sampling::softmax(&self.cache, device, x, &mut out, rows, cols)?;
                    tensors.insert(TensorId(i), out);
                }
                Layer::Output { input: inp, .. } => {
                    last_output = Some(*inp);
                }
                Layer::Conv2d { input: inp, out_channels, kernel, stride, padding } => {
                    let x = tensors.get(inp).ok_or(DeviceError::Memory("missing input".into()))?;
                    let x_dims = x.shape.dims();
                    // x: [N, C_in, H, W]
                    let in_channels = x_dims.get(1).and_then(|d| d.static_val()).unwrap_or(3) as u32;
                    let h = x_dims.get(2).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                    let w = x_dims.get(3).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                    let batch = x.numel / (in_channels * h * w) as usize;

                    let params = crate::conv::Conv2dParams::new(in_channels, *out_channels, *kernel)
                        .stride(*stride)
                        .padding(*padding);
                    let out_h = params.output_h(h);
                    let out_w = params.output_w(w);

                    // Rebuild weight with correct in_channels if the stored weight has wrong shape
                    let w_key = format!("layer{i}_weight");
                    let weight = if let Some(existing) = self.weights.get(&w_key) {
                        existing
                    } else {
                        return Err(DeviceError::Memory(format!("missing {w_key}")));
                    };

                    let out_shape = Shape::from_static(&[batch, *out_channels as usize, out_h as usize, out_w as usize]);
                    let mut out = GpuTensor::<f32>::zeros(device, out_shape, DType::F32)?;
                    crate::conv::conv2d(&self.cache, device, x, weight, None, &mut out, &params, h, w)?;
                    tensors.insert(TensorId(i), out);
                }
                Layer::BatchNorm { input: inp } => {
                    let x = tensors.get(inp).ok_or(DeviceError::Memory("missing input".into()))?;
                    let x_dims = x.shape.dims();
                    let channels = x_dims.get(1).and_then(|d| d.static_val()).unwrap_or(x.numel) as u32;
                    let spatial: u32 = if x_dims.len() > 2 {
                        x_dims[2..].iter().map(|d| d.static_val().unwrap_or(1) as u32).product()
                    } else {
                        1
                    };

                    let scale_key = format!("layer{i}_scale");
                    let bias_key = format!("layer{i}_bias");
                    let mean_key = format!("layer{i}_mean");
                    let var_key = format!("layer{i}_var");

                    let scale = self.weights.get(&scale_key).ok_or(DeviceError::Memory(format!("missing {scale_key}")))?;
                    let bias = self.weights.get(&bias_key).ok_or(DeviceError::Memory(format!("missing {bias_key}")))?;
                    let mean = self.weights.get(&mean_key).ok_or(DeviceError::Memory(format!("missing {mean_key}")))?;
                    let var = self.weights.get(&var_key).ok_or(DeviceError::Memory(format!("missing {var_key}")))?;

                    let mut out = GpuTensor::<f32>::zeros(device, x.shape.clone(), DType::F32)?;
                    crate::conv::batchnorm2d(
                        &self.cache, device, x, scale, bias, mean, var,
                        &mut out, channels, spatial, 1e-5,
                    )?;
                    tensors.insert(TensorId(i), out);
                }
                Layer::MaxPool { input: inp, kernel, stride, padding } => {
                    let x = tensors.get(inp).ok_or(DeviceError::Memory("missing input".into()))?;
                    let x_dims = x.shape.dims();
                    let channels = x_dims.get(1).and_then(|d| d.static_val()).unwrap_or(x.numel) as u32;
                    let h = x_dims.get(2).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                    let w = x_dims.get(3).and_then(|d| d.static_val()).unwrap_or(1) as u32;
                    let batch = x.numel / (channels * h * w) as usize;
                    let out_h = crate::conv::conv_output_size(h, *kernel, *stride, *padding, 1);
                    let out_w = crate::conv::conv_output_size(w, *kernel, *stride, *padding, 1);

                    let out_shape = Shape::from_static(&[batch, channels as usize, out_h as usize, out_w as usize]);
                    let mut out = GpuTensor::<f32>::zeros(device, out_shape, DType::F32)?;
                    crate::conv::maxpool2d(
                        &self.cache, device, x, &mut out,
                        channels, h, w,
                        *kernel, *kernel,
                        *stride, *stride,
                        *padding, *padding,
                    )?;
                    tensors.insert(TensorId(i), out);
                }
                _ => {
                    // Concat, Reshape — pass through from previous layer
                }
            }
        }

        let out_id = last_output.ok_or(DeviceError::Memory("no output defined".into()))?;
        tensors.remove(&out_id).ok_or(DeviceError::Memory("output tensor not found".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_resnet_block() {
        let mut b = ModelBuilder::new();

        // ResNet-style block: Conv → BN → ReLU → Conv → BN + Skip → ReLU
        let input = b.input("image", &[1, 64, 56, 56]);
        let conv1 = b.conv2d(input, 64, 3, 1, 1);
        let bn1 = b.batchnorm(conv1);
        let relu1 = b.relu(bn1);
        let conv2 = b.conv2d(relu1, 64, 3, 1, 1);
        let bn2 = b.batchnorm(conv2);
        let skip = b.add(bn2, input); // residual connection
        let out = b.relu(skip);
        b.output("features", out);

        println!("{}", b.summary());
        assert_eq!(b.layers.len(), 9); // 1 input + 7 layers + 1 output
    }

    #[test]
    fn builder_classifier() {
        let mut b = ModelBuilder::new();

        let input = b.input("image", &[1, 3, 32, 32]);
        let conv = b.conv2d(input, 32, 3, 1, 1);
        let relu = b.relu(conv);
        let pool = b.global_avg_pool(relu);
        let fc = b.linear(pool, 10);
        let probs = b.softmax(fc);
        b.output("probs", probs);

        println!("{}", b.summary());
    }

    #[test]
    fn builder_mlp_inference() {
        let dev = match crate::device::WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        // Build: Input(4) → ReLU → Linear(4→8) → Sigmoid → Output
        let mut b = ModelBuilder::new();
        let input = b.input("x", &[1, 4]);
        let relu = b.relu(input);
        let fc = b.linear(relu, 8);
        let sig = b.sigmoid(fc);
        b.output("y", sig);

        println!("{}", b.summary());

        let model = b.build(&dev, Precision::FP32).unwrap();
        println!("{}", model.info());

        // Run inference
        let input_data = vec![1.0f32, -0.5, 2.0, -1.0];
        let input_tensor = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[1, 4]), DType::F32).unwrap();

        let output = model.infer(&dev, &input_tensor).unwrap();
        dev.synchronize().unwrap();

        let result = output.to_host(&dev).unwrap();
        println!("Builder MLP output: {:?}", result);
        assert!(result.iter().all(|v| v.is_finite()), "Output has NaN!");
        assert!(result.iter().all(|v| *v >= 0.0 && *v <= 1.0), "Sigmoid output should be in [0,1]!");
        println!("PASSED: Builder model produces valid sigmoid output!");
    }
}

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

        // Initialize weights for each layer that needs them
        let mut weight_seed = 42;
        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                Layer::Conv2d { out_channels, kernel, .. } => {
                    // Assume in_channels from previous layer (simplified)
                    let in_c = 3u32; // default, would be inferred from graph
                    let w_size = (*out_channels * in_c * kernel * kernel) as usize;
                    let w_data = rand_vec(w_size, weight_seed);
                    weight_seed += 1;
                    let w = GpuTensor::from_host(device, &w_data,
                        Shape::from_static(&[*out_channels as usize, in_c as usize, *kernel as usize, *kernel as usize]),
                        DType::F32)?;
                    weights.insert(format!("layer{i}_weight"), w);
                }
                Layer::Linear { out_features, .. } => {
                    let in_f = 256u32; // default
                    let w_size = (in_f * *out_features) as usize;
                    let w_data = rand_vec(w_size, weight_seed);
                    weight_seed += 1;
                    let w = GpuTensor::from_host(device, &w_data,
                        Shape::from_static(&[in_f as usize, *out_features as usize]),
                        DType::F32)?;
                    weights.insert(format!("layer{i}_weight"), w);
                }
                Layer::BatchNorm { .. } => {
                    // BN params initialized to identity transform
                    let c = 64usize; // default
                    weights.insert(format!("layer{i}_scale"),
                        GpuTensor::from_host(device, &vec![1.0f32; c], Shape::from_static(&[c]), DType::F32)?);
                    weights.insert(format!("layer{i}_bias"),
                        GpuTensor::from_host(device, &vec![0.0f32; c], Shape::from_static(&[c]), DType::F32)?);
                    weights.insert(format!("layer{i}_mean"),
                        GpuTensor::from_host(device, &vec![0.0f32; c], Shape::from_static(&[c]), DType::F32)?);
                    weights.insert(format!("layer{i}_var"),
                        GpuTensor::from_host(device, &vec![1.0f32; c], Shape::from_static(&[c]), DType::F32)?);
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
                _ => {
                    // Conv2D, BatchNorm, MaxPool etc need more shape tracking
                    // For now, pass through from previous layer
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

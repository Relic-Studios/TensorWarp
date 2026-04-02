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

        // Simple classifier: Conv → ReLU → Pool → Linear → Softmax
        let input = b.input("image", &[1, 3, 32, 32]);
        let conv = b.conv2d(input, 32, 3, 1, 1);
        let relu = b.relu(conv);
        let pool = b.global_avg_pool(relu);
        let fc = b.linear(pool, 10);
        let probs = b.softmax(fc);
        b.output("probs", probs);

        println!("{}", b.summary());
    }
}

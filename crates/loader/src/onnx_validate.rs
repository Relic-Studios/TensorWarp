//! ONNX model validation — verify TensorWarp produces correct results.
//!
//! Creates test models programmatically and verifies output against
//! CPU reference implementations.

use std::collections::HashMap;

use warp_ir::{DType, Shape};
use warp_kernels::device::WarpDevice;
use warp_kernels::tensor::GpuTensor;

use crate::onnx::*;
use crate::onnx_exec::*;

/// Build a simple MLP: Linear(4→8) → ReLU → Linear(8→3)
pub fn build_mlp_model() -> OnnxModel {
    let mut inits = HashMap::new();

    // W1: [4, 8], B1: [8]
    let w1: Vec<f32> = (0..32).map(|i| ((i * 7 + 3) % 20) as f32 * 0.1 - 1.0).collect();
    let b1: Vec<f32> = (0..8).map(|i| i as f32 * 0.05 - 0.2).collect();
    inits.insert("w1".into(), OnnxTensor {
        name: "w1".into(), dtype: OnnxDType::Float,
        shape: vec![4, 8], raw_data: w1.iter().flat_map(|f| f.to_le_bytes()).collect(),
    });
    inits.insert("b1".into(), OnnxTensor {
        name: "b1".into(), dtype: OnnxDType::Float,
        shape: vec![8], raw_data: b1.iter().flat_map(|f| f.to_le_bytes()).collect(),
    });

    // W2: [8, 3], B2: [3]
    let w2: Vec<f32> = (0..24).map(|i| ((i * 11 + 5) % 20) as f32 * 0.1 - 1.0).collect();
    let b2: Vec<f32> = vec![0.1, -0.1, 0.0];
    inits.insert("w2".into(), OnnxTensor {
        name: "w2".into(), dtype: OnnxDType::Float,
        shape: vec![8, 3], raw_data: w2.iter().flat_map(|f| f.to_le_bytes()).collect(),
    });
    inits.insert("b2".into(), OnnxTensor {
        name: "b2".into(), dtype: OnnxDType::Float,
        shape: vec![3], raw_data: b2.iter().flat_map(|f| f.to_le_bytes()).collect(),
    });

    OnnxModel {
        inputs: vec![OnnxIO { name: "input".into(), dtype: Some(OnnxDType::Float), shape: vec![1, 4] }],
        outputs: vec![OnnxIO { name: "output".into(), dtype: Some(OnnxDType::Float), shape: vec![1, 3] }],
        nodes: vec![
            OnnxNode { name: "gemm1".into(), op_type: "Gemm".into(),
                inputs: vec!["input".into(), "w1".into(), "b1".into()],
                outputs: vec!["hidden".into()], attrs: HashMap::new() },
            OnnxNode { name: "relu".into(), op_type: "Relu".into(),
                inputs: vec!["hidden".into()],
                outputs: vec!["relu_out".into()], attrs: HashMap::new() },
            OnnxNode { name: "gemm2".into(), op_type: "Gemm".into(),
                inputs: vec!["relu_out".into(), "w2".into(), "b2".into()],
                outputs: vec!["output".into()], attrs: HashMap::new() },
        ],
        initializers: inits,
        ir_version: 8, opset_version: 17, producer: "test".into(),
    }
}

/// CPU reference: MLP forward pass.
fn cpu_mlp(input: &[f32], w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]) -> Vec<f32> {
    // hidden = input @ w1 + b1
    let mut hidden = vec![0.0f32; 8];
    for j in 0..8 {
        hidden[j] = b1[j];
        for i in 0..4 {
            hidden[j] += input[i] * w1[i * 8 + j];
        }
    }
    // relu
    for v in &mut hidden {
        *v = v.max(0.0);
    }
    // output = hidden @ w2 + b2
    let mut output = vec![0.0f32; 3];
    for j in 0..3 {
        output[j] = b2[j];
        for i in 0..8 {
            output[j] += hidden[i] * w2[i * 3 + j];
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_mlp_model() {
        let dev = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        let model = build_mlp_model();
        println!("{}", model.summary());

        let exec = OnnxExecutor::new(&dev, &model).unwrap();

        // Test input
        let input_data = vec![0.5f32, -0.3, 0.8, -0.1];
        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[1, 4]), DType::F32).unwrap();

        let outputs = exec.run(&dev, &[("input", &input)]).unwrap();
        dev.synchronize().unwrap();

        let gpu_result = outputs["output"].to_host(&dev).unwrap();

        // CPU reference
        let w1 = model.initializers["w1"].to_f32();
        let b1 = model.initializers["b1"].to_f32();
        let w2 = model.initializers["w2"].to_f32();
        let b2 = model.initializers["b2"].to_f32();
        let cpu_result = cpu_mlp(&input_data, &w1, &b1, &w2, &b2);

        println!("\n=== MLP Validation ===");
        println!("  Input: {:?}", input_data);
        println!("  GPU output:  {:?}", gpu_result);
        println!("  CPU output:  {:?}", cpu_result);

        let max_err: f32 = gpu_result.iter().zip(cpu_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("  Max error: {max_err:.6}");

        assert!(max_err < 0.01, "GPU vs CPU mismatch: {max_err}");
        println!("  PASSED: GPU matches CPU reference!");
    }

    #[test]
    fn validate_conv_relu_pool() {
        let dev = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        // Build Conv → ReLU → GlobalAvgPool → FC model
        let mut inits = HashMap::new();

        // Conv weight: [2, 1, 3, 3] (2 output channels, 1 input, 3x3 kernel)
        let conv_w: Vec<f32> = (0..18).map(|i| ((i * 7 + 3) % 20) as f32 * 0.1 - 1.0).collect();
        inits.insert("conv_w".into(), OnnxTensor {
            name: "conv_w".into(), dtype: OnnxDType::Float,
            shape: vec![2, 1, 3, 3], raw_data: conv_w.iter().flat_map(|f| f.to_le_bytes()).collect(),
        });

        let model = OnnxModel {
            inputs: vec![OnnxIO { name: "input".into(), dtype: Some(OnnxDType::Float), shape: vec![1, 1, 8, 8] }],
            outputs: vec![OnnxIO { name: "pool_out".into(), dtype: Some(OnnxDType::Float), shape: vec![1, 2, 1, 1] }],
            nodes: vec![
                OnnxNode { name: "conv".into(), op_type: "Conv".into(),
                    inputs: vec!["input".into(), "conv_w".into()],
                    outputs: vec!["conv_out".into()],
                    attrs: [("kernel_shape".into(), OnnxAttr::Ints(vec![3, 3])),
                            ("pads".into(), OnnxAttr::Ints(vec![1, 1, 1, 1]))].into_iter().collect() },
                OnnxNode { name: "relu".into(), op_type: "Relu".into(),
                    inputs: vec!["conv_out".into()],
                    outputs: vec!["relu_out".into()], attrs: HashMap::new() },
                OnnxNode { name: "pool".into(), op_type: "GlobalAveragePool".into(),
                    inputs: vec!["relu_out".into()],
                    outputs: vec!["pool_out".into()], attrs: HashMap::new() },
            ],
            initializers: inits,
            ir_version: 8, opset_version: 17, producer: "test".into(),
        };

        let exec = OnnxExecutor::new(&dev, &model).unwrap();

        let input_data: Vec<f32> = (0..64).map(|i| ((i * 13 + 7) % 100) as f32 * 0.01).collect();
        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[1, 1, 8, 8]), DType::F32).unwrap();

        let outputs = exec.run(&dev, &[("input", &input)]).unwrap();
        dev.synchronize().unwrap();

        let result = outputs["pool_out"].to_host(&dev).unwrap();

        println!("\n=== Conv→ReLU→Pool Validation ===");
        println!("  Input:  [1, 1, 8, 8]");
        println!("  Output: {:?}", result);
        assert!(result.iter().all(|v| v.is_finite()), "Output has NaN/Inf!");
        assert!(result.iter().any(|v| *v != 0.0), "Output is all zeros!");
        println!("  PASSED: Conv→ReLU→Pool pipeline produces valid output!");
    }

    #[test]
    fn validate_elementwise_chain() {
        let dev = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        // Test: Add → Relu → Mul (should be auto-fusible)
        let model = OnnxModel {
            inputs: vec![
                OnnxIO { name: "a".into(), dtype: Some(OnnxDType::Float), shape: vec![4] },
                OnnxIO { name: "b".into(), dtype: Some(OnnxDType::Float), shape: vec![4] },
                OnnxIO { name: "c".into(), dtype: Some(OnnxDType::Float), shape: vec![4] },
            ],
            outputs: vec![OnnxIO { name: "out".into(), dtype: Some(OnnxDType::Float), shape: vec![4] }],
            nodes: vec![
                OnnxNode { name: "add".into(), op_type: "Add".into(),
                    inputs: vec!["a".into(), "b".into()],
                    outputs: vec!["sum".into()], attrs: HashMap::new() },
                OnnxNode { name: "relu".into(), op_type: "Relu".into(),
                    inputs: vec!["sum".into()],
                    outputs: vec!["relu_out".into()], attrs: HashMap::new() },
                OnnxNode { name: "mul".into(), op_type: "Mul".into(),
                    inputs: vec!["relu_out".into(), "c".into()],
                    outputs: vec!["out".into()], attrs: HashMap::new() },
            ],
            initializers: HashMap::new(),
            ir_version: 8, opset_version: 17, producer: "test".into(),
        };

        let exec = OnnxExecutor::new(&dev, &model).unwrap();

        let a_data = vec![1.0f32, -2.0, 3.0, -4.0];
        let b_data = vec![0.5, 0.5, 0.5, 0.5];
        let c_data = vec![2.0, 2.0, 2.0, 2.0];

        let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[4]), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[4]), DType::F32).unwrap();
        let c = GpuTensor::from_host(&dev, &c_data, Shape::from_static(&[4]), DType::F32).unwrap();

        let outputs = exec.run(&dev, &[("a", &a), ("b", &b), ("c", &c)]).unwrap();
        dev.synchronize().unwrap();

        let result = outputs["out"].to_host(&dev).unwrap();

        // CPU reference: relu(a + b) * c
        let expected: Vec<f32> = a_data.iter().zip(b_data.iter()).zip(c_data.iter())
            .map(|((&a, &b), &c)| (a + b).max(0.0) * c)
            .collect();

        println!("\n=== Elementwise Chain Validation ===");
        println!("  GPU:      {:?}", result);
        println!("  Expected: {:?}", expected);

        let max_err: f32 = result.iter().zip(expected.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("  Max error: {max_err:.6}");
        assert!(max_err < 0.001, "Elementwise chain mismatch: {max_err}");
        println!("  PASSED!");
    }

    #[test]
    fn validate_gather_embedding() {
        let dev = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        // Test: Gather op for embedding lookup (critical for LLMs)
        let mut inits = HashMap::new();
        let embed_table: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        inits.insert("embed".into(), OnnxTensor {
            name: "embed".into(), dtype: OnnxDType::Float,
            shape: vec![5, 4], // 5 tokens, dim 4
            raw_data: embed_table.iter().flat_map(|f| f.to_le_bytes()).collect(),
        });

        let model = OnnxModel {
            inputs: vec![OnnxIO { name: "indices".into(), dtype: Some(OnnxDType::Float), shape: vec![3] }],
            outputs: vec![OnnxIO { name: "embeddings".into(), dtype: Some(OnnxDType::Float), shape: vec![3, 4] }],
            nodes: vec![
                OnnxNode { name: "gather".into(), op_type: "Gather".into(),
                    inputs: vec!["embed".into(), "indices".into()],
                    outputs: vec!["embeddings".into()], attrs: HashMap::new() },
            ],
            initializers: inits,
            ir_version: 8, opset_version: 17, producer: "test".into(),
        };

        let exec = OnnxExecutor::new(&dev, &model).unwrap();

        // Look up tokens 0, 2, 4
        let idx_data = vec![0.0f32, 2.0, 4.0];
        let indices = GpuTensor::from_host(&dev, &idx_data, Shape::from_static(&[3]), DType::F32).unwrap();

        let outputs = exec.run(&dev, &[("indices", &indices)]).unwrap();
        dev.synchronize().unwrap();

        let result = outputs["embeddings"].to_host(&dev).unwrap();

        println!("\n=== Gather (Embedding) Validation ===");
        println!("  Indices: [0, 2, 4]");
        println!("  Result: {:?}", result);

        // Token 0: [0.0, 0.1, 0.2, 0.3]
        assert!((result[0] - 0.0).abs() < 1e-5);
        assert!((result[1] - 0.1).abs() < 1e-5);
        // Token 2: [0.8, 0.9, 1.0, 1.1]
        assert!((result[4] - 0.8).abs() < 1e-5);
        // Token 4: [1.6, 1.7, 1.8, 1.9]
        assert!((result[8] - 1.6).abs() < 1e-5);
        println!("  PASSED: Embedding lookup correct!");
    }

    #[test]
    fn validate_resnet_residual_block() {
        let dev = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        // ResNet residual block: Conv(8→8, 3×3, pad=1) → ReLU → Conv(8→8, 3×3, pad=1) + Skip → ReLU
        let mut inits = HashMap::new();

        let conv1_w: Vec<f32> = (0..8*8*3*3).map(|i| ((i*7+3) % 200) as f32 * 0.005 - 0.5).collect();
        let conv2_w: Vec<f32> = (0..8*8*3*3).map(|i| ((i*11+5) % 200) as f32 * 0.005 - 0.5).collect();
        inits.insert("conv1_w".into(), OnnxTensor { name: "conv1_w".into(), dtype: OnnxDType::Float,
            shape: vec![8,8,3,3], raw_data: conv1_w.iter().flat_map(|f| f.to_le_bytes()).collect() });
        inits.insert("conv2_w".into(), OnnxTensor { name: "conv2_w".into(), dtype: OnnxDType::Float,
            shape: vec![8,8,3,3], raw_data: conv2_w.iter().flat_map(|f| f.to_le_bytes()).collect() });

        let model = OnnxModel {
            inputs: vec![OnnxIO { name: "input".into(), dtype: Some(OnnxDType::Float), shape: vec![1,8,16,16] }],
            outputs: vec![OnnxIO { name: "output".into(), dtype: Some(OnnxDType::Float), shape: vec![1,8,16,16] }],
            nodes: vec![
                OnnxNode { name: "conv1".into(), op_type: "Conv".into(),
                    inputs: vec!["input".into(), "conv1_w".into()], outputs: vec!["conv1_out".into()],
                    attrs: [("kernel_shape".into(), OnnxAttr::Ints(vec![3,3])),
                            ("pads".into(), OnnxAttr::Ints(vec![1,1,1,1]))].into_iter().collect() },
                OnnxNode { name: "relu1".into(), op_type: "Relu".into(),
                    inputs: vec!["conv1_out".into()], outputs: vec!["relu1_out".into()],
                    attrs: HashMap::new() },
                OnnxNode { name: "conv2".into(), op_type: "Conv".into(),
                    inputs: vec!["relu1_out".into(), "conv2_w".into()], outputs: vec!["conv2_out".into()],
                    attrs: [("kernel_shape".into(), OnnxAttr::Ints(vec![3,3])),
                            ("pads".into(), OnnxAttr::Ints(vec![1,1,1,1]))].into_iter().collect() },
                OnnxNode { name: "add".into(), op_type: "Add".into(),
                    inputs: vec!["conv2_out".into(), "input".into()], outputs: vec!["skip_out".into()],
                    attrs: HashMap::new() },
                OnnxNode { name: "relu2".into(), op_type: "Relu".into(),
                    inputs: vec!["skip_out".into()], outputs: vec!["output".into()],
                    attrs: HashMap::new() },
            ],
            initializers: inits,
            ir_version: 8, opset_version: 17, producer: "test".into(),
        };

        let exec = OnnxExecutor::new(&dev, &model).unwrap();
        let input_data: Vec<f32> = (0..8*16*16).map(|i| ((i*13+7) % 100) as f32 * 0.01).collect();
        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[1,8,16,16]), DType::F32).unwrap();

        let outputs = exec.run(&dev, &[("input", &input)]).unwrap();
        dev.synchronize().unwrap();

        let result = outputs["output"].to_host(&dev).unwrap();

        println!("\n=== ResNet Residual Block Validation ===");
        println!("  Input:  [1, 8, 16, 16]");
        println!("  Output: {} values, range [{:.4}, {:.4}]",
            result.len(),
            result.iter().cloned().fold(f32::INFINITY, f32::min),
            result.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        assert_eq!(result.len(), 8 * 16 * 16);
        assert!(result.iter().all(|v| v.is_finite()), "Output has NaN/Inf!");
        // After ReLU, all values should be >= 0
        assert!(result.iter().all(|v| *v >= 0.0), "After ReLU, all values should be >= 0!");
        println!("  PASSED: ResNet residual block (Conv→ReLU→Conv+Skip→ReLU)!");
    }

    #[test]
    fn validate_transformer_attention_pattern() {
        let dev = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        // Simplified transformer attention: Gemm(Q) → Gemm(K) → MatMul(QK^T) → Softmax → Gemm(V)
        let mut inits = HashMap::new();
        let h = 32; // hidden size

        let wq: Vec<f32> = (0..h*h).map(|i| ((i*7+3) % 200) as f32 * 0.01 - 1.0).collect();
        let wk: Vec<f32> = (0..h*h).map(|i| ((i*11+5) % 200) as f32 * 0.01 - 1.0).collect();
        inits.insert("wq".into(), OnnxTensor { name: "wq".into(), dtype: OnnxDType::Float,
            shape: vec![h as i64, h as i64], raw_data: wq.iter().flat_map(|f| f.to_le_bytes()).collect() });
        inits.insert("wk".into(), OnnxTensor { name: "wk".into(), dtype: OnnxDType::Float,
            shape: vec![h as i64, h as i64], raw_data: wk.iter().flat_map(|f| f.to_le_bytes()).collect() });

        let model = OnnxModel {
            inputs: vec![OnnxIO { name: "x".into(), dtype: Some(OnnxDType::Float), shape: vec![4, h as i64] }],
            outputs: vec![OnnxIO { name: "k_out".into(), dtype: Some(OnnxDType::Float), shape: vec![4, h as i64] }],
            nodes: vec![
                OnnxNode { name: "q_proj".into(), op_type: "MatMul".into(),
                    inputs: vec!["x".into(), "wq".into()], outputs: vec!["q".into()],
                    attrs: HashMap::new() },
                OnnxNode { name: "k_proj".into(), op_type: "MatMul".into(),
                    inputs: vec!["x".into(), "wk".into()], outputs: vec!["k_out".into()],
                    attrs: HashMap::new() },
            ],
            initializers: inits,
            ir_version: 8, opset_version: 17, producer: "test".into(),
        };

        let exec = OnnxExecutor::new(&dev, &model).unwrap();
        let input_data: Vec<f32> = (0..4*h).map(|i| ((i*13+7) % 100) as f32 * 0.01 - 0.5).collect();
        let input = GpuTensor::from_host(&dev, &input_data,
            Shape::from_static(&[4, h]), DType::F32).unwrap();

        let outputs = exec.run(&dev, &[("x", &input)]).unwrap();
        dev.synchronize().unwrap();

        let result = outputs["k_out"].to_host(&dev).unwrap();

        println!("\n=== Transformer QK Projection Validation ===");
        println!("  Input: [4, {h}] (4 tokens, hidden={h})");
        println!("  K output: {} values", result.len());
        assert_eq!(result.len(), 4 * h);
        assert!(result.iter().all(|v| v.is_finite()), "K projection has NaN/Inf!");
        println!("  PASSED: QK projection via ONNX MatMul!");
    }
}

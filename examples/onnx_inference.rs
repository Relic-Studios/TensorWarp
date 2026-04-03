//! onnx_inference — Run a synthetic ONNX model on the GPU with TensorWarp.
//!
//! Demonstrates:
//!   1. Constructing an ONNX model in-memory (no .onnx file needed)
//!   2. Loading it into the OnnxExecutor
//!   3. Running inference and reading outputs
//!
//! In production you would use `OnnxModel::load("model.onnx")` instead
//! of building the model struct by hand.
//!
//! Run: cargo run --example onnx_inference

use std::collections::HashMap;

use warp_ir::{DType, Shape};
use warp_kernels::device::WarpDevice;
use warp_kernels::tensor::GpuTensor;
use warp_loader::onnx::{OnnxAttr, OnnxDType, OnnxIO, OnnxModel, OnnxNode, OnnxTensor};
use warp_loader::onnx_exec::OnnxExecutor;

/// Build a small ONNX model programmatically.
///
/// Architecture: input[1,3,8,8] -> Conv(3->8, 3x3, pad=1) -> ReLU -> GlobalAvgPool -> Gemm -> output[1,10]
fn build_demo_model() -> OnnxModel {
    let mut initializers = HashMap::new();

    // Conv weight: [out_channels=8, in_channels=3, kH=3, kW=3]
    let conv_w: Vec<f32> = (0..8 * 3 * 3 * 3)
        .map(|i| ((i * 7 + 3) % 200) as f32 * 0.01 - 1.0)
        .collect();
    initializers.insert(
        "conv.weight".to_string(),
        OnnxTensor {
            name: "conv.weight".to_string(),
            dtype: OnnxDType::Float,
            shape: vec![8, 3, 3, 3],
            raw_data: conv_w.iter().flat_map(|f| f.to_le_bytes()).collect(),
        },
    );

    // Fully-connected weight: [in_features=8, out_features=10]
    let fc_w: Vec<f32> = (0..8 * 10)
        .map(|i| ((i * 11 + 5) % 200) as f32 * 0.01 - 1.0)
        .collect();
    initializers.insert(
        "fc.weight".to_string(),
        OnnxTensor {
            name: "fc.weight".to_string(),
            dtype: OnnxDType::Float,
            shape: vec![8, 10],
            raw_data: fc_w.iter().flat_map(|f| f.to_le_bytes()).collect(),
        },
    );

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
            // Conv2D: input -> conv_out
            OnnxNode {
                name: "conv1".to_string(),
                op_type: "Conv".to_string(),
                inputs: vec!["input".into(), "conv.weight".into()],
                outputs: vec!["conv_out".into()],
                attrs: [
                    ("kernel_shape".into(), OnnxAttr::Ints(vec![3, 3])),
                    ("pads".into(), OnnxAttr::Ints(vec![1, 1, 1, 1])),
                    ("strides".into(), OnnxAttr::Ints(vec![1, 1])),
                ]
                .into_iter()
                .collect(),
            },
            // ReLU: conv_out -> relu_out
            OnnxNode {
                name: "relu1".to_string(),
                op_type: "Relu".to_string(),
                inputs: vec!["conv_out".into()],
                outputs: vec!["relu_out".into()],
                attrs: HashMap::new(),
            },
            // GlobalAveragePool: relu_out -> gap_out
            OnnxNode {
                name: "gap".to_string(),
                op_type: "GlobalAveragePool".to_string(),
                inputs: vec!["relu_out".into()],
                outputs: vec!["gap_out".into()],
                attrs: HashMap::new(),
            },
            // Reshape: gap_out -> flat_out (flatten spatial dims)
            OnnxNode {
                name: "reshape".to_string(),
                op_type: "Reshape".to_string(),
                inputs: vec!["gap_out".into()],
                outputs: vec!["flat_out".into()],
                attrs: HashMap::new(),
            },
            // Gemm (FC layer): flat_out x fc.weight -> output
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
        producer: "tensorwarp-example".to_string(),
    }
}

fn main() {
    // ── Step 1: Initialize CUDA device ────────────────────────────
    let device = WarpDevice::new(0).expect("Failed to initialize CUDA device");
    println!("Device: {}", device.summary());

    // ── Step 2: Build the ONNX model ──────────────────────────────
    let model = build_demo_model();
    println!("\n{}", model.summary());

    // ── Step 3: Create the executor (uploads weights to GPU) ──────
    let executor = OnnxExecutor::new(&device, &model).expect("Failed to create ONNX executor");
    println!(
        "Executor ready  |  Weight memory: {:.2} KB",
        executor.weight_memory_bytes() as f64 / 1024.0
    );

    // ── Step 4: Prepare input tensor [1, 3, 8, 8] ────────────────
    let input_data: Vec<f32> = (0..1 * 3 * 8 * 8)
        .map(|i| ((i * 13 + 7) % 100) as f32 * 0.01)
        .collect();
    let input_tensor = GpuTensor::from_host(
        &device,
        &input_data,
        Shape::from_static(&[1, 3, 8, 8]),
        DType::F32,
    )
    .expect("Failed to upload input tensor");

    // ── Step 5: Run inference ─────────────────────────────────────
    let outputs = executor
        .run(&device, &[("input", &input_tensor)])
        .expect("Inference failed");
    device.synchronize().expect("Sync failed");

    // ── Step 6: Read and print the output ─────────────────────────
    let output_tensor = outputs
        .get("output")
        .expect("Model did not produce 'output' tensor");
    let result = output_tensor.to_host(&device).expect("Failed to read output");

    println!("\n=== Inference Results ===");
    println!("  Model:  Conv(3->8, 3x3) -> ReLU -> GlobalAvgPool -> FC(8->10)");
    println!("  Input:  [1, 3, 8, 8]  ({} elements)", input_data.len());
    println!("  Output: [1, 10]  ({} elements)", result.len());
    println!("  Values: {:?}", result);

    // Basic sanity check
    assert!(
        result.iter().all(|v| v.is_finite()),
        "Output contains NaN or Inf!"
    );
    println!("\nONNX inference complete — all outputs finite.");
}

//! builder_api — Construct and run a model using TensorWarp's Builder SDK.
//!
//! Demonstrates:
//!   1. Using ModelBuilder to define a neural network graph
//!   2. Building (compiling) the model with random weights
//!   3. Running inference through the compiled model
//!
//! The Builder API is TensorWarp's programmatic model construction interface,
//! similar to TensorRT's Builder/NetworkDefinition API.
//!
//! Run: cargo run --example builder_api

use warp_ir::{DType, Shape};
use warp_kernels::builder::{ModelBuilder, Precision};
use warp_kernels::device::WarpDevice;
use warp_kernels::tensor::GpuTensor;

fn main() {
    // ── Step 1: Initialize the CUDA device ────────────────────────
    let device = WarpDevice::new(0).expect("Failed to initialize CUDA device");
    println!("Device: {}", device.summary());

    // ── Step 2: Define the model graph ────────────────────────────
    // Architecture: Input(1,4) -> ReLU -> Linear(4->8) -> Softmax -> Output
    let mut builder = ModelBuilder::new();

    // Define a 2D input: batch=1, features=4
    let input = builder.input("x", &[1, 4]);

    // ReLU activation (zeroes out negatives)
    let relu = builder.relu(input);

    // Fully-connected layer: maps 4 features to 8
    let fc = builder.linear(relu, 8);

    // Softmax: turns logits into a probability distribution
    let probs = builder.softmax(fc);

    // Mark the softmax output as the model output
    builder.output("probabilities", probs);

    // Print the model summary before building
    println!("\n=== Model Graph ===");
    println!("{}", builder.summary());

    // ── Step 3: Build (compile) the model ─────────────────────────
    // This initializes random weights and prepares the execution plan.
    // In a real scenario, weights would be loaded from a file.
    let model = builder
        .build(&device, Precision::FP32)
        .expect("Failed to build model");
    println!("\n{}", model.info());

    // ── Step 4: Prepare input data ────────────────────────────────
    // Mix of positive and negative values to show ReLU in action.
    let input_data = vec![1.5f32, -0.5, 2.0, -1.0];
    println!("\nInput: {:?}", input_data);

    let input_tensor = GpuTensor::from_host(
        &device,
        &input_data,
        Shape::from_static(&[1, 4]),
        DType::F32,
    )
    .expect("Failed to upload input tensor");

    // ── Step 5: Run inference ─────────────────────────────────────
    let output = model
        .infer(&device, &input_tensor)
        .expect("Inference failed");
    device.synchronize().expect("Sync failed");

    // ── Step 6: Read and display results ──────────────────────────
    let result = output.to_host(&device).expect("Failed to read output");

    println!("\n=== Inference Results ===");
    println!("  Output shape: [1, 8]");
    println!("  Probabilities: {:?}", result);

    // Verify softmax properties: all values in [0,1] and sum to ~1.0
    let sum: f32 = result.iter().sum();
    println!("  Sum of probabilities: {:.6} (should be ~1.0)", sum);

    assert!(
        result.iter().all(|v| *v >= 0.0 && *v <= 1.0),
        "Softmax output should be in [0, 1]"
    );
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Softmax probabilities should sum to ~1.0, got {}",
        sum
    );

    println!("\nBuilder API inference complete — valid probability distribution.");
}

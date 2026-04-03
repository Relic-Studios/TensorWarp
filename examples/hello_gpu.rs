//! hello_gpu — Minimal TensorWarp example: GPU matrix multiply.
//!
//! Demonstrates:
//!   1. Creating a CUDA device
//!   2. Allocating GPU tensors from host data
//!   3. Running a GEMM (General Matrix Multiply) kernel
//!   4. Reading results back to the CPU
//!
//! Run: cargo run --example hello_gpu

use warp_ir::{DType, Shape};
use warp_kernels::cache::KernelCache;
use warp_kernels::device::WarpDevice;
use warp_kernels::ops;
use warp_kernels::tensor::GpuTensor;

fn main() {
    // ── Step 1: Initialize the CUDA device (GPU 0) ────────────────
    let device = WarpDevice::new(0).expect("Failed to initialize CUDA device");
    println!("Initialized: {}", device.summary());

    // ── Step 2: Prepare host data ─────────────────────────────────
    // A is a 4x3 matrix, B is a 3x2 matrix => C will be 4x2.
    let (m, n, k) = (4u32, 2u32, 3u32);

    // A = [[1, 2, 3],
    //      [4, 5, 6],
    //      [7, 8, 9],
    //      [10, 11, 12]]
    let a_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();

    // B = [[1, 0],
    //      [0, 1],
    //      [1, 1]]
    let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    // ── Step 3: Upload tensors to the GPU ─────────────────────────
    let a_gpu = GpuTensor::from_host(
        &device,
        &a_data,
        Shape::from_static(&[m as usize, k as usize]),
        DType::F32,
    )
    .expect("Failed to upload matrix A");

    let b_gpu = GpuTensor::from_host(
        &device,
        &b_data,
        Shape::from_static(&[k as usize, n as usize]),
        DType::F32,
    )
    .expect("Failed to upload matrix B");

    // Allocate a zero-initialized output tensor on the GPU.
    let mut c_gpu = GpuTensor::<f32>::zeros(
        &device,
        Shape::from_static(&[m as usize, n as usize]),
        DType::F32,
    )
    .expect("Failed to allocate output tensor");

    // ── Step 4: Launch the GEMM kernel (C = A * B) ────────────────
    let cache = KernelCache::new();
    ops::gemm(&cache, &device, &a_gpu, &b_gpu, &mut c_gpu, m, n, k)
        .expect("GEMM kernel failed");

    // Wait for GPU to finish.
    device.synchronize().expect("Sync failed");

    // ── Step 5: Read the result back to the CPU ───────────────────
    let c_host = c_gpu.to_host(&device).expect("Failed to read result");

    // ── Step 6: Print the result ──────────────────────────────────
    println!("\nA ({m}x{k}):");
    for row in 0..m as usize {
        let slice = &a_data[row * k as usize..(row + 1) * k as usize];
        println!("  {:?}", slice);
    }

    println!("\nB ({k}x{n}):");
    for row in 0..k as usize {
        let slice = &b_data[row * n as usize..(row + 1) * n as usize];
        println!("  {:?}", slice);
    }

    println!("\nC = A * B ({m}x{n}):");
    for row in 0..m as usize {
        let slice = &c_host[row * n as usize..(row + 1) * n as usize];
        println!("  {:?}", slice);
    }

    // Verify: row 0 should be [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
    println!("\nExpected first row: [4.0, 5.0], got: {:?}", &c_host[0..2]);
    println!("TensorWarp GPU GEMM complete!");
}

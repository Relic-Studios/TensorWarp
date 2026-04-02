//! # warp-kernels
//!
//! Real GPU kernel implementations for the Warp inference engine.
//!
//! This crate bridges the gap between codegen (which produces PTX text)
//! and actual GPU execution. It handles:
//! - CUDA device management via cudarc
//! - PTX compilation and kernel loading
//! - Typed tensor buffers on GPU
//! - Kernel launch with proper parameter binding
//! - GEMM, attention, and elementwise kernel templates

pub mod device;
pub mod tensor;
pub mod elementwise;
pub mod gemm;

pub use device::WarpDevice;
pub use tensor::GpuTensor;

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

pub mod attention;
pub mod cache;
pub mod cublas_bench;
pub mod device;
pub mod elementwise;
pub mod gemm;
pub mod ops;
pub mod tensor;

pub use cache::KernelCache;
pub use device::WarpDevice;
pub use tensor::GpuTensor;

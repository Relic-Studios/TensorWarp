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
pub mod attention_ext;
pub mod autotune;
pub mod bench;
pub mod builder;
pub mod cache;
pub mod calibrate;
pub mod conv;
pub mod cublas_bench;
pub mod cuda_graph;
pub mod detect;
pub mod device;
pub mod elementwise;
pub mod engine;
pub mod fp16;
pub mod gather;
pub mod gemm;
pub mod generate;
pub mod mem_pool;
pub mod missing_ops;
pub mod multi_gpu;
pub mod gemm_fast;
pub mod kv_cache;
pub mod gemm_tc;
pub mod gemm_v2;
pub mod ops;
pub mod profiler;
pub mod quantize;
pub mod rope;
pub mod sampling;
pub mod swiglu;
pub mod tensor;
pub mod transformer;
pub mod transformer_f16;

pub use cache::KernelCache;
pub use device::WarpDevice;
pub use tensor::GpuTensor;

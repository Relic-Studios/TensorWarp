//! # warp-codegen
//!
//! GPU kernel code generation for the Warp inference engine.
//!
//! This crate defines the `Backend` trait — the interface between the
//! optimized IR graph and actual GPU code. Each backend (CUDA/PTX, Metal,
//! SPIR-V) implements this trait to generate kernels for its target.
//!
//! The codegen is *shape-specialized*: it generates different kernels for
//! different tensor shapes, baking in exact tile sizes and loop bounds.
//! This is slower to compile than TRT's approach (selecting from a kernel
//! library) but produces faster kernels because there's zero runtime
//! shape dispatch.

pub mod backend;
pub mod ptx;
pub mod metal;
pub mod kernel;

pub use backend::{Backend, CompiledKernel, KernelConfig};

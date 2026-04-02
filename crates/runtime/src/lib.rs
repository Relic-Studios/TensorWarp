//! # warp-runtime
//!
//! Execution runtime for the Warp inference engine.
//!
//! The runtime manages:
//! - Memory allocation (arena-based GPU memory pools)
//! - Kernel execution scheduling
//! - KV cache management for autoregressive decoding
//! - CUDA Graph capture and replay
//!
//! The runtime is backend-agnostic — it works with any codegen backend
//! through the `Backend` trait.

pub mod memory;
pub mod engine;
pub mod schedule;

pub use engine::Engine;
pub use memory::{MemoryPool, TensorBuffer};

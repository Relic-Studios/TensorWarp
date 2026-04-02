//! # warp-runtime
//!
//! Execution runtime for the Warp inference engine.
//!
//! The runtime manages:
//! - Memory allocation (arena-based GPU memory pools)
//! - Kernel execution scheduling
//! - Runtime profiling for profile-guided recompilation
//! - Tiered compilation (Tier 0→3 with hot-swapping)
//! - KV cache management for autoregressive decoding
//! - CUDA Graph capture and replay
//!
//! The runtime is backend-agnostic — it works with any codegen backend
//! through the `Backend` trait.

pub mod engine;
pub mod memory;
pub mod profile;
pub mod schedule;
pub mod tiered;

pub use engine::Engine;
pub use memory::{MemoryPool, TensorBuffer};
pub use profile::Profiler;
pub use tiered::{Tier, TierPolicy, TieredCompiler};

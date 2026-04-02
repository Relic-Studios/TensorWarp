//! # warp-optimizer
//!
//! Graph optimization passes for the Warp inference engine.
//!
//! The optimizer works by running a pipeline of passes over the IR graph.
//! Each pass is a function `Graph -> Graph` that rewrites the graph.
//! Passes are composable and ordered — fusion runs after constant folding,
//! layout optimization runs after fusion, etc.

pub mod fusion;
pub mod pass;
pub mod pattern;

pub use pass::{OptimizationLevel, PassPipeline};

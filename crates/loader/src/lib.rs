//! # warp-loader
//!
//! Model weight loading for TensorWarp.
//! Supports SafeTensors format (the HuggingFace standard).
//!
//! Usage:
//! ```ignore
//! let model = SafeTensorsLoader::open("model.safetensors")?;
//! let weight = model.load_f32("model.layers.0.self_attn.q_proj.weight", &device)?;
//! ```

pub mod safetensors_loader;
pub mod llama;

pub use safetensors_loader::SafeTensorsLoader;
pub use llama::{LlamaConfig, LlamaModel};

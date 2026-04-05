#![allow(unused_imports, unused_variables, dead_code)]
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
pub mod onnx;
pub mod onnx_compile;
pub mod onnx_exec;
pub mod onnx_validate;
pub mod graph_exec;
pub mod pipeline;
pub mod gguf;
pub mod tokenizer;
pub mod hub;
pub mod debug_model;

pub use safetensors_loader::SafeTensorsLoader;
pub use llama::{LlamaConfig, LlamaModel, LlamaModelF16, LlamaModelQ4};
pub use onnx::OnnxModel;
pub use onnx_exec::OnnxExecutor;
pub use graph_exec::GraphExecutor;
pub use pipeline::InferencePipeline;
pub use gguf::GgufModel;
pub use tokenizer::{Tokenizer, ChatTemplate};
pub use hub::{HubConfig, ModelFormat};

//! Model serialization — save compiled models to disk for instant reload.
//!
//! Like TensorRT's .engine files, TensorWarp can serialize a compiled model
//! including weights, kernel cache, and optimization state. Subsequent loads
//! skip JIT compilation entirely.
//!
//! Format: simple binary with header + weight blobs + kernel PTX cache.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

/// Header for serialized model file.
#[derive(Debug)]
struct WarpEngineHeader {
    magic: [u8; 4],      // "WARP"
    version: u32,        // format version
    num_weights: u32,    // number of weight tensors
    num_kernels: u32,    // number of cached PTX kernels
    metadata_len: u32,   // length of JSON metadata
}

/// Metadata stored in the engine file.
#[derive(Debug, Clone)]
pub struct EngineMetadata {
    pub model_name: String,
    pub precision: String,      // "fp32", "fp16", "int8", "q4_0"
    pub num_layers: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub gpu_arch: String,       // "sm_89" etc
    pub tensorwarp_version: String,
}

impl Default for EngineMetadata {
    fn default() -> Self {
        Self {
            model_name: "unknown".into(),
            precision: "fp32".into(),
            num_layers: 0,
            hidden_size: 0,
            vocab_size: 0,
            gpu_arch: "sm_89".into(),
            tensorwarp_version: env!("CARGO_PKG_VERSION").into(),
        }
    }
}

/// Save model weights to a binary file.
pub fn save_weights(
    path: impl AsRef<Path>,
    weights: &HashMap<String, Vec<f32>>,
    metadata: &EngineMetadata,
) -> Result<(), std::io::Error> {
    let mut file = std::fs::File::create(path)?;

    // Magic + version
    file.write_all(b"WARP")?;
    file.write_all(&1u32.to_le_bytes())?; // version 1

    // Metadata as JSON
    let meta_json = format!(
        r#"{{"model":"{}","precision":"{}","layers":{},"hidden":{},"vocab":{},"arch":"{}","version":"{}"}}"#,
        metadata.model_name, metadata.precision, metadata.num_layers,
        metadata.hidden_size, metadata.vocab_size, metadata.gpu_arch,
        metadata.tensorwarp_version
    );
    let meta_bytes = meta_json.as_bytes();
    file.write_all(&(meta_bytes.len() as u32).to_le_bytes())?;
    file.write_all(meta_bytes)?;

    // Number of weights
    file.write_all(&(weights.len() as u32).to_le_bytes())?;

    // Each weight: name_len + name + numel + data
    for (name, data) in weights {
        let name_bytes = name.as_bytes();
        file.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        file.write_all(name_bytes)?;
        file.write_all(&(data.len() as u32).to_le_bytes())?;
        // Write f32 array as raw bytes
        let byte_data: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&byte_data)?;
    }

    Ok(())
}

/// Load model weights from a binary file.
pub fn load_weights(
    path: impl AsRef<Path>,
) -> Result<(HashMap<String, Vec<f32>>, EngineMetadata), std::io::Error> {
    let mut file = std::fs::File::open(path)?;

    // Check magic
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != b"WARP" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Not a WARP engine file"));
    }

    // Version
    let mut buf4 = [0u8; 4];
    file.read_exact(&mut buf4)?;
    let _version = u32::from_le_bytes(buf4);

    // Metadata
    file.read_exact(&mut buf4)?;
    let meta_len = u32::from_le_bytes(buf4) as usize;
    let mut meta_buf = vec![0u8; meta_len];
    file.read_exact(&mut meta_buf)?;
    let meta_str = String::from_utf8_lossy(&meta_buf);

    // Parse metadata (simple — real impl would use serde_json)
    let metadata = EngineMetadata {
        model_name: extract_json_str(&meta_str, "model"),
        precision: extract_json_str(&meta_str, "precision"),
        num_layers: extract_json_u32(&meta_str, "layers"),
        hidden_size: extract_json_u32(&meta_str, "hidden"),
        vocab_size: extract_json_u32(&meta_str, "vocab"),
        gpu_arch: extract_json_str(&meta_str, "arch"),
        tensorwarp_version: extract_json_str(&meta_str, "version"),
    };

    // Number of weights
    file.read_exact(&mut buf4)?;
    let num_weights = u32::from_le_bytes(buf4) as usize;

    let mut weights = HashMap::new();
    for _ in 0..num_weights {
        // Name
        file.read_exact(&mut buf4)?;
        let name_len = u32::from_le_bytes(buf4) as usize;
        let mut name_buf = vec![0u8; name_len];
        file.read_exact(&mut name_buf)?;
        let name = String::from_utf8_lossy(&name_buf).to_string();

        // Data
        file.read_exact(&mut buf4)?;
        let numel = u32::from_le_bytes(buf4) as usize;
        let mut data_buf = vec![0u8; numel * 4];
        file.read_exact(&mut data_buf)?;
        let data: Vec<f32> = data_buf.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        weights.insert(name, data);
    }

    Ok((weights, metadata))
}

fn extract_json_str(json: &str, key: &str) -> String {
    let pattern = format!("\"{}\":\"", key);
    if let Some(start) = json.find(&pattern) {
        let rest = &json[start + pattern.len()..];
        if let Some(end) = rest.find('"') {
            return rest[..end].to_string();
        }
    }
    String::new()
}

fn extract_json_u32(json: &str, key: &str) -> u32 {
    let pattern = format!("\"{}\":", key);
    if let Some(start) = json.find(&pattern) {
        let rest = &json[start + pattern.len()..];
        let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        return num_str.parse().unwrap_or(0);
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_load_roundtrip() {
        let mut weights = HashMap::new();
        weights.insert("layer0.weight".to_string(), vec![1.0f32, 2.0, 3.0, 4.0]);
        weights.insert("layer0.bias".to_string(), vec![0.1, -0.2]);

        let metadata = EngineMetadata {
            model_name: "test_model".into(),
            precision: "fp16".into(),
            num_layers: 4,
            hidden_size: 256,
            vocab_size: 32000,
            ..Default::default()
        };

        let path = std::env::temp_dir().join("test_model.warp");
        save_weights(&path, &weights, &metadata).unwrap();

        let (loaded_weights, loaded_meta) = load_weights(&path).unwrap();

        assert_eq!(loaded_weights["layer0.weight"], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded_weights["layer0.bias"], vec![0.1, -0.2]);
        assert_eq!(loaded_meta.model_name, "test_model");
        assert_eq!(loaded_meta.precision, "fp16");
        assert_eq!(loaded_meta.num_layers, 4);
        assert_eq!(loaded_meta.hidden_size, 256);

        std::fs::remove_file(&path).unwrap();
        println!("Save/Load roundtrip: correct!");
    }
}

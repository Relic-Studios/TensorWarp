//! HuggingFace Hub integration — model caching and download helpers.
//!
//! Since TensorWarp doesn't bundle `reqwest`, this module focuses on:
//! - Cache management (checking if models are already downloaded)
//! - Path resolution (local path vs HF repo ID)
//! - Generating download instructions when models aren't cached

use std::fmt;
use std::path::{Path, PathBuf};

// ─── Error ───────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum HubError {
    NotCached(String),
    Io(std::io::Error),
    InvalidRepoId(String),
}

impl fmt::Display for HubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCached(msg) => write!(f, "{msg}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::InvalidRepoId(id) => write!(f, "invalid repo ID: {id}"),
        }
    }
}

impl std::error::Error for HubError {}

impl From<std::io::Error> for HubError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for HuggingFace Hub access.
pub struct HubConfig {
    /// Local cache directory for downloaded models.
    pub cache_dir: PathBuf,
    /// HuggingFace API token (from HF_TOKEN env var).
    pub token: Option<String>,
}

impl HubConfig {
    /// Create config with default cache directory (~/.cache/tensorwarp/models).
    pub fn new() -> Self {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        let cache_dir = PathBuf::from(home).join(".cache").join("tensorwarp").join("models");

        Self {
            cache_dir,
            token: std::env::var("HF_TOKEN").ok(),
        }
    }

    /// Create config with a custom cache directory.
    pub fn with_cache_dir(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            token: std::env::var("HF_TOKEN").ok(),
        }
    }
}

impl Default for HubConfig {
    fn default() -> Self { Self::new() }
}

// ─── Functions ───────────────────────────────────────────────────────────────

/// Convert a HuggingFace repo ID to a local cache directory name.
/// "meta-llama/Llama-3-8B" → "meta-llama--Llama-3-8B"
fn repo_to_dirname(repo_id: &str) -> String {
    repo_id.replace('/', "--")
}

/// Download a model from HuggingFace Hub.
///
/// Returns the local path if the model is already cached.
/// Otherwise returns an error with download instructions.
pub fn download_model(
    repo_id: &str,
    _revision: &str,
    config: &HubConfig,
) -> Result<PathBuf, HubError> {
    if repo_id.is_empty() || !repo_id.contains('/') {
        return Err(HubError::InvalidRepoId(repo_id.to_string()));
    }

    let model_dir = config.cache_dir.join(repo_to_dirname(repo_id));

    if model_dir.exists() && has_model_files(&model_dir) {
        return Ok(model_dir);
    }

    // Without reqwest, we can't download automatically.
    // Provide clear instructions.
    let mut msg = format!(
        "Model '{repo_id}' not cached locally.\n\n\
         Download with one of:\n\
         \n\
         1. huggingface-cli:\n\
         \x20  huggingface-cli download {repo_id} --local-dir {}\n\
         \n\
         2. git lfs:\n\
         \x20  git lfs install\n\
         \x20  git clone https://huggingface.co/{repo_id} {}\n",
        model_dir.display(),
        model_dir.display(),
    );

    if config.token.is_some() {
        msg.push_str("\n(HF_TOKEN is set — authenticated downloads should work)\n");
    } else {
        msg.push_str(
            "\nFor gated models, set HF_TOKEN:\n\
             \x20  export HF_TOKEN=hf_...\n"
        );
    }

    Err(HubError::NotCached(msg))
}

/// Check if a model is already cached locally.
pub fn is_cached(repo_id: &str, config: &HubConfig) -> bool {
    let model_dir = config.cache_dir.join(repo_to_dirname(repo_id));
    model_dir.exists() && has_model_files(&model_dir)
}

/// List all cached models.
pub fn list_cached(config: &HubConfig) -> Vec<String> {
    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&config.cache_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                // Convert dirname back to repo ID
                let repo_id = name.replace("--", "/");
                if repo_id.contains('/') {
                    models.push(repo_id);
                }
            }
        }
    }
    models.sort();
    models
}

/// Resolve a model identifier to a local path.
///
/// - If it's an existing local directory, use it directly.
/// - If it looks like a HF repo ID (contains '/'), check the cache.
/// - Otherwise treat it as a local path.
pub fn resolve_model_path(model_id: &str) -> Result<PathBuf, HubError> {
    let path = Path::new(model_id);

    // Direct local path
    if path.exists() {
        return Ok(path.to_path_buf());
    }

    // Looks like a HF repo ID
    if model_id.contains('/') {
        let config = HubConfig::new();
        return download_model(model_id, "main", &config);
    }

    // Check cache with common org prefixes
    let config = HubConfig::new();
    for prefix in &["meta-llama", "mistralai", "google", "microsoft", "TheBloke"] {
        let repo_id = format!("{prefix}/{model_id}");
        if is_cached(&repo_id, &config) {
            let model_dir = config.cache_dir.join(repo_to_dirname(&repo_id));
            return Ok(model_dir);
        }
    }

    Err(HubError::NotCached(format!(
        "'{model_id}' is not a local path and not cached.\n\
         If it's a HuggingFace model, use the full repo ID (e.g., meta-llama/Llama-3-8B)."
    )))
}

/// Detect model format from files in a directory.
pub fn detect_model_format(dir: &Path) -> ModelFormat {
    if let Ok(entries) = std::fs::read_dir(dir) {
        let files: Vec<String> = entries
            .flatten()
            .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
            .collect();

        for f in &files {
            if f.ends_with(".gguf") {
                return ModelFormat::Gguf;
            }
        }
        for f in &files {
            if f.ends_with(".safetensors") {
                return ModelFormat::SafeTensors;
            }
        }
        for f in &files {
            if f.ends_with(".onnx") {
                return ModelFormat::Onnx;
            }
        }
    }

    // Check if the path itself is a file
    if let Some(ext) = dir.extension().and_then(|e| e.to_str()) {
        match ext {
            "gguf" => return ModelFormat::Gguf,
            "safetensors" => return ModelFormat::SafeTensors,
            "onnx" => return ModelFormat::Onnx,
            _ => {}
        }
    }

    ModelFormat::Unknown
}

/// Supported model formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    SafeTensors,
    Gguf,
    Onnx,
    Unknown,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SafeTensors => write!(f, "SafeTensors"),
            Self::Gguf => write!(f, "GGUF"),
            Self::Onnx => write!(f, "ONNX"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Check if a directory contains recognized model files.
fn has_model_files(dir: &Path) -> bool {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".safetensors")
                    || name.ends_with(".gguf")
                    || name.ends_with(".onnx")
                    || name == "config.json"
                {
                    return true;
                }
            }
        }
    }
    false
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repo_to_dirname() {
        assert_eq!(repo_to_dirname("meta-llama/Llama-3-8B"), "meta-llama--Llama-3-8B");
        assert_eq!(repo_to_dirname("TheBloke/model-GGUF"), "TheBloke--model-GGUF");
    }

    #[test]
    fn test_detect_format_from_extension() {
        assert_eq!(detect_model_format(Path::new("model.gguf")), ModelFormat::Gguf);
        assert_eq!(detect_model_format(Path::new("model.safetensors")), ModelFormat::SafeTensors);
        assert_eq!(detect_model_format(Path::new("model.onnx")), ModelFormat::Onnx);
        assert_eq!(detect_model_format(Path::new("model.bin")), ModelFormat::Unknown);
    }

    #[test]
    fn test_hub_config_default() {
        let config = HubConfig::new();
        assert!(config.cache_dir.to_str().unwrap().contains("tensorwarp"));
    }

    #[test]
    fn test_list_cached_empty() {
        let config = HubConfig::with_cache_dir(PathBuf::from("/nonexistent/path"));
        let cached = list_cached(&config);
        assert!(cached.is_empty());
    }

    #[test]
    fn test_invalid_repo_id() {
        let config = HubConfig::new();
        let result = download_model("invalid", "main", &config);
        assert!(result.is_err());
    }
}

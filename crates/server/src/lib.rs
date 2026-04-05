//! TensorWarp HTTP Server — vLLM-style OpenAI-compatible inference API.
//!
//! Endpoints:
//!   POST /v1/completions         Text completion (OpenAI format)
//!   POST /v1/chat/completions    Chat completion (OpenAI format)
//!   GET  /v1/models              List available models
//!   GET  /health                 Health check

use axum::{Json, Router, extract::State, routing};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use warp_kernels::device::WarpDevice;
use warp_kernels::generate::{GenerateConfig, GenerationEngine};
use warp_loader::Tokenizer;

// ─── Request / Response types (OpenAI format) ───────────────────────────────

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
}

#[derive(Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

#[derive(Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

// ─── Server State ───────────────────────────────────────────────────────────

pub struct ServerState {
    pub model_name: String,
    pub engine: Option<Arc<Mutex<GenerationEngine>>>,
    pub tokenizer: Option<Arc<Tokenizer>>,
    pub device: Option<Arc<WarpDevice>>,
    pub gen_config: GenerateConfig,
}

impl ServerState {
    pub fn new(model_name: String) -> Self {
        Self {
            model_name,
            engine: None,
            tokenizer: None,
            device: None,
            gen_config: GenerateConfig::default(),
        }
    }

    pub fn with_engine(
        model_name: String,
        engine: GenerationEngine,
        tokenizer: Tokenizer,
        device: WarpDevice,
    ) -> Self {
        Self {
            model_name,
            engine: Some(Arc::new(Mutex::new(engine))),
            tokenizer: Some(Arc::new(tokenizer)),
            device: Some(Arc::new(device)),
            gen_config: GenerateConfig::default(),
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn short_uuid() -> String {
    uuid::Uuid::new_v4()
        .to_string()
        .split('-')
        .next()
        .unwrap_or("0")
        .to_string()
}

// ─── Handlers ───────────────────────────────────────────────────────────────

async fn completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CompletionRequest>,
) -> Json<CompletionResponse> {
    let prompt = req.prompt.unwrap_or_default();
    let max_tokens = req.max_tokens.unwrap_or(128);

    // If engine + tokenizer are loaded, run actual generation
    if let (Some(engine), Some(tokenizer), Some(device)) =
        (&state.engine, &state.tokenizer, &state.device)
    {
        let engine = engine.clone();
        let tokenizer = tokenizer.clone();
        let device = device.clone();
        let model_name = state.model_name.clone();

        let mut gen_config = state.gen_config.clone();
        gen_config.max_tokens = max_tokens;
        if let Some(temp) = req.temperature {
            gen_config.temperature = temp;
            gen_config.greedy = temp <= 0.01;
        }

        let prompt_clone = prompt.clone();
        // Run GPU generation on a blocking thread (GPU ops are synchronous)
        let result = tokio::task::spawn_blocking(move || {
            let token_ids: Vec<i32> = tokenizer
                .encode(&prompt_clone)
                .iter()
                .map(|&id| id as i32)
                .collect();
            let prompt_len = token_ids.len();

            let eng = engine.lock().unwrap();
            let gen_result = eng.generate_with_cache(&device, &token_ids, &gen_config, 2048)?;

            let output_ids: Vec<u32> = gen_result.tokens.iter().map(|&t| t as u32).collect();
            let decoded = tokenizer.decode(&output_ids);

            Ok::<_, warp_kernels::device::DeviceError>((decoded, prompt_len, gen_result.tokens_generated))
        })
        .await;

        match result {
            Ok(Ok((generated_text, prompt_tokens, completion_tokens))) => {
                return Json(CompletionResponse {
                    id: format!("cmpl-{}", short_uuid()),
                    object: "text_completion".into(),
                    created: now_secs(),
                    model: model_name,
                    choices: vec![CompletionChoice {
                        text: generated_text,
                        index: 0,
                        finish_reason: Some("stop".into()),
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                });
            }
            Ok(Err(e)) => {
                log::error!("Generation error: {e}");
                // Fall through to placeholder
            }
            Err(e) => {
                log::error!("Task join error: {e}");
                // Fall through to placeholder
            }
        }
    }

    // Placeholder fallback (no model loaded)
    let preview = &prompt[..prompt.len().min(50)];
    let generated_text = format!(
        "[TensorWarp: would generate {max_tokens} tokens from prompt '{preview}']"
    );

    let prompt_tokens = prompt.split_whitespace().count();
    Json(CompletionResponse {
        id: format!("cmpl-{}", short_uuid()),
        object: "text_completion".into(),
        created: now_secs(),
        model: state.model_name.clone(),
        choices: vec![CompletionChoice {
            text: generated_text,
            index: 0,
            finish_reason: Some("length".into()),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: max_tokens,
            total_tokens: prompt_tokens + max_tokens,
        },
    })
}

async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Json<ChatCompletionResponse> {
    let last_msg = req
        .messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default();
    let max_tokens = req.max_tokens.unwrap_or(128);

    // If engine + tokenizer are loaded, run actual generation
    if let (Some(engine), Some(tokenizer), Some(device)) =
        (&state.engine, &state.tokenizer, &state.device)
    {
        let engine = engine.clone();
        let tokenizer = tokenizer.clone();
        let device = device.clone();
        let model_name = state.model_name.clone();

        let mut gen_config = state.gen_config.clone();
        gen_config.max_tokens = max_tokens;
        if let Some(temp) = req.temperature {
            gen_config.temperature = temp;
            gen_config.greedy = temp <= 0.01;
        }

        // Concatenate all messages into a single prompt string
        let full_prompt: String = req
            .messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let result = tokio::task::spawn_blocking(move || {
            let token_ids: Vec<i32> = tokenizer
                .encode(&full_prompt)
                .iter()
                .map(|&id| id as i32)
                .collect();
            let prompt_len = token_ids.len();

            let eng = engine.lock().unwrap();
            let gen_result = eng.generate_with_cache(&device, &token_ids, &gen_config, 2048)?;

            let output_ids: Vec<u32> = gen_result.tokens.iter().map(|&t| t as u32).collect();
            let decoded = tokenizer.decode(&output_ids);

            Ok::<_, warp_kernels::device::DeviceError>((decoded, prompt_len, gen_result.tokens_generated))
        })
        .await;

        match result {
            Ok(Ok((generated_text, prompt_tokens, completion_tokens))) => {
                return Json(ChatCompletionResponse {
                    id: format!("chatcmpl-{}", short_uuid()),
                    object: "chat.completion".into(),
                    created: now_secs(),
                    model: model_name,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".into(),
                            content: generated_text,
                        },
                        finish_reason: Some("stop".into()),
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                });
            }
            Ok(Err(e)) => {
                log::error!("Chat generation error: {e}");
            }
            Err(e) => {
                log::error!("Chat task join error: {e}");
            }
        }
    }

    // Placeholder fallback (no model loaded)
    let preview = &last_msg[..last_msg.len().min(50)];
    let generated = format!(
        "[TensorWarp: would generate {max_tokens} tokens for '{preview}']"
    );

    let prompt_tokens = last_msg.split_whitespace().count();
    Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", short_uuid()),
        object: "chat.completion".into(),
        created: now_secs(),
        model: state.model_name.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: generated,
            },
            finish_reason: Some("length".into()),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: max_tokens,
            total_tokens: prompt_tokens + max_tokens,
        },
    })
}

async fn list_models(State(state): State<Arc<ServerState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".into(),
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model".into(),
            owned_by: "tensorwarp".into(),
        }],
    })
}

async fn health() -> &'static str {
    "ok"
}

// ─── Router ─────────────────────────────────────────────────────────────────

pub fn create_router(state: Arc<ServerState>) -> Router {
    Router::new()
        .route("/v1/completions", routing::post(completions))
        .route("/v1/chat/completions", routing::post(chat_completions))
        .route("/v1/models", routing::get(list_models))
        .route("/health", routing::get(health))
        .with_state(state)
}

/// Start the OpenAI-compatible inference server.
///
/// If `model_path` is provided, loads SafeTensors + tokenizer from that directory
/// and creates a GenerationEngine for real inference. Otherwise runs in placeholder
/// mode (useful for testing the API without a GPU model).
pub async fn serve(
    host: &str,
    port: u16,
    model_name: &str,
    model_path: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = if let Some(path) = model_path {
        println!("Loading model from {path}...");

        // Load model on a blocking thread (GPU init + weight loading is synchronous)
        let path = path.to_string();
        let model_name = model_name.to_string();
        let state = tokio::task::spawn_blocking(move || -> Result<ServerState, Box<dyn std::error::Error + Send + Sync>> {
            let device = WarpDevice::new(0)?;

            // Load config
            let config_path = std::path::Path::new(&path).join("config.json");
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| format!("Failed to read config.json: {e}"))?;
            let llama_config: warp_loader::LlamaConfig = serde_json::from_str(&config_str)
                .map_err(|e| format!("Failed to parse config.json: {e}"))?;

            // Find and load SafeTensors file(s) from the model directory
            let model_dir = std::path::Path::new(&path);
            let st_file = std::fs::read_dir(model_dir)
                .map_err(|e| format!("Failed to read model directory: {e}"))?
                .filter_map(|e| e.ok())
                .find(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
                .ok_or_else(|| "No .safetensors file found in model directory".to_string())?;
            let loader = warp_loader::safetensors_loader::ShardedSafeTensorsLoader::open(&[st_file.path()])
                .map_err(|e| format!("Failed to load SafeTensors: {e}"))?;
            let llama_model = warp_loader::LlamaModel::load(&loader, &llama_config, &device)
                .map_err(|e| format!("Failed to load LLaMA model: {e}"))?;

            // Build GenerationEngine from loaded model
            let engine = GenerationEngine {
                config: llama_model.transformer_config.clone(),
                vocab_size: llama_config.vocab_size,
                embed_tokens: llama_model.embed_tokens,
                layers: llama_model.layers,
                final_norm: llama_model.final_norm,
                lm_head: llama_model.lm_head,
                cache: warp_kernels::cache::KernelCache::new(),
                pool: warp_kernels::mem_pool::GpuMemPool::new(),
            };

            // Load tokenizer
            let tok_path = std::path::Path::new(&path).join("tokenizer.json");
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

            println!("Model loaded: {} layers, vocab={}, hidden={}",
                engine.layers.len(), engine.vocab_size, engine.config.hidden_size);

            Ok(ServerState::with_engine(model_name, engine, tokenizer, device))
        })
        .await.map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?
         .map_err(|e| -> Box<dyn std::error::Error> { e })?;

        Arc::new(state)
    } else {
        Arc::new(ServerState::new(model_name.to_string()))
    };

    let app = create_router(state);
    let addr = format!("{host}:{port}");

    println!("TensorWarp server starting on {addr}");
    println!("  POST /v1/completions");
    println!("  POST /v1/chat/completions");
    println!("  GET  /v1/models");
    println!("  GET  /health");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

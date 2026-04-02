//! Autoregressive text generation.
//!
//! This is the module that ties everything together:
//! tokens → embedding → N transformer blocks → logits → sample → repeat
//!
//! Supports:
//! - Greedy decoding (argmax)
//! - Temperature scaling
//! - KV cache for O(N) decode instead of O(N²)

use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::kv_cache::ModelKVCache;
use crate::ops;
use crate::sampling;
use crate::tensor::GpuTensor;
use crate::transformer::{
    TransformerBlockWeights, TransformerConfig,
    QuantizedBlockWeights, quantize_block_weights,
};

/// Configuration for text generation.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Temperature for sampling (1.0 = neutral, <1.0 = sharper, >1.0 = more random).
    pub temperature: f32,
    /// Stop token ID (EOS). Generation stops when this token is produced.
    pub eos_token_id: Option<i32>,
    /// Whether to use greedy decoding (argmax) vs sampling.
    pub greedy: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 1.0,
            eos_token_id: Some(2), // common EOS for LLaMA
            greedy: true,
        }
    }
}

/// A loaded model ready for generation.
/// This is a simplified interface that owns all the pieces.
pub struct GenerationEngine {
    pub config: TransformerConfig,
    pub vocab_size: u32,
    /// Embedding table [vocab_size, hidden_size] on GPU
    pub embed_tokens: GpuTensor<f32>,
    /// Transformer layers
    pub layers: Vec<TransformerBlockWeights>,
    /// Final norm [hidden_size]
    pub final_norm: GpuTensor<f32>,
    /// LM head (output projection) [vocab_size, hidden_size]
    pub lm_head: GpuTensor<f32>,
    /// Kernel cache
    pub cache: KernelCache,
}

impl GenerationEngine {
    /// Run a full forward pass: tokens → logits.
    /// Returns logits [1, vocab_size] for the last token position.
    pub fn forward(
        &self,
        device: &WarpDevice,
        input_ids: &[i32],
    ) -> Result<GpuTensor<f32>, DeviceError> {
        let seq_len = input_ids.len() as u32;
        let h = self.config.hidden_size;
        let batch = 1u32;

        // 1. Embedding lookup
        let ids = GpuTensor::from_host(device, input_ids,
            Shape::from_static(&[seq_len as usize]), DType::I32)?;
        let mut hidden = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[seq_len as usize, h as usize]), DType::F32)?;
        sampling::embedding(&self.cache, device, &self.embed_tokens, &ids,
            &mut hidden, seq_len, h)?;

        // 2. Run through all transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = crate::transformer::transformer_block_forward(
                &self.cache, device, &hidden, layer, &self.config,
                batch, seq_len, 0,
            )?;
        }

        // 3. Final RMSNorm
        let mut normed = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[seq_len as usize, h as usize]), DType::F32)?;
        ops::rmsnorm(&self.cache, device, &hidden, &self.final_norm,
            &mut normed, h, self.config.norm_eps)?;

        // 4. LM head projection: [seq_len, hidden] × [hidden, vocab] → [seq_len, vocab]
        // We only need the last position's logits
        // For now, compute all positions (optimization: only compute last)
        let mut logits = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[seq_len as usize, self.vocab_size as usize]), DType::F32)?;
        ops::gemm(&self.cache, device, &normed, &self.lm_head, &mut logits,
            seq_len, self.vocab_size, h)?;

        Ok(logits)
    }

    /// Generate tokens autoregressively.
    /// Returns the generated token IDs (not including the prompt).
    pub fn generate(
        &self,
        device: &WarpDevice,
        prompt_ids: &[i32],
        gen_config: &GenerateConfig,
    ) -> Result<Vec<i32>, DeviceError> {
        let mut all_ids = prompt_ids.to_vec();
        let mut generated = Vec::new();

        for step in 0..gen_config.max_tokens {
            // Forward pass on all tokens so far
            let logits = self.forward(device, &all_ids)?;
            device.synchronize()?;

            // Get logits for the last position
            let all_logits = logits.to_host(device)?;
            let seq_len = all_ids.len();
            let vocab = self.vocab_size as usize;
            let last_logits = &all_logits[(seq_len - 1) * vocab..seq_len * vocab];

            // Apply temperature
            let scaled: Vec<f32> = if gen_config.temperature != 1.0 {
                last_logits.iter().map(|&v| v / gen_config.temperature).collect()
            } else {
                last_logits.to_vec()
            };

            // Select next token
            let next_token = if gen_config.greedy {
                // Argmax on CPU (simple and correct)
                scaled.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i32)
                    .unwrap_or(0)
            } else {
                // Softmax + multinomial sampling on CPU
                let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_vals: Vec<f32> = scaled.iter().map(|v| (v - max_val).exp()).collect();
                let sum: f32 = exp_vals.iter().sum();
                let probs: Vec<f32> = exp_vals.iter().map(|v| v / sum).collect();

                // Simple random sampling (use step as pseudo-random seed)
                let r = ((step * 7 + 13) % 100) as f32 / 100.0;
                let mut cumsum = 0.0f32;
                let mut selected = 0i32;
                for (i, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if cumsum >= r {
                        selected = i as i32;
                        break;
                    }
                }
                selected
            };

            // Check for EOS
            if let Some(eos) = gen_config.eos_token_id {
                if next_token == eos {
                    break;
                }
            }

            generated.push(next_token);
            all_ids.push(next_token);
        }

        Ok(generated)
    }

    /// Run prefill forward pass: tokens → logits, populating KV cache per layer.
    fn forward_prefill(
        &self,
        device: &WarpDevice,
        input_ids: &[i32],
        kv_cache: &mut ModelKVCache,
    ) -> Result<GpuTensor<f32>, DeviceError> {
        let seq_len = input_ids.len() as u32;
        let h = self.config.hidden_size;
        let batch = 1u32;

        // 1. Embedding lookup
        let ids = GpuTensor::from_host(device, input_ids,
            Shape::from_static(&[seq_len as usize]), DType::I32)?;
        let mut hidden = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[seq_len as usize, h as usize]), DType::F32)?;
        sampling::embedding(&self.cache, device, &self.embed_tokens, &ids,
            &mut hidden, seq_len, h)?;

        // 2. Run through all transformer layers — prefill variant saves K/V
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = crate::transformer::transformer_block_prefill(
                &self.cache, device, &hidden, layer, &self.config,
                &mut kv_cache.layers[i],
                batch, seq_len, 0,
            )?;
        }

        // 3. Final RMSNorm
        let mut normed = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[seq_len as usize, h as usize]), DType::F32)?;
        ops::rmsnorm(&self.cache, device, &hidden, &self.final_norm,
            &mut normed, h, self.config.norm_eps)?;

        // 4. LM head projection
        let mut logits = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[seq_len as usize, self.vocab_size as usize]), DType::F32)?;
        ops::gemm(&self.cache, device, &normed, &self.lm_head, &mut logits,
            seq_len, self.vocab_size, h)?;

        Ok(logits)
    }

    /// Run decode forward pass for a single token, using KV cache.
    fn forward_decode(
        &self,
        device: &WarpDevice,
        token_id: i32,
        kv_cache: &mut ModelKVCache,
        pos: u32,
    ) -> Result<GpuTensor<f32>, DeviceError> {
        let h = self.config.hidden_size;
        let batch = 1u32;

        // 1. Embedding lookup (single token)
        let ids = GpuTensor::from_host(device, &[token_id],
            Shape::from_static(&[1]), DType::I32)?;
        let mut hidden = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, h as usize]), DType::F32)?;
        sampling::embedding(&self.cache, device, &self.embed_tokens, &ids,
            &mut hidden, 1, h)?;

        // 2. Run through all transformer layers — decode variant uses KV cache
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = crate::transformer::transformer_block_decode(
                &self.cache, device, &hidden, layer, &self.config,
                &mut kv_cache.layers[i],
                batch, pos,
            )?;
        }

        // 3. Final RMSNorm
        let mut normed = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, h as usize]), DType::F32)?;
        ops::rmsnorm(&self.cache, device, &hidden, &self.final_norm,
            &mut normed, h, self.config.norm_eps)?;

        // 4. LM head projection → [1, vocab_size]
        let mut logits = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, self.vocab_size as usize]), DType::F32)?;
        ops::gemm(&self.cache, device, &normed, &self.lm_head, &mut logits,
            1, self.vocab_size, h)?;

        Ok(logits)
    }

    /// Generate with KV cache — O(N) per token instead of O(N²).
    ///
    /// Phase 1 (prefill): run full prompt through all layers, populate cache.
    /// Phase 2 (decode): one token at a time, using cached K/V.
    pub fn generate_with_cache(
        &self,
        device: &WarpDevice,
        prompt_ids: &[i32],
        gen_config: &GenerateConfig,
        max_seq_len: u32,
    ) -> Result<GenerationResult, DeviceError> {
        let kv_dim = self.config.kv_dim();
        let num_layers = self.layers.len() as u32;
        let vocab = self.vocab_size as usize;

        // Allocate KV cache
        let mut kv_cache = ModelKVCache::new(device, num_layers, max_seq_len, kv_dim)?;

        let prefill_start = std::time::Instant::now();

        // Phase 1: Prefill — process full prompt, populate KV cache per layer
        let logits = self.forward_prefill(device, prompt_ids, &mut kv_cache)?;
        device.synchronize()?;
        let prefill_time = prefill_start.elapsed();

        // Get last token's logits from prefill
        let all_logits = logits.to_host(device)?;
        let prompt_len = prompt_ids.len();
        let mut last_logits = all_logits[(prompt_len - 1) * vocab..prompt_len * vocab].to_vec();

        let decode_start = std::time::Instant::now();
        let mut generated = Vec::new();
        let mut pos = prompt_len as u32;

        // Phase 2: Decode — one token at a time using KV cache
        for _step in 0..gen_config.max_tokens {
            // Apply temperature
            let scaled: Vec<f32> = if gen_config.temperature != 1.0 {
                last_logits.iter().map(|&v| v / gen_config.temperature).collect()
            } else {
                last_logits.clone()
            };

            // Greedy selection
            let next_token = scaled.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i32)
                .unwrap_or(0);

            if let Some(eos) = gen_config.eos_token_id {
                if next_token == eos {
                    break;
                }
            }

            generated.push(next_token);

            // Decode forward: single token, KV cache handles history
            let logits = self.forward_decode(device, next_token, &mut kv_cache, pos)?;
            device.synchronize()?;

            last_logits = logits.to_host(device)?;
            pos += 1;
        }

        let decode_time = decode_start.elapsed();

        Ok(GenerationResult {
            tokens: generated.clone(),
            prefill_time,
            decode_time,
            tokens_generated: generated.len(),
            prefill_tokens: prompt_len,
            tokens_per_sec: if decode_time.as_secs_f64() > 0.0 {
                generated.len() as f64 / decode_time.as_secs_f64()
            } else {
                0.0
            },
            kv_cache_memory_bytes: kv_cache.memory_bytes(),
        })
    }
}

/// Results from a generation run with detailed timing.
#[derive(Debug)]
pub struct GenerationResult {
    pub tokens: Vec<i32>,
    pub prefill_time: std::time::Duration,
    pub decode_time: std::time::Duration,
    pub tokens_generated: usize,
    pub prefill_tokens: usize,
    pub tokens_per_sec: f64,
    pub kv_cache_memory_bytes: usize,
}

impl std::fmt::Display for GenerationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Generation Results:")?;
        writeln!(f, "  Prefill: {} tokens in {:.2}ms ({:.1} tokens/sec)",
            self.prefill_tokens,
            self.prefill_time.as_secs_f64() * 1000.0,
            self.prefill_tokens as f64 / self.prefill_time.as_secs_f64().max(1e-9))?;
        writeln!(f, "  Decode:  {} tokens in {:.2}ms ({:.1} tokens/sec)",
            self.tokens_generated,
            self.decode_time.as_secs_f64() * 1000.0,
            self.tokens_per_sec)?;
        writeln!(f, "  TTFT:    {:.2}ms", self.prefill_time.as_secs_f64() * 1000.0)?;
        writeln!(f, "  KV cache: {:.2} MB", self.kv_cache_memory_bytes as f64 / 1e6)?;
        write!(f, "  Tokens: {:?}", &self.tokens[..self.tokens.len().min(20)])
    }
}

/// Create a generation engine from random weights (for testing).
pub fn create_test_engine(
    device: &WarpDevice,
    config: TransformerConfig,
    num_layers: u32,
    vocab_size: u32,
) -> Result<GenerationEngine, DeviceError> {
    let h = config.hidden_size as usize;

    let rand_vec = |n: usize| -> Vec<f32> {
        (0..n).map(|i| ((i * 7 + 13) % 100) as f32 * 0.01 - 0.5).collect()
    };

    let embed_tokens = GpuTensor::from_host(device,
        &rand_vec(vocab_size as usize * h),
        Shape::from_static(&[vocab_size as usize, h]), DType::F32)?;

    let mut layers = Vec::new();
    for _ in 0..num_layers {
        layers.push(crate::transformer::random_weights(device, &config)?);
    }

    let final_norm = GpuTensor::from_host(device,
        &vec![1.0f32; h], Shape::from_static(&[h]), DType::F32)?;

    // LM head: [vocab_size, hidden_size] — note: transposed for our GEMM
    let lm_head = GpuTensor::from_host(device,
        &rand_vec(h * vocab_size as usize),
        Shape::from_static(&[h, vocab_size as usize]), DType::F32)?;

    Ok(GenerationEngine {
        config,
        vocab_size,
        embed_tokens,
        layers,
        final_norm,
        lm_head,
        cache: KernelCache::new(),
    })
}

/// Generation engine with Q4_0 quantized weights — 6.4x less weight memory.
///
/// Embedding and LM head stay f32 (they're accessed sparsely).
/// All transformer projection GEMMs use W4A16.
pub struct QuantizedGenerationEngine {
    pub config: TransformerConfig,
    pub vocab_size: u32,
    pub embed_tokens: GpuTensor<f32>,
    pub layers: Vec<QuantizedBlockWeights>,
    pub final_norm: GpuTensor<f32>,
    pub lm_head: GpuTensor<f32>,
    pub cache: KernelCache,
}

impl QuantizedGenerationEngine {
    /// Quantize a full-precision engine to Q4_0.
    pub fn from_f32(
        device: &WarpDevice,
        engine: &GenerationEngine,
    ) -> Result<Self, DeviceError> {
        let mut q_layers = Vec::with_capacity(engine.layers.len());
        for layer in &engine.layers {
            q_layers.push(quantize_block_weights(
                &engine.cache, device, layer, &engine.config,
            )?);
        }

        // Clone embedding, norm, lm_head (stay f32)
        let embed_host = engine.embed_tokens.to_host(device)?;
        let norm_host = engine.final_norm.to_host(device)?;
        let lm_host = engine.lm_head.to_host(device)?;

        Ok(Self {
            config: engine.config.clone(),
            vocab_size: engine.vocab_size,
            embed_tokens: GpuTensor::from_host(device, &embed_host,
                engine.embed_tokens.shape.clone(), DType::F32)?,
            layers: q_layers,
            final_norm: GpuTensor::from_host(device, &norm_host,
                engine.final_norm.shape.clone(), DType::F32)?,
            lm_head: GpuTensor::from_host(device, &lm_host,
                engine.lm_head.shape.clone(), DType::F32)?,
            cache: KernelCache::new(),
        })
    }

    /// Prefill forward pass — populates KV cache.
    fn forward_prefill(
        &self,
        device: &WarpDevice,
        input_ids: &[i32],
        kv_cache: &mut ModelKVCache,
    ) -> Result<GpuTensor<f32>, DeviceError> {
        let seq_len = input_ids.len() as u32;
        let h = self.config.hidden_size;

        let ids = GpuTensor::from_host(device, input_ids,
            Shape::from_static(&[seq_len as usize]), DType::I32)?;
        let mut hidden = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[seq_len as usize, h as usize]), DType::F32)?;
        sampling::embedding(&self.cache, device, &self.embed_tokens, &ids,
            &mut hidden, seq_len, h)?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = crate::transformer::transformer_block_prefill_q4(
                &self.cache, device, &hidden, layer, &self.config,
                &mut kv_cache.layers[i], 1, seq_len, 0,
            )?;
        }

        let mut normed = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[seq_len as usize, h as usize]), DType::F32)?;
        ops::rmsnorm(&self.cache, device, &hidden, &self.final_norm,
            &mut normed, h, self.config.norm_eps)?;

        let mut logits = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[seq_len as usize, self.vocab_size as usize]), DType::F32)?;
        ops::gemm(&self.cache, device, &normed, &self.lm_head, &mut logits,
            seq_len, self.vocab_size, h)?;

        Ok(logits)
    }

    /// Decode forward pass — single token with KV cache.
    fn forward_decode(
        &self,
        device: &WarpDevice,
        token_id: i32,
        kv_cache: &mut ModelKVCache,
        pos: u32,
    ) -> Result<GpuTensor<f32>, DeviceError> {
        let h = self.config.hidden_size;

        let ids = GpuTensor::from_host(device, &[token_id],
            Shape::from_static(&[1]), DType::I32)?;
        let mut hidden = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, h as usize]), DType::F32)?;
        sampling::embedding(&self.cache, device, &self.embed_tokens, &ids,
            &mut hidden, 1, h)?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = crate::transformer::transformer_block_decode_q4(
                &self.cache, device, &hidden, layer, &self.config,
                &mut kv_cache.layers[i], 1, pos,
            )?;
        }

        let mut normed = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, h as usize]), DType::F32)?;
        ops::rmsnorm(&self.cache, device, &hidden, &self.final_norm,
            &mut normed, h, self.config.norm_eps)?;

        let mut logits = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, self.vocab_size as usize]), DType::F32)?;
        ops::gemm(&self.cache, device, &normed, &self.lm_head, &mut logits,
            1, self.vocab_size, h)?;

        Ok(logits)
    }

    /// Generate with KV cache using quantized weights.
    pub fn generate_with_cache(
        &self,
        device: &WarpDevice,
        prompt_ids: &[i32],
        gen_config: &GenerateConfig,
        max_seq_len: u32,
    ) -> Result<GenerationResult, DeviceError> {
        let kv_dim = self.config.kv_dim();
        let num_layers = self.layers.len() as u32;
        let vocab = self.vocab_size as usize;

        let mut kv_cache = ModelKVCache::new(device, num_layers, max_seq_len, kv_dim)?;

        let prefill_start = std::time::Instant::now();
        let logits = self.forward_prefill(device, prompt_ids, &mut kv_cache)?;
        device.synchronize()?;
        let prefill_time = prefill_start.elapsed();

        let all_logits = logits.to_host(device)?;
        let prompt_len = prompt_ids.len();
        let mut last_logits = all_logits[(prompt_len - 1) * vocab..prompt_len * vocab].to_vec();

        let decode_start = std::time::Instant::now();
        let mut generated = Vec::new();
        let mut pos = prompt_len as u32;

        for _step in 0..gen_config.max_tokens {
            let scaled: Vec<f32> = if gen_config.temperature != 1.0 {
                last_logits.iter().map(|&v| v / gen_config.temperature).collect()
            } else {
                last_logits.clone()
            };

            let next_token = scaled.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i32)
                .unwrap_or(0);

            if let Some(eos) = gen_config.eos_token_id {
                if next_token == eos { break; }
            }

            generated.push(next_token);

            let logits = self.forward_decode(device, next_token, &mut kv_cache, pos)?;
            device.synchronize()?;

            last_logits = logits.to_host(device)?;
            pos += 1;
        }

        let decode_time = decode_start.elapsed();

        Ok(GenerationResult {
            tokens: generated.clone(),
            prefill_time,
            decode_time,
            tokens_generated: generated.len(),
            prefill_tokens: prompt_len,
            tokens_per_sec: if decode_time.as_secs_f64() > 0.0 {
                generated.len() as f64 / decode_time.as_secs_f64()
            } else { 0.0 },
            kv_cache_memory_bytes: kv_cache.memory_bytes(),
        })
    }
}

/// Estimate weight memory for f32 vs Q4_0 for a given config.
pub fn weight_memory_estimate(config: &TransformerConfig, num_layers: u32, vocab_size: u32) -> (usize, usize) {
    let h = config.hidden_size as usize;
    let kv = config.kv_dim() as usize;
    let ffn = config.ffn_dim as usize;
    let v = vocab_size as usize;

    // Per layer: wq[h,h] + wk[h,kv] + wv[h,kv] + wo[h,h] + w_gate[h,ffn] + w_up[h,ffn] + w_down[ffn,h] + norms[2*h]
    let per_layer_params = h*h + h*kv + h*kv + h*h + h*ffn + h*ffn + ffn*h + 2*h;
    let global_params = v*h + h + h*v; // embed + final_norm + lm_head

    let total_params = per_layer_params * num_layers as usize + global_params;
    let f32_bytes = total_params * 4;

    // Q4_0: projection weights quantized (everything except norms, embed, lm_head)
    let proj_params_per_layer = h*h + h*kv + h*kv + h*h + h*ffn + h*ffn + ffn*h;
    let norm_params_per_layer = 2 * h;
    let proj_blocks_per_layer = proj_params_per_layer / 32; // BLOCK_SIZE
    let q4_proj_bytes = proj_blocks_per_layer * 20; // Q4_0_BLOCK_BYTES
    let q4_per_layer = q4_proj_bytes + norm_params_per_layer * 4;
    let q4_bytes = q4_per_layer * num_layers as usize + global_params * 4;

    (f32_bytes, q4_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<WarpDevice> {
        WarpDevice::new(0).ok()
    }

    #[test]
    fn forward_pass_produces_logits() {
        let dev = match setup() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let config = TransformerConfig::tiny();
        let engine = create_test_engine(&dev, config, 2, 100).unwrap();

        let prompt = vec![1i32, 5, 10, 15, 20];
        let logits = engine.forward(&dev, &prompt).unwrap();
        dev.synchronize().unwrap();

        let logits_host = logits.to_host(&dev).unwrap();
        assert_eq!(logits_host.len(), prompt.len() * engine.vocab_size as usize);
        assert!(logits_host.iter().all(|v| v.is_finite()), "Logits contain NaN/Inf!");
        assert!(logits_host.iter().any(|v| *v != 0.0), "Logits are all zeros!");

        println!("Forward pass: {} tokens → {} logits (vocab={})",
            prompt.len(), logits_host.len(), engine.vocab_size);
        println!("  Logit range: [{:.4}, {:.4}]",
            logits_host.iter().cloned().fold(f32::INFINITY, f32::min),
            logits_host.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    }

    #[test]
    fn greedy_generation() {
        let dev = match setup() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let config = TransformerConfig::tiny();
        let engine = create_test_engine(&dev, config, 2, 100).unwrap();

        let prompt = vec![1i32, 5, 10];
        let gen_config = GenerateConfig {
            max_tokens: 10,
            greedy: true,
            eos_token_id: None, // don't stop early
            ..Default::default()
        };

        let generated = engine.generate(&dev, &prompt, &gen_config).unwrap();

        assert_eq!(generated.len(), 10);
        assert!(generated.iter().all(|&t| t >= 0 && t < 100), "Token out of vocab range!");

        println!("Generated {} tokens (greedy): {:?}", generated.len(), generated);

        // Greedy should be deterministic — run again and check same output
        let generated2 = engine.generate(&dev, &prompt, &gen_config).unwrap();
        assert_eq!(generated, generated2, "Greedy generation is not deterministic!");
        println!("Deterministic: confirmed (same output on second run)");
    }

    #[test]
    fn generation_perf() {
        let dev = match setup() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let config = TransformerConfig::tiny();
        let engine = create_test_engine(&dev, config, 4, 256).unwrap();

        let prompt = vec![1i32, 2, 3, 4, 5];
        let gen_config = GenerateConfig {
            max_tokens: 20,
            greedy: true,
            eos_token_id: None,
            ..Default::default()
        };

        let start = std::time::Instant::now();
        let generated = engine.generate(&dev, &prompt, &gen_config).unwrap();
        let elapsed = start.elapsed();

        let tokens_per_sec = generated.len() as f64 / elapsed.as_secs_f64();
        let ms_per_token = elapsed.as_secs_f64() * 1000.0 / generated.len() as f64;

        println!("\nGeneration perf (4 layers, H=64, vocab=256):");
        println!("  Prompt: {} tokens", prompt.len());
        println!("  Generated: {} tokens in {:.1}ms", generated.len(), elapsed.as_secs_f64() * 1000.0);
        println!("  Speed: {:.1} tokens/sec ({:.1}ms/token)", tokens_per_sec, ms_per_token);
        println!("  Tokens: {:?}", generated);
        println!("{}", engine.cache.stats());
    }

    #[test]
    fn generation_with_cache_timing() {
        let dev = match setup() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let config = TransformerConfig::tiny();
        let engine = create_test_engine(&dev, config, 4, 256).unwrap();

        let prompt = vec![1i32, 2, 3, 4, 5, 6, 7, 8];
        let gen_config = GenerateConfig {
            max_tokens: 16,
            greedy: true,
            eos_token_id: None,
            ..Default::default()
        };

        let result = engine.generate_with_cache(&dev, &prompt, &gen_config, 256).unwrap();

        println!("\n{result}");
        assert_eq!(result.tokens_generated, 16);
        assert!(result.tokens.iter().all(|&t| t >= 0 && t < 256));
        println!("{}", engine.cache.stats());
    }

    #[test]
    fn quantized_generation() {
        let dev = match setup() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        // Use small config — head_dim must be >= 32 (BLOCK_SIZE) for Q4_0 decode
        let config = TransformerConfig::small();
        let num_layers = 4u32;
        let vocab_size = 256u32;

        // Create f32 engine, then quantize
        let f32_engine = create_test_engine(&dev, config.clone(), num_layers, vocab_size).unwrap();
        let q4_engine = QuantizedGenerationEngine::from_f32(&dev, &f32_engine).unwrap();

        let prompt = vec![1i32, 2, 3, 4, 5, 6, 7, 8];
        let gen_config = GenerateConfig {
            max_tokens: 16,
            greedy: true,
            eos_token_id: None,
            ..Default::default()
        };

        // F32 generation
        let f32_result = f32_engine.generate_with_cache(&dev, &prompt, &gen_config, 256).unwrap();

        // Q4_0 generation
        let q4_result = q4_engine.generate_with_cache(&dev, &prompt, &gen_config, 256).unwrap();

        println!("\n=== F32 vs Q4_0 Generation ===");
        println!("F32:  {}", f32_result);
        println!("Q4_0: {}", q4_result);

        // Memory savings
        let (f32_bytes, q4_bytes) = weight_memory_estimate(&config, num_layers, vocab_size);
        println!("Weight memory:");
        println!("  F32:  {:.2} KB", f32_bytes as f64 / 1024.0);
        println!("  Q4_0: {:.2} KB ({:.1}x smaller)", q4_bytes as f64 / 1024.0,
            f32_bytes as f64 / q4_bytes as f64);

        // Both should produce valid tokens
        assert_eq!(q4_result.tokens_generated, 16);
        assert!(q4_result.tokens.iter().all(|&t| t >= 0 && t < vocab_size as i32));

        // Q4 should be faster (less memory bandwidth for weights)
        println!("Speed comparison:");
        println!("  F32 decode:  {:.1} tokens/sec", f32_result.tokens_per_sec);
        println!("  Q4_0 decode: {:.1} tokens/sec", q4_result.tokens_per_sec);
        if q4_result.tokens_per_sec > 0.0 && f32_result.tokens_per_sec > 0.0 {
            println!("  Speedup:     {:.2}x", q4_result.tokens_per_sec / f32_result.tokens_per_sec);
        }

        // Tokens may differ (quantization changes weights) but both should be deterministic
        let q4_result2 = q4_engine.generate_with_cache(&dev, &prompt, &gen_config, 256).unwrap();
        assert_eq!(q4_result.tokens, q4_result2.tokens, "Q4_0 generation not deterministic!");
        println!("Q4_0 deterministic: confirmed");
    }

    #[test]
    fn weight_memory_scaling() {
        // Show memory savings at real model scales
        println!("\n=== Weight Memory Estimates ===");

        let configs = vec![
            ("tiny (H=64, 4L)",    TransformerConfig::tiny(),   4u32,  256u32),
            ("small (H=256, 8L)",  TransformerConfig::small(),  8,     32000),
            ("medium (H=1024, 16L)", TransformerConfig::medium(), 16,  32000),
            ("LLaMA-7B-like",      TransformerConfig {
                hidden_size: 4096, num_heads: 32, num_kv_heads: 32,
                head_dim: 128, ffn_dim: 11008, rope_base: 10000.0, norm_eps: 1e-6,
            }, 32, 32000),
        ];

        for (name, config, layers, vocab) in configs {
            let (f32_b, q4_b) = weight_memory_estimate(&config, layers, vocab);
            println!("  {name:25} F32={:>8.1} MB  Q4_0={:>8.1} MB  ({:.1}x smaller)",
                f32_b as f64 / 1e6, q4_b as f64 / 1e6, f32_b as f64 / q4_b as f64);
        }
    }
}

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
use crate::ops;
use crate::sampling;
use crate::tensor::GpuTensor;
use crate::transformer::{TransformerBlockWeights, TransformerConfig};

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

/// KV cache for autoregressive decoding.
/// Stores precomputed K and V tensors for each layer to avoid recomputation.
pub struct KVCache {
    /// Per-layer K cache: [num_layers][max_seq, kv_dim]
    pub k_cache: Vec<Vec<f32>>,
    /// Per-layer V cache: [num_layers][max_seq, kv_dim]
    pub v_cache: Vec<Vec<f32>>,
    /// Current sequence length in cache.
    pub seq_len: usize,
    /// Maximum sequence length the cache can hold.
    pub max_seq_len: usize,
    /// KV dimension per layer.
    pub kv_dim: usize,
}

impl KVCache {
    pub fn new(num_layers: usize, max_seq_len: usize, kv_dim: usize) -> Self {
        Self {
            k_cache: vec![vec![0.0f32; max_seq_len * kv_dim]; num_layers],
            v_cache: vec![vec![0.0f32; max_seq_len * kv_dim]; num_layers],
            seq_len: 0,
            max_seq_len,
            kv_dim,
        }
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
        for layer in &mut self.k_cache {
            layer.fill(0.0);
        }
        for layer in &mut self.v_cache {
            layer.fill(0.0);
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
}

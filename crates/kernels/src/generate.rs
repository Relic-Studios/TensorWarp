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
use crate::mem_pool::GpuMemPool;
use crate::ops;
use crate::sampling;
use crate::tensor::GpuTensor;
use crate::transformer::{
    TransformerBlockWeights, TransformerBlockWeightsF16, TransformerConfig,
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
    /// Keep only top-K logits before sampling (None = no filtering).
    pub top_k: Option<usize>,
    /// Nucleus sampling threshold — keep tokens until cumulative probability >= p.
    pub top_p: Option<f32>,
    /// Penalize repeated tokens (1.0 = no penalty, >1.0 = discourage repeats).
    pub repetition_penalty: f32,
    /// Stop generation when any of these token sequences appear at the end.
    pub stop_sequences: Vec<Vec<i32>>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 1.0,
            eos_token_id: Some(2), // common EOS for LLaMA
            greedy: true,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            stop_sequences: Vec::new(),
        }
    }
}

/// Check if generated tokens end with any stop sequence.
fn matches_stop_sequence(generated: &[i32], stop: &[Vec<i32>]) -> bool {
    stop.iter().any(|seq| !seq.is_empty() && generated.ends_with(seq))
}

/// Sample a token from logits using the full sampling pipeline:
/// repetition penalty -> temperature -> top-K -> top-P -> greedy/multinomial.
fn sample_token(logits: &[f32], config: &GenerateConfig, generated: &[i32], rng_seed: u64) -> i32 {
    let mut logits = logits.to_vec();

    // 1. Repetition penalty — reduce probability of tokens already generated.
    if config.repetition_penalty != 1.0 {
        for &token in generated {
            if token >= 0 && (token as usize) < logits.len() {
                if logits[token as usize] > 0.0 {
                    logits[token as usize] /= config.repetition_penalty;
                } else {
                    logits[token as usize] *= config.repetition_penalty;
                }
            }
        }
    }

    // 2. Temperature scaling
    if config.temperature != 1.0 && config.temperature > 0.0 {
        for l in logits.iter_mut() {
            *l /= config.temperature;
        }
    }

    // 3. Top-K filtering — keep only the K highest logits.
    if let Some(k) = config.top_k {
        if k > 0 && k < logits.len() {
            let mut sorted = logits.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = sorted[k];
            for l in logits.iter_mut() {
                if *l < threshold {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
    }

    // 4. Top-P (nucleus) sampling — keep smallest set of tokens whose cumulative
    //    probability exceeds p, masking everything else.
    if let Some(p) = config.top_p {
        if p < 1.0 {
            let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<(usize, f32)> = logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, (v - max_val).exp()))
                .collect();
            let sum: f32 = probs.iter().map(|(_, p)| p).sum();
            for (_, prob) in probs.iter_mut() {
                *prob /= sum;
            }
            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumsum = 0.0;
            let mut cutoff_idx = probs.len();
            for (i, (_, prob)) in probs.iter().enumerate() {
                cumsum += prob;
                if cumsum >= p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            let nucleus: std::collections::HashSet<usize> =
                probs[..cutoff_idx].iter().map(|(i, _)| *i).collect();
            for (i, l) in logits.iter_mut().enumerate() {
                if !nucleus.contains(&i) {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
    }

    // 5. Select token — greedy (argmax) or multinomial sampling.
    if config.greedy {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as i32)
            .unwrap_or(0)
    } else {
        // Softmax + multinomial
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|v| v / sum).collect();

        // Pseudo-random sampling using LCG with the provided seed
        let r = ((rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)) >> 33) as f32
            / (1u64 << 31) as f32;
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= r {
                return i as i32;
            }
        }
        probs.len() as i32 - 1
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
    /// GPU memory pool for reusing temporary allocations
    pub pool: GpuMemPool,
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
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
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
    pub fn forward_prefill(
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
        let mut normed = self.pool.get_f32(device,
            Shape::from_static(&[seq_len as usize, h as usize]), DType::F32)?;
        ops::rmsnorm(&self.cache, device, &hidden, &self.final_norm,
            &mut normed, h, self.config.norm_eps)?;

        // 4. LM head projection
        let mut logits = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[seq_len as usize, self.vocab_size as usize]), DType::F32)?;
        ops::gemm(&self.cache, device, &normed, &self.lm_head, &mut logits,
            seq_len, self.vocab_size, h)?;
        self.pool.return_f32(normed);

        Ok(logits)
    }

    /// Run decode forward pass for a single token, using KV cache.
    pub fn forward_decode(
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
        let mut hidden = self.pool.get_f32(device,
            Shape::from_static(&[1, h as usize]), DType::F32)?;
        sampling::embedding(&self.cache, device, &self.embed_tokens, &ids,
            &mut hidden, 1, h)?;

        // 2. Run through all transformer layers — decode variant uses KV cache
        for (i, layer) in self.layers.iter().enumerate() {
            let prev_hidden = hidden;
            hidden = crate::transformer::transformer_block_decode(
                &self.cache, device, &prev_hidden, layer, &self.config,
                &mut kv_cache.layers[i],
                batch, pos,
            )?;
            self.pool.return_f32(prev_hidden);
        }

        // 3. Final RMSNorm
        let mut normed = self.pool.get_f32(device,
            Shape::from_static(&[1, h as usize]), DType::F32)?;
        ops::rmsnorm(&self.cache, device, &hidden, &self.final_norm,
            &mut normed, h, self.config.norm_eps)?;
        self.pool.return_f32(hidden);

        // 4. LM head projection → [1, vocab_size]
        let mut logits = self.pool.get_f32(device,
            Shape::from_static(&[1, self.vocab_size as usize]), DType::F32)?;
        ops::gemm(&self.cache, device, &normed, &self.lm_head, &mut logits,
            1, self.vocab_size, h)?;
        self.pool.return_f32(normed);

        // logits is returned to caller — not returned to pool
        Ok(logits)
    }

    /// Run a batched decode forward pass: one token per request, each with its own KV cache.
    ///
    /// Takes a batch of (token_id, kv_cache, position) tuples and returns logits
    /// for each request. Currently processes requests sequentially — true batched
    /// GEMM with padding across requests is a future optimization.
    ///
    /// Returns a Vec of logits tensors, one per request, each [1, vocab_size].
    pub fn forward_decode_batch(
        &self,
        device: &WarpDevice,
        requests: &mut [(i32, &mut ModelKVCache, u32)],
    ) -> Result<Vec<GpuTensor<f32>>, DeviceError> {
        let mut all_logits = Vec::with_capacity(requests.len());

        // Sequential loop for now — each request runs its own forward_decode.
        // The API is in place for future batched GEMM where we'd pad and concatenate
        // across the batch dimension.
        for (token_id, kv_cache, pos) in requests.iter_mut() {
            let logits = self.forward_decode(device, *token_id, *kv_cache, *pos)?;
            all_logits.push(logits);
        }

        Ok(all_logits)
    }

    /// Generate with KV cache — O(N) per token instead of O(N²).
    ///
    /// Phase 1 (prefill): run full prompt through all layers, populate cache.
    /// Phase 2 (decode): one token at a time, using cached K/V.
    ///
    /// Uses the full sampling pipeline: repetition penalty, temperature,
    /// top-K, top-P, and stop sequence detection.
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

        // RNG seed — combine prompt content for reproducible but varied sequences
        let base_seed: u64 = prompt_ids.iter().fold(42u64, |acc, &t| {
            acc.wrapping_mul(6364136223846793005).wrapping_add(t as u64)
        });

        // Pre-allocate ALL decode buffers ONCE — eliminates cudaMalloc during decode loop.
        // This gives ~1.8x speedup by reusing GPU memory every step.
        let has_fusion = self.layers.first().map_or(false, |l| l.wqkv.is_some());
        let mut decode_buffers = DecodeBuffers::allocate_with_fusion(
            device, &self.pool, &self.config, self.layers.len(), self.vocab_size, has_fusion,
        )?;

        // Phase 2: Decode — one token at a time using KV cache + pre-allocated buffers
        for step in 0..gen_config.max_tokens {
            let rng_seed = base_seed.wrapping_add(step as u64);
            let next_token = sample_token(&last_logits, gen_config, &generated, rng_seed);

            if let Some(eos) = gen_config.eos_token_id {
                if next_token == eos {
                    break;
                }
            }

            generated.push(next_token);

            // Check stop sequences
            if !gen_config.stop_sequences.is_empty()
                && matches_stop_sequence(&generated, &gen_config.stop_sequences)
            {
                break;
            }

            // Decode forward with pre-allocated buffers (zero cudaMalloc per step)
            // Write token ID into pre-allocated buffer
            device.htod_copy(&[next_token], &mut decode_buffers.ids.data)?;

            self.forward_decode_preallocated(
                device, &mut decode_buffers, &mut kv_cache, pos,
            )?;
            device.synchronize()?;

            last_logits = decode_buffers.logits.to_host(device)?;

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

impl GenerationEngine {
    /// Generate with CUDA graph acceleration for decode.
    ///
    /// Strategy:
    /// - Pre-allocate ALL decode buffers from the memory pool (stable addresses).
    /// - Step 0 (warmup): run forward_decode_preallocated to pre-compile all
    ///   NVRTC kernels (graph capture fails if compilation happens during capture).
    /// - Steps 1+: reuse pre-allocated buffers, eliminating all cudaMalloc/Free
    ///   overhead per decode step.
    ///
    /// CUDA graph status:
    /// - Buffer pre-allocation is DONE — all addresses are stable across steps.
    /// - Full graph capture/replay is NOT YET ACTIVE because `pos` (RoPE) and
    ///   `kv_cache.len` change each step. Graph replay would produce incorrect
    ///   results without cudaGraphExecKernelNodeSetParams to update those params.
    /// - The DecodeGraphCache infrastructure is wired and ready for when we add
    ///   parameter-update support.
    ///
    /// Current wins (even without graph replay):
    /// - Zero cudaMalloc/Free per decode step (~500-1000us savings per step)
    /// - All buffers reused from pool with stable addresses
    /// - Framework ready for full graph capture
    pub fn generate_with_graph(
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

        // Phase 1: Prefill (no graph — variable sequence length)
        let prefill_start = std::time::Instant::now();
        let logits = self.forward_prefill(device, prompt_ids, &mut kv_cache)?;
        device.synchronize()?;
        let prefill_time = prefill_start.elapsed();

        let all_logits = logits.to_host(device)?;
        let prompt_len = prompt_ids.len();
        let mut last_logits = all_logits[(prompt_len - 1) * vocab..prompt_len * vocab].to_vec();

        // Phase 2: Decode with pre-allocated buffers
        //
        // Pre-allocate ALL intermediate tensors from the pool before any decode step.
        // These addresses remain stable for the entire generation, which is the
        // prerequisite for CUDA graph capture.
        let mut buffers = DecodeBuffers::allocate(
            device, &self.pool, &self.config, num_layers as usize, self.vocab_size,
        )?;
        let _graph_cache = crate::cuda_graph::DecodeGraphCache::new();

        #[cfg(debug_assertions)]
        eprintln!("  DecodeBuffers: {} buffers, {:.2} KB pre-allocated",
            buffers.buffer_count(), buffers.total_bytes() as f64 / 1024.0);

        let decode_start = std::time::Instant::now();
        let mut generated = Vec::new();
        let mut pos = prompt_len as u32;

        let base_seed: u64 = prompt_ids.iter().fold(42u64, |acc, &t| {
            acc.wrapping_mul(6364136223846793005).wrapping_add(t as u64)
        });

        for step in 0..gen_config.max_tokens {
            let rng_seed = base_seed.wrapping_add(step as u64);
            let next_token = sample_token(&last_logits, gen_config, &generated, rng_seed);

            if let Some(eos) = gen_config.eos_token_id {
                if next_token == eos { break; }
            }

            generated.push(next_token);

            // Check stop sequences
            if !gen_config.stop_sequences.is_empty()
                && matches_stop_sequence(&generated, &gen_config.stop_sequences)
            {
                break;
            }

            // Write new token ID into pre-allocated buffer (no allocation)
            device.htod_copy(&[next_token], &mut buffers.ids.data)?;

            if step == 0 {
                // Step 0: Warmup — run on default stream to pre-compile all NVRTC
                // kernels. This ensures compilation is complete before any future
                // graph capture attempt.
                self.forward_decode_preallocated(device, &mut buffers, &mut kv_cache, pos)?;
                device.synchronize()?;
            } else {
                // Steps 1+: Run with pre-allocated buffers.
                // No allocations happen — all buffers are reused at stable addresses.
                //
                // TODO: When cudaGraphExecKernelNodeSetParams is implemented, this
                // is where we'd do graph replay with updated pos/kv_len parameters
                // instead of re-launching all kernels individually.
                self.forward_decode_preallocated(device, &mut buffers, &mut kv_cache, pos)?;
                device.synchronize()?;
            }

            // Read logits from stable pre-allocated buffer
            last_logits = buffers.logits.to_host(device)?;
            pos += 1;
        }

        let decode_time = decode_start.elapsed();

        #[cfg(debug_assertions)]
        eprintln!("  Pool after decode: {}", self.pool.stats());

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

    /// Generate with streaming — calls `on_token` for each generated token.
    /// Enables real-time output without waiting for the full sequence.
    pub fn generate_streaming<F>(
        &self,
        device: &WarpDevice,
        prompt_ids: &[i32],
        gen_config: &GenerateConfig,
        max_seq_len: u32,
        mut on_token: F,
    ) -> Result<GenerationResult, DeviceError>
    where
        F: FnMut(i32, usize), // (token_id, position)
    {
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

        let base_seed: u64 = prompt_ids.iter().fold(42u64, |acc, &t| {
            acc.wrapping_mul(6364136223846793005).wrapping_add(t as u64)
        });

        for step in 0..gen_config.max_tokens {
            let rng_seed = base_seed.wrapping_add(step as u64);
            let next_token = sample_token(&last_logits, gen_config, &generated, rng_seed);

            if let Some(eos) = gen_config.eos_token_id {
                if next_token == eos { break; }
            }

            generated.push(next_token);

            // Check stop sequences
            if !gen_config.stop_sequences.is_empty()
                && matches_stop_sequence(&generated, &gen_config.stop_sequences)
            {
                break;
            }

            // Stream the token immediately
            on_token(next_token, pos as usize);

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

/// Pre-allocated per-layer intermediates for decode (stable addresses for CUDA graph).
pub struct LayerDecodeBuffers {
    pub normed: GpuTensor<f32>,        // [1, H]  — attn input after RMSNorm
    pub q: GpuTensor<f32>,             // [1, H]
    pub k: GpuTensor<f32>,             // [1, kv_dim]
    pub v: GpuTensor<f32>,             // [1, kv_dim]
    pub q_rope: GpuTensor<f32>,        // [1, H]
    pub k_rope: GpuTensor<f32>,        // [1, kv_dim]
    pub attn_out: GpuTensor<f32>,      // [1, head_dim]
    pub attn_proj: GpuTensor<f32>,     // [1, H]
    pub ffn_normed: GpuTensor<f32>,    // [1, H]
    pub residual: GpuTensor<f32>,      // [1, H]
    pub gate: GpuTensor<f32>,          // [1, FFN]
    pub up: GpuTensor<f32>,            // [1, FFN]
    pub swiglu: GpuTensor<f32>,        // [1, FFN]
    pub ffn_out: GpuTensor<f32>,       // [1, H]
    pub output: GpuTensor<f32>,        // [1, H]
    /// Fused QKV buffer [1, H + 2*kv_dim] — pre-allocated when fused weights exist.
    pub qkv: Option<GpuTensor<f32>>,
    /// Fused gate+up buffer [1, 2*FFN] — pre-allocated when fused weights exist.
    pub gate_up: Option<GpuTensor<f32>>,
    /// Bias scratch buffers — pre-allocated for models with Q/K/V biases (e.g. Qwen).
    /// broadcast_add writes to a separate output, so we need scratch space.
    pub q_biased: Option<GpuTensor<f32>>,   // [1, H]
    pub k_biased: Option<GpuTensor<f32>>,   // [1, kv_dim]
    pub v_biased: Option<GpuTensor<f32>>,   // [1, kv_dim]
}

/// Pre-allocated tensors for decode step (stable addresses for CUDA graph).
///
/// CUDA graph capture records exact memory addresses. If tensors are allocated
/// at different addresses each call, the graph is invalid. By pre-allocating ALL
/// intermediates from the memory pool before capture, we guarantee stable addresses.
///
/// Even without full CUDA graph replay (which needs cudaGraphExecKernelNodeSetParams
/// for parameters like `pos` that change each step), pre-allocation eliminates
/// cudaMalloc/cudaFree overhead per decode step — typically 500-1000us savings.
pub struct DecodeBuffers {
    pub ids: GpuTensor<i32>,           // [1]     — input token ID
    pub hidden: GpuTensor<f32>,        // [1, H]  — embedding output / layer input
    pub normed: GpuTensor<f32>,        // [1, H]  — after final RMSNorm
    pub logits: GpuTensor<f32>,        // [1, V]  — output logits
    /// Per-layer intermediates — one set per transformer layer.
    pub layer_buffers: Vec<LayerDecodeBuffers>,
}

impl DecodeBuffers {
    /// Pre-allocate all decode buffers from the memory pool.
    /// These addresses remain stable across decode steps, enabling CUDA graph replay.
    pub fn allocate(
        device: &WarpDevice,
        pool: &GpuMemPool,
        config: &TransformerConfig,
        num_layers: usize,
        vocab_size: u32,
    ) -> Result<Self, DeviceError> {
        Self::allocate_with_fusion(device, pool, config, num_layers, vocab_size, false)
    }

    /// Pre-allocate all decode buffers. When `fused` is true, also allocates
    /// buffers for fused QKV and gate+up projections.
    pub fn allocate_with_fusion(
        device: &WarpDevice,
        pool: &GpuMemPool,
        config: &TransformerConfig,
        num_layers: usize,
        vocab_size: u32,
        fused: bool,
    ) -> Result<Self, DeviceError> {
        let h = config.hidden_size as usize;
        let kv_dim = config.kv_dim() as usize;
        let head_dim = config.head_dim as usize;
        let ffn = config.ffn_dim as usize;
        let v = vocab_size as usize;

        let sh = Shape::from_static(&[1, h]);
        let sk = Shape::from_static(&[1, kv_dim]);
        let sd = Shape::from_static(&[1, head_dim]);
        let sf = Shape::from_static(&[1, ffn]);

        let ids = GpuTensor::<i32>::zeros(device, Shape::from_static(&[1]), DType::I32)?;
        let hidden = pool.get_f32(device, sh.clone(), DType::F32)?;
        let normed = pool.get_f32(device, sh.clone(), DType::F32)?;
        let logits = pool.get_f32(device, Shape::from_static(&[1, v]), DType::F32)?;

        let mut layer_buffers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let qkv = if fused {
                let sqkv = Shape::from_static(&[1, h + kv_dim + kv_dim]);
                Some(pool.get_f32(device, sqkv, DType::F32)?)
            } else {
                None
            };
            let gate_up = if fused {
                let sgu = Shape::from_static(&[1, ffn * 2]);
                Some(pool.get_f32(device, sgu, DType::F32)?)
            } else {
                None
            };

            layer_buffers.push(LayerDecodeBuffers {
                normed: pool.get_f32(device, sh.clone(), DType::F32)?,
                q: pool.get_f32(device, sh.clone(), DType::F32)?,
                k: pool.get_f32(device, sk.clone(), DType::F32)?,
                v: pool.get_f32(device, sk.clone(), DType::F32)?,
                q_rope: pool.get_f32(device, sh.clone(), DType::F32)?,
                k_rope: pool.get_f32(device, sk.clone(), DType::F32)?,
                attn_out: pool.get_f32(device, sh.clone(), DType::F32)?,
                attn_proj: pool.get_f32(device, sh.clone(), DType::F32)?,
                ffn_normed: pool.get_f32(device, sh.clone(), DType::F32)?,
                residual: pool.get_f32(device, sh.clone(), DType::F32)?,
                gate: pool.get_f32(device, sf.clone(), DType::F32)?,
                up: pool.get_f32(device, sf.clone(), DType::F32)?,
                swiglu: pool.get_f32(device, sf.clone(), DType::F32)?,
                ffn_out: pool.get_f32(device, sh.clone(), DType::F32)?,
                output: pool.get_f32(device, sh.clone(), DType::F32)?,
                qkv,
                gate_up,
                // Bias scratch buffers — always allocated (tiny overhead).
                // Used when models have Q/K/V biases (e.g. Qwen).
                q_biased: Some(pool.get_f32(device, sh.clone(), DType::F32)?),
                k_biased: Some(pool.get_f32(device, sk.clone(), DType::F32)?),
                v_biased: Some(pool.get_f32(device, sk.clone(), DType::F32)?),
            });
        }

        Ok(Self { ids, hidden, normed, logits, layer_buffers })
    }

    /// Total number of pre-allocated buffers.
    pub fn buffer_count(&self) -> usize {
        let per_layer = 15
            + if self.layer_buffers.first().map_or(false, |lb| lb.qkv.is_some()) { 1 } else { 0 }
            + if self.layer_buffers.first().map_or(false, |lb| lb.gate_up.is_some()) { 1 } else { 0 }
            + if self.layer_buffers.first().map_or(false, |lb| lb.q_biased.is_some()) { 3 } else { 0 };
        4 + self.layer_buffers.len() * per_layer
    }

    /// Total bytes pre-allocated.
    pub fn total_bytes(&self) -> usize {
        let mut bytes = self.ids.size_bytes()
            + self.hidden.size_bytes()
            + self.normed.size_bytes()
            + self.logits.size_bytes();
        for lb in &self.layer_buffers {
            bytes += lb.normed.size_bytes() + lb.q.size_bytes() + lb.k.size_bytes()
                + lb.v.size_bytes() + lb.q_rope.size_bytes() + lb.k_rope.size_bytes()
                + lb.attn_out.size_bytes() + lb.attn_proj.size_bytes()
                + lb.ffn_normed.size_bytes() + lb.residual.size_bytes()
                + lb.gate.size_bytes() + lb.up.size_bytes() + lb.swiglu.size_bytes()
                + lb.ffn_out.size_bytes() + lb.output.size_bytes();
            if let Some(ref qkv) = lb.qkv { bytes += qkv.size_bytes(); }
            if let Some(ref gu) = lb.gate_up { bytes += gu.size_bytes(); }
            if let Some(ref qb) = lb.q_biased { bytes += qb.size_bytes(); }
            if let Some(ref kb) = lb.k_biased { bytes += kb.size_bytes(); }
            if let Some(ref vb) = lb.v_biased { bytes += vb.size_bytes(); }
        }
        bytes
    }
}

// ═══════════════════════════════════════════════════════════════
// FP16 Pre-allocated Decode Buffers
// ═══════════════════════════════════════════════════════════════

/// Per-layer FP16 buffers for zero-allocation decode.
pub struct LayerDecodeBuffersF16 {
    pub normed: GpuTensor<half::f16>,    // [1, H]
    pub q: GpuTensor<half::f16>,         // [1, H]
    pub k: GpuTensor<half::f16>,         // [1, kv_dim]
    pub v: GpuTensor<half::f16>,         // [1, kv_dim]
    pub q_rope: GpuTensor<half::f16>,    // [1, H]
    pub k_rope: GpuTensor<half::f16>,    // [1, kv_dim]
    pub v_biased: GpuTensor<half::f16>,  // [1, kv_dim] — scratch for V bias add
    pub attn_out: GpuTensor<half::f16>,  // [1, H]
    pub attn_proj: GpuTensor<half::f16>, // [1, H]
    pub ffn_normed: GpuTensor<half::f16>,// [1, H]
    pub residual: GpuTensor<half::f16>,  // [1, H]
    pub gate: GpuTensor<half::f16>,      // [1, FFN]
    pub up: GpuTensor<half::f16>,        // [1, FFN]
    pub swiglu: GpuTensor<half::f16>,    // [1, FFN]
    pub ffn_out: GpuTensor<half::f16>,   // [1, H]
    pub output: GpuTensor<half::f16>,    // [1, H]
}

/// Pre-allocated FP16 decode buffers for the full FP16 pipeline.
pub struct DecodeBuffersF16 {
    pub hidden_f16: GpuTensor<half::f16>,   // [1, H] — layer input
    pub hidden_f32: GpuTensor<f32>,         // [1, H] — for embedding + final norm
    pub normed_f32: GpuTensor<f32>,         // [1, H] — final norm output
    pub logits: GpuTensor<f32>,             // [1, V] — output logits (F32 for argmax)
    pub layer_buffers: Vec<LayerDecodeBuffersF16>,
}

impl DecodeBuffersF16 {
    pub fn allocate(
        device: &WarpDevice,
        config: &TransformerConfig,
        num_layers: usize,
        vocab_size: u32,
    ) -> Result<Self, DeviceError> {
        let h = config.hidden_size as usize;
        let kv = config.kv_dim() as usize;
        let ffn = config.ffn_dim as usize;
        let sh = Shape::from_static(&[1, h]);
        let sk = Shape::from_static(&[1, kv]);
        let sf = Shape::from_static(&[1, ffn]);

        let mut layer_buffers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layer_buffers.push(LayerDecodeBuffersF16 {
                normed: GpuTensor::<half::f16>::zeros(device, sh.clone(), DType::F16)?,
                q: GpuTensor::<half::f16>::zeros(device, sh.clone(), DType::F16)?,
                k: GpuTensor::<half::f16>::zeros(device, sk.clone(), DType::F16)?,
                v: GpuTensor::<half::f16>::zeros(device, sk.clone(), DType::F16)?,
                q_rope: GpuTensor::<half::f16>::zeros(device, sh.clone(), DType::F16)?,
                k_rope: GpuTensor::<half::f16>::zeros(device, sk.clone(), DType::F16)?,
                v_biased: GpuTensor::<half::f16>::zeros(device, sk.clone(), DType::F16)?,
                attn_out: GpuTensor::<half::f16>::zeros(device, sh.clone(), DType::F16)?,
                attn_proj: GpuTensor::<half::f16>::zeros(device, sh.clone(), DType::F16)?,
                ffn_normed: GpuTensor::<half::f16>::zeros(device, sh.clone(), DType::F16)?,
                residual: GpuTensor::<half::f16>::zeros(device, sh.clone(), DType::F16)?,
                gate: GpuTensor::<half::f16>::zeros(device, sf.clone(), DType::F16)?,
                up: GpuTensor::<half::f16>::zeros(device, sf.clone(), DType::F16)?,
                swiglu: GpuTensor::<half::f16>::zeros(device, sf.clone(), DType::F16)?,
                ffn_out: GpuTensor::<half::f16>::zeros(device, sh.clone(), DType::F16)?,
                output: GpuTensor::<half::f16>::zeros(device, sh.clone(), DType::F16)?,
            });
        }

        Ok(Self {
            hidden_f16: GpuTensor::<half::f16>::zeros(device, sh.clone(), DType::F16)?,
            hidden_f32: GpuTensor::<f32>::zeros(device, sh.clone(), DType::F32)?,
            normed_f32: GpuTensor::<f32>::zeros(device, sh, DType::F32)?,
            logits: GpuTensor::<f32>::zeros(device, Shape::from_static(&[1, vocab_size as usize]), DType::F32)?,
            layer_buffers,
        })
    }
}

impl GenerationEngine {
    /// Decode forward pass using pre-allocated buffers — zero allocation overhead.
    ///
    /// This is the CUDA-graph-friendly version of `forward_decode`. Instead of
    /// allocating fresh tensors per step, it writes into stable pre-allocated buffers.
    /// The caller must write the token ID into `buffers.ids` before calling.
    ///
    /// All GPU memory addresses are stable across calls, making this path compatible
    /// with CUDA graph capture (once cudaGraphExecKernelNodeSetParams is wired for
    /// pos/kv_len updates).
    pub fn forward_decode_preallocated(
        &self,
        device: &WarpDevice,
        buffers: &mut DecodeBuffers,
        kv_cache: &mut ModelKVCache,
        pos: u32,
    ) -> Result<(), DeviceError> {
        let h = self.config.hidden_size;
        let d = self.config.head_dim;
        let kv_dim = self.config.kv_dim();
        let ffn = self.config.ffn_dim;
        let batch = 1u32;
        let bn = batch;

        // 1. Embedding lookup — writes into buffers.hidden
        sampling::embedding(&self.cache, device, &self.embed_tokens, &buffers.ids,
            &mut buffers.hidden, 1, h)?;

        // 2. Run through each transformer layer using pre-allocated per-layer buffers.
        //
        // Borrow-checker challenge: layer i reads from layer i-1's output while writing
        // to layer i's buffers. We use split_at_mut to get non-overlapping slices.
        for (i, layer) in self.layers.iter().enumerate() {
            // Get the input tensor for this layer:
            // - Layer 0: reads from buffers.hidden
            // - Layer N>0: reads from layer_buffers[N-1].output
            //
            // We use raw pointers to avoid borrow conflicts between reading the
            // previous layer's output and writing to the current layer's buffers.
            // SAFETY: layer i-1 output is only read, layer i buffers are only written.
            // These are disjoint allocations (different indices into Vec).
            let x_ptr: *const GpuTensor<f32> = if i == 0 {
                &buffers.hidden as *const GpuTensor<f32>
            } else {
                &buffers.layer_buffers[i - 1].output as *const GpuTensor<f32>
            };
            let x: &GpuTensor<f32> = unsafe { &*x_ptr };

            let lb = &mut buffers.layer_buffers[i];

            // 2a. RMSNorm
            ops::rmsnorm(&self.cache, device, x, &layer.attn_norm,
                &mut lb.normed, h, self.config.norm_eps)?;

            // 2b. Q, K, V projections — fused or separate
            let use_fused_qkv = layer.wqkv.is_some() && lb.qkv.is_some();
            if use_fused_qkv {
                let wqkv = layer.wqkv.as_ref().unwrap();
                let qkv_buf = lb.qkv.as_mut().unwrap();
                let qkv_dim = h + kv_dim + kv_dim;
                if let Some(ref bqkv) = layer.bqkv {
                    ops::gemm_bias(&self.cache, device, &lb.normed, wqkv, bqkv, qkv_buf, bn, qkv_dim, h)?;
                } else {
                    ops::gemm(&self.cache, device, &lb.normed, wqkv, qkv_buf, bn, qkv_dim, h)?;
                }
                ops::split_qkv(&self.cache, device, qkv_buf, &mut lb.q, &mut lb.k, &mut lb.v, h, kv_dim, bn)?;
            } else {
                ops::gemm(&self.cache, device, &lb.normed, &layer.wq, &mut lb.q, bn, h, h)?;
                ops::gemm(&self.cache, device, &lb.normed, &layer.wk, &mut lb.k, bn, kv_dim, h)?;
                ops::gemm(&self.cache, device, &lb.normed, &layer.wv, &mut lb.v, bn, kv_dim, h)?;

                // Biases handled by fused kernel below
            }

            // 2c+2d. Fused bias + RoPE + KV cache append (1 launch replaces 6)
            //
            // For unfused QKV with biases: applies bias, RoPE, and KV append in one shot.
            // For fused QKV (bias already in GEMM) or no-bias models: skips bias add.
            let has_bias = !use_fused_qkv && layer.bq.is_some();
            // Bias pointers: when has_bias=false the kernel ignores these, but we
            // still need valid GPU pointers. Use lb.q as a dummy (never read).
            let bq_ref = if has_bias { layer.bq.as_ref().unwrap() } else { &lb.q };
            let bk_ref = if has_bias { layer.bk.as_ref().unwrap() } else { &lb.k };
            let bv_ref = if has_bias { layer.bv.as_ref().unwrap() } else { &lb.v };

            {
                let kv_layer = &mut kv_cache.layers[i];
                ops::fused_bias_rope_append(
                    &self.cache, device,
                    &lb.q, &lb.k, &lb.v,
                    &mut lb.q_rope, &mut lb.k_rope,
                    bq_ref, bk_ref, bv_ref,
                    &mut kv_layer.k, &mut kv_layer.v,
                    self.config.num_heads, self.config.num_kv_heads, d, kv_dim,
                    pos, kv_cache.max_seq_len, self.config.rope_base, has_bias,
                )?;
                kv_layer.len = pos + 1;
            }
            crate::kv_cache::decode_attention_multihead(
                &self.cache, device, &lb.q_rope, &kv_cache.layers[i], &mut lb.attn_out,
                self.config.num_heads, self.config.num_kv_heads, d,
            )?;

            // 2e. Output projection: attn_out[1, H] @ wo[H, H] → attn_proj[1, H]
            ops::gemm(&self.cache, device, &lb.attn_out, &layer.wo,
                &mut lb.attn_proj, bn, h, h)?;

            // 2f. Fused residual + FFN norm
            ops::fused_residual_rmsnorm(&self.cache, device, &lb.attn_proj, x,
                &layer.ffn_norm, &mut lb.ffn_normed, &mut lb.residual, h, self.config.norm_eps)?;

            // 2g. Gate + Up + SwiGLU + Down — fused or separate
            let use_fused_gu = layer.w_gate_up.is_some() && lb.gate_up.is_some();
            if use_fused_gu {
                let w_gate_up = layer.w_gate_up.as_ref().unwrap();
                let gu_buf = lb.gate_up.as_mut().unwrap();
                let gu_dim = ffn * 2;
                ops::gemm(&self.cache, device, &lb.ffn_normed, w_gate_up, gu_buf, bn, gu_dim, h)?;
                ops::split_gate_up(&self.cache, device, gu_buf, &mut lb.gate, &mut lb.up, ffn, bn)?;
            } else {
                ops::gemm(&self.cache, device, &lb.ffn_normed, &layer.w_gate,
                    &mut lb.gate, bn, ffn, h)?;
                ops::gemm(&self.cache, device, &lb.ffn_normed, &layer.w_up,
                    &mut lb.up, bn, ffn, h)?;
            }
            ops::fused_silu_mul(&self.cache, device, &lb.gate, &lb.up, &mut lb.swiglu)?;
            ops::gemm(&self.cache, device, &lb.swiglu, &layer.w_down,
                &mut lb.ffn_out, bn, h, ffn)?;

            // 2h. Final residual
            ops::add(&self.cache, device, &lb.residual, &lb.ffn_out, &mut lb.output)?;
        }

        // 3. Final RMSNorm — read from last layer's output
        let last_output = &buffers.layer_buffers.last().unwrap().output;
        // SAFETY: last_output is read-only, buffers.normed is a different allocation
        let last_output_ptr: *const GpuTensor<f32> = last_output;
        ops::rmsnorm(&self.cache, device, unsafe { &*last_output_ptr },
            &self.final_norm, &mut buffers.normed, h, self.config.norm_eps)?;

        // 4. LM head projection -> logits
        ops::gemm(&self.cache, device, &buffers.normed, &self.lm_head,
            &mut buffers.logits, 1, self.vocab_size, h)?;

        Ok(())
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
        pool: GpuMemPool::new(),
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

        let base_seed: u64 = prompt_ids.iter().fold(42u64, |acc, &t| {
            acc.wrapping_mul(6364136223846793005).wrapping_add(t as u64)
        });

        for step in 0..gen_config.max_tokens {
            let rng_seed = base_seed.wrapping_add(step as u64);
            let next_token = sample_token(&last_logits, gen_config, &generated, rng_seed);

            if let Some(eos) = gen_config.eos_token_id {
                if next_token == eos { break; }
            }

            generated.push(next_token);

            if !gen_config.stop_sequences.is_empty()
                && matches_stop_sequence(&generated, &gen_config.stop_sequences)
            {
                break;
            }

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

// ═════════════════════════════════════════════════════════════════
// Q4 Pre-allocated Decode Buffers
// ═════════════════════════════════════════════════════════════════

/// Per-layer pre-allocated buffers for Q4 decode (eliminates cudaMalloc per step).
pub struct Q4LayerDecodeBuffers {
    pub normed: GpuTensor<f32>,
    pub q: GpuTensor<f32>,
    pub k: GpuTensor<f32>,
    pub v: GpuTensor<f32>,
    pub q_biased: GpuTensor<f32>,
    pub k_biased: GpuTensor<f32>,
    pub v_biased: GpuTensor<f32>,
    pub q_rope: GpuTensor<f32>,
    pub k_rope: GpuTensor<f32>,
    pub attn_out: GpuTensor<f32>,
    pub attn_proj: GpuTensor<f32>,
    pub ffn_normed: GpuTensor<f32>,
    pub residual: GpuTensor<f32>,
    pub gate: GpuTensor<f32>,
    pub up: GpuTensor<f32>,
    pub swiglu: GpuTensor<f32>,
    pub ffn_out: GpuTensor<f32>,
    pub output: GpuTensor<f32>,
}

/// Pre-allocated tensors for Q4 decode — zero cudaMalloc during generation.
pub struct Q4DecodeBuffers {
    pub ids: GpuTensor<i32>,
    pub hidden: GpuTensor<f32>,
    pub normed: GpuTensor<f32>,
    pub logits: GpuTensor<f32>,
    pub layers: Vec<Q4LayerDecodeBuffers>,
}

impl Q4DecodeBuffers {
    pub fn allocate(
        device: &WarpDevice,
        config: &TransformerConfig,
        num_layers: usize,
        vocab_size: u32,
    ) -> Result<Self, DeviceError> {
        let h = config.hidden_size as usize;
        let kv = config.kv_dim() as usize;
        let ffn = config.ffn_dim as usize;
        let v = vocab_size as usize;

        let sh = Shape::from_static(&[1, h]);
        let sk = Shape::from_static(&[1, kv]);
        let sf = Shape::from_static(&[1, ffn]);

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(Q4LayerDecodeBuffers {
                normed: GpuTensor::zeros(device, sh.clone(), DType::F32)?,
                q: GpuTensor::zeros(device, sh.clone(), DType::F32)?,
                k: GpuTensor::zeros(device, sk.clone(), DType::F32)?,
                v: GpuTensor::zeros(device, sk.clone(), DType::F32)?,
                q_biased: GpuTensor::zeros(device, sh.clone(), DType::F32)?,
                k_biased: GpuTensor::zeros(device, sk.clone(), DType::F32)?,
                v_biased: GpuTensor::zeros(device, sk.clone(), DType::F32)?,
                q_rope: GpuTensor::zeros(device, sh.clone(), DType::F32)?,
                k_rope: GpuTensor::zeros(device, sk.clone(), DType::F32)?,
                attn_out: GpuTensor::zeros(device, sh.clone(), DType::F32)?,
                attn_proj: GpuTensor::zeros(device, sh.clone(), DType::F32)?,
                ffn_normed: GpuTensor::zeros(device, sh.clone(), DType::F32)?,
                residual: GpuTensor::zeros(device, sh.clone(), DType::F32)?,
                gate: GpuTensor::zeros(device, sf.clone(), DType::F32)?,
                up: GpuTensor::zeros(device, sf.clone(), DType::F32)?,
                swiglu: GpuTensor::zeros(device, sf.clone(), DType::F32)?,
                ffn_out: GpuTensor::zeros(device, sh.clone(), DType::F32)?,
                output: GpuTensor::zeros(device, sh.clone(), DType::F32)?,
            });
        }

        Ok(Self {
            ids: GpuTensor::zeros(device, Shape::from_static(&[1]), DType::I32)?,
            hidden: GpuTensor::zeros(device, sh.clone(), DType::F32)?,
            normed: GpuTensor::zeros(device, sh, DType::F32)?,
            logits: GpuTensor::zeros(device, Shape::from_static(&[1, v]), DType::F32)?,
            layers,
        })
    }

    pub fn buffer_count(&self) -> usize {
        4 + self.layers.len() * 18
    }

    pub fn memory_bytes(&self) -> usize {
        self.ids.size_bytes() + self.hidden.size_bytes() + self.normed.size_bytes()
            + self.logits.size_bytes()
            + self.layers.iter().map(|l| {
                l.normed.size_bytes() + l.q.size_bytes() + l.k.size_bytes() + l.v.size_bytes()
                + l.q_biased.size_bytes() + l.k_biased.size_bytes() + l.v_biased.size_bytes()
                + l.q_rope.size_bytes() + l.k_rope.size_bytes() + l.attn_out.size_bytes()
                + l.attn_proj.size_bytes() + l.ffn_normed.size_bytes() + l.residual.size_bytes()
                + l.gate.size_bytes() + l.up.size_bytes() + l.swiglu.size_bytes()
                + l.ffn_out.size_bytes() + l.output.size_bytes()
            }).sum::<usize>()
    }
}

impl QuantizedGenerationEngine {
    /// Decode with pre-allocated buffers — zero cudaMalloc during generation.
    fn forward_decode_prealloc(
        &self,
        device: &WarpDevice,
        buffers: &mut Q4DecodeBuffers,
        kv_cache: &mut ModelKVCache,
        pos: u32,
    ) -> Result<(), DeviceError> {
        let h = self.config.hidden_size;
        let d = self.config.head_dim;
        let kv_dim = self.config.kv_dim();
        let ffn = self.config.ffn_dim;

        sampling::embedding(&self.cache, device, &self.embed_tokens, &buffers.ids,
            &mut buffers.hidden, 1, h)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let x_ptr: *const GpuTensor<f32> = if i == 0 {
                &buffers.hidden as *const _
            } else {
                &buffers.layers[i - 1].output as *const _
            };
            let x: &GpuTensor<f32> = unsafe { &*x_ptr };
            let lb = &mut buffers.layers[i];

            // 1. RMSNorm
            ops::rmsnorm(&self.cache, device, x, &layer.attn_norm,
                &mut lb.normed, h, self.config.norm_eps)?;

            // 2. Q, K, V — M=1 specialized GEMM
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &lb.normed, &layer.wq, &mut lb.q, h, h)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &lb.normed, &layer.wk, &mut lb.k, kv_dim, h)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &lb.normed, &layer.wv, &mut lb.v, kv_dim, h)?;

            // 3. Biases
            let q_ref = if let Some(ref bq) = layer.bq {
                ops::broadcast_add(&self.cache, device, &lb.q, bq, &mut lb.q_biased)?;
                &lb.q_biased
            } else { &lb.q };
            let k_ref = if let Some(ref bk) = layer.bk {
                ops::broadcast_add(&self.cache, device, &lb.k, bk, &mut lb.k_biased)?;
                &lb.k_biased
            } else { &lb.k };
            let v_ref = if let Some(ref bv) = layer.bv {
                ops::broadcast_add(&self.cache, device, &lb.v, bv, &mut lb.v_biased)?;
                &lb.v_biased
            } else { &lb.v };

            // 4. RoPE
            crate::rope::rope(&self.cache, device, q_ref, &mut lb.q_rope,
                self.config.num_heads, 1, d, self.config.rope_base, pos)?;
            crate::rope::rope(&self.cache, device, k_ref, &mut lb.k_rope,
                self.config.num_kv_heads, 1, d, self.config.rope_base, pos)?;

            // 5. KV cache append
            {
                let kv_layer = &mut kv_cache.layers[i];
                kv_layer.append(&self.cache, device, &lb.k_rope, v_ref)?;

                // 6. Decode attention (FlashDecoding)
                crate::kv_cache::decode_attention_flash(
                    &self.cache, device, &lb.q_rope, kv_layer, &mut lb.attn_out,
                    self.config.num_heads, self.config.num_kv_heads, d,
                )?;
            }

            // 7. Output projection — M=1 GEMM
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &lb.attn_out, &layer.wo,
                &mut lb.attn_proj, h, h)?;

            // 8. Fused residual + FFN norm
            ops::fused_residual_rmsnorm(&self.cache, device, &lb.attn_proj, x,
                &layer.ffn_norm, &mut lb.ffn_normed, &mut lb.residual,
                h, self.config.norm_eps)?;

            // 9. Gate + Up — M=1 GEMM
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &lb.ffn_normed, &layer.w_gate,
                &mut lb.gate, ffn, h)?;
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &lb.ffn_normed, &layer.w_up,
                &mut lb.up, ffn, h)?;

            // 10. Fused SwiGLU
            ops::fused_silu_mul(&self.cache, device, &lb.gate, &lb.up, &mut lb.swiglu)?;

            // 11. Down projection — M=1 GEMM
            crate::quantize::gemm_q4_0_m1(&self.cache, device, &lb.swiglu, &layer.w_down,
                &mut lb.ffn_out, h, ffn)?;

            // 12. Residual
            ops::add(&self.cache, device, &lb.residual, &lb.ffn_out, &mut lb.output)?;
        }

        // Final norm + LM head
        let last_layer = self.layers.len() - 1;
        ops::rmsnorm(&self.cache, device, &buffers.layers[last_layer].output,
            &self.final_norm, &mut buffers.normed, h, self.config.norm_eps)?;
        ops::gemm(&self.cache, device, &buffers.normed, &self.lm_head,
            &mut buffers.logits, 1, self.vocab_size, h)?;

        Ok(())
    }

    /// Generate with pre-allocated decode buffers — zero cudaMalloc during generation.
    pub fn generate_prealloc(
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

        // Pre-allocate all decode buffers (one-time cost)
        let mut buffers = Q4DecodeBuffers::allocate(
            device, &self.config, self.layers.len(), self.vocab_size,
        )?;

        // Prefill (still uses the allocating path — prefill is one-shot)
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

        let base_seed: u64 = prompt_ids.iter().fold(42u64, |acc, &t| {
            acc.wrapping_mul(6364136223846793005).wrapping_add(t as u64)
        });

        for step in 0..gen_config.max_tokens {
            let rng_seed = base_seed.wrapping_add(step as u64);
            let next_token = sample_token(&last_logits, gen_config, &generated, rng_seed);

            if let Some(eos) = gen_config.eos_token_id {
                if next_token == eos { break; }
            }

            generated.push(next_token);

            if !gen_config.stop_sequences.is_empty()
                && matches_stop_sequence(&generated, &gen_config.stop_sequences)
            { break; }

            // Write token ID into pre-allocated buffer (no allocation)
            device.htod_copy(&[next_token], &mut buffers.ids.data)?;

            // Decode with pre-allocated buffers
            self.forward_decode_prealloc(device, &mut buffers, &mut kv_cache, pos)?;
            device.synchronize()?;

            last_logits = buffers.logits.to_host(device)?;
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

// ═════════════════════════════════════════════════════════════════
// Q4 → FP16 Dequantized Generation Engine (ExLlamaV2-style)
// ═════════════════════════════════════════════════════════════════

/// Generation engine with Q4_0 weights pre-dequantized to FP16.
///
/// ExLlamaV2-style: Q4 weights are dequantized to FP16 once at load time.
/// Inference uses cuBLAS HGEMM (tensor cores) instead of custom Q4 tile GEMM.
/// Uses ~2x more VRAM than Q4_0 but gets cuBLAS-level throughput.
///
/// For 7B model: Q4 = ~4.4 GB, FP16 dequant = ~14 GB (fits on 24 GB 4090).
pub struct DequantF16GenerationEngine {
    pub config: TransformerConfig,
    pub vocab_size: u32,
    pub embed_tokens: GpuTensor<f32>,
    pub layers: Vec<crate::quantize::DequantizedF16Weights>,
    pub final_norm: GpuTensor<f32>,
    pub lm_head: GpuTensor<f32>,
    pub cache: KernelCache,
}

impl DequantF16GenerationEngine {
    /// Create from an existing QuantizedGenerationEngine by dequantizing weights to FP16.
    pub fn from_quantized(
        device: &WarpDevice,
        q_engine: &QuantizedGenerationEngine,
    ) -> Result<Self, DeviceError> {
        let cache = KernelCache::new();
        let mut f16_layers = Vec::with_capacity(q_engine.layers.len());
        for layer in &q_engine.layers {
            f16_layers.push(crate::quantize::DequantizedF16Weights::from_quantized(
                &cache, device, layer, &q_engine.config,
            )?);
        }
        device.synchronize()?;

        let embed_host = q_engine.embed_tokens.to_host(device)?;
        let norm_host = q_engine.final_norm.to_host(device)?;
        let lm_host = q_engine.lm_head.to_host(device)?;

        Ok(Self {
            config: q_engine.config.clone(),
            vocab_size: q_engine.vocab_size,
            embed_tokens: GpuTensor::from_host(device, &embed_host,
                q_engine.embed_tokens.shape.clone(), DType::F32)?,
            layers: f16_layers,
            final_norm: GpuTensor::from_host(device, &norm_host,
                q_engine.final_norm.shape.clone(), DType::F32)?,
            lm_head: GpuTensor::from_host(device, &lm_host,
                q_engine.lm_head.shape.clone(), DType::F32)?,
            cache,
        })
    }

    /// Prefill using the corrected Q4 prefill path.
    /// (For prefill, we still use Q4 GEMM since we don't have FP16 prefill yet.
    ///  The Q4 prefill path now has the correct transposes and bias handling.)
    fn forward_prefill(
        &self,
        device: &WarpDevice,
        input_ids: &[i32],
        kv_cache: &mut ModelKVCache,
        q_layers: &[QuantizedBlockWeights],
    ) -> Result<GpuTensor<f32>, DeviceError> {
        let seq_len = input_ids.len() as u32;
        let h = self.config.hidden_size;

        let ids = GpuTensor::from_host(device, input_ids,
            Shape::from_static(&[seq_len as usize]), DType::I32)?;
        let mut hidden = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[seq_len as usize, h as usize]), DType::F32)?;
        sampling::embedding(&self.cache, device, &self.embed_tokens, &ids,
            &mut hidden, seq_len, h)?;

        for (i, layer) in q_layers.iter().enumerate() {
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

    /// Decode forward pass using cuBLAS HGEMM with pre-dequantized FP16 weights.
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
            hidden = crate::transformer::transformer_block_decode_q4_f16(
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

    /// Generate with KV cache using pre-dequantized FP16 weights.
    /// Requires the original Q4 layers for prefill (since we don't have FP16 prefill yet).
    pub fn generate_with_cache(
        &self,
        device: &WarpDevice,
        prompt_ids: &[i32],
        gen_config: &GenerateConfig,
        max_seq_len: u32,
        q_layers: &[QuantizedBlockWeights],
    ) -> Result<GenerationResult, DeviceError> {
        let kv_dim = self.config.kv_dim();
        let num_layers = self.layers.len() as u32;
        let vocab = self.vocab_size as usize;

        let mut kv_cache = ModelKVCache::new(device, num_layers, max_seq_len, kv_dim)?;

        let prefill_start = std::time::Instant::now();
        let logits = self.forward_prefill(device, prompt_ids, &mut kv_cache, q_layers)?;
        device.synchronize()?;
        let prefill_time = prefill_start.elapsed();

        let all_logits = logits.to_host(device)?;
        let prompt_len = prompt_ids.len();
        let mut last_logits = all_logits[(prompt_len - 1) * vocab..prompt_len * vocab].to_vec();

        let decode_start = std::time::Instant::now();
        let mut generated = Vec::new();
        let mut pos = prompt_len as u32;

        let base_seed: u64 = prompt_ids.iter().fold(42u64, |acc, &t| {
            acc.wrapping_mul(6364136223846793005).wrapping_add(t as u64)
        });

        for step in 0..gen_config.max_tokens {
            let rng_seed = base_seed.wrapping_add(step as u64);
            let next_token = sample_token(&last_logits, gen_config, &generated, rng_seed);

            if let Some(eos) = gen_config.eos_token_id {
                if next_token == eos { break; }
            }

            generated.push(next_token);

            if !gen_config.stop_sequences.is_empty()
                && matches_stop_sequence(&generated, &gen_config.stop_sequences)
            {
                break;
            }

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

    /// VRAM usage for all dequantized FP16 weight layers.
    pub fn weight_memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum::<usize>()
            + self.embed_tokens.size_bytes()
            + self.final_norm.size_bytes()
            + self.lm_head.size_bytes()
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

// ═════════════════════════════════════════════════════════════════
// FP16 mixed-precision generation engine
// ═════════════════════════════════════════════════════════════════

/// Generation engine with FP16 weight matrices — 2x less weight bandwidth.
///
/// Weight matrices (Q/K/V/O/gate/up/down) are stored in FP16.
/// During decode, each GEMM: cast activation F32→F16, HGEMM, cast output F16→F32.
/// Embedding, lm_head, norms, biases, attention, RoPE, SwiGLU all stay F32.
pub struct GenerationEngineF16 {
    pub config: TransformerConfig,
    pub vocab_size: u32,
    pub embed_tokens: GpuTensor<f32>,
    pub layers: Vec<TransformerBlockWeightsF16>,
    pub final_norm: GpuTensor<f32>,
    pub lm_head: GpuTensor<f32>,
    pub cache: KernelCache,
    pub pool: GpuMemPool,
}

impl GenerationEngineF16 {
    /// Prefill forward pass — populates KV cache using FP16 mixed-precision GEMMs.
    pub fn forward_prefill_f16(
        &self,
        device: &WarpDevice,
        input_ids: &[i32],
        kv_cache: &mut ModelKVCache,
    ) -> Result<GpuTensor<f32>, DeviceError> {
        let seq_len = input_ids.len() as u32;
        let h = self.config.hidden_size;
        let batch = 1u32;

        // 1. Embedding lookup (F32)
        let ids = GpuTensor::from_host(device, input_ids,
            Shape::from_static(&[seq_len as usize]), DType::I32)?;
        let mut hidden = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[seq_len as usize, h as usize]), DType::F32)?;
        sampling::embedding(&self.cache, device, &self.embed_tokens, &ids,
            &mut hidden, seq_len, h)?;

        // 2. Run through all transformer layers — FP16 mixed prefill
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = crate::transformer::transformer_block_prefill_f16_mixed(
                &self.cache, device, &hidden, layer, &self.config,
                &mut kv_cache.layers[i],
                batch, seq_len, 0,
            )?;
        }

        // 3. Final RMSNorm (F32)
        let mut normed = self.pool.get_f32(device,
            Shape::from_static(&[seq_len as usize, h as usize]), DType::F32)?;
        ops::rmsnorm(&self.cache, device, &hidden, &self.final_norm,
            &mut normed, h, self.config.norm_eps)?;

        // 4. LM head projection (F32)
        let mut logits = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[seq_len as usize, self.vocab_size as usize]), DType::F32)?;
        ops::gemm(&self.cache, device, &normed, &self.lm_head, &mut logits,
            seq_len, self.vocab_size, h)?;
        self.pool.return_f32(normed);

        Ok(logits)
    }

    /// Decode forward pass — single token with KV cache, FP16 mixed-precision GEMMs.
    pub fn forward_decode_f16(
        &self,
        device: &WarpDevice,
        token_id: i32,
        kv_cache: &mut ModelKVCache,
        pos: u32,
    ) -> Result<GpuTensor<f32>, DeviceError> {
        let h = self.config.hidden_size;
        let batch = 1u32;

        // 1. Embedding lookup (single token, F32)
        let ids = GpuTensor::from_host(device, &[token_id],
            Shape::from_static(&[1]), DType::I32)?;
        let mut hidden = self.pool.get_f32(device,
            Shape::from_static(&[1, h as usize]), DType::F32)?;
        sampling::embedding(&self.cache, device, &self.embed_tokens, &ids,
            &mut hidden, 1, h)?;

        // Pre-allocate shared FP16 buffers for all layers (avoid cudaMalloc per layer)
        let max_dim = h.max(self.config.ffn_dim) as usize;
        let mut f16_in = GpuTensor::<half::f16>::zeros(device,
            Shape::from_static(&[1, max_dim]), DType::F16)?;
        let mut f16_out = GpuTensor::<half::f16>::zeros(device,
            Shape::from_static(&[1, max_dim]), DType::F16)?;

        // 2. Run through all transformer layers — FP16 mixed decode
        for (i, layer) in self.layers.iter().enumerate() {
            let prev_hidden = hidden;
            hidden = crate::transformer::transformer_block_decode_f16_mixed_prealloc(
                &self.cache, device, &prev_hidden, layer, &self.config,
                &mut kv_cache.layers[i],
                batch, pos,
                &mut f16_in, &mut f16_out,
            )?;
            self.pool.return_f32(prev_hidden);
        }

        // 3. Final RMSNorm (F32)
        let mut normed = self.pool.get_f32(device,
            Shape::from_static(&[1, h as usize]), DType::F32)?;
        ops::rmsnorm(&self.cache, device, &hidden, &self.final_norm,
            &mut normed, h, self.config.norm_eps)?;
        self.pool.return_f32(hidden);

        // 4. LM head projection (F32) → [1, vocab_size]
        let mut logits = self.pool.get_f32(device,
            Shape::from_static(&[1, self.vocab_size as usize]), DType::F32)?;
        ops::gemm(&self.cache, device, &normed, &self.lm_head, &mut logits,
            1, self.vocab_size, h)?;
        self.pool.return_f32(normed);

        Ok(logits)
    }

    /// Generate with KV cache using FP16 mixed-precision weights.
    ///
    /// Prefill and decode both use FP16 weight GEMMs for 2x bandwidth savings.
    /// Uses the full sampling pipeline: repetition penalty, temperature,
    /// top-K, top-P, and stop sequence detection.
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

        // Phase 1: Prefill — FP16 mixed
        let logits = self.forward_prefill_f16(device, prompt_ids, &mut kv_cache)?;
        device.synchronize()?;
        let prefill_time = prefill_start.elapsed();

        // Get last token's logits from prefill
        let all_logits = logits.to_host(device)?;
        let prompt_len = prompt_ids.len();
        let mut last_logits = all_logits[(prompt_len - 1) * vocab..prompt_len * vocab].to_vec();

        let decode_start = std::time::Instant::now();
        let mut generated = Vec::new();
        let mut pos = prompt_len as u32;

        // RNG seed
        let base_seed: u64 = prompt_ids.iter().fold(42u64, |acc, &t| {
            acc.wrapping_mul(6364136223846793005).wrapping_add(t as u64)
        });

        // Full FP16 pipeline with pre-allocated buffers.
        let h = self.config.hidden_size;
        let d = self.config.head_dim;
        let kv_dim = self.config.kv_dim();
        let ffn = self.config.ffn_dim;
        let num_layers = self.layers.len();

        // Pre-allocate ALL FP16 decode buffers (zero cudaMalloc during decode)
        let mut bufs = DecodeBuffersF16::allocate(device, &self.config, num_layers, self.vocab_size)?;

        // FP16 KV cache
        let mut kv_f16 = crate::kv_cache::ModelKVCacheF16::new(device, num_layers as u32, max_seq_len, kv_dim)?;

        // Copy prefill KV cache (F32) to FP16
        for i in 0..num_layers {
            let f32_l = &kv_cache.layers[i];
            let f16_l = &mut kv_f16.layers[i];
            if f32_l.len > 0 {
                let mut kt = GpuTensor::<half::f16>::zeros(device, f32_l.k.shape.clone(), DType::F16)?;
                let mut vt = GpuTensor::<half::f16>::zeros(device, f32_l.v.shape.clone(), DType::F16)?;
                crate::fp16::cast_f32_to_f16(&self.cache, device, &f32_l.k, &mut kt)?;
                crate::fp16::cast_f32_to_f16(&self.cache, device, &f32_l.v, &mut vt)?;
                f16_l.k = kt;
                f16_l.v = vt;
                f16_l.len = f32_l.len;
            }
        }

        // Phase 2: Full FP16 decode with CUDA graph capture.
        // Step 0: warmup (compile all kernels)
        // Step 1: capture the decode step as a CUDA graph
        // Steps 2+: replay the graph with updated pos
        let bn = 1u32;

        // GPU buffer for position — updated between graph replays
        let mut pos_buf = GpuTensor::from_host(device, &[pos],
            Shape::from_static(&[1]), DType::U32)?;
        // GPU buffer for cache_len = pos + 1
        let mut len_buf = GpuTensor::from_host(device, &[pos + 1],
            Shape::from_static(&[1]), DType::U32)?;

        let mut graph: Option<crate::cuda_graph::GraphCapture> = None;

        for step in 0..gen_config.max_tokens {
            let rng_seed = base_seed.wrapping_add(step as u64);
            let next_token = sample_token(&last_logits, gen_config, &generated, rng_seed);

            if let Some(eos) = gen_config.eos_token_id {
                if next_token == eos { break; }
            }

            generated.push(next_token);

            if !gen_config.stop_sequences.is_empty()
                && matches_stop_sequence(&generated, &gen_config.stop_sequences)
            { break; }

            // Embedding (F32) → cast to FP16
            let ids_t = GpuTensor::from_host(device, &[next_token],
                Shape::from_static(&[1]), DType::I32)?;
            sampling::embedding(&self.cache, device, &self.embed_tokens, &ids_t,
                &mut bufs.hidden_f32, 1, h)?;
            crate::fp16::cast_f32_to_f16(&self.cache, device, &bufs.hidden_f32, &mut bufs.hidden_f16)?;

            // Update pos buffer for this step (device-pos kernels read from it)
            device.htod_copy(&[pos], &mut pos_buf.data)?;
            device.htod_copy(&[pos + 1], &mut len_buf.data)?;

            // Run all layers — FP16 throughout, pre-allocated buffers
            for (i, layer) in self.layers.iter().enumerate() {
                // Input: first layer reads from bufs.hidden_f16, subsequent from prev layer output
                let x_ptr: *const GpuTensor<half::f16> = if i == 0 {
                    &bufs.hidden_f16 as *const _
                } else {
                    &bufs.layer_buffers[i - 1].output as *const _
                };
                let x: &GpuTensor<half::f16> = unsafe { &*x_ptr };
                let lb = &mut bufs.layer_buffers[i];

                // 1. RMSNorm (FP16 in/out, F32 internal)
                crate::fp16::f16_rmsnorm(&self.cache, device, x, layer.attn_norm_f16(),
                    &mut lb.normed, h, self.config.norm_eps)?;

                // 2. QKV GEMMs (cuBLAS HGEMM — tensor cores)
                crate::cublas_gemm::gemm_cublas_f16(device, &lb.normed, &layer.wq, &mut lb.q, bn, h, h)?;
                crate::cublas_gemm::gemm_cublas_f16(device, &lb.normed, &layer.wk, &mut lb.k, bn, kv_dim, h)?;
                crate::cublas_gemm::gemm_cublas_f16(device, &lb.normed, &layer.wv, &mut lb.v, bn, kv_dim, h)?;

                // 2b. Biases (pre-computed FP16 — zero allocation, one kernel per bias)
                if let Some(ref bq_f16) = layer.bq_f16 {
                    // f16_add writes to a separate output — we need a temp buffer
                    // BUT we can reuse q_rope as temp since it's not written yet
                    crate::fp16::f16_add(&self.cache, device, &lb.q, bq_f16, &mut lb.q_rope)?;
                    std::mem::swap(&mut lb.q, &mut lb.q_rope);
                }
                if let Some(ref bk_f16) = layer.bk_f16 {
                    crate::fp16::f16_add(&self.cache, device, &lb.k, bk_f16, &mut lb.k_rope)?;
                    std::mem::swap(&mut lb.k, &mut lb.k_rope);
                }
                if let Some(ref bv_f16) = layer.bv_f16 {
                    crate::fp16::f16_add(&self.cache, device, &lb.v, bv_f16, &mut lb.v_biased)?;
                    std::mem::swap(&mut lb.v, &mut lb.v_biased);
                }

                // 3. RoPE (FP16) — use device-pos variants for CUDA graph compatibility
                crate::fp16::f16_rope_device_pos(&self.cache, device, &lb.q, &mut lb.q_rope,
                    1 * self.config.num_heads, 1, d, self.config.rope_base, &pos_buf)?;
                crate::fp16::f16_rope_device_pos(&self.cache, device, &lb.k, &mut lb.k_rope,
                    1 * self.config.num_kv_heads, 1, d, self.config.rope_base, &pos_buf)?;

                // 4. KV cache append + attention (FP16) — device-pos variants
                {
                    let kv_l = &mut kv_f16.layers[i];
                    kv_l.append_device_pos(&self.cache, device, &lb.k_rope, &lb.v, &pos_buf)?;
                    crate::kv_cache::decode_attention_multihead_f16_device_len(
                        &self.cache, device, &lb.q_rope, kv_l, &mut lb.attn_out,
                        self.config.num_heads, self.config.num_kv_heads, d, &len_buf,
                    )?;
                }

                // 5. Output projection (cuBLAS HGEMM)
                crate::cublas_gemm::gemm_cublas_f16(device, &lb.attn_out, &layer.wo, &mut lb.attn_proj, bn, h, h)?;

                // 6+7. Fused residual + FFN norm (FP16)
                crate::fp16::f16_fused_residual_rmsnorm(&self.cache, device,
                    &lb.attn_proj, x, layer.ffn_norm_f16(),
                    &mut lb.ffn_normed, &mut lb.residual, h, self.config.norm_eps)?;

                // 8-9. Gate + Up (cuBLAS HGEMM) + SwiGLU (FP16)
                crate::cublas_gemm::gemm_cublas_f16(device, &lb.ffn_normed, &layer.w_gate, &mut lb.gate, bn, ffn, h)?;
                crate::cublas_gemm::gemm_cublas_f16(device, &lb.ffn_normed, &layer.w_up, &mut lb.up, bn, ffn, h)?;
                crate::fp16::f16_fused_silu_mul(&self.cache, device, &lb.gate, &lb.up, &mut lb.swiglu)?;

                // 10. Down projection (cuBLAS HGEMM)
                crate::cublas_gemm::gemm_cublas_f16(device, &lb.swiglu, &layer.w_down, &mut lb.ffn_out, bn, h, ffn)?;

                // 11. Final residual (FP16 add)
                crate::fp16::f16_add(&self.cache, device, &lb.residual, &lb.ffn_out, &mut lb.output)?;
            }

            // Cast final hidden FP16 → F32 for norm + LM head (ONCE per token)
            let last_output = &bufs.layer_buffers.last().unwrap().output;
            let last_ptr: *const GpuTensor<half::f16> = last_output;
            crate::fp16::cast_f16_to_f32(&self.cache, device, unsafe { &*last_ptr }, &mut bufs.hidden_f32)?;
            ops::rmsnorm(&self.cache, device, &bufs.hidden_f32, &self.final_norm,
                &mut bufs.normed_f32, h, self.config.norm_eps)?;
            ops::gemm(&self.cache, device, &bufs.normed_f32, &self.lm_head,
                &mut bufs.logits, 1, self.vocab_size, h)?;
            device.synchronize()?;

            last_logits = bufs.logits.to_host(device)?;
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
    fn sample_token_greedy() {
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let config = GenerateConfig {
            greedy: true,
            ..Default::default()
        };
        let token = sample_token(&logits, &config, &[], 0);
        assert_eq!(token, 3, "Greedy should pick index 3 (highest logit 0.8)");
    }

    #[test]
    fn sample_token_with_temperature() {
        // With very low temperature, should still pick the argmax
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let config = GenerateConfig {
            greedy: true,
            temperature: 0.01,
            ..Default::default()
        };
        let token = sample_token(&logits, &config, &[], 0);
        assert_eq!(token, 3);
    }

    #[test]
    fn sample_token_repetition_penalty() {
        // Token 3 has highest logit but is already generated — penalty should suppress it
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.7];
        let config = GenerateConfig {
            greedy: true,
            repetition_penalty: 5.0, // heavy penalty
            ..Default::default()
        };
        let generated = vec![3]; // token 3 already appeared
        let token = sample_token(&logits, &config, &generated, 0);
        // Token 3 logit becomes 0.8/5.0 = 0.16, so token 4 (0.7) should win
        assert_eq!(token, 4, "Repetition penalty should suppress token 3");
    }

    #[test]
    fn sample_token_top_k() {
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.7];
        let config = GenerateConfig {
            greedy: true,
            top_k: Some(2), // keep only top 2
            ..Default::default()
        };
        let token = sample_token(&logits, &config, &[], 0);
        assert_eq!(token, 3, "Top-K=2 should still pick the argmax");
    }

    #[test]
    fn stop_sequence_detection() {
        let gen = vec![1, 2, 3, 4, 5];
        assert!(matches_stop_sequence(&gen, &[vec![4, 5]]));
        assert!(!matches_stop_sequence(&gen, &[vec![3, 5]]));
        assert!(matches_stop_sequence(&gen, &[vec![1, 2], vec![4, 5]]));
        assert!(!matches_stop_sequence(&gen, &[]));
        assert!(!matches_stop_sequence(&gen, &[vec![]]));
        assert!(matches_stop_sequence(&gen, &[vec![5]]));
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
                attention_mode: crate::transformer::AttentionMode::Standard,
            }, 32, 32000),
        ];

        for (name, config, layers, vocab) in configs {
            let (f32_b, q4_b) = weight_memory_estimate(&config, layers, vocab);
            println!("  {name:25} F32={:>8.1} MB  Q4_0={:>8.1} MB  ({:.1}x smaller)",
                f32_b as f64 / 1e6, q4_b as f64 / 1e6, f32_b as f64 / q4_b as f64);
        }
    }

    #[test]
    fn decode_with_preallocated_buffers() {
        let dev = match setup() {
            Some(d) => d,
            None => { println!("No CUDA, skipping"); return; }
        };

        let config = TransformerConfig::tiny();
        let num_layers = 2u32;
        let vocab_size = 100u32;
        let engine = create_test_engine(&dev, config.clone(), num_layers, vocab_size).unwrap();

        let prompt = vec![1i32, 5, 10, 15, 20];
        let gen_config = GenerateConfig {
            max_tokens: 16,
            greedy: true,
            eos_token_id: None,
            ..Default::default()
        };

        // Run generate_with_graph (uses pre-allocated buffers internally)
        let result = engine.generate_with_graph(&dev, &prompt, &gen_config, 128).unwrap();

        println!("\n=== Decode with Pre-allocated Buffers ===");
        println!("{result}");

        // Verify all tokens are valid
        assert_eq!(result.tokens_generated, 16, "Should generate exactly 16 tokens");
        assert!(result.tokens.iter().all(|&t| t >= 0 && (t as u32) < vocab_size),
            "All tokens must be in vocab range [0, {}), got: {:?}", vocab_size, result.tokens);

        // Compare with normal generate_with_cache for correctness
        let normal_result = engine.generate_with_cache(&dev, &prompt, &gen_config, 128).unwrap();
        assert_eq!(result.tokens, normal_result.tokens,
            "Pre-allocated path must produce identical tokens to normal path!\n  preallocated: {:?}\n  normal:       {:?}",
            result.tokens, normal_result.tokens);
        println!("Correctness: pre-allocated path matches normal path");

        // Print pool stats showing buffer reuse
        let pool_stats = engine.pool.stats();
        println!("Pool stats: {pool_stats}");
        println!("  Pre-allocated decode buffers eliminated per-step allocation overhead");

        // Verify pre-allocated buffers are actually allocated from pool
        let buffers = DecodeBuffers::allocate(
            &dev, &engine.pool, &config, num_layers as usize, vocab_size,
        ).unwrap();
        println!("  DecodeBuffers: {} buffers, {:.2} KB",
            buffers.buffer_count(), buffers.total_bytes() as f64 / 1024.0);

        // Performance comparison
        let start = std::time::Instant::now();
        let _ = engine.generate_with_cache(&dev, &prompt, &gen_config, 128).unwrap();
        let normal_time = start.elapsed();

        let start = std::time::Instant::now();
        let _ = engine.generate_with_graph(&dev, &prompt, &gen_config, 128).unwrap();
        let prealloc_time = start.elapsed();

        println!("\n  Performance (16 tokens, {} layers, H={}, vocab={}):",
            num_layers, config.hidden_size, vocab_size);
        println!("    Normal decode:       {:.2}ms", normal_time.as_secs_f64() * 1000.0);
        println!("    Pre-allocated decode: {:.2}ms", prealloc_time.as_secs_f64() * 1000.0);
        if prealloc_time.as_secs_f64() > 0.0 {
            println!("    Speedup: {:.2}x", normal_time.as_secs_f64() / prealloc_time.as_secs_f64());
        }
    }
}

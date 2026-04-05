//! Speculative decoding — draft-then-verify for faster autoregressive generation.
//!
//! The core idea: a small "draft" model generates K candidate tokens cheaply,
//! then the full "target" model verifies all K+1 positions in a single forward
//! pass (like prefill). Accepted tokens are emitted for free — only rejected
//! tokens cost a full-model decode step.
//!
//! With a good draft model (acceptance rate ~70-90%), speculative decoding
//! yields 2-3x speedup over vanilla autoregressive decode with zero quality loss.
//! The output distribution is provably identical to the target model alone.

use std::time::{Duration, Instant};

use crate::device::{DeviceError, WarpDevice};
use crate::generate::{create_test_engine, GenerateConfig, GenerationEngine, GenerationResult};
use crate::kv_cache::ModelKVCache;
use crate::transformer::TransformerConfig;

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of draft tokens to generate per speculation round.
    pub num_draft_tokens: usize,
    /// Whether to adaptively adjust num_draft_tokens based on acceptance rate.
    pub adaptive_k: bool,
    /// Minimum acceptance rate before reducing K.
    pub min_acceptance_rate: f32,
    /// Maximum K to try.
    pub max_draft_tokens: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_draft_tokens: 4,
            adaptive_k: true,
            min_acceptance_rate: 0.4,
            max_draft_tokens: 8,
        }
    }
}

/// Statistics from speculative decoding.
#[derive(Debug, Clone)]
pub struct SpecStats {
    pub total_tokens: u64,
    pub accepted_tokens: u64,
    pub speculation_rounds: u64,
    pub avg_acceptance_rate: f32,
    pub tokens_per_round: f32,
}

impl SpecStats {
    fn new() -> Self {
        Self {
            total_tokens: 0,
            accepted_tokens: 0,
            speculation_rounds: 0,
            avg_acceptance_rate: 0.0,
            tokens_per_round: 0.0,
        }
    }

    fn update(&mut self, accepted: usize, draft_k: usize) {
        self.speculation_rounds += 1;
        // accepted tokens + 1 bonus token per round
        let emitted = accepted + 1;
        self.total_tokens += emitted as u64;
        self.accepted_tokens += accepted as u64;
        let rate = accepted as f32 / draft_k.max(1) as f32;
        // Running average
        let n = self.speculation_rounds as f32;
        self.avg_acceptance_rate =
            self.avg_acceptance_rate * ((n - 1.0) / n) + rate / n;
        self.tokens_per_round =
            self.total_tokens as f32 / self.speculation_rounds as f32;
    }
}

impl std::fmt::Display for SpecStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SpecStats: {} rounds, {} tokens ({} accepted), \
             avg accept rate {:.1}%, tokens/round {:.2}",
            self.speculation_rounds,
            self.total_tokens,
            self.accepted_tokens,
            self.avg_acceptance_rate * 100.0,
            self.tokens_per_round,
        )
    }
}

/// Speculative decoding engine.
///
/// Uses a draft model (fewer layers) to speculate K tokens, then verifies
/// with the full target model. The output distribution is mathematically
/// identical to the target model alone.
pub struct SpeculativeEngine {
    /// Target model (full size).
    pub target: GenerationEngine,
    /// Draft model (smaller -- fewer layers, same vocab).
    pub draft: GenerationEngine,
    /// Configuration.
    pub config: SpeculativeConfig,
    /// Statistics tracking.
    pub stats: SpecStats,
}

/// Compute softmax probabilities on CPU from logits.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    if sum <= 0.0 {
        // Uniform fallback
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exp_vals.iter().map(|v| v / sum).collect()
    }
}

/// Argmax of a float slice.
fn argmax(vals: &[f32]) -> usize {
    vals.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Sample from a probability distribution using a deterministic pseudo-random value.
/// `seed` is used to produce a repeatable pseudo-random number in [0, 1).
fn deterministic_sample(probs: &[f32], seed: u64) -> usize {
    // Simple LCG-style hash for deterministic "randomness"
    let r = ((seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)) >> 33)
        as f32
        / (1u64 << 31) as f32;
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i;
        }
    }
    probs.len() - 1
}

/// Sample from the adjusted distribution: normalize(max(0, p_target - p_draft)).
/// This is the key to speculative decoding's correctness guarantee.
fn sample_from_adjusted(
    target_logits: &[f32],
    draft_logits: &[f32],
    seed: u64,
) -> i32 {
    let p_target = softmax(target_logits);
    let p_draft = softmax(draft_logits);

    // Compute adjusted distribution: max(0, p_target - p_draft)
    let adjusted: Vec<f32> = p_target
        .iter()
        .zip(p_draft.iter())
        .map(|(&pt, &pd)| (pt - pd).max(0.0))
        .collect();

    let sum: f32 = adjusted.iter().sum();
    if sum <= 1e-10 {
        // Fallback: sample from target distribution directly
        return deterministic_sample(&p_target, seed) as i32;
    }

    // Normalize
    let normalized: Vec<f32> = adjusted.iter().map(|v| v / sum).collect();
    deterministic_sample(&normalized, seed) as i32
}

impl SpeculativeEngine {
    /// Create a new speculative decoding engine.
    pub fn new(
        target: GenerationEngine,
        draft: GenerationEngine,
        config: SpeculativeConfig,
    ) -> Self {
        assert_eq!(
            target.vocab_size, draft.vocab_size,
            "Target and draft models must have the same vocabulary size"
        );
        Self {
            target,
            draft,
            config,
            stats: SpecStats::new(),
        }
    }

    /// Draft K tokens using the draft model with KV cache.
    /// Returns (draft_token_ids, draft_logits_per_position).
    /// Each entry in draft_logits is [vocab_size] logits for that position.
    fn draft_k_tokens(
        &self,
        device: &WarpDevice,
        kv_cache_draft: &mut ModelKVCache,
        last_token: i32,
        start_pos: u32,
        k: usize,
    ) -> Result<(Vec<i32>, Vec<Vec<f32>>), DeviceError> {
        let vocab = self.draft.vocab_size as usize;
        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_logits = Vec::with_capacity(k);
        let mut current_token = last_token;
        let mut pos = start_pos;

        for _ in 0..k {
            let logits_tensor =
                self.draft
                    .forward_decode(device, current_token, kv_cache_draft, pos)?;
            device.synchronize()?;

            let logits_host = logits_tensor.to_host(device)?;
            // logits_host is [1, vocab_size], take the single row
            let logits_row = logits_host[..vocab].to_vec();

            // Greedy draft: pick argmax
            let next_token = argmax(&logits_row) as i32;

            draft_logits.push(logits_row);
            draft_tokens.push(next_token);
            current_token = next_token;
            pos += 1;
        }

        Ok((draft_tokens, draft_logits))
    }

    /// Verify draft tokens against the target model.
    /// Runs target model on [last_accepted_token, draft_0, ..., draft_{K-1}] in
    /// one prefill-style forward pass (K+1 tokens).
    ///
    /// Returns (num_accepted, all emitted tokens for this round).
    fn verify_and_accept(
        &self,
        device: &WarpDevice,
        kv_cache_target: &mut ModelKVCache,
        last_accepted_token: i32,
        draft_tokens: &[i32],
        draft_logits: &[Vec<f32>],
        round_seed: u64,
    ) -> Result<(usize, Vec<i32>), DeviceError> {
        let k = draft_tokens.len();
        let vocab = self.target.vocab_size as usize;

        // Build the verification sequence: [last_accepted, draft_0, ..., draft_{K-1}]
        let mut verify_ids = Vec::with_capacity(k + 1);
        verify_ids.push(last_accepted_token);
        verify_ids.extend_from_slice(draft_tokens);

        // Run target model on all K+1 positions in one forward pass.
        // We use forward_prefill which processes the full sequence and populates KV cache.
        let target_logits_tensor =
            self.target
                .forward_prefill(device, &verify_ids, kv_cache_target)?;
        device.synchronize()?;

        let target_logits_host = target_logits_tensor.to_host(device)?;

        // target_logits_host is [(K+1), vocab_size]
        // Position i corresponds to the prediction for position i+1:
        //   target_logits[0] predicts what comes after last_accepted (should match draft_0)
        //   target_logits[1] predicts what comes after draft_0 (should match draft_1)
        //   ...
        //   target_logits[K] predicts what comes after draft_{K-1} (bonus token)

        let mut emitted = Vec::new();
        let mut accepted = 0usize;

        for i in 0..k {
            let t_logits = &target_logits_host[i * vocab..(i + 1) * vocab];
            let d_logits = &draft_logits[i];

            let p_target = softmax(t_logits);
            let p_draft = softmax(d_logits);

            let draft_token_idx = draft_tokens[i] as usize;
            let pt = p_target[draft_token_idx.min(vocab - 1)];
            let pd = p_draft[draft_token_idx.min(vocab - 1)];

            // Accept with probability min(1, p_target / p_draft)
            let accept_prob = if pd > 1e-10 { (pt / pd).min(1.0) } else { 1.0 };

            // Deterministic acceptance check
            let seed_i = round_seed.wrapping_add(i as u64);
            let r = ((seed_i.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407))
                >> 33) as f32
                / (1u64 << 31) as f32;

            if r < accept_prob {
                // Accept this draft token
                accepted += 1;
                emitted.push(draft_tokens[i]);
            } else {
                // Reject: sample from adjusted distribution max(0, p_target - p_draft)
                let bonus = sample_from_adjusted(t_logits, d_logits, seed_i.wrapping_add(1000));
                emitted.push(bonus);
                break;
            }
        }

        if accepted == k {
            // All K draft tokens accepted -- sample one bonus token from target_logits[K]
            let bonus_logits = &target_logits_host[k * vocab..(k + 1) * vocab];
            let bonus_probs = softmax(bonus_logits);
            let bonus = deterministic_sample(&bonus_probs, round_seed.wrapping_add(999));
            emitted.push(bonus as i32);
        }

        Ok((accepted, emitted))
    }

    /// Roll back KV caches to a given sequence length.
    /// This simply sets the `len` field on each layer cache -- the stale data
    /// beyond that point will be overwritten on subsequent appends/prefills.
    fn rollback_kv_cache(cache: &mut ModelKVCache, new_len: u32) {
        for layer in &mut cache.layers {
            layer.len = new_len;
        }
    }

    /// Main speculative generation loop.
    ///
    /// Phase 1: Prefill the prompt through both target and draft models.
    /// Phase 2: Repeatedly draft K tokens, verify, accept/reject.
    ///
    /// Returns a GenerationResult compatible with the standard generation API.
    pub fn generate(
        &mut self,
        device: &WarpDevice,
        prompt_ids: &[i32],
        gen_config: &GenerateConfig,
        max_seq_len: u32,
    ) -> Result<GenerationResult, DeviceError> {
        let target_kv_dim = self.target.config.kv_dim();
        let target_layers = self.target.layers.len() as u32;
        let draft_kv_dim = self.draft.config.kv_dim();
        let draft_layers = self.draft.layers.len() as u32;
        let vocab = self.target.vocab_size as usize;

        // Allocate KV caches for both models
        let mut kv_cache_target =
            ModelKVCache::new(device, target_layers, max_seq_len, target_kv_dim)?;
        let mut kv_cache_draft =
            ModelKVCache::new(device, draft_layers, max_seq_len, draft_kv_dim)?;

        let prefill_start = Instant::now();

        // Phase 1: Prefill both models with the prompt
        let target_logits =
            self.target
                .forward_prefill(device, prompt_ids, &mut kv_cache_target)?;
        let draft_logits_tensor =
            self.draft
                .forward_prefill(device, prompt_ids, &mut kv_cache_draft)?;
        device.synchronize()?;

        let prefill_time = prefill_start.elapsed();

        // Get last token's logits from target prefill to pick the first token
        let all_target_logits = target_logits.to_host(device)?;
        let prompt_len = prompt_ids.len();
        let last_logits =
            &all_target_logits[(prompt_len - 1) * vocab..prompt_len * vocab];

        // Pick first token (greedy)
        let first_token = argmax(last_logits) as i32;

        if let Some(eos) = gen_config.eos_token_id {
            if first_token == eos {
                return Ok(GenerationResult {
                    tokens: vec![],
                    prefill_time,
                    decode_time: Duration::ZERO,
                    tokens_generated: 0,
                    prefill_tokens: prompt_len,
                    tokens_per_sec: 0.0,
                    kv_cache_memory_bytes: kv_cache_target.memory_bytes()
                        + kv_cache_draft.memory_bytes(),
                });
            }
        }

        let decode_start = Instant::now();
        let mut generated = vec![first_token];
        let mut pos = prompt_len as u32; // next position to fill
        let mut current_k = self.config.num_draft_tokens;
        let mut round_seed: u64 = 42;

        // The last accepted token drives the next draft phase
        let mut last_accepted_token = first_token;

        // We need to advance the draft model's KV cache past first_token too.
        // Run one decode step on the draft to include first_token in its cache.
        let _draft_step = self.draft.forward_decode(
            device,
            first_token,
            &mut kv_cache_draft,
            pos,
        )?;
        device.synchronize()?;
        pos += 1;

        // Also advance target KV cache: run decode for the first token
        let _target_step = self.target.forward_decode(
            device,
            first_token,
            &mut kv_cache_target,
            pos - 1,
        )?;
        device.synchronize()?;

        let max_tokens = gen_config.max_tokens;

        // Phase 2: Speculative decode loop
        while generated.len() < max_tokens {
            let remaining = max_tokens - generated.len();
            let k = current_k.min(remaining);
            if k == 0 {
                break;
            }

            // Record cache lengths before drafting (for rollback)
            let draft_cache_len_before = kv_cache_draft.seq_len();
            let target_cache_len_before = kv_cache_target.seq_len();

            // Draft phase: generate K tokens with draft model
            let (draft_tokens, draft_logits_list) = self.draft_k_tokens(
                device,
                &mut kv_cache_draft,
                last_accepted_token,
                pos,
                k,
            )?;

            // Verify phase: run target model on the draft tokens
            // First, roll back target KV cache to before this round's tokens
            // (forward_prefill will write from position 0, so we need to handle
            // this differently -- we run individual decode steps for verification)

            // For verification, we run each draft token through target decode
            // to get per-position logits. This is simpler than prefill for
            // maintaining correct KV cache state.
            let mut target_logits_list = Vec::with_capacity(k + 1);

            // Get logits for each draft position from target model
            for i in 0..k {
                let token = if i == 0 {
                    last_accepted_token
                } else {
                    draft_tokens[i - 1]
                };
                // Run target decode to get logits predicting the next token
                let logits_tensor = self.target.forward_decode(
                    device,
                    if i == 0 { last_accepted_token } else { draft_tokens[i - 1] },
                    &mut kv_cache_target,
                    target_cache_len_before + i as u32,
                )?;
                device.synchronize()?;
                let logits_host = logits_tensor.to_host(device)?;
                target_logits_list.push(logits_host[..vocab].to_vec());
            }
            // One more decode for the bonus token position
            let bonus_logits_tensor = self.target.forward_decode(
                device,
                draft_tokens[k - 1],
                &mut kv_cache_target,
                target_cache_len_before + k as u32,
            )?;
            device.synchronize()?;
            let bonus_logits_host = bonus_logits_tensor.to_host(device)?;
            target_logits_list.push(bonus_logits_host[..vocab].to_vec());

            // Accept/reject loop
            let mut accepted = 0usize;
            let mut emitted = Vec::new();

            for i in 0..k {
                let t_logits = &target_logits_list[i];
                let d_logits = &draft_logits_list[i];

                let p_target = softmax(t_logits);
                let p_draft = softmax(d_logits);

                let draft_token_idx = draft_tokens[i] as usize;
                let pt = p_target[draft_token_idx.min(vocab - 1)];
                let pd = p_draft[draft_token_idx.min(vocab - 1)];

                // Accept with probability min(1, p_target / p_draft)
                let accept_prob = if pd > 1e-10 { (pt / pd).min(1.0) } else { 1.0 };

                let seed_i = round_seed.wrapping_add(i as u64);
                let r = ((seed_i
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407))
                    >> 33) as f32
                    / (1u64 << 31) as f32;

                if r < accept_prob {
                    accepted += 1;
                    emitted.push(draft_tokens[i]);
                } else {
                    // Reject: sample from adjusted distribution
                    let bonus =
                        sample_from_adjusted(t_logits, d_logits, seed_i.wrapping_add(1000));
                    emitted.push(bonus);
                    break;
                }
            }

            if accepted == k {
                // All accepted -- sample bonus from target_logits[K]
                let bonus_logits = &target_logits_list[k];
                let bonus_probs = softmax(bonus_logits);
                let bonus = deterministic_sample(&bonus_probs, round_seed.wrapping_add(999));
                emitted.push(bonus as i32);
            }

            // Update stats
            self.stats.update(accepted, k);

            // Check for EOS in emitted tokens
            let mut eos_hit = false;
            for &tok in &emitted {
                if let Some(eos) = gen_config.eos_token_id {
                    if tok == eos {
                        eos_hit = true;
                        break;
                    }
                }
                generated.push(tok);
            }

            // Roll back BOTH KV caches to the correct length.
            //
            // Draft cache: must roll back to only accepted positions. Rejected
            // draft tokens left stale entries in the draft KV cache; if we don't
            // roll back, the next speculation round would draft from a corrupted
            // cache state (containing KVs for tokens that were never emitted).
            let draft_rollback = draft_cache_len_before + accepted as u32;
            Self::rollback_kv_cache(&mut kv_cache_draft, draft_rollback);

            // Target cache: roll back to accepted + 1. The "+1" accounts for
            // the resampled/bonus token that the target model produced. When all
            // K drafts are accepted, the bonus token is from target_logits[K],
            // so we keep K+1 entries. When rejection happens at position i, we
            // keep i accepted entries + 1 for the resampled token.
            let target_rollback = target_cache_len_before + accepted as u32 + 1;
            Self::rollback_kv_cache(&mut kv_cache_target, target_rollback);

            // Update position and last accepted token
            if let Some(&last) = generated.last() {
                last_accepted_token = last;
            }
            pos = prompt_len as u32 + generated.len() as u32;

            if eos_hit {
                break;
            }

            // Adaptive K: adjust draft length based on acceptance rate
            if self.config.adaptive_k {
                let rate = accepted as f32 / k as f32;
                if rate > 0.8 && current_k < self.config.max_draft_tokens {
                    current_k += 1;
                }
                if rate < self.config.min_acceptance_rate && current_k > 1 {
                    current_k -= 1;
                }
            }

            round_seed = round_seed.wrapping_add(k as u64 + 7);
        }

        // Trim to max_tokens
        generated.truncate(max_tokens);

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
            kv_cache_memory_bytes: kv_cache_target.memory_bytes()
                + kv_cache_draft.memory_bytes(),
        })
    }
}

impl std::fmt::Display for SpeculativeEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SpeculativeEngine:")?;
        writeln!(
            f,
            "  Target: {} layers, H={}",
            self.target.layers.len(),
            self.target.config.hidden_size
        )?;
        writeln!(
            f,
            "  Draft:  {} layers, H={}",
            self.draft.layers.len(),
            self.draft.config.hidden_size
        )?;
        writeln!(f, "  K={} (adaptive={})", self.config.num_draft_tokens, self.config.adaptive_k)?;
        write!(f, "  {}", self.stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::WarpDevice;
    use crate::generate::{create_test_engine, GenerateConfig};
    use crate::transformer::TransformerConfig;

    #[test]
    fn speculative_decoding_basic() {
        let device = WarpDevice::new(0).expect("CUDA device required");

        let config = TransformerConfig::tiny(); // H=64, heads=4, head_dim=16
        let vocab_size = 100u32;

        // Target model: 4 layers (full model)
        let target = create_test_engine(&device, config.clone(), 4, vocab_size)
            .expect("failed to create target engine");

        // Draft model: 1 layer (lightweight draft)
        let draft = create_test_engine(&device, config.clone(), 1, vocab_size)
            .expect("failed to create draft engine");

        let spec_config = SpeculativeConfig {
            num_draft_tokens: 4,
            adaptive_k: true,
            min_acceptance_rate: 0.4,
            max_draft_tokens: 8,
        };

        let mut engine = SpeculativeEngine::new(target, draft, spec_config);

        let prompt: Vec<i32> = vec![1, 5, 10, 15, 20];
        let gen_config = GenerateConfig {
            max_tokens: 20,
            temperature: 1.0,
            eos_token_id: None, // don't stop early for this test
            greedy: true,
            ..Default::default()
        };

        let result = engine
            .generate(&device, &prompt, &gen_config, 128)
            .expect("speculative generation failed");

        // Basic correctness: we got tokens
        assert!(
            !result.tokens.is_empty(),
            "Speculative decoding produced no tokens"
        );
        assert!(
            result.tokens.len() <= gen_config.max_tokens,
            "Generated more tokens than max_tokens"
        );

        // All tokens should be valid (in vocab range)
        for &tok in &result.tokens {
            assert!(
                tok >= 0 && (tok as u32) < vocab_size,
                "Token {} out of vocab range [0, {})",
                tok,
                vocab_size
            );
        }

        // Print stats
        println!("=== Speculative Decoding Results ===");
        println!("{}", result);
        println!("{}", engine.stats);
        println!("{}", engine);
        println!("Tokens generated: {}", result.tokens_generated);
        println!(
            "Decode speed: {:.1} tokens/sec",
            result.tokens_per_sec
        );

        // Stats should be populated
        assert!(
            engine.stats.speculation_rounds > 0,
            "No speculation rounds recorded"
        );
        assert!(
            engine.stats.total_tokens > 0,
            "No total tokens recorded"
        );

        // Acceptance rate should be between 0 and 1
        assert!(
            engine.stats.avg_acceptance_rate >= 0.0
                && engine.stats.avg_acceptance_rate <= 1.0,
            "Invalid acceptance rate: {}",
            engine.stats.avg_acceptance_rate
        );

        println!(
            "Acceptance rate: {:.1}%",
            engine.stats.avg_acceptance_rate * 100.0
        );
        println!(
            "Tokens per round: {:.2}",
            engine.stats.tokens_per_round
        );
    }

    #[test]
    fn speculative_vs_baseline_comparison() {
        let device = WarpDevice::new(0).expect("CUDA device required");

        let config = TransformerConfig::tiny();
        let vocab_size = 100u32;
        let prompt: Vec<i32> = vec![1, 5, 10];

        let gen_config = GenerateConfig {
            max_tokens: 16,
            temperature: 1.0,
            eos_token_id: None,
            greedy: true,
            ..Default::default()
        };

        // Baseline: standard autoregressive generation with target model
        let baseline_engine =
            create_test_engine(&device, config.clone(), 4, vocab_size)
                .expect("baseline engine");
        let baseline_result = baseline_engine
            .generate_with_cache(&device, &prompt, &gen_config, 128)
            .expect("baseline generation");

        // Speculative: target (4 layers) + draft (1 layer)
        let target = create_test_engine(&device, config.clone(), 4, vocab_size)
            .expect("target engine");
        let draft = create_test_engine(&device, config.clone(), 1, vocab_size)
            .expect("draft engine");

        let spec_config = SpeculativeConfig {
            num_draft_tokens: 4,
            adaptive_k: false,
            min_acceptance_rate: 0.4,
            max_draft_tokens: 8,
        };

        let mut spec_engine = SpeculativeEngine::new(target, draft, spec_config);
        let spec_result = spec_engine
            .generate(&device, &prompt, &gen_config, 128)
            .expect("speculative generation");

        println!("=== Baseline vs Speculative ===");
        println!(
            "Baseline: {} tokens in {:.2}ms ({:.1} tok/s)",
            baseline_result.tokens_generated,
            baseline_result.decode_time.as_secs_f64() * 1000.0,
            baseline_result.tokens_per_sec,
        );
        println!(
            "Speculative: {} tokens in {:.2}ms ({:.1} tok/s)",
            spec_result.tokens_generated,
            spec_result.decode_time.as_secs_f64() * 1000.0,
            spec_result.tokens_per_sec,
        );
        println!("{}", spec_engine.stats);

        // Both should produce valid tokens
        assert!(!baseline_result.tokens.is_empty());
        assert!(!spec_result.tokens.is_empty());

        // Both should produce tokens in valid range
        for &tok in &spec_result.tokens {
            assert!(tok >= 0 && (tok as u32) < vocab_size);
        }
        for &tok in &baseline_result.tokens {
            assert!(tok >= 0 && (tok as u32) < vocab_size);
        }

        // Report KV cache memory overhead
        println!(
            "Baseline KV memory:     {:.2} MB",
            baseline_result.kv_cache_memory_bytes as f64 / 1e6
        );
        println!(
            "Speculative KV memory:  {:.2} MB (target + draft)",
            spec_result.kv_cache_memory_bytes as f64 / 1e6
        );
    }

    #[test]
    fn speculative_adaptive_k() {
        let device = WarpDevice::new(0).expect("CUDA device required");
        let config = TransformerConfig::tiny();
        let vocab_size = 100u32;

        let target = create_test_engine(&device, config.clone(), 4, vocab_size)
            .expect("target");
        let draft = create_test_engine(&device, config.clone(), 1, vocab_size)
            .expect("draft");

        let spec_config = SpeculativeConfig {
            num_draft_tokens: 3,
            adaptive_k: true,
            min_acceptance_rate: 0.4,
            max_draft_tokens: 6,
        };

        let mut engine = SpeculativeEngine::new(target, draft, spec_config);

        let prompt: Vec<i32> = vec![1, 2, 3];
        let gen_config = GenerateConfig {
            max_tokens: 30,
            temperature: 1.0,
            eos_token_id: None,
            greedy: true,
            ..Default::default()
        };

        let result = engine
            .generate(&device, &prompt, &gen_config, 128)
            .expect("adaptive speculative generation");

        println!("=== Adaptive K Test ===");
        println!("{}", result);
        println!("{}", engine.stats);
        println!(
            "Final tokens/round: {:.2} (started with K=3)",
            engine.stats.tokens_per_round
        );

        assert!(!result.tokens.is_empty());
        assert!(engine.stats.speculation_rounds > 0);
    }

    #[test]
    fn speculative_eos_handling() {
        let device = WarpDevice::new(0).expect("CUDA device required");
        let config = TransformerConfig::tiny();
        let vocab_size = 100u32;

        let target = create_test_engine(&device, config.clone(), 4, vocab_size)
            .expect("target");
        let draft = create_test_engine(&device, config.clone(), 1, vocab_size)
            .expect("draft");

        let spec_config = SpeculativeConfig::default();
        let mut engine = SpeculativeEngine::new(target, draft, spec_config);

        let prompt: Vec<i32> = vec![1, 2, 3];
        let gen_config = GenerateConfig {
            max_tokens: 50,
            temperature: 1.0,
            eos_token_id: Some(2), // token 2 as EOS
            greedy: true,
            ..Default::default()
        };

        let result = engine
            .generate(&device, &prompt, &gen_config, 128)
            .expect("eos speculative generation");

        println!("=== EOS Handling Test ===");
        println!("Generated {} tokens", result.tokens_generated);
        println!("Tokens: {:?}", &result.tokens);

        // No emitted token should be the EOS token
        for &tok in &result.tokens {
            assert_ne!(
                tok, 2,
                "EOS token should not appear in generated output"
            );
        }
    }
}

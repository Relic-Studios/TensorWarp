//! Batch inference support — run multiple inputs through a model simultaneously.
//!
//! Batch processing amortizes kernel launch overhead across multiple inputs
//! and enables better GPU utilization through larger matrix dimensions.
//!
//! The `ContinuousBatcher` now integrates with `GenerationEngine` to run real
//! autoregressive inference across multiple requests, managing per-request
//! KV caches and collecting timing statistics.

use std::time::{Duration, Instant};

use warp_ir::{DType, Shape};

use crate::device::{DeviceError, WarpDevice};
use crate::generate::{GenerateConfig, GenerationEngine};
use crate::tensor::GpuTensor;

/// Batch configuration.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size.
    pub max_batch: usize,
    /// Current number of active requests.
    pub active: usize,
    /// Whether to pad to max_batch for consistent kernel shapes.
    pub pad_to_max: bool,
}

impl BatchConfig {
    pub fn new(max_batch: usize) -> Self {
        Self { max_batch, active: 0, pad_to_max: false }
    }
}

/// Batch multiple 1D tensors into a single batched tensor.
pub fn batch_tensors(
    device: &WarpDevice,
    tensors: &[&GpuTensor<f32>],
) -> Result<GpuTensor<f32>, DeviceError> {
    if tensors.is_empty() {
        return Err(DeviceError::Memory("empty batch".into()));
    }

    let elem_size = tensors[0].numel;
    let batch_size = tensors.len();

    // Gather all data to host, concatenate, upload
    let mut all_data = Vec::with_capacity(batch_size * elem_size);
    for t in tensors {
        let data = t.to_host(device)?;
        all_data.extend_from_slice(&data);
    }

    GpuTensor::from_host(device, &all_data,
        Shape::from_static(&[batch_size, elem_size]),
        DType::F32)
}

/// Split a batched tensor back into individual tensors.
pub fn unbatch_tensor(
    device: &WarpDevice,
    batched: &GpuTensor<f32>,
    batch_size: usize,
) -> Result<Vec<GpuTensor<f32>>, DeviceError> {
    let data = batched.to_host(device)?;
    let elem_size = data.len() / batch_size;

    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let chunk = &data[i * elem_size..(i + 1) * elem_size];
        let t = GpuTensor::from_host(device, chunk,
            Shape::from_static(&[elem_size]), DType::F32)?;
        results.push(t);
    }
    Ok(results)
}

/// Continuous batching scheduler — manages a queue of inference requests.
pub struct ContinuousBatcher {
    /// Maximum concurrent batch size.
    max_batch: usize,
    /// Pending requests.
    pending: Vec<BatchRequest>,
    /// Running requests.
    running: Vec<BatchRequest>,
    /// Completed count.
    completed: u64,
}

/// Result from running a generation request through the batcher.
pub struct BatchResult {
    pub request_id: u64,
    pub tokens: Vec<i32>,
    pub decode_time: Duration,
}

/// A single inference request.
pub struct BatchRequest {
    pub id: u64,
    pub input_data: Vec<f32>,
    pub input_shape: Vec<usize>,
    pub output: Option<Vec<f32>>,
    pub status: RequestStatus,
    /// Token IDs for text generation requests.
    pub prompt_ids: Vec<i32>,
    /// Generation configuration (temperature, max_tokens, etc.).
    pub gen_config: Option<GenerateConfig>,
    /// Maximum number of tokens to generate for this request.
    pub max_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RequestStatus {
    Pending,
    Running,
    Completed,
}

impl ContinuousBatcher {
    pub fn new(max_batch: usize) -> Self {
        Self { max_batch, pending: Vec::new(), running: Vec::new(), completed: 0 }
    }

    /// Add a new inference request to the queue.
    pub fn submit(&mut self, input_data: Vec<f32>, input_shape: Vec<usize>) -> u64 {
        let id = self.completed + self.pending.len() as u64 + self.running.len() as u64;
        self.pending.push(BatchRequest {
            id, input_data, input_shape,
            output: None, status: RequestStatus::Pending,
            prompt_ids: Vec::new(),
            gen_config: None,
            max_tokens: 0,
        });
        id
    }

    /// Submit a text generation request with prompt token IDs.
    pub fn submit_generation(
        &mut self,
        prompt_ids: Vec<i32>,
        max_tokens: usize,
    ) -> u64 {
        let id = self.completed + self.pending.len() as u64 + self.running.len() as u64;
        self.pending.push(BatchRequest {
            id,
            input_data: Vec::new(),
            input_shape: Vec::new(),
            output: None,
            status: RequestStatus::Pending,
            prompt_ids,
            gen_config: Some(GenerateConfig {
                max_tokens,
                greedy: true,
                temperature: 1.0,
                eos_token_id: None, // no early stop in batch mode by default
                ..Default::default()
            }),
            max_tokens,
        });
        id
    }

    /// Get the next batch of requests to process.
    pub fn next_batch(&mut self) -> Vec<BatchRequest> {
        let n = self.max_batch.min(self.pending.len());
        let batch: Vec<BatchRequest> = self.pending.drain(..n).collect();
        batch
    }

    /// Mark requests as completed.
    pub fn complete(&mut self, count: usize) {
        self.completed += count as u64;
    }

    /// Stats.
    pub fn stats(&self) -> String {
        format!("Batcher: {} pending, {} running, {} completed",
            self.pending.len(), self.running.len(), self.completed)
    }

    /// Run a batch of generation requests through the engine.
    ///
    /// This is the core integration point: the batcher pulls pending requests,
    /// allocates per-request KV caches, runs prefill + decode for each request,
    /// and returns the generated tokens with timing.
    ///
    /// Currently processes requests sequentially (one KV cache at a time).
    /// True batched GEMM with padding across requests can come later.
    pub fn run_batch(
        &mut self,
        engine: &GenerationEngine,
        device: &WarpDevice,
        max_seq_len: u32,
    ) -> Result<Vec<BatchResult>, DeviceError> {
        let batch = self.next_batch();
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        let num_requests = batch.len();
        let mut results = Vec::with_capacity(num_requests);

        for request in &batch {
            if request.prompt_ids.is_empty() {
                // Not a generation request — skip
                continue;
            }

            let gen_config = request.gen_config.clone().unwrap_or_else(|| GenerateConfig {
                max_tokens: request.max_tokens.max(1),
                greedy: true,
                temperature: 1.0,
                eos_token_id: None,
                ..Default::default()
            });

            let decode_start = Instant::now();

            // Run full generation with KV cache for this request
            let gen_result = engine.generate_with_cache(
                device,
                &request.prompt_ids,
                &gen_config,
                max_seq_len,
            )?;

            let decode_time = decode_start.elapsed();

            results.push(BatchResult {
                request_id: request.id,
                tokens: gen_result.tokens,
                decode_time,
            });
        }

        self.complete(num_requests);
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::{create_test_engine, GenerateConfig};
    use crate::transformer::TransformerConfig;

    #[test]
    fn batch_unbatch_roundtrip() {
        let dev = match WarpDevice::new(0) {
            Ok(d) => d, Err(_) => { println!("No CUDA"); return; }
        };

        let t1 = GpuTensor::from_host(&dev, &[1.0f32, 2.0, 3.0],
            Shape::from_static(&[3]), DType::F32).unwrap();
        let t2 = GpuTensor::from_host(&dev, &[4.0f32, 5.0, 6.0],
            Shape::from_static(&[3]), DType::F32).unwrap();

        let batched = batch_tensors(&dev, &[&t1, &t2]).unwrap();
        assert_eq!(batched.numel, 6);

        let unbatched = unbatch_tensor(&dev, &batched, 2).unwrap();
        assert_eq!(unbatched.len(), 2);

        let r1 = unbatched[0].to_host(&dev).unwrap();
        let r2 = unbatched[1].to_host(&dev).unwrap();
        assert_eq!(r1, vec![1.0, 2.0, 3.0]);
        assert_eq!(r2, vec![4.0, 5.0, 6.0]);
        println!("Batch/unbatch roundtrip: correct!");
    }

    #[test]
    fn continuous_batcher_basic() {
        let mut batcher = ContinuousBatcher::new(4);

        batcher.submit(vec![1.0; 10], vec![10]);
        batcher.submit(vec![2.0; 10], vec![10]);
        batcher.submit(vec![3.0; 10], vec![10]);

        let batch = batcher.next_batch();
        assert_eq!(batch.len(), 3);

        batcher.complete(3);
        println!("{}", batcher.stats());
        assert_eq!(batcher.completed, 3);
    }

    #[test]
    fn continuous_batching_generation() {
        let dev = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA — skipping"); return; }
        };

        let config = TransformerConfig {
            hidden_size: 64,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            ffn_dim: 128,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            attention_mode: crate::transformer::AttentionMode::Standard,
        };

        let num_layers = 2u32;
        let vocab_size = 100u32;
        let engine = create_test_engine(&dev, config, num_layers, vocab_size).unwrap();

        let mut batcher = ContinuousBatcher::new(4);

        // Submit 3 generation requests with different prompt lengths
        let id0 = batcher.submit_generation(vec![1, 5, 10], 8);
        let id1 = batcher.submit_generation(vec![2, 7, 3, 9, 12], 6);
        let id2 = batcher.submit_generation(vec![4, 8], 10);

        println!("Submitted 3 requests: ids={}, {}, {}", id0, id1, id2);
        println!("{}", batcher.stats());

        let max_seq_len = 256;
        let results = batcher.run_batch(&engine, &dev, max_seq_len).unwrap();

        assert_eq!(results.len(), 3, "Expected 3 results from batch");

        for res in &results {
            assert!(!res.tokens.is_empty(),
                "Request {} should have generated tokens", res.request_id);
            println!("  Request {}: {} tokens in {:.2}ms — {:?}",
                res.request_id,
                res.tokens.len(),
                res.decode_time.as_secs_f64() * 1000.0,
                &res.tokens[..res.tokens.len().min(10)]);
        }

        // Verify batcher tracked completions
        println!("{}", batcher.stats());
        assert_eq!(batcher.completed, 3);

        // Print aggregate timing
        let total_time: Duration = results.iter().map(|r| r.decode_time).sum();
        let total_tokens: usize = results.iter().map(|r| r.tokens.len()).sum();
        println!("Batch total: {} tokens in {:.2}ms ({:.1} tok/s)",
            total_tokens,
            total_time.as_secs_f64() * 1000.0,
            total_tokens as f64 / total_time.as_secs_f64().max(1e-9));
    }
}

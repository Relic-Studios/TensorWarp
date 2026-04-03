//! Batch inference support — run multiple inputs through a model simultaneously.
//!
//! Batch processing amortizes kernel launch overhead across multiple inputs
//! and enables better GPU utilization through larger matrix dimensions.

use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
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

/// A single inference request.
pub struct BatchRequest {
    pub id: u64,
    pub input_data: Vec<f32>,
    pub input_shape: Vec<usize>,
    pub output: Option<Vec<f32>>,
    pub status: RequestStatus,
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

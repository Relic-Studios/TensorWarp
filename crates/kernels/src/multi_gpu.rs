//! Multi-GPU support — device enumeration and tensor parallelism.
//!
//! Supports:
//! - Device enumeration and selection
//! - Tensor parallelism: split large matrices across GPUs
//! - Pipeline parallelism: different layers on different GPUs
//!
//! Usage:
//! ```ignore
//! let devices = MultiDevice::new()?;
//! println!("Found {} GPUs", devices.count());
//!
//! // Tensor parallel: split a GEMM across 2 GPUs
//! let result = devices.parallel_gemm(&a, &b, m, n, k)?;
//! ```

use cudarc::driver::sys;
use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::ops;
use crate::tensor::GpuTensor;

/// Multi-GPU device manager.
pub struct MultiDevice {
    /// Available GPU devices.
    pub devices: Vec<WarpDevice>,
    /// Per-device kernel caches.
    pub caches: Vec<KernelCache>,
}

impl MultiDevice {
    /// Initialize all available GPU devices.
    pub fn new() -> Result<Self, DeviceError> {
        let count = WarpDevice::device_count()?;
        if count == 0 {
            return Err(DeviceError::Init("No CUDA devices found".into()));
        }

        let mut devices = Vec::new();
        let mut caches = Vec::new();
        for i in 0..count {
            devices.push(WarpDevice::new(i)?);
            caches.push(KernelCache::new());
        }

        Ok(Self { devices, caches })
    }

    /// Number of available GPUs.
    pub fn count(&self) -> usize {
        self.devices.len()
    }

    /// Get device and cache for a specific GPU.
    pub fn get(&self, ordinal: usize) -> Option<(&WarpDevice, &KernelCache)> {
        self.devices.get(ordinal).map(|d| (d, &self.caches[ordinal]))
    }

    /// Summary of all devices.
    pub fn summary(&self) -> String {
        let mut lines = vec![format!("MultiDevice: {} GPUs", self.count())];
        for (i, dev) in self.devices.iter().enumerate() {
            lines.push(format!("  [{}] {}", i, dev.summary()));
        }
        lines.join("\n")
    }

    /// Create an isolated execution context for a model.
    /// Each model gets its own CUDA stream for concurrent execution.
    pub fn create_model_context(&self, device_idx: usize) -> Result<ModelContext, DeviceError> {
        let (dev, _cache) = self.get(device_idx)
            .ok_or(DeviceError::Init(format!("device {} not found", device_idx)))?;

        let stream = dev.ctx.new_stream()
            .map_err(|e| DeviceError::Init(format!("create stream: {e}")))?;

        Ok(ModelContext {
            device: dev.with_stream(stream),
            cache: KernelCache::new(),
            model_name: String::new(),
        })
    }

    /// Synchronize all devices.
    pub fn synchronize_all(&self) -> Result<(), DeviceError> {
        for dev in &self.devices {
            dev.synchronize()?;
        }
        Ok(())
    }

    /// Select the best device based on SM count and compute capability.
    ///
    /// Queries the number of streaming multiprocessors and compute capability
    /// for each device, and returns the ordinal with the highest score
    /// (SM count * compute_major * 10 + compute_minor).
    pub fn best_device(&self) -> usize {
        let mut best_idx = 0;
        let mut best_score: i64 = -1;

        for (i, dev) in self.devices.iter().enumerate() {
            let sm_count = dev.ctx
                .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                .unwrap_or(1);
            let (major, minor) = dev.compute_capability;
            // Score: SM count weighted by compute capability generation.
            // Higher SM count and higher compute capability both contribute.
            let score = (sm_count as i64) * (major as i64 * 10 + minor as i64);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Tensor-parallel GEMM: split N dimension across GPUs.
    ///
    /// Computes C[M,N] = A[M,K] x B[K,N] by splitting the N dimension:
    /// - Each GPU i gets the full A matrix and B[:, i*N/nGPU : (i+1)*N/nGPU]
    /// - Each GPU computes its slice of C
    /// - Results are gathered back to GPU 0 via host-mediated copy
    ///
    /// Falls back to single-GPU GEMM when only 1 GPU is available.
    pub fn parallel_gemm(
        &self,
        a_data: &[f32],  // [M, K] on host
        b_data: &[f32],  // [K, N] on host
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, DeviceError> {
        let ngpu = self.count();
        assert_eq!(a_data.len(), (m * k) as usize, "a_data length mismatch");
        assert_eq!(b_data.len(), (k * n) as usize, "b_data length mismatch");

        // Single GPU fast path
        if ngpu == 1 {
            let dev = &self.devices[0];
            let cache = &self.caches[0];
            let a_gpu = GpuTensor::from_host(dev, a_data, Shape::from_static(&[m as usize, k as usize]), DType::F32)?;
            let b_gpu = GpuTensor::from_host(dev, b_data, Shape::from_static(&[k as usize, n as usize]), DType::F32)?;
            let mut c_gpu = GpuTensor::<f32>::zeros(dev, Shape::from_static(&[m as usize, n as usize]), DType::F32)?;
            ops::gemm(cache, dev, &a_gpu, &b_gpu, &mut c_gpu, m, n, k)?;
            dev.synchronize()?;
            return c_gpu.to_host(dev);
        }

        // Multi-GPU: split B along N dimension
        let n_per_gpu = n / ngpu as u32;
        let remainder = n % ngpu as u32;

        // Shard B columns: B is row-major [K, N], so row i is b_data[i*N .. (i+1)*N]
        // GPU j gets columns [col_start .. col_end) from each row.
        let mut c_slices: Vec<Vec<f32>> = Vec::with_capacity(ngpu);

        for gpu_idx in 0..ngpu {
            let col_start = gpu_idx as u32 * n_per_gpu + (gpu_idx as u32).min(remainder);
            let col_end = (gpu_idx as u32 + 1) * n_per_gpu + ((gpu_idx as u32) + 1).min(remainder);
            let local_n = col_end - col_start;

            if local_n == 0 {
                continue;
            }

            // Extract B shard: columns [col_start, col_end) for each of K rows
            let b_shard = extract_columns(b_data, k, n, col_start, col_end);

            let dev = &self.devices[gpu_idx];
            let cache = &self.caches[gpu_idx];

            let a_gpu = GpuTensor::from_host(dev, a_data, Shape::from_static(&[m as usize, k as usize]), DType::F32)?;
            let b_gpu = GpuTensor::from_host(dev, &b_shard, Shape::from_static(&[k as usize, local_n as usize]), DType::F32)?;
            let mut c_gpu = GpuTensor::<f32>::zeros(dev, Shape::from_static(&[m as usize, local_n as usize]), DType::F32)?;

            ops::gemm(cache, dev, &a_gpu, &b_gpu, &mut c_gpu, m, local_n, k)?;
            dev.synchronize()?;

            let c_host = c_gpu.to_host(dev)?;
            c_slices.push(c_host);
        }

        // Gather: stitch column slices back into full C[M, N]
        let mut result = vec![0.0f32; (m * n) as usize];
        let mut col_offset = 0u32;
        for (gpu_idx, c_slice) in c_slices.iter().enumerate() {
            let col_start = gpu_idx as u32 * n_per_gpu + (gpu_idx as u32).min(remainder);
            let col_end = (gpu_idx as u32 + 1) * n_per_gpu + ((gpu_idx as u32) + 1).min(remainder);
            let local_n = col_end - col_start;
            let _ = col_offset;

            for row in 0..m {
                let dst_start = (row * n + col_start) as usize;
                let src_start = (row * local_n) as usize;
                result[dst_start..dst_start + local_n as usize]
                    .copy_from_slice(&c_slice[src_start..src_start + local_n as usize]);
            }
            col_offset += local_n;
        }

        Ok(result)
    }

    /// Pre-shard a weight matrix [K, N] across GPUs along the N dimension.
    ///
    /// Returns one GpuTensor per device, each holding columns
    /// [i*N/nGPU, (i+1)*N/nGPU) of the weight matrix.
    /// This avoids re-uploading weights on every inference call.
    pub fn shard_weights(
        &self,
        weight_data: &[f32], // [K, N] on host
        k: u32,
        n: u32,
    ) -> Result<Vec<GpuTensor<f32>>, DeviceError> {
        assert_eq!(weight_data.len(), (k * n) as usize, "weight_data length mismatch");
        let ngpu = self.count();
        let n_per_gpu = n / ngpu as u32;
        let remainder = n % ngpu as u32;

        let mut shards = Vec::with_capacity(ngpu);

        for gpu_idx in 0..ngpu {
            let col_start = gpu_idx as u32 * n_per_gpu + (gpu_idx as u32).min(remainder);
            let col_end = (gpu_idx as u32 + 1) * n_per_gpu + ((gpu_idx as u32) + 1).min(remainder);
            let local_n = col_end - col_start;

            let shard_data = extract_columns(weight_data, k, n, col_start, col_end);
            let dev = &self.devices[gpu_idx];
            let tensor = GpuTensor::from_host(
                dev,
                &shard_data,
                Shape::from_static(&[k as usize, local_n as usize]),
                DType::F32,
            )?;
            shards.push(tensor);
        }

        Ok(shards)
    }

    /// Tensor-parallel GEMM using pre-sharded weights.
    ///
    /// Computes C[M,N] = A[M,K] x B[K,N] where B is already sharded
    /// across GPUs (from `shard_weights`).
    ///
    /// - `a` lives on device 0; it is copied to other devices as needed.
    /// - `b_shards` are the pre-sharded weight tensors, one per GPU.
    /// - Returns the full C[M,N] on device 0.
    pub fn parallel_gemm_sharded(
        &self,
        a: &GpuTensor<f32>,         // [M, K] on device 0
        b_shards: &[GpuTensor<f32>], // pre-sharded B, one per GPU
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<GpuTensor<f32>, DeviceError> {
        let ngpu = self.count();
        assert_eq!(b_shards.len(), ngpu, "b_shards length must equal GPU count");

        // Single GPU fast path
        if ngpu == 1 {
            let dev = &self.devices[0];
            let cache = &self.caches[0];
            let mut c_gpu = GpuTensor::<f32>::zeros(dev, Shape::from_static(&[m as usize, n as usize]), DType::F32)?;
            ops::gemm(cache, dev, a, &b_shards[0], &mut c_gpu, m, n, k)?;
            dev.synchronize()?;
            return Ok(c_gpu);
        }

        // Copy A to host once, then upload to each device
        let a_host = a.to_host(&self.devices[0])?;

        let n_per_gpu = n / ngpu as u32;
        let remainder = n % ngpu as u32;

        // Compute each shard's partial result
        let mut c_host_slices: Vec<Vec<f32>> = Vec::with_capacity(ngpu);
        let mut local_ns: Vec<u32> = Vec::with_capacity(ngpu);
        let mut col_starts: Vec<u32> = Vec::with_capacity(ngpu);

        for gpu_idx in 0..ngpu {
            let col_start = gpu_idx as u32 * n_per_gpu + (gpu_idx as u32).min(remainder);
            let col_end = (gpu_idx as u32 + 1) * n_per_gpu + ((gpu_idx as u32) + 1).min(remainder);
            let local_n = col_end - col_start;

            let dev = &self.devices[gpu_idx];
            let cache = &self.caches[gpu_idx];

            let a_local = GpuTensor::from_host(dev, &a_host, Shape::from_static(&[m as usize, k as usize]), DType::F32)?;
            let mut c_local = GpuTensor::<f32>::zeros(dev, Shape::from_static(&[m as usize, local_n as usize]), DType::F32)?;

            ops::gemm(cache, dev, &a_local, &b_shards[gpu_idx], &mut c_local, m, local_n, k)?;
            dev.synchronize()?;

            let c_host = c_local.to_host(dev)?;
            c_host_slices.push(c_host);
            local_ns.push(local_n);
            col_starts.push(col_start);
        }

        // Gather column slices into full C[M, N] on host
        let mut full_c = vec![0.0f32; (m * n) as usize];
        for (gpu_idx, c_slice) in c_host_slices.iter().enumerate() {
            let col_start = col_starts[gpu_idx];
            let local_n = local_ns[gpu_idx];
            for row in 0..m {
                let dst_start = (row * n + col_start) as usize;
                let src_start = (row * local_n) as usize;
                full_c[dst_start..dst_start + local_n as usize]
                    .copy_from_slice(&c_slice[src_start..src_start + local_n as usize]);
            }
        }

        // Upload full C to device 0
        let dev0 = &self.devices[0];
        GpuTensor::from_host(dev0, &full_c, Shape::from_static(&[m as usize, n as usize]), DType::F32)
    }

    /// All-reduce sum: gather partial results from all GPUs and sum them.
    ///
    /// Each partial result is downloaded to host, element-wise summed,
    /// and the result is uploaded to device 0.
    ///
    /// This is the host-mediated approach. Peer-to-peer or NCCL would be
    /// faster for production use.
    pub fn all_reduce_sum(
        &self,
        partial_results: &[GpuTensor<f32>], // one per GPU
    ) -> Result<GpuTensor<f32>, DeviceError> {
        if partial_results.is_empty() {
            return Err(DeviceError::Init("all_reduce_sum: no partial results".into()));
        }

        let numel = partial_results[0].numel;
        let shape = partial_results[0].shape.clone();

        // Verify all partials have the same size
        for (i, p) in partial_results.iter().enumerate() {
            if p.numel != numel {
                return Err(DeviceError::Init(format!(
                    "all_reduce_sum: partial {} has {} elements, expected {}",
                    i, p.numel, numel
                )));
            }
        }

        // Download all partials to host and sum
        let mut sum = vec![0.0f32; numel];
        for (gpu_idx, partial) in partial_results.iter().enumerate() {
            // Use the device that owns this partial
            let dev_idx = gpu_idx.min(self.devices.len() - 1);
            let dev = &self.devices[dev_idx];
            let host_data = partial.to_host(dev)?;

            for (s, &val) in sum.iter_mut().zip(host_data.iter()) {
                *s += val;
            }
        }

        // Upload summed result to device 0
        let dev0 = &self.devices[0];
        GpuTensor::from_host(dev0, &sum, shape, DType::F32)
    }

    /// Tensor-parallel info string.
    pub fn parallel_info(&self) -> String {
        let n = self.count();
        if n <= 1 {
            return "Single GPU — no parallelism available".into();
        }
        format!(
            "Tensor Parallelism: {} GPUs available\n\
             Strategy: split GEMM N-dimension across {} devices\n\
             Each GPU processes N/{} columns of the weight matrix\n\
             Results gathered via host-mediated copy to device 0",
            n, n, n
        )
    }
}

/// Extract columns [col_start, col_end) from a row-major matrix [rows, cols].
fn extract_columns(data: &[f32], rows: u32, cols: u32, col_start: u32, col_end: u32) -> Vec<f32> {
    let local_n = (col_end - col_start) as usize;
    let mut shard = Vec::with_capacity(rows as usize * local_n);
    for row in 0..rows as usize {
        let row_start = row * cols as usize + col_start as usize;
        shard.extend_from_slice(&data[row_start..row_start + local_n]);
    }
    shard
}

/// Isolated execution context for a single model.
/// Each model runs on its own CUDA stream for zero-conflict concurrency.
pub struct ModelContext {
    pub device: WarpDevice,
    pub cache: KernelCache,
    pub model_name: String,
}

impl ModelContext {
    pub fn set_name(&mut self, name: &str) {
        self.model_name = name.to_string();
    }

    pub fn synchronize(&self) -> Result<(), DeviceError> {
        self.device.synchronize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_devices() {
        match MultiDevice::new() {
            Ok(md) => {
                println!("{}", md.summary());
                assert!(md.count() >= 1);
            }
            Err(e) => println!("No CUDA devices: {e}"),
        }
    }

    #[test]
    fn device_count() {
        let count = WarpDevice::device_count().unwrap_or(0);
        println!("CUDA devices: {}", count);
    }

    #[test]
    fn best_device_selection() {
        let md = match MultiDevice::new() {
            Ok(md) => md,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };
        let best = md.best_device();
        println!("Best device: {} (out of {})", best, md.count());
        assert!(best < md.count());
    }

    /// Verify parallel_gemm produces correct results.
    ///
    /// - Creates random A[64,128] and B[128,256]
    /// - Computes C with parallel_gemm (split across available GPUs)
    /// - Computes C with single-GPU GEMM
    /// - Verifies max error < 0.01
    /// - Works with 1 GPU (falls back to single-GPU path)
    #[test]
    fn parallel_gemm_correctness() {
        let md = match MultiDevice::new() {
            Ok(md) => md,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        let m: u32 = 64;
        let k: u32 = 128;
        let n: u32 = 256;

        // Deterministic pseudo-random data (no rand crate dependency)
        let a_data: Vec<f32> = (0..(m * k) as usize)
            .map(|i| ((i * 1103515245 + 12345) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let b_data: Vec<f32> = (0..(k * n) as usize)
            .map(|i| (((i as u64).wrapping_mul(48271).wrapping_add(12345)) % 1000) as f32 / 1000.0 - 0.5)
            .collect();

        // --- Parallel GEMM ---
        let c_parallel = md.parallel_gemm(&a_data, &b_data, m, n, k).unwrap();
        assert_eq!(c_parallel.len(), (m * n) as usize);

        // --- Reference: single-GPU GEMM on device 0 ---
        let dev = &md.devices[0];
        let cache = &md.caches[0];
        let a_gpu = GpuTensor::from_host(dev, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();
        let b_gpu = GpuTensor::from_host(dev, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let mut c_gpu = GpuTensor::<f32>::zeros(dev, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        ops::gemm(cache, dev, &a_gpu, &b_gpu, &mut c_gpu, m, n, k).unwrap();
        dev.synchronize().unwrap();
        let c_reference = c_gpu.to_host(dev).unwrap();

        // --- Compare ---
        let max_err = c_parallel.iter()
            .zip(c_reference.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!(
            "parallel_gemm_correctness: {} GPUs, max error = {:.6}",
            md.count(), max_err
        );
        assert!(
            max_err < 0.01,
            "Max error {max_err} exceeds threshold 0.01"
        );
    }

    /// Verify shard_weights + parallel_gemm_sharded round-trip.
    #[test]
    fn sharded_gemm_correctness() {
        let md = match MultiDevice::new() {
            Ok(md) => md,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        let m: u32 = 32;
        let k: u32 = 64;
        let n: u32 = 128;

        let a_data: Vec<f32> = (0..(m * k) as usize)
            .map(|i| ((i * 48271 + 1) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let b_data: Vec<f32> = (0..(k * n) as usize)
            .map(|i| ((i * 16807 + 7) % 1000) as f32 / 1000.0 - 0.5)
            .collect();

        // Pre-shard weights
        let b_shards = md.shard_weights(&b_data, k, n).unwrap();
        assert_eq!(b_shards.len(), md.count());

        // Upload A to device 0
        let dev0 = &md.devices[0];
        let a_gpu = GpuTensor::from_host(dev0, &a_data, Shape::from_static(&[m as usize, k as usize]), DType::F32).unwrap();

        // Sharded GEMM
        let c_sharded = md.parallel_gemm_sharded(&a_gpu, &b_shards, m, n, k).unwrap();
        let c_result = c_sharded.to_host(dev0).unwrap();

        // Reference single-GPU
        let cache = &md.caches[0];
        let b_gpu = GpuTensor::from_host(dev0, &b_data, Shape::from_static(&[k as usize, n as usize]), DType::F32).unwrap();
        let mut c_ref_gpu = GpuTensor::<f32>::zeros(dev0, Shape::from_static(&[m as usize, n as usize]), DType::F32).unwrap();
        ops::gemm(cache, dev0, &a_gpu, &b_gpu, &mut c_ref_gpu, m, n, k).unwrap();
        dev0.synchronize().unwrap();
        let c_reference = c_ref_gpu.to_host(dev0).unwrap();

        let max_err = c_result.iter()
            .zip(c_reference.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("sharded_gemm_correctness: {} GPUs, max error = {:.6}", md.count(), max_err);
        assert!(max_err < 0.01, "Max error {max_err} exceeds threshold 0.01");
    }

    /// Verify all_reduce_sum correctness.
    #[test]
    fn all_reduce_sum_correctness() {
        let md = match MultiDevice::new() {
            Ok(md) => md,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        let n = 256usize;
        let ngpu = md.count();

        // Create partial results: GPU i has values [i+1, i+1, ..., i+1]
        let mut partials = Vec::new();
        for gpu_idx in 0..ngpu {
            let dev = &md.devices[gpu_idx];
            let val = (gpu_idx + 1) as f32;
            let data = vec![val; n];
            let tensor = GpuTensor::from_host(
                dev, &data, Shape::from_static(&[n]), DType::F32
            ).unwrap();
            partials.push(tensor);
        }

        let result = md.all_reduce_sum(&partials).unwrap();
        let result_host = result.to_host(&md.devices[0]).unwrap();

        // Expected sum: 1 + 2 + ... + ngpu = ngpu*(ngpu+1)/2
        let expected = (ngpu * (ngpu + 1) / 2) as f32;
        for &val in &result_host {
            assert!((val - expected).abs() < 1e-5,
                "all_reduce_sum: expected {expected}, got {val}");
        }
        println!("all_reduce_sum: {} GPUs, sum = {expected}", ngpu);
    }
}

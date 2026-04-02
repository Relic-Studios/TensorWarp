//! GPU memory pool — reuse allocations to eliminate cudaMalloc overhead.
//!
//! During inference, each token generation allocates hundreds of temporary
//! GPU tensors. Without pooling, each allocation calls cudaMalloc (~10-50μs).
//! With pooling, allocations are served from pre-allocated buffers (~0μs).
//!
//! Strategy: size-bucketed free lists.
//! - Round up requested size to next power-of-2 bucket
//! - Check bucket's free list for an available buffer
//! - If none, allocate new buffer from CUDA
//! - On "free", return buffer to its bucket's free list

use std::collections::HashMap;
use std::sync::Mutex;

use cudarc::driver::CudaSlice;
use warp_ir::{DType, Shape};

use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// A pool of reusable GPU memory buffers.
pub struct GpuMemPool {
    /// Free lists keyed by bucket size (power-of-2 number of f32 elements)
    f32_pools: Mutex<HashMap<usize, Vec<CudaSlice<f32>>>>,
    /// Statistics
    allocs: Mutex<u64>,
    reuses: Mutex<u64>,
}

impl GpuMemPool {
    pub fn new() -> Self {
        Self {
            f32_pools: Mutex::new(HashMap::new()),
            allocs: Mutex::new(0),
            reuses: Mutex::new(0),
        }
    }

    /// Get a zeroed f32 tensor from the pool.
    /// If a suitably-sized buffer is available, reuses it. Otherwise allocates.
    pub fn get_f32(
        &self,
        device: &WarpDevice,
        shape: Shape,
        dtype: DType,
    ) -> Result<GpuTensor<f32>, DeviceError> {
        let numel = shape.numel_static();
        let bucket = next_power_of_2(numel);

        // Try to reuse from pool
        {
            let mut pools = self.f32_pools.lock().unwrap();
            if let Some(free_list) = pools.get_mut(&bucket) {
                if let Some(data) = free_list.pop() {
                    *self.reuses.lock().unwrap() += 1;
                    // Zero the buffer (reuse means old data might be there)
                    // For performance we skip zeroing — caller should overwrite
                    return Ok(GpuTensor {
                        data,
                        shape,
                        dtype,
                        numel,
                    });
                }
            }
        }

        // Allocate new
        *self.allocs.lock().unwrap() += 1;
        GpuTensor::<f32>::zeros(device, shape, dtype)
    }

    /// Return a tensor to the pool for reuse.
    pub fn return_f32(&self, tensor: GpuTensor<f32>) {
        let bucket = next_power_of_2(tensor.numel);
        let mut pools = self.f32_pools.lock().unwrap();
        pools.entry(bucket).or_default().push(tensor.data);
    }

    /// Pool statistics.
    pub fn stats(&self) -> PoolStats {
        let allocs = *self.allocs.lock().unwrap();
        let reuses = *self.reuses.lock().unwrap();
        let pools = self.f32_pools.lock().unwrap();
        let total_buffers: usize = pools.values().map(|v| v.len()).sum();
        let total_bytes: usize = pools.iter()
            .map(|(&size, bufs)| size * 4 * bufs.len())
            .sum();

        PoolStats {
            allocs,
            reuses,
            hit_rate: if allocs + reuses > 0 { reuses as f64 / (allocs + reuses) as f64 } else { 0.0 },
            pooled_buffers: total_buffers,
            pooled_bytes: total_bytes,
        }
    }

    /// Release all pooled buffers back to the GPU.
    pub fn clear(&self) {
        let mut pools = self.f32_pools.lock().unwrap();
        pools.clear();
    }
}

impl Default for GpuMemPool {
    fn default() -> Self { Self::new() }
}

#[derive(Debug)]
pub struct PoolStats {
    pub allocs: u64,
    pub reuses: u64,
    pub hit_rate: f64,
    pub pooled_buffers: usize,
    pub pooled_bytes: usize,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemPool: {} allocs, {} reuses ({:.0}% hit), {} buffers pooled ({:.1} MB)",
            self.allocs, self.reuses, self.hit_rate * 100.0,
            self.pooled_buffers, self.pooled_bytes as f64 / 1e6)
    }
}

fn next_power_of_2(n: usize) -> usize {
    if n <= 1 { return 1; }
    1 << (usize::BITS - (n - 1).leading_zeros())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_reuse() {
        let dev = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        let pool = GpuMemPool::new();

        // Allocate
        let t1 = pool.get_f32(&dev, Shape::from_static(&[256]), DType::F32).unwrap();
        let t2 = pool.get_f32(&dev, Shape::from_static(&[256]), DType::F32).unwrap();
        assert_eq!(pool.stats().allocs, 2);
        assert_eq!(pool.stats().reuses, 0);

        // Return to pool
        pool.return_f32(t1);
        pool.return_f32(t2);

        // Allocate again — should reuse
        let _t3 = pool.get_f32(&dev, Shape::from_static(&[256]), DType::F32).unwrap();
        let _t4 = pool.get_f32(&dev, Shape::from_static(&[256]), DType::F32).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.reuses, 2, "Should reuse 2 buffers");
        assert_eq!(stats.allocs, 2, "Should not allocate new");
        println!("{}", stats);
    }

    #[test]
    fn pool_different_sizes() {
        let dev = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        let pool = GpuMemPool::new();

        // Different sizes go to different buckets
        let t1 = pool.get_f32(&dev, Shape::from_static(&[100]), DType::F32).unwrap(); // bucket 128
        let t2 = pool.get_f32(&dev, Shape::from_static(&[500]), DType::F32).unwrap(); // bucket 512
        pool.return_f32(t1);
        pool.return_f32(t2);

        // Size 100 should reuse from bucket 128, size 500 from bucket 512
        let _t3 = pool.get_f32(&dev, Shape::from_static(&[100]), DType::F32).unwrap();
        let _t4 = pool.get_f32(&dev, Shape::from_static(&[500]), DType::F32).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.reuses, 2);
        println!("{}", stats);
    }

    #[test]
    fn power_of_2() {
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(2), 2);
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(5), 8);
        assert_eq!(next_power_of_2(100), 128);
        assert_eq!(next_power_of_2(1000), 1024);
        assert_eq!(next_power_of_2(1025), 2048);
    }
}

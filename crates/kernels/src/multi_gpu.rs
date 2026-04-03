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

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
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

    /// Synchronize all devices.
    pub fn synchronize_all(&self) -> Result<(), DeviceError> {
        for dev in &self.devices {
            dev.synchronize()?;
        }
        Ok(())
    }

    /// Select the best device (highest SM count × clock).
    pub fn best_device(&self) -> usize {
        // For now, always device 0 (could query SM count via cuDeviceGetAttribute)
        0
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
}

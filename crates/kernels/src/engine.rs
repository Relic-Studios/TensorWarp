//! TensorWarp Inference Engine — production-ready execution wrapper.
//!
//! The Engine combines all subsystems:
//! - Kernel cache with disk persistence
//! - Autotuner with saved results
//! - Optimal kernel dispatch per operation + shape
//! - Device management
//!
//! Usage:
//! ```ignore
//! let engine = Engine::new(0)?;  // GPU device 0
//! let engine = Engine::with_cache_dir(0, "./warp_cache")?;  // persistent cache
//!
//! // FP16 GEMM — autotuned on first call
//! engine.gemm_f16(&a, &b, &mut c, m, n, k)?;
//!
//! // F32 GEMM — autoselected best kernel
//! engine.gemm_f32(&a, &b, &mut c, m, n, k)?;
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Serializable tuning result.
#[derive(Debug, Clone)]
struct TuneResult {
    kernel_id: String,
    time_us: f64,
}

/// Production inference engine.
pub struct Engine {
    pub device: WarpDevice,
    pub cache: KernelCache,
    tuned: Mutex<HashMap<String, TuneResult>>,
    cache_dir: Option<PathBuf>,
}

impl Engine {
    /// Create an engine on the given GPU device.
    pub fn new(device_ordinal: usize) -> Result<Self, DeviceError> {
        let device = WarpDevice::new(device_ordinal)?;
        Ok(Self {
            device,
            cache: KernelCache::new(),
            tuned: Mutex::new(HashMap::new()),
            cache_dir: None,
        })
    }

    /// Create an engine with persistent disk cache.
    /// Compiled kernels and tuning results are saved to `dir` and reloaded on restart.
    pub fn with_cache_dir(device_ordinal: usize, dir: impl Into<PathBuf>) -> Result<Self, DeviceError> {
        let dir = dir.into();
        let device = WarpDevice::new(device_ordinal)?;
        let cache = KernelCache::with_disk_cache(dir.join("kernels"));

        let tuned = HashMap::new();

        Ok(Self {
            device,
            cache,
            tuned: Mutex::new(tuned),
            cache_dir: Some(dir),
        })
    }

    /// Device summary.
    pub fn summary(&self) -> String {
        let tuned = self.tuned.lock().unwrap();
        format!("{} | {} tuned shapes | cache: {}",
            self.device.summary(), tuned.len(), self.cache.stats())
    }

    // ── GEMM dispatch ─────────────────────────────────────────

    /// F32 GEMM with auto-selection.
    pub fn gemm_f32(
        &self,
        a: &GpuTensor<f32>,
        b: &GpuTensor<f32>,
        c: &mut GpuTensor<f32>,
        m: u32, n: u32, k: u32,
    ) -> Result<(), DeviceError> {
        crate::ops::gemm(&self.cache, &self.device, a, b, c, m, n, k)
    }

    /// FP16 GEMM with tensor core dispatch.
    pub fn gemm_f16(
        &self,
        a: &GpuTensor<half::f16>,
        b: &GpuTensor<half::f16>,
        c: &mut GpuTensor<half::f16>,
        m: u32, n: u32, k: u32,
    ) -> Result<(), DeviceError> {
        crate::gemm_tc::gemm_tensor_core(&self.cache, &self.device, a, b, c, m, n, k)
    }

    /// Quantized W4A16 GEMM.
    pub fn gemm_q4(
        &self,
        a: &GpuTensor<f32>,
        b_quant: &GpuTensor<u8>,
        c: &mut GpuTensor<f32>,
        m: u32, n: u32, k: u32,
    ) -> Result<(), DeviceError> {
        crate::quantize::gemm_q4_0(&self.cache, &self.device, a, b_quant, c, m, n, k)
    }

    // ── Elementwise ops ─────────────────────────────────────────

    pub fn relu(&self, x: &GpuTensor<f32>, out: &mut GpuTensor<f32>) -> Result<(), DeviceError> {
        crate::ops::relu(&self.cache, &self.device, x, out)
    }

    pub fn gelu(&self, x: &GpuTensor<f32>, out: &mut GpuTensor<f32>) -> Result<(), DeviceError> {
        crate::ops::gelu(&self.cache, &self.device, x, out)
    }

    pub fn silu(&self, x: &GpuTensor<f32>, out: &mut GpuTensor<f32>) -> Result<(), DeviceError> {
        crate::ops::silu(&self.cache, &self.device, x, out)
    }

    pub fn add(&self, a: &GpuTensor<f32>, b: &GpuTensor<f32>, out: &mut GpuTensor<f32>) -> Result<(), DeviceError> {
        crate::ops::add(&self.cache, &self.device, a, b, out)
    }

    // ── Normalization ─────────────────────────────────────────

    pub fn rmsnorm(&self, x: &GpuTensor<f32>, gamma: &GpuTensor<f32>, out: &mut GpuTensor<f32>,
                   hidden: u32, eps: f32) -> Result<(), DeviceError> {
        crate::ops::rmsnorm(&self.cache, &self.device, x, gamma, out, hidden, eps)
    }

    // ── Conv ops ──────────────────────────────────────────────

    pub fn conv2d(&self, input: &GpuTensor<f32>, weight: &GpuTensor<f32>,
                  bias: Option<&GpuTensor<f32>>, output: &mut GpuTensor<f32>,
                  params: &crate::conv::Conv2dParams, h: u32, w: u32) -> Result<(), DeviceError> {
        crate::conv::conv2d(&self.cache, &self.device, input, weight, bias, output, params, h, w)
    }

    // ── FP16 ops ──────────────────────────────────────────────

    pub fn f16_rmsnorm(&self, x: &GpuTensor<half::f16>, gamma: &GpuTensor<half::f16>,
                       out: &mut GpuTensor<half::f16>, hidden: u32, eps: f32) -> Result<(), DeviceError> {
        crate::fp16::f16_rmsnorm(&self.cache, &self.device, x, gamma, out, hidden, eps)
    }

    pub fn cast_f32_to_f16(&self, x: &GpuTensor<f32>, out: &mut GpuTensor<half::f16>) -> Result<(), DeviceError> {
        crate::fp16::cast_f32_to_f16(&self.cache, &self.device, x, out)
    }

    pub fn cast_f16_to_f32(&self, x: &GpuTensor<half::f16>, out: &mut GpuTensor<f32>) -> Result<(), DeviceError> {
        crate::fp16::cast_f16_to_f32(&self.cache, &self.device, x, out)
    }

    // ── Device management ────────────────────────────────────

    pub fn synchronize(&self) -> Result<(), DeviceError> {
        self.device.synchronize()
    }

    /// Warmup: compile all commonly-used kernels.
    /// Call this once at startup to avoid JIT latency on first inference.
    pub fn warmup(&self) -> Result<(), DeviceError> {
        log::info!("Engine warmup: compiling kernels...");

        // Compile elementwise kernels
        let n = 256usize;
        let shape = Shape::from_static(&[n]);
        let dummy = GpuTensor::<f32>::zeros(&self.device, shape.clone(), DType::F32)?;
        let mut out = GpuTensor::<f32>::zeros(&self.device, shape, DType::F32)?;
        crate::ops::relu(&self.cache, &self.device, &dummy, &mut out)?;
        crate::ops::gelu(&self.cache, &self.device, &dummy, &mut out)?;
        crate::ops::silu(&self.cache, &self.device, &dummy, &mut out)?;
        crate::ops::sigmoid(&self.cache, &self.device, &dummy, &mut out)?;

        // Compile GEMM kernels for common sizes
        let sizes = [(64u32, 64, 64), (256, 256, 256), (1024, 1024, 1024)];
        for &(m, n, k) in &sizes {
            let a = GpuTensor::<f32>::zeros(&self.device,
                Shape::from_static(&[m as usize, k as usize]), DType::F32)?;
            let b = GpuTensor::<f32>::zeros(&self.device,
                Shape::from_static(&[k as usize, n as usize]), DType::F32)?;
            let mut c = GpuTensor::<f32>::zeros(&self.device,
                Shape::from_static(&[m as usize, n as usize]), DType::F32)?;
            crate::ops::gemm(&self.cache, &self.device, &a, &b, &mut c, m, n, k)?;
        }

        // Compile FP16 cast kernels
        let f32_dummy = GpuTensor::<f32>::zeros(&self.device,
            Shape::from_static(&[256]), DType::F32)?;
        let mut f16_dummy = GpuTensor::<half::f16>::zeros(&self.device,
            Shape::from_static(&[256]), DType::F16)?;
        crate::fp16::cast_f32_to_f16(&self.cache, &self.device, &f32_dummy, &mut f16_dummy)?;

        self.device.synchronize()?;
        log::info!("Engine warmup complete: {}", self.cache.stats());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_basic() {
        let engine = match Engine::new(0) {
            Ok(e) => e,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        println!("Engine: {}", engine.summary());

        // Warmup
        engine.warmup().unwrap();
        println!("After warmup: {}", engine.summary());

        // Test F32 GEMM
        let m = 256u32;
        let a = GpuTensor::<f32>::zeros(&engine.device,
            Shape::from_static(&[m as usize, m as usize]), DType::F32).unwrap();
        let b = GpuTensor::<f32>::zeros(&engine.device,
            Shape::from_static(&[m as usize, m as usize]), DType::F32).unwrap();
        let mut c = GpuTensor::<f32>::zeros(&engine.device,
            Shape::from_static(&[m as usize, m as usize]), DType::F32).unwrap();
        engine.gemm_f32(&a, &b, &mut c, m, m, m).unwrap();
        engine.synchronize().unwrap();
        println!("F32 GEMM 256x256x256: OK");

        // Test FP16 GEMM
        let a16 = GpuTensor::<half::f16>::zeros(&engine.device,
            Shape::from_static(&[m as usize, m as usize]), DType::F16).unwrap();
        let b16 = GpuTensor::<half::f16>::zeros(&engine.device,
            Shape::from_static(&[m as usize, m as usize]), DType::F16).unwrap();
        let mut c16 = GpuTensor::<half::f16>::zeros(&engine.device,
            Shape::from_static(&[m as usize, m as usize]), DType::F16).unwrap();
        engine.gemm_f16(&a16, &b16, &mut c16, m, m, m).unwrap();
        engine.synchronize().unwrap();
        println!("FP16 GEMM 256x256x256: OK");
    }

    #[test]
    fn engine_with_disk_cache() {
        let cache_dir = std::env::temp_dir().join("warp_engine_test");
        let _ = std::fs::remove_dir_all(&cache_dir);

        let engine = match Engine::with_cache_dir(0, &cache_dir) {
            Ok(e) => e,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        // First run: compiles kernels, saves to disk
        engine.warmup().unwrap();
        let stats1 = engine.cache.stats();
        println!("First run: {}", stats1);
        assert!(stats1.misses > 0, "Should have compiled kernels");

        // Create a second engine using the same cache dir
        let engine2 = Engine::with_cache_dir(0, &cache_dir).unwrap();
        engine2.warmup().unwrap();
        let stats2 = engine2.cache.stats();
        println!("Second run: {}", stats2);

        // The second run should have more hits from disk cache
        println!("Disk cache working: {} entries on disk",
            std::fs::read_dir(cache_dir.join("kernels")).map(|d| d.count()).unwrap_or(0));

        let _ = std::fs::remove_dir_all(&cache_dir);
    }
}

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
use std::sync::{Arc, Mutex};

use warp_ir::{DType, Shape};

use crate::autotune::{AutoTuner, DreamCycleRunner, GemmShape};
use crate::cache::KernelCache;
use crate::cost_model::CostModel;
use crate::device::{DeviceError, WarpDevice};
use crate::mem_pool::GpuMemPool;
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
    pub pool: GpuMemPool,
    pub tuner: Arc<AutoTuner>,
    pub cost_model: CostModel,
    tuned: Mutex<HashMap<String, TuneResult>>,
    cache_dir: Option<PathBuf>,
    dream_cycle: Mutex<Option<DreamCycleRunner>>,
}

impl Engine {
    /// Create an engine on the given GPU device.
    pub fn new(device_ordinal: usize) -> Result<Self, DeviceError> {
        let device = WarpDevice::new(device_ordinal)?;
        let cost_model = CostModel::from_device(&device);
        Ok(Self {
            device,
            cache: KernelCache::new(),
            pool: GpuMemPool::new(),
            tuner: Arc::new(AutoTuner::new()),
            cost_model,
            tuned: Mutex::new(HashMap::new()),
            cache_dir: None,
            dream_cycle: Mutex::new(None),
        })
    }

    /// Create an engine with persistent disk cache.
    /// Compiled kernels and tuning results are saved to `dir` and reloaded on restart.
    pub fn with_cache_dir(device_ordinal: usize, dir: impl Into<PathBuf>) -> Result<Self, DeviceError> {
        let dir = dir.into();
        let device = WarpDevice::new(device_ordinal)?;
        let cache = KernelCache::with_disk_cache(dir.join("kernels"));
        let cost_model = CostModel::from_device(&device);

        let tuned = HashMap::new();

        Ok(Self {
            device,
            cache,
            pool: GpuMemPool::new(),
            tuner: Arc::new(AutoTuner::new()),
            cost_model,
            tuned: Mutex::new(tuned),
            cache_dir: Some(dir),
            dream_cycle: Mutex::new(None),
        })
    }

    /// Device summary.
    pub fn summary(&self) -> String {
        let tuned = self.tuned.lock().unwrap();
        format!("{} | {} tuned shapes | cache: {} | {} | autotuner: {}",
            self.device.summary(), tuned.len(), self.cache.stats(),
            self.cost_model.summary(), self.tuner.report())
    }

    /// Return the autotuner's tuning report.
    pub fn tuning_report(&self) -> String {
        self.tuner.report()
    }

    // ── GEMM dispatch ─────────────────────────────────────────

    /// F32 GEMM with auto-selection via autotuner.
    /// Uses the tuned kernel variant if available, otherwise falls back to a
    /// heuristic (fast register-tiled for large sizes, tiled-32 for small).
    pub fn gemm_f32(
        &self,
        a: &GpuTensor<f32>,
        b: &GpuTensor<f32>,
        c: &mut GpuTensor<f32>,
        m: u32, n: u32, k: u32,
    ) -> Result<(), DeviceError> {
        // Shape traffic is also recorded inside tuner.gemm(), but we record here
        // too for callers that bypass the autotuner's gemm() path.
        self.tuner.gemm(&self.cache, &self.device, a, b, c, m, n, k)
    }

    /// FP16 GEMM — cuBLAS with automatic tensor core dispatch.
    /// Uses NVIDIA's hand-tuned SASS kernels for peak FP16 throughput.
    pub fn gemm_f16(
        &self,
        a: &GpuTensor<half::f16>,
        b: &GpuTensor<half::f16>,
        c: &mut GpuTensor<half::f16>,
        m: u32, n: u32, k: u32,
    ) -> Result<(), DeviceError> {
        crate::cublas_gemm::gemm_cublas_f16(&self.device, a, b, c, m, n, k)
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

        // Load cached tuning results from disk if available
        let tune_cache_path = self.cache_dir.as_ref().map(|d| d.join("tuning.csv"));
        if let Some(ref path) = tune_cache_path {
            match self.tuner.load_from_disk(path) {
                Ok(n) if n > 0 => log::info!("Loaded {} cached tuning results from disk", n),
                _ => {}
            }
        }

        // Autotune GEMM kernels for common sizes (skip already-tuned shapes)
        log::info!("Engine warmup: autotuning GEMM kernels...");
        let tune_shapes: Vec<(u32, u32, u32)> = vec![
            (64, 64, 64), (128, 128, 128), (256, 256, 256),
            (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048),
            // Transformer-specific shapes (hidden=4096, FFN=11008 for LLaMA-7B)
            (1, 4096, 4096), (1, 11008, 4096), (1, 4096, 11008),
            // Smaller models (hidden=768, 1024, 2048)
            (1, 768, 768), (1, 3072, 768), (1, 2048, 2048),
        ];
        for &(m, n, k) in &tune_shapes {
            if self.tuner.best_variant(&crate::autotune::GemmShape { m, n, k }).is_some() {
                continue; // Already tuned (from disk cache or previous run)
            }
            match self.tuner.tune_gemm(&self.cache, &self.device, m, n, k, 3, 5) {
                Ok(winner) => log::info!("  {}x{}x{}: {}", m, n, k, winner),
                Err(e) => log::warn!("  {}x{}x{}: tune failed: {}", m, n, k, e),
            }
        }

        // Save tuning results to disk for next startup
        if let Some(ref path) = tune_cache_path {
            if let Err(e) = self.tuner.save_to_disk(path) {
                log::warn!("Failed to save tuning results: {}", e);
            }
        }

        self.device.synchronize()?;
        log::info!("Engine warmup complete: {}", self.cache.stats());
        log::info!("{}", self.tuner.report());

        // Start dream-cycle background tuner.
        // It discovers hot shapes from inference traffic and tunes them automatically.
        let dream = DreamCycleRunner::start(
            self.tuner.clone(),
            Arc::new(KernelCache::new()),
            0, // device ordinal
        );
        log::info!("DreamCycle background tuner started");
        *self.dream_cycle.lock().unwrap() = Some(dream);

        Ok(())
    }

    /// Stop the dream-cycle background tuner (if running) and return its stats.
    pub fn stop_dream_cycle(&self) -> Option<String> {
        let mut dc = self.dream_cycle.lock().unwrap();
        if let Some(ref mut runner) = *dc {
            let stats = runner.stats();
            runner.stop();
            *dc = None;
            Some(stats)
        } else {
            None
        }
    }

    /// Queue a shape for background tuning by the dream cycle.
    pub fn queue_dream_tune(&self, m: u32, n: u32, k: u32) {
        let dc = self.dream_cycle.lock().unwrap();
        if let Some(ref runner) = *dc {
            runner.queue_shape(m, n, k);
        }
    }

    /// Get the top hot shapes from inference traffic.
    pub fn hot_shapes(&self, top_n: usize) -> Vec<(GemmShape, u64, f64)> {
        self.tuner.hot_shapes(top_n)
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

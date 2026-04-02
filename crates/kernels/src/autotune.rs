//! Kernel autotuner — empirical performance optimization.
//!
//! The autotuner tries multiple kernel configurations for a given
//! operation and shape, benchmarks each, and caches the winner.
//! This is Tier 3 of tiered compilation — the final optimization pass.
//!
//! Inspired by:
//! - Thompson sampling (explore vs exploit) from Didymus
//! - Dream-cycle optimization (tune during idle time)
//! - tritonBLAS cost models

use rustc_hash::FxHashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Identifies a GEMM shape for autotuning lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

/// A kernel variant that can be benchmarked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmVariant {
    /// Simple 32×32 tiled kernel
    Tiled32,
    /// Fast register-tiled 128×128 with 8×8 per thread
    RegTiled128,
    /// Tensor Core wmma (FP16)
    TensorCore,
}

impl std::fmt::Display for GemmVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GemmVariant::Tiled32 => write!(f, "Tiled-32"),
            GemmVariant::RegTiled128 => write!(f, "RegTiled-128"),
            GemmVariant::TensorCore => write!(f, "TensorCore"),
        }
    }
}

/// Result of benchmarking a kernel variant.
#[derive(Debug, Clone)]
pub struct BenchEntry {
    pub variant: GemmVariant,
    pub avg_time: Duration,
    pub tflops: f64,
    pub runs: u32,
}

/// The autotuner state.
pub struct AutoTuner {
    /// Best variant per shape.
    best: Mutex<FxHashMap<GemmShape, BenchEntry>>,
    /// All benchmark results per shape.
    history: Mutex<FxHashMap<GemmShape, Vec<BenchEntry>>>,
    /// Total tuning time spent.
    total_tune_time: Mutex<Duration>,
}

impl AutoTuner {
    pub fn new() -> Self {
        Self {
            best: Mutex::new(FxHashMap::default()),
            history: Mutex::new(FxHashMap::default()),
            total_tune_time: Mutex::new(Duration::ZERO),
        }
    }

    /// Get the best known variant for a shape, or None if not yet tuned.
    pub fn best_variant(&self, shape: &GemmShape) -> Option<GemmVariant> {
        self.best.lock().unwrap().get(shape).map(|e| e.variant)
    }

    /// Benchmark all GEMM variants for a given shape and record the winner.
    pub fn tune_gemm(
        &self,
        cache: &KernelCache,
        device: &WarpDevice,
        m: u32,
        n: u32,
        k: u32,
        warmup_iters: usize,
        bench_iters: usize,
    ) -> Result<GemmVariant, DeviceError> {
        let shape = GemmShape { m, n, k };
        let tune_start = Instant::now();

        let a_data: Vec<f32> = (0..(m * k) as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let b_data: Vec<f32> = (0..(k * n) as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();

        let a = GpuTensor::from_host(device, &a_data,
            warp_ir::Shape::from_static(&[m as usize, k as usize]), warp_ir::DType::F32)?;
        let b = GpuTensor::from_host(device, &b_data,
            warp_ir::Shape::from_static(&[k as usize, n as usize]), warp_ir::DType::F32)?;
        let mut c = GpuTensor::<f32>::zeros(device,
            warp_ir::Shape::from_static(&[m as usize, n as usize]), warp_ir::DType::F32)?;

        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let mut results = Vec::new();

        // Benchmark: Tiled-32
        {
            let variant = GemmVariant::Tiled32;
            // Warmup
            for _ in 0..warmup_iters {
                crate::ops::gemm_tiled32(cache, device, &a, &b, &mut c, m, n, k)?;
            }
            device.synchronize()?;

            let start = Instant::now();
            for _ in 0..bench_iters {
                crate::ops::gemm_tiled32(cache, device, &a, &b, &mut c, m, n, k)?;
            }
            device.synchronize()?;
            let elapsed = start.elapsed();

            let avg = elapsed / bench_iters as u32;
            let tflops = flops * bench_iters as f64 / elapsed.as_secs_f64() / 1e12;
            results.push(BenchEntry { variant, avg_time: avg, tflops, runs: bench_iters as u32 });
        }

        // Benchmark: RegTiled-128
        if m >= 128 && n >= 128 {
            let variant = GemmVariant::RegTiled128;
            for _ in 0..warmup_iters {
                crate::gemm_fast::gemm_fast(cache, device, &a, &b, &mut c, m, n, k)?;
            }
            device.synchronize()?;

            let start = Instant::now();
            for _ in 0..bench_iters {
                crate::gemm_fast::gemm_fast(cache, device, &a, &b, &mut c, m, n, k)?;
            }
            device.synchronize()?;
            let elapsed = start.elapsed();

            let avg = elapsed / bench_iters as u32;
            let tflops = flops * bench_iters as f64 / elapsed.as_secs_f64() / 1e12;
            results.push(BenchEntry { variant, avg_time: avg, tflops, runs: bench_iters as u32 });
        }

        // Find winner
        results.sort_by(|a, b| a.avg_time.cmp(&b.avg_time));
        let winner = results[0].clone();

        // Store results
        {
            let mut best = self.best.lock().unwrap();
            best.insert(shape, winner.clone());
        }
        {
            let mut history = self.history.lock().unwrap();
            history.insert(shape, results);
        }
        *self.total_tune_time.lock().unwrap() += tune_start.elapsed();

        Ok(winner.variant)
    }

    /// Auto-selecting GEMM: uses tuned variant if available, fast heuristic otherwise.
    pub fn gemm(
        &self,
        cache: &KernelCache,
        device: &WarpDevice,
        a: &GpuTensor<f32>,
        b: &GpuTensor<f32>,
        c: &mut GpuTensor<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), DeviceError> {
        let shape = GemmShape { m, n, k };
        let variant = self.best_variant(&shape).unwrap_or_else(|| {
            // Heuristic: use fast kernel for large sizes
            if m >= 128 && n >= 128 {
                GemmVariant::RegTiled128
            } else {
                GemmVariant::Tiled32
            }
        });

        match variant {
            GemmVariant::Tiled32 => crate::ops::gemm_tiled32(cache, device, a, b, c, m, n, k),
            GemmVariant::RegTiled128 => crate::gemm_fast::gemm_fast(cache, device, a, b, c, m, n, k),
            GemmVariant::TensorCore => {
                // FP32 tensors can't use TC directly, fall back
                crate::gemm_fast::gemm_fast(cache, device, a, b, c, m, n, k)
            }
        }
    }

    /// Print tuning report.
    pub fn report(&self) -> String {
        let best = self.best.lock().unwrap();
        let total_time = self.total_tune_time.lock().unwrap();
        let mut lines = vec![
            format!("=== AutoTuner Report ({} shapes tuned, {:.1}ms total) ===",
                best.len(), total_time.as_secs_f64() * 1000.0),
        ];

        let mut entries: Vec<_> = best.iter().collect();
        entries.sort_by_key(|(s, _)| (s.m, s.n, s.k));

        for (shape, entry) in entries {
            lines.push(format!(
                "  {}x{}x{}: {} ({:.3} TFLOPS, {:.3}ms)",
                shape.m, shape.n, shape.k,
                entry.variant, entry.tflops,
                entry.avg_time.as_secs_f64() * 1000.0,
            ));
        }
        lines.join("\n")
    }
}

impl Default for AutoTuner {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    #[test]
    fn autotune_gemm_shapes() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let tuner = AutoTuner::new();

        println!("\n=== Autotuning GEMM shapes ===");
        for &(m, n, k) in &[
            (64u32, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
        ] {
            let winner = tuner.tune_gemm(&cache, &dev, m, n, k, 5, 20).unwrap();
            println!("  {m}x{n}x{k}: winner = {winner}");
        }

        println!("\n{}", tuner.report());
    }

    #[test]
    fn autotune_transformer_shapes() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let tuner = AutoTuner::new();

        // Real transformer GEMM shapes (LLaMA-7B-like)
        println!("\n=== Autotuning transformer GEMM shapes ===");
        let shapes = vec![
            // QKV projections: [seq*batch, hidden] × [hidden, hidden]
            (128u32, 4096, 4096),  // seq=128, hidden=4096
            (256, 4096, 4096),
            (512, 4096, 4096),
            // FFN: [seq*batch, hidden] × [hidden, ffn_dim]
            (128, 11008, 4096),
            (256, 11008, 4096),
            // FFN down: [seq*batch, ffn_dim] × [ffn_dim, hidden]
            (128, 4096, 11008),
        ];

        for (m, n, k) in shapes {
            let winner = tuner.tune_gemm(&cache, &dev, m, n, k, 3, 10).unwrap();
            println!("  {m:4}x{n:5}x{k:5}: {winner}");
        }

        println!("\n{}", tuner.report());
    }
}

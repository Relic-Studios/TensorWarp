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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

// ── Shape Buckets ───────────────────────────────────────────────
// Shapes with similar M dimensions share tuning results.
// N and K are kept exact because they correspond to model weights
// (hidden_dim, ffn_dim) which don't change across batches.

/// A bucketed GEMM shape for approximate lookup.
/// M is bucketed to the next power-of-2-ish boundary;
/// N and K are kept exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShapeBucket {
    pub m_bucket: u32, // power-of-2 bucket for M
    pub n: u32,        // exact N
    pub k: u32,        // exact K
}

impl ShapeBucket {
    pub fn from_shape(m: u32, n: u32, k: u32) -> Self {
        let m_bucket = if m <= 1 {
            1
        } else if m <= 4 {
            4
        } else if m <= 8 {
            8
        } else if m <= 16 {
            16
        } else if m <= 32 {
            32
        } else if m <= 64 {
            64
        } else if m <= 128 {
            128
        } else if m <= 256 {
            256
        } else {
            512
        };
        Self { m_bucket, n, k }
    }
}

// ── Thompson Sampling ───────────────────────────────────────────
// Bayesian bandit-style exploration for kernel variant selection.
// Each variant accumulates wins/trials, and UCB scoring balances
// exploitation of known-good variants with exploration of untested ones.

/// Thompson sampling state for a kernel variant.
#[derive(Debug, Clone)]
pub struct VariantBelief {
    /// Number of times this variant was the fastest.
    pub wins: f64,
    /// Number of times this variant was benchmarked.
    pub trials: f64,
}

impl VariantBelief {
    /// Create with a weak prior (1 win out of 2 trials).
    pub fn new() -> Self {
        Self {
            wins: 1.0,
            trials: 2.0,
        }
    }

    /// Upper confidence bound score.
    /// Balances mean performance with exploration bonus.
    pub fn ucb(&self, total_trials: f64) -> f64 {
        let mean = self.wins / self.trials;
        let exploration = (2.0 * total_trials.ln() / self.trials).sqrt();
        mean + exploration
    }

    /// Update belief after a benchmark round.
    pub fn update(&mut self, won: bool) {
        self.trials += 1.0;
        if won {
            self.wins += 1.0;
        }
    }
}

impl Default for VariantBelief {
    fn default() -> Self {
        Self::new()
    }
}

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
    /// Split-K: partition K across blocks for small-M decode
    SplitK { splits: u32 },
    /// cuBLAS: NVIDIA's hand-tuned SASS kernels — the gold standard
    CuBLAS,
}

impl std::fmt::Display for GemmVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GemmVariant::Tiled32 => write!(f, "Tiled-32"),
            GemmVariant::RegTiled128 => write!(f, "RegTiled-128"),
            GemmVariant::TensorCore => write!(f, "TensorCore"),
            GemmVariant::SplitK { splits } => write!(f, "SplitK-{splits}"),
            GemmVariant::CuBLAS => write!(f, "cuBLAS"),
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

/// Track which shapes are actually being used in inference, and how often.
pub struct ShapeTraffic {
    /// Shape -> (call_count, total_time_estimate_us)
    pub traffic: Mutex<FxHashMap<GemmShape, (u64, f64)>>,
}

impl ShapeTraffic {
    pub fn new() -> Self {
        Self {
            traffic: Mutex::new(FxHashMap::default()),
        }
    }
}

impl Default for ShapeTraffic {
    fn default() -> Self { Self::new() }
}

/// The autotuner state.
pub struct AutoTuner {
    /// Best variant per exact shape.
    best: Mutex<FxHashMap<GemmShape, BenchEntry>>,
    /// Best variant per shape bucket (approximate lookup).
    bucket_results: Mutex<FxHashMap<ShapeBucket, BenchEntry>>,
    /// Thompson sampling beliefs per variant name.
    beliefs: Mutex<FxHashMap<String, VariantBelief>>,
    /// All benchmark results per shape.
    history: Mutex<FxHashMap<GemmShape, Vec<BenchEntry>>>,
    /// Total tuning time spent.
    total_tune_time: Mutex<Duration>,
    /// Whether to tune on first encounter of new shapes during inference.
    pub auto_tune_on_miss: bool,
    /// Warmup/bench iterations for on-demand tuning (fewer than explicit tuning).
    pub on_demand_warmup: usize,
    pub on_demand_bench: usize,
    /// Total number of gemm() dispatch calls (for exploration decay).
    total_calls: AtomicU64,
    /// Shape traffic tracking — records which shapes are dispatched during inference.
    traffic: ShapeTraffic,
}

impl AutoTuner {
    pub fn new() -> Self {
        Self {
            best: Mutex::new(FxHashMap::default()),
            bucket_results: Mutex::new(FxHashMap::default()),
            beliefs: Mutex::new(FxHashMap::default()),
            history: Mutex::new(FxHashMap::default()),
            total_tune_time: Mutex::new(Duration::ZERO),
            auto_tune_on_miss: true,
            on_demand_warmup: 2,
            on_demand_bench: 3,
            total_calls: AtomicU64::new(0),
            traffic: ShapeTraffic::new(),
        }
    }

    /// Record that a shape was dispatched during inference.
    /// Call this from gemm() so the dream cycle can discover hot shapes.
    pub fn record_shape_usage(&self, shape: &GemmShape, estimated_time_us: f64) {
        let mut traffic = self.traffic.traffic.lock().unwrap();
        let entry = traffic.entry(*shape).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += estimated_time_us;
    }

    /// Get shapes sorted by total estimated time (most impactful first).
    pub fn hot_shapes(&self, top_n: usize) -> Vec<(GemmShape, u64, f64)> {
        let traffic = self.traffic.traffic.lock().unwrap();
        let mut shapes: Vec<_> = traffic.iter()
            .map(|(s, (count, time))| (*s, *count, *time))
            .collect();
        shapes.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        shapes.truncate(top_n);
        shapes
    }

    /// Get shapes that are frequently used but not yet optimally tuned.
    /// These are shapes where we used the heuristic fallback instead of a tuned result.
    pub fn untuned_hot_shapes(&self, top_n: usize) -> Vec<GemmShape> {
        let hot = self.hot_shapes(top_n * 2);
        let best = self.best.lock().unwrap();
        hot.into_iter()
            .filter(|(shape, _, _)| !best.contains_key(shape))
            .map(|(shape, _, _)| shape)
            .take(top_n)
            .collect()
    }

    /// Get the best known variant for a shape, or None if not yet tuned.
    /// Checks exact match first, then falls back to bucket match.
    pub fn best_variant(&self, shape: &GemmShape) -> Option<GemmVariant> {
        // Exact match first
        if let Some(entry) = self.best.lock().unwrap().get(shape) {
            return Some(entry.variant);
        }
        // Bucket match (approximate)
        let bucket = ShapeBucket::from_shape(shape.m, shape.n, shape.k);
        self.bucket_results.lock().unwrap().get(&bucket).map(|e| e.variant)
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

        // Benchmark: Split-K variants (only for small M where it shines)
        if m <= 8 && k >= 256 {
            for &splits in &[4u32, 8, 16, 32] {
                let variant = GemmVariant::SplitK { splits };
                for _ in 0..warmup_iters {
                    crate::gemm_splitk::gemm_splitk(cache, device, &a, &b, &mut c, m, n, k, splits)?;
                }
                device.synchronize()?;

                let start = Instant::now();
                for _ in 0..bench_iters {
                    crate::gemm_splitk::gemm_splitk(cache, device, &a, &b, &mut c, m, n, k, splits)?;
                }
                device.synchronize()?;
                let elapsed = start.elapsed();

                let avg = elapsed / bench_iters as u32;
                let tflops = flops * bench_iters as f64 / elapsed.as_secs_f64() / 1e12;
                results.push(BenchEntry { variant, avg_time: avg, tflops, runs: bench_iters as u32 });
            }
        }

        // Benchmark: cuBLAS (the gold standard — NVIDIA's hand-tuned SASS kernels)
        {
            let variant = GemmVariant::CuBLAS;
            for _ in 0..warmup_iters {
                crate::cublas_gemm::gemm_cublas_f32(device, &a, &b, &mut c, m, n, k)?;
            }
            device.synchronize()?;

            let start = Instant::now();
            for _ in 0..bench_iters {
                crate::cublas_gemm::gemm_cublas_f32(device, &a, &b, &mut c, m, n, k)?;
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
        let winner_name = winner.variant.to_string();

        // Update Thompson sampling beliefs
        {
            let mut beliefs = self.beliefs.lock().unwrap();
            for result in &results {
                let name = result.variant.to_string();
                let belief = beliefs.entry(name.clone()).or_insert_with(VariantBelief::new);
                belief.update(name == winner_name);
            }
        }

        // Store results in exact-match table
        {
            let mut best = self.best.lock().unwrap();
            best.insert(shape, winner.clone());
        }
        // Store results in bucket table
        {
            let bucket = ShapeBucket::from_shape(m, n, k);
            let mut bucket_map = self.bucket_results.lock().unwrap();
            bucket_map.insert(bucket, winner.clone());
        }
        {
            let mut history = self.history.lock().unwrap();
            history.insert(shape, results);
        }
        *self.total_tune_time.lock().unwrap() += tune_start.elapsed();

        Ok(winner.variant)
    }

    /// Decide whether to explore a non-optimal variant for this shape.
    /// Uses epsilon-greedy with decay: explore with probability
    /// epsilon = 0.1 / (1 + num_calls/100).
    fn should_explore(&self, shape: &GemmShape) -> bool {
        let calls = self.total_calls.fetch_add(1, Ordering::Relaxed);
        // Explore ~10% of the time for the first 100 calls, then decay
        let epsilon = 0.1 / (1.0 + calls as f64 / 100.0);
        let hash = (shape.m as u64 * 7 + shape.n as u64 * 13 + shape.k as u64 * 17 + calls as u64) % 1000;
        (hash as f64 / 1000.0) < epsilon
    }

    /// Pick a random non-best variant for exploration.
    fn pick_exploration_variant(&self, m: u32, n: u32, k: u32, best: GemmVariant) -> GemmVariant {
        let mut candidates = vec![GemmVariant::Tiled32, GemmVariant::CuBLAS];
        if m >= 128 && n >= 128 {
            candidates.push(GemmVariant::RegTiled128);
        }
        if m <= 8 && k >= 256 {
            let splits = crate::gemm_splitk::auto_splits(k);
            candidates.push(GemmVariant::SplitK { splits });
        }
        // Remove the best variant, pick from remaining
        candidates.retain(|v| std::mem::discriminant(v) != std::mem::discriminant(&best));
        if candidates.is_empty() {
            best
        } else {
            // Simple deterministic selection based on call count
            let idx = self.total_calls.load(Ordering::Relaxed) as usize % candidates.len();
            candidates[idx]
        }
    }

    /// Auto-selecting GEMM: uses tuned variant if available.
    /// If `auto_tune_on_miss` is true, tunes on first encounter of a new shape.
    /// Otherwise falls back to a heuristic.
    ///
    /// Occasionally explores non-optimal variants (epsilon-greedy) to discover
    /// if a better kernel has become available for this shape.
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

        // Record shape traffic so the dream cycle can discover hot shapes.
        // Estimate time from FLOPs (rough: 1 TFLOP/s baseline -> us = 2*M*N*K / 1e6).
        let estimated_us = 2.0 * m as f64 * n as f64 * k as f64 / 1e6;
        self.record_shape_usage(&shape, estimated_us);

        let variant = match self.best_variant(&shape) {
            Some(best) => {
                // Exploration: occasionally try a non-optimal variant to discover
                // if a different kernel has become better (e.g. due to thermal
                // throttling, different data patterns, etc.)
                if self.should_explore(&shape) {
                    let explore_variant = self.pick_exploration_variant(m, n, k, best);
                    log::debug!(
                        "AutoTuner: exploring {} for {}x{}x{} (best={})",
                        explore_variant, m, n, k, best
                    );

                    // Benchmark the exploration variant and update beliefs
                    let start = Instant::now();
                    let result = Self::dispatch_variant(cache, device, a, b, c, m, n, k, explore_variant);
                    if result.is_ok() {
                        device.synchronize()?;
                        let elapsed = start.elapsed();
                        let explore_name = explore_variant.to_string();
                        let best_name = best.to_string();
                        let mut beliefs = self.beliefs.lock().unwrap();
                        // Update: the exploration variant "wins" if it ran (we
                        // don't have the best time to compare, so just record a trial)
                        let belief = beliefs.entry(explore_name).or_insert_with(VariantBelief::new);
                        belief.update(false); // conservative: count as a trial, not a win
                        let best_belief = beliefs.entry(best_name).or_insert_with(VariantBelief::new);
                        best_belief.update(true);
                    }
                    return result;
                }
                best
            }
            None if self.auto_tune_on_miss && m >= 32 && n >= 32 && k >= 32 => {
                // On-demand tuning for shapes worth benchmarking.
                // Use fewer iterations than explicit tuning to keep latency low.
                match self.tune_gemm(cache, device, m, n, k,
                                     self.on_demand_warmup, self.on_demand_bench) {
                    Ok(v) => v,
                    Err(_) => Self::heuristic_variant(m, n, k),
                }
            }
            None => Self::heuristic_variant(m, n, k),
        };

        Self::dispatch_variant(cache, device, a, b, c, m, n, k, variant)
    }

    /// Static heuristic for when tuning is disabled or fails.
    /// cuBLAS is the default for everything — it's NVIDIA's hand-tuned SASS
    /// and wins on raw GEMM throughput. Our custom kernels remain as fallbacks
    /// and for fused operations where cuBLAS can't help.
    fn heuristic_variant(m: u32, n: u32, k: u32) -> GemmVariant {
        if m <= 8 && k >= 256 {
            // Split-K still wins for tiny-M decode shapes where cuBLAS
            // launch overhead dominates
            GemmVariant::SplitK { splits: crate::gemm_splitk::auto_splits(k) }
        } else {
            // cuBLAS is fastest for everything else
            GemmVariant::CuBLAS
        }
    }

    /// Dispatch to the chosen kernel variant.
    fn dispatch_variant(
        cache: &KernelCache,
        device: &WarpDevice,
        a: &GpuTensor<f32>,
        b: &GpuTensor<f32>,
        c: &mut GpuTensor<f32>,
        m: u32, n: u32, k: u32,
        variant: GemmVariant,
    ) -> Result<(), DeviceError> {
        match variant {
            GemmVariant::Tiled32 => crate::ops::gemm_tiled32(cache, device, a, b, c, m, n, k),
            GemmVariant::RegTiled128 => crate::gemm_fast::gemm_fast(cache, device, a, b, c, m, n, k),
            GemmVariant::TensorCore => {
                // FP32 tensors can't use TC directly, fall back to fastest F32
                crate::gemm_fast::gemm_fast(cache, device, a, b, c, m, n, k)
            }
            GemmVariant::SplitK { splits } => {
                crate::gemm_splitk::gemm_splitk(cache, device, a, b, c, m, n, k, splits)
            }
            GemmVariant::CuBLAS => {
                crate::cublas_gemm::gemm_cublas_f32(device, a, b, c, m, n, k)
            }
        }
    }

    /// Save tuning results to disk for persistence across runs.
    pub fn save_to_disk(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let best = self.best.lock().unwrap();
        let mut lines = Vec::new();
        for (shape, entry) in best.iter() {
            lines.push(format!("{},{},{},{},{:.6},{:.6}",
                shape.m, shape.n, shape.k,
                match entry.variant {
                    GemmVariant::Tiled32 => "tiled32".to_string(),
                    GemmVariant::RegTiled128 => "regtiled128".to_string(),
                    GemmVariant::TensorCore => "tensorcore".to_string(),
                    GemmVariant::SplitK { splits } => format!("splitk:{splits}"),
                    GemmVariant::CuBLAS => "cublas".to_string(),
                },
                entry.avg_time.as_secs_f64(),
                entry.tflops,
            ));
        }
        std::fs::write(path, lines.join("\n"))
    }

    /// Load tuning results from disk.
    pub fn load_from_disk(&self, path: &std::path::Path) -> Result<usize, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        let mut loaded = 0;
        let mut best = self.best.lock().unwrap();
        for line in content.lines() {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 4 {
                let m: u32 = parts[0].parse().unwrap_or(0);
                let n: u32 = parts[1].parse().unwrap_or(0);
                let k: u32 = parts[2].parse().unwrap_or(0);
                let variant = if let Some(s) = parts[3].strip_prefix("splitk:") {
                    match s.parse::<u32>() {
                        Ok(splits) => GemmVariant::SplitK { splits },
                        Err(_) => continue,
                    }
                } else {
                    match parts[3] {
                        "tiled32" => GemmVariant::Tiled32,
                        "regtiled128" => GemmVariant::RegTiled128,
                        "tensorcore" => GemmVariant::TensorCore,
                        "cublas" => GemmVariant::CuBLAS,
                        _ => continue,
                    }
                };
                let avg_time = if parts.len() > 4 {
                    Duration::from_secs_f64(parts[4].parse().unwrap_or(0.0))
                } else {
                    Duration::ZERO
                };
                let tflops = if parts.len() > 5 {
                    parts[5].parse().unwrap_or(0.0)
                } else {
                    0.0
                };
                let shape = GemmShape { m, n, k };
                best.insert(shape, BenchEntry { variant, avg_time, tflops, runs: 0 });
                loaded += 1;
            }
        }
        Ok(loaded)
    }

    /// Number of shapes currently tuned.
    pub fn num_tuned(&self) -> usize {
        self.best.lock().unwrap().len()
    }

    /// Number of shape buckets with cached results.
    pub fn num_buckets(&self) -> usize {
        self.bucket_results.lock().unwrap().len()
    }

    /// Get the best variant for a bucket directly.
    pub fn best_variant_for_bucket(&self, bucket: &ShapeBucket) -> Option<GemmVariant> {
        self.bucket_results.lock().unwrap().get(bucket).map(|e| e.variant)
    }

    /// Print Thompson sampling beliefs report.
    pub fn beliefs_report(&self) -> String {
        let beliefs = self.beliefs.lock().unwrap();
        if beliefs.is_empty() {
            return "No variant beliefs yet.".to_string();
        }
        let total_trials: f64 = beliefs.values().map(|b| b.trials).sum();
        let mut lines = vec![format!("=== Variant Beliefs ({:.0} total trials) ===", total_trials)];
        let mut entries: Vec<_> = beliefs.iter().collect();
        entries.sort_by(|a, b| {
            b.1.ucb(total_trials)
                .partial_cmp(&a.1.ucb(total_trials))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (name, belief) in entries {
            lines.push(format!(
                "  {}: win_rate={:.2}, trials={:.0}, UCB={:.3}",
                name,
                belief.wins / belief.trials,
                belief.trials,
                belief.ucb(total_trials),
            ));
        }
        lines.join("\n")
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

// AutoTuner is Send+Sync: all mutable fields are behind Mutex, which is
// Send+Sync when the inner type is Send (FxHashMap<K, V> is Send when K, V are Send).
// CudaFunction (in BenchEntry) wraps Arc internals and is Send+Sync.
// We assert this explicitly so DreamCycleRunner can share it across threads.
// Safety: AutoTuner has no raw pointers or non-threadsafe interior mutability
// outside of Mutex.
unsafe impl Send for AutoTuner {}
unsafe impl Sync for AutoTuner {}

impl AutoTuner {
    /// Create with on-demand tuning disabled (heuristic fallback only).
    pub fn heuristic_only() -> Self {
        let mut t = Self::new();
        t.auto_tune_on_miss = false;
        t
    }
}

// ── Dream-Cycle Background Tuning ──────────────────────────────
// Background tuning that runs when the GPU is idle.
// The DreamCycleRunner spawns a thread that iterates through a list of
// shapes and tunes each one. In production, it would monitor GPU utilization
// via NVML and only tune during idle periods.

/// Background tuning runner that explores kernel configurations when the GPU is idle.
///
/// Named after the "dream cycle" optimization concept: the engine continuously
/// improves its performance by tuning kernels in the background, like dreaming
/// about better strategies between active inference runs.
///
/// The runner discovers shapes from real inference traffic (via AutoTuner's shape
/// tracking) and prioritizes them by estimated GPU time impact. It also accepts
/// manually queued shapes for on-demand background tuning.
pub struct DreamCycleRunner {
    /// Whether the runner is active.
    running: Arc<AtomicBool>,
    /// Handle to the background tuning thread.
    handle: Option<thread::JoinHandle<()>>,
    /// Shapes manually queued for background tuning.
    pending_shapes: Arc<Mutex<Vec<GemmShape>>>,
    /// Number of shapes successfully tuned by this runner.
    shapes_tuned: Arc<AtomicU64>,
    /// Total time spent tuning (milliseconds).
    total_tune_time_ms: Arc<AtomicU64>,
}

impl DreamCycleRunner {
    /// Start dream-cycle background tuning with dynamic shape discovery.
    ///
    /// Spawns a background thread that continuously discovers hot shapes from
    /// inference traffic and tunes them. Shapes are prioritized by total estimated
    /// GPU time (most impactful first). Already-tuned shapes are skipped.
    ///
    /// # Arguments
    /// * `tuner` - Shared autotuner (traffic is read from here, results stored here).
    /// * `cache` - Shared kernel compilation cache.
    /// * `device_ordinal` - GPU device index to tune on.
    pub fn start(
        tuner: Arc<AutoTuner>,
        cache: Arc<KernelCache>,
        device_ordinal: usize,
    ) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let pending = Arc::new(Mutex::new(Vec::new()));
        let shapes_tuned = Arc::new(AtomicU64::new(0));
        let total_tune_time = Arc::new(AtomicU64::new(0));

        let running_c = running.clone();
        let pending_c = pending.clone();
        let tuned_c = shapes_tuned.clone();
        let time_c = total_tune_time.clone();

        let handle = thread::spawn(move || {
            // Create a CUDA context on this thread.
            let device = match WarpDevice::new(device_ordinal) {
                Ok(d) => d,
                Err(e) => {
                    log::warn!("DreamCycle: failed to init device {}: {}", device_ordinal, e);
                    return;
                }
            };

            while running_c.load(Ordering::Relaxed) {
                // Phase 1: Discover hot shapes from inference traffic that need tuning.
                let hot_shapes = tuner.untuned_hot_shapes(10);

                // Phase 2: Drain manually queued shapes.
                let manual_shapes: Vec<GemmShape> = {
                    let mut p = pending_c.lock().unwrap();
                    p.drain(..).collect()
                };

                // Merge and deduplicate.
                let mut to_tune: Vec<GemmShape> = hot_shapes;
                for s in manual_shapes {
                    if !to_tune.contains(&s) {
                        to_tune.push(s);
                    }
                }

                if to_tune.is_empty() {
                    // Nothing to tune — sleep longer before re-checking.
                    thread::sleep(Duration::from_secs(5));
                    continue;
                }

                // Tune each discovered shape.
                for shape in to_tune {
                    if !running_c.load(Ordering::Relaxed) { break; }

                    let start = Instant::now();
                    match tuner.tune_gemm(&cache, &device, shape.m, shape.n, shape.k, 3, 5) {
                        Ok(winner) => {
                            let elapsed = start.elapsed();
                            tuned_c.fetch_add(1, Ordering::Relaxed);
                            time_c.fetch_add(elapsed.as_millis() as u64, Ordering::Relaxed);
                            log::info!("DreamCycle: {}x{}x{} -> {} ({:.1}ms)",
                                shape.m, shape.n, shape.k, winner, elapsed.as_secs_f64() * 1000.0);
                        }
                        Err(e) => {
                            log::warn!("DreamCycle: {}x{}x{} failed: {}",
                                shape.m, shape.n, shape.k, e);
                        }
                    }

                    // Pause between shapes to avoid hogging the GPU.
                    thread::sleep(Duration::from_millis(200));
                }
            }
        });

        Self {
            running,
            handle: Some(handle),
            pending_shapes: pending,
            shapes_tuned,
            total_tune_time_ms: total_tune_time,
        }
    }

    /// Start dream-cycle with an initial static shape list (legacy API).
    ///
    /// The shapes are queued for tuning, and the runner also discovers additional
    /// shapes from inference traffic.
    pub fn start_with_shapes(
        tuner: Arc<AutoTuner>,
        cache: Arc<KernelCache>,
        device_ordinal: usize,
        shapes_to_tune: Vec<(u32, u32, u32)>,
    ) -> Self {
        let runner = Self::start(tuner, cache, device_ordinal);
        // Queue the initial shapes.
        {
            let mut pending = runner.pending_shapes.lock().unwrap();
            for (m, n, k) in shapes_to_tune {
                pending.push(GemmShape { m, n, k });
            }
        }
        runner
    }

    /// Manually queue a shape for background tuning.
    pub fn queue_shape(&self, m: u32, n: u32, k: u32) {
        self.pending_shapes.lock().unwrap().push(GemmShape { m, n, k });
    }

    /// Stats about the dream cycle.
    pub fn stats(&self) -> String {
        format!("DreamCycle: {} shapes tuned, {:.1}ms total tuning time",
            self.shapes_tuned.load(Ordering::Relaxed),
            self.total_tune_time_ms.load(Ordering::Relaxed) as f64)
    }

    /// Stop the background tuner and wait for the thread to exit.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }

    /// Check if the background tuner is still active.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

impl Drop for DreamCycleRunner {
    fn drop(&mut self) {
        self.stop();
    }
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
        println!("\n{}", tuner.beliefs_report());
    }

    #[test]
    fn shape_bucket_lookup() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let tuner = AutoTuner::new();

        // Tune shape (128, 4096, 4096)
        let winner = tuner.tune_gemm(&cache, &dev, 128, 4096, 4096, 3, 5).unwrap();
        println!("Tuned 128x4096x4096: {}", winner);

        // Verify exact match works
        let exact = tuner.best_variant(&GemmShape { m: 128, n: 4096, k: 4096 });
        assert!(exact.is_some(), "Exact match should exist");

        // Verify bucket (128, 4096, 4096) has result
        let bucket = ShapeBucket::from_shape(128, 4096, 4096);
        assert_eq!(bucket.m_bucket, 128);
        let bucket_result = tuner.best_variant_for_bucket(&bucket);
        assert!(bucket_result.is_some(), "Bucket should have result after tuning");

        // Verify shape (100, 4096, 4096) matches bucket (128, 4096, 4096)
        let bucket_100 = ShapeBucket::from_shape(100, 4096, 4096);
        assert_eq!(bucket_100.m_bucket, 128, "M=100 should bucket to 128");
        assert_eq!(bucket_100, bucket, "Buckets should match");

        // Verify best_variant falls back to bucket for untested exact shape
        let fallback = tuner.best_variant(&GemmShape { m: 100, n: 4096, k: 4096 });
        assert!(fallback.is_some(), "Should fall back to bucket match for M=100");
        assert_eq!(fallback.unwrap(), exact.unwrap(), "Bucket fallback should match tuned variant");

        println!("Shape bucket lookup: OK");
        println!("  Tuned shapes: {}, Buckets: {}", tuner.num_tuned(), tuner.num_buckets());
    }

    #[test]
    fn dream_cycle_basic() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let tuner = Arc::new(AutoTuner::new());
        let cache = Arc::new(KernelCache::new());

        let shapes = vec![(64u32, 64, 64), (128, 128, 128)];

        // Start dream-cycle tuning in the background.
        let mut runner = DreamCycleRunner::start_with_shapes(
            tuner.clone(),
            cache.clone(),
            0,
            shapes.clone(),
        );
        assert!(runner.is_running(), "Runner should be active");

        // Wait for tuning to happen.
        std::thread::sleep(std::time::Duration::from_secs(2));

        // Stop the runner.
        runner.stop();
        assert!(!runner.is_running(), "Runner should be stopped");

        // Verify at least one shape was tuned.
        let tuned = tuner.num_tuned();
        println!("DreamCycle: tuned {} shapes", tuned);
        assert!(tuned >= 1, "DreamCycle should have tuned at least 1 shape, got {}", tuned);
    }
}

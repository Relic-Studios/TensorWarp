//! Runtime profiling for profile-guided recompilation.
//!
//! The profiler collects execution data from live inference:
//! - Per-kernel wall clock timings
//! - Input shape frequencies (which shapes actually appear)
//! - GPU hardware counters (occupancy, bandwidth, cache rates)
//! - Memory access patterns
//!
//! This data feeds into the tiered compiler — Tier 2 uses profiling
//! data from Tier 1 to make better optimization decisions.

use rustc_hash::FxHashMap;
use std::time::{Duration, Instant};
use warp_ir::{NodeId, Shape};

/// Hardware performance counters for a kernel execution.
/// In a real GPU implementation these come from CUPTI (CUDA) or
/// MTLCounterSampleBuffer (Metal). For now we track what we can
/// measure from the host side and define the interface for GPU counters.
#[derive(Debug, Clone, Default)]
pub struct HwCounters {
    /// Fraction of peak SM occupancy achieved (0.0 - 1.0).
    pub sm_occupancy: f32,
    /// L2 cache hit rate (0.0 - 1.0).
    pub l2_hit_rate: f32,
    /// DRAM bandwidth utilization as fraction of peak (0.0 - 1.0).
    pub dram_bandwidth_util: f32,
    /// Compute throughput as fraction of peak TFLOPS (0.0 - 1.0).
    pub compute_throughput: f32,
    /// Shared memory utilization (0.0 - 1.0).
    pub shared_mem_util: f32,
}

/// Profiling data collected for a single kernel execution.
#[derive(Debug, Clone)]
pub struct KernelSample {
    /// Wall-clock execution time.
    pub duration: Duration,
    /// Input shapes for this execution.
    pub input_shapes: Vec<Shape>,
    /// Hardware counters (if available).
    pub hw_counters: Option<HwCounters>,
}

/// Accumulated profile for a single kernel across many executions.
#[derive(Debug, Clone)]
pub struct KernelProfile {
    /// All observed execution times.
    samples: Vec<Duration>,
    /// Shape frequency: which input shapes appeared and how often.
    shape_histogram: FxHashMap<Vec<Shape>, u64>,
    /// Accumulated hardware counters (averaged over samples).
    hw_samples: Vec<HwCounters>,
    /// Total invocation count.
    pub invocation_count: u64,
}

impl KernelProfile {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            shape_histogram: FxHashMap::default(),
            hw_samples: Vec::new(),
            invocation_count: 0,
        }
    }

    pub fn record(&mut self, sample: KernelSample) {
        self.samples.push(sample.duration);
        *self
            .shape_histogram
            .entry(sample.input_shapes)
            .or_insert(0) += 1;
        if let Some(hw) = sample.hw_counters {
            self.hw_samples.push(hw);
        }
        self.invocation_count += 1;
    }

    /// Median execution time (more stable than mean for GPU timings).
    pub fn median_time(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let mut sorted = self.samples.clone();
        sorted.sort();
        sorted[sorted.len() / 2]
    }

    /// P99 execution time.
    pub fn p99_time(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let mut sorted = self.samples.clone();
        sorted.sort();
        let idx = (sorted.len() as f64 * 0.99) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// The most frequently observed input shape combination.
    pub fn dominant_shapes(&self) -> Option<(&Vec<Shape>, u64)> {
        self.shape_histogram
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(shapes, count)| (shapes, *count))
    }

    /// What fraction of invocations use the dominant shape.
    pub fn shape_concentration(&self) -> f64 {
        if self.invocation_count == 0 {
            return 0.0;
        }
        self.dominant_shapes()
            .map(|(_, count)| count as f64 / self.invocation_count as f64)
            .unwrap_or(0.0)
    }

    /// Top-N most common shape combinations with their frequencies.
    pub fn top_shapes(&self, n: usize) -> Vec<(Vec<Shape>, u64)> {
        let mut entries: Vec<_> = self
            .shape_histogram
            .iter()
            .map(|(s, c)| (s.clone(), *c))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(n);
        entries
    }

    /// Average hardware counters across all samples.
    pub fn avg_hw_counters(&self) -> Option<HwCounters> {
        if self.hw_samples.is_empty() {
            return None;
        }
        let n = self.hw_samples.len() as f32;
        let mut avg = HwCounters::default();
        for hw in &self.hw_samples {
            avg.sm_occupancy += hw.sm_occupancy;
            avg.l2_hit_rate += hw.l2_hit_rate;
            avg.dram_bandwidth_util += hw.dram_bandwidth_util;
            avg.compute_throughput += hw.compute_throughput;
            avg.shared_mem_util += hw.shared_mem_util;
        }
        avg.sm_occupancy /= n;
        avg.l2_hit_rate /= n;
        avg.dram_bandwidth_util /= n;
        avg.compute_throughput /= n;
        avg.shared_mem_util /= n;
        Some(avg)
    }

    /// Is this kernel memory-bound? (high compute idle, low bandwidth util)
    pub fn is_memory_bound(&self) -> bool {
        self.avg_hw_counters()
            .map(|hw| hw.dram_bandwidth_util > 0.7 && hw.compute_throughput < 0.5)
            .unwrap_or(false)
    }

    /// Is this kernel compute-bound? (high compute, low bandwidth)
    pub fn is_compute_bound(&self) -> bool {
        self.avg_hw_counters()
            .map(|hw| hw.compute_throughput > 0.7 && hw.dram_bandwidth_util < 0.5)
            .unwrap_or(false)
    }

    /// Is this kernel underutilizing the GPU?
    pub fn is_low_occupancy(&self) -> bool {
        self.avg_hw_counters()
            .map(|hw| hw.sm_occupancy < 0.5)
            .unwrap_or(false)
    }
}

/// The runtime profiler. Collects data across all kernels.
pub struct Profiler {
    /// Per-kernel profiles, keyed by NodeId.
    profiles: FxHashMap<NodeId, KernelProfile>,
    /// When profiling started.
    started_at: Instant,
    /// Total inference calls profiled.
    pub total_inferences: u64,
    /// Whether profiling is active.
    pub active: bool,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            profiles: FxHashMap::default(),
            started_at: Instant::now(),
            total_inferences: 0,
            active: true,
        }
    }

    /// Record a kernel execution sample.
    pub fn record_kernel(&mut self, node: NodeId, sample: KernelSample) {
        if !self.active {
            return;
        }
        self.profiles
            .entry(node)
            .or_insert_with(KernelProfile::new)
            .record(sample);
    }

    /// Record that one full inference pass completed.
    pub fn record_inference(&mut self) {
        self.total_inferences += 1;
    }

    /// Get the profile for a specific kernel.
    pub fn kernel_profile(&self, node: NodeId) -> Option<&KernelProfile> {
        self.profiles.get(&node)
    }

    /// Get all kernel profiles.
    pub fn all_profiles(&self) -> &FxHashMap<NodeId, KernelProfile> {
        &self.profiles
    }

    /// Total wall time since profiling started.
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Generate optimization hints from profiling data.
    /// These hints feed into the Tier 2 compiler.
    pub fn generate_hints(&self) -> Vec<OptimizationHint> {
        let mut hints = Vec::new();

        for (&node_id, profile) in &self.profiles {
            // Hint: specialize for dominant shape if concentration is high
            if profile.shape_concentration() > 0.8 {
                if let Some((shapes, _)) = profile.dominant_shapes() {
                    hints.push(OptimizationHint::SpecializeShape {
                        node: node_id,
                        shapes: shapes.clone(),
                        confidence: profile.shape_concentration(),
                    });
                }
            }

            // Hint: fuse with neighbors if memory-bound
            if profile.is_memory_bound() {
                hints.push(OptimizationHint::FuseNeighbors {
                    node: node_id,
                    reason: BottleneckReason::MemoryBound,
                });
            }

            // Hint: retile if low occupancy
            if profile.is_low_occupancy() {
                hints.push(OptimizationHint::Retile {
                    node: node_id,
                    current_occupancy: profile
                        .avg_hw_counters()
                        .map(|hw| hw.sm_occupancy)
                        .unwrap_or(0.0),
                });
            }

            // Hint: this kernel is hot (called very frequently), prioritize optimization
            if profile.invocation_count > 100 {
                let total_time: Duration = profile.samples.iter().sum();
                hints.push(OptimizationHint::HotKernel {
                    node: node_id,
                    total_time,
                    invocations: profile.invocation_count,
                });
            }
        }

        // Sort hints by impact — hot kernels first, then memory-bound, etc.
        hints.sort_by(|a, b| {
            let priority = |h: &OptimizationHint| -> u32 {
                match h {
                    OptimizationHint::HotKernel { .. } => 0,
                    OptimizationHint::FuseNeighbors { .. } => 1,
                    OptimizationHint::SpecializeShape { .. } => 2,
                    OptimizationHint::Retile { .. } => 3,
                }
            };
            priority(a).cmp(&priority(b))
        });

        hints
    }

    /// Reset all profiling data (for starting a new tier).
    pub fn reset(&mut self) {
        self.profiles.clear();
        self.started_at = Instant::now();
        self.total_inferences = 0;
    }

    /// Generate a human-readable profiling report.
    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "=== Profiler Report ({} inferences in {:.1}s) ===",
            self.total_inferences,
            self.elapsed().as_secs_f64()
        ));

        // Sort kernels by total time (hottest first)
        let mut kernel_times: Vec<_> = self
            .profiles
            .iter()
            .map(|(&node, profile)| {
                let total: Duration = profile.samples.iter().sum();
                (node, profile, total)
            })
            .collect();
        kernel_times.sort_by(|a, b| b.2.cmp(&a.2));

        for (node, profile, total) in &kernel_times {
            let shape_info = if let Some((shapes, count)) = profile.dominant_shapes() {
                let shapes_str: Vec<String> = shapes.iter().map(|s| s.to_string()).collect();
                format!(
                    "dominant: [{}] ({:.0}%)",
                    shapes_str.join(", "),
                    count as f64 / profile.invocation_count as f64 * 100.0,
                )
            } else {
                "no shape data".to_string()
            };

            let bound = if profile.is_memory_bound() {
                " [MEM-BOUND]"
            } else if profile.is_compute_bound() {
                " [COMPUTE-BOUND]"
            } else {
                ""
            };

            let occ = profile
                .avg_hw_counters()
                .map(|hw| format!(" occ={:.0}%", hw.sm_occupancy * 100.0))
                .unwrap_or_default();

            lines.push(format!(
                "  Node {:3}: {:8.1}μs total, {:6.1}μs median, {:5} calls | {}{}{} ",
                node.0,
                total.as_secs_f64() * 1e6,
                profile.median_time().as_secs_f64() * 1e6,
                profile.invocation_count,
                shape_info,
                bound,
                occ,
            ));
        }

        let hints = self.generate_hints();
        if !hints.is_empty() {
            lines.push(String::new());
            lines.push(format!("  {} optimization hints generated", hints.len()));
            for hint in hints.iter().take(5) {
                lines.push(format!("    - {hint}"));
            }
            if hints.len() > 5 {
                lines.push(format!("    ... and {} more", hints.len() - 5));
            }
        }

        lines.join("\n")
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// An optimization hint generated from profiling data.
/// Fed into the Tier 2+ compiler to guide recompilation.
#[derive(Debug, Clone)]
pub enum OptimizationHint {
    /// This kernel should be specialized for a specific shape
    /// because that shape appears in >80% of invocations.
    SpecializeShape {
        node: NodeId,
        shapes: Vec<Shape>,
        confidence: f64,
    },
    /// This kernel should be fused with its neighbors
    /// because it's bottlenecked on memory bandwidth.
    FuseNeighbors {
        node: NodeId,
        reason: BottleneckReason,
    },
    /// This kernel should be retiled for better occupancy.
    Retile {
        node: NodeId,
        current_occupancy: f32,
    },
    /// This kernel is invoked very frequently — prioritize optimizing it.
    HotKernel {
        node: NodeId,
        total_time: Duration,
        invocations: u64,
    },
}

impl std::fmt::Display for OptimizationHint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SpecializeShape {
                node, confidence, ..
            } => write!(
                f,
                "Specialize Node {} for dominant shape ({:.0}% confidence)",
                node.0,
                confidence * 100.0
            ),
            Self::FuseNeighbors { node, reason } => {
                write!(f, "Fuse neighbors of Node {} ({reason:?})", node.0)
            }
            Self::Retile {
                node,
                current_occupancy,
            } => write!(
                f,
                "Retile Node {} (occupancy: {:.0}%)",
                node.0,
                current_occupancy * 100.0
            ),
            Self::HotKernel {
                node,
                total_time,
                invocations,
            } => write!(
                f,
                "Hot kernel Node {} ({} calls, {:.1}μs total)",
                node.0,
                invocations,
                total_time.as_secs_f64() * 1e6
            ),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BottleneckReason {
    MemoryBound,
    LowOccupancy,
    KernelLaunchOverhead,
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_ir::Shape;

    #[test]
    fn basic_profiling() {
        let mut profiler = Profiler::new();
        let node = NodeId(5);

        for _ in 0..100 {
            profiler.record_kernel(
                node,
                KernelSample {
                    duration: Duration::from_micros(50),
                    input_shapes: vec![Shape::from_static(&[1, 768])],
                    hw_counters: None,
                },
            );
        }
        profiler.record_inference();

        let profile = profiler.kernel_profile(node).unwrap();
        assert_eq!(profile.invocation_count, 100);
        assert_eq!(profile.median_time(), Duration::from_micros(50));
        assert!(profile.shape_concentration() > 0.99);
    }

    #[test]
    fn shape_specialization_hint() {
        let mut profiler = Profiler::new();
        let node = NodeId(3);

        // 90% one shape, 10% another
        for _ in 0..90 {
            profiler.record_kernel(
                node,
                KernelSample {
                    duration: Duration::from_micros(100),
                    input_shapes: vec![Shape::from_static(&[1, 32, 4096])],
                    hw_counters: None,
                },
            );
        }
        for _ in 0..10 {
            profiler.record_kernel(
                node,
                KernelSample {
                    duration: Duration::from_micros(100),
                    input_shapes: vec![Shape::from_static(&[1, 64, 4096])],
                    hw_counters: None,
                },
            );
        }

        let hints = profiler.generate_hints();
        let has_specialize = hints.iter().any(|h| {
            matches!(h, OptimizationHint::SpecializeShape { node: n, .. } if n.0 == 3)
        });
        assert!(has_specialize, "Should generate shape specialization hint");
    }

    #[test]
    fn memory_bound_detection() {
        let mut profiler = Profiler::new();
        let node = NodeId(7);

        profiler.record_kernel(
            node,
            KernelSample {
                duration: Duration::from_micros(200),
                input_shapes: vec![Shape::from_static(&[1, 4096])],
                hw_counters: Some(HwCounters {
                    sm_occupancy: 0.8,
                    l2_hit_rate: 0.3,
                    dram_bandwidth_util: 0.85,
                    compute_throughput: 0.2,
                    shared_mem_util: 0.4,
                }),
            },
        );

        let profile = profiler.kernel_profile(node).unwrap();
        assert!(profile.is_memory_bound());
        assert!(!profile.is_compute_bound());
    }

    #[test]
    fn profiler_report() {
        let mut profiler = Profiler::new();
        let node = NodeId(0);

        for i in 0..50 {
            profiler.record_kernel(
                node,
                KernelSample {
                    duration: Duration::from_micros(40 + i),
                    input_shapes: vec![Shape::from_static(&[1, 768])],
                    hw_counters: Some(HwCounters {
                        sm_occupancy: 0.3,
                        l2_hit_rate: 0.5,
                        dram_bandwidth_util: 0.9,
                        compute_throughput: 0.15,
                        shared_mem_util: 0.2,
                    }),
                },
            );
        }
        profiler.total_inferences = 50;

        let report = profiler.report();
        assert!(report.contains("Profiler Report"));
        assert!(report.contains("MEM-BOUND"));
    }
}

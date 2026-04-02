//! Tiered compilation — the core innovation of Warp.
//!
//! Like JVM's C1/C2 or V8's Ignition/TurboFan, Warp compiles models
//! through multiple tiers of increasing optimization. Each tier runs
//! live inference while the next tier compiles in the background.
//! When the new tier is ready, the execution plan is hot-swapped
//! atomically — zero downtime, the model just gets faster.
//!
//! Tier 0: Naive (instant startup, no fusion)
//! Tier 1: Pattern-matched fusion (seconds)
//! Tier 2: Profile-guided recompilation (uses real execution data)
//! Tier 3: Autotuned (empirically tests configurations)
//!
//! Design insight from cognitive architecture work:
//! This is the same pattern as sense→process→learn→adapt.
//! The profiler is the sensory system, the optimizer is cognition,
//! the hot-swap is action, and the tiered progression is learning.

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use warp_codegen::{Backend, CompiledKernel, KernelConfig};
use warp_ir::{Graph, NodeId};
use warp_optimizer::OptimizationLevel;

use crate::engine::{CompilationResult, Engine, EngineError, KernelInfo};
use crate::profile::{OptimizationHint, Profiler};
use crate::schedule::ExecutionPlan;

/// Which compilation tier is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// No optimization. Instant startup.
    Tier0 = 0,
    /// Pattern-matched fusion (O1).
    Tier1 = 1,
    /// Profile-guided recompilation.
    Tier2 = 2,
    /// Autotuned kernels.
    Tier3 = 3,
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tier::Tier0 => write!(f, "Tier 0 (naive)"),
            Tier::Tier1 => write!(f, "Tier 1 (fused)"),
            Tier::Tier2 => write!(f, "Tier 2 (profile-guided)"),
            Tier::Tier3 => write!(f, "Tier 3 (autotuned)"),
        }
    }
}

/// A versioned execution plan that can be atomically swapped.
#[derive(Debug)]
pub struct VersionedPlan {
    /// The execution plan.
    pub plan: ExecutionPlan,
    /// Compiled kernels for this plan.
    pub kernels: Vec<(NodeId, CompiledKernel)>,
    /// Which tier produced this plan.
    pub tier: Tier,
    /// When this plan was compiled.
    pub compiled_at: Instant,
    /// Total memory bytes required.
    pub memory_bytes: usize,
}

/// Current state of the background compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerState {
    /// No background compilation running.
    Idle,
    /// Profiling the current tier before recompilation.
    Profiling { samples_collected: u64, target: u64 },
    /// Compiling the next tier.
    Compiling { target_tier: Tier },
    /// New plan ready, waiting for swap.
    Ready { target_tier: Tier },
    /// Reached maximum tier.
    Complete,
}

/// Thresholds that control when tier transitions happen.
#[derive(Debug, Clone)]
pub struct TierPolicy {
    /// Minimum inferences before considering Tier 1→2 transition.
    pub min_samples_for_tier2: u64,
    /// Minimum inferences before considering Tier 2→3 transition.
    pub min_samples_for_tier3: u64,
    /// Minimum time at a tier before allowing promotion.
    pub min_time_at_tier: Duration,
    /// Shape concentration threshold for specialization.
    pub shape_specialization_threshold: f64,
    /// Maximum number of autotuning configurations to try.
    pub max_autotune_configs: usize,
}

impl Default for TierPolicy {
    fn default() -> Self {
        Self {
            min_samples_for_tier2: 50,
            min_samples_for_tier3: 500,
            min_time_at_tier: Duration::from_secs(2),
            shape_specialization_threshold: 0.8,
            max_autotune_configs: 64,
        }
    }
}

/// Autotuning configuration for a kernel.
#[derive(Debug, Clone)]
pub struct TuneConfig {
    /// Block size options to try.
    pub block_sizes: Vec<[u32; 3]>,
    /// Tile size options (for matmul-like ops).
    pub tile_sizes: Vec<(usize, usize, usize)>,
    /// Number of pipeline stages to try.
    pub stages: Vec<usize>,
}

impl Default for TuneConfig {
    fn default() -> Self {
        Self {
            block_sizes: vec![
                [64, 1, 1],
                [128, 1, 1],
                [256, 1, 1],
                [512, 1, 1],
                [1024, 1, 1],
                [32, 4, 1],
                [64, 4, 1],
                [128, 2, 1],
            ],
            tile_sizes: vec![
                (32, 32, 16),
                (64, 64, 16),
                (64, 64, 32),
                (128, 64, 32),
                (128, 128, 32),
                (128, 128, 64),
                (256, 128, 32),
            ],
            stages: vec![1, 2, 3, 4],
        }
    }
}

/// The tiered compilation engine.
///
/// This is the heart of Warp. It manages the lifecycle:
/// compile Tier 0 → run → profile → compile Tier 1 → swap → profile → ...
pub struct TieredCompiler {
    /// The currently active execution plan (shared with inference threads).
    active_plan: Arc<RwLock<VersionedPlan>>,

    /// The runtime profiler.
    profiler: Arc<RwLock<Profiler>>,

    /// Current compiler state.
    state: CompilerState,

    /// Current active tier.
    current_tier: Tier,

    /// When the current tier was activated.
    tier_activated_at: Instant,

    /// Policy controlling tier transitions.
    policy: TierPolicy,

    /// The original (unoptimized) graph — kept for recompilation.
    /// Each tier recompiles from the original, not from the previous tier's output.
    original_graph: Graph,

    /// History of all compiled plans (for rollback).
    plan_history: Vec<PlanRecord>,
}

/// Record of a compiled plan for history tracking.
#[derive(Debug)]
pub struct PlanRecord {
    pub tier: Tier,
    pub compiled_at: Instant,
    pub kernel_count: usize,
    pub estimated_time_us: f64,
    pub memory_bytes: usize,
    pub hints_used: Vec<String>,
}

impl TieredCompiler {
    /// Create a new tiered compiler and immediately compile Tier 0.
    pub fn new(
        graph: Graph,
        backend: &dyn Backend,
        config: &KernelConfig,
        policy: TierPolicy,
    ) -> Result<Self, EngineError> {
        // Tier 0: compile with no optimizations — instant startup
        let mut t0_graph = graph.clone();
        let engine = Engine::new(config.clone());
        let result = engine.compile(&mut t0_graph, backend, OptimizationLevel::O0)?;

        let plan = VersionedPlan {
            plan: result.execution_plan,
            kernels: result.compiled_kernels,
            tier: Tier::Tier0,
            compiled_at: Instant::now(),
            memory_bytes: result.memory_bytes,
        };

        let record = PlanRecord {
            tier: Tier::Tier0,
            compiled_at: Instant::now(),
            kernel_count: plan.kernels.len(),
            estimated_time_us: plan.plan.estimated_time_us,
            memory_bytes: plan.memory_bytes,
            hints_used: vec![],
        };

        Ok(Self {
            active_plan: Arc::new(RwLock::new(plan)),
            profiler: Arc::new(RwLock::new(Profiler::new())),
            state: CompilerState::Profiling {
                samples_collected: 0,
                target: policy.min_samples_for_tier2,
            },
            current_tier: Tier::Tier0,
            tier_activated_at: Instant::now(),
            policy,
            original_graph: graph,
            plan_history: vec![record],
        })
    }

    /// Get a reference to the active plan (for inference threads).
    pub fn active_plan(&self) -> Arc<RwLock<VersionedPlan>> {
        Arc::clone(&self.active_plan)
    }

    /// Get a reference to the profiler (for inference threads to record data).
    pub fn profiler(&self) -> Arc<RwLock<Profiler>> {
        Arc::clone(&self.profiler)
    }

    /// Current tier.
    pub fn current_tier(&self) -> Tier {
        self.current_tier
    }

    /// Current compiler state.
    pub fn state(&self) -> CompilerState {
        self.state
    }

    /// Check if it's time to advance to the next tier, and if so, compile it.
    ///
    /// This is the main "tick" of the tiered compiler. Call it periodically
    /// (e.g., after every N inferences, or on a timer).
    ///
    /// Returns Some(new_tier) if a tier transition happened.
    pub fn maybe_advance(
        &mut self,
        backend: &dyn Backend,
        config: &KernelConfig,
    ) -> Result<Option<Tier>, EngineError> {
        // Check if we're already at max tier
        if self.current_tier == Tier::Tier3 {
            self.state = CompilerState::Complete;
            return Ok(None);
        }

        // Check time gate
        if self.tier_activated_at.elapsed() < self.policy.min_time_at_tier {
            return Ok(None);
        }

        let profiler = self.profiler.read().unwrap();
        let inferences = profiler.total_inferences;
        drop(profiler);

        let next_tier = match self.current_tier {
            Tier::Tier0 => {
                // Tier 0 → 1: always advance after minimum time (Tier 1 is fast)
                Some(Tier::Tier1)
            }
            Tier::Tier1 => {
                // Tier 1 → 2: need enough profiling samples
                if inferences >= self.policy.min_samples_for_tier2 {
                    Some(Tier::Tier2)
                } else {
                    self.state = CompilerState::Profiling {
                        samples_collected: inferences,
                        target: self.policy.min_samples_for_tier2,
                    };
                    None
                }
            }
            Tier::Tier2 => {
                // Tier 2 → 3: need many samples for autotuning
                if inferences >= self.policy.min_samples_for_tier3 {
                    Some(Tier::Tier3)
                } else {
                    self.state = CompilerState::Profiling {
                        samples_collected: inferences,
                        target: self.policy.min_samples_for_tier3,
                    };
                    None
                }
            }
            Tier::Tier3 => None,
        };

        let Some(target) = next_tier else {
            return Ok(None);
        };

        self.state = CompilerState::Compiling {
            target_tier: target,
        };

        // Compile the next tier
        let result = self.compile_tier(target, backend, config)?;

        // Hot-swap the plan
        self.swap_plan(result, target);

        Ok(Some(target))
    }

    /// Compile a specific tier.
    fn compile_tier(
        &self,
        tier: Tier,
        backend: &dyn Backend,
        config: &KernelConfig,
    ) -> Result<CompilationResult, EngineError> {
        let mut graph = self.original_graph.clone();
        let engine = Engine::new(config.clone());

        match tier {
            Tier::Tier0 => engine.compile(&mut graph, backend, OptimizationLevel::O0),
            Tier::Tier1 => engine.compile(&mut graph, backend, OptimizationLevel::O1),
            Tier::Tier2 => {
                // Profile-guided: use hints from profiler
                let profiler = self.profiler.read().unwrap();
                let hints = profiler.generate_hints();
                drop(profiler);

                // Apply shape specialization hints to the graph
                self.apply_hints(&mut graph, &hints);

                // Compile with full optimization
                engine.compile(&mut graph, backend, OptimizationLevel::O2)
            }
            Tier::Tier3 => {
                // Autotuned: compile at O3 with autotuning
                let profiler = self.profiler.read().unwrap();
                let hints = profiler.generate_hints();
                drop(profiler);

                self.apply_hints(&mut graph, &hints);
                engine.compile(&mut graph, backend, OptimizationLevel::O3)
            }
        }
    }

    /// Apply profiling hints to a graph before recompilation.
    fn apply_hints(&self, graph: &mut Graph, hints: &[OptimizationHint]) {
        for hint in hints {
            match hint {
                OptimizationHint::SpecializeShape { node, shapes, confidence } => {
                    if *confidence >= self.policy.shape_specialization_threshold {
                        // Replace dynamic dimensions with the observed static shape
                        // in the node's output values.
                        let node_data = graph.node(*node);
                        for &output in &node_data.outputs.clone() {
                            if let Some(target_shape) = shapes.first() {
                                let val = graph.value_mut(output);
                                if !val.shape.is_static() {
                                    val.shape = target_shape.clone();
                                }
                            }
                        }
                    }
                }
                // Other hints are used by the optimizer passes, not graph mutation
                _ => {}
            }
        }
    }

    /// Atomically swap to a new execution plan.
    fn swap_plan(&mut self, result: CompilationResult, tier: Tier) {
        let new_plan = VersionedPlan {
            plan: result.execution_plan,
            kernels: result.compiled_kernels,
            tier,
            compiled_at: Instant::now(),
            memory_bytes: result.memory_bytes,
        };

        let record = PlanRecord {
            tier,
            compiled_at: Instant::now(),
            kernel_count: new_plan.kernels.len(),
            estimated_time_us: new_plan.plan.estimated_time_us,
            memory_bytes: new_plan.memory_bytes,
            hints_used: vec![], // TODO: record which hints were applied
        };

        // Atomic swap — inference threads see the new plan on next read
        {
            let mut plan = self.active_plan.write().unwrap();
            *plan = new_plan;
        }

        // Reset profiler for next tier's data collection
        {
            let mut profiler = self.profiler.write().unwrap();
            profiler.reset();
        }

        self.plan_history.push(record);
        self.current_tier = tier;
        self.tier_activated_at = Instant::now();
        self.state = if tier == Tier::Tier3 {
            CompilerState::Complete
        } else {
            CompilerState::Profiling {
                samples_collected: 0,
                target: match tier {
                    Tier::Tier1 => self.policy.min_samples_for_tier2,
                    Tier::Tier2 => self.policy.min_samples_for_tier3,
                    _ => u64::MAX,
                },
            }
        };
    }

    /// Get the compilation history.
    pub fn history(&self) -> &[PlanRecord] {
        &self.plan_history
    }

    /// Generate a summary of the tiered compilation state.
    pub fn summary(&self) -> String {
        let plan = self.active_plan.read().unwrap();
        let profiler = self.profiler.read().unwrap();

        let mut lines = vec![
            format!("=== Warp Tiered Compiler ==="),
            format!("  Active tier:   {}", self.current_tier),
            format!("  State:         {:?}", self.state),
            format!("  Time at tier:  {:.1}s", self.tier_activated_at.elapsed().as_secs_f64()),
            format!(
                "  Active plan:   {} kernels, {:.2} MB, ~{:.1}μs estimated",
                plan.kernels.len(),
                plan.memory_bytes as f64 / (1024.0 * 1024.0),
                plan.plan.estimated_time_us,
            ),
            format!("  Inferences:    {}", profiler.total_inferences),
        ];

        if self.plan_history.len() > 1 {
            lines.push(String::new());
            lines.push("  Tier History:".to_string());
            for (i, record) in self.plan_history.iter().enumerate() {
                let improvement = if i > 0 {
                    let prev = &self.plan_history[i - 1];
                    if prev.estimated_time_us > 0.0 {
                        let speedup = prev.estimated_time_us / record.estimated_time_us;
                        format!(" ({:.2}x speedup)", speedup)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                lines.push(format!(
                    "    {}: {} kernels, ~{:.1}μs{}",
                    record.tier, record.kernel_count, record.estimated_time_us, improvement,
                ));
            }
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_codegen::ptx::PtxBackend;
    use warp_ir::*;

    fn build_test_graph() -> Graph {
        let mut g = Graph::new();
        let x = g.add_input(shape![1, 768], DType::F32, Some("x"));
        let w = g.add_input(shape![768, 3072], DType::F32, Some("w"));
        let bias = g.add_input(shape![3072], DType::F32, Some("bias"));

        let (_, mm) = g.add_node(
            Op::MatMul { transpose_a: false, transpose_b: false },
            &[x, w],
            &[(shape![1, 3072], DType::F32)],
            None,
        );
        let (_, add) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[mm[0], bias],
            &[(shape![1, 3072], DType::F32)],
            None,
        );
        let (_, act) = g.add_node(
            Op::Activate { activation: Activation::GeluTanh },
            &[add[0]],
            &[(shape![1, 3072], DType::F32)],
            None,
        );
        g.mark_output(act[0]);
        g
    }

    #[test]
    fn tier0_instant_startup() {
        let graph = build_test_graph();
        let backend = PtxBackend::new(89);
        let config = KernelConfig::default();
        let policy = TierPolicy::default();

        let compiler = TieredCompiler::new(graph, &backend, &config, policy).unwrap();
        assert_eq!(compiler.current_tier(), Tier::Tier0);

        let plan = compiler.active_plan();
        let plan = plan.read().unwrap();
        assert_eq!(plan.tier, Tier::Tier0);
    }

    #[test]
    fn tier0_to_tier1_advance() {
        let graph = build_test_graph();
        let backend = PtxBackend::new(89);
        let config = KernelConfig::default();
        let policy = TierPolicy {
            min_time_at_tier: Duration::from_millis(0), // no time gate for test
            ..Default::default()
        };

        let mut compiler = TieredCompiler::new(graph, &backend, &config, policy).unwrap();
        assert_eq!(compiler.current_tier(), Tier::Tier0);

        // Advance to Tier 1 (always happens after min time)
        let result = compiler.maybe_advance(&backend, &config).unwrap();
        assert_eq!(result, Some(Tier::Tier1));
        assert_eq!(compiler.current_tier(), Tier::Tier1);

        // Plan should now be Tier 1
        let plan = compiler.active_plan();
        let plan = plan.read().unwrap();
        assert_eq!(plan.tier, Tier::Tier1);
    }

    #[test]
    fn tier1_to_tier2_needs_profiling() {
        let graph = build_test_graph();
        let backend = PtxBackend::new(89);
        let config = KernelConfig::default();
        let policy = TierPolicy {
            min_time_at_tier: Duration::from_millis(0),
            min_samples_for_tier2: 10,
            ..Default::default()
        };

        let mut compiler = TieredCompiler::new(graph, &backend, &config, policy).unwrap();

        // Advance to Tier 1
        compiler.maybe_advance(&backend, &config).unwrap();
        assert_eq!(compiler.current_tier(), Tier::Tier1);

        // Try to advance to Tier 2 — should fail (not enough samples)
        let result = compiler.maybe_advance(&backend, &config).unwrap();
        assert_eq!(result, None);

        // Simulate profiling data
        {
            let profiler = compiler.profiler();
            let mut p = profiler.write().unwrap();
            for _ in 0..10 {
                p.record_inference();
            }
        }

        // Now should advance to Tier 2
        let result = compiler.maybe_advance(&backend, &config).unwrap();
        assert_eq!(result, Some(Tier::Tier2));
        assert_eq!(compiler.current_tier(), Tier::Tier2);
    }

    #[test]
    fn full_tier_progression() {
        let graph = build_test_graph();
        let backend = PtxBackend::new(89);
        let config = KernelConfig::default();
        let policy = TierPolicy {
            min_time_at_tier: Duration::from_millis(0),
            min_samples_for_tier2: 5,
            min_samples_for_tier3: 10,
            ..Default::default()
        };

        let mut compiler = TieredCompiler::new(graph, &backend, &config, policy).unwrap();

        // Tier 0 → 1
        compiler.maybe_advance(&backend, &config).unwrap();
        assert_eq!(compiler.current_tier(), Tier::Tier1);

        // Simulate profiling
        {
            let p = compiler.profiler();
            let mut p = p.write().unwrap();
            for _ in 0..5 {
                p.record_inference();
            }
        }

        // Tier 1 → 2
        compiler.maybe_advance(&backend, &config).unwrap();
        assert_eq!(compiler.current_tier(), Tier::Tier2);

        // More profiling
        {
            let p = compiler.profiler();
            let mut p = p.write().unwrap();
            for _ in 0..10 {
                p.record_inference();
            }
        }

        // Tier 2 → 3
        compiler.maybe_advance(&backend, &config).unwrap();
        assert_eq!(compiler.current_tier(), Tier::Tier3);

        // Should not advance past Tier 3
        let result = compiler.maybe_advance(&backend, &config).unwrap();
        assert_eq!(result, None);
        assert_eq!(compiler.state(), CompilerState::Complete);

        // Should have 4 records in history
        assert_eq!(compiler.history().len(), 4);

        let summary = compiler.summary();
        assert!(summary.contains("Tier 3"));
        assert!(summary.contains("Tier History"));
    }

    #[test]
    fn concurrent_plan_access() {
        let graph = build_test_graph();
        let backend = PtxBackend::new(89);
        let config = KernelConfig::default();
        let policy = TierPolicy::default();

        let compiler = TieredCompiler::new(graph, &backend, &config, policy).unwrap();
        let plan = compiler.active_plan();

        // Simulate concurrent reads (inference threads)
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let plan = Arc::clone(&plan);
                std::thread::spawn(move || {
                    let p = plan.read().unwrap();
                    assert_eq!(p.tier, Tier::Tier0);
                    p.plan.num_kernel_launches()
                })
            })
            .collect();

        for handle in handles {
            let _ = handle.join().unwrap();
        }
    }
}

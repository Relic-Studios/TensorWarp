//! Analytical cost model for GEMM kernel configuration scoring.
//!
//! Predicts runtime without actually running the kernel.
//! Used to pre-screen 100+ configs down to top-10 for empirical benchmarking.
//! This is the brain behind TensorWarp's autotuner — it eliminates >90% of
//! the search space before any GPU time is spent.

use crate::device::WarpDevice;

/// GPU hardware parameters used for cost estimation.
pub struct CostModel {
    /// Peak FP32 throughput in TFLOPS (e.g., 82.6 for RTX 4090)
    pub peak_tflops_f32: f64,
    /// Peak memory bandwidth in GB/s (e.g., 1008 for RTX 4090)
    pub peak_bandwidth_gb: f64,
    /// Number of streaming multiprocessors (e.g., 128 for RTX 4090)
    pub num_sms: u32,
    /// Maximum shared memory per SM in bytes (e.g., 100KB)
    pub max_shared_mem_per_sm: u32,
    /// Maximum registers per SM (e.g., 65536)
    pub max_registers_per_sm: u32,
    /// Maximum warps per SM (e.g., 48)
    pub max_warps_per_sm: u32,
}

/// A kernel configuration to score.
#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub num_warps: u32,     // 1-8
    pub num_stages: u32,    // 1-4 (pipeline depth)
    pub split_k: u32,       // 1 = no split
}

/// Predicted performance for a config.
pub struct CostPrediction {
    pub config: KernelConfig,
    pub predicted_time_us: f64,
    pub compute_bound_time_us: f64,
    pub memory_bound_time_us: f64,
    pub occupancy: f64,         // 0.0 - 1.0
    pub grid_efficiency: f64,   // how well grid fills the GPU
}

impl CostModel {
    /// Build a cost model from device properties.
    /// Queries compute capability and sets architecture-appropriate parameters.
    pub fn from_device(device: &WarpDevice) -> Self {
        let (major, _minor) = device.compute_capability;
        // RTX 4090 (SM89): 82.6 TFLOPS F32, 1008 GB/s, 128 SMs
        // RTX 3090 (SM86): 35.6 TFLOPS F32, 936 GB/s, 82 SMs
        // Fallback to reasonable defaults
        let (tflops, bw, sms) = match major {
            8 => (40.0, 900.0, 80),    // Ampere
            9 => (80.0, 1000.0, 128),  // Ada Lovelace
            _ => (20.0, 500.0, 40),    // fallback
        };
        Self {
            peak_tflops_f32: tflops,
            peak_bandwidth_gb: bw,
            num_sms: sms,
            max_shared_mem_per_sm: 100 * 1024,
            max_registers_per_sm: 65536,
            max_warps_per_sm: 48,
        }
    }

    /// Score a kernel configuration for a given GEMM shape (M, N, K).
    /// Returns a prediction with estimated time, occupancy, and grid efficiency.
    pub fn score(&self, config: &KernelConfig, m: u32, n: u32, k: u32) -> CostPrediction {
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let bytes = ((m * k + k * n + m * n) * 4) as f64; // f32

        // Compute time: how long the arithmetic takes at peak throughput
        let compute_time = flops / (self.peak_tflops_f32 * 1e12) * 1e6; // microseconds

        // Memory time: how long the data movement takes at peak bandwidth
        let memory_time = bytes / (self.peak_bandwidth_gb * 1e9) * 1e6;

        // Occupancy: limited by shared memory and registers
        let smem_per_block = config.block_m * config.block_k * 4 * config.num_stages
                           + config.block_k * config.block_n * 4 * config.num_stages; // bytes
        let regs_per_block = config.num_warps * 32 * 64; // estimate 64 regs per thread

        let blocks_per_sm_by_smem = self.max_shared_mem_per_sm / smem_per_block.max(1);
        let blocks_per_sm_by_regs = self.max_registers_per_sm / regs_per_block.max(1);
        let blocks_per_sm_by_warps = self.max_warps_per_sm / config.num_warps.max(1);
        let blocks_per_sm = blocks_per_sm_by_smem
            .min(blocks_per_sm_by_regs)
            .min(blocks_per_sm_by_warps)
            .max(1);
        let occupancy = (blocks_per_sm * config.num_warps) as f64 / self.max_warps_per_sm as f64;

        // Grid size and efficiency
        let grid_m = (m + config.block_m - 1) / config.block_m;
        let grid_n = (n + config.block_n - 1) / config.block_n;
        let total_blocks = grid_m * grid_n * config.split_k;
        let max_concurrent = self.num_sms as f64 * blocks_per_sm as f64;
        let waves = (total_blocks as f64 / max_concurrent).ceil();
        let grid_efficiency = total_blocks as f64 / (waves * max_concurrent);

        // Predicted time = max(compute, memory) / occupancy / grid_efficiency
        let predicted = compute_time.max(memory_time) / occupancy.max(0.1) / grid_efficiency.max(0.1);

        CostPrediction {
            config: config.clone(),
            predicted_time_us: predicted,
            compute_bound_time_us: compute_time,
            memory_bound_time_us: memory_time,
            occupancy,
            grid_efficiency,
        }
    }

    /// Generate all valid kernel configurations for a given GEMM shape.
    /// Filters out configs that exceed hardware limits (shared memory, etc.).
    pub fn generate_configs(&self, m: u32, n: u32, _k: u32) -> Vec<KernelConfig> {
        let mut configs = Vec::new();
        for &bm in &[32u32, 64, 128, 256] {
            for &bn in &[32u32, 64, 128, 256] {
                for &bk in &[16u32, 32, 64] {
                    for &warps in &[2u32, 4, 8] {
                        for &stages in &[1u32, 2, 3] {
                            let split_ks: Vec<u32> = if m <= 8 {
                                vec![1, 4, 8, 16]
                            } else {
                                vec![1]
                            };
                            for &sk in &split_ks {
                                // Filter invalid configs
                                let smem = (bm * bk + bk * bn) * 4 * stages;
                                if smem > self.max_shared_mem_per_sm {
                                    continue;
                                }
                                // Don't use tiles bigger than the problem dimension
                                if bm > m && m > 0 {
                                    continue;
                                }
                                if bn > n && n > 0 {
                                    continue;
                                }
                                configs.push(KernelConfig {
                                    block_m: bm,
                                    block_n: bn,
                                    block_k: bk,
                                    num_warps: warps,
                                    num_stages: stages,
                                    split_k: sk,
                                });
                            }
                        }
                    }
                }
            }
        }
        configs
    }

    /// Score all configs and return the top-k by predicted time (fastest first).
    pub fn top_k(&self, configs: &[KernelConfig], m: u32, n: u32, k: u32, top: usize) -> Vec<CostPrediction> {
        let mut predictions: Vec<CostPrediction> = configs.iter()
            .map(|c| self.score(c, m, n, k))
            .collect();
        predictions.sort_by(|a, b| {
            a.predicted_time_us
                .partial_cmp(&b.predicted_time_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        predictions.truncate(top);
        predictions
    }

    /// Summary string for display in engine info.
    pub fn summary(&self) -> String {
        format!(
            "CostModel(peak={:.1} TFLOPS, BW={:.0} GB/s, SMs={})",
            self.peak_tflops_f32, self.peak_bandwidth_gb, self.num_sms
        )
    }
}

impl std::fmt::Display for KernelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}x{}x{}_w{}_s{}_sk{}",
            self.block_m, self.block_n, self.block_k,
            self.num_warps, self.num_stages, self.split_k
        )
    }
}

impl std::fmt::Display for CostPrediction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {:.2}us (compute={:.2}us, mem={:.2}us, occ={:.2}, grid_eff={:.2})",
            self.config, self.predicted_time_us,
            self.compute_bound_time_us, self.memory_bound_time_us,
            self.occupancy, self.grid_efficiency
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_model_scoring() {
        let device = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        let model = CostModel::from_device(&device);
        println!("Cost model: {}", model.summary());

        // Generate configs for 256x256x256
        let configs = model.generate_configs(256, 256, 256);
        println!("Generated {} configs for 256x256x256", configs.len());
        assert!(!configs.is_empty(), "Should generate at least some configs");

        // Score all and get top 5
        let top5 = model.top_k(&configs, 256, 256, 256, 5);
        assert_eq!(top5.len(), 5.min(configs.len()));

        println!("\nTop-5 configs for 256x256x256:");
        for (i, pred) in top5.iter().enumerate() {
            println!("  #{}: {}", i + 1, pred);
        }

        // Verify predictions are reasonable
        let best = &top5[0];
        assert!(best.predicted_time_us > 0.0, "Predicted time must be positive");
        assert!(best.occupancy > 0.0, "Occupancy must be positive");
        assert!(best.grid_efficiency > 0.0, "Grid efficiency must be positive");
        assert!(best.grid_efficiency <= 1.0, "Grid efficiency must be <= 1.0");
    }
}

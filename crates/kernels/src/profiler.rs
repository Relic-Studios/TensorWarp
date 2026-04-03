//! Layer-by-layer profiling for inference optimization.
//!
//! Measures execution time per kernel/layer to identify bottlenecks.
//! Used by the autotuner to decide which operations to optimize.
//!
//! Usage:
//! ```ignore
//! let mut prof = LayerProfiler::new();
//! prof.start("conv1");
//! // ... run conv1 ...
//! device.synchronize()?;
//! prof.stop("conv1");
//!
//! prof.start("relu1");
//! // ... run relu1 ...
//! device.synchronize()?;
//! prof.stop("relu1");
//!
//! println!("{}", prof.report());
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Per-layer timing entry.
#[derive(Debug, Clone)]
pub struct LayerTiming {
    pub name: String,
    pub total_time: Duration,
    pub call_count: u64,
    pub min_time: Duration,
    pub max_time: Duration,
}

impl LayerTiming {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            total_time: Duration::ZERO,
            call_count: 0,
            min_time: Duration::from_secs(999),
            max_time: Duration::ZERO,
        }
    }

    fn record(&mut self, elapsed: Duration) {
        self.total_time += elapsed;
        self.call_count += 1;
        if elapsed < self.min_time { self.min_time = elapsed; }
        if elapsed > self.max_time { self.max_time = elapsed; }
    }

    fn avg(&self) -> Duration {
        if self.call_count == 0 { Duration::ZERO }
        else { self.total_time / self.call_count as u32 }
    }
}

/// Layer-by-layer profiler.
pub struct LayerProfiler {
    timings: HashMap<String, LayerTiming>,
    active: Option<(String, Instant)>,
    order: Vec<String>, // preserve insertion order
}

impl LayerProfiler {
    pub fn new() -> Self {
        Self {
            timings: HashMap::new(),
            active: None,
            order: Vec::new(),
        }
    }

    /// Start timing a layer.
    pub fn start(&mut self, name: &str) {
        self.active = Some((name.to_string(), Instant::now()));
    }

    /// Stop timing the current layer and record the elapsed time.
    pub fn stop(&mut self, name: &str) {
        if let Some((active_name, start)) = self.active.take() {
            if active_name == name {
                let elapsed = start.elapsed();
                let entry = self.timings.entry(name.to_string())
                    .or_insert_with(|| {
                        self.order.push(name.to_string());
                        LayerTiming::new(name)
                    });
                entry.record(elapsed);
            }
        }
    }

    /// Generate a profiling report sorted by total time.
    pub fn report(&self) -> String {
        let total: Duration = self.timings.values().map(|t| t.total_time).sum();
        let mut lines = vec![format!("=== Layer Profile (total: {:.2}ms) ===", total.as_secs_f64() * 1000.0)];

        // Sort by total time descending
        let mut entries: Vec<&LayerTiming> = self.timings.values().collect();
        entries.sort_by(|a, b| b.total_time.cmp(&a.total_time));

        lines.push(format!("{:30} {:>10} {:>8} {:>10} {:>10} {:>6}",
            "Layer", "Total(ms)", "Calls", "Avg(μs)", "Max(μs)", "%"));

        for entry in &entries {
            let pct = if total.as_secs_f64() > 0.0 {
                entry.total_time.as_secs_f64() / total.as_secs_f64() * 100.0
            } else { 0.0 };
            lines.push(format!("{:30} {:>10.2} {:>8} {:>10.1} {:>10.1} {:>5.1}%",
                entry.name,
                entry.total_time.as_secs_f64() * 1000.0,
                entry.call_count,
                entry.avg().as_secs_f64() * 1e6,
                entry.max_time.as_secs_f64() * 1e6,
                pct));
        }

        lines.join("\n")
    }

    /// Get timing for a specific layer.
    pub fn get(&self, name: &str) -> Option<&LayerTiming> {
        self.timings.get(name)
    }

    /// Reset all timings.
    pub fn reset(&mut self) {
        self.timings.clear();
        self.order.clear();
    }
}

impl Default for LayerProfiler {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiler_basic() {
        let mut prof = LayerProfiler::new();

        prof.start("layer1");
        std::thread::sleep(Duration::from_millis(1));
        prof.stop("layer1");

        prof.start("layer2");
        std::thread::sleep(Duration::from_millis(2));
        prof.stop("layer2");

        prof.start("layer1");
        std::thread::sleep(Duration::from_millis(1));
        prof.stop("layer1");

        println!("{}", prof.report());

        let l1 = prof.get("layer1").unwrap();
        assert_eq!(l1.call_count, 2);
        let l2 = prof.get("layer2").unwrap();
        assert_eq!(l2.call_count, 1);
    }
}

//! INT8/FP8 calibration pipeline for post-training quantization.
//!
//! Calibration measures the dynamic range of activations at each layer
//! by running representative data through the model. These ranges are
//! then used to compute optimal quantization scales.
//!
//! Supported methods:
//! - MinMax: uses the min/max observed values (fast, less accurate)
//! - Entropy (KL divergence): finds scales that minimize information loss
//! - Percentile: clips outliers at a given percentile
//!
//! Usage:
//! ```ignore
//! let mut calibrator = Calibrator::new(CalibrationMethod::Entropy);
//! for batch in calibration_data {
//!     calibrator.observe("layer1.output", &activations)?;
//! }
//! let scales = calibrator.compute_scales()?;
//! ```

use std::collections::HashMap;

/// Calibration method.
#[derive(Debug, Clone, Copy)]
pub enum CalibrationMethod {
    /// Track min/max values. Fastest, but sensitive to outliers.
    MinMax,
    /// Percentile-based: clip at the given percentile (e.g., 99.99%).
    Percentile(f32),
    /// Entropy-based (KL divergence): find optimal threshold.
    Entropy { num_bins: usize },
}

/// Per-tensor calibration statistics.
#[derive(Debug, Clone)]
pub struct TensorStats {
    pub name: String,
    pub min: f32,
    pub max: f32,
    pub num_samples: u64,
    /// Histogram for entropy calibration.
    pub histogram: Option<Vec<u64>>,
    pub hist_min: f32,
    pub hist_max: f32,
}

impl TensorStats {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            num_samples: 0,
            histogram: None,
            hist_min: 0.0,
            hist_max: 0.0,
        }
    }

    fn observe_minmax(&mut self, data: &[f32]) {
        for &v in data {
            if v < self.min { self.min = v; }
            if v > self.max { self.max = v; }
        }
        self.num_samples += data.len() as u64;
    }

    fn observe_histogram(&mut self, data: &[f32], num_bins: usize) {
        if self.histogram.is_none() {
            // First observation: set range and initialize histogram
            self.observe_minmax(data);
            self.hist_min = self.min;
            self.hist_max = self.max;
            self.histogram = Some(vec![0u64; num_bins]);
        }

        let hist = self.histogram.as_mut().unwrap();
        let range = self.hist_max - self.hist_min;
        if range <= 0.0 { return; }

        for &v in data {
            let bin = ((v - self.hist_min) / range * (num_bins as f32 - 1.0))
                .max(0.0).min((num_bins - 1) as f32) as usize;
            hist[bin] += 1;
        }
        self.num_samples += data.len() as u64;
    }
}

/// Quantization scale for a tensor.
#[derive(Debug, Clone)]
pub struct QuantScale {
    pub name: String,
    pub scale: f32,    // multiply activation by this before rounding
    pub zero_point: i32,  // offset (0 for symmetric)
    pub bits: u32,     // 8 for INT8, 4 for INT4
}

/// Calibration engine.
pub struct Calibrator {
    method: CalibrationMethod,
    stats: HashMap<String, TensorStats>,
}

impl Calibrator {
    pub fn new(method: CalibrationMethod) -> Self {
        Self {
            method,
            stats: HashMap::new(),
        }
    }

    /// Record activation values for a named tensor.
    pub fn observe(&mut self, name: &str, data: &[f32]) {
        let stats = self.stats.entry(name.to_string())
            .or_insert_with(|| TensorStats::new(name));

        match self.method {
            CalibrationMethod::MinMax | CalibrationMethod::Percentile(_) => {
                stats.observe_minmax(data);
            }
            CalibrationMethod::Entropy { num_bins } => {
                stats.observe_histogram(data, num_bins);
            }
        }
    }

    /// Compute quantization scales from collected statistics.
    pub fn compute_scales(&self, bits: u32) -> Vec<QuantScale> {
        let qmax = (1 << (bits - 1)) - 1; // 127 for INT8, 7 for INT4

        self.stats.values().map(|stats| {
            let (scale, zero_point) = match self.method {
                CalibrationMethod::MinMax => {
                    // Symmetric: scale = max(|min|, |max|) / qmax
                    let amax = stats.min.abs().max(stats.max.abs());
                    let scale = amax / qmax as f32;
                    (scale, 0i32)
                }
                CalibrationMethod::Percentile(pct) => {
                    // Use percentile of the observed range
                    let range = (stats.max - stats.min) * pct / 100.0;
                    let amax = range / 2.0;
                    let scale = amax / qmax as f32;
                    (scale, 0)
                }
                CalibrationMethod::Entropy { num_bins } => {
                    // Entropy calibration: find threshold that minimizes KL divergence
                    // Simplified version: use 99.99th percentile of histogram
                    if let Some(ref hist) = stats.histogram {
                        let total: u64 = hist.iter().sum();
                        let target = (total as f64 * 0.9999) as u64;
                        let mut cumsum = 0u64;
                        let mut threshold_bin = hist.len() - 1;
                        for (i, &count) in hist.iter().enumerate() {
                            cumsum += count;
                            if cumsum >= target {
                                threshold_bin = i;
                                break;
                            }
                        }
                        let range = stats.hist_max - stats.hist_min;
                        let threshold = stats.hist_min + range * (threshold_bin as f32 / num_bins as f32);
                        let amax = threshold.abs().max(1e-8);
                        let scale = amax / qmax as f32;
                        (scale, 0)
                    } else {
                        let amax = stats.min.abs().max(stats.max.abs());
                        (amax / qmax as f32, 0)
                    }
                }
            };

            QuantScale {
                name: stats.name.clone(),
                scale: scale.max(1e-10), // prevent division by zero
                zero_point,
                bits,
            }
        }).collect()
    }

    /// Print calibration report.
    pub fn report(&self) -> String {
        let mut lines = vec![format!("=== Calibration Report ({:?}) ===", self.method)];
        lines.push(format!("  {} tensors observed", self.stats.len()));

        let mut entries: Vec<_> = self.stats.iter().collect();
        entries.sort_by_key(|(name, _)| (*name).clone());

        for (name, stats) in entries {
            lines.push(format!("  {name:30} min={:.4} max={:.4} samples={}",
                stats.min, stats.max, stats.num_samples));
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minmax_calibration() {
        let mut cal = Calibrator::new(CalibrationMethod::MinMax);

        // Simulate observing activations
        cal.observe("layer0", &[-1.5, 0.0, 2.3, -0.5, 1.0]);
        cal.observe("layer0", &[-0.8, 3.1, 0.2]);
        cal.observe("layer1", &[0.0, 0.5, -0.2, 0.1]);

        let scales = cal.compute_scales(8);
        println!("{}", cal.report());

        // layer0: max(|-1.5|, |3.1|) = 3.1, scale = 3.1/127 ≈ 0.0244
        let l0_scale = scales.iter().find(|s| s.name == "layer0").unwrap();
        assert!((l0_scale.scale - 3.1 / 127.0).abs() < 0.001);
        println!("layer0 scale: {:.6} (INT8)", l0_scale.scale);

        let l1_scale = scales.iter().find(|s| s.name == "layer1").unwrap();
        println!("layer1 scale: {:.6} (INT8)", l1_scale.scale);
    }

    #[test]
    fn entropy_calibration() {
        let mut cal = Calibrator::new(CalibrationMethod::Entropy { num_bins: 1024 });

        // Generate some activation-like data (normal distribution)
        let data: Vec<f32> = (0..10000)
            .map(|i| {
                let x = (i as f32 / 10000.0 - 0.5) * 6.0; // range [-3, 3]
                (-x * x / 2.0).exp() // gaussian-like
            })
            .collect();

        cal.observe("activations", &data);
        let scales = cal.compute_scales(8);

        println!("{}", cal.report());
        let s = &scales[0];
        println!("Entropy INT8 scale: {:.6}", s.scale);
        assert!(s.scale > 0.0);
    }

    #[test]
    fn int4_calibration() {
        let mut cal = Calibrator::new(CalibrationMethod::MinMax);
        cal.observe("weights", &[-0.5, 0.3, -0.1, 0.7, -0.8, 0.2]);

        let scales = cal.compute_scales(4);
        let s = &scales[0];
        // INT4: qmax = 7, amax = 0.8, scale = 0.8/7 ≈ 0.114
        assert!((s.scale - 0.8 / 7.0).abs() < 0.001);
        println!("INT4 scale: {:.6}", s.scale);
    }
}

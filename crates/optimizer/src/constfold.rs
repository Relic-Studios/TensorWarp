//! Constant folding — precompute operations with known constant inputs.
//!
//! Key optimizations:
//! - BatchNorm folding into Conv: precompute scale/bias, eliminate BN entirely
//! - Constant expression evaluation: 2+3 → 5 at compile time
//! - Shape computation: eliminate Shape/Gather/Reshape chains with known dims
//!
//! BatchNorm folding is critical for CNN inference:
//! Conv + BN = 2 kernel launches → 1 Conv (with modified weights)
//!
//! The math: BN(Conv(x)) = gamma * (Conv(x) - mean) / sqrt(var + eps) + beta
//!         = gamma/sqrt(var+eps) * Conv(x) + (beta - gamma*mean/sqrt(var+eps))
//!         = Conv_new(x) where:
//!           W_new = W * gamma / sqrt(var + eps)
//!           b_new = beta - gamma * mean / sqrt(var + eps) + original_bias * gamma / sqrt(var + eps)

/// Fold BatchNorm parameters into Conv weights.
/// Returns (new_weight, new_bias) that produce the same output as Conv+BN.
pub fn fold_batchnorm_into_conv(
    conv_weight: &[f32],   // [C_out, C_in, kH, kW]
    conv_bias: Option<&[f32]>,  // [C_out] or None
    bn_scale: &[f32],     // gamma [C_out]
    bn_bias: &[f32],      // beta [C_out]
    bn_mean: &[f32],      // running_mean [C_out]
    bn_var: &[f32],       // running_var [C_out]
    eps: f32,
    c_out: usize,
    kernel_size: usize,    // C_in * kH * kW
) -> (Vec<f32>, Vec<f32>) {
    let mut new_weight = vec![0.0f32; conv_weight.len()];
    let mut new_bias = vec![0.0f32; c_out];

    for c in 0..c_out {
        let inv_std = 1.0 / (bn_var[c] + eps).sqrt();
        let scale = bn_scale[c] * inv_std;

        // Scale the conv weights for this output channel
        for k in 0..kernel_size {
            new_weight[c * kernel_size + k] = conv_weight[c * kernel_size + k] * scale;
        }

        // Compute new bias: beta - gamma * mean * inv_std + original_bias * scale
        let original_bias = conv_bias.map_or(0.0, |b| b[c]);
        new_bias[c] = bn_bias[c] - bn_scale[c] * bn_mean[c] * inv_std + original_bias * scale;
    }

    (new_weight, new_bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batchnorm_folding_correctness() {
        let c_out = 4;
        let c_in = 3;
        let k = 3;
        let kernel_size = c_in * k * k;

        // Random conv weights
        let conv_w: Vec<f32> = (0..c_out * kernel_size)
            .map(|i| ((i * 7 + 3) % 200) as f32 * 0.01 - 1.0)
            .collect();
        let conv_b: Vec<f32> = vec![0.1, -0.2, 0.3, -0.1];

        // BN params
        let bn_scale = vec![1.0f32; c_out];
        let bn_bias = vec![0.0f32; c_out];
        let bn_mean = vec![0.5, -0.3, 0.1, 0.2];
        let bn_var = vec![1.0, 2.0, 0.5, 1.5];
        let eps = 1e-5;

        let (new_w, new_b) = fold_batchnorm_into_conv(
            &conv_w, Some(&conv_b), &bn_scale, &bn_bias, &bn_mean, &bn_var,
            eps, c_out, kernel_size);

        // Verify: for a dummy input, Conv+BN should equal folded Conv
        let x = vec![0.5f32; kernel_size]; // one "pixel" with all channels

        for c in 0..c_out {
            // Original: conv then BN
            let mut conv_out = conv_b[c];
            for k in 0..kernel_size {
                conv_out += conv_w[c * kernel_size + k] * x[k];
            }
            let inv_std = 1.0 / (bn_var[c] + eps).sqrt();
            let bn_out = bn_scale[c] * (conv_out - bn_mean[c]) * inv_std + bn_bias[c];

            // Folded: single conv
            let mut folded_out = new_b[c];
            for k in 0..kernel_size {
                folded_out += new_w[c * kernel_size + k] * x[k];
            }

            let err = (bn_out - folded_out).abs();
            assert!(err < 1e-5, "Channel {c}: BN={bn_out:.6} vs folded={folded_out:.6}, err={err}");
        }

        println!("BatchNorm folding: correct for all {} channels!", c_out);
    }
}

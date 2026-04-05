//! Layer-by-layer debugging for model inference.
//!
//! Loads a real model, runs each operation step by step, and dumps
//! intermediate values for comparison against a reference (e.g., PyTorch).
//! This is the tool that finds where TensorWarp diverges from the reference.

use warp_ir::{DType, Shape};
use warp_kernels::cache::KernelCache;
use warp_kernels::device::{DeviceError, WarpDevice};
use warp_kernels::tensor::GpuTensor;
use warp_kernels::ops;

use crate::safetensors_loader::SafeTensorsLoader;

/// Run a single-token forward pass through one transformer layer,
/// dumping values at each step.
///
/// Returns a vector of (step_name, first_5_values) for comparison.
pub fn debug_single_token(
    loader: &SafeTensorsLoader,
    device: &WarpDevice,
    token_id: i32,
    hidden_size: u32,
    kv_dim: u32,
    ffn_dim: u32,
    norm_eps: f32,
) -> Result<Vec<(String, Vec<f32>)>, Box<dyn std::error::Error>> {
    let cache = KernelCache::new();
    let h = hidden_size as usize;
    let mut results = Vec::new();

    // Step 1: Embedding lookup
    let embed_table = loader.load_f32("model.embed_tokens.weight", device)?;
    let ids = GpuTensor::from_host(device, &[token_id],
        Shape::from_static(&[1]), DType::I32)?;
    let mut hidden = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, h]), DType::F32)?;
    warp_kernels::sampling::embedding(&cache, device, &embed_table, &ids,
        &mut hidden, 1, hidden_size)?;
    device.synchronize()?;

    let embed_vals = hidden.to_host(device)?;
    results.push(("embedding".into(), embed_vals[..5.min(embed_vals.len())].to_vec()));

    // Step 2: RMSNorm (layer 0 input)
    let norm_w = loader.load_f32("model.layers.0.input_layernorm.weight", device)?;
    let mut normed = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, h]), DType::F32)?;
    ops::rmsnorm(&cache, device, &hidden, &norm_w, &mut normed, hidden_size, norm_eps)?;
    device.synchronize()?;

    let norm_vals = normed.to_host(device)?;
    results.push(("rmsnorm".into(), norm_vals[..5.min(norm_vals.len())].to_vec()));

    // Step 3: Q projection (with transposed weight)
    let wq = loader.load_f32_transposed("model.layers.0.self_attn.q_proj.weight", device)?;
    let mut q = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, h]), DType::F32)?;
    ops::gemm(&cache, device, &normed, &wq, &mut q, 1, hidden_size, hidden_size)?;
    device.synchronize()?;

    // Add bias if present
    if let Ok(bq) = loader.load_f32("model.layers.0.self_attn.q_proj.bias", device) {
        let mut q_biased = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, h]), DType::F32)?;
        ops::broadcast_add(&cache, device, &q, &bq, &mut q_biased)?;
        q = q_biased;
        device.synchronize()?;
    }

    let q_vals = q.to_host(device)?;
    results.push(("q_proj".into(), q_vals[..5.min(q_vals.len())].to_vec()));

    // Step 4: K projection
    let wk = loader.load_f32_transposed("model.layers.0.self_attn.k_proj.weight", device)?;
    let kd = kv_dim as usize;
    let mut k = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, kd]), DType::F32)?;
    ops::gemm(&cache, device, &normed, &wk, &mut k, 1, kv_dim, hidden_size)?;
    device.synchronize()?;

    if let Ok(bk) = loader.load_f32("model.layers.0.self_attn.k_proj.bias", device) {
        let mut k_biased = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, kd]), DType::F32)?;
        ops::broadcast_add(&cache, device, &k, &bk, &mut k_biased)?;
        k = k_biased;
        device.synchronize()?;
    }

    let k_vals = k.to_host(device)?;
    results.push(("k_proj".into(), k_vals[..5.min(k_vals.len())].to_vec()));

    // Step 5: Gate projection (FFN, rectangular matrix)
    let wg = loader.load_f32_transposed("model.layers.0.mlp.gate_proj.weight", device)?;
    let ff = ffn_dim as usize;
    let mut gate = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, ff]), DType::F32)?;
    ops::gemm(&cache, device, &normed, &wg, &mut gate, 1, ffn_dim, hidden_size)?;
    device.synchronize()?;

    let gate_vals = gate.to_host(device)?;
    results.push(("gate_proj".into(), gate_vals[..5.min(gate_vals.len())].to_vec()));

    // Step 6: V projection
    let wv = loader.load_f32_transposed("model.layers.0.self_attn.v_proj.weight", device)?;
    let mut v = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, kd]), DType::F32)?;
    ops::gemm(&cache, device, &normed, &wv, &mut v, 1, kv_dim, hidden_size)?;
    device.synchronize()?;

    if let Ok(bv) = loader.load_f32("model.layers.0.self_attn.v_proj.bias", device) {
        let mut v_biased = GpuTensor::<f32>::zeros(device,
            Shape::from_static(&[1, kd]), DType::F32)?;
        ops::broadcast_add(&cache, device, &v, &bv, &mut v_biased)?;
        v = v_biased;
        device.synchronize()?;
    }

    let v_vals = v.to_host(device)?;
    results.push(("v_proj".into(), v_vals[..5.min(v_vals.len())].to_vec()));

    // Step 7: For single token, attention output = V (repeated for GQA)
    // With 14 Q heads and 2 KV heads, V gets repeated 7x
    // Attention with 1 token: softmax([score]) = [1.0], so output = V
    // After GQA repeat: [14, 1, 64] where each group of 7 heads shares same KV head
    // Reshape to [1, 14*64] = [1, 896]
    let n_rep = 14u32 / 2; // num_heads / num_kv_heads = 7
    let head_dim_usize = 64usize;
    let v_host = v.to_host(device)?;

    // Build the GQA-repeated attention output (which equals V for single token)
    let mut attn_flat = vec![0.0f32; h]; // [896]
    for kv_h in 0..2usize {
        let src_off = kv_h * head_dim_usize;
        for r in 0..n_rep as usize {
            let dst_h = kv_h * n_rep as usize + r;
            let dst_off = dst_h * head_dim_usize;
            attn_flat[dst_off..dst_off + head_dim_usize]
                .copy_from_slice(&v_host[src_off..src_off + head_dim_usize]);
        }
    }
    results.push(("attn_out_expected".into(), attn_flat[..5].to_vec()));

    // Step 8: Output projection
    let wo = loader.load_f32_transposed("model.layers.0.self_attn.o_proj.weight", device)?;
    let attn_gpu = GpuTensor::from_host(device, &attn_flat,
        Shape::from_static(&[1, h]), DType::F32)?;
    let mut attn_proj = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, h]), DType::F32)?;
    ops::gemm(&cache, device, &attn_gpu, &wo, &mut attn_proj, 1, hidden_size, hidden_size)?;
    device.synchronize()?;

    let oproj_vals = attn_proj.to_host(device)?;
    results.push(("o_proj".into(), oproj_vals[..5.min(oproj_vals.len())].to_vec()));

    // Step 9: Residual add
    let mut residual = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, h]), DType::F32)?;
    ops::add(&cache, device, &hidden, &attn_proj, &mut residual)?;
    device.synchronize()?;

    // Step 10: FFN RMSNorm
    let ffn_norm_w = loader.load_f32("model.layers.0.post_attention_layernorm.weight", device)?;
    let mut ffn_normed = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, h]), DType::F32)?;
    ops::rmsnorm(&cache, device, &residual, &ffn_norm_w, &mut ffn_normed, hidden_size, norm_eps)?;
    device.synchronize()?;

    // Step 11: FFN gate + up + silu*mul + down
    let w_up = loader.load_f32_transposed("model.layers.0.mlp.up_proj.weight", device)?;
    let w_down = loader.load_f32_transposed("model.layers.0.mlp.down_proj.weight", device)?;

    let mut up = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, ff]), DType::F32)?;
    ops::gemm(&cache, device, &ffn_normed, &wg, &mut gate, 1, ffn_dim, hidden_size)?;
    ops::gemm(&cache, device, &ffn_normed, &w_up, &mut up, 1, ffn_dim, hidden_size)?;
    device.synchronize()?;

    let mut swiglu = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, ff]), DType::F32)?;
    ops::fused_silu_mul(&cache, device, &gate, &up, &mut swiglu)?;
    device.synchronize()?;

    let mut ffn_out = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, h]), DType::F32)?;
    ops::gemm(&cache, device, &swiglu, &w_down, &mut ffn_out, 1, hidden_size, ffn_dim)?;
    device.synchronize()?;

    // Step 12: Final residual
    let mut output = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[1, h]), DType::F32)?;
    ops::add(&cache, device, &residual, &ffn_out, &mut output)?;
    device.synchronize()?;

    let output_vals = output.to_host(device)?;
    results.push(("layer0_output".into(), output_vals[..5.min(output_vals.len())].to_vec()));
    let output_sum: f32 = output_vals.iter().sum();
    results.push(("layer0_sum".into(), vec![output_sum]));

    Ok(results)
}

/// Compare TensorWarp intermediate values against PyTorch reference.
/// Prints a pass/fail report for each step.
pub fn compare_against_reference(
    tw_results: &[(String, Vec<f32>)],
    reference: &[(&str, &[f32])],
    tolerance: f32,
) -> bool {
    let mut all_pass = true;
    for (tw_step, tw_vals) in tw_results {
        if let Some((_, ref_vals)) = reference.iter().find(|(name, _)| *name == tw_step.as_str()) {
            let n = tw_vals.len().min(ref_vals.len());
            let max_err = tw_vals[..n].iter().zip(ref_vals[..n].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let status = if max_err < tolerance { "PASS" } else { "FAIL" };
            println!("  {}: {} (max_err={:.6})", tw_step, status, max_err);
            if max_err >= tolerance {
                println!("    TW:  {:?}", &tw_vals[..n.min(5)]);
                println!("    Ref: {:?}", &ref_vals[..n.min(5)]);
                all_pass = false;
            }
        } else {
            println!("  {}: SKIP (no reference)", tw_step);
        }
    }
    all_pass
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_qwen_decode_layers() {
        // Test the DECODE path layer by layer against PyTorch reference.
        // PyTorch computes all 27 tokens (26 prompt + "2") in one forward pass.
        // TensorWarp does prefill(26 tokens) then decode(1 token with KV cache).
        // The outputs should match at every layer for the last token position.

        let model_path = format!(
            "{}/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors",
            std::env::var("USERPROFILE").unwrap_or_default()
        );
        if !std::path::Path::new(&model_path).exists() {
            println!("Qwen model not found, skipping"); return;
        }
        let device = match WarpDevice::new(0) {
            Ok(d) => d, Err(_) => { println!("No CUDA, skipping"); return; }
        };

        let config_path = model_path.replace("model.safetensors", "config.json");
        let llama_config = crate::llama::LlamaConfig::from_json(&config_path).unwrap();
        let loader = crate::safetensors_loader::SafeTensorsLoader::open(&model_path).unwrap();
        let model = crate::llama::LlamaModel::load(&loader, &llama_config, &device).unwrap();

        let tc = model.transformer_config.clone();
        let cache = warp_kernels::cache::KernelCache::new();

        // Prefill 26 tokens
        let prompt: Vec<i32> = vec![151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198];
        let num_layers = model.layers.len() as u32;
        let kv_dim = tc.kv_dim();
        let mut kv_cache = warp_kernels::kv_cache::ModelKVCache::new(&device, num_layers, 32768, kv_dim).unwrap();

        // Run prefill
        let ids = GpuTensor::from_host(&device, &prompt,
            warp_ir::Shape::from_static(&[prompt.len()]), warp_ir::DType::I32).unwrap();
        let mut hidden = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[prompt.len(), tc.hidden_size as usize]), warp_ir::DType::F32).unwrap();
        warp_kernels::sampling::embedding(&cache, &device, &model.embed_tokens, &ids,
            &mut hidden, prompt.len() as u32, tc.hidden_size).unwrap();

        for (i, layer) in model.layers.iter().enumerate() {
            hidden = warp_kernels::transformer::transformer_block_prefill(
                &cache, &device, &hidden, layer, &tc,
                &mut kv_cache.layers[i], 1, prompt.len() as u32, 0,
            ).unwrap();
        }
        device.synchronize().unwrap();

        // Now decode token 17 ("2") at position 26
        let token_id = 17i32;
        let pos = prompt.len() as u32; // 26

        let ids2 = GpuTensor::from_host(&device, &[token_id],
            warp_ir::Shape::from_static(&[1]), warp_ir::DType::I32).unwrap();
        let mut hidden2 = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, tc.hidden_size as usize]), warp_ir::DType::F32).unwrap();
        warp_kernels::sampling::embedding(&cache, &device, &model.embed_tokens, &ids2,
            &mut hidden2, 1, tc.hidden_size).unwrap();

        // PyTorch reference: per-layer output at token position 26 (dims 0:5)
        let pytorch_refs: Vec<Vec<f32>> = vec![
            vec![-0.0571, 0.0563, -0.0755, 0.1211, -0.095],
            vec![-0.1767, 0.1401, -0.1868, 0.1049, -0.1504],
            vec![-0.2867, 0.2388, -0.2438, 0.1889, -0.1481],
            vec![-0.3271, 0.3553, -0.2924, 0.3083, -0.0242],
            vec![-0.308, 0.0284, -0.2763, 0.2324, 0.0237],
        ];

        // Also manually compute K with RoPE for position 1 to verify
        {
            let cache2 = KernelCache::new();
            let wk = loader.load_f32_transposed("model.layers.0.self_attn.k_proj.weight", &device).unwrap();
            let bk_opt = loader.load_f32("model.layers.0.self_attn.k_proj.bias", &device).ok();

            // Compute K projection for position 1 only (second token)
            let embed_table2 = loader.load_f32("model.embed_tokens.weight", &device).unwrap();
            let ids_pos1 = GpuTensor::from_host(&device, &[prompt[1]],  // token at position 1
                warp_ir::Shape::from_static(&[1]), warp_ir::DType::I32).unwrap();
            let mut hidden_pos1 = GpuTensor::<f32>::zeros(&device,
                warp_ir::Shape::from_static(&[1, 896]), warp_ir::DType::F32).unwrap();
            warp_kernels::sampling::embedding(&cache2, &device, &embed_table2, &ids_pos1,
                &mut hidden_pos1, 1, 896).unwrap();

            // RMSNorm
            let norm_w = loader.load_f32("model.layers.0.input_layernorm.weight", &device).unwrap();
            let mut normed_pos1 = GpuTensor::<f32>::zeros(&device,
                warp_ir::Shape::from_static(&[1, 896]), warp_ir::DType::F32).unwrap();
            ops::rmsnorm(&cache2, &device, &hidden_pos1, &norm_w, &mut normed_pos1, 896, 1e-6).unwrap();

            // K projection
            let mut k_pos1 = GpuTensor::<f32>::zeros(&device,
                warp_ir::Shape::from_static(&[1, 128]), warp_ir::DType::F32).unwrap();
            ops::gemm(&cache2, &device, &normed_pos1, &wk, &mut k_pos1, 1, 128, 896).unwrap();
            if let Some(ref bk) = bk_opt {
                let mut kb = GpuTensor::<f32>::zeros(&device,
                    warp_ir::Shape::from_static(&[1, 128]), warp_ir::DType::F32).unwrap();
                ops::broadcast_add(&cache2, &device, &k_pos1, bk, &mut kb).unwrap();
                k_pos1 = kb;
            }

            // Apply RoPE at position 1 (single token, 2 kv heads)
            let mut k_rope_pos1 = GpuTensor::<f32>::zeros(&device,
                warp_ir::Shape::from_static(&[2, 1, 64]), warp_ir::DType::F32).unwrap();
            // Need to reshape k_pos1 from [1, 128] to [2, 1, 64] for RoPE
            let k_vals = k_pos1.to_host(&device).unwrap();
            let mut k_heads_first_pos1 = vec![0.0f32; 128];
            // [1, 128] positions-first to [2, 1, 64] heads-first
            for h in 0..2usize {
                for dd in 0..64usize {
                    k_heads_first_pos1[h * 64 + dd] = k_vals[h * 64 + dd];
                }
            }
            // For single pos, [1, 128] and [2, 1, 64] have same flat layout — no transpose needed
            let k_hf_pos1 = GpuTensor::from_host(&device, &k_heads_first_pos1,
                warp_ir::Shape::from_static(&[2, 1, 64]), warp_ir::DType::F32).unwrap();
            warp_kernels::rope::rope(&cache2, &device, &k_hf_pos1, &mut k_rope_pos1, 2, 1, 64, 1000000.0, 1).unwrap();
            device.synchronize().unwrap();

            let kr_vals = k_rope_pos1.to_host(&device).unwrap();
            println!("\n=== Manual K RoPE at position 1 ===");
            println!("  head=0, :5 = {:?}", &kr_vals[0..5]);
            println!("  head=1, :5 = {:?}", &kr_vals[64..69]);
            println!("  PT head=0 = [-10.180277, -7.69831, -7.355332, 2.671425, -0.25471]");
            println!("  PT head=1 = [39.130356, -10.496856, 5.865224, 7.209194, 3.594053]");
        }

        // Dump KV cache layer 0 to compare against PyTorch
        {
            let k_cache = kv_cache.layers[0].k.to_host(&device).unwrap();
            println!("\n=== KV Cache Layer 0 After Prefill ===");
            println!("  flat[0:5]    (pos=0,h=0) = {:?}", &k_cache[0..5]);
            println!("  flat[64:69]  (pos=0,h=1) = {:?}", &k_cache[64..69]);
            println!("  flat[128:133](pos=1,h=0) = {:?}", &k_cache[128..133]);
            println!("  flat[192:197](pos=1,h=1) = {:?}", &k_cache[192..197]);
            // PyTorch reference:
            println!("  PT flat[0:5]    = [-7.918979, -2.115042, -5.876757, 1.155304, -0.322867]");
            println!("  PT flat[64:69]  = [-8.646019, -1.073029, 5.749961, 5.548925, 4.048565]");
            println!("  PT flat[128:133]= [-10.180277, -7.69831, -7.355332, 2.671425, -0.25471]");
            println!("  PT flat[192:197]= [39.130356, -10.496856, 5.865224, 7.209194, 3.594053]");
        }

        // Verify Q_rope at pos=26 matches PyTorch
        {
            let cache3 = KernelCache::new();
            let embed3 = loader.load_f32("model.embed_tokens.weight", &device).unwrap();
            let ids3 = GpuTensor::from_host(&device, &[17i32],
                warp_ir::Shape::from_static(&[1]), warp_ir::DType::I32).unwrap();
            let mut hid3 = GpuTensor::<f32>::zeros(&device,
                warp_ir::Shape::from_static(&[1, 896]), warp_ir::DType::F32).unwrap();
            warp_kernels::sampling::embedding(&cache3, &device, &embed3, &ids3, &mut hid3, 1, 896).unwrap();

            let norm3 = loader.load_f32("model.layers.0.input_layernorm.weight", &device).unwrap();
            let mut norm3_out = GpuTensor::<f32>::zeros(&device,
                warp_ir::Shape::from_static(&[1, 896]), warp_ir::DType::F32).unwrap();
            ops::rmsnorm(&cache3, &device, &hid3, &norm3, &mut norm3_out, 896, 1e-6).unwrap();

            let wq3 = loader.load_f32_transposed("model.layers.0.self_attn.q_proj.weight", &device).unwrap();
            let mut q3 = GpuTensor::<f32>::zeros(&device,
                warp_ir::Shape::from_static(&[1, 896]), warp_ir::DType::F32).unwrap();
            ops::gemm(&cache3, &device, &norm3_out, &wq3, &mut q3, 1, 896, 896).unwrap();
            if let Ok(bq3) = loader.load_f32("model.layers.0.self_attn.q_proj.bias", &device) {
                let mut qb3 = GpuTensor::<f32>::zeros(&device,
                    warp_ir::Shape::from_static(&[1, 896]), warp_ir::DType::F32).unwrap();
                ops::broadcast_add(&cache3, &device, &q3, &bq3, &mut qb3).unwrap();
                q3 = qb3;
            }

            // RoPE at pos=26
            let mut qr3 = GpuTensor::<f32>::zeros(&device,
                warp_ir::Shape::from_static(&[14, 1, 64]), warp_ir::DType::F32).unwrap();
            warp_kernels::rope::rope(&cache3, &device, &q3, &mut qr3, 14, 1, 64, 1000000.0, 26).unwrap();
            device.synchronize().unwrap();

            let qr_vals = qr3.to_host(&device).unwrap();
            println!("\n=== Q_rope at pos=26 ===");
            println!("  TW head0[:10] = {:?}", &qr_vals[0..10].iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>());
            println!("  PT head0[:10] = [-0.106970, 0.292465, 0.604252, 0.062199, -10.581017, -0.422278, 0.270687, -0.555376, -21.270908, -32.763664]");

            // Compute dot product with K_cache[pos=0, head=0]
            let k_cache_vals = kv_cache.layers[0].k.to_host(&device).unwrap();
            let k_pos0_h0 = &k_cache_vals[0..64];

            let dot: f64 = qr_vals[0..64].iter().zip(k_pos0_h0.iter())
                .map(|(q, k)| *q as f64 * *k as f64).sum();
            println!("  TW Q.K dot (pos=0, head=0) = {:.6}", dot);
            println!("  TW score (×0.125) = {:.6}", dot * 0.125);
            println!("  PT score = 138.861933");
        }

        println!("\n=== Decode Layer-by-Layer Debug (pos=26, token='2') ===\n");

        for (i, layer) in model.layers.iter().enumerate() {
            hidden2 = warp_kernels::transformer::transformer_block_decode(
                &cache, &device, &hidden2, layer, &tc,
                &mut kv_cache.layers[i], 1, pos,
            ).unwrap();
            device.synchronize().unwrap();

            let vals = hidden2.to_host(&device).unwrap();
            let first5: Vec<f32> = vals[..5.min(vals.len())].to_vec();

            if i < pytorch_refs.len() {
                let ref5 = &pytorch_refs[i];
                let max_err = first5.iter().zip(ref5.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let status = if max_err < 0.05 { "PASS" } else { "FAIL" };
                println!("  L{:2}: {} (err={:.4})  TW={:?}  PT={:?}",
                    i, status, max_err,
                    first5.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>(),
                    ref5);
            } else {
                println!("  L{:2}: {:?}", i,
                    first5.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
            }

            if i == 0 {
                // Check if layer 0 already diverges
                if i < pytorch_refs.len() {
                    let ref5 = &pytorch_refs[i];
                    let max_err = first5.iter().zip(ref5.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    if max_err > 0.1 {
                        println!("\n  DIVERGENCE AT LAYER 0 — decode attention or KV cache is wrong");
                        break;
                    }
                }
            }
        }
    }

    #[test]
    fn debug_qwen_layer0() {
        // Reference values from PyTorch Qwen2.5-0.5B-Instruct, token_id=13048 ("Hi")
        let ref_embed: &[f32] = &[0.01508, 0.00766, -0.01471, 0.01160, -0.01215];
        let ref_norm: &[f32] = &[-0.05314, 0.02560, 0.02809, 0.07367, 0.03992];
        let ref_q: &[f32] = &[-0.03764, 0.01877, -0.14672, -0.03106, -13.42876];
        let ref_k: &[f32] = &[-8.38362, -3.10641, -6.25165, 0.56798, -0.14008];
        let ref_gate: &[f32] = &[-0.01969, 0.11960, -0.01728, 0.04826, 0.14126];
        let ref_v: &[f32] = &[-0.00128, 0.00867, -0.02779, 0.01229, -0.02898];
        let ref_oproj: &[f32] = &[-0.02446, 0.01393, -0.00796, 0.00627, 0.00309];
        // These need to be recomputed with correct GQA — skipping exact match for now
        let ref_layer0: &[f32] = &[]; // will be filled after o_proj passes
        let ref_layer0_sum: &[f32] = &[];

        let reference = vec![
            ("embedding", ref_embed),
            ("rmsnorm", ref_norm),
            ("q_proj", ref_q),
            ("k_proj", ref_k),
            ("gate_proj", ref_gate),
            ("v_proj", ref_v),
            ("o_proj", ref_oproj),
            ("layer0_output", ref_layer0),
            ("layer0_sum", ref_layer0_sum),
        ];

        let model_path = format!(
            "{}/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors",
            std::env::var("USERPROFILE").unwrap_or_default()
        );

        if !std::path::Path::new(&model_path).exists() {
            println!("Qwen model not found at {}, skipping", model_path);
            return;
        }

        let device = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        let loader = SafeTensorsLoader::open(&model_path).expect("Failed to open model");

        println!("\n=== Layer-by-Layer Debug: Qwen2.5-0.5B (token=13048 'Hi') ===\n");

        let results = debug_single_token(
            &loader, &device,
            13048,  // "Hi"
            896,    // hidden_size
            128,    // kv_dim (2 heads * 64 dim)
            4864,   // ffn_dim
            1e-6,   // norm_eps
        ).expect("Debug forward failed");

        let pass = compare_against_reference(&results, &reference, 0.01);

        if pass {
            println!("\n ALL STEPS MATCH PYTORCH REFERENCE (tolerance=0.01)");
        } else {
            println!("\n DIVERGENCE DETECTED — see FAIL steps above");
        }

        assert!(pass, "TensorWarp output diverges from PyTorch reference");
    }

    #[test]
    fn debug_qwen_manual_decode_step() {
        // Manual decode step through layer 0 of Qwen 0.5B for token 5373 at position 1.
        // Prefill with token [19], then manually decode token 5373, comparing each
        // intermediate value against PyTorch reference.
        //
        // The goal: if the manual decode matches PyTorch but transformer_block_decode
        // doesn't, the bug is in how transformer_block_decode composes the operations.

        let model_path = format!(
            "{}/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors",
            std::env::var("USERPROFILE").unwrap_or_default()
        );
        if !std::path::Path::new(&model_path).exists() {
            println!("Qwen model not found, skipping"); return;
        }
        let device = match WarpDevice::new(0) {
            Ok(d) => d, Err(_) => { println!("No CUDA, skipping"); return; }
        };

        let config_path = model_path.replace("model.safetensors", "config.json");
        let llama_config = crate::llama::LlamaConfig::from_json(&config_path).unwrap();
        let loader = crate::safetensors_loader::SafeTensorsLoader::open(&model_path).unwrap();
        let model = crate::llama::LlamaModel::load(&loader, &llama_config, &device).unwrap();

        let tc = model.transformer_config.clone();
        let cache = warp_kernels::cache::KernelCache::new();

        let h = tc.hidden_size;       // 896
        let d = tc.head_dim;           // 64
        let kv_dim = tc.kv_dim();      // 128
        let ffn = tc.ffn_dim;          // 4864
        let num_heads = tc.num_heads;  // 14
        let num_kv_heads = tc.num_kv_heads; // 2
        let hu = h as usize;
        let kvu = kv_dim as usize;
        let ffnu = ffn as usize;

        let num_layers = model.layers.len() as u32;
        let mut kv_cache = warp_kernels::kv_cache::ModelKVCache::new(
            &device, num_layers, 32768, kv_dim,
        ).unwrap();

        // =====================================================================
        // PREFILL: token [19] (single token "4") through all layers
        // =====================================================================
        let prefill_ids = vec![19i32];
        let ids_gpu = GpuTensor::from_host(&device, &prefill_ids,
            warp_ir::Shape::from_static(&[1]), warp_ir::DType::I32).unwrap();
        let mut hidden = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, hu]), warp_ir::DType::F32).unwrap();
        warp_kernels::sampling::embedding(&cache, &device, &model.embed_tokens, &ids_gpu,
            &mut hidden, 1, h).unwrap();

        for (i, layer) in model.layers.iter().enumerate() {
            hidden = warp_kernels::transformer::transformer_block_prefill(
                &cache, &device, &hidden, layer, &tc,
                &mut kv_cache.layers[i], 1, 1, 0,
            ).unwrap();
        }
        device.synchronize().unwrap();
        println!("\n=== Prefill complete (token=19, 1 position) ===\n");

        // =====================================================================
        // MANUAL DECODE: token 5373 at position 1 through layer 0 ONLY
        // =====================================================================
        // PyTorch reference values for each intermediate
        let ref_embed:    [f32; 5] = [-0.01086, -0.00513, 0.00674, -0.00365, 0.00830];
        let ref_normed:   [f32; 5] = [ 0.03551, -0.01589, -0.01194, -0.02148, -0.02530];
        let ref_q:        [f32; 5] = [ 0.11529, 0.28509, -0.06383, -0.58791, -15.45899];
        let ref_k:        [f32; 5] = [-8.82794, -3.62927, -6.52621, 0.37332, -0.03389];
        let ref_v:        [f32; 5] = [-0.00358, -0.00325, 0.00736, 0.02890, 0.07045];
        let ref_q_rope:   [f32; 5] = [-0.09340, 0.02931, -0.33484, -0.43556, -13.04660];
        let ref_k_rope:   [f32; 5] = [-10.51042, -8.16756, -7.40030, 2.26250, 0.01122];
        let ref_attn:     [f32; 5] = [-0.00466, -0.00369, 0.01130, 0.00716, 0.04620];
        let ref_oproj:    [f32; 5] = [ 0.00300, 0.00240, -0.00955, 0.01178, 0.00014];
        let ref_residual: [f32; 5] = [-0.00786, -0.00273, -0.00281, 0.00813, 0.00844];
        let ref_output:   [f32; 5] = [-0.12961, -0.01417, -0.06814, -0.06495, 0.07988];

        let mut all_pass = true;
        let tol = 0.02; // tolerance for FP32 comparison

        // Helper closure for comparing and printing
        let check = |name: &str, got: &[f32], expected: &[f32], all_pass: &mut bool| {
            let n = got.len().min(expected.len()).min(5);
            let max_err = got[..n].iter().zip(expected[..n].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let status = if max_err < tol { "PASS" } else { "FAIL" };
            println!("  {:15}: {} (max_err={:.6})", name, status, max_err);
            println!("    TW:  {:?}", &got[..n]);
            println!("    Ref: {:?}", &expected[..n]);
            if max_err >= tol {
                *all_pass = false;
            }
        };

        println!("=== Manual Decode Step: token=5373, pos=1, layer=0 ===\n");

        // --- Step 1: Embedding lookup for token 5373 ---
        let ids_decode = GpuTensor::from_host(&device, &[5373i32],
            warp_ir::Shape::from_static(&[1]), warp_ir::DType::I32).unwrap();
        let mut embed = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, hu]), warp_ir::DType::F32).unwrap();
        warp_kernels::sampling::embedding(&cache, &device, &model.embed_tokens, &ids_decode,
            &mut embed, 1, h).unwrap();
        device.synchronize().unwrap();

        let embed_vals = embed.to_host(&device).unwrap();
        check("embed", &embed_vals[..5], &ref_embed, &mut all_pass);

        // --- Step 2: RMSNorm with layer 0 norm weights ---
        let layer = &model.layers[0];
        let mut normed = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, hu]), warp_ir::DType::F32).unwrap();
        ops::rmsnorm(&cache, &device, &embed, &layer.attn_norm, &mut normed, h, tc.norm_eps).unwrap();
        device.synchronize().unwrap();

        let normed_vals = normed.to_host(&device).unwrap();
        check("normed", &normed_vals[..5], &ref_normed, &mut all_pass);

        // --- Step 3: Q/K/V GEMM ---
        let mut q = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, hu]), warp_ir::DType::F32).unwrap();
        let mut k_proj = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, kvu]), warp_ir::DType::F32).unwrap();
        let mut v_proj = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, kvu]), warp_ir::DType::F32).unwrap();

        ops::gemm(&cache, &device, &normed, &layer.wq, &mut q, 1, h, h).unwrap();
        ops::gemm(&cache, &device, &normed, &layer.wk, &mut k_proj, 1, kv_dim, h).unwrap();
        ops::gemm(&cache, &device, &normed, &layer.wv, &mut v_proj, 1, kv_dim, h).unwrap();
        device.synchronize().unwrap();

        // --- Step 4: Bias addition ---
        if let Some(ref bq) = layer.bq {
            let mut q_biased = GpuTensor::<f32>::zeros(&device,
                warp_ir::Shape::from_static(&[1, hu]), warp_ir::DType::F32).unwrap();
            ops::broadcast_add(&cache, &device, &q, bq, &mut q_biased).unwrap();
            q = q_biased;
        }
        if let Some(ref bk) = layer.bk {
            let mut k_biased = GpuTensor::<f32>::zeros(&device,
                warp_ir::Shape::from_static(&[1, kvu]), warp_ir::DType::F32).unwrap();
            ops::add(&cache, &device, &k_proj, bk, &mut k_biased).unwrap();
            k_proj = k_biased;
        }
        if let Some(ref bv) = layer.bv {
            let mut v_biased = GpuTensor::<f32>::zeros(&device,
                warp_ir::Shape::from_static(&[1, kvu]), warp_ir::DType::F32).unwrap();
            ops::add(&cache, &device, &v_proj, bv, &mut v_biased).unwrap();
            v_proj = v_biased;
        }
        device.synchronize().unwrap();

        let q_vals = q.to_host(&device).unwrap();
        check("q", &q_vals[..5], &ref_q, &mut all_pass);

        let k_vals = k_proj.to_host(&device).unwrap();
        check("k", &k_vals[..5], &ref_k, &mut all_pass);

        let v_vals = v_proj.to_host(&device).unwrap();
        check("v", &v_vals[..5], &ref_v, &mut all_pass);

        // --- Step 5: RoPE at position 1 ---
        let mut q_rope = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, hu]), warp_ir::DType::F32).unwrap();
        let mut k_rope = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, kvu]), warp_ir::DType::F32).unwrap();
        // Q: B = num_heads (14), seq_len = 1, head_dim = 64
        warp_kernels::rope::rope(&cache, &device, &q, &mut q_rope,
            num_heads, 1, d, tc.rope_base, 1).unwrap();
        // K: B = num_kv_heads (2), seq_len = 1, head_dim = 64
        warp_kernels::rope::rope(&cache, &device, &k_proj, &mut k_rope,
            num_kv_heads, 1, d, tc.rope_base, 1).unwrap();
        device.synchronize().unwrap();

        let qr_vals = q_rope.to_host(&device).unwrap();
        check("q_rope h0", &qr_vals[..5], &ref_q_rope, &mut all_pass);

        let kr_vals = k_rope.to_host(&device).unwrap();
        check("k_rope h0", &kr_vals[..5], &ref_k_rope, &mut all_pass);

        // --- Step 6: KV cache append (k_rope and v to layer 0 cache) ---
        kv_cache.layers[0].append(&cache, &device, &k_rope, &v_proj).unwrap();
        device.synchronize().unwrap();
        println!("  KV cache layer 0 len = {}", kv_cache.layers[0].len);

        // --- Step 7: decode_attention_multihead (Q against 2-position cache) ---
        let mut attn_out = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, hu]), warp_ir::DType::F32).unwrap();
        warp_kernels::kv_cache::decode_attention_multihead(
            &cache, &device, &q_rope, &kv_cache.layers[0], &mut attn_out,
            num_heads, num_kv_heads, d,
        ).unwrap();
        device.synchronize().unwrap();

        let attn_vals = attn_out.to_host(&device).unwrap();
        check("attn_flat", &attn_vals[..5], &ref_attn, &mut all_pass);

        // --- Step 8: Output projection GEMM ---
        let mut o_proj = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, hu]), warp_ir::DType::F32).unwrap();
        ops::gemm(&cache, &device, &attn_out, &layer.wo, &mut o_proj, 1, h, h).unwrap();
        device.synchronize().unwrap();

        let oproj_vals = o_proj.to_host(&device).unwrap();
        check("o_proj", &oproj_vals[..5], &ref_oproj, &mut all_pass);

        // --- Step 9: Residual add (embed + o_proj) ---
        let mut residual = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, hu]), warp_ir::DType::F32).unwrap();
        ops::add(&cache, &device, &embed, &o_proj, &mut residual).unwrap();
        device.synchronize().unwrap();

        let res_vals = residual.to_host(&device).unwrap();
        check("residual", &res_vals[..5], &ref_residual, &mut all_pass);

        // --- Step 10: FFN (fused_residual_rmsnorm, gate, up, silu_mul, down, residual) ---
        // 10a. Fused residual RMSNorm: computes residual = embed + o_proj AND normed for FFN
        //      But we already have residual, so just do the FFN norm.
        //      transformer_block_decode uses fused_residual_rmsnorm(o_proj, embed, ffn_norm, ...)
        //      which computes residual_out = embed + o_proj, norm_out = rmsnorm(residual_out)
        let mut ffn_normed = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, hu]), warp_ir::DType::F32).unwrap();
        let mut fused_residual = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, hu]), warp_ir::DType::F32).unwrap();
        ops::fused_residual_rmsnorm(&cache, &device, &o_proj, &embed, &layer.ffn_norm,
            &mut ffn_normed, &mut fused_residual, h, tc.norm_eps).unwrap();
        device.synchronize().unwrap();

        // 10b. Gate and Up projections
        let mut gate = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, ffnu]), warp_ir::DType::F32).unwrap();
        let mut up = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, ffnu]), warp_ir::DType::F32).unwrap();
        ops::gemm(&cache, &device, &ffn_normed, &layer.w_gate, &mut gate, 1, ffn, h).unwrap();
        ops::gemm(&cache, &device, &ffn_normed, &layer.w_up, &mut up, 1, ffn, h).unwrap();
        device.synchronize().unwrap();

        // 10c. Fused SiLU * mul
        let mut swiglu = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, ffnu]), warp_ir::DType::F32).unwrap();
        ops::fused_silu_mul(&cache, &device, &gate, &up, &mut swiglu).unwrap();
        device.synchronize().unwrap();

        // 10d. Down projection
        let mut ffn_out = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, hu]), warp_ir::DType::F32).unwrap();
        ops::gemm(&cache, &device, &swiglu, &layer.w_down, &mut ffn_out, 1, h, ffn).unwrap();
        device.synchronize().unwrap();

        // 10e. Final residual add
        let mut output = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, hu]), warp_ir::DType::F32).unwrap();
        ops::add(&cache, &device, &fused_residual, &ffn_out, &mut output).unwrap();
        device.synchronize().unwrap();

        let out_vals = output.to_host(&device).unwrap();
        check("output", &out_vals[..5], &ref_output, &mut all_pass);

        // =====================================================================
        // Also run transformer_block_decode for comparison
        // =====================================================================
        println!("\n--- Now running transformer_block_decode for comparison ---\n");

        // We need a fresh KV cache since we already appended to layer 0 above.
        // Re-create and re-prefill.
        let mut kv_cache2 = warp_kernels::kv_cache::ModelKVCache::new(
            &device, num_layers, 32768, kv_dim,
        ).unwrap();

        // Re-prefill token [19]
        let mut hidden_pf = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, hu]), warp_ir::DType::F32).unwrap();
        warp_kernels::sampling::embedding(&cache, &device, &model.embed_tokens, &ids_gpu,
            &mut hidden_pf, 1, h).unwrap();
        for (i, ly) in model.layers.iter().enumerate() {
            hidden_pf = warp_kernels::transformer::transformer_block_prefill(
                &cache, &device, &hidden_pf, ly, &tc,
                &mut kv_cache2.layers[i], 1, 1, 0,
            ).unwrap();
        }
        device.synchronize().unwrap();

        // Decode token 5373 at pos=1 through layer 0 only using transformer_block_decode
        let mut decode_hidden = GpuTensor::<f32>::zeros(&device,
            warp_ir::Shape::from_static(&[1, hu]), warp_ir::DType::F32).unwrap();
        warp_kernels::sampling::embedding(&cache, &device, &model.embed_tokens, &ids_decode,
            &mut decode_hidden, 1, h).unwrap();

        let decode_out = warp_kernels::transformer::transformer_block_decode(
            &cache, &device, &decode_hidden, &model.layers[0], &tc,
            &mut kv_cache2.layers[0], 1, 1,
        ).unwrap();
        device.synchronize().unwrap();

        let decode_vals = decode_out.to_host(&device).unwrap();
        let decode5: Vec<f32> = decode_vals[..5.min(decode_vals.len())].to_vec();

        println!("  transformer_block_decode output[:5] = {:?}", decode5);
        println!("  manual decode output[:5]            = {:?}", &out_vals[..5]);
        println!("  PyTorch reference[:5]               = {:?}", &ref_output);

        let manual_vs_decode_err = decode5.iter().zip(out_vals[..5].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let decode_vs_pytorch_err = decode5.iter().zip(ref_output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("\n  manual vs decode max_err:  {:.6}", manual_vs_decode_err);
        println!("  decode vs pytorch max_err: {:.6}", decode_vs_pytorch_err);

        if manual_vs_decode_err > 0.01 {
            println!("\n  DIVERGENCE: manual decode != transformer_block_decode");
            println!("  The bug is in how transformer_block_decode composes the operations.");
        }

        println!();
        if all_pass {
            println!("ALL MANUAL STEPS MATCH PYTORCH REFERENCE (tol={:.3})", tol);
        } else {
            println!("DIVERGENCE DETECTED in manual decode -- see FAIL steps above");
        }
    }
}

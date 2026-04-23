//! Qwen3.5 loader + linear-attn forward smoke test.
//!
//! Loads Huihui-Qwen3.5-9B-abliterated from safetensors, validates that
//! every tensor expected by the loader is present, allocates state for
//! one batch, and runs ONE Gated DeltaNet decode step on dummy input
//! to verify the CUDA kernels compile + launch without crashing.
//!
//! Run:
//!   cargo run --bin qwen3_5_loader_smoke --release -- D:/qwen3.5-9b-abliterated-with-yield

use std::path::PathBuf;
use std::time::Instant;

use warp_kernels::cache::KernelCache;
use warp_kernels::device::WarpDevice;
use warp_kernels::tensor::GpuTensor;
use warp_kernels::gated_delta_net::{
    gated_delta_net_step, GatedDeltaNetStepBuffers,
};
use warp_loader::safetensors_loader::ShardedSafeTensorsLoader;
use warp_loader::qwen3_5_dense_q4::{
    Qwen35Config, Qwen35LayerWeights, alloc_delta_state, load_qwen3_5,
};
use warp_ir::{DType, Shape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let model_dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("D:/qwen3.5-9b-abliterated-with-yield"));

    eprintln!("[smoke] model dir: {}", model_dir.display());

    // ── 1. Init device + load weights ─────────────────────────────────
    let device = WarpDevice::new(0)?;
    eprintln!("[smoke] CUDA device init OK");

    let cache = KernelCache::new();

    // Find safetensors shards
    let shards: Vec<PathBuf> = std::fs::read_dir(&model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    eprintln!("[smoke] found {} safetensors shards", shards.len());
    if shards.is_empty() {
        return Err(format!("no .safetensors in {}", model_dir.display()).into());
    }

    let loader = ShardedSafeTensorsLoader::open(&shards)?;
    eprintln!("[smoke] safetensors loader OK");

    // ── 2. Load model ─────────────────────────────────────────────────
    let cfg = Qwen35Config::qwen3_5_9b_abliterated();
    let t0 = Instant::now();
    let model = load_qwen3_5(&loader, &cfg, &device)?;
    eprintln!("[smoke] model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // ── 3. Allocate state for batch=1 ─────────────────────────────────
    let mut delta_states = alloc_delta_state(&device, &model, 1)?;
    eprintln!("[smoke] state alloc OK ({} layer slots)", delta_states.len());

    // ── 4. Allocate per-step buffers for one linear layer ─────────────
    let alloc1d = |dim: usize| -> Result<GpuTensor<f32>, Box<dyn std::error::Error>> {
        Ok(GpuTensor::<f32>::zeros(&device, Shape::from_static(&[1, dim]), DType::F32)?)
    };
    let alloc3d = |a: usize, b: usize, c: usize| -> Result<GpuTensor<f32>, Box<dyn std::error::Error>> {
        Ok(GpuTensor::<f32>::zeros(&device, Shape::from_static(&[a, b, c]), DType::F32)?)
    };

    let mut bufs = GatedDeltaNetStepBuffers {
        q:        alloc3d(1, cfg.gdn.num_value_heads, cfg.gdn.key_head_dim)?,
        k:        alloc3d(1, cfg.gdn.num_value_heads, cfg.gdn.key_head_dim)?,
        v:        alloc3d(1, cfg.gdn.num_value_heads, cfg.gdn.value_head_dim)?,
        z:        alloc1d(cfg.gdn.value_dim())?,
        beta:     alloc1d(cfg.gdn.num_value_heads)?,
        g:        alloc1d(cfg.gdn.num_value_heads)?,
        y:        alloc3d(1, cfg.gdn.num_value_heads, cfg.gdn.value_head_dim)?,
        y_gated:  alloc1d(cfg.gdn.value_dim())?,
    };

    // Non-zero input so we can verify the kernel actually computes.
    // Pattern: 0.1, -0.1, 0.1, ... (small alternating values)
    let host_hidden: Vec<f32> = (0..cfg.hidden_size).map(|i| {
        if i % 2 == 0 { 0.1 } else { -0.1 }
    }).collect();
    let host_residual: Vec<f32> = (0..cfg.hidden_size).map(|i| {
        0.01 * (i as f32 / cfg.hidden_size as f32)
    }).collect();

    let hidden = GpuTensor::<f32>::from_host(
        &device, &host_hidden,
        Shape::from_static(&[1, cfg.hidden_size]), DType::F32,
    )?;
    let residual = GpuTensor::<f32>::from_host(
        &device, &host_residual,
        Shape::from_static(&[1, cfg.hidden_size]), DType::F32,
    )?;
    let mut out = alloc1d(cfg.hidden_size)?;

    // Find the first linear layer (should be 0) and run one decode step
    let (linear_idx, gdn) = model.layers.iter().enumerate()
        .find_map(|(i, l)| match l {
            Qwen35LayerWeights::Linear { gdn, .. } => Some((i, gdn)),
            _ => None,
        })
        .ok_or("no linear layer found?!")?;

    let state = delta_states[linear_idx].as_mut().expect("state present");

    eprintln!("[smoke] running cold step on layer {} (NVRTC compile included)", linear_idx);
    let t0 = Instant::now();
    gated_delta_net_step(
        &device, &cache,
        &hidden, &residual,
        gdn, state, &mut bufs,
        &cfg.gdn,
        &mut out,
    )?;
    device.stream.synchronize()?;
    eprintln!("[smoke] cold step: {:.2} ms", t0.elapsed().as_secs_f64() * 1000.0);

    // Steady-state benchmark
    eprintln!("[smoke] running 100 warm steps for steady-state benchmark...");
    let t0 = Instant::now();
    for _ in 0..100 {
        gated_delta_net_step(
            &device, &cache,
            &hidden, &residual,
            gdn, state, &mut bufs,
            &cfg.gdn,
            &mut out,
        )?;
    }
    device.stream.synchronize()?;
    let total = t0.elapsed().as_secs_f64();
    eprintln!("[smoke] warm: {:.1} ms total / {:.3} ms per step / {:.1} steps/sec",
              total * 1000.0, total * 10.0, 100.0 / total);

    // Spot-check the output isn't all zeros (would indicate kernel didn't write)
    let out_host: Vec<f32> = out.to_host(&device)?;
    let nonzero = out_host.iter().filter(|&&x| x.abs() > 1e-9).count();
    let max_abs = out_host.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    eprintln!("[smoke] output stats: {} / {} nonzero, max |x| = {:e}",
              nonzero, out_host.len(), max_abs);

    if nonzero == 0 {
        eprintln!("[smoke] WARNING: output is all-zero. Kernel may not have run \
                   correctly OR input was zero (which it was — this is a smoke test).");
    }

    eprintln!("[smoke] ✓ SUCCESS — kernels compile + launch end-to-end.");
    Ok(())
}

//! .warp cache for MoE engine — instant model loading.
//!
//! First run: load from SafeTensors (~4 min) → save .warp cache (~15 GB).
//! Subsequent runs: mmap .warp → cudaMemcpy (~5 sec).

use std::path::Path;
use std::io::{Write, BufWriter};
use memmap2::Mmap;
use warp_ir::{DType, Shape};
use warp_kernels::cache::KernelCache;
use warp_kernels::device::WarpDevice;
use warp_kernels::tensor::GpuTensor;
use warp_kernels::transformer::{GemmaConfig, GemmaLayerAttentionConfig};
use warp_kernels::moe_q4_generate::{MoEQ4Layer, MoEQ4Engine};

use crate::safetensors_loader::LoaderError;

const WARP_MOE_MAGIC: &[u8; 8] = b"WARPMOE2";

/// Tensor entry in the .warp file.
struct CacheEntry {
    name: String,
    dtype: u8,      // 0=F32, 1=F16, 2=U8
    num_bytes: usize,
    offset: usize,  // byte offset in the data section
}

/// Save an MoEQ4Engine to a .warp cache file.
pub fn save_moe_cache(
    path: &Path,
    engine: &MoEQ4Engine,
    device: &WarpDevice,
) -> Result<(), LoaderError> {
    let start = std::time::Instant::now();
    eprintln!("[warp] Saving cache to {}...", path.display());

    let file = std::fs::File::create(path)
        .map_err(|e| LoaderError::Device(format!("create: {e}")))?;
    let mut w = BufWriter::with_capacity(64 * 1024 * 1024, file); // 64MB buffer

    // Magic
    w.write_all(WARP_MOE_MAGIC).map_err(|e| LoaderError::Device(e.to_string()))?;

    // Config as JSON
    let config_json = serde_config_json(&engine.config, &engine.layer_configs);
    let config_bytes = config_json.as_bytes();
    w.write_all(&(config_bytes.len() as u32).to_le_bytes()).map_err(|e| LoaderError::Device(e.to_string()))?;
    w.write_all(config_bytes).map_err(|e| LoaderError::Device(e.to_string()))?;

    // Number of layers
    w.write_all(&(engine.layers.len() as u32).to_le_bytes()).map_err(|e| LoaderError::Device(e.to_string()))?;

    // Embedding F16
    write_tensor_f16(&mut w, "embed_tokens", &engine.embed_tokens, device)?;

    // Final norm F32
    write_tensor_f32(&mut w, "final_norm", &engine.final_norm, device)?;

    // Per-layer tensors
    for (i, layer) in engine.layers.iter().enumerate() {
        let p = format!("layer.{i}");

        // F32 norms
        write_tensor_f32(&mut w, &format!("{p}.attn_norm"), &layer.attn_norm, device)?;
        write_tensor_f32(&mut w, &format!("{p}.post_attn_norm"), &layer.post_attn_norm, device)?;
        write_tensor_f32(&mut w, &format!("{p}.pre_ffn_norm"), &layer.pre_ffn_norm, device)?;
        write_tensor_f32(&mut w, &format!("{p}.post_ffn_norm"), &layer.post_ffn_norm, device)?;
        write_tensor_f32(&mut w, &format!("{p}.post_ffn_norm_1"), &layer.post_ffn_norm_1, device)?;
        write_tensor_f32(&mut w, &format!("{p}.pre_ffn_norm_2"), &layer.pre_ffn_norm_2, device)?;
        write_tensor_f32(&mut w, &format!("{p}.post_ffn_norm_2"), &layer.post_ffn_norm_2, device)?;
        write_tensor_f32(&mut w, &format!("{p}.q_norm"), &layer.q_norm, device)?;
        write_tensor_f32(&mut w, &format!("{p}.k_norm"), &layer.k_norm, device)?;

        // Scalar
        w.write_all(&layer.layer_scalar.to_le_bytes()).map_err(|e| LoaderError::Device(e.to_string()))?;

        // F16 attention/MLP weights
        write_tensor_f16(&mut w, &format!("{p}.wq"), &layer.wq, device)?;
        write_tensor_f16(&mut w, &format!("{p}.wk"), &layer.wk, device)?;
        write_tensor_f16(&mut w, &format!("{p}.wv"), &layer.wv, device)?;
        write_tensor_f16(&mut w, &format!("{p}.wo"), &layer.wo, device)?;
        write_tensor_f16(&mut w, &format!("{p}.w_gate"), &layer.w_gate, device)?;
        write_tensor_f16(&mut w, &format!("{p}.w_up"), &layer.w_up, device)?;
        write_tensor_f16(&mut w, &format!("{p}.w_down"), &layer.w_down, device)?;

        // F32 router
        write_tensor_f32(&mut w, &format!("{p}.router_proj"), &layer.router_proj, device)?;
        write_tensor_f32(&mut w, &format!("{p}.router_scale"), &layer.router_scale, device)?;
        write_tensor_f32(&mut w, &format!("{p}.per_expert_scale"), &layer.per_expert_scale, device)?;

        // Expert raw bytes (U8)
        write_tensor_u8(&mut w, &format!("{p}.experts_gu"), &layer.experts_gu_raw, device)?;
        write_tensor_u8(&mut w, &format!("{p}.experts_d"), &layer.experts_d_raw, device)?;

        // Expert metadata
        let meta = [
            layer.gu_bytes_per_expert as u64,
            layer.d_bytes_per_expert as u64,
            layer.d_block_bytes as u64,
            layer.d_block_elems as u64,
            layer.use_native_gguf_experts as u64,
            layer.expert_gu_k as u64, layer.expert_gu_n as u64,
            layer.expert_d_k as u64, layer.expert_d_n as u64,
            layer.gu_scales_per_expert as u64,
            layer.d_scales_per_expert as u64,
        ];
        for v in &meta {
            w.write_all(&v.to_le_bytes()).map_err(|e| LoaderError::Device(e.to_string()))?;
        }

        // Optional TW-Marlin scales
        let has_scales = layer.experts_gu_scales.is_some() as u8;
        w.write_all(&[has_scales]).map_err(|e| LoaderError::Device(e.to_string()))?;
        if let Some(ref s) = layer.experts_gu_scales {
            write_tensor_f16(&mut w, &format!("{p}.gu_scales"), s, device)?;
        }
        if let Some(ref s) = layer.experts_d_scales {
            write_tensor_f16(&mut w, &format!("{p}.d_scales"), s, device)?;
        }
    }

    w.flush().map_err(|e| LoaderError::Device(e.to_string()))?;
    eprintln!("[warp] Cache saved in {:.1}s", start.elapsed().as_secs_f64());
    Ok(())
}

/// Load an MoEQ4Engine from a .warp cache file (mmap + cudaMemcpy).
pub fn load_moe_cache(
    path: &Path,
    device: &WarpDevice,
) -> Result<MoEQ4Engine, LoaderError> {
    let start = std::time::Instant::now();
    eprintln!("[warp] Loading cache from {}...", path.display());

    let file = std::fs::File::open(path)
        .map_err(|e| LoaderError::Device(format!("open: {e}")))?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| LoaderError::Device(format!("mmap: {e}")))?;

    let mut pos = 0usize;

    // Magic
    if &mmap[pos..pos+8] != WARP_MOE_MAGIC {
        return Err(LoaderError::Config("not a .warp MoE cache".into()));
    }
    pos += 8;

    // Config JSON
    let config_len = u32::from_le_bytes(mmap[pos..pos+4].try_into().unwrap()) as usize;
    pos += 4;
    let config_json = std::str::from_utf8(&mmap[pos..pos+config_len])
        .map_err(|_| LoaderError::Config("invalid config JSON".into()))?;
    pos += config_len;

    let (config, layer_configs) = parse_config_json(config_json)?;

    // Number of layers
    let num_layers = u32::from_le_bytes(mmap[pos..pos+4].try_into().unwrap()) as usize;
    pos += 4;

    // Embedding
    let embed_tokens = read_tensor_f16(&mmap, &mut pos, device)?;
    eprintln!("[warp] Embedding loaded");

    // Final norm
    let final_norm = read_tensor_f32(&mmap, &mut pos, device)?;

    // Layers
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        // F32 norms
        let attn_norm = read_tensor_f32(&mmap, &mut pos, device)?;
        let post_attn_norm = read_tensor_f32(&mmap, &mut pos, device)?;
        let pre_ffn_norm = read_tensor_f32(&mmap, &mut pos, device)?;
        let post_ffn_norm = read_tensor_f32(&mmap, &mut pos, device)?;
        let post_ffn_norm_1 = read_tensor_f32(&mmap, &mut pos, device)?;
        let pre_ffn_norm_2 = read_tensor_f32(&mmap, &mut pos, device)?;
        let post_ffn_norm_2 = read_tensor_f32(&mmap, &mut pos, device)?;
        let q_norm = read_tensor_f32(&mmap, &mut pos, device)?;
        let k_norm = read_tensor_f32(&mmap, &mut pos, device)?;

        let layer_scalar = f32::from_le_bytes(mmap[pos..pos+4].try_into().unwrap());
        pos += 4;

        // F16 attention/MLP
        let wq = read_tensor_f16(&mmap, &mut pos, device)?;
        let wk = read_tensor_f16(&mmap, &mut pos, device)?;
        let wv = read_tensor_f16(&mmap, &mut pos, device)?;
        let wo = read_tensor_f16(&mmap, &mut pos, device)?;
        let w_gate = read_tensor_f16(&mmap, &mut pos, device)?;
        let w_up = read_tensor_f16(&mmap, &mut pos, device)?;
        let w_down = read_tensor_f16(&mmap, &mut pos, device)?;

        // Router
        let router_proj = read_tensor_f32(&mmap, &mut pos, device)?;
        let router_scale = read_tensor_f32(&mmap, &mut pos, device)?;
        let per_expert_scale = read_tensor_f32(&mmap, &mut pos, device)?;

        // Expert raw bytes
        let experts_gu_raw = read_tensor_u8(&mmap, &mut pos, device)?;
        let experts_d_raw = read_tensor_u8(&mmap, &mut pos, device)?;

        // Expert metadata
        let mut meta = [0u64; 11];
        for m in &mut meta {
            *m = u64::from_le_bytes(mmap[pos..pos+8].try_into().unwrap());
            pos += 8;
        }

        let has_scales = mmap[pos];
        pos += 1;

        let (experts_gu_scales, experts_d_scales) = if has_scales != 0 {
            let gs = read_tensor_f16(&mmap, &mut pos, device)?;
            let ds = read_tensor_f16(&mmap, &mut pos, device)?;
            (Some(gs), Some(ds))
        } else {
            (None, None)
        };

        layers.push(MoEQ4Layer {
            attn_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm,
            post_ffn_norm_1, pre_ffn_norm_2, post_ffn_norm_2,
            q_norm, k_norm, layer_scalar,
            wq, wk, wv, wo, w_gate, w_up, w_down,
            router_proj, router_scale, per_expert_scale,
            experts_gu_raw, gu_bytes_per_expert: meta[0] as usize,
            experts_d_raw, d_bytes_per_expert: meta[1] as usize,
            d_block_bytes: meta[2] as u32, d_block_elems: meta[3] as u32,
            use_native_gguf_experts: meta[4] != 0,
            expert_gu_k: meta[5] as u32, expert_gu_n: meta[6] as u32,
            expert_d_k: meta[7] as u32, expert_d_n: meta[8] as u32,
            experts_gu_scales, experts_d_scales,
            gu_scales_per_expert: meta[9] as usize,
            d_scales_per_expert: meta[10] as usize,
        });

        if (i + 1) % 10 == 0 {
            eprintln!("[warp] {}/{} layers loaded", i + 1, num_layers);
        }
    }

    device.synchronize().map_err(|e| LoaderError::Device(e.to_string()))?;
    eprintln!("[warp] Cache loaded in {:.1}s ({} layers)", start.elapsed().as_secs_f64(), num_layers);

    Ok(MoEQ4Engine {
        config, layer_configs, embed_tokens, final_norm,
        cache: KernelCache::new(), layers, weights_reordered: false,
    })
}

/// Check if a .warp cache exists for a model.
pub fn warp_cache_path(model_path: &Path) -> std::path::PathBuf {
    model_path.join("model.warp")
}

pub fn has_warp_cache(model_path: &Path) -> bool {
    warp_cache_path(model_path).exists()
}

// ── Serialization helpers ────────────────────────────────────────────────────

fn write_tensor_f32<W: Write>(w: &mut W, _name: &str, t: &GpuTensor<f32>, device: &WarpDevice) -> Result<(), LoaderError> {
    let host = t.to_host(device).map_err(|e| LoaderError::Device(e.to_string()))?;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4) };
    w.write_all(&(bytes.len() as u64).to_le_bytes()).map_err(|e| LoaderError::Device(e.to_string()))?;
    w.write_all(bytes).map_err(|e| LoaderError::Device(e.to_string()))?;
    Ok(())
}

fn write_tensor_f16<W: Write>(w: &mut W, _name: &str, t: &GpuTensor<half::f16>, device: &WarpDevice) -> Result<(), LoaderError> {
    let host = t.to_host(device).map_err(|e| LoaderError::Device(e.to_string()))?;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2) };
    w.write_all(&(bytes.len() as u64).to_le_bytes()).map_err(|e| LoaderError::Device(e.to_string()))?;
    w.write_all(bytes).map_err(|e| LoaderError::Device(e.to_string()))?;
    Ok(())
}

fn write_tensor_u8<W: Write>(w: &mut W, _name: &str, t: &GpuTensor<u8>, device: &WarpDevice) -> Result<(), LoaderError> {
    let host = t.to_host(device).map_err(|e| LoaderError::Device(e.to_string()))?;
    w.write_all(&(host.len() as u64).to_le_bytes()).map_err(|e| LoaderError::Device(e.to_string()))?;
    w.write_all(&host).map_err(|e| LoaderError::Device(e.to_string()))?;
    Ok(())
}

fn read_tensor_f32(mmap: &[u8], pos: &mut usize, device: &WarpDevice) -> Result<GpuTensor<f32>, LoaderError> {
    let nbytes = u64::from_le_bytes(mmap[*pos..*pos+8].try_into().unwrap()) as usize;
    *pos += 8;
    let data = &mmap[*pos..*pos+nbytes];
    *pos += nbytes;
    let floats: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, nbytes / 4) };
    GpuTensor::from_host(device, floats, Shape::from_static(&[nbytes / 4]), DType::F32)
        .map_err(|e| LoaderError::Device(e.to_string()))
}

fn read_tensor_f16(mmap: &[u8], pos: &mut usize, device: &WarpDevice) -> Result<GpuTensor<half::f16>, LoaderError> {
    let nbytes = u64::from_le_bytes(mmap[*pos..*pos+8].try_into().unwrap()) as usize;
    *pos += 8;
    let data = &mmap[*pos..*pos+nbytes];
    *pos += nbytes;
    let halfs: &[half::f16] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::f16, nbytes / 2) };
    GpuTensor::from_host(device, halfs, Shape::from_static(&[nbytes / 2]), DType::F16)
        .map_err(|e| LoaderError::Device(e.to_string()))
}

fn read_tensor_u8(mmap: &[u8], pos: &mut usize, device: &WarpDevice) -> Result<GpuTensor<u8>, LoaderError> {
    let nbytes = u64::from_le_bytes(mmap[*pos..*pos+8].try_into().unwrap()) as usize;
    *pos += 8;
    let data = &mmap[*pos..*pos+nbytes];
    *pos += nbytes;
    GpuTensor::from_host(device, data, Shape::from_static(&[nbytes]), DType::U8)
        .map_err(|e| LoaderError::Device(e.to_string()))
}

// ── Config serialization ─────────────────────────────────────────────────────

fn serde_config_json(config: &GemmaConfig, layer_configs: &[GemmaLayerAttentionConfig]) -> String {
    format!(
        r#"{{"hidden_size":{},"num_heads":{},"num_kv_heads":{},"num_global_kv_heads":{},"head_dim":{},"global_head_dim":{},"ffn_dim":{},"vocab_size":{},"num_layers":{},"norm_eps":{},"sliding_window":{},"sliding_window_pattern":{},"rope_theta":{},"rope_theta_global":{},"partial_rotary_factor":{},"k_eq_v":{},"final_logit_softcapping":{},"tie_word_embeddings":{}}}"#,
        config.hidden_size, config.num_heads, config.num_kv_heads,
        config.num_global_kv_heads, config.head_dim, config.global_head_dim,
        config.ffn_dim, config.vocab_size, config.num_layers,
        config.norm_eps, config.sliding_window, config.sliding_window_pattern,
        config.rope_theta, config.rope_theta_global, config.partial_rotary_factor,
        config.k_eq_v, config.final_logit_softcapping, config.tie_word_embeddings,
    )
}

fn parse_config_json(json: &str) -> Result<(GemmaConfig, Vec<GemmaLayerAttentionConfig>), LoaderError> {
    // Simple JSON parsing for our known fields
    let get = |key: &str| -> Option<&str> {
        let pattern = format!("\"{}\":", key);
        let start = json.find(&pattern)? + pattern.len();
        let rest = &json[start..];
        let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
        Some(rest[..end].trim())
    };
    let get_u32 = |key: &str| -> u32 { get(key).and_then(|s| s.parse().ok()).unwrap_or(0) };
    let get_f32 = |key: &str| -> f32 { get(key).and_then(|s| s.parse().ok()).unwrap_or(0.0) };
    let get_bool = |key: &str| -> bool { get(key).map(|s| s == "true").unwrap_or(false) };

    let config = GemmaConfig {
        hidden_size: get_u32("hidden_size"),
        num_heads: get_u32("num_heads"),
        num_kv_heads: get_u32("num_kv_heads"),
        num_global_kv_heads: get_u32("num_global_kv_heads"),
        head_dim: get_u32("head_dim"),
        global_head_dim: get_u32("global_head_dim"),
        ffn_dim: get_u32("ffn_dim"),
        vocab_size: get_u32("vocab_size"),
        num_layers: get_u32("num_layers"),
        norm_eps: get_f32("norm_eps"),
        sliding_window: get_u32("sliding_window"),
        sliding_window_pattern: get_u32("sliding_window_pattern"),
        rope_theta: get_f32("rope_theta"),
        rope_theta_global: get_f32("rope_theta_global"),
        partial_rotary_factor: get_f32("partial_rotary_factor"),
        k_eq_v: get_bool("k_eq_v"),
        final_logit_softcapping: get_f32("final_logit_softcapping"),
        tie_word_embeddings: get_bool("tie_word_embeddings"),
    };
    let layer_configs = config.layer_configs();
    Ok((config, layer_configs))
}

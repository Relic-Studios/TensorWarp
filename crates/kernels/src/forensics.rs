//! Forensic capture for the two MoE-Q4 generation paths.
//!
//! Two engines, two failure modes:
//!   * `generate_with_hook` — eager `forward_decode`. Correct for arbitrary
//!     prefill lengths but has collapsed to reserved tokens (`<unused6226>`)
//!     in observed runs.
//!   * `generate` — graph-replay via `forward_graph_capturable`. Fast, but
//!     trips `CUDA_ERROR_INVALID_VALUE` on long ForgeCode-style prompts.
//!
//! This module is a side-channel that records per-step state from both
//! paths so the two symptoms can be diffed on the same input during
//! ForgeCode operation. The recorder is env-gated on `DIDYMUS_FORENSICS_DIR`.
//! When the env var is unset, `ForensicRecorder::maybe_new` returns `None`
//! and every recording call is a cheap `Option::is_some` check.
//!
//! Output is a JSONL file at `{dir}/run_{run_id}.jsonl`. Hand-rolled
//! serialization — no serde dep on the kernels crate.
//!
//! The recorder does NOT control which path runs. It observes. Wiring it
//! into `generate_with_hook` and `generate` is the caller's job; a
//! recorder attached to both in one process produces paired records.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

/// Number of top logits to record per step. Small so readback is cheap.
const TOP_K_DEFAULT: usize = 8;

/// A per-run forensic log. Thread-safe — internal writer is behind a mutex.
///
/// Construct with [`ForensicRecorder::maybe_new`]; returns `None` when the
/// env var is unset so callers don't pay any cost in the normal path.
pub struct ForensicRecorder {
    writer: Mutex<BufWriter<File>>,
    run_id: String,
    path_file: PathBuf,
    top_k: usize,
}

impl ForensicRecorder {
    /// Returns `Some` when `DIDYMUS_FORENSICS_DIR` is set and the output
    /// file opened successfully. Returns `None` otherwise. Any I/O failure
    /// while opening the file is swallowed with a `log::warn` — forensics
    /// must never crash the inference path.
    pub fn maybe_new(run_id: impl Into<String>) -> Option<Self> {
        let dir = std::env::var("DIDYMUS_FORENSICS_DIR").ok()?;
        let dir = PathBuf::from(dir);
        if let Err(e) = std::fs::create_dir_all(&dir) {
            log::warn!("forensics: create_dir_all({:?}) failed: {}", dir, e);
            return None;
        }
        let run_id = run_id.into();
        let path_file = dir.join(format!("run_{}.jsonl", sanitize(&run_id)));
        let file = match OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path_file)
        {
            Ok(f) => f,
            Err(e) => {
                log::warn!("forensics: open({:?}) failed: {}", path_file, e);
                return None;
            }
        };
        let top_k = std::env::var("DIDYMUS_FORENSICS_TOPK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(TOP_K_DEFAULT);
        Some(Self {
            writer: Mutex::new(BufWriter::new(file)),
            run_id,
            path_file,
            top_k,
        })
    }

    /// The JSONL file being written. Useful for logging at startup.
    pub fn file(&self) -> &std::path::Path { &self.path_file }

    /// Record one step of the eager `generate_with_hook` path.
    ///
    /// `logits` is the host-side f32 vocab vector that `sample_token`
    /// just chose from. `residual` is the `bufs.output_scaled` tensor
    /// exposed by the residual hook — read back to host and summarized
    /// as {mean, max_abs, nan_count}. Readback uses the existing
    /// `GpuTensor::to_host` path; it is synchronous, so only enable
    /// forensics when you want the bench trace.
    pub fn record_eager_step(
        &self,
        device: &WarpDevice,
        step: usize,
        pos: u32,
        token: i32,
        prompt_len: usize,
        logits: &[f32],
        residual: &GpuTensor<f32>,
    ) {
        let stats = match residual.to_host(device) {
            Ok(v) => residual_stats(&v),
            Err(e) => {
                log::warn!("forensics: residual readback failed: {}", e);
                ResidualStats::unavailable()
            }
        };
        let top = top_k_logits(logits, self.top_k);
        let ts = now_ns();
        let mut line = String::with_capacity(256);
        line.push('{');
        write_kv_str(&mut line, "ts_ns", &ts.to_string(), true);
        write_kv_str(&mut line, "path", "\"eager\"", false);
        write_kv_str(&mut line, "run", &quoted(&self.run_id), false);
        write_kv_str(&mut line, "step", &step.to_string(), false);
        write_kv_str(&mut line, "pos", &pos.to_string(), false);
        write_kv_str(&mut line, "token", &token.to_string(), false);
        write_kv_str(&mut line, "prompt_len", &prompt_len.to_string(), false);
        write_kv_str(&mut line, "residual_mean", &f32_json(stats.mean), false);
        write_kv_str(&mut line, "residual_max_abs", &f32_json(stats.max_abs), false);
        write_kv_str(&mut line, "residual_nan", &stats.nan_count.to_string(), false);
        write_kv_str(&mut line, "residual_len", &stats.len.to_string(), false);
        line.push_str(",\"top\":");
        write_top_k(&mut line, &top);
        line.push('}');
        self.write_line(&line);
    }

    /// Record one step of the graph-replay `generate` path.
    ///
    /// Called unconditionally around `graph.replay()`. `cuda_err` is
    /// `Some(message)` when the replay failed — that is the forensically
    /// interesting case. `kv_lens` is one u32 per layer, captured before
    /// the replay so post-mortem analysis can see the state we fed in.
    pub fn record_graph_replay_step(
        &self,
        step: usize,
        pos: u32,
        prompt_len: usize,
        kv_lens: &[u32],
        cuda_err: Option<&str>,
    ) {
        let ts = now_ns();
        let mut line = String::with_capacity(192);
        line.push('{');
        write_kv_str(&mut line, "ts_ns", &ts.to_string(), true);
        write_kv_str(&mut line, "path", "\"graph_replay\"", false);
        write_kv_str(&mut line, "run", &quoted(&self.run_id), false);
        write_kv_str(&mut line, "step", &step.to_string(), false);
        write_kv_str(&mut line, "pos", &pos.to_string(), false);
        write_kv_str(&mut line, "prompt_len", &prompt_len.to_string(), false);
        line.push_str(",\"kv_lens\":");
        write_u32_array(&mut line, kv_lens);
        match cuda_err {
            Some(e) => {
                line.push_str(",\"cuda_err\":");
                line.push_str(&quoted(e));
            }
            None => line.push_str(",\"cuda_err\":null"),
        }
        line.push('}');
        self.write_line(&line);
    }

    fn write_line(&self, line: &str) {
        if let Ok(mut w) = self.writer.lock() {
            if let Err(e) = writeln!(w, "{}", line) {
                log::warn!("forensics: write failed: {}", e);
            }
        }
    }
}

impl Drop for ForensicRecorder {
    fn drop(&mut self) {
        if let Ok(mut w) = self.writer.lock() {
            let _ = w.flush();
        }
    }
}

struct ResidualStats {
    mean: f32,
    max_abs: f32,
    nan_count: u32,
    len: usize,
}

impl ResidualStats {
    fn unavailable() -> Self {
        Self { mean: f32::NAN, max_abs: f32::NAN, nan_count: 0, len: 0 }
    }
}

fn residual_stats(v: &[f32]) -> ResidualStats {
    let mut sum = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut nan_count: u32 = 0;
    let mut finite_n: usize = 0;
    for &x in v {
        if x.is_nan() {
            nan_count += 1;
            continue;
        }
        sum += x as f64;
        let a = x.abs();
        if a > max_abs { max_abs = a; }
        finite_n += 1;
    }
    let mean = if finite_n > 0 { (sum / finite_n as f64) as f32 } else { f32::NAN };
    ResidualStats { mean, max_abs, nan_count, len: v.len() }
}

fn top_k_logits(logits: &[f32], k: usize) -> Vec<(i32, f32)> {
    if logits.is_empty() || k == 0 { return Vec::new(); }
    let k = k.min(logits.len());
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.into_iter().map(|i| (i as i32, logits[i])).collect()
}

fn now_ns() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

fn quoted(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn f32_json(x: f32) -> String {
    if x.is_nan() || x.is_infinite() { "null".to_string() } else { format!("{}", x) }
}

fn write_kv_str(out: &mut String, k: &str, v: &str, first: bool) {
    if !first { out.push(','); }
    out.push('"'); out.push_str(k); out.push('"'); out.push(':');
    out.push_str(v);
}

fn write_top_k(out: &mut String, top: &[(i32, f32)]) {
    out.push('[');
    for (i, (tok, score)) in top.iter().enumerate() {
        if i > 0 { out.push(','); }
        out.push('[');
        out.push_str(&tok.to_string());
        out.push(',');
        out.push_str(&f32_json(*score));
        out.push(']');
    }
    out.push(']');
}

fn write_u32_array(out: &mut String, xs: &[u32]) {
    out.push('[');
    for (i, x) in xs.iter().enumerate() {
        if i > 0 { out.push(','); }
        out.push_str(&x.to_string());
    }
    out.push(']');
}

/// Thin helper so call sites can do `let rec = new_run_recorder(...)` and
/// stuff the `Option` into both `generate_with_hook` and `generate`.
pub fn new_run_recorder(prompt_hash: u64) -> Option<ForensicRecorder> {
    ForensicRecorder::maybe_new(format!("{:016x}", prompt_hash))
}

// Silence unused-field lint — DeviceError import keeps residual readback
// errors readable in log::warn.
#[allow(dead_code)]
fn _anchor_device_error(e: DeviceError) -> String { e.to_string() }

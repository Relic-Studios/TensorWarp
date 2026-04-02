//! Kernel compilation cache.
//!
//! NVRTC compilation takes 10-50ms per kernel. The cache stores compiled
//! CUDA functions keyed by source hash, so each unique kernel is compiled
//! exactly once per session.

use cudarc::driver::CudaFunction;
use rustc_hash::FxHashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::device::{DeviceError, WarpDevice};

/// Thread-safe kernel compilation cache with optional disk persistence.
///
/// Session cache: in-memory FxHashMap, ~1μs lookups.
/// Disk cache: PTX files in a cache directory, eliminates NVRTC on restart.
pub struct KernelCache {
    cache: Mutex<FxHashMap<u64, (CudaFunction, Duration)>>,
    hits: Mutex<u64>,
    misses: Mutex<u64>,
    total_compile_time: Mutex<Duration>,
    /// Directory for persistent PTX cache. None = session-only.
    disk_cache_dir: Option<std::path::PathBuf>,
}

impl KernelCache {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(FxHashMap::default()),
            hits: Mutex::new(0),
            misses: Mutex::new(0),
            total_compile_time: Mutex::new(Duration::ZERO),
            disk_cache_dir: None,
        }
    }

    /// Create a cache with disk persistence.
    /// Compiled PTX is saved to `dir` and reloaded on subsequent runs,
    /// skipping NVRTC compilation entirely.
    pub fn with_disk_cache(dir: impl Into<std::path::PathBuf>) -> Self {
        let dir = dir.into();
        let _ = std::fs::create_dir_all(&dir);
        Self {
            cache: Mutex::new(FxHashMap::default()),
            hits: Mutex::new(0),
            misses: Mutex::new(0),
            total_compile_time: Mutex::new(Duration::ZERO),
            disk_cache_dir: Some(dir),
        }
    }

    /// Get or compile a CUDA kernel. Returns cached version if available.
    ///
    /// Lookup order: memory cache → disk cache → NVRTC compile.
    pub fn get_or_compile(
        &self,
        device: &WarpDevice,
        cuda_src: &str,
        func_name: &str,
    ) -> Result<CudaFunction, DeviceError> {
        let key = hash_source(cuda_src, func_name);

        // Fast path: memory cache hit
        {
            let cache = self.cache.lock().unwrap();
            if let Some((func, _)) = cache.get(&key) {
                *self.hits.lock().unwrap() += 1;
                return Ok(func.clone());
            }
        }

        // Medium path: disk cache hit (load pre-compiled PTX file)
        if let Some(ref dir) = self.disk_cache_dir {
            let ptx_path = dir.join(format!("{:016x}.ptx", key));
            if ptx_path.exists() {
                let start = Instant::now();
                let ptx = cudarc::nvrtc::Ptx::from_file(&ptx_path);
                if let Ok(module) = device.ctx.load_module(ptx) {
                    if let Ok(function) = module.load_function(func_name) {
                        let load_time = start.elapsed();
                        let mut cache = self.cache.lock().unwrap();
                        cache.insert(key, (function.clone(), load_time));
                        *self.hits.lock().unwrap() += 1;
                        return Ok(function);
                    }
                }
                // Corrupted cache file — fall through to recompile
            }
        }

        // Slow path: NVRTC compile
        let start = Instant::now();
        let ptx = cudarc::nvrtc::compile_ptx(cuda_src)
            .map_err(|e| DeviceError::PtxLoad(e.to_string()))?;

        // Save PTX to disk cache for next startup
        if let Some(ref dir) = self.disk_cache_dir {
            let ptx_path = dir.join(format!("{:016x}.ptx", key));
            let ptx_text = ptx.to_src();
            let _ = std::fs::write(&ptx_path, ptx_text);
        }

        let module = device.ctx.load_module(ptx)
            .map_err(|e| DeviceError::PtxLoad(e.to_string()))?;
        let function = module.load_function(func_name)
            .map_err(|e| DeviceError::FuncNotFound(e.to_string()))?;
        let compile_time = start.elapsed();

        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key, (function.clone(), compile_time));
        }

        *self.misses.lock().unwrap() += 1;
        *self.total_compile_time.lock().unwrap() += compile_time;

        Ok(function)
    }

    /// Get or compile with custom include paths and arch.
    pub fn get_or_compile_with_opts(
        &self,
        device: &WarpDevice,
        cuda_src: &str,
        func_name: &str,
        include_paths: &[String],
        arch: Option<&'static str>,
    ) -> Result<CudaFunction, DeviceError> {
        let key = hash_source(cuda_src, func_name);

        {
            let cache = self.cache.lock().unwrap();
            if let Some((func, _)) = cache.get(&key) {
                *self.hits.lock().unwrap() += 1;
                return Ok(func.clone());
            }
        }

        let start = Instant::now();
        let (_module, function) = device.load_cuda_source_with_opts(cuda_src, func_name, include_paths, arch)?;
        let compile_time = start.elapsed();

        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key, (function.clone(), compile_time));
        }

        *self.misses.lock().unwrap() += 1;
        *self.total_compile_time.lock().unwrap() += compile_time;

        Ok(function)
    }

    pub fn len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn hit_rate(&self) -> f64 {
        let hits = *self.hits.lock().unwrap();
        let misses = *self.misses.lock().unwrap();
        let total = hits + misses;
        if total == 0 { 0.0 } else { hits as f64 / total as f64 }
    }

    pub fn stats(&self) -> CacheStats {
        let hits = *self.hits.lock().unwrap();
        let misses = *self.misses.lock().unwrap();
        let total_compile = *self.total_compile_time.lock().unwrap();
        let avg_compile = if misses > 0 { total_compile / misses as u32 } else { Duration::ZERO };
        CacheStats {
            entries: self.len(),
            hits,
            misses,
            total_compile_time: total_compile,
            hit_rate: self.hit_rate(),
            estimated_time_saved: avg_compile * hits as u32,
        }
    }
}

impl Default for KernelCache {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub total_compile_time: Duration,
    pub hit_rate: f64,
    pub estimated_time_saved: Duration,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KernelCache: {} entries, {}/{} hits ({:.1}%), {:.1}ms compile, ~{:.1}ms saved",
            self.entries, self.hits, self.hits + self.misses,
            self.hit_rate * 100.0,
            self.total_compile_time.as_secs_f64() * 1000.0,
            self.estimated_time_saved.as_secs_f64() * 1000.0,
        )
    }
}

fn hash_source(src: &str, func_name: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    src.hash(&mut hasher);
    func_name.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_hit_and_miss() {
        let dev = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        let cache = KernelCache::new();
        let src = r#"extern "C" __global__ void test_kern(float *out) { out[0] = 42.0f; }"#;

        // First: miss
        let _k1 = cache.get_or_compile(&dev, src, "test_kern").unwrap();

        // Second: hit
        let start = Instant::now();
        let _k2 = cache.get_or_compile(&dev, src, "test_kern").unwrap();
        let cached_time = start.elapsed();

        let stats = cache.stats();
        println!("Cache lookup: {:.3}μs", cached_time.as_secs_f64() * 1e6);
        println!("{stats}");
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }
}

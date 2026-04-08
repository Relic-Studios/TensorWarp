//! `.warp` file format — pre-quantized weight cache for fast model loading.
//!
//! First run: load BF16/F32 weights → quantize to Q4 → save `.warp` file.
//! Subsequent runs: read `.warp` → upload to GPU. Skips all quantization.
//!
//! Format:
//! ```text
//! [8 bytes]  magic: b"WARP0001"
//! [4 bytes]  config_len: u32 (little-endian)
//! [N bytes]  config_json: UTF-8 JSON (model config + layer configs)
//! [4 bytes]  num_tensors: u32
//! For each tensor:
//!   [4 bytes]  name_len: u32
//!   [N bytes]  name: UTF-8
//!   [4 bytes]  dtype: u32 (0=F32, 1=F16, 2=U8/Q4)
//!   [4 bytes]  ndims: u32
//!   [ndims*8]  shape: [u64; ndims]
//!   [8 bytes]  data_len: u64
//!   [N bytes]  data: raw bytes
//! ```

use std::io::{Read, Write, BufWriter, BufReader};
use std::path::Path;

const MAGIC: &[u8; 8] = b"WARP0001";

/// Write a .warp cache file from pre-quantized tensors.
pub fn save_warp_cache(
    path: &Path,
    config_json: &str,
    tensors: &[(&str, u32, &[usize], &[u8])], // (name, dtype, shape, data)
) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    // Magic
    w.write_all(MAGIC)?;

    // Config JSON
    let config_bytes = config_json.as_bytes();
    w.write_all(&(config_bytes.len() as u32).to_le_bytes())?;
    w.write_all(config_bytes)?;

    // Tensor count
    w.write_all(&(tensors.len() as u32).to_le_bytes())?;

    // Tensors
    for (name, dtype, shape, data) in tensors {
        let name_bytes = name.as_bytes();
        w.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        w.write_all(name_bytes)?;
        w.write_all(&dtype.to_le_bytes())?;
        w.write_all(&(shape.len() as u32).to_le_bytes())?;
        for &dim in *shape {
            w.write_all(&(dim as u64).to_le_bytes())?;
        }
        w.write_all(&(data.len() as u64).to_le_bytes())?;
        w.write_all(data)?;
    }

    w.flush()?;
    Ok(())
}

/// Tensor entry read from a .warp file.
pub struct WarpTensor {
    pub name: String,
    pub dtype: u32,    // 0=F32, 1=F16, 2=U8/Q4
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

/// Read a .warp cache file.
pub fn load_warp_cache(path: &Path) -> std::io::Result<(String, Vec<WarpTensor>)> {
    let file = std::fs::File::open(path)?;
    let mut r = BufReader::new(file);

    // Magic
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("not a .warp file (magic: {:?})", magic)));
    }

    // Config JSON
    let mut config_len_buf = [0u8; 4];
    r.read_exact(&mut config_len_buf)?;
    let config_len = u32::from_le_bytes(config_len_buf) as usize;
    let mut config_bytes = vec![0u8; config_len];
    r.read_exact(&mut config_bytes)?;
    let config_json = String::from_utf8_lossy(&config_bytes).to_string();

    // Tensor count
    let mut count_buf = [0u8; 4];
    r.read_exact(&mut count_buf)?;
    let num_tensors = u32::from_le_bytes(count_buf) as usize;

    // Read tensors
    let mut tensors = Vec::with_capacity(num_tensors);
    for _ in 0..num_tensors {
        // Name
        let mut name_len_buf = [0u8; 4];
        r.read_exact(&mut name_len_buf)?;
        let name_len = u32::from_le_bytes(name_len_buf) as usize;
        let mut name_bytes = vec![0u8; name_len];
        r.read_exact(&mut name_bytes)?;
        let name = String::from_utf8_lossy(&name_bytes).to_string();

        // Dtype
        let mut dtype_buf = [0u8; 4];
        r.read_exact(&mut dtype_buf)?;
        let dtype = u32::from_le_bytes(dtype_buf);

        // Shape
        let mut ndims_buf = [0u8; 4];
        r.read_exact(&mut ndims_buf)?;
        let ndims = u32::from_le_bytes(ndims_buf) as usize;
        let mut shape = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            let mut dim_buf = [0u8; 8];
            r.read_exact(&mut dim_buf)?;
            shape.push(u64::from_le_bytes(dim_buf) as usize);
        }

        // Data
        let mut data_len_buf = [0u8; 8];
        r.read_exact(&mut data_len_buf)?;
        let data_len = u64::from_le_bytes(data_len_buf) as usize;
        let mut data = vec![0u8; data_len];
        r.read_exact(&mut data)?;

        tensors.push(WarpTensor { name, dtype, shape, data });
    }

    Ok((config_json, tensors))
}

/// Check if a .warp cache exists for a model directory.
pub fn warp_cache_path(model_dir: &Path) -> std::path::PathBuf {
    model_dir.join("model.warp")
}

/// Check if .warp cache exists and is valid.
pub fn has_warp_cache(model_dir: &Path) -> bool {
    let cache_path = warp_cache_path(model_dir);
    if !cache_path.exists() { return false; }

    // Quick validation: check magic bytes
    if let Ok(mut f) = std::fs::File::open(&cache_path) {
        let mut magic = [0u8; 8];
        if f.read_exact(&mut magic).is_ok() && &magic == MAGIC {
            return true;
        }
    }
    false
}

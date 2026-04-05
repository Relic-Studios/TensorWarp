//! GGUF model loader — parse llama.cpp's GGUF binary format.
//!
//! GGUF is the standard format for quantized LLM weights used by llama.cpp,
//! ollama, and many other local inference tools.

use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

/// Magic number for GGUF files: "GGUF" in little-endian.
const GGUF_MAGIC: u32 = 0x46475547;

// ─── Error ───────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    InvalidMagic(u32),
    UnsupportedVersion(u32),
    InvalidValueType(u32),
    InvalidDType(u32),
    InvalidUtf8,
    TensorNotFound(String),
    UnsupportedDType(GgufDType),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::InvalidMagic(m) => write!(f, "invalid GGUF magic: 0x{m:08X} (expected 0x{GGUF_MAGIC:08X})"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported GGUF version: {v}"),
            Self::InvalidValueType(t) => write!(f, "invalid metadata value type: {t}"),
            Self::InvalidDType(t) => write!(f, "invalid tensor dtype: {t}"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8 in string"),
            Self::TensorNotFound(name) => write!(f, "tensor not found: {name}"),
            Self::UnsupportedDType(dt) => write!(f, "dequantization not supported for {dt:?}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<io::Error> for GgufError {
    fn from(e: io::Error) -> Self { Self::Io(e) }
}

// ─── Types ───────────────────────────────────────────────────────────────────

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    /// Try to extract a u64 value (coercing from smaller unsigned types).
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(v) => Some(*v as u64),
            Self::Uint16(v) => Some(*v as u64),
            Self::Uint32(v) => Some(*v as u64),
            Self::Uint64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract a string reference.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// GGUF tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
}

impl GgufDType {
    fn from_u32(v: u32) -> Result<Self, GgufError> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            _ => Err(GgufError::InvalidDType(v)),
        }
    }

    /// Bytes per element (for unquantized types) or bytes per block.
    pub fn block_size(&self) -> (usize, usize) {
        // (elements_per_block, bytes_per_block)
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q4_0 => (32, 18),   // 32 elements: 2 bytes scale + 16 bytes quants
            Self::Q4_1 => (32, 20),   // 32 elements: 2+2 bytes scale/min + 16 bytes quants
            Self::Q5_0 => (32, 22),   // 32 elements: 2 bytes scale + 4 bytes high bits + 16 bytes quants
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),   // 32 elements: 2 bytes scale + 32 bytes quants
            Self::Q8_1 => (32, 36),
        }
    }
}

/// Information about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgufDType,
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total number of elements.
    pub fn numel(&self) -> u64 {
        self.dims.iter().product::<u64>().max(1)
    }

    /// Size in bytes of the tensor data.
    pub fn byte_size(&self) -> u64 {
        let n = self.numel() as usize;
        let (elems_per_block, bytes_per_block) = self.dtype.block_size();
        let n_blocks = (n + elems_per_block - 1) / elems_per_block;
        (n_blocks * bytes_per_block) as u64
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// A parsed GGUF model file.
pub struct GgufModel {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data: Vec<u8>,
}

impl GgufModel {
    /// Load a GGUF file from disk.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let mut file = std::fs::File::open(path.as_ref())?;
        Self::read_from(&mut file)
    }

    /// Parse GGUF from any reader that supports Read + Seek.
    pub fn read_from<R: Read + Seek>(r: &mut R) -> Result<Self, GgufError> {
        // ── Header ───────────────────────────────────────────────────────
        let magic = read_u32(r)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = read_u32(r)?;
        if version < 2 || version > 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let n_tensors = read_u64(r)?;
        let n_kv = read_u64(r)?;

        // ── Metadata KV ──────────────────────────────────────────────────
        let mut metadata = HashMap::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = read_gguf_string(r)?;
            let value = read_gguf_value(r)?;
            metadata.insert(key, value);
        }

        // ── Tensor infos ─────────────────────────────────────────────────
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = read_gguf_string(r)?;
            let n_dims = read_u32(r)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(r)?);
            }
            let dtype = GgufDType::from_u32(read_u32(r)?)?;
            let offset = read_u64(r)?;
            tensors.push(GgufTensorInfo { name, dims, dtype, offset });
        }

        // ── Data section ─────────────────────────────────────────────────
        // Alignment (default 32)
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as u64;

        // Current position in the file
        let pos = r.stream_position()?;
        // Align to the next boundary
        let aligned_pos = (pos + alignment - 1) / alignment * alignment;
        r.seek(SeekFrom::Start(aligned_pos))?;

        // Read remaining data
        let mut data = Vec::new();
        r.read_to_end(&mut data)?;

        Ok(Self { version, metadata, tensors, data })
    }

    /// Get tensor data as f32 (dequantizing if needed). Only F32 supported for now.
    pub fn get_tensor_f32(&self, name: &str) -> Option<Vec<f32>> {
        let info = self.tensors.iter().find(|t| t.name == name)?;
        let offset = info.offset as usize;
        let byte_size = info.byte_size() as usize;

        if offset + byte_size > self.data.len() {
            return None;
        }

        let raw = &self.data[offset..offset + byte_size];
        let numel = info.numel() as usize;

        match info.dtype {
            GgufDType::F32 => {
                let mut out = vec![0.0f32; numel];
                for (i, chunk) in raw.chunks_exact(4).enumerate().take(numel) {
                    out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                Some(out)
            }
            GgufDType::F16 => {
                let mut out = vec![0.0f32; numel];
                for (i, chunk) in raw.chunks_exact(2).enumerate().take(numel) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out[i] = half::f16::from_bits(bits).to_f32();
                }
                Some(out)
            }
            GgufDType::Q8_0 => {
                // Q8_0: blocks of 32 elements, each block = 2 bytes scale (f16) + 32 bytes quants (i8)
                let mut out = vec![0.0f32; numel];
                let (elems_per_block, bytes_per_block) = info.dtype.block_size();
                let n_blocks = numel / elems_per_block;
                for bi in 0..n_blocks {
                    let block_start = bi * bytes_per_block;
                    let scale_bits = u16::from_le_bytes([raw[block_start], raw[block_start + 1]]);
                    let scale = half::f16::from_bits(scale_bits).to_f32();
                    for j in 0..32 {
                        let q = raw[block_start + 2 + j] as i8;
                        out[bi * 32 + j] = q as f32 * scale;
                    }
                }
                Some(out)
            }
            _ => None, // Other quantization formats not yet implemented
        }
    }

    /// Human-readable summary of the model.
    pub fn summary(&self) -> String {
        let arch = self.architecture().unwrap_or("unknown");
        let ctx = self.context_length().map(|v| v.to_string()).unwrap_or_else(|| "?".into());
        let total_bytes: u64 = self.tensors.iter().map(|t| t.byte_size()).sum();
        let mb = total_bytes as f64 / 1024.0 / 1024.0;

        let mut s = format!(
            "GGUF v{} | arch={} | ctx={} | {} tensors | {:.1} MB\n",
            self.version, arch, ctx, self.tensors.len(), mb
        );

        // Show first 20 tensors
        for (i, t) in self.tensors.iter().take(20).enumerate() {
            s.push_str(&format!(
                "  [{:3}] {:50} {:?} {:?}\n",
                i, t.name, t.dtype, t.dims
            ));
        }
        if self.tensors.len() > 20 {
            s.push_str(&format!("  ... and {} more tensors\n", self.tensors.len() - 20));
        }
        s
    }

    /// Get the model architecture from metadata.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture")?.as_str()
    }

    /// Get the context length from metadata.
    pub fn context_length(&self) -> Option<u64> {
        // Try architecture-specific key first, then generic
        if let Some(arch) = self.architecture() {
            let key = format!("{arch}.context_length");
            if let Some(v) = self.metadata.get(&key) {
                return v.as_u64();
            }
        }
        self.metadata.get("general.context_length")?.as_u64()
    }
}

// ─── Binary reader helpers ───────────────────────────────────────────────────

fn read_u8<R: Read>(r: &mut R) -> Result<u8, GgufError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8<R: Read>(r: &mut R) -> Result<i8, GgufError> {
    Ok(read_u8(r)? as i8)
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16, GgufError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16<R: Read>(r: &mut R) -> Result<i16, GgufError> {
    Ok(read_u16(r)? as i16)
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, GgufError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32, GgufError> {
    Ok(read_u32(r)? as i32)
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, GgufError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64, GgufError> {
    Ok(read_u64(r)? as i64)
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32, GgufError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(r: &mut R) -> Result<f64, GgufError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_gguf_string<R: Read>(r: &mut R) -> Result<String, GgufError> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| GgufError::InvalidUtf8)
}

fn read_gguf_value<R: Read>(r: &mut R) -> Result<GgufValue, GgufError> {
    let vtype = read_u32(r)?;
    match vtype {
        0 => Ok(GgufValue::Uint8(read_u8(r)?)),
        1 => Ok(GgufValue::Int32(read_i8(r)? as i32)), // int8 → store as i32
        2 => Ok(GgufValue::Uint32(read_u16(r)? as u32)),
        3 => Ok(GgufValue::Int32(read_i16(r)? as i32)),
        4 => Ok(GgufValue::Uint32(read_u32(r)?)),
        5 => Ok(GgufValue::Int32(read_i32(r)?)),
        6 => Ok(GgufValue::Float32(read_f32(r)?)),
        7 => {
            let b = read_u8(r)?;
            Ok(GgufValue::Bool(b != 0))
        }
        8 => Ok(GgufValue::String(read_gguf_string(r)?)),
        9 => {
            // Array: element type (u32) + count (u64) + elements
            let elem_type = read_u32(r)?;
            let count = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                // Read element of the declared type (without re-reading the type tag)
                let val = read_gguf_value_of_type(r, elem_type)?;
                arr.push(val);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::Uint64(read_u64(r)?)),
        11 => Ok(GgufValue::Int64(read_i64(r)?)),
        12 => Ok(GgufValue::Float64(read_f64(r)?)),
        _ => Err(GgufError::InvalidValueType(vtype)),
    }
}

fn read_gguf_value_of_type<R: Read>(r: &mut R, vtype: u32) -> Result<GgufValue, GgufError> {
    match vtype {
        0 => Ok(GgufValue::Uint8(read_u8(r)?)),
        1 => Ok(GgufValue::Int32(read_i8(r)? as i32)),
        2 => Ok(GgufValue::Uint32(read_u16(r)? as u32)),
        3 => Ok(GgufValue::Int32(read_i16(r)? as i32)),
        4 => Ok(GgufValue::Uint32(read_u32(r)?)),
        5 => Ok(GgufValue::Int32(read_i32(r)?)),
        6 => Ok(GgufValue::Float32(read_f32(r)?)),
        7 => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        8 => Ok(GgufValue::String(read_gguf_string(r)?)),
        10 => Ok(GgufValue::Uint64(read_u64(r)?)),
        11 => Ok(GgufValue::Int64(read_i64(r)?)),
        12 => Ok(GgufValue::Float64(read_f64(r)?)),
        _ => Err(GgufError::InvalidValueType(vtype)),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Write helpers for constructing test GGUF files.
    fn write_u32(buf: &mut Vec<u8>, v: u32) { buf.extend_from_slice(&v.to_le_bytes()); }
    fn write_u64(buf: &mut Vec<u8>, v: u64) { buf.extend_from_slice(&v.to_le_bytes()); }
    fn write_f32(buf: &mut Vec<u8>, v: f32) { buf.extend_from_slice(&v.to_le_bytes()); }
    fn write_string(buf: &mut Vec<u8>, s: &str) {
        write_u64(buf, s.len() as u64);
        buf.extend_from_slice(s.as_bytes());
    }

    fn build_minimal_gguf() -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        write_u32(&mut buf, GGUF_MAGIC);   // magic
        write_u32(&mut buf, 3);            // version
        write_u64(&mut buf, 1);            // n_tensors
        write_u64(&mut buf, 2);            // n_kv

        // KV 0: general.architecture = "llama"
        write_string(&mut buf, "general.architecture");
        write_u32(&mut buf, 8); // string type
        write_string(&mut buf, "llama");

        // KV 1: llama.context_length = 4096
        write_string(&mut buf, "llama.context_length");
        write_u32(&mut buf, 4); // uint32 type
        write_u32(&mut buf, 4096);

        // Tensor info: "weight" — 2x3 F32
        write_string(&mut buf, "weight");
        write_u32(&mut buf, 2); // n_dims
        write_u64(&mut buf, 2); // dim 0
        write_u64(&mut buf, 3); // dim 1
        write_u32(&mut buf, 0); // F32
        write_u64(&mut buf, 0); // offset in data section

        // Align to 32 bytes for data section
        let pos = buf.len();
        let aligned = (pos + 31) / 32 * 32;
        buf.resize(aligned, 0);

        // Tensor data: 6 f32 values
        for i in 0..6 {
            write_f32(&mut buf, i as f32 * 1.5);
        }

        buf
    }

    #[test]
    fn gguf_header_parsing() {
        let data = build_minimal_gguf();
        let mut cursor = Cursor::new(data);
        let model = GgufModel::read_from(&mut cursor).expect("failed to parse GGUF");

        assert_eq!(model.version, 3);
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.architecture(), Some("llama"));
        assert_eq!(model.context_length(), Some(4096));

        // Check tensor info
        let t = &model.tensors[0];
        assert_eq!(t.name, "weight");
        assert_eq!(t.dims, vec![2, 3]);
        assert_eq!(t.dtype, GgufDType::F32);
        assert_eq!(t.numel(), 6);

        // Read tensor data
        let values = model.get_tensor_f32("weight").expect("failed to read tensor");
        assert_eq!(values.len(), 6);
        for i in 0..6 {
            assert!((values[i] - i as f32 * 1.5).abs() < 1e-6, "mismatch at {i}");
        }

        // Summary should not panic
        let summary = model.summary();
        assert!(summary.contains("llama"));
        assert!(summary.contains("weight"));
    }
}

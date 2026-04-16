//! GGUF model loader — parse llama.cpp's GGUF binary format.
//!
//! GGUF is the standard format for quantized LLM weights used by llama.cpp,
//! ollama, and many other local inference tools.

use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;
use memmap2::Mmap;

/// Magic number for GGUF files: "GGUF" in little-endian.
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as u32 little-endian

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
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
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
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            _ => Err(GgufError::InvalidDType(v)),
        }
    }

    /// Bytes per element (for unquantized types) or bytes per block.
    pub fn block_size(&self) -> (usize, usize) {
        // (elements_per_block, bytes_per_block)
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q4_0 => (32, 18),    // 2B scale + 16B quants
            Self::Q4_1 => (32, 20),    // 2+2B scale/min + 16B quants
            Self::Q5_0 => (32, 22),    // 2B scale + 4B high + 16B quants
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),    // 2B scale + 32B quants
            Self::Q8_1 => (32, 36),
            Self::Q2_K => (256, 84),   // 256 elems: scales + quants
            Self::Q3_K => (256, 110),
            Self::Q4_K => (256, 144),  // 2B d + 2B dmin + 12B scales + 128B qs
            Self::Q5_K => (256, 176),
            Self::Q6_K => (256, 210),  // 128B ql + 64B qh + 16B scales + 2B d
            Self::Q8_K => (256, 292),
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

/// A parsed GGUF model file. Uses mmap for large files.
pub struct GgufModel {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    /// Raw tensor data — either owned Vec or mmap'd slice
    data: GgufData,
    /// Offset from start of file to the data section (for mmap mode)
    data_offset: u64,
}

enum GgufData {
    Owned(Vec<u8>),
    Mmap(Mmap),
}

impl GgufData {
    fn as_slice(&self, data_offset: u64) -> &[u8] {
        match self {
            GgufData::Owned(v) => v.as_slice(),
            GgufData::Mmap(m) => &m[data_offset as usize..],
        }
    }
}

impl GgufModel {
    /// Load a GGUF file from disk. Uses mmap for files > 1 GB.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path.as_ref())?;
        let file_size = file.metadata()?.len();

        if file_size > 1_000_000_000 {
            // Large file: mmap + parse header from the mmap
            Self::load_mmap(file)
        } else {
            let mut file = file;
            Self::read_from(&mut file)
        }
    }

    /// Load using memory-mapped I/O (for large GGUF files).
    fn load_mmap(file: std::fs::File) -> Result<Self, GgufError> {
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| GgufError::Io(e))?;
        let mut cursor = std::io::Cursor::new(&mmap[..]);

        let magic = read_u32(&mut cursor)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = read_u32(&mut cursor)?;
        if version < 2 || version > 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let n_tensors = read_u64(&mut cursor)?;
        let n_kv = read_u64(&mut cursor)?;

        let mut metadata = HashMap::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = read_gguf_string(&mut cursor)?;
            let value = read_gguf_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = read_gguf_string(&mut cursor)?;
            let n_dims = read_u32(&mut cursor)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut cursor)?);
            }
            let dtype = GgufDType::from_u32(read_u32(&mut cursor)?)?;
            let offset = read_u64(&mut cursor)?;
            tensors.push(GgufTensorInfo { name, dims, dtype, offset });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as u64;

        let pos = cursor.position();
        let data_offset = (pos + alignment - 1) / alignment * alignment;

        Ok(Self {
            version,
            metadata,
            tensors,
            data: GgufData::Mmap(mmap),
            data_offset,
        })
    }

    /// Parse GGUF from any reader that supports Read + Seek.
    pub fn read_from<R: Read + Seek>(r: &mut R) -> Result<Self, GgufError> {
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

        let mut metadata = HashMap::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = read_gguf_string(r)?;
            let value = read_gguf_value(r)?;
            metadata.insert(key, value);
        }

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

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as u64;

        let pos = r.stream_position()?;
        let aligned_pos = (pos + alignment - 1) / alignment * alignment;
        r.seek(SeekFrom::Start(aligned_pos))?;

        let mut data = Vec::new();
        r.read_to_end(&mut data)?;

        Ok(Self { version, metadata, tensors, data: GgufData::Owned(data), data_offset: 0 })
    }

    /// Get the raw bytes for a tensor.
    fn tensor_raw(&self, name: &str) -> Option<(&GgufTensorInfo, &[u8])> {
        let info = self.tensors.iter().find(|t| t.name == name)?;
        let data_slice = self.data.as_slice(self.data_offset);
        let offset = info.offset as usize;
        let byte_size = info.byte_size() as usize;
        if offset + byte_size > data_slice.len() {
            return None;
        }
        Some((info, &data_slice[offset..offset + byte_size]))
    }

    /// Get raw bytes for a tensor (no dequantization).
    pub fn get_tensor_raw(&self, name: &str) -> Option<&[u8]> {
        let (info, raw) = self.tensor_raw(name)?;
        Some(raw)
    }

    /// Get tensor data as f32 (dequantizing if needed).
    pub fn get_tensor_f32(&self, name: &str) -> Option<Vec<f32>> {
        let (info, raw) = self.tensor_raw(name)?;
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
            GgufDType::Q4_K => Some(dequantize_q4_k(raw, numel)),
            GgufDType::Q6_K => Some(dequantize_q6_k(raw, numel)),
            GgufDType::Q5_K => Some(dequantize_q5_k(raw, numel)),
            GgufDType::Q8_K => Some(dequantize_q8_k(raw, numel)),
            GgufDType::Q4_0 => Some(dequantize_q4_0(raw, numel)),
            GgufDType::Q5_0 => Some(dequantize_q5_0(raw, numel)),
            _ => None,
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

// ─── K-quant dequantization ──────────────────────────────────────────────────

/// Extract 6-bit scale and min for Q4_K sub-block `j` from the 12-byte scales array.
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, mn)
    }
}

/// Dequantize Q4_K blocks to f32.
/// Block layout (144 bytes per 256 elements):
///   d: f16 (2B), dmin: f16 (2B), scales: [12]u8, qs: [128]u8
fn dequantize_q4_k(raw: &[u8], numel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; numel];
    let block_bytes = 144;
    let n_blocks = numel / 256;

    for bi in 0..n_blocks {
        let b = &raw[bi * block_bytes..(bi + 1) * block_bytes];
        let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        let dmin = half::f16::from_bits(u16::from_le_bytes([b[2], b[3]])).to_f32();
        let scales = &b[4..16];
        let qs = &b[16..144];

        let out_block = &mut out[bi * 256..(bi + 1) * 256];
        let mut is = 0usize;
        let mut q_off = 0usize;
        // 4 chunks of 64 elements each
        for _chunk in 0..4 {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * m0 as f32;
            let (sc1, m1_val) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * m1_val as f32;

            let elem_off = _chunk * 64;
            for l in 0..32 {
                out_block[elem_off + l] = d1 * (qs[q_off + l] & 0xF) as f32 - m1;
            }
            for l in 0..32 {
                out_block[elem_off + 32 + l] = d2 * (qs[q_off + l] >> 4) as f32 - m2;
            }
            q_off += 32;
            is += 2;
        }
    }
    out
}

/// Dequantize Q6_K blocks to f32.
/// Block layout (210 bytes per 256 elements):
///   ql: [128]u8, qh: [64]u8, scales: [16]i8, d: f16 (2B)
fn dequantize_q6_k(raw: &[u8], numel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; numel];
    let block_bytes = 210;
    let n_blocks = numel / 256;

    for bi in 0..n_blocks {
        let b = &raw[bi * block_bytes..(bi + 1) * block_bytes];
        let ql = &b[0..128];
        let qh = &b[128..192];
        let sc = &b[192..208]; // 16 x i8
        let d = half::f16::from_bits(u16::from_le_bytes([b[208], b[209]])).to_f32();

        let out_block = &mut out[bi * 256..(bi + 1) * 256];

        // Process in 2 halves of 128 elements each
        for half_idx in 0..2usize {
            let ql_off = half_idx * 64;
            let qh_off = half_idx * 32;
            let sc_off = half_idx * 8;
            let out_off = half_idx * 128;

            for l in 0..32 {
                let q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_off + 32 + l] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[ql_off + 32 + l] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

                out_block[out_off + l]      = d * sc[sc_off] as i8 as f32 * q1 as f32;
                out_block[out_off + 32 + l] = d * sc[sc_off + 2] as i8 as f32 * q2 as f32;
                out_block[out_off + 64 + l] = d * sc[sc_off + 4] as i8 as f32 * q3 as f32;
                out_block[out_off + 96 + l] = d * sc[sc_off + 6] as i8 as f32 * q4 as f32;
            }
        }
    }
    out
}

/// Dequantize Q5_K blocks to f32.
/// Block layout (176 bytes per 256 elements):
///   d: f16 (2B), dmin: f16 (2B), scales: [12]u8, qh: [32]u8, qs: [128]u8
fn dequantize_q5_k(raw: &[u8], numel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; numel];
    let block_bytes = 176;
    let n_blocks = numel / 256;

    for bi in 0..n_blocks {
        let b = &raw[bi * block_bytes..(bi + 1) * block_bytes];
        let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        let dmin = half::f16::from_bits(u16::from_le_bytes([b[2], b[3]])).to_f32();
        let scales = &b[4..16];
        let qh = &b[16..48];  // 32 bytes = 256 high bits
        let qs = &b[48..176]; // 128 bytes = 256 low nibbles

        let out_block = &mut out[bi * 256..(bi + 1) * 256];
        let mut is = 0usize;
        let mut q_off = 0usize;

        for chunk in 0..4 {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * m0 as f32;
            let (sc1, m1_val) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * m1_val as f32;

            let elem_off = chunk * 64;
            let qh_bit_base = chunk * 64; // bit offset into qh

            for l in 0..32 {
                // Low nibble + high bit for first 32 elements
                let qh_byte = qh_bit_base + l;
                let hb0 = ((qh[qh_byte / 8] >> (qh_byte % 8)) & 1) as u8;
                let q_low = qs[q_off + l] & 0xF;
                out_block[elem_off + l] = d1 * (q_low | (hb0 << 4)) as f32 - m1;
            }
            for l in 0..32 {
                let qh_byte = qh_bit_base + 32 + l;
                let hb1 = ((qh[qh_byte / 8] >> (qh_byte % 8)) & 1) as u8;
                let q_high = qs[q_off + l] >> 4;
                out_block[elem_off + 32 + l] = d2 * (q_high | (hb1 << 4)) as f32 - m2;
            }
            q_off += 32;
            is += 2;
        }
    }
    out
}

/// Dequantize Q8_K blocks to f32.
/// Block layout (292 bytes per 256 elements):
///   d: f32 (4B), qs: [256]i8, bsums: [16]i16
fn dequantize_q8_k(raw: &[u8], numel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; numel];
    let block_bytes = 292;
    let n_blocks = numel / 256;

    for bi in 0..n_blocks {
        let b = &raw[bi * block_bytes..(bi + 1) * block_bytes];
        let d = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let qs = &b[4..260]; // 256 bytes of i8

        for j in 0..256 {
            out[bi * 256 + j] = d * qs[j] as i8 as f32;
        }
    }
    out
}

/// Dequantize Q4_0 blocks to f32.
/// Block layout (18 bytes per 32 elements):
///   scale: f16 (2B), qs: [16]u8 (packed 4-bit, unsigned with bias -8)
fn dequantize_q4_0(raw: &[u8], numel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; numel];
    let block_bytes = 18;
    let n_blocks = numel / 32;

    for bi in 0..n_blocks {
        let b = &raw[bi * block_bytes..(bi + 1) * block_bytes];
        let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        let qs = &b[2..18];

        for j in 0..16 {
            let lo = (qs[j] & 0xF) as i32 - 8;
            let hi = (qs[j] >> 4) as i32 - 8;
            out[bi * 32 + 2 * j] = d * lo as f32;
            out[bi * 32 + 2 * j + 1] = d * hi as f32;
        }
    }
    out
}

/// Dequantize Q5_0 blocks to f32.
/// Block layout (22 bytes per 32 elements):
///   scale: f16 (2B), qh: [4]u8 (32 high bits), qs: [16]u8 (packed 4-bit low nibbles)
fn dequantize_q5_0(raw: &[u8], numel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; numel];
    let block_bytes = 22;
    let n_blocks = numel / 32;

    for bi in 0..n_blocks {
        let b = &raw[bi * block_bytes..(bi + 1) * block_bytes];
        let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        let qh_bytes = &b[2..6]; // 4 bytes = 32 bits
        let qh = u32::from_le_bytes([qh_bytes[0], qh_bytes[1], qh_bytes[2], qh_bytes[3]]);
        let qs = &b[6..22];

        for j in 0..16 {
            let lo_nibble = (qs[j] & 0xF) as u32;
            let hi_nibble = (qs[j] >> 4) as u32;
            let hb0 = (qh >> (2 * j)) & 1;
            let hb1 = (qh >> (2 * j + 1)) & 1;
            let q0 = (lo_nibble | (hb0 << 4)) as i32 - 16;
            let q1 = (hi_nibble | (hb1 << 4)) as i32 - 16;
            out[bi * 32 + 2 * j] = d * q0 as f32;
            out[bi * 32 + 2 * j + 1] = d * q1 as f32;
        }
    }
    out
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

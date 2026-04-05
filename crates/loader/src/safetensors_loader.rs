//! SafeTensors format loader.
//!
//! SafeTensors is a simple, safe format for storing tensors:
//! - 8-byte header length (u64 LE)
//! - JSON header with tensor metadata (name, dtype, shape, offsets)
//! - Raw tensor data (contiguous, no padding)
//!
//! We memory-map the file for zero-copy access and convert dtypes
//! on the fly when uploading to GPU.

use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use warp_ir::{DType, Shape};
use warp_kernels::device::{DeviceError, WarpDevice};
use warp_kernels::tensor::GpuTensor;

/// A loaded SafeTensors file, memory-mapped for zero-copy access.
pub struct SafeTensorsLoader {
    /// Memory-mapped file data.
    mmap: Mmap,
    /// Path for error messages.
    path: String,
}

impl SafeTensorsLoader {
    /// Open a SafeTensors file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LoaderError> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file = File::open(path.as_ref())
            .map_err(|e| LoaderError::Io(format!("{path_str}: {e}")))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| LoaderError::Io(format!("{path_str}: mmap failed: {e}")))?;

        Ok(Self {
            mmap,
            path: path_str,
        })
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> Result<Vec<String>, LoaderError> {
        let st = SafeTensors::deserialize(&self.mmap)
            .map_err(|e| LoaderError::Parse(e.to_string()))?;
        Ok(st.names().into_iter().map(|s| s.to_string()).collect())
    }

    /// Get tensor info (shape, dtype) without loading data.
    pub fn tensor_info(&self, name: &str) -> Result<TensorInfo, LoaderError> {
        let st = SafeTensors::deserialize(&self.mmap)
            .map_err(|e| LoaderError::Parse(e.to_string()))?;
        let view = st.tensor(name)
            .map_err(|e| LoaderError::TensorNotFound(name.to_string(), e.to_string()))?;

        let shape: Vec<usize> = view.shape().to_vec();
        let dtype = convert_dtype(view.dtype());

        Ok(TensorInfo { name: name.to_string(), shape, dtype })
    }

    /// Load a tensor as F32 onto GPU, converting from file dtype if needed.
    pub fn load_f32(
        &self,
        name: &str,
        device: &WarpDevice,
    ) -> Result<GpuTensor<f32>, LoaderError> {
        let st = SafeTensors::deserialize(&self.mmap)
            .map_err(|e| LoaderError::Parse(e.to_string()))?;
        let view = st.tensor(name)
            .map_err(|e| LoaderError::TensorNotFound(name.to_string(), e.to_string()))?;

        let shape: Vec<usize> = view.shape().to_vec();
        let data = view.data();
        let file_dtype = view.dtype();

        // Convert to f32
        let f32_data = match file_dtype {
            safetensors::Dtype::F32 => {
                // Direct reinterpret
                let ptr = data.as_ptr() as *const f32;
                let len = data.len() / 4;
                unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
            }
            safetensors::Dtype::F16 => {
                // Convert f16 → f32
                let ptr = data.as_ptr() as *const half::f16;
                let len = data.len() / 2;
                let f16_slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                f16_slice.iter().map(|v| v.to_f32()).collect()
            }
            safetensors::Dtype::BF16 => {
                // Convert bf16 → f32
                let ptr = data.as_ptr() as *const half::bf16;
                let len = data.len() / 2;
                let bf16_slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                bf16_slice.iter().map(|v| v.to_f32()).collect()
            }
            other => {
                return Err(LoaderError::UnsupportedDtype(format!("{other:?}")));
            }
        };

        let warp_shape = Shape::from_static(&shape);
        let tensor = GpuTensor::from_host(device, &f32_data, warp_shape, DType::F32)
            .map_err(|e| LoaderError::Device(e.to_string()))?;

        Ok(tensor)
    }

    /// Load a 2D weight tensor as F32, transposed from [out, in] to [in, out].
    ///
    /// HuggingFace stores linear weights as [out_features, in_features] (row-major).
    /// PyTorch's F.linear(x, W) computes x @ W^T internally.
    /// Our GEMM computes x @ B directly, so we need B = W^T = [in, out].
    pub fn load_f32_transposed(
        &self,
        name: &str,
        device: &WarpDevice,
    ) -> Result<GpuTensor<f32>, LoaderError> {
        let st = SafeTensors::deserialize(&self.mmap)
            .map_err(|e| LoaderError::Parse(e.to_string()))?;
        let view = st.tensor(name)
            .map_err(|e| LoaderError::TensorNotFound(name.to_string(), e.to_string()))?;

        let shape: Vec<usize> = view.shape().to_vec();
        if shape.len() != 2 {
            // Not a 2D weight — load without transposition
            return self.load_f32(name, device);
        }

        let rows = shape[0]; // out_features
        let cols = shape[1]; // in_features
        let data = view.data();
        let file_dtype = view.dtype();

        // Convert to f32
        let f32_data: Vec<f32> = match file_dtype {
            safetensors::Dtype::F32 => {
                let ptr = data.as_ptr() as *const f32;
                let len = data.len() / 4;
                unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
            }
            safetensors::Dtype::F16 => {
                let ptr = data.as_ptr() as *const half::f16;
                let len = data.len() / 2;
                let f16_slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                f16_slice.iter().map(|v| v.to_f32()).collect()
            }
            safetensors::Dtype::BF16 => {
                let ptr = data.as_ptr() as *const half::bf16;
                let len = data.len() / 2;
                let bf16_slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                bf16_slice.iter().map(|v| v.to_f32()).collect()
            }
            other => {
                return Err(LoaderError::UnsupportedDtype(format!("{other:?}")));
            }
        };

        // Transpose [rows, cols] → [cols, rows]
        let mut transposed = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = f32_data[r * cols + c];
            }
        }

        // Transposed shape: [in_features, out_features]
        let warp_shape = Shape::from_static(&[cols, rows]);
        let tensor = GpuTensor::from_host(device, &transposed, warp_shape, DType::F32)
            .map_err(|e| LoaderError::Device(e.to_string()))?;

        Ok(tensor)
    }

    /// Load a tensor as F16 onto GPU.
    pub fn load_f16(
        &self,
        name: &str,
        device: &WarpDevice,
    ) -> Result<GpuTensor<half::f16>, LoaderError> {
        let st = SafeTensors::deserialize(&self.mmap)
            .map_err(|e| LoaderError::Parse(e.to_string()))?;
        let view = st.tensor(name)
            .map_err(|e| LoaderError::TensorNotFound(name.to_string(), e.to_string()))?;

        let shape: Vec<usize> = view.shape().to_vec();
        let data = view.data();
        let file_dtype = view.dtype();

        let f16_data = match file_dtype {
            safetensors::Dtype::F16 => {
                let ptr = data.as_ptr() as *const half::f16;
                let len = data.len() / 2;
                unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
            }
            safetensors::Dtype::F32 => {
                let ptr = data.as_ptr() as *const f32;
                let len = data.len() / 4;
                let f32_slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                f32_slice.iter().map(|v| half::f16::from_f32(*v)).collect()
            }
            safetensors::Dtype::BF16 => {
                let ptr = data.as_ptr() as *const half::bf16;
                let len = data.len() / 2;
                let bf16_slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                bf16_slice.iter().map(|v| half::f16::from_f32(v.to_f32())).collect()
            }
            other => {
                return Err(LoaderError::UnsupportedDtype(format!("{other:?}")));
            }
        };

        let warp_shape = Shape::from_static(&shape);
        let tensor = GpuTensor::from_host(device, &f16_data, warp_shape, DType::F16)
            .map_err(|e| LoaderError::Device(e.to_string()))?;

        Ok(tensor)
    }

    /// Load a 2D weight tensor as F16, transposed from [out, in] to [in, out].
    ///
    /// Same as `load_f32_transposed` but outputs F16 — halves memory bandwidth
    /// for weight GEMMs during decode. Converts from any file dtype (F32/F16/BF16).
    pub fn load_f16_transposed(
        &self,
        name: &str,
        device: &WarpDevice,
    ) -> Result<GpuTensor<half::f16>, LoaderError> {
        let st = SafeTensors::deserialize(&self.mmap)
            .map_err(|e| LoaderError::Parse(e.to_string()))?;
        let view = st.tensor(name)
            .map_err(|e| LoaderError::TensorNotFound(name.to_string(), e.to_string()))?;

        let shape: Vec<usize> = view.shape().to_vec();
        if shape.len() != 2 {
            // Not a 2D weight — load without transposition
            return self.load_f16(name, device);
        }

        let rows = shape[0]; // out_features
        let cols = shape[1]; // in_features
        let data = view.data();
        let file_dtype = view.dtype();

        // Convert to f16
        let f16_data: Vec<half::f16> = match file_dtype {
            safetensors::Dtype::F16 => {
                let ptr = data.as_ptr() as *const half::f16;
                let len = data.len() / 2;
                unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
            }
            safetensors::Dtype::F32 => {
                let ptr = data.as_ptr() as *const f32;
                let len = data.len() / 4;
                let f32_slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                f32_slice.iter().map(|v| half::f16::from_f32(*v)).collect()
            }
            safetensors::Dtype::BF16 => {
                let ptr = data.as_ptr() as *const half::bf16;
                let len = data.len() / 2;
                let bf16_slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                bf16_slice.iter().map(|v| half::f16::from_f32(v.to_f32())).collect()
            }
            other => {
                return Err(LoaderError::UnsupportedDtype(format!("{other:?}")));
            }
        };

        // Transpose [rows, cols] → [cols, rows]
        let mut transposed = vec![half::f16::from_f32(0.0); rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = f16_data[r * cols + c];
            }
        }

        // Transposed shape: [in_features, out_features]
        let warp_shape = Shape::from_static(&[cols, rows]);
        let tensor = GpuTensor::from_host(device, &transposed, warp_shape, DType::F16)
            .map_err(|e| LoaderError::Device(e.to_string()))?;

        Ok(tensor)
    }

    /// Print a summary of all tensors in the file.
    pub fn summary(&self) -> Result<String, LoaderError> {
        let st = SafeTensors::deserialize(&self.mmap)
            .map_err(|e| LoaderError::Parse(e.to_string()))?;

        let mut lines = vec![format!("SafeTensors: {}", self.path)];
        let mut total_params = 0usize;
        let mut total_bytes = 0usize;

        let mut names: Vec<_> = st.names().into_iter().collect();
        names.sort();

        for name in &names {
            let view = st.tensor(name).unwrap();
            let shape: Vec<usize> = view.shape().to_vec();
            let numel: usize = shape.iter().product();
            let dtype = view.dtype();
            let bytes = view.data().len();

            total_params += numel;
            total_bytes += bytes;

            let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
            lines.push(format!(
                "  {name:60} [{:20}] {:?} ({:.2} MB)",
                shape_str.join(", "),
                dtype,
                bytes as f64 / 1e6,
            ));
        }

        lines.push(format!(
            "\n  Total: {} tensors, {:.1}M parameters, {:.1} MB",
            names.len(),
            total_params as f64 / 1e6,
            total_bytes as f64 / 1e6,
        ));

        Ok(lines.join("\n"))
    }
}

/// Info about a tensor in the file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

fn convert_dtype(dt: safetensors::Dtype) -> DType {
    match dt {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::I32 => DType::I32,
        safetensors::Dtype::I64 => DType::I64,
        safetensors::Dtype::I8 => DType::I8,
        safetensors::Dtype::U8 => DType::U8,
        _ => DType::F32, // fallback
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("IO error: {0}")]
    Io(String),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Tensor not found: {0} ({1})")]
    TensorNotFound(String, String),
    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),
    #[error("Device error: {0}")]
    Device(String),
    #[error("Config error: {0}")]
    Config(String),
}

/// Loader that searches across multiple sharded SafeTensors files.
/// Used for large models (7B+) that split weights across multiple files.
pub struct ShardedSafeTensorsLoader {
    shards: Vec<SafeTensorsLoader>,
}

impl ShardedSafeTensorsLoader {
    /// Open multiple SafeTensors shard files.
    pub fn open(paths: &[impl AsRef<Path>]) -> Result<Self, LoaderError> {
        let mut shards = Vec::new();
        for path in paths {
            shards.push(SafeTensorsLoader::open(path)?);
        }
        Ok(Self { shards })
    }

    /// Open all shard files in a directory (matching model-*.safetensors pattern).
    pub fn open_dir(dir: impl AsRef<Path>) -> Result<Self, LoaderError> {
        let dir = dir.as_ref();
        let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| LoaderError::Io(format!("{}: {e}", dir.display())))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().map_or(false, |ext| ext == "safetensors")
            })
            .collect();
        paths.sort();

        if paths.is_empty() {
            return Err(LoaderError::Io(format!("no .safetensors files in {}", dir.display())));
        }

        Self::open(&paths)
    }

    /// Load a tensor as F32, searching across all shards.
    pub fn load_f32(&self, name: &str, device: &WarpDevice) -> Result<GpuTensor<f32>, LoaderError> {
        for shard in &self.shards {
            match shard.load_f32(name, device) {
                Ok(t) => return Ok(t),
                Err(LoaderError::TensorNotFound(_, _)) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(LoaderError::TensorNotFound(name.to_string(), "not found in any shard".to_string()))
    }

    /// Load a tensor as F32, transposed from [out, in] to [in, out].
    pub fn load_f32_transposed(&self, name: &str, device: &WarpDevice) -> Result<GpuTensor<f32>, LoaderError> {
        for shard in &self.shards {
            match shard.load_f32_transposed(name, device) {
                Ok(t) => return Ok(t),
                Err(LoaderError::TensorNotFound(_, _)) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(LoaderError::TensorNotFound(name.to_string(), "not found in any shard".to_string()))
    }

    /// Load a tensor as F16, transposed.
    pub fn load_f16_transposed(&self, name: &str, device: &WarpDevice) -> Result<GpuTensor<half::f16>, LoaderError> {
        for shard in &self.shards {
            match shard.load_f16_transposed(name, device) {
                Ok(t) => return Ok(t),
                Err(LoaderError::TensorNotFound(_, _)) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(LoaderError::TensorNotFound(name.to_string(), "not found in any shard".to_string()))
    }

    /// Number of shards.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_load_safetensors() {
        // Create a test safetensors file
        use safetensors::tensor::{serialize, TensorView};
        use std::collections::HashMap;

        let data_a: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let data_b: Vec<f32> = (0..8).map(|i| i as f32 * -0.5).collect();

        let bytes_a: &[u8] = unsafe {
            std::slice::from_raw_parts(data_a.as_ptr() as *const u8, data_a.len() * 4)
        };
        let bytes_b: &[u8] = unsafe {
            std::slice::from_raw_parts(data_b.as_ptr() as *const u8, data_b.len() * 4)
        };

        let mut tensors = HashMap::new();
        tensors.insert(
            "weight_a".to_string(),
            TensorView::new(safetensors::Dtype::F32, vec![4, 4], bytes_a).unwrap(),
        );
        tensors.insert(
            "weight_b".to_string(),
            TensorView::new(safetensors::Dtype::F32, vec![2, 4], bytes_b).unwrap(),
        );

        let serialized = serialize(&tensors, None).unwrap();

        // Write to temp file
        let path = std::env::temp_dir().join("warp_test.safetensors");
        std::fs::write(&path, &serialized).unwrap();

        // Load it back
        let loader = SafeTensorsLoader::open(&path).unwrap();

        let names = loader.tensor_names().unwrap();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"weight_a".to_string()));

        let info = loader.tensor_info("weight_a").unwrap();
        assert_eq!(info.shape, vec![4, 4]);
        assert_eq!(info.dtype, DType::F32);

        // Load onto GPU
        let dev = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping GPU load"); return; }
        };

        let tensor = loader.load_f32("weight_a", &dev).unwrap();
        let result = tensor.to_host(&dev).unwrap();
        assert_eq!(result, data_a);

        println!("SafeTensors: loaded 2 tensors from file → GPU successfully!");
        println!("{}", loader.summary().unwrap());

        // Cleanup
        std::fs::remove_file(&path).ok();
    }
}

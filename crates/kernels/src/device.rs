//! CUDA device management via cudarc.
//!
//! WarpDevice wraps cudarc's CudaContext with higher-level operations:
//! - PTX compilation and module loading
//! - Kernel launch with grid/block configuration
//! - Device memory allocation
//! - Device info queries

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, sys};
use cudarc::driver::result::DriverError;
use cudarc::nvrtc;
use std::sync::Arc;

/// Warp's GPU device handle.
pub struct WarpDevice {
    /// The underlying cudarc CUDA context.
    pub ctx: Arc<CudaContext>,
    /// Default stream for kernel launches.
    pub stream: Arc<CudaStream>,
    /// Device ordinal (0, 1, ...).
    pub ordinal: usize,
    /// Compute capability (major, minor) e.g., (8, 9) for Ada.
    pub compute_capability: (u32, u32),
    /// SM version as single number (e.g., 89).
    pub sm_version: u32,
}

impl WarpDevice {
    /// Initialize a CUDA device by ordinal.
    pub fn new(ordinal: usize) -> Result<Self, DeviceError> {
        let ctx = CudaContext::new(ordinal).map_err(|e| DeviceError::Init(e.to_string()))?;
        let stream = ctx.default_stream();

        let major = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| DeviceError::Init(format!("failed to query compute capability major: {e}")))?
            as u32;
        let minor = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| DeviceError::Init(format!("failed to query compute capability minor: {e}")))?
            as u32;
        let sm_version = major * 10 + minor;

        Ok(Self {
            ctx,
            stream,
            ordinal,
            compute_capability: (major, minor),
            sm_version,
        })
    }

    /// Compile CUDA C source to PTX and load it as a module.
    pub fn load_cuda_source(
        &self,
        cuda_src: &str,
        func_name: &str,
    ) -> Result<(Arc<CudaModule>, CudaFunction), DeviceError> {
        let ptx = nvrtc::compile_ptx(cuda_src)
            .map_err(|e| DeviceError::PtxLoad(e.to_string()))?;
        let module = self.ctx.load_module(ptx)
            .map_err(|e| DeviceError::PtxLoad(e.to_string()))?;
        let func = module.load_function(func_name)
            .map_err(|e| DeviceError::FuncNotFound(e.to_string()))?;
        Ok((module, func))
    }

    /// Compile CUDA C source with include paths and arch flags.
    /// Needed for Tensor Core wmma kernels that require mma.h.
    pub fn load_cuda_source_with_opts(
        &self,
        cuda_src: &str,
        func_name: &str,
        include_paths: &[String],
        arch: Option<&'static str>,
    ) -> Result<(Arc<CudaModule>, CudaFunction), DeviceError> {
        let opts = nvrtc::CompileOptions {
            include_paths: include_paths.to_vec(),
            arch,
            ..Default::default()
        };
        let ptx = nvrtc::compile_ptx_with_opts(cuda_src, opts)
            .map_err(|e| DeviceError::PtxLoad(e.to_string()))?;
        let module = self.ctx.load_module(ptx)
            .map_err(|e| DeviceError::PtxLoad(e.to_string()))?;
        let func = module.load_function(func_name)
            .map_err(|e| DeviceError::FuncNotFound(e.to_string()))?;
        Ok((module, func))
    }

    /// CUDA toolkit include path (for mma.h, cuda_fp16.h, etc.)
    pub fn cuda_include_path() -> String {
        // Check common locations
        for version in &["v12.8", "v12.6", "v12.4", "v12.2", "v12.0", "v11.8"] {
            let path = format!("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/{version}/include");
            if std::path::Path::new(&path).exists() {
                return path;
            }
        }
        // Fallback: try environment variable
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            return format!("{cuda_path}/include");
        }
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include".to_string()
    }

    /// Allocate device memory and copy host data to it.
    pub fn htod<T: cudarc::driver::DeviceRepr + Clone>(
        &self,
        data: &[T],
    ) -> Result<cudarc::driver::CudaSlice<T>, DeviceError> {
        self.stream
            .memcpy_stod(data)
            .map_err(|e| DeviceError::Memory(e.to_string()))
    }

    /// Copy device data back to host.
    pub fn dtoh<T: cudarc::driver::DeviceRepr + Clone + Default>(
        &self,
        src: &cudarc::driver::CudaSlice<T>,
    ) -> Result<Vec<T>, DeviceError> {
        self.stream
            .memcpy_dtov(src)
            .map_err(|e| DeviceError::Memory(e.to_string()))
    }

    /// Allocate zeroed device memory.
    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>, DeviceError> {
        self.stream
            .alloc_zeros(len)
            .map_err(|e| DeviceError::Memory(e.to_string()))
    }

    /// Synchronize the default stream.
    pub fn synchronize(&self) -> Result<(), DeviceError> {
        self.stream
            .synchronize()
            .map_err(|e| DeviceError::Sync(e.to_string()))
    }

    /// NVRTC arch string for this device (e.g., "compute_89").
    pub fn arch_flag(&self) -> String {
        format!("compute_{}", self.sm_version)
    }

    /// Create a clone of this device pointing to a different stream.
    /// Used for CUDA graph capture — the capture stream records kernel launches.
    pub fn with_stream(&self, stream: std::sync::Arc<CudaStream>) -> Self {
        Self {
            ctx: self.ctx.clone(),
            stream,
            ordinal: self.ordinal,
            compute_capability: self.compute_capability,
            sm_version: self.sm_version,
        }
    }

    /// Summary string for this device.
    pub fn summary(&self) -> String {
        format!(
            "CUDA Device {} (SM {}.{})",
            self.ordinal,
            self.compute_capability.0,
            self.compute_capability.1,
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error("Device initialization failed: {0}")]
    Init(String),
    #[error("PTX/CUDA compilation failed: {0}")]
    PtxLoad(String),
    #[error("Function not found: {0}")]
    FuncNotFound(String),
    #[error("Device synchronization failed: {0}")]
    Sync(String),
    #[error("Memory operation failed: {0}")]
    Memory(String),
    #[error("Kernel launch failed: {0}")]
    Launch(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_init() {
        match WarpDevice::new(0) {
            Ok(dev) => {
                println!("Device: {}", dev.summary());
            }
            Err(e) => {
                println!("No CUDA device available: {e}");
            }
        }
    }
}

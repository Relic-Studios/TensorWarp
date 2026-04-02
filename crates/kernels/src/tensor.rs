//! GPU tensor — typed device memory with shape metadata.

use cudarc::driver::CudaSlice;
use warp_ir::{DType, Shape};

use crate::device::{DeviceError, WarpDevice};

/// A tensor living on the GPU with type and shape metadata.
pub struct GpuTensor<T> {
    /// Device memory buffer.
    pub data: CudaSlice<T>,
    /// Tensor shape.
    pub shape: Shape,
    /// Data type.
    pub dtype: DType,
    /// Number of elements.
    pub numel: usize,
}

impl<T: cudarc::driver::DeviceRepr + Clone> GpuTensor<T> {
    /// Create a GPU tensor by copying host data to device.
    pub fn from_host(device: &WarpDevice, data: &[T], shape: Shape, dtype: DType) -> Result<Self, DeviceError> {
        let numel = shape.numel_static();
        assert_eq!(data.len(), numel, "data length {} != shape numel {}", data.len(), numel);
        let gpu_data = device.htod(data)?;
        Ok(Self { data: gpu_data, shape, dtype, numel })
    }
}

impl<T: cudarc::driver::DeviceRepr + Clone + Default> GpuTensor<T> {
    /// Copy tensor data back to host.
    pub fn to_host(&self, device: &WarpDevice) -> Result<Vec<T>, DeviceError> {
        device.dtoh(&self.data)
    }
}

impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits> GpuTensor<T> {
    /// Create a zero-initialized GPU tensor.
    pub fn zeros(device: &WarpDevice, shape: Shape, dtype: DType) -> Result<Self, DeviceError> {
        let numel = shape.numel_static();
        let data = device.alloc_zeros(numel)?;
        Ok(Self { data, shape, dtype, numel })
    }
}

impl<T> GpuTensor<T> {
    pub fn size_bytes(&self) -> usize {
        self.numel * std::mem::size_of::<T>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_f32() {
        let dev = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        let host_data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let shape = Shape::from_static(&[16, 16]);
        let tensor = GpuTensor::from_host(&dev, &host_data, shape, DType::F32).unwrap();

        assert_eq!(tensor.numel, 256);
        let result = tensor.to_host(&dev).unwrap();
        assert_eq!(result, host_data);
    }

    #[test]
    fn zeros_tensor() {
        let dev = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        let shape = Shape::from_static(&[1024]);
        let tensor: GpuTensor<f32> = GpuTensor::zeros(&dev, shape, DType::F32).unwrap();
        let result = tensor.to_host(&dev).unwrap();
        assert!(result.iter().all(|&x| x == 0.0));
    }
}

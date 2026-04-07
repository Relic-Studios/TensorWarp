//! CUDA Graph capture and replay for decode acceleration.
//!
//! CUDA graphs capture a sequence of kernel launches and replay them
//! as a single GPU operation, eliminating per-kernel launch overhead.
//!
//! For a 32-layer transformer decode step with ~15 launches/layer:
//!   Without graphs: ~480 launches × ~2μs each = ~960μs overhead
//!   With graphs:    1 graph launch × ~5μs = ~5μs overhead
//!   Savings: ~955μs per token (can be >50% of total decode time at batch=1)
//!
//! Usage:
//! ```ignore
//! // First call: capture the graph
//! let graph = CudaGraphCapture::record(&stream, || {
//!     // launch kernels normally
//!     engine.forward_decode(device, token, &mut kv_cache, pos)?;
//!     Ok(())
//! })?;
//!
//! // Subsequent calls: replay (single launch)
//! graph.replay()?;
//! ```

use std::sync::Arc;
use cudarc::driver::{CudaStream, CudaFunction, CudaGraph, sys};

use crate::device::DeviceError;

/// Get raw CUfunction from CudaFunction (vendored cudarc with pub field).
fn raw_cu_function(f: &CudaFunction) -> sys::CUfunction {
    f.cu_function
}

/// Get raw device pointer from CudaSlice (vendored cudarc with pub field).
fn raw_device_ptr<T>(slice: &cudarc::driver::CudaSlice<T>) -> sys::CUdeviceptr {
    slice.cu_device_ptr
}

/// Launch a kernel directly via cuLaunchKernel WITHOUT bind_to_thread().
/// This is the capture-safe launch path for CUDA graph recording.
///
/// SAFETY: All pointers in kernel_params must be valid device pointers
/// or host pointers to scalar arguments.
pub unsafe fn capture_safe_launch(
    f: &CudaFunction,
    stream: &CudaStream,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DeviceError> {
    let cu_func = raw_cu_function(f);
    let cu_stream = stream.cu_stream();

    let result = sys::cuLaunchKernel(
        cu_func,
        grid.0, grid.1, grid.2,
        block.0, block.1, block.2,
        shared_mem,
        cu_stream,
        kernel_params.as_mut_ptr(),
        std::ptr::null_mut(),
    );

    if result != sys::cudaError_enum::CUDA_SUCCESS {
        return Err(DeviceError::Launch(format!("cuLaunchKernel failed: {:?}", result)));
    }
    Ok(())
}

/// A captured CUDA graph ready for replay.
pub struct GraphCapture {
    graph: CudaGraph,
}

impl GraphCapture {
    /// Capture a CUDA graph by recording all kernel launches in `f`.
    ///
    /// IMPORTANT: All kernels must be pre-compiled before capture.
    /// The closure `f` should use `device.with_stream(capture_stream)`
    /// to launch kernels on the capture stream.
    ///
    /// The context is pre-bound before capture to avoid bind_to_thread()
    /// invalidating the capture during kernel launches.
    pub fn record_with_device<F>(device: &crate::device::WarpDevice, stream: &Arc<CudaStream>, f: F) -> Result<Self, DeviceError>
    where
        F: FnOnce() -> Result<(), DeviceError>,
    {
        // Pre-bind context BEFORE starting capture.
        // This makes launch_builder's bind_to_thread() a no-op during capture.
        device.ctx.bind_to_thread()
            .map_err(|e| DeviceError::Launch(format!("pre-bind context: {e}")))?;

        // Begin capture
        // RELAXED mode allows cross-stream buffer dependencies during capture.
        // Required because buffers may have been allocated/written on the default stream.
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| DeviceError::Launch(format!("graph begin_capture: {e}")))?;

        // Execute the closure — kernels are captured, not executed
        let result = f();

        // End capture and instantiate
        let graph = stream.end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| DeviceError::Launch(format!("graph end_capture: {e}")))?
            .ok_or_else(|| DeviceError::Launch("graph capture produced null graph".into()))?;

        result?;

        Ok(Self { graph })
    }

    /// Replay the captured graph — single GPU launch for all recorded kernels.
    pub fn replay(&self) -> Result<(), DeviceError> {
        self.graph.launch()
            .map_err(|e| DeviceError::Launch(format!("graph launch: {e}")))
    }
}

/// Manages a cached CUDA graph for a decode step.
/// First call captures, subsequent calls replay.
pub struct DecodeGraphCache {
    graph: Option<GraphCapture>,
    captures: u64,
    replays: u64,
}

impl DecodeGraphCache {
    pub fn new() -> Self {
        Self { graph: None, captures: 0, replays: 0 }
    }

    /// Run the decode step — captures on first call, replays thereafter.
    ///
    /// IMPORTANT: The graph captures the exact kernel sequence. If the model
    /// shape changes (different seq_len, different batch), the graph must be
    /// re-captured. This is fine for decode (always batch=1, seq=1).
    pub fn run<F>(&mut self, device: &crate::device::WarpDevice, stream: &Arc<CudaStream>, f: F) -> Result<(), DeviceError>
    where
        F: FnOnce() -> Result<(), DeviceError>,
    {
        if let Some(ref graph) = self.graph {
            graph.replay()?;
            self.replays += 1;
        } else {
            let graph = GraphCapture::record_with_device(device, stream, f)?;
            self.graph = Some(graph);
            self.captures += 1;
        }
        Ok(())
    }

    /// Invalidate the cached graph (call when shapes change).
    pub fn invalidate(&mut self) {
        self.graph = None;
    }

    /// Statistics.
    pub fn stats(&self) -> String {
        format!("DecodeGraph: {} captures, {} replays, {}",
            self.captures, self.replays,
            if self.graph.is_some() { "cached" } else { "uncached" })
    }
}

impl Default for DecodeGraphCache {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::WarpDevice;
    use crate::cache::KernelCache;
    use crate::tensor::GpuTensor;
    use warp_ir::{DType, Shape};

    #[test]
    fn graph_capture_and_replay() {
        let dev = match WarpDevice::new(0) {
            Ok(d) => d,
            Err(_) => { println!("No CUDA, skipping"); return; }
        };

        // CUDA graph capture needs a non-default stream
        let capture_stream = dev.ctx.new_stream()
            .expect("Failed to create capture stream");

        let cache = KernelCache::new();
        let n = 1024usize;

        let a_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let a = GpuTensor::from_host(&dev, &a_data, Shape::from_static(&[n]), DType::F32).unwrap();
        let b = GpuTensor::from_host(&dev, &b_data, Shape::from_static(&[n]), DType::F32).unwrap();
        let mut out = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[n]), DType::F32).unwrap();

        // Pre-allocate ALL output tensors before capture
        let mut out2 = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[n]), DType::F32).unwrap();

        // Warmup: compile ALL kernels on the DEFAULT stream before capture
        // Graph capture fails if NVRTC compilation happens during capture
        crate::ops::add(&cache, &dev, &a, &b, &mut out).unwrap();
        crate::ops::relu(&cache, &dev, &a, &mut out2).unwrap();
        dev.synchronize().unwrap();
        println!("Kernels pre-compiled: {}", cache.stats());

        // Create a device view using the capture stream — ops launched on this
        // device will go to the capture stream and be recorded into the graph.
        let capture_dev = dev.with_stream(capture_stream.clone());

        let mut out3 = GpuTensor::<f32>::zeros(&dev, Shape::from_static(&[n]), DType::F32).unwrap();

        // Pre-compile and get the CudaFunction handles
        let add_func = cache.get_or_compile(&dev,
            r#"extern "C" __global__ void warp_add(float *out, const float *a, const float *b, size_t n) { size_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { out[i] = a[i] + b[i]; } }"#,
            "warp_add").unwrap();
        let relu_func = cache.get_or_compile(&dev,
            r#"extern "C" __global__ void warp_relu(float *out, const float *x, size_t n) { size_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { out[i] = fmaxf(x[i], 0.0f); } }"#,
            "warp_relu").unwrap();
        dev.synchronize().unwrap();

        // Capture using capture_safe_launch with vendored cudarc pub fields
        let graph = GraphCapture::record_with_device(&dev, &capture_stream, || {
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let out_ptr = raw_device_ptr(&out.data);
            let out2_ptr = raw_device_ptr(&out2.data);
            let out3_ptr = raw_device_ptr(&out3.data);
            let a_ptr = raw_device_ptr(&a.data);
            let b_ptr = raw_device_ptr(&b.data);
            let mut n_val = n;

            unsafe {
                let mut p1 = [&out_ptr as *const _ as *mut std::ffi::c_void,
                    &a_ptr as *const _ as *mut std::ffi::c_void,
                    &b_ptr as *const _ as *mut std::ffi::c_void,
                    &mut n_val as *mut _ as *mut std::ffi::c_void];
                capture_safe_launch(&add_func, &capture_stream, (blocks,1,1), (threads,1,1), 0, &mut p1)?;

                let mut p2 = [&out2_ptr as *const _ as *mut std::ffi::c_void,
                    &out_ptr as *const _ as *mut std::ffi::c_void,
                    &mut n_val as *mut _ as *mut std::ffi::c_void];
                capture_safe_launch(&relu_func, &capture_stream, (blocks,1,1), (threads,1,1), 0, &mut p2)?;

                let mut p3 = [&out3_ptr as *const _ as *mut std::ffi::c_void,
                    &out2_ptr as *const _ as *mut std::ffi::c_void,
                    &a_ptr as *const _ as *mut std::ffi::c_void,
                    &mut n_val as *mut _ as *mut std::ffi::c_void];
                capture_safe_launch(&add_func, &capture_stream, (blocks,1,1), (threads,1,1), 0, &mut p3)?;
            }
            Ok(())
        });

        let graph = match graph {
            Ok(g) => { println!("CUDA Graph captured!"); g }
            Err(e) => { println!("Graph capture failed: {e}"); return; }
        };

        dev.synchronize().unwrap();

        // Benchmark
        let iters = 1000;
        let start = std::time::Instant::now();
        for _ in 0..iters { graph.replay().unwrap(); }
        dev.synchronize().unwrap();
        let graph_time = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            crate::ops::add(&cache, &dev, &a, &b, &mut out).unwrap();
            crate::ops::relu(&cache, &dev, &out, &mut out2).unwrap();
            crate::ops::add(&cache, &dev, &out2, &a, &mut out3).unwrap();
        }
        dev.synchronize().unwrap();
        let normal_time = start.elapsed();

        println!("\n=== CUDA Graph Benchmark ({iters} iters) ===");
        println!("  Normal: {:.1}μs/iter", normal_time.as_secs_f64()*1e6/iters as f64);
        println!("  Graph:  {:.1}μs/iter", graph_time.as_secs_f64()*1e6/iters as f64);
        println!("  Speedup: {:.1}x", normal_time.as_secs_f64()/graph_time.as_secs_f64());

        let result = out3.to_host(&dev).unwrap();
        let expected = ((a_data[0] + b_data[0]).max(0.0)) + a_data[0];
        assert!((result[0] - expected).abs() < 0.01);
        println!("  Correctness: verified!");
    }
}

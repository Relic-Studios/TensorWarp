//! C API / FFI — allows C, C++, and other languages to use TensorWarp.
//!
//! Provides extern "C" functions for:
//! - Device initialization
//! - Tensor allocation and data transfer
//! - GEMM, Conv2D, and other core operations
//! - ONNX model loading and execution
//! - Engine lifecycle management
//!
//! Usage from C:
//!   WarpDevice* dev = warp_device_create(0);
//!   WarpTensor* a = warp_tensor_create(dev, dims, 2, WARP_F32);
//!   warp_gemm(dev, a, b, c, m, n, k);
//!   warp_tensor_destroy(a);
//!   warp_device_destroy(dev);

use std::ffi::c_void;

/// Opaque handle types for C API.
pub type WarpDeviceHandle = *mut c_void;
pub type WarpTensorHandle = *mut c_void;
pub type WarpEngineHandle = *mut c_void;

/// Data type enum for C API.
#[repr(C)]
pub enum WarpDType {
    F32 = 0,
    F16 = 1,
    I32 = 2,
    I8 = 3,
    U8 = 4,
}

/// Error codes for C API.
#[repr(C)]
pub enum WarpError {
    Ok = 0,
    DeviceError = 1,
    MemoryError = 2,
    ShapeError = 3,
    UnsupportedOp = 4,
    InvalidHandle = 5,
}

/// Create a CUDA device. Returns null on failure.
#[no_mangle]
pub extern "C" fn warp_device_create(ordinal: i32) -> WarpDeviceHandle {
    match crate::device::WarpDevice::new(ordinal as usize) {
        Ok(dev) => Box::into_raw(Box::new(dev)) as WarpDeviceHandle,
        Err(_) => std::ptr::null_mut(),
    }
}

/// Destroy a device handle.
#[no_mangle]
pub extern "C" fn warp_device_destroy(handle: WarpDeviceHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle as *mut crate::device::WarpDevice)); }
    }
}

/// Get device info as a null-terminated string.
/// Caller must free the returned string with warp_string_free.
#[no_mangle]
pub extern "C" fn warp_device_info(handle: WarpDeviceHandle) -> *mut std::ffi::c_char {
    if handle.is_null() { return std::ptr::null_mut(); }
    let dev = unsafe { &*(handle as *const crate::device::WarpDevice) };
    let info = dev.summary();
    match std::ffi::CString::new(info) {
        Ok(cs) => cs.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a string returned by warp_device_info.
#[no_mangle]
pub extern "C" fn warp_string_free(s: *mut std::ffi::c_char) {
    if !s.is_null() {
        unsafe { drop(std::ffi::CString::from_raw(s)); }
    }
}

/// Synchronize the device (wait for all GPU work to complete).
#[no_mangle]
pub extern "C" fn warp_device_sync(handle: WarpDeviceHandle) -> WarpError {
    if handle.is_null() { return WarpError::InvalidHandle; }
    let dev = unsafe { &*(handle as *const crate::device::WarpDevice) };
    match dev.synchronize() {
        Ok(()) => WarpError::Ok,
        Err(_) => WarpError::DeviceError,
    }
}

/// Get number of available CUDA devices.
#[no_mangle]
pub extern "C" fn warp_device_count() -> i32 {
    crate::device::WarpDevice::device_count().unwrap_or(0) as i32
}

/// Get TensorWarp version string.
#[no_mangle]
pub extern "C" fn warp_version() -> *const std::ffi::c_char {
    // Static string — doesn't need freeing
    b"0.1.0\0".as_ptr() as *const std::ffi::c_char
}

// ── C Header Generation ─────────────────────────────────────────

/// Generate the C header for the TensorWarp API.
pub fn generate_c_header() -> String {
    r#"/* TensorWarp C API — auto-generated header */
#ifndef TENSORWARP_H
#define TENSORWARP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles */
typedef void* WarpDevice;
typedef void* WarpTensor;
typedef void* WarpEngine;

/* Data types */
typedef enum {
    WARP_F32 = 0,
    WARP_F16 = 1,
    WARP_I32 = 2,
    WARP_I8  = 3,
    WARP_U8  = 4,
} WarpDType;

/* Error codes */
typedef enum {
    WARP_OK = 0,
    WARP_DEVICE_ERROR = 1,
    WARP_MEMORY_ERROR = 2,
    WARP_SHAPE_ERROR = 3,
    WARP_UNSUPPORTED_OP = 4,
    WARP_INVALID_HANDLE = 5,
} WarpError;

/* Device management */
WarpDevice warp_device_create(int ordinal);
void warp_device_destroy(WarpDevice dev);
char* warp_device_info(WarpDevice dev);
void warp_string_free(char* s);
WarpError warp_device_sync(WarpDevice dev);
int warp_device_count(void);
const char* warp_version(void);

#ifdef __cplusplus
}
#endif

#endif /* TENSORWARP_H */
"#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_api_device_lifecycle() {
        let dev = warp_device_create(0);
        if dev.is_null() {
            println!("No CUDA device, skipping C API test");
            return;
        }

        let info = warp_device_info(dev);
        assert!(!info.is_null());
        let info_str = unsafe { std::ffi::CStr::from_ptr(info).to_string_lossy().to_string() };
        println!("C API device info: {}", info_str);
        warp_string_free(info);

        assert_eq!(warp_device_sync(dev) as i32, WarpError::Ok as i32);

        let count = warp_device_count();
        assert!(count >= 1);
        println!("C API device count: {}", count);

        warp_device_destroy(dev);
        println!("C API lifecycle: correct!");
    }

    #[test]
    fn c_header_generation() {
        let header = generate_c_header();
        assert!(header.contains("TENSORWARP_H"));
        assert!(header.contains("warp_device_create"));
        println!("Generated C header ({} bytes)", header.len());
    }
}

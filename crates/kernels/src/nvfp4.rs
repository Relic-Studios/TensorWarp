//! NVIDIA FP4 (E2M1) dequantization kernel.
//!
//! Dequantizes NVFP4 packed weights to F16 for cuBLAS HGEMM.
//! Format: 2 FP4 values per uint8 byte, with FP8 per-group scales
//! and a global F32 second-level scale.
//!
//! dequant(byte) = fp4_lut[nibble] * fp8_scale[group] * global_scale

use warp_ir::{DType, Shape};
use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;
use cudarc::driver::{LaunchConfig, PushKernelArg};

const DEQUANT_FP4_SRC: &str = r#"
// FP4 E2M1 lookup table: maps 4-bit value to float
__device__ __constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// FP8 E4M3 to float conversion via PTX
__device__ float fp8_to_float(unsigned char bits) {
    // FP8 E4M3: 1 sign + 4 exp + 3 mantissa
    unsigned int sign = (bits >> 7) & 1;
    unsigned int exp = (bits >> 3) & 0xF;
    unsigned int mant = bits & 0x7;

    if (exp == 0) {
        // Subnormal
        float val = mant / 8.0f * (1.0f / 64.0f);  // 2^(1-bias) * mant/8
        return sign ? -val : val;
    } else if (exp == 15) {
        // NaN (no inf in E4M3)
        return 0.0f;
    } else {
        // Normal: (-1)^sign * 2^(exp-7) * (1 + mant/8)
        float val = (1.0f + mant / 8.0f);
        // 2^(exp-7): exp ranges 1-14, so power ranges -6 to 7
        int power = (int)exp - 7;
        if (power >= 0) {
            for (int i = 0; i < power; i++) val *= 2.0f;
        } else {
            for (int i = 0; i < -power; i++) val *= 0.5f;
        }
        return sign ? -val : val;
    }
}

extern "C" __global__ void warp_dequant_fp4_to_f16(
    unsigned short *out,           // [rows, cols] as F16 bits
    const unsigned char *packed,   // [rows, cols/2] packed FP4
    const unsigned char *scales,   // [rows, cols/16] FP8 E4M3 scales
    float global_scale,            // weight_scale_2
    unsigned int rows,
    unsigned int cols,             // original cols (2× packed cols)
    unsigned int group_size        // typically 16
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = rows * (cols / 2);
    if (idx >= total) return;

    unsigned int row = idx / (cols / 2);
    unsigned int byte_col = idx % (cols / 2);
    unsigned int col_lo = byte_col * 2;
    unsigned int col_hi = byte_col * 2 + 1;

    unsigned char byte_val = packed[idx];
    unsigned char lo_nib = byte_val & 0xF;
    unsigned char hi_nib = byte_val >> 4;

    // Scale group: each group covers `group_size` elements
    unsigned int group_lo = col_lo / group_size;
    unsigned int group_hi = col_hi / group_size;
    unsigned int scale_cols = cols / group_size;

    float scale_lo = fp8_to_float(scales[row * scale_cols + group_lo]) * global_scale;
    float scale_hi = fp8_to_float(scales[row * scale_cols + group_hi]) * global_scale;

    float val_lo = FP4_LUT[lo_nib] * scale_lo;
    float val_hi = FP4_LUT[hi_nib] * scale_hi;

    // Convert to F16 via PTX
    unsigned short f16_lo, f16_hi;
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(f16_lo) : "f"(val_lo));
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(f16_hi) : "f"(val_hi));

    out[row * cols + col_lo] = f16_lo;
    out[row * cols + col_hi] = f16_hi;
}
"#;

/// Dequantize NVFP4 packed weights to F16.
///
/// packed: [rows, cols/2] uint8 (2 FP4 values per byte)
/// scales: [rows, cols/16] float8_e4m3
/// global_scale: scalar f32
/// out: [rows, cols] f16
pub fn dequant_fp4_to_f16(
    cache: &KernelCache,
    device: &WarpDevice,
    packed: &GpuTensor<u8>,
    scales: &GpuTensor<u8>,     // FP8 stored as raw bytes
    global_scale: f32,
    out: &mut GpuTensor<half::f16>,
    rows: u32,
    cols: u32,                   // original columns (2× packed cols)
) -> Result<(), DeviceError> {
    let f = cache.get_or_compile(device, DEQUANT_FP4_SRC, "warp_dequant_fp4_to_f16")?;
    let total = rows * (cols / 2);
    let cfg = LaunchConfig::for_num_elems(total);
    let group_size = 16u32;

    unsafe {
        device.stream.launch_builder(&f)
            .arg(&mut out.data)
            .arg(&packed.data)
            .arg(&scales.data)
            .arg(&global_scale)
            .arg(&rows)
            .arg(&cols)
            .arg(&group_size)
            .launch(cfg)
            .map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))?;
    }
    Ok(())
}

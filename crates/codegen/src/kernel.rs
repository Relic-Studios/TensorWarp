//! Kernel templates and generation utilities.
//!
//! Rather than hand-writing every kernel, Warp uses parameterized templates.
//! A template is a kernel with "holes" for tile sizes, unroll factors,
//! data types, and fusion boundaries. The codegen fills these holes based
//! on the specific shapes and dtypes from the IR.

use warp_ir::DType;

/// PTX type name for a given dtype.
pub fn ptx_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => ".f32",
        DType::F16 => ".f16",
        DType::BF16 => ".bf16",
        DType::F8E4M3 => ".b8", // no native PTX type, treated as bytes
        DType::F8E5M2 => ".b8",
        DType::I64 => ".s64",
        DType::I32 => ".s32",
        DType::I16 => ".s16",
        DType::I8 => ".s8",
        DType::I4 => ".b8",  // packed
        DType::U8 => ".u8",
        DType::U32 => ".u32",
        DType::Bool => ".pred",
        DType::Q8_0 | DType::Q4_0 | DType::Q4_1 => ".b8", // block-quantized, custom handling
    }
}

/// Metal shader type name for a given dtype.
pub fn metal_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        DType::I64 => "long",
        DType::I32 => "int",
        DType::I16 => "short",
        DType::I8 => "char",
        DType::U8 => "uchar",
        DType::U32 => "uint",
        DType::Bool => "bool",
        _ => "uchar", // quantized/fp8 use raw bytes
    }
}

/// Compute optimal tile size for a matrix multiply.
/// Balances shared memory usage, register pressure, and occupancy.
pub fn optimal_matmul_tiles(m: usize, n: usize, k: usize, dtype: DType) -> MatMulTiling {
    let elem_bytes = dtype.byte_size();

    // Heuristic: pick tiles that fill shared memory ~50% to leave room
    // for double-buffering.
    let (tm, tn, tk) = if m * n * k < 1024 * 1024 {
        // Small matmul: 64x64 tiles
        (64, 64, 32)
    } else if elem_bytes <= 2 {
        // FP16/BF16: 128x128 tiles with k=32 (fits in 128KB smem with double-buffer)
        (128, 128, 32)
    } else {
        // FP32: 64x64 tiles
        (64, 64, 32)
    };

    MatMulTiling {
        tile_m: tm.min(m),
        tile_n: tn.min(n),
        tile_k: tk.min(k),
        warp_tile_m: 32,
        warp_tile_n: 32,
        stages: 2, // double-buffering
    }
}

/// Tiling configuration for matrix multiply kernels.
#[derive(Debug, Clone)]
pub struct MatMulTiling {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub warp_tile_m: usize,
    pub warp_tile_n: usize,
    pub stages: usize, // pipeline stages for async copy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_sizes_scale_with_problem() {
        let small = optimal_matmul_tiles(32, 32, 32, DType::F16);
        assert!(small.tile_m <= 64);

        let large = optimal_matmul_tiles(4096, 4096, 4096, DType::F16);
        assert_eq!(large.tile_m, 128);
        assert_eq!(large.tile_n, 128);
    }

    #[test]
    fn ptx_types() {
        assert_eq!(ptx_type(DType::F16), ".f16");
        assert_eq!(ptx_type(DType::F32), ".f32");
        assert_eq!(ptx_type(DType::I8), ".s8");
    }
}

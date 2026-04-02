//! Tensor data types with GPU-native precision support.
//!
//! Warp treats precision as a first-class optimization axis.
//! Every dtype knows its size, whether it needs emulation on specific
//! hardware, and how to promote/demote between precisions.

use serde::{Deserialize, Serialize};

/// All data types supported by the Warp IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    // Floating point
    F32,
    F16,
    BF16,
    F8E4M3,  // FP8 (NVIDIA Ada+)
    F8E5M2,  // FP8 (NVIDIA Ada+)

    // Integer
    I64,
    I32,
    I16,
    I8,
    I4, // Packed 4-bit integer (2 per byte)
    U8,
    U32,

    // Quantized (block-scaled)
    Q8_0,  // 8-bit with per-block f16 scale (GGUF-style)
    Q4_0,  // 4-bit with per-block f16 scale
    Q4_1,  // 4-bit with per-block f16 scale + min

    // Special
    Bool,
}

impl DType {
    /// Size in bits of a single element.
    pub const fn bit_width(self) -> u32 {
        match self {
            DType::Bool => 1,
            DType::I4 => 4,
            DType::Q4_0 | DType::Q4_1 => 4, // per-element; block overhead amortized
            DType::F8E4M3 | DType::F8E5M2 | DType::I8 | DType::U8 | DType::Q8_0 => 8,
            DType::F16 | DType::BF16 | DType::I16 => 16,
            DType::F32 | DType::I32 | DType::U32 => 32,
            DType::I64 => 64,
        }
    }

    /// Size in bytes, rounded up for sub-byte types.
    pub const fn byte_size(self) -> usize {
        ((self.bit_width() + 7) / 8) as usize
    }

    /// Whether this type is a floating-point type.
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            DType::F32 | DType::F16 | DType::BF16 | DType::F8E4M3 | DType::F8E5M2
        )
    }

    /// Whether this type is a block-quantized type.
    pub const fn is_quantized(self) -> bool {
        matches!(self, DType::Q8_0 | DType::Q4_0 | DType::Q4_1)
    }

    /// Whether this type requires special hardware support (Ada+ for FP8).
    pub const fn requires_hw_support(self) -> bool {
        matches!(self, DType::F8E4M3 | DType::F8E5M2)
    }

    /// The "compute type" — what this dtype promotes to for arithmetic.
    /// Sub-f32 types accumulate into f32 on most hardware.
    pub const fn compute_type(self) -> DType {
        match self {
            DType::F32 | DType::I32 | DType::U32 | DType::I64 => self,
            DType::F16 | DType::BF16 | DType::F8E4M3 | DType::F8E5M2 => DType::F32,
            DType::I8 | DType::U8 | DType::I4 | DType::I16 => DType::I32,
            DType::Q8_0 | DType::Q4_0 | DType::Q4_1 => DType::F32,
            DType::Bool => DType::I32,
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::F8E4M3 => write!(f, "f8e4m3"),
            DType::F8E5M2 => write!(f, "f8e5m2"),
            DType::I64 => write!(f, "i64"),
            DType::I32 => write!(f, "i32"),
            DType::I16 => write!(f, "i16"),
            DType::I8 => write!(f, "i8"),
            DType::I4 => write!(f, "i4"),
            DType::U8 => write!(f, "u8"),
            DType::U32 => write!(f, "u32"),
            DType::Q8_0 => write!(f, "q8_0"),
            DType::Q4_0 => write!(f, "q4_0"),
            DType::Q4_1 => write!(f, "q4_1"),
            DType::Bool => write!(f, "bool"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_type_promotion() {
        assert_eq!(DType::F16.compute_type(), DType::F32);
        assert_eq!(DType::BF16.compute_type(), DType::F32);
        assert_eq!(DType::F8E4M3.compute_type(), DType::F32);
        assert_eq!(DType::I8.compute_type(), DType::I32);
        assert_eq!(DType::Q4_0.compute_type(), DType::F32);
        assert_eq!(DType::F32.compute_type(), DType::F32);
    }

    #[test]
    fn bit_widths() {
        assert_eq!(DType::F32.bit_width(), 32);
        assert_eq!(DType::F16.bit_width(), 16);
        assert_eq!(DType::I4.bit_width(), 4);
        assert_eq!(DType::Bool.bit_width(), 1);
    }
}

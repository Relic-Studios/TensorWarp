//! Tensor shapes — static, dynamic, and symbolic.
//!
//! Warp distinguishes between fully-known shapes (all dims concrete),
//! partially-known shapes (some dims symbolic), and fully dynamic shapes.
//! Static shapes enable the most aggressive optimizations — kernel codegen
//! can bake in exact tile sizes, unroll loops, and eliminate bounds checks.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// A single dimension — either a concrete size or a symbolic variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dim {
    /// Known at graph construction time.
    Static(usize),
    /// Resolved at runtime. The u32 is a symbolic variable ID —
    /// all dims sharing the same ID must have the same runtime value.
    Dynamic(u32),
}

impl Dim {
    pub const fn is_static(self) -> bool {
        matches!(self, Dim::Static(_))
    }

    pub const fn static_val(self) -> Option<usize> {
        match self {
            Dim::Static(v) => Some(v),
            Dim::Dynamic(_) => None,
        }
    }

    pub const fn is_dynamic(self) -> bool {
        matches!(self, Dim::Dynamic(_))
    }
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dim::Static(v) => write!(f, "{v}"),
            Dim::Dynamic(id) => write!(f, "?{id}"),
        }
    }
}

/// Tensor shape. SmallVec<[Dim; 4]> because most tensors are 2-4D,
/// and we want to avoid heap allocation in the common case.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape {
    dims: SmallVec<[Dim; 4]>,
}

impl Shape {
    /// Create a fully-static shape from concrete dimensions.
    pub fn from_static(dims: &[usize]) -> Self {
        Self {
            dims: dims.iter().map(|&d| Dim::Static(d)).collect(),
        }
    }

    /// Create a shape from mixed static/dynamic dims.
    pub fn new(dims: impl Into<SmallVec<[Dim; 4]>>) -> Self {
        Self { dims: dims.into() }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn dims(&self) -> &[Dim] {
        &self.dims
    }

    /// Returns true if all dimensions are statically known.
    pub fn is_static(&self) -> bool {
        self.dims.iter().all(|d| d.is_static())
    }

    /// Total number of elements, if fully static.
    pub fn numel(&self) -> Option<usize> {
        if !self.is_static() {
            return None;
        }
        Some(
            self.dims
                .iter()
                .map(|d| d.static_val().unwrap())
                .product(),
        )
    }

    /// Total number of elements, panics if shape is dynamic.
    pub fn numel_static(&self) -> usize {
        self.numel().expect("shape has dynamic dimensions")
    }

    /// Get a specific dimension.
    pub fn dim(&self, idx: usize) -> Dim {
        self.dims[idx]
    }

    /// Check if two shapes are broadcast-compatible.
    pub fn broadcast_compatible(&self, other: &Shape) -> bool {
        let (a, b) = (&self.dims, &other.dims);
        let max_rank = a.len().max(b.len());

        for i in 0..max_rank {
            let da = if i < max_rank - a.len() {
                Dim::Static(1)
            } else {
                a[i - (max_rank - a.len())]
            };
            let db = if i < max_rank - b.len() {
                Dim::Static(1)
            } else {
                b[i - (max_rank - b.len())]
            };

            match (da, db) {
                (Dim::Static(1), _) | (_, Dim::Static(1)) => continue,
                (Dim::Static(a), Dim::Static(b)) if a == b => continue,
                (Dim::Dynamic(a), Dim::Dynamic(b)) if a == b => continue,
                (Dim::Dynamic(_), _) | (_, Dim::Dynamic(_)) => continue, // assume compatible at runtime
                _ => return false,
            }
        }
        true
    }

    /// Compute broadcast result shape.
    pub fn broadcast_shape(&self, other: &Shape) -> Option<Shape> {
        if !self.broadcast_compatible(other) {
            return None;
        }
        let (a, b) = (&self.dims, &other.dims);
        let max_rank = a.len().max(b.len());
        let mut result = SmallVec::with_capacity(max_rank);

        for i in 0..max_rank {
            let da = if i < max_rank - a.len() {
                Dim::Static(1)
            } else {
                a[i - (max_rank - a.len())]
            };
            let db = if i < max_rank - b.len() {
                Dim::Static(1)
            } else {
                b[i - (max_rank - b.len())]
            };

            let out = match (da, db) {
                (Dim::Static(1), d) | (d, Dim::Static(1)) => d,
                (Dim::Static(a), Dim::Static(_)) => Dim::Static(a), // equal, checked above
                (d @ Dim::Dynamic(_), _) | (_, d @ Dim::Dynamic(_)) => d,
            };
            result.push(out);
        }
        Some(Shape::new(result))
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// Convenience macro for building shapes.
#[macro_export]
macro_rules! shape {
    ($($d:expr),* $(,)?) => {
        $crate::Shape::from_static(&[$($d),*])
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_shape() {
        let s = Shape::from_static(&[2, 3, 4]);
        assert_eq!(s.rank(), 3);
        assert!(s.is_static());
        assert_eq!(s.numel(), Some(24));
        assert_eq!(s.to_string(), "[2, 3, 4]");
    }

    #[test]
    fn dynamic_shape() {
        let s = Shape::new(SmallVec::from_vec(vec![
            Dim::Dynamic(0),
            Dim::Static(128),
            Dim::Static(768),
        ]));
        assert_eq!(s.rank(), 3);
        assert!(!s.is_static());
        assert_eq!(s.numel(), None);
        assert_eq!(s.to_string(), "[?0, 128, 768]");
    }

    #[test]
    fn broadcast() {
        let a = Shape::from_static(&[1, 3, 1]);
        let b = Shape::from_static(&[2, 1, 4]);
        assert!(a.broadcast_compatible(&b));
        let c = a.broadcast_shape(&b).unwrap();
        assert_eq!(c.to_string(), "[2, 3, 4]");
    }

    #[test]
    fn broadcast_rank_mismatch() {
        let a = Shape::from_static(&[3, 4]);
        let b = Shape::from_static(&[2, 3, 4]);
        assert!(a.broadcast_compatible(&b));
        let c = a.broadcast_shape(&b).unwrap();
        assert_eq!(c.to_string(), "[2, 3, 4]");
    }

    #[test]
    fn broadcast_incompatible() {
        let a = Shape::from_static(&[3, 4]);
        let b = Shape::from_static(&[3, 5]);
        assert!(!a.broadcast_compatible(&b));
    }
}

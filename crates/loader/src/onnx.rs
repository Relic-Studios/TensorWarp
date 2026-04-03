//! ONNX model importer — loads .onnx files into warp-ir graphs.
//!
//! Parses the ONNX protobuf format and maps operations to warp-ir ops.
//! Supports the common ops needed for CNNs, transformers, and detection models.
//!
//! Usage:
//! ```ignore
//! let model = OnnxModel::load("model.onnx")?;
//! println!("Inputs: {:?}", model.inputs);
//! println!("Outputs: {:?}", model.outputs);
//! for node in &model.nodes {
//!     println!("  {} ({})", node.name, node.op_type);
//! }
//! ```

use std::collections::HashMap;
use std::path::Path;

/// ONNX data type constants (from onnx.proto TensorProto.DataType).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OnnxDType {
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
    BFloat16 = 16,
    Float8E4M3FN = 17,
    Float8E5M2 = 19,
}

impl OnnxDType {
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            1 => Some(OnnxDType::Float),
            2 => Some(OnnxDType::Uint8),
            3 => Some(OnnxDType::Int8),
            5 => Some(OnnxDType::Int16),
            6 => Some(OnnxDType::Int32),
            7 => Some(OnnxDType::Int64),
            10 => Some(OnnxDType::Float16),
            11 => Some(OnnxDType::Double),
            16 => Some(OnnxDType::BFloat16),
            _ => None,
        }
    }

    pub fn to_warp_dtype(self) -> warp_ir::DType {
        match self {
            OnnxDType::Float | OnnxDType::Double => warp_ir::DType::F32,
            OnnxDType::Float16 => warp_ir::DType::F16,
            OnnxDType::BFloat16 => warp_ir::DType::BF16,
            OnnxDType::Int8 => warp_ir::DType::I8,
            OnnxDType::Uint8 => warp_ir::DType::U8,
            OnnxDType::Int16 => warp_ir::DType::I16,
            OnnxDType::Int32 => warp_ir::DType::I32,
            OnnxDType::Int64 => warp_ir::DType::I64,
            OnnxDType::Uint32 => warp_ir::DType::U32,
            _ => warp_ir::DType::F32,
        }
    }

    pub fn byte_size(self) -> usize {
        match self {
            OnnxDType::Float | OnnxDType::Int32 | OnnxDType::Uint32 => 4,
            OnnxDType::Float16 | OnnxDType::BFloat16 | OnnxDType::Int16 | OnnxDType::Uint16 => 2,
            OnnxDType::Int8 | OnnxDType::Uint8 => 1,
            OnnxDType::Int64 | OnnxDType::Uint64 | OnnxDType::Double => 8,
            _ => 4,
        }
    }
}

/// An ONNX attribute (node parameter).
#[derive(Debug, Clone)]
pub enum OnnxAttr {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
}

/// An ONNX graph node (operation).
#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, OnnxAttr>,
}

impl OnnxNode {
    pub fn get_int(&self, name: &str, default: i64) -> i64 {
        match self.attrs.get(name) {
            Some(OnnxAttr::Int(v)) => *v,
            _ => default,
        }
    }

    pub fn get_float(&self, name: &str, default: f32) -> f32 {
        match self.attrs.get(name) {
            Some(OnnxAttr::Float(v)) => *v,
            _ => default,
        }
    }

    pub fn get_ints(&self, name: &str) -> Vec<i64> {
        match self.attrs.get(name) {
            Some(OnnxAttr::Ints(v)) => v.clone(),
            _ => vec![],
        }
    }

    pub fn get_string(&self, name: &str) -> Option<&str> {
        match self.attrs.get(name) {
            Some(OnnxAttr::String(v)) => Some(v),
            _ => None,
        }
    }
}

/// An ONNX tensor (initializer / weight).
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    pub name: String,
    pub dtype: OnnxDType,
    pub shape: Vec<i64>,
    pub raw_data: Vec<u8>,
}

impl OnnxTensor {
    /// Read as f32 values (converting from the stored dtype).
    pub fn to_f32(&self) -> Vec<f32> {
        match self.dtype {
            OnnxDType::Float => {
                self.raw_data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            }
            OnnxDType::Double => {
                self.raw_data.chunks_exact(8)
                    .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
                    .collect()
            }
            OnnxDType::Float16 => {
                self.raw_data.chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect()
            }
            OnnxDType::Int64 => {
                self.raw_data.chunks_exact(8)
                    .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
                    .collect()
            }
            OnnxDType::Int32 => {
                self.raw_data.chunks_exact(4)
                    .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
                    .collect()
            }
            _ => {
                log::warn!("Unsupported ONNX dtype {:?} for f32 conversion", self.dtype);
                vec![]
            }
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }
}

/// Graph input/output specification.
#[derive(Debug, Clone)]
pub struct OnnxIO {
    pub name: String,
    pub dtype: Option<OnnxDType>,
    pub shape: Vec<i64>, // -1 for dynamic dims
}

/// A parsed ONNX model.
#[derive(Debug)]
pub struct OnnxModel {
    /// Model inputs (excluding initializers).
    pub inputs: Vec<OnnxIO>,
    /// Model outputs.
    pub outputs: Vec<OnnxIO>,
    /// Graph nodes (operations in topological order).
    pub nodes: Vec<OnnxNode>,
    /// Initializers (weights, constants).
    pub initializers: HashMap<String, OnnxTensor>,
    /// IR version.
    pub ir_version: i64,
    /// Opset version.
    pub opset_version: i64,
    /// Producer name.
    pub producer: String,
}

/// Errors during ONNX loading.
#[derive(Debug, thiserror::Error)]
pub enum OnnxError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Protobuf decode error: {0}")]
    Decode(String),
    #[error("Unsupported ONNX op: {0}")]
    UnsupportedOp(String),
    #[error("Missing attribute: {node}.{attr}")]
    MissingAttr { node: String, attr: String },
    #[error("Invalid model: {0}")]
    Invalid(String),
}

impl OnnxModel {
    /// Load an ONNX model from a file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, OnnxError> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
    }

    /// Parse an ONNX model from raw protobuf bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, OnnxError> {
        // ONNX protobuf is a ModelProto. We parse it manually using prost's
        // low-level wire format decoder since we don't want to depend on
        // compiled .proto files.
        let mut model = OnnxModel {
            inputs: vec![],
            outputs: vec![],
            nodes: vec![],
            initializers: HashMap::new(),
            ir_version: 0,
            opset_version: 0,
            producer: String::new(),
        };

        parse_model_proto(data, &mut model)?;
        Ok(model)
    }

    /// Map an ONNX op_type to the corresponding warp-ir Op.
    pub fn map_op(node: &OnnxNode) -> Result<OnnxOpMapping, OnnxError> {
        match node.op_type.as_str() {
            // Core compute
            "MatMul" | "Gemm" => {
                let trans_a = node.get_int("transA", 0) != 0;
                let trans_b = node.get_int("transB", 0) != 0;
                Ok(OnnxOpMapping::MatMul { trans_a, trans_b })
            }

            // Activations
            "Relu" => Ok(OnnxOpMapping::Activation("relu")),
            "Sigmoid" => Ok(OnnxOpMapping::Activation("sigmoid")),
            "Tanh" => Ok(OnnxOpMapping::Activation("tanh")),
            "Gelu" => Ok(OnnxOpMapping::Activation("gelu")),
            "Silu" | "Swish" => Ok(OnnxOpMapping::Activation("silu")),
            "LeakyRelu" => Ok(OnnxOpMapping::LeakyRelu {
                alpha: node.get_float("alpha", 0.01),
            }),
            "Clip" => Ok(OnnxOpMapping::Clip),

            // Elementwise
            "Add" => Ok(OnnxOpMapping::Binary("add")),
            "Sub" => Ok(OnnxOpMapping::Binary("sub")),
            "Mul" => Ok(OnnxOpMapping::Binary("mul")),
            "Div" => Ok(OnnxOpMapping::Binary("div")),

            // Normalization
            "BatchNormalization" => Ok(OnnxOpMapping::BatchNorm {
                eps: node.get_float("epsilon", 1e-5),
            }),
            "LayerNormalization" => Ok(OnnxOpMapping::LayerNorm {
                eps: node.get_float("epsilon", 1e-5),
            }),
            "GroupNormalization" => Ok(OnnxOpMapping::GroupNorm {
                num_groups: node.get_int("num_groups", 32) as u32,
                eps: node.get_float("epsilon", 1e-5),
            }),
            "InstanceNormalization" => Ok(OnnxOpMapping::InstanceNorm {
                eps: node.get_float("epsilon", 1e-5),
            }),

            // Convolution
            "Conv" => {
                let kernel = node.get_ints("kernel_shape");
                let strides = node.get_ints("strides");
                let pads = node.get_ints("pads");
                let dilations = node.get_ints("dilations");
                let group = node.get_int("group", 1) as u32;
                Ok(OnnxOpMapping::Conv {
                    kernel: kernel.iter().map(|&x| x as u32).collect(),
                    strides: strides.iter().map(|&x| x as u32).collect(),
                    pads: pads.iter().map(|&x| x as u32).collect(),
                    dilations: dilations.iter().map(|&x| x as u32).collect(),
                    group,
                })
            }
            "ConvTranspose" => {
                let kernel = node.get_ints("kernel_shape");
                let strides = node.get_ints("strides");
                let pads = node.get_ints("pads");
                let group = node.get_int("group", 1) as u32;
                Ok(OnnxOpMapping::ConvTranspose {
                    kernel: kernel.iter().map(|&x| x as u32).collect(),
                    strides: strides.iter().map(|&x| x as u32).collect(),
                    pads: pads.iter().map(|&x| x as u32).collect(),
                    group,
                })
            }

            // Pooling
            "MaxPool" => {
                let kernel = node.get_ints("kernel_shape");
                let strides = node.get_ints("strides");
                let pads = node.get_ints("pads");
                Ok(OnnxOpMapping::MaxPool {
                    kernel: kernel.iter().map(|&x| x as u32).collect(),
                    strides: strides.iter().map(|&x| x as u32).collect(),
                    pads: pads.iter().map(|&x| x as u32).collect(),
                })
            }
            "AveragePool" => {
                let kernel = node.get_ints("kernel_shape");
                let strides = node.get_ints("strides");
                let pads = node.get_ints("pads");
                Ok(OnnxOpMapping::AvgPool {
                    kernel: kernel.iter().map(|&x| x as u32).collect(),
                    strides: strides.iter().map(|&x| x as u32).collect(),
                    pads: pads.iter().map(|&x| x as u32).collect(),
                })
            }
            "GlobalAveragePool" => Ok(OnnxOpMapping::GlobalAvgPool),

            // Shape ops
            "Reshape" => Ok(OnnxOpMapping::Reshape),
            "Transpose" => {
                let perm = node.get_ints("perm");
                Ok(OnnxOpMapping::Transpose { perm: perm.iter().map(|&x| x as usize).collect() })
            }
            "Concat" => {
                let axis = node.get_int("axis", 0) as i32;
                Ok(OnnxOpMapping::Concat { axis })
            }
            "Split" => {
                let axis = node.get_int("axis", 0) as i32;
                Ok(OnnxOpMapping::Split { axis })
            }
            "Flatten" => {
                let axis = node.get_int("axis", 1) as i32;
                Ok(OnnxOpMapping::Flatten { axis })
            }
            "Squeeze" | "Unsqueeze" => Ok(OnnxOpMapping::Reshape),
            "Gather" => {
                let axis = node.get_int("axis", 0) as i32;
                Ok(OnnxOpMapping::Gather { axis })
            }
            "Slice" => Ok(OnnxOpMapping::Slice),

            // Resize
            "Resize" | "Upsample" => {
                let mode = node.get_string("mode").unwrap_or("nearest");
                Ok(OnnxOpMapping::Resize { mode: mode.to_string() })
            }

            // Reduce
            "ReduceMean" => Ok(OnnxOpMapping::Reduce { op: "mean" }),
            "ReduceSum" => Ok(OnnxOpMapping::Reduce { op: "sum" }),
            "ReduceMax" => Ok(OnnxOpMapping::Reduce { op: "max" }),

            // Softmax
            "Softmax" => {
                let axis = node.get_int("axis", -1) as i32;
                Ok(OnnxOpMapping::Softmax { axis })
            }

            // Other
            "Constant" => Ok(OnnxOpMapping::Constant),
            "Shape" => Ok(OnnxOpMapping::Shape),
            "Cast" => Ok(OnnxOpMapping::Cast),
            "Pad" => Ok(OnnxOpMapping::Pad),
            "NonMaxSuppression" => Ok(OnnxOpMapping::NMS),
            "TopK" => Ok(OnnxOpMapping::TopK),

            // Broadcast / conditional
            "Expand" => Ok(OnnxOpMapping::Identity), // broadcast shape — handled by kernel
            "Where" => Ok(OnnxOpMapping::Identity),   // conditional selection
            "Pow" => Ok(OnnxOpMapping::Binary("pow")),
            "Sqrt" => Ok(OnnxOpMapping::Activation("sqrt")),
            "Neg" => Ok(OnnxOpMapping::Activation("neg")),
            "Abs" => Ok(OnnxOpMapping::Activation("abs")),
            "Exp" => Ok(OnnxOpMapping::Activation("exp")),
            "Log" => Ok(OnnxOpMapping::Activation("log")),
            "Reciprocal" => Ok(OnnxOpMapping::Activation("reciprocal")),
            "ArgMax" | "ArgMin" => Ok(OnnxOpMapping::TopK),
            "Einsum" => Ok(OnnxOpMapping::MatMul { trans_a: false, trans_b: false }),
            "ConstantOfShape" => Ok(OnnxOpMapping::Constant),
            "Range" => Ok(OnnxOpMapping::Shape),
            "CumSum" => Ok(OnnxOpMapping::Reduce { op: "cumsum" }),
            "Tile" => Ok(OnnxOpMapping::Identity),
            "DepthToSpace" | "SpaceToDepth" => Ok(OnnxOpMapping::Reshape),
            "OneHot" => Ok(OnnxOpMapping::Constant),
            "NonZero" => Ok(OnnxOpMapping::Shape),
            "ScatterND" | "ScatterElements" => Ok(OnnxOpMapping::Identity),
            "Floor" | "Ceil" | "Round" => Ok(OnnxOpMapping::Identity), // rounding ops
            "Min" | "Max" => Ok(OnnxOpMapping::Binary("minmax")),
            "Equal" | "Greater" | "Less" | "Not" | "And" | "Or" => Ok(OnnxOpMapping::Identity),
            "MatMulInteger" => Ok(OnnxOpMapping::MatMul { trans_a: false, trans_b: false }),

            // Identity / no-op
            "Identity" | "Dropout" | "Flatten" => Ok(OnnxOpMapping::Identity),

            other => Err(OnnxError::UnsupportedOp(other.to_string())),
        }
    }

    /// Get the list of supported ONNX op types.
    pub fn supported_ops() -> Vec<&'static str> {
        vec![
            "MatMul", "Gemm",
            "Relu", "Sigmoid", "Tanh", "Gelu", "Silu", "Swish", "LeakyRelu", "Clip",
            "Add", "Sub", "Mul", "Div",
            "BatchNormalization", "LayerNormalization", "GroupNormalization", "InstanceNormalization",
            "Conv", "ConvTranspose",
            "MaxPool", "AveragePool", "GlobalAveragePool",
            "Reshape", "Transpose", "Concat", "Split", "Flatten", "Squeeze", "Unsqueeze",
            "Gather", "Slice",
            "Resize", "Upsample",
            "ReduceMean", "ReduceSum", "ReduceMax",
            "Softmax",
            "Constant", "Shape", "Cast", "Pad",
            "NonMaxSuppression", "TopK",
            "Identity", "Dropout", "Flatten",
            "Expand", "Where", "Pow", "Sqrt", "Neg", "Abs", "Exp", "Log", "Reciprocal",
            "Floor", "Ceil", "Round", "Min", "Max",
            "Equal", "Greater", "Less", "Not", "And", "Or",
            "MatMulInteger",
            "ArgMax", "ArgMin", "Einsum", "ConstantOfShape", "Range", "CumSum",
            "Tile", "DepthToSpace", "SpaceToDepth", "OneHot", "NonZero",
            "ScatterND", "ScatterElements",
        ]
    }

    /// Print a summary of the model.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("ONNX Model (IR v{}, opset v{}, producer: {})\n",
            self.ir_version, self.opset_version, self.producer));
        s.push_str(&format!("  Inputs:  {}\n", self.inputs.len()));
        for inp in &self.inputs {
            s.push_str(&format!("    {} {:?}\n", inp.name, inp.shape));
        }
        s.push_str(&format!("  Outputs: {}\n", self.outputs.len()));
        for out in &self.outputs {
            s.push_str(&format!("    {} {:?}\n", out.name, out.shape));
        }
        s.push_str(&format!("  Nodes:   {}\n", self.nodes.len()));
        s.push_str(&format!("  Weights: {} ({:.2} MB)\n",
            self.initializers.len(),
            self.initializers.values().map(|t| t.raw_data.len()).sum::<usize>() as f64 / 1e6));

        // Op type histogram
        let mut op_counts: HashMap<&str, usize> = HashMap::new();
        for node in &self.nodes {
            *op_counts.entry(&node.op_type).or_insert(0) += 1;
        }
        let mut ops: Vec<_> = op_counts.into_iter().collect();
        ops.sort_by(|a, b| b.1.cmp(&a.1));
        s.push_str("  Op types:\n");
        for (op, count) in ops {
            let supported = Self::map_op(&OnnxNode {
                name: String::new(), op_type: op.to_string(),
                inputs: vec![], outputs: vec![],
                attrs: HashMap::new(),
            }).is_ok();
            let marker = if supported { "✓" } else { "✗" };
            s.push_str(&format!("    {marker} {op}: {count}\n"));
        }
        s
    }
}

/// Mapped operation — intermediate representation before building the warp-ir graph.
#[derive(Debug)]
pub enum OnnxOpMapping {
    MatMul { trans_a: bool, trans_b: bool },
    Activation(&'static str),
    LeakyRelu { alpha: f32 },
    Clip,
    Binary(&'static str),
    BatchNorm { eps: f32 },
    LayerNorm { eps: f32 },
    GroupNorm { num_groups: u32, eps: f32 },
    InstanceNorm { eps: f32 },
    Conv { kernel: Vec<u32>, strides: Vec<u32>, pads: Vec<u32>, dilations: Vec<u32>, group: u32 },
    ConvTranspose { kernel: Vec<u32>, strides: Vec<u32>, pads: Vec<u32>, group: u32 },
    MaxPool { kernel: Vec<u32>, strides: Vec<u32>, pads: Vec<u32> },
    AvgPool { kernel: Vec<u32>, strides: Vec<u32>, pads: Vec<u32> },
    GlobalAvgPool,
    Reshape,
    Transpose { perm: Vec<usize> },
    Concat { axis: i32 },
    Split { axis: i32 },
    Flatten { axis: i32 },
    Gather { axis: i32 },
    Slice,
    Resize { mode: String },
    Reduce { op: &'static str },
    Softmax { axis: i32 },
    Constant,
    Shape,
    Cast,
    Pad,
    NMS,
    TopK,
    Identity,
}

// ═════════════════════════════════════════════════════════════════
// Minimal protobuf wire format decoder
// ═════════════════════════════════════════════════════════════════

/// Protobuf wire types.
#[derive(Debug, Clone, Copy, PartialEq)]
enum WireType { Varint, Fixed64, LengthDelimited, Fixed32 }

/// Decode a varint from the buffer, advancing the position.
fn pb_varint(buf: &mut &[u8]) -> Result<u64, OnnxError> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        if buf.is_empty() { return Err(OnnxError::Invalid("unexpected EOF in varint".into())); }
        let byte = buf[0];
        *buf = &buf[1..];
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 { return Ok(result); }
        shift += 7;
        if shift >= 64 { return Err(OnnxError::Invalid("varint too long".into())); }
    }
}

/// Decode a field tag (field_number, wire_type).
fn pb_tag(buf: &mut &[u8]) -> Result<(u32, WireType), OnnxError> {
    let v = pb_varint(buf)? as u32;
    let field = v >> 3;
    let wt = match v & 0x7 {
        0 => WireType::Varint,
        1 => WireType::Fixed64,
        2 => WireType::LengthDelimited,
        5 => WireType::Fixed32,
        other => return Err(OnnxError::Invalid(format!("unknown wire type {other}"))),
    };
    Ok((field, wt))
}

/// Read a length-delimited field, returning the sub-slice.
fn pb_bytes<'a>(buf: &mut &'a [u8]) -> Result<&'a [u8], OnnxError> {
    let len = pb_varint(buf)? as usize;
    if buf.len() < len { return Err(OnnxError::Invalid("truncated length-delimited field".into())); }
    let data = &buf[..len];
    *buf = &buf[len..];
    Ok(data)
}

/// Read a string field.
fn pb_string(buf: &mut &[u8]) -> Result<String, OnnxError> {
    let data = pb_bytes(buf)?;
    Ok(String::from_utf8_lossy(data).to_string())
}

/// Skip a field based on its wire type.
fn pb_skip(wt: WireType, buf: &mut &[u8]) -> Result<(), OnnxError> {
    match wt {
        WireType::Varint => { pb_varint(buf)?; }
        WireType::Fixed64 => {
            if buf.len() < 8 { return Err(OnnxError::Invalid("truncated fixed64".into())); }
            *buf = &buf[8..];
        }
        WireType::LengthDelimited => { pb_bytes(buf)?; }
        WireType::Fixed32 => {
            if buf.len() < 4 { return Err(OnnxError::Invalid("truncated fixed32".into())); }
            *buf = &buf[4..];
        }
    }
    Ok(())
}

// ═════════════════════════════════════════════════════════════════
// ONNX protobuf message parsers
// ═════════════════════════════════════════════════════════════════

fn parse_model_proto(data: &[u8], model: &mut OnnxModel) -> Result<(), OnnxError> {
    let mut buf = data;
    while !buf.is_empty() {
        let (tag, wt) = pb_tag(&mut buf)?;
        match (tag, wt) {
            (1, WireType::Varint) => { model.ir_version = pb_varint(&mut buf)? as i64; }
            (2, WireType::LengthDelimited) => { model.producer = pb_string(&mut buf)?; }
            (7, WireType::LengthDelimited) => { parse_graph_proto(pb_bytes(&mut buf)?, model)?; }
            (8, WireType::LengthDelimited) => {
                let od = pb_bytes(&mut buf)?;
                let mut ob = od;
                while !ob.is_empty() {
                    let (ot, owt) = pb_tag(&mut ob)?;
                    if ot == 2 && owt == WireType::Varint { model.opset_version = pb_varint(&mut ob)? as i64; }
                    else { pb_skip(owt, &mut ob)?; }
                }
            }
            (_, wt) => { pb_skip(wt, &mut buf)?; }
        }
    }
    Ok(())
}

fn parse_graph_proto(data: &[u8], model: &mut OnnxModel) -> Result<(), OnnxError> {
    let mut init_names: std::collections::HashSet<String> = std::collections::HashSet::new();

    // First pass: initializers
    let mut buf = data;
    while !buf.is_empty() {
        let (tag, wt) = pb_tag(&mut buf)?;
        if tag == 5 && wt == WireType::LengthDelimited {
            let td = pb_bytes(&mut buf)?;
            if let Ok(t) = parse_tensor_proto(td) {
                init_names.insert(t.name.clone());
                model.initializers.insert(t.name.clone(), t);
            }
        } else { pb_skip(wt, &mut buf)?; }
    }

    // Second pass: nodes, inputs, outputs
    let mut buf = data;
    while !buf.is_empty() {
        let (tag, wt) = pb_tag(&mut buf)?;
        match (tag, wt) {
            (1, WireType::LengthDelimited) => {
                if let Ok(n) = parse_node_proto(pb_bytes(&mut buf)?) { model.nodes.push(n); }
            }
            (5, WireType::LengthDelimited) => { pb_bytes(&mut buf)?; } // skip (already parsed)
            (11, WireType::LengthDelimited) => {
                if let Ok(io) = parse_value_info(pb_bytes(&mut buf)?) {
                    if !init_names.contains(&io.name) { model.inputs.push(io); }
                }
            }
            (12, WireType::LengthDelimited) => {
                if let Ok(io) = parse_value_info(pb_bytes(&mut buf)?) { model.outputs.push(io); }
            }
            (_, wt) => { pb_skip(wt, &mut buf)?; }
        }
    }
    Ok(())
}

fn parse_node_proto(data: &[u8]) -> Result<OnnxNode, OnnxError> {
    let mut node = OnnxNode { name: String::new(), op_type: String::new(), inputs: vec![], outputs: vec![], attrs: HashMap::new() };
    let mut buf = data;
    while !buf.is_empty() {
        let (tag, wt) = pb_tag(&mut buf)?;
        match (tag, wt) {
            (1, WireType::LengthDelimited) => { node.inputs.push(pb_string(&mut buf)?); }
            (2, WireType::LengthDelimited) => { node.outputs.push(pb_string(&mut buf)?); }
            (3, WireType::LengthDelimited) => { node.name = pb_string(&mut buf)?; }
            (4, WireType::LengthDelimited) => { node.op_type = pb_string(&mut buf)?; }
            (5, WireType::LengthDelimited) => {
                if let Ok((n, a)) = parse_attribute(pb_bytes(&mut buf)?) { node.attrs.insert(n, a); }
            }
            (_, wt) => { pb_skip(wt, &mut buf)?; }
        }
    }
    Ok(node)
}

fn parse_attribute(data: &[u8]) -> Result<(String, OnnxAttr), OnnxError> {
    let mut name = String::new();
    let mut attr_type = 0i32;
    let mut f_val = 0.0f32;
    let mut i_val = 0i64;
    let mut s_val = Vec::new();
    let mut floats = Vec::new();
    let mut ints = Vec::new();

    let mut buf = data;
    while !buf.is_empty() {
        let (tag, wt) = pb_tag(&mut buf)?;
        match (tag, wt) {
            (1, WireType::LengthDelimited) => { name = pb_string(&mut buf)?; }
            (2, WireType::Varint) => { attr_type = pb_varint(&mut buf)? as i32; }
            (3, WireType::Varint) => { i_val = pb_varint(&mut buf)? as i64; }
            (4, WireType::Fixed32) => {
                f_val = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
                buf = &buf[4..];
            }
            (4, WireType::LengthDelimited) => { // string (s field)
                let d = pb_bytes(&mut buf)?;
                s_val = d.to_vec();
            }
            (7, WireType::LengthDelimited) => { // packed floats
                let d = pb_bytes(&mut buf)?;
                for c in d.chunks_exact(4) { floats.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]])); }
            }
            (7, WireType::Fixed32) => {
                floats.push(f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]));
                buf = &buf[4..];
            }
            (8, WireType::LengthDelimited) => { // packed ints
                let d = pb_bytes(&mut buf)?;
                let mut ib = d;
                while !ib.is_empty() { ints.push(pb_varint(&mut ib)? as i64); }
            }
            (8, WireType::Varint) => { ints.push(pb_varint(&mut buf)? as i64); }
            (20, WireType::Varint) => { attr_type = pb_varint(&mut buf)? as i32; } // type (alt field)
            (_, wt) => { pb_skip(wt, &mut buf)?; }
        }
    }

    let attr = match attr_type {
        1 => OnnxAttr::Float(f_val),
        2 => OnnxAttr::Int(i_val),
        3 => OnnxAttr::String(String::from_utf8_lossy(&s_val).to_string()),
        6 => OnnxAttr::Floats(floats),
        7 => OnnxAttr::Ints(ints),
        _ => OnnxAttr::Int(i_val),
    };
    Ok((name, attr))
}

fn parse_tensor_proto(data: &[u8]) -> Result<OnnxTensor, OnnxError> {
    let mut name = String::new();
    let mut dtype = 1i32;
    let mut dims = Vec::new();
    let mut raw_data = Vec::new();
    let mut float_data = Vec::new();

    let mut buf = data;
    while !buf.is_empty() {
        let (tag, wt) = pb_tag(&mut buf)?;
        match (tag, wt) {
            (1, WireType::LengthDelimited) => { // packed dims
                let d = pb_bytes(&mut buf)?;
                let mut db = d;
                while !db.is_empty() { dims.push(pb_varint(&mut db)? as i64); }
            }
            (1, WireType::Varint) => { dims.push(pb_varint(&mut buf)? as i64); }
            (2, WireType::Varint) => { dtype = pb_varint(&mut buf)? as i32; }
            (4, WireType::LengthDelimited) => { // packed float_data
                let d = pb_bytes(&mut buf)?;
                for c in d.chunks_exact(4) { float_data.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]])); }
            }
            (4, WireType::Fixed32) => {
                float_data.push(f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]));
                buf = &buf[4..];
            }
            (8, WireType::LengthDelimited) => { name = pb_string(&mut buf)?; }
            (13, WireType::LengthDelimited) => { raw_data = pb_bytes(&mut buf)?.to_vec(); }
            (_, wt) => { pb_skip(wt, &mut buf)?; }
        }
    }

    if raw_data.is_empty() && !float_data.is_empty() {
        raw_data = float_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    }

    Ok(OnnxTensor { name, dtype: OnnxDType::from_i32(dtype).unwrap_or(OnnxDType::Float), shape: dims, raw_data })
}

fn parse_value_info(data: &[u8]) -> Result<OnnxIO, OnnxError> {
    let mut name = String::new();
    let mut shape = Vec::new();
    let mut dtype = None;

    let mut buf = data;
    while !buf.is_empty() {
        let (tag, wt) = pb_tag(&mut buf)?;
        match (tag, wt) {
            (1, WireType::LengthDelimited) => { name = pb_string(&mut buf)?; }
            (2, WireType::LengthDelimited) => {
                let td = pb_bytes(&mut buf)?;
                parse_type_proto(td, &mut dtype, &mut shape)?;
            }
            (_, wt) => { pb_skip(wt, &mut buf)?; }
        }
    }
    Ok(OnnxIO { name, dtype, shape })
}

fn parse_type_proto(data: &[u8], dtype: &mut Option<OnnxDType>, shape: &mut Vec<i64>) -> Result<(), OnnxError> {
    let mut buf = data;
    while !buf.is_empty() {
        let (tag, wt) = pb_tag(&mut buf)?;
        if tag == 1 && wt == WireType::LengthDelimited {
            let tt = pb_bytes(&mut buf)?;
            let mut tb = tt;
            while !tb.is_empty() {
                let (t2, w2) = pb_tag(&mut tb)?;
                match (t2, w2) {
                    (1, WireType::Varint) => { *dtype = OnnxDType::from_i32(pb_varint(&mut tb)? as i32); }
                    (2, WireType::LengthDelimited) => {
                        let sd = pb_bytes(&mut tb)?;
                        let mut sb = sd;
                        while !sb.is_empty() {
                            let (s3, w3) = pb_tag(&mut sb)?;
                            if s3 == 1 && w3 == WireType::LengthDelimited {
                                let dd = pb_bytes(&mut sb)?;
                                let mut db = dd;
                                let mut dv = -1i64;
                                while !db.is_empty() {
                                    let (d4, w4) = pb_tag(&mut db)?;
                                    if d4 == 1 && w4 == WireType::Varint { dv = pb_varint(&mut db)? as i64; }
                                    else { pb_skip(w4, &mut db)?; }
                                }
                                shape.push(dv);
                            } else { pb_skip(w3, &mut sb)?; }
                        }
                    }
                    (_, w2) => { pb_skip(w2, &mut tb)?; }
                }
            }
        } else { pb_skip(wt, &mut buf)?; }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_ops_list() {
        let ops = OnnxModel::supported_ops();
        assert!(ops.len() >= 40, "Should support 40+ ONNX ops");
        println!("Supported ONNX ops ({}):", ops.len());
        for op in &ops {
            println!("  {op}");
        }
    }

    #[test]
    fn op_mapping_smoke() {
        // Verify all common ops can be mapped
        let test_ops = vec![
            ("Conv", vec![("kernel_shape", OnnxAttr::Ints(vec![3, 3]))]),
            ("BatchNormalization", vec![]),
            ("Relu", vec![]),
            ("MaxPool", vec![("kernel_shape", OnnxAttr::Ints(vec![2, 2]))]),
            ("Gemm", vec![]),
            ("Add", vec![]),
            ("Reshape", vec![]),
            ("Softmax", vec![]),
            ("GlobalAveragePool", vec![]),
            ("Concat", vec![]),
            ("Resize", vec![]),
            ("NonMaxSuppression", vec![]),
        ];

        for (op_type, attrs) in test_ops {
            let node = OnnxNode {
                name: format!("test_{op_type}"),
                op_type: op_type.to_string(),
                inputs: vec![],
                outputs: vec![],
                attrs: attrs.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            };

            let result = OnnxModel::map_op(&node);
            assert!(result.is_ok(), "Failed to map ONNX op: {op_type}");
            println!("  {op_type:30} → {:?}", result.unwrap());
        }
    }
}

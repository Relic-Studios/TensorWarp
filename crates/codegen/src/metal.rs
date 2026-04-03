//! Metal Shading Language code generation backend.
//!
//! Targets Apple GPUs (M1/M2/M3/M4 series).
//! Metal compute shaders are written in a C++-like language and compiled
//! to GPU-specific code by the Metal compiler at load time.
//!
//! Key differences from CUDA:
//! - Unified memory (no explicit host↔device copies)
//! - SIMD group width is 32 (same as CUDA warp)
//! - Threadgroup memory = shared memory
//! - No global atomic on FP16 (must use FP32 atomics)

use warp_ir::*;

use crate::backend::*;
use crate::kernel::metal_type;

/// Metal compute shader backend.
pub struct MetalBackend;

impl MetalBackend {
    pub fn new() -> Self {
        Self
    }

    fn gen_binary(
        &self,
        op: &BinaryOp,
        shape: &Shape,
        dtype: DType,
    ) -> Result<CompiledKernel, CodegenError> {
        let n = shape.numel_static();
        let ty = metal_type(dtype);
        let op_str = match op {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            _ => return Err(CodegenError::UnsupportedOp(format!("binary {:?}", op))),
        };

        let threads_per_group = 256;
        let grid_size = (n + threads_per_group - 1) / threads_per_group;

        let shader = format!(
            r#"#include <metal_stdlib>
using namespace metal;

kernel void warp_binary_{n}(
    device {ty}* out [[buffer(0)]],
    device const {ty}* a [[buffer(1)]],
    device const {ty}* b [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= {n}) return;
    out[tid] = a[tid] {op_str} b[tid];
}}
"#,
            n = n,
            ty = ty,
            op_str = op_str,
        );

        Ok(CompiledKernel {
            code: shader.into_bytes(),
            entry_point: format!("warp_binary_{n}"),
            grid: [grid_size as u32, 1, 1],
            block: [threads_per_group as u32, 1, 1],
            shared_mem_bytes: 0,
            description: format!("Metal binary {op:?} on {n} elements ({dtype})"),
        })
    }

    fn gen_activation(
        &self,
        activation: &Activation,
        shape: &Shape,
        dtype: DType,
    ) -> Result<CompiledKernel, CodegenError> {
        let n = shape.numel_static();
        let ty = metal_type(dtype);
        let threads_per_group = 256;
        let grid_size = (n + threads_per_group - 1) / threads_per_group;

        let body = match activation {
            Activation::Relu => format!("out[tid] = max(x, ({ty})0);"),
            Activation::Gelu | Activation::GeluTanh => format!(
                "float xf = float(x);\n    float g = 0.5f * xf * (1.0f + tanh(0.7978845608f * (xf + 0.044715f * xf * xf * xf)));\n    out[tid] = ({ty})g;"
            ),
            Activation::Silu => format!(
                "float xf = float(x);\n    out[tid] = ({ty})(xf / (1.0f + exp(-xf)));"
            ),
            Activation::Sigmoid => format!(
                "float xf = float(x);\n    out[tid] = ({ty})(1.0f / (1.0f + exp(-xf)));"
            ),
            _ => return Err(CodegenError::UnsupportedOp(format!("activation {:?}", activation))),
        };

        let shader = format!(
            r#"#include <metal_stdlib>
using namespace metal;

kernel void warp_activate_{n}(
    device {ty}* out [[buffer(0)]],
    device const {ty}* input [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= {n}) return;
    {ty} x = input[tid];
    {body}
}}
"#,
            n = n,
            ty = ty,
            body = body,
        );

        Ok(CompiledKernel {
            code: shader.into_bytes(),
            entry_point: format!("warp_activate_{n}"),
            grid: [grid_size as u32, 1, 1],
            block: [threads_per_group as u32, 1, 1],
            shared_mem_bytes: 0,
            description: format!("Metal {:?} activation on {n} elements", activation),
        })
    }

    fn gen_gemm(
        &self,
        m: usize, n: usize, k: usize,
        dtype: DType,
    ) -> Result<CompiledKernel, CodegenError> {
        let ty = metal_type(dtype);
        let tile = 16;

        let shader = format!(
            r#"#include <metal_stdlib>
using namespace metal;

#define M {m}
#define N {n}
#define K {k}
#define TILE {tile}

kernel void warp_gemm_{m}x{n}x{k}(
    device {ty}* C [[buffer(0)]],
    device const {ty}* A [[buffer(1)]],
    device const {ty}* B [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {{
    if (gid.y >= M || gid.x >= N) return;

    float sum = 0.0f;
    for (uint kk = 0; kk < K; kk++) {{
        sum += float(A[gid.y * K + kk]) * float(B[kk * N + gid.x]);
    }}
    C[gid.y * N + gid.x] = ({ty})sum;
}}
"#);

        Ok(CompiledKernel {
            code: shader.into_bytes(),
            entry_point: format!("warp_gemm_{m}x{n}x{k}"),
            grid: [n as u32, m as u32, 1],
            block: [tile as u32, tile as u32, 1],
            shared_mem_bytes: 0,
            description: format!("Metal GEMM {m}x{n}x{k} ({dtype})"),
        })
    }

    fn gen_rmsnorm(
        &self,
        hidden: usize,
        dtype: DType,
    ) -> Result<CompiledKernel, CodegenError> {
        let ty = metal_type(dtype);

        let shader = format!(
            r#"#include <metal_stdlib>
using namespace metal;

kernel void warp_rmsnorm_{hidden}(
    device {ty}* out [[buffer(0)]],
    device const {ty}* x [[buffer(1)]],
    device const {ty}* gamma [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {{
    threadgroup float shared_sum[32];
    uint H = {hidden};
    device const {ty}* x_row = x + row * H;
    device {ty}* out_row = out + row * H;

    float sum_sq = 0.0f;
    for (uint i = tid; i < H; i += 32) {{
        float v = float(x_row[i]);
        sum_sq += v * v;
    }}
    shared_sum[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {{
        float total = 0.0f;
        for (uint i = 0; i < 32; i++) total += shared_sum[i];
        shared_sum[0] = rsqrt(total / float(H) + eps);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms = shared_sum[0];

    for (uint i = tid; i < H; i += 32) {{
        out_row[i] = ({ty})(float(x_row[i]) * rms * float(gamma[i]));
    }}
}}
"#);

        Ok(CompiledKernel {
            code: shader.into_bytes(),
            entry_point: format!("warp_rmsnorm_{hidden}"),
            grid: [1, 1, 1], // set per-row at dispatch
            block: [32, 1, 1],
            shared_mem_bytes: 128,
            description: format!("Metal RMSNorm H={hidden} ({dtype})"),
        })
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn generate_kernel(
        &self,
        graph: &Graph,
        node: NodeId,
        _config: &KernelConfig,
    ) -> Result<CompiledKernel, CodegenError> {
        let n = graph.node(node);
        match &n.op {
            Op::Binary { op } => {
                let output = n.outputs[0];
                let info = graph.value(output);
                self.gen_binary(op, &info.shape, info.dtype)
            }
            Op::Activate { activation } => {
                let output = n.outputs[0];
                let info = graph.value(output);
                self.gen_activation(activation, &info.shape, info.dtype)
            }
            Op::Input { .. } | Op::Constant { .. } => {
                Err(CodegenError::UnsupportedOp("input/constant".into()))
            }
            other => Err(CodegenError::UnsupportedOp(format!("{:?}", std::mem::discriminant(other)))),
        }
    }

    fn estimate_cost(
        &self,
        graph: &Graph,
        node: NodeId,
        _input_shapes: &[Shape],
        dtype: DType,
    ) -> f64 {
        let n = graph.node(node);
        let output = n.outputs.first().copied();
        let numel = output
            .map(|o| graph.value(o).shape.numel().unwrap_or(0))
            .unwrap_or(0) as f64;

        match &n.op {
            Op::Binary { .. } | Op::Unary { .. } | Op::Activate { .. } => {
                let bytes = numel * dtype.byte_size() as f64 * 3.0;
                // M3 Ultra bandwidth: ~800 GB/s
                bytes / 800e3
            }
            Op::MatMul { .. } => {
                // M3 Ultra: ~28 TFLOPS FP16
                let flops = numel * 768.0 * 2.0;
                flops / 28e6
            }
            _ => 10.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_metal_add() {
        let backend = MetalBackend::new();
        let mut g = Graph::new();
        let a = g.add_input(shape![512], DType::F32, Some("a"));
        let b = g.add_input(shape![512], DType::F32, Some("b"));
        let (node_id, _) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[a, b],
            &[(shape![512], DType::F32)],
            None,
        );

        let config = KernelConfig::default();
        let kernel = backend.generate_kernel(&g, node_id, &config).unwrap();

        let shader = String::from_utf8(kernel.code).unwrap();
        assert!(shader.contains("metal_stdlib"));
        assert!(shader.contains("a[tid] + b[tid]"));
    }

    #[test]
    fn generate_metal_gelu() {
        let backend = MetalBackend::new();
        let mut g = Graph::new();
        let x = g.add_input(shape![1024], DType::F16, Some("x"));
        let (node_id, _) = g.add_node(
            Op::Activate { activation: Activation::GeluTanh },
            &[x],
            &[(shape![1024], DType::F16)],
            None,
        );

        let config = KernelConfig::default();
        let kernel = backend.generate_kernel(&g, node_id, &config).unwrap();

        let shader = String::from_utf8(kernel.code).unwrap();
        assert!(shader.contains("tanh"));
        assert!(shader.contains("half")); // F16 -> "half" in Metal
    }
}

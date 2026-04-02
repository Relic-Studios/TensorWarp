//! CUDA PTX code generation backend.
//!
//! Generates PTX assembly text that targets NVIDIA GPUs.
//! PTX is NVIDIA's virtual ISA — it gets JIT-compiled to actual GPU
//! machine code (SASS) by the CUDA driver at load time.
//!
//! Why PTX instead of SASS? PTX is forward-compatible across GPU
//! generations and readable for debugging. The JIT overhead is one-time
//! and cached by the CUDA driver.

use warp_ir::*;

use crate::backend::*;
use crate::kernel::{optimal_matmul_tiles, ptx_type};

/// CUDA PTX backend.
pub struct PtxBackend {
    /// Target SM version (e.g., 89 for Ada/4090, 90 for Hopper).
    pub sm_version: u32,
}

impl PtxBackend {
    pub fn new(sm_version: u32) -> Self {
        Self { sm_version }
    }

    /// Generate PTX for an elementwise binary operation.
    fn gen_binary(
        &self,
        op: &BinaryOp,
        shape: &Shape,
        dtype: DType,
    ) -> Result<CompiledKernel, CodegenError> {
        let n = shape.numel_static();
        let ty = ptx_type(dtype);
        let op_inst = match op {
            BinaryOp::Add => format!("add{ty}"),
            BinaryOp::Sub => format!("sub{ty}"),
            BinaryOp::Mul => format!("mul{ty}"),
            BinaryOp::Div => format!("div.rn{ty}"),
            _ => return Err(CodegenError::UnsupportedOp(format!("binary {:?}", op))),
        };

        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;

        let ptx = format!(
            r#".version 8.0
.target sm_{sm}
.address_size 64

.visible .entry warp_binary_{n}(
    .param .u64 out,
    .param .u64 a,
    .param .u64 b
)
{{
    .reg .u64 %rd<8>;
    .reg {ty} %f<4>;
    .reg .u32 %r<4>;
    .reg .pred %p<2>;

    // Thread index
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mad.lo.u32 %r2, %r1, {block_size}, %r0;

    // Bounds check
    setp.ge.u32 %p0, %r2, {n};
    @%p0 bra EXIT;

    // Load params
    ld.param.u64 %rd0, [out];
    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];

    // Offset
    mul.wide.u32 %rd3, %r2, {elem_size};
    add.u64 %rd4, %rd1, %rd3;
    add.u64 %rd5, %rd2, %rd3;
    add.u64 %rd6, %rd0, %rd3;

    // Load, compute, store
    ld.global{ty} %f0, [%rd4];
    ld.global{ty} %f1, [%rd5];
    {op_inst} %f2, %f0, %f1;
    st.global{ty} [%rd6], %f2;

EXIT:
    ret;
}}
"#,
            sm = self.sm_version,
            n = n,
            block_size = block_size,
            elem_size = dtype.byte_size(),
            ty = ty,
            op_inst = op_inst,
        );

        Ok(CompiledKernel {
            code: ptx.into_bytes(),
            entry_point: format!("warp_binary_{n}"),
            grid: [grid_size as u32, 1, 1],
            block: [block_size as u32, 1, 1],
            shared_mem_bytes: 0,
            description: format!("binary {op:?} on {n} elements ({dtype})"),
        })
    }

    /// Generate PTX for an activation function.
    fn gen_activation(
        &self,
        activation: &Activation,
        shape: &Shape,
        dtype: DType,
    ) -> Result<CompiledKernel, CodegenError> {
        let n = shape.numel_static();
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;

        // For now, generate F32 compute kernels (promote from F16 if needed)
        let compute_body = match activation {
            Activation::Relu => "max.f32 %f1, %f0, 0f00000000;".to_string(),
            Activation::Gelu | Activation::GeluTanh => {
                // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                r#"mul.f32 %f1, %f0, %f0;           // x²
    mul.f32 %f1, %f1, %f0;           // x³
    mul.f32 %f1, %f1, 0f3D372713;    // 0.044715 * x³
    add.f32 %f1, %f0, %f1;           // x + 0.044715 * x³
    mul.f32 %f1, %f1, 0f3F4C422A;    // sqrt(2/π) * (...)
    // tanh approximation via ex and e-x
    mul.f32 %f2, %f1, 0fC0000000;    // -2x
    ex2.approx.f32 %f3, %f2;         // e^(-2x) approx
    add.f32 %f2, %f3, 0f3F800000;    // 1 + e^(-2x)
    rcp.approx.f32 %f2, %f2;         // 1/(1+e^(-2x))
    mul.f32 %f2, %f2, 0f40000000;    // 2/(1+e^(-2x))
    sub.f32 %f1, %f2, 0f3F800000;    // tanh ≈ 2/(1+e^(-2x)) - 1
    add.f32 %f1, %f1, 0f3F800000;    // 1 + tanh(...)
    mul.f32 %f1, %f1, 0f3F000000;    // 0.5 * (1 + tanh(...))
    mul.f32 %f1, %f0, %f1;           // x * 0.5 * (1 + tanh(...))"#.to_string()
            }
            Activation::Silu => {
                // SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))
                r#"neg.f32 %f1, %f0;                // -x
    ex2.approx.f32 %f1, %f1;         // e^(-x) approx
    add.f32 %f1, %f1, 0f3F800000;    // 1 + e^(-x)
    rcp.approx.f32 %f1, %f1;         // sigmoid = 1/(1+e^(-x))
    mul.f32 %f1, %f0, %f1;           // x * sigmoid(x)"#.to_string()
            }
            _ => return Err(CodegenError::UnsupportedOp(format!("activation {:?}", activation))),
        };

        let ptx = format!(
            r#".version 8.0
.target sm_{sm}
.address_size 64

.visible .entry warp_activate_{n}(
    .param .u64 out,
    .param .u64 input
)
{{
    .reg .u64 %rd<6>;
    .reg .f32 %f<8>;
    .reg .u32 %r<4>;
    .reg .pred %p<2>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mad.lo.u32 %r2, %r1, {block_size}, %r0;

    setp.ge.u32 %p0, %r2, {n};
    @%p0 bra EXIT;

    ld.param.u64 %rd0, [out];
    ld.param.u64 %rd1, [input];

    mul.wide.u32 %rd2, %r2, 4;
    add.u64 %rd3, %rd1, %rd2;
    add.u64 %rd4, %rd0, %rd2;

    ld.global.f32 %f0, [%rd3];
    {compute_body}
    st.global.f32 [%rd4], %f1;

EXIT:
    ret;
}}
"#,
            sm = self.sm_version,
            n = n,
            block_size = block_size,
            compute_body = compute_body,
        );

        Ok(CompiledKernel {
            code: ptx.into_bytes(),
            entry_point: format!("warp_activate_{n}"),
            grid: [grid_size as u32, 1, 1],
            block: [block_size as u32, 1, 1],
            shared_mem_bytes: 0,
            description: format!("{:?} activation on {n} elements", activation),
        })
    }
}

impl Backend for PtxBackend {
    fn name(&self) -> &str {
        "cuda-ptx"
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
        match &n.op {
            // Elementwise: ~memory bound, estimate by bandwidth
            Op::Binary { .. } | Op::Unary { .. } | Op::Activate { .. } => {
                let output = n.outputs[0];
                let numel = graph.value(output).shape.numel().unwrap_or(0) as f64;
                let bytes = numel * dtype.byte_size() as f64 * 3.0; // read 2, write 1
                // 4090 bandwidth: ~1 TB/s = 1e12 B/s → 1e6 B/μs
                bytes / 1e6
            }
            // MatMul: ~compute bound
            Op::MatMul { .. } | Op::FusedMatMulBias { .. } | Op::FusedMatMulBiasAct { .. } => {
                let output = n.outputs[0];
                let shape = &graph.value(output).shape;
                if let (Some(m), Some(n_dim)) = (
                    shape.dim(0).static_val(),
                    shape.dim(1).static_val(),
                ) {
                    // Rough estimate: 2*M*N*K FLOPs / peak TFLOPS
                    // 4090: ~330 TFLOPS FP16 → 330e6 MFLOP/μs
                    let flops = 2.0 * m as f64 * n_dim as f64 * 768.0; // assume K=768
                    flops / 330e6
                } else {
                    100.0 // unknown, assume 100μs
                }
            }
            Op::Attention { num_heads, head_dim, .. } => {
                // FlashAttention-style: O(N² * d) but memory-efficient
                let output = n.outputs[0];
                let numel = graph.value(output).shape.numel().unwrap_or(0) as f64;
                numel * *num_heads as f64 * *head_dim as f64 / 330e6
            }
            _ => 10.0, // default estimate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_binary_add() {
        let backend = PtxBackend::new(89);
        let mut g = Graph::new();
        let a = g.add_input(shape![1024], DType::F32, Some("a"));
        let b = g.add_input(shape![1024], DType::F32, Some("b"));
        let (node_id, _) = g.add_node(
            Op::Binary { op: BinaryOp::Add },
            &[a, b],
            &[(shape![1024], DType::F32)],
            None,
        );

        let config = KernelConfig::default();
        let kernel = backend.generate_kernel(&g, node_id, &config).unwrap();

        let ptx = String::from_utf8(kernel.code).unwrap();
        assert!(ptx.contains("sm_89"));
        assert!(ptx.contains("add.f32"));
        assert_eq!(kernel.grid, [4, 1, 1]); // 1024/256 = 4 blocks
        assert_eq!(kernel.block, [256, 1, 1]);
    }

    #[test]
    fn generate_gelu_activation() {
        let backend = PtxBackend::new(89);
        let mut g = Graph::new();
        let x = g.add_input(shape![2048], DType::F32, Some("x"));
        let (node_id, _) = g.add_node(
            Op::Activate { activation: Activation::GeluTanh },
            &[x],
            &[(shape![2048], DType::F32)],
            None,
        );

        let config = KernelConfig::default();
        let kernel = backend.generate_kernel(&g, node_id, &config).unwrap();

        let ptx = String::from_utf8(kernel.code).unwrap();
        assert!(ptx.contains("ex2.approx")); // tanh uses exp
        assert_eq!(kernel.grid, [8, 1, 1]); // 2048/256
    }
}

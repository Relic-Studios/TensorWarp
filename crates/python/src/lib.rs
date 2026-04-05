use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use warp_ir::{DType, Shape};
use warp_kernels::conv::Conv2dParams;
use warp_kernels::device::WarpDevice;
use warp_kernels::engine::Engine;
use warp_kernels::tensor::GpuTensor;
use warp_loader::OnnxModel;

// ── Helpers ──────────────────────────────────────────────────

fn to_pyerr(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// ── PyEngine ─────────────────────────────────────────────────

/// GPU inference engine backed by TensorWarp.
#[pyclass(name = "Engine")]
struct PyEngine {
    inner: Engine,
}

#[pymethods]
impl PyEngine {
    /// Create a new Engine on the given GPU device.
    ///
    /// Args:
    ///     device: GPU ordinal (default 0).
    ///     cache_dir: Optional path to persist compiled kernels.
    #[new]
    #[pyo3(signature = (device=0, cache_dir=None))]
    fn new(device: usize, cache_dir: Option<String>) -> PyResult<Self> {
        let inner = match cache_dir {
            Some(dir) => Engine::with_cache_dir(device, dir).map_err(to_pyerr)?,
            None => Engine::new(device).map_err(to_pyerr)?,
        };
        Ok(Self { inner })
    }

    /// Return a human-readable summary of the engine state.
    fn summary(&self) -> String {
        self.inner.summary()
    }

    /// Pre-compile commonly used kernels to avoid first-call JIT latency.
    fn warmup(&self) -> PyResult<()> {
        self.inner.warmup().map_err(to_pyerr)
    }

    /// Block until all GPU work on this engine's stream has completed.
    fn synchronize(&self) -> PyResult<()> {
        self.inner.synchronize().map_err(to_pyerr)
    }

    /// Run Conv2D: output = conv2d(input, weight, bias).
    ///
    /// Args:
    ///     input: Input tensor [N, C_in, H, W].
    ///     weight: Weight tensor [C_out, C_in, kH, kW].
    ///     bias: Optional bias tensor [C_out], or None.
    ///     kh, kw: Kernel height and width.
    ///     sh, sw: Stride height and width.
    ///     ph, pw: Padding height and width.
    ///
    /// Returns:
    ///     Output Tensor [N, C_out, out_H, out_W].
    #[pyo3(signature = (input, weight, bias=None, kh=3, kw=3, sh=1, sw=1, ph=0, pw=0))]
    fn conv2d(
        &self,
        input: &PyTensor,
        weight: &PyTensor,
        bias: Option<&PyTensor>,
        kh: u32, kw: u32,
        sh: u32, sw: u32,
        ph: u32, pw: u32,
    ) -> PyResult<PyTensor> {
        // Infer channels from weight shape [C_out, C_in, kH, kW]
        if weight.shape_vec.len() != 4 {
            return Err(PyRuntimeError::new_err("weight must be 4D [C_out, C_in, kH, kW]"));
        }
        if input.shape_vec.len() != 4 {
            return Err(PyRuntimeError::new_err("input must be 4D [N, C_in, H, W]"));
        }
        let out_channels = weight.shape_vec[0] as u32;
        let in_channels = weight.shape_vec[1] as u32;
        let h = input.shape_vec[2] as u32;
        let w = input.shape_vec[3] as u32;
        let n = input.shape_vec[0];

        let params = Conv2dParams {
            in_channels,
            out_channels,
            kernel_h: kh,
            kernel_w: kw,
            stride_h: sh,
            stride_w: sw,
            padding_h: ph,
            padding_w: pw,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        };

        let out_h = ((h + 2 * ph - kh) / sh + 1) as usize;
        let out_w = ((w + 2 * pw - kw) / sw + 1) as usize;
        let out_shape_vec = vec![n, out_channels as usize, out_h, out_w];
        let out_shape = Shape::from_static(&out_shape_vec);
        let mut out = GpuTensor::<f32>::zeros(&self.inner.device, out_shape, DType::F32)
            .map_err(to_pyerr)?;

        let bias_ref = bias.map(|b| &b.inner);
        self.inner.conv2d(&input.inner, &weight.inner, bias_ref, &mut out, &params, h, w)
            .map_err(to_pyerr)?;

        Ok(PyTensor { shape_vec: out_shape_vec, inner: out })
    }

    /// Run RMSNorm: out = rmsnorm(x, gamma).
    ///
    /// Args:
    ///     x: Input tensor.
    ///     gamma: Scale parameter tensor [hidden].
    ///     hidden: Hidden dimension size.
    ///     eps: Epsilon for numerical stability.
    ///
    /// Returns:
    ///     Normalized Tensor.
    #[pyo3(signature = (x, gamma, hidden, eps=1e-5))]
    fn rmsnorm(&self, x: &PyTensor, gamma: &PyTensor, hidden: u32, eps: f32) -> PyResult<PyTensor> {
        let out_shape = Shape::from_static(&x.shape_vec);
        let mut out = GpuTensor::<f32>::zeros(&self.inner.device, out_shape, DType::F32)
            .map_err(to_pyerr)?;
        self.inner.rmsnorm(&x.inner, &gamma.inner, &mut out, hidden, eps)
            .map_err(to_pyerr)?;
        Ok(PyTensor { shape_vec: x.shape_vec.clone(), inner: out })
    }

    /// Run Softmax along the last dimension.
    ///
    /// Args:
    ///     x: Input tensor of shape [rows, cols].
    ///
    /// Returns:
    ///     Tensor with softmax applied.
    fn softmax(&self, x: &PyTensor) -> PyResult<PyTensor> {
        if x.shape_vec.len() < 1 {
            return Err(PyRuntimeError::new_err("tensor must have at least 1 dimension"));
        }
        let cols = *x.shape_vec.last().unwrap() as u32;
        let rows = (x.inner.numel / cols as usize) as u32;
        let out_shape = Shape::from_static(&x.shape_vec);
        let mut out = GpuTensor::<f32>::zeros(&self.inner.device, out_shape, DType::F32)
            .map_err(to_pyerr)?;
        warp_kernels::sampling::softmax(&self.inner.cache, &self.inner.device,
                                         &x.inner, &mut out, rows, cols)
            .map_err(to_pyerr)?;
        Ok(PyTensor { shape_vec: x.shape_vec.clone(), inner: out })
    }
}

// ── PyTensor ─────────────────────────────────────────────────

/// A GPU tensor backed by device memory.
#[pyclass(name = "Tensor")]
struct PyTensor {
    inner: GpuTensor<f32>,
    shape_vec: Vec<usize>,
}

#[pymethods]
impl PyTensor {
    /// Upload a flat list of f32 values with the given shape to the GPU.
    ///
    /// Args:
    ///     engine: The Engine whose device will own this tensor.
    ///     data: Flat list of f32 values.
    ///     shape: Tuple describing the tensor dimensions.
    #[staticmethod]
    #[pyo3(signature = (engine, data, shape))]
    fn from_numpy(engine: &PyEngine, data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(PyRuntimeError::new_err(format!(
                "data length {} does not match shape {:?} (expected {})",
                data.len(), shape, expected
            )));
        }
        let ir_shape = Shape::from_static(&shape);
        let gpu = GpuTensor::<f32>::from_host(&engine.inner.device, &data, ir_shape, DType::F32)
            .map_err(to_pyerr)?;
        Ok(Self {
            inner: gpu,
            shape_vec: shape,
        })
    }

    /// Download tensor data from GPU back to a flat Python list of floats.
    fn to_numpy(&self, engine: &PyEngine) -> PyResult<Vec<f32>> {
        self.inner.to_host(&engine.inner.device).map_err(to_pyerr)
    }

    /// Tensor shape as a tuple.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape_vec.clone()
    }

    /// Total number of elements.
    #[getter]
    fn numel(&self) -> usize {
        self.inner.numel
    }
}

// ── PyOnnxModel ─────────────────────────────────────────────

/// A loaded ONNX model.
#[pyclass(name = "OnnxModel")]
struct PyOnnxModel {
    inner: OnnxModel,
}

#[pymethods]
impl PyOnnxModel {
    /// Load an ONNX model from a file path.
    ///
    /// Args:
    ///     path: Path to the .onnx file.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = OnnxModel::load(path).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Return a human-readable summary of the model.
    fn summary(&self) -> String {
        self.inner.summary()
    }

    /// Number of model inputs.
    fn num_inputs(&self) -> usize {
        self.inner.inputs.len()
    }

    /// Number of model outputs.
    fn num_outputs(&self) -> usize {
        self.inner.outputs.len()
    }

    /// List of input tensor names.
    fn input_names(&self) -> Vec<String> {
        self.inner.inputs.iter().map(|io| io.name.clone()).collect()
    }
}

// ── Free functions (GEMM, ReLU) ──────────────────────────────

/// Perform f32 GEMM: C = A @ B.
///
/// Args:
///     engine: The Engine to run on.
///     a: Input tensor A of shape [m, k].
///     b: Input tensor B of shape [k, n].
///     m, n, k: Matrix dimensions.
///
/// Returns:
///     New Tensor of shape [m, n].
#[pyfunction]
fn gemm_f32(engine: &PyEngine, a: &PyTensor, b: &PyTensor, m: u32, n: u32, k: u32) -> PyResult<PyTensor> {
    let out_shape = Shape::from_static(&[m as usize, n as usize]);
    let mut c = GpuTensor::<f32>::zeros(&engine.inner.device, out_shape, DType::F32)
        .map_err(to_pyerr)?;
    engine.inner.gemm_f32(&a.inner, &b.inner, &mut c, m, n, k)
        .map_err(to_pyerr)?;
    Ok(PyTensor {
        shape_vec: vec![m as usize, n as usize],
        inner: c,
    })
}

/// Apply ReLU element-wise: out = max(0, x).
///
/// Args:
///     engine: The Engine to run on.
///     x: Input tensor.
///
/// Returns:
///     New Tensor with ReLU applied.
#[pyfunction]
fn relu(engine: &PyEngine, x: &PyTensor) -> PyResult<PyTensor> {
    let out_shape = Shape::from_static(&x.shape_vec);
    let mut out = GpuTensor::<f32>::zeros(&engine.inner.device, out_shape, DType::F32)
        .map_err(to_pyerr)?;
    engine.inner.relu(&x.inner, &mut out).map_err(to_pyerr)?;
    Ok(PyTensor {
        shape_vec: x.shape_vec.clone(),
        inner: out,
    })
}

/// Return the number of CUDA-capable GPUs detected.
#[pyfunction]
fn device_count() -> PyResult<usize> {
    WarpDevice::device_count().map_err(to_pyerr)
}

/// Return the TensorWarp version string.
#[pyfunction]
fn version() -> &'static str {
    "0.1.0"
}

/// Demo token generation (for testing/demo).
///
/// Creates a toy transformer and generates tokens from prompt IDs.
/// This is NOT a real LLM — it's a smoke test for the generation pipeline.
///
/// Args:
///     engine: The Engine to run on.
///     prompt_ids: List of input token IDs.
///     max_tokens: Maximum number of tokens to generate.
///     temperature: Sampling temperature.
///
/// Returns:
///     List of generated token IDs.
#[pyfunction]
#[pyo3(signature = (engine, prompt_ids, max_tokens=32, temperature=1.0))]
fn generate(
    engine: &PyEngine,
    prompt_ids: Vec<i32>,
    max_tokens: usize,
    temperature: f32,
) -> PyResult<Vec<i32>> {
    use warp_kernels::generate::GenerateConfig;
    use warp_kernels::transformer::{TransformerConfig, TransformerBlockWeights};
    use warp_kernels::cache::KernelCache;
    use warp_kernels::mem_pool::GpuMemPool;
    use warp_kernels::generate::GenerationEngine;

    // Build a tiny transformer for demo purposes
    let config = TransformerConfig::tiny();
    let dev = &engine.inner.device;
    let h = config.hidden_size as usize;
    let vocab_size = 256u32;

    let make_zeros = |rows: usize, cols: usize| -> PyResult<GpuTensor<f32>> {
        GpuTensor::<f32>::zeros(dev,
            Shape::from_static(&[rows, cols]), DType::F32).map_err(to_pyerr)
    };
    let make_zeros_1d = |len: usize| -> PyResult<GpuTensor<f32>> {
        GpuTensor::<f32>::zeros(dev,
            Shape::from_static(&[len]), DType::F32).map_err(to_pyerr)
    };

    let ffn_dim = config.ffn_dim as usize;
    let block = TransformerBlockWeights {
        attn_norm: make_zeros_1d(h)?,
        wq: make_zeros(h, h)?,
        wk: make_zeros(h, h)?,
        wv: make_zeros(h, h)?,
        wo: make_zeros(h, h)?,
        ffn_norm: make_zeros_1d(h)?,
        w_gate: make_zeros(h, ffn_dim)?,
        w_up: make_zeros(h, ffn_dim)?,
        w_down: make_zeros(ffn_dim, h)?,
        bq: None,
        bk: None,
        bv: None,
        wqkv: None,
        bqkv: None,
        w_gate_up: None,
    };

    let tok_embed_data = vec![0.01f32; vocab_size as usize * h];
    let tok_embed = GpuTensor::from_host(dev, &tok_embed_data,
        Shape::from_static(&[vocab_size as usize, h]), DType::F32).map_err(to_pyerr)?;
    let final_norm_data = vec![1.0f32; h];
    let final_norm = GpuTensor::from_host(dev, &final_norm_data,
        Shape::from_static(&[h]), DType::F32).map_err(to_pyerr)?;
    let lm_head = make_zeros(h, vocab_size as usize)?;

    let gen_engine = GenerationEngine {
        config,
        vocab_size,
        embed_tokens: tok_embed,
        layers: vec![block],
        final_norm,
        lm_head,
        cache: KernelCache::new(),
        pool: GpuMemPool::new(),
    };

    let gen_config = GenerateConfig {
        max_tokens,
        temperature,
        eos_token_id: None,
        greedy: temperature < 0.01,
        ..Default::default()
    };

    let tokens = gen_engine.generate(dev, &prompt_ids, &gen_config).map_err(to_pyerr)?;
    Ok(tokens)
}

/// Get memory pool statistics as a human-readable string.
///
/// Args:
///     engine: The Engine to query.
///
/// Returns:
///     String with pool statistics.
#[pyfunction]
fn pool_stats(engine: &PyEngine) -> String {
    format!("{}", engine.inner.pool.stats())
}

// ── Module ───────────────────────────────────────────────────

/// TensorWarp — a self-optimizing GPU inference engine.
#[pymodule]
fn tensorwarp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyOnnxModel>()?;
    m.add_function(wrap_pyfunction!(gemm_f32, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(generate, m)?)?;
    m.add_function(wrap_pyfunction!(pool_stats, m)?)?;
    m.add_function(wrap_pyfunction!(device_count, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

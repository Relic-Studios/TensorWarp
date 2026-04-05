"""
TensorWarp — A self-optimizing GPU inference engine.

Faster than TensorRT with automatic fusion, W4A16 quantization,
and 130+ ONNX ops.

Usage:
    import tensorwarp as tw

    # Create an engine and run ops directly
    engine = tw.Engine(device=0)
    engine.warmup()

    a = tw.Tensor.from_numpy(engine, [1.0]*1024, [32, 32])
    b = tw.Tensor.from_numpy(engine, [1.0]*1024, [32, 32])
    c = tw.gemm_f32(engine, a, b, 32, 32, 32)
    result = c.to_numpy(engine)
"""

__version__ = "0.1.0"

try:
    from .tensorwarp import *  # native PyO3 bindings
    _NATIVE = True
except ImportError:
    _NATIVE = False

    # Stub fallback so users get a clear error message
    class Engine:
        """GPU inference engine (stub — native bindings not compiled)."""
        def __init__(self, device=0, cache_dir=None):
            raise RuntimeError(
                "TensorWarp native bindings not found. "
                "Build with: cd crates/python && maturin develop --release"
            )

    class Tensor:
        """GPU tensor (stub — native bindings not compiled)."""
        @staticmethod
        def from_numpy(engine, data, shape):
            raise RuntimeError(
                "TensorWarp native bindings not found. "
                "Build with: cd crates/python && maturin develop --release"
            )

    def gemm_f32(engine, a, b, m, n, k):
        raise RuntimeError("TensorWarp native bindings not found.")

    def relu(engine, x):
        raise RuntimeError("TensorWarp native bindings not found.")

    def device_count():
        raise RuntimeError("TensorWarp native bindings not found.")

    def version():
        return __version__

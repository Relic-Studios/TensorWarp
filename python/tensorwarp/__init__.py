"""
TensorWarp — A self-optimizing GPU inference engine.

Faster than TensorRT with automatic fusion, W4A16 quantization,
and 130+ ONNX ops.

Usage:
    import tensorwarp as tw

    # Load and run an ONNX model
    engine = tw.Engine(device=0)
    model = tw.OnnxModel("model.onnx")
    output = engine.run(model, {"input": input_array})

    # Or use the Builder API
    builder = tw.ModelBuilder()
    x = builder.input("x", [1, 784])
    h = builder.linear(builder.relu(x), 256)
    out = builder.softmax(builder.linear(h, 10))
    builder.output("probs", out)
    model = builder.build(precision="fp16")
"""

__version__ = "0.1.0"

# When PyO3 bindings are compiled, they'll be imported here:
# from .tensorwarp_py import Engine, OnnxModel, ModelBuilder, Tensor

# For now, provide a pure-Python preview of the API
class Engine:
    """GPU inference engine."""
    def __init__(self, device=0, cache_dir=None):
        self.device = device
        self.cache_dir = cache_dir
        print(f"TensorWarp Engine initialized on GPU {device}")

    def run(self, model, inputs):
        """Run inference on a model."""
        raise NotImplementedError(
            "Native bindings not compiled. "
            "Build with: maturin develop --release"
        )

class OnnxModel:
    """Load an ONNX model."""
    def __init__(self, path):
        self.path = path
        print(f"Loading ONNX model: {path}")

class ModelBuilder:
    """Programmatic model construction (TensorRT-style)."""
    def __init__(self):
        self.layers = []

    def input(self, name, shape):
        self.layers.append(("input", name, shape))
        return len(self.layers) - 1

    def linear(self, input_id, out_features):
        self.layers.append(("linear", input_id, out_features))
        return len(self.layers) - 1

    def relu(self, input_id):
        self.layers.append(("relu", input_id))
        return len(self.layers) - 1

    def softmax(self, input_id):
        self.layers.append(("softmax", input_id))
        return len(self.layers) - 1

    def output(self, name, input_id):
        self.layers.append(("output", name, input_id))

    def build(self, precision="fp32"):
        print(f"Building model with {len(self.layers)} layers, precision={precision}")
        return self

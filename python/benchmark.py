#!/usr/bin/env python3
"""
TensorWarp Benchmark Script

Compares TensorWarp inference performance against baselines.
Run: python benchmark.py

Requires: pip install tensorwarp numpy
Optional: pip install onnxruntime-gpu (for comparison)
"""

import subprocess
import sys
import time
import os

def run_tensorwarp_bench():
    """Run TensorWarp's built-in benchmarks via CLI."""
    print("=" * 60)
    print("  TensorWarp Inference Benchmarks")
    print("=" * 60)
    print()

    # Run the Rust benchmark suite
    tensorwarp_path = os.path.join(os.path.dirname(__file__), "..")
    result = subprocess.run(
        ["cargo", "run", "--release", "--", "bench"],
        cwd=tensorwarp_path,
        capture_output=True, text=True, timeout=300
    )

    if result.returncode == 0:
        print(result.stdout)
    else:
        print("TensorWarp bench failed:")
        print(result.stderr)

def compare_with_onnxruntime():
    """Compare against ONNX Runtime if available."""
    try:
        import onnxruntime as ort
        import numpy as np

        print("\n" + "=" * 60)
        print("  ONNX Runtime Comparison")
        print("=" * 60)

        providers = ort.get_available_providers()
        print(f"  Providers: {providers}")

        # Create a simple test model
        try:
            import onnx
            from onnx import helper, TensorProto

            # Build: MatMul(1024×1024 × 1024×1024)
            X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1024, 1024])
            Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1024, 1024])
            Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1024, 1024])

            matmul = helper.make_node('MatMul', ['X', 'Y'], ['Z'])
            graph = helper.make_graph([matmul], 'test', [X, Y], [Z])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

            model_path = "/tmp/test_gemm.onnx"
            onnx.save(model, model_path)

            session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

            x = np.random.randn(1024, 1024).astype(np.float32)
            y = np.random.randn(1024, 1024).astype(np.float32)

            # Warmup
            session.run(None, {'X': x, 'Y': y})

            # Benchmark
            iters = 100
            start = time.time()
            for _ in range(iters):
                session.run(None, {'X': x, 'Y': y})
            elapsed = time.time() - start

            ms = elapsed * 1000.0 / iters
            flops = 2 * 1024 * 1024 * 1024
            tflops = flops * iters / elapsed / 1e12

            print(f"  ONNX Runtime GEMM 1024³: {ms:.3f}ms ({tflops:.2f} TFLOPS)")
            print(f"  Provider: {session.get_providers()[0]}")

        except ImportError:
            print("  (onnx package not installed — skipping model comparison)")

    except ImportError:
        print("\n  ONNX Runtime not installed. Install with:")
        print("    pip install onnxruntime-gpu")

def main():
    print("TensorWarp Benchmark Suite v0.1.0")
    print()

    run_tensorwarp_bench()
    compare_with_onnxruntime()

    print("\n" + "=" * 60)
    print("  Benchmark complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
Hugging Face model loader for TensorWarp.

Downloads and converts Hugging Face models to TensorWarp format.

Usage:
    from tensorwarp.hf_loader import load_from_hub

    # Load a model from HF Hub
    model = load_from_hub("microsoft/resnet-18", export_onnx=True)

    # Or load a local ONNX file
    model = load_onnx("path/to/model.onnx")
"""

import os
import subprocess
import sys


def load_from_hub(model_id, export_onnx=True, cache_dir=None):
    """
    Load a model from Hugging Face Hub.

    Args:
        model_id: HF model identifier (e.g., "microsoft/resnet-18")
        export_onnx: Whether to export to ONNX format
        cache_dir: Directory to cache downloaded models

    Returns:
        Path to the ONNX model file
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/tensorwarp")
    os.makedirs(cache_dir, exist_ok=True)

    model_dir = os.path.join(cache_dir, model_id.replace("/", "--"))
    onnx_path = os.path.join(model_dir, "model.onnx")

    if os.path.exists(onnx_path):
        print(f"Using cached ONNX model: {onnx_path}")
        return onnx_path

    if export_onnx:
        print(f"Exporting {model_id} to ONNX...")
        os.makedirs(model_dir, exist_ok=True)

        try:
            # Try using optimum for export
            subprocess.run([
                sys.executable, "-m", "optimum.exporters.onnx",
                "--model", model_id,
                model_dir
            ], check=True)
            print(f"Exported to: {onnx_path}")
            return onnx_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("optimum not installed. Install with: pip install optimum[exporters]")
            print("Or export manually with torch.onnx.export()")

    return None


def load_onnx(path):
    """Load a local ONNX model and inspect it with TensorWarp."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX model not found: {path}")

    # Use TensorWarp CLI to inspect
    tensorwarp_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    result = subprocess.run(
        ["cargo", "run", "--release", "--", "onnx", path],
        cwd=tensorwarp_dir,
        capture_output=True, text=True
    )

    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error inspecting model: {result.stderr}")

    return path


def export_pytorch_to_onnx(model, dummy_input, output_path, opset=17):
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch nn.Module
        dummy_input: Example input tensor
        output_path: Where to save the .onnx file
        opset: ONNX opset version
    """
    try:
        import torch
        torch.onnx.export(
            model, dummy_input, output_path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
        )
        print(f"Exported PyTorch model to: {output_path}")
        return output_path
    except ImportError:
        print("PyTorch not installed. Install with: pip install torch")
        return None

"""
TensorWarp Python package setup.

For development (pure Python preview):
    pip install -e .

For native bindings (requires Rust + CUDA):
    pip install maturin
    maturin develop --release
"""

from setuptools import setup, find_packages

setup(
    name="tensorwarp",
    version="0.1.0",
    description="A self-optimizing GPU inference engine — faster than TensorRT",
    long_description=open("../README.md").read() if __import__("os").path.exists("../README.md") else "",
    long_description_content_type="text/markdown",
    author="Relic Studios",
    url="https://github.com/Relic-Studios/TensorWarp",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "dev": ["pytest", "onnx", "torch"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Rust",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

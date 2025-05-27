#!/usr/bin/env python3
"""Setup script for Qwen3 Optimizer Comparison Study."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
        "peft>=0.10.0",
        "evaluate>=0.4.1",
        "bitsandbytes>=0.43.0",
        "adabound>=0.0.5",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
    ]

setup(
    name="qwen3-optimizer-study",
    version="1.0.0",
    author="Qwen3 Optimizer Study Team",
    author_email="your-email@example.com",
    description="A comprehensive benchmarking framework comparing AdamW, SGD+Momentum, and AdaBound optimizers for fine-tuning Qwen3-8B on CommonsenseQA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/qwen3-optimizer-study",
    packages=find_packages(exclude=["tests", "experiments", "results", "data", "models"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "plotly>=5.17.0",
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
        ],
        "gpu": [
            "flash-attn>=2.5.0",
            "gpustat>=1.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qwen3-study=utils.analyze_results:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/qwen3-optimizer-study/issues",
        "Source": "https://github.com/your-username/qwen3-optimizer-study",
        "Documentation": "https://github.com/your-username/qwen3-optimizer-study/blob/main/README.md",
    },
    keywords=[
        "machine learning",
        "deep learning",
        "natural language processing",
        "transformers",
        "optimization",
        "fine-tuning",
        "benchmark",
        "qwen",
        "lora",
        "pytorch",
    ],
    include_package_data=True,
    zip_safe=False,
) 
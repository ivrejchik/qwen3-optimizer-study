# Qwen3 Optimizer Comparison Study

A state-of-the-art benchmarking framework comparing AdamW, SGD+Momentum, AdaBound, and a novel Hybrid Adam+SGD optimizer for fine-tuning Qwen2.5-7B on CommonsenseQA.

## 🎯 Overview

This repository implements a comprehensive six-phase pipeline to systematically compare different optimizers when fine-tuning large language models using LoRA (Low-Rank Adaptation). The study evaluates:

- **AdamW**: Adaptive moment estimation with weight decay (baseline)
- **SGD + Momentum**: Stochastic gradient descent with momentum
- **AdaBound**: Smooth transition from Adam to SGD
- **Hybrid Adam+SGD** (NEW): Novel optimizer dynamically blending Adam's adaptivity with SGD's stability

## ✨ New Features

- 🔥 **Hybrid Adam+SGD Optimizer**: Research-grade implementation combining best of both worlds
- 📊 **GPU Timeline Tracking**: Real-time GPU memory and utilization monitoring during training
- 🎨 **Interactive Visualizations**: Plotly-powered interactive charts (radar plots, GPU timelines, loss curves)
- 🤗 **HuggingFace Benchmark Format**: Auto-generated benchmark-compatible JSON outputs
- 📈 **Enhanced Analytics**: Comprehensive performance metrics with multiple visualization types

## 🚀 Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd qwen3-optimizer-study

# One-command execution
bash run_all.sh
```

## 📋 Requirements

### Hardware
- **Recommended**: 2 × A100 80GB or 4 × RTX 4090 24GB
- **Minimum**: 1 × RTX 3090 24GB (with adjusted batch sizes)

### Software
- Ubuntu 22.04 LTS (tested)
- CUDA 12.3+
- Python 3.11
- ~50GB free disk space

## 🔧 Installation

```bash
# Create conda environment
conda create -y -n qwen_optim python=3.11
conda activate qwen_optim

# Install dependencies
pip install -r requirements.txt

# Configure accelerate
accelerate config default

# Login to Hugging Face (required for model access)
huggingface-cli login
```

## 📊 Project Structure

```
qwen3-optimizer-study/
├── phases/                 # Modular pipeline scripts
│   ├── 0_env.sh           # Environment setup
│   ├── 1_data.py          # Data acquisition
│   ├── 2_train.py         # LoRA training (with GPU tracking)
│   ├── 3_merge.py         # Adapter merging
│   └── 4_eval.py          # Model evaluation (HF benchmark export)
├── experiments/           # Training outputs
│   ├── adamw/
│   ├── sgd/
│   ├── adabound/
│   └── hybrid/            # NEW: Hybrid optimizer results
├── data/                  # Dataset cache
├── models/                # Model cache
├── results/               # Evaluation outputs
│   └── analysis/          # Visualizations (PNG + HTML)
├── configs/               # Configuration files
│   └── training_config.yaml  # All optimizer configs
├── utils/                 # Helper utilities
│   ├── optimizers.py      # Optimizer registry
│   ├── hybrid_adam_sgd.py # NEW: Hybrid optimizer implementation
│   └── analyze_results.py # Enhanced with GPU timeline & radar plots
├── tests/                 # Unit tests (pytest)
└── notebooks/             # Analysis notebooks
```

## 🔄 Pipeline Phases

### Phase 0: Environment Setup
```bash
bash phases/0_env.sh
```

### Phase 1: Data & Model Acquisition
```bash
python phases/1_data.py
```
Downloads and caches CommonsenseQA dataset and Qwen3-8B model.

### Phase 2: LoRA Fine-tuning
```bash
python phases/2_train.py --optimizer adamw --output_dir ./experiments/adamw
python phases/2_train.py --optimizer sgd --output_dir ./experiments/sgd
python phases/2_train.py --optimizer adabound --output_dir ./experiments/adabound
python phases/2_train.py --optimizer hybrid --output_dir ./experiments/hybrid  # NEW!
```
Training now includes GPU tracking and saves metrics to `gpu_metrics.json`.

### Phase 3: Adapter Merging
```bash
python phases/3_merge.py
```

### Phase 4: Evaluation
```bash
python phases/4_eval.py
```

### Phase 5: Analysis & Visualization
```bash
python utils/analyze_results.py
```
Generates comprehensive visualizations including:
- Static plots (accuracy comparison, performance metrics, radar charts)
- Interactive HTML visualizations (GPU timeline, training loss comparison)
- HuggingFace benchmark-compatible JSON outputs

## 📈 Expected Results

The pipeline generates:
- **Accuracy scores** on CommonsenseQA validation set
- **Training metrics** (loss curves, memory usage, GPU utilization)
- **Inference speed** comparisons
- **Resource utilization** logs with GPU timeline tracking
- **Interactive visualizations** (HTML + PNG formats)
- **HuggingFace benchmark JSON** files for each optimizer

## 🔧 Configuration

Edit `configs/training_config.yaml` to adjust:
- Learning rates
- Batch sizes
- LoRA parameters
- Training epochs
- Optimizer-specific parameters (including hybrid transition settings)

## 🔬 Hybrid Adam+SGD Optimizer

The novel hybrid optimizer dynamically blends Adam and SGD behaviors:

**Key Features**:
- Starts with Adam-like adaptive learning (fast initial convergence)
- Gradually transitions to SGD-like momentum (stable final performance)
- Configurable transition schedule via `transition_steps`
- Configurable final blend ratio via `final_ratio`

**Configuration** (in `configs/training_config.yaml`):
```yaml
hybrid:
  lr: 1e-5
  beta1: 0.9           # Adam first moment decay
  beta2: 0.999         # Adam second moment decay
  momentum: 0.9        # SGD momentum factor
  transition_steps: 1000  # Steps to transition from Adam to SGD
  final_ratio: 0.1     # Final Adam ratio (0.1 = 10% Adam, 90% SGD)
```

**Theory**: Combines the fast early convergence of adaptive methods with the better generalization of momentum-based methods.

## 🎨 Customization

### Adding New Optimizers
1. Add optimizer to `utils/optimizers.py`
2. Update `phases/2_train.py` mapping
3. Run the pipeline

### Different Models
Update model paths in `configs/model_config.yaml`

## 📖 Reproducibility

All experiments use fixed seeds and deterministic operations. The complete pipeline can be reproduced with:

```bash
bash run_all.sh --seed 42
```

## 🐛 Troubleshooting

### Common Issues

**OOM Errors**:
```bash
# Reduce batch size
export TRAIN_BATCH_SIZE=4
export GRAD_ACCUM_STEPS=4
```

**Slow Training**:
```bash
# Enable mixed precision
export USE_BF16=true
export USE_FLASH_ATTN=true
```

**Model Download Issues**:
```bash
# Use mirror
export HF_ENDPOINT=https://hf-mirror.com
```

## 📚 References

- [Qwen3 Official Repository](https://github.com/QwenLM/Qwen)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [AdaBound Paper](https://arxiv.org/abs/1902.09843)
- [CommonsenseQA Dataset](https://arxiv.org/abs/1811.00937)

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: [your-email]

---

*Built with ❤️ for the ML research community* 
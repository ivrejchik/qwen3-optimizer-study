#!/usr/bin/env bash

# Phase 0: Environment Setup
# Sets up conda environment and installs all dependencies

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[SETUP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda not found. Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_status "Creating conda environment 'qwen_optim' with Python 3.11..."

# Create environment if it doesn't exist
if ! conda env list | grep -q "qwen_optim"; then
    conda create -y -n qwen_optim python=3.11
    print_status "Environment created successfully"
else
    print_warning "Environment 'qwen_optim' already exists, skipping creation"
fi

print_status "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate qwen_optim

print_status "Installing core dependencies..."

# Install PyTorch with CUDA support
print_status "Installing PyTorch with CUDA support..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other requirements
print_status "Installing additional dependencies..."
pip install --upgrade pip

# Install from requirements.txt
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
else
    print_warning "requirements.txt not found, installing core packages manually..."
    pip install transformers datasets accelerate peft evaluate bitsandbytes adabound
    pip install numpy pandas matplotlib seaborn jupyter
    pip install pyyaml tqdm wandb tensorboard
fi

print_status "Configuring accelerate..."
# Create default accelerate config
accelerate config default --config_file ~/.cache/huggingface/accelerate/default_config.yaml

print_status "Verifying installation..."

# Test PyTorch installation
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Test other critical packages
python -c "
try:
    import transformers, datasets, accelerate, peft, evaluate, bitsandbytes
    print('✓ All core packages imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
"

print_status "Environment setup completed successfully!"
echo ""
echo "To activate this environment in the future, run:"
echo "  conda activate qwen_optim"
echo ""
echo "Next steps:"
echo "1. Login to Hugging Face: huggingface-cli login"
echo "2. Run the full pipeline: bash run_all.sh" 
#!/usr/bin/env bash

# Qwen3 Optimizer Comparison Study - Master Pipeline
# Usage: bash run_all.sh [--seed SEED] [--skip-env] [--phase N]

set -e

# Default configurations
SEED=42
SKIP_ENV=false
PHASE_START=1
PHASE_END=5

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEED="$2"
            shift 2
            ;;
        --skip-env)
            SKIP_ENV=true
            shift
            ;;
        --phase)
            PHASE_START="$2"
            PHASE_END="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--seed SEED] [--skip-env] [--phase N]"
            echo "  --seed SEED     Set random seed (default: 42)"
            echo "  --skip-env      Skip environment setup"
            echo "  --phase N       Run only phase N (1-5)"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Export seed for all scripts
export PYTHONHASHSEED=$SEED

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found. Please install Miniconda or Anaconda."
        exit 1
    fi
}

# Check if we're in the right environment
check_environment() {
    # Check for conda environment
    if [[ -n "$CONDA_DEFAULT_ENV" ]] && [[ "$CONDA_DEFAULT_ENV" != "qwen_optim" ]] && [[ "$SKIP_ENV" == false ]]; then
        print_warning "Not in qwen_optim environment. Run 'conda activate qwen_optim' first."
        print_warning "Or use --skip-env flag if you're using a different environment."
        exit 1
    fi

    # Check for venv (common on macOS)
    if [[ -z "$CONDA_DEFAULT_ENV" ]] && [[ -n "$VIRTUAL_ENV" ]]; then
        print_success "Using Python virtual environment: $(basename $VIRTUAL_ENV)"
        return 0
    fi

    # If neither conda nor venv, warn but continue with --skip-env
    if [[ -z "$CONDA_DEFAULT_ENV" ]] && [[ -z "$VIRTUAL_ENV" ]] && [[ "$SKIP_ENV" == false ]]; then
        print_warning "No virtual environment detected. Use --skip-env to continue anyway."
        print_warning "Recommended: create a venv with 'python -m venv venv && source venv/bin/activate'"
        exit 1
    fi
}

# Log system information
log_system_info() {
    print_header "System Information"
    echo "Date: $(date)"
    echo "User: $(whoami)"
    echo "Hostname: $(hostname)"
    echo "OS: $(uname -s) $(uname -m)"
    echo "Python: $(python --version 2>&1 || echo 'Not found')"
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

    # Check for different accelerators
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
    echo "MPS available: $(python -c 'import torch; print(torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False)' 2>/dev/null || echo 'Unknown')"

    # Determine which device will be used
    echo "Device: $(python -c '
import torch
if torch.cuda.is_available():
    print(f\"CUDA ({torch.cuda.device_count()} GPUs)\")
elif hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available():
    print(\"MPS (Apple Silicon)\")
else:
    print(\"CPU\")
' 2>/dev/null || echo 'Unknown')"

    echo "Seed: $SEED"
    echo ""
}

# Phase 0: Environment Setup (optional)
phase0_environment() {
    if [[ "$SKIP_ENV" == true ]]; then
        print_warning "Skipping environment setup"
        return 0
    fi
    
    print_header "Phase 0: Environment Setup"
    check_conda
    
    if [[ -f "phases/0_env.sh" ]]; then
        bash phases/0_env.sh
        print_success "Environment setup completed"
    else
        print_warning "Environment setup script not found, skipping"
    fi
}

# Phase 1: Data and Model Acquisition
phase1_data() {
    print_header "Phase 1: Data & Model Acquisition"
    python phases/1_data.py --seed $SEED
    print_success "Data and model acquisition completed"
}

# Phase 2: LoRA Fine-tuning
phase2_training() {
    print_header "Phase 2: LoRA Fine-tuning"

    # Create experiment directories
    mkdir -p experiments/{adamw,sgd,adabound,hybrid}

    # Training jobs (including new hybrid optimizer)
    optimizers=("adamw" "sgd" "adabound" "hybrid")

    for optimizer in "${optimizers[@]}"; do
        echo -e "${YELLOW}Training with $optimizer optimizer...${NC}"
        python phases/2_train.py \
            --optimizer $optimizer \
            --output_dir ./experiments/$optimizer \
            --seed $SEED
        print_success "$optimizer training completed"
    done

    print_success "All training jobs completed"
}

# Phase 3: Adapter Merging
phase3_merging() {
    print_header "Phase 3: Adapter Merging"
    python phases/3_merge.py
    print_success "Adapter merging completed"
}

# Phase 4: Evaluation
phase4_evaluation() {
    print_header "Phase 4: Model Evaluation"
    python phases/4_eval.py --seed $SEED
    print_success "Model evaluation completed"
}

# Phase 5: Analysis and Reporting
phase5_analysis() {
    print_header "Phase 5: Analysis & Reporting"

    # Generate plots and analysis
    if [[ -f "utils/analyze_results.py" ]]; then
        python utils/analyze_results.py \
            --optimizers adamw sgd adabound hybrid \
            --experiments_dir ./experiments
        print_success "Analysis completed"
    fi

    # Show results summary
    if [[ -f "results/results.csv" ]]; then
        echo -e "${GREEN}Results Summary:${NC}"
        python -c "
import pandas as pd
df = pd.read_csv('results/results.csv')
print(df.to_string(index=False))
print(f'\nBest performing optimizer: {df.loc[df[\"accuracy\"].idxmax(), \"model\"]}')
print(f'Best accuracy: {df[\"accuracy\"].max():.4f}')
"
    fi

    # Display link to interactive visualizations
    echo -e "\n${GREEN}Interactive visualizations available in:${NC}"
    echo "  - results/analysis/interactive_radar.html"
    echo "  - results/analysis/gpu_timeline.html"
    echo "  - results/analysis/training_loss_comparison.html"
}

# Error handling
error_handler() {
    print_error "Pipeline failed at line $1"
    echo "Check the logs above for details."
    exit 1
}

trap 'error_handler $LINENO' ERR

# Main execution
main() {
    print_header "Qwen3 Optimizer Comparison Study"
    echo "Starting pipeline with seed: $SEED"
    echo "Phase range: $PHASE_START-$PHASE_END"
    echo ""
    
    log_system_info
    
    # Environment check
    if [[ "$SKIP_ENV" == false ]]; then
        check_environment
    fi
    
    # Execute phases
    for phase in $(seq $PHASE_START $PHASE_END); do
        case $phase in
            0) phase0_environment ;;
            1) phase1_data ;;
            2) phase2_training ;;
            3) phase3_merging ;;
            4) phase4_evaluation ;;
            5) phase5_analysis ;;
            *) print_error "Invalid phase: $phase" ;;
        esac
    done
    
    print_header "Pipeline Completed Successfully!"
    echo "Results available in: results/"
    echo "Models available in: experiments/"
    echo ""
    echo "Next steps:"
    echo "1. Review results.csv for accuracy comparisons"
    echo "2. Check notebooks/ for detailed analysis"
    echo "3. Upload results to Hugging Face Hub (optional)"
}

# Run main function
main "$@" 
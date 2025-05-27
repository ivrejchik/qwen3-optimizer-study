# Contributing to Qwen3 Optimizer Comparison Study

Thank you for your interest in contributing to this project! This guide will help you get started.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/qwen3-optimizer-study.git
   cd qwen3-optimizer-study
   ```
3. **Set up development environment**:
   ```bash
   bash phases/0_env.sh
   conda activate qwen_optim
   pip install -e ".[dev]"
   ```

## 🛠️ Development Setup

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (recommended)
- ~50GB free disk space
- Git

### Environment Setup
```bash
# Create development environment
conda create -n qwen_optim_dev python=3.11
conda activate qwen_optim_dev

# Install in development mode
pip install -e ".[dev,viz,gpu]"

# Set up pre-commit hooks (optional)
pre-commit install
```

## 📋 Types of Contributions

We welcome several types of contributions:

### 🐛 Bug Reports
- Use the GitHub issue tracker
- Include system information, error logs, and reproduction steps
- Check existing issues first

### ✨ Feature Requests
- Describe the feature and its use case
- Explain why it would be valuable
- Consider implementation complexity

### 🔧 Code Contributions
- New optimizers
- Performance improvements
- Documentation improvements
- Bug fixes
- Test coverage improvements

### 📚 Documentation
- README improvements
- Code comments
- Tutorials and examples
- Configuration guides

## 🔄 Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run basic tests
python -m pytest tests/

# Run specific phase
python phases/1_data.py --dataset-only

# Check code style
black --check .
isort --check-only .
flake8 .
```

### 4. Commit Changes
```bash
git add .
git commit -m "feat: add support for new optimizer"
# or
git commit -m "fix: resolve memory leak in evaluation"
```

#### Commit Message Convention
We follow conventional commits:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Test additions/modifications
- `chore:` - Maintenance tasks

### 5. Push and Create PR
```bash
git push origin your-branch-name
```

Then create a Pull Request on GitHub.

## 📝 Coding Standards

### Python Style
- Follow PEP 8
- Use Black for formatting: `black .`
- Use isort for imports: `isort .`
- Use type hints where possible
- Maximum line length: 100 characters

### Code Quality
- Write docstrings for all functions and classes
- Include type hints
- Add comments for complex logic
- Follow existing patterns in the codebase

### Example Function
```python
def evaluate_model(
    model_path: str,
    dataset_path: str,
    batch_size: int = 8
) -> Dict[str, float]:
    """
    Evaluate a model on the given dataset.
    
    Args:
        model_path: Path to the model directory
        dataset_path: Path to the dataset
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary containing evaluation metrics
        
    Raises:
        FileNotFoundError: If model or dataset not found
    """
    # Implementation here
    pass
```

## 🧪 Testing

### Running Tests
```bash
# All tests
python -m pytest

# Specific test file
python -m pytest tests/test_optimizers.py

# With coverage
python -m pytest --cov=utils --cov-report=html
```

### Writing Tests
- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock expensive operations (model loading, etc.)

### Test Example
```python
def test_optimizer_creation():
    """Test that optimizers can be created with default parameters."""
    from utils.optimizers import OptimizerConfig
    
    # Test each optimizer
    for optimizer_name in ["adamw", "sgd", "adabound"]:
        optimizer_class = OptimizerConfig.get_optimizer_class(optimizer_name)
        assert optimizer_class is not None
```

## 📂 Project Structure

```
qwen3-optimizer-study/
├── phases/           # Pipeline scripts (0-4)
├── utils/           # Utility functions
├── configs/         # Configuration files
├── notebooks/       # Jupyter notebooks
├── tests/          # Test files
├── experiments/    # Training outputs (gitignored)
├── results/        # Evaluation results (gitignored)
├── data/           # Dataset cache (gitignored)
└── models/         # Model cache (gitignored)
```

## 🎯 Specific Contribution Areas

### Adding New Optimizers

1. **Update `utils/optimizers.py`**:
   ```python
   OPTIMIZER_REGISTRY = {
       # ... existing optimizers
       "your_optimizer": {
           "class": YourOptimizer,
           "default_params": {...},
           "description": "Your optimizer description"
       }
   }
   ```

2. **Update training script** (`phases/2_train.py`)
3. **Add tests** for the new optimizer
4. **Update documentation**

### Adding New Evaluation Metrics

1. **Update `phases/4_eval.py`**
2. **Update analysis tools** (`utils/analyze_results.py`)
3. **Update visualizations**
4. **Add tests**

### Improving Performance

- Profile code to identify bottlenecks
- Optimize memory usage
- Improve GPU utilization
- Add caching where appropriate

## 🐛 Debugging Tips

### Common Issues
- **CUDA out of memory**: Reduce batch size or use gradient accumulation
- **Model download fails**: Check internet connection and HF token
- **Import errors**: Ensure all dependencies are installed

### Debugging Tools
- Use `python -m pdb` for debugging
- Add logging statements
- Check GPU memory with `nvidia-smi`
- Monitor with `htop` or `psutil`

## 📖 Documentation

### Updating README
- Keep it concise but comprehensive
- Include examples
- Update installation instructions if needed

### Code Documentation
- Use clear, descriptive docstrings
- Include examples in docstrings when helpful
- Document parameters and return values

### Configuration Documentation
- Document all configuration options
- Provide examples for common use cases
- Explain the impact of different settings

## 🔍 Review Process

### What We Look For
- Code quality and style
- Test coverage
- Documentation completeness
- Performance impact
- Backward compatibility

### Review Timeline
- Initial review within 3-5 days
- Follow-up reviews within 1-2 days
- Merge after approval and CI passes

## 🆘 Getting Help

- **Issues**: Create a GitHub issue for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Chat**: Join our Discord/Slack (if available)
- **Email**: Contact maintainers directly for sensitive issues

## 🏆 Recognition

Contributors will be:
- Listed in the CONTRIBUTORS.md file
- Mentioned in release notes
- Given appropriate credit in academic papers (if applicable)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Qwen3 Optimizer Comparison Study! 🎉 
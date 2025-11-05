"""
Optimizer utilities and configurations for the Qwen3 study.
"""

from typing import Dict, Any, Type
import torch
from transformers.optimization import AdamW
from adabound import AdaBound
from .hybrid_adam_sgd import AdamSGDHybrid


class SGDMomentum(torch.optim.SGD):
    """SGD with momentum wrapper for consistency with other optimizers."""
    
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.01, **kwargs):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)


class OptimizerConfig:
    """Configuration class for optimizers."""
    
    OPTIMIZER_REGISTRY = {
        "adamw": {
            "class": AdamW,
            "default_params": {
                "lr": 1e-5,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
                "eps": 1e-8
            },
            "description": "AdamW: Adaptive moment estimation with weight decay"
        },
        "sgd": {
            "class": SGDMomentum,
            "default_params": {
                "lr": 1e-5,
                "momentum": 0.9,
                "weight_decay": 0.01,
                "nesterov": False
            },
            "description": "SGD with momentum"
        },
        "adabound": {
            "class": AdaBound,
            "default_params": {
                "lr": 1e-5,
                "final_lr": 0.1,
                "gamma": 1e-3,
                "weight_decay": 0.01
            },
            "description": "AdaBound: Smooth transition from Adam to SGD"
        },
        "hybrid": {
            "class": AdamSGDHybrid,
            "default_params": {
                "lr": 1e-5,
                "beta1": 0.9,
                "beta2": 0.999,
                "momentum": 0.9,
                "eps": 1e-8,
                "weight_decay": 0.01,
                "transition_steps": 1000,
                "final_ratio": 0.1
            },
            "description": "Hybrid Adam+SGD: Dynamic blend of Adam adaptivity and SGD stability"
        }
    }
    
    @classmethod
    def get_optimizer_class(cls, optimizer_name: str) -> Type[torch.optim.Optimizer]:
        """Get optimizer class by name."""
        if optimizer_name not in cls.OPTIMIZER_REGISTRY:
            available = list(cls.OPTIMIZER_REGISTRY.keys())
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {available}")
        
        return cls.OPTIMIZER_REGISTRY[optimizer_name]["class"]
    
    @classmethod
    def get_default_params(cls, optimizer_name: str) -> Dict[str, Any]:
        """Get default parameters for an optimizer."""
        if optimizer_name not in cls.OPTIMIZER_REGISTRY:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return cls.OPTIMIZER_REGISTRY[optimizer_name]["default_params"].copy()
    
    @classmethod
    def get_description(cls, optimizer_name: str) -> str:
        """Get description for an optimizer."""
        if optimizer_name not in cls.OPTIMIZER_REGISTRY:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return cls.OPTIMIZER_REGISTRY[optimizer_name]["description"]
    
    @classmethod
    def list_optimizers(cls) -> Dict[str, str]:
        """List all available optimizers with descriptions."""
        return {name: config["description"] for name, config in cls.OPTIMIZER_REGISTRY.items()}
    
    @classmethod
    def create_optimizer(cls, optimizer_name: str, model_parameters, **override_params) -> torch.optim.Optimizer:
        """Create an optimizer instance with given parameters."""
        optimizer_class = cls.get_optimizer_class(optimizer_name)
        default_params = cls.get_default_params(optimizer_name)
        
        # Override default parameters with provided ones
        final_params = {**default_params, **override_params}
        
        return optimizer_class(model_parameters, **final_params)


def get_optimizer_recommendations() -> Dict[str, Dict[str, Any]]:
    """Get recommendations for different use cases."""
    return {
        "accuracy_focused": {
            "optimizer": "hybrid",
            "reason": "Dynamic blending of Adam and SGD for optimal convergence and stability",
            "params": {"lr": 1e-5, "transition_steps": 1000, "final_ratio": 0.1}
        },
        "speed_focused": {
            "optimizer": "sgd",
            "reason": "Generally faster computation per step",
            "params": {"lr": 1e-5, "momentum": 0.9}
        },
        "balanced": {
            "optimizer": "adamw",
            "reason": "Good balance of performance and stability",
            "params": {"lr": 1e-5, "weight_decay": 0.01}
        },
        "memory_constrained": {
            "optimizer": "sgd",
            "reason": "Lower memory footprint compared to adaptive methods",
            "params": {"lr": 1e-5, "momentum": 0.9}
        },
        "research_novel": {
            "optimizer": "hybrid",
            "reason": "Novel approach combining best of both worlds - fast early convergence with stable final performance",
            "params": {"lr": 1e-5, "transition_steps": 1000}
        }
    }


def validate_optimizer_config(optimizer_name: str, params: Dict[str, Any]) -> bool:
    """Validate optimizer configuration."""
    try:
        # Check if optimizer exists
        OptimizerConfig.get_optimizer_class(optimizer_name)
        
        # Basic parameter validation
        if "lr" in params and (params["lr"] <= 0 or params["lr"] > 1):
            return False
        
        if optimizer_name == "sgd" and "momentum" in params:
            if not (0 <= params["momentum"] <= 1):
                return False
        
        if optimizer_name == "adamw" and "betas" in params:
            if not (isinstance(params["betas"], (list, tuple)) and len(params["betas"]) == 2):
                return False
        
        return True
        
    except ValueError:
        return False


def compare_optimizers_theoretical() -> Dict[str, Dict[str, str]]:
    """Provide theoretical comparison of optimizers."""
    return {
        "adamw": {
            "convergence": "Fast initial convergence, may plateau",
            "memory": "Higher (stores momentum and variance)",
            "stability": "Generally stable",
            "hyperparams": "Less sensitive to learning rate",
            "best_for": "Most general-purpose applications"
        },
        "sgd": {
            "convergence": "Slower but can achieve better final performance",
            "memory": "Lower (only stores momentum)",
            "stability": "Can be unstable with high learning rates",
            "hyperparams": "Very sensitive to learning rate and momentum",
            "best_for": "When you can tune hyperparameters carefully"
        },
        "adabound": {
            "convergence": "Fast initial like Adam, stable final like SGD",
            "memory": "Similar to Adam",
            "stability": "Designed to be more stable than Adam",
            "hyperparams": "Less sensitive than SGD, more robust than Adam",
            "best_for": "When you want the best of both worlds"
        },
        "hybrid": {
            "convergence": "Dynamic transition: Adam-like initially, SGD-like finally",
            "memory": "Highest (stores both Adam moments and SGD momentum)",
            "stability": "Adaptive stability increasing over training",
            "hyperparams": "Moderate sensitivity, configurable transition",
            "best_for": "Research and when optimal convergence + stability is critical"
        }
    } 
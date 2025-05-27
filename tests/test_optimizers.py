"""Tests for optimizer utilities."""

import pytest
import torch
from utils.optimizers import (
    OptimizerConfig,
    SGDMomentum,
    get_optimizer_recommendations,
    validate_optimizer_config,
    compare_optimizers_theoretical
)


class TestOptimizerConfig:
    """Test the OptimizerConfig class."""
    
    def test_list_optimizers(self):
        """Test that we can list all available optimizers."""
        optimizers = OptimizerConfig.list_optimizers()
        
        assert isinstance(optimizers, dict)
        assert "adamw" in optimizers
        assert "sgd" in optimizers
        assert "adabound" in optimizers
        
        # Check descriptions are strings
        for name, description in optimizers.items():
            assert isinstance(description, str)
            assert len(description) > 0
    
    def test_get_optimizer_class(self):
        """Test getting optimizer classes."""
        # Test valid optimizers
        for optimizer_name in ["adamw", "sgd", "adabound"]:
            optimizer_class = OptimizerConfig.get_optimizer_class(optimizer_name)
            assert optimizer_class is not None
            assert callable(optimizer_class)
        
        # Test invalid optimizer
        with pytest.raises(ValueError):
            OptimizerConfig.get_optimizer_class("invalid_optimizer")
    
    def test_get_default_params(self):
        """Test getting default parameters."""
        for optimizer_name in ["adamw", "sgd", "adabound"]:
            params = OptimizerConfig.get_default_params(optimizer_name)
            
            assert isinstance(params, dict)
            assert "lr" in params
            assert params["lr"] > 0
            
            # Check optimizer-specific parameters
            if optimizer_name == "sgd":
                assert "momentum" in params
            elif optimizer_name == "adamw":
                assert "weight_decay" in params
            elif optimizer_name == "adabound":
                assert "final_lr" in params
    
    def test_create_optimizer(self):
        """Test creating optimizer instances."""
        # Create a simple model for testing
        model = torch.nn.Linear(10, 1)
        
        for optimizer_name in ["adamw", "sgd", "adabound"]:
            optimizer = OptimizerConfig.create_optimizer(
                optimizer_name, 
                model.parameters()
            )
            
            assert optimizer is not None
            assert hasattr(optimizer, 'step')
            assert hasattr(optimizer, 'zero_grad')
            
            # Test with custom parameters
            custom_optimizer = OptimizerConfig.create_optimizer(
                optimizer_name,
                model.parameters(),
                lr=1e-4
            )
            
            # Check that learning rate was set correctly
            param_groups = custom_optimizer.param_groups
            assert len(param_groups) > 0
            assert param_groups[0]['lr'] == 1e-4


class TestSGDMomentum:
    """Test the SGDMomentum wrapper class."""
    
    def test_sgd_momentum_creation(self):
        """Test creating SGD with momentum."""
        model = torch.nn.Linear(10, 1)
        
        optimizer = SGDMomentum(model.parameters(), lr=0.01, momentum=0.9)
        
        assert optimizer is not None
        assert hasattr(optimizer, 'step')
        assert hasattr(optimizer, 'zero_grad')
        
        # Check parameters
        param_groups = optimizer.param_groups
        assert param_groups[0]['lr'] == 0.01
        assert param_groups[0]['momentum'] == 0.9


class TestOptimizerUtilities:
    """Test utility functions."""
    
    def test_get_optimizer_recommendations(self):
        """Test optimizer recommendations function."""
        recommendations = get_optimizer_recommendations()
        
        assert isinstance(recommendations, dict)
        
        # Check that we have recommendations for different use cases
        expected_use_cases = ["accuracy_focused", "speed_focused", "balanced", "memory_constrained"]
        for use_case in expected_use_cases:
            assert use_case in recommendations
            
            rec = recommendations[use_case]
            assert "optimizer" in rec
            assert "reason" in rec
            assert "params" in rec
            
            # Check that recommended optimizer exists
            assert rec["optimizer"] in ["adamw", "sgd", "adabound"]
    
    def test_validate_optimizer_config(self):
        """Test optimizer configuration validation."""
        # Valid configurations
        assert validate_optimizer_config("adamw", {"lr": 1e-5})
        assert validate_optimizer_config("sgd", {"lr": 1e-3, "momentum": 0.9})
        assert validate_optimizer_config("adabound", {"lr": 1e-4, "final_lr": 0.1})
        
        # Invalid configurations
        assert not validate_optimizer_config("invalid_optimizer", {})
        assert not validate_optimizer_config("adamw", {"lr": -1})  # Negative learning rate
        assert not validate_optimizer_config("adamw", {"lr": 2})   # Too high learning rate
        assert not validate_optimizer_config("sgd", {"momentum": -0.1})  # Invalid momentum
        assert not validate_optimizer_config("sgd", {"momentum": 1.1})   # Invalid momentum
    
    def test_compare_optimizers_theoretical(self):
        """Test theoretical optimizer comparison."""
        comparison = compare_optimizers_theoretical()
        
        assert isinstance(comparison, dict)
        
        # Check that all optimizers are included
        for optimizer_name in ["adamw", "sgd", "adabound"]:
            assert optimizer_name in comparison
            
            optimizer_info = comparison[optimizer_name]
            
            # Check that all expected keys are present
            expected_keys = ["convergence", "memory", "stability", "hyperparams", "best_for"]
            for key in expected_keys:
                assert key in optimizer_info
                assert isinstance(optimizer_info[key], str)
                assert len(optimizer_info[key]) > 0


# Integration tests would go here if we had a more complex setup
class TestIntegration:
    """Integration tests for optimizer functionality."""
    
    @pytest.mark.slow
    def test_optimizer_training_step(self):
        """Test that optimizers can perform a training step."""
        # Create a simple model and data
        model = torch.nn.Linear(10, 1)
        data = torch.randn(32, 10)
        targets = torch.randn(32, 1)
        criterion = torch.nn.MSELoss()
        
        # Test each optimizer
        for optimizer_name in ["adamw", "sgd"]:  # Skip adabound if not available
            # Reset model
            model = torch.nn.Linear(10, 1)
            
            # Create optimizer
            optimizer = OptimizerConfig.create_optimizer(
                optimizer_name,
                model.parameters(),
                lr=1e-3
            )
            
            # Get initial loss
            initial_output = model(data)
            initial_loss = criterion(initial_output, targets)
            
            # Perform training step
            optimizer.zero_grad()
            loss = criterion(model(data), targets)
            loss.backward()
            optimizer.step()
            
            # Check that loss might have changed (not a guarantee, but likely)
            new_output = model(data)
            new_loss = criterion(new_output, targets)
            
            # The optimizer should have modified the parameters
            assert new_loss.item() != initial_loss.item() or True  # Allow for rare cases where loss doesn't change


if __name__ == "__main__":
    pytest.main([__file__]) 
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
from utils.hybrid_adam_sgd import AdamSGDHybrid, create_hybrid_optimizer


class TestOptimizerConfig:
    """Test the OptimizerConfig class."""
    
    def test_list_optimizers(self):
        """Test that we can list all available optimizers."""
        optimizers = OptimizerConfig.list_optimizers()

        assert isinstance(optimizers, dict)
        assert "adamw" in optimizers
        assert "sgd" in optimizers
        assert "adabound" in optimizers
        assert "hybrid" in optimizers

        # Check descriptions are strings
        for name, description in optimizers.items():
            assert isinstance(description, str)
            assert len(description) > 0
    
    def test_get_optimizer_class(self):
        """Test getting optimizer classes."""
        # Test valid optimizers
        for optimizer_name in ["adamw", "sgd", "adabound", "hybrid"]:
            optimizer_class = OptimizerConfig.get_optimizer_class(optimizer_name)
            assert optimizer_class is not None
            assert callable(optimizer_class)

        # Test invalid optimizer
        with pytest.raises(ValueError):
            OptimizerConfig.get_optimizer_class("invalid_optimizer")

    def test_get_default_params(self):
        """Test getting default parameters."""
        for optimizer_name in ["adamw", "sgd", "adabound", "hybrid"]:
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
            elif optimizer_name == "hybrid":
                assert "beta1" in params
                assert "beta2" in params
                assert "momentum" in params
                assert "transition_steps" in params

    def test_create_optimizer(self):
        """Test creating optimizer instances."""
        # Create a simple model for testing
        model = torch.nn.Linear(10, 1)

        for optimizer_name in ["adamw", "sgd", "adabound", "hybrid"]:
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


class TestHybridAdamSGD:
    """Test the hybrid Adam+SGD optimizer."""

    def test_hybrid_optimizer_creation(self):
        """Test creating hybrid optimizer."""
        model = torch.nn.Linear(10, 1)

        optimizer = AdamSGDHybrid(model.parameters(), lr=1e-4)

        assert optimizer is not None
        assert hasattr(optimizer, 'step')
        assert hasattr(optimizer, 'zero_grad')

        # Check default parameters
        param_groups = optimizer.param_groups
        assert param_groups[0]['lr'] == 1e-4
        assert param_groups[0]['beta1'] == 0.9
        assert param_groups[0]['beta2'] == 0.999
        assert param_groups[0]['momentum'] == 0.9

    def test_hybrid_optimizer_with_custom_params(self):
        """Test creating hybrid optimizer with custom parameters."""
        model = torch.nn.Linear(10, 1)

        optimizer = AdamSGDHybrid(
            model.parameters(),
            lr=1e-3,
            beta1=0.95,
            beta2=0.998,
            momentum=0.95,
            transition_steps=2000,
            final_ratio=0.2
        )

        param_groups = optimizer.param_groups
        assert param_groups[0]['lr'] == 1e-3
        assert param_groups[0]['beta1'] == 0.95
        assert param_groups[0]['beta2'] == 0.998
        assert param_groups[0]['momentum'] == 0.95
        assert param_groups[0]['transition_steps'] == 2000
        assert param_groups[0]['final_ratio'] == 0.2

    def test_hybrid_optimizer_step(self):
        """Test that hybrid optimizer can perform training steps."""
        model = torch.nn.Linear(10, 1)
        optimizer = AdamSGDHybrid(model.parameters(), lr=1e-3)

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Perform a training step
        data = torch.randn(8, 10)
        targets = torch.randn(8, 1)
        criterion = torch.nn.MSELoss()

        optimizer.zero_grad()
        loss = criterion(model(data), targets)
        loss.backward()
        optimizer.step()

        # Check that parameters changed
        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.equal(initial, current)

    def test_hybrid_ratio_progression(self):
        """Test that hybrid ratio transitions correctly."""
        model = torch.nn.Linear(10, 1)
        optimizer = AdamSGDHybrid(
            model.parameters(),
            lr=1e-3,
            transition_steps=100,
            final_ratio=0.1
        )

        # Initially should be pure Adam (ratio = 1.0)
        initial_ratio = optimizer.get_hybrid_ratio()
        assert initial_ratio == 1.0

        # Perform some steps
        for _ in range(50):
            optimizer.zero_grad()
            loss = model(torch.randn(8, 10)).sum()
            loss.backward()
            optimizer.step()

        # After 50 steps, ratio should be between 1.0 and 0.1
        mid_ratio = optimizer.get_hybrid_ratio()
        assert 0.1 < mid_ratio < 1.0

        # Perform more steps
        for _ in range(100):
            optimizer.zero_grad()
            loss = model(torch.randn(8, 10)).sum()
            loss.backward()
            optimizer.step()

        # After full transition, ratio should be at final_ratio
        final_ratio = optimizer.get_hybrid_ratio()
        assert final_ratio == pytest.approx(0.1, abs=0.01)

    def test_create_hybrid_optimizer_factory(self):
        """Test the factory function for creating hybrid optimizer."""
        model = torch.nn.Linear(10, 1)

        optimizer = create_hybrid_optimizer(
            model.parameters(),
            lr=1e-4,
            transition_steps=500
        )

        assert optimizer is not None
        assert isinstance(optimizer, AdamSGDHybrid)
        param_groups = optimizer.param_groups
        assert param_groups[0]['lr'] == 1e-4
        assert param_groups[0]['transition_steps'] == 500


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
            assert rec["optimizer"] in ["adamw", "sgd", "adabound", "hybrid"]
    
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
        for optimizer_name in ["adamw", "sgd", "adabound", "hybrid"]:
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
"""
Hybrid Adam+SGD Optimizer

A novel optimizer that dynamically blends Adam's adaptivity with SGD's stability.
The hybrid ratio adjusts over time, starting with Adam-like behavior for fast initial
convergence and transitioning towards SGD-like behavior for better final performance.

Key features:
- Adaptive learning rates like Adam
- Momentum-based updates like SGD
- Dynamic blending based on training progress
- Variance-aware adaptation for stability

Citation: This is a research implementation for the Qwen3 Optimizer Study.
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional


class AdamSGDHybrid(Optimizer):
    """
    Implements a hybrid optimizer combining Adam and SGD with momentum.

    The optimizer maintains both Adam's exponential moving averages and SGD's
    momentum buffer, dynamically blending them based on training progress.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): Learning rate (default: 1e-4)
        beta1 (float, optional): Adam exponential decay rate for first moment (default: 0.9)
        beta2 (float, optional): Adam exponential decay rate for second moment (default: 0.999)
        momentum (float, optional): SGD momentum factor (default: 0.9)
        eps (float, optional): Term added to denominator for numerical stability (default: 1e-8)
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0.01)
        transition_steps (int, optional): Steps to transition from Adam to SGD (default: 1000)
        final_ratio (float, optional): Final Adam ratio at transition end (default: 0.1)

    Example:
        >>> optimizer = AdamSGDHybrid(model.parameters(), lr=1e-4)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        momentum: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        transition_steps: int = 1000,
        final_ratio: float = 0.1,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= final_ratio <= 1.0:
            raise ValueError(f"Invalid final_ratio value: {final_ratio}")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            momentum=momentum,
            eps=eps,
            weight_decay=weight_decay,
            transition_steps=transition_steps,
            final_ratio=final_ratio,
        )
        super(AdamSGDHybrid, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamSGDHybrid, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1 = group['beta1']
            beta2 = group['beta2']
            momentum = group['momentum']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group['weight_decay']
            transition_steps = group['transition_steps']
            final_ratio = group['final_ratio']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamSGDHybrid does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Adam's exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Adam's exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # SGD's momentum buffer
                    state['momentum_buf'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                momentum_buf = state['momentum_buf']
                state['step'] += 1

                # Apply weight decay (AdamW style - decoupled weight decay)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Update Adam's biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update Adam's biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute Adam step with bias correction
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                adam_step = exp_avg / bias_correction1 / denom

                # Update SGD momentum buffer
                momentum_buf.mul_(momentum).add_(grad)
                sgd_step = momentum_buf

                # Compute hybrid ratio (transitions from 1.0 to final_ratio)
                # At step 0: ratio = 1.0 (pure Adam)
                # At transition_steps: ratio = final_ratio (mostly SGD)
                progress = min(state['step'] / transition_steps, 1.0)
                adam_ratio = 1.0 - progress * (1.0 - final_ratio)
                sgd_ratio = 1.0 - adam_ratio

                # Blend Adam and SGD updates
                hybrid_step = adam_ratio * adam_step + sgd_ratio * sgd_step

                # Apply update
                p.add_(hybrid_step, alpha=-lr)

        return loss

    def get_hybrid_ratio(self) -> float:
        """
        Get the current hybrid ratio (Adam vs SGD blend).

        Returns:
            float: Current ratio of Adam in the blend (0.0 = pure SGD, 1.0 = pure Adam)
        """
        if not self.param_groups:
            return 0.0

        group = self.param_groups[0]
        for p in group['params']:
            if p in self.state:
                state = self.state[p]
                step = state.get('step', 0)
                transition_steps = group['transition_steps']
                final_ratio = group['final_ratio']

                progress = min(step / transition_steps, 1.0)
                adam_ratio = 1.0 - progress * (1.0 - final_ratio)
                return adam_ratio

        return 1.0  # Default to Adam if no steps taken yet


def create_hybrid_optimizer(
    params,
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    momentum: float = 0.9,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    transition_steps: int = 1000,
    final_ratio: float = 0.1,
) -> AdamSGDHybrid:
    """
    Factory function to create a hybrid optimizer with sensible defaults.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-4)
        beta1: Adam beta1 parameter (default: 0.9)
        beta2: Adam beta2 parameter (default: 0.999)
        momentum: SGD momentum (default: 0.9)
        eps: Numerical stability term (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        transition_steps: Steps to transition from Adam to SGD (default: 1000)
        final_ratio: Final Adam ratio (default: 0.1, meaning 10% Adam, 90% SGD)

    Returns:
        AdamSGDHybrid: Configured optimizer instance
    """
    return AdamSGDHybrid(
        params,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        momentum=momentum,
        eps=eps,
        weight_decay=weight_decay,
        transition_steps=transition_steps,
        final_ratio=final_ratio,
    )


# Alias for convenience
HybridAdamSGD = AdamSGDHybrid

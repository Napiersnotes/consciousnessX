"""
Learning rate scheduler for ConsciousnessX training.

Provides various learning rate scheduling strategies.
"""

import numpy as np
from typing import Optional, Union, List


class LRScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, optimizer, initial_lr: float):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer instance
            initial_lr: Initial learning rate
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.step_count = 0
    
    def step(self) -> float:
        """
        Update learning rate.
        
        Returns:
            Current learning rate
        """
        self.step_count += 1
        lr = self.get_lr(self.step_count)
        self.current_lr = lr
        self.optimizer.lr = lr
        return lr
    
    def get_lr(self, step: int) -> float:
        """
        Get learning rate at given step.
        
        Args:
            step: Current step
            
        Returns:
            Learning rate
        """
        raise NotImplementedError("Subclasses must implement get_lr")
    
    def state_dict(self) -> dict:
        """Get scheduler state."""
        return {
            'initial_lr': self.initial_lr,
            'current_lr': self.current_lr,
            'step_count': self.step_count
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state."""
        self.initial_lr = state_dict['initial_lr']
        self.current_lr = state_dict['current_lr']
        self.step_count = state_dict['step_count']


class StepLR(LRScheduler):
    """Decays learning rate by step_size every step_size epochs."""
    
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        """
        Initialize step scheduler.
        
        Args:
            optimizer: Optimizer instance
            step_size: Period of learning rate decay
            gamma: Multiplicative factor of learning rate decay
        """
        super().__init__(optimizer, optimizer.lr)
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self, step: int) -> float:
        """Get learning rate at given step."""
        return self.initial_lr * (self.gamma ** (step // self.step_size))


class ExponentialLR(LRScheduler):
    """Decays learning rate exponentially."""
    
    def __init__(self, optimizer, gamma: float = 0.95):
        """
        Initialize exponential scheduler.
        
        Args:
            optimizer: Optimizer instance
            gamma: Multiplicative factor of decay
        """
        super().__init__(optimizer, optimizer.lr)
        self.gamma = gamma
    
    def get_lr(self, step: int) -> float:
        """Get learning rate at given step."""
        return self.initial_lr * (self.gamma ** step)


class CosineAnnealingLR(LRScheduler):
    """Set learning rate using cosine annealing schedule."""
    
    def __init__(self, optimizer, T_max: int, eta_min: float = 0):
        """
        Initialize cosine annealing scheduler.
        
        Args:
            optimizer: Optimizer instance
            T_max: Maximum number of iterations
            eta_min: Minimum learning rate
        """
        super().__init__(optimizer, optimizer.lr)
        self.T_max = T_max
        self.eta_min = eta_min
    
    def get_lr(self, step: int) -> float:
        """Get learning rate at given step."""
        return self.eta_min + (self.initial_lr - self.eta_min) * \
               (1 + np.cos(np.pi * step / self.T_max)) / 2


class ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when a metric has stopped improving."""
    
    def __init__(
        self,
        optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        min_lr: float = 0
    ):
        """
        Initialize plateau scheduler.
        
        Args:
            optimizer: Optimizer instance
            mode: 'min' or 'max'
            factor: Factor by which to reduce learning rate
            patience: Number of epochs with no improvement
            threshold: Threshold for measuring new optimum
            min_lr: Lower bound on learning rate
        """
        super().__init__(optimizer, optimizer.lr)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        
        self.best = np.inf if mode == 'min' else -np.inf
        self.num_bad_epochs = 0
    
    def step(self, metric: Optional[float] = None) -> float:
        """
        Update learning rate based on metric.
        
        Args:
            metric: Current metric value
            
        Returns:
            Current learning rate
        """
        if metric is None:
            self.step_count += 1
            return self.current_lr
        
        improved = False
        if self.mode == 'min':
            if metric < self.best - self.threshold:
                improved = True
                self.best = metric
        else:
            if metric > self.best + self.threshold:
                improved = True
                self.best = metric
        
        if improved:
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.optimizer.lr = self.current_lr
            self.num_bad_epochs = 0
        
        self.step_count += 1
        return self.current_lr


class CyclicLR(LRScheduler):
    """Sets learning rate according to cyclical learning rate policy."""
    
    def __init__(
        self,
        optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        mode: str = 'triangular'
    ):
        """
        Initialize cyclic scheduler.
        
        Args:
            optimizer: Optimizer instance
            base_lr: Lower bound of learning rate
            max_lr: Upper bound of learning rate
            step_size_up: Number of steps to increase learning rate
            mode: 'triangular', 'triangular2', or 'exp_range'
        """
        super().__init__(optimizer, base_lr)
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.mode = mode
    
    def get_lr(self, step: int) -> float:
        """Get learning rate at given step."""
        cycle = np.floor(1 + step / (2 * self.step_size_up))
        x = np.abs(step / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            base_height = (self.max_lr - self.base_lr)
            return self.base_lr + base_height * np.maximum(0, (1 - x))
        elif self.mode == 'triangular2':
            base_height = (self.max_lr - self.base_lr) / (2 ** (cycle - 1))
            return self.base_lr + base_height * np.maximum(0, (1 - x))
        elif self.mode == 'exp_range':
            base_height = (self.max_lr - self.base_lr) * (0.99994 ** step)
            return self.base_lr + base_height * np.maximum(0, (1 - x))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class WarmupScheduler(LRScheduler):
    """Linear warmup followed by another scheduler."""
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        scheduler: LRScheduler
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            scheduler: Scheduler to use after warmup
        """
        super().__init__(optimizer, optimizer.lr)
        self.warmup_steps = warmup_steps
        self.scheduler = scheduler
        self.warmup_lr = optimizer.lr
    
    def get_lr(self, step: int) -> float:
        """Get learning rate at given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.warmup_lr * (step + 1) / self.warmup_steps
        else:
            # Use wrapped scheduler
            return self.scheduler.get_lr(step - self.warmup_steps)
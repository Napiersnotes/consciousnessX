"""
Random number generation utilities for ConsciousnessX.

Provides reproducible random number generation across different components.
"""

import random
import numpy as np
from typing import Optional, Union, List
from numpy.random import Generator, PCG64


# Global random state
_global_rng: Optional[Generator] = None


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    global _global_rng
    
    # Set seeds for all random libraries
    random.seed(seed)
    np.random.seed(seed)
    
    # Create NumPy Generator with specific algorithm
    _global_rng = np.random.Generator(PCG64(seed))
    
    # Set torch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_random_state() -> Generator:
    """
    Get global random state generator.
    
    Returns:
        NumPy random Generator instance
    """
    global _global_rng
    if _global_rng is None:
        _global_rng = np.random.Generator(PCG64())
    return _global_rng


def rand_float(
    low: float = 0.0,
    high: float = 1.0,
    size: Optional[Union[int, tuple]] = None
) -> Union[float, np.ndarray]:
    """
    Generate random float(s) in range [low, high).
    
    Args:
        low: Lower bound (inclusive)
        high: Upper bound (exclusive)
        size: Output shape (None for scalar)
        
    Returns:
        Random float or array of floats
    """
    rng = get_random_state()
    return rng.uniform(low, high, size=size)


def rand_int(
    low: int = 0,
    high: Optional[int] = None,
    size: Optional[Union[int, tuple]] = None
) -> Union[int, np.ndarray]:
    """
    Generate random integer(s) in range [low, high).
    
    Args:
        low: Lower bound (inclusive)
        high: Upper bound (exclusive)
        size: Output shape (None for scalar)
        
    Returns:
        Random int or array of ints
    """
    rng = get_random_state()
    return rng.integers(low, high, size=size)


def rand_normal(
    loc: float = 0.0,
    scale: float = 1.0,
    size: Optional[Union[int, tuple]] = None
) -> Union[float, np.ndarray]:
    """
    Generate random float(s) from normal distribution.
    
    Args:
        loc: Mean of distribution
        scale: Standard deviation
        size: Output shape (None for scalar)
        
    Returns:
        Random float or array from normal distribution
    """
    rng = get_random_state()
    return rng.normal(loc, scale, size=size)


def rand_choice(
    a: Union[int, List, np.ndarray],
    size: Optional[int] = None,
    replace: bool = True,
    p: Optional[np.ndarray] = None
) -> Union[int, np.ndarray]:
    """
    Generate random sample(s) from given array.
    
    Args:
        a: If int, random sample from np.arange(a); if array-like, sample from elements
        size: Number of samples
        replace: Whether to sample with replacement
        p: Probabilities associated with each entry
        
    Returns:
        Random sample(s)
    """
    rng = get_random_state()
    return rng.choice(a, size=size, replace=replace, p=p)


def rand_permutation(n: int) -> np.ndarray:
    """
    Generate random permutation of integers [0, n).
    
    Args:
        n: Number of elements
        
    Returns:
        Random permutation
    """
    rng = get_random_state()
    return rng.permutation(n)


def rand_shuffle(x: Union[List, np.ndarray]) -> Union[List, np.ndarray]:
    """
    Shuffle array in place.
    
    Args:
        x: Array to shuffle
        
    Returns:
        Shuffled array (same object)
    """
    rng = get_random_state()
    rng.shuffle(x)
    return x


def random_split(
    total: int,
    ratios: List[float],
    shuffle: bool = True
) -> List[np.ndarray]:
    """
    Split indices into multiple groups with given ratios.
    
    Args:
        total: Total number of items
        ratios: List of ratios (sum should be 1.0)
        shuffle: Whether to shuffle before splitting
        
    Returns:
        List of index arrays for each split
    """
    indices = np.arange(total)
    
    if shuffle:
        indices = rand_permutation(total)
    
    # Normalize ratios
    ratios = np.array(ratios)
    ratios = ratios / ratios.sum()
    
    splits = []
    start = 0
    
    for ratio in ratios[:-1]:
        end = start + int(total * ratio)
        splits.append(indices[start:end])
        start = end
    
    # Add remaining indices to last split
    splits.append(indices[start:])
    
    return splits


def bernoulli(p: float = 0.5, size: Optional[Union[int, tuple]] = None) -> Union[int, np.ndarray]:
    """
    Sample from Bernoulli distribution.
    
    Args:
        p: Probability of success
        size: Output shape (None for scalar)
        
    Returns:
        0 or 1 (or array of 0s and 1s)
    """
    rng = get_random_state()
    return rng.binomial(1, p, size=size)


def poisson(lam: float = 1.0, size: Optional[Union[int, tuple]] = None) -> Union[int, np.ndarray]:
    """
    Sample from Poisson distribution.
    
    Args:
        lam: Expected number of events
        size: Output shape (None for scalar)
        
    Returns:
        Sample from Poisson distribution
    """
    rng = get_random_state()
    return rng.poisson(lam, size=size)


def exponential(scale: float = 1.0, size: Optional[Union[int, tuple]] = None) -> Union[float, np.ndarray]:
    """
    Sample from exponential distribution.
    
    Args:
        scale: Scale parameter (inverse of rate)
        size: Output shape (None for scalar)
        
    Returns:
        Sample from exponential distribution
    """
    rng = get_random_state()
    return rng.exponential(scale, size=size)


def lognormal(
    mean: float = 0.0,
    sigma: float = 1.0,
    size: Optional[Union[int, tuple]] = None
) -> Union[float, np.ndarray]:
    """
    Sample from log-normal distribution.
    
    Args:
        mean: Mean of underlying normal distribution
        sigma: Standard deviation of underlying normal distribution
        size: Output shape (None for scalar)
        
    Returns:
        Sample from log-normal distribution
    """
    rng = get_random_state()
    return rng.lognormal(mean, sigma, size=size)


def random_unit_vector(dim: int) -> np.ndarray:
    """
    Generate random unit vector in n-dimensional space.
    
    Args:
        dim: Dimension of vector
        
    Returns:
        Unit vector of shape (dim,)
    """
    rng = get_random_state()
    vec = rng.standard_normal(dim)
    return vec / np.linalg.norm(vec)


def random_rotation_matrix(dim: int = 3) -> np.ndarray:
    """
    Generate random rotation matrix using QR decomposition.
    
    Args:
        dim: Dimension of rotation matrix
        
    Returns:
        Orthogonal rotation matrix
    """
    rng = get_random_state()
    A = rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    
    return Q


class RandomContext:
    """
    Context manager for temporary random state changes.
    """
    
    def __init__(self, seed: int):
        """
        Initialize random context.
        
        Args:
            seed: Seed to use within context
        """
        self.seed = seed
        self.old_state = None
    
    def __enter__(self):
        """Enter context and set new seed."""
        # Save current state
        rng = get_random_state()
        self.old_state = rng.bit_generator.state
        
        # Set new seed
        set_seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore old state."""
        global _global_rng
        if self.old_state is not None:
            _global_rng = np.random.Generator(PCG64())
            _global_rng.bit_generator.state = self.old_state
        return False
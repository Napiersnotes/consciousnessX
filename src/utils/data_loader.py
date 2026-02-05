"""
Data loading utilities for ConsciousnessX.

Provides data loading and batching functionality for neural networks.
"""

import numpy as np
from typing import Optional, Tuple, List, Iterator, Union
from pathlib import Path


class DataLoader:
    """
    Generic data loader for batch-based training.
    """

    def __init__(
        self,
        data: Union[np.ndarray, List, Tuple],
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize data loader.

        Args:
            data: Input data (array, list, or tuple of arrays)
            batch_size: Batch size
            shuffle: Whether to shuffle data each epoch
            drop_last: Whether to drop last incomplete batch
            seed: Random seed for reproducibility
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        # Handle different data types
        if isinstance(data, tuple):
            self.data = [np.asarray(d) for d in data]
            self.num_samples = len(self.data[0])
        else:
            self.data = [np.asarray(data)]
            self.num_samples = len(self.data[0])

        # Verify all arrays have same length
        for d in self.data[1:]:
            if len(d) != self.num_samples:
                raise ValueError("All data arrays must have the same length")

        self.indices = np.arange(self.num_samples)
        self.current_position = 0

        if seed is not None:
            np.random.seed(seed)

    def __len__(self) -> int:
        """Get number of batches."""
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator:
        """Iterate over batches."""
        self.reset()
        return self

    def __next__(self) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Get next batch."""
        if self.current_position >= self.num_samples:
            raise StopIteration

        # Get batch indices
        end_idx = min(self.current_position + self.batch_size, self.num_samples)
        batch_indices = self.indices[self.current_position : end_idx]

        # Handle incomplete batch
        if not self.drop_last and len(batch_indices) < self.batch_size:
            # Pad last batch if needed
            padding_size = self.batch_size - len(batch_indices)
            padding_indices = np.random.choice(self.num_samples, padding_size)
            batch_indices = np.concatenate([batch_indices, padding_indices])

        # Get batch data
        batches = [d[batch_indices] for d in self.data]

        self.current_position += self.batch_size

        if len(batches) == 1:
            return batches[0]
        else:
            return tuple(batches)

    def reset(self) -> None:
        """Reset data loader for new epoch."""
        self.current_position = 0

        if self.shuffle:
            self.indices = np.random.permutation(self.num_samples)

    def get_batch(self, batch_idx: int) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Get specific batch by index.

        Args:
            batch_idx: Batch index

        Returns:
            Batch data
        """
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]

        batches = [d[batch_indices] for d in self.data]

        if len(batches) == 1:
            return batches[0]
        else:
            return tuple(batches)

    def to(self, device: str) -> "DataLoader":
        """
        Move data to device (placeholder for PyTorch compatibility).

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        return self


class BatchSampler:
    """
    Sampler for creating batches with custom sampling strategies.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize batch sampler.

        Args:
            num_samples: Total number of samples
            batch_size: Batch size
            shuffle: Whether to shuffle samples
            drop_last: Whether to drop last incomplete batch
            seed: Random seed
        """
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        self.indices = np.arange(num_samples)
        self.current_position = 0

        if seed is not None:
            np.random.seed(seed)

    def __len__(self) -> int:
        """Get number of batches."""
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batch indices."""
        self.reset()
        return self

    def __next__(self) -> List[int]:
        """Get next batch indices."""
        if self.current_position >= self.num_samples:
            raise StopIteration

        end_idx = min(self.current_position + self.batch_size, self.num_samples)
        batch_indices = self.indices[self.current_position : end_idx].tolist()

        self.current_position += self.batch_size

        return batch_indices

    def reset(self) -> None:
        """Reset sampler for new epoch."""
        self.current_position = 0

        if self.shuffle:
            self.indices = np.random.permutation(self.num_samples)


def load_data_from_file(filepath: str, delimiter: Optional[str] = None) -> np.ndarray:
    """
    Load data from text file.

    Args:
        filepath: Path to data file
        delimiter: Delimiter for text files (None for numpy loadtxt default)

    Returns:
        Data array
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        return np.load(filepath)
    elif suffix == ".npz":
        data = np.load(filepath)
        # Return first array in npz file
        return data[data.files[0]]
    elif suffix in [".txt", ".csv"]:
        return np.loadtxt(filepath, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def save_data_to_file(data: np.ndarray, filepath: str) -> None:
    """
    Save data to file.

    Args:
        data: Data array to save
        filepath: Path to save file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix == ".npy":
        np.save(filepath, data)
    elif suffix == ".npz":
        np.savez(filepath, data=data)
    elif suffix == ".txt":
        np.savetxt(filepath, data)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def train_test_split(
    data: Union[np.ndarray, Tuple[np.ndarray, ...]],
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, ...]], Union[np.ndarray, Tuple[np.ndarray, ...]]]:
    """
    Split data into train and test sets.

    Args:
        data: Input data (array or tuple of arrays)
        test_size: Fraction of data for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_data, test_data)
    """
    if isinstance(data, tuple):
        num_samples = len(data[0])
    else:
        num_samples = len(data)

    indices = np.arange(num_samples)

    if random_state is not None:
        np.random.seed(random_state)

    np.random.shuffle(indices)

    split_idx = int(num_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    if isinstance(data, tuple):
        train_data = tuple(d[train_indices] for d in data)
        test_data = tuple(d[test_indices] for d in data)
    else:
        train_data = data[train_indices]
        test_data = data[test_indices]

    return train_data, test_data


def create_sequences(data: np.ndarray, sequence_length: int, stride: int = 1) -> np.ndarray:
    """
    Create sequences from time series data.

    Args:
        data: Input time series data
        sequence_length: Length of each sequence
        stride: Step between sequences

    Returns:
        Array of sequences
    """
    num_sequences = (len(data) - sequence_length) // stride + 1
    sequences = np.zeros((num_sequences, sequence_length, *data.shape[1:]))

    for i in range(num_sequences):
        start_idx = i * stride
        end_idx = start_idx + sequence_length
        sequences[i] = data[start_idx:end_idx]

    return sequences

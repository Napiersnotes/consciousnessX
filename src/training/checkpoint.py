"""
CheckpointManager module for managing training checkpoints
"""

import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class CheckpointManager:
    """
    Manager for saving and loading training checkpoints.

    This class handles automatic checkpoint management, including
    saving, loading, and cleanup of old checkpoints.
    """

    def __init__(
        self, save_dir: str = "data/checkpoints", max_to_keep: int = 5, save_best_only: bool = True
    ):
        """
        Initialize the CheckpointManager.

        Args:
            save_dir: Directory to save checkpoints
            max_to_keep: Maximum number of checkpoints to keep
            save_best_only: Whether to save only the best checkpoints
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.max_to_keep = max_to_keep
        self.save_best_only = save_best_only

        self.best_metric = float("inf")
        self.checkpoint_history: List[Dict[str, Any]] = []

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        metric_to_optimize: str = "loss",
        filename: Optional[str] = None,
    ):
        """
        Save a training checkpoint.

        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer to save
            metrics: Dictionary of metrics
            metric_to_optimize: Metric to optimize for best checkpoint
            filename: Optional custom filename
        """
        # Determine if this is the best checkpoint
        current_metric = metrics.get(metric_to_optimize, float("inf"))
        is_best = current_metric < self.best_metric

        if self.save_best_only and not is_best:
            return

        # Update best metric
        if is_best:
            self.best_metric = current_metric

        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_epoch_{epoch}_{timestamp}.pt"

        filepath = self.save_dir / filename

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_metric": self.best_metric,
            "timestamp": datetime.now().isoformat(),
        }

        torch.save(checkpoint, filepath)

        # Record checkpoint
        self.checkpoint_history.append(
            {"epoch": epoch, "filepath": filepath, "metrics": metrics, "is_best": is_best}
        )

        # Clean up old checkpoints
        self._cleanup()

        print(f"Saved checkpoint: {filepath}")
        if is_best:
            print(f"  Best {metric_to_optimize}: {current_metric:.4f}")

    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load a training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to load checkpoint on

        Returns:
            Dictionary containing checkpoint information
        """
        filepath = Path(checkpoint_path)

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"  Metrics: {checkpoint['metrics']}")

        return checkpoint

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load the best checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to load checkpoint on

        Returns:
            Dictionary containing checkpoint information
        """
        best_checkpoint = self._find_best_checkpoint()

        if best_checkpoint is None:
            raise FileNotFoundError("No best checkpoint found")

        return self.load(best_checkpoint["filepath"], model, optimizer, device)

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load the latest checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to load checkpoint on

        Returns:
            Dictionary containing checkpoint information
        """
        if not self.checkpoint_history:
            raise FileNotFoundError("No checkpoints found")

        latest_checkpoint = self.checkpoint_history[-1]
        return self.load(latest_checkpoint["filepath"], model, optimizer, device)

    def _find_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Find the best checkpoint.

        Returns:
            Best checkpoint dictionary or None
        """
        best_checkpoints = [cp for cp in self.checkpoint_history if cp["is_best"]]

        if not best_checkpoints:
            return None

        return best_checkpoints[-1]

    def _cleanup(self):
        """
        Clean up old checkpoints, keeping only the most recent ones.
        """
        if len(self.checkpoint_history) <= self.max_to_keep:
            return

        # Get checkpoints to remove (oldest non-best)
        checkpoints_to_remove = []
        best_checkpoints = [cp for cp in self.checkpoint_history if cp["is_best"]]

        for cp in self.checkpoint_history:
            if not cp["is_best"]:
                checkpoints_to_remove.append(cp)
                if len(checkpoints_to_remove) + len(best_checkpoints) > self.max_to_keep:
                    break

        # Remove old checkpoints
        for cp in checkpoints_to_remove:
            if cp["filepath"].exists():
                cp["filepath"].unlink()
                print(f"Removed old checkpoint: {cp['filepath']}")

        # Update checkpoint history
        self.checkpoint_history = [
            cp for cp in self.checkpoint_history if cp not in checkpoints_to_remove or cp["is_best"]
        ]

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint information dictionaries
        """
        return self.checkpoint_history.copy()

    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get information about a checkpoint without loading it.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary containing checkpoint information
        """
        filepath = Path(checkpoint_path)

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint metadata only
        checkpoint = torch.load(filepath, map_location="cpu")

        return {
            "epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics"),
            "best_metric": checkpoint.get("best_metric"),
            "timestamp": checkpoint.get("timestamp"),
            "filesize": filepath.stat().st_size,
        }

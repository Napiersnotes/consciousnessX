"""
Performance tracker for ConsciousnessX evaluation.

Tracks and aggregates training metrics over time.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict, deque
import time


class PerformanceTracker:
    """
    Track and aggregate performance metrics during training and evaluation.
    """

    def __init__(self, window_size: int = 100, store_history: bool = True):
        """
        Initialize performance tracker.

        Args:
            window_size: Size of rolling window for statistics
            store_history: Whether to store full history of metrics
        """
        self.window_size = window_size
        self.store_history = store_history

        # Current values
        self.metrics: Dict[str, float] = {}

        # History storage
        self.history: Dict[str, List[float]] = defaultdict(list)

        # Rolling windows
        self.windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

        # Best values
        self.best_metrics: Dict[str, float] = {}
        self.best_epochs: Dict[str, int] = {}

        # Timing
        self.start_times: Dict[str, float] = {}
        self.durations: Dict[str, List[float]] = defaultdict(list)

        self.step_count = 0

    def update(self, metrics: Dict[str, float]) -> None:
        """
        Update tracker with new metrics.

        Args:
            metrics: Dictionary of metric names to values
        """
        self.step_count += 1

        for name, value in metrics.items():
            # Update current value
            self.metrics[name] = value

            # Add to window
            self.windows[name].append(value)

            # Store history
            if self.store_history:
                self.history[name].append(value)

            # Update best
            if name not in self.best_metrics:
                self.best_metrics[name] = value
                self.best_epochs[name] = self.step_count
            else:
                # Assume lower is better for most metrics
                # Can be customized with mode parameter
                if value < self.best_metrics[name]:
                    self.best_metrics[name] = value
                    self.best_epochs[name] = self.step_count

    def get_metric(self, name: str) -> Optional[float]:
        """
        Get current value of a metric.

        Args:
            name: Metric name

        Returns:
            Current metric value or None if not tracked
        """
        return self.metrics.get(name)

    def get_average(self, name: str) -> Optional[float]:
        """
        Get rolling average of a metric.

        Args:
            name: Metric name

        Returns:
            Rolling average or None if not tracked
        """
        if name not in self.windows or len(self.windows[name]) == 0:
            return None
        return np.mean(list(self.windows[name]))

    def get_std(self, name: str) -> Optional[float]:
        """
        Get rolling standard deviation of a metric.

        Args:
            name: Metric name

        Returns:
            Rolling std or None if not tracked
        """
        if name not in self.windows or len(self.windows[name]) == 0:
            return None
        return np.std(list(self.windows[name]))

    def get_history(self, name: str) -> List[float]:
        """
        Get full history of a metric.

        Args:
            name: Metric name

        Returns:
            List of metric values
        """
        return self.history[name].copy()

    def get_best(self, name: str) -> Optional[float]:
        """
        Get best value of a metric.

        Args:
            name: Metric name

        Returns:
            Best value or None if not tracked
        """
        return self.best_metrics.get(name)

    def get_best_epoch(self, name: str) -> Optional[int]:
        """
        Get step/epoch when metric achieved best value.

        Args:
            name: Metric name

        Returns:
            Step number or None if not tracked
        """
        return self.best_epochs.get(name)

    def start_timer(self, name: str) -> None:
        """
        Start timing an operation.

        Args:
            name: Timer name
        """
        self.start_times[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """
        Stop timing an operation and record duration.

        Args:
            name: Timer name

        Returns:
            Duration in seconds
        """
        if name not in self.start_times:
            raise ValueError(f"Timer {name} not started")

        duration = time.time() - self.start_times[name]
        self.durations[name].append(duration)

        # Update as a metric
        self.update({f"{name}_time": duration})

        return duration

    def get_average_duration(self, name: str) -> Optional[float]:
        """
        Get average duration of a timed operation.

        Args:
            name: Timer name

        Returns:
            Average duration or None if not tracked
        """
        if name not in self.durations or len(self.durations[name]) == 0:
            return None
        return np.mean(self.durations[name])

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all tracked metrics.

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "step_count": self.step_count,
            "metrics": {},
            "averages": {},
            "best": {},
            "timings": {},
        }

        for name in self.metrics.keys():
            summary["metrics"][name] = self.get_metric(name)
            summary["averages"][name] = self.get_average(name)
            summary["best"][name] = self.get_best(name)

        for name in self.durations.keys():
            summary["timings"][name] = self.get_average_duration(name)

        return summary

    def reset(self) -> None:
        """Reset all tracking state."""
        self.metrics.clear()
        self.history.clear()
        self.windows.clear()
        self.best_metrics.clear()
        self.best_epochs.clear()
        self.start_times.clear()
        self.durations.clear()
        self.step_count = 0

    def __repr__(self) -> str:
        """String representation of tracker."""
        summary = self.get_summary()
        return f"PerformanceTracker(step_count={summary['step_count']}, metrics={list(summary['metrics'].keys())})"


class MetricLogger:
    """
    Log metrics to console with formatted output.
    """

    def __init__(self, tracker: PerformanceTracker, log_interval: int = 10):
        """
        Initialize metric logger.

        Args:
            tracker: Performance tracker instance
            log_interval: Log every N steps
        """
        self.tracker = tracker
        self.log_interval = log_interval

    def log(self, force: bool = False) -> None:
        """
        Log current metrics.

        Args:
            force: Force logging regardless of interval
        """
        if not force and self.tracker.step_count % self.log_interval != 0:
            return

        summary = self.tracker.get_summary()
        msg = f"Step {self.tracker.step_count:6d}"

        for name, value in summary["metrics"].items():
            msg += f" | {name}: {value:.6f}"

        for name, value in summary["timings"].items():
            msg += f" | {name}: {value:.3f}s"

        print(msg)

    def log_summary(self) -> None:
        """Log summary of best metrics."""
        summary = self.tracker.get_summary()
        msg = "Summary:"

        for name, value in summary["best"].items():
            epoch = self.tracker.get_best_epoch(name)
            if value is not None:
                msg += f" | Best {name}: {value:.6f} (epoch {epoch})"

        print(msg)

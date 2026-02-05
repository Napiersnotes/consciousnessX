"""
Model comparison utilities for ConsciousnessX.

Compare different models and configurations.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class ModelResult:
    """Results from a single model evaluation."""

    name: str
    metrics: Dict[str, float]
    config: Dict[str, Any] = field(default_factory=dict)
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0


@dataclass
class ComparisonResult:
    """Results from comparing multiple models."""

    results: List[ModelResult]
    best_models: Dict[str, str] = field(default_factory=dict)
    rankings: Dict[str, List[str]] = field(default_factory=dict)

    def get_best_model(self, metric: str, lower_is_better: bool = True) -> Optional[ModelResult]:
        """
        Get best model for a given metric.

        Args:
            metric: Metric name
            lower_is_better: Whether lower values are better

        Returns:
            Best model result or None
        """
        if not self.results:
            return None

        sorted_results = sorted(
            self.results,
            key=lambda r: r.metrics.get(metric, np.inf if lower_is_better else -np.inf),
            reverse=not lower_is_better,
        )

        return sorted_results[0]

    def rank_models(self, metric: str, lower_is_better: bool = True) -> List[str]:
        """
        Rank models by a given metric.

        Args:
            metric: Metric name
            lower_is_better: Whether lower values are better

        Returns:
            List of model names in ranked order
        """
        if not self.results:
            return []

        sorted_results = sorted(
            self.results,
            key=lambda r: r.metrics.get(metric, np.inf if lower_is_better else -np.inf),
            reverse=not lower_is_better,
        )

        return [r.name for r in sorted_results]


class ModelComparator:
    """
    Compare multiple models across different metrics.
    """

    def __init__(self, lower_is_better: Optional[Dict[str, bool]] = None):
        """
        Initialize model comparator.

        Args:
            lower_is_better: Dictionary mapping metric names to whether lower is better
        """
        self.lower_is_better = lower_is_better or {
            "loss": True,
            "error": True,
            "mae": True,
            "mse": True,
            "accuracy": False,
            "f1": False,
            "precision": False,
            "recall": False,
            "phi": False,
            "consciousness_score": False,
        }

        self.results: List[ModelResult] = []

    def add_result(self, result: ModelResult) -> None:
        """
        Add a model result.

        Args:
            result: Model result to add
        """
        self.results.append(result)

    def add_results_from_dict(
        self,
        name: str,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Add result from dictionary.

        Args:
            name: Model name
            metrics: Dictionary of metrics
            config: Optional configuration dictionary
            **kwargs: Additional parameters for ModelResult
        """
        result = ModelResult(name=name, metrics=metrics, config=config or {}, **kwargs)
        self.add_result(result)

    def compare(self) -> ComparisonResult:
        """
        Compare all added models.

        Returns:
            Comparison result
        """
        comparison = ComparisonResult(results=self.results)

        # Find best model for each metric
        for metric in self._get_all_metrics():
            lower_better = self.lower_is_better.get(metric, True)
            best_model = comparison.get_best_model(metric, lower_better)

            if best_model:
                comparison.best_models[metric] = best_model.name
                comparison.rankings[metric] = comparison.rank_models(metric, lower_better)

        return comparison

    def _get_all_metrics(self) -> List[str]:
        """Get all unique metric names across all results."""
        metrics = set()
        for result in self.results:
            metrics.update(result.metrics.keys())
        return list(metrics)

    def get_summary_table(self) -> str:
        """
        Get summary table as string.

        Returns:
            Formatted summary table
        """
        comparison = self.compare()

        if not self.results:
            return "No results to compare"

        metrics = self._get_all_metrics()

        # Create table header
        header = ["Model"] + metrics + ["Train Time", "Inf Time"]
        rows = [header]

        # Add row for each model
        for result in self.results:
            row = [result.name]

            for metric in metrics:
                value = result.metrics.get(metric, None)
                if value is None:
                    row.append("N/A")
                else:
                    row.append(f"{value:.4f}")

            row.append(f"{result.training_time:.2f}s")
            row.append(f"{result.inference_time:.4f}s")
            rows.append(row)

        # Format table
        col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(header))]

        table_lines = []
        for row in rows:
            formatted_row = " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))
            table_lines.append(formatted_row)

        # Add separator line
        separator = "-+-".join("-" * w for w in col_widths)
        table_lines.insert(1, separator)

        return "\n".join(table_lines)

    def save_comparison(self, filepath: str) -> None:
        """
        Save comparison results to file.

        Args:
            filepath: Path to save file
        """
        comparison = self.compare()

        # Convert to serializable format
        data = {
            "results": [
                {
                    "name": r.name,
                    "metrics": r.metrics,
                    "config": r.config,
                    "training_time": r.training_time,
                    "inference_time": r.inference_time,
                    "memory_usage": r.memory_usage,
                }
                for r in comparison.results
            ],
            "best_models": comparison.best_models,
            "rankings": comparison.rankings,
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def load_comparison(self, filepath: str) -> None:
        """
        Load comparison results from file.

        Args:
            filepath: Path to load file from
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)

            self.results = [
                ModelResult(
                    name=r["name"],
                    metrics=r["metrics"],
                    config=r.get("config", {}),
                    training_time=r.get("training_time", 0.0),
                    inference_time=r.get("inference_time", 0.0),
                    memory_usage=r.get("memory_usage", 0.0),
                )
                for r in data["results"]
            ]
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def statistical_test(
        self, metric: str, model1: str, model2: str, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical test comparing two models.

        Args:
            metric: Metric to test
            model1: First model name
            model2: Second model name
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        # Find results for both models
        result1 = next((r for r in self.results if r.name == model1), None)
        result2 = next((r for r in self.results if r.name == model2), None)

        if result1 is None or result2 is None:
            raise ValueError("One or both models not found")

        value1 = result1.metrics.get(metric)
        value2 = result2.metrics.get(metric)

        if value1 is None or value2 is None:
            raise ValueError(f"Metric {metric} not found for one or both models")

        # Simple comparison (for single values)
        # For multiple runs, would use t-test or other statistical tests
        diff = abs(value1 - value2)
        better = model1 if value1 < value2 else model2

        return {
            "model1": model1,
            "model2": model2,
            "metric": metric,
            "value1": value1,
            "value2": value2,
            "difference": diff,
            "better_model": better,
            "significant": diff > alpha,  # Simplified significance test
        }

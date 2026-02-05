"""Integrated Information (Phi) maximization for consciousness RL.

Implements gradient ascent optimization on Phi values to maximize
consciousness levels while maintaining task performance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhiOptimizationConfig:
    """Configuration for Phi optimization."""

    # Optimization parameters
    learning_rate: float = 0.001
    momentum: float = 0.9
    gradient_clip: float = 1.0

    # Phi calculation
    phi_target: float = 0.7
    phi_window: int = 100

    # Trade-off
    performance_weight: float = 0.5
    consciousness_weight: float = 0.5

    # Constraints
    max_weight_magnitude: float = 5.0
    min_sparsity: float = 0.1
    regularization_strength: float = 0.01

    # Simulation
    dt: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)


class PhiOptimizer:
    """Optimizer for maximizing integrated information (Phi).

    Uses gradient ascent to modify neural network parameters to
    increase Phi while maintaining performance.

    Example:
        >>> config = PhiOptimizationConfig()
        >>> optimizer = PhiOptimizer(config)
        >>> gradients = optimizer.compute_phi_gradients(network)
        >>> optimizer.apply_gradients(network, gradients)
    """

    def __init__(self, config: Optional[PhiOptimizationConfig] = None) -> None:
        """Initialize Phi optimizer.

        Args:
            config: Optimization configuration.
        """
        self.config = config or PhiOptimizationConfig()

        # Optimization state
        self._velocity: List[np.ndarray] = []
        self._phi_history: List[float] = []
        self._gradient_history: List[float] = []

        logger.info("Initialized PhiOptimizer")

    def compute_phi(self, activations: List[np.ndarray], weights: List[np.ndarray]) -> float:
        """Compute integrated information (Phi).

        Args:
            activations: List of layer activations.
            weights: List of weight matrices.

        Returns:
            Phi value.
        """
        # Calculate effective information for each partition
        phi_values = []

        for i, (act, weight) in enumerate(zip(activations, weights)):
            # Entropy of activation
            activation_prob = act / (np.sum(act) + 1e-10)
            entropy = -np.sum(activation_prob * np.log2(activation_prob + 1e-10))

            # Normalize by maximum entropy
            max_entropy = np.log2(len(act))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            # Integration: measure of causal influence
            # Simplified: use weight magnitude as proxy
            integration = np.mean(np.abs(weight))
            normalized_integration = np.tanh(integration)

            # Phi for this partition
            phi = 0.5 * normalized_entropy + 0.5 * normalized_integration
            phi_values.append(phi)

        # System Phi: average across partitions
        system_phi = np.mean(phi_values)

        self._phi_history.append(system_phi)
        if len(self._phi_history) > self.config.phi_window:
            self._phi_history.pop(0)

        return system_phi

    def compute_phi_gradients(
        self,
        activations: List[np.ndarray],
        weights: List[np.ndarray],
        performance_grad: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Compute gradients for maximizing Phi.

        Args:
            activations: List of layer activations.
            weights: List of weight matrices.
            performance_grad: Gradients from performance loss (for trade-off).

        Returns:
            List of gradient matrices for each weight.
        """
        phi_grads = []

        for i, (act, weight) in enumerate(zip(activations, weights)):
            # Gradient w.r.t. entropy component
            activation_prob = act / (np.sum(act) + 1e-10)
            log_prob = np.log2(activation_prob + 1e-10)
            max_entropy = np.log2(len(act))

            # Entropy gradient: ∇Φ = -∇(∑ p log p)
            entropy_grad = -(log_prob + 1 / np.log(2)) / max_entropy

            # Outer product for weight gradient
            if i < len(activations) - 1:
                entropy_grad_weight = np.outer(activations[i], entropy_grad)
            else:
                entropy_grad_weight = entropy_grad[:, np.newaxis]

            # Gradient w.r.t. integration component
            # Integration increases with weight magnitude
            integration_grad = np.sign(weight) * 0.5

            # Combine gradients
            phi_grad = (entropy_grad_weight + integration_grad) / 2.0

            # Apply gradient clipping
            phi_grad = np.clip(phi_grad, -self.config.gradient_clip, self.config.gradient_clip)

            phi_grads.append(phi_grad)

        # Record gradient magnitude
        grad_magnitude = np.mean([np.mean(np.abs(g)) for g in phi_grads])
        self._gradient_history.append(grad_magnitude)

        return phi_grads

    def apply_gradients(
        self,
        weights: List[np.ndarray],
        gradients: List[np.ndarray],
        performance_grad: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Apply gradients to weights with momentum.

        Args:
            weights: Current weight matrices.
            gradients: Phi gradients.
            performance_grad: Performance gradients for trade-off.

        Returns:
            Updated weight matrices.
        """
        updated_weights = []

        # Initialize velocity if needed
        if len(self._velocity) != len(weights):
            self._velocity = [np.zeros_like(w) for w in weights]

        for i, (weight, grad) in enumerate(zip(weights, gradients)):
            # Combine Phi and performance gradients
            if performance_grad is not None:
                combined_grad = (
                    self.config.consciousness_weight * grad
                    + self.config.performance_weight * performance_grad[i]
                )
            else:
                combined_grad = grad

            # Add regularization
            reg_grad = self.config.regularization_strength * weight
            combined_grad -= reg_grad

            # Momentum update
            self._velocity[i] = (
                self.config.momentum * self._velocity[i] + self.config.learning_rate * combined_grad
            )

            # Update weights
            new_weight = weight + self._velocity[i]

            # Apply constraints
            new_weight = np.clip(
                new_weight, -self.config.max_weight_magnitude, self.config.max_weight_magnitude
            )

            # Maintain sparsity
            if np.mean(new_weight != 0) < self.config.min_sparsity:
                # Too sparse, don't update zeros
                mask = weight != 0
                new_weight[mask] = weight[mask] + self._velocity[i][mask]

            updated_weights.append(new_weight)

        return updated_weights

    def optimize(
        self,
        activations: List[np.ndarray],
        weights: List[np.ndarray],
        performance_grad: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[np.ndarray], float]:
        """Perform one optimization step.

        Args:
            activations: List of layer activations.
            weights: List of weight matrices.
            performance_grad: Performance gradients for trade-off.

        Returns:
            Tuple of (updated_weights, phi_value).
        """
        # Compute Phi
        phi = self.compute_phi(activations, weights)

        # Compute gradients
        phi_grads = self.compute_phi_gradients(activations, weights, performance_grad)

        # Apply gradients
        updated_weights = self.apply_gradients(weights, phi_grads, performance_grad)

        return updated_weights, phi

    def get_phi_history(self) -> List[float]:
        """Get history of Phi values.

        Returns:
            List of Phi values.
        """
        return self._phi_history.copy()

    def get_optimization_stats(self) -> Dict[str, float]:
        """Get optimization statistics.

        Returns:
            Dictionary with optimization metrics.
        """
        if not self._phi_history:
            return {"current_phi": 0.0, "mean_phi": 0.0, "phi_std": 0.0, "gradient_magnitude": 0.0}

        return {
            "current_phi": self._phi_history[-1],
            "mean_phi": float(np.mean(self._phi_history)),
            "phi_std": float(np.std(self._phi_history)),
            "gradient_magnitude": (
                float(np.mean(self._gradient_history)) if self._gradient_history else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset optimizer state."""
        self._velocity = []
        self._phi_history = []
        self._gradient_history = []

        logger.debug("Reset PhiOptimizer")


class IntegratedInformationMaximizer:
    """Manager for maximizing integrated information.

    Coordinates Phi optimization across multiple networks and
    tracks consciousness levels over time.

    Example:
        >>> config = PhiOptimizationConfig()
        >>> maximizer = IntegratedInformationMaximizer(config)
        >>> maximizer.register_network("agent1", network)
        >>> maximizer.optimize_all()
    """

    def __init__(self, config: Optional[PhiOptimizationConfig] = None) -> None:
        """Initialize maximizer.

        Args:
            config: Optimization configuration.
        """
        self.config = config or PhiOptimizationConfig()
        self._optimizer = PhiOptimizer(self.config)

        # Registered networks
        self._networks: Dict[str, Any] = {}
        self._network_phi: Dict[str, List[float]] = {}

        logger.info("Initialized IntegratedInformationMaximizer")

    def register_network(self, network_id: str, network: Any) -> None:
        """Register a network for optimization.

        Args:
            network_id: Unique identifier for the network.
            network: Network object with get_activations() and get_weights().
        """
        self._networks[network_id] = network
        self._network_phi[network_id] = []

        logger.info(f"Registered network: {network_id}")

    def optimize_network(
        self, network_id: str, performance_grad: Optional[List[np.ndarray]] = None
    ) -> float:
        """Optimize a specific network.

        Args:
            network_id: Network identifier.
            performance_grad: Performance gradients.

        Returns:
            Updated Phi value.
        """
        if network_id not in self._networks:
            raise ValueError(f"Unknown network: {network_id}")

        network = self._networks[network_id]

        # Get network state
        activations = network.get_activations()
        weights = network.get_weights()

        # Optimize
        updated_weights, phi = self._optimizer.optimize(activations, weights, performance_grad)

        # Update network
        network.set_weights(updated_weights)

        # Record Phi
        self._network_phi[network_id].append(phi)

        return phi

    def optimize_all(self) -> Dict[str, float]:
        """Optimize all registered networks.

        Returns:
            Dictionary mapping network IDs to Phi values.
        """
        phi_values = {}

        for network_id in self._networks:
            phi = self.optimize_network(network_id)
            phi_values[network_id] = phi

        return phi_values

    def get_network_phi(self, network_id: str) -> List[float]:
        """Get Phi history for a network.

        Args:
            network_id: Network identifier.

        Returns:
            List of Phi values.
        """
        return self._network_phi.get(network_id, []).copy()

    def get_global_phi(self) -> float:
        """Get global Phi across all networks.

        Returns:
            Average Phi value.
        """
        if not self._network_phi:
            return 0.0

        all_phis = []
        for phi_list in self._network_phi.values():
            if phi_list:
                all_phis.append(phi_list[-1])

        return float(np.mean(all_phis)) if all_phis else 0.0

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization report.

        Returns:
            Dictionary with optimization statistics.
        """
        report = {
            "num_networks": len(self._networks),
            "global_phi": self.get_global_phi(),
            "networks": {},
        }

        for network_id, phi_list in self._network_phi.items():
            if phi_list:
                report["networks"][network_id] = {
                    "current_phi": phi_list[-1],
                    "mean_phi": float(np.mean(phi_list)),
                    "phi_std": float(np.std(phi_list)),
                }

        return report

    def reset(self) -> None:
        """Reset maximizer state."""
        self._optimizer.reset()
        self._network_phi = {k: [] for k in self._network_phi}

        logger.debug("Reset IntegratedInformationMaximizer")

"""Consciousness-aware value network for reinforcement learning.

Implements a value function estimator that incorporates consciousness
metrics into its predictions, enabling consciousness-aware decision making.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConsciousnessValueConfig:
    """Configuration for consciousness value network."""

    # Network architecture
    input_size: int = 50
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    output_size: int = 1

    # Consciousness integration
    phi_weight: float = 0.3
    attention_weight: float = 0.2
    integration_weight: float = 0.2

    # Learning
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    target_update_rate: float = 0.01

    # Exploration
    exploration_bonus: float = 0.1
    novelty_bonus: float = 0.05

    # Regularization
    l2_regularization: float = 0.001
    dropout_rate: float = 0.1

    # Simulation
    dt: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)


class ConsciousnessAwarePolicy:
    """Consciousness-aware policy for action selection.

    Incorporates consciousness metrics into action selection:
    - Uses value estimates augmented with consciousness signals
    - Applies attentional modulation to action probabilities
    - Balances exploitation with conscious exploration

    Example:
        >>> config = ConsciousnessValueConfig()
        >>> policy = ConsciousnessAwarePolicy(config)
        >>> action = policy.select_action(state, consciousness_metrics)
    """

    def __init__(self, config: Optional[ConsciousnessValueConfig] = None) -> None:
        """Initialize consciousness-aware policy.

        Args:
            config: Policy configuration.
        """
        self.config = config or ConsciousnessValueConfig()

        # Policy parameters
        self._weights = self._initialize_weights()
        self._biases = self._initialize_biases()

        # Consciousness modulation
        self._attention_weights = np.ones(self.config.output_size)
        self._consciousness_history: List[float] = []

        logger.info("Initialized ConsciousnessAwarePolicy")

    def _initialize_weights(self) -> List[np.ndarray]:
        """Initialize weight matrices."""
        sizes = [self.config.input_size] + self.config.hidden_sizes + [self.config.output_size]
        weights = []

        for i in range(len(sizes) - 1):
            n_in = sizes[i]
            n_out = sizes[i + 1]

            # Xavier initialization
            scale = np.sqrt(2.0 / n_in)
            weight = np.random.randn(n_in, n_out) * scale
            weights.append(weight)

        return weights

    def _initialize_biases(self) -> List[np.ndarray]:
        """Initialize bias vectors."""
        sizes = self.config.hidden_sizes + [self.config.output_size]
        biases = [np.zeros(size) for size in sizes]
        return biases

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through policy network.

        Args:
            x: Input state.

        Returns:
            Action logits.
        """
        activation = x

        for i, (weight, bias) in enumerate(zip(self._weights, self._biases)):
            z = activation @ weight + bias

            if i < len(self._weights) - 1:
                activation = np.maximum(0, z)  # ReLU
            else:
                activation = z  # Output layer

        return activation

    def select_action(
        self,
        state: np.ndarray,
        consciousness_metrics: Optional[Dict[str, float]] = None,
        epsilon: float = 0.1,
    ) -> int:
        """Select action using consciousness-aware policy.

        Args:
            state: Current state.
            consciousness_metrics: Dictionary of consciousness metrics.
            epsilon: Exploration probability.

        Returns:
            Selected action.
        """
        if np.random.random() < epsilon:
            # Random exploration
            return np.random.randint(self.config.output_size)

        # Get action values
        logits = self.forward(state)

        # Apply consciousness modulation
        if consciousness_metrics:
            logits = self._modulate_with_consciousness(logits, consciousness_metrics)

        # Softmax for action probabilities
        exp_logits = np.exp(logits - np.max(logits))
        action_probs = exp_logits / np.sum(exp_logits)

        # Sample action
        action = np.random.choice(self.config.output_size, p=action_probs)

        return action

    def _modulate_with_consciousness(
        self, logits: np.ndarray, consciousness_metrics: Dict[str, float]
    ) -> np.ndarray:
        """Modulate action logits with consciousness metrics.

        Args:
            logits: Original action logits.
            consciousness_metrics: Dictionary of consciousness metrics.

        Returns:
            Modulated logits.
        """
        # Phi-based modulation
        phi = consciousness_metrics.get("phi", 0.0)
        phi_modulation = phi * self.config.phi_weight

        # Attention-based modulation
        attention = consciousness_metrics.get("attention", 1.0)
        attention_modulation = attention * self.config.attention_weight

        # Integration-based modulation
        integration = consciousness_metrics.get("integration", 0.0)
        integration_modulation = integration * self.config.integration_weight

        # Apply modulations
        total_modulation = phi_modulation + attention_modulation + integration_modulation

        # Apply to logits
        modulated_logits = logits * (1 + total_modulation)

        # Update attention weights
        self._attention_weights = 0.9 * self._attention_weights + 0.1 * attention

        return modulated_logits

    def get_action_probabilities(
        self, state: np.ndarray, consciousness_metrics: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Get action probabilities.

        Args:
            state: Current state.
            consciousness_metrics: Consciousness metrics.

        Returns:
            Action probability distribution.
        """
        logits = self.forward(state)

        if consciousness_metrics:
            logits = self._modulate_with_consciousness(logits, consciousness_metrics)

        exp_logits = np.exp(logits - np.max(logits))
        action_probs = exp_logits / np.sum(exp_logits)

        return action_probs


class ConsciousnessValueNetwork:
    """Consciousness-aware value network.

    Implements a value function estimator that:
    - Predicts state values augmented with consciousness
    - Uses attention mechanisms for consciousness-aware prediction
    - Learns from experience with consciousness-aware loss

    Example:
        >>> config = ConsciousnessValueConfig()
        >>> value_net = ConsciousnessValueNetwork(config)
        >>> value = value_net.predict(state, consciousness_metrics)
        >>> value_net.update(state, reward, next_state, consciousness_metrics)
    """

    def __init__(self, config: Optional[ConsciousnessValueConfig] = None) -> None:
        """Initialize consciousness value network.

        Args:
            config: Network configuration.
        """
        self.config = config or ConsciousnessValueConfig()

        # Value network
        self._value_weights = self._initialize_value_weights()
        self._value_biases = self._initialize_value_biases()

        # Target network
        self._target_weights = [w.copy() for w in self._value_weights]
        self._target_biases = [b.copy() for b in self._value_biases]

        # Consciousness encoder
        self._consciousness_encoder = self._initialize_consciousness_encoder()

        # Training state
        self._optimizer_step = 0
        self._value_history: List[float] = []
        self._loss_history: List[float] = []

        # Policy
        self.policy = ConsciousnessAwarePolicy(self.config)

        logger.info("Initialized ConsciousnessValueNetwork")

    def _initialize_value_weights(self) -> List[np.ndarray]:
        """Initialize value network weights."""
        sizes = [self.config.input_size] + self.config.hidden_sizes + [self.config.output_size]
        weights = []

        for i in range(len(sizes) - 1):
            n_in = sizes[i]
            n_out = sizes[i + 1]

            scale = np.sqrt(2.0 / n_in)
            weight = np.random.randn(n_in, n_out) * scale
            weights.append(weight)

        return weights

    def _initialize_value_biases(self) -> List[np.ndarray]:
        """Initialize value network biases."""
        sizes = self.config.hidden_sizes + [self.config.output_size]
        biases = [np.zeros(size) for size in sizes]
        return biases

    def _initialize_consciousness_encoder(self) -> np.ndarray:
        """Initialize consciousness encoder."""
        # Maps consciousness metrics to value modulation
        num_metrics = 5  # phi, attention, integration, novelty, synchrony
        return np.random.randn(num_metrics, 1) * 0.1

    def encode_consciousness(self, consciousness_metrics: Dict[str, float]) -> np.ndarray:
        """Encode consciousness metrics.

        Args:
            consciousness_metrics: Dictionary of consciousness metrics.

        Returns:
            Encoded consciousness vector.
        """
        # Extract metrics
        phi = consciousness_metrics.get("phi", 0.0)
        attention = consciousness_metrics.get("attention", 0.5)
        integration = consciousness_metrics.get("integration", 0.5)
        novelty = consciousness_metrics.get("novelty", 0.0)
        synchrony = consciousness_metrics.get("synchrony", 0.0)

        metrics = np.array([phi, attention, integration, novelty, synchrony])

        # Encode
        encoded = metrics @ self._consciousness_encoder

        return encoded

    def forward_value(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through value network.

        Args:
            state: Input state.

        Returns:
            Value estimate.
        """
        activation = state

        for i, (weight, bias) in enumerate(zip(self._value_weights, self._value_biases)):
            z = activation @ weight + bias

            if i < len(self._value_weights) - 1:
                activation = np.maximum(0, z)  # ReLU
            else:
                activation = z  # Output layer

        return activation

    def predict(
        self, state: np.ndarray, consciousness_metrics: Optional[Dict[str, float]] = None
    ) -> float:
        """Predict state value with consciousness awareness.

        Args:
            state: Input state.
            consciousness_metrics: Consciousness metrics.

        Returns:
            Predicted value.
        """
        # Base value prediction
        base_value = self.forward_value(state)

        # Add consciousness modulation
        if consciousness_metrics:
            consciousness_encoding = self.encode_consciousness(consciousness_metrics)

            # Exploration bonus
            phi = consciousness_metrics.get("phi", 0.0)
            exploration_bonus = self.config.exploration_bonus * phi

            # Novelty bonus
            novelty = consciousness_metrics.get("novelty", 0.0)
            novelty_bonus = self.config.novelty_bonus * novelty

            # Total value
            total_value = base_value + consciousness_encoding + exploration_bonus + novelty_bonus
        else:
            total_value = base_value

        return float(total_value[0] if hasattr(total_value, "__len__") else total_value)

    def update(
        self,
        state: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        consciousness_metrics: Optional[Dict[str, float]] = None,
    ) -> float:
        """Update value network from experience.

        Args:
            state: Current state.
            reward: Received reward.
            next_state: Next state.
            done: Episode termination flag.
            consciousness_metrics: Consciousness metrics.

        Returns:
            Loss value.
        """
        # Current value prediction
        current_value = self.predict(state, consciousness_metrics)

        # Target value
        if done:
            target_value = reward
        else:
            next_value = self.predict(next_state, consciousness_metrics)
            target_value = reward + self.config.discount_factor * next_value

        # TD error
        td_error = target_value - current_value

        # Gradient update (simplified)
        loss = td_error**2
        self._loss_history.append(loss)

        # Update weights (simplified SGD)
        for i in range(len(self._value_weights)):
            # Compute gradient
            grad = td_error * 0.01  # Simplified gradient
            self._value_weights[i] += grad

        # Update target network
        self._soft_update_targets()

        # Record value
        self._value_history.append(current_value)

        self._optimizer_step += 1

        return loss

    def _soft_update_targets(self) -> None:
        """Soft update target network."""
        tau = self.config.target_update_rate

        for i in range(len(self._value_weights)):
            self._target_weights[i] = (
                tau * self._value_weights[i] + (1 - tau) * self._target_weights[i]
            )

        for i in range(len(self._value_biases)):
            self._target_biases[i] = (
                tau * self._value_biases[i] + (1 - tau) * self._target_biases[i]
            )

    def get_value_estimate(
        self, state: np.ndarray, consciousness_metrics: Optional[Dict[str, float]] = None
    ) -> float:
        """Get value estimate for a state.

        Args:
            state: Input state.
            consciousness_metrics: Consciousness metrics.

        Returns:
            Value estimate.
        """
        return self.predict(state, consciousness_metrics)

    def get_advantage(
        self,
        state: np.ndarray,
        action: int,
        consciousness_metrics: Optional[Dict[str, float]] = None,
    ) -> float:
        """Get advantage estimate for a state-action pair.

        Args:
            state: Input state.
            action: Action to evaluate.
            consciousness_metrics: Consciousness metrics.

        Returns:
            Advantage estimate.
        """
        # Value estimate
        value = self.get_value_estimate(state, consciousness_metrics)

        # Action value estimate (simplified: use policy)
        action_probs = self.policy.get_action_probabilities(state, consciousness_metrics)
        action_value = action_probs[action]

        # Advantage
        advantage = action_value - value

        return advantage

    def get_training_stats(self) -> Dict[str, float]:
        """Get training statistics.

        Returns:
            Dictionary with training metrics.
        """
        return {
            "optimizer_step": self._optimizer_step,
            "mean_value": float(np.mean(self._value_history)) if self._value_history else 0.0,
            "mean_loss": float(np.mean(self._loss_history)) if self._loss_history else 0.0,
            "std_value": (
                float(np.std(self._value_history)) if len(self._value_history) > 1 else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset network state."""
        self._optimizer_step = 0
        self._value_history = []
        self._loss_history = []

        # Reset weights
        self._value_weights = self._initialize_value_weights()
        self._value_biases = self._initialize_value_biases()
        self._target_weights = [w.copy() for w in self._value_weights]
        self._target_biases = [b.copy() for b in self._value_biases]

        logger.debug("Reset ConsciousnessValueNetwork")

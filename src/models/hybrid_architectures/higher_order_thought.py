"""Higher-Order Thought (HOT) theory implementation for consciousness.

Implements Rosenthal-style HOT models where consciousness arises from
higher-order representations of first-order states, with self-awareness
and meta-representational structures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class HOTLevel(Enum):
    """Levels of higher-order thought."""

    FIRST_ORDER = "first_order"
    SECOND_ORDER = "second_order"
    THIRD_ORDER = "third_order"
    META = "meta"


@dataclass
class MetaRepresentation:
    """Meta-representation of a mental state."""

    level: HOTLevel
    content: np.ndarray
    source_representation: Optional[str] = None
    confidence: float = 1.0
    timestamp: float = 0.0


@dataclass
class HOTConfig:
    """Configuration for Higher-Order Thought model."""

    # Representation dimensions
    first_order_dim: int = 50
    second_order_dim: int = 50
    third_order_dim: int = 20
    num_first_order: int = 10
    num_second_order: int = 5

    # HOT formation
    formation_threshold: float = 0.5
    integration_weight: float = 0.3
    abstraction_weight: float = 0.5

    # Self-awareness
    self_monitoring_rate: float = 0.1
    reflection_strength: float = 0.5

    # Consciousness
    consciousness_threshold: float = 0.7
    access_consciousness_duration: float = 100.0  # ms

    # Simulation
    dt: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)


class HigherOrderThought:
    """Higher-Order Thought consciousness model.

    Implements Rosenthal's HOT theory where consciousness requires
    higher-order representations of first-order mental states:
    - First-order: Basic perceptions, thoughts, feelings
    - Second-order: Representations of first-order states
    - Third-order: Representations of second-order states
    - Meta: Global self-awareness and reflection

    Example:
        >>> config = HOTConfig()
        >>> hot = HigherOrderThought(config)
        >>> hot.add_first_order_state("perception", data)
        >>> consciousness = hot.compute_consciousness()
    """

    def __init__(self, config: Optional[HOTConfig] = None) -> None:
        """Initialize HOT model.

        Args:
            config: Model configuration.
        """
        self.config = config or HOTConfig()

        # First-order states
        self._first_order_states: Dict[str, MetaRepresentation] = {}
        self._first_order_matrix: np.ndarray = np.zeros(
            self.config.num_first_order * self.config.first_order_dim
        )

        # Transformation matrices
        self._first_to_second = self._initialize_transformation(
            self.config.first_order_dim, self.config.second_order_dim, self.config.num_first_order
        )
        self._second_to_third = self._initialize_transformation(
            self.config.second_order_dim, self.config.third_order_dim, self.config.num_second_order
        )

        # Second-order representations
        self._second_order_states: List[MetaRepresentation] = []

        # Third-order representations
        self._third_order_states: List[MetaRepresentation] = []

        # Meta-awareness
        self._meta_awareness: Optional[MetaRepresentation] = None
        self._self_monitoring_active: bool = False

        # Consciousness state
        self._conscious_state = False
        self._consciousness_level = 0.0
        self._conscious_history: List[float] = []

        logger.info("Initialized HigherOrderThought model")

    def _initialize_transformation(
        self, input_dim: int, output_dim: int, num_inputs: int
    ) -> np.ndarray:
        """Initialize transformation matrix.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            num_inputs: Number of input states.

        Returns:
            Transformation matrix.
        """
        matrix = np.random.randn(num_inputs * input_dim, output_dim) * 0.1

        # Normalize
        matrix = matrix / (np.linalg.norm(matrix, axis=0, keepdims=True) + 1e-10)

        return matrix

    def add_first_order_state(self, state_id: str, content: np.ndarray) -> None:
        """Add a first-order mental state.

        Args:
            state_id: Unique identifier for the state.
            content: State content vector.
        """
        if len(content) != self.config.first_order_dim:
            raise ValueError(f"Content dimension must be {self.config.first_order_dim}")

        representation = MetaRepresentation(
            level=HOTLevel.FIRST_ORDER, content=content.copy(), source_representation=state_id
        )

        self._first_order_states[state_id] = representation

        # Update first-order matrix
        idx = list(self._first_order_states.keys()).index(state_id)
        start = idx * self.config.first_order_dim
        end = start + self.config.first_order_dim
        self._first_order_matrix[start:end] = content

        logger.debug(f"Added first-order state: {state_id}")

    def _form_second_order_representations(self) -> List[MetaRepresentation]:
        """Form second-order representations of first-order states.

        Returns:
            List of second-order representations.
        """
        if not self._first_order_states:
            return []

        second_order_reps = []

        # Create second-order states for combinations of first-order states
        first_order_ids = list(self._first_order_states.keys())

        for i in range(min(self.config.num_second_order, len(first_order_ids))):
            # Select first-order state
            fo_id = first_order_ids[i]
            fo_state = self._first_order_states[fo_id]

            # Transform to second-order
            fo_idx = first_order_ids.index(fo_id)
            start = fo_idx * self.config.first_order_dim
            end = start + self.config.first_order_dim

            slice_matrix = self._first_to_second[start:end]
            transformed = fo_state.content @ slice_matrix

            # Normalize
            transformed = transformed / (np.linalg.norm(transformed) + 1e-10)

            rep = MetaRepresentation(
                level=HOTLevel.SECOND_ORDER,
                content=transformed,
                source_representation=fo_id,
                confidence=np.mean(np.abs(fo_state.content)),
            )

            second_order_reps.append(rep)

        self._second_order_states = second_order_reps

        return second_order_reps

    def _form_third_order_representations(self) -> List[MetaRepresentation]:
        """Form third-order representations of second-order states.

        Returns:
            List of third-order representations.
        """
        if not self._second_order_states:
            return []

        third_order_reps = []

        for i, second_rep in enumerate(self._second_order_states):
            # Transform to third-order
            transformed = second_rep.content @ self._second_to_third

            # Normalize
            transformed = transformed / (np.linalg.norm(transformed) + 1e-10)

            rep = MetaRepresentation(
                level=HOTLevel.THIRD_ORDER,
                content=transformed,
                source_representation=f"second_{i}",
                confidence=second_rep.confidence * self.config.integration_weight,
            )

            third_order_reps.append(rep)

        self._third_order_states = third_order_reps

        return third_order_reps

    def _form_meta_awareness(self) -> MetaRepresentation:
        """Form meta-awareness from third-order representations.

        Returns:
            Meta-awareness representation.
        """
        if not self._third_order_states:
            # Create empty meta-awareness
            return MetaRepresentation(
                level=HOTLevel.META, content=np.zeros(self.config.third_order_dim)
            )

        # Aggregate third-order representations
        aggregated = np.zeros(self.config.third_order_dim)

        for to_rep in self._third_order_states:
            aggregated += to_rep.content * to_rep.confidence

        # Normalize
        aggregated = aggregated / (np.linalg.norm(aggregated) + 1e-10)

        meta_rep = MetaRepresentation(
            level=HOTLevel.META,
            content=aggregated,
            confidence=np.mean([r.confidence for r in self._third_order_states]),
        )

        self._meta_awareness = meta_rep

        return meta_rep

    def compute_consciousness(self) -> float:
        """Compute consciousness level based on HOT structure.

        Returns:
            Consciousness level (0 to 1).
        """
        # Form representations
        self._form_second_order_representations()
        self._form_third_order_representations()
        self._form_meta_awareness()

        # Calculate consciousness based on:
        # 1. Strength of meta-awareness
        # 2. Integration between levels
        # 3. Coherence of representations

        meta_strength = np.linalg.norm(self._meta_awareness.content) if self._meta_awareness else 0

        # Integration: measure of causal influence
        integration = np.mean(
            [np.mean(np.abs(self._first_to_second)), np.mean(np.abs(self._second_to_third))]
        )

        # Coherence: average confidence
        coherence = 0.0
        if self._second_order_states and self._third_order_states:
            coherence = (
                np.mean([r.confidence for r in self._second_order_states])
                + np.mean([r.confidence for r in self._third_order_states])
            ) / 2

        # Combined consciousness
        consciousness = 0.4 * meta_strength + 0.3 * integration + 0.3 * coherence

        consciousness = np.clip(consciousness, 0.0, 1.0)

        self._consciousness_level = consciousness
        self._conscious_history.append(consciousness)

        # Check consciousness threshold
        self._conscious_state = consciousness > self.config.consciousness_threshold

        return consciousness

    def self_reflect(self) -> Dict[str, float]:
        """Perform self-reflection on current mental state.

        Returns:
            Dictionary with reflection results.
        """
        # Monitor current state
        if np.random.random() < self.config.self_monitoring_rate:
            self._self_monitoring_active = True

        if not self._self_monitoring_active:
            return {"monitoring_active": False, "reflection_strength": 0.0}

        # Calculate reflection strength
        reflection = self.config.reflection_strength * self._consciousness_level

        # Update meta-awareness with reflection
        if self._meta_awareness:
            noise = np.random.normal(0, 0.01, self._meta_awareness.content.shape)
            self._meta_awareness.content += reflection * noise

        return {
            "monitoring_active": True,
            "reflection_strength": reflection,
            "consciousness_level": self._consciousness_level,
        }

    def get_first_order_states(self) -> Dict[str, np.ndarray]:
        """Get first-order states.

        Returns:
            Dictionary mapping state IDs to content.
        """
        return {k: v.content.copy() for k, v in self._first_order_states.items()}

    def get_second_order_states(self) -> List[Dict[str, Any]]:
        """Get second-order states.

        Returns:
            List of second-order state dictionaries.
        """
        return [
            {
                "content": rep.content.copy(),
                "source": rep.source_representation,
                "confidence": rep.confidence,
            }
            for rep in self._second_order_states
        ]

    def get_third_order_states(self) -> List[Dict[str, Any]]:
        """Get third-order states.

        Returns:
            List of third-order state dictionaries.
        """
        return [
            {"content": rep.content.copy(), "confidence": rep.confidence}
            for rep in self._third_order_states
        ]

    def get_meta_awareness(self) -> Optional[Dict[str, Any]]:
        """Get meta-awareness.

        Returns:
            Meta-awareness dictionary or None.
        """
        if not self._meta_awareness:
            return None

        return {
            "content": self._meta_awareness.content.copy(),
            "confidence": self._meta_awareness.confidence,
        }

    def get_consciousness_metrics(self) -> Dict[str, float]:
        """Get consciousness metrics.

        Returns:
            Dictionary with consciousness metrics.
        """
        return {
            "consciousness_level": self._consciousness_level,
            "conscious_state": float(self._conscious_state),
            "num_first_order": len(self._first_order_states),
            "num_second_order": len(self._second_order_states),
            "num_third_order": len(self._third_order_states),
            "has_meta_awareness": self._meta_awareness is not None,
            "monitoring_active": self._self_monitoring_active,
        }

    def get_consciousness_history(self, n: int = 100) -> List[float]:
        """Get consciousness level history.

        Args:
            n: Number of recent values.

        Returns:
            List of consciousness levels.
        """
        return self._conscious_history[-n:]

    def reset(self) -> None:
        """Reset HOT model state."""
        self._first_order_states = {}
        self._first_order_matrix = np.zeros(
            self.config.num_first_order * self.config.first_order_dim
        )
        self._second_order_states = []
        self._third_order_states = []
        self._meta_awareness = None
        self._self_monitoring_active = False
        self._conscious_state = False
        self._consciousness_level = 0.0
        self._conscious_history = []

        logger.debug("Reset HigherOrderThought model")

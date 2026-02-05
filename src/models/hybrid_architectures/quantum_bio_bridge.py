"""Quantum-classical-biological interface layer.

Implements a bidirectional interface between quantum systems,
classical neural networks, and biological simulations for
integrated consciousness modeling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class InterfaceDirection(Enum):
    """Direction of interface signal."""

    QUANTUM_TO_CLASSICAL = "quantum_to_classical"
    QUANTUM_TO_BIOLOGICAL = "quantum_to_biological"
    CLASSICAL_TO_QUANTUM = "classical_to_quantum"
    CLASSICAL_TO_BIOLOGICAL = "classical_to_biological"
    BIOLOGICAL_TO_QUANTUM = "biological_to_quantum"
    BIOLOGICAL_TO_CLASSICAL = "biological_to_classical"


@dataclass
class InterfaceSignal:
    """Signal passing through the quantum-biological interface."""

    timestamp: float
    source: str
    destination: str
    direction: InterfaceDirection
    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_signal_strength(self) -> float:
        """Get signal strength."""
        return float(np.linalg.norm(self.data))

    def normalize(self) -> InterfaceSignal:
        """Normalize signal data."""
        norm = np.linalg.norm(self.data)
        if norm > 0:
            self.data = self.data / norm
        return self


@dataclass
class QuantumBioConfig:
    """Configuration for quantum-biological bridge."""

    # Interface dimensions
    quantum_dim: int = 8
    classical_dim: int = 100
    biological_dim: int = 50

    # Transformation parameters
    quantum_to_classical_gain: float = 1.0
    classical_to_quantum_gain: float = 1.0
    biological_to_quantum_gain: float = 0.5
    quantum_to_biological_gain: float = 0.5

    # Coupling
    coupling_strength: float = 0.3
    coherence_threshold: float = 0.7
    entanglement_probability: float = 0.1

    # Noise and decoherence
    thermal_noise: float = 0.01
    decoherence_rate: float = 0.001

    # Buffer management
    buffer_size: int = 100
    signal_timeout: float = 10.0  # seconds

    # Simulation
    dt: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)


class QuantumBioBridge:
    """Quantum-classical-biological interface.

    Implements a bidirectional interface that:
    - Translates quantum states to classical representations
    - Maps classical signals to quantum operations
    - Interfaces with biological neuronal simulations
    - Maintains coherence across all three domains

    Example:
        >>> config = QuantumBioConfig()
        >>> bridge = QuantumBioBridge(config)
        >>> signal = bridge.quantum_to_classical(quantum_state)
        >>> biological_response = bridge.classical_to_biological(classical_state)
    """

    def __init__(self, config: Optional[QuantumBioConfig] = None) -> None:
        """Initialize quantum-biological bridge.

        Args:
            config: Bridge configuration.
        """
        self.config = config or QuantumBioConfig()

        # Transformation matrices
        self._quantum_to_classical = self._initialize_quantum_to_classical()
        self._classical_to_quantum = self._initialize_classical_to_quantum()
        self._quantum_to_biological = self._initialize_quantum_to_biological()
        self._biological_to_quantum = self._initialize_biological_to_quantum()
        self._classical_to_biological = self._initialize_classical_to_biological()
        self._biological_to_classical = self._initialize_biological_to_classical()

        # Interface state
        self._signal_buffer: List[InterfaceSignal] = []
        self._coherence_timers: Dict[str, float] = {}
        self._entanglement_pairs: List[Tuple[str, str]] = []

        # Metrics
        self._signal_counts: Dict[InterfaceDirection, int] = {
            direction: 0 for direction in InterfaceDirection
        }
        self._total_signal_energy: float = 0.0

        logger.info("Initialized QuantumBioBridge")

    def _initialize_quantum_to_classical(self) -> np.ndarray:
        """Initialize quantum to classical transformation matrix."""
        # Pauli measurements as basis
        n_q = self.config.quantum_dim
        n_c = self.config.classical_dim

        # Create measurement operators
        transform = np.random.randn(n_q * 2, n_c) * 0.1

        # Normalize
        transform = transform / (np.linalg.norm(transform, axis=0, keepdims=True) + 1e-10)

        return transform

    def _initialize_classical_to_quantum(self) -> np.ndarray:
        """Initialize classical to quantum transformation matrix."""
        n_c = self.config.classical_dim
        n_q = self.config.quantum_dim

        # State preparation operators
        transform = np.random.randn(n_c, n_q * 2) * 0.1

        # Normalize
        transform = transform / (np.linalg.norm(transform, axis=1, keepdims=True) + 1e-10)

        return transform

    def _initialize_quantum_to_biological(self) -> np.ndarray:
        """Initialize quantum to biological transformation matrix."""
        n_q = self.config.quantum_dim
        n_b = self.config.biological_dim

        # Microtubule coupling
        transform = np.random.randn(n_q * 2, n_b) * 0.1

        return transform

    def _initialize_biological_to_quantum(self) -> np.ndarray:
        """Initialize biological to quantum transformation matrix."""
        n_b = self.config.biological_dim
        n_q = self.config.quantum_dim

        # Neuronal quantum coupling
        transform = np.random.randn(n_b, n_q * 2) * 0.1

        return transform

    def _initialize_classical_to_biological(self) -> np.ndarray:
        """Initialize classical to biological transformation matrix."""
        n_c = self.config.classical_dim
        n_b = self.config.biological_dim

        # Neural encoding
        transform = np.random.randn(n_c, n_b) * 0.1

        return transform

    def _initialize_biological_to_classical(self) -> np.ndarray:
        """Initialize biological to classical transformation matrix."""
        n_b = self.config.biological_dim
        n_c = self.config.classical_dim

        # Population decoding
        transform = np.random.randn(n_b, n_c) * 0.1

        return transform

    def quantum_to_classical(
        self, quantum_state: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> InterfaceSignal:
        """Translate quantum state to classical representation.

        Args:
            quantum_state: Quantum state vector (complex).
            metadata: Optional metadata.

        Returns:
            InterfaceSignal with classical data.
        """
        # Separate real and imaginary parts
        state_real = np.real(quantum_state)
        state_imag = np.imag(quantum_state)
        combined = np.concatenate([state_real, state_imag])

        # Transform to classical
        classical_data = combined @ self._quantum_to_classical

        # Apply gain
        classical_data *= self.config.quantum_to_classical_gain

        # Add thermal noise
        noise = np.random.normal(0, self.config.thermal_noise, classical_data.shape)
        classical_data += noise

        # Create signal
        signal = InterfaceSignal(
            timestamp=0.0,  # Will be set by simulation
            source="quantum",
            destination="classical",
            direction=InterfaceDirection.QUANTUM_TO_CLASSICAL,
            data=classical_data,
            metadata=metadata or {},
        )

        # Record signal
        self._record_signal(signal)

        return signal

    def classical_to_quantum(
        self, classical_data: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> InterfaceSignal:
        """Translate classical data to quantum state.

        Args:
            classical_data: Classical data vector.
            metadata: Optional metadata.

        Returns:
            InterfaceSignal with quantum state.
        """
        # Transform to quantum
        quantum_flat = classical_data @ self._classical_to_quantum

        # Separate real and imaginary parts
        n = len(quantum_flat) // 2
        real_part = quantum_flat[:n]
        imag_part = quantum_flat[n:]

        # Form complex state
        quantum_state = real_part + 1j * imag_part

        # Apply gain
        quantum_state *= self.config.classical_to_quantum_gain

        # Normalize
        norm = np.linalg.norm(quantum_state)
        if norm > 0:
            quantum_state = quantum_state / norm

        # Create signal
        signal = InterfaceSignal(
            timestamp=0.0,
            source="classical",
            destination="quantum",
            direction=InterfaceDirection.CLASSICAL_TO_QUANTUM,
            data=quantum_state,
            metadata=metadata or {},
        )

        # Record signal
        self._record_signal(signal)

        return signal

    def quantum_to_biological(
        self, quantum_state: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> InterfaceSignal:
        """Translate quantum state to biological representation.

        Args:
            quantum_state: Quantum state vector.
            metadata: Optional metadata.

        Returns:
            InterfaceSignal with biological data.
        """
        # Separate real and imaginary parts
        state_real = np.real(quantum_state)
        state_imag = np.imag(quantum_state)
        combined = np.concatenate([state_real, state_imag])

        # Transform to biological
        biological_data = combined @ self._quantum_to_biological

        # Apply gain
        biological_data *= self.config.quantum_to_biological_gain

        # Create signal
        signal = InterfaceSignal(
            timestamp=0.0,
            source="quantum",
            destination="biological",
            direction=InterfaceDirection.QUANTUM_TO_BIOLOGICAL,
            data=biological_data,
            metadata=metadata or {},
        )

        # Record signal
        self._record_signal(signal)

        return signal

    def biological_to_quantum(
        self, biological_data: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> InterfaceSignal:
        """Translate biological data to quantum state.

        Args:
            biological_data: Biological activity vector.
            metadata: Optional metadata.

        Returns:
            InterfaceSignal with quantum state.
        """
        # Transform to quantum
        quantum_flat = biological_data @ self._biological_to_quantum

        # Separate real and imaginary parts
        n = len(quantum_flat) // 2
        real_part = quantum_flat[:n]
        imag_part = quantum_flat[n:]

        # Form complex state
        quantum_state = real_part + 1j * imag_part

        # Apply gain
        quantum_state *= self.config.biological_to_quantum_gain

        # Normalize
        norm = np.linalg.norm(quantum_state)
        if norm > 0:
            quantum_state = quantum_state / norm

        # Create signal
        signal = InterfaceSignal(
            timestamp=0.0,
            source="biological",
            destination="quantum",
            direction=InterfaceDirection.BIOLOGICAL_TO_QUANTUM,
            data=quantum_state,
            metadata=metadata or {},
        )

        # Record signal
        self._record_signal(signal)

        return signal

    def classical_to_biological(
        self, classical_data: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> InterfaceSignal:
        """Translate classical data to biological representation.

        Args:
            classical_data: Classical data vector.
            metadata: Optional metadata.

        Returns:
            InterfaceSignal with biological data.
        """
        # Transform to biological
        biological_data = classical_data @ self._classical_to_biological

        # Create signal
        signal = InterfaceSignal(
            timestamp=0.0,
            source="classical",
            destination="biological",
            direction=InterfaceDirection.CLASSICAL_TO_BIOLOGICAL,
            data=biological_data,
            metadata=metadata or {},
        )

        # Record signal
        self._record_signal(signal)

        return signal

    def biological_to_classical(
        self, biological_data: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> InterfaceSignal:
        """Translate biological data to classical representation.

        Args:
            biological_data: Biological activity vector.
            metadata: Optional metadata.

        Returns:
            InterfaceSignal with classical data.
        """
        # Transform to classical
        classical_data = biological_data @ self._biological_to_classical

        # Create signal
        signal = InterfaceSignal(
            timestamp=0.0,
            source="biological",
            destination="classical",
            direction=InterfaceDirection.BIOLOGICAL_TO_CLASSICAL,
            data=classical_data,
            metadata=metadata or {},
        )

        # Record signal
        self._record_signal(signal)

        return signal

    def _record_signal(self, signal: InterfaceSignal) -> None:
        """Record signal in buffer."""
        self._signal_buffer.append(signal)

        # Update counts
        self._signal_counts[signal.direction] += 1

        # Update energy
        self._total_signal_energy += signal.get_signal_strength() ** 2

        # Manage buffer size
        if len(self._signal_buffer) > self.config.buffer_size:
            self._signal_buffer.pop(0)

    def entangle(self, source: str, destination: str) -> bool:
        """Entangle two interface endpoints.

        Args:
            source: Source endpoint.
            destination: Destination endpoint.

        Returns:
            True if entanglement created, False otherwise.
        """
        if np.random.random() < self.config.entanglement_probability:
            self._entanglement_pairs.append((source, destination))

            # Initialize coherence timer
            key = f"{source}_{destination}"
            self._coherence_timers[key] = 1.0

            logger.info(f"Entangled {source} with {destination}")
            return True

        return False

    def disentangle(self, source: str, destination: str) -> bool:
        """Remove entanglement between endpoints.

        Args:
            source: Source endpoint.
            destination: Destination endpoint.

        Returns:
            True if entanglement removed, False otherwise.
        """
        pair = (source, destination)
        if pair in self._entanglement_pairs:
            self._entanglement_pairs.remove(pair)

            # Remove coherence timer
            key = f"{source}_{destination}"
            if key in self._coherence_timers:
                del self._coherence_timers[key]

            logger.info(f"Disentangled {source} from {destination}")
            return True

        return False

    def get_coherence(self, source: str, destination: str) -> float:
        """Get coherence between endpoints.

        Args:
            source: Source endpoint.
            destination: Destination endpoint.

        Returns:
            Coherence value (0 to 1).
        """
        key = f"{source}_{destination}"
        return self._coherence_timers.get(key, 0.0)

    def update_coherence(self, dt: float = 1.0) -> None:
        """Update coherence timers.

        Args:
            dt: Time step.
        """
        keys_to_remove = []

        for key in self._coherence_timers:
            # Decay coherence
            self._coherence_timers[key] *= 1 - self.config.decoherence_rate

            # Remove if below threshold
            if self._coherence_timers[key] < self.config.coherence_threshold:
                keys_to_remove.append(key)

        # Remove expired coherences
        for key in keys_to_remove:
            del self._coherence_timers[key]
            # Remove entanglement pairs
            source, destination = key.split("_")
            pair = (source, destination)
            if pair in self._entanglement_pairs:
                self._entanglement_pairs.remove(pair)

    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get signal transmission statistics.

        Returns:
            Dictionary with signal statistics.
        """
        direction_stats = {}
        for direction, count in self._signal_counts.items():
            direction_stats[direction.value] = count

        return {
            "total_signals": len(self._signal_buffer),
            "by_direction": direction_stats,
            "total_energy": self._total_signal_energy,
            "active_entanglements": len(self._entanglement_pairs),
            "mean_coherence": (
                float(np.mean(list(self._coherence_timers.values())))
                if self._coherence_timers
                else 0.0
            ),
        }

    def get_recent_signals(self, n: int = 10) -> List[InterfaceSignal]:
        """Get recent signals from buffer.

        Args:
            n: Number of recent signals.

        Returns:
            List of recent signals.
        """
        return self._signal_buffer[-n:]

    def reset(self) -> None:
        """Reset bridge state."""
        self._signal_buffer = []
        self._coherence_timers = {}
        self._entanglement_pairs = []
        self._signal_counts = {direction: 0 for direction in InterfaceDirection}
        self._total_signal_energy = 0.0

        logger.debug("Reset QuantumBioBridge")

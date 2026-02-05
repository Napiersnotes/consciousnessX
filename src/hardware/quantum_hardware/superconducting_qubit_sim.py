"""Superconducting qubit simulation for consciousnessX.

Implements transmon qubit dynamics with realistic T1/T2 decoherence,
microwave drive control, and readout resonators for Orch-OR simulations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy.linalg import expm

logger = logging.getLogger(__name__)


class QubitState(Enum):
    """Qubit computational basis states."""

    ZERO = "0"
    ONE = "1"
    PLUS = "+"
    MINUS = "-"
    UNKNOWN = "unknown"


@dataclass
class QubitConfig:
    """Configuration for superconducting qubit."""

    frequency: float = 5.0e9  # 5 GHz
    anharmonicity: float = -0.3e9  # -300 MHz
    t1: float = 100e-6  # 100 microseconds
    t2: float = 50e-6  # 50 microseconds
    drive_amplitude: float = 1e6  # 1 MHz Rabi rate
    temperature: float = 0.015  # 15 mK
    readout_freq: float = 6.5e9  # 6.5 GHz
    readout_linewidth: float = 1e6  # 1 MHz
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)


class TransmonQubit:
    """Transmon superconducting qubit simulation.

    Implements a realistic transmon qubit model with:
    - Anharmonic oscillator energy levels
    - T1/T2 decoherence
    - Microwave drive control
    - Readout via resonator coupling

    Example:
        >>> config = QubitConfig(frequency=5e9)
        >>> qubit = TransmonQubit(config)
        >>> qubit.apply_x_pulse(duration=20e-9)
        >>> result = qubit.measure()
    """

    def __init__(self, config: Optional[QubitConfig] = None) -> None:
        """Initialize transmon qubit.

        Args:
            config: Qubit configuration. Uses defaults if None.
        """
        self.config = config or QubitConfig()
        self.state = np.array([1.0, 0.0], dtype=complex)  # |0⟩ state
        self._time = 0.0
        self._drive_phase = 0.0
        self._readout_signal = 0.0

        # Precompute operators
        self._x = np.array([[0, 1], [1, 0]], dtype=complex)
        self._y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self._z = np.array([[1, 0], [0, -1]], dtype=complex)
        self._h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

        logger.info(f"Initialized TransmonQubit at {self.config.frequency/1e9:.2f} GHz")

    def apply_x_pulse(self, duration: float, amplitude: Optional[float] = None) -> None:
        """Apply X-gate via microwave drive.

        Args:
            duration: Pulse duration in seconds.
            amplitude: Optional amplitude override.
        """
        amp = amplitude or self.config.drive_amplitude
        theta = 2 * np.pi * amp * duration
        self._time += duration

        # Apply rotation
        rotation = expm(-1j * theta * self._x / 2)
        self.state = rotation @ self.state

        # Apply decoherence
        self._apply_decoherence(duration)

        logger.debug(f"Applied X pulse: duration={duration*1e9:.1f}ns, theta={theta:.2f}")

    def apply_y_pulse(self, duration: float, amplitude: Optional[float] = None) -> None:
        """Apply Y-gate via microwave drive.

        Args:
            duration: Pulse duration in seconds.
            amplitude: Optional amplitude override.
        """
        amp = amplitude or self.config.drive_amplitude
        theta = 2 * np.pi * amp * duration
        self._time += duration

        # Apply rotation
        rotation = expm(-1j * theta * self._y / 2)
        self.state = rotation @ self.state

        # Apply decoherence
        self._apply_decoherence(duration)

        logger.debug(f"Applied Y pulse: duration={duration*1e9:.1f}ns, theta={theta:.2f}")

    def apply_z_pulse(self, duration: float, amplitude: Optional[float] = None) -> None:
        """Apply Z-gate via virtual Z rotation.

        Args:
            duration: Pulse duration in seconds.
            amplitude: Optional amplitude override.
        """
        amp = amplitude or self.config.drive_amplitude
        theta = 2 * np.pi * amp * duration
        self._time += duration

        # Apply rotation
        rotation = expm(-1j * theta * self._z / 2)
        self.state = rotation @ self.state

        logger.debug(f"Applied Z pulse: duration={duration*1e9:.1f}ns, theta={theta:.2f}")

    def apply_hadamard(self) -> None:
        """Apply Hadamard gate."""
        self._time += self.config.t1 / 100  # Typical gate time
        self.state = self._h @ self.state
        self._apply_decoherence(self._time)
        logger.debug("Applied Hadamard gate")

    def apply_arbitrary_rotation(self, theta: float, phi: float) -> None:
        """Apply arbitrary single-qubit rotation R(θ, φ).

        Args:
            theta: Rotation angle.
            phi: Phase angle.
        """
        self._time += self.config.t1 / 100

        # R(θ, φ) = cos(θ/2)I - i*sin(θ/2)*(cos(φ)X + sin(φ)Y)
        rotation = np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * (
            np.cos(phi) * self._x + np.sin(phi) * self._y
        )

        self.state = rotation @ self.state
        self._apply_decoherence(self._time)
        logger.debug(f"Applied rotation: theta={theta:.2f}, phi={phi:.2f}")

    def _apply_decoherence(self, duration: float) -> None:
        """Apply T1/T2 decoherence."""
        # Amplitude damping (T1)
        gamma1 = 1.0 / self.config.t1
        prob_decay = 1 - np.exp(-gamma1 * duration)

        if np.random.random() < prob_decay * np.abs(self.state[1]) ** 2:
            # |1⟩ -> |0⟩
            self.state[0] += self.state[1]
            self.state[1] = 0

        # Dephasing (T2)
        gamma2 = 1.0 / self.config.t2
        gamma_phi = gamma2 - gamma1 / 2
        prob_dephase = 1 - np.exp(-gamma_phi * duration)

        if np.random.random() < prob_dephase:
            # Random phase flip
            self.state[1] *= -1

        # Normalize
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state = self.state / norm

    def measure(self) -> int:
        """Measure qubit in computational basis.

        Returns:
            Measurement result (0 or 1).
        """
        # Calculate probability of |1⟩
        prob_1 = np.abs(self.state[1]) ** 2

        # Simulate readout error (typical 95% fidelity)
        readout_error = 0.05
        if np.random.random() < readout_error:
            prob_1 = 1 - prob_1

        result = 1 if np.random.random() < prob_1 else 0

        # Collapse state
        if result == 0:
            self.state = np.array([1.0, 0.0], dtype=complex)
        else:
            self.state = np.array([0.0, 1.0], dtype=complex)

        logger.debug(f"Measurement result: {result}")
        return result

    def get_readout_signal(self) -> Tuple[complex, float]:
        """Get simulated readout signal from resonator.

        Returns:
            Tuple of (complex amplitude, phase shift).
        """
        # Probability of |1⟩ shifts resonator frequency
        prob_1 = np.abs(self.state[1]) ** 2

        # Frequency shift due to dispersive coupling
        chi = 2 * np.pi * 2e6  # 2 MHz dispersive shift
        phase_shift = chi * prob_1 / self.config.readout_linewidth

        # Amplitude with readout noise
        amplitude = 1.0 + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)

        return amplitude * np.exp(1j * phase_shift), phase_shift

    def reset(self) -> None:
        """Reset qubit to |0⟩ state."""
        self.state = np.array([1.0, 0.0], dtype=complex)
        self._time = 0.0
        self._drive_phase = 0.0
        logger.debug("Qubit reset to |0⟩")

    def get_state(self) -> np.ndarray:
        """Get current state vector."""
        return self.state.copy()

    def get_expectation_value(self, operator: np.ndarray) -> float:
        """Calculate expectation value of an operator.

        Args:
            operator: Hermitian operator to measure.

        Returns:
            Expectation value.
        """
        return float(np.real(self.state.conj().T @ operator @ self.state))

    def get_density_matrix(self) -> np.ndarray:
        """Get density matrix of the qubit."""
        return np.outer(self.state, self.state.conj())

    def get_coherence_time_remaining(self) -> Dict[str, float]:
        """Get remaining coherence times."""
        return {
            "t1_remaining": max(0, self.config.t1 - self._time),
            "t2_remaining": max(0, self.config.t2 - self._time),
            "elapsed_time": self._time,
        }

    def get_fidelity(self, target_state: np.ndarray) -> float:
        """Calculate fidelity with target state.

        Args:
            target_state: Target state vector.

        Returns:
            Fidelity (0 to 1).
        """
        overlap = np.abs(np.vdot(target_state, self.state)) ** 2
        return float(overlap)


class SuperconductingQubit(TransmonQubit):
    """Superconducting qubit with advanced features.

    Extends TransmonQubit with additional capabilities for
    consciousness simulation including tunable couplers and
    multi-level operations.
    """

    def __init__(self, config: Optional[QubitConfig] = None, num_levels: int = 3) -> None:
        """Initialize superconducting qubit.

        Args:
            config: Qubit configuration.
            num_levels: Number of energy levels to simulate.
        """
        super().__init__(config)
        self.num_levels = num_levels
        self._multilevel_state = np.zeros(num_levels, dtype=complex)
        self._multilevel_state[0] = 1.0

    def apply_flux_pulse(self, flux: float, duration: float) -> None:
        """Apply flux pulse to tune qubit frequency.

        Args:
            flux: Flux bias (in units of Φ₀).
            duration: Pulse duration.
        """
        # Simplified: flux changes effective frequency
        detuning = 2 * np.pi * 100e6 * flux  # 100 MHz per flux quantum
        self._time += duration

        # Apply Z rotation proportional to detuning
        theta = detuning * duration
        rotation = expm(-1j * theta * self._z / 2)
        self.state = rotation @ self.state

        logger.debug(f"Applied flux pulse: flux={flux:.2f}, duration={duration*1e9:.1f}ns")

    def get_anharmonicity_spectrum(self) -> np.ndarray:
        """Get energy level spectrum."""
        levels = np.arange(self.num_levels)
        frequencies = self.config.frequency + self.config.anharmonicity * levels * (levels - 1) / 2
        return frequencies

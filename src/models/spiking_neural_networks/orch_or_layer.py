"""Orchestrated Objective Reduction (Orch-OR) neural layer.

Implements a neural network layer compatible with Orch-OR theory,
integrating quantum collapse detection with neural activation dynamics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class CollapseType(Enum):
    """Types of quantum collapse events."""
    OBJECTIVE = "objective"
    SUBJECTIVE = "subjective"
    SPONTANEOUS = "spontaneous"
    STIMULATED = "stimulated"


@dataclass
class CollapseEvent:
    """Quantum collapse event in Orch-OR layer."""
    timestamp: float
    layer_id: int
    neuron_indices: List[int]
    collapse_type: CollapseType
    phi_value: float  # Integrated information
    quantum_state: np.ndarray
    consciousness_level: float


@dataclass
class OrchORConfig:
    """Configuration for Orch-OR neural layer."""
    # Layer architecture
    num_neurons: int = 100
    input_size: int = 50
    output_size: int = 50
    
    # Neural dynamics
    activation_threshold: float = 0.5
    membrane_time_constant: float = 20.0  # ms
    synaptic_time_constant: float = 5.0   # ms
    
    # Orch-OR parameters
    collapse_probability: float = 0.01  # Probability per timestep
    phi_threshold: float = 0.3  # Phi threshold for consciousness
    quantum_coherence_time: float = 50.0  # ms
    entanglement_radius: int = 5  # Neurons to entangle
    
    # Consciousness metrics
    baseline_phi: float = 0.1
    consciousness_gain: float = 1.0
    
    # Simulation
    dt: float = 1.0  # ms
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)


class OrchORLayer:
    """Orch-OR compatible neural layer.
    
    Implements a neural network layer that:
    - Detects quantum collapse events
    - Modulates neural activity based on collapse events
    - Calculates integrated information (Phi)
    - Broadcasts consciousness-related signals
    
    Example:
        >>> config = OrchORConfig(num_neurons=100)
        >>> layer = OrchORLayer(0, config)
        >>> output = layer.forward(input_data)
        >>> collapse = layer.detect_collapse()
    """
    
    def __init__(self, layer_id: int, config: Optional[OrchORConfig] = None) -> None:
        """Initialize Orch-OR layer.
        
        Args:
            layer_id: Unique identifier for the layer.
            config: Layer configuration.
        """
        self.layer_id = layer_id
        self.config = config or OrchORConfig()
        
        # Neural state
        self._activations = np.zeros(self.config.num_neurons)
        self._membrane_potentials = np.zeros(self.config.num_neurons)
        self._synaptic_currents = np.zeros(self.config.num_neurons)
        
        # Weights
        self._weights = self._initialize_weights()
        self._biases = np.zeros(self.config.num_neurons)
        
        # Quantum state
        self._quantum_amplitudes = np.random.random(self.config.num_neurons)
        self._coherence_timers = np.zeros(self.config.num_neurons)
        self._entanglement_matrix = np.zeros(
            (self.config.num_neurons, self.config.num_neurons)
        )
        
        # Collapse history
        self._collapse_history: List[CollapseEvent] = []
        
        # Phi values
        self._phi_values = np.zeros(self.config.num_neurons)
        
        logger.info(f"Initialized OrchORLayer {layer_id} with "
                   f"{self.config.num_neurons} neurons")
    
    def _initialize_weights(self) -> np.ndarray:
        """Initialize synaptic weights."""
        # Xavier initialization
        input_dim = self.config.input_size
        fan_in = input_dim
        scale = np.sqrt(2.0 / fan_in)
        
        weights = np.random.randn(self.config.num_neurons, input_dim) * scale
        
        # Add sparse connectivity (20% connection probability)
        mask = np.random.random(weights.shape) < 0.2
        weights *= mask
        
        return weights
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the layer.
        
        Args:
            x: Input data of shape (batch_size, input_size).
            
        Returns:
            Layer activations of shape (batch_size, num_neurons).
        """
        # Compute synaptic currents
        synaptic = x @ self._weights.T + self._biases
        
        # Update membrane potentials
        self._membrane_potentials = self._update_membrane_potentials(synaptic)
        
        # Apply quantum modulation
        quantum_modulation = self._apply_quantum_modulation()
        self._membrane_potentials += quantum_modulation
        
        # Compute activations
        self._activations = self._compute_activations(self._membrane_potentials)
        
        # Update Phi values
        self._update_phi_values()
        
        # Check for collapse events
        collapse = self._check_for_collapse()
        
        return self._activations
    
    def _update_membrane_potentials(self, synaptic: np.ndarray) -> np.ndarray:
        """Update membrane potentials with leaky integration.
        
        Args:
            synaptic: Synaptic input currents.
            
        Returns:
            Updated membrane potentials.
        """
        tau_m = self.config.membrane_time_constant
        dt = self.config.dt
        
        # Leaky integration
        dV = (-self._membrane_potentials + synaptic) / tau_m * dt
        new_potentials = self._membrane_potentials + dV
        
        return new_potentials
    
    def _apply_quantum_modulation(self) -> np.ndarray:
        """Apply quantum modulation to membrane potentials.
        
        Returns:
            Quantum modulation array.
        """
        modulation = np.zeros(self.config.num_neurons)
        
        for i in range(self.config.num_neurons):
            if self._coherence_timers[i] > 0:
                # Coherent quantum state - add quantum fluctuations
                quantum_amplitude = self._quantum_amplitudes[i]
                entanglement_effect = np.sum(
                    self._entanglement_matrix[i] * self._quantum_amplitudes
                )
                
                modulation[i] = quantum_amplitude * entanglement_effect
                
                # Decay coherence
                self._coherence_timers[i] -= self.config.dt
        
        return modulation
    
    def _compute_activations(self, potentials: np.ndarray) -> np.ndarray:
        """Compute neural activations from membrane potentials.
        
        Args:
            potentials: Membrane potentials.
            
        Returns:
            Activation values.
        """
        # Sigmoid activation
        activations = 1.0 / (1.0 + np.exp(-potentials))
        
        # Thresholding
        activations = np.where(
            activations > self.config.activation_threshold,
            activations,
            0.0
        )
        
        return activations
    
    def _update_phi_values(self) -> None:
        """Update integrated information (Phi) values for each neuron."""
        # Phi based on activation pattern and quantum coherence
        activation_entropy = -np.sum(
            self._activations * np.log2(self._activations + 1e-10)
        )
        
        # Normalize
        max_entropy = np.log2(self.config.num_neurons)
        normalized_entropy = activation_entropy / max_entropy
        
        # Combine with quantum coherence
        coherence = np.mean(self._quantum_amplitudes)
        
        for i in range(self.config.num_neurons):
            baseline = self.config.baseline_phi
            contribution = normalized_entropy * self._activations[i]
            quantum_factor = self._quantum_amplitudes[i] * coherence
            
            self._phi_values[i] = baseline + contribution + quantum_factor
    
    def _check_for_collapse(self) -> Optional[CollapseEvent]:
        """Check for quantum collapse events.
        
        Returns:
            CollapseEvent if collapse occurred, None otherwise.
        """
        # Find neurons above Phi threshold
        conscious_neurons = np.where(self._phi_values > self.config.phi_threshold)[0]
        
        if len(conscious_neurons) == 0:
            return None
        
        # Check for collapse probability
        if np.random.random() < self.config.collapse_probability:
            # Determine collapse type
            if np.mean(self._phi_values[conscious_neurons]) > 0.7:
                collapse_type = CollapseType.SUBJECTIVE
            elif np.any(self._activations[conscious_neurons] > 0.8):
                collapse_type = CollapseType.STIMULATED
            else:
                collapse_type = CollapseType.SPONTANEOUS
            
            # Calculate average Phi
            avg_phi = np.mean(self._phi_values[conscious_neurons])
            
            # Get quantum state
            quantum_state = self._quantum_amplitudes[conscious_neurons].copy()
            
            # Calculate consciousness level
            consciousness_level = avg_phi * self.config.consciousness_gain
            
            # Create collapse event
            collapse = CollapseEvent(
                timestamp=0.0,  # Will be set by simulation
                layer_id=self.layer_id,
                neuron_indices=conscious_neurons.tolist(),
                collapse_type=collapse_type,
                phi_value=avg_phi,
                quantum_state=quantum_state,
                consciousness_level=consciousness_level
            )
            
            # Record collapse
            self._collapse_history.append(collapse)
            
            # Reset coherence for affected neurons
            for i in conscious_neurons:
                self._coherence_timers[i] = self.config.quantum_coherence_time
                self._entanglement_matrix[i] = 0
            
            logger.info(f"Collapse detected in layer {self.layer_id}: "
                       f"{len(conscious_neurons)} neurons, Phi={avg_phi:.3f}")
            
            return collapse
        
        return None
    
    def detect_collapse(self) -> Optional[CollapseEvent]:
        """Manually trigger collapse detection.
        
        Returns:
            CollapseEvent if collapse occurred, None otherwise.
        """
        return self._check_for_collapse()
    
    def entangle_neurons(self, i: int, j: int) -> None:
        """Entangle two neurons.
        
        Args:
            i: Index of first neuron.
            j: Index of second neuron.
        """
        self._entanglement_matrix[i, j] = 1.0
        self._entanglement_matrix[j, i] = 1.0
        
        # Update coherence
        self._coherence_timers[i] = self.config.quantum_coherence_time
        self._coherence_timers[j] = self.config.quantum_coherence_time
        
        logger.debug(f"Entangled neurons {i} and {j}")
    
    def broadcast_consciousness(self) -> Dict[str, Any]:
        """Broadcast consciousness-related information.
        
        Returns:
            Dictionary containing consciousness metrics.
        """
        if not self._collapse_history:
            return {
                "layer_id": self.layer_id,
                "has_collapse": False,
                "phi_mean": float(np.mean(self._phi_values)),
                "phi_max": float(np.max(self._phi_values)),
                "consciousness_level": 0.0
            }
        
        latest_collapse = self._collapse_history[-1]
        
        return {
            "layer_id": self.layer_id,
            "has_collapse": True,
            "phi_mean": float(np.mean(self._phi_values)),
            "phi_max": float(np.max(self._phi_values)),
            "consciousness_level": latest_collapse.consciousness_level,
            "collapse_type": latest_collapse.collapse_type.value,
            "affected_neurons": len(latest_collapse.neuron_indices)
        }
    
    def get_phi_values(self) -> np.ndarray:
        """Get current Phi values for all neurons.
        
        Returns:
            Array of Phi values.
        """
        return self._phi_values.copy()
    
    def get_collapse_history(self) -> List[CollapseEvent]:
        """Get history of collapse events.
        
        Returns:
            List of CollapseEvent objects.
        """
        return self._collapse_history.copy()
    
    def reset(self) -> None:
        """Reset layer to initial state."""
        self._activations = np.zeros(self.config.num_neurons)
        self._membrane_potentials = np.zeros(self.config.num_neurons)
        self._synaptic_currents = np.zeros(self.config.num_neurons)
        self._quantum_amplitudes = np.random.random(self.config.num_neurons)
        self._coherence_timers = np.zeros(self.config.num_neurons)
        self._entanglement_matrix = np.zeros(
            (self.config.num_neurons, self.config.num_neurons)
        )
        self._phi_values = np.zeros(self.config.num_neurons)
        self._collapse_history.clear()
        
        logger.debug(f"Reset OrchORLayer {self.layer_id}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current layer state.
        
        Returns:
            Dictionary containing layer state variables.
        """
        return {
            "layer_id": self.layer_id,
            "activations": self._activations.copy(),
            "phi_values": self._phi_values.copy(),
            "quantum_amplitudes": self._quantum_amplitudes.copy(),
            "collapse_count": len(self._collapse_history),
            "mean_phi": float(np.mean(self._phi_values))
        }
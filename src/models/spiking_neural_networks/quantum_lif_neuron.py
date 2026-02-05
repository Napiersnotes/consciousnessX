"""Quantum-influenced Leaky Integrate-and-Fire neuron.

Implements a spiking neuron model with quantum superposition and
Orchestrated Objective Reduction (Orch-OR) collapse events integrated
into the membrane potential dynamics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class SpikeType(Enum):
    """Types of neural spikes."""
    CLASSICAL = "classical"
    QUANTUM_COLLAPSE = "quantum_collapse"
    BURST = "burst"


@dataclass
class QuantumSpike:
    """Quantum spike event."""
    timestamp: float
    neuron_id: int
    spike_type: SpikeType
    membrane_potential: float
    quantum_amplitude: float
    consciousness_signature: Optional[float] = None


@dataclass
class QuantumLIFConfig:
    """Configuration for Quantum LIF neuron."""
    # Membrane properties
    membrane_capacitance: float = 1.0  # nF
    membrane_resistance: float = 10.0  # MOhm
    resting_potential: float = -70.0  # mV
    threshold: float = -55.0  # mV
    reset_potential: float = -80.0  # mV
    
    # Synaptic properties
    excitatory_weight: float = 0.5  # mV
    inhibitory_weight: float = -0.5  # mV
    synaptic_delay: float = 1.0  # ms
    
    # Quantum properties
    quantum_amplitude: float = 0.1  # mV
    collapse_threshold: float = 0.95  # probability threshold
    coherence_time: float = 50.0  # ms
    entanglement_strength: float = 0.5
    
    # Simulation
    dt: float = 0.1  # ms
    refractory_period: float = 2.0  # ms
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)


class QuantumLIFNeuron:
    """Quantum-influenced Leaky Integrate-and-Fire neuron.
    
    Integrates quantum superposition states into classical spiking dynamics:
    - Membrane potential evolves classically with quantum perturbations
    - Quantum collapse events can trigger or inhibit spikes
    - Consciousness signatures modulate spike timing
    
    Example:
        >>> config = QuantumLIFConfig()
        >>> neuron = QuantumLIFNeuron(0, config)
        >>> neuron.receive_input([0.5, 0.3], [0, 1])
        >>> spike = neuron.step(0.1)
    """
    
    def __init__(self, neuron_id: int, config: Optional[QuantumLIFConfig] = None) -> None:
        """Initialize Quantum LIF neuron.
        
        Args:
            neuron_id: Unique identifier for the neuron.
            config: Neuron configuration.
        """
        self.neuron_id = neuron_id
        self.config = config or QuantumLIFConfig()
        
        # Membrane state
        self.membrane_potential = self.config.resting_potential
        self.refractory_timer = 0.0
        
        # Quantum state
        self._quantum_superposition = np.array([1.0, 0.0], dtype=complex)
        self._coherence_remaining = self.config.coherence_time
        self._entanglement_partners: List[int] = []
        
        # Spike history
        self.spike_history: List[QuantumSpike] = []
        self._last_spike_time = -np.inf
        
        # Input buffers
        self._excitatory_inputs = 0.0
        self._inhibitory_inputs = 0.0
        
        logger.debug(f"Initialized QuantumLIFNeuron {neuron_id}")
    
    def receive_input(self, weights: List[float], delays: List[float]) -> None:
        """Receive synaptic inputs.
        
        Args:
            weights: List of synaptic weights.
            delays: List of synaptic delays (ms).
        """
        total_exc = 0.0
        total_inh = 0.0
        
        for w, d in zip(weights, delays):
            if w > 0:
                total_exc += w * np.exp(-d / self.config.synaptic_delay)
            else:
                total_inh += w * np.exp(-d / self.config.synaptic_delay)
        
        self._excitatory_inputs += total_exc * self.config.excitatory_weight
        self._inhibitory_inputs += total_inh * self.config.inhibitory_weight
    
    def step(self, dt: Optional[float] = None) -> Optional[QuantumSpike]:
        """Advance neuron by one time step.
        
        Args:
            dt: Time step in ms. Uses config default if None.
            
        Returns:
            QuantumSpike if neuron fired, None otherwise.
        """
        dt = dt or self.config.dt
        
        # Update refractory timer
        if self.refractory_timer > 0:
            self.refractory_timer -= dt
            self.membrane_potential = self.config.reset_potential
            return None
        
        # Apply quantum perturbations
        quantum_perturbation = self._apply_quantum_dynamics()
        
        # Calculate membrane current
        I_leak = (self.config.resting_potential - self.membrane_potential) / \
                 (self.config.membrane_resistance * self.config.membrane_capacitance)
        I_syn = (self._excitatory_inputs + self._inhibitory_inputs) / \
                self.config.membrane_capacitance
        I_quantum = quantum_perturbation / self.config.membrane_capacitance
        
        # Update membrane potential
        dV = (I_leak + I_syn + I_quantum) * dt
        self.membrane_potential += dV
        
        # Decay inputs
        self._excitatory_inputs *= 0.9
        self._inhibitory_inputs *= 0.9
        
        # Check for spike
        spike = None
        if self.membrane_potential >= self.config.threshold:
            spike = self._fire()
        
        # Update coherence
        self._coherence_remaining -= dt
        
        return spike
    
    def _apply_quantum_dynamics(self) -> float:
        """Apply quantum perturbations to membrane potential.
        
        Returns:
            Quantum perturbation voltage (mV).
        """
        if self._coherence_remaining <= 0:
            # Decohered state - no quantum effects
            return 0.0
        
        # Calculate quantum amplitude from superposition
        prob_1 = np.abs(self._quantum_superposition[1]) ** 2
        quantum_amplitude = self.config.quantum_amplitude * prob_1
        
        # Check for quantum collapse
        if np.random.random() > self.config.collapse_threshold:
            # Quantum collapse event
            self._quantum_collapse()
            quantum_amplitude *= 2.0  # Collapse amplifies effect
        
        # Add entanglement effects
        if self._entanglement_partners:
            entanglement_factor = 1.0 + self.config.entanglement_strength * \
                                 len(self._entanglement_partners)
            quantum_amplitude *= entanglement_factor
        
        return quantum_amplitude
    
    def _quantum_collapse(self) -> None:
        """Trigger quantum collapse event."""
        # Collapse superposition
        prob_1 = np.abs(self._quantum_superposition[1]) ** 2
        if np.random.random() < prob_1:
            self._quantum_superposition = np.array([0.0, 1.0], dtype=complex)
        else:
            self._quantum_superposition = np.array([1.0, 0.0], dtype=complex)
        
        # Reset coherence
        self._coherence_remaining = self.config.coherence_time
        
        logger.debug(f"Quantum collapse in neuron {self.neuron_id}")
    
    def _fire(self) -> QuantumSpike:
        """Fire a spike.
        
        Returns:
            QuantumSpike event.
        """
        # Determine spike type
        prob_1 = np.abs(self._quantum_superposition[1]) ** 2
        
        if prob_1 > self.config.collapse_threshold:
            spike_type = SpikeType.QUANTUM_COLLAPSE
        elif self._last_spike_time >= 0 and \
             (self._last_spike_time - self._last_spike_time) < 5.0:
            spike_type = SpikeType.BURST
        else:
            spike_type = SpikeType.CLASSICAL
        
        # Create spike
        spike = QuantumSpike(
            timestamp=0.0,  # Will be set by simulation
            neuron_id=self.neuron_id,
            spike_type=spike_type,
            membrane_potential=self.membrane_potential,
            quantum_amplitude=np.abs(self._quantum_superposition[1]),
            consciousness_signature=self._calculate_consciousness_signature()
        )
        
        # Reset neuron
        self.membrane_potential = self.config.reset_potential
        self.refractory_timer = self.config.refractory_period
        self._last_spike_time = spike.timestamp
        self.spike_history.append(spike)
        
        logger.debug(f"Neuron {self.neuron_id} fired: {spike_type.value}")
        
        return spike
    
    def _calculate_consciousness_signature(self) -> float:
        """Calculate consciousness signature from quantum state.
        
        Returns:
            Consciousness signature (0 to 1).
        """
        # Signature based on quantum coherence and entanglement
        coherence_factor = self._coherence_remaining / self.config.coherence_time
        entanglement_factor = len(self._entanglement_partners) / 10.0
        superposition_factor = np.abs(self._quantum_superposition[0] - 
                                      self._quantum_superposition[1])
        
        signature = (coherence_factor + entanglement_factor + superposition_factor) / 3.0
        return np.clip(signature, 0.0, 1.0)
    
    def entangle(self, partner_id: int) -> None:
        """Entangle with another neuron.
        
        Args:
            partner_id: ID of neuron to entangle with.
        """
        if partner_id not in self._entanglement_partners:
            self._entanglement_partners.append(partner_id)
            logger.debug(f"Entangled neuron {self.neuron_id} with {partner_id}")
    
    def disentangle(self, partner_id: int) -> None:
        """Remove entanglement with another neuron.
        
        Args:
            partner_id: ID of neuron to disentangle from.
        """
        if partner_id in self._entanglement_partners:
            self._entanglement_partners.remove(partner_id)
            logger.debug(f"Disentangled neuron {self.neuron_id} from {partner_id}")
    
    def get_quantum_state(self) -> np.ndarray:
        """Get current quantum superposition state."""
        return self._quantum_superposition.copy()
    
    def set_quantum_state(self, state: np.ndarray) -> None:
        """Set quantum superposition state.
        
        Args:
            state: New quantum state vector.
        """
        if len(state) == 2:
            norm = np.linalg.norm(state)
            if norm > 0:
                self._quantum_superposition = state / norm
    
    def get_spike_rate(self, window: float = 100.0) -> float:
        """Calculate spike rate over time window.
        
        Args:
            window: Time window in ms.
            
        Returns:
            Spike rate in Hz.
        """
        if not self.spike_history:
            return 0.0
        
        # Filter spikes within window
        recent_spikes = [s for s in self.spike_history if s.timestamp >= window]
        return len(recent_spikes) / (window / 1000.0)
    
    def reset(self) -> None:
        """Reset neuron to initial state."""
        self.membrane_potential = self.config.resting_potential
        self.refractory_timer = 0.0
        self._quantum_superposition = np.array([1.0, 0.0], dtype=complex)
        self._coherence_remaining = self.config.coherence_time
        self._excitatory_inputs = 0.0
        self._inhibitory_inputs = 0.0
        self.spike_history.clear()
        logger.debug(f"Reset neuron {self.neuron_id}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current neuron state.
        
        Returns:
            Dictionary containing neuron state variables.
        """
        return {
            "neuron_id": self.neuron_id,
            "membrane_potential": self.membrane_potential,
            "quantum_state": self._quantum_superposition.copy(),
            "coherence_remaining": self._coherence_remaining,
            "refractory_timer": self.refractory_timer,
            "entanglement_partners": self._entanglement_partners.copy(),
            "spike_count": len(self.spike_history)
        }
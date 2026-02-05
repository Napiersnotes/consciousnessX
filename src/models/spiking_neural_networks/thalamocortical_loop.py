"""Thalamocortical loop simulation for consciousnessX.

Implements TRN-gated thalamocortical feedback loops with realistic
thalamic relay nuclei, thalamic reticular nucleus, and cortical interactions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ThalamicNucleus(Enum):
    """Thalamic nuclei types."""
    LGN = "lateral_geniculate"  # Visual
    MGN = "medial_geniculate"   # Auditory
    VPL = "ventral_posterolateral"  # Somatosensory
    VPM = "ventral_posteromedial"  # Trigeminal
    MD = "mediodorsal"         # Prefrontal
    Pulvinar = "pulvinar"      # Association


class TRNNeuronType(Enum):
    """Thalamic reticular nucleus neuron types."""
    CORE = "core"
    MATRIX = "matrix"


@dataclass
class TRNNeuron:
    """Thalamic reticular nucleus neuron."""
    neuron_id: int
    neuron_type: TRNNeuronType
    membrane_potential: float = -70.0  # mV
    threshold: float = -50.0  # mV
    activation: float = 0.0
    
    def update(self, input_current: float, dt: float = 1.0) -> bool:
        """Update neuron dynamics.
        
        Args:
            input_current: Input synaptic current.
            dt: Time step in ms.
            
        Returns:
            True if neuron fired, False otherwise.
        """
        # Leaky integration
        tau = 20.0  # ms
        dV = (-self.membrane_potential + input_current) / tau * dt
        self.membrane_potential += dV
        
        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = -70.0  # Reset
            self.activation = 1.0
            return True
        
        # Decay activation
        self.activation *= 0.9
        return False


@dataclass
class ThalamocorticalConfig:
    """Configuration for thalamocortical loop."""
    # Architecture
    num_cortical_units: int = 100
    num_thalamic_units: int = 50
    num_trn_units: int = 25
    
    # Connectivity
    corticothalamic_strength: float = 0.8
    thalamocortical_strength: float = 0.7
    trn_inhibition_strength: float = 1.2
    
    # Dynamics
    membrane_time_constant: float = 20.0  # ms
    synaptic_delay: float = 2.0  # ms
    refractory_period: float = 2.0  # ms
    
    # Oscillations
    alpha_frequency: float = 10.0  # Hz
    gamma_frequency: float = 40.0  # Hz
    
    # Gating
    trn_threshold: float = 0.5
    attention_gain: float = 1.5
    
    # Simulation
    dt: float = 1.0  # ms
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)


class ThalamocorticalLoop:
    """Thalamocortical loop with TRN gating.
    
    Implements the canonical thalamocortical circuit:
    1. Cortical projections to thalamus
    2. Thalamic relay to cortex (gated by TRN)
    3. TRN receives inputs from both cortex and thalamus
    4. TRN inhibits thalamus (attentional gating)
    
    Example:
        >>> config = ThalamocorticalConfig()
        >>> loop = ThalamocorticalLoop(0, config)
        >>> cortical_input = np.random.rand(100)
        >>> thalamic_output = loop.process(cortical_input)
    """
    
    def __init__(self, loop_id: int, 
                 config: Optional[ThalamocorticalConfig] = None) -> None:
        """Initialize thalamocortical loop.
        
        Args:
            loop_id: Unique identifier for the loop.
            config: Loop configuration.
        """
        self.loop_id = loop_id
        self.config = config or ThalamocorticalConfig()
        
        # Neural populations
        self._cortical_units = np.zeros(self.config.num_cortical_units)
        self._thalamic_units = np.zeros(self.config.num_thalamic_units)
        self._trn_neurons: List[TRNNeuron] = []
        self._initialize_trn()
        
        # Connectivity
        self._corticothalamic_weights = self._initialize_corticothalamic()
        self._thalamocortical_weights = self._initialize_thalamocortical()
        self._cortico_trn_weights = self._initialize_cortico_trn()
        self._thalamo_trn_weights = self._initialize_thalamo_trn()
        self._trn_thalamic_weights = self._initialize_trn_thalamic()
        
        # Input buffers
        self._sensory_input = np.zeros(self.config.num_thalamic_units)
        self._cortical_feedback = np.zeros(self.config.num_cortical_units)
        
        # Oscillatory state
        self._phase_alpha = 0.0
        self._phase_gamma = 0.0
        
        # Gating state
        self._attentional_gain = np.ones(self.config.num_thalamic_units)
        self._gating_history: List[float] = []
        
        logger.info(f"Initialized ThalamocorticalLoop {loop_id}")
    
    def _initialize_trn(self) -> None:
        """Initialize TRN neurons."""
        for i in range(self.config.num_trn_units):
            # Mix of core and matrix neurons
            neuron_type = (TRNNeuronType.CORE if i < self.config.num_trn_units // 2 
                          else TRNNeuronType.MATRIX)
            
            neuron = TRNNeuron(
                neuron_id=i,
                neuron_type=neuron_type
            )
            
            self._trn_neurons.append(neuron)
    
    def _initialize_corticothalamic(self) -> np.ndarray:
        """Initialize corticothalamic connection weights."""
        n_cort = self.config.num_cortical_units
        n_thal = self.config.num_thalamic_units
        
        # Sparse connectivity
        weights = np.random.rand(n_cort, n_thal) * self.config.corticothalamic_strength
        mask = np.random.random(weights.shape) < 0.3
        weights *= mask
        
        return weights
    
    def _initialize_thalamocortical(self) -> np.ndarray:
        """Initialize thalamocortical connection weights."""
        n_thal = self.config.num_thalamic_units
        n_cort = self.config.num_cortical_units
        
        # Topographic organization
        weights = np.zeros((n_thal, n_cort))
        
        for i in range(n_thal):
            # Each thalamic unit connects to local cortical region
            center = (i / n_thal) * n_cort
            width = n_cort / n_thal
            
            for j in range(n_cort):
                distance = abs(j - center)
                weights[i, j] = np.exp(-(distance / width) ** 2) * \
                               self.config.thalamocortical_strength
        
        return weights
    
    def _initialize_cortico_trn(self) -> np.ndarray:
        """Initialize cortical to TRN connection weights."""
        n_cort = self.config.num_cortical_units
        n_trn = self.config.num_trn_units
        
        weights = np.random.rand(n_cort, n_trn) * 0.5
        return weights
    
    def _initialize_thalamo_trn(self) -> np.ndarray:
        """Initialize thalamic to TRN connection weights."""
        n_thal = self.config.num_thalamic_units
        n_trn = self.config.num_trn_units
        
        weights = np.random.rand(n_thal, n_trn) * 0.5
        return weights
    
    def _initialize_trn_thalamic(self) -> np.ndarray:
        """Initialize TRN to thalamic inhibitory weights."""
        n_trn = self.config.num_trn_units
        n_thal = self.config.num_thalamic_units
        
        # Strong, sparse inhibition
        weights = np.random.rand(n_trn, n_thal) * self.config.trn_inhibition_strength
        mask = np.random.random(weights.shape) < 0.2
        weights *= mask
        
        return weights
    
    def process(self, sensory_input: np.ndarray, 
                cortical_feedback: Optional[np.ndarray] = None) -> np.ndarray:
        """Process input through thalamocortical loop.
        
        Args:
            sensory_input: Sensory input to thalamus.
            cortical_feedback: Cortical feedback (top-down).
            
        Returns:
            Thalamocortical output to cortex.
        """
        # Update inputs
        self._sensory_input = sensory_input
        
        if cortical_feedback is not None:
            self._cortical_feedback = cortical_feedback
        
        # Step 1: Cortical projections to thalamus
        corticothalamic_input = self._cortical_feedback @ self._corticothalamic_weights
        
        # Step 2: Update thalamic units
        self._update_thalamic_units(corticothalamic_input)
        
        # Step 3: Update TRN neurons
        self._update_trn()
        
        # Step 4: Apply TRN inhibition (gating)
        self._apply_trn_gating()
        
        # Step 5: Update oscillatory phase
        self._update_oscillations()
        
        # Step 6: Thalamocortical output
        thalamocortical_output = self._thalamic_units @ self._thalamocortical_weights
        
        # Add oscillatory modulation
        alpha_modulation = np.sin(2 * np.pi * self._phase_alpha)
        thalamocortical_output *= (1 + 0.2 * alpha_modulation)
        
        return thalamocortical_output
    
    def _update_thalamic_units(self, corticothalamic_input: np.ndarray) -> None:
        """Update thalamic relay unit dynamics."""
        # Total input to thalamus
        total_input = (self._sensory_input + corticothalamic_input) * \
                     self._attentional_gain
        
        # Leaky integration
        tau = self.config.membrane_time_constant
        dt = self.config.dt
        
        dV = (-self._thalamic_units + total_input) / tau * dt
        self._thalamic_units += dV
        
        # ReLU activation
        self._thalamic_units = np.maximum(0, self._thalamic_units)
    
    def _update_trn(self) -> None:
        """Update TRN neuron dynamics."""
        # Calculate inputs to TRN
        cortico_input = self._cortical_feedback @ self._cortico_trn_weights
        thalamo_input = self._thalamic_units @ self._thalamo_trn_weights
        
        total_input = cortico_input + thalamo_input
        
        # Update each TRN neuron
        trn_activations = np.zeros(len(self._trn_neurons))
        
        for i, neuron in enumerate(self._trn_neurons):
            fired = neuron.update(total_input[i], self.config.dt)
            if fired:
                trn_activations[i] = 1.0
            else:
                trn_activations[i] = neuron.activation
        
        self._trn_activations = trn_activations
    
    def _apply_trn_gating(self) -> None:
        """Apply TRN inhibitory gating to thalamus."""
        # Calculate TRN inhibition
        trn_inhibition = self._trn_activations @ self._trn_thalamic_weights
        
        # Update attentional gain
        # Higher TRN activity = lower thalamic gain = attentional selection
        inhibition_factor = 1.0 / (1.0 + trn_inhibition)
        self._attentional_gain = np.clip(inhibition_factor, 0.1, self.config.attention_gain)
        
        # Record gating strength
        mean_gate = np.mean(trn_inhibition)
        self._gating_history.append(mean_gate)
        
        if len(self._gating_history) > 1000:
            self._gating_history.pop(0)
    
    def _update_oscillations(self) -> None:
        """Update oscillatory phase for alpha and gamma rhythms."""
        dt = self.config.dt / 1000.0  # Convert to seconds
        
        self._phase_alpha += self.config.alpha_frequency * dt
        self._phase_gamma += self.config.gamma_frequency * dt
        
        # Wrap phases
        self._phase_alpha %= 1.0
        self._phase_gamma %= 1.0
    
    def set_attention(self, attention_map: np.ndarray) -> None:
        """Set attentional modulation of thalamic gain.
        
        Args:
            attention_map: Array of attentional gains (0 to 2).
        """
        if len(attention_map) == self.config.num_thalamic_units:
            self._attentional_gain = np.clip(attention_map, 0.1, 2.0)
            logger.debug("Updated attention map")
    
    def get_thalamic_activity(self) -> np.ndarray:
        """Get current thalamic unit activities."""
        return self._thalamic_units.copy()
    
    def get_trn_activity(self) -> np.ndarray:
        """Get current TRN neuron activities."""
        return np.array([n.activation for n in self._trn_neurons])
    
    def get_gating_strength(self) -> float:
        """Get current TRN gating strength."""
        if not self._gating_history:
            return 0.0
        return self._gating_history[-1]
    
    def get_oscillatory_state(self) -> Dict[str, float]:
        """Get current oscillatory state.
        
        Returns:
            Dictionary with alpha and gamma phases.
        """
        return {
            "alpha_phase": self._phase_alpha,
            "gamma_phase": self._phase_gamma,
            "alpha_amplitude": np.sin(2 * np.pi * self._phase_alpha),
            "gamma_amplitude": np.sin(2 * np.pi * self._phase_gamma)
        }
    
    def get_coherence(self) -> float:
        """Calculate thalamocortical coherence."""
        # Cross-correlation between thalamus and cortex
        thalamic_activity = self._thalamic_units
        cortical_projection = self._thalamic_units @ self._thalamocortical_weights
        
        if np.std(thalamic_activity) > 0 and np.std(cortical_projection) > 0:
            coherence = np.corrcoef(thalamic_activity, cortical_projection)[0, 1]
            return abs(coherence)
        
        return 0.0
    
    def reset(self) -> None:
        """Reset loop to initial state."""
        self._cortical_units = np.zeros(self.config.num_cortical_units)
        self._thalamic_units = np.zeros(self.config.num_thalamic_units)
        self._sensory_input = np.zeros(self.config.num_thalamic_units)
        self._cortical_feedback = np.zeros(self.config.num_cortical_units)
        self._attentional_gain = np.ones(self.config.num_thalamic_units)
        self._phase_alpha = 0.0
        self._phase_gamma = 0.0
        self._gating_history.clear()
        
        for neuron in self._trn_neurons:
            neuron.membrane_potential = -70.0
            neuron.activation = 0.0
        
        logger.debug(f"Reset ThalamocorticalLoop {self.loop_id}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current loop state.
        
        Returns:
            Dictionary containing loop state variables.
        """
        return {
            "loop_id": self.loop_id,
            "mean_thalamic_activity": float(np.mean(self._thalamic_units)),
            "mean_trn_activity": float(np.mean(self.get_trn_activity())),
            "mean_attentional_gain": float(np.mean(self._attentional_gain)),
            "gating_strength": self.get_gating_strength(),
            "coherence": self.get_coherence(),
            "oscillatory_state": self.get_oscillatory_state()
        }
"""Cortical column simulation for consciousnessX.

Implements minicolumn-based cortical simulation with realistic connectivity
patterns, layer-specific dynamics, and dendritic processing for consciousness modeling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """Cortical layer types."""

    L1 = "layer1"  # Molecular layer (input)
    L2_3 = "layer2_3"  # Supragranular (intracortical)
    L4 = "layer4"  # Granular (thalamic input)
    L5 = "layer5"  # Infragranular (output)
    L6 = "layer6"  # Infragranular (corticothalamic)


class NeuronType(Enum):
    """Neuron types in cortical columns."""

    EXCITATORY = "pyramidal"
    INHIBITORY = "interneuron"


@dataclass
class Minicolumn:
    """Single minicolumn within a cortical column."""

    minicolumn_id: int
    neurons: List[int]
    layer: LayerType
    activation_level: float = 0.0
    synchronization_index: float = 0.0

    def get_activation(self) -> float:
        """Get current activation level."""
        return self.activation_level

    def update_activation(self, new_level: float) -> None:
        """Update activation level with decay."""
        self.activation_level = 0.9 * self.activation_level + 0.1 * new_level


@dataclass
class CorticalColumnConfig:
    """Configuration for cortical column simulation."""

    # Architecture
    num_minicolumns: int = 100
    neurons_per_minicolumn: int = 100
    num_layers: int = 6

    # Connectivity
    intra_column_connectivity: float = 0.3
    inter_column_connectivity: float = 0.1
    thalamic_connectivity: float = 0.5

    # Dynamics
    membrane_time_constant: float = 20.0  # ms
    synaptic_time_constant: float = 5.0  # ms
    refractory_period: float = 2.0  # ms

    # Plasticity
    learning_rate: float = 0.01
    synaptic_decay: float = 0.001

    # Consciousness
    phi_threshold: float = 0.3
    integration_window: float = 50.0  # ms

    # Simulation
    dt: float = 1.0  # ms
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)


class CorticalColumn:
    """Cortical column simulation with minicolumns.

    Implements a realistic cortical column with:
    - Multiple layers with specific connectivity patterns
    - Minicolumn organization (100 neurons/minicolumn)
    - Layer-specific neuron types and properties
    - Dendritic integration and spike generation
    - Synaptic plasticity

    Example:
        >>> config = CorticalColumnConfig(num_minicolumns=100)
        >>> column = CorticalColumn(0, config)
        >>> column.receive_thalamic_input(input_data)
        >>> activations = column.step()
    """

    def __init__(self, column_id: int, config: Optional[CorticalColumnConfig] = None) -> None:
        """Initialize cortical column.

        Args:
            column_id: Unique identifier for the column.
            config: Column configuration.
        """
        self.column_id = column_id
        self.config = config or CorticalColumnConfig()

        # Initialize minicolumns
        self.minicolumns: List[Minicolumn] = []
        self._initialize_minicolumns()

        # Neural state
        self.num_neurons = self.config.num_minicolumns * self.config.neurons_per_minicolumn
        self._activations = np.zeros(self.num_neurons)
        self._membrane_potentials = np.zeros(self.num_neurons)
        self._spike_times = np.full(self.num_neurons, -np.inf)

        # Layer assignment
        self._neuron_layers = self._assign_layers()

        # Synaptic weights
        self._weights = self._initialize_weights()

        # Plasticity
        self._synaptic_strengths = np.ones_like(self._weights)

        # Consciousness metrics
        self._phi_value = 0.0
        self._integration_history: List[float] = []

        logger.info(
            f"Initialized CorticalColumn {column_id} with "
            f"{self.config.num_minicolumns} minicolumns, "
            f"{self.num_neurons} neurons"
        )

    def _initialize_minicolumns(self) -> None:
        """Initialize minicolumns across layers."""
        minicolumns_per_layer = self.config.num_minicolumns // self.config.num_layers

        for layer_idx, layer_type in enumerate(LayerType):
            for i in range(minicolumns_per_layer):
                mc_id = layer_idx * minicolumns_per_layer + i

                # Neuron indices for this minicolumn
                start_idx = mc_id * self.config.neurons_per_minicolumn
                end_idx = start_idx + self.config.neurons_per_minicolumn
                neuron_indices = list(range(start_idx, min(end_idx, self.num_neurons)))

                minicolumn = Minicolumn(
                    minicolumn_id=mc_id, neurons=neuron_indices, layer=layer_type
                )

                self.minicolumns.append(minicolumn)

    def _assign_layers(self) -> np.ndarray:
        """Assign neurons to cortical layers."""
        layers = np.zeros(self.num_neurons, dtype=int)
        neurons_per_layer = self.num_neurons // self.config.num_layers

        for layer_idx in range(self.config.num_layers):
            start = layer_idx * neurons_per_layer
            end = start + neurons_per_layer
            layers[start:end] = layer_idx

        return layers

    def _initialize_weights(self) -> np.ndarray:
        """Initialize synaptic weights with realistic connectivity."""
        # Sparse connectivity
        connection_prob = self.config.intra_column_connectivity
        weights = np.random.rand(self.num_neurons, self.num_neurons)

        # Apply sparsity
        mask = np.random.random(weights.shape) < connection_prob
        weights = weights * mask

        # Normalize
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-10)

        # Layer-specific connectivity
        for i in range(self.num_neurons):
            source_layer = self._neuron_layers[i]

            for j in range(self.num_neurons):
                target_layer = self._neuron_layers[j]

                # Feedforward connections (lower to higher layers)
                if target_layer > source_layer:
                    weights[i, j] *= 1.5
                # Feedback connections (higher to lower layers)
                elif target_layer < source_layer:
                    weights[i, j] *= 0.8
                # Lateral connections (same layer)
                else:
                    weights[i, j] *= 1.0

        return weights

    def receive_thalamic_input(self, input_data: np.ndarray) -> None:
        """Receive thalamic input to layer 4.

        Args:
            input_data: Input activations.
        """
        # Target layer 4 neurons
        l4_neurons = np.where(self._neuron_layers == 3)[0]  # Layer 4

        if len(l4_neurons) > 0:
            # Distribute input
            input_size = len(input_data)
            neurons_per_input = len(l4_neurons) // input_size

            for i in range(input_size):
                start = i * neurons_per_input
                end = start + neurons_per_input
                target_neurons = l4_neurons[start:end]

                self._activations[target_neurons] += (
                    input_data[i] * self.config.thalamic_connectivity
                )

    def step(self) -> np.ndarray:
        """Advance simulation by one time step.

        Returns:
            Activations of all neurons.
        """
        # Update membrane potentials
        self._update_membrane_potentials()

        # Compute new activations
        self._activations = self._compute_activations()

        # Update minicolumn activations
        self._update_minicolumns()

        # Update plasticity
        self._update_plasticity()

        # Calculate Phi
        self._calculate_phi()

        return self._activations

    def _update_membrane_potentials(self) -> None:
        """Update membrane potentials with leaky integration."""
        # Compute synaptic inputs
        synaptic_inputs = self._activations @ self._weights * self._synaptic_strengths

        # Leaky integration
        tau_m = self.config.membrane_time_constant
        dt = self.config.dt

        dV = (-self._membrane_potentials + synaptic_inputs) / tau_m * dt
        self._membrane_potentials += dV

    def _compute_activations(self) -> np.ndarray:
        """Compute neural activations from membrane potentials."""
        # Sigmoid activation
        activations = 1.0 / (1.0 + np.exp(-self._membrane_potentials))

        # Refractory period
        time_since_spike = 0.0 - self._spike_times  # Current time is 0
        refractory_mask = time_since_spike < self.config.refractory_period
        activations[refractory_mask] = 0.0

        # Thresholding
        threshold_mask = activations < 0.5
        activations[threshold_mask] = 0.0

        # Update spike times
        spiking_neurons = np.where(activations > 0.5)[0]
        self._spike_times[spiking_neurons] = 0.0

        return activations

    def _update_minicolumns(self) -> None:
        """Update activation levels for all minicolumns."""
        for minicolumn in self.minicolumns:
            # Calculate mean activation of minicolumn
            mc_activations = self._activations[minicolumn.neurons]
            mean_activation = np.mean(mc_activations)

            # Update synchronization index
            std_activation = np.std(mc_activations)
            if mean_activation > 0:
                sync_index = 1.0 - std_activation / (mean_activation + 1e-10)
                minicolumn.synchronization_index = np.clip(sync_index, 0.0, 1.0)

            # Update activation level
            minicolumn.update_activation(mean_activation)

    def _update_plasticity(self) -> None:
        """Update synaptic strengths with Hebbian learning."""
        # Hebbian rule: Δw = η * (pre * post - decay)
        pre = self._activations[:, np.newaxis]
        post = self._activations[np.newaxis, :]

        delta = self.config.learning_rate * (pre * post - self.config.synaptic_decay)
        self._synaptic_strengths = np.clip(
            self._synaptic_strengths + delta, 0.1, 2.0  # Minimum strength  # Maximum strength
        )

    def _calculate_phi(self) -> None:
        """Calculate integrated information (Phi)."""
        # Entropy of activation pattern
        activation_prob = self._activations / (np.sum(self._activations) + 1e-10)
        activation_entropy = -np.sum(activation_prob * np.log2(activation_prob + 1e-10))

        # Normalize
        max_entropy = np.log2(self.num_neurons)
        normalized_entropy = activation_entropy / max_entropy

        # Integration: measure of causal influence
        integration = np.sum(np.abs(self._weights))
        normalized_integration = integration / self.num_neurons

        # Phi combines entropy and integration
        phi = 0.5 * normalized_entropy + 0.5 * normalized_integration

        self._phi_value = phi
        self._integration_history.append(phi)

        # Keep limited history
        if len(self._integration_history) > 1000:
            self._integration_history.pop(0)

    def get_phi(self) -> float:
        """Get current Phi value."""
        return self._phi_value

    def get_minicolumn_activations(self) -> np.ndarray:
        """Get activations of all minicolumns.

        Returns:
            Array of minicolumn activation levels.
        """
        return np.array([mc.get_activation() for mc in self.minicolumns])

    def get_layer_activations(self) -> Dict[int, np.ndarray]:
        """Get activations for each cortical layer.

        Returns:
            Dictionary mapping layer index to activation array.
        """
        layer_activations = {}

        for layer_idx in range(self.config.num_layers):
            neurons = np.where(self._neuron_layers == layer_idx)[0]
            layer_activations[layer_idx] = self._activations[neurons].copy()

        return layer_activations

    def get_synchronization_matrix(self) -> np.ndarray:
        """Get synchronization matrix between minicolumns.

        Returns:
            NxN matrix of synchronization indices.
        """
        n_mc = len(self.minicolumns)
        sync_matrix = np.zeros((n_mc, n_mc))

        for i in range(n_mc):
            for j in range(n_mc):
                # Calculate correlation between minicolumns
                mc_i = self.minicolumns[i]
                mc_j = self.minicolumns[j]

                act_i = self._activations[mc_i.neurons]
                act_j = self._activations[mc_j.neurons]

                if np.std(act_i) > 0 and np.std(act_j) > 0:
                    corr = np.corrcoef(act_i, act_j)[0, 1]
                    sync_matrix[i, j] = abs(corr)

        return sync_matrix

    def stimulate(self, minicolumn_id: int, intensity: float = 1.0) -> None:
        """Stimulate a specific minicolumn.

        Args:
            minicolumn_id: ID of minicolumn to stimulate.
            intensity: Stimulation intensity.
        """
        if 0 <= minicolumn_id < len(self.minicolumns):
            minicolumn = self.minicolumns[minicolumn_id]
            self._activations[minicolumn.neurons] += intensity

            logger.debug(f"Stimulated minicolumn {minicolumn_id} " f"with intensity {intensity}")

    def reset(self) -> None:
        """Reset column to initial state."""
        self._activations = np.zeros(self.num_neurons)
        self._membrane_potentials = np.zeros(self.num_neurons)
        self._spike_times = np.full(self.num_neurons, -np.inf)
        self._synaptic_strengths = np.ones_like(self._weights)
        self._phi_value = 0.0
        self._integration_history.clear()

        for mc in self.minicolumns:
            mc.activation_level = 0.0
            mc.synchronization_index = 0.0

        logger.debug(f"Reset CorticalColumn {self.column_id}")

    def get_state(self) -> Dict[str, Any]:
        """Get current column state.

        Returns:
            Dictionary containing column state variables.
        """
        return {
            "column_id": self.column_id,
            "num_neurons": self.num_neurons,
            "num_minicolumns": len(self.minicolumns),
            "mean_activation": float(np.mean(self._activations)),
            "max_activation": float(np.max(self._activations)),
            "phi": self._phi_value,
            "synchrony": float(np.mean([mc.synchronization_index for mc in self.minicolumns])),
        }

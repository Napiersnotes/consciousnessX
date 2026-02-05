"""
Spiking neural networks with quantum consciousness integration.

This module implements biologically-inspired spiking neural networks
integrated with Orch-OR quantum consciousness models, including
quantum-influenced neurons, cortical columns, and thalamocortical loops.
"""

from .quantum_lif_neuron import (
    QuantumLIFNeuron,
    QuantumLIFConfig,
    QuantumSpike,
)
from .orch_or_layer import (
    OrchORLayer,
    OrchORConfig,
    CollapseEvent,
)
from .cortical_column_sim import (
    CorticalColumn,
    CorticalColumnConfig,
    Minicolumn,
)
from .thalamocortical_loop import (
    ThalamocorticalLoop,
    ThalamocorticalConfig,
    TRNNeuron,
)

__all__ = [
    # Quantum LIF neuron
    "QuantumLIFNeuron",
    "QuantumLIFConfig",
    "QuantumSpike",
    # Orch-OR layer
    "OrchORLayer",
    "OrchORConfig",
    "CollapseEvent",
    # Cortical column
    "CorticalColumn",
    "CorticalColumnConfig",
    "Minicolumn",
    # Thalamocortical loop
    "ThalamocorticalLoop",
    "ThalamocorticalConfig",
    "TRNNeuron",
]

__version__ = "0.1.0"

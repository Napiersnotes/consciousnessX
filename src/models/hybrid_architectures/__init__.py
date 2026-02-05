"""
Hybrid architectures for consciousness modeling.

This module implements hybrid quantum-classical-biological architectures
for consciousness, including Global Workspace Theory, Higher-Order Thought,
and quantum-biological interface layers.
"""

from .quantum_bio_bridge import (
    QuantumBioBridge,
    QuantumBioConfig,
    InterfaceSignal,
)
from .global_workspace_theory import (
    GlobalWorkspaceTheory,
    GlobalWorkspaceConfig,
    WorkspaceBroadcast,
)
from .higher_order_thought import (
    HigherOrderThought,
    HOTConfig,
    MetaRepresentation,
    HOTLevel,
)

__all__ = [
    # Quantum-biological bridge
    "QuantumBioBridge",
    "QuantumBioConfig",
    "InterfaceSignal",
    # Global workspace theory
    "GlobalWorkspaceTheory",
    "GlobalWorkspaceConfig",
    "WorkspaceBroadcast",
    # Higher-order thought
    "HigherOrderThought",
    "HOTConfig",
    "MetaRepresentation",
    "HOTLevel",
]

__version__ = "0.1.0"
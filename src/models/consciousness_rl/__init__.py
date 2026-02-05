"""
Consciousness-informed reinforcement learning models.

This module implements RL algorithms enhanced with consciousness metrics,
including self-evolving architectures, Phi maximization, and recursive
self-organization for autonomous agents.
"""

from .self_evolving_consciousness import (
    SelfEvolvingConsciousness,
    ConsciousnessConfig,
    ArchitecturalChange,
)
from .integrated_information_maximizer import (
    IntegratedInformationMaximizer,
    PhiOptimizer,
    PhiOptimizationConfig,
)
from .recursive_self_organization import (
    RecursiveSelfOrganization,
    SelfOrganizationConfig,
    EmergenceEvent,
)
from .consciousness_value_network import (
    ConsciousnessValueNetwork,
    ConsciousnessValueConfig,
    ConsciousnessAwarePolicy,
)

__all__ = [
    # Self-evolving consciousness
    "SelfEvolvingConsciousness",
    "ConsciousnessConfig",
    "ArchitecturalChange",
    # Phi maximization
    "IntegratedInformationMaximizer",
    "PhiOptimizer",
    "PhiOptimizationConfig",
    # Recursive self-organization
    "RecursiveSelfOrganization",
    "SelfOrganizationConfig",
    "EmergenceEvent",
    # Consciousness value network
    "ConsciousnessValueNetwork",
    "ConsciousnessValueConfig",
    "ConsciousnessAwarePolicy",
]

__version__ = "0.1.0"

"""
Quantum hardware simulation module for consciousnessX.

This module provides virtual quantum processor implementations for
Orchestrated Objective Reduction (Orch-OR) consciousness simulations,
including superconducting qubits, gate operations, and error correction.
"""

from .virtual_quantum_processor import (
    VirtualQuantumProcessor,
    QuantumProcessorConfig,
    GateType,
)
from .superconducting_qubit_sim import (
    SuperconductingQubit,
    TransmonQubit,
    QubitState,
    QubitConfig,
)
from .quantum_error_correction import (
    QuantumErrorCorrection,
    SurfaceCode,
    ErrorCorrectionConfig,
)

__all__ = [
    # Virtual quantum processor
    "VirtualQuantumProcessor",
    "QuantumProcessorConfig",
    "GateType",
    # Superconducting qubits
    "SuperconductingQubit",
    "TransmonQubit",
    "QubitState",
    "QubitConfig",
    # Error correction
    "QuantumErrorCorrection",
    "SurfaceCode",
    "ErrorCorrectionConfig",
]

__version__ = "0.1.0"
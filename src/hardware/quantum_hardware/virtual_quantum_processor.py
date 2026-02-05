"""Virtual quantum processor simulation for consciousnessX.

Implements a gate-based quantum processor with realistic noise models,
supporting superconducting qubit operations for Orch-OR simulations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class GateType(Enum):
    """Supported quantum gate types."""
    H = "hadamard"
    X = "pauli_x"
    Y = "pauli_y"
    Z = "pauli_z"
    CNOT = "cnot"
    RZ = "rotation_z"
    RX = "rotation_x"
    RY = "rotation_y"
    CZ = "controlled_z"
    SWAP = "swap"


@dataclass
class QuantumProcessorConfig:
    """Configuration for virtual quantum processor."""
    num_qubits: int = 8
    gate_fidelity: float = 0.999
    readout_fidelity: float = 0.95
    coherence_time_t1: float = 100e-6  # 100 microseconds
    coherence_time_t2: float = 50e-6   # 50 microseconds
    gate_time: float = 20e-9           # 20 nanoseconds
    temperature: float = 0.015         # 15 mK
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)


class VirtualQuantumProcessor:
    """Simulated quantum processor for Orch-OR consciousness models.
    
    This class simulates a superconducting quantum processor with realistic
    noise models including T1/T2 decoherence and gate fidelity errors.
    
    Attributes:
        config: Processor configuration parameters.
        state: Current quantum state vector.
        gate_history: Record of applied gates.
        
    Example:
        >>> config = QuantumProcessorConfig(num_qubits=4)
        >>> processor = VirtualQuantumProcessor(config)
        >>> processor.apply_gate(GateType.H, target=0)
        >>> result = processor.measure()
    """
    
    def __init__(self, config: Optional[QuantumProcessorConfig] = None) -> None:
        """Initialize quantum processor.
        
        Args:
            config: Processor configuration. Uses defaults if None.
        """
        self.config = config or QuantumProcessorConfig()
        self.state = self._initialize_state()
        self.gate_history: List[Dict[str, Any]] = []
        self._decoherence_rates = self._calculate_decoherence_rates()
        self._time_elapsed = 0.0
        logger.info(f"Initialized VirtualQuantumProcessor with "
                   f"{self.config.num_qubits} qubits")
    
    def _initialize_state(self) -> np.ndarray:
        """Initialize |0...0⟩ state."""
        state = np.zeros(2 ** self.config.num_qubits, dtype=complex)
        state[0] = 1.0
        return state
    
    def _calculate_decoherence_rates(self) -> Dict[str, float]:
        """Calculate decoherence rates from T1/T2 times."""
        gamma1 = 1.0 / self.config.coherence_time_t1
        gamma2 = 1.0 / self.config.coherence_time_t2
        gamma_phi = gamma2 - gamma1 / 2
        return {"T1": gamma1, "T2": gamma2, "dephasing": gamma_phi}
    
    def apply_gate(self, gate: GateType, target: int, 
                   control: Optional[int] = None,
                   params: Optional[Dict[str, float]] = None) -> None:
        """Apply quantum gate with noise simulation.
        
        Args:
            gate: Type of gate to apply.
            target: Target qubit index.
            control: Control qubit for multi-qubit gates.
            params: Optional gate parameters (e.g., rotation angles).
            
        Raises:
            ValueError: If qubit indices are invalid.
        """
        if target >= self.config.num_qubits:
            raise ValueError(f"Target qubit {target} out of range")
        
        # Update time elapsed
        self._time_elapsed += self.config.gate_time
        
        # Get gate matrix
        gate_matrix = self._get_gate_matrix(gate, params)
        
        # Apply gate to state
        self._apply_matrix(gate_matrix, target, control)
        
        # Apply decoherence
        self._apply_decoherence()
        
        # Record gate operation
        self.gate_history.append({
            "gate": gate.value,
            "target": target,
            "control": control,
            "params": params,
            "time": self._time_elapsed
        })
        
        logger.debug(f"Applied {gate.value} to qubit {target}")
    
    def _get_gate_matrix(self, gate: GateType, 
                         params: Optional[Dict[str, float]]) -> np.ndarray:
        """Get unitary matrix for gate type."""
        if gate == GateType.H:
            return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        elif gate == GateType.X:
            return np.array([[0, 1], [1, 0]])
        elif gate == GateType.Y:
            return np.array([[0, -1j], [1j, 0]])
        elif gate == GateType.Z:
            return np.array([[1, 0], [0, -1]])
        elif gate == GateType.CNOT:
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], 
                            [0, 0, 0, 1], [0, 0, 1, 0]])
        elif gate == GateType.CZ:
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], 
                            [0, 0, 1, 0], [0, 0, 0, -1]])
        elif gate == GateType.SWAP:
            return np.array([[1, 0, 0, 0], [0, 0, 1, 0], 
                            [0, 1, 0, 0], [0, 0, 0, 1]])
        elif gate == GateType.RZ and params:
            theta = params.get("theta", 0)
            return np.array([
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)]
            ])
        elif gate == GateType.RX and params:
            theta = params.get("theta", 0)
            return np.array([
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)]
            ])
        elif gate == GateType.RY and params:
            theta = params.get("theta", 0)
            return np.array([
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)]
            ])
        else:
            raise NotImplementedError(f"Gate {gate} not implemented")
    
    def _apply_matrix(self, matrix: np.ndarray, target: int,
                      control: Optional[int]) -> None:
        """Apply gate matrix to state vector using tensor operations."""
        n = self.config.num_qubits
        
        if control is None:
            # Single qubit gate
            # Create full operator by tensoring identities
            full_op = np.eye(1, dtype=complex)
            for i in range(n):
                if i == target:
                    full_op = np.kron(full_op, matrix)
                else:
                    full_op = np.kron(full_op, np.eye(2))
        else:
            # Two qubit gate
            if target < control:
                q1, q2 = target, control
            else:
                q1, q2 = control, target
            
            full_op = np.eye(1, dtype=complex)
            for i in range(n):
                if i == q1:
                    full_op = np.kron(full_op, matrix.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4))
                elif i == q2:
                    full_op = np.kron(full_op, np.eye(2))
                else:
                    full_op = np.kron(full_op, np.eye(2))
        
        self.state = full_op @ self.state
        
        # Normalize
        self.state = self.state / np.linalg.norm(self.state)
    
    def _apply_decoherence(self) -> None:
        """Apply T1/T2 decoherence noise using Kraus operators."""
        # Amplitude damping (T1)
        if np.random.random() < self._decoherence_rates["T1"] * self.config.gate_time:
            for i in range(len(self.state)):
                if i % 2 == 1:  # |1⟩ state
                    prob_decay = self.config.gate_time / self.config.coherence_time_t1
                    if np.random.random() < prob_decay:
                        self.state[i - 1] += self.state[i]
                        self.state[i] = 0
        
        # Dephasing (T2)
        dephasing_prob = self._decoherence_rates["dephasing"] * self.config.gate_time
        if np.random.random() < dephasing_prob:
            # Apply random phase
            phase = np.random.choice([-1, 1]) * np.pi
            for i in range(len(self.state)):
                self.state[i] *= np.exp(1j * phase * (i % 2))
        
        # Normalize
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state = self.state / norm
    
    def measure(self, qubits: Optional[List[int]] = None) -> Dict[int, int]:
        """Measure qubits with readout noise.
        
        Args:
            qubits: List of qubits to measure. Measures all if None.
            
        Returns:
            Dictionary mapping qubit index to measurement result (0 or 1).
        """
        qubits = qubits or list(range(self.config.num_qubits))
        results = {}
        
        for q in sorted(qubits, reverse=True):  # Measure from highest to lowest
            # Calculate probability of |1⟩ for this qubit
            prob_1 = 0.0
            for i in range(len(self.state)):
                if (i >> (self.config.num_qubits - 1 - q)) & 1:
                    prob_1 += np.abs(self.state[i]) ** 2
            
            # Add readout error
            prob_1 = prob_1 * self.config.readout_fidelity + \
                     (1 - prob_1) * (1 - self.config.readout_fidelity)
            
            result = 1 if np.random.random() < prob_1 else 0
            results[q] = result
            
            # Collapse state
            self._collapse(q, result)
        
        logger.debug(f"Measurement results: {results}")
        return results
    
    def _collapse(self, qubit: int, result: int) -> None:
        """Collapse state after measurement."""
        mask = 1 << (self.config.num_qubits - 1 - qubit)
        
        for i in range(len(self.state)):
            if ((i & mask) != 0) != result:
                self.state[i] = 0
        
        # Normalize
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state = self.state / norm
    
    def get_coherence_metrics(self) -> Dict[str, float]:
        """Calculate current coherence metrics."""
        density_matrix = np.outer(self.state, self.state.conj())
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 0]
        
        # Von Neumann entropy
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        # Purity
        purity = np.real(np.trace(density_matrix @ density_matrix))
        
        return {
            "purity": float(purity),
            "von_neumann_entropy": float(entropy),
            "t1_remaining": max(0, self.config.coherence_time_t1 - self._time_elapsed),
            "t2_remaining": max(0, self.config.coherence_time_t2 - self._time_elapsed),
            "gate_count": len(self.gate_history)
        }
    
    def reset(self) -> None:
        """Reset processor to |0...0⟩ state."""
        self.state = self._initialize_state()
        self.gate_history.clear()
        self._time_elapsed = 0.0
        logger.info("Processor reset to initial state")
    
    def get_state_vector(self) -> np.ndarray:
        """Get current state vector."""
        return self.state.copy()
    
    def apply_circuit(self, circuit: List[Tuple[GateType, int, Optional[int], 
                                                Optional[Dict[str, float]]]]) -> None:
        """Apply a sequence of gates as a circuit.
        
        Args:
            circuit: List of tuples (gate_type, target, control, params).
        """
        for gate, target, control, params in circuit:
            self.apply_gate(gate, target, control, params)
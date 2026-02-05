"""
Unit tests for QuantumOrchOR
"""

import pytest
import numpy as np
from src.core.quantum_orch_or import QuantumOrchOR


class TestQuantumOrchOR:
    """Test suite for QuantumOrchOR class"""

    @pytest.fixture
    def orch_or(self):
        """Create a QuantumOrchOR instance"""
        return QuantumOrchOR(num_qubits=8, reduction_time=1e-3)

    def test_initialization(self, orch_or):
        """Test OrchOR initialization"""
        assert orch_or.num_qubits == 8
        assert orch_or.reduction_time == 1e-3
        assert hasattr(orch_or, "quantum_state")

    def test_initialize_superposition(self, orch_or):
        """Test quantum superposition initialization"""
        orch_or.initialize_superposition()

        assert orch_or.quantum_state is not None
        assert isinstance(orch_or.quantum_state, np.ndarray)
        assert orch_or.quantum_state.shape == (2**8,)

    def test_compute_phi(self, orch_or):
        """Test phi (integrated information) computation"""
        orch_or.initialize_superposition()
        phi = orch_or.compute_phi()

        assert 0 <= phi <= 1
        assert isinstance(phi, float)

    def test_simulate_orchestration(self, orch_or):
        """Test orchestration simulation"""
        orch_or.initialize_superposition()
        result = orch_or.simulate_orchestration(dt=1e-4, steps=100)

        assert result is not None
        assert "phi_history" in result
        assert "quantum_state_history" in result
        assert len(result["phi_history"]) == 100

    def test_simulate_reduction(self, orch_or):
        """Test objective reduction simulation"""
        orch_or.initialize_superposition()
        final_state = orch_or.simulate_reduction()

        assert final_state is not None
        assert isinstance(final_state, np.ndarray)
        assert len(final_state) == 2**8

    def test_compute_consciousness_moment(self, orch_or):
        """Test consciousness moment computation"""
        orch_or.initialize_superposition()
        consciousness_moment = orch_or.compute_consciousness_moment()

        assert consciousness_moment is not None
        assert isinstance(consciousness_moment, dict)
        assert "phi" in consciousness_moment
        assert "duration" in consciousness_moment
        assert "content" in consciousness_moment

    def test_save_consciousness_moment(self, orch_or, temp_dir):
        """Test saving consciousness moment"""
        orch_or.initialize_superposition()
        moment = orch_or.compute_consciousness_moment()

        filepath = temp_dir / "consciousness_moment.npz"
        orch_or.save_consciousness_moment(moment, str(filepath))

        assert filepath.exists()

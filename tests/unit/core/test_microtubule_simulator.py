"""
Unit tests for MicrotubuleSimulator
"""

import pytest
import numpy as np
from src.core.microtubule_simulator import MicrotubuleSimulator, MicrotubuleConfig


class TestMicrotubuleSimulator:
    """Test suite for MicrotubuleSimulator class"""

    @pytest.fixture
    def simulator(self):
        """Create a MicrotubuleSimulator instance"""
        config = MicrotubuleConfig(
            num_tubulins_per_filament=100,
            microtubule_length_nm=1000.0
        )
        return MicrotubuleSimulator(config)

    def test_initialization(self, simulator):
        """Test simulator initialization"""
        assert simulator.config.num_tubulins_per_filament == 100
        assert simulator.config.microtubule_length_nm == 1000.0
        assert hasattr(simulator, "lattice")
        assert hasattr(simulator, "coherence_sim")

    def test_simulate_quantum_dynamics(self, simulator):
        """Test quantum dynamics simulation"""
        dt = 0.1
        steps = 10

        result = simulator.simulate_quantum_dynamics(dt, steps)

        assert result is not None
        assert "quantum_states" in result
        assert "coherence" in result
        assert len(result["quantum_states"]) == steps + 1

    def test_compute_coherence(self, simulator):
        """Test coherence computation"""
        simulator.initialize_quantum_state()
        coherence = simulator.compute_coherence()

        assert 0 <= coherence <= 1
        assert isinstance(coherence, float)

    def test_update_microtubule_state(self, simulator):
        """Test microtubule state update"""
        initial_state = simulator.tubulin_states.copy()

        simulator.update_microtubule_state(temperature=300.0)

        assert not np.array_equal(simulator.tubulin_states, initial_state)
        assert simulator.tubulin_states.shape == (100,)

    def test_compute_orch_or(self, simulator):
        """Test Orchestrated Objective Reduction computation"""
        simulator.initialize_quantum_state()

        phi = simulator.compute_orch_or()

        assert phi >= 0
        assert isinstance(phi, float)

    def test_save_state(self, simulator, temp_dir):
        """Test saving simulator state"""
        simulator.initialize_quantum_state()

        filepath = temp_dir / "microtubule_state.npz"
        simulator.save_state(str(filepath))

        assert filepath.exists()

    def test_load_state(self, simulator, temp_dir):
        """Test loading simulator state"""
        simulator.initialize_quantum_state()

        filepath = temp_dir / "microtubule_state.npz"
        simulator.save_state(str(filepath))

        new_simulator = MicrotubuleSimulator(num_tubulins=100, length=1000.0)
        new_simulator.load_state(str(filepath))

        np.testing.assert_array_equal(simulator.quantum_states, new_simulator.quantum_states)

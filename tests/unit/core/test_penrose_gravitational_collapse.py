"""
Unit tests for PenroseGravitationalCollapse
"""

import pytest
import numpy as np
from src.core.penrose_gravitational_collapse import GravitationalCollapseCalculator, PenroseParameters


class TestPenroseGravitationalCollapse:
    """Test suite for PenroseGravitationalCollapse class"""

    @pytest.fixture
    def collapse(self):
        """Create a PenroseGravitationalCollapse instance"""
        params = PenroseParameters(
            mass_kg=1e-26,
            energy_joules=1e-10
        )
        return GravitationalCollapseCalculator(params)

    def test_initialization(self, collapse):
        """Test collapse model initialization"""
        assert collapse.params.mass_kg == 1e-26
        assert collapse.params.energy_joules == 1e-10
        assert hasattr(collapse, "calculate_collapse_time")

    def test_compute_reduction_time(self, collapse):
        """Test reduction time computation"""
        t_g = collapse.compute_reduction_time()

        assert t_g > 0
        assert isinstance(t_g, float)

    def test_simulate_collapse(self, collapse):
        """Test gravitational collapse simulation"""
        initial_state = np.array([0.7, 0.7j, 0.0, 0.0])
        final_state = collapse.simulate_collapse(initial_state)

        assert final_state is not None
        assert isinstance(final_state, np.ndarray)
        # Final state should be a basis state (only one non-zero component)
        non_zero = np.abs(final_state) > 1e-10
        assert np.sum(non_zero) == 1

    def test_compute_gravitational_energy(self, collapse):
        """Test gravitational energy computation"""
        energy = collapse.compute_gravitational_energy()

        assert energy > 0
        assert isinstance(energy, float)

    def test_apply_uncertainty_principle(self, collapse):
        """Test uncertainty principle application"""
        state = np.array([0.7, 0.7j, 0.0, 0.0])
        uncertain_state = collapse.apply_uncertainty_principle(state)

        assert uncertain_state is not None
        assert isinstance(uncertain_state, np.ndarray)
        # Uncertainty should preserve normalization
        assert np.abs(np.linalg.norm(uncertain_state) - 1.0)

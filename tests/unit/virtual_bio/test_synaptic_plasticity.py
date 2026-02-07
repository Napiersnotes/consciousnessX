"""
Unit tests for SynapticPlasticity
"""

import pytest
import numpy as np
from src.virtual_bio.synaptic_plasticity import SynapticPlasticity


class TestSynapticPlasticity:
    """Test suite for SynapticPlasticity class"""

    @pytest.fixture
    def plasticity(self):
        """Create a SynapticPlasticity instance"""
        config = {
            "stdp_learning_rate": 0.01,
            "stdp_tau_plus": 20.0,
            "stdp_tau_minus": 20.0,
            "stdp_a_plus": 0.1,
            "stdp_a_minus": 0.1,
            "homeostatic_target": 1.0,
            "homeostatic_rate": 0.001,
        }
        plasticity = SynapticPlasticity(config)
        plasticity.initialize(n_neurons=100)
        return plasticity

    def test_initialization(self, plasticity):
        """Test synaptic plasticity initialization"""
        assert plasticity.n_neurons == 100
        assert plasticity.config["stdp_learning_rate"] == 0.01
        assert plasticity.synaptic_weights is not None
        assert plasticity.synaptic_weights.shape == (100, 100)

    def test_apply_hebbian_learning(self, plasticity):
        """Test Hebbian learning application"""
        pre_synaptic = np.random.rand(100)
        post_synaptic = np.random.rand(100)

        initial_weights = plasticity.synaptic_weights.copy()
        plasticity.apply_hebbian_learning(pre_synaptic, post_synaptic)

        assert not np.array_equal(plasticity.synaptic_weights, initial_weights)

    def test_apply_stdp(self, plasticity):
        """Test Spike-Timing-Dependent Plasticity"""
        pre_times = np.sort(np.random.rand(100) * 100)
        post_times = np.sort(np.random.rand(100) * 100)

        initial_weights = plasticity.synaptic_weights.copy()
        plasticity.apply_stdp(pre_times, post_times)

        assert not np.array_equal(plasticity.synaptic_weights, initial_weights)

    def test_apply_anti_hebbian(self, plasticity):
        """Test anti-Hebbian learning"""
        pre_synaptic = np.random.rand(100)
        post_synaptic = np.random.rand(100)

        initial_weights = plasticity.synaptic_weights.copy()
        plasticity.apply_anti_hebbian(pre_synaptic, post_synaptic)

        assert not np.array_equal(plasticity.synaptic_weights, initial_weights)

    def test_normalize_weights(self, plasticity):
        """Test weight normalization"""
        plasticity.synaptic_weights = np.random.rand(100, 100) * 10
        plasticity.normalize_weights()

        max_weight = np.max(plasticity.synaptic_weights)
        assert max_weight <= 1.0

    def test_simulate_long_term_potentiation(self, plasticity):
        """Test LTP simulation"""
        stimulus_strength = 0.8
        duration = 100

        initial_weights = plasticity.synaptic_weights.copy()
        plasticity.simulate_long_term_potentiation(stimulus_strength, duration)

        assert not np.array_equal(plasticity.synaptic_weights, initial_weights)
        # LTP should strengthen weights
        assert np.mean(plasticity.synaptic_weights) > np.mean(initial_weights)

    def test_simulate_long_term_depression(self, plasticity):
        """Test LTD simulation"""
        stimulus_strength = 0.8
        duration = 100

        initial_weights = plasticity.synaptic_weights.copy()
        plasticity.simulate_long_term_depression(stimulus_strength, duration)

        assert not np.array_equal(plasticity.synaptic_weights, initial_weights)
        # LTD should weaken weights
        assert np.mean(plasticity.synaptic_weights) < np.mean(initial_weights)

    def test_save_weights(self, plasticity, temp_dir):
        """Test saving weights"""
        filepath = temp_dir / "synaptic_weights.npy"
        plasticity.save_weights(str(filepath))

        assert filepath.exists()

    def test_load_weights(self, plasticity, temp_dir):
        """Test loading weights"""
        filepath = temp_dir / "synaptic_weights.npy"
        plasticity.save_weights(str(filepath))

        new_plasticity = SynapticPlasticity(num_neurons=100)
        new_plasticity.load_weights(str(filepath))

        np.testing.assert_array_equal(plasticity.synaptic_weights, new_plasticity.synaptic_weights)

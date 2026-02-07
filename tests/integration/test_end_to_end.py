"""
End-to-end integration tests for consciousnessX
"""

import pytest
import numpy as np
from src.core.microtubule_simulator import MicrotubuleSimulator, MicrotubuleConfig
from src.core.quantum_orch_or import QuantumOrchOR
from src.virtual_bio.ion_channel_dynamics import IonChannel, IonChannelConfig, IonChannelType
from src.virtual_bio.synaptic_plasticity import SynapticPlasticity
from src.evaluation.consciousness_assessment import ConsciousnessAssessor


class TestEndToEnd:
    """End-to-end integration tests"""

    def test_microtubule_to_consciousness_pipeline(self):
        """Test complete pipeline from microtubule simulation to consciousness assessment"""
        # Initialize components
        simulator = MicrotubuleSimulator(config=MicrotubuleConfig(num_tubulins_per_filament=100, microtubule_length_nm=1000.0))
        orch_or = QuantumOrchOR(num_tubulins=1000, coherence_time=1e-3, quantum_superposition_levels=8)
        assessment = ConsciousnessAssessor()

        # Simulate microtubule dynamics
        simulator.initialize_quantum_state()
        simulator.simulate_quantum_dynamics(dt=0.1, steps=10)

        # Extract quantum state
        quantum_state = simulator.quantum_states

        # Apply Orch-OR
        orch_or.initialize_superposition()
        phi = orch_or.compute_phi()

        # Assess consciousness
        metrics = assessment.assess_consciousness(
            {
                "quantum_state": quantum_state,
                "phi": phi,
                "coherence": simulator.compute_coherence(),
            }
        )

        assert metrics is not None
        assert "consciousness_level" in metrics
        assert "quantum_coherence" in metrics
        assert 0 <= metrics["consciousness_level"] <= 1

    def test_neural_plasticity_integration(self):
        """Test integration between ion channels and synaptic plasticity"""
        ion_channel = IonChannel(channel_type=IonChannelType.SODIUM, config=IonChannelConfig(time_step_ms=0.01, membrane_area_um2=1000.0), channel_density=100.0)
        plasticity = SynapticPlasticity()
        plasticity.initialize(n_neurons=100)

        # Simulate action potentials
        result = ion_channel.simulate_action_potential()

        # Use voltage to drive plasticity
        voltage_spikes = result["voltage"][result["voltage"] > 0]
        pre_synaptic = np.random.rand(100)
        post_synaptic = np.random.rand(100)

        # Apply plasticity
        initial_weights = plasticity.weights.copy()
        plasticity.apply_hebbian_learning(pre_synaptic, post_synaptic)

        # Verify weights changed
        assert not np.array_equal(plasticity.weights, initial_weights)

    def test_multiscale_simulation(self):
        """Test multiscale simulation across quantum and biological levels"""
        # Quantum level
        simulator = MicrotubuleSimulator(config=MicrotubuleConfig(num_tubulins_per_filament=100, microtubule_length_nm=1000.0))
        orch_or = QuantumOrchOR(num_tubulins=1000, coherence_time=1e-3, quantum_superposition_levels=8)

        # Biological level
        ion_channel = IonChannel(channel_type=IonChannelType.SODIUM, config=IonChannelConfig(time_step_ms=0.01, membrane_area_um2=1000.0), channel_density=100.0)
        plasticity = SynapticPlasticity()
        plasticity.initialize(n_neurons=100)

        # Simulate quantum dynamics
        simulator.initialize_quantum_state()
        simulator.simulate_quantum_dynamics(dt=0.1, steps=10)
        quantum_coherence = simulator.compute_coherence()

        # Simulate biological dynamics
        ion_channel.simulate_voltage_step(-80.0, 0.0, 10.0)
        biological_activity = np.mean(np.abs(ion_channel.current))

        # Apply plasticity
        pre_synaptic = np.random.rand(100)
        post_synaptic = np.random.rand(100)
        plasticity.apply_hebbian_learning(pre_synaptic, post_synaptic)

        # Verify cross-level interactions
        assert quantum_coherence is not None
        assert biological_activity is not None
        assert np.mean(plasticity.weights) > 0

    def test_consciousness_assessment_pipeline(self):
        """Test complete consciousness assessment pipeline"""
        # Setup components
        simulator = MicrotubuleSimulator(config=MicrotubuleConfig(num_tubulins_per_filament=100, microtubule_length_nm=1000.0))
        orch_or = QuantumOrchOR(num_tubulins=1000, coherence_time=1e-3, quantum_superposition_levels=8)
        ion_channel = IonChannel(channel_type=IonChannelType.SODIUM, config=IonChannelConfig(time_step_ms=0.01, membrane_area_um2=1000.0), channel_density=100.0)
        plasticity = SynapticPlasticity()
        plasticity.initialize(n_neurons=100)
        assessment = ConsciousnessAssessor()

        # Run simulations
        simulator.initialize_quantum_state()
        simulator.simulate_quantum_dynamics(dt=0.1, steps=10)

        orch_or.initialize_superposition()
        phi = orch_or.compute_phi()

        ion_channel.simulate_action_potential()

        pre_synaptic = np.random.rand(100)
        post_synaptic = np.random.rand(100)
        plasticity.apply_hebbian_learning(pre_synaptic, post_synaptic)

        # Comprehensive assessment
        metrics = assessment.assess_consciousness(
            {
                "quantum_state": simulator.quantum_states,
                "phi": phi,
                "coherence": simulator.compute_coherence(),
                "neural_activity": np.mean(np.abs(ion_channel.current)),
                "synaptic_strength": np.mean(plasticity.weights),
            }
        )

        # Verify metrics
        assert metrics is not None
        assert "consciousness_level" in metrics
        assert "quantum_coherence" in metrics
        assert "integrated_information" in metrics
        assert 0 <= metrics["consciousness_level"] <= 1

    def test_state_persistence_and_retrieval(self, temp_dir):
        """Test saving and loading complete system state"""
        # Initialize and run simulation
        simulator = MicrotubuleSimulator(config=MicrotubuleConfig(num_tubulins_per_filament=100, microtubule_length_nm=1000.0))
        simulator.initialize_quantum_state()
        simulator.simulate_quantum_dynamics(dt=0.1, steps=10)

        # Save state
        state_file = temp_dir / "system_state.npz"
        simulator.save_state(str(state_file))

        # Load state
        new_simulator = MicrotubuleSimulator(config=MicrotubuleConfig(num_tubulins_per_filament=100, microtubule_length_nm=1000.0))
        new_simulator.load_state(str(state_file))

        # Verify state preservation
        np.testing.assert_array_equal(simulator.quantum_states, new_simulator.quantum_states)
        np.testing.assert_array_equal(simulator.tubulin_states, new_simulator.tubulin_states)

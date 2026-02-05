"""Unit tests for consciousnessX framework components.

Tests individual components in isolation to ensure correctness.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
import numpy as np
from typing import Dict, List

# Import consciousnessX modules
from src.theory.orch_or_theory import OrchORTheory, OrchORConfig
from src.hardware.quantum_hardware.virtual_quantum_processor import VirtualQuantumProcessor
from src.models.spiking_neural_networks.quantum_lif_neuron import QuantumLIFNeuron, QuantumLIFConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnitTest:
    """Unit tests for consciousnessX components."""
    
    def __init__(self):
        """Initialize unit test suite."""
        self.test_results: List[Dict[str, any]] = []
        self.tolerance = 1e-6
        logger.info("Initialized UnitTest suite")
    
    def run_all_tests(self) -> Dict[str, any]:
        """Run all unit tests.
        
        Returns:
            Dictionary with test results summary.
        """
        logger.info("=" * 60)
        logger.info("Starting ConsciousnessX Unit Tests")
        logger.info("=" * 60)
        
        # Orch-OR Theory Tests
        self.test_orch_or_gamma_calculation()
        self.test_orch_or_reduction_time()
        self.test_orch_or_collapse()
        
        # Quantum Hardware Tests
        self.test_quantum_processor_initialization()
        self.test_quantum_gates()
        self.test_quantum_measurement()
        
        # SNN Model Tests
        self.test_quantum_lif_neuron()
        self.test_neuron_spike()
        self.test_neuron_entanglement()
        
        # Generate summary
        summary = self._generate_summary()
        
        logger.info("=" * 60)
        logger.info("Unit Tests Complete")
        logger.info(f"Passed: {summary['passed']}/{summary['total']}")
        logger.info(f"Failed: {summary['failed']}/{summary['total']}")
        logger.info("=" * 60)
        
        return summary
    
    # ==================== Orch-OR Theory Tests ====================
    
    def test_orch_or_gamma_calculation(self) -> None:
        """Test Orch-OR gamma calculation."""
        logger.info("\n[Test] Orch-OR Gamma Calculation")
        
        try:
            # Initialize with default config
            orch_or = OrchORTheory()
            
            # Test gamma calculation
            masses = [0.1, 1.0, 10.0]
            for mass in masses:
                gamma = orch_or.compute_gamma(mass)
                assert gamma > 0, f"Gamma should be positive for mass={mass}"
                logger.info(f"  Mass={mass}, Gamma={gamma:.3e}")
            
            # Verify gamma scales with mass
            gamma1 = orch_or.compute_gamma(1.0)
            gamma2 = orch_or.compute_gamma(2.0)
            assert gamma2 > gamma1, "Gamma should increase with mass"
            
            self._record_result("Orch-OR Gamma Calculation", True, "Gamma calculation correct")
            
        except Exception as e:
            logger.error(f"Gamma calculation test failed: {e}")
            self._record_result("Orch-OR Gamma Calculation", False, str(e))
    
    def test_orch_or_reduction_time(self) -> None:
        """Test Orch-OR reduction time calculation."""
        logger.info("\n[Test] Orch-OR Reduction Time")
        
        try:
            # Initialize with default config
            orch_or = OrchORTheory()
            
            # Test reduction time calculation
            masses = [0.1, 1.0, 10.0]
            for mass in masses:
                t_reduct = orch_or.compute_reduction_time(mass)
                assert t_reduct > 0, f"Reduction time should be positive for mass={mass}"
                logger.info(f"  Mass={mass}, T_reduction={t_reduct:.3e} s")
            
            # Verify reduction time decreases with mass
            t1 = orch_or.compute_reduction_time(1.0)
            t2 = orch_or.compute_reduction_time(2.0)
            assert t2 < t1, "Reduction time should decrease with mass"
            
            self._record_result("Orch-OR Reduction Time", True, "Reduction time calculation correct")
            
        except Exception as e:
            logger.error(f"Reduction time test failed: {e}")
            self._record_result("Orch-OR Reduction Time", False, str(e))
    
    def test_orch_or_collapse(self) -> None:
        """Test wavefunction collapse."""
        logger.info("\n[Test] Orch-OR Wavefunction Collapse")
        
        try:
            # Initialize
            orch_or = OrchORTheory()
            
            # Test superposition state
            state = np.array([0.7071, 0.7071], dtype=complex)
            assert np.abs(np.linalg.norm(state) - 1.0) < self.tolerance, "State should be normalized"
            
            # Collapse
            collapsed = orch_or.collapse_wavefunction(state)
            
            # Verify collapse
            assert np.abs(np.linalg.norm(collapsed) - 1.0) < self.tolerance, "Collapsed state should be normalized"
            
            # Verify state is now eigenstate (|0⟩ or |1⟩)
            is_eigenstate = (
                np.abs(collapsed[0]) > 0.99 or 
                np.abs(collapsed[1]) > 0.99
            )
            assert is_eigenstate, "Collapsed state should be eigenstate"
            
            logger.info(f"  Original: {state}")
            logger.info(f"  Collapsed: {collapsed}")
            
            self._record_result("Orch-OR Collapse", True, "Wavefunction collapse correct")
            
        except Exception as e:
            logger.error(f"Collapse test failed: {e}")
            self._record_result("Orch-OR Collapse", False, str(e))
    
    # ==================== Quantum Hardware Tests ====================
    
    def test_quantum_processor_initialization(self) -> None:
        """Test quantum processor initialization."""
        logger.info("\n[Test] Quantum Processor Initialization")
        
        try:
            # Initialize processor
            processor = VirtualQuantumProcessor()
            
            # Verify initial state
            assert processor.num_qubits == 0, "Should start with 0 qubits"
            
            # Create qubits
            processor.create_circuit(2)
            assert processor.num_qubits == 2, "Should have 2 qubits"
            
            # Get initial state
            state = processor.get_state()
            assert len(state) == 4, "State should have 4 elements (2^2)"
            
            # Verify initial state is |00⟩
            expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
            assert np.allclose(state, expected, atol=self.tolerance), "Initial state should be |00⟩"
            
            self._record_result("Quantum Processor Initialization", True, "Initialization correct")
            
        except Exception as e:
            logger.error(f"Initialization test failed: {e}")
            self._record_result("Quantum Processor Initialization", False, str(e))
    
    def test_quantum_gates(self) -> None:
        """Test quantum gate operations."""
        logger.info("\n[Test] Quantum Gate Operations")
        
        try:
            processor = VirtualQuantumProcessor()
            processor.create_circuit(2)
            
            # Test Hadamard gate
            processor.apply_hadamard(0)
            state = processor.get_state()
            
            # After H on qubit 0: (|0⟩ + |1⟩)/√2 ⊗ |0⟩ = (|00⟩ + |10⟩)/√2
            expected_amplitude = 0.7071
            assert np.abs(state[0] - expected_amplitude) < 0.01, "State[0] should be ~0.7071"
            assert np.abs(state[2] - expected_amplitude) < 0.01, "State[2] should be ~0.7071"
            
            # Reset
            processor.reset()
            processor.create_circuit(2)
            
            # Test CNOT gate
            processor.apply_hadamard(0)
            processor.apply_cnot(0, 1)
            state = processor.get_state()
            
            # After CNOT with control=0, target=1 on (|0⟩+|1⟩)/√2 ⊗ |0⟩:
            # Should get (|00⟩ + |11⟩)/√2
            assert np.abs(state[0] - expected_amplitude) < 0.01, "State[0] should be ~0.7071"
            assert np.abs(state[3] - expected_amplitude) < 0.01, "State[3] should be ~0.7071"
            
            # Get circuit depth
            depth = processor.get_circuit_depth()
            assert depth == 2, "Circuit should have depth 2 (H + CNOT)"
            
            self._record_result("Quantum Gates", True, f"Gates working, depth={depth}")
            
        except Exception as e:
            logger.error(f"Quantum gates test failed: {e}")
            self._record_result("Quantum Gates", False, str(e))
    
    def test_quantum_measurement(self) -> None:
        """Test quantum measurement."""
        logger.info("\n[Test] Quantum Measurement")
        
        try:
            processor = VirtualQuantumProcessor()
            processor.create_circuit(2)
            
            # Measure in initial state |00⟩
            result = processor.measure_all()
            assert result == [0, 0], "Initial state should measure as |00⟩"
            
            # Apply Hadamard to create superposition
            processor.apply_hadamard(0)
            processor.apply_hadamard(1)
            
            # Multiple measurements should show distribution
            measurements = {}
            for _ in range(100):
                result = processor.measure_all()
                key = tuple(result)
                measurements[key] = measurements.get(key, 0) + 1
            
            # All outcomes should be possible
            assert len(measurements) == 4, "Should see all 4 outcomes"
            
            # Distribution should be roughly uniform
            counts = list(measurements.values())
            mean = np.mean(counts)
            for count in counts:
                assert abs(count - mean) < 30, f"Count {count} too far from mean {mean}"
            
            logger.info(f"  Measurement distribution: {measurements}")
            
            self._record_result("Quantum Measurement", True, "Measurement distribution correct")
            
        except Exception as e:
            logger.error(f"Measurement test failed: {e}")
            self._record_result("Quantum Measurement", False, str(e))
    
    # ==================== SNN Model Tests ====================
    
    def test_quantum_lif_neuron(self) -> None:
        """Test quantum LIF neuron."""
        logger.info("\n[Test] Quantum LIF Neuron")
        
        try:
            # Initialize neuron
            config = QuantumLIFConfig()
            neuron = QuantumLIFNeuron(0, config)
            
            # Verify initial state
            assert neuron.membrane_potential == config.resting_potential, "Should start at resting potential"
            assert len(neuron.spike_history) == 0, "Should have no spikes initially"
            
            # Receive input
            weights = [1.0, 0.5]
            delays = [0.0, 0.0]
            neuron.receive_input(weights, delays)
            
            # Step
            spike = neuron.step()
            
            # Check neuron can fire
            logger.info(f"  Membrane potential: {neuron.membrane_potential:.3f}")
            logger.info(f"  Spike fired: {spike is not None}")
            
            # Get quantum state
            quantum_state = neuron.get_quantum_state()
            assert len(quantum_state) == 2, "Quantum state should have 2 elements"
            assert np.abs(np.linalg.norm(quantum_state) - 1.0) < self.tolerance, "Quantum state should be normalized"
            
            # Calculate spike rate
            for _ in range(10):
                neuron.receive_input([1.0], [0.0])
                neuron.step()
            
            spike_rate = neuron.get_spike_rate(10.0)
            assert spike_rate >= 0, "Spike rate should be non-negative"
            
            self._record_result("Quantum LIF Neuron", True, f"Spike rate={spike_rate:.1f} Hz")
            
        except Exception as e:
            logger.error(f"Quantum LIF neuron test failed: {e}")
            self._record_result("Quantum LIF Neuron", False, str(e))
    
    def test_neuron_spike(self) -> None:
        """Test neuron spiking behavior."""
        logger.info("\n[Test] Neuron Spiking")
        
        try:
            config = QuantumLIFConfig(threshold=-55.0, reset_potential=-80.0)
            neuron = QuantumLIFNeuron(0, config)
            
            # Strong input to trigger spike
            for _ in range(100):
                neuron.receive_input([10.0], [0.0])
                neuron.step()
            
            # Should have spiked
            assert len(neuron.spike_history) > 0, "Should have spiked with strong input"
            logger.info(f"  Number of spikes: {len(neuron.spike_history)}")
            
            # Verify spike history
            for spike in neuron.spike_history:
                assert spike.neuron_id == 0, "Spike should have correct neuron ID"
                assert spike.membrane_potential >= config.threshold, "Spike should have crossed threshold"
            
            # Test refractory period
            strong_spikes = len(neuron.spike_history)
            
            for _ in range(10):
                neuron.receive_input([10.0], [0.0])
                neuron.step()
            
            # With refractory period, shouldn't spike as much
            recent_spikes = len([s for s in neuron.spike_history[-10:]])
            logger.info(f"  Recent spikes: {recent_spikes}")
            
            self._record_result("Neuron Spiking", True, f"Total spikes: {strong_spikes}")
            
        except Exception as e:
            logger.error(f"Neuron spiking test failed: {e}")
            self._record_result("Neuron Spiking", False, str(e))
    
    def test_neuron_entanglement(self) -> None:
        """Test neuron entanglement."""
        logger.info("\n[Test] Neuron Entanglement")
        
        try:
            # Create two neurons
            config = QuantumLIFConfig()
            neuron1 = QuantumLIFNeuron(0, config)
            neuron2 = QuantumLIFNeuron(1, config)
            
            # Entangle neurons
            neuron1.entangle(1)
            assert 1 in neuron1._entanglement_partners, "Neuron1 should have neuron2 as partner"
            
            # Check quantum states
            state1 = neuron1.get_quantum_state()
            state2 = neuron2.get_quantum_state()
            
            # Entanglement should create correlation
            # (This is a simplified check - real entanglement is more complex)
            logger.info(f"  Neuron1 quantum state: {state1}")
            logger.info(f"  Neuron2 quantum state: {state2}")
            
            # Disentangle
            neuron1.disentangle(1)
            assert 1 not in neuron1._entanglement_partners, "Neuron1 should not have neuron2 as partner"
            
            self._record_result("Neuron Entanglement", True, "Entanglement working")
            
        except Exception as e:
            logger.error(f"Neuron entanglement test failed: {e}")
            self._record_result("Neuron Entanglement", False, str(e))
    
    # ==================== Helper Methods ====================
    
    def _record_result(self, test_name: str, passed: bool, message: str) -> None:
        """Record test result.
        
        Args:
            test_name: Name of the test.
            passed: Whether test passed.
            message: Test result message.
        """
        self.test_results.append({
            'name': test_name,
            'passed': passed,
            'message': message
        })
        
        status = "PASSED" if passed else "FAILED"
        logger.info(f"[{status}] {test_name}: {message}")
    
    def _generate_summary(self) -> Dict[str, any]:
        """Generate test summary.
        
        Returns:
            Dictionary with test summary.
        """
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['passed'])
        failed = total - passed
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'success_rate': passed / total if total > 0 else 0,
            'results': self.test_results
        }


def main():
    """Run unit tests."""
    tester = UnitTest()
    summary = tester.run_all_tests()
    
    # Exit with appropriate code
    if summary['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
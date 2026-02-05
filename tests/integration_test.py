"""Integration tests for consciousnessX framework.

Tests the complete system with all components working together
to simulate quantum consciousness phenomena.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
from typing import Dict, List

# Import consciousnessX modules
from src.theory.orch_or_theory import OrchORTheory
from src.hardware.quantum_hardware.virtual_quantum_processor import VirtualQuantumProcessor
from src.models.spiking_neural_networks.cortical_column_sim import CorticalColumn
from src.models.consciousness_rl.self_evolving_consciousness import SelfEvolvingConsciousness
from src.models.hybrid_architectures.quantum_bio_bridge import QuantumBioBridge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegrationTest:
    """Integration tests for consciousnessX."""

    def __init__(self):
        """Initialize integration test suite."""
        self.test_results: List[Dict[str, any]] = []
        logger.info("Initialized IntegrationTest suite")

    def run_all_tests(self) -> Dict[str, any]:
        """Run all integration tests.

        Returns:
            Dictionary with test results summary.
        """
        logger.info("=" * 60)
        logger.info("Starting ConsciousnessX Integration Tests")
        logger.info("=" * 60)

        # Test 1: Orch-OR Theory Integration
        self.test_orch_or_integration()

        # Test 2: Quantum Hardware Integration
        self.test_quantum_hardware_integration()

        # Test 3: Cortical Column Simulation
        self.test_cortical_column_integration()

        # Test 4: Consciousness RL Integration
        self.test_consciousness_rl_integration()

        # Test 5: Hybrid Architecture Integration
        self.test_hybrid_architecture_integration()

        # Test 6: End-to-End Simulation
        self.test_end_to_end_simulation()

        # Generate summary
        summary = self._generate_summary()

        logger.info("=" * 60)
        logger.info("Integration Tests Complete")
        logger.info(f"Passed: {summary['passed']}/{summary['total']}")
        logger.info(f"Failed: {summary['failed']}/{summary['total']}")
        logger.info("=" * 60)

        return summary

    def test_orch_or_integration(self) -> None:
        """Test Orch-OR theory integration."""
        logger.info("\n[Test 1] Orch-OR Theory Integration")

        try:
            # Initialize Orch-OR theory
            orch_or = OrchORTheory()

            # Compute reduction
            gamma = orch_or.compute_gamma(1.0)
            reduction_time = orch_or.compute_reduction_time(1.0)

            # Verify results
            assert gamma > 0, "Gamma should be positive"
            assert reduction_time > 0, "Reduction time should be positive"

            # Test collapse
            state = np.array([1.0, 0.0], dtype=complex)
            collapsed = orch_or.collapse_wavefunction(state)

            assert (
                np.abs(np.linalg.norm(collapsed) - 1.0) < 1e-6
            ), "Collapsed state should be normalized"

            self._record_result("Orch-OR Integration", True, "All tests passed")

        except Exception as e:
            logger.error(f"Orch-OR integration test failed: {e}")
            self._record_result("Orch-OR Integration", False, str(e))

    def test_quantum_hardware_integration(self) -> None:
        """Test quantum hardware integration."""
        logger.info("\n[Test 2] Quantum Hardware Integration")

        try:
            # Initialize virtual quantum processor
            processor = VirtualQuantumProcessor()

            # Create circuit
            processor.create_circuit(2)

            # Apply gates
            processor.apply_hadamard(0)
            processor.apply_cnot(0, 1)

            # Measure
            result = processor.measure_all()

            # Verify result
            assert len(result) == 2, "Should measure 2 qubits"

            # Get circuit depth
            depth = processor.get_circuit_depth()
            assert depth > 0, "Circuit should have depth > 0"

            # Reset
            processor.reset()

            self._record_result(
                "Quantum Hardware Integration", True, "Circuit operations successful"
            )

        except Exception as e:
            logger.error(f"Quantum hardware integration test failed: {e}")
            self._record_result("Quantum Hardware Integration", False, str(e))

    def test_cortical_column_integration(self) -> None:
        """Test cortical column simulation integration."""
        logger.info("\n[Test 3] Cortical Column Integration")

        try:
            # Initialize cortical column
            column = CorticalColumn(0)

            # Receive thalamic input
            input_data = np.random.rand(50)
            column.receive_thalamic_input(input_data)

            # Step simulation
            activations = column.step()

            # Verify results
            assert (
                len(activations) == column.num_neurons
            ), "Activations length should match number of neurons"

            # Get Phi value
            phi = column.get_phi()
            assert phi >= 0 and phi <= 1, "Phi should be in [0, 1]"

            # Get minicolumn activations
            mc_activations = column.get_minicolumn_activations()
            assert len(mc_activations) == len(
                column.minicolumns
            ), "Should have one activation per minicolumn"

            # Reset
            column.reset()

            self._record_result("Cortical Column Integration", True, f"Phi={phi:.3f}")

        except Exception as e:
            logger.error(f"Cortical column integration test failed: {e}")
            self._record_result("Cortical Column Integration", False, str(e))

    def test_consciousness_rl_integration(self) -> None:
        """Test consciousness RL integration."""
        logger.info("\n[Test 4] Consciousness RL Integration")

        try:
            # Initialize self-evolving consciousness
            from src.models.consciousness_rl.self_evolving_consciousness import ConsciousnessConfig

            config = ConsciousnessConfig(input_size=50, hidden_sizes=[50, 25])
            agent = SelfEvolvingConsciousness(config)

            # Test action selection
            observation = np.random.rand(50)
            action = agent.act(observation)

            assert 0 <= action < config.output_size, "Action should be valid"

            # Test learning
            reward = 1.0
            next_observation = np.random.rand(50)
            loss = agent.learn(reward, next_observation)

            assert loss >= 0, "Loss should be non-negative"

            # Test evolution
            change = agent.evolve()

            # Calculate Phi
            phi = agent.calculate_phi()
            assert phi >= 0 and phi <= 1, "Phi should be in [0, 1]"

            # Get architecture summary
            summary = agent.get_architecture_summary()
            assert "total_neurons" in summary, "Summary should include total_neurons"

            self._record_result("Consciousness RL Integration", True, f"Phi={phi:.3f}")

        except Exception as e:
            logger.error(f"Consciousness RL integration test failed: {e}")
            self._record_result("Consciousness RL Integration", False, str(e))

    def test_hybrid_architecture_integration(self) -> None:
        """Test hybrid architecture integration."""
        logger.info("\n[Test 5] Hybrid Architecture Integration")

        try:
            # Initialize quantum-biological bridge
            from src.models.hybrid_architectures.quantum_bio_bridge import QuantumBioConfig

            config = QuantumBioConfig(quantum_dim=4, classical_dim=20, biological_dim=10)
            bridge = QuantumBioBridge(config)

            # Test quantum to classical transformation
            quantum_state = np.array([0.7071, 0.7071, 0.0, 0.0], dtype=complex)
            signal = bridge.quantum_to_classical(quantum_state)

            assert (
                signal.data.shape[0] == config.classical_dim
            ), "Classical data dimension should match"

            # Test classical to quantum transformation
            classical_data = np.random.rand(config.classical_dim)
            signal = bridge.classical_to_quantum(classical_data)

            assert len(signal.data) == config.quantum_dim * 2, "Quantum data dimension should match"

            # Test entanglement
            success = bridge.entangle("source1", "dest1")
            coherence = bridge.get_coherence("source1", "dest1")

            # Test biological transformation
            biological_data = np.random.rand(config.biological_dim)
            signal = bridge.classical_to_biological(biological_data)

            assert (
                signal.data.shape[0] == config.biological_dim
            ), "Biological data dimension should match"

            # Get statistics
            stats = bridge.get_signal_statistics()
            assert "total_signals" in stats, "Statistics should include total_signals"

            self._record_result(
                "Hybrid Architecture Integration", True, "Signal transformations successful"
            )

        except Exception as e:
            logger.error(f"Hybrid architecture integration test failed: {e}")
            self._record_result("Hybrid Architecture Integration", False, str(e))

    def test_end_to_end_simulation(self) -> None:
        """Test end-to-end simulation."""
        logger.info("\n[Test 6] End-to-End Simulation")

        try:
            # Initialize all components
            orch_or = OrchORTheory()
            processor = VirtualQuantumProcessor()
            column = CorticalColumn(0)

            from src.models.consciousness_rl.self_evolving_consciousness import ConsciousnessConfig

            config = ConsciousnessConfig(input_size=50, hidden_sizes=[50, 25])
            agent = SelfEvolvingConsciousness(config)

            from src.models.hybrid_architectures.quantum_bio_bridge import QuantumBioConfig

            bridge_config = QuantumBioConfig(quantum_dim=4, classical_dim=20, biological_dim=10)
            bridge = QuantumBioBridge(bridge_config)

            # Run simulation steps
            results = []

            for step in range(10):
                # 1. Create quantum state
                processor.create_circuit(2)
                processor.apply_hadamard(0)
                state = processor.get_state()

                # 2. Bridge to classical
                classical_signal = bridge.quantum_to_classical(state)

                # 3. Process through cortical column
                input_data = (
                    classical_signal.data[:50]
                    if len(classical_signal.data) >= 50
                    else np.random.rand(50)
                )
                column.receive_thalamic_input(input_data)
                activations = column.step()
                phi = column.get_phi()

                # 4. RL agent action
                action = agent.act(activations[:50])
                agent.learn(1.0, activations[:50])
                agent_phi = agent.calculate_phi()

                # 5. Store results
                results.append(
                    {"step": step, "phi_column": phi, "phi_agent": agent_phi, "action": action}
                )

                logger.info(
                    f"Step {step}: Phi_column={phi:.3f}, Phi_agent={agent_phi:.3f}, Action={action}"
                )

            # Verify results
            assert len(results) == 10, "Should have 10 simulation steps"
            assert all("phi_column" in r for r in results), "All results should have phi_column"

            # Check Phi evolution
            phi_values = [r["phi_column"] for r in results]
            logger.info(f"Phi range: {min(phi_values):.3f} - {max(phi_values):.3f}")

            self._record_result("End-to-End Simulation", True, f"Completed {len(results)} steps")

        except Exception as e:
            logger.error(f"End-to-end simulation test failed: {e}")
            self._record_result("End-to-End Simulation", False, str(e))

    def _record_result(self, test_name: str, passed: bool, message: str) -> None:
        """Record test result.

        Args:
            test_name: Name of the test.
            passed: Whether test passed.
            message: Test result message.
        """
        self.test_results.append({"name": test_name, "passed": passed, "message": message})

        status = "PASSED" if passed else "FAILED"
        logger.info(f"[{status}] {test_name}: {message}")

    def _generate_summary(self) -> Dict[str, any]:
        """Generate test summary.

        Returns:
            Dictionary with test summary.
        """
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["passed"])
        failed = total - passed

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0,
            "results": self.test_results,
        }


def main():
    """Run integration tests."""
    tester = IntegrationTest()
    summary = tester.run_all_tests()

    # Exit with appropriate code
    if summary["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

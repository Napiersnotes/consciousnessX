"""
Demo: Quantum-Neural Hybrid Architecture

Demonstrates integration of quantum hardware simulation with neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.hardware.quantum_hardware.virtual_quantum_processor import VirtualQuantumProcessor
from src.models.spiking_neural_networks.quantum_lif_neuron import QuantumLIFNeuron
from src.models.hybrid_architectures.quantum_bio_bridge import QuantumBioBridge
from src.utils.logging_utils import setup_logger


def main():
    """Run quantum-neural hybrid demonstration."""
    
    # Setup logging
    logger = setup_logger('quantum_neural_demo')
    logger.info("Starting Quantum-Neural Hybrid Demo")
    
    # Initialize quantum processor
    logger.info("Initializing quantum processor...")
    quantum_proc = VirtualQuantumProcessor(
        num_qubits=8,
        gate_fidelity=0.99,
        noise_level=0.01
    )
    
    # Create entangled Bell state
    logger.info("Creating Bell state...")
    state = quantum_proc.create_bell_state(0, 1)
    logger.info(f"Bell state prepared: {state[:4]}")
    
    # Apply noise simulation
    logger.info("Applying quantum noise...")
    noisy_state = quantum_proc.apply_noise(state)
    logger.info(f"Noisy state: {noisy_state[:4]}")
    
    # Initialize quantum-enhanced LIF neuron
    logger.info("Initializing quantum LIF neuron...")
    neuron = QuantumLIFNeuron(
        tau_m=20.0,
        threshold=1.0,
        collapse_threshold=0.8
    )
    
    # Create quantum-bio bridge
    logger.info("Creating quantum-biological bridge...")
    bridge = QuantumBioBridge(
        quantum_dim=8,
        neural_dim=100
    )
    
    # Simulate integration
    logger.info("Simulating quantum-neural integration...")
    time_steps = 1000
    membrane_potentials = []
    collapse_events = []
    phi_values = []
    
    for t in range(time_steps):
        # Get input from quantum state
        quantum_state = noisy_state if t < time_steps // 2 else state
        
        # Convert to neural input
        neural_input = bridge.quantum_to_neural(quantum_state)
        
        # Process through neuron
        spike, membrane_pot, collapsed = neuron.step(neural_input[:neuron.num_neurons])
        
        # Track metrics
        membrane_potentials.append(membrane_pot)
        if collapsed:
            collapse_events.append(t)
        
        # Calculate integrated information (phi)
        phi = bridge.calculate_phi(neural_input)
        phi_values.append(phi)
        
        if t % 100 == 0:
            logger.info(f"Step {t}: Membrane Potential = {membrane_pot:.4f}, Phi = {phi:.4f}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot membrane potential
    axes[0].plot(membrane_potentials)
    axes[0].set_ylabel('Membrane Potential')
    axes[0].set_title('Quantum-Enhanced LIF Neuron Activity')
    axes[0].grid(True, alpha=0.3)
    
    # Plot collapse events
    if collapse_events:
        axes[0].scatter(collapse_events, [membrane_potentials[i] for i in collapse_events],
                       color='red', s=50, marker='x', label='Quantum Collapse', zorder=5)
        axes[0].legend()
    
    # Plot phi dynamics
    axes[1].plot(phi_values, color='purple')
    axes[1].axhline(1.0, color='red', linestyle='--', label='Consciousness Threshold')
    axes[1].set_ylabel('Phi (Integrated Information)')
    axes[1].set_title('Integrated Information Dynamics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot quantum state evolution
    from src.visualization.quantum_visualizer import QuantumVisualizer
    qvis = QuantumVisualizer()
    qvis.plot_quantum_state(state, title="Initial Quantum State")
    
    plt.tight_layout()
    plt.savefig('quantum_neural_demo.png', dpi=300, bbox_inches='tight')
    logger.info("Visualization saved to quantum_neural_demo.png")
    
    # Summary statistics
    logger.info("\n=== Summary Statistics ===")
    logger.info(f"Total collapse events: {len(collapse_events)}")
    logger.info(f"Mean membrane potential: {np.mean(membrane_potentials):.4f}")
    logger.info(f"Mean phi value: {np.mean(phi_values):.4f}")
    logger.info(f"Max phi value: {np.max(phi_values):.4f}")
    logger.info(f"Time above threshold: {sum(1 for p in phi_values if p > 1.0)} steps")
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()
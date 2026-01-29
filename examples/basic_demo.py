#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Demo for consciousnessX
Example usage of the microtubule quantum consciousness simulation
"""

import sys
import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.microtubule_simulator import MicrotubuleSimulator, MicrotubuleConfig
from core.penrose_gravitational_collapse import GravitationalCollapseCalculator, CollapseModel
from virtual_bio.virtual_neuronal_culture import VirtualNeuronalCulture

def demo_microtubule_simulation():
    """Demo microtubule quantum simulation"""
    print("\n" + "="*60)
    print("MICROTUBULE QUANTUM SIMULATION DEMO")
    print("="*60)
    
    # Create configuration
    config = MicrotubuleConfig(
        num_protofilaments=13,
        num_tubulins_per_filament=50,  # Smaller for demo
        simulation_duration_s=0.05,     # 50 ms
        time_step_s=1e-4,              # 0.1 ms
        use_gpu=False                   # Disable GPU for demo
    )
    
    print(f"Configuration:")
    print(f"  Protofilaments: {config.num_protofilaments}")
    print(f"  Tubulins per filament: {config.num_tubulins_per_filament}")
    print(f"  Total tubulins: {config.num_protofilaments * config.num_tubulins_per_filament}")
    print(f"  Simulation time: {config.simulation_duration_s:.3f} s")
    print(f"  Time step: {config.time_step_s:.1e} s")
    print(f"  Device: {config.device}")
    
    # Create and run simulator
    print(f"\nInitializing simulator...")
    simulator = MicrotubuleSimulator(config)
    
    print(f"Running simulation...")
    start_time = time.time()
    results = simulator.simulate()
    elapsed = time.time() - start_time
    
    print(f"Simulation completed in {elapsed:.2f} seconds")
    print(f"Real-time factor: {results['summary']['total_time'] / elapsed:.2f}")
    
    # Display results
    summary = results['summary']
    print(f"\nResults:")
    print(f"  Final coherence: {summary['final_coherence']:.3f}")
    print(f"  Integrated Information (Φ): {summary['integrated_information']:.4f}")
    print(f"  Consciousness level: {summary['consciousness_level']}")
    
    collapse_stats = summary['collapse_statistics']
    print(f"  Total collapses: {collapse_stats['total_collapses']}")
    print(f"  Collapse rate: {collapse_stats['collapse_rate']:.1f} Hz")
    print(f"  Regularity index: {collapse_stats['regularity_index']:.3f}")
    
    return results, simulator

def demo_penrose_calculations():
    """Demo Penrose gravitational collapse calculations"""
    print("\n" + "="*60)
    print("PENROSE COLLAPSE CALCULATIONS DEMO")
    print("="*60)
    
    calculator = GravitationalCollapseCalculator()
    
    # Test different masses
    masses = [110e-24, 550e-24, 1100e-24]  # 1x, 5x, 10x tubulin mass
    separations = [1.0e-9, 2.4e-9, 5.0e-9]  # Different separations
    
    print(f"Testing Penrose collapse times:")
    print(f"\nVarying mass (separation = 2.4 nm):")
    for mass in masses:
        result = calculator.calculate_collapse_time(
            model=CollapseModel.PENROSE_SIMPLE,
            mass=mass,
            separation=2.4e-9
        )
        mass_ratio = mass / 110e-24
        print(f"  Mass: {mass_ratio:.1f}x tubulin = {result['collapse_time']:.3e} s")
    
    print(f"\nVarying separation (mass = tubulin):")
    for sep in separations:
        result = calculator.calculate_collapse_time(
            model=CollapseModel.PENROSE_SIMPLE,
            mass=110e-24,
            separation=sep
        )
        print(f"  Separation: {sep*1e9:.1f} nm = {result['collapse_time']:.3e} s")
    
    # Compare all models
    print(f"\nComparing all collapse models:")
    comparison = calculator.compare_models()
    
    for model_name, result in comparison.items():
        if model_name.startswith('_'):
            continue
        if 'collapse_time' in result:
            time = result['collapse_time']
            print(f"  {model_name:30s}: {time:.3e} s")
    
    # Calculate probabilities
    print(f"\nCollapse probabilities for 1 ms intervals:")
    intervals = [1e-4, 1e-3, 1e-2, 1e-1]  # 0.1 ms to 100 ms
    
    for interval in intervals:
        prob_result = calculator.calculate_collapse_probability(
            time_interval=interval,
            model=CollapseModel.PENROSE_SIMPLE
        )
        print(f"  {interval*1e3:6.1f} ms: P = {prob_result['probability']:.6f}")
    
    return calculator

def demo_visualization(results):
    """Create visualization of simulation results"""
    print("\n" + "="*60)
    print("VISUALIZATION DEMO")
    print("="*60)
    
    # Extract data
    time_array = np.array(results['simulation_time'])
    metrics = results['metrics']
    
    coherence_history = [m['coherence_fraction'] for m in metrics]
    order_history = [m['order_parameter'] for m in metrics]
    energy_history = [m['energy_metric'] for m in metrics]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot coherence
    axes[0, 0].plot(time_array * 1e3, coherence_history, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Coherence Fraction')
    axes[0, 0].set_title('Quantum Coherence Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot order parameter
    axes[0, 1].plot(time_array * 1e3, order_history, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Order Parameter')
    axes[0, 1].set_title('System Order (Magnetization-like)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot energy
    axes[1, 0].plot(time_array * 1e3, energy_history, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Energy Metric')
    axes[1, 0].set_title('System Energy Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot state distribution at final time
    final_state = results['final_state']
    state_labels = ['Superposition', 'Collapsed Up', 'Collapsed Down']
    state_counts = [
        np.sum(final_state == 0),
        np.sum(final_state == 1),
        np.sum(final_state == 2)
    ]
    
    colors = ['blue', 'green', 'red']
    axes[1, 1].bar(state_labels, state_counts, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Final State Distribution')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to 'demo_results.png'")
    
    plt.show()

def demo_neuronal_culture():
    """Demo virtual neuronal culture"""
    print("\n" + "="*60)
    print("VIRTUAL NEURONAL CULTURE DEMO")
    print("="*60)
    
    # Create culture
    culture = VirtualNeuronalCulture(
        num_neurons=100,
        num_electrodes=16,
        culture_type="hippocampal"
    )
    
    print(f"Created {culture.culture_type} culture with {culture.num_neurons} neurons")
    print(f"Electrode array: {culture.num_electrodes} electrodes")
    
    # Run simulation
    print(f"\nRunning simulation...")
    results = culture.run_simulation(
        duration_ms=1000.0,
        stimulation_protocol="theta_rhythm"
    )
    
    analysis = results['analysis']
    print(f"\nResults:")
    print(f"  Total spikes: {analysis['total_spikes']}")
    print(f"  Mean firing rate: {analysis['mean_firing_rate']:.2f} Hz")
    print(f"  Network bursts: {analysis['burst_analysis']['num_bursts']}")
    
    if 'synchrony' in analysis:
        print(f"  Synchrony index: {analysis['synchrony']['mean_synchrony']:.3f}")
    
    return results

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='consciousnessX Demo')
    parser.add_argument('--demo', type=str, default='all',
                       choices=['microtubule', 'penrose', 'neuron', 'all'],
                       help='Which demo to run')
    parser.add_argument('--duration', type=float, default=0.05,
                       help='Simulation duration in seconds')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CONSCIOUSNESSX DEMONSTRATION")
    print("Quantum-Biological AGI Framework")
    print("="*70)
    
    results = {}
    
    try:
        if args.demo in ['microtubule', 'all']:
            micro_results, simulator = demo_microtubule_simulation()
            results['microtubule'] = micro_results
            
            if args.save:
                simulator.save_results('microtubule_demo_results.pkl.gz')
                print(f"\nResults saved to 'microtubule_demo_results.pkl.gz'")
        
        if args.demo in ['penrose', 'all']:
            calculator = demo_penrose_calculations()
            results['penrose'] = calculator
        
        if args.demo in ['neuron', 'all']:
            neuron_results = demo_neuronal_culture()
            results['neuron'] = neuron_results
        
        # Visualization for microtubule results
        if 'microtubule' in results and args.demo != 'penrose':
            demo_visualization(results['microtubule'])
        
        print("\n" + "="*70)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except Exception as e:
        print(f"\nERROR: Demo failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

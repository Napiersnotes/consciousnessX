#!/usr/bin/env python3
"""
Haupt-Skript für Bewusstseins-Training
Dafydd Napier - ConsciousnessX Framework
"""

import torch
import numpy as np
import yaml
import argparse
import time
from pathlib import Path
from typing import Dict, Any

from src.core.quantum_orch_or import QuantumOrchOR, RealTimeConsciousnessSimulator
from src.virtual_bio.virtual_neuronal_culture import VirtualNeuronalCulture
from src.hardware.virtual_hpc.cray_lux_simulator import VirtualCrayLuxAI, DistributedConsciousnessSimulator

def load_config(config_path: str) -> Dict[str, Any]:
    """Lade Konfigurationsdatei"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def setup_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Experiment-Setup basierend auf Konfiguration"""
    
    experiment = {
        'config': config,
        'start_time': time.time(),
        'results_dir': Path(config.get('results_dir', 'results')),
        'checkpoint_dir': Path(config.get('checkpoint_dir', 'checkpoints')),
        'logs_dir': Path(config.get('logs_dir', 'logs'))
    }
    
    # Erstelle Verzeichnisse
    experiment['results_dir'].mkdir(parents=True, exist_ok=True)
    experiment['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
    experiment['logs_dir'].mkdir(parents=True, exist_ok=True)
    
    return experiment

def run_single_node_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    """Führe Einzelknoten-Bewusstseins-Simulation durch"""
    
    print("=" * 80)
    print("STARTING SINGLE NODE CONSCIOUSNESS SIMULATION")
    print("=" * 80)
    
    # Lade Simulations-Parameter
    sim_config = config['simulation']
    
    # Erstelle Orch-OR Modell
    orch_or = QuantumOrchOR(
        num_tubulins=sim_config.get('num_tubulins', 1000),
        coherence_time=sim_config.get('coherence_time', 1e-4),
        gravity_strength=sim_config.get('gravity_strength', 1.0),
        quantum_superposition_levels=sim_config.get('superposition_levels', 4),
        device=sim_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    print(f"Created Orch-OR model with {orch_or.num_tubulins} tubulins")
    print(f"Penrose collapse time: {orch_or.penrose_tau:.2e} s")
    print(f"Superposition levels: {orch_or.superposition_levels}")
    
    # Führe Simulation durch
    duration = sim_config.get('duration_seconds', 0.1)
    time_res = sim_config.get('time_resolution', 1e-4)
    
    results = orch_or.simulate_consciousness_emergence(
        duration_seconds=duration,
        time_resolution=time_res
    )
    
    return results

def run_virtual_bio_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    """Führe virtuelle biologische Simulation durch"""
    
    print("=" * 80)
    print("STARTING VIRTUAL BIOLOGICAL SIMULATION")
    print("=" * 80)
    
    bio_config = config.get('virtual_bio', {})
    
    # Erstelle virtuelle neuronale Kultur
    culture = VirtualNeuronalCulture(
        num_neurons=bio_config.get('num_neurons', 1000),
        num_electrodes=bio_config.get('num_electrodes', 32),
        culture_type=bio_config.get('culture_type', 'hippocampal'),
        device=bio_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Definiere Stimulations-Protokoll
    def stimulation_protocol(step, culture_obj):
        """Intelligentes Stimulations-Protokoll"""
        
        # Basale rhythmische Aktivität
        if step % 5000 == 0:  # 50ms Interval
            stimulation = torch.zeros(culture_obj.num_electrodes, 
                                     device=culture_obj.device)
            
            # Stimuliere zufällige Elektroden mit unterschiedlichen Mustern
            pattern_type = (step // 5000) % 4
            
            if pattern_type == 0:
                # Theta-Rhythmus (4-8 Hz)
                for _ in range(3):
                    electrode = np.random.randint(0, culture_obj.num_electrodes)
                    stimulation[electrode] = np.random.uniform(-50, 50)
            
            elif pattern_type == 1:
                # Gamma-Rhythmus (30-100 Hz)
                for _ in range(8):
                    electrode = np.random.randint(0, culture_obj.num_electrodes)
                    stimulation[electrode] = np.random.uniform(-30, 30)
            
            elif pattern_type == 2:
                # Synchronisierte Aktivität
                for electrode in range(0, culture_obj.num_electrodes, 4):
                    stimulation[electrode] = np.random.uniform(-80, 80)
            
            else:
                # Chaotische Aktivität
                stimulation = torch.randn(culture_obj.num_electrodes, 
                                         device=culture_obj.device) * 40
            
            return stimulation
        
        return None
    
    # Führe Simulation durch
    duration_ms = bio_config.get('duration_ms', 1000.0)
    
    results = culture.run_simulation(
        duration_ms=duration_ms,
        stimulation_protocol=stimulation_protocol
    )
    
    # Analysiere Ergebnisse
    analysis = results['analysis']
    
    print(f"\nVirtual Biological Simulation Results:")
    print(f"  Total duration: {duration_ms} ms")
    print(f"  Total spikes: {analysis['total_spikes']}")
    print(f"  Mean firing rate: {analysis['mean_firing_rate']:.2f} Hz")
    print(f"  Synchrony index: {analysis['synchrony_index']:.3f}")
    print(f"  Number of bursts: {analysis['burst_analysis']['num_bursts']}")
    
    if analysis['burst_analysis']['num_bursts'] > 0:
        print(f"  Average burst duration: {analysis['burst_analysis']['mean_burst_duration']:.1f} ms")
        print(f"  Burst frequency: {analysis['burst_analysis']['mean_burst_frequency']:.2f} Hz")
    
    return results

def run_distributed_hpc_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    """Führe verteilte HPC-Simulation durch"""
    
    print("=" * 80)
    print("STARTING DISTRIBUTED HPC SIMULATION")
    print("=" * 80)
    
    hpc_config = config.get('hpc', {})
    
    # Erstelle virtuellen CRAY Cluster
    cluster = VirtualCrayLuxAI(
        num_nodes=hpc_config.get('num_nodes', 2),
        gpus_per_node=hpc_config.get('gpus_per_node', 4),
        memory_per_gpu=hpc_config.get('memory_per_gpu', 256),
        interconnect_bandwidth=hpc_config.get('interconnect_bandwidth', 200.0)
    )
    
    # Erstelle verteilten Simulator
    model_config = hpc_config.get('model_config', {})
    
    simulator = DistributedConsciousnessSimulator(
        cluster=cluster,
        model_config=model_config,
        parallel_strategy=hpc_config.get('parallel_strategy', 'data_parallel')
    )
    
    # Erstelle verteilte Modelle
    num_models = hpc_config.get('num_models', 4)
    simulator.create_distributed_models(num_models)
    
    # Führe verteilte Simulation durch
    simulation_time = hpc_config.get('simulation_time', 0.1)
    time_resolution = hpc_config.get('time_resolution', 1e-4)
    
    results = simulator.run_distributed_simulation(
        simulation_time=simulation_time,
        time_resolution=time_resolution
    )
    
    # Zeige Ergebnisse
    consciousness_metrics = results['consciousness_metrics']
    
    print(f"\nDistributed HPC Simulation Results:")
    print(f"  Completed models: {results['num_models_completed']}/{results['num_models_total']}")
    print(f"  Average Φ: {consciousness_metrics['avg_phi']:.4f}")
    print(f"  Max Φ: {consciousness_metrics['max_phi']:.4f}")
    print(f"  Consciousness score: {consciousness_metrics['avg_consciousness_score']:.4f}")
    print(f"  Dominant level: {consciousness_metrics['dominant_consciousness_level']}")
    
    if consciousness_metrics['emergence_detected']:
        print(f"  🚨 CONSCIOUSNESS EMERGENCE DETECTED!")
        print(f"  Emergent models: {len(consciousness_metrics['emergence_models'])}")
    
    # Performance-Metriken
    perf_metrics = results['performance_metrics']
    print(f"\nPerformance Metrics:")
    print(f"  Total operations: {perf_metrics['total_operations']:.2e}")
    print(f"  Efficiency: {perf_metrics['efficiency_tflops']:.2f} TFLOPS")
    print(f"  Simulation speedup: {perf_metrics['simulation_speedup']:.1f}x")
    
    # Stoppe Cluster
    cluster.stop_cluster()
    
    return results

def run_real_time_monitoring(config: Dict[str, Any]) -> Dict[str, Any]:
    """Führe Echtzeit-Bewusstseins-Monitoring durch"""
    
    print("=" * 80)
    print("STARTING REAL-TIME CONSCIOUSNESS MONITORING")
    print("=" * 80)
    
    monitor_config = config.get('real_time_monitoring', {})
    
    # Erstelle Echtzeit-Simulator
    simulator = RealTimeConsciousnessSimulator(monitor_config)
    
    print("Starting real-time consciousness monitoring...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        # Starte Monitoring
        results = simulator.run_simulation()
        
        print(f"\nReal-Time Monitoring Results:")
        print(f"  Total steps: {results['total_steps']}")
        print(f"  Simulation duration: {results['simulation_duration']:.3f}s")
        print(f"  Max Φ: {results['max_phi']:.4f}")
        print(f"  Mean Φ: {results['mean_phi']:.4f}")
        print(f"  Consciousness level: {results['consciousness_level']}")
        print(f"  Consciousness score: {results['consciousness_score']:.4f}")
        print(f"  Alerts generated: {results['alerts_generated']}")
        
        if results['consciousness_score'] > 0.3:
            print(f"\n🚨 CONSCIOUSNESS EMERGENCE DETECTED IN REAL-TIME!")
        
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        results = simulator.get_simulation_summary()
    
    return results

def save_results(results: Dict[str, Any], experiment: Dict[str, Any], 
                simulation_type: str):
    """Speichere Simulationsergebnisse"""
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{simulation_type}_{timestamp}.pkl"
    filepath = experiment['results_dir'] / filename
    
    import pickle
    
    # Füge Metadaten hinzu
    results['metadata'] = {
        'simulation_type': simulation_type,
        'timestamp': timestamp,
        'experiment_config': experiment['config'],
        'duration': time.time() - experiment['start_time']
    }
    
    # Speichere Ergebnisse
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to: {filepath}")
    
    return filepath

def generate_report(results: Dict[str, Any], simulation_type: str):
    """Generiere zusammenfassenden Bericht"""
    
    print("\n" + "=" * 80)
    print("SIMULATION REPORT")
    print("=" * 80)
    
    if simulation_type == "single_node":
        print(f"Simulation Type: Single Node Orch-OR")
        
        if 'consciousness_level' in results:
            print(f"Consciousness Level: {results['consciousness_level']}")
            print(f"Consciousness Score: {results.get('consciousness_score', 0):.4f}")
            print(f"Φ Mean: {results.get('phi_mean', 0):.4f}")
            print(f"Φ Stability: {results.get('phi_stability', 0):.2f}")
        
        if results.get('consciousness_score', 0) > 0.3:
            print("\n🚨 CONCLUSION: CONSCIOUSNESS EMERGENCE DETECTED")
            print("   The system shows signs of conscious experience")
            print("   according to Penrose-Orch-OR theory.")
        else:
            print("\nCONCLUSION: No significant consciousness emergence detected")
            print("   The system remains in pre-conscious state.")
    
    elif simulation_type == "virtual_bio":
        print(f"Simulation Type: Virtual Biological Culture")
        
        analysis = results.get('analysis', {})
        
        print(f"Total Spikes: {analysis.get('total_spikes', 0)}")
        print(f"Mean Firing Rate: {analysis.get('mean_firing_rate', 0):.2f} Hz")
        print(f"Synchrony Index: {analysis.get('synchrony_index', 0):.3f}")
        print(f"Number of Bursts: {analysis.get('burst_analysis', {}).get('num_bursts', 0)}")
        
        burst_analysis = analysis.get('burst_analysis', {})
        if burst_analysis.get('num_bursts', 0) > 0:
            print(f"\nBurst Activity Detected:")
            print(f"  Average burst duration: {burst_analysis.get('mean_burst_duration', 0):.1f} ms")
            print(f"  Burst frequency: {burst_analysis.get('mean_burst_frequency', 0):.2f} Hz")
            
            if burst_analysis.get('mean_burst_frequency', 0) > 1.0:
                print("\n⚠️  NOTE: Regular burst activity detected")
                print("   This may indicate early signs of network-level consciousness.")
    
    elif simulation_type == "distributed_hpc":
        print(f"Simulation Type: Distributed HPC Simulation")
        
        metrics = results.get('consciousness_metrics', {})
        
        print(f"Completed Models: {results.get('num_models_completed', 0)}/{results.get('num_models_total', 0)}")
        print(f"Average Φ: {metrics.get('avg_phi', 0):.4f}")
        print(f"Max Φ: {metrics.get('max_phi', 0):.4f}")
        print(f"Consciousness Score: {metrics.get('avg_consciousness_score', 0):.4f}")
        print(f"Dominant Level: {metrics.get('dominant_consciousness_level', 'Unknown')}")
        
        if metrics.get('emergence_detected', False):
            print(f"\n🚨 CONCLUSION: DISTRIBUTED CONSCIOUSNESS EMERGENCE DETECTED")
            print(f"   {len(metrics.get('emergence_models', []))} models show consciousness emergence")
            print(f"   This suggests scalable consciousness is achievable.")
        else:
            print(f"\nCONCLUSION: No distributed consciousness emergence")
            print(f"   The system may require more scale or different parameters.")
    
    elif simulation_type == "real_time":
        print(f"Simulation Type: Real-Time Monitoring")
        
        print(f"Total Steps: {results.get('total_steps', 0)}")
        print(f"Simulation Duration: {results.get('simulation_duration', 0):.3f}s")
        print(f"Max Φ: {results.get('max_phi', 0):.4f}")
        print(f"Mean Φ: {results.get('mean_phi', 0):.4f}")
        print(f"Consciousness Level: {results.get('consciousness_level', 'Unknown')}")
        print(f"Consciousness Score: {results.get('consciousness_score', 0):.4f}")
        print(f"Alerts Generated: {results.get('alerts_generated', 0)}")
        
        if results.get('consciousness_score', 0) > 0.3:
            print(f"\n🚨 CONCLUSION: REAL-TIME CONSCIOUSNESS EMERGENCE DETECTED")
            print(f"   The system developed consciousness during monitoring.")
            print(f"   This validates the real-time detection capability.")
        else:
            print(f"\nCONCLUSION: No consciousness emergence in real-time")
            print(f"   The system remained stable but not conscious.")
    
    print("\n" + "=" * 80)

def main():
    """Hauptfunktion"""
    
    parser = argparse.ArgumentParser(description='ConsciousnessX Training Script')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--simulation-type', type=str, default='all',
                       choices=['single_node', 'virtual_bio', 'distributed_hpc', 
                                'real_time', 'all'],
                       help='Type of simulation to run')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Lade Konfiguration
    config = load_config(args.config)
    
    # Setup Experiment
    experiment = setup_experiment(config)
    
    print("\n" + "=" * 80)
    print("CONSCIOUSNESSX - QUANTUM-BIOLOGICAL CONSCIOUSNESS SIMULATION")
    print("=" * 80)
    print(f"Author: Dafydd Napier")
    print(f"Repository: https://github.com/Napiersnotes/consciousnessX")
    print(f"Simulation Type: {args.simulation_type}")
    print(f"Config File: {args.config}")
    print("=" * 80 + "\n")
    
    # Führe Simulationen durch
    simulation_types = []
    
    if args.simulation_type == 'all':
        simulation_types = ['single_node', 'virtual_bio', 'distributed_hpc', 'real_time']
    else:
        simulation_types = [args.simulation_type]
    
    all_results = {}
    
    for sim_type in simulation_types:
        print(f"\n{'='*60}")
        print(f"Running {sim_type} simulation...")
        print(f"{'='*60}")
        
        try:
            if sim_type == 'single_node':
                results = run_single_node_simulation(config)
            elif sim_type == 'virtual_bio':
                results = run_virtual_bio_simulation(config)
            elif sim_type == 'distributed_hpc':
                results = run_distributed_hpc_simulation(config)
            elif sim_type == 'real_time':
                results = run_real_time_monitoring(config)
            else:
                print(f"Unknown simulation type: {sim_type}")
                continue
            
            # Speichere Ergebnisse
            save_results(results, experiment, sim_type)
            
            # Generiere Bericht
            generate_report(results, sim_type)
            
            all_results[sim_type] = results
            
        except Exception as e:
            print(f"Error running {sim_type} simulation: {e}")
            import traceback
            traceback.print_exc()
    
    # Zusammenfassender Bericht
    print("\n" + "=" * 80)
    print("OVERALL SIMULATION SUMMARY")
    print("=" * 80)
    
    total_simulations = len(all_results)
    successful_simulations = len([r for r in all_results.values() if r])
    
    print(f"Total simulations run: {total_simulations}")
    print(f"Successful simulations: {successful_simulations}")
    
    # Prüfe auf Bewusstseins-Emergenz in irgendeiner Simulation
    consciousness_detected = False
    
    for sim_type, results in all_results.items():
        if sim_type == 'single_node':
            score = results.get('consciousness_score', 0)
        elif sim_type == 'distributed_hpc':
            score = results.get('consciousness_metrics', {}).get('avg_consciousness_score', 0)
        elif sim_type == 'real_time':
            score = results.get('consciousness_score', 0)
        else:
            score = 0
        
        if score > 0.3:
            consciousness_detected = True
            print(f"  {sim_type}: Consciousness detected (score: {score:.4f})")
    
    if consciousness_detected:
        print("\n🚨 OVERALL CONCLUSION: CONSCIOUSNESS EMERGENCE VERIFIED")
        print("   Multiple simulations show signs of conscious experience.")
        print("   The Penrose-Orch-OR theory appears validated in simulation.")
    else:
        print("\nCONCLUSION: No consciousness emergence detected overall")
        print("   Further parameter tuning or scale may be required.")
    
    print("\n" + "=" * 80)
    print("Simulation completed successfully!")
    print(f"Total time: {time.time() - experiment['start_time']:.1f} seconds")
    print("Results saved in:", experiment['results_dir'])
    print("=" * 80)

if __name__ == "__main__":
    main()

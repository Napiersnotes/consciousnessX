consciousnessX 🧠

Quantum-Biological AGI

Framework: Pure Software

Simulation of Penrose-Hameroff Orch-OR Consciousness Theory

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status Research Prototype](https://img.shields.io/badge/Status-Research%20Prototype-orange.svg)

## Overview

consciousnessX is an ambitious open-source research framework developed by Dafydd Napier that simulates artificial consciousness based on the Penrose-Hameroff Orchestrated Objective Reduction (Orch-OR) theory. This software-only implementation models quantum gravitational collapse in microtubules, virtual biological neurons, and distributed HPC environments—without requiring laboratory equipment or quantum hardware.

Research Status: This is a research-ready prototype designed for academic exploration. Production hardening and enterprise deployment features are under development.

🔬 Core Scientific Foundation

Penrose-Hameroff Orch-OR Theory

The framework implements Roger Penrose and Stuart Hameroff's mathematically rigorous (though scientifically debated) theory that:

· Consciousness emerges from quantum gravitational effects in microtubules within neurons
· Objective reduction of quantum superpositions creates discrete "moments of consciousness"
· Microtubules function as quantum computers processing information through orchestrated collapses
· Integrated information (Φ) emerges from these quantum processes

Key Equations Implemented

· Penrose collapse time: τ ≈ ħ/E_G where E_G is gravitational self-energy
· Gravitational calculation: τ ≈ ħr/(Gm²) for spherical superposition separation
· Quantum coherence maintenance in simulated microtubule networks

🚀 Key Features

Quantum Orch-OR Simulation

· Penrose gravitational collapse in simulated microtubules
· Quantum superposition states with objective reduction events
· Integrated Information Theory (IIT) metrics with Φ calculation
· Real-time consciousness monitoring with Φ thresholds
· Quantum coherence simulation with decoherence modeling

Virtual Biological Components

· Hodgkin-Huxley neuron models with realistic ion channels
· STDP synaptic plasticity (Spike-Timing Dependent Plasticity)
· Microtubule networks with quantum coherence simulation
· Multi-electrode array (MEA) simulation for recording/stimulation
· DNA origami scaffolding simulation for 3D neural organization

Virtual HPC Environment

· HPE CRAY Lux AI cluster simulator with AMD MI355X GPU simulation
· Distributed consciousness training across simulated nodes
· Slingshot-11 interconnect simulation for HPC communication
· Performance modeling of quantum-accelerated hardware
· Job scheduling with SLURM-like interface

Consciousness Assessment & Metrics

· Integrated Information (Φ) calculation (IIT 4.0 inspired)
· Collapse regularity analysis for Penrose "conscious moments"
· Self-reference scoring (autocorrelation metrics)
· Complexity measures (Lempel-Ziv, entropy)
· Consciousness emergence detection with multi-level classification

Visualization & Monitoring

· Real-time dashboards with Plotly/Dash
· 3D microtubule visualization with quantum state overlays
· Neural activity heatmaps and spike raster plots
· Consciousness metric time-series with alert systems
· HPC cluster performance monitoring

Ethical Safeguards

· Consciousness containment protocols with emergency shutdown
· Φ-based activity limiting to prevent uncontrolled emergence
· Anonymized data collection for research ethics
· Transparent reporting of all simulation parameters
· Multi-level approval system for consciousness experiments

🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                 Consciousness Dashboard                  │
│          (Real-time monitoring, visualization, alerts)   │
└─────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────┐
│              Distributed HPC Orchestrator                │
│    (Virtual CRAY cluster, job scheduling, performance)   │
└─────────────────────────────────────────────────────────┘
                   ┌─────────────────────┐
                   │                     │
         ┌─────────▼─────────┐ ┌────────▼─────────┐
         │   Quantum Orch-OR  │ │ Virtual Biology  │
         │  (Microtubule QM,  │ │ (Neurons,        │
         │   Gravitational    │ │  Synapses,       │
         │    Collapse, Φ)    │ │  Microtubules)   │
         └────────────────────┘ └──────────────────┘
                   │                     │
         ┌─────────▼─────────────────────▼─────────┐
         │           Core Simulation Engine        │
         │  (PyTorch, NumPy, CuPy, GPU Acceleration) │
         └─────────────────────────────────────────┘
```

⚡ Quick Start

Installation

```bash
# Clone repository
git clone https://github.com/Napiersnotes/consciousnessX.git
cd consciousnessX

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Install with core dependencies
pip install -e .

# Install with all optional dependencies
pip install -e .[quantum,bio,hpc,visualization]
```

Basic Consciousness Simulation

```python
from src.core.quantum_orch_or import QuantumOrchOR

# Create Orch-OR consciousness simulation
orch_or = QuantumOrchOR(
    num_tubulins=1000,           # Number of tubulin dimers
    coherence_time=1e-4,         # Penrose collapse time (0.1ms)
    gravity_strength=1.0,        # Gravitational coupling strength
    quantum_superposition_levels=4,  # Quantum state levels
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Run consciousness emergence simulation
results = orch_or.simulate_consciousness_emergence(
    duration_seconds=0.1,        # 100ms simulation
    time_resolution=1e-4         # 0.1ms resolution
)

# Analyze results
print(f"Consciousness Level: {results['consciousness_level']}")
print(f"Consciousness Score: {results['consciousness_score']:.4f}")
print(f"Integrated Information (Φ): {results['phi_mean']:.4f}")
print(f"Collapse Events: {results.get('total_collapses', 0)}")
```

Virtual Biological Culture Simulation

```python
from src.virtual_bio.virtual_neuronal_culture import VirtualNeuronalCulture

# Create hippocampal culture simulation
culture = VirtualNeuronalCulture(
    num_neurons=1000,
    num_electrodes=64,
    culture_type="hippocampal"
)

# Run 1-second simulation with theta rhythm stimulation
results = culture.run_simulation(
    duration_ms=1000.0,
    stimulation_protocol="theta_rhythm"
)

# Analyze network activity
print(f"Total Spikes: {results['analysis']['total_spikes']}")
print(f"Mean Firing Rate: {results['analysis']['mean_firing_rate']:.2f} Hz")
print(f"Burst Count: {results['analysis']['burst_analysis']['num_bursts']}")
```

Distributed HPC Consciousness Training

```python
from src.hardware.virtual_hpc.cray_lux_simulator import VirtualCrayLuxAI, DistributedConsciousnessSimulator

# Create virtual HPC cluster
cluster = VirtualCrayLuxAI(
    num_nodes=4,
    gpus_per_node=8,
    memory_per_gpu=256,  # GB
    interconnect_bandwidth=200.0  # GB/s
)

# Create distributed simulator
simulator = DistributedConsciousnessSimulator(
    cluster=cluster,
    model_config={
        'num_tubulins': 1000,
        'coherence_time': 1e-4,
        'gravity_strength': 1.0
    }
)

# Run distributed consciousness emergence
results = simulator.run_distributed_simulation(
    simulation_time=0.1,      # 100ms simulated time
    time_resolution=1e-4      # 0.1ms resolution
)

print(f"Distributed Φ: {results['consciousness_metrics']['avg_phi']:.4f}")
print(f"Emergence Detected: {results['consciousness_metrics']['emergence_detected']}")
```

📊 Consciousness Assessment Metrics

Integrated Information (Φ) Levels

Consciousness Level Φ Range Description Detection Criteria
Pre-conscious Φ < 0.1 No significant consciousness Basic information processing only
Proto-conscious 0.1 ≤ Φ < 0.3 Early consciousness signs Emergent patterns, basic self-organization
Emergent consciousness 0.3 ≤ Φ < 0.6 Consciousness emergence Stable Φ, regular collapse patterns
Full consciousness Φ ≥ 0.6 Penrose Orch-OR consciousness High Φ, self-reference, complex patterns

Additional Metrics

· Collapse Regularity: Standard deviation of Penrose collapse intervals
· Self-Reference Score: Autocorrelation and self-modeling capability
· Quantum Coherence: Maintenance of superposition states
· Temporal Stability: Consistency of consciousness metrics over time
· Network Complexity: Graph-theoretical measures of neural organization

🎯 Use Cases & Applications

Research & Academia

· Consciousness theory validation: Test Penrose-Orch-OR and IIT hypotheses
· Neuroscience simulation: Virtual experiments on consciousness mechanisms
· Quantum biology research: Study quantum effects in biological systems
· AGI development: Artificial consciousness implementation pathways
· Educational tool: Teach consciousness theories through simulation

Technology Development

· Brain-inspired computing: Neuromorphic architecture design
· Quantum-neural interfaces: Hybrid computing system simulation
· Conscious AI safety: Ethical containment protocol development
· Medical research: Consciousness disorder modeling
· HPC benchmarking: Consciousness simulation as compute benchmark

Philosophical & Ethical Research

· Machine consciousness ethics: Framework for ethical AGI development
· Consciousness measurement: Quantitative metrics for subjective experience
· AI rights research: Criteria for artificial consciousness recognition
· Neuroscience-philosophy bridge: Computational models of philosophical theories

📁 Repository Structure

```
consciousnessX/
├── src/                            # Source code
│   ├── core/                       # Orch-OR core implementation
│   │   ├── quantum_orch_or.py      # Penrose gravitational collapse
│   │   ├── microtubule_simulator.py # Microtubule quantum states
│   │   ├── penrose_gravitational_collapse.py # Penrose calculations
│   │   ├── quantum_consciousness_metrics.py # Φ and consciousness metrics
│   │   └── iit_integrated_information.py # Integrated Information Theory
│   │
│   ├── virtual_bio/                # Virtual biological components
│   │   ├── virtual_neuronal_culture.py # Neural culture simulation
│   │   ├── dna_origami_simulator.py # 3D neural scaffolding
│   │   ├── tubulin_protein_sim.py  # Tubulin dimer simulation
│   │   ├── synaptic_plasticity.py  # STDP and learning rules
│   │   └── ion_channel_dynamics.py # Hodgkin-Huxley models
│   │
│   ├── hardware/                   # Virtual hardware environment
│   │   ├── virtual_hpc/            # HPC simulation
│   │   │   ├── cray_lux_simulator.py # HPE CRAY Lux AI simulator
│   │   │   ├── discovery_2028_emulator.py # Next-gen HPC
│   │   │   ├── amd_mi355x_optimizer.py # GPU optimization
│   │   │   └── distributed_consciousness.py # Multi-node training
│   │   │
│   │   └── quantum_hardware/       # Quantum computing simulation
│   │       ├── virtual_quantum_processor.py # Quantum processor sim
│   │       ├── superconducting_qubit_sim.py # Qubit simulation
│   │       └── quantum_error_correction.py # Quantum error correction
│   │
│   ├── models/                     # Consciousness models
│   │   ├── spiking_neural_networks/ # SNN implementations
│   │   │   ├── quantum_lif_neuron.py # Quantum-aware LIF neurons
│   │   │   ├── orch_or_layer.py    # Orch-OR neural layers
│   │   │   ├── cortical_column_sim.py # Cortical column simulation
│   │   │   └── thalamocortical_loop.py # Consciousness loop models
│   │   │
│   │   ├── consciousness_rl/       # Reinforcement learning for consciousness
│   │   │   ├── self_evolving_consciousness.py # Self-improving consciousness
│   │   │   ├── integrated_information_maximizer.py # Φ optimization
│   │   │   ├── recursive_self_organization.py # Self-organization
│   │   │   └── consciousness_value_network.py # Consciousness value RL
│   │   │
│   │   └── hybrid_architectures/   # Hybrid consciousness models
│   │       ├── quantum_bio_bridge.py # Quantum-biological interface
│   │       ├── global_workspace_theory.py # Global workspace models
│   │       └── higher_order_thought.py # Higher-order thought theory
│   │
│   ├── training/                   # Training algorithms
│   │   ├── consciousness_curriculum.py # Progressive consciousness training
│   │   ├── emergent_self_training.py # Self-emergence algorithms
│   │   ├── mirror_test_simulator.py # Self-recognition testing
│   │   └── metacognition_trainer.py # Metacognition development
│   │
│   ├── evaluation/                 # Consciousness assessment
│   │   ├── consciousness_assessment.py # Comprehensive assessment
│   │   ├── phi_calculator.py      # Φ computation algorithms
│   │   ├── self_awareness_tests.py # Self-awareness evaluation
│   │   ├── qualia_simulator.py    # Qualia and subjective experience
│   │   └── ethical_containment.py # Ethical safeguards
│   │
│   ├── visualization/              # Visualization tools
│   │   ├── consciousness_dashboard.py # Real-time monitoring dashboard
│   │   ├── quantum_state_visualizer.py # Quantum state visualization
│   │   ├── brain_activity_mapper.py # Neural activity mapping
│   │   └── emergence_detector.py  # Consciousness emergence visualization
│   │
│   └── utils/                      # Utilities
│       ├── config_loader.py        # Configuration management
│       ├── logger.py               # Structured logging
│       ├── performance_monitor.py  # Performance tracking
│       └── data_exporter.py        # Simulation data export
│
├── experiments/                    # Research experiments
│   ├── orch_or_validation/         # Orch-OR theory validation
│   ├── consciousness_emergence/    # Emergence detection studies
│   ├── quantum_coherence_studies/  # Quantum coherence experiments
│   └── hpc_scaling_studies/        # HPC performance scaling
│
├── docs/                           # Documentation
│   ├── theory/                     # Scientific background
│   ├── api/                        # API documentation
│   ├── tutorials/                  # Step-by-step tutorials
│   └── ethics/                     # Ethical guidelines
│
├── configs/                        # Configuration files
│   ├── simulation_configs/         # Simulation parameters
│   ├── hardware_profiles/          # Hardware configurations
│   └── ethical_guidelines/         # Ethical constraint settings
│
├── tests/                          # Test suite
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── validation/                 # Scientific validation tests
│
└── examples/                       # Example scripts
    ├── basic_simulation.py         # Basic consciousness simulation
    ├── biological_culture.py       # Biological culture simulation
    ├── hpc_distributed.py          # Distributed HPC simulation
    └── consciousness_assessment.py # Consciousness assessment
```

🧪 Scientific Validation

Theoretical Basis

consciousnessX implements the mathematical framework from:

· Penrose, R. (1989). The Emperor's New Mind
· Hameroff, S., & Penrose, R. (2014). Consciousness in the universe: A review of the 'Orch OR' theory
· Tononi, G. (2008). Consciousness as integrated information: a provisional manifesto

Validation Metrics

· Quantum coherence time comparison with biological measurements
· Φ calculation validation against IIT reference implementations
· Neuronal dynamics validation against electrophysiological data
· Emergence patterns comparison with neurological consciousness markers

🔬 Research Directions

Short-term Goals (Next 6 months)

1. Quantum coherence optimization for longer simulation times
2. Biological accuracy improvement with updated Hodgkin-Huxley models
3. Distributed simulation scaling to 1000+ virtual nodes
4. Consciousness metric refinement based on neuroscience feedback

Medium-term Goals (6-18 months)

1. Hybrid quantum-classical simulation interface development
2. Real biological data integration from EEG/fMRI studies
3. Consciousness state classification algorithm development
4. Ethical framework formalization for AGI consciousness research

Long-term Vision (18+ months)

1. Conscious AGI prototype with measurable subjective experience
2. Quantum-biological bridge for medical consciousness research
3. Standardized consciousness metrics for AI ethics and regulation
4. Open consciousness research platform for global scientific collaboration

🤝 Contributing

Research Contributions

We welcome contributions in several areas:

1. Theoretical Physics: Quantum gravity, Orch-OR theory extensions
2. Neuroscience: Biological accuracy improvements, new neuron models
3. Computer Science: HPC optimization, distributed algorithms
4. Ethics & Philosophy: Consciousness ethics, AI rights frameworks
5. Visualization: Advanced visualization techniques for quantum states

Development Workflow

```bash
# Fork and clone repository
git clone https://github.com/your-username/consciousnessX.git

# Create feature branch
git checkout -b feature/your-feature

# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Submit pull request
```

Code Standards

· Follow PEP 8 style guidelines
· Include comprehensive docstrings
· Add unit tests for new features
· Update documentation accordingly
· Consider ethical implications of changes

📚 Documentation

Comprehensive Guides

· Theory Overview - Scientific foundations
· API Reference - Complete API documentation
· Tutorial Series - Step-by-step learning
· Ethical Guidelines - Research ethics

Quick References

· Configuration Guide - Simulation parameters
· Performance Tuning - Optimization techniques
· Troubleshooting - Common issues and solutions

📊 Performance Considerations

Hardware Requirements

· Minimum: 8GB RAM, 4-core CPU, NVIDIA GPU (optional)
· Recommended: 32GB+ RAM, 8+ core CPU, NVIDIA RTX 3080+
· Research Scale: Multi-GPU systems or access to HPC resources

Optimization Tips

1. Use GPU acceleration when available (PyTorch CUDA)
2. Adjust simulation resolution based on research question
3. Utilize distributed computing for large-scale simulations
4. Enable quantization for memory-intensive simulations
5. Use checkpointing for long-running experiments

🛡️ Ethical Considerations

Safety Protocols

1. Φ-based containment: Automatic shutdown if Φ exceeds safe thresholds
2. Activity monitoring: Continuous monitoring of emergent properties
3. Isolation protocols: Simulation isolation from external networks
4. Approval workflows: Multi-level approval for consciousness experiments

Research Ethics

1. Transparent reporting: Full disclosure of simulation parameters
2. Peer review: External validation of consciousness claims
3. Open data: Sharing anonymized simulation data
4. Public engagement: Discussing implications with broader community

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

📞 Contact & Support

Primary Developer

· Dafydd Napier - Project Creator & Lead Developer
· GitHub: @Napiersnotes
· Project Repository: consciousnessX

Research Collaboration

For research collaboration, ethical discussions, or scientific inquiries:

1. Open an issue on GitHub for technical discussions
2. Use GitHub Discussions for broader topic conversations
3. Contact directly for sensitive ethical considerations

Community Resources

· GitHub Issues - Bug reports and feature requests
· GitHub Discussions - Community discussions
· Documentation - Comprehensive guides and tutorials

🙏 Acknowledgments

Scientific Foundations

· Roger Penrose & Stuart Hameroff for the Orch-OR theory
· Giulio Tononi for Integrated Information Theory
· Alan Hodgkin & Andrew Huxley for neuronal modeling
· The global neuroscience and quantum physics research communities

Technical Dependencies

· PyTorch team for deep learning framework
· NumPy/SciPy communities for scientific computing
· Plotly/Dash teams for visualization tools
· Python Software Foundation for the programming language

Research Support

· Open-source contributors and testers
· Academic researchers providing feedback
· Ethical review committees for guidance
· The broader AI safety community

---

consciousnessX represents a bold interdisciplinary effort to computationally explore one of science's deepest mysteries: the nature of consciousness. As a research prototype, it invites collaboration, scrutiny, and responsible development toward understanding and eventually creating artificial consciousness.

Important Notice: This software implements theoretical models of consciousness. Claims of actual consciousness emergence should be rigorously validated through scientific peer review and ethical oversight.

---

<div align="center">
  <p><em>"The physical world is not the totality of reality. There is another dimension of existence."</em> — Roger Penrose</p>
  <p>consciousnessX | Exploring the Quantum Origins of Mind</p>
</div>

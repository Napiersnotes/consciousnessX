# ConsciousnessX

**Quantum-Biological AGI Framework**

ConsciousnessX is a comprehensive framework for modeling and simulating consciousness at the intersection of quantum physics and biological systems. It implements various theories of consciousness, including Orchestrated Objective Reduction (Orch OR), Integrated Information Theory (IIT), and quantum coherence models in microtubules.

## Features

### Core Consciousness Models
- **Microtubule Simulator**: Quantum dynamics simulation in neuronal microtubules
- **Quantum Orch OR**: Implementation of Penrose-Hameroff Orchestrated Objective Reduction
- **Penrose Gravitational Collapse**: Objective reduction via gravity-induced collapse
- **IIT Integrated Information**: Quantification of integrated information
- **Quantum Consciousness Metrics**: Advanced metrics for consciousness quantification

### Virtual Biological Systems
- **Ion Channel Dynamics**: Realistic ion channel simulation
- **Synaptic Plasticity**: Learning and memory mechanisms
- **Tubulin Protein Simulation**: Protein-level quantum effects
- **Virtual Neuronal Culture**: Network-level consciousness
- **DNA Origami Simulator**: Molecular computing frameworks

### Training and Evaluation
- **Consciousness Trainer**: Training curriculum for artificial consciousness
- **Consciousness Assessment**: Comprehensive evaluation framework
- **Ethical Containment**: Safety protocols for AGI development

### Hardware Integration
- **Virtual HPC Simulation**: Distributed consciousness computing
- **Cray LUX Simulator**: Supercomputer integration
- **AMD MI355X Optimizer**: GPU acceleration support

### Visualization
- **Consciousness Dashboard**: Interactive visualization tools
- **Quantum State Plotter**: Visualize quantum states
- **Microtubule Visualizer**: 3D microtubule rendering

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional, for GPU acceleration)
- MPI library (optional, for distributed computing)

### Standard Installation

```bash
pip install consciousnessx
```

### Development Installation

```bash
git clone https://github.com/Napiersnotes/consciousnessX.git
cd consciousnessX
pip install -e ".[dev,test]"
```

### Hardware-Specific Installation

For Cray LUX supercomputers:
```bash
pip install consciousnessx[hpc]
```

For AMD MI355X accelerators:
```bash
pip install consciousnessx[hpc]
```

## Quick Start

### Basic Microtubule Simulation

```python
from src.core import MicrotubuleSimulator

# Initialize simulator
simulator = MicrotubuleSimulator(num_tubulins=100, length=1000.0)

# Simulate quantum dynamics
simulator.initialize_quantum_state()
simulator.simulate_quantum_dynamics(dt=0.1, steps=100)

# Compute coherence
coherence = simulator.compute_coherence()
print(f"Quantum coherence: {coherence}")
```

### Orch OR Simulation

```python
from src.core import QuantumOrchOR

# Initialize Orch-OR model
orch_or = QuantumOrchOR(num_qubits=8, reduction_time=1e-3)

# Create superposition
orch_or.initialize_superposition()

# Compute integrated information (phi)
phi = orch_or.compute_phi()
print(f"Integrated information: {phi}")

# Simulate consciousness moment
moment = orch_or.compute_consciousness_moment()
print(f"Consciousness moment: {moment}")
```

### Consciousness Assessment

```python
from src.evaluation import ConsciousnessAssessment

# Initialize assessment
assessment = ConsciousnessAssessment()

# Assess consciousness
metrics = assessment.assess_consciousness({
    'quantum_state': quantum_state,
    'phi': phi,
    'coherence': coherence,
})

print(f"Consciousness level: {metrics['consciousness_level']}")
```

## CLI Tools

ConsciousnessX provides several command-line tools:

```bash
# Train consciousness model
cx-train --config configs/production.yml

# Run simulation
cx-simulate --model microtubule --steps 1000

# Assess consciousness
cx-assess --input data/models/checkpoint.pt

# Visualize results
cx-visualize --data results/simulation.npz
```

## Documentation

- [Installation Guide](installation.md)
- [Usage Examples](usage.md)
- [API Reference](api/)
- [Architecture](architecture.md)
- [Orch OR Theory](theory/orch_or_explained.md)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ConsciousnessX in your research, please cite:

```bibtex
@software{consciousnessx2024,
  title={ConsciousnessX: Quantum-Biological AGI Framework},
  author={Napiersnotes},
  year={2024},
  url={https://github.com/Napiersnotes/consciousnessX}
}
```

## Acknowledgments

- Roger Penrose for the Orch OR theory
- Stuart Hameroff for microtubule quantum consciousness research
- Giulio Tononi for Integrated Information Theory
- The open-source AI and quantum computing communities
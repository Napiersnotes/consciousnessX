# Usage Guide

This guide provides comprehensive usage examples and best practices for ConsciousnessX.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Modules](#core-modules)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)

## Quick Start

### Basic Simulation

```python
from src.core import MicrotubuleSimulator

# Initialize
simulator = MicrotubuleSimulator(num_tubulins=100, length=1000.0)

# Run simulation
simulator.initialize_quantum_state()
simulator.simulate_quantum_dynamics(dt=0.1, steps=100)

# Analyze results
coherence = simulator.compute_coherence()
phi = simulator.compute_orch_or()

print(f"Quantum coherence: {coherence:.4f}")
print(f"Integrated information: {phi:.4f}")
```

## Core Modules

### Microtubule Simulator

#### Initialization

```python
from src.core import MicrotubuleSimulator

# Basic initialization
simulator = MicrotubuleSimulator(
    num_tubulins=100,      # Number of tubulin proteins
    length=1000.0,         # Microtubule length in nm
    temperature=300.0      # Temperature in Kelvin
)
```

#### Quantum Dynamics

```python
# Initialize quantum state
simulator.initialize_quantum_state()

# Simulate dynamics
result = simulator.simulate_quantum_dynamics(
    dt=0.1,        # Time step
    steps=100      # Number of steps
)

# Access results
quantum_states = result['quantum_states']
coherence_history = result['coherence']
```

#### Analysis

```python
# Compute coherence
coherence = simulator.compute_coherence()

# Compute Orch-OR
phi = simulator.compute_orch_or()

# Update microtubule state
simulator.update_microtubule_state(temperature=310.0)
```

### Quantum Orch OR

```python
from src.core import QuantumOrchOR

# Initialize
orch_or = QuantumOrchOR(
    num_qubits=8,          # Number of qubits
    reduction_time=1e-3    # Reduction time in seconds
)

# Create superposition
orch_or.initialize_superposition()

# Compute integrated information
phi = orch_or.compute_phi()

# Simulate orchestration
result = orch_or.simulate_orchestration(
    dt=1e-4,
    steps=100
)

# Simulate reduction
final_state = orch_or.simulate_reduction()

# Compute consciousness moment
moment = orch_or.compute_consciousness_moment()
print(f"Phi: {moment['phi']}")
print(f"Duration: {moment['duration']} s")
```

### Penrose Gravitational Collapse

```python
from src.core import PenroseGravitationalCollapse

# Initialize
collapse = PenroseGravitationalCollapse(
    mass=1e-26,    # Mass in kg
    energy=1e-10   # Energy in J
)

# Compute reduction time
t_g = collapse.compute_reduction_time()

# Simulate collapse
initial_state = np.array([0.7, 0.7j, 0.0, 0.0])
final_state = collapse.simulate_collapse(initial_state)
```

### Integrated Information Theory

```python
from src.core import IITIntegratedInformation

# Initialize
iit = IITIntegratedInformation(
    num_elements=10
)

# Compute phi
transition_matrix = np.random.rand(10, 10)
phi = iit.compute_phi(transition_matrix)

# Compute conceptual structure
concepts = iit.compute_conceptual_structure(transition_matrix)
```

## Virtual Biological Systems

### Ion Channel Dynamics

```python
from src.virtual_bio import IonChannelDynamics

# Initialize
ion_channel = IonChannelDynamics(
    num_channels=100,
    voltage_range=(-80, 40),
    dt=0.01
)

# Simulate voltage step
result = ion_channel.simulate_voltage_step(
    initial_voltage=-80.0,
    final_voltage=0.0,
    duration=10.0
)

# Simulate action potential
result = ion_channel.simulate_action_potential()

# Access results
voltage = result['voltage']
current = result['current']
```

### Synaptic Plasticity

```python
from src.virtual_bio import SynapticPlasticity

# Initialize
plasticity = SynapticPlasticity(
    num_neurons=100,
    initial_weight=0.5,
    learning_rate=0.01
)

# Apply Hebbian learning
pre_synaptic = np.random.rand(100)
post_synaptic = np.random.rand(100)
plasticity.apply_hebbian_learning(pre_synaptic, post_synaptic)

# Apply STDP
pre_times = np.sort(np.random.rand(100) * 100)
post_times = np.sort(np.random.rand(100) * 100)
plasticity.apply_stdp(pre_times, post_times)

# Simulate LTP
plasticity.simulate_long_term_potentiation(stimulus_strength=0.8, duration=100)
```

### Virtual Neuronal Culture

```python
from src.virtual_bio import VirtualNeuronalCulture

# Initialize
culture = VirtualNeuronalCulture(
    num_neurons=1000,
    connectivity=0.1
)

# Simulate network activity
activity = culture.simulate_network(dt=0.1, steps=1000)

# Analyze
firing_rate = culture.compute_firing_rate()
synchrony = culture.compute_synchrony()
```

## Training

### Consciousness Trainer

```python
from src.training import ConsciousnessTrainer, CurriculumScheduler

# Initialize trainer
trainer = ConsciousnessTrainer(
    model=your_model,
    config='configs/production.yml'
)

# Initialize curriculum
curriculum = CurriculumScheduler(
    num_stages=5,
    difficulty_function=lambda stage: 2**stage
)

# Train
trainer.train(
    curriculum=curriculum,
    epochs=100,
    save_dir='data/checkpoints'
)

# Validate
metrics = trainer.validate()
print(f"Validation loss: {metrics['loss']}")
```

### Checkpoint Management

```python
from src.training import CheckpointManager

# Initialize
checkpoint_manager = CheckpointManager(
    save_dir='data/checkpoints',
    max_to_keep=5
)

# Save checkpoint
checkpoint_manager.save(
    epoch=10,
    model=model,
    optimizer=optimizer,
    metrics={'loss': 0.5}
)

# Load checkpoint
checkpoint = checkpoint_manager.load('checkpoint_epoch_10.pt')
```

## Evaluation

### Consciousness Assessment

```python
from src.evaluation import ConsciousnessAssessment

# Initialize
assessment = ConsciousnessAssessment()

# Assess consciousness
metrics = assessment.assess_consciousness({
    'quantum_state': quantum_state,
    'phi': phi,
    'coherence': coherence,
    'neural_activity': neural_activity,
    'synaptic_strength': synaptic_strength,
})

# Access metrics
print(f"Consciousness level: {metrics['consciousness_level']}")
print(f"Quantum coherence: {metrics['quantum_coherence']}")
print(f"Integrated information: {metrics['integrated_information']}")
```

### Ethical Containment

```python
from src.evaluation import EthicalContainment

# Initialize
containment = EthicalContainment(
    threshold=0.8
)

# Check safety
is_safe, risk_level = containment.check_safety(
    consciousness_level=0.75,
    capabilities=['reasoning', 'learning'],
    behavior_history=history
)

if not is_safe:
    print(f"Risk level: {risk_level}")
    containment.trigger_shutdown()
```

## Visualization

### Consciousness Dashboard

```python
from src.visualization import ConsciousnessDashboard

# Initialize
dashboard = ConsciousnessDashboard()

# Add data
dashboard.add_metric('phi', phi_history)
dashboard.add_metric('coherence', coherence_history)

# Display
dashboard.show()

# Save
dashboard.save('dashboard.html')
```

### Quantum State Plotter

```python
from src.visualization.plotters import QuantumStatePlotter

# Initialize
plotter = QuantumStatePlotter()

# Plot state
plotter.plot_quantum_state(quantum_state)

# Plot Bloch sphere
plotter.plot_bloch_sphere(quantum_state)

# Show
plotter.show()
```

### Microtubule Visualizer

```python
from src.visualization.plotters import MicrotubuleVisualizer

# Initialize
visualizer = MicrotubuleVisualizer()

# Visualize microtubule
visualizer.visualize_microtubule(
    positions=simulator.positions,
    quantum_states=simulator.quantum_states
)

# Show
visualizer.show()
```

## Advanced Usage

### Distributed Computing

```python
from src.hardware.virtual_hpc import DistributedConsciousness

# Initialize distributed system
distributed = DistributedConsciousness(
    num_workers=8,
    backend='nccl'
)

# Run distributed simulation
results = distributed.run_simulation(
    simulator_class=MicrotubuleSimulator,
    params={'num_tubulins': 100}
)
```

### Hardware Acceleration

```python
import torch

# Use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = MicrotubuleSimulator(num_tubulins=1000).to(device)

# Use multiple GPUs
if torch.cuda.device_count() > 1:
    simulator = torch.nn.DataParallel(simulator)
```

### Custom Models

```python
from src.core import MicrotubuleSimulator

class CustomSimulator(MicrotubuleSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_parameter = kwargs.get('custom_param', 0.5)
    
    def custom_method(self):
        # Your custom implementation
        pass

# Use custom simulator
simulator = CustomSimulator(
    num_tubulins=100,
    custom_param=0.8
)
```

## Best Practices

### 1. Start Small

```python
# Good: Start with small systems
simulator = MicrotubuleSimulator(num_tubulins=50)

# Avoid: Starting too large
simulator = MicrotubuleSimulator(num_tubulins=10000)  # May be slow
```

### 2. Use Checkpoints

```python
# Save frequently
checkpoint_manager.save(epoch=epoch, model=model, ...)

# Load from checkpoint
checkpoint_manager.load('checkpoint_epoch_50.pt')
```

### 3. Monitor Resources

```python
import psutil
import torch

# Check CPU usage
print(f"CPU: {psutil.cpu_percent()}%")

# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### 4. Validate Results

```python
# Always validate
results = simulator.simulate_quantum_dynamics(dt=0.1, steps=100)

# Check for NaN
if np.any(np.isnan(results['quantum_states'])):
    print("Warning: NaN values detected")
```

### 5. Use Configuration Files

```python
import yaml

# Load config
with open('configs/production.yml') as f:
    config = yaml.safe_load(f)

# Use config
simulator = MicrotubuleSimulator(
    num_tubulins=config['model']['num_tubulins']
)
```

## Next Steps

- Explore the [API Reference](api/)
- Check out [Examples](../examples/)
- Read the [Architecture Guide](architecture.md)
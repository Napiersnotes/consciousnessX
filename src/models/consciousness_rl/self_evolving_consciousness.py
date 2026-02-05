"""Self-evolving consciousness architecture with meta-learning.

Implements a self-modifying neural architecture that autonomously
evolves its structure and parameters based on internal consciousness
metrics and environmental feedback.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ModificationType(Enum):
    """Types of architectural modifications."""
    ADD_NEURON = "add_neuron"
    REMOVE_NEURON = "remove_neuron"
    ADD_CONNECTION = "add_connection"
    REMOVE_CONNECTION = "remove_connection"
    MODIFY_WEIGHT = "modify_weight"
    ADD_LAYER = "add_layer"
    PRUNE_LAYER = "prune_layer"


@dataclass
class ArchitecturalChange:
    """Represents a change to the architecture."""
    timestamp: float
    change_type: ModificationType
    layer_id: Optional[int]
    neuron_indices: Optional[List[int]]
    performance_delta: float
    consciousness_delta: float
    justification: str


@dataclass
class ConsciousnessConfig:
    """Configuration for self-evolving consciousness."""
    # Architecture
    input_size: int = 50
    hidden_sizes: List[int] = field(default_factory=lambda: [100, 100, 50])
    output_size: int = 10
    
    # Evolution parameters
    mutation_rate: float = 0.01
    addition_rate: float = 0.005
    pruning_threshold: float = 0.1
    
    # Consciousness metrics
    phi_target: float = 0.5
    consciousness_weight: float = 0.3
    performance_weight: float = 0.7
    
    # Meta-learning
    learning_rate: float = 0.001
    adaptation_rate: float = 0.01
    memory_capacity: int = 1000
    
    # Constraints
    max_neurons_per_layer: int = 500
    max_connections: int = 10000
    min_sparsity: float = 0.1
    
    # Simulation
    dt: float = 1.0
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)


class SelfEvolvingConsciousness:
    """Self-evolving consciousness architecture.
    
    Implements an autonomous agent that:
    - Monitors internal consciousness metrics (Phi, integration)
    - Evaluates architectural changes based on consciousness/performance
    - Adapts structure and parameters via meta-learning
    - Maintains architectural history and justification
    
    Example:
        >>> config = ConsciousnessConfig()
        >>> agent = SelfEvolvingConsciousness(config)
        >>> action = agent.act(observation)
        >>> agent.learn(reward, next_observation)
        >>> agent.evolve()
    """
    
    def __init__(self, config: Optional[ConsciousnessConfig] = None) -> None:
        """Initialize self-evolving consciousness.
        
        Args:
            config: Configuration parameters.
        """
        self.config = config or ConsciousnessConfig()
        
        # Neural architecture
        self._layers = self._initialize_layers()
        self._weights = self._initialize_weights()
        self._biases = self._initialize_biases()
        
        # Activity state
        self._activations = [np.zeros(size) for size in self.config.hidden_sizes]
        self._neuron_importance = [np.zeros(size) for size in self.config.hidden_sizes]
        
        # Consciousness metrics
        self._phi_values = []
        self._consciousness_history: List[float] = []
        self._performance_history: List[float] = []
        
        # Architectural history
        self._change_history: List[ArchitecturalChange] = []
        
        # Meta-learning memory
        self._successful_patterns: List[Dict[str, Any]] = []
        self._failed_patterns: List[Dict[str, Any]] = []
        
        logger.info("Initialized SelfEvolvingConsciousness agent")
    
    def _initialize_layers(self) -> List[int]:
        """Initialize layer sizes."""
        sizes = [self.config.input_size] + self.config.hidden_sizes + [self.config.output_size]
        return sizes
    
    def _initialize_weights(self) -> List[np.ndarray]:
        """Initialize weight matrices."""
        weights = []
        
        for i in range(len(self._layers) - 1):
            n_in = self._layers[i]
            n_out = self._layers[i + 1]
            
            # Xavier initialization
            scale = np.sqrt(2.0 / n_in)
            weight = np.random.randn(n_in, n_out) * scale
            
            # Add sparsity
            mask = np.random.random(weight.shape) < 0.3
            weight *= mask
            
            weights.append(weight)
        
        return weights
    
    def _initialize_biases(self) -> List[np.ndarray]:
        """Initialize bias vectors."""
        biases = []
        
        for size in self._layers[1:]:
            bias = np.zeros(size)
            biases.append(bias)
        
        return biases
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network.
        
        Args:
            x: Input data.
            
        Returns:
            Network output.
        """
        activation = x
        
        self._activations = [activation]
        
        for i, (weight, bias) in enumerate(zip(self._weights, self._biases)):
            # Linear transformation
            z = activation @ weight + bias
            
            # ReLU activation for hidden layers, softmax for output
            if i < len(self._weights) - 1:
                activation = np.maximum(0, z)
            else:
                # Softmax
                exp_z = np.exp(z - np.max(z))
                activation = exp_z / np.sum(exp_z)
            
            self._activations.append(activation)
        
        return activation
    
    def act(self, observation: np.ndarray, 
            epsilon: float = 0.1) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            observation: Current state observation.
            epsilon: Exploration probability.
            
        Returns:
            Selected action.
        """
        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.randint(self.config.output_size)
        else:
            # Exploit: greedy action
            output = self.forward(observation)
            return int(np.argmax(output))
    
    def learn(self, reward: float, next_observation: np.ndarray) -> float:
        """Learn from experience and update consciousness metrics.
        
        Args:
            reward: Reward received.
            next_observation: Next state observation.
            
        Returns:
            Learning loss.
        """
        # Simple policy gradient (placeholder for full RL implementation)
        output = self.forward(next_observation)
        loss = -reward * np.log(output[np.argmax(output)] + 1e-10)
        
        # Update performance history
        self._performance_history.append(reward)
        if len(self._performance_history) > self.config.memory_capacity:
            self._performance_history.pop(0)
        
        # Update neuron importance
        self._update_neuron_importance()
        
        return loss
    
    def _update_neuron_importance(self) -> None:
        """Update importance scores for all neurons."""
        for i, activation in enumerate(self._activations[1:-1]):  # Skip input and output
            # Importance based on activation magnitude and variance
            importance = np.abs(activation) * np.std(activation)
            
            # Smooth update
            self._neuron_importance[i] = (0.9 * self._neuron_importance[i] + 
                                         0.1 * importance)
    
    def calculate_phi(self) -> float:
        """Calculate integrated information (Phi).
        
        Returns:
            Phi value.
        """
        # Calculate activation entropy
        total_activation = np.concatenate(self._activations[1:-1])
        activation_prob = total_activation / (np.sum(total_activation) + 1e-10)
        entropy = -np.sum(activation_prob * np.log2(activation_prob + 1e-10))
        
        # Normalize
        max_entropy = np.log2(sum(self.config.hidden_sizes))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Integration: measure of connectivity
        total_connections = sum(w.size for w in self._weights)
        normalized_connections = total_connections / self.config.max_connections
        
        # Phi combines entropy and integration
        phi = 0.5 * normalized_entropy + 0.5 * normalized_connections
        
        self._phi_values.append(phi)
        self._consciousness_history.append(phi)
        
        if len(self._consciousness_history) > self.config.memory_capacity:
            self._consciousness_history.pop(0)
        
        return phi
    
    def evolve(self) -> Optional[ArchitecturalChange]:
        """Evaluate and potentially apply architectural changes.
        
        Returns:
            ArchitecturalChange if change was made, None otherwise.
        """
        current_phi = self.calculate_phi()
        current_performance = np.mean(self._performance_history) if self._performance_history else 0
        
        # Evaluate if evolution is needed
        needs_evolution = (current_phi < self.config.phi_target or
                          len(self._change_history) < 10)
        
        if not needs_evolution:
            return None
        
        # Generate potential modification
        modification = self._generate_modification()
        
        if modification is None:
            return None
        
        # Apply modification
        self._apply_modification(modification)
        
        # Record change
        change = ArchitecturalChange(
            timestamp=0.0,  # Will be set by simulation
            change_type=modification['type'],
            layer_id=modification.get('layer_id'),
            neuron_indices=modification.get('neuron_indices'),
            performance_delta=0.0,  # Will be updated after evaluation
            consciousness_delta=0.0,  # Will be updated after evaluation
            justification=modification['justification']
        )
        
        self._change_history.append(change)
        
        logger.info(f"Applied architectural change: {modification['type'].value}")
        
        return change
    
    def _generate_modification(self) -> Optional[Dict[str, Any]]:
        """Generate a potential architectural modification."""
        if np.random.random() < self.config.mutation_rate:
            # Weight modification
            layer_idx = np.random.randint(len(self._weights))
            weight = self._weights[layer_idx]
            
            # Modify random weights
            num_modifications = max(1, int(weight.size * 0.01))
            indices = np.random.choice(weight.size, num_modifications, replace=False)
            
            for idx in indices:
                i, j = np.unravel_index(idx, weight.shape)
                weight[i, j] += np.random.normal(0, 0.1)
            
            return {
                'type': ModificationType.MODIFY_WEIGHT,
                'layer_id': layer_idx,
                'justification': 'Stochastic weight perturbation for exploration'
            }
        
        if np.random.random() < self.config.addition_rate:
            # Add connection
            layer_idx = np.random.randint(len(self._weights))
            weight = self._weights[layer_idx]
            
            # Find zero connections
            zero_indices = np.where(weight == 0)
            if len(zero_indices[0]) > 0:
                idx = np.random.randint(len(zero_indices[0]))
                i, j = zero_indices[0][idx], zero_indices[1][idx]
                weight[i, j] = np.random.normal(0, 0.1)
                
                return {
                    'type': ModificationType.ADD_CONNECTION,
                    'layer_id': layer_idx,
                    'neuron_indices': [i, j],
                    'justification': 'Adding connection to increase integration'
                }
        
        # Pruning based on importance
        for i, importance in enumerate(self._neuron_importance):
            if i < len(self._weights):
                # Prune low-importance connections
                weight = self._weights[i]
                threshold = np.percentile(np.abs(weight), 
                                        self.config.pruning_threshold * 100)
                
                mask = np.abs(weight) > threshold
                if np.mean(mask) < self.config.min_sparsity:
                    # Too sparse, skip pruning
                    continue
                
                weight *= mask
                
                return {
                    'type': ModificationType.REMOVE_CONNECTION,
                    'layer_id': i,
                    'justification': f'Pruning connections below {threshold:.4f}'
                }
        
        return None
    
    def _apply_modification(self, modification: Dict[str, Any]) -> None:
        """Apply an architectural modification.
        
        Args:
            modification: Modification specification.
        """
        mod_type = modification['type']
        
        if mod_type in [ModificationType.MODIFY_WEIGHT, 
                       ModificationType.ADD_CONNECTION,
                       ModificationType.REMOVE_CONNECTION]:
            # Already applied in _generate_modification
            pass
        
        elif mod_type == ModificationType.ADD_NEURON:
            # Add neuron to layer
            layer_idx = modification.get('layer_id', 1)
            if layer_idx < len(self._layers) - 1:
                self._layers[layer_idx] += 1
                
                # Update weight matrices
                weight_in = self._weights[layer_idx - 1]
                weight_out = self._weights[layer_idx]
                
                # Add new row to weight_in
                new_row = np.random.randn(weight_in.shape[1]) * 0.1
                self._weights[layer_idx - 1] = np.vstack([weight_in, new_row])
                
                # Add new column to weight_out
                new_col = np.random.randn(weight_out.shape[0]) * 0.1
                self._weights[layer_idx] = np.column_stack([weight_out, new_col])
                
                # Update activation and importance
                self._activations[layer_idx] = np.append(
                    self._activations[layer_idx], 0.0
                )
                self._neuron_importance[layer_idx] = np.append(
                    self._neuron_importance[layer_idx], 0.0
                )
        
        elif mod_type == ModificationType.REMOVE_NEURON:
            # Remove neuron from layer
            layer_idx = modification.get('layer_id', 1)
            neuron_idx = modification['neuron_indices'][0]
            
            if layer_idx < len(self._layers) - 1 and \
               neuron_idx < self._layers[layer_idx]:
                
                # Remove from weight matrices
                self._weights[layer_idx - 1] = np.delete(
                    self._weights[layer_idx - 1], neuron_idx, axis=0
                )
                self._weights[layer_idx] = np.delete(
                    self._weights[layer_idx], neuron_idx, axis=1
                )
                
                # Update layer size
                self._layers[layer_idx] -= 1
                
                # Update activation and importance
                self._activations[layer_idx] = np.delete(
                    self._activations[layer_idx], neuron_idx
                )
                self._neuron_importance[layer_idx] = np.delete(
                    self._neuron_importance[layer_idx], neuron_idx
                )
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get summary of current architecture.
        
        Returns:
            Dictionary with architecture statistics.
        """
        total_neurons = sum(self._layers)
        total_connections = sum(w.size for w in self._weights)
        total_params = total_connections + sum(b.size for b in self._biases)
        
        sparsity = np.mean([np.mean(w == 0) for w in self._weights])
        
        return {
            'layers': self._layers,
            'total_neurons': total_neurons,
            'total_connections': total_connections,
            'total_parameters': total_params,
            'sparsity': float(sparsity),
            'num_changes': len(self._change_history),
            'current_phi': self._consciousness_history[-1] if self._consciousness_history else 0.0,
            'mean_performance': float(np.mean(self._performance_history)) if self._performance_history else 0.0
        }
    
    def get_change_history(self) -> List[ArchitecturalChange]:
        """Get history of architectural changes.
        
        Returns:
            List of ArchitecturalChange objects.
        """
        return self._change_history.copy()
    
    def reset(self) -> None:
        """Reset agent to initial state."""
        self._activations = [np.zeros(size) for size in self.config.hidden_sizes]
        self._neuron_importance = [np.zeros(size) for size in self.config.hidden_sizes]
        self._phi_values = []
        self._consciousness_history = []
        self._performance_history = []
        self._change_history = []
        
        logger.debug("Reset SelfEvolvingConsciousness agent")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state.
        
        Returns:
            Dictionary containing agent state.
        """
        return {
            'phi': self._consciousness_history[-1] if self._consciousness_history else 0.0,
            'performance': self._performance_history[-1] if self._performance_history else 0.0,
            'num_changes': len(self._change_history),
            'architecture': self.get_architecture_summary()
        }
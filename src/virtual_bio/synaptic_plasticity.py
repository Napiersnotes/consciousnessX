"""
Synaptic plasticity simulation including STDP.
"""

import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)

class SynapticPlasticity:
    """Simulate synaptic plasticity including STDP."""
    
    def __init__(self, config=None):
        self.config = config or {
            'stdp_learning_rate': 0.01,
            'stdp_tau_plus': 20.0,  # ms
            'stdp_tau_minus': 20.0,  # ms
            'stdp_a_plus': 0.1,
            'stdp_a_minus': 0.1,
            'homeostatic_target': 1.0,
            'homeostatic_rate': 0.001
        }
        
        # Initialize state
        self.synaptic_weights = None
        self.last_spike_times = None
        self.trace_plus = None
        self.trace_minus = None
        
    def initialize(self, n_neurons):
        """Initialize plasticity simulation."""
        self.n_neurons = n_neurons
        
        # Random synaptic weights
        self.synaptic_weights = np.random.randn(n_neurons, n_neurons) * 0.1
        np.fill_diagonal(self.synaptic_weights, 0)  # No self-connections
        
        # Last spike times
        self.last_spike_times = np.full((n_neurons, n_neurons), -np.inf)
        
        # STDP traces
        self.trace_plus = np.zeros((n_neurons, n_neurons))
        self.trace_minus = np.zeros((n_neurons, n_neurons))
        
        # Homeostatic scaling
        self.firing_rates = np.zeros(n_neurons)
        
        logger.info(f"Initialized synaptic plasticity for {n_neurons} neurons")
    
    def update_stdp(self, spikes, time_ms):
        """
        Update weights using Spike-Timing Dependent Plasticity.
        
        Args:
            spikes: Binary array of spikes (1=spike, 0=no spike)
            time_ms: Current time in milliseconds
        """
        if self.synaptic_weights is None:
            self.initialize(len(spikes))
        
        # Update STDP traces
        dt = 1.0  # Assuming 1ms time step
        
        # Decay traces
        self.trace_plus *= np.exp(-dt / self.config['stdp_tau_plus'])
        self.trace_minus *= np.exp(-dt / self.config['stdp_tau_minus'])
        
        # Update traces for spiking neurons
        spike_indices = np.where(spikes)[0]
        
        for i in spike_indices:
            # Update plus trace for presynaptic spikes
            self.trace_plus[i, :] += self.config['stdp_a_plus']
            
            # Update minus trace for postsynaptic spikes
            self.trace_minus[:, i] += self.config['stdp_a_minus']
        
        # Update weights based on trace correlations
        for i in spike_indices:
            for j in spike_indices:
                if i != j:
                    # STDP rule: LTP when pre before post, LTD when post before pre
                    # Using trace-based implementation
                    dw_plus = self.trace_minus[i, j] * self.config['stdp_learning_rate']
                    dw_minus = -self.trace_plus[j, i] * self.config['stdp_learning_rate']
                    
                    self.synaptic_weights[i, j] += dw_plus + dw_minus
        
        # Update last spike times
        for i in spike_indices:
            for j in spike_indices:
                if i != j:
                    self.last_spike_times[i, j] = time_ms
        
        # Apply weight bounds
        self.synaptic_weights = np.clip(self.synaptic_weights, -1.0, 1.0)
        
        # Homeostatic scaling
        self.update_homeostasis(spikes)
        
        return self.synaptic_weights.copy()
    
    def update_homeostasis(self, spikes):
        """Update homeostatic scaling of weights."""
        # Update firing rates (exponential moving average)
        alpha = 0.01
        self.firing_rates = (1 - alpha) * self.firing_rates + alpha * spikes
        
        # Calculate scaling factor
        target_rate = self.config['homeostatic_target']
        scale_factor = 1.0 + self.config['homeostatic_rate'] * (target_rate - self.firing_rates[:, np.newaxis])
        
        # Scale weights
        self.synaptic_weights *= scale_factor
        
        # Normalize weights to maintain total strength
        row_sums = np.sum(np.abs(self.synaptic_weights), axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.synaptic_weights /= row_sums
    
    def apply_hebbian_learning(self, pre_activity, post_activity):
        """
        Apply Hebbian learning rule.
        
        Args:
            pre_activity: Presynaptic neuron activity
            post_activity: Postsynaptic neuron activity
            
        Returns:
            updated_weights: New synaptic weights
        """
        if self.synaptic_weights is None:
            n = len(pre_activity)
            self.initialize(n)
        
        # Simple Hebbian rule: Δw = η * pre * post
        dw = self.config['stdp_learning_rate'] * np.outer(post_activity, pre_activity)
        
        # Update weights
        self.synaptic_weights += dw
        
        # Normalize
        self.normalize_weights()
        
        return self.synaptic_weights.copy()
    
    def normalize_weights(self):
        """Normalize weights to maintain stability."""
        # L2 normalization
        norms = np.linalg.norm(self.synaptic_weights, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.synaptic_weights /= norms
    
    def get_plasticity_metrics(self):
        """Calculate plasticity-related metrics."""
        if self.synaptic_weights is None:
            return {}
        
        metrics = {
            'mean_weight': float(np.mean(self.synaptic_weights)),
            'weight_std': float(np.std(self.synaptic_weights)),
            'weight_entropy': self.calculate_weight_entropy(),
            'connectivity': float(np.mean(self.synaptic_weights != 0)),
            'symmetry': self.calculate_symmetry(),
            'clustering_coefficient': self.calculate_clustering()
        }
        
        return metrics
    
    def calculate_weight_entropy(self):
        """Calculate entropy of weight distribution."""
        # Flatten and normalize weights
        weights_flat = self.synaptic_weights.flatten()
        weights_abs = np.abs(weights_flat)
        weights_abs = weights_abs / np.sum(weights_abs)
        
        # Calculate entropy
        entropy = -np.sum(weights_abs * np.log(weights_abs + 1e-10))
        
        # Normalize by max entropy
        max_entropy = np.log(len(weights_abs))
        if max_entropy > 0:
            entropy /= max_entropy
        
        return float(entropy)
    
    def calculate_symmetry(self):
        """Calculate symmetry of weight matrix."""
        symmetry = 0.0
        n = self.n_neurons
        
        for i in range(n):
            for j in range(i + 1, n):
                symmetry += abs(self.synaptic_weights[i, j] - self.synaptic_weights[j, i])
        
        # Normalize
        if n > 1:
            max_possible = n * (n - 1) / 2
            symmetry = 1.0 - (symmetry / max_possible)
        
        return float(symmetry)
    
    def calculate_clustering(self):
        """Calculate clustering coefficient."""
        # Binarize weights
        binary_weights = (np.abs(self.synaptic_weights) > 0.01).astype(float)
        
        clustering_sum = 0.0
        n = self.n_neurons
        
        for i in range(n):
            # Get neighbors
            neighbors = np.where(binary_weights[i, :] > 0)[0]
            k = len(neighbors)
            
            if k < 2:
                continue
            
            # Count triangles
            triangles = 0
            for j in neighbors:
                for k in neighbors:
                    if j < k and binary_weights[j, k] > 0:
                        triangles += 1
            
            # Local clustering coefficient
            max_triangles = k * (k - 1) / 2
            if max_triangles > 0:
                clustering_sum += triangles / max_triangles
        
        # Average clustering coefficient
        if n > 0:
            avg_clustering = clustering_sum / n
        else:
            avg_clustering = 0.0
        
        return float(avg_clustering)

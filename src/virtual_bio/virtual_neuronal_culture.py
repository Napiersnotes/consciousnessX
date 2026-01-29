"""
Virtual neuronal culture simulation.
"""

import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)

class VirtualNeuronalCulture:
    """Simulate a virtual neuronal culture."""
    
    def __init__(self, config=None):
        self.config = config or {
            'n_neurons': 1000,
            'connection_probability': 0.1,
            'simulation_dt': 0.001,  # seconds
            'membrane_time_constant': 0.02,  # seconds
            'resting_potential': -65.0,  # mV
            'threshold_potential': -50.0,  # mV
            'reset_potential': -70.0,  # mV
            'refractory_period': 0.002,  # seconds
            'noise_amplitude': 2.0  # mV
        }
        
        # Initialize state
        self.n_neurons = self.config['n_neurons']
        self.initialize_network()
        
        logger.info(f"Initialized virtual neuronal culture with {self.n_neurons} neurons")
    
    def initialize_network(self):
        """Initialize neuronal network."""
        # Membrane potentials
        self.membrane_potentials = np.ones(self.n_neurons) * self.config['resting_potential']
        
        # Synaptic weights
        self.synaptic_weights = np.random.randn(self.n_neurons, self.n_neurons) * 0.5
        
        # Apply connection probability
        mask = np.random.rand(self.n_neurons, self.n_neurons) < self.config['connection_probability']
        self.synaptic_weights = self.synaptic_weights * mask
        
        # No self-connections
        np.fill_diagonal(self.synaptic_weights, 0)
        
        # Refractory period tracking
        self.refractory_timers = np.zeros(self.n_neurons)
        
        # Spike history
        self.spike_history = []
        self.membrane_history = []
        
        # Input currents
        self.input_currents = np.zeros(self.n_neurons)
        
        # Connectivity metrics
        self.calculate_connectivity_metrics()
    
    def calculate_connectivity_metrics(self):
        """Calculate initial connectivity metrics."""
        self.metrics = {
            'mean_degree': np.mean(np.sum(self.synaptic_weights != 0, axis=1)),
            'clustering': self.calculate_clustering_coefficient(),
            'small_worldness': self.calculate_small_worldness(),
            'modularity': self.calculate_modularity()
        }
    
    def update(self, external_input=None, time_step=None):
        """
        Update neuronal culture simulation.
        
        Args:
            external_input: External input to neurons (optional)
            time_step: Custom time step (optional)
            
        Returns:
            spikes: Array of spikes for this time step
        """
        dt = time_step or self.config['simulation_dt']
        
        # Add external input if provided
        if external_input is not None:
            self.input_currents += external_input
        
        # Add noise
        noise = np.random.randn(self.n_neurons) * self.config['noise_amplitude']
        self.input_currents += noise
        
        # Update membrane potentials (Leaky Integrate-and-Fire model)
        dV = (-(self.membrane_potentials - self.config['resting_potential']) + self.input_currents) / self.config['membrane_time_constant']
        self.membrane_potentials += dV * dt
        
        # Apply refractory period
        self.refractory_timers = np.maximum(0, self.refractory_timers - dt)
        self.membrane_potentials[self.refractory_timers > 0] = self.config['reset_potential']
        
        # Check for spikes
        spikes = self.membrane_potentials >= self.config['threshold_potential']
        
        # Reset spiking neurons
        self.membrane_potentials[spikes] = self.config['reset_potential']
        self.refractory_timers[spikes] = self.config['refractory_period']
        
        # Propagate spikes through network
        if np.any(spikes):
            spike_indices = np.where(spikes)[0]
            
            # Add synaptic inputs from spiking neurons
            for i in spike_indices:
                self.input_currents += self.synaptic_weights[i, :] * 10.0  # Scale synaptic input
            
            # Reset input currents for spiking neurons
            self.input_currents[spikes] = 0
        
        # Decay input currents
        self.input_currents *= np.exp(-dt / 0.005)  # Fast decay
        
        # Store history
        self.spike_history.append(spikes.copy())
        self.membrane_history.append(self.membrane_potentials.copy())
        
        # Limit history size
        if len(self.spike_history) > 10000:
            self.spike_history = self.spike_history[-10000:]
            self.membrane_history = self.membrane_history[-10000:]
        
        return spikes
    
    def stimulate_pattern(self, pattern_indices, amplitude=20.0, duration=0.01):
        """
        Stimulate specific neurons with a pattern.
        
        Args:
            pattern_indices: Indices of neurons to stimulate
            amplitude: Stimulation amplitude (mV)
            duration: Stimulation duration (seconds)
            
        Returns:
            response: Network response to stimulation
        """
        # Create stimulation pattern
        stimulation = np.zeros(self.n_neurons)
        stimulation[pattern_indices] = amplitude
        
        # Apply stimulation for multiple time steps
        n_steps = int(duration / self.config['simulation_dt'])
        responses = []
        
        for _ in range(n_steps):
            spikes = self.update(external_input=stimulation)
            responses.append(spikes)
        
        return np.array(responses)
    
    def calculate_connectivity_metrics(self):
        """Calculate connectivity metrics."""
        # Binary connectivity matrix
        binary_weights = (np.abs(self.synaptic_weights) > 0.01).astype(float)
        
        # Degree distribution
        degrees = np.sum(binary_weights, axis=1)
        mean_degree = np.mean(degrees)
        
        # Clustering coefficient
        clustering = self.calculate_clustering_coefficient(binary_weights)
        
        # Small-worldness
        small_worldness = self.calculate_small_worldness(binary_weights)
        
        # Modularity
        modularity = self.calculate_modularity(binary_weights)
        
        self.metrics = {
            'mean_degree': float(mean_degree),
            'clustering': float(clustering),
            'small_worldness': float(small_worldness),
            'modularity': float(modularity),
            'degree_std': float(np.std(degrees))
        }
        
        return self.metrics
    
    def calculate_clustering_coefficient(self, binary_weights=None):
        """Calculate average clustering coefficient."""
        if binary_weights is None:
            binary_weights = (np.abs(self.synaptic_weights) > 0.01).astype(float)
        
        n = self.n_neurons
        clustering_sum = 0.0
        valid_neurons = 0
        
        for i in range(n):
            neighbors = np.where(binary_weights[i, :] > 0)[0]
            k = len(neighbors)
            
            if k < 2:
                continue
            
            # Count triangles among neighbors
            triangles = 0
            for j in range(len(neighbors)):
                for l in range(j + 1, len(neighbors)):
                    if binary_weights[neighbors[j], neighbors[l]] > 0:
                        triangles += 1
            
            max_triangles = k * (k - 1) / 2
            if max_triangles > 0:
                clustering_sum += triangles / max_triangles
                valid_neurons += 1
        
        if valid_neurons > 0:
            return clustering_sum / valid_neurons
        else:
            return 0.0
    
    def calculate_small_worldness(self, binary_weights=None):
        """Calculate small-worldness metric."""
        if binary_weights is None:
            binary_weights = (np.abs(self.synaptic_weights) > 0.01).astype(float)
        
        # Calculate characteristic path length
        path_length = self.calculate_characteristic_path_length(binary_weights)
        
        # Calculate clustering coefficient
        clustering = self.calculate_clustering_coefficient(binary_weights)
        
        # Generate random network with same degree distribution
        random_clustering, random_path_length = self.generate_random_network_metrics(binary_weights)
        
        if random_clustering > 0 and random_path_length > 0:
            small_worldness = (clustering / random_clustering) / (path_length / random_path_length)
        else:
            small_worldness = 1.0
        
        return small_worldness
    
    def calculate_characteristic_path_length(self, binary_weights):
        """Calculate characteristic path length using Floyd-Warshall."""
        n = self.n_neurons
        
        # Initialize distance matrix
        dist = np.where(binary_weights > 0, 1, np.inf)
        np.fill_diagonal(dist, 0)
        
        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, j] > dist[i, k] + dist[k, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        
        # Calculate average path length
        mask = np.isfinite(dist)
        if np.sum(mask) > 0:
            avg_path_length = np.mean(dist[mask])
        else:
            avg_path_length = np.inf
        
        return avg_path_length
    
    def generate_random_network_metrics(self, binary_weights):
        """Generate metrics for random network with same degree distribution."""
        # Simple approximation - in reality would generate actual random networks
        n = self.n_neurons
        mean_degree = np.mean(np.sum(binary_weights, axis=1))
        
        # Expected clustering for random network
        random_clustering = mean_degree / n
        
        # Expected path length for random network
        random_path_length = np.log(n) / np.log(mean_degree) if mean_degree > 1 else n
        
        return random_clustering, random_path_length
    
    def calculate_modularity(self, binary_weights=None):
        """Calculate modularity using simple community detection."""
        if binary_weights is None:
            binary_weights = (np.abs(self.synaptic_weights) > 0.01).astype(float)
        
        n = self.n_neurons
        
        # Simple community detection using spectral clustering
        try:
            from sklearn.cluster import SpectralClustering
            
            # Use first few eigenvectors for clustering
            n_clusters = min(10, n // 20)
            if n_clusters < 2:
                return 0.0
            
            clustering = SpectralClustering(n_clusters=n_clusters, 
                                          affinity='precomputed',
                                          random_state=42)
            communities = clustering.fit_predict(binary_weights + binary_weights.T)
            
            # Calculate modularity
            m = np.sum(binary_weights)  # Total number of edges
            if m == 0:
                return 0.0
            
            modularity = 0.0
            for c in range(n_clusters):
                nodes_in_c = np.where(communities == c)[0]
                l_c = np.sum(binary_weights[np.ix_(nodes_in_c, nodes_in_c)])
                d_c = np.sum(np.sum(binary_weights[nodes_in_c, :], axis=1))
                
                if m > 0:
                    modularity += (l_c / m) - (d_c / (2 * m)) ** 2
            
            return modularity
            
        except Exception as e:
            logger.warning(f"Could not calculate modularity: {e}")
            return 0.0
    
    def get_activity_metrics(self, window=1000):
        """Calculate activity metrics from recent history."""
        if len(self.spike_history) < window:
            window = len(self.spike_history)
        
        if window == 0:
            return {}
        
        # Get recent spikes
        recent_spikes = np.array(self.spike_history[-window:])
        
        # Calculate metrics
        firing_rates = np.mean(recent_spikes, axis=0) / self.config['simulation_dt']
        
        # Synchrony (mean pairwise correlation)
        if len(recent_spikes) > 1:
            correlations = np.corrcoef(recent_spikes.T)
            np.fill_diagonal(correlations, 0)
            synchrony = np.mean(correlations)
        else:
            synchrony = 0.0
        
        # Burst detection
        total_spikes = np.sum(recent_spikes)
        burst_threshold = np.mean(firing_rates) + 2 * np.std(firing_rates)
        burst_neurons = np.sum(firing_rates > burst_threshold)
        
        # Information entropy
        spike_prob = np.mean(recent_spikes)
        if 0 < spike_prob < 1:
            entropy = - (spike_prob * np.log(spike_prob) + (1 - spike_prob) * np.log(1 - spike_prob))
            entropy /= np.log(2)  # Convert to bits
        else:
            entropy = 0.0
        
        metrics = {
            'mean_firing_rate': float(np.mean(firing_rates)),
            'firing_rate_std': float(np.std(firing_rates)),
            'synchrony': float(synchrony),
            'total_spikes': int(total_spikes),
            'burst_neurons': int(burst_neurons),
            'burst_fraction': float(burst_neurons / self.n_neurons) if self.n_neurons > 0 else 0.0,
            'activity_entropy': float(entropy),
            'active_neurons': int(np.sum(firing_rates > 0))
        }
        
        return metrics

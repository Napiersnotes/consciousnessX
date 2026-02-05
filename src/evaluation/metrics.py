"""
ConsciousnessMetrics module for quantifying consciousness states
"""

import numpy as np
from typing import Dict, Any, List, Optional
import torch


class ConsciousnessMetrics:
    """
    Metrics for quantifying consciousness states in artificial systems.
    
    This class provides various metrics inspired by theories of consciousness,
    including Integrated Information Theory (IIT), Orch OR, and quantum coherence.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ConsciousnessMetrics.
        
        Args:
            config: Configuration dictionary for metrics
        """
        self.config = config or {}
        
        # Metric weights
        self.weights = self.config.get('weights', {
            'phi': 0.3,
            'quantum_coherence': 0.25,
            'integrated_information': 0.25,
            'neural_complexity': 0.2
        })
    
    def compute_phi(
        self,
        transition_matrix: np.ndarray,
        partition: Optional[List[int]] = None
    ) -> float:
        """
        Compute Phi (integrated information) using IIT principles.
        
        Args:
            transition_matrix: State transition matrix
            partition: Optional partition for effective information
            
        Returns:
            Phi value between 0 and 1
        """
        # Compute effective information
        effective_info = self._compute_effective_information(transition_matrix)
        
        # Normalize to [0, 1]
        phi = min(1.0, effective_info / np.log2(transition_matrix.shape[0]))
        
        return phi
    
    def compute_quantum_coherence(
        self,
        quantum_state: np.ndarray
    ) -> float:
        """
        Compute quantum coherence of a state.
        
        Args:
            quantum_state: Complex quantum state vector
            
        Returns:
            Coherence value between 0 and 1
        """
        # Normalize state
        normalized_state = quantum_state / np.linalg.norm(quantum_state)
        
        # Compute density matrix
        density_matrix = np.outer(normalized_state, np.conj(normalized_state))
        
        # Compute coherence as off-diagonal elements
        diag_elements = np.diag(density_matrix)
        coherence = 1.0 - np.sum(np.abs(diag_elements))
        
        return max(0.0, min(1.0, coherence))
    
    def compute_integrated_information(
        self,
        neural_activity: np.ndarray,
        connectivity_matrix: np.ndarray
    ) -> float:
        """
        Compute integrated information from neural activity.
        
        Args:
            neural_activity: Neural activity patterns
            connectivity_matrix: Neural connectivity matrix
            
        Returns:
            Integrated information value between 0 and 1
        """
        # Compute mutual information across network
        mutual_info = self._compute_mutual_information(
            neural_activity,
            connectivity_matrix
        )
        
        # Normalize
        integrated_info = min(1.0, mutual_info / np.log2(neural_activity.shape[0]))
        
        return integrated_info
    
    def compute_neural_complexity(
        self,
        neural_activity: np.ndarray
    ) -> float:
        """
        Compute neural complexity measure.
        
        Args:
            neural_activity: Neural activity patterns
            
        Returns:
            Complexity value between 0 and 1
        """
        # Compute entropy
        entropy = self._compute_entropy(neural_activity)
        
        # Compute correlation-based complexity
        correlation_matrix = np.corrcoef(neural_activity)
        complexity = np.mean(np.abs(correlation_matrix))
        
        # Combine metrics
        neural_complexity = (entropy + complexity) / 2
        
        return min(1.0, neural_complexity)
    
    def compute_consciousness_level(
        self,
        metrics: Dict[str, float]
    ) -> float:
        """
        Compute overall consciousness level from individual metrics.
        
        Args:
            metrics: Dictionary of individual metrics
            
        Returns:
            Overall consciousness level between 0 and 1
        """
        # Compute weighted sum
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric_name, value in metrics.items():
            if metric_name in self.weights:
                weight = self.weights[metric_name]
                weighted_sum += weight * value
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def compute_all_metrics(
        self,
        quantum_state: np.ndarray,
        neural_activity: np.ndarray,
        connectivity_matrix: np.ndarray,
        transition_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all consciousness metrics.
        
        Args:
            quantum_state: Complex quantum state vector
            neural_activity: Neural activity patterns
            connectivity_matrix: Neural connectivity matrix
            transition_matrix: State transition matrix
            
        Returns:
            Dictionary of all computed metrics
        """
        metrics = {
            'phi': self.compute_phi(transition_matrix),
            'quantum_coherence': self.compute_quantum_coherence(quantum_state),
            'integrated_information': self.compute_integrated_information(
                neural_activity, connectivity_matrix
            ),
            'neural_complexity': self.compute_neural_complexity(neural_activity)
        }
        
        # Compute overall consciousness level
        metrics['consciousness_level'] = self.compute_consciousness_level(metrics)
        
        return metrics
    
    def _compute_effective_information(
        self,
        transition_matrix: np.ndarray
    ) -> float:
        """
        Compute effective information from transition matrix.
        
        Args:
            transition_matrix: State transition matrix
            
        Returns:
            Effective information value
        """
        # Compute entropy of each row
        row_entropies = []
        for row in transition_matrix:
            # Avoid log(0)
            row = row + 1e-10
            row = row / np.sum(row)
            entropy = -np.sum(row * np.log2(row))
            row_entropies.append(entropy)
        
        # Return average effective information
        return np.mean(row_entropies)
    
    def _compute_mutual_information(
        self,
        neural_activity: np.ndarray,
        connectivity_matrix: np.ndarray
    ) -> float:
        """
        Compute mutual information across neural network.
        
        Args:
            neural_activity: Neural activity patterns
            connectivity_matrix: Neural connectivity matrix
            
        Returns:
            Mutual information value
        """
        # Compute correlation-based mutual information
        correlation_matrix = np.corrcoef(neural_activity)
        
        # Weight by connectivity
        weighted_correlation = correlation_matrix * connectivity_matrix
        
        # Sum of absolute correlations
        mutual_info = np.sum(np.abs(weighted_correlation)) / 2
        
        return mutual_info
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """
        Compute entropy of data.
        
        Args:
            data: Data array
            
        Returns:
            Entropy value
        """
        # Compute histogram
        hist, _ = np.histogram(data, bins=50, density=True)
        
        # Avoid log(0)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)
        
        # Compute entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        return min(1.0, entropy / np.log2(50))  # Normalize
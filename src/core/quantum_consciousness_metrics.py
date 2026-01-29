"""
Quantum consciousness metrics calculation.
"""

import numpy as np
from scipy import linalg
import logging

logger = logging.getLogger(__name__)

class QuantumConsciousnessMetrics:
    """Calculate quantum consciousness metrics."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.hbar = self.config.get('hbar', 1.054571817e-34)
        
    def calculate_quantum_coherence(self, density_matrix):
        """
        Calculate quantum coherence from density matrix.
        
        Args:
            density_matrix: Complex numpy array representing density matrix
            
        Returns:
            coherence: Measure of quantum coherence
        """
        try:
            # Calculate von Neumann entropy
            eigenvalues = linalg.eigvalsh(density_matrix)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = eigenvalues[eigenvalues > 0]
            entropy = -np.sum(eigenvalues * np.log(eigenvalues))
            
            # Calculate purity
            purity = np.trace(density_matrix @ density_matrix)
            
            # Coherence measure (based on off-diagonal elements)
            off_diag = density_matrix - np.diag(np.diag(density_matrix))
            coherence = np.linalg.norm(off_diag, 'fro')
            
            # Combined metric
            quantum_coherence = coherence * (1 - entropy) * purity
            
            return {
                'coherence': float(coherence),
                'entropy': float(entropy),
                'purity': float(purity),
                'quantum_coherence': float(quantum_coherence)
            }
            
        except Exception as e:
            logger.error(f"Error calculating quantum coherence: {e}")
            return {'coherence': 0.0, 'entropy': 0.0, 'purity': 0.0, 'quantum_coherence': 0.0}
    
    def calculate_quantum_integration(self, system_states):
        """
        Calculate quantum integration (precursor to phi).
        
        Args:
            system_states: List of subsystem quantum states
            
        Returns:
            integration: Measure of quantum integration
        """
        if not system_states:
            return 0.0
        
        try:
            # Calculate mutual information between subsystems
            n_subsystems = len(system_states)
            mutual_info_matrix = np.zeros((n_subsystems, n_subsystems))
            
            for i in range(n_subsystems):
                for j in range(i + 1, n_subsystems):
                    # Simplified mutual information calculation
                    # In real implementation, would use quantum mutual information
                    state_i = system_states[i]
                    state_j = system_states[j]
                    
                    if isinstance(state_i, dict) and 'entropy' in state_i:
                        entropy_i = state_i['entropy']
                        entropy_j = state_j['entropy']
                        
                        # Simplified joint entropy (would need actual joint state)
                        joint_entropy = max(entropy_i, entropy_j)
                        
                        mutual_info = entropy_i + entropy_j - joint_entropy
                        mutual_info_matrix[i, j] = mutual_info
                        mutual_info_matrix[j, i] = mutual_info
            
            # Total integration is sum of mutual information
            total_integration = np.sum(mutual_info_matrix) / 2
            
            # Normalize by number of connections
            max_possible = (n_subsystems * (n_subsystems - 1)) / 2
            if max_possible > 0:
                normalized_integration = total_integration / max_possible
            else:
                normalized_integration = 0.0
            
            return {
                'total_integration': float(total_integration),
                'normalized_integration': float(normalized_integration),
                'mutual_info_matrix': mutual_info_matrix.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error calculating quantum integration: {e}")
            return {'total_integration': 0.0, 'normalized_integration': 0.0, 'mutual_info_matrix': []}
    
    def calculate_orch_or_metric(self, microtubule_states, collapse_probabilities):
        """
        Calculate Orchestrated Objective Reduction metric.
        
        Args:
            microtubule_states: Array of microtubule quantum states
            collapse_probabilities: Array of collapse probabilities
            
        Returns:
            orch_or_score: ORCH-OR consciousness metric
        """
        try:
            # Calculate coherence across microtubules
            coherence_scores = []
            for state in microtubule_states:
                if isinstance(state, dict) and 'coherence' in state:
                    coherence_scores.append(state['coherence'])
                elif hasattr(state, 'coherence'):
                    coherence_scores.append(state.coherence)
                else:
                    coherence_scores.append(0.0)
            
            avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
            
            # Calculate synchronization of collapses
            collapse_std = np.std(collapse_probabilities) if len(collapse_probabilities) > 1 else 0.0
            collapse_sync = 1.0 / (1.0 + collapse_std)  # Higher sync = lower std
            
            # Calculate orchestration (pattern complexity)
            if len(collapse_probabilities) > 1:
                # Use Fourier transform to analyze patterns
                fft_result = np.fft.fft(collapse_probabilities)
                power_spectrum = np.abs(fft_result) ** 2
                # Entropy of power spectrum measures pattern complexity
                power_spectrum = power_spectrum / np.sum(power_spectrum)
                power_spectrum = power_spectrum[power_spectrum > 0]
                pattern_complexity = -np.sum(power_spectrum * np.log(power_spectrum))
                pattern_complexity = pattern_complexity / np.log(len(power_spectrum)) if len(power_spectrum) > 1 else 0.0
            else:
                pattern_complexity = 0.0
            
            # Combined ORCH-OR score
            orch_or_score = avg_coherence * collapse_sync * pattern_complexity
            
            return {
                'orch_or_score': float(orch_or_score),
                'avg_coherence': float(avg_coherence),
                'collapse_sync': float(collapse_sync),
                'pattern_complexity': float(pattern_complexity)
            }
            
        except Exception as e:
            logger.error(f"Error calculating ORCH-OR metric: {e}")
            return {'orch_or_score': 0.0, 'avg_coherence': 0.0, 'collapse_sync': 0.0, 'pattern_complexity': 0.0}
    
    def assess_consciousness_level(self, metrics_dict):
        """
        Assess overall consciousness level from quantum metrics.
        
        Args:
            metrics_dict: Dictionary containing quantum metrics
            
        Returns:
            consciousness_level: 0-1 consciousness score
        """
        try:
            # Extract metrics with defaults
            quantum_coherence = metrics_dict.get('quantum_coherence', 0.0)
            normalized_integration = metrics_dict.get('normalized_integration', 0.0)
            orch_or_score = metrics_dict.get('orch_or_score', 0.0)
            
            # Normalize each metric to 0-1 range
            qc_norm = min(max(quantum_coherence, 0.0), 1.0)
            int_norm = min(max(normalized_integration, 0.0), 1.0)
            orch_norm = min(max(orch_or_score, 0.0), 1.0)
            
            # Weighted combination
            weights = {
                'coherence': 0.4,
                'integration': 0.3,
                'orch_or': 0.3
            }
            
            consciousness_level = (
                weights['coherence'] * qc_norm +
                weights['integration'] * int_norm +
                weights['orch_or'] * orch_norm
            )
            
            # Apply nonlinear scaling (consciousness might emerge nonlinearly)
            consciousness_level = np.tanh(consciousness_level * 2)  # Squash to 0-1
            
            return float(consciousness_level)
            
        except Exception as e:
            logger.error(f"Error assessing consciousness level: {e}")
            return 0.0

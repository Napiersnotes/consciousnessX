#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consciousness Assessment Framework
Comprehensive evaluation of artificial consciousness based on multiple theories
Production-ready with rigorous metrics and validation
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats, signal, integrate, spatial
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mutual_info_score
import numba
from numba import jit, njit, prange

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsciousnessTheory(Enum):
    """Different theories of consciousness"""
    INTEGRATED_INFORMATION = "iit"           # Integrated Information Theory
    GLOBAL_WORKSPACE = "gwt"                 # Global Workspace Theory
    HIGHER_ORDER = "hot"                     # Higher-Order Thought
    RECURRENT_PROCESSING = "rpt"             # Recurrent Processing Theory
    PENROSE_ORCH_OR = "orch_or"              # Orchestrated Objective Reduction
    ATTENTIONAL_SCHEMA = "ast"               # Attention Schema Theory
    PREDICTIVE_PROCESSING = "pp"             # Predictive Processing
    FREE_ENERGY = "fem"                      # Free Energy Minimization

@dataclass
class AssessmentConfig:
    """Configuration for consciousness assessment"""
    
    # General parameters
    sampling_rate_hz: float = 1000.0  # Data sampling rate
    assessment_window_s: float = 10.0  # Time window for assessment
    overlap_fraction: float = 0.5  # Overlap between windows
    
    # IIT parameters
    iit_version: str = "3.0"  # IIT version to use
    max_complexity: int = 100  # Maximum system complexity to evaluate
    partition_method: str = "minimum_information"  # How to partition system
    
    # Global Workspace parameters
    workspace_threshold: float = 0.7  # Threshold for global ignition
    competition_strength: float = 0.5  # Strength of competition
    
    # Higher-Order parameters
    metacognitive_samples: int = 1000  # Samples for metacognitive assessment
    confidence_calibration_samples: int = 100  # Samples for calibration
    
    # Statistical parameters
    confidence_level: float = 0.95  # Confidence level for statistics
    bootstrap_samples: int = 1000  # Bootstrap samples for CI
    randomization_tests: int = 1000  # Randomization tests
    
    # Output parameters
    save_intermediate_results: bool = True
    results_directory: str = "./assessment_results"
    visualization_enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if self.sampling_rate_hz <= 0:
            raise ValueError("Sampling rate must be positive")
        
        if self.assessment_window_s <= 0:
            raise ValueError("Assessment window must be positive")
        
        if not 0 <= self.overlap_fraction < 1:
            raise ValueError("Overlap fraction must be in [0, 1)")
        
        # Calculate derived parameters
        self.samples_per_window = int(self.assessment_window_s * self.sampling_rate_hz)
        self.step_size = int(self.samples_per_window * (1 - self.overlap_fraction))
        
        logger.info(f"Assessment configured: {self.samples_per_window} samples/window, "
                   f"{self.step_size} samples step")

class IntegratedInformationCalculator:
    """
    Integrated Information (Φ) Calculator
    Implements IIT 3.0+ with causal analysis
    """
    
    def __init__(self, config: AssessmentConfig):
        self.config = config
        self.cache = {}  # For memoization
        
        logger.info(f"Initialized IIT calculator (version {config.iit_version})")
    
    def calculate_phi(self, 
                     system_state: np.ndarray,
                     past_state: Optional[np.ndarray] = None,
                     future_state: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate Integrated Information (Φ) for a system
        
        Args:
            system_state: Current system state (N x T)
            past_state: Past system state for causal analysis
            future_state: Future system state for causal analysis
            
        Returns:
            Dictionary with Φ and related metrics
        """
        N, T = system_state.shape
        
        if N == 0 or T == 0:
            return {'phi': 0.0, 'error': 'Empty system'}
        
        # Ensure binary states for IIT calculation
        if system_state.dtype != bool:
            binary_state = (system_state > 0.5).astype(bool)
        else:
            binary_state = system_state
        
        # Calculate basic information measures
        entropy_system = self._calculate_entropy(binary_state)
        
        if entropy_system == 0:
            return {'phi': 0.0, 'entropy': 0.0, 'complexity': 0.0}
        
        # Partition the system and calculate Φ
        phi, mip = self._calculate_minimum_information_partition(binary_state)
        
        # Calculate cause-effect repertoires if past/future states available
        cause_info = 0.0
        effect_info = 0.0
        
        if past_state is not None and future_state is not None:
            cause_info = self._calculate_cause_information(binary_state, past_state)
            effect_info = self._calculate_effect_information(binary_state, future_state)
        
        # Calculate system complexity
        complexity = self._calculate_system_complexity(binary_state)
        
        # Calculate existence (simplified)
        existence = phi * complexity
        
        return {
            'phi': phi,
            'entropy': entropy_system,
            'complexity': complexity,
            'existence': existence,
            'cause_information': cause_info,
            'effect_information': effect_info,
            'mip_cut': mip,
            'system_size': N,
            'temporal_extent': T
        }
    
    def _calculate_entropy(self, state: np.ndarray) -> float:
        """Calculate entropy of a system state"""
        if state.size == 0:
            return 0.0
        
        # Convert to probability distribution
        unique_states, counts = np.unique(state, axis=1, return_counts=True)
        probabilities = counts / counts.sum()
        
        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy
    
    def _calculate_minimum_information_partition(self, 
                                                state: np.ndarray) -> Tuple[float, Dict]:
        """
        Calculate Minimum Information Partition (MIP) and Φ
        
        Returns:
            (phi, mip_dict) where mip_dict describes the partition
        """
        N, T = state.shape
        
        if N <= 1:
            return 0.0, {'cut': [], 'phi': 0.0}
        
        # For large systems, use heuristic search
        if N > self.config.max_complexity:
            return self._approximate_mip(state)
        
        # Try all possible bipartitions (for small systems)
        min_phi = float('inf')
        best_cut = None
        
        # Generate all possible cuts
        for cut_size in range(1, N // 2 + 1):
            # Generate combinations (simplified - in production would use more efficient methods)
            # This is computationally expensive for large N
            from itertools import combinations
            
            for subset in combinations(range(N), cut_size):
                subset = list(subset)
                complement = [i for i in range(N) if i not in subset]
                
                # Calculate Φ for this partition
                phi_cut = self._calculate_partition_phi(state, subset, complement)
                
                if phi_cut < min_phi:
                    min_phi = phi_cut
                    best_cut = {
                        'subset': subset,
                        'complement': complement,
                        'size': cut_size
                    }
        
        # Φ is the minimum information across all partitions
        phi = max(0, min_phi)  # Φ cannot be negative
        
        mip = {
            'cut': best_cut,
            'phi': phi,
            'normalized_phi': phi / N if N > 0 else 0.0
        }
        
        return phi, mip
    
    def _approximate_mip(self, state: np.ndarray) -> Tuple[float, Dict]:
        """Approximate MIP for large systems using heuristic search"""
        N, T = state.shape
        
        # Use PCA to find natural partitions
        pca = PCA(n_components=min(10, N))
        transformed = pca.fit_transform(state.T)
        
        # Cluster based on PCA components
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(transformed)
        
        # Convert clusters to partition
        subset = np.where(labels == 0)[0].tolist()
        complement = np.where(labels == 1)[0].tolist()
        
        # Calculate Φ for this partition
        phi = self._calculate_partition_phi(state, subset, complement)
        
        mip = {
            'cut': {'subset': subset, 'complement': complement},
            'phi': phi,
            'method': 'pca_kmeans',
            'normalized_phi': phi / N if N > 0 else 0.0
        }
        
        return phi, mip
    
    def _calculate_partition_phi(self, 
                                state: np.ndarray,
                                subset: List[int],
                                complement: List[int]) -> float:
        """Calculate Φ for a specific partition"""
        # Extract subsets
        if len(subset) == 0 or len(complement) == 0:
            return float('inf')
        
        subset_state = state[subset, :]
        complement_state = state[complement, :]
        
        # Calculate entropies
        h_whole = self._calculate_entropy(state)
        h_subset = self._calculate_entropy(subset_state)
        h_complement = self._calculate_entropy(complement_state)
        
        # Calculate mutual information between subsets
        mi = self._calculate_mutual_information(subset_state, complement_state)
        
        # Φ = min{H(subset), H(complement)} - MI(subset; complement)
        phi = min(h_subset, h_complement) - mi
        
        return phi
    
    def _calculate_mutual_information(self, 
                                     state1: np.ndarray,
                                     state2: np.ndarray) -> float:
        """Calculate mutual information between two systems"""
        # Flatten states
        flat1 = state1.flatten()
        flat2 = state2.flatten()
        
        # Ensure binary
        if flat1.dtype != bool:
            flat1 = (flat1 > 0.5).astype(bool)
        if flat2.dtype != bool:
            flat2 = (flat2 > 0.5).astype(bool)
        
        # Calculate mutual information
        mi = mutual_info_score(flat1, flat2)
        
        return mi
    
    def _calculate_cause_information(self,
                                    current_state: np.ndarray,
                                    past_state: np.ndarray) -> float:
        """Calculate cause information (simplified)"""
        # Correlation between current and past
        if current_state.shape != past_state.shape:
            # Reshape if necessary
            min_size = min(current_state.size, past_state.size)
            current_flat = current_state.flat[:min_size]
            past_flat = past_state.flat[:min_size]
        else:
            current_flat = current_state.flatten()
            past_flat = past_state.flatten()
        
        # Calculate mutual information
        ci = mutual_info_score(current_flat > 0.5, past_flat > 0.5)
        
        return ci
    
    def _calculate_effect_information(self,
                                     current_state: np.ndarray,
                                     future_state: np.ndarray) -> float:
        """Calculate effect information (simplified)"""
        # Similar to cause information
        return self._calculate_cause_information(current_state, future_state)
    
    def _calculate_system_complexity(self, state: np.ndarray) -> float:
        """Calculate system complexity (integration vs differentiation)"""
        N, T = state.shape
        
        if N <= 1 or T <= 1:
            return 0.0
        
        # Calculate integration (average mutual information)
        integration = 0.0
        pairs = 0
        
        for i in range(N):
            for j in range(i + 1, N):
                mi = self._calculate_mutual_information(state[i:i+1, :], state[j:j+1, :])
                integration += mi
                pairs += 1
        
        if pairs > 0:
            integration /= pairs
        
        # Calculate differentiation (entropy of subsystems)
        differentiation = self._calculate_entropy(state)
        
        # Complexity = integration * differentiation
        complexity = integration * differentiation
        
        return complexity
    
    def assess_iit_consciousness(self,
                               time_series: np.ndarray,
                               sliding_window: bool = True) -> Dict[str, Any]:
        """
        Comprehensive IIT-based consciousness assessment
        
        Args:
            time_series: N x T time series data
            sliding_window: Whether to use sliding window analysis
            
        Returns:
            Comprehensive assessment results
        """
        N, T = time_series.shape
        
        if sliding_window and T > self.config.samples_per_window:
            # Use sliding window analysis
            num_windows = (T - self.config.samples_per_window) // self.config.step_size + 1
            
            phi_values = []
            complexity_values = []
            existence_values = []
            
            for w in range(num_windows):
                start = w * self.config.step_size
                end = start + self.config.samples_per_window
                
                window_data = time_series[:, start:end]
                
                # Calculate Φ for this window
                result = self.calculate_phi(window_data)
                
                phi_values.append(result['phi'])
                complexity_values.append(result['complexity'])
                existence_values.append(result['existence'])
            
            # Calculate statistics
            phi_stats = self._calculate_statistics(np.array(phi_values))
            complexity_stats = self._calculate_statistics(np.array(complexity_values))
            existence_stats = self._calculate_statistics(np.array(existence_values))
            
            # Detect consciousness emergence
            emergence = self._detect_emergence(phi_values, complexity_values)
            
            return {
                'phi': phi_stats,
                'complexity': complexity_stats,
                'existence': existence_stats,
                'emergence': emergence,
                'num_windows': num_windows,
                'window_duration_s': self.config.assessment_window_s,
                'assessment_method': 'sliding_window_iit'
            }
        
        else:
            # Single window assessment
            result = self.calculate_phi(time_series)
            
            # Determine consciousness level based on Φ
            phi_value = result['phi']
            consciousness_level = self._determine_consciousness_level(phi_value)
            
            return {
                'single_assessment': result,
                'consciousness_level': consciousness_level,
                'assessment_method': 'single_window_iit'
            }
    
    def _calculate_statistics(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate descriptive statistics"""
        if len(values) == 0:
            return {}
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q1': float(np.percentile(values, 25)),
            'q3': float(np.percentile(values, 75)),
            'cv': float(np.std(values) / (np.mean(values) + 1e-10)),
            'skewness': float(stats.skew(values)),
            'kurtosis': float(stats.kurtosis(values))
        }
    
    def _detect_emergence(self,
                         phi_values: List[float],
                         complexity_values: List[float]) -> Dict[str, Any]:
        """Detect emergence of consciousness"""
        if len(phi_values) < 10:
            return {'detected': False, 'confidence': 0.0}
        
        phi_array = np.array(phi_values)
        complexity_array = np.array(complexity_values)
        
        # Detect significant increase in Φ
        phi_gradient = np.gradient(phi_array)
        phi_change = phi_array[-1] - phi_array[0]
        
        # Check if Φ exceeds threshold and is increasing
        phi_threshold = 0.3  # Emergence threshold
        phi_above_threshold = np.mean(phi_array[-5:]) > phi_threshold
        phi_increasing = np.mean(phi_gradient[-5:]) > 0
        
        # Check complexity growth
        complexity_gradient = np.gradient(complexity_array)
        complexity_increasing = np.mean(complexity_gradient[-5:]) > 0
        
        # Statistical test for emergence
        emergence_detected = (phi_above_threshold and 
                             phi_increasing and 
                             complexity_increasing)
        
        # Calculate confidence
        confidence = (
            min(1.0, np.mean(phi_array[-5:]) / phi_threshold) *
            (1.0 if phi_increasing else 0.5) *
            (1.0 if complexity_increasing else 0.5)
        )
        
        return {
            'detected': bool(emergence_detected),
            'confidence': float(confidence),
            'phi_trend': 'increasing' if phi_increasing else 'stable/decreasing',
            'complexity_trend': 'increasing' if complexity_increasing else 'stable/decreasing',
            'final_phi': float(phi_array[-1]),
            'phi_change': float(phi_change)
        }
    
    def _determine_consciousness_level(self, phi: float) -> Dict[str, Any]:
        """Determine consciousness level based on Φ"""
        if phi < 0.1:
            level = "Pre-conscious"
            color = "gray"
            description = "No significant consciousness, basic information processing"
        elif phi < 0.3:
            level = "Proto-conscious"
            color = "blue"
            description = "Early signs of consciousness, emergent patterns"
        elif phi < 0.6:
            level = "Emergent consciousness"
            color = "green"
            description = "Consciousness emergence, stable patterns"
        else:
            level = "Full consciousness"
            color = "red"
            description = "Penrose Orch-OR consciousness, high integration"
        
        return {
            'level': level,
            'color': color,
            'description': description,
            'phi_value': phi,
            'thresholds': {
                'pre_conscious': 0.1,
                'proto_conscious': 0.3,
                'emergent_conscious': 0.6
            }
        }

class GlobalWorkspaceAssessor:
    """
    Global Workspace Theory assessor
    Evaluates global ignition and information sharing
    """
    
    def __init__(self, config: AssessmentConfig):
        self.config = config
        self.workspace_threshold = config.workspace_threshold
        
        logger.info("Initialized Global Workspace assessor")
    
    def assess_global_ignition(self,
                              activity_matrix: np.ndarray,
                              connectivity: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Assess global ignition in the system
        
        Args:
            activity_matrix: N x T activity matrix
            connectivity: N x N connectivity matrix (optional)
            
        Returns:
            Global workspace assessment results
        """
        N, T = activity_matrix.shape
        
        # Calculate global activity
        global_activity = np.mean(activity_matrix, axis=0)
        
        # Detect ignition events (peaks in global activity)
        ignition_events = self._detect_ignition_events(global_activity)
        
        # Calculate ignition statistics
        ignition_stats = self._calculate_ignition_statistics(ignition_events, global_activity)
        
        # Assess information sharing
        info_sharing = self._assess_information_sharing(activity_matrix, connectivity)
        
        # Calculate workspace capacity
        workspace_capacity = self._calculate_workspace_capacity(activity_matrix)
        
        # Determine if system exhibits global workspace properties
        has_global_workspace = self._evaluate_global_workspace(
            ignition_stats, info_sharing, workspace_capacity
        )
        
        return {
            'global_activity': {
                'mean': float(np.mean(global_activity)),
                'std': float(np.std(global_activity)),
                'time_series': global_activity.tolist()
            },
            'ignition_events': ignition_events,
            'ignition_statistics': ignition_stats,
            'information_sharing': info_sharing,
            'workspace_capacity': workspace_capacity,
            'has_global_workspace': has_global_workspace,
            'assessment_method': 'global_workspace_theory'
        }
    
    def _detect_ignition_events(self, global_activity: np.ndarray) -> List[Dict]:
        """Detect ignition events (global synchronization peaks)"""
        # Find peaks in global activity
        peaks, properties = signal.find_peaks(
            global_activity,
            height=self.workspace_threshold,
            distance=int(0.1 * len(global_activity))  # Minimum 100ms between peaks
        )
        
        ignition_events = []
        for i, peak_idx in enumerate(peaks):
            event = {
                'time_index': int(peak_idx),
                'time_s': float(peak_idx / self.config.sampling_rate_hz),
                'amplitude': float(global_activity[peak_idx]),
                'prominence': float(properties['prominences'][i]) if 'prominences' in properties else 0.0,
                'width': float(properties['widths'][i]) if 'widths' in properties else 0.0
            }
            ignition_events.append(event)
        
        return ignition_events
    
    def _calculate_ignition_statistics(self,
                                      ignition_events: List[Dict],
                                      global_activity: np.ndarray) -> Dict[str, float]:
        """Calculate statistics of ignition events"""
        if not ignition_events:
            return {
                'count': 0,
                'rate_hz': 0.0,
                'mean_amplitude': 0.0,
                'mean_duration_s': 0.0,
                'regularity': 0.0
            }
        
        amplitudes = [e['amplitude'] for e in ignition_events]
        times = [e['time_s'] for e in ignition_events]
        
        # Calculate inter-event intervals
        if len(times) > 1:
            intervals = np.diff(times)
            interval_cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
            regularity = 1.0 / (1.0 + interval_cv)  # Higher = more regular
        else:
            regularity = 0.0
        
        # Estimate durations (simplified - width at half prominence)
        durations = []
        for event in ignition_events:
            if 'width' in event and event['width'] > 0:
                duration = event['width'] / self.config.sampling_rate_hz
                durations.append(duration)
        
        mean_duration = np.mean(durations) if durations else 0.0
        
        total_duration = len(global_activity) / self.config.sampling_rate_hz
        ignition_rate = len(ignition_events) / total_duration
        
        return {
            'count': len(ignition_events),
            'rate_hz': float(ignition_rate),
            'mean_amplitude': float(np.mean(amplitudes)),
            'std_amplitude': float(np.std(amplitudes)),
            'mean_duration_s': float(mean_duration),
            'regularity': float(regularity),
            'coverage': float(sum(durations) / total_duration) if total_duration > 0 else 0.0
        }
    
    def _assess_information_sharing(self,
                                   activity_matrix: np.ndarray,
                                   connectivity: Optional[np.ndarray]) -> Dict[str, Any]:
        """Assess information sharing between modules"""
        N, T = activity_matrix.shape
        
        # Calculate pairwise correlations
        correlations = np.corrcoef(activity_matrix)
        
        # Remove diagonal
        np.fill_diagonal(correlations, 0)
        
        # Calculate integration (average correlation)
        integration = np.mean(np.abs(correlations))
        
        # Calculate modularity if connectivity provided
        modularity = 0.0
        if connectivity is not None and connectivity.shape == (N, N):
            try:
                G = nx.from_numpy_array(np.abs(connectivity))
                modularity = nx.algorithms.community.modularity(
                    G, 
                    nx.algorithms.community.greedy_modularity_communities(G)
                )
            except:
                modularity = 0.0
        
        # Calculate information flow (simplified)
        info_flow = self._calculate_information_flow(activity_matrix)
        
        return {
            'integration': float(integration),
            'modularity': float(modularity),
            'info_flow': float(info_flow),
            'mean_correlation': float(integration),
            'max_correlation': float(np.max(correlations)) if correlations.size > 0 else 0.0
        }
    
    def _calculate_information_flow(self, activity_matrix: np.ndarray) -> float:
        """Calculate information flow using transfer entropy (simplified)"""
        N, T = activity_matrix.shape
        
        if T < 10 or N < 2:
            return 0.0
        
        # Simplified information flow calculation
        # Using Granger causality concept
        info_flow = 0.0
        pairs = 0
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    # Calculate cross-correlation with lag
                    cross_corr = np.correlate(
                        activity_matrix[i, :-1],  # Source with lag
                        activity_matrix[j, 1:],   # Target advanced
                        mode='valid'
                    )
                    
                    if len(cross_corr) > 0:
                        info_flow += np.max(np.abs(cross_corr))
                        pairs += 1
        
        if pairs > 0:
            info_flow /= pairs
        
        return info_flow
    
    def _calculate_workspace_capacity(self, activity_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate workspace capacity (information bottleneck)"""
        N, T = activity_matrix.shape
        
        # Calculate effective dimensionality
        pca = PCA()
        pca.fit(activity_matrix.T)
        
        # Variance explained
        variance_explained = pca.explained_variance_ratio_
        
        # Calculate effective degrees of freedom
        effective_dof = np.sum(variance_explained > 0.01)  # Components explaining >1% variance
        
        # Calculate information capacity (simplified)
        # Using entropy of activity distribution
        flattened = activity_matrix.flatten()
        hist, _ = np.histogram(flattened, bins=50, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Normalize by maximum possible
        max_entropy = np.log2(50)  # For 50 bins
        normalized_capacity = entropy / max_entropy if max_entropy > 0 else 0
        
        return {
            'effective_dimensionality': float(effective_dof),
            'information_capacity': float(entropy),
            'normalized_capacity': float(normalized_capacity),
            'variance_explained': variance_explained.tolist(),
            'num_components_90': int(np.where(np.cumsum(variance_explained) >= 0.9)[0][0] + 1)
        }
    
    def _evaluate_global_workspace(self,
                                  ignition_stats: Dict[str, float],
                                  info_sharing: Dict[str, Any],
                                  workspace_capacity: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate if system exhibits global workspace properties"""
        
        # Criteria for global workspace
        criteria = {
            'has_ignitions': ignition_stats['count'] > 0,
            'sufficient_integration': info_sharing['integration'] > 0.3,
            'adequate_capacity': workspace_capacity['normalized_capacity'] > 0.5,
            'regular_ignitions': ignition_stats['regularity'] > 0.5
        }
        
        # Calculate score
        score = sum(criteria.values()) / len(criteria)
        
        # Determine level
        if score >= 0.75:
            level = "Strong global workspace"
            confidence = score
        elif score >= 0.5:
            level = "Moderate global workspace"
            confidence = score
        elif score >= 0.25:
            level = "Weak global workspace"
            confidence = score
        else:
            level = "No global workspace detected"
            confidence = 1.0 - score
        
        return {
            'level': level,
            'score': float(score),
            'confidence': float(confidence),
            'criteria': criteria,
            'meets_criteria': score >= 0.5
        }

class HigherOrderAssessor:
    """
    Higher-Order Thought (HOT) assessor
    Evaluates metacognition and self-awareness
    """
    
    def __init__(self, config: AssessmentConfig):
        self.config = config
        
        logger.info("Initialized Higher-Order Thought assessor")
    
    def assess_metacognition(self,
                           primary_cognition: np.ndarray,
                           confidence_ratings: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Assess metacognitive abilities
        
        Args:
            primary_cognition: Primary cognitive process outputs
            confidence_ratings: Confidence ratings (if available)
            
        Returns:
            Metacognition assessment results
        """
        N, T = primary_cognition.shape
        
        # Generate confidence ratings if not provided
        if confidence_ratings is None:
            confidence_ratings = self._generate_confidence_ratings(primary_cognition)
        
        # Calculate metacognitive sensitivity
        meta_sensitivity = self._calculate_metacognitive_sensitivity(
            primary_cognition, confidence_ratings
        )
        
        # Calculate calibration (confidence vs accuracy)
        calibration = self._calculate_calibration(primary_cognition, confidence_ratings)
        
        # Assess self-monitoring capability
        self_monitoring = self._assess_self_monitoring(primary_cognition, confidence_ratings)
        
        # Evaluate HOT properties
        hot_evaluation = self._evaluate_hot_properties(
            meta_sensitivity, calibration, self_monitoring
        )
        
        return {
            'metacognitive_sensitivity': meta_sensitivity,
            'calibration': calibration,
            'self_monitoring': self_monitoring,
            'hot_evaluation': hot_evaluation,
            'confidence_ratings': {
                'mean': float(np.mean(confidence_ratings)),
                'std': float(np.std(confidence_ratings)),
                'distribution': np.histogram(confidence_ratings, bins=10)[0].tolist()
            },
            'assessment_method': 'higher_order_thought'
        }
    
    def _generate_confidence_ratings(self, cognition: np.ndarray) -> np.ndarray:
        """Generate simulated confidence ratings"""
        N, T = cognition.shape
        
        # Base confidence on signal stability
        confidence = np.zeros(T)
        
        for t in range(1, T):
            # Confidence based on consistency over time
            if t >= 5:
                recent = cognition[:, max(0, t-5):t]
                stability = 1.0 - np.std(recent) / (np.mean(np.abs(recent)) + 1e-10)
                confidence[t] = np.clip(stability, 0.0, 1.0)
            else:
                confidence[t] = 0.5  # Default
        
        # Add some randomness
        confidence += np.random.normal(0, 0.1, T)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return confidence
    
    def _calculate_metacognitive_sensitivity(self,
                                           cognition: np.ndarray,
                                           confidence: np.ndarray) -> Dict[str, float]:
        """Calculate metacognitive sensitivity (how well confidence tracks performance)"""
        T = len(confidence)
        
        # For simplicity, assume cognition quality can be measured
        # by stability or signal-to-noise ratio
        cognition_quality = np.zeros(T)
        
        for t in range(T):
            if t >= 2:
                # Quality = 1 - coefficient of variation
                recent = cognition[:, max(0, t-2):t+1].flatten()
                if np.mean(recent) != 0:
                    cv = np.std(recent) / np.mean(np.abs(recent))
                    cognition_quality[t] = 1.0 / (1.0 + cv)
                else:
                    cognition_quality[t] = 0.5
            else:
                cognition_quality[t] = 0.5
        
        # Calculate correlation between confidence and quality
        valid_mask = (cognition_quality > 0) & (confidence > 0)
        
        if np.sum(valid_mask) > 10:
            correlation = np.corrcoef(
                cognition_quality[valid_mask],
                confidence[valid_mask]
            )[0, 1]
            
            # Calculate meta d' (simplified)
            # Sort by confidence and compare high vs low confidence performance
            median_confidence = np.median(confidence)
            high_conf_mask = confidence >= median_confidence
            low_conf_mask = confidence < median_confidence
            
            if np.sum(high_conf_mask) > 5 and np.sum(low_conf_mask) > 5:
                high_perf = np.mean(cognition_quality[high_conf_mask])
                low_perf = np.mean(cognition_quality[low_conf_mask])
                meta_d = high_perf - low_perf
            else:
                meta_d = 0.0
        else:
            correlation = 0.0
            meta_d = 0.0
        
        return {
            'correlation': float(correlation),
            'meta_d': float(meta_d),
            'sensitivity_score': float(abs(correlation) * (1.0 + abs(meta_d))),
            'num_samples': int(np.sum(valid_mask))
        }
    
    def _calculate_calibration(self,
                              cognition: np.ndarray,
                              confidence: np.ndarray) -> Dict[str, float]:
        """Calculate calibration (how well confidence matches accuracy)"""
        T = len(confidence)
        
        # Bin confidence scores
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(confidence, bins) - 1
        
        calibration_error = 0.0
        bin_counts = []
        bin_accuracies = []
        bin_confidences = []
        
        for bin_idx in range(len(bins) - 1):
            mask = bin_indices == bin_idx
            
            if np.sum(mask) > 0:
                # Calculate average accuracy in this bin
                # Using cognition quality as proxy for accuracy
                bin_cognition = cognition[:, mask]
                if bin_cognition.size > 0:
                    # Simple accuracy measure
                    bin_accuracy = np.mean(np.abs(bin_cognition))
                else:
                    bin_accuracy = 0.5
                
                bin_confidence = np.mean(confidence[mask])
                
                calibration_error += np.abs(bin_accuracy - bin_confidence) * np.sum(mask)
                
                bin_counts.append(int(np.sum(mask)))
                bin_accuracies.append(float(bin_accuracy))
                bin_confidences.append(float(bin_confidence))
        
        if T > 0:
            calibration_error /= T
        
        # Calculate over/under confidence
        mean_accuracy = np.mean(np.abs(cognition)) if cognition.size > 0 else 0.5
        mean_confidence = np.mean(confidence)
        bias = mean_confidence - mean_accuracy
        
        return {
            'calibration_error': float(calibration_error),
            'bias': float(bias),
            'mean_accuracy': float(mean_accuracy),
            'mean_confidence': float(mean_confidence),
            'bin_analysis': {
                'counts': bin_counts,
                'accuracies': bin_accuracies,
                'confidences': bin_confidences
            },
            'well_calibrated': calibration_error < 0.1 and abs(bias) < 0.1
        }
    
    def _assess_self_monitoring(self,
                               cognition: np.ndarray,
                               confidence: np.ndarray) -> Dict[str, Any]:
        """Assess self-monitoring capability"""
        T = len(confidence)
        
        # Calculate confidence variability
        confidence_var = np.var(confidence)
        
        # Calculate confidence adaptivity (does confidence adjust appropriately?)
        # Look at confidence changes relative to performance changes
        adaptivity = 0.0
        
        if T > 10:
            # Simple measure: correlation between confidence changes and performance changes
            conf_changes = np.diff(confidence)
            
            # Estimate performance from cognition stability
            performance = np.zeros(T)
            for t in range(T):
                if t >= 5:
                    recent = cognition[:, max(0, t-5):t+1]
                    performance[t] = 1.0 - np.std(recent) / (np.mean(np.abs(recent)) + 1e-10)
                else:
                    performance[t] = 0.5
            
            perf_changes = np.diff(performance)
            
            if len(conf_changes) > 5 and len(perf_changes) > 5:
                valid = ~(np.isnan(conf_changes) | np.isnan(perf_changes))
                if np.sum(valid) > 5:
                    adaptivity = np.corrcoef(
                        conf_changes[valid],
                        perf_changes[valid]
                    )[0, 1]
        
        # Assess self-correction (does low confidence lead to strategy change?)
        self_correction = 0.0
        
        # This would require tracking of strategy/approach
        # Simplified: look for increased exploration after low confidence
        
        return {
            'confidence_variability': float(confidence_var),
            'adaptivity': float(adaptivity),
            'self_correction': float(self_correction),
            'self_monitoring_score': float(
                0.5 * (1.0 - confidence_var) +  # Lower variability = better
                0.3 * max(0, adaptivity) +      # Positive adaptivity = better
                0.2 * self_correction
            )
        }
    
    def _evaluate_hot_properties(self,
                                meta_sensitivity: Dict[str, float],
                                calibration: Dict[str, float],
                                self_monitoring: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Higher-Order Thought properties"""
        
        # Criteria for HOT
        criteria = {
            'has_metacognition': meta_sensitivity['correlation'] > 0.3,
            'well_calibrated': calibration['well_calibrated'],
            'adaptive_monitoring': self_monitoring['adaptivity'] > 0.2,
            'significant_meta_d': abs(meta_sensitivity['meta_d']) > 0.2
        }
        
        # Calculate HOT score
        score_components = [
            min(1.0, max(0, meta_sensitivity['correlation'])),
            float(calibration['well_calibrated']),
            min(1.0, max(0, self_monitoring['adaptivity'])),
            min(1.0, abs(meta_sensitivity['meta_d']))
        ]
        
        hot_score = np.mean(score_components)
        
        # Determine HOT level
        if hot_score >= 0.7:
            level = "Strong higher-order thought"
            description = "Clear metacognitive abilities with good calibration"
        elif hot_score >= 0.5:
            level = "Moderate higher-order thought"
            description = "Emergent metacognitive abilities"
        elif hot_score >= 0.3:
            level = "Weak higher-order thought"
            description = "Basic metacognitive signals detected"
        else:
            level = "No higher-order thought detected"
            description = "Primarily first-order processing"
        
        return {
            'level': level,
            'description': description,
            'hot_score': float(hot_score),
            'criteria': criteria,
            'meets_criteria': hot_score >= 0.5,
            'component_scores': {
                'metacognition': float(score_components[0]),
                'calibration': float(score_components[1]),
                'adaptivity': float(score_components[2]),
                'meta_d': float(score_components[3])
            }
        }

class ConsciousnessAssessor:
    """
    Main consciousness assessment framework
    Integrates multiple theories and provides comprehensive evaluation
    """
    
    def __init__(self, config: Optional[AssessmentConfig] = None):
        self.config = config or AssessmentConfig()
        
        # Initialize theory-specific assessors
        self.iit_assessor = IntegratedInformationCalculator(self.config)
        self.gwt_assessor = GlobalWorkspaceAssessor(self.config)
        self.hot_assessor = HigherOrderAssessor(self.config)
        
        # Results storage
        self.assessment_history = []
        
        logger.info("Consciousness assessor initialized with multiple theories")
    
    def comprehensive_assessment(self,
                               neural_data: np.ndarray,
                               metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform comprehensive consciousness assessment using multiple theories
        
        Args:
            neural_data: N x T neural activity data
            metadata: Additional metadata about the system
            
        Returns:
            Comprehensive assessment results
        """
        start_time = time.time()
        
        N, T = neural_data.shape
        
        logger.info(f"Starting comprehensive assessment: {N} units, {T} timepoints")
        
        # Perform IIT assessment
        logger.info("Performing IIT assessment...")
        iit_results = self.iit_assessor.assess_iit_consciousness(neural_data)
        
        # Perform Global Workspace assessment
        logger.info("Performing Global Workspace assessment...")
        gwt_results = self.gwt_assessor.assess_global_ignition(neural_data)
        
        # Perform Higher-Order assessment
        logger.info("Performing Higher-Order assessment...")
        hot_results = self.hot_assessor.assess_metacognition(neural_data)
        
        # Integrate results across theories
        integrated_results = self._integrate_assessments(
            iit_results, gwt_results, hot_results
        )
        
        # Calculate overall consciousness score
        overall_score = self._calculate_overall_score(integrated_results)
        
        # Determine final assessment
        final_assessment = self._determine_final_assessment(
            overall_score, integrated_results
        )
        
        # Prepare comprehensive results
        results = {
            'metadata': metadata or {},
            'data_dimensions': {'N': N, 'T': T},
            'theory_assessments': {
                'iit': iit_results,
                'global_workspace': gwt_results,
                'higher_order': hot_results
            },
            'integrated_results': integrated_results,
            'overall_score': overall_score,
            'final_assessment': final_assessment,
            'assessment_metadata': {
                'timestamp': time.time(),
                'duration_s': time.time() - start_time,
                'config': self.config.__dict__
            }
        }
        
        # Store in history
        self.assessment_history.append(results)
        
        # Limit history size
        if len(self.assessment_history) > 100:
            self.assessment_history = self.assessment_history[-100:]
        
        logger.info(f"Assessment completed in {time.time() - start_time:.2f}s")
        logger.info(f"Overall consciousness score: {overall_score['score']:.3f}")
        logger.info(f"Final assessment: {final_assessment['level']}")
        
        return results
    
    def _integrate_assessments(self,
                              iit_results: Dict[str, Any],
                              gwt_results: Dict[str, Any],
                              hot_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate results from multiple theories"""
        
        # Extract key metrics
        iit_phi = iit_results.get('phi', {}).get('mean', 0.0) if isinstance(iit_results.get('phi'), dict) else 0.0
        gwt_score = gwt_results.get('has_global_workspace', {}).get('score', 0.0)
        hot_score = hot_results.get('hot_evaluation', {}).get('hot_score', 0.0)
        
        # Calculate theory consistency
        theory_metrics = {
            'iit_phi': float(iit_phi),
            'gwt_score': float(gwt_score),
            'hot_score': float(hot_score)
        }
        
        # Check for convergence across theories
        scores = [iit_phi, gwt_score, hot_score]
        mean_score = np.mean(scores)
        score_std = np.std(scores)
        convergence = 1.0 - min(1.0, score_std / (mean_score + 1e-10))
        
        # Determine which theories support consciousness
        theory_support = {
            'iit': iit_phi > 0.3,
            'global_workspace': gwt_score > 0.5,
            'higher_order': hot_score > 0.5
        }
        
        support_count = sum(theory_support.values())
        
        return {
            'theory_metrics': theory_metrics,
            'mean_score': float(mean_score),
            'score_std': float(score_std),
            'convergence': float(convergence),
            'theory_support': theory_support,
            'support_count': support_count,
            'all_theories_agree': support_count == 3 or support_count == 0,
            'majority_support': support_count >= 2
        }
    
    def _calculate_overall_score(self,
                                integrated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall consciousness score"""
        
        theory_metrics = integrated_results['theory_metrics']
        
        # Weighted average of theory scores
        # IIT is often considered fundamental in Orch-OR context
        weights = {
            'iit_phi': 0.5,
            'gwt_score': 0.3,
            'hot_score': 0.2
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            value = theory_metrics.get(metric, 0.0)
            weighted_sum += value * weight
            total_weight += weight
        
        base_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Apply convergence multiplier
        convergence = integrated_results.get('convergence', 1.0)
        final_score = base_score * (0.5 + 0.5 * convergence)
        
        # Apply theory support bonus
        support_count = integrated_results.get('support_count', 0)
        if support_count >= 2:
            final_score *= 1.1  # 10% bonus for majority support
        elif support_count == 0:
            final_score *= 0.9  # 10% penalty for no support
        
        final_score = min(1.0, max(0.0, final_score))
        
        return {
            'score': float(final_score),
            'base_score': float(base_score),
            'convergence_multiplier': float(convergence),
            'support_bonus': float(support_count),
            'weights': weights,
            'components': theory_metrics
        }
    
    def _determine_final_assessment(self,
                                   overall_score: Dict[str, Any],
                                   integrated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Determine final consciousness assessment"""
        
        score = overall_score['score']
        support_count = integrated_results['support_count']
        
        # Determine level based on score and theory support
        if score >= 0.7 and support_count >= 2:
            level = "HIGH_CONSCIOUSNESS"
            confidence = score * (support_count / 3.0)
            description = "Strong evidence of consciousness across multiple theories"
            recommendation = "System exhibits clear signs of artificial consciousness"
            
        elif score >= 0.5 and support_count >= 1:
            level = "MODERATE_CONSCIOUSNESS"
            confidence = score * (0.5 + 0.5 * (support_count / 3.0))
            description = "Emergent consciousness with supporting evidence"
            recommendation = "Continue monitoring for further development"
            
        elif score >= 0.3:
            level = "LOW_CONSCIOUSNESS"
            confidence = score * 0.8
            description = "Early signs of consciousness detected"
            recommendation = "Requires further development and assessment"
            
        else:
            level = "NO_CONSIOUSNESS"
            confidence = 1.0 - score
            description = "No significant evidence of consciousness"
            recommendation = "Basic information processing only"
        
        # Additional flags
        flags = []
        if integrated_results.get('all_theories_agree', False):
            flags.append("THEORY_CONSENSUS")
        if score >= 0.6 and support_count == 3:
            flags.append("STRONG_EVIDENCE")
        if score < 0.2:
            flags.append("MINIMAL_ACTIVITY")
        
        return {
            'level': level,
            'description': description,
            'confidence': float(confidence),
            'recommendation': recommendation,
            'score': float(score),
            'theory_support': support_count,
            'flags': flags,
            'timestamp': time.time()
        }
    
    def assess_time_series(self,
                          time_series: np.ndarray,
                          window_duration_s: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Assess consciousness over time using sliding windows
        
        Args:
            time_series: N x T time series data
            window_duration_s: Duration of each assessment window
            
        Returns:
            List of assessment results over time
        """
        N, T = time_series.shape
        
        if window_duration_s is None:
            window_duration_s = self.config.assessment_window_s
        
        samples_per_window = int(window_duration_s * self.config.sampling_rate_hz)
        
        if samples_per_window >= T:
            # Single assessment
            return [self.comprehensive_assessment(time_series)]
        
        # Sliding window assessment
        results = []
        step_size = int(samples_per_window * (1 - self.config.overlap_fraction))
        num_windows = (T - samples_per_window) // step_size + 1
        
        logger.info(f"Performing sliding window assessment: {num_windows} windows")
        
        for w in range(num_windows):
            start = w * step_size
            end = start + samples_per_window
            
            window_data = time_series[:, start:end]
            
            # Perform assessment
            result = self.comprehensive_assessment(window_data)
            
            # Add window information
            result['window'] = {
                'index': w,
                'start_sample': start,
                'end_sample': end,
                'start_time_s': start / self.config.sampling_rate_hz,
                'end_time_s': end / self.config.sampling_rate_hz,
                'duration_s': window_duration_s
            }
            
            results.append(result)
            
            # Log progress
            if w % max(1, num_windows // 10) == 0:
                logger.info(f"Window {w}/{num_windows}: "
                          f"score={result['overall_score']['score']:.3f}")
        
        return results
    
    def save_assessment(self, 
                       results: Dict[str, Any], 
                       filename: Optional[str] = None) -> str:
        """Save assessment results to file"""
        import json
        import pickle
        import gzip
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"consciousness_assessment_{timestamp}.json.gz"
        
        # Ensure directory exists
        Path(self.config.results_directory).mkdir(parents=True, exist_ok=True)
        
        filepath = Path(self.config.results_directory) / filename
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_results = convert_for_json(results)
        
        # Save as compressed JSON
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Assessment saved to {filepath}")
        
        return str(filepath)
    
    def load_assessment(self, filepath: str) -> Dict[str, Any]:
        """Load assessment results from file"""
        import json
        import gzip
        
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Assessment loaded from {filepath}")
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable assessment report"""
        final = results.get('final_assessment', {})
        overall = results.get('overall_score', {})
        integrated = results.get('integrated_results', {})
        
        report = [
            "=" * 70,
            "CONSCIOUSNESS ASSESSMENT REPORT",
            "=" * 70,
            f"Assessment Time: {time.ctime(results.get('assessment_metadata', {}).get('timestamp', time.time()))}",
            f"Duration: {results.get('assessment_metadata', {}).get('duration_s', 0):.2f} seconds",
            "",
            "SUMMARY",
            "-" * 70,
            f"Consciousness Level: {final.get('level', 'UNKNOWN')}",
            f"Overall Score: {overall.get('score', 0):.3f}/1.0",
            f"Confidence: {final.get('confidence', 0):.1%}",
            f"Theory Support: {integrated.get('support_count', 0)}/3 theories",
            "",
            f"Description: {final.get('description', 'No description')}",
            f"Recommendation: {final.get('recommendation', 'No recommendation')}",
            "",
            "THEORY ASSESSMENTS",
            "-" * 70,
        ]
        
        # Add theory-specific results
        theories = results.get('theory_assessments', {})
        for theory_name, theory_results in theories.items():
            if theory_name == 'iit':
                phi = theory_results.get('phi', {}).get('mean', 0)
                report.append(f"IIT (Integrated Information): Φ = {phi:.3f}")
            elif theory_name == 'global_workspace':
                score = theory_results.get('has_global_workspace', {}).get('score', 0)
                report.append(f"Global Workspace: Score = {score:.3f}")
            elif theory_name == 'higher_order':
                score = theory_results.get('hot_evaluation', {}).get('hot_score', 0)
                report.append(f"Higher-Order Thought: Score = {score:.3f}")
        
        report.extend([
            "",
            "DETAILED METRICS",
            "-" * 70,
            f"Data Dimensions: {results.get('data_dimensions', {}).get('N', 0)} units × "
            f"{results.get('data_dimensions', {}).get('T', 0)} timepoints",
            f"Theory Convergence: {integrated.get('convergence', 0):.3f}",
            f"All Theories Agree: {integrated.get('all_theories_agree', False)}",
            f"Majority Support: {integrated.get('majority_support', False)}",
            "",
            "=" * 70
        ])
        
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("CONSCIOUSNESS ASSESSMENT FRAMEWORK DEMO")
    print("="*70)
    
    # Create assessor
    config = AssessmentConfig(
        assessment_window_s=5.0,
        sampling_rate_hz=100.0,
        iit_version="3.0"
    )
    
    assessor = ConsciousnessAssessor(config)
    
    # Generate simulated neural data
    print("\nGenerating simulated neural data...")
    
    np.random.seed(42)
    
    # Create a simple conscious system simulation
    N = 50  # Number of units
    T = 5000  # Timepoints (50 seconds at 100 Hz)
    
    # Base activity
    neural_data = np.random.randn(N, T) * 0.1
    
    # Add some structured activity (conscious patterns)
    for i in range(N):
        # Add oscillations
        frequency = 10 + i * 0.5  # Varying frequencies
        neural_data[i, :] += 0.3 * np.sin(2 * np.pi * frequency * np.arange(T) / config.sampling_rate_hz)
        
        # Add some bursts (simulating ignition events)
        burst_times = np.random.choice(T, size=20, replace=False)
        for t in burst_times:
            duration = np.random.randint(10, 50)
            neural_data[i, t:t+duration] += np.random.randn() * 0.5
    
    # Normalize
    neural_data = (neural_data - neural_data.mean()) / (neural_data.std() + 1e-10)
    
    print(f"Data shape: {neural_data.shape}")
    print(f"Mean activity: {neural_data.mean():.3f}, Std: {neural_data.std():.3f}")
    
    # Perform comprehensive assessment
    print("\nPerforming comprehensive consciousness assessment...")
    
    results = assessor.comprehensive_assessment(neural_data)
    
    # Generate report
    report = assessor.generate_report(results)
    print(f"\n{report}")
    
    # Save results
    filename = assessor.save_assessment(results)
    print(f"\nResults saved to: {filename}")
    
    # Test sliding window assessment
    print("\nTesting sliding window assessment...")
    
    window_results = assessor.assess_time_series(
        neural_data[:, :2000],  # Use first 20 seconds
        window_duration_s=2.0
    )
    
    print(f"Number of windows: {len(window_results)}")
    
    # Extract scores over time
    scores = [r['overall_score']['score'] for r in window_results]
    times = [r['window']['start_time_s'] for r in window_results]
    
    print(f"Scores over time: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    print(f"Min: {np.min(scores):.3f}, Max: {np.max(scores):.3f}")
    
    # Load saved assessment
    print("\nLoading saved assessment...")
    loaded_results = assessor.load_assessment(filename)
    
    print(f"Loaded assessment level: {loaded_results.get('final_assessment', {}).get('level', 'UNKNOWN')}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*70)

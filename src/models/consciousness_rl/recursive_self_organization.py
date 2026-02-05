"""Recursive self-organization for emergent consciousness.

Implements emergence-driven self-organization where the system
autonomously restructures itself based on detected emergence patterns.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class EmergenceType(Enum):
    """Types of emergence events."""
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    BEHAVIORAL = "behavioral"
    PHENOMENAL = "phenomenal"


@dataclass
class EmergenceEvent:
    """Represents an emergence event."""
    timestamp: float
    emergence_type: EmergenceType
    location: str  # Layer/module identifier
    strength: float
    description: str
    reorganization_trigger: bool = False


@dataclass
class SelfOrganizationConfig:
    """Configuration for self-organization."""
    # Emergence detection
    emergence_threshold: float = 0.7
    detection_window: int = 50
    pattern_memory: int = 1000
    
    # Reorganization
    reorganization_rate: float = 0.01
    max_reorganizations: int = 10
    
    # Pattern recognition
    pattern_similarity_threshold: float = 0.8
    novelty_threshold: float = 0.5
    
    # Recursive depth
    max_recursion_depth: int = 5
    recursion_decay: float = 0.9
    
    # Constraints
    min_module_size: int = 10
    max_module_size: int = 500
    sparsity_target: float = 0.2
    
    # Simulation
    dt: float = 1.0
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)


class RecursiveSelfOrganization:
    """Recursive self-organization system.
    
    Implements autonomous self-organization based on:
    - Emergence detection (structural, functional, behavioral)
    - Pattern recognition and novelty detection
    - Recursive reorganization at multiple scales
    - Emergence-driven structural changes
    
    Example:
        >>> config = SelfOrganizationConfig()
        >>> rso = RecursiveSelfOrganization(config)
        >>> emergence = rso.detect_emergence(activity_patterns)
        >>> if emergence.reorganization_trigger:
        ...     rso.reorganize()
    """
    
    def __init__(self, config: Optional[SelfOrganizationConfig] = None) -> None:
        """Initialize self-organization system.
        
        Args:
            config: Configuration parameters.
        """
        self.config = config or SelfOrganizationConfig()
        
        # Emergence detection
        self._pattern_memory: deque = deque(maxlen=self.config.pattern_memory)
        self._emergence_history: List[EmergenceEvent] = []
        self._novelty_scores: List[float] = []
        
        # Self-organization state
        self._modules: Dict[str, Dict[str, Any]] = {}
        self._module_hierarchy: Dict[str, List[str]] = {}
        self._recursion_depth = 0
        
        # Structural state
        self._connectivity_patterns: Dict[str, np.ndarray] = {}
        self._activity_patterns: Dict[str, np.ndarray] = {}
        
        logger.info("Initialized RecursiveSelfOrganization")
    
    def register_module(self, module_id: str, size: int,
                       parent: Optional[str] = None) -> None:
        """Register a module for organization.
        
        Args:
            module_id: Unique identifier for the module.
            size: Number of units in the module.
            parent: Parent module identifier.
        """
        self._modules[module_id] = {
            'size': size,
            'activity': np.zeros(size),
            'connectivity': np.random.rand(size, size) * 0.1,
            'organization_level': 0,
            'emergence_count': 0
        }
        
        if parent:
            if parent not in self._module_hierarchy:
                self._module_hierarchy[parent] = []
            self._module_hierarchy[parent].append(module_id)
        else:
            self._module_hierarchy[module_id] = []
        
        logger.info(f"Registered module: {module_id} (size={size})")
    
    def update_module_activity(self, module_id: str, activity: np.ndarray) -> None:
        """Update activity for a module.
        
        Args:
            module_id: Module identifier.
            activity: Activity vector.
        """
        if module_id in self._modules:
            self._modules[module_id]['activity'] = activity.copy()
            self._activity_patterns[module_id] = activity.copy()
    
    def update_module_connectivity(self, module_id: str, 
                                  connectivity: np.ndarray) -> None:
        """Update connectivity for a module.
        
        Args:
            module_id: Module identifier.
            connectivity: Connectivity matrix.
        """
        if module_id in self._modules:
            self._modules[module_id]['connectivity'] = connectivity.copy()
            self._connectivity_patterns[module_id] = connectivity.copy()
    
    def detect_emergence(self, module_id: Optional[str] = None) -> Optional[EmergenceEvent]:
        """Detect emergence in system.
        
        Args:
            module_id: Specific module to check, or None for all.
            
        Returns:
            EmergenceEvent if emergence detected, None otherwise.
        """
        modules_to_check = [module_id] if module_id else list(self._modules.keys())
        
        for mod_id in modules_to_check:
            emergence = self._detect_module_emergence(mod_id)
            if emergence:
                return emergence
        
        return None
    
    def _detect_module_emergence(self, module_id: str) -> Optional[EmergenceEvent]:
        """Detect emergence in a specific module."""
        if module_id not in self._modules:
            return None
        
        module = self._modules[module_id]
        activity = module['activity']
        
        # Detect structural emergence (connectivity patterns)
        structural_emergence = self._detect_structural_emergence(module_id)
        
        # Detect functional emergence (activity patterns)
        functional_emergence = self._detect_functional_emergence(module_id)
        
        # Combine emergence scores
        emergence_strength = max(structural_emergence, functional_emergence)
        
        if emergence_strength < self.config.emergence_threshold:
            return None
        
        # Determine emergence type
        if structural_emergence > functional_emergence:
            emergence_type = EmergenceType.STRUCTURAL
            description = f"Emergent connectivity pattern in {module_id}"
        else:
            emergence_type = EmergenceType.FUNCTIONAL
            description = f"Emergent activity pattern in {module_id}"
        
        # Create emergence event
        event = EmergenceEvent(
            timestamp=0.0,  # Will be set by simulation
            emergence_type=emergence_type,
            location=module_id,
            strength=emergence_strength,
            description=description,
            reorganization_trigger=True
        )
        
        self._emergence_history.append(event)
        module['emergence_count'] += 1
        
        logger.info(f"Emergence detected in {module_id}: {emergence_type.value} "
                   f"(strength={emergence_strength:.3f})")
        
        return event
    
    def _detect_structural_emergence(self, module_id: str) -> float:
        """Detect structural emergence from connectivity patterns."""
        if module_id not in self._connectivity_patterns:
            return 0.0
        
        connectivity = self._connectivity_patterns[module_id]
        
        # Calculate network statistics
        degree_centrality = np.sum(connectivity, axis=1)
        clustering_coefficient = self._calculate_clustering(connectivity)
        
        # Emergence: deviation from random network
        random_clustering = np.mean(degree_centrality) / connectivity.size
        emergence_score = abs(clustering_coefficient - random_clustering) / \
                         (random_clustering + 1e-10)
        
        # Normalize to [0, 1]
        emergence_score = np.tanh(emergence_score)
        
        return emergence_score
    
    def _detect_functional_emergence(self, module_id: str) -> float:
        """Detect functional emergence from activity patterns."""
        if module_id not in self._modules:
            return 0.0
        
        activity = self._modules[module_id]['activity']
        
        # Calculate entropy
        activity_prob = activity / (np.sum(activity) + 1e-10)
        entropy = -np.sum(activity_prob * np.log2(activity_prob + 1e-10))
        max_entropy = np.log2(len(activity))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Calculate synchrony
        synchrony = np.std(activity) / (np.mean(activity) + 1e-10)
        
        # Emergence: high entropy with some synchrony
        emergence_score = normalized_entropy * (1 - np.exp(-synchrony))
        
        return emergence_score
    
    def _calculate_clustering(self, adjacency: np.ndarray) -> float:
        """Calculate clustering coefficient."""
        n = adjacency.shape[0]
        if n < 3:
            return 0.0
        
        # Simplified clustering: fraction of triangles
        triangles = 0
        possible_triangles = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if adjacency[i, j] and adjacency[j, k] and adjacency[k, i]:
                        triangles += 1
                    if adjacency[i, j] + adjacency[j, k] + adjacency[k, i] > 0:
                        possible_triangles += 1
        
        return triangles / possible_triangles if possible_triangles > 0 else 0.0
    
    def detect_novelty(self, pattern: np.ndarray) -> float:
        """Detect novelty of a pattern.
        
        Args:
            pattern: Pattern to evaluate.
            
        Returns:
            Novelty score (0 to 1).
        """
        if len(self._pattern_memory) == 0:
            self._pattern_memory.append(pattern)
            return 1.0
        
        # Compare with stored patterns
        similarities = []
        for stored_pattern in self._pattern_memory:
            # Normalized cross-correlation
            similarity = np.corrcoef(pattern.flatten(), 
                                   stored_pattern.flatten())[0, 1]
            similarities.append(abs(similarity))
        
        max_similarity = max(similarities) if similarities else 0
        novelty = 1.0 - max_similarity
        
        # Store if novel enough
        if novelty > self.config.novelty_threshold:
            self._pattern_memory.append(pattern)
        
        self._novelty_scores.append(novelty)
        
        return novelty
    
    def reorganize(self, trigger_event: Optional[EmergenceEvent] = None) -> bool:
        """Perform self-organization/reorganization.
        
        Args:
            trigger_event: Emergence event that triggered reorganization.
            
        Returns:
            True if reorganization occurred, False otherwise.
        """
        if self._recursion_depth >= self.config.max_recursion_depth:
            logger.warning("Max recursion depth reached")
            return False
        
        if np.random.random() > self.config.reorganization_rate:
            return False
        
        self._recursion_depth += 1
        
        try:
            # Select module to reorganize
            if trigger_event:
                module_id = trigger_event.location
            else:
                module_id = np.random.choice(list(self._modules.keys()))
            
            if module_id not in self._modules:
                return False
            
            # Perform reorganization
            success = self._reorganize_module(module_id)
            
            if success:
                logger.info(f"Reorganized module: {module_id} "
                           f"(depth={self._recursion_depth})")
                
                # Recursive reorganization of sub-modules
                if module_id in self._module_hierarchy:
                    for sub_module in self._module_hierarchy[module_id]:
                        self.reorganize()
            
            return success
        
        finally:
            self._recursion_depth -= 1
    
    def _reorganize_module(self, module_id: str) -> bool:
        """Reorganize a specific module."""
        module = self._modules[module_id]
        connectivity = module['connectivity']
        
        # Reorganization strategy: strengthen strong connections,
        # prune weak connections
        threshold = np.percentile(np.abs(connectivity), 
                                 (1 - self.config.sparsity_target) * 100)
        
        # Prune weak connections
        mask = np.abs(connectivity) > threshold
        new_connectivity = connectivity * mask
        
        # Strengthen remaining connections
        new_connectivity = new_connectivity * 1.1
        
        # Apply constraints
        new_connectivity = np.clip(new_connectivity, -1.0, 1.0)
        
        # Update module
        module['connectivity'] = new_connectivity
        module['organization_level'] += 1
        
        # Update connectivity pattern
        self._connectivity_patterns[module_id] = new_connectivity
        
        return True
    
    def split_module(self, module_id: str, split_ratio: float = 0.5) -> bool:
        """Split a module into two sub-modules.
        
        Args:
            module_id: Module to split.
            split_ratio: Ratio for first sub-module.
            
        Returns:
            True if split successful, False otherwise.
        """
        if module_id not in self._modules:
            return False
        
        module = self._modules[module_id]
        original_size = module['size']
        
        if original_size < self.config.min_module_size * 2:
            return False
        
        # Create sub-modules
        size_1 = int(original_size * split_ratio)
        size_2 = original_size - size_1
        
        sub_id_1 = f"{module_id}_sub1"
        sub_id_2 = f"{module_id}_sub2"
        
        # Register sub-modules
        self.register_module(sub_id_1, size_1, parent=module_id)
        self.register_module(sub_id_2, size_2, parent=module_id)
        
        # Distribute activity and connectivity
        activity_1 = module['activity'][:size_1]
        activity_2 = module['activity'][size_1:]
        
        conn = module['connectivity']
        connectivity_1 = conn[:size_1, :size_1]
        connectivity_2 = conn[size_1:, size_1:]
        
        self.update_module_activity(sub_id_1, activity_1)
        self.update_module_activity(sub_id_2, activity_2)
        self.update_module_connectivity(sub_id_1, connectivity_1)
        self.update_module_connectivity(sub_id_2, connectivity_2)
        
        # Update hierarchy
        self._module_hierarchy[module_id] = [sub_id_1, sub_id_2]
        
        logger.info(f"Split module {module_id} into {sub_id_1} and {sub_id_2}")
        
        return True
    
    def merge_modules(self, module_id_1: str, module_id_2: str) -> bool:
        """Merge two modules.
        
        Args:
            module_id_1: First module to merge.
            module_id_2: Second module to merge.
            
        Returns:
            True if merge successful, False otherwise.
        """
        if module_id_1 not in self._modules or module_id_2 not in self._modules:
            return False
        
        module_1 = self._modules[module_id_1]
        module_2 = self._modules[module_id_2]
        
        # Check size constraint
        new_size = module_1['size'] + module_2['size']
        if new_size > self.config.max_module_size:
            return False
        
        # Create merged module
        merged_id = f"{module_id_1}_{module_id_2}_merged"
        self.register_module(merged_id, new_size)
        
        # Merge activity
        merged_activity = np.concatenate([
            module_1['activity'],
            module_2['activity']
        ])
        self.update_module_activity(merged_id, merged_activity)
        
        # Merge connectivity
        conn_1 = module_1['connectivity']
        conn_2 = module_2['connectivity']
        merged_connectivity = np.zeros((new_size, new_size))
        
        size_1 = module_1['size']
        merged_connectivity[:size_1, :size_1] = conn_1
        merged_connectivity[size_1:, size_1:] = conn_2
        
        # Add inter-module connections
        inter_connections = np.random.rand(size_1, new_size - size_1) * 0.1
        merged_connectivity[:size_1, size_1:] = inter_connections
        merged_connectivity[size_1:, :size_1] = inter_connections.T
        
        self.update_module_connectivity(merged_id, merged_connectivity)
        
        logger.info(f"Merged modules {module_id_1} and {module_id_2} into {merged_id}")
        
        return True
    
    def get_emergence_summary(self) -> Dict[str, Any]:
        """Get summary of emergence events.
        
        Returns:
            Dictionary with emergence statistics.
        """
        if not self._emergence_history:
            return {
                'total_emergence': 0,
                'emergence_by_type': {},
                'mean_strength': 0.0,
                'most_active_module': None
            }
        
        emergence_by_type = {}
        for event in self._emergence_history:
            etype = event.emergence_type.value
            emergence_by_type[etype] = emergence_by_type.get(etype, 0) + 1
        
        module_counts = {}
        for event in self._emergence_history:
            module_counts[event.location] = module_counts.get(event.location, 0) + 1
        
        most_active = max(module_counts, key=module_counts.get) if module_counts else None
        
        return {
            'total_emergence': len(self._emergence_history),
            'emergence_by_type': emergence_by_type,
            'mean_strength': float(np.mean([e.strength for e in self._emergence_history])),
            'most_active_module': most_active,
            'mean_novelty': float(np.mean(self._novelty_scores)) if self._novelty_scores else 0.0
        }
    
    def get_module_structure(self) -> Dict[str, Any]:
        """Get current module structure.
        
        Returns:
            Dictionary with module hierarchy and statistics.
        """
        structure = {
            'modules': {},
            'hierarchy': self._module_hierarchy.copy(),
            'total_modules': len(self._modules)
        }
        
        for module_id, module in self._modules.items():
            structure['modules'][module_id] = {
                'size': module['size'],
                'organization_level': module['organization_level'],
                'emergence_count': module['emergence_count'],
                'has_children': module_id in self._module_hierarchy and len(self._module_hierarchy[module_id]) > 0
            }
        
        return structure
    
    def reset(self) -> None:
        """Reset self-organization state."""
        self._pattern_memory.clear()
        self._emergence_history = []
        self._novelty_scores = []
        self._recursion_depth = 0
        
        # Reset module states
        for module in self._modules.values():
            module['activity'] = np.zeros(module['size'])
            module['organization_level'] = 0
            module['emergence_count'] = 0
        
        logger.debug("Reset RecursiveSelfOrganization")
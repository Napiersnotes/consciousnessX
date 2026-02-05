"""Global Workspace Theory implementation for consciousness.

Implements Baars-style global workspace with broadcasting mechanism
for conscious information integration and access across modules.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class WorkspaceModuleType(Enum):
    """Types of modules in the global workspace."""
    PERCEPTUAL = "perceptual"
    ATTENTIONAL = "attentional"
    MEMORY = "memory"
    EXECUTIVE = "executive"
    MOTOR = "motor"
    LANGUAGE = "language"
    EMOTIONAL = "emotional"


@dataclass
class WorkspaceBroadcast:
    """Information broadcast in the global workspace."""
    timestamp: float
    source_module: str
    content: np.ndarray
    priority: float
    recipients: List[str]
    broadcast_id: int
    
    def get_strength(self) -> float:
        """Get broadcast strength."""
        return float(np.linalg.norm(self.content) * self.priority)


@dataclass
class GlobalWorkspaceConfig:
    """Configuration for global workspace."""
    # Workspace architecture
    num_modules: int = 10
    workspace_size: int = 100
    broadcast_capacity: int = 50
    
    # Broadcasting
    broadcast_threshold: float = 0.5
    broadcast_decay: float = 0.1
    priority_weight: float = 0.3
    
    # Attention
    attention_capacity: float = 0.7
    attention_decay: float = 0.05
    
    # Consciousness
    ignition_threshold: float = 0.8
    consciousness_duration: float = 100.0  # ms
    
    # Connectivity
    default_connectivity: float = 0.2
    learning_rate: float = 0.01
    
    # Simulation
    dt: float = 1.0
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)


class GlobalWorkspaceTheory:
    """Global Workspace Theory implementation.
    
    Implements the Baars global workspace model:
    - Multiple specialized modules compete for global access
    - Winning module broadcasts to all other modules
    - Broadcasting creates conscious experience
    - Modules can cooperate through the workspace
    
    Example:
        >>> config = GlobalWorkspaceConfig()
        >>> gwt = GlobalWorkspaceTheory(config)
        >>> gwt.register_module("visual", WorkspaceModuleType.PERCEPTUAL, 50)
        >>> gwt.set_module_activity("visual", activity_data)
        >>> broadcast = gwt.broadcast_conscious()
    """
    
    def __init__(self, config: Optional[GlobalWorkspaceConfig] = None) -> None:
        """Initialize global workspace.
        
        Args:
            config: Workspace configuration.
        """
        self.config = config or GlobalWorkspaceConfig()
        
        # Module registry
        self._modules: Dict[str, Dict[str, Any]] = {}
        self._module_types: Dict[str, WorkspaceModuleType] = {}
        self._connectivity: np.ndarray = np.zeros((0, 0))
        
        # Workspace state
        self._workspace_content = np.zeros(self.config.workspace_size)
        self._broadcast_history: List[WorkspaceBroadcast] = []
        self._broadcast_counter = 0
        
        # Attention state
        self._attention_distribution = np.zeros(self.config.num_modules)
        self._conscious_state = False
        self._conscious_timer = 0.0
        
        # Metrics
        self._broadcast_counts: Dict[str, int] = {}
        self._cooperation_scores: Dict[Tuple[str, str], float] = {}
        
        logger.info("Initialized GlobalWorkspaceTheory")
    
    def register_module(self, module_id: str, module_type: WorkspaceModuleType,
                       size: int) -> None:
        """Register a module in the workspace.
        
        Args:
            module_id: Unique identifier for the module.
            module_type: Type of module.
            size: Size of module's representation.
        """
        self._modules[module_id] = {
            'size': size,
            'activity': np.zeros(size),
            'output': np.zeros(self.config.workspace_size),
            'input_buffer': np.zeros(self.config.workspace_size),
            'activation_level': 0.0
        }
        
        self._module_types[module_id] = module_type
        self._broadcast_counts[module_id] = 0
        
        # Update connectivity matrix
        self._update_connectivity_matrix()
        
        logger.info(f"Registered module: {module_id} ({module_type.value}, size={size})")
    
    def _update_connectivity_matrix(self) -> None:
        """Update connectivity matrix for all modules."""
        n = len(self._modules)
        self._connectivity = np.random.rand(n, n) * self.config.default_connectivity
        
        # Remove self-connections
        np.fill_diagonal(self._connectivity, 0)
    
    def set_module_activity(self, module_id: str, activity: np.ndarray) -> None:
        """Set activity for a module.
        
        Args:
            module_id: Module identifier.
            activity: Activity vector.
        """
        if module_id in self._modules:
            self._modules[module_id]['activity'] = activity.copy()
            self._modules[module_id]['activation_level'] = np.linalg.norm(activity)
    
    def update_module(self, module_id: str) -> np.ndarray:
        """Update a module's state based on workspace input.
        
        Args:
            module_id: Module identifier.
            
        Returns:
            Module output.
        """
        if module_id not in self._modules:
            return np.zeros(self.config.workspace_size)
        
        module = self._modules[module_id]
        
        # Process input from workspace
        input_data = module['input_buffer']
        
        # Simple transformation (would be module-specific in full implementation)
        output = np.tanh(input_data + module['activity'])
        
        # Resize if needed
        if len(output) != self.config.workspace_size:
            output = np.interp(
                np.linspace(0, 1, self.config.workspace_size),
                np.linspace(0, 1, len(output)),
                output
            )
        
        module['output'] = output
        
        # Clear input buffer
        module['input_buffer'] = np.zeros(self.config.workspace_size)
        
        return output
    
    def compute_competition(self) -> Dict[str, float]:
        """Compute competition for global access.
        
        Returns:
            Dictionary mapping module IDs to competition scores.
        """
        scores = {}
        
        for module_id, module in self._modules.items():
            # Base score from activation
            activation = module['activation_level']
            
            # Attention boost
            module_idx = list(self._modules.keys()).index(module_id)
            attention_boost = self._attention_distribution[module_idx]
            
            # Novelty bonus (based on recent broadcasts)
            recent_broadcasts = [b for b in self._broadcast_history[-10:] 
                               if b.source_module == module_id]
            novelty = 1.0 / (1.0 + len(recent_broadcasts))
            
            # Combined score
            score = (activation + attention_boost * self.config.priority_weight + 
                    novelty * 0.1)
            
            scores[module_id] = score
        
        return scores
    
    def broadcast_conscious(self) -> Optional[WorkspaceBroadcast]:
        """Broadcast conscious content to workspace.
        
        Returns:
            WorkspaceBroadcast if broadcast occurred, None otherwise.
        """
        # Compute competition
        scores = self.compute_competition()
        
        if not scores:
            return None
        
        # Find winning module
        winner = max(scores, key=scores.get)
        winner_score = scores[winner]
        
        # Check threshold
        if winner_score < self.config.broadcast_threshold:
            return None
        
        # Create broadcast
        module = self._modules[winner]
        content = module['output']
        
        # Get recipients (all modules except source)
        recipients = [mid for mid in self._modules.keys() if mid != winner]
        
        broadcast = WorkspaceBroadcast(
            timestamp=0.0,  # Will be set by simulation
            source_module=winner,
            content=content,
            priority=winner_score,
            recipients=recipients,
            broadcast_id=self._broadcast_counter
        )
        
        self._broadcast_counter += 1
        
        # Update workspace content
        self._workspace_content = content.copy()
        
        # Record broadcast
        self._broadcast_history.append(broadcast)
        if len(self._broadcast_history) > self.config.broadcast_capacity:
            self._broadcast_history.pop(0)
        
        # Update module count
        self._broadcast_counts[winner] += 1
        
        # Trigger conscious state
        if winner_score > self.config.ignition_threshold:
            self._conscious_state = True
            self._conscious_timer = self.config.consciousness_duration
        
        # Distribute to recipients
        for recipient in recipients:
            if recipient in self._modules:
                self._modules[recipient]['input_buffer'] += content * 0.5
        
        logger.info(f"Broadcast from {winner}: score={winner_score:.3f}, "
                   f"recipients={len(recipients)}")
        
        return broadcast
    
    def update_attention(self) -> None:
        """Update attention distribution."""
        # Decay attention
        self._attention_distribution *= (1 - self.config.attention_decay)
        
        # Redistribute based on current activations
        total_activation = sum(m['activation_level'] for m in self._modules.values())
        
        if total_activation > 0:
            for i, module_id in enumerate(self._modules.keys()):
                activation = self._modules[module_id]['activation_level']
                new_attention = (activation / total_activation) * self.config.attention_capacity
                
                # Mix with existing attention
                self._attention_distribution[i] = (
                    0.7 * self._attention_distribution[i] + 0.3 * new_attention
                )
        
        # Normalize
        total = np.sum(self._attention_distribution)
        if total > 0:
            self._attention_distribution = self._attention_distribution / total
    
    def step(self) -> Optional[WorkspaceBroadcast]:
        """Advance simulation by one time step.
        
        Returns:
            WorkspaceBroadcast if broadcast occurred, None otherwise.
        """
        # Update all modules
        for module_id in self._modules:
            self.update_module(module_id)
        
        # Update attention
        self.update_attention()
        
        # Check for conscious broadcast
        broadcast = self.broadcast_conscious()
        
        # Update conscious timer
        if self._conscious_state:
            self._conscious_timer -= self.config.dt
            if self._conscious_timer <= 0:
                self._conscious_state = False
        
        # Decay workspace content
        self._workspace_content *= (1 - self.config.broadcast_decay)
        
        return broadcast
    
    def get_workspace_state(self) -> Dict[str, Any]:
        """Get current workspace state.
        
        Returns:
            Dictionary with workspace state variables.
        """
        return {
            'workspace_content': self._workspace_content.copy(),
            'conscious_state': self._conscious_state,
            'conscious_timer': self._conscious_timer,
            'attention_distribution': self._attention_distribution.copy(),
            'num_broadcasts': len(self._broadcast_history)
        }
    
    def get_module_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all modules.
        
        Returns:
            Dictionary mapping module IDs to their states.
        """
        module_states = {}
        
        for module_id, module in self._modules.items():
            module_states[module_id] = {
                'type': self._module_types[module_id].value,
                'activation_level': module['activation_level'],
                'broadcast_count': self._broadcast_counts[module_id]
            }
        
        return module_states
    
    def get_broadcast_history(self, n: int = 10) -> List[WorkspaceBroadcast]:
        """Get recent broadcast history.
        
        Args:
            n: Number of recent broadcasts.
            
        Returns:
            List of recent broadcasts.
        """
        return self._broadcast_history[-n:]
    
    def get_cooperation_matrix(self) -> np.ndarray:
        """Get cooperation matrix between modules.
        
        Returns:
            NxN matrix of cooperation scores.
        """
        n = len(self._modules)
        cooperation = np.zeros((n, n))
        
        module_ids = list(self._modules.keys())
        
        for i, mid1 in enumerate(module_ids):
            for j, mid2 in enumerate(module_ids):
                if i == j:
                    continue
                
                # Cooperation based on broadcast frequency
                broadcasts_from_i = self._broadcast_counts.get(mid1, 0)
                broadcasts_to_j = sum(1 for b in self._broadcast_history 
                                    if b.source_module == mid1 and mid2 in b.recipients)
                
                if broadcasts_from_i > 0:
                    cooperation[i, j] = broadcasts_to_j / broadcasts_from_i
        
        return cooperation
    
    def get_consciousness_metrics(self) -> Dict[str, float]:
        """Get consciousness-related metrics.
        
        Returns:
            Dictionary with consciousness metrics.
        """
        # Integration: mean connectivity
        integration = np.mean(self._connectivity)
        
        # Information sharing: broadcast diversity
        broadcast_sources = set(b.source_module for b in self._broadcast_history)
        diversity = len(broadcast_sources) / len(self._modules) if self._modules else 0
        
        # Global access: mean attention entropy
        if np.sum(self._attention_distribution) > 0:
            attention_entropy = -np.sum(
                self._attention_distribution * 
                np.log2(self._attention_distribution + 1e-10)
            )
            max_entropy = np.log2(len(self._modules))
            normalized_entropy = attention_entropy / max_entropy if max_entropy > 0 else 0
        else:
            normalized_entropy = 0
        
        return {
            'integration': float(integration),
            'diversity': float(diversity),
            'global_access': float(normalized_entropy),
            'conscious_state': float(self._conscious_state),
            'conscious_duration': float(self._conscious_timer)
        }
    
    def reset(self) -> None:
        """Reset workspace state."""
        self._workspace_content = np.zeros(self.config.workspace_size)
        self._broadcast_history = []
        self._broadcast_counter = 0
        self._attention_distribution = np.zeros(self.config.num_modules)
        self._conscious_state = False
        self._conscious_timer = 0.0
        self._broadcast_counts = {mid: 0 for mid in self._modules.keys()}
        
        # Reset module buffers
        for module in self._modules.values():
            module['input_buffer'] = np.zeros(self.config.workspace_size)
        
        logger.debug("Reset GlobalWorkspaceTheory")
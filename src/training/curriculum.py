"""
CurriculumScheduler module for progressive training
"""

import numpy as np
from typing import Dict, Any, Callable


class CurriculumScheduler:
    """
    Scheduler for curriculum learning in consciousness training.
    
    This class manages progressive difficulty scaling during training,
    gradually increasing the complexity of training tasks.
    """
    
    def __init__(
        self,
        num_stages: int = 5,
        difficulty_function: Optional[Callable[[int], float]] = None,
        stage_duration: int = 10
    ):
        """
        Initialize the CurriculumScheduler.
        
        Args:
            num_stages: Number of curriculum stages
            difficulty_function: Function to compute difficulty for each stage
            stage_duration: Number of epochs per stage
        """
        self.num_stages = num_stages
        self.stage_duration = stage_duration
        self.current_stage = 0
        
        # Set difficulty function
        if difficulty_function is None:
            # Default: exponential difficulty scaling
            self.difficulty_function = lambda stage: 2**stage
        else:
            self.difficulty_function = difficulty_function
        
        # Initialize stage parameters
        self.stage_parameters = self._initialize_stages()
    
    def _initialize_stages(self) -> Dict[int, Dict[str, Any]]:
        """
        Initialize parameters for each curriculum stage.
        
        Returns:
            Dictionary mapping stage numbers to parameters
        """
        stages = {}
        
        for stage in range(self.num_stages):
            difficulty = self.difficulty_function(stage)
            
            stages[stage] = {
                'difficulty': difficulty,
                'batch_size': 32 * (stage + 1),
                'learning_rate': 0.001 / (2**stage),
                'sequence_length': 100 * (stage + 1),
                'num_components': 10 * (stage + 1),
                'noise_level': 0.1 / (stage + 1),
                'constraint_strength': 0.5 * (1 - stage / self.num_stages)
            }
        
        return stages
    
    def update(self, epoch: int) -> Dict[str, Any]:
        """
        Update the curriculum stage based on epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of current stage parameters
        """
        # Calculate current stage
        new_stage = min(epoch // self.stage_duration, self.num_stages - 1)
        
        if new_stage != self.current_stage:
            print(f"\nAdvancing to curriculum stage {new_stage + 1}/{self.num_stages}")
            self.current_stage = new_stage
        
        return self.get_current_parameters()
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for the current curriculum stage.
        
        Returns:
            Dictionary of current stage parameters
        """
        return self.stage_parameters[self.current_stage]
    
    def get_difficulty(self, stage: Optional[int] = None) -> float:
        """
        Get the difficulty level for a given stage.
        
        Args:
            stage: Stage number (default: current stage)
            
        Returns:
            Difficulty level
        """
        if stage is None:
            stage = self.current_stage
        
        return self.difficulty_function(stage)
    
    def should_advance(self, metrics: Dict[str, float]) -> bool:
        """
        Determine if the model should advance to the next stage.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            True if should advance, False otherwise
        """
        # Check if current stage is complete
        if self.current_stage >= self.num_stages - 1:
            return False
        
        # Check performance threshold
        if 'loss' in metrics and metrics['loss'] < 0.1:
            return True
        
        if 'accuracy' in metrics and metrics['accuracy'] > 0.95:
            return True
        
        if 'consciousness_level' in metrics and metrics['consciousness_level'] > 0.8:
            return True
        
        return False
    
    def get_stage_description(self, stage: Optional[int] = None) -> str:
        """
        Get a description of the specified stage.
        
        Args:
            stage: Stage number (default: current stage)
            
        Returns:
            Description string
        """
        if stage is None:
            stage = self.current_stage
        
        params = self.stage_parameters[stage]
        
        descriptions = [
            "Basic quantum state simulation",
            "Multi-scale integration",
            "Complex consciousness modeling",
            "Advanced self-awareness",
            "Full consciousness emergence"
        ]
        
        desc = descriptions[min(stage, len(descriptions) - 1)]
        
        return (
            f"Stage {stage + 1}/{self.num_stages}: {desc}\n"
            f"  Difficulty: {params['difficulty']:.2f}\n"
            f"  Batch size: {params['batch_size']}\n"
            f"  Learning rate: {params['learning_rate']:.6f}\n"
            f"  Sequence length: {params['sequence_length']}\n"
            f"  Noise level: {params['noise_level']:.3f}"
        )
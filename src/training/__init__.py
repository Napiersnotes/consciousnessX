"""
Training and curriculum modules for consciousness development.
"""

from .trainer import ConsciousnessTrainer
from .curriculum import CurriculumScheduler
from .checkpoint import CheckpointManager
from .consciousness_curriculum import ConsciousnessCurriculum

__all__ = [
    'ConsciousnessTrainer',
    'CurriculumScheduler',
    'CheckpointManager',
    'ConsciousnessCurriculum'
]

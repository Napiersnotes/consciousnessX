"""
Training and curriculum modules for consciousness development.
"""

from .trainer import ConsciousnessTrainer
from .curriculum import CurriculumScheduler
from .checkpoint import CheckpointManager
from .consciousness_curriculum import ConsciousnessCurriculum
from .scheduler import (
    LRScheduler,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CyclicLR,
    WarmupScheduler,
)

__all__ = [
    "ConsciousnessTrainer",
    "CurriculumScheduler",
    "CheckpointManager",
    "ConsciousnessCurriculum",
    "LRScheduler",
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "ReduceLROnPlateau",
    "CyclicLR",
    "WarmupScheduler",
]

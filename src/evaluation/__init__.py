"""
Consciousness evaluation and assessment modules.
"""

from .consciousness_assessment import ConsciousnessAssessor
from .ethical_containment import EthicalContainment
from .metrics import ConsciousnessMetrics
from .performance_tracker import PerformanceTracker, MetricLogger
from .model_comparison import ModelComparator, ModelResult, ComparisonResult

__all__ = [
    "ConsciousnessAssessor",
    "EthicalContainment",
    "ConsciousnessMetrics",
    "PerformanceTracker",
    "MetricLogger",
    "ModelComparator",
    "ModelResult",
    "ComparisonResult",
]

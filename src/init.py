"""
ConsciousnessX - Quantum-Biological AGI Framework
"""

__version__ = "0.1.0"
__author__ = "ConsciousnessX Team"
__license__ = "MIT"

from src.core.microtubule_simulator import MicrotubuleSimulator
from src.core.penrose_gravitational_collapse import PenroseCollapse
from src.virtual_bio.ion_channel_dynamics import IonChannelDynamics
from src.visualization.consciousness_dashboard import launch_dashboard
from src.evaluation.consciousness_assessment import ConsciousnessAssessor
from src.training.consciousness_curriculum import ConsciousnessCurriculum

__all__ = [
    'MicrotubuleSimulator',
    'PenroseCollapse', 
    'IonChannelDynamics',
    'launch_dashboard',
    'ConsciousnessAssessor',
    'ConsciousnessCurriculum'
]

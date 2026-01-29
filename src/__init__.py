"""
ConsciousnessX - Quantum-Biological AGI Framework
Penrose-Orch-OR Consciousness Simulation
"""

__version__ = "1.0.0"
__author__ = "Dafydd Napier"
__license__ = "MIT"

from consciousnessx.core.microtubule_simulator import MicrotubuleSimulator
from consciousnessx.core.penrose_gravitational_collapse import GravitationalCollapseCalculator
from src.virtual_bio.ion_channel_dynamics import HodgkinHuxleyNeuron
from src.hardware.virtual_hpc.cray_lux_simulator import VirtualCrayLuxAI
from src.visualization.consciousness_dashboard import ConsciousnessDashboard
from src.evaluation.consciousness_assessment import ConsciousnessAssessor
from src.training.consciousness_curriculum import ConsciousnessCurriculum

__all__ = [
    'MicrotubuleSimulator',
    'GravitationalCollapseCalculator',
    'HodgkinHuxleyNeuron',
    'VirtualCrayLuxAI',
    'ConsciousnessDashboard',
    'ConsciousnessAssessor',
    'ConsciousnessCurriculum'
]

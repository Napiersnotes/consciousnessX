"""
Hardware simulation and virtual HPC modules.
"""

from .virtual_hpc.distributed_consciousness import DistributedConsciousness
from .virtual_hpc.cray_lux_simulator import CrayLuxSimulator

__all__ = [
    'DistributedConsciousness',
    'CrayLuxSimulator'
]

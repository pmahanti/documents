"""
Lunar Regolith Simulation Package

A physics-based simulation of regolith flow on lunar slopes,
modeling the formation of elephant hide textures.
"""

__version__ = "0.1.0"

from .physics import RegolithPhysics
from .slope import SlopeGeometry
from .simulation import RegolithFlowSimulation
from .visualization import SimulationVisualizer

__all__ = [
    "RegolithPhysics",
    "SlopeGeometry",
    "RegolithFlowSimulation",
    "SimulationVisualizer",
]

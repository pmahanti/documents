"""
Lunar Regolith Simulation Package

A physics-based simulation of regolith flow on lunar slopes,
modeling the formation of elephant hide textures through thermal
cycling and seismic perturbations over geological timescales.
"""

__version__ = "0.2.0"

from .physics import RegolithPhysics
from .slope import SlopeGeometry
from .simulation import RegolithFlowSimulation
from .visualization import SimulationVisualizer
from .thermal import LunarThermalCycle, ThermalCreepSimulator
from .seismic import MoonquakeSimulator, SeismicPerturbation, RandomPerturbation
from .geological_simulation import GeologicalRegolithSimulation, AcceleratedSimulation

__all__ = [
    "RegolithPhysics",
    "SlopeGeometry",
    "RegolithFlowSimulation",
    "SimulationVisualizer",
    "LunarThermalCycle",
    "ThermalCreepSimulator",
    "MoonquakeSimulator",
    "SeismicPerturbation",
    "RandomPerturbation",
    "GeologicalRegolithSimulation",
    "AcceleratedSimulation",
]

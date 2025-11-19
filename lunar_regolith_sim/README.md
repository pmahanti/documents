# Lunar Regolith Flow Simulator v0.2

A physics-based simulation of regolith flow on lunar slopes, modeling the formation of distinctive "elephant hide" textures through **thermal cycling** and **seismic perturbations** over **geological timescales**.

## Overview

This package provides a comprehensive simulation framework for studying how elephant hide textures form on the Moon through slow, cumulative downslope creep over millions of years. The simulator accounts for:

- **Realistic regolith properties** (40-50% porosity, 40-800 μm grain sizes, low cohesion)
- **Lunar thermal cycling** (100-400 K day/night temperature swings, 29.5-day cycle)
- **Seismic perturbations** (moonquakes M 2-5, impact-induced ground motion)
- **Slope-dependent formation** (threshold >8°, optimal 15-25°)
- **Geological timescales** (millions of years of cumulative creep)

## What are Elephant Hide Textures?

Elephant hide textures are distinctive wrinkled surface patterns observed on steep lunar slopes, particularly on crater walls. They form through:

### Formation Mechanism

The texture develops via **two primary forces**:

1. **Thermal Cycling** - Temperature variations from lunar day/night cycles cause slow downslope creep:
   - Daytime: ~400 K (127°C)
   - Nighttime: ~100 K (-173°C)
   - Thermal expansion/contraction drives gradual material movement

2. **Seismic Shaking** - Mass movement involves ground shaking from:
   - Impact events (meteorite strikes)
   - Moonquakes (M 2-5 typical, rare M >5 events)
   - Triggers regolith landslides on steep slopes

### Characteristics

- **Appearance**: Wrinkled, anastomosing (branching and rejoining) patterns
- **Orientation**: Aligned downslope
- **Slope dependence**:
  - Threshold: >8° for any texture formation
  - Optimal: 15-25° for maximum development
  - Too steep (>35°): Fresh avalanches dominate
- **Timescale**: Forms over millions of years
- **Location**: Common on crater walls, terrace slopes

## Features

### Physics Engine

- **Enhanced regolith properties**:
  - Porosity: 40-50% (loose, unconsolidated particles)
  - Internal friction angle: 35-40°
  - Cohesion: 0.1-1 kPa (very low)
  - Grain sizes: 40-800 μm median
  - Bulk density consistent with porosity

- **Slope-dependent texture formation**:
  - Automatic intensity calculation based on slope angle
  - Threshold detection (8°)
  - Optimal range identification (15-25°)

### Thermal Cycling Module

- **Lunar day/night cycle simulation**:
  - 29.5 Earth day period
  - Temperature range: 100-400 K
  - Latitude-dependent variations
  - Subsurface thermal diffusion

- **Thermal creep calculation**:
  - Temperature-dependent deformation
  - Cumulative displacement over cycles
  - Thermal stress computation

### Seismic Perturbation Module

- **Moonquake simulation**:
  - Deep moonquakes (~700 km depth, M 2-3.5)
  - Shallow moonquakes (~50 km depth, M 3-5, rare)
  - Thermal moonquakes (~20 km depth, M 1.5-2.5)
  - Impact-induced quakes (variable depth, M 2-4)

- **Ground motion calculation**:
  - Peak ground acceleration
  - Distance attenuation
  - Regolith mobilization thresholds

### Geological Timescale Simulation

- **GeologicalRegolithSimulation**: Full geological evolution
  - Simulate millions to billions of years
  - Tracks thermal cycles and seismic events
  - Fresh vs. aged crater comparison

- **AcceleratedSimulation**: Time-compressed demonstration
  - Time acceleration factors (e.g., 10^6x)
  - Preserves essential physics
  - Faster testing and parameter exploration

### Visualization Tools

- Comprehensive plotting functions
- Fresh vs. aged crater comparisons
- Slope-texture relationship analysis
- Temporal evolution tracking
- Publication-quality figures

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from source

```bash
cd lunar_regolith_sim
pip install -e .
```

### Dependencies

- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- pillow >= 8.3.0
- numba >= 0.54.0

## Quick Start

### Geological Timescale Simulation

```python
from lunar_regolith_sim import (
    RegolithPhysics,
    SlopeGeometry,
    LunarThermalCycle,
    MoonquakeSimulator,
    GeologicalRegolithSimulation
)

# Create crater wall geometry
slope = SlopeGeometry(width=200, height=200, resolution=1.0)
slope.create_crater_wall(
    crater_x=100, crater_y=100,
    inner_radius=40, outer_radius=85,
    rim_height=20, floor_depth=10
)

# Initialize physics with realistic parameters
physics = RegolithPhysics(
    porosity=0.45,                # 45% porosity
    cohesion=0.5,                 # 0.5 kPa
    internal_friction_angle=37.5, # 35-40° range
    grain_size=60e-6              # 60 μm
)

# Thermal cycle (100-400 K)
thermal = LunarThermalCycle(temp_max=400, temp_min=100)

# Moonquakes
moonquakes = MoonquakeSimulator()

# Create geological simulation
geo_sim = GeologicalRegolithSimulation(
    slope_geometry=slope,
    physics=physics,
    thermal_cycle=thermal,
    moonquake_sim=moonquakes,
    initial_thickness=2.0
)

# Simulate fresh crater (smooth regolith)
fresh_state = geo_sim.simulate_fresh_crater()

# Evolve over 3 million years
final_state = geo_sim.advance_geological_time(duration_years=3e6)

# Get elephant hide texture
texture = geo_sim.get_elephant_hide_texture()
```

### Run Example Scripts

```bash
# Geological timescale simulation (millions of years)
python examples/geological_timescale.py

# Realistic parameter study
python examples/realistic_parameters.py

# Original fast-flow examples
python examples/simple_slope.py
python examples/crater_wall.py
python examples/interactive_demo.py
```

## Documentation

### Physics Module

Enhanced with realistic lunar regolith parameters:

```python
physics = RegolithPhysics(
    porosity=0.45,                # 40-50% typical
    cohesion=0.5,                 # 0.1-1 kPa range
    internal_friction_angle=37.5, # 35-40° range
    grain_size=60e-6,            # 40-800 μm median
    gravity=1.62                  # Lunar gravity
)

# Check texture formation potential
intensity = physics.get_texture_formation_intensity(slope_angle=20)

# Calculate creep rate
creep_rate = physics.calculate_creep_rate(
    slope_angle=20,
    temperature_variation=150,  # K
    seismic_activity=0.1       # 0-1 scale
)
```

### Thermal Cycling Module

```python
from lunar_regolith_sim import LunarThermalCycle, ThermalCreepSimulator

# Create thermal cycle
thermal = LunarThermalCycle(
    temp_max=400,   # Daytime temperature (K)
    temp_min=100,   # Nighttime temperature (K)
    latitude=0.0    # Equatorial (max variation)
)

# Get temperature at time
temp = thermal.get_surface_temperature(time_seconds)

# Get temperature at depth
temp_subsurface = thermal.get_temperature_at_depth(time_seconds, depth=0.5)

# Thermal creep simulator
creep_sim = ThermalCreepSimulator(thermal, physics)
result = creep_sim.simulate_texture_development(slope_angles, duration_years)
```

### Seismic Perturbation Module

```python
from lunar_regolith_sim import MoonquakeSimulator, SeismicPerturbation

# Create moonquake simulator
moonquakes = MoonquakeSimulator(
    quake_rate_multiplier=1.0,  # 1.0 = typical rate
    seed=42                      # Reproducible
)

# Generate quake sequence
quakes = moonquakes.generate_quake_sequence(duration_years=1e6)

# Calculate ground motion
pga = moonquakes.calculate_ground_motion(
    magnitude=3.5,      # Moonquake magnitude
    distance_km=50,     # Distance from epicenter
    depth_km=700        # Depth of quake
)

# Seismic perturbation
seismic = SeismicPerturbation(moonquakes)
perturbation = seismic.apply_to_grid(
    slope_angles, epicenter_x, epicenter_y,
    magnitude, depth_km, grid_resolution
)
```

### Geological Simulation Module

```python
from lunar_regolith_sim import GeologicalRegolithSimulation

geo_sim = GeologicalRegolithSimulation(
    slope_geometry=slope,
    physics=physics,
    thermal_cycle=thermal,
    moonquake_sim=moonquakes
)

# Fresh crater (t=0)
fresh_state = geo_sim.simulate_fresh_crater()

# Advance geological time
final_state = geo_sim.advance_geological_time(duration_years=3e6)

# Compare fresh vs aged
comparison = geo_sim.compare_fresh_vs_aged()

# Get texture
texture = geo_sim.get_elephant_hide_texture()
```

## Scientific Background

### Formation Timescales

Elephant hide textures form through **slow, cumulative downslope creep** over millions of years. Fresh impact craters have smooth regolith that gradually develops texture over time:

- **0-100 kyr**: Minimal texture, fresh appearance
- **100 kyr - 1 Myr**: Initial texture nucleation
- **1-10 Myr**: Well-developed texture patterns
- **>10 Myr**: Mature, complex textures

### Physical Parameters

Based on Apollo samples and remote sensing:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Porosity | 40-50% | Loose, unconsolidated |
| Bulk density | 1700-1900 kg/m³ | Depends on porosity |
| Grain density | ~3100 kg/m³ | Solid particles |
| Median grain size | 40-800 μm | Size distribution |
| Internal friction | 35-40° | Angle of internal friction |
| Cohesion | 0.1-1 kPa | Very low |
| Angle of repose | ~35° | Critical for avalanching |

### Thermal Cycling

| Parameter | Value |
|-----------|-------|
| Daytime temp | ~400 K (127°C) |
| Nighttime temp | ~100 K (-173°C) |
| Temperature swing | ~300 K |
| Cycle period | 29.5 Earth days |
| Thermal skin depth | ~0.5 m |

### Seismic Activity

| Type | Depth | Magnitude | Rate |
|------|-------|-----------|------|
| Deep moonquakes | ~700 km | M 2-3.5 | ~600/yr |
| Shallow moonquakes | ~50 km | M 3-5 | ~2/yr |
| Thermal moonquakes | ~20 km | M 1.5-2.5 | ~100/yr |
| Impact quakes | <10 km | M 2-4 | ~50/yr |

## Applications

- **Planetary Science**: Understanding lunar surface evolution
- **Mission Planning**: Slope stability assessment for rovers/landers
- **Age Dating**: Using texture development to estimate crater ages
- **Regolith Properties**: Inferring properties from texture patterns
- **Comparative Planetology**: Studying similar processes on other airless bodies
- **Education**: Teaching geological timescales and planetary processes

## Limitations and Assumptions

- 2D surface simulation (vertical dimension simplified)
- Homogeneous regolith properties
- Simplified thermal and seismic models
- No impact gardening effects
- No space weathering effects
- Statistical representation of long-term processes

## Contributing

Contributions welcome! Areas for enhancement:

- 3D particle-based simulation
- Heterogeneous regolith layers
- Impact gardening integration
- Validation against LRO/Kaguya data
- Machine learning texture recognition

## References

1. Kumar, P.S., et al. (2016). "Gullies and landslides on the Moon"
2. Senthil Kumar, P., et al. (2013). "Recent shallow moonquake and impact-triggered boulder falls"
3. Xiao, Z., et al. (2013). "Mass wasting features on the Moon"
4. Apollo Lunar Surface Experiments Package (ALSEP) seismic data
5. Lunar Reconnaissance Orbiter Camera (LROC) observations

## License

MIT License - See LICENSE file

## Authors

Lunar Regolith Research Team

---

**Version 0.2.0** - Enhanced with thermal cycling, seismic perturbations, and geological timescale simulation

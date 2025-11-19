# Lunar Regolith Flow Simulator

A physics-based simulation of regolith flow on lunar slopes, modeling the formation of distinctive "elephant hide" textures observed on the Moon.

## Overview

This package provides a comprehensive simulation framework for studying granular flow dynamics of lunar regolith on slopes. The simulator accounts for:

- **Lunar gravity conditions** (1/6 of Earth's gravity)
- **Granular flow mechanics** (angle of repose, friction, cohesion)
- **Cellular automata avalanching** for discrete flow events
- **Continuum mechanics** for smooth flow processes
- **Texture formation** through cumulative deformation

## What are Elephant Hide Textures?

Elephant hide textures are distinctive surface patterns observed on steep lunar slopes, particularly on crater walls. They appear as:

- Wrinkled, anastomosing (branching and rejoining) patterns
- Oriented downslope
- Alternating ridges and troughs
- Result from repeated granular flow and avalanching

These textures provide valuable information about:
- Regolith properties and behavior
- Slope stability and dynamics
- Lunar surface processes and evolution

## Features

- **Physics Engine**: Realistic regolith mechanics under lunar gravity
- **Flexible Geometry**: Support for linear slopes, crater walls, and terraced slopes
- **Flow Simulation**: Hybrid cellular automata + continuum mechanics approach
- **Visualization**: Comprehensive plotting tools for analysis
- **Extensible**: Modular design for easy customization

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from source

```bash
cd lunar_regolith_sim
pip install -e .
```

This will install the package and all dependencies.

### Dependencies

- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- pillow >= 8.3.0
- numba >= 0.54.0 (for performance optimization)

## Quick Start

### Basic Example

```python
from lunar_regolith_sim import (
    RegolithPhysics,
    SlopeGeometry,
    RegolithFlowSimulation,
    SimulationVisualizer
)

# Create a slope
slope = SlopeGeometry(width=100, height=100, resolution=1.0)
slope.create_linear_slope(angle=35, direction='y')
slope.add_roughness(amplitude=0.2, wavelength=5.0)

# Set up physics (lunar conditions)
physics = RegolithPhysics()

# Create simulation
sim = RegolithFlowSimulation(
    slope_geometry=slope,
    physics=physics,
    initial_thickness=1.5
)

# Run simulation
sim.run(duration=1000)

# Visualize results
viz = SimulationVisualizer(sim)
viz.create_summary_figure()
```

### Run Example Scripts

The package includes several example scripts:

```bash
# Simple linear slope
python examples/simple_slope.py

# Crater wall (most realistic for elephant hide)
python examples/crater_wall.py

# Parameter study
python examples/interactive_demo.py
```

## Documentation

### Physics Module

The `RegolithPhysics` class handles physical properties and mechanics:

```python
physics = RegolithPhysics(
    gravity=1.62,              # m/s² (lunar gravity)
    particle_density=1800,     # kg/m³
    angle_of_repose=35.0,      # degrees
    cohesion=0.1               # kPa
)

# Check stability
is_stable = physics.is_stable(slope_angle=30, thickness=1.0)

# Calculate flow velocity
velocity = physics.calculate_flow_velocity(slope_angle=35, thickness=1.5)
```

### Slope Geometry Module

The `SlopeGeometry` class creates terrain configurations:

```python
slope = SlopeGeometry(width=100, height=100, resolution=1.0)

# Linear slope
slope.create_linear_slope(angle=35, direction='y')

# Crater wall
slope.create_crater_wall(
    crater_x=50, crater_y=50,
    inner_radius=20, outer_radius=40,
    rim_height=10, floor_depth=5
)

# Terraced slope
slope.create_terrace(
    terrace_y=50, terrace_height=5,
    slope_angle_upper=30, slope_angle_lower=35
)

# Add surface features
slope.add_roughness(amplitude=0.2, wavelength=5.0)
slope.add_perturbation(x=50, y=50, radius=5, amplitude=2)
```

### Simulation Module

The `RegolithFlowSimulation` class runs the simulation:

```python
sim = RegolithFlowSimulation(
    slope_geometry=slope,
    physics=physics,
    initial_thickness=1.5
)

# Single time step
state = sim.step(dt=0.1)

# Run for duration
history = sim.run(duration=1000, progress_interval=100)

# Extract elephant hide texture
texture = sim.get_elephant_hide_texture()

# Reset simulation
sim.reset(initial_thickness=2.0)
```

### Visualization Module

The `SimulationVisualizer` class provides comprehensive plotting:

```python
viz = SimulationVisualizer(sim)

# Individual plots
viz.plot_elevation(include_regolith=True)
viz.plot_slope_angle()
viz.plot_elephant_hide(enhance=True)
viz.plot_flow_velocity()
viz.plot_3d_surface()

# Comprehensive summary
viz.create_summary_figure(save_path='summary.png')

# Save all figures
viz.save_all_figures(output_dir='output/')
```

## Physical Background

### Granular Flow on the Moon

Regolith flow on lunar slopes is governed by several factors:

1. **Low Gravity**: 1.62 m/s² vs 9.81 m/s² on Earth
   - Affects flow velocity and runout distance
   - Influences particle interactions

2. **No Atmosphere**: No air resistance or moisture
   - Purely mechanical interactions
   - Low cohesion between particles

3. **Angle of Repose**: Critical angle (~35° for lunar regolith)
   - Below: stable, no flow
   - At/above: unstable, avalanching occurs

4. **Savage-Hutter Flow Model**: Describes granular avalanches
   - Velocity depends on thickness and slope
   - Energy dissipation through friction

### Elephant Hide Formation

The distinctive texture forms through:

1. **Initial Perturbation**: Small-scale roughness or irregularities
2. **Flow Nucleation**: Flow begins at steepest/most unstable points
3. **Channelization**: Flow concentrates in channels
4. **Deposition**: Material deposits in patterns
5. **Repetition**: Multiple flow events create texture

## Scientific Applications

This simulator can be used for:

- **Planetary Science**: Understanding lunar surface processes
- **Mission Planning**: Assessing slope stability for rovers/landers
- **Regolith Characterization**: Inferring properties from textures
- **Comparative Planetology**: Studying flow on other airless bodies
- **Education**: Teaching granular mechanics and planetary processes

## Performance Notes

- **Grid Resolution**: Use 0.5-1.0 m for good balance of speed/accuracy
- **Time Steps**: Default 0.1 s is generally appropriate
- **JIT Compilation**: First run may be slow (Numba compilation)
- **Large Domains**: Consider coarser resolution or smaller area

## Output

The simulation generates:

1. **Elevation maps**: Topography with hillshading
2. **Slope angle maps**: Distribution and stability analysis
3. **Elephant hide textures**: Enhanced texture patterns
4. **Flow velocity fields**: Magnitude and direction
5. **3D surface views**: Perspective visualization
6. **Cross-sections**: Profile views
7. **Statistics**: Flow events, deformation, etc.

## Limitations and Assumptions

- **2D Simulation**: Vertical dimension simplified
- **Simplified Physics**: Real regolith behavior is more complex
- **No Temperature Effects**: Assumes isothermal conditions
- **Homogeneous Regolith**: Uniform properties
- **Steady Gravity**: No tidal or rotational effects

## Contributing

Contributions are welcome! Areas for improvement:

- More sophisticated flow models
- 3D particle-based simulation
- Temperature-dependent properties
- Heterogeneous regolith layers
- Validation against real lunar data

## References

1. Bart, G. D. (2007). "Comparison of small lunar landslides and martian gullies"
2. Senthil Kumar, P., et al. (2013). "Lunar regolith mass movement on crater wall"
3. Xiao, Z., et al. (2013). "Mass wasting features on the Moon"
4. Savage, S. B., & Hutter, K. (1989). "The motion of a finite mass of granular material"

## License

MIT License - See LICENSE file for details

## Authors

Lunar Regolith Research Team

## Acknowledgments

- Based on observations from Apollo missions, Lunar Reconnaissance Orbiter, and other lunar missions
- Physics models adapted from terrestrial granular flow research
- Inspired by planetary surface processes research

## Support

For issues, questions, or suggestions:
- Open an issue on the GitHub repository
- Contact the development team

---

**Happy Simulating!** 🌙

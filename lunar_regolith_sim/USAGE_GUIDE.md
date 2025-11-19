# Usage Guide: Generating Elephant Hide Textures

## Quick Start

### 1. Simple Texture Generation

Generate a texture with default parameters:

```bash
cd lunar_regolith_sim
python generate_texture.py
```

This creates `texture_output.png` showing a 3 million year evolution on a 20° slope.

### 2. Custom Parameters

Generate texture with custom slope angle and duration:

```bash
python generate_texture.py --slope 18 --duration 5.0 --output my_texture.png
```

### 3. Crater Geometry

Generate texture on a crater wall:

```bash
python generate_texture.py --geometry crater --duration 3.0 --output crater_texture.png
```

### 4. Parameter Study

Test different porosity values:

```bash
python generate_texture.py --porosity 0.40 --output compact_regolith.png
python generate_texture.py --porosity 0.50 --output loose_regolith.png
```

## Command-Line Options

### Slope Parameters
- `--slope ANGLE` - Slope angle in degrees (default: 20.0)
  - Optimal range: 15-25°
  - Threshold: >8° for texture formation
  - Example: `--slope 22`

### Regolith Properties
- `--porosity VALUE` - Porosity 0-1 (default: 0.45)
  - Typical range: 0.40-0.50
  - Higher = looser regolith
  - Example: `--porosity 0.48`

- `--cohesion VALUE` - Cohesion in kPa (default: 0.5)
  - Typical range: 0.1-1.0
  - Lower = easier mobilization
  - Example: `--cohesion 0.3`

- `--grain-size VALUE` - Median grain size in μm (default: 60)
  - Typical range: 40-800
  - Example: `--grain-size 100`

### Thermal Cycling
- `--temp-max VALUE` - Maximum daytime temperature in K (default: 400)
  - Equatorial: ~400 K
  - High latitude: ~300 K
  - Example: `--temp-max 380`

- `--temp-min VALUE` - Minimum nighttime temperature in K (default: 100)
  - Typical: 100 K
  - Example: `--temp-min 120`

### Simulation Parameters
- `--duration VALUE` - Evolution time in million years (default: 3.0)
  - Young crater: 0.5-1.0 Myr
  - Mature crater: 3.0-10.0 Myr
  - Example: `--duration 5.0`

- `--geometry TYPE` - Slope geometry: `linear` or `crater` (default: linear)
  - `linear`: Simple planar slope
  - `crater`: Crater wall geometry
  - Example: `--geometry crater`

- `--domain-size VALUE` - Domain size in meters (default: 100)
  - Example: `--domain-size 200`

- `--resolution VALUE` - Grid resolution in m/cell (default: 1.0)
  - Finer resolution = more detail, slower computation
  - Example: `--resolution 0.5`

### Output Options
- `--output PATH` - Output image path (default: texture_output.png)
  - Example: `--output results/my_simulation.png`

- `--show` - Display plot window interactively
  - Example: `--show`

## Example Scenarios

### Optimal Conditions (Maximum Texture Development)
```bash
python generate_texture.py \
  --slope 20 \
  --porosity 0.45 \
  --cohesion 0.5 \
  --duration 3.0 \
  --geometry crater \
  --output optimal_texture.png
```

### Young Crater (Minimal Texture)
```bash
python generate_texture.py \
  --slope 20 \
  --duration 0.5 \
  --output young_crater.png
```

### Steep Slope (Fresh Avalanches)
```bash
python generate_texture.py \
  --slope 35 \
  --duration 3.0 \
  --output steep_slope.png
```

### Low Slope (Below Threshold)
```bash
python generate_texture.py \
  --slope 5 \
  --duration 3.0 \
  --output low_slope.png
```

### High Latitude (Reduced Thermal Cycling)
```bash
python generate_texture.py \
  --temp-max 300 \
  --temp-min 150 \
  --duration 3.0 \
  --output high_latitude.png
```

### Very Loose Regolith
```bash
python generate_texture.py \
  --porosity 0.50 \
  --cohesion 0.1 \
  --duration 3.0 \
  --output loose_regolith.png
```

## Animated Evolution

### Generate Animation Over Geological Time

```bash
cd examples
python generate_animated_texture.py
```

This creates an animated GIF showing texture evolution from 0 to 3 Myr.

### Customize Animation

Edit `generate_animated_texture.py` and modify parameters in the `main()` function:

```python
result = generate_texture_animation(
    # Adjust these parameters
    total_duration_myr=5.0,    # 5 million years
    num_snapshots=30,          # 30 frames
    animation_fps=3,           # 3 frames per second
    # ... other parameters
)
```

## Python API Usage

### Basic Texture Generation

```python
from lunar_regolith_sim import (
    RegolithPhysics,
    SlopeGeometry,
    GeologicalRegolithSimulation,
    LunarThermalCycle,
    MoonquakeSimulator
)

# Create slope (20° linear slope)
slope = SlopeGeometry(100, 100, 1.0)
slope.create_linear_slope(angle=20, direction='y')
slope.add_roughness(amplitude=0.3, wavelength=5.0)

# Initialize physics (realistic parameters)
physics = RegolithPhysics(
    porosity=0.45,
    cohesion=0.5,
    grain_size=60e-6
)

# Thermal cycle (100-400 K)
thermal = LunarThermalCycle(temp_max=400, temp_min=100)

# Moonquakes
moonquakes = MoonquakeSimulator()

# Create simulation
sim = GeologicalRegolithSimulation(
    slope, physics, thermal, moonquakes
)

# Simulate evolution
sim.simulate_fresh_crater()
result = sim.advance_geological_time(3e6)  # 3 million years

# Get texture
texture = sim.get_elephant_hide_texture()
```

### Cycle Over Multiple Timescales

```python
# Simulate different evolutionary stages
stages = [0.1, 0.5, 1.0, 3.0, 10.0]  # Million years

for duration_myr in stages:
    sim.reset()
    sim.simulate_fresh_crater()
    result = sim.advance_geological_time(duration_myr * 1e6)

    texture = result['texture_intensity']
    print(f"t={duration_myr} Myr: mean texture = {texture.mean():.3f}")
```

### Parameter Sweep

```python
import numpy as np
import matplotlib.pyplot as plt

slopes = np.linspace(5, 40, 8)
textures = []

for slope_deg in slopes:
    slope = SlopeGeometry(100, 100, 1.0)
    slope.create_linear_slope(angle=slope_deg, direction='y')

    sim = GeologicalRegolithSimulation(slope, physics, thermal, moonquakes)
    sim.simulate_fresh_crater()
    result = sim.advance_geological_time(3e6)

    textures.append(result['texture_intensity'].mean())

plt.plot(slopes, textures)
plt.xlabel('Slope Angle (degrees)')
plt.ylabel('Mean Texture Intensity')
plt.title('Texture Development vs. Slope')
plt.show()
```

## Understanding Output

### Output Image Contains 6 Panels:

1. **Topography** - Crater/slope elevation
2. **Slope Angles** - With 8°, 15°, 25° contours
3. **Formation Potential** - Likelihood of texture formation
4. **Fresh (t=0)** - Initial smooth state
5. **Aged (t=duration)** - Final textured state
6. **Statistics** - Simulation parameters and results

### Key Metrics:

- **Mean texture intensity** - Average texture strength (0-1)
  - <0.1: Very little texture
  - 0.3-0.6: Moderate texture
  - >0.6: Well-developed texture

- **Textured area** - Percentage with intensity >0.3
  - Indicates spatial extent of texture

- **Max displacement** - Maximum cumulative movement
  - Shows total downslope transport

## Interpretation Guide

### Slope Dependency:
- **<8°**: No significant texture (below threshold)
- **8-15°**: Gradual texture development
- **15-25°**: Optimal texture formation
- **25-35°**: Moderate texture (competing with avalanches)
- **>35°**: Limited texture (dominated by fresh avalanches)

### Temporal Evolution:
- **<0.1 Myr**: Fresh appearance, minimal texture
- **0.1-1 Myr**: Initial texture nucleation
- **1-3 Myr**: Well-developed texture patterns
- **>3 Myr**: Mature, complex textures

### Porosity Effects:
- **40%**: More consolidated, slower creep
- **45%**: Typical loose regolith
- **50%**: Very loose, faster creep

### Thermal Cycling:
- **ΔT = 300 K**: Maximum thermal cycling (equatorial)
- **ΔT = 150 K**: Reduced cycling (high latitude)
- Greater ΔT → More thermal creep → Stronger textures

## Tips for Best Results

1. **For realistic simulations**: Use default parameters
2. **For faster testing**: Reduce `domain-size` and increase `resolution`
3. **For animations**: Use `examples/generate_animated_texture.py`
4. **For parameter studies**: Use `examples/realistic_parameters.py`
5. **For crater walls**: Always use `--geometry crater`

## Troubleshooting

### Simulation is slow
- Reduce `--domain-size` (e.g., 50 instead of 100)
- Increase `--resolution` (e.g., 2.0 instead of 1.0)
- Reduce `--duration` for testing

### Little or no texture
- Check slope angle (must be >8°)
- Increase `--duration` (try 5-10 Myr)
- Check porosity (higher = more creep)

### Too much texture everywhere
- May be too steep (check slope angles)
- May be too long duration
- Adjust parameters toward more consolidated regolith

## Citation

If you use this simulation in research, please cite:
- Apollo Lunar Surface Experiments Package (ALSEP) seismic data
- Lunar Reconnaissance Orbiter Camera (LROC) observations
- This simulation package: lunar-regolith-sim v0.2.0

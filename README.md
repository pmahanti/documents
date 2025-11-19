# Lunar Regolith Flow Simulation

Physics-based simulation of regolith flow on lunar slopes, modeling the formation of elephant hide textures.

## Project: lunar_regolith_sim

A comprehensive Python package for simulating granular flow of lunar regolith on slopes under lunar gravity conditions. The simulation models the formation of distinctive "elephant hide" textures commonly observed on steep lunar slopes, particularly on crater walls.

### Features

- **Realistic physics** with thermal cycling (100-400 K) and seismic perturbations (moonquakes M 2-5)
- **Geological timescales** simulating millions of years of cumulative creep
- **Slope-dependent formation** (>8° threshold, 15-25° optimal range)
- **Accurate regolith properties** (40-50% porosity, 40-800 μm grain sizes, 0.1-1 kPa cohesion)
- **CLI texture generator** for quick parameter exploration
- **Animation generation** showing evolution from fresh to aged craters
- **Multiple slope geometries** (linear slopes, crater walls, terraces)
- **Comprehensive visualization** tools for analysis

### Quick Start

**Generate texture with default parameters:**
```bash
cd lunar_regolith_sim
pip install -e .
python generate_texture.py
```

**Generate with custom parameters:**
```bash
python generate_texture.py --slope 20 --duration 3.0 --porosity 0.45 --geometry crater
```

**Create animated evolution:**
```bash
cd examples
python generate_animated_texture.py
```

**Run example simulations:**
```bash
python examples/geological_timescale.py    # 3 million year evolution
python examples/realistic_parameters.py    # Parameter study
```

See `lunar_regolith_sim/USAGE_GUIDE.md` for detailed usage and `lunar_regolith_sim/README.md` for scientific background.

### Scientific Context

Elephant hide textures are wrinkled, anastomosing surface patterns found on steep lunar slopes. They result from repeated granular flow and avalanching events, providing insights into:
- Regolith properties and behavior
- Slope stability dynamics
- Lunar surface evolution processes

This simulation helps researchers understand these processes and can support mission planning for lunar exploration.

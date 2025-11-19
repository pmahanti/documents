# Lunar Regolith Flow Simulation

Physics-based simulation of regolith flow on lunar slopes, modeling the formation of elephant hide textures.

## Project: lunar_regolith_sim

A comprehensive Python package for simulating granular flow of lunar regolith on slopes under lunar gravity conditions. The simulation models the formation of distinctive "elephant hide" textures commonly observed on steep lunar slopes, particularly on crater walls.

### Features

- **Physics-based simulation** accounting for lunar gravity (1/6 Earth)
- **Granular flow mechanics** with realistic angle of repose and friction
- **Multiple slope geometries** (linear slopes, crater walls, terraces)
- **Hybrid simulation approach** combining cellular automata and continuum mechanics
- **Comprehensive visualization** tools for analysis
- **Example scripts** demonstrating various scenarios

### Quick Start

```bash
cd lunar_regolith_sim
pip install -e .
python examples/crater_wall.py
```

See the full documentation in `lunar_regolith_sim/README.md` for detailed usage instructions.

### Scientific Context

Elephant hide textures are wrinkled, anastomosing surface patterns found on steep lunar slopes. They result from repeated granular flow and avalanching events, providing insights into:
- Regolith properties and behavior
- Slope stability dynamics
- Lunar surface evolution processes

This simulation helps researchers understand these processes and can support mission planning for lunar exploration.

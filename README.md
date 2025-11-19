# Comet 3I Trajectory Simulation

Simulation and visualization of an interstellar comet (3I) approaching the Earth-Moon system.

## Overview

This project simulates the trajectory of a hypothetical interstellar comet designated "3I" as it passes through the inner solar system with a close approach to the Earth-Moon system. The simulation uses N-body gravitational dynamics to accurately model the interactions between the Sun, Earth, Moon, and the comet.

## Features

- **Orbital Mechanics**: Simulates hyperbolic (interstellar) comet trajectory using Keplerian orbital elements
- **N-Body Dynamics**: Accurate gravitational interactions between Sun, Earth, Moon, and comet
- **3D Visualization**: Beautiful 3D plots showing the complete trajectory through space
- **Close Approach Analysis**: Detailed view of the comet's closest approach to Earth and Moon
- **Distance Tracking**: Real-time distance calculations and plots

## Installation

1. Clone this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:

```bash
python comet_3i_trajectory.py
```

The script will:
1. Simulate the comet's trajectory for 100 days
2. Print a summary of the encounter including closest approach distances
3. Generate two high-resolution PNG images:
   - `comet_3i_trajectory_3d.png` - Full 3D trajectory view
   - `comet_3i_close_approach.png` - Detailed close approach analysis
4. Display interactive plots

## Orbital Parameters

The simulation uses the following parameters for Comet 3I:

- **Eccentricity**: 3.5 (hyperbolic orbit, indicating interstellar origin)
- **Perihelion**: 1.0 AU (closest approach to the Sun, near Earth's orbit)
- **Inclination**: 30° (orbital tilt relative to Earth's orbit)
- **Longitude of Ascending Node**: 45°
- **Argument of Perihelion**: 60°
- **Hyperbolic Excess Velocity**: 26 km/s

These parameters represent a comet passing through the inner solar system on a hyperbolic trajectory, similar to interstellar objects like 1I/'Oumuamua or 2I/Borisov.

## Output Examples

### 3D Trajectory
Shows the complete path of the comet through the solar system relative to the Sun, Earth, and Moon.

### Close Approach
- Earth-centered reference frame showing detailed encounter geometry
- Distance vs. time plots for both Earth and Moon
- Closest approach points marked

## Customization

You can modify the comet's orbital parameters in `comet_3i_trajectory.py`:

```python
COMET_PARAMS = {
    'eccentricity': 3.5,
    'perihelion': 1.0,
    'inclination': 30.0,
    'longitude_ascending': 45.0,
    'argument_perihelion': 60.0,
    'v_infinity': 26.0,
}
```

You can also adjust simulation parameters:

```python
t, comet_pos, earth_pos, moon_pos = simulate_comet_trajectory(
    duration_days=100,  # Simulation length
    timesteps=2000      # Number of calculation steps
)
```

## Technical Details

- **Coordinate System**: Heliocentric (Sun-centered) inertial reference frame
- **Units**: AU for distance, years for time, solar masses for mass
- **Integration Method**: SciPy's `odeint` using LSODA algorithm
- **Gravitational Model**: Full N-body with relativistic corrections omitted (suitable for these distances and velocities)

## Dependencies

- Python 3.7+
- NumPy: Numerical computations
- Matplotlib: Visualization and plotting
- SciPy: Numerical integration

## References

For more information about interstellar objects:
- [1I/'Oumuamua](https://en.wikipedia.org/wiki/%CA%BBOumuamua) - First confirmed interstellar object
- [2I/Borisov](https://en.wikipedia.org/wiki/2I/Borisov) - First confirmed interstellar comet

## License

Open source project for educational and research purposes.
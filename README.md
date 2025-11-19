# Comet Trajectory Visualization

Visualization of comet trajectories approaching the Earth-Moon system using real NAIF SPICE ephemeris data and simulated N-body dynamics.

## Overview

This project provides tools to visualize and analyze comet trajectories as they pass through the inner solar system:

1. **Real Ephemeris Data**: Visualize **Comet C/2025 N1 (ATLAS)** using actual NAIF SPICE kernel data
2. **Simulated Trajectories**: Simulate hypothetical interstellar comets using N-body gravitational dynamics

## Features

### Real SPICE Data (Comet ATLAS)
- **Actual Ephemeris**: Uses NASA NAIF SPICE kernels with real orbital data
- **Comet C/2025 N1 (ATLAS)**: Data from Nov 17 - Dec 17, 2025
- **3D Visualization**: Complete trajectory through the solar system
- **Close Approach Analysis**: Earth-centered views with multiple perspectives
- **Distance Tracking**: Precise distance calculations to Earth and Moon
- **Automatic Kernel Management**: Downloads required planetary ephemeris

### Simulated N-Body Dynamics
- **Orbital Mechanics**: Hyperbolic (interstellar) comet trajectory using Keplerian elements
- **N-Body Integration**: Gravitational interactions between Sun, Earth, Moon, and comet
- **Customizable Parameters**: Adjust orbital elements for different scenarios

## Installation

1. Clone this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Real Comet ATLAS Data (Recommended)

Visualize the actual trajectory of Comet C/2025 N1 (ATLAS):

```bash
python comet_atlas_spice.py
```

The script will:
1. Download required SPICE kernels (leap seconds, planetary ephemeris)
2. Load the comet's SPK kernel: `3latlas_C2025N1_1004083_2025-11-17_2025-12-17.bsp`
3. Extract real trajectory data for Nov 17 - Dec 17, 2025
4. Print detailed summary including closest approach distances
5. Generate two high-resolution PNG images:
   - `comet_atlas_trajectory_3d.png` - Full 3D trajectory view
   - `comet_atlas_close_approach.png` - Detailed close approach analysis
6. Display interactive plots

**Note**: First run will download ~150MB of SPICE kernels (cached for future use).

### Option 2: Simulated Hypothetical Comet

Run the N-body simulation for a hypothetical interstellar comet:

```bash
python comet_3i_trajectory.py
```

The script will:
1. Simulate a hyperbolic comet trajectory for 100 days
2. Print summary of the simulated encounter
3. Generate two high-resolution PNG images:
   - `comet_3i_trajectory_3d.png` - Full 3D trajectory view
   - `comet_3i_close_approach.png` - Detailed close approach analysis
4. Display interactive plots

## Data Sources

### Comet C/2025 N1 (ATLAS)

**SPICE Kernel**: `3latlas_C2025N1_1004083_2025-11-17_2025-12-17.bsp`
- **Source**: NASA NAIF SPICE kernel (binary SPK format)
- **NAIF ID**: 1004083
- **Time Coverage**: November 17, 2025 - December 17, 2025
- **Discovery**: ATLAS survey
- **Type**: Real observational data with computed ephemeris

The kernel contains precise position and velocity vectors computed from actual observations.

### Simulated Hypothetical Comet

The simulation uses the following orbital parameters:

- **Eccentricity**: 3.5 (hyperbolic orbit, indicating interstellar origin)
- **Perihelion**: 1.0 AU (closest approach to the Sun, near Earth's orbit)
- **Inclination**: 30° (orbital tilt relative to Earth's orbit)
- **Longitude of Ascending Node**: 45°
- **Argument of Perihelion**: 60°
- **Hyperbolic Excess Velocity**: 26 km/s

These parameters represent a comet passing through the inner solar system on a hyperbolic trajectory, similar to interstellar objects like 1I/'Oumuamua or 2I/Borisov.

## Output Examples

Both scripts generate comprehensive visualizations:

### 3D Trajectory Plot
- Complete path of the comet through the solar system
- Heliocentric (Sun-centered) reference frame
- Earth and Moon orbits for context
- Start and end positions clearly marked
- Scale in Astronomical Units (AU)

### Close Approach Analysis
Three complementary views:
1. **3D Earth-Centered View**: Detailed geometry of the encounter
2. **Distance vs Time**: Logarithmic plot showing distances to Earth and Moon
3. **XY Plane View**: Top-down perspective with lunar distance circle

Features:
- Closest approach point highlighted
- Distance measurements in multiple units (km, AU, Lunar Distances)
- Date/time stamps for key events

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

### SPICE-Based Visualization (comet_atlas_spice.py)
- **Data Format**: NASA NAIF SPICE SPK (binary ephemeris kernel)
- **Library**: spiceypy (Python wrapper for CSPICE)
- **Coordinate System**: J2000 inertial reference frame
- **Reference Point**: Heliocentric (Sun-centered)
- **Precision**: High-precision ephemeris from observational data
- **Required Kernels**:
  - `naif0012.tls`: Leap seconds kernel
  - `de440.bsp`: JPL planetary ephemeris (Sun, Earth, Moon)
  - Comet-specific SPK kernel (included)

### N-Body Simulation (comet_3i_trajectory.py)
- **Coordinate System**: Heliocentric (Sun-centered) inertial reference frame
- **Units**: AU for distance, years for time, solar masses for mass
- **Integration Method**: SciPy's `odeint` using LSODA algorithm
- **Gravitational Model**: Full N-body (Sun, Earth, Moon, comet)
- **Initial Conditions**: Computed from Keplerian orbital elements
- **Numerical Precision**: Adaptive step size integration

## Dependencies

- **Python 3.7+**
- **NumPy** (≥1.21.0): Numerical computations and array operations
- **Matplotlib** (≥3.4.0): Visualization and plotting (2D and 3D)
- **SciPy** (≥1.7.0): Numerical integration for N-body simulation
- **spiceypy** (≥6.0.0): Python interface to NASA NAIF SPICE toolkit

Install all dependencies:
```bash
pip install -r requirements.txt
```

## SPICE Kernel Information

### What is SPICE?

SPICE (Spacecraft Planet Instrument C-matrix Events) is NASA's toolkit for computing geometric information for space science missions. SPK (Spacecraft and Planet Kernel) files contain ephemeris data - precise position and velocity information over time.

### Obtaining Additional Kernels

- **NAIF Website**: https://naif.jpl.nasa.gov/naif/
- **Generic Kernels**: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/
- **Small Body Database**: https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html

To use a different comet, update the `COMET_KERNEL` and `COMET_ID` variables in `comet_atlas_spice.py`.

## References

### Comets and Interstellar Objects
- [1I/'Oumuamua](https://en.wikipedia.org/wiki/%CA%BBOumuamua) - First confirmed interstellar object
- [2I/Borisov](https://en.wikipedia.org/wiki/2I/Borisov) - First confirmed interstellar comet
- [ATLAS Survey](https://atlas.fallingstar.com/) - Asteroid Terrestrial-impact Last Alert System

### SPICE Documentation
- [NAIF SPICE Toolkit](https://naif.jpl.nasa.gov/naif/)
- [spiceypy Documentation](https://spiceypy.readthedocs.io/)
- [SPICE Tutorial](https://naif.jpl.nasa.gov/naif/tutorials.html)

## License

Open source project for educational and research purposes.
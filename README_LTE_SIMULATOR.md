# Lunar Surface 4G LTE Communication Simulator

A Python application for simulating 4G LTE wireless communication coverage on the lunar surface, accounting for topography, line-of-sight constraints, and RF propagation characteristics in the lunar vacuum environment.

## Features

- **Topography-Aware Analysis**: Uses Digital Elevation Models (DEM) from lunar surface imagery
- **Line-of-Sight Calculations**: Accounts for terrain blocking and shadowing
- **Fresnel Zone Clearance**: Evaluates first Fresnel zone obstructions
- **Knife-Edge Diffraction**: Models signal diffraction over terrain obstacles
- **Free-Space Path Loss**: Calculates propagation loss in lunar vacuum (no atmospheric absorption)
- **4G LTE Parameters**: Configurable transmitter and receiver parameters
- **Coverage Visualization**: Generates detailed coverage maps and analysis plots
- **GeoTIFF Export**: Exports results as georeferenced raster files

## Physical Model

### Propagation Environment

The simulator models RF propagation in the lunar environment, which differs from Earth:

- **No atmosphere**: Pure free-space propagation (no atmospheric absorption)
- **No ionosphere**: No ionospheric refraction or scattering
- **Vacuum propagation**: Speed of light = 299,792,458 m/s exactly
- **Terrain dominates**: Line-of-sight and diffraction are primary factors

### Key Equations

#### Free-Space Path Loss (FSPL)
```
FSPL(dB) = 20·log₁₀(d) + 20·log₁₀(f) + 32.45
```
where:
- d = distance in kilometers
- f = frequency in MHz

#### First Fresnel Zone Radius
```
r₁ = √(λ · d₁ · d₂ / (d₁ + d₂))
```
where:
- λ = wavelength
- d₁ = distance from transmitter to point
- d₂ = distance from point to receiver

#### Knife-Edge Diffraction Loss
Uses ITU-R P.526 model:
```
v = h · √(2(d₁ + d₂) / (λ · d₁ · d₂))
L(dB) = 6.9 + 20·log₁₀(√((v-0.1)² + 1) + v - 0.1)  for v > -0.78
```
where h is the obstacle height above the line-of-sight path.

#### Link Budget
```
P_rx(dBm) = P_tx(dBm) + G_tx(dBi) + G_rx(dBi) - PathLoss(dB) - AdditionalLosses(dB)
```

## Installation

### Requirements

- Python 3.8 or higher
- NumPy
- SciPy
- Rasterio (for GeoTIFF handling)
- Matplotlib (for visualization)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy scipy rasterio matplotlib
```

## Usage

### Basic Usage

```python
from lunar_lte_simulator import LunarLTESimulator, TransmitterConfig

# Configure transmitter
config = TransmitterConfig(
    lat=-89.5,              # Latitude (degrees)
    lon=0.0,                # Longitude (degrees)
    height_above_ground=10.0,  # Antenna height (m)
    frequency_mhz=2600.0,   # 4G LTE frequency
    transmit_power_dbm=46.0,   # TX power (40W)
    antenna_gain_dbi=18.0,  # Antenna gain
    max_range_km=20.0,      # Analysis radius
    resolution_m=100.0      # Grid resolution
)

# Create simulator
simulator = LunarLTESimulator("path/to/dem.tif", config)

# Run analysis
simulator.run_analysis(verbose=True)

# Visualize results
simulator.plot_results(save_path="coverage_map.png")

# Export results
simulator.export_results("output_prefix")
```

### Running Examples

The repository includes several example scenarios:

```bash
python example_usage.py
```

This provides an interactive menu with:
1. Basic coverage analysis (15 km, standard parameters)
2. High-power long-range (50 km, 100W transmitter)
3. Low-power local network (5 km, 1W transmitter)
4. Custom parameters

## Configuration Parameters

### Transmitter Location
- `lat`: Latitude in degrees
- `lon`: Longitude in degrees
- `height_above_ground`: Antenna height above local terrain (meters)

### RF Parameters
- `frequency_mhz`: Operating frequency in MHz (default: 2600 for LTE Band 7)
- `transmit_power_dbm`: Transmitter power in dBm (46 dBm = 40W)
- `antenna_gain_dbi`: Transmitter antenna gain in dBi
- `antenna_tilt_deg`: Antenna tilt from horizontal (degrees)

### Receiver Parameters
- `receiver_sensitivity_dbm`: Minimum detectable signal (default: -110 dBm)
- `receiver_gain_dbi`: Receiver antenna gain in dBi

### Analysis Parameters
- `max_range_km`: Maximum analysis radius in kilometers
- `resolution_m`: Analysis grid resolution in meters

### Propagation Parameters
- `polarization`: 'vertical' or 'horizontal'
- `include_diffraction`: Enable/disable knife-edge diffraction modeling

## 4G LTE Frequency Bands

Common 4G LTE bands that could be used on the Moon:

| Band | Frequency (MHz) | Use Case |
|------|----------------|----------|
| Band 7 | 2500-2690 | High capacity, medium range |
| Band 3 | 1710-1880 | Balanced coverage/capacity |
| Band 20 | 791-821 | Extended range, lower capacity |
| Band 28 | 703-803 | Maximum range |

Lower frequencies provide better range but lower data rates.
Higher frequencies provide higher data rates but shorter range.

## Output Files

### Plots
The `plot_results()` function generates a 4-panel visualization:
1. **Topography**: Lunar surface elevation with transmitter location
2. **Line-of-Sight**: Areas with direct LOS to transmitter
3. **Received Signal Strength**: Signal power in dBm across the area
4. **Coverage Area**: Binary coverage mask (adequate signal yes/no)

### GeoTIFF Exports
The `export_results()` function creates:
- `*_received_power.tif`: Received signal strength (dBm) at each location
- `*_coverage.tif`: Binary coverage mask (1 = covered, 0 = not covered)

Both files are georeferenced and can be opened in GIS software (QGIS, ArcGIS, etc.).

## Example Scenarios

### Scenario 1: Artemis Base Station
```python
config = TransmitterConfig(
    lat=-89.5, lon=0.0,
    height_above_ground=15.0,
    frequency_mhz=1800.0,  # Extended range
    transmit_power_dbm=50.0,  # 100W
    antenna_gain_dbi=21.0,
    max_range_km=50.0
)
```
**Use case**: Main communication hub for lunar base

### Scenario 2: Rover-to-Rover Communication
```python
config = TransmitterConfig(
    lat=-89.5, lon=0.0,
    height_above_ground=2.0,  # Rover height
    frequency_mhz=2600.0,
    transmit_power_dbm=30.0,  # 1W
    antenna_gain_dbi=3.0,  # Omnidirectional
    max_range_km=5.0
)
```
**Use case**: Direct communication between surface vehicles

### Scenario 3: EVA Suit Communication
```python
config = TransmitterConfig(
    lat=-89.5, lon=0.0,
    height_above_ground=2.0,  # Astronaut height
    frequency_mhz=2600.0,
    transmit_power_dbm=23.0,  # 200mW
    antenna_gain_dbi=0.0,  # Omnidirectional
    max_range_km=2.0
)
```
**Use case**: Astronaut-to-base communication during EVA

## Limitations and Assumptions

1. **Coordinates**: Assumes input coordinates are in the same CRS as the DEM
2. **Antenna Pattern**: Assumes isotropic radiation (omnidirectional)
3. **Polarization**: Assumes matched polarization between TX and RX
4. **Surface Roughness**: Does not model small-scale surface scattering
5. **Multi-path**: Does not include ground reflections (could be significant on Moon)
6. **Thermal Noise**: Uses standard thermal noise floor assumptions
7. **Regolith Properties**: Does not model regolith electromagnetic properties

## Future Enhancements

Potential improvements for more accurate modeling:

- [ ] Two-ray ground reflection model
- [ ] Antenna pattern modeling (directional beams)
- [ ] Multiple transmitter optimization
- [ ] Network capacity analysis
- [ ] Doppler shift for moving receivers
- [ ] Integration with SPICE kernels for spacecraft positions
- [ ] PSR (Permanently Shadowed Region) special handling
- [ ] Lunar regolith dielectric property modeling

## Data Sources

This simulator works with:
- **Lunar Reconnaissance Orbiter (LRO)** DEMs
- **ShadowCam** 60m resolution imagery (SDC60_COG files)
- Any GeoTIFF format lunar elevation data

## References

1. ITU-R P.526: "Propagation by diffraction"
2. ITU-R P.525: "Calculation of free-space attenuation"
3. Rappaport, T. S. (2001). "Wireless Communications: Principles and Practice"
4. NASA Lunar Communications and Navigation studies

## Contributing

Feel free to extend this simulator with:
- Additional propagation models
- Network optimization algorithms
- Integration with mission planning tools
- Performance improvements for large-scale analysis

## License

This is a research/educational tool. Modify and use as needed for your lunar communication planning!

## Contact

For questions about lunar communication system design, consult:
- NASA Space Communications and Navigation (SCaN)
- Lunar exploration mission planning teams

---

**Note**: This simulator is for planning and research purposes. Actual lunar communication systems require detailed engineering analysis and testing.

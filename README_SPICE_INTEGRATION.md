# Lunar Communication SPICE Integration

Extension to the Lunar LTE Simulator adding SPICE-based analysis for:
- **Direct-to-Earth (DTE)** communication via Deep Space Network
- **Surface asset tracking** and multi-point communication
- **Earth visibility windows** from lunar surface locations
- **Integrated multi-link analysis** combining surface and DTE

## Features

### 1. Earth Visibility Analysis
- Calculate when Earth is visible from any lunar surface location
- Account for lunar horizon constraints
- Generate visibility windows over arbitrary time periods
- Compute elevation and azimuth angles to Earth

### 2. Direct-to-Earth Links
- Link budget calculations to DSN stations (Goldstone, Canberra, Madrid)
- Support for 70m and 34m antenna configurations
- X-band and S-band frequency support
- Real Earth-Moon geometry using SPICE ephemeris

### 3. Surface Asset Tracking
- Multi-asset communication analysis
- Link budgets between base station and surface assets (rovers, landers, relays)
- Geometric line-of-sight calculations
- Support for different asset types and configurations

### 4. Integrated Analysis
- Combine surface coverage maps with DTE windows
- Multi-link scheduling and optimization
- Comprehensive reporting and visualization

## New Files

### Core Modules

1. **lunar_comm_spice.py** - SPICE-based communication analysis
   - `LunarCommSPICE` class for Earth visibility and DTE
   - `SurfaceAsset` dataclass for asset configuration
   - `DSNStation` definitions for all major DSN antennas

2. **integrated_comm_analysis.py** - Combined analysis framework
   - `IntegratedCommAnalysis` class
   - Merges LTE surface coverage with SPICE-based DTE
   - Multi-asset and multi-link visualization

3. **example_spice_analysis.py** - Interactive examples
   - Earth visibility windows
   - DTE link budget calculations
   - Surface asset communication
   - Full integrated analysis

## Installation

### Additional Dependencies

```bash
pip install spiceypy>=6.0.0
```

Or update from requirements.txt:
```bash
pip install -r requirements.txt
```

### SPICE Kernels

The simulator looks for SPICE kernels in the `kernels/` directory. Your repository already contains:
- `de430.bsp` - JPL planetary ephemeris
- `kplo_*.bsp` - Korea Pathfinder Lunar Orbiter kernels

For full functionality, you may want to add:
- Leap seconds kernel (LSK): `naif0012.tls` or similar
- Planetary constants (PCK): `pck00010.tpc`
- Moon orientation frames (FK): `moon_080317.tf` or `moon_pa_de421_1900-2050.bpc`

Download from: https://naif.jpl.nasa.gov/naif/data.html

**Note**: The simulator will work with approximations if kernels are missing.

## Usage

### Quick Start - Earth Visibility

```python
from lunar_comm_spice import LunarCommSPICE

# Initialize
comm = LunarCommSPICE(kernel_dir='kernels')

# Find visibility windows
vis_data = comm.find_earth_visibility_windows(
    tx_lat=-89.5,  # Lunar south pole
    tx_lon=0.0,
    tx_alt=10.0,
    start_time="2025-11-22T00:00:00",
    duration_hours=720,  # 30 days
    time_step_minutes=30
)

# Display results
windows = vis_data['windows']
print(f"Found {len(windows)} visibility windows")

# Plot
comm.plot_earth_visibility(vis_data, save_path="earth_visibility.png")
```

### DTE Link Budget

```python
from lunar_comm_spice import LunarCommSPICE
import spiceypy as spice

comm = LunarCommSPICE(kernel_dir='kernels')

# Select DSN station
station = comm.DSN_STATIONS['Goldstone']

# Calculate link at specific time
et = spice.str2et("2025-11-22T12:00:00")

link = comm.calculate_dte_link_budget(
    tx_lat=-89.5,
    tx_lon=0.0,
    tx_alt=10.0,
    tx_power_dbm=50.0,      # 100W
    tx_gain_dbi=30.0,       # High-gain to Earth
    frequency_mhz=8450.0,   # X-band
    et=et,
    dsn_station=station
)

print(f"Distance: {link['distance_km']:.0f} km")
print(f"RX Power: {link['rx_power_dbm']:.1f} dBm")
print(f"Link Margin: {link['link_margin_db']:.1f} dB")
print(f"Link Available: {link['link_available']}")
```

### Surface Asset Communication

```python
from lunar_comm_spice import LunarCommSPICE, SurfaceAsset

comm = LunarCommSPICE(kernel_dir='kernels')

# Define asset
rover = SurfaceAsset(
    name="Rover-Alpha",
    lat=-89.3,
    lon=10.0,
    altitude=2.0,
    receiver_sensitivity_dbm=-115.0,
    antenna_gain_dbi=8.0
)

# Check link
link = comm.check_asset_link(
    tx_lat=-89.5, tx_lon=0.0, tx_alt=10.0,
    asset=rover,
    tx_power_dbm=40.0,      # 10W
    tx_gain_dbi=12.0,
    frequency_mhz=2400.0    # S-band
)

print(f"Distance: {link['distance_km']:.2f} km")
print(f"Link Margin: {link['link_margin_db']:.1f} dB")
```

### Integrated Analysis

```python
from integrated_comm_analysis import IntegratedCommAnalysis
from lunar_lte_simulator import TransmitterConfig
from lunar_comm_spice import SurfaceAsset

# Configure base station
config = TransmitterConfig(
    lat=-89.5, lon=0.0,
    height_above_ground=10.0,
    frequency_mhz=2600.0,
    transmit_power_dbm=46.0,
    antenna_gain_dbi=18.0,
    max_range_km=20.0,
    resolution_m=150.0
)

# Create analyzer
analyzer = IntegratedCommAnalysis(
    dem_path="SDC60_COG/M012728826S.60m.COG.tif",
    tx_config=config,
    kernel_dir='kernels'
)

# Add assets
analyzer.add_surface_asset(SurfaceAsset(
    name="Rover-1", lat=-89.3, lon=10.0, altitude=2.0
))

# Run complete analysis
analyzer.analyze_surface_coverage()
analyzer.analyze_asset_links()
analyzer.analyze_dte_windows(
    start_time="2025-11-22T00:00:00",
    duration_hours=240
)
analyzer.analyze_dte_link_budget()

# Generate outputs
analyzer.plot_integrated_coverage(save_path="integrated.png")
analyzer.generate_report("report.txt")
```

## Running Examples

### Interactive Menu
```bash
python example_spice_analysis.py
```

This provides:
1. Earth visibility windows
2. DTE link budget to DSN
3. Surface asset communication
4. Full integrated analysis

### Run Standalone Modules
```bash
# SPICE analysis only
python lunar_comm_spice.py

# Integrated analysis
python integrated_comm_analysis.py
```

## Deep Space Network (DSN) Stations

### 70-meter Antennas (High Performance)

| Station | Location | Coordinates | Frequency Coverage |
|---------|----------|-------------|-------------------|
| DSS-14 Goldstone | California, USA | 35.43°N, 116.89°W | S/X/Ka-band |
| DSS-43 Canberra | Australia | 35.40°S, 148.98°E | S/X/Ka-band |
| DSS-63 Madrid | Spain | 40.43°N, 4.25°W | S/X/Ka-band |

**Capabilities**:
- RX Gain: ~74 dBi @ X-band
- Sensitivity: ~-160 dBm
- Used for deep space missions

### 34-meter Antennas (Standard)

Each complex has multiple 34m dishes for operational flexibility.

**Capabilities**:
- RX Gain: ~68 dBi @ X-band
- Sensitivity: ~-150 dBm
- Used for near-Earth and lunar missions

## Frequency Bands for Lunar Communications

### Surface-to-Surface (LTE)
- **S-band** (2-4 GHz): Best for surface mobility, good range
- **UHF** (400-900 MHz): Maximum range, lower data rate
- **L-band** (1-2 GHz): Balance of range and capacity

### Direct-to-Earth (DTE)
- **X-band** (8-12 GHz): Standard for deep space, high data rate
  - Uplink: 7.1-7.2 GHz
  - Downlink: 8.4-8.5 GHz
- **Ka-band** (32-34 GHz): Very high data rate, weather sensitive
- **S-band** (2-4 GHz): Backup, lower data rate, more robust

## Link Budget Equations

### Free-Space Path Loss (Vacuum)
```
FSPL(dB) = 20·log₁₀(d_km) + 20·log₁₀(f_MHz) + 32.45
```

### Received Power
```
P_rx(dBm) = P_tx(dBm) + G_tx(dBi) + G_rx(dBi) - FSPL(dB)
```

### Link Margin
```
Margin(dB) = P_rx(dBm) - Sensitivity(dBm)
```

**Required margins**:
- Minimum: 3 dB (marginal)
- Good: 6-10 dB (reliable)
- Excellent: >15 dB (robust)

## Earth-Moon Geometry

### Earth Visibility from Lunar Poles

**South Pole Region (-85° to -90°)**:
- Earth visibility highly dependent on exact location
- Near true pole: Very limited or no visibility
- At -89°: Periodic visibility with low elevation angles
- Relay stations or high antennas may be needed

**North Pole Region (+85° to +90°)**:
- Similar constraints to south pole
- Earth appears at low elevation when visible

### Key Factors
1. **Libration**: Apparent wobble of Moon (±6.5°)
2. **Local terrain**: Crater rims and mountains block low-angle views
3. **Antenna height**: Critical for horizon clearance

## Output Files

### From SPICE Analysis

1. **earth_visibility.png** - Multi-panel visibility plot
   - Visibility timeline
   - Elevation angle over time
   - Azimuth angle over time

2. **Integrated coverage maps** - Combined visualization
   - Surface LTE coverage
   - Asset link status
   - DTE link budget summary
   - Earth visibility timeline

3. **Text reports** - Detailed numerical results
   - Coverage statistics
   - Link budgets for all links
   - Visibility window summaries

## Application Scenarios

### Scenario 1: Artemis Base Camp (South Pole)
```python
config = TransmitterConfig(
    lat=-89.5, lon=0.0,      # Near Shackleton Crater
    height_above_ground=50.0, # Tall mast for Earth visibility
    frequency_mhz=2600.0,
    transmit_power_dbm=46.0,
    max_range_km=50.0
)
```

**Communications Architecture**:
- S-band for local surface communications (rovers, EVA)
- X-band high-gain antenna for DTE to DSN
- Relay satellite for extended surface coverage
- High antenna mast to overcome terrain blocking

### Scenario 2: Polar Exploration Network
Multiple base stations for complete coverage:
- Primary station with DTE capability
- Secondary relay stations for surface coverage
- Mobile assets (rovers) communicate via nearest station
- Scheduled DTE passes during Earth visibility windows

### Scenario 3: Commercial Lunar Operations
```python
# Multiple assets requiring coordination
assets = [
    SurfaceAsset("Mining-Rover-1", lat=-89.2, lon=15.0),
    SurfaceAsset("Processing-Plant", lat=-89.6, lon=-10.0),
    SurfaceAsset("Landing-Pad", lat=-89.4, lon=5.0),
]

# Optimize communication schedule
# - Local traffic uses S-band surface network
# - Earth uplinks scheduled during visibility windows
# - Data buffering during Earth occultation
```

## Advanced Features

### Time-Varying Analysis
```python
# Analyze how links change over time
for hour in range(0, 720, 6):  # Every 6 hours for 30 days
    et = et_start + hour * 3600
    link = comm.calculate_dte_link_budget(...)
    # Store and analyze link availability patterns
```

### Multi-Asset Scheduling
```python
# Find optimal communication schedule
# Prioritize assets based on:
# - Data volume requirements
# - Power constraints
# - Mission criticality
```

### Relay Optimization
```python
# Place relay stations to maximize coverage
# Consider:
# - Terrain line-of-sight
# - Earth visibility
# - Power/mass constraints
```

## Limitations and Future Work

### Current Limitations
1. **Terrain LOS**: Asset links use geometric LOS only (no DEM-based terrain blocking yet)
2. **Antenna patterns**: Assumes isotropic or simple gain models
3. **Atmospheric effects**: Not applicable (vacuum), but Earth-based DSN weather not modeled
4. **Doppler**: Not calculated for moving assets
5. **Regolith scattering**: Surface reflections not modeled

### Planned Enhancements
- [ ] Full terrain-aware LOS for asset links (integrate with DEM)
- [ ] Antenna pattern modeling (beamwidth, pointing)
- [ ] Doppler shift calculations
- [ ] Multi-path propagation (surface reflections)
- [ ] Network capacity and scheduling optimization
- [ ] Integration with spacecraft relay orbits (LRO, Gateway, etc.)
- [ ] Real-time ephemeris updates
- [ ] Machine learning for optimal station placement

## SPICE Resources

### Kernel Sources
- **NAIF**: https://naif.jpl.nasa.gov/naif/data.html
- **Generic kernels**: LSK, PCK, FK
- **Planetary ephemeris**: DE430, DE440
- **Lunar missions**: LRO, KPLO, Artemis

### Tools
- **SPICE Toolkit**: https://naif.jpl.nasa.gov/naif/toolkit.html
- **WebGeoCalc**: https://wgc.jpl.nasa.gov:8443/webgeocalc/
- **Cosmographia**: Visualization tool for SPICE data

### Documentation
- **SPICE Tutorials**: https://naif.jpl.nasa.gov/naif/tutorials.html
- **Required Reading**: Essential SPICE concepts
- **SpiceyPy Docs**: https://spiceypy.readthedocs.io/

## References

1. **DSN Documentation**:
   - "DSN Telecommunications Link Design Handbook" (810-005)
   - https://deepspace.jpl.nasa.gov/

2. **Lunar Communications**:
   - NASA LunaNet Interoperability Specification
   - Lunar Exploration Architecture Studies

3. **RF Propagation**:
   - ITU-R P.525: Free-space attenuation
   - ITU-R P.526: Diffraction

4. **Mission Design**:
   - "Lunar Surface Communications System Architecture" (NASA/TM-2019)
   - Artemis communications and navigation concepts

## Contributing

Areas for contribution:
- Additional SPICE kernels for lunar missions
- Improved propagation models
- Network optimization algorithms
- Integration with mission planning tools
- GUI development
- Performance optimization for large-scale analysis

## License

Research and educational tool. Use and modify as needed.

## Support

For questions:
- SPICE: naif@jpl.nasa.gov
- Lunar communications: NASA SCaN
- This simulator: See main repository

---

**Note**: This is a planning and analysis tool. Actual mission implementation requires detailed engineering analysis, testing, and verification.

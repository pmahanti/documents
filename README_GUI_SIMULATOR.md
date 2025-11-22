# Lunar Communication Simulator - GUI Application

**Comprehensive Web-Based Interface for Lunar Surface Communication Analysis**

## Overview

The Lunar Communication Simulator provides a complete web-based GUI for analyzing RF communication coverage on the lunar surface. It integrates all previously developed capabilities into an intuitive, deployable application.

### Key Features

✅ **4 Operational Scenarios (ConOps)**:
- Surface TX → Surface RX (Rover/lander communications)
- Surface TX → Earth RX (Direct-to-Earth via DSN)
- Crater TX → Earth RX (Communication from PSRs)
- Rover Path → Earth RX (Moving asset DTE coverage)

✅ **9 Propagation Models**:
- Free-Space Path Loss (baseline)
- Two-Ray Ground Reflection (multipath)
- Knife-Edge Diffraction
- Crater Rim Diffraction
- Plane Earth Loss
- Egli Model (lunar adapted)
- Longley-Rice (lunar terrain)
- COST-231 Hata (lunar adapted)
- Surface Scattering

✅ **Interactive Controls**:
- Adjustable transmitter parameters (power, gain, frequency, location)
- Configurable receiver specifications
- Propagation model selection
- Surface asset management
- Rover path waypoint editor
- DTE time window configuration

✅ **Multiple Output Formats**:
- PNG (coverage maps, timelines)
- GeoTIFF (georeferenced rasters)
- CSV (tabular data for analysis)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check Streamlit installation
streamlit --version

# Test import of modules
python3 -c "import streamlit; import plotly; import rasterio; print('All imports successful!')"
```

## Running the GUI

### Local Deployment (Recommended for First Use)

```bash
# Navigate to project directory
cd /path/to/lunar-communication-simulator

# Launch the web GUI
streamlit run lunar_comm_gui.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Network Deployment

To make the GUI accessible on your local network:

```bash
streamlit run lunar_comm_gui.py --server.address 0.0.0.0 --server.port 8501
```

Access from other devices: `http://YOUR_IP_ADDRESS:8501`

### Production Deployment

For production deployment on a web server:

```bash
# With custom port and headless mode
streamlit run lunar_comm_gui.py \
  --server.port 80 \
  --server.headless true \
  --server.enableCORS false
```

## User Guide

### Workflow

1. **Select Scenario**
   - Choose from 4 operational scenarios in the sidebar
   - Each scenario optimized for specific use cases

2. **Configure Parameters** (Setup Tab)
   - Set transmitter location (lat/lon)
   - Adjust TX power, gain, frequency
   - Select propagation model
   - Configure scenario-specific parameters
   - Add surface assets or rover waypoints

3. **Run Simulation** (Run Tab)
   - Click "🚀 Run Simulation"
   - Wait for processing (10 seconds to 2 minutes)
   - Check for success message

4. **View Results** (Results Tab)
   - Examine coverage maps (interactive Plotly charts)
   - Review statistics and metrics
   - Check asset link budgets or DSN availability
   - Analyze visibility windows

5. **Download Outputs** (Results Tab)
   - Generate PNG for reports/presentations
   - Create GeoTIFF for GIS analysis
   - Export CSV for further processing

### Scenario Details

#### 1. Surface TX → Surface RX

**Purpose**: Surface-to-surface LTE communication analysis

**Use Cases**:
- Rover-to-base station communication
- Rover-to-rover relay networks
- Lander-to-rover links

**Parameters**:
- Analysis range: 1-100 km
- Grid resolution: 10-500 m
- RX sensitivity: -150 to -50 dBm
- Surface assets: up to 10 receivers

**Outputs**:
- 2D coverage map (received power)
- Path loss distribution
- Coverage percentage
- Asset-specific link budgets
- GeoTIFF rasters

**Example Configuration**:
```
Frequency: 2600 MHz (S-band LTE)
TX Power: 46 dBm (40W)
TX Gain: 18 dBi (directional)
Analysis Range: 20 km
Propagation: Two-Ray (multipath)
```

#### 2. Surface TX → Earth RX (DTE)

**Purpose**: Direct-to-Earth communication via Deep Space Network

**Use Cases**:
- Telemetry downlink from lunar base
- Science data transmission
- Command uplink verification

**Parameters**:
- Start time: Any UTC date/time
- Duration: 1-720 hours (up to 30 days)
- DTE frequency: Typically 8450 MHz (X-band)
- DTE power: 30-60 dBm (1-1000W)

**Outputs**:
- Earth visibility timeline
- DSN station link budgets (3 complexes)
- Visibility window statistics
- Elevation/azimuth profiles

**Example Configuration**:
```
Location: Shackleton Crater (-89.5°, 0.0°)
Frequency: 8450 MHz (X-band)
TX Power: 50 dBm (100W)
TX Gain: 30 dBi (Earth-pointing HGA)
Duration: 240 hours (10 days)
```

#### 3. Crater TX → Earth RX (DTE)

**Purpose**: DTE from inside crater (e.g., PSR ice mining)

**Use Cases**:
- Communication from permanently shadowed regions
- Ice prospecting missions
- Sub-surface operations

**Additional Parameters**:
- Crater depth: 10-5000 m
- Crater radius: 50-10,000 m
- TX inside crater: Yes/No

**Outputs**:
- Standard DTE analysis
- Additional crater diffraction loss
- Adjusted link margins

**Example Configuration**:
```
Crater: 500m radius, 100m depth
TX inside: Yes, 10m above floor
Frequency: 8450 MHz (X-band)
Additional Loss: ~15-30 dB (rim diffraction)
```

#### 4. Rover Path → Earth RX (DTE)

**Purpose**: Moving rover DTE coverage analysis

**Use Cases**:
- Mission planning for traverses
- Data transfer scheduling
- Science stop selection

**Parameters**:
- Waypoints: 2-20 locations (lat/lon)
- Rover speed: 0.1-10 km/h
- Mission duration: 1-720 hours
- Time resolution: 1-60 minutes

**Outputs**:
- Minute-by-minute CSV coverage data
- Coverage timeline (Earth visibility, DSN availability)
- Path map with coverage indicators
- Station handover events

**Example Configuration**:
```
Waypoints: 7 science stops
Speed: 1.5 km/h (VIPER-class)
Duration: 120 hours (5 days)
Frequency: 8450 MHz (X-band)
Output: 7,200 rows (minute-by-minute)
```

### Propagation Models

#### Free-Space (Baseline)
```
FSPL(dB) = 20·log₁₀(d_km) + 20·log₁₀(f_MHz) + 32.45
```
- **Use**: Baseline, ideal vacuum
- **Accuracy**: High for LOS, smooth terrain
- **Speed**: Fastest

#### Two-Ray Ground Reflection (Recommended)
```
Includes direct path + ground-reflected path
Phase difference from path length
Fresnel reflection coefficients
```
- **Use**: Realistic surface scenarios
- **Accuracy**: Very good for lunar surface
- **Speed**: Fast
- **Notes**: Accounts for multipath interference

#### Knife-Edge Diffraction
```
ITU-R P.526 model
v = h·√(2(d₁+d₂)/(λ·d₁·d₂))
L(dB) = 6.9 + 20·log₁₀(√((v-0.1)²+1) + v - 0.1)
```
- **Use**: Single obstacle (ridge, crater rim)
- **Accuracy**: Good for isolated obstacles
- **Speed**: Fast

#### Crater Diffraction
```
Models crater rim as knife-edge
Calculates additional loss for TX inside
```
- **Use**: PSR communication
- **Accuracy**: Approximate (simplified geometry)
- **Speed**: Fast

### Parameter Ranges

| Parameter | Min | Max | Typical | Units |
|-----------|-----|-----|---------|-------|
| Frequency | 100 | 30,000 | 2,600 / 8,450 | MHz |
| TX Power | 0 | 60 | 40-50 | dBm |
| TX Gain | 0 | 50 | 12-30 | dBi |
| TX Height | 0.1 | 100 | 2-20 | m |
| Range (S2S) | 1 | 100 | 10-50 | km |
| Duration (DTE) | 1 | 720 | 48-240 | hours |
| Rover Speed | 0.1 | 10 | 0.5-3 | km/h |

### Tunable Controls

#### Transmitter
- ✓ Location (lat/lon, anywhere on Moon)
- ✓ Height (antenna mast height)
- ✓ Power (0.001W to 1kW)
- ✓ Gain (omnidirectional to high-gain dish)
- ✓ Frequency (VHF to Ka-band)

#### Propagation
- ✓ Model selection (9 models)
- ✓ Multipath enable/disable
- ✓ Diffraction enable/disable
- ✓ Surface roughness (0-1m RMS)

#### Analysis
- ✓ Range/duration
- ✓ Resolution/time step
- ✓ Receiver sensitivity
- ✓ Start time (UTC)

#### Assets
- ✓ Number of assets (0-10)
- ✓ Per-asset location, height, gain, sensitivity
- ✓ Rover waypoints (2-20)
- ✓ Rover speed

#### Crater (Crater scenario)
- ✓ Depth (10-5000m)
- ✓ Radius (50-10,000m)
- ✓ TX inside/outside

## Architecture

### Component Structure

```
lunar_comm_gui.py           # Streamlit web interface
├── simulation_engine.py    # Backend simulation engine
│   ├── lunar_lte_simulator.py      # Surface-to-surface LTE
│   ├── lunar_comm_spice.py         # DTE and Earth visibility
│   ├── rover_path_dte_coverage.py  # Rover path analysis
│   └── propagation_models.py       # RF propagation models
└── output_manager.py       # PNG/GeoTIFF/CSV generation
```

### Data Flow

```
User Input (GUI)
  ↓
SimulationConfig
  ↓
LunarCommSimulationEngine
  ↓
Scenario-Specific Simulator
  ↓
Propagation Models
  ↓
Results Dictionary
  ↓
OutputManager → Files (PNG/GeoTIFF/CSV)
  ↓
User Downloads
```

## Output Formats

### PNG Images

**Surface-to-Surface**:
- 4-panel plot:
  - Received signal strength (heatmap)
  - Path loss distribution
  - Coverage map (binary)
  - Statistics table

**DTE**:
- 3-panel plot:
  - Visibility timeline
  - Elevation angle profile
  - Azimuth angle profile

**Rover Path**:
- 3-panel plot:
  - Earth visibility
  - DSN station availability
  - Distance traveled

**Specifications**:
- Format: PNG
- Resolution: 300 DPI
- Size: ~500-2000 KB
- Suitable for: Reports, presentations

### GeoTIFF Rasters

**Available For**: Surface-to-surface scenario

**Layers**:
- Received power (dBm) - Float32
- Path loss (dB) - Float32
- Coverage mask - UInt8

**Specifications**:
- CRS: WGS84 lat/lon
- Compression: LZW
- NoData: -9999 (power/loss), 255 (mask)
- Metadata: Scenario, timestamp, parameters

**Use Cases**:
- Import to QGIS, ArcGIS
- Overlay with lunar basemaps
- Spatial analysis
- Mission planning overlays

### CSV Tables

**Surface-to-Surface**:
- Summary table with parameters and statistics

**DTE**:
- Summary table with visibility windows

**Rover Path**:
- Minute-by-minute timeline:
  - Timestamp (UTC)
  - Rover position (lat/lon)
  - Distance traveled
  - Earth visibility
  - DSN station availability
  - Link margins

**Specifications**:
- Format: Standard CSV (UTF-8)
- Delim iter: Comma
- Decimal: Period
- Headers: Descriptive names

## Performance

### Processing Times

| Scenario | Grid/Duration | Time |
|----------|---------------|------|
| Surface (10km, 100m res) | 100×100 grid | ~10 sec |
| Surface (50km, 200m res) | 250×250 grid | ~60 sec |
| DTE (10 days, 30min step) | 480 samples | ~15 sec |
| DTE (30 days, 10min step) | 4,320 samples | ~90 sec |
| Rover (5 days, 1min step) | 7,200 samples | ~120 sec |

### Memory Usage

- Typical: 200-500 MB
- Peak (large grids): 1-2 GB
- CSV export: 500 bytes/record

### Recommendations

**For Quick Preview**:
- Coarser resolution (200-500m)
- Shorter duration (24-48 hrs)
- Larger time steps (30-60 min)

**For Final Analysis**:
- Fine resolution (50-100m)
- Full duration
- Minute-level time steps

## Troubleshooting

### GUI Won't Launch

**Problem**: `streamlit: command not found`

**Solution**:
```bash
pip install streamlit
# or
pip install -r requirements.txt
```

---

**Problem**: Port 8501 already in use

**Solution**:
```bash
streamlit run lunar_comm_gui.py --server.port 8502
```

### Simulation Errors

**Problem**: "SPICE not initialized"

**Solution**:
- Ensure `kernels/` directory exists
- Check that `.bsp` files are present
- DTE scenarios require SPICE kernels

---

**Problem**: "No DEM files found"

**Solution**:
- Surface-to-surface can run without DEM (uses analytical models)
- For terrain-aware analysis, place DEM in `SDC60_COG/` directory
- Set DEM path in config

---

**Problem**: Simulation very slow

**Solution**:
- Reduce grid resolution
- Decrease analysis range
- Use coarser time steps
- Select faster propagation model (free-space)

### Output Issues

**Problem**: GeoTIFF not displaying correctly

**Solution**:
- Check CRS in GIS software (should be WGS84)
- Verify NoData values are set correctly
- Some values may be inf (filtered in output)

---

**Problem**: CSV too large

**Solution**:
- Increase time step (5-10 minutes instead of 1)
- Reduce mission duration
- Use summary statistics instead

## Advanced Usage

### Custom Scenarios

Modify `simulation_engine.py` to add custom scenarios:

```python
# Add to create_example_config()
elif scenario == 'my_scenario':
    return SimulationConfig(
        scenario='my_scenario',
        # ... custom parameters
    )
```

### Batch Processing

Run multiple simulations programmatically:

```python
from simulation_engine import LunarCommSimulationEngine, SimulationConfig

configs = [config1, config2, config3]

for i, config in enumerate(configs):
    engine = LunarCommSimulationEngine(config)
    results = engine.run_simulation()
    # Process results
```

### Integration with Other Tools

Export results to:
- **MATLAB**: Use CSV exports
- **Python/Jupyter**: Import `simulation_engine` directly
- **GIS**: Use GeoTIFF exports
- **Excel**: Open CSV files

## Development

### Adding New Propagation Models

1. Add model to `propagation_models.py`:
```python
@staticmethod
def my_new_model(params: PropagationParameters) -> float:
    # Implementation
    return path_loss_db
```

2. Add to `list_available_models()`
3. Add description to `get_model_description()`
4. Update GUI dropdown

### Adding New Outputs

1. Add generator to `output_manager.py`:
```python
def save_my_format(self, results: Dict) -> str:
    # Implementation
    return output_path
```

2. Add button to GUI results display
3. Update `generate_all_outputs()` if needed

## References

### Propagation Models
- ITU-R P.525: Free-space attenuation
- ITU-R P.526: Diffraction
- ITU-R P.1546: Point-to-area prediction
- Two-ray ground reflection model

### Lunar Communications
- NASA DSN Telecommunications Link Design Handbook
- LunaNet Interoperability Specification
- Artemis communications architecture

### Software
- Streamlit: https://streamlit.io/
- Plotly: https://plotly.com/
- SPICE Toolkit: https://naif.jpl.nasa.gov/

## Support

For issues, questions, or contributions:
- Check troubleshooting section above
- Review example scenarios in Setup tab
- See main repository documentation

## License

Research and educational tool. See main repository for license details.

---

**Version**: 1.0.0
**Last Updated**: 2025
**Developed for**: Lunar surface communication analysis and mission planning

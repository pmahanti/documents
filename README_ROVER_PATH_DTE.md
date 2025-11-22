# Rover Path DTE Coverage Analysis

Analyzes Direct-to-Earth (DTE) communication coverage for a rover traveling along a planned path on the lunar surface. Generates **minute-by-minute CSV output** showing DSN station availability at each point along the rover's trajectory.

## Overview

This tool simulates a rover mission and calculates:
- **When** Earth is visible from each location
- **Which** DSN stations can communicate at each time
- **Where** coverage gaps occur along the path
- **How** link quality varies with position and time

Perfect for mission planning, communication scheduling, and traverse optimization.

## Key Features

### ✓ Minute-by-Minute Analysis
- Samples coverage every minute (configurable)
- Tracks rover position as it moves along path
- Calculates Earth visibility at each timestep
- Evaluates link budget to all DSN stations

### ✓ Realistic Rover Motion
- Interpolates between waypoints
- Accounts for rover speed
- Handles multi-day traverses
- Supports stationary science stops

### ✓ Complete DSN Coverage
- Analyzes all 3 DSN complexes (Goldstone, Canberra, Madrid)
- Identifies best station at each moment
- Tracks station handovers
- Calculates link margins

### ✓ CSV Output
- One row per minute
- Complete coverage data
- Easy to import to Excel, MATLAB, Python
- Ready for mission planning tools

## Quick Start

### Basic Example

```python
from rover_path_dte_coverage import RoverPathDTEAnalyzer

# Define rover path (waypoints as lat/lon)
waypoints = [
    (-89.50, 0.00),   # Start
    (-89.55, 5.00),   # Waypoint 1
    (-89.60, 10.00),  # Waypoint 2
    (-89.50, 0.00),   # Return to start
]

# Create analyzer
analyzer = RoverPathDTEAnalyzer(kernel_dir='kernels')

# Run analysis
coverage_df = analyzer.analyze_coverage(
    waypoints=waypoints,
    start_time="2026-03-15T06:00:00",
    duration_hours=48,  # 2-day mission
    rover_speed_kmh=1.5,
    time_step_minutes=1.0  # Every minute
)

# Save to CSV
analyzer.save_csv("rover_coverage.csv")

# Print summary
analyzer.print_summary()
```

### Run Example
```bash
# Simple example
python rover_path_dte_coverage.py

# Realistic VIPER mission test
python test_rover_path_coverage.py
```

## CSV Output Format

### Columns Included

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | String | UTC time (YYYY-MM-DD HH:MM:SS) |
| `elapsed_minutes` | Float | Minutes since mission start |
| `rover_lat` | Float | Rover latitude (degrees) |
| `rover_lon` | Float | Rover longitude (degrees) |
| `distance_traveled_km` | Float | Cumulative distance (km) |
| `earth_visible` | Boolean | Is Earth above horizon? |
| `earth_elevation_deg` | Float | Earth elevation angle (°) |
| `earth_azimuth_deg` | Float | Earth azimuth angle (°) |
| `goldstone_available` | Boolean | Goldstone link available? |
| `goldstone_margin_db` | Float | Goldstone link margin (dB) |
| `goldstone_distance_km` | Float | Distance to Earth (km) |
| `canberra_available` | Boolean | Canberra link available? |
| `canberra_margin_db` | Float | Canberra link margin (dB) |
| `canberra_distance_km` | Float | Distance to Earth (km) |
| `madrid_available` | Boolean | Madrid link available? |
| `madrid_margin_db` | Float | Madrid link margin (dB) |
| `madrid_distance_km` | Float | Distance to Earth (km) |
| `num_stations_available` | Int | Count of available stations |
| `best_station` | String | Station with best margin |
| `best_margin_db` | Float | Best link margin (dB) |
| `any_dsn_available` | Boolean | Any station available? |

### Example CSV Data

```csv
timestamp,elapsed_minutes,rover_lat,rover_lon,distance_traveled_km,earth_visible,earth_elevation_deg,earth_azimuth_deg,goldstone_available,goldstone_margin_db,canberra_available,madrid_available,num_stations_available,best_station,best_margin_db,any_dsn_available
2026-03-15 06:00:00,0.0,-89.500000,0.000000,0.000,True,8.45,234.12,True,18.7,False,True,2,Goldstone,18.7,True
2026-03-15 06:01:00,1.0,-89.500083,0.002500,0.025,True,8.46,234.15,True,18.8,False,True,2,Goldstone,18.8,True
2026-03-15 06:02:00,2.0,-89.500167,0.005000,0.050,True,8.47,234.18,True,18.9,False,True,2,Goldstone,18.9,True
...
```

## Configuration Parameters

### Mission Parameters

```python
analyzer.analyze_coverage(
    # Path definition
    waypoints=waypoints,              # List of (lat, lon) tuples

    # Time parameters
    start_time="2026-03-15T06:00:00", # Mission start (ISO format)
    duration_hours=72,                 # Mission duration (hours)

    # Rover characteristics
    rover_speed_kmh=1.5,              # Rover speed (km/h)
    rover_antenna_height=2.5,         # Antenna height (m)

    # RF parameters
    tx_power_dbm=40.0,                # TX power (dBm)
    tx_gain_dbi=20.0,                 # TX antenna gain (dBi)
    frequency_mhz=8450.0,             # Frequency (MHz)

    # Analysis parameters
    time_step_minutes=1.0,            # Sampling rate (minutes)
    interpolate_path=True,            # Interpolate waypoints?
    path_samples=100                  # Interpolation samples
)
```

### Typical Rover Configurations

#### VIPER-class Rover
```python
rover_speed_kmh = 0.8          # Slow science pace
rover_antenna_height = 2.5     # Mast-mounted HGA
tx_power_dbm = 43.0            # 20W SSPA
tx_gain_dbi = 25.0             # High-gain steerable dish
frequency_mhz = 8450.0         # X-band
```

#### Fast Scout Rover
```python
rover_speed_kmh = 3.0          # Fast traverse
rover_antenna_height = 1.5     # Short mast
tx_power_dbm = 37.0            # 5W transmitter
tx_gain_dbi = 15.0             # Medium-gain antenna
frequency_mhz = 2200.0         # S-band
```

#### Heavy Cargo Rover
```python
rover_speed_kmh = 0.5          # Very slow
rover_antenna_height = 3.0     # Tall mast
tx_power_dbm = 46.0            # 40W transmitter
tx_gain_dbi = 30.0             # Large dish
frequency_mhz = 8450.0         # X-band
```

## Use Cases

### 1. Mission Planning

**Question**: When can we send data during the traverse?

```python
# Analyze full mission
coverage_df = analyzer.analyze_coverage(...)

# Find communication windows
windows = coverage_df[coverage_df['any_dsn_available']]

# Schedule data downlinks
print(f"Available for communication: {len(windows)} out of {len(coverage_df)} minutes")
```

### 2. Traverse Optimization

**Question**: Which path provides best Earth visibility?

```python
# Test multiple paths
paths = [path_A, path_B, path_C]
results = []

for path in paths:
    df = analyzer.analyze_coverage(waypoints=path, ...)
    coverage_pct = 100 * df['any_dsn_available'].sum() / len(df)
    results.append(coverage_pct)

# Pick best path
best_path = paths[np.argmax(results)]
```

### 3. Science Stop Planning

**Question**: Where should we stop for extended science operations?

```python
# Analyze coverage at each waypoint
for i, waypoint in enumerate(waypoints):
    # Check coverage at this location
    single_point = [waypoint, waypoint]  # Stationary
    df = analyzer.analyze_coverage(
        waypoints=single_point,
        duration_hours=24  # 1-day science stop
    )

    coverage_pct = 100 * df['any_dsn_available'].sum() / len(df)
    print(f"Waypoint {i}: {coverage_pct:.1f}% DSN coverage")
```

### 4. Communication Scheduling

**Question**: When should we schedule high-bandwidth data transfers?

```python
# Find periods with best link margin
df = analyzer.coverage_data
best_periods = df[df['best_margin_db'] > 20.0]  # >20 dB margin

# Extract time windows
for i, row in best_periods.iterrows():
    print(f"{row['timestamp']}: {row['best_station']} @ {row['best_margin_db']:.1f} dB")
```

## Test Case: VIPER Mission

The included test simulates a realistic 5-day VIPER-style ice prospecting mission:

**Mission Profile**:
- **Duration**: 5 days (120 hours)
- **Path**: 15 waypoints covering ~30 km
- **Speed**: 0.8 km/h (science pace)
- **Stops**: Multiple PSR investigations
- **DTE**: 20W X-band transmitter

**Run Test**:
```bash
python test_rover_path_coverage.py
```

**Outputs**:
1. **viper_mission_dte_coverage.csv** - Full minute-by-minute data
2. **viper_mission_coverage_timeline.png** - 4-panel visualization
3. **viper_mission_coverage_map.png** - Path map with coverage

**Expected Results**:
- ~7,200 data records (120 hours × 60 min/hr)
- Coverage varies by location along path
- Multiple DSN station handovers
- Typical coverage: 50-70% depending on latitude

## Understanding the Results

### Earth Visibility

**Earth Elevation Angle**:
- **> 10°**: Good - clear line of sight
- **5-10°**: Fair - near horizon, may have terrain blocking
- **< 5°**: Poor - very low angle, likely blocked

**From Lunar South Pole** (-89.5°):
- Earth appears near horizon
- Visibility varies with libration
- Maximum elevation: ~15-20°
- Coverage: ~50-70% of time

### Link Quality

**Link Margin** (dB):
- **> 20**: Excellent - robust link
- **15-20**: Very good - reliable
- **10-15**: Good - adequate
- **6-10**: Fair - may have issues
- **< 6**: Poor - marginal

**DSN Station Availability**:
- Depends on Earth's rotation
- Stations rise/set from rover perspective
- Typical: 1-2 stations available at once
- Best: 3 stations during favorable geometry

### Coverage Gaps

**Common Causes**:
1. **Earth below horizon**: Rover on far side or in deep crater
2. **Low elevation**: Terrain blocking
3. **Link budget**: Insufficient power/gain
4. **DSN scheduling**: Station not available (not modeled)

**Solutions**:
- Increase antenna gain
- Use higher power transmitter
- Add relay satellite
- Plan stops at high-visibility locations

## Output Files

### CSV File
- **Size**: ~1 MB per 10,000 minutes
- **Format**: Standard CSV (UTF-8)
- **Import**: Excel, MATLAB, Python pandas
- **Use**: Mission planning, analysis, visualization

### Visualization (if enabled)
- **Timeline Plot**: Coverage vs. time (4 panels)
  - Earth visibility
  - DSN station availability
  - Number of stations
  - Distance traveled

- **Coverage Map**: Path with color-coded coverage
  - Green: DSN available
  - Red: No DSN
  - Waypoints marked

## Advanced Usage

### Custom Analysis

```python
# Load saved CSV
import pandas as pd
df = pd.read_csv('rover_coverage.csv')

# Find longest coverage gap
df['gap'] = ~df['any_dsn_available']
gaps = []
current_gap = 0

for i, row in df.iterrows():
    if row['gap']:
        current_gap += 1
    else:
        if current_gap > 0:
            gaps.append(current_gap)
        current_gap = 0

longest_gap = max(gaps)
print(f"Longest coverage gap: {longest_gap} minutes")
```

### Integration with Other Tools

```python
# Export for MATLAB
df.to_csv('coverage.csv', index=False)

# Export for Excel pivot tables
with pd.ExcelWriter('coverage.xlsx') as writer:
    df.to_excel(writer, sheet_name='Raw Data')
    summary.to_excel(writer, sheet_name='Summary')

# Export for Google Earth (KML)
# ... custom KML generation ...
```

## Performance

### Computational Requirements

| Mission Duration | Path Points | Time Steps | Processing Time |
|------------------|-------------|------------|-----------------|
| 24 hours | 10 waypoints | 1,440 | ~30 seconds |
| 120 hours (5 days) | 15 waypoints | 7,200 | ~2 minutes |
| 720 hours (30 days) | 50 waypoints | 43,200 | ~15 minutes |

**Optimization Tips**:
- Use `time_step_minutes=5` for faster preview analysis
- Reduce `path_samples` for fewer interpolation points
- Use `interpolate_path=False` for waypoint-only analysis

### Memory Usage

- Typical: < 100 MB for 10,000 records
- CSV file: ~500 bytes per record
- In-memory DataFrame: ~2 KB per record

## Troubleshooting

### "SPICE kernels not found"
- Ensure `kernels/` directory exists
- Check that `.bsp` files are present
- Analysis will run with approximations if kernels missing

### "Rover doesn't complete path"
- Check `rover_speed_kmh` vs `duration_hours`
- Calculate: `time_needed = distance / speed`
- Increase duration or speed as needed

### "All coverage is False"
- Check rover location (near pole?)
- Verify Earth visibility from that latitude
- Try different start time
- Increase TX power or antenna gain

### "CSV too large"
- Reduce `time_step_minutes` (e.g., 5 or 10 minutes)
- Shorten `duration_hours`
- Use fewer `path_samples`

## Examples

### Example 1: Short Scout Mission
```python
# 4-hour reconnaissance
waypoints = [(-89.5, 0.0), (-89.4, 10.0), (-89.5, 0.0)]

df = analyzer.analyze_coverage(
    waypoints=waypoints,
    start_time="2026-04-01T12:00:00",
    duration_hours=4,
    rover_speed_kmh=3.0,  # Fast
    time_step_minutes=1.0
)
```

### Example 2: Multi-Day Exploration
```python
# 7-day detailed survey
waypoints = [
    (-89.5, 0.0),   # Base
    (-89.6, 10.0),  # Site 1
    (-89.7, 15.0),  # Site 2
    (-89.6, 20.0),  # Site 3
    (-89.5, 0.0),   # Return
]

df = analyzer.analyze_coverage(
    waypoints=waypoints,
    start_time="2026-05-15T00:00:00",
    duration_hours=168,  # 7 days
    rover_speed_kmh=0.5,  # Very slow
    tx_power_dbm=46.0,    # High power
    time_step_minutes=1.0
)
```

### Example 3: Stationary Science Platform
```python
# 30-day stationary observation
location = [(-89.55, 5.0)]  # Single point

df = analyzer.analyze_coverage(
    waypoints=location * 2,  # Stay at same point
    start_time="2026-06-01T00:00:00",
    duration_hours=720,  # 30 days
    rover_speed_kmh=0.0,  # Stationary
    time_step_minutes=10.0  # Coarser sampling
)
```

## API Reference

### RoverPathDTEAnalyzer Class

#### Methods

**`__init__(kernel_dir='kernels')`**
- Initialize analyzer with SPICE kernels

**`interpolate_path(waypoints, num_samples=100)`**
- Interpolate between waypoints
- Returns list of (lat, lon) points

**`calculate_path_distance(waypoints)`**
- Calculate total path distance in km

**`analyze_coverage(...)`**
- Main analysis function
- Returns pandas DataFrame
- See Parameters section above

**`generate_summary()`**
- Generate coverage statistics
- Returns dictionary

**`print_summary()`**
- Print formatted summary to console

**`save_csv(output_path)`**
- Save coverage data to CSV file

## Citation

If you use this tool for research or mission planning, please reference:

```
Lunar Surface Communication Analysis Tool
Rover Path DTE Coverage Analyzer
[Your Repository/Paper]
2025
```

## License

Research and educational use. See main repository for license details.

## Support

For questions or issues:
- Check troubleshooting section above
- Review example code
- See test case for realistic usage

## Related Tools

- `lunar_comm_spice.py` - Base SPICE analysis module
- `test_surface_to_earth.py` - Static location DTE test
- `integrated_comm_analysis.py` - Full system analysis

---

**Ready to analyze your rover mission!** 🌙🚀📡

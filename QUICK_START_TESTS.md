# Quick Start: Running Communication Tests

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

## Test 1: Surface-to-Surface Communication

### Run the Test
```bash
python test_surface_to_surface.py
```

### What It Does
Tests communication from **Artemis Base Station** at Shackleton Crater to **6 surface assets**:
- 🚙 VIPER Rover (16.7 km)
- 🛸 Artemis Lander (16.7 km)
- 🚙 Exploration Rover (28.1 km)
- 📡 Communications Relay (56.6 km)
- 🏭 ISRU Water Plant (27.8 km)
- 🔬 Science Lander (80.2 km)

### Test Configuration
- **Location**: -89.5°, 0.0° (Shackleton Crater rim)
- **Date/Time**: December 15, 2025, 12:00:00 UTC
- **Frequency**: 2600 MHz (S-band LTE)
- **Power**: 40W transmitter

### Expected Output
```
✓ VIPER Rover
   Distance: 16.73 km
   RX Power: -81.2 dBm
   Link Margin: +33.8 dB
   Status: LINK OK

✓ Artemis Lander 1
   Distance: 16.68 km
   RX Power: -78.5 dBm
   Link Margin: +41.5 dB
   Status: LINK OK

[... results for all 6 assets ...]

SUMMARY STATISTICS
Total Assets: 6
Links Available: 5 (83.3%)
Links Unavailable: 1 (16.7%)
```

### Generated Files
- **surface_to_surface_test_report.txt** - Detailed analysis report

---

## Test 2: Surface-to-Earth (DTE) Communication

### Run the Test
```bash
python test_surface_to_earth.py
```

### What It Does
Tests **Direct-to-Earth** communication from lunar base to **DSN ground stations**:
- 📡 Goldstone 70m & 34m (California)
- 📡 Canberra 70m & 34m (Australia)
- 📡 Madrid 70m & 34m (Spain)

### Test Configuration
- **Location**: -89.5°, 0.0° (Shackleton Crater rim)
- **Period**: December 15-25, 2025 (10 days)
- **Frequency**: 8450 MHz (X-band)
- **Power**: 100W transmitter
- **Distance**: ~384,400 km (Earth-Moon)

### Expected Output
```
EARTH VISIBILITY ANALYSIS
Total Windows: 18
Total Visible Time: 156.34 hours (6.51 days)
Visibility Coverage: 65.1%

Window Statistics:
  Longest Window: 12.45 hours
  Shortest Window: 5.23 hours
  Average Window: 8.69 hours

DSN LINK BUDGET ANALYSIS
✓ Goldstone DSS-14 (70m)
   Distance: 384,235 km
   Path Loss: 270.23 dB
   RX Power: -136.23 dBm
   Link Margin: +23.77 dB
   Status: LINK AVAILABLE

[... results for all 6 DSN stations ...]
```

### Generated Files
- **surface_to_earth_test_report.txt** - Detailed analysis report
- **earth_visibility_test.png** - Earth visibility plot (3 panels)

---

## Understanding the Results

### Link Margin Guide

| Margin | Quality | What It Means |
|--------|---------|---------------|
| > 20 dB | ⭐⭐⭐⭐⭐ Excellent | Very robust, high reliability |
| 15-20 dB | ⭐⭐⭐⭐ Very Good | Good operational margin |
| 10-15 dB | ⭐⭐⭐ Good | Adequate for most conditions |
| 6-10 dB | ⭐⭐ Fair | May have occasional issues |
| 3-6 dB | ⭐ Marginal | Frequent packet loss |
| 0-3 dB | ⚠️ Poor | Barely usable |
| < 0 dB | ❌ None | No communication |

### Link Status Symbols
- ✓ Link Available
- ✗ Link Unavailable
- ⚠ Warning (low margin)

---

## Quick Performance Summary

### Surface-to-Surface (S-band @ 2.6 GHz)
- **Typical Range**: 10-50 km
- **Data Rate**: 1-100 Mbps (LTE)
- **Latency**: Milliseconds
- **Main Challenge**: Line-of-sight terrain blocking

### Surface-to-Earth (X-band @ 8.45 GHz)
- **Range**: 384,400 km (fixed Earth-Moon distance)
- **Data Rate**: 1-100 Mbps (depends on link margin)
- **Latency**: 2.5 seconds round-trip
- **Main Challenge**: Earth visibility windows, path loss

---

## Customizing the Tests

### Change Base Station Location
Edit in `test_surface_to_surface.py`:
```python
self.base_station = {
    'lat': -85.0,  # Your latitude
    'lon': 45.0,   # Your longitude
    ...
}
```

### Change Test Time
Edit in `test_surface_to_earth.py`:
```python
self.test_start = "2026-06-01T00:00:00"  # Your UTC time
self.test_duration_hours = 720  # 30 days
```

### Add More Assets
Add to asset list in `test_surface_to_surface.py`:
```python
{
    'name': 'My Rover',
    'type': 'Light Rover',
    'lat': -89.0,
    'lon': 30.0,
    'altitude': 2.0,
    'rx_sensitivity_dbm': -110.0,
    'antenna_gain_dbi': 6.0,
    'mission': 'Exploration'
}
```

---

## Typical Run Times

- **Surface-to-Surface**: ~10 seconds
- **Surface-to-Earth**: ~30-60 seconds (analyzing 10 days of visibility)

---

## Troubleshooting

### "No module named 'spiceypy'"
```bash
pip install spiceypy
```

### "No SPICE kernels loaded"
This is just a warning. Tests will run with approximations.
For full accuracy, add SPICE kernels to `kernels/` directory.

### Import errors
```bash
pip install -r requirements.txt
```

---

## What The Reports Contain

### Surface-to-Surface Report
1. **Base Station Config**: Location, power, frequency
2. **Asset List**: All 6 assets with coordinates and specs
3. **Link Analysis**: Detailed RF link budget for each
4. **Statistics**: Mean, min, max link margins and distances
5. **Recommendations**: Which assets need relay stations

### Surface-to-Earth Report
1. **Transmitter Config**: Location, power, frequency
2. **Visibility Windows**: All windows with UTC start/end times
3. **DSN Link Budgets**: Performance to all 6 stations
4. **Coverage Summary**: Which stations available when
5. **Recommendations**: Operations planning guidance

---

## Example Use Cases

### Mission Planning
```bash
# Run both tests to understand full comm architecture
python test_surface_to_surface.py
python test_surface_to_earth.py

# Review reports for:
# - Surface network coverage gaps
# - Earth contact schedules
# - Required relay stations
# - DSN scheduling needs
```

### Network Design
```bash
# Modify asset locations
# Re-run test
# Compare link margins
# Optimize relay placement
```

### Link Budget Verification
```bash
# Change RF parameters
# Re-run tests
# Verify margins meet requirements
# Iterate design
```

---

## Next Steps

After running these tests:

1. ✅ **Review Reports** - Analyze detailed link budgets
2. ✅ **Check Coverage** - Identify gaps in surface network
3. ✅ **Plan Operations** - Use Earth visibility for scheduling
4. ✅ **Optimize Design** - Adjust power/antennas as needed
5. ✅ **Run Integrated Analysis** - Use full system simulator

For more advanced analysis:
```bash
# Full integrated analysis with terrain
python integrated_comm_analysis.py

# Interactive examples
python example_spice_analysis.py
```

---

## Questions?

See detailed documentation:
- **README_TESTS.md** - Complete test documentation
- **README_SPICE_INTEGRATION.md** - SPICE/DTE documentation
- **README_LTE_SIMULATOR.md** - LTE simulator documentation

---

**Happy Testing! 🌙📡🌍**

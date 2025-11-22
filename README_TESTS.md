# Lunar Communication System Test Suite

Comprehensive test cases demonstrating surface-to-surface and surface-to-Earth communication analysis with specific lunar locations and UTC times.

## Test Cases

### Test 1: Surface-to-Surface Communication
**File**: `test_surface_to_surface.py`

#### Test Configuration

**Base Station**:
- **Location**: Shackleton Crater rim (-89.5°, 0.0°)
- **Site**: Proposed Artemis Base Camp location
- **Antenna Height**: 10 meters
- **Frequency**: 2600 MHz (S-band LTE, Band 7)
- **TX Power**: 46 dBm (40 W)
- **TX Gain**: 18 dBi (sector antenna)

**Test Date**: December 15, 2025, 12:00:00 UTC

**Surface Assets** (6 total):

1. **VIPER Rover**
   - Location: -89.35°, 12.0° (~16.7 km from base)
   - Mission: Ice prospecting in permanently shadowed regions
   - Antenna: 2m height, 8 dBi gain, -115 dBm sensitivity

2. **Artemis Lander 1**
   - Location: -89.65°, -8.0° (~16.7 km from base)
   - Mission: Crew habitat and operations
   - Antenna: 8m height, 12 dBi gain, -120 dBm sensitivity

3. **Exploration Rover Alpha**
   - Location: -89.25°, 25.0° (~28.1 km from base)
   - Mission: Geological survey
   - Antenna: 1.5m height, 6 dBi gain, -110 dBm sensitivity

4. **Communications Relay 1**
   - Location: -89.0°, 45.0° (~56.6 km from base)
   - Mission: Extended range relay
   - Antenna: 15m mast, 18 dBi gain, -125 dBm sensitivity

5. **Resource Extractor 1**
   - Location: -89.75°, 3.0° (~27.8 km from base)
   - Mission: Water ice extraction (ISRU)
   - Antenna: 5m height, 10 dBi gain, -112 dBm sensitivity

6. **Science Lander Beta**
   - Location: -88.8°, 60.0° (~80.2 km from base)
   - Mission: Seismic monitoring
   - Antenna: 3m height, 9 dBi gain, -108 dBm sensitivity

#### What Gets Analyzed

- **Link Budgets**: Complete RF link budget for each asset
- **Path Loss**: Free-space path loss calculations
- **Received Power**: Signal strength at each asset
- **Link Margins**: Margin above sensitivity threshold
- **Line-of-Sight**: Geometric LOS determination
- **Link Quality**: Performance categorization (Excellent/Good/Fair/Marginal/Poor)

#### Outputs

**Console Output**:
- Base station configuration
- Asset list with missions
- Link analysis for each asset
- Summary statistics
- Recommendations

**Report File**: `surface_to_surface_test_report.txt`
- Complete test metadata
- Detailed link budgets
- Performance statistics (mean, min, max, std dev)
- Link quality assessments
- Operational recommendations

#### Running the Test

```bash
python test_surface_to_surface.py
```

#### Example Output

```
✓ VIPER Rover
   Type: Heavy Rover
   Distance: 16.73 km
   Path Loss: 145.2 dB
   RX Power: -81.2 dBm
   Link Margin: +33.8 dB
   Status: LINK OK

✗ Science Lander Beta
   Distance: 80.15 km
   RX Power: -95.3 dBm
   Link Margin: -12.7 dB
   Status: NO LINK
   → Recommendation: Relay station needed (beyond horizon)
```

---

### Test 2: Surface-to-Earth (DTE) Communication
**File**: `test_surface_to_earth.py`

#### Test Configuration

**Transmitter** (Lunar Surface):
- **Location**: Shackleton Crater rim (-89.5°, 0.0°)
- **Site**: Artemis Base Camp DTE Terminal
- **Antenna Height**: 20 meters (tall mast for Earth visibility)
- **Frequency**: 8450 MHz (X-band downlink)
- **TX Power**: 50 dBm (100 W SSPA)
- **TX Gain**: 30 dBi (1.2m dish pointed at Earth)
- **Modulation**: QPSK
- **Target Data Rate**: 10 Mbps

**Test Period**:
- **Start**: December 15, 2025, 00:00:00 UTC
- **End**: December 25, 2025, 00:00:00 UTC
- **Duration**: 10 days (240 hours)
- **Sampling**: Every 30 minutes

**DSN Stations Analyzed** (6 total):

**Goldstone Complex** (California, USA):
- DSS-14: 70m dish, 74 dBi gain @ X-band, -160 dBm sensitivity
- DSS-24: 34m dish, 68 dBi gain @ X-band, -150 dBm sensitivity

**Canberra Complex** (Australia):
- DSS-43: 70m dish, 74 dBi gain @ X-band, -160 dBm sensitivity
- DSS-34: 34m dish, 68 dBi gain @ X-band, -150 dBm sensitivity

**Madrid Complex** (Spain):
- DSS-63: 70m dish, 74 dBi gain @ X-band, -160 dBm sensitivity
- DSS-54: 34m dish, 68 dBi gain @ X-band, -150 dBm sensitivity

#### What Gets Analyzed

**Earth Visibility**:
- Visibility windows over 10-day period
- Window duration statistics
- Elevation and azimuth angles
- Total visibility percentage
- Longest/shortest/average windows

**DTE Link Budgets**:
- Distance (Earth-Moon: ~384,400 km)
- Free-space path loss (~270 dB @ X-band)
- Received power at each DSN station
- Link margins
- Link quality assessment
- Theoretical data rate capacity

**DSN Coverage Timeline**:
- Which stations are available during each window
- Station availability percentages
- Coverage gaps
- Best/worst case scenarios

#### Outputs

**Console Output**:
- Earth visibility analysis
- DSN link budgets for all stations
- Coverage timeline (first 20 windows)
- Station availability summary

**Report File**: `surface_to_earth_test_report.txt`
- Complete test metadata
- Transmitter configuration
- All visibility windows with times
- Detailed link budgets for each DSN station
- Coverage statistics
- Operational recommendations

**Plot**: `earth_visibility_test.png`
- 3-panel visualization:
  1. Visibility timeline (10 days)
  2. Elevation angle profile
  3. Azimuth angle profile

#### Running the Test

```bash
python test_surface_to_earth.py
```

#### Example Output

```
Visibility Results:
  Total Windows: 18
  Total Visible Time: 156.34 hours (6.51 days)
  Visibility Coverage: 65.1%

Window Statistics:
  Longest Window: 12.45 hours
  Shortest Window: 5.23 hours
  Average Window: 8.69 hours

✓ Goldstone DSS-14 (70m)
   Distance: 384,235 km
   Free-Space Path Loss: 270.23 dB
   RX Power: -136.23 dBm
   Link Margin: +23.77 dB
   Status: LINK AVAILABLE
   Link Quality: EXCELLENT

DSN Coverage Summary:
  Goldstone (70m):        16 / 18 (88.9%)
  Canberra (70m):         15 / 18 (83.3%)
  Madrid (70m):           17 / 18 (94.4%)
```

---

## Running Both Tests

### Sequential Execution
```bash
# Run surface-to-surface test
python test_surface_to_surface.py

# Run surface-to-Earth test
python test_surface_to_earth.py
```

### Batch Execution
```bash
# Run both tests (create wrapper script)
python -c "
import test_surface_to_surface
import test_surface_to_earth

print('\\n' + '='*80)
print('RUNNING ALL TESTS')
print('='*80)

test_surface_to_surface.main()
test_surface_to_earth.main()

print('\\n' + '='*80)
print('ALL TESTS COMPLETE')
print('='*80)
"
```

## Output Files Generated

After running both tests, you will have:

1. **surface_to_surface_test_report.txt** - Detailed S2S analysis
2. **surface_to_earth_test_report.txt** - Detailed DTE analysis
3. **earth_visibility_test.png** - Earth visibility visualization

## Key Metrics Reported

### Surface-to-Surface
- **Coverage**: Percentage of assets with available links
- **Link Margins**: Average, min, max for available links
- **Distances**: Range coverage achieved
- **Availability**: Which assets need relay stations

### Surface-to-Earth
- **Visibility**: Percentage of time Earth is visible
- **Window Duration**: How long each communication pass lasts
- **DSN Coverage**: Which stations provide coverage
- **Link Quality**: Margins to each DSN complex
- **Data Capacity**: Theoretical maximum data rates

## Understanding the Results

### Link Margin Interpretation

| Margin (dB) | Quality | Reliability | Notes |
|-------------|---------|-------------|-------|
| > 20 | Excellent | Very High | Robust against fading |
| 15-20 | Very Good | High | Good operational margin |
| 10-15 | Good | Good | Adequate for most conditions |
| 6-10 | Fair | Moderate | May have occasional dropouts |
| 3-6 | Marginal | Low | Frequent packet loss expected |
| 0-3 | Poor | Very Low | Barely usable |
| < 0 | None | N/A | Link unavailable |

### Typical Performance Expectations

**Surface-to-Surface (S-band, 2.6 GHz)**:
- **Range**: 10-50 km (depends on terrain and power)
- **Data Rate**: 1-100 Mbps (LTE standard)
- **Latency**: ~milliseconds (light speed @ distance)
- **Challenges**: Line-of-sight, terrain blocking

**Surface-to-Earth (X-band, 8.45 GHz)**:
- **Range**: ~384,400 km (Earth-Moon distance)
- **Data Rate**: 1-100 Mbps (depends on SNR and bandwidth)
- **Latency**: ~2.5 seconds round-trip
- **Challenges**: Visibility windows, path loss, DSN scheduling

## Test Scenarios Explained

### Why Shackleton Crater?

The test location (-89.5°, 0.0°) near Shackleton Crater was chosen because:
- Proposed Artemis landing site
- Near south pole - scientifically valuable
- Access to permanently shadowed regions (ice)
- Representative of challenging comm environment
- Low Earth elevation angles when visible

### Why These Times?

**December 15-25, 2025**:
- Representative future mission timeframe
- 10-day period captures full lunar rotation effects
- Shows variation in Earth visibility
- Demonstrates DSN handover between complexes

### Why These Frequencies?

**2.6 GHz (S-band)** for surface:
- Good balance of range and data rate
- Standard LTE band
- Less atmospheric absorption (not relevant on Moon, but standardized hardware)
- Proven space heritage

**8.45 GHz (X-band)** for DTE:
- Standard deep space downlink band
- High data rate capability
- DSN optimized for X-band
- Well-characterized performance

## Customizing the Tests

### Change Location
Edit the transmitter coordinates:
```python
self.base_station = {
    'lat': -85.0,  # Change latitude
    'lon': 45.0,   # Change longitude
    ...
}
```

### Change Time Period
Modify test period:
```python
self.test_start = "2026-01-01T00:00:00"
self.test_duration_hours = 720  # 30 days
```

### Add More Assets
Add to `self.surface_assets` list:
```python
{
    'name': 'New Rover',
    'type': 'Light Rover',
    'lat': -89.0,
    'lon': 30.0,
    'altitude': 2.0,
    'rx_sensitivity_dbm': -110.0,
    'antenna_gain_dbi': 6.0,
    'mission': 'Exploration'
}
```

### Change RF Parameters
Modify transmitter settings:
```python
'frequency_mhz': 1800.0,  # Different frequency
'tx_power_dbm': 40.0,     # Different power
'tx_gain_dbi': 15.0,      # Different antenna
```

## Interpretation Guide

### When Surface-to-Surface Links Fail

**Beyond Horizon**:
- Distance exceeds geometric line-of-sight
- **Solution**: Deploy relay station or increase antenna height

**Insufficient Power**:
- Link margin is negative even with LOS
- **Solution**: Increase TX power, use higher gain antennas, or reduce distance

**Marginal Quality**:
- Link available but margin < 6 dB
- **Solution**: Improve link margin for reliability

### When Earth Visibility is Limited

**Low Visibility %**:
- Earth visible < 50% of time
- **Solution**: Use store-and-forward operations, relay satellite, or different landing site

**Short Windows**:
- Average window < 4 hours
- **Solution**: Schedule frequent contact opportunities, prioritize data

**Low Elevation**:
- Max elevation < 10°
- **Solution**: Tall antenna mast, high-gain steerable dish

### When DSN Links Are Weak

**High Path Loss**:
- Distance-dependent (can't change)
- **Solution**: Increase TX power or antenna gain

**No Station Coverage**:
- No DSN station has positive margin
- **Solution**: Increase EIRP (power × gain)

**Limited Station Access**:
- Only one complex available
- **Risk**: DSN scheduling conflicts
- **Solution**: Improve margin to enable 34m dishes

## References

### Test Methodology
Based on:
- ITU-R P.525: Free-space attenuation
- ITU-R P.526: Diffraction
- NASA DSN Telecommunications Link Design Handbook (810-005)
- LTE specifications (3GPP)

### Lunar Mission Context
- Artemis mission planning documents
- NASA Lunar Communications and Navigation Architecture
- LunaNet Interoperability Specification

### DSN Information
- https://deepspace.jpl.nasa.gov/
- DSN station capabilities and availability
- Deep space communication standards

## Troubleshooting

### "No SPICE kernels loaded"
- Ensure `kernels/` directory exists
- Check that `.bsp` files are present
- Tests will still run with approximations

### "No DEM files found"
- `test_surface_to_surface.py` doesn't require DEMs (uses geometric LOS)
- Only integrated analysis needs DEMs

### Import errors
```bash
pip install -r requirements.txt
```

### Results seem unrealistic
- Check that coordinates are valid (-90 to 90 lat, -180 to 180 lon)
- Verify frequencies are in MHz
- Ensure power is in dBm, not watts

## Next Steps

After running these tests:

1. **Analyze Reports**: Review detailed link budgets
2. **Optimize Design**: Adjust parameters for better coverage
3. **Run Integrated Analysis**: Use `integrated_comm_analysis.py` for full system view
4. **Mission Planning**: Use visibility windows for operations timeline
5. **Network Design**: Place relay stations based on coverage gaps

---

**Note**: These are simulation tools for planning purposes. Actual mission implementation requires detailed engineering analysis, hardware testing, and mission-specific optimization.

# ConOps Test Outputs

Generated outputs from all 4 operational scenarios.

## ConOps 1: Surface TX → Surface RX
- **Coverage Map**: surface_coverage_20251122_065151.png
- **CSV Report**: report_20251122_065152.csv
- **Configuration**:
  - Frequency: 2600 MHz (S-band LTE)
  - TX Power: 46 dBm (39.8 W)
  - Propagation Model: Two-ray with multipath
  - Assets: 3 (VIPER Rover, Artemis Lander, Science Platform)
- **Results**:
  - Coverage: 100%
  - Max Range: 15.00 km
  - All asset links available with positive margins

## ConOps 2: Surface TX → Earth RX (DTE)
- **Coverage Timeline**: dte_coverage_20251122_065153.png
- **CSV Report**: report_20251122_065154.csv
- **Configuration**:
  - Frequency: 8450 MHz (X-band)
  - TX Power: 50 dBm (100 W)
  - TX Gain: 30 dBi (HGA)
  - Duration: 10 days
- **Results**:
  - Earth visibility windows calculated
  - DSN station link budgets computed

## ConOps 3: Crater TX → Earth RX (DTE)
- **Coverage Timeline**: dte_coverage_20251122_065154.png
- **CSV Report**: report_20251122_065155.csv
- **Configuration**:
  - Crater: 600m radius, 150m depth
  - TX inside crater at 10m above floor
  - Frequency: 8450 MHz (X-band)
  - Additional diffraction loss modeled
- **Results**:
  - Crater effects on link margins analyzed

## ConOps 4: Rover Path → Earth RX (DTE)
- **Coverage Timeline**: rover_path_20251122_065322.png
- **Detailed CSV**: report_20251122_065322.csv (576 time steps)
- **Configuration**:
  - 7 waypoints, 8.11 km total path
  - Mission: 48 hours at 1.2 km/h
  - Time resolution: 5 minutes
  - Frequency: 8450 MHz (X-band)
- **Results**:
  - Mission Duration: 576 minutes
  - Earth Visible: 100%
  - DSN Available: 100%
  - All 3 DSN complexes available throughout mission

## Files

| File | Size | Description |
|------|------|-------------|
| surface_coverage_20251122_065151.png | 490 KB | ConOps 1 coverage map |
| dte_coverage_20251122_065153.png | 191 KB | ConOps 2 visibility timeline |
| dte_coverage_20251122_065154.png | 191 KB | ConOps 3 visibility timeline |
| rover_path_20251122_065322.png | 172 KB | ConOps 4 rover path timeline |
| report_20251122_065152.csv | 723 B | ConOps 1 summary |
| report_20251122_065154.csv | 618 B | ConOps 2 summary |
| report_20251122_065155.csv | 617 B | ConOps 3 summary |
| report_20251122_065322.csv | 112 KB | ConOps 4 detailed data (576 rows) |


# KPLO Location Tracker

Python applications for tracking the Korea Pathfinder Lunar Orbiter (KPLO) position using SPICE kernels.

## Overview

The KPLO is equipped with ShadowCam, a high-resolution imaging instrument designed to map permanently shadowed regions (PSRs) near the lunar poles. These applications visualize KPLO's current orbital position.

## Applications

### 1. kplo_location.py (Full Version)
Real-time KPLO tracking using NASA NAIF SPICE kernels.

**Requirements:**
- Git LFS to download binary SPICE kernel files
- spiceypy library
- All kernels in the `kernels/` directory

**Usage:**
```bash
python3 kplo_location.py
```

**Note:** Currently requires Git LFS setup to download the large binary SPICE kernel files (BSP files are stored as Git LFS objects).

### 2. kplo_location_demo.py (Demonstration Version)
Simulated KPLO tracking using typical polar orbit parameters.

**Features:**
- Works without SPICE kernels
- Simulates realistic polar orbit (100 km altitude, ~2 hour period)
- Shows typical KPLO ground track
- Includes UTC and Arizona time stamps

**Usage:**
```bash
python3 kplo_location_demo.py
```

## Output

Both applications generate `kplo_location.png` containing:

**Left Panel - Moon Projection:**
- Mollweide projection of lunar surface
- KPLO position marker (green)
- Orbital ground track
- Latitude/longitude grid

**Right Panel - Telemetry:**
- Position (lat/lon/alt and XYZ)
- Velocity (3D vector and magnitude)
- Orbital parameters
- Time stamps (UTC and Arizona MST)

## SPICE Kernels

The `kernels/` directory contains NAIF SPICE files for KPLO:

### ✓ Available Kernels (Real Binary Files - 12/15)

**Spacecraft Data:**
- `kplo_scp_*.bc` (1.5 MB, 301 KB) - KPLO orientation (attitude) ✓
- `kplo_sclkscet_*.tsc` (173 KB) - Spacecraft clock ✓

**Instrument:**
- `kplo_shadowcam_v01.ti` (14 KB) - ShadowCam instrument kernel ✓
- `kplo_v00_shc_a.tf` (74 KB) - ShadowCam frames ✓
- `SHC_REFERENCE.tf` (1.1 KB) - Reference frames ✓

**Lunar/Planetary:**
- `moon_*.tf` (21 KB, 8.3 KB) - Moon reference frames ✓
- `moon_pa_de421_1900-2050.bpc` (1.7 MB) - Moon orientation ✓

**Constants:**
- `naif0012.tls` (5.2 KB) - Leap seconds ✓
- `pck00010.tpc` (124 KB) - Physical constants ✓
- `gm_de431.tpc` (6.2 KB) - Gravitational parameters ✓

### ✗ Require Git LFS (Currently LFS Pointers - 3/15)

These files are stored as Git LFS pointers and need `git lfs pull`:

- `kplo_pm_*.bsp` (2 files) - **KPLO position/velocity ephemeris** (CRITICAL)
- `de430.bsp` (119 MB) - Planetary ephemeris

**Impact:** 80% of kernels are available. Missing only the spacecraft position data needed for real-time tracking.

## Setup for Real-Time Tracking

To use real SPICE kernels:

1. **Install Git LFS:**
   ```bash
   sudo apt-get install git-lfs
   git lfs install
   ```

2. **Pull kernel files:**
   ```bash
   git lfs pull
   ```

3. **Verify kernels:**
   ```bash
   file kernels/*.bsp  # Should show "data" not "ASCII text"
   ```

4. **Run tracker:**
   ```bash
   python3 kplo_location.py
   ```

## KPLO Mission Details

- **Launch:** August 4, 2022
- **Lunar Orbit Insertion:** December 16, 2022
- **Orbit:** Polar orbit, ~100 km altitude
- **Period:** ~2 hours
- **Inclination:** ~90° (polar)
- **Primary Instrument:** ShadowCam (NASA-provided)
- **Mission:** Map lunar PSRs, search for water ice

## Metadata

The `fnnames_sslatlon_time.xlsx` file contains metadata linking:
- COG filenames (SDC60 files)
- Spacecraft sub-solar latitude/longitude
- Acquisition times

This enables correlation between KPLO position and ShadowCam imagery.

## Related Applications

- `visualize_psr_cog.py` - Visualize ShadowCam imagery with PSR overlays
- `create_psr_geodatabase.py` - Build PSR spatial database
- `extract_cog_footprints.py` - Extract COG image footprints

## References

- [NAIF KPLO Data](https://naif.jpl.nasa.gov/naif/data_kplo.html)
- [SPICE Toolkit Documentation](https://naif.jpl.nasa.gov/naif/toolkit.html)
- [KPLO Mission (KARI)](https://www.kari.re.kr/eng.do)
- [ShadowCam (ASU)](https://www.shadowcam.asu.edu/)

## Version

- **Version:** 1.0
- **Date:** 2025-11-21
- **Author:** PSR-SDC1 Analysis Team

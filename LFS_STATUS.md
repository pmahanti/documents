# Git LFS Status for KPLO Kernels

## Attempted Installation

✓ Git LFS successfully installed (v3.4.1)
✓ Git LFS initialized in repository

## Download Attempts

### Attempt 1: Local Proxy (Failed)
```
git lfs pull
```
**Result:** HTTP 502 error - Local git proxy does not support LFS batch API

### Attempt 2: Direct from GitHub (Failed)
```
git remote add github https://github.com/pmahanti/documents.git
git lfs fetch github claude/psr-sdc1-work-01Xo7bsFPQ7aVi9trog3sdFj
```
**Result:** Authorization error - LFS objects require authentication

## Why LFS Downloads Failed

1. **Local Proxy Limitation:** The local git proxy server doesn't support Git LFS protocol
2. **GitHub Authentication:** LFS objects on GitHub require authenticated access
3. **Private Repository:** The repository appears to require credentials for LFS downloads

## Current Status

**Working:**
- 12/15 kernels (80%) available as binary files
- Demo visualization working perfectly (`kplo_location_demo.py`)
- All attitude, Moon, instrument, and time kernels functional

**Not Working:**
- Real-time position tracking requires missing ephemeris kernels
- 3 BSP files stored as LFS pointers (132 bytes each)

## Available Options

### Option 1: Manual Download (Recommended)

If you have GitHub access to this repository:

1. Visit: https://github.com/pmahanti/documents/tree/claude/psr-sdc1-work-01Xo7bsFPQ7aVi9trog3sdFj/kernels

2. Click on each file and download:
   - `kplo_pm_20251113_20260108_v00.bsp` (4.2 MB)
   - `kplo_pm_20251116_20260111_v00.bsp` (4.2 MB)
   - `de430.bsp` (119.7 MB)

3. Place in `/home/user/documents/kernels/`

### Option 2: Request Files from Repository Owner

Ask the repository owner (pmahanti) to provide the BSP files through another method:
- Direct file transfer
- Cloud storage link
- Alternative hosting

### Option 3: Download from NAIF

Original SPICE kernels available from NASA:
- NAIF SPICE Data: https://naif.jpl.nasa.gov/naif/data_kplo.html
- Download KPLO ephemeris kernels directly
- Place in kernels/ directory

## Using Demo Mode (Currently Available)

While waiting for kernel files:

```bash
python3 kplo_location_demo.py
```

This generates realistic KPLO position visualizations using simulated orbital parameters:
- 100 km altitude polar orbit
- ~2 hour orbital period  
- 90° inclination
- UTC and Arizona timestamps
- Moon surface projection

Output: `kplo_location.png` (310 KB)

## Summary

Git LFS is installed and configured but cannot download files due to:
- Local proxy lacking LFS support
- GitHub requiring authentication for LFS objects

**Recommendation:** Use demo mode or manually download the 3 BSP files if you have repository access.

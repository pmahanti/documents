# Downloading KPLO Ephemeris Kernels

## Current Status

**Available Kernels (12/15):** ✓ Working
All attitude, orientation, Moon, instrument, and time kernels are already available as binary files.

**Missing Kernels (3/15):** Need to download
These are stored as Git LFS objects and need to be downloaded:

1. `kplo_pm_20251113_20260108_v00.bsp` (4.2 MB)
2. `kplo_pm_20251116_20260111_v00.bsp` (4.2 MB)
3. `de430.bsp` (119.7 MB)

## Why Can't I Download Automatically?

The BSP files are stored in Git LFS (Large File Storage). GitHub's media endpoint for LFS files requires authentication, so automatic download scripts fail with 403 errors.

## How to Get the Files

### Option 1: Install Git LFS (Recommended)

```bash
# Install Git LFS
sudo apt-get update
sudo apt-get install git-lfs

# Initialize Git LFS
git lfs install

# Pull the LFS files
git lfs pull

# Verify
file kernels/kplo_pm_*.bsp kernels/de430.bsp
# Should show "data" not "ASCII text"
```

### Option 2: Manual Download from GitHub

1. Visit: https://github.com/pmahanti/documents/tree/claude/psr-sdc1-work-01Xo7bsFPQ7aVi9trog3sdFj/kernels

2. Click on each BSP file:
   - `kplo_pm_20251113_20260108_v00.bsp`
   - `kplo_pm_20251116_20260111_v00.bsp`
   - `de430.bsp`

3. Click the "Download" button (or right-click → Save As)

4. Place the downloaded files in: `/home/user/documents/kernels/`

5. Verify the files are binary:
   ```bash
   file kernels/*.bsp
   ```
   Should show "data" not "ASCII text"

### Option 3: Use Git Clone with LFS

If starting fresh:

```bash
# Install git-lfs first
sudo apt-get install git-lfs
git lfs install

# Clone with LFS
git clone https://github.com/pmahanti/documents.git
cd documents
git checkout claude/psr-sdc1-work-01Xo7bsFPQ7aVi9trog3sdFj

# Verify kernels
ls -lh kernels/*.bsp
```

## Verification

After downloading, verify the kernels:

```bash
cd /home/user/documents

# Check file types (should be "data")
file kernels/*.bsp

# Check file sizes
ls -lh kernels/*.bsp

# Expected output:
# kplo_pm_20251113_20260108_v00.bsp: data (4.2 MB)
# kplo_pm_20251116_20260111_v00.bsp: data (4.2 MB)
# de430.bsp: data (119.7 MB)

# Test KPLO location tracker
python3 kplo_location.py
```

## Why These Files Are Critical

The missing BSP files contain:

- **kplo_pm_*.bsp**: KPLO spacecraft position and velocity (ephemeris)
  - Required for real-time location tracking
  - Without these: Can only use simulated orbit

- **de430.bsp**: Planetary ephemeris (Earth, Moon, Sun positions)
  - Provides Moon position relative to Earth
  - Used for coordinate transformations

## Current Workaround

Until the BSP files are downloaded, use the demo version:

```bash
python3 kplo_location_demo.py
```

This uses simulated orbital parameters (100 km altitude, ~2 hour period) and generates realistic visualizations.

## Troubleshooting

### "ASCII text" instead of "data"

If `file` shows "ASCII text", the file is still a Git LFS pointer:

```bash
# Check content
head kernels/kplo_pm_20251113_20260108_v00.bsp
# If it shows "version https://git-lfs.github.com/spec/v1", it's a pointer

# Solution: Use one of the download options above
```

### Permission denied / 403 errors

GitHub's raw and media URLs don't serve LFS files without authentication for private repos. Use Git LFS or manual download from the GitHub UI.

### Out of disk space

The de430.bsp file is 119.7 MB. Ensure you have at least 150 MB free:

```bash
df -h /home/user/documents
```

## Additional Resources

- [Git LFS Documentation](https://git-lfs.github.com/)
- [NAIF SPICE Data for KPLO](https://naif.jpl.nasa.gov/naif/data_kplo.html)
- [GitHub LFS Guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files)

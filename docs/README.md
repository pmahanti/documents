# Documentation Directory

Comprehensive explanations of the crater analysis code and algorithms.

## Step-by-Step Guides

### **Step 1: Prepare Geometries**
- [`STEP1_PREPARE_GEOMETRIES_EXPLAINED.md`](STEP1_PREPARE_GEOMETRIES_EXPLAINED.md) - Complete explanation
- [`step1_example.py`](step1_example.py) - Interactive demo (requires geopandas)
- [`step1_simple_demo.py`](step1_simple_demo.py) - Mathematical demo (no dependencies)

**What it does:** Converts crater point locations to circular polygon geometries

**Key operation:** `gdf.geometry.buffer(gdf['D_m'] / 2)`

---

### **Step 2: Refine Crater Rims**
- [`STEP2_REFINE_RIMS_EXPLAINED.md`](STEP2_REFINE_RIMS_EXPLAINED.md) - Complete code explanation
- [`step2_visual_guide.md`](step2_visual_guide.md) - Visual diagrams and examples

**What it does:** Uses DEM topography to detect actual rim positions

**Key functions:**
- `update_crater_rims()` - High-level orchestration
- `fit_crater_rim()` - Core rim detection algorithm
- `compute_E_matrix()` - Elevation sampling in polar coordinates

---

### **Step 3: Analyze Morphometry**
- Documentation coming soon...

**What it does:** Computes crater depth, diameter, and d/D ratio

---

## Quick Reference

### Algorithm Overview

```
INPUT                STEP 1              STEP 2              STEP 3              OUTPUT
Crater points   →   Circles        →    Refined rims   →    Morphometry    →    Results
+ diameters         (approximate)       (topographic)       (depth, d/D)        (shapefile + CSV)
```

### Code Structure

```
Notebook Cell          Modular Code
─────────────────────────────────────────────────────────────
Cell 0 (config)    →  config/regions.json
                      src/crater_analysis/config.py

Cell 2 (prep)      →  src/crater_analysis/refinement.py
                      └── prepare_crater_geometries()

Cell 1 (refine)    →  src/crater_analysis/refinement.py
                      └── update_crater_rims()
                          └── cratools.fit_crater_rim()

Cell 4 (analyze)   →  src/crater_analysis/morphometry.py
                      └── compute_depth_diameter_ratios()
```

---

## Running the Examples

### Step 1 Demo (Mathematical)
```bash
python3 docs/step1_simple_demo.py
```

**Output:** Shows transformation of points to circles with actual numbers

### Step 1 Demo (Full)
```bash
python3 docs/step1_example.py  # Requires geopandas
```

**Output:** Complete example with GeoPandas operations

---

## Key Concepts

### Coordinate Systems

**Polar Coordinates (r, θ):**
- Used for rim detection
- r = radius from center
- θ = azimuth angle (0° = North)

**Cartesian Coordinates (x, y):**
- Used for DEM sampling
- x = East-West position
- y = North-South position

**Conversion:**
```python
x = r × cos(θ)
y = r × sin(θ)
```

### Elevation Matrix

```
E_rim[radius, azimuth] = elevation at that position

           Azimuth
         0°   5°   10° ...
Radius ┌─────────────────
0.8R   │ elevations...
0.9R   │ elevations...
1.0R   │ elevations...  ← Rim typically here
1.1R   │ elevations...
1.2R   │ elevations...
```

### Circle Fitting

**Objective:** Minimize residuals between detected points and circle

**Equation:** `(x - x₀)² + (y - y₀)² = r²`

**Parameters:**
- (x₀, y₀) = center position
- r = radius

**Output:**
- Fitted geometry
- Uncertainty estimates

---

## Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_diameter` | 60 | Minimum crater size (meters) |
| `inner_radius` | 0.8 | Start rim search (fraction of R) |
| `outer_radius` | 1.2 | End rim search (fraction of R) |
| `azimuth_step` | 5° | Angular sampling interval |
| `radius_step` | 0.01R | Radial sampling interval |
| `remove_ext` | True | Remove regional topography |

---

## Glossary

**Azimuth:** Angle measured clockwise from North (0° to 360°)

**Buffer:** Create polygon by expanding geometry by radius

**CRS:** Coordinate Reference System (projection + datum)

**DEM:** Digital Elevation Model (raster of elevations)

**GeoPandas:** Python library for spatial data (extends pandas)

**Prominence:** Height of peak above surrounding valleys

**Radial Profile:** Elevation values along line from center

**Spline:** Smooth curve through points (for interpolation)

**Topography:** Shape and features of land surface

---

## Algorithm Performance

### Computational Complexity

**Step 1 (Prepare):**
- O(n) where n = number of craters
- Very fast (~0.1s for 100 craters)

**Step 2 (Refine):**
- O(n × m × k) where:
  - n = number of craters
  - m = number of azimuths (72)
  - k = number of radii (41)
- Moderate (~2-5s per crater with plotting)

**Step 3 (Analyze):**
- O(n × p) where:
  - n = number of craters
  - p = pixels within crater
- Moderate (~1-3s per crater)

### Memory Usage

- DEM size dominates memory
- Typical: 1-2 GB for large regions
- Processing: ~100 MB per crater

---

## Troubleshooting

### No rim detected
**Cause:** Degraded crater or wrong search range
**Solution:** Adjust `inner_radius` and `outer_radius`

### Poor circle fit
**Cause:** Irregular crater shape
**Solution:** Check diagnostic plots, may need manual inspection

### High uncertainties
**Cause:** Noisy DEM or scattered rim detections
**Solution:** Check DEM quality, consider smoothing

### Memory errors
**Cause:** Large DEM or many craters
**Solution:** Process regions separately, reduce DEM resolution

---

## Further Reading

- **Main README:** [`../README.md`](../README.md) - Project overview
- **Usage Examples:** [`../USAGE_EXAMPLE.md`](../USAGE_EXAMPLE.md) - How to use the code
- **Data Summary:** [`../data/DATA_SUMMARY.md`](../data/DATA_SUMMARY.md) - Input data format
- **Reorganization:** [`../REORGANIZATION_SUMMARY.md`](../REORGANIZATION_SUMMARY.md) - How code was modularized

---

## Contributing

To add documentation:
1. Create `.md` file in this directory
2. Add link to this README
3. Use clear headings and examples
4. Include code snippets and visuals

---

## Questions?

Check the detailed explanations:
- **Step 1:** Buffer operation, coordinate systems
- **Step 2:** Rim detection, peak finding, circle fitting
- **Step 3:** Depth calculation, uncertainty propagation

Each guide includes:
- Code breakdown
- Mathematical details
- Visual diagrams
- Real examples
- Troubleshooting tips

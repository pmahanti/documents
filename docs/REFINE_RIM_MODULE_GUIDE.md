# Crater Rim Refinement Module - Complete Guide

## Overview

The `refine_crater_rim.py` module performs **Step 2** of the crater analysis workflow. It refines crater rim positions using both topographic analysis and computer vision techniques, producing confidence scores for each detection.

## Key Features

✓ **Dual-Method Rim Detection**
- Topographic analysis (elevation peaks from DEM)
- Computer vision (edge detection from contrast image)
- Combined scoring for robustness

✓ **Probability Scoring**
- Each crater gets a rim detection probability (0-1)
- Based on topographic quality, edge strength, method agreement, and fit error
- Allows filtering by confidence level

✓ **Four Output Products**
1. **Shapefile** with refined rims + probability scores
2. **PNG** showing refined rim positions (colored by probability)
3. **CSFD plot** for refined craters
4. **Difference PNG** showing original vs refined rims

✓ **Enhanced Algorithm**
- Builds on existing topographic rim fitting
- Adds Canny edge detection
- Incorporates gradient analysis
- Calculates multiple quality metrics

---

## Required Inputs

### 1. Input Shapefile (from Step 0)

**Source:** Output from `input_module.py`

**File:** `craters_initial.shp`

**Required Fields:**
- `geometry`: Crater circles (polygons)
- `diameter`: Crater diameter in meters

**Optional Fields:**
- `UFID`: Crater identifier
- `lat`, `lon`: Geographic coordinates

### 2. Contrast Image

**Format:** GeoTIFF (`.tif`)

**Purpose:** Edge detection and visual rim cues

**Requirements:**
- Same extent as DTM
- Same pixel scale as DTM
- Georeferenced

### 3. DTM (Digital Terrain Model)

**Format:** GeoTIFF (`.tif`)

**Purpose:** Topographic rim detection

**Requirements:**
- Elevation raster
- Same CRS as shapefile

---

## Usage

### Basic Command

```bash
python refine_crater_rims.py \
    --shapefile results/craters_initial.shp \
    --image data/image.tif \
    --dtm data/dtm.tif \
    --output results/
```

### With Custom Parameters

```bash
python refine_crater_rims.py \
    --shapefile craters_initial.shp \
    --image image.tif \
    --dtm dtm.tif \
    --output refined/ \
    --min-diameter 50 \
    --inner-radius 0.75 \
    --outer-radius 1.25 \
    --plot-individual  # Generate plots for each crater
```

### From Python

```python
from crater_analysis.refine_crater_rim import refine_crater_rims

results = refine_crater_rims(
    input_shapefile='results/craters_initial.shp',
    dem_path='data/dtm.tif',
    image_path='data/image.tif',
    output_dir='results/refined/',
    min_diameter=60,
    inner_radius=0.8,
    outer_radius=1.2
)

print(f"Refined: {results['shapefile']}")
print(f"Mean probability: {results['statistics']['mean_probability']:.3f}")
```

---

## Algorithm Details

### Step-by-Step Process

#### **For Each Crater:**

**1. Topographic Rim Detection** (Existing Algorithm)
```
- Extract DEM region (3R × 3R box)
- Remove regional slope (optional)
- Sample elevation along 72 radial profiles
- Find peaks at each azimuth
- Fit optimal circle to detected rim points
- Calculate position errors
```

**2. Computer Vision Enhancement** (NEW)
```
- Extract contrast image region
- Apply Canny edge detection
- Compute edge strength in annulus (0.8R-1.2R)
- Detect rim radius from edge profiles
- Calculate edge-based confidence
```

**3. Quality Assessment** (NEW)
```
Topographic Quality:
  - Count azimuths with clear peaks
  - Measure peak prominence
  - Score: 0-1 based on rim clarity

Edge Strength:
  - Fraction of annulus pixels with edges
  - Consistency of edge radius
  - Score: 0-1 based on edge clarity

Radius Agreement:
  - Compare topographic vs edge-detected radii
  - Score: exp(-difference/tolerance)
  - High agreement = high score
```

**4. Probability Calculation** (NEW)
```python
probability = (
    0.4 × topographic_quality +
    0.3 × edge_strength +
    0.2 × radius_agreement +
    0.1 × error_score
)
```

**5. Final Rim Position**
```
- Primary: Topographic center (most reliable)
- Radius: Weighted combination (70% topo, 30% edge)
- Fallback to topography if edge detection fails
```

---

## Probability Scoring

### Components

| Component | Weight | Description |
|-----------|--------|-------------|
| **Topographic Quality** | 40% | Clarity of elevation peaks |
| **Edge Strength** | 30% | Strength of image edges |
| **Radius Agreement** | 20% | Agreement between methods |
| **Error Score** | 10% | Low fitting error |

### Probability Ranges

| Range | Confidence | Interpretation |
|-------|-----------|----------------|
| **> 0.7** | High | Strong rim signal in both DEM and image |
| **0.5 - 0.7** | Medium | Moderate rim signal, some uncertainty |
| **< 0.5** | Low | Weak/degraded rim, high uncertainty |

### Example Scores

**High Probability (0.85):**
```
- Topographic quality: 0.9 (clear peaks)
- Edge strength: 0.8 (strong edges)
- Radius agreement: 0.9 (methods agree)
- Error: 2m (small)
→ High confidence, reliable rim
```

**Low Probability (0.35):**
```
- Topographic quality: 0.3 (degraded peaks)
- Edge strength: 0.2 (weak edges)
- Radius agreement: 0.4 (methods disagree)
- Error: 15m (large)
→ Low confidence, questionable rim
```

---

## Outputs

### 1. Refined Shapefile: `craters_refined.shp`

**New Fields Added:**

| Field | Type | Description |
|-------|------|-------------|
| `rim_probability` | Float | Overall detection probability (0-1) |
| `topo_quality` | Float | Topographic rim quality (0-1) |
| `edge_strength` | Float | Image edge strength (0-1) |
| `radius_agreement` | Float | Agreement between methods (0-1) |
| `err_x0` | Float | X-position error (meters) |
| `err_y0` | Float | Y-position error (meters) |
| `err_r` | Float | Radius error (meters) |
| `radius_orig` | Float | Original radius (meters) |
| `radius_refined` | Float | Refined radius (meters) |
| `center_shift` | Float | Center displacement (meters) |

**Example Data:**
```
UFID  | diameter | rim_probability | topo_quality | edge_strength | radius_refined
------|----------|-----------------|--------------|---------------|---------------
c001  | 169.1    | 0.78           | 0.85         | 0.72          | 86.2
c002  | 307.7    | 0.82           | 0.88         | 0.78          | 155.4
c003  | 97.5     | 0.55           | 0.62         | 0.48          | 49.8
```

### 2. Refined Positions Plot: `craters_refined_positions.png`

**Content:**
- Contrast image (grayscale background)
- Crater circles colored by rim probability
- Color scale: Red (low) → Yellow (medium) → Green (high)
- Title: "Refined Crater Rims, N = X, Mean Probability = Y"

**Example:**
```
┌──────────────────────────────────────────────┐
│ Refined Crater Rims, N = 19                  │
│ Mean Probability = 0.72                      │
│                                              │
│    [Image with colored crater circles]       │
│                                              │
│  ○ Green (high prob)                         │
│     ○ Yellow (medium)                        │
│        ○ Red (low prob)                      │
│                                              │
│  Color scale: 0.0 ───────────── 1.0         │
│               Red    Yellow   Green          │
└──────────────────────────────────────────────┘
```

### 3. CSFD Plot: `craters_refined_csfd.png`

**Content:**
- Crater Size-Frequency Distribution
- Log-log plot
- Uses refined diameters
- Same format as Step 0 CSFD

### 4. Rim Differences Plot: `craters_rim_differences.png`

**Content:**
- Contrast image (faded background)
- Original rims (blue, dashed lines)
- Refined rims (red, solid lines)
- Yellow arrows showing center shifts
- Statistics box with:
  - Mean center shift
  - Max center shift
  - Mean radius change

**Example:**
```
┌──────────────────────────────────────────────┐
│ Crater Rim Refinement Comparison             │
│                                              │
│  ╭─ ─ ─╮  Blue dashed = Original            │
│ ╱       ╲                                    │
│ ─ ─ ─ ─ ─ Red solid = Refined               │
│ ╲   →  ╱  Yellow arrow = Center shift       │
│  ╰─────╯                                     │
│                                              │
│ ┌─────────────────────────┐                 │
│ │ Mean center shift: 8.5m │                 │
│ │ Max center shift: 18.9m │                 │
│ │ Mean radius change: +2% │                 │
│ └─────────────────────────┘                 │
└──────────────────────────────────────────────┘
```

---

## Computer Vision Enhancements

### Edge Detection

**Method:** Canny Edge Detector

**Process:**
1. Extract image region around crater (3R box)
2. Normalize to 0-255
3. Apply Canny edge detection (thresholds: 50, 150)
4. Compute edge density in annulus (0.8R-1.2R)
5. Sample edges along radial profiles
6. Detect rim from edge peaks

**Edge Strength Score:**
```python
edge_strength = (pixels_with_edges) / (total_annulus_pixels)
```

**Example:**
```
Annulus contains 1000 pixels
200 pixels have edges
→ Edge strength = 0.20
```

### Gradient Analysis

**Purpose:** Find inflection points when peak detection fails

**Process:**
1. Compute elevation gradient: `dE/dr`
2. Compute second derivative: `d²E/dr²`
3. Rim = position of minimum second derivative
4. Used as fallback when no clear peak

**Mathematical:**
```
First derivative:  dE/dr
Second derivative: d²E/dr²
Rim position:      argmin(d²E/dr²)
```

### Radius Detection from Edges

**Process:**
1. For each of 72 azimuths:
   - Sample edges along radial line
   - Find peaks in edge profile
   - Use strongest peak as rim position
2. Calculate median of detected radii
3. Compare with topographic radius

**Agreement Calculation:**
```python
radius_diff = abs(radius_topo - radius_edge)
agreement = exp(-radius_diff / (0.2 * radius_original))
```

---

## Parameters

### Search Range

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_diameter` | 60 | Minimum crater diameter (meters) |
| `inner_radius` | 0.8 | Start search at 0.8R |
| `outer_radius` | 1.2 | End search at 1.2R |

**Effect of Range:**
```
Narrow (0.9R - 1.1R):      Wide (0.7R - 1.3R):
  More precise               More robust
  May miss degraded rims     May include noise
```

### Topography Processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `remove_external_topo` | True | Remove regional slope |

**Effect:**
```
With removal:              Without removal:
  Rim peaks clearer          May miss rim on slope
  Better for tilted terrain  Faster processing
```

### Edge Detection

**Canny Thresholds:**
- Low threshold: 50
- High threshold: 150

**Adjusting for different images:**
- Bright, high-contrast: Increase thresholds
- Dark, low-contrast: Decrease thresholds

---

## Quality Metrics

### Topographic Quality

**Calculation:**
```python
# For each azimuth, detect if peak exists
peak_fraction = azimuths_with_peaks / total_azimuths

# Average prominence of detected peaks
avg_prominence = mean(peak_prominences)
prominence_score = min(avg_prominence / (0.1 * radius), 1.0)

# Combined quality
quality = 0.6 × peak_fraction + 0.4 × prominence_score
```

**Interpretation:**
- High (>0.7): Clear rim at most azimuths
- Medium (0.5-0.7): Rim visible but degraded
- Low (<0.5): Weak or missing rim signal

### Edge Strength

**Calculation:**
```python
# Extract annulus region
annulus = image_region[(dist >= 0.8R) & (dist <= 1.2R)]

# Apply edge detection
edges = canny_edge_detection(annulus)

# Compute strength
edge_strength = edges.sum() / annulus.size
```

**Interpretation:**
- High (>0.3): Strong edges around rim
- Medium (0.15-0.3): Moderate edges
- Low (<0.15): Weak or no edges

---

## Integration with Workflow

### Complete Pipeline

```bash
# Step 0: Input Processing
python process_crater_inputs.py \
    --image data/image.tif \
    --dtm data/dtm.tif \
    --craters data/craters.csv \
    --output step0/

# Step 2: Rim Refinement (NEW)
python refine_crater_rims.py \
    --shapefile step0/craters_initial.shp \
    --image data/image.tif \
    --dtm data/dtm.tif \
    --output step2/

# Step 3: Morphometry Analysis (TODO)
python analyze_morphometry.py \
    --shapefile step2/craters_refined.shp \
    --dtm data/dtm.tif \
    --output step3/
```

### Filtering by Probability

```python
import geopandas as gpd

# Load refined craters
craters = gpd.read_file('craters_refined.shp')

# Filter by probability
high_conf = craters[craters['rim_probability'] > 0.7]
print(f"High confidence craters: {len(high_conf)}")

# Save filtered subset
high_conf.to_file('craters_high_confidence.shp')
```

### Analyzing Quality

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load refined data
craters = gpd.read_file('craters_refined.shp')

# Plot probability vs diameter
plt.figure(figsize=(10, 6))
plt.scatter(craters['diameter'], craters['rim_probability'], alpha=0.5)
plt.xlabel('Diameter (m)')
plt.ylabel('Rim Detection Probability')
plt.title('Rim Detection Quality vs Crater Size')
plt.grid(True, alpha=0.3)
plt.show()

# Summary statistics
print(f"Mean probability: {craters['rim_probability'].mean():.3f}")
print(f"Std deviation: {craters['rim_probability'].std():.3f}")
print(f"High confidence (>0.7): {(craters['rim_probability'] > 0.7).sum()}")
```

---

## Troubleshooting

### Issue: All probabilities are low

**Possible Causes:**
1. Poor contrast in image
2. Heavily degraded craters
3. Resolution too coarse
4. Wrong coordinate system

**Solutions:**
```bash
# Check image quality
gdalinfo image.tif

# Try without removing regional topography
python refine_crater_rims.py ... --keep-regional-topo

# Increase search range
python refine_crater_rims.py ... --inner-radius 0.7 --outer-radius 1.3
```

### Issue: Edge detection fails

**Causes:**
- Low contrast image
- Image and DEM misalignment
- Wrong CRS

**Solutions:**
```bash
# Check CRS match
gdalinfo image.tif | grep "Coordinate System"
gdalinfo dtm.tif | grep "Coordinate System"

# Check alignment
gdalinfo image.tif | grep "Origin"
gdalinfo dtm.tif | grep "Origin"
```

### Issue: Large center shifts

**Causes:**
- Initial positions inaccurate
- Overlapping craters
- Non-circular craters

**Solutions:**
```python
# Filter by center shift
craters = gpd.read_file('craters_refined.shp')
reasonable = craters[craters['center_shift'] < 20]  # < 20m shift
```

### Issue: Memory errors

**Cause:** Processing too many large craters

**Solution:**
```bash
# Process in batches by diameter
python refine_crater_rims.py ... --min-diameter 100  # Large craters
python refine_crater_rims.py ... --min-diameter 60 --max-diameter 100  # Medium
# (Note: --max-diameter would need to be added to code)
```

---

## Advanced Usage

### Custom Probability Weights

Modify `compute_rim_probability()` in the code:

```python
# In refine_crater_rim.py, adjust weights:
w_topo = 0.5      # Increase topography weight
w_edge = 0.2      # Decrease edge weight
w_agree = 0.2
w_error = 0.1
```

### Different Edge Detection

Replace Canny with other methods:

```python
# In compute_edge_strength(), replace Canny with:

# Sobel edges
edges = cv2.Sobel(region_norm, cv2.CV_64F, 1, 1, ksize=3)

# Laplacian
edges = cv2.Laplacian(region_norm, cv2.CV_64F)
```

### Export Probability Statistics

```python
from crater_analysis.refine_crater_rim import refine_crater_rims

results = refine_crater_rims(...)

# Export statistics
stats_df = pd.DataFrame([results['statistics']])
stats_df.to_csv('refinement_statistics.csv', index=False)

# Export per-crater metrics
craters = results['refined_data']
metrics = craters[['UFID', 'rim_probability', 'topo_quality',
                   'edge_strength', 'radius_agreement']]
metrics.to_csv('crater_quality_metrics.csv', index=False)
```

---

## Performance

### Computational Cost

**Per Crater:**
- Topographic refinement: ~2-5 seconds
- Edge detection: ~0.5-1 second
- Total: ~3-6 seconds per crater

**For 100 Craters:**
- Expected time: 5-10 minutes
- Memory: ~1-2 GB (depends on DEM size)

### Optimization Tips

1. **Disable individual plotting:**
   ```bash
   # Don't use --plot-individual flag
   ```

2. **Process subset first:**
   ```bash
   # Test on small diameter range
   python refine_crater_rims.py ... --min-diameter 150
   ```

3. **Parallel processing (future):**
   ```python
   # Could add multiprocessing in future version
   ```

---

## See Also

- **Input Module:** [INPUT_MODULE_GUIDE.md](INPUT_MODULE_GUIDE.md)
- **Step 1:** [STEP1_PREPARE_GEOMETRIES_EXPLAINED.md](STEP1_PREPARE_GEOMETRIES_EXPLAINED.md)
- **Step 2 Original:** [STEP2_REFINE_RIMS_EXPLAINED.md](STEP2_REFINE_RIMS_EXPLAINED.md)
- **Main README:** [../README.md](../README.md)

---

## Summary

**Inputs:**
- Shapefile from Step 0
- Contrast image (GeoTIFF)
- DTM (GeoTIFF)

**Algorithm:**
- Topographic rim detection (existing)
- Computer vision edge detection (NEW)
- Probability scoring (NEW)
- Quality metrics (NEW)

**Outputs:**
- Refined shapefile with probability scores
- 3 PNG visualizations

**Key Innovation:**
Combines multiple detection methods and provides confidence scores, allowing users to filter results by quality.

# Crater Morphometry Analysis Module - Complete Guide

## Overview

The `analyze_morphometry.py` module performs **Step 3** of the crater analysis workflow. It measures crater morphology using two complementary methods, providing comprehensive depth and degradation analysis with full error propagation.

## Key Features

✓ **Dual-Method Depth Estimation**
- Method 1: Rim perimeter analysis (proven, reliable)
- Method 2: 2D Gaussian floor fitting (NEW, robust to noise)
- Combined estimate with uncertainty weighting

✓ **Enhanced Error Propagation**
- Incorporates rim detection probability from Step 2
- Propagates all measurement uncertainties
- Provides total error including all sources

✓ **Five Output Products**
1. **Shapefile** with morphometry measurements (both methods)
2. **Scatter plots** (depth vs diameter, d/D vs diameter) with error bars
3. **Probability distributions** (joint and marginal)
4. **CSV** with all morphometry data
5. **CSV** with conditional probabilities P(d|D) and P(D|d)

✓ **Quality Metrics**
- Gaussian fit quality score (R²)
- Floor uncertainty estimates
- Method agreement indicators

---

## Required Inputs

### 1. Refined Shapefile (from Step 2)

**Source:** Output from `refine_crater_rim.py`

**File:** `craters_refined.shp`

**Required Fields:**
- `geometry`: Refined crater circles
- `diameter`: Crater diameter (meters)
- `rim_probability`: Rim detection confidence (0-1)
- `err_r`: Radius error from refinement (meters)

**Used For:**
- Crater locations and sizes
- Rim probability for error propagation
- Radius uncertainty estimates

### 2. DTM (Digital Terrain Model)

**Format:** GeoTIFF (`.tif`)

**Purpose:** Elevation measurements for depth calculation

**Requirements:**
- Same CRS as shapefile
- Covers all crater locations
- Sufficient resolution for floor detection

### 3. Contrast Image/Orthophoto

**Format:** GeoTIFF (`.tif`)

**Purpose:** Required by existing crater tools (method 1)

**Requirements:**
- Same extent and resolution as DTM
- Georeferenced

---

## Usage

### Basic Command

```bash
python analyze_crater_morphometry.py \
    --shapefile results/craters_refined.shp \
    --dtm data/dtm.tif \
    --image data/image.tif \
    --output results/morphometry/
```

### With Custom Parameters

```bash
python analyze_crater_morphometry.py \
    --shapefile craters_refined.shp \
    --dtm dtm.tif \
    --image image.tif \
    --output morphometry/ \
    --min-diameter 100 \
    --plot-individual
```

### From Python

```python
from crater_analysis.analyze_morphometry import analyze_crater_morphometry

results = analyze_crater_morphometry(
    input_shapefile='results/craters_refined.shp',
    output_shapefile='results/morphometry/craters_morphometry.shp',
    dem_path='data/dtm.tif',
    orthophoto_path='data/image.tif',
    output_dir='results/morphometry/',
    min_diameter=60.0,
    remove_external_topo=True
)

print(f"Analyzed: {results['statistics']['total_craters']} craters")
print(f"Method 1 mean d/D: {results['statistics']['method1']['mean_d_D']:.3f}")
print(f"Method 2 mean d/D: {results['statistics']['method2']['mean_d_D']:.3f}")
```

---

## Algorithm Details

### Method 1: Rim Perimeter Analysis (Existing)

**Process:**
```
1. Extract crater DEM region (3R × 3R box)
2. Remove regional slope (optional, default=True)
3. Detect rim as perimeter pixels
4. Compute mean rim height ± std dev
5. Find floor as minimum elevation
6. Calculate depth = rim_height - floor_height
7. Compute d/D = depth / diameter
```

**Advantages:**
- Proven, reliable method
- Uses entire rim perimeter
- Natural uncertainty from rim variance

**Limitations:**
- Floor = single minimum pixel (sensitive to noise)
- Assumes rim is at geometry boundary
- Can be affected by outliers

---

### Method 2: 2D Gaussian Floor Fitting (NEW)

**Mathematical Foundation:**

The 2D Gaussian function:
```
G(x, y) = A × exp(-[a(x-x₀)² + 2b(x-x₀)(y-y₀) + c(y-y₀)²]) + offset

where:
  a = cos²θ/(2σₓ²) + sin²θ/(2σᵧ²)
  b = -sin(2θ)/(4σₓ²) + sin(2θ)/(4σᵧ²)
  c = sin²θ/(2σₓ²) + cos²θ/(2σᵧ²)

Parameters:
  A = amplitude (peak height)
  (x₀, y₀) = center position
  σₓ, σᵧ = standard deviations in x, y
  θ = rotation angle
  offset = baseline
```

**Process:**

**Step 1: Invert Crater Elevation**
```python
# Transform crater depression into peak
min_elev = np.min(crater_dem)
max_elev = np.max(crater_dem)
inverted_dem = max_elev - crater_dem

# Now crater floor (lowest point) becomes highest point
```

**Step 2: Initial Parameter Guesses**
```python
# Amplitude: max height of inverted crater
amplitude_guess = np.max(inverted_dem)

# Center: position of deepest point (now highest)
deepest_idx = np.argmin(original_crater_dem)
xo_guess, yo_guess = get_coordinates(deepest_idx)

# Sigma: crater floor typically ~1/3 of diameter
crater_radius_pixels = sqrt(crater_area / π)
sigma_guess = crater_radius_pixels / 3.0

# No rotation initially
theta_guess = 0.0

# Offset: should be near zero (constrained later)
offset_guess = 0.0
```

**Step 3: Constrained Fitting**
```python
# Bounds to prevent overshoot/undershoot
lower_bounds = [
    0,                      # Amplitude ≥ 0
    0,                      # xo ≥ 0
    0,                      # yo ≥ 0
    radius/10,              # σx min
    radius/10,              # σy min
    -π,                     # θ
    0                       # Offset ≥ 0 (NO UNDERSHOOT)
]

upper_bounds = [
    amplitude_guess × 1.2,  # Amplitude limited (NO OVERSHOOT)
    image_width,            # xo max
    image_height,           # yo max
    crater_radius,          # σx max
    crater_radius,          # σy max
    π,                      # θ
    amplitude_guess × 0.1   # Small offset allowed
]

# Fit using trust region reflective algorithm
params_optimal, covariance = curve_fit(
    gaussian_2d,
    (x_coords, y_coords),
    inverted_elevations,
    p0=initial_guess,
    bounds=(lower_bounds, upper_bounds),
    method='trf'
)
```

**Step 4: Extract Floor Elevation**
```python
# Gaussian peak in inverted space
gaussian_peak_inverted = amplitude + offset

# Transform back to original elevation
floor_elevation = max_elev - gaussian_peak_inverted

# Safety check: floor cannot be below actual minimum
if floor_elevation < min_elev - tolerance:
    floor_elevation = min_elev
```

**Step 5: Compute Depth**
```python
# Get rim height from Method 1 (more reliable for rim)
rim_height = method1_rim_height

# Depth with Gaussian floor
depth = rim_height - floor_elevation

# Uncertainty: combine rim uncertainty + Gaussian fit uncertainty
floor_uncertainty = sqrt(covariance[0][0])  # From amplitude error
total_uncertainty = sqrt(rim_unc² + floor_unc²)

depth = depth ± total_uncertainty
```

**Step 6: Quality Assessment**
```python
# Compute fit quality (R-squared)
fitted_values = gaussian_2d((x, y), *params_optimal)
residuals = data - fitted_values

SS_res = sum(residuals²)
SS_tot = sum((data - mean(data))²)

R² = 1 - (SS_res / SS_tot)

# R² → 1.0 means excellent fit
# R² < 0.7 means poor fit
```

**Advantages:**
- Robust to noise (averages over entire floor region)
- Provides uncertainty estimate from fit covariance
- Handles irregular crater shapes
- Fit quality metric identifies problematic cases

**Limitations:**
- Assumes approximately bowl-shaped floor
- Requires sufficient floor area for reliable fit
- Can fail for highly irregular craters

---

## Error Propagation

### Sources of Uncertainty

| Source | Method 1 | Method 2 |
|--------|----------|----------|
| **Rim height** | Std dev of rim pixels | From Method 1 |
| **Floor height** | None (single minimum) | Gaussian fit uncertainty |
| **Radius** | From Step 2 refinement | From Step 2 refinement |
| **Rim probability** | Step 2 confidence | Step 2 confidence |

### Rim Probability Error Contribution

**Concept:** Low rim probability → higher depth uncertainty

```python
# Convert rim probability to uncertainty factor
prob_uncertainty_factor = 1.0 - rim_probability

# Examples:
# rim_probability = 0.9 → factor = 0.1 (low additional uncertainty)
# rim_probability = 0.5 → factor = 0.5 (medium additional uncertainty)
# rim_probability = 0.3 → factor = 0.7 (high additional uncertainty)

# Additional depth uncertainty
probability_error = abs(depth) × prob_uncertainty_factor × 0.5

# Total uncertainty
total_uncertainty = sqrt(measurement_unc² + probability_error²)
```

**Example:**

**Case 1: High rim probability**
```
Depth: 20.0 ± 1.0 m (measurement uncertainty)
Rim probability: 0.9
Probability error: 20.0 × 0.1 × 0.5 = 1.0 m
Total uncertainty: sqrt(1.0² + 1.0²) = 1.41 m
Final depth: 20.0 ± 1.41 m
```

**Case 2: Low rim probability**
```
Depth: 20.0 ± 1.0 m (measurement uncertainty)
Rim probability: 0.3
Probability error: 20.0 × 0.7 × 0.5 = 7.0 m
Total uncertainty: sqrt(1.0² + 7.0²) = 7.07 m
Final depth: 20.0 ± 7.07 m
```

### Combined Method Uncertainty

Both methods are combined using **inverse-variance weighting**:

```python
# Variances
var1 = (uncertainty_method1)²
var2 = (uncertainty_method2)²

# Weights (inverse of variance)
w1 = 1 / var1
w2 = 1 / var2
w_total = w1 + w2

# Combined estimate
d_D_combined = (w1 × d_D_method1 + w2 × d_D_method2) / w_total

# Combined uncertainty
uncertainty_combined = sqrt(1 / w_total)
```

**Interpretation:**
- More precise method gets higher weight
- Combined uncertainty is smaller than either individual uncertainty
- If methods disagree significantly, uncertainty increases

---

## Outputs

### Output 1: Morphometry Shapefile (`craters_morphometry.shp`)

**New Fields Added:**

#### Method 1 Fields
| Field | Type | Description |
|-------|------|-------------|
| `diam_m1` | Float | Diameter (m) |
| `depth_m1` | Float | Crater depth (m) |
| `depth_err_m1` | Float | Depth uncertainty (m) |
| `d_D_m1` | Float | Depth-to-diameter ratio |
| `d_D_err_m1` | Float | d/D uncertainty |
| `rim_height_m1` | Float | Mean rim elevation (m) |
| `floor_height_m1` | Float | Floor elevation (m) |
| `total_error_m1` | Float | Total uncertainty including rim probability |
| `prob_error_m1` | Float | Uncertainty contribution from rim probability |

#### Method 2 Fields
| Field | Type | Description |
|-------|------|-------------|
| `diam_m2` | Float | Diameter (m) |
| `depth_m2` | Float | Crater depth (m) |
| `depth_err_m2` | Float | Depth uncertainty (m) |
| `d_D_m2` | Float | Depth-to-diameter ratio |
| `d_D_err_m2` | Float | d/D uncertainty |
| `rim_height_m2` | Float | Mean rim elevation (m) |
| `floor_height_m2` | Float | Gaussian-fitted floor elevation (m) |
| `floor_unc_m2` | Float | Floor uncertainty from fit (m) |
| `fit_quality_m2` | Float | Gaussian fit quality (R², 0-1) |
| `total_error_m2` | Float | Total uncertainty including rim probability |
| `prob_error_m2` | Float | Uncertainty contribution from rim probability |

#### Combined Fields
| Field | Type | Description |
|-------|------|-------------|
| `d_D_combined` | Float | Weighted average d/D |
| `d_D_err_combined` | Float | Combined uncertainty |

**Example Data:**
```
UFID | diam_m1 | depth_m1 | d_D_m1 | total_error_m1 | depth_m2 | d_D_m2 | fit_quality_m2 | d_D_combined
-----|---------|----------|--------|----------------|----------|--------|----------------|-------------
c001 | 169.1   | 23.3     | 0.138  | 1.2            | 22.8     | 0.135  | 0.92           | 0.136
c002 | 307.7   | 45.2     | 0.147  | 2.1            | 44.9     | 0.146  | 0.88           | 0.146
c003 | 97.5    | 12.1     | 0.124  | 3.5            | 11.8     | 0.121  | 0.75           | 0.122
```

---

### Output 2: Scatter Plots (`morphometry_scatter_plots.png`)

**Two-panel figure:**

**Panel 1: Depth vs Diameter**
```
Features:
- Blue circles: Method 1 (rim perimeter)
- Red squares: Method 2 (Gaussian fitting)
- Error bars on both axes
- Grid for easy reading
- Legend identifying methods
```

**Panel 2: d/D Ratio vs Diameter**
```
Features:
- Blue circles: Method 1
- Red squares: Method 2
- Error bars showing total uncertainty
- Horizontal dashed line at d/D = 0.2 (fresh crater reference)
- Helps identify degradation trends
```

**Interpretation:**
- Fresh craters: d/D ≈ 0.15-0.20
- Degraded craters: d/D < 0.10
- Large error bars → low rim probability
- Method disagreement → irregular crater shape

---

### Output 3: Probability Distributions (`probability_distributions.png`)

**Two-panel figure:**

**Panel 1: Joint Probability Distribution P(depth, diameter)**
```
Features:
- 2D filled contour plot
- Colors represent probability density (viridis colormap)
- White dots: actual crater measurements
- Shows correlation between depth and diameter
```

**Uses:**
- Identify depth-diameter relationships
- Detect outliers (craters far from high-density regions)
- Understand crater population characteristics

**Panel 2: Marginal Probability Distribution P(d/D)**
```
Features:
- Blue filled curve: Kernel Density Estimate (KDE)
- Gray histogram: Binned data
- Red dashed line: Mean d/D
- Orange dotted lines: ± 1 standard deviation
```

**Uses:**
- Estimate population mean degradation state
- Identify bimodal distributions (multiple crater populations)
- Compare to theoretical/other datasets

---

### Output 4: Morphometry Data CSV (`morphometry_data.csv`)

**All shapefile fields except geometry**

**Format:**
```csv
crater_id,diam_m1,depth_m1,depth_err_m1,d_D_m1,d_D_err_m1,total_error_m1,diam_m2,depth_m2,depth_err_m2,d_D_m2,d_D_err_m2,fit_quality_m2,total_error_m2,d_D_combined,d_D_err_combined
c001,169.1,23.3,0.5,0.138,0.003,1.2,169.1,22.8,0.8,0.135,0.005,0.92,1.5,0.136,0.004
c002,307.7,45.2,1.2,0.147,0.004,2.1,307.7,44.9,1.5,0.146,0.005,0.88,2.3,0.146,0.004
```

**Uses:**
- Import into Excel, MATLAB, R, Python
- Statistical analysis
- Data visualization
- Machine learning applications

---

### Output 5: Conditional Probability CSV (`conditional_probability.csv`)

**Estimates P(d|D) and P(D|d) in bins**

**Format:**
```csv
diameter_bin,depth_bin,diameter_center,depth_center,P_d_given_D,P_D_given_d,count,mean_d_D,std_d_D
100-150,10-15,125.0,12.5,0.3,0.25,5,0.110,0.015
100-150,15-20,125.0,17.5,0.5,0.40,8,0.142,0.020
150-200,15-20,175.0,17.5,0.4,0.35,6,0.102,0.018
```

**Fields:**
- `diameter_bin`: Diameter range (m)
- `depth_bin`: Depth range (m)
- `diameter_center`: Bin center for diameter
- `depth_center`: Bin center for depth
- `P_d_given_D`: Probability of depth given diameter
- `P_D_given_d`: Probability of diameter given depth
- `count`: Number of craters in bin
- `mean_d_D`: Mean d/D ratio in bin
- `std_d_D`: Std dev of d/D in bin

**Uses:**
- Predict expected depth for a given diameter
- Estimate crater size from observed depth
- Validate theoretical models
- Quality control (identify unusual crater relationships)

**Example Query:**
"For a 150m diameter crater, what is the most likely depth?"

→ Find all bins with diameter_center ≈ 150m, select bin with highest P_d_given_D

---

## Parameters

### Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_diameter` | 60.0 | Minimum crater diameter (m) |
| `remove_external_topo` | True | Remove regional slope |
| `plot_individual` | False | Plot each crater individually |

### Gaussian Fitting Parameters (Internal)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_iterations` | 10 | Maximum fitting iterations |
| `tolerance` | 0.1 m | Floor elevation convergence tolerance |
| `sigma_guess` | radius/3 | Initial floor width estimate |

**Adjusting for different datasets:**

```python
# For larger craters (>500m)
sigma_guess = crater_radius / 2.5  # Wider floors

# For smaller craters (<100m)
sigma_guess = crater_radius / 3.5  # Narrower floors

# For highly degraded craters
lower_bounds[6] = 0  # Allow small offset
max_iterations = 20  # More iterations
```

---

## Quality Metrics

### Method 1: Rim Perimeter Reliability

**Indicators:**
- Low `depth_err_m1`: Consistent rim height around perimeter
- `total_error_m1` close to `depth_err_m1`: High rim probability

**Good quality:**
```
depth_m1 = 20.0 ± 0.8 m  (4% uncertainty)
total_error_m1 = 1.2 m
rim_probability = 0.85
```

**Poor quality:**
```
depth_m1 = 20.0 ± 3.5 m  (17.5% uncertainty)
total_error_m1 = 8.2 m
rim_probability = 0.35
```

### Method 2: Gaussian Fit Quality

**`fit_quality_m2` (R² value):**

| Range | Interpretation |
|-------|----------------|
| **> 0.90** | Excellent fit, crater is bowl-shaped |
| **0.80 - 0.90** | Good fit, reliable floor estimate |
| **0.70 - 0.80** | Moderate fit, floor estimate usable |
| **< 0.70** | Poor fit, irregular floor or insufficient data |

**Causes of poor fit:**
- Flat-floored crater (not Gaussian)
- Multiple overlapping craters
- Insufficient floor area
- High noise in DEM

**Action for poor fits:**
- Trust Method 1 over Method 2
- Increase `max_iterations`
- Check for data quality issues

### Method Agreement

**Compare `d_D_m1` and `d_D_m2`:**

```python
agreement = abs(d_D_m1 - d_D_m2) / mean([d_D_m1, d_D_m2])

# Good agreement: < 10%
# Moderate: 10-20%
# Poor: > 20%
```

**High agreement:**
```
d_D_m1 = 0.145
d_D_m2 = 0.148
→ Agreement: 2% (excellent)
```

**Poor agreement:**
```
d_D_m1 = 0.180
d_D_m2 = 0.120
→ Agreement: 40% (investigate crater)
```

---

## Integration with Workflow

### Complete Pipeline (Steps 0 → 2 → 3)

```bash
# Step 0: Input Processing
python process_crater_inputs.py \
    --image data/image.tif \
    --dtm data/dtm.tif \
    --craters data/craters.csv \
    --output step0/

# Step 2: Rim Refinement
python refine_crater_rims.py \
    --shapefile step0/craters_initial.shp \
    --image data/image.tif \
    --dtm data/dtm.tif \
    --output step2/

# Step 3: Morphometry Analysis (NEW)
python analyze_crater_morphometry.py \
    --shapefile step2/craters_refined.shp \
    --dtm data/dtm.tif \
    --image data/image.tif \
    --output step3/
```

### Filtering Results by Quality

```python
import geopandas as gpd

# Load morphometry results
craters = gpd.read_file('craters_morphometry.shp')

# Filter by Gaussian fit quality
high_quality = craters[craters['fit_quality_m2'] > 0.85]
print(f"High quality fits: {len(high_quality)}/{len(craters)}")

# Filter by total uncertainty
precise = craters[craters['total_error_m2'] < 2.0]
print(f"Precise measurements (error < 2m): {len(precise)}/{len(craters)}")

# Filter by method agreement
def compute_agreement(row):
    return abs(row['d_D_m1'] - row['d_D_m2']) / ((row['d_D_m1'] + row['d_D_m2']) / 2)

craters['agreement'] = craters.apply(compute_agreement, axis=1)
consistent = craters[craters['agreement'] < 0.15]  # < 15% difference
print(f"Consistent methods: {len(consistent)}/{len(craters)}")

# Combined quality filter
best_craters = craters[
    (craters['fit_quality_m2'] > 0.8) &
    (craters['total_error_m2'] < 3.0) &
    (craters['agreement'] < 0.20)
]
best_craters.to_file('craters_best_quality.shp')
```

### Analyzing Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load morphometry CSV
df = pd.read_csv('morphometry_data.csv')

# Compare methods
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(df['d_D_m1'], df['d_D_m2'], alpha=0.6)
ax.plot([0, 0.2], [0, 0.2], 'r--', label='1:1 line')
ax.set_xlabel('d/D Method 1 (rim perimeter)')
ax.set_ylabel('d/D Method 2 (Gaussian)')
ax.set_title('Method Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('method_comparison.png', dpi=300)

# Degradation vs size
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['diam_m2'], df['d_D_m2'],
                    c=df['fit_quality_m2'], cmap='RdYlGn',
                    s=50, alpha=0.7)
plt.colorbar(scatter, label='Fit Quality')
ax.set_xlabel('Diameter (m)')
ax.set_ylabel('d/D Ratio')
ax.set_title('Crater Degradation (colored by fit quality)')
ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Fresh')
ax.legend()
plt.savefig('degradation_analysis.png', dpi=300)
```

---

## Troubleshooting

### Issue: All Gaussian fits fail (fit_quality_m2 < 0.5)

**Possible Causes:**
1. DEM resolution too coarse
2. Craters too small
3. Heavily degraded population

**Solutions:**
```bash
# Use higher resolution DEM if available

# Increase minimum diameter
python analyze_crater_morphometry.py ... --min-diameter 150

# Rely on Method 1 for this dataset
```

### Issue: Large discrepancy between Method 1 and Method 2

**Causes:**
- Irregular crater shape (non-circular floor)
- Floor not bowl-shaped (flat or stepped)
- Overlapping craters

**Diagnosis:**
```python
# Identify problematic craters
craters['diff'] = abs(craters['d_D_m1'] - craters['d_D_m2'])
outliers = craters[craters['diff'] > 0.05]

# Plot individual crater
python analyze_crater_morphometry.py ... --plot-individual
```

**Solution:**
- For irregular craters: Trust Method 1
- For overlapping craters: Exclude from analysis
- For flat floors: Method 2 will fail (low fit_quality_m2)

### Issue: High uncertainty (total_error > 5m)

**Causes:**
- Low rim probability from Step 2
- Poor rim detection
- Large measurement errors

**Solutions:**
```bash
# Review Step 2 results
# Filter craters by rim probability > 0.7

# In Python:
good_rims = craters[craters['rim_probability'] > 0.7]
```

### Issue: Conditional probability CSV is sparse

**Cause:** Not enough craters for binning

**Solution:**
```python
# Reduce number of bins
# In compute_conditional_probabilities():
n_bins = min(5, len(valid) // 2)  # Fewer bins
```

### Issue: Memory errors with large datasets

**Cause:** Too many craters with `plot_individual=True`

**Solution:**
```bash
# Disable individual plotting
python analyze_crater_morphometry.py ... --output results/
# (Don't use --plot-individual)

# Or process in batches by diameter
python analyze_crater_morphometry.py ... --min-diameter 200  # Large
python analyze_crater_morphometry.py ... --min-diameter 100 --max-diameter 200  # Medium
# (Would need to add --max-diameter to code)
```

---

## Advanced Usage

### Custom Gaussian Fitting Strategy

Modify `fit_gaussian_floor()` for specific crater types:

```python
# For large craters with wide floors
sigma_guess = crater_radius_pixels / 2.0  # Larger sigma

# For small, sharp craters
sigma_guess = crater_radius_pixels / 4.0  # Smaller sigma

# For asymmetric craters
# Allow more rotation freedom (already implemented via theta parameter)
```

### Export to Different Formats

```python
import geopandas as gpd

# Load shapefile
gdf = gpd.read_file('craters_morphometry.shp')

# Export to GeoJSON
gdf.to_file('craters_morphometry.geojson', driver='GeoJSON')

# Export to KML (for Google Earth)
gdf.to_file('craters_morphometry.kml', driver='KML')

# Export geometry to WKT
with open('crater_geometries.wkt', 'w') as f:
    for idx, row in gdf.iterrows():
        f.write(f"{row['crater_id']}\t{row.geometry.wkt}\n")
```

### Batch Processing Multiple Regions

```bash
#!/bin/bash
# Process multiple crater datasets

for region in faustini_a faustini_b faustini_c; do
    echo "Processing ${region}..."

    python analyze_crater_morphometry.py \
        --shapefile regions/${region}/craters_refined.shp \
        --dtm regions/${region}/dtm.tif \
        --image regions/${region}/image.tif \
        --output results/${region}/
done

# Combine results
python -c "
import geopandas as gpd
import glob

files = glob.glob('results/*/craters_morphometry.shp')
gdfs = [gpd.read_file(f) for f in files]
combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
combined.to_file('results/all_regions_morphometry.shp')
"
```

---

## Performance

### Computational Cost

**Per Crater:**
- Method 1: ~2-5 seconds
- Method 2 (Gaussian fitting): ~1-3 seconds
- Total: ~3-8 seconds per crater

**For 100 Craters:**
- Expected time: 5-15 minutes
- Memory: ~1-2 GB (depends on DEM size)

### Optimization Tips

1. **Disable individual plots:**
   ```bash
   # Much faster without --plot-individual
   ```

2. **Filter small craters:**
   ```bash
   python analyze_crater_morphometry.py ... --min-diameter 100
   ```

3. **Parallel processing (future enhancement):**
   ```python
   # Could add multiprocessing for independent craters
   from multiprocessing import Pool
   with Pool(8) as p:
       results = p.map(process_crater, crater_list)
   ```

---

## See Also

- **Input Module:** [INPUT_MODULE_GUIDE.md](INPUT_MODULE_GUIDE.md)
- **Rim Refinement:** [REFINE_RIM_MODULE_GUIDE.md](REFINE_RIM_MODULE_GUIDE.md)
- **Crater Tools:** [STEP2_REFINE_RIMS_EXPLAINED.md](STEP2_REFINE_RIMS_EXPLAINED.md)
- **Main README:** [../README.md](../README.md)

---

## Summary

**Inputs:**
- Refined shapefile from Step 2
- DTM (elevation raster)
- Contrast image

**Dual-Method Algorithm:**
- Method 1: Rim perimeter analysis (proven)
- Method 2: 2D Gaussian floor fitting (robust, NEW)
- Combined estimate with uncertainty weighting

**Error Propagation:**
- Measurement uncertainties
- Rim probability from Step 2
- Gaussian fit uncertainties
- Total error for each crater

**Outputs:**
- Shapefile with 20+ morphometry fields
- 2 scatter plots with error bars
- 2 probability distribution plots
- CSV with all measurements
- CSV with conditional probabilities

**Key Innovation:**
Dual-method approach provides robust depth estimates with comprehensive uncertainty quantification, enabling quality-based filtering and statistical analysis of crater populations.

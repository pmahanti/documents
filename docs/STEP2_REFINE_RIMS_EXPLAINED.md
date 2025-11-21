# Step 2: Refine Crater Rims - Detailed Code Explanation

## Overview

**Purpose**: Use Digital Elevation Model (DEM) topography to find the actual crater rim positions and refine the circular geometries from Step 1.

**Why it's needed**: The circular geometries from Step 1 are just estimates based on diameter. Real craters are not perfect circles, and their actual rim positions vary due to degradation, overlapping, and other geological processes.

---

## The Code Architecture

Step 2 involves **three levels** of code:

1. **High-level orchestration**: `update_crater_rims()` - Loops through craters
2. **Core algorithm**: `fit_crater_rim()` - Finds rim for one crater
3. **Helper functions**: `compute_E_matrix()`, `remove_external_topography()` - Support functions

---

## Level 1: High-Level Orchestration

### Location: `src/crater_analysis/refinement.py`

```python
def update_crater_rims(input_shapefile, output_shapefile, dem_path,
                       orthophoto_path, min_diameter=60,
                       inner_radius=0.8, outer_radius=1.2,
                       plot=True, remove_external_topo=True):
    """
    Refine crater rim positions using topographic data.
    """
    with rio.open(dem_path, 'r') as src:
        # Initialize result containers
        updated_craters = []
        metadata = {
            'UFID': [], 'Diam_m': [], 'davg': [], 'old_dD': [],
            'rim_AJS': [], 'fl_AJS': [],
            'err_x0': [], 'err_y0': [], 'err_r': []
        }

        # Read input craters (circular geometries from Step 1)
        df_crater = gpd.read_file(input_shapefile)
        df_crater = df_crater[df_crater['D_m'] > min_diameter]

        # Process each crater
        for idx, geom in enumerate(df_crater['geometry']):
            # Fit crater rim using topography
            refined_geom, err = cratools.fit_crater_rim(
                geom=geom,
                dem_src=src,
                crs=crs,
                orthophoto=orthophoto_path,
                inner_radius=inner_radius,     # Search from 0.8R
                outer_radius=outer_radius,     # Search to 1.2R
                plot=plot,
                remove_ext=remove_external_topo
            )

            # Store results
            updated_craters.append(refined_geom)
            metadata['err_x0'].append(err[0])  # X position error
            metadata['err_y0'].append(err[1])  # Y position error
            metadata['err_r'].append(err[2])   # Radius error

        # Create output GeoDataFrame
        gdf_refined = gpd.GeoDataFrame(
            data=metadata,
            geometry=updated_craters,
            crs=crs
        )

        gdf_refined.to_file(output_shapefile)
        return gdf_refined
```

### What This Does:

1. **Opens DEM** - Loads the elevation data
2. **Reads input craters** - Gets circular geometries from Step 1
3. **Loops through each crater** - Processes one at a time
4. **Calls fit_crater_rim()** - Does the actual rim detection (see Level 2)
5. **Stores results** - Saves refined geometry + error estimates
6. **Saves output** - Writes refined shapefile

---

## Level 2: Core Rim Fitting Algorithm

### Location: `src/crater_analysis/cratools.py`

This is the **heart of Step 2**. It implements the rim detection algorithm.

### Function Signature:

```python
def fit_crater_rim(geom, dem_src, crs, orthophoto,
                   inner_radius=0.8, outer_radius=1.2,
                   remove_ext=True, plot=False):
```

### Algorithm Steps:

#### **Step 2.1: Extract Crater Parameters**

```python
# Compute radius and center from circular geometry
radius = np.sqrt(geom.area / np.pi)
center = geom.centroid
center = np.array([center.x, center.y])  # [x, y] in map coordinates
```

**What it does:**
- Calculates radius from circle area: `r = √(A/π)`
- Gets center point coordinates

**Example:**
```python
# For crater with area = 22,458 m²
radius = √(22458 / 3.14159) = 84.55 meters
center = [89601.29, 8829.40]
```

---

#### **Step 2.2: Remove Regional Topography (Optional)**

```python
if remove_ext:
    (out_image, out_transform), _ = remove_external_topography(
        geom=geom,
        dem_src=dem_src,
        orthophoto=orthophoto,
        crs=crs
    )
```

**What it does:**
- Removes regional slope/tilt from DEM
- Isolates crater-specific topography
- Fits a plane to far-field topography (at 3R distance)
- Subtracts this plane from the DEM

**Why it's important:**
- Craters often sit on sloped terrain
- Regional tilt can mask the rim signal
- Detrending makes rim peaks more obvious

**Visual:**
```
Before detrending:          After detrending:
  \  Crater on slope         Crater isolated
   \ /‾‾\                       /‾‾\
    ────────                  ────────
Regional slope removed       Flat reference
```

---

#### **Step 2.3: Define Search Parameters**

```python
# Define azimuthal range and radius range
azimuths = np.arange(0., 359., 5)            # Every 5 degrees
r_dist = np.arange(inner_radius, outer_radius, .01)  # 0.8R to 1.2R
```

**What it does:**
- Creates array of azimuth angles: [0°, 5°, 10°, ..., 355°]
- Creates array of radii: [0.8R, 0.81R, 0.82R, ..., 1.2R]

**Example:**
```python
# For crater with R = 84.55m
azimuths = [0, 5, 10, ..., 355]  # 72 angles
r_dist = [0.8, 0.81, 0.82, ..., 1.2] * 84.55  # Search 67.6m to 101.5m
```

**Visual:**
```
        N (0°)
        |
   NW   |   NE
     \  |  /
      \ | /
W ------○------ E (Center)
      / | \
     /  |  \
   SW   |   SE
        |
        S (180°)

At each angle, sample elevation from 0.8R to 1.2R
```

---

#### **Step 2.4: Create Elevation Matrix**

```python
# Compute matrix with rows = distance, cols = azimuth
E_rim = compute_E_matrix(
    image=out_image,
    transform=out_transform,
    center=center,
    azimuths=azimuths,
    r_dist=r_dist
)
```

**What it does:**
- Samples DEM elevation along radial profiles at each azimuth
- Creates 2D matrix: rows = radius, columns = azimuth
- Uses spline interpolation for sub-pixel accuracy
- Applies smoothing to reduce noise

**Matrix structure:**
```
E_rim matrix:
                Azimuth (degrees)
              0°    5°   10°  ...  355°
r_dist  0.8R  [ elevation values... ]
        0.9R  [ elevation values... ]
        1.0R  [ elevation values... ]
        1.1R  [ elevation values... ]
        1.2R  [ elevation values... ]
```

**Example values:**
```
At azimuth 0° (North direction):
  r=0.8R (67.6m): elevation = 1450.2m
  r=0.9R (76.1m): elevation = 1452.5m
  r=1.0R (84.5m): elevation = 1455.8m ← RIM PEAK
  r=1.1R (93.0m): elevation = 1453.2m
  r=1.2R (101.5m): elevation = 1451.0m
```

---

#### **Step 2.5: Detect Rim Peaks at Each Azimuth**

```python
rim_idx = np.full(E_rim.shape[1], circlefit_idx)  # Initialize

for idx, az in enumerate(azimuths):
    # Get elevation profile at this azimuth
    prof = E_rim[:, idx]

    # Find peaks using signal processing
    (local_max_idx) = scipy.signal.find_peaks(
        prof,
        prominence=1e-5 * radius
    )

    if local_max_idx[0].size == 0:
        # No clear peak - use second derivative minimum
        second_deriv = np.gradient(np.gradient(prof))
        rim_idx[idx] = np.argmin(second_deriv)
    else:
        # Use peak with highest prominence
        rim_idx[idx] = local_max_idx[0][np.argmax(local_max_idx[1]["prominences"])]
```

**What it does:**
- For each azimuth, analyzes the elevation profile
- Looks for local maximum (rim peak)
- If no clear peak, uses inflection point (second derivative minimum)

**Peak Detection Methods:**

**Method 1: Peak finding** (preferred)
```
Elevation profile at azimuth θ:

    E
    │     Peak!
    │      ╱╲
    │     ╱  ╲
    │    ╱    ╲___
    │___╱
    └────────────── r
        Rim position
```

**Method 2: Second derivative** (fallback)
```
If no obvious peak, find where curvature changes:

    E         Inflection
    │         point
    │        /
    │      /‾
    │    /
    │___/
    └────────────── r
        Rim estimate
```

**Example:**
```python
# At azimuth 0°
profile = [1450.2, 1452.5, 1455.8, 1453.2, 1451.0]  # Elevations
peaks = find_peaks(profile)  # Returns index 2
rim_radius[0°] = r_dist[2]  # = 1.0R (at the peak)

# At azimuth 90°
profile = [1449.8, 1451.2, 1454.5, 1452.8, 1450.5]
peaks = find_peaks(profile)  # Returns index 2
rim_radius[90°] = r_dist[2]  # = 1.0R
```

---

#### **Step 2.6: Convert to Cartesian Coordinates**

```python
# Convert polar (r, θ) to Cartesian (x, y)
rim_r_pos = r_dist[rim_idx]  # Rim radius at each azimuth

# Filter out invalid values
rim_r_pos[rim_r_pos < inner_radius] = np.nan
rim_r_pos[rim_r_pos > outer_radius] = np.nan

# Convert to x, y coordinates (relative to center)
rim_x = rim_r_pos * np.cos(azimuths * np.pi/180)
rim_y = rim_r_pos * np.sin(azimuths * np.pi/180)
```

**What it does:**
- Converts detected rim positions from polar to Cartesian
- Filters out outliers outside the search range

**Example:**
```python
# Rim detected at:
azimuth = 0°, radius = 1.0R = 84.55m

# Convert to Cartesian:
rim_x = 84.55 * cos(0°) = 84.55
rim_y = 84.55 * sin(0°) = 0.0

# Point relative to center: (84.55, 0.0)
```

**Visual:**
```
Detected rim points (relative to center):
        y
        │
   rim  │  rim
    ○   │   ○
        │
────○───●───○──── x
        │
    ○   │   ○
   rim  │  rim
        │
```

---

#### **Step 2.7: Fit Circle to Rim Points**

```python
# Define objective function for circle fitting
def obj_circle(params, x, y):
    x0, y0, rad = params
    return rad**2 - ((x - x0)**2 + (y - y0)**2)

# Fit circle using least squares
fit_result = scipy.optimize.least_squares(
    obj_circle,
    x0=[0, 0, 0.5],  # Initial guess: centered, R=0.5
    args=(rim_x[~np.isnan(rim_x)], rim_y[~np.isnan(rim_x)]),
    gtol=1e-10
)

# Extract fitted parameters
(x0_fit, y0_fit, rad_fit_fac) = fit_result.x

# Compute fit errors
J = fit_result.jac
cov = np.linalg.inv(J.T @ J)
err = np.sqrt(np.diag(cov))
```

**What it does:**
- Fits an optimal circle to the detected rim points
- Uses least-squares optimization
- Minimizes residuals: `r² - (x-x₀)² - (y-y₀)²`
- Computes uncertainties from Jacobian matrix

**Circle Equation:**
```
(x - x₀)² + (y - y₀)² = r²

Where:
  (x₀, y₀) = center offset from original center
  r = fitted radius
```

**Example:**
```python
# Detected rim points (scattered around circle)
rim_points = [
    (84.2, 0.5),   # 0°
    (59.5, 60.1),  # 45°
    (0.3, 85.0),   # 90°
    ...
]

# Fitted circle:
x0_fit = 0.2      # Center shifted 0.2m east
y0_fit = -0.1     # Center shifted 0.1m south
rad_fit_fac = 1.02  # Radius is 2% larger than initial

# Fitted radius:
radius_fit = 84.55 * 1.02 = 86.24 meters

# Fitted center:
center_fit = [89601.29 + 0.2*84.55, 8829.40 + (-0.1)*84.55]
           = [89618.20, 8820.95]

# Errors:
err_x0 = ±0.5 meters
err_y0 = ±0.4 meters
err_r = ±0.8 meters
```

---

#### **Step 2.8: Create Refined Geometry**

```python
# Update original parameters using fit
radius_fit = radius * rad_fit_fac
center_fit = [
    center[0] + x0_fit * radius,
    center[1] + y0_fit * radius
]

# Create new circular polygon at refined position
new_crater = shapely.geometry.Point(
    center_fit[0],
    center_fit[1]
).buffer(radius_fit)

return new_crater, err
```

**What it does:**
- Creates new circular geometry at refined center
- Uses fitted radius
- Returns refined geometry + error estimates

---

## Level 3: Helper Function - compute_E_matrix()

### Location: `src/crater_analysis/cratools.py` (lines 294-339)

This function creates the elevation matrix used for rim detection.

```python
def compute_E_matrix(image, transform, center,
                     r_dist=np.arange(0.01, 3.01, .01),
                     azimuths=np.arange(0., 359., 5)):

    # Convert metric coordinates to pixel coordinates
    (row_center, col_center) = ~transform * center
    (_, row, col) = image.shape

    # Create coordinate arrays
    row_s = np.linspace(-row_center/row*6, (row-row_center)/row*6, row)
    col_s = np.linspace(-col_center/col*6, (col-col_center)/col*6, col)

    # Create 2D interpolator (bicubic spline, degree 5)
    interpolator_spline = scipy.interpolate.RectBivariateSpline(
        row_s, col_s, image[0, :, :], kx=5, ky=5
    )

    # Create meshgrid of (azimuth, radius) pairs
    azaz, rr = np.meshgrid(azimuths, r_dist)
    azaz *= np.pi/180  # Convert to radians

    # Convert polar to Cartesian
    xs = rr * np.cos(azaz)
    ys = rr * np.sin(azaz)

    # Sample elevations using interpolator
    E_mat = interpolator_spline.ev(-ys, xs)

    # Apply 1D smoothing spline to each azimuthal profile
    E_sm = np.array(E_mat, copy=True)
    k = 3    # Spline degree
    s = 2.0  # Smoothing parameter

    for j in range(E_mat.shape[1]):  # For each azimuth
        y = E_mat[:, j]
        try:
            spline1d = scipy.interpolate.UnivariateSpline(r_dist, y, s=s, k=k)
            E_sm[:, j] = spline1d(r_dist)
        except Exception:
            E_sm[:, j] = y  # Keep original if smoothing fails

    return E_sm
```

### What This Does:

1. **Coordinate Transformation**
   - Converts crater center from map coordinates to pixel coordinates
   - Uses the DEM's geotransform

2. **Spline Interpolation**
   - Creates bicubic spline interpolator for the DEM
   - Allows sampling at sub-pixel locations
   - Higher accuracy than nearest-neighbor sampling

3. **Radial Sampling**
   - Creates grid of (radius, azimuth) points in polar coordinates
   - Converts to Cartesian (x, y)
   - Samples elevation at each point using interpolator

4. **Smoothing**
   - Applies 1D smoothing spline to each radial profile
   - Reduces noise while preserving rim signal
   - Uses degree-3 spline with smoothing parameter s=2.0

### Matrix Output:

```
E_sm (smoothed elevation matrix):
               Azimuth
            0°    5°   10°  ...  355°
Radius  0.01R [ 1445.2, 1445.3, 1445.1, ..., 1445.2 ]
        0.02R [ 1445.5, 1445.6, 1445.4, ..., 1445.5 ]
        ...
        1.00R [ 1455.8, 1455.9, 1455.7, ..., 1455.8 ] ← Rim
        ...
        3.00R [ 1450.1, 1450.2, 1450.0, ..., 1450.1 ]
```

---

## Complete Workflow Example

### Input:
- Circular crater geometry: center = (89601.29, 8829.40), radius = 84.55m
- DEM with 1m resolution
- Search range: 0.8R to 1.2R (67.6m to 101.5m)

### Process:

1. **Sample elevations** at 72 azimuths × 41 radii = 2,952 points
2. **Find rim** at each azimuth using peak detection
3. **Fit circle** to 72 detected rim points
4. **Result:**
   - New center: (89618.20, 8820.95) - shifted 18.9m east, 8.5m south
   - New radius: 86.24m - 2% larger
   - Errors: ±0.5m (x), ±0.4m (y), ±0.8m (r)

### Output:
- Refined circular geometry at new position
- Error estimates for position and size

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inner_radius` | 0.8 | Start search at 80% of initial radius |
| `outer_radius` | 1.2 | End search at 120% of initial radius |
| `azimuth_step` | 5° | Sample every 5 degrees (72 profiles) |
| `radius_step` | 0.01R | Sample every 1% of radius |
| `remove_ext` | True | Remove regional topography |
| `plot` | False | Generate diagnostic plots |

---

## Summary

**Step 2 Algorithm:**

1. For each crater from Step 1
2. Extract DEM around crater (3R × 3R box)
3. Remove regional topography (optional)
4. Sample elevation along 72 radial profiles
5. Detect rim peak at each profile
6. Fit optimal circle to detected rim points
7. Calculate uncertainties
8. Return refined geometry

**Key Innovation:**
Instead of assuming craters are perfect circles, this algorithm **detects the actual rim** using topographic signatures, then fits the best circle to those detected points.

**Result:**
More accurate crater positions and sizes for morphometry analysis in Step 3.

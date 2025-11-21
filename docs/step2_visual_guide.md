# Step 2: Visual Guide to Rim Refinement

## The Big Picture

```
INPUT (from Step 1):          OUTPUT (after Step 2):
Approximate Circle            Refined Circle

      ╭─────╮                      ╭─────╮
      │  ·  │                      │  ·  │
      │     │  R=84.5m            │     │  R=86.2m
      ╰─────╯                      ╰─────╯
  Center: (89601, 8829)        Center: (89618, 8821)

  Based on reported            Based on actual
  diameter only                topography
```

---

## The Algorithm in 8 Steps

### Step 1: Extract DEM around Crater

```
DEM Extraction:

    ╔══════════════════════════════╗
    ║         DEM Image            ║
    ║                              ║
    ║        ┌─────────┐           ║
    ║        │ Crater  │  3R × 3R  ║
    ║        │  Area   │  box      ║
    ║        └─────────┘           ║
    ║                              ║
    ╚══════════════════════════════╝
```

### Step 2: Remove Regional Slope (Optional)

```
Before:                    After:
Crater on slope            Crater isolated

  \
   \  /‾‾\                     /‾‾\
    \/    \        →         /    \
     ──────                ──────────
  Tilted plane           Flat reference
```

### Step 3: Define Search Range

```
Search annulus (0.8R to 1.2R):

         Outer limit (1.2R)
              ╱────╲
            ╱        ╲
          ╱   ╭────╮  ╲
         │   ╱      ╲  │
         │  │   ·   │  │  ← Center
         │   ╲      ╱  │
          ╲   ╰────╯  ╱
            ╲        ╱
              ╲────╱
         Inner limit (0.8R)

    Search for rim in this ring
```

### Step 4: Sample at Multiple Azimuths

```
Azimuthal sampling (every 5°):

           N (0°)
           │
     NW    │    NE
   (315°)  │  (45°)
       ╲   │   ╱
        ╲  │  ╱
    W ───●───── E
   (270°) │ (90°)
        ╱  │  ╲
       ╱   │   ╲
     SW    │    SE
   (225°)  │  (135°)
           │
           S (180°)

72 radial lines (0°, 5°, 10°, ..., 355°)
```

### Step 5: Sample Along Each Radial Line

```
Elevation profile at azimuth θ:

    Elevation
    │
    │         Rim peak
    │          ╱╲
    │         ╱  ╲
    │        ╱    ╲
    │       ╱      ╲___
    │______╱
    └────────────────────────
         0.8R  1.0R  1.2R
              Radius →

Sample 41 points from 0.8R to 1.2R
```

### Step 6: Create Elevation Matrix

```
E_rim matrix (elevation values):

           Azimuth (columns)
         0°   5°   10°  15° ... 355°
      ┌─────────────────────────────┐
0.80R │ 1450 1450 1451 1449 ... 1450│
0.85R │ 1451 1451 1452 1450 ... 1451│
0.90R │ 1453 1453 1454 1452 ... 1453│
0.95R │ 1455 1455 1456 1454 ... 1455│
1.00R │ 1456 1456 1457 1455 ... 1456│ ← Rim
1.05R │ 1454 1454 1455 1453 ... 1454│
1.10R │ 1452 1452 1453 1451 ... 1452│
1.15R │ 1451 1451 1452 1450 ... 1451│
1.20R │ 1450 1450 1451 1449 ... 1450│
      └─────────────────────────────┘

Each column = one radial profile
Each row = constant radius
```

### Step 7: Detect Rim at Each Azimuth

```
For each azimuth, find the peak:

Profile at 0°:        Profile at 45°:       Profile at 90°:
    E                     E                     E
    │   Peak              │   Peak              │    Peak
    │    ▲                │     ▲               │     ▲
    │   ╱╲                │    ╱╲               │    ╱╲
    │  ╱  ╲               │   ╱  ╲              │   ╱  ╲
    └──────────           └───────────          └──────────
        1.0R                  0.98R                 1.02R

Rim detected at:      Rim detected at:      Rim detected at:
r = 1.0R             r = 0.98R             r = 1.02R
```

### Step 8: Fit Circle to Detected Points

```
Detected rim points (×) and fitted circle (○):

           N
           ×
      ×    │    ×
   ×       │       ×
  ×        │        ×
───×───────●───────×─── E
  ×        │        ×
   ×       │       ×
      ×    │    ×
           ×
           S

Least-squares circle fit:
  Center: (x₀, y₀) = offset from original
  Radius: r_fit = adjusted radius
```

---

## Detailed: Elevation Sampling

### How We Sample the DEM

```
DEM pixels (discrete):         Interpolated values (continuous):

┌───┬───┬───┬───┐              Smooth surface
│1450│1451│1452│1453│              ╱╲
├───┼───┼───┼───┤               ╱    ╲
│1449│1450│1451│1452│            ╱        ╲
├───┼───┼───┼───┤             ╱            ╲
│1448│1449│1450│1451│        ╱                ╲
└───┴───┴───┴───┘

Bicubic spline interpolation
allows sampling at any (x, y)
```

### Radial Profile Creation

```
Polar coordinates (r, θ) → Cartesian (x, y):

For azimuth θ = 45°, sample at radii [0.8R, 0.81R, ..., 1.2R]:

Radius  Angle  →  x_offset  y_offset  →  x_map      y_map
0.80R   45°       56.57     56.57        89657.86   8886.97
0.85R   45°       60.10     60.10        89661.39   8890.50
0.90R   45°       63.64     63.64        89664.93   8894.04
...
1.00R   45°       70.71     70.71        89672.00   8901.11
...
1.20R   45°       84.85     84.85        89686.14   8915.25

Then sample DEM elevation at each (x_map, y_map)
```

---

## Peak Detection Methods

### Method 1: Find Peaks (Preferred)

```python
scipy.signal.find_peaks(profile, prominence=threshold)
```

```
Elevation profile with clear rim:

    E
    │                Peak detected!
    │                 ▲
    │                ╱╲
    │     Prominence╱  ╲
    │      ↕↕↕↕↕↕↕╱    ╲___
    │____________╱
    └──────────────────────────
                 Rim position

Prominence = height difference from peak to surrounding valleys
```

### Method 2: Second Derivative (Fallback)

```python
second_deriv = np.gradient(np.gradient(profile))
rim_idx = np.argmin(second_deriv)
```

```
Elevation profile with degraded rim:

    E          Second derivative:
    │              │
    │      ╱‾      │   ╱╲
    │    ╱         │  ╱  ╲
    │  ╱           │_╱    ╲___
    │_╱                    ▼ Min
    └─────────             └─────────

Inflection point = where curvature changes most
(minimum of second derivative)
```

---

## Circle Fitting

### Objective Function

Minimize residuals between detected points and circle:

```
Circle equation: (x - x₀)² + (y - y₀)² = r²

Residual for point i:
    residual_i = r² - (x_i - x₀)² - (y_i - y₀)²

Least squares minimizes:
    Σ(residual_i)²
```

### Visual Representation

```
Detected rim points (not quite circular):

           ×
      ×         ×
   ×               ×
  ×                 ×
───×─────●─────×────── Fitted circle (solid line)
  ×      |      ×      tries to minimize total
   ×     |     ×       distance to all points
      ×  |  ×
         ×

● = Fitted center (may be offset from original)
```

---

## Error Estimation

### Uncertainty Sources

```
Error propagation:

DEM Resolution    Interpolation    Peak Detection
    ±0.5m      +     ±0.3m      +      ±0.8m
                         ↓
              Total Position Error: ±1.0m
                         ↓
              Fitted Uncertainties:
                err_x0 = ±0.5m
                err_y0 = ±0.4m
                err_r  = ±0.8m
```

### Jacobian Matrix

```
Fit quality from Jacobian J:

        ∂residual/∂x₀  ∂residual/∂y₀  ∂residual/∂r
J =   [ ...              ...            ...        ]
      [ ...              ...            ...        ]

Covariance: Cov = (J'J)⁻¹
Errors: σ = √diag(Cov)
```

---

## Parameter Effects

### Inner/Outer Radius Effect

```
Narrow search (0.9R - 1.1R):    Wide search (0.7R - 1.3R):
      More precise                  More robust
      ╭──╮                          ╭────────╮
      │  │                          │        │
      │ ● │                         │   ●    │
      │  │                          │        │
      ╰──╯                          ╰────────╯
   May miss rim               May include noise
```

### Azimuth Sampling Effect

```
Coarse (every 15°):         Fine (every 5°):
      24 points                   72 points
       ×   ×                     × × × ×
     ×       ×                 ×         ×
    ×    ●    ×               ×     ●     ×
     ×       ×                 ×         ×
       ×   ×                     × × × ×
   Less accurate             More accurate
```

---

## Real-World Example

### Crater p003c2322

**Initial (Step 1):**
- Center: (89601.29, 8829.40)
- Radius: 84.55m (from diameter 169.1m)
- Shape: Perfect circle

**Rim Detection:**
- 72 azimuthal profiles analyzed
- Rim found at radii ranging from 0.96R to 1.04R
- Average rim elevation: 1455.8m
- Interior elevation: ~1447.2m (8.6m deep)

**Refined (Step 2):**
- Center: (89618.20, 8820.95) — shifted 18.9m
- Radius: 86.24m — 2% larger
- Errors: ±0.5m (x), ±0.4m (y), ±0.8m (r)
- Shape: Still circular, but better positioned

---

## Summary

**What Step 2 Does:**

1. ✓ Extracts topography around each crater
2. ✓ Removes regional slope
3. ✓ Samples elevation along 72 radial profiles
4. ✓ Detects rim peak at each azimuth
5. ✓ Fits optimal circle to detected rim points
6. ✓ Estimates position and size uncertainties

**Result:**
Refined crater geometries with actual rim positions for accurate morphometry in Step 3.

**Key Innovation:**
Uses **actual topography** instead of just reported diameters to find crater rims.

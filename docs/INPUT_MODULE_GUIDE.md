# Input Module Guide

## Overview

The `input_module.py` provides comprehensive input processing for crater analysis, handling multiple data formats and coordinate systems.

## Features

✓ **Multiple Input Formats**
- GeoTIFF images (.tif, .tiff)
- ISIS cube files (.cub) - requires GDAL ISIS driver
- CSV crater files
- .diam crater files

✓ **Flexible Coordinate Systems**
- Latitude/Longitude
- Projected X/Y coordinates
- Automatic detection from headers

✓ **Lunar Projections**
- Equirectangular (Simple Cylindrical)
- Stereographic (Polar)
- Orthographic
- Automatic coordinate conversion

✓ **Outputs**
- ESRI Shapefile with crater geometries
- PNG: Crater locations on image
- PNG: Crater Size-Frequency Distribution (CSFD)

---

## Required Inputs

### 1. Contrast Image (Mandatory)

**Purpose:** Visual context for crater identification

**Formats:**
- GeoTIFF: `.tif`, `.tiff`
- ISIS Cube: `.cub` (requires GDAL with ISIS driver)

**Requirements:**
- Same spatial extent as DTM
- Same pixel scale as DTM
- Georeferenced (has coordinate system)

**Example:**
```bash
image.tif
  Size: 2048 x 2048 pixels
  Pixel size: 20 m/pixel
  CRS: IAU2000:30100 (Moon equirectangular)
```

### 2. DTM/DEM (Mandatory)

**Purpose:** Elevation data for crater morphometry

**Format:** GeoTIFF (`.tif`, `.tiff`)

**Requirements:**
- Raster with elevation values
- Same size and pixel scale as contrast image
- Same coordinate system as contrast image

**Example:**
```bash
dtm.tif
  Size: 2048 x 2048 pixels
  Pixel size: 20 m/pixel
  Values: Elevation in meters
  CRS: IAU2000:30100
```

### 3. Crater Location File (Mandatory)

**Purpose:** Initial crater positions and sizes

**Formats:**
- CSV (`.csv`)
- .diam files

**Requirements:**
- **Column 1:** Latitude OR X-coordinate
- **Column 2:** Longitude OR Y-coordinate
- **Column 3:** Diameter
- **Headers:** Must identify coordinate type

**Supported Headers:**

**Lat/Lon:**
```csv
lat, lon, diameter
-87.031, 84.372, 169.1
```

**X/Y:**
```csv
x_m, y_m, diam_m
89601.29, 8829.40, 169.1
```

**Alternative Headers:**
```
Latitude/longitude: lat, latitude, lat_deg, lat_d, lon, long, longitude, lon_deg, lon_d
X/Y: x, x_m, x_km, easting, y, y_m, y_km, northing
Diameter: d, diam, diameter, d_m, d_km, diam_m, diam_km
```

---

## Coordinate Systems

### Lunar Projections

The module supports three standard lunar projections:

#### 1. Equirectangular (Simple Cylindrical)

**When to use:** Global or low-latitude regions

**Formula:**
```
x = R × (lon - lon₀) × cos(lat₀)
y = R × (lat - lat₀)

Where:
  R = 1737400 m (Moon radius)
  lon₀, lat₀ = projection center
```

**Example:**
```bash
python process_crater_inputs.py \
    --projection equirectangular \
    --center-lon 0 --center-lat 0
```

#### 2. Stereographic

**When to use:** Polar regions (lat > 60° or lat < -60°)

**Formula:**
```
k = 2R / (1 + sin(lat₀)×sin(lat) + cos(lat₀)×cos(lat)×cos(lon-lon₀))
x = k × cos(lat) × sin(lon - lon₀)
y = k × (cos(lat₀)×sin(lat) - sin(lat₀)×cos(lat)×cos(lon-lon₀))
```

**Example (South Pole):**
```bash
python process_crater_inputs.py \
    --projection stereographic \
    --center-lon 0 --center-lat -90
```

#### 3. Orthographic

**When to use:** Hemisphere views, visible disk

**Formula:**
```
x = R × cos(lat) × sin(lon - lon₀)
y = R × (cos(lat₀)×sin(lat) - sin(lat₀)×cos(lat)×cos(lon-lon₀))
```

**Example:**
```bash
python process_crater_inputs.py \
    --projection orthographic \
    --center-lon 0 --center-lat 0
```

---

## Usage

### Basic Usage

```bash
python process_crater_inputs.py \
    --image path/to/image.tif \
    --dtm path/to/dtm.tif \
    --craters path/to/craters.csv \
    --output results/
```

### With Specific Projection

```bash
python process_crater_inputs.py \
    --image data/south_pole_image.tif \
    --dtm data/south_pole_dtm.tif \
    --craters data/craters_latlon.csv \
    --output results/ \
    --projection stereographic \
    --center-lon 0 \
    --center-lat -90
```

### From Python

```python
from crater_analysis.input_module import process_crater_inputs

results = process_crater_inputs(
    image_path='data/image.tif',
    dtm_path='data/dtm.tif',
    crater_file_path='data/craters.csv',
    output_dir='results/',
    projection='equirectangular',
    center_lon=0.0,
    center_lat=0.0
)

print(f"Created shapefile: {results['shapefile']}")
print(f"Total craters: {results['crater_count']}")
```

---

## Input File Examples

### Example 1: Lat/Lon CSV

**File:** `craters_latlon.csv`

```csv
lat,lon,diameter
-87.031491,84.372185,169.1
-87.025907,84.669165,307.7
-87.018277,84.798228,97.5
```

**Usage:**
```bash
python process_crater_inputs.py \
    --craters craters_latlon.csv \
    ...
```

**Output:**
```
Detected lat/lon coordinates: lat, lon
Loaded 3 craters
Coordinate type: latlon
Converting lat/lon to X/Y...
```

### Example 2: X/Y CSV

**File:** `craters_xy.csv`

```csv
x_m,y_m,diam_m
89601.29,8829.40,169.1
89814.57,8380.60,307.7
90063.78,8199.25,97.5
```

**Usage:**
```bash
python process_crater_inputs.py \
    --craters craters_xy.csv \
    ...
```

**Output:**
```
Detected X/Y coordinates: x_m, y_m
Loaded 3 craters
Coordinate type: xy
Converting X/Y to lat/lon...
```

### Example 3: Diameter in Kilometers

**File:** `craters_km.csv`

```csv
lat,lon,diam_km
-87.031,84.372,0.169
-87.026,84.669,0.308
```

**Automatic conversion:**
```
Converting diameter from km to meters
Diameter range: 169.0 - 308.0 m
```

### Example 4: .diam Format

**File:** `craters.diam`

```
# Crater diameter file
latitude  longitude  diameter_m
-87.031   84.372     169.1
-87.026   84.669     307.7
```

**Usage:**
```bash
python process_crater_inputs.py \
    --craters craters.diam \
    ...
```

---

## Outputs

### 1. Shapefile: `craters_initial.shp`

**Content:**
- Crater geometries as circles (polygons)
- Attributes: lat, lon, x, y, diameter

**Files created:**
```
craters_initial.shp     # Geometry
craters_initial.shx     # Spatial index
craters_initial.dbf     # Attributes
craters_initial.prj     # Projection
craters_initial.cpg     # Character encoding
```

**Load in QGIS/ArcGIS:**
```
Open → Add Vector Layer → craters_initial.shp
```

**Load in Python:**
```python
import geopandas as gpd
craters = gpd.read_file('craters_initial.shp')
print(craters.head())
```

### 2. Location Plot: `craters_initial_locations.png`

**Content:**
- Contrast image as background (grayscale)
- Crater circles overlaid in red
- Title: "Initial location of craters, N = XX"
- Axes labeled with coordinates

**Example:**
```
┌───────────────────────────────────┐
│ Initial location of craters, N=19│
│                                   │
│         ○  ○                      │
│            Image with             │
│       ○         ○                 │
│         red crater                │
│   ○         ○       ○             │
│           circles                 │
│     ○            ○                │
│                                   │
└───────────────────────────────────┘
```

### 3. CSFD Plot: `craters_csfd.png`

**Content:**
- Log-log plot of crater size-frequency distribution
- X-axis: Diameter (km)
- Y-axis: Cumulative frequency (craters/km²) or count
- Data points connected by lines
- Grid for reference

**Example:**
```
Cumulative Frequency
    │
10² │     ●
    │      ●
10¹ │       ●
    │        ●
10⁰ │         ●●●
    │            ●●●●
10⁻¹└─────────────────────
      0.1    1     10
       Diameter (km)
```

---

## Coordinate Conversion Details

### Lat/Lon → X/Y

When crater file has lat/lon coordinates:

```python
# For each crater:
x, y = converter.latlon_to_xy(lat, lon)

# Example:
lat, lon = -87.031, 84.372
x, y = 89601.29, 8829.40  # meters
```

### X/Y → Lat/Lon

When crater file has X/Y coordinates:

```python
# For each crater:
lat, lon = converter.xy_to_latlon(x, y)

# Example:
x, y = 89601.29, 8829.40  # meters
lat, lon = -87.031, 84.372  # degrees
```

### Round-Trip Accuracy

```python
# Original
lat1, lon1 = -87.031, 84.372

# Convert to XY and back
x, y = latlon_to_xy(lat1, lon1)
lat2, lon2 = xy_to_latlon(x, y)

# Difference (should be < 0.001°)
error = abs(lat2 - lat1), abs(lon2 - lon1)
# Typical: (1e-6, 1e-6)
```

---

## Error Handling

### Missing Headers

If headers are unclear:

```
Warning: Coordinate type not clearly identified
Headers: ['col1', 'col2', 'col3']
Assuming lat/lon based on value ranges
```

**Solution:** Add clear headers to CSV file

### Wrong Coordinate Type

If lat/lon values are outside valid range:

```
Warning: Latitude values > 90 detected
Possible coordinate type mismatch
```

**Solution:** Check if X/Y was labeled as lat/lon

### Unit Mismatch

If diameters are in wrong units:

```
Diameter range: 0.1 - 0.4 m  # Too small!
```

**Solution:** Check if diameters are in km (should be meters)

### ISIS Cube Not Supported

```
Error reading ISIS cube: ...
Note: ISIS cube support requires GDAL with ISIS driver
Consider converting to GeoTIFF using gdal_translate
```

**Solution:**
```bash
gdal_translate input.cub output.tif
```

---

## Integration with Main Workflow

### Step 0: Input Processing (NEW)

```bash
python process_crater_inputs.py \
    --image data/image.tif \
    --dtm data/dtm.tif \
    --craters data/craters.csv \
    --output results/
```

**Outputs:**
- `results/craters_initial.shp`
- `results/craters_initial_locations.png`
- `results/craters_csfd.png`

### Step 1: Prepare Geometries

**Input:** `craters_initial.shp` (from Step 0)

```bash
python main.py --steps prepare \
    --input results/craters_initial.shp
```

### Step 2: Refine Rims

```bash
python main.py --steps refine
```

### Step 3: Analyze Morphometry

```bash
python main.py --steps analyze
```

---

## Advanced Usage

### Custom Coordinate System

If your data uses a non-standard projection:

```python
from crater_analysis.input_module import read_crater_file, create_crater_shapefile
import geopandas as gpd

# Read crater file
crater_df = read_crater_file('craters.csv')

# Custom CRS
custom_crs = 'EPSG:104903'  # Moon 2000 South Pole Stereographic

# Create shapefile with custom CRS
gdf = create_crater_shapefile(
    crater_df,
    'craters.shp',
    crs=custom_crs
)
```

### Batch Processing

Process multiple regions:

```python
regions = ['faustini', 'sverdrup', 'shackleton']

for region in regions:
    process_crater_inputs(
        image_path=f'data/{region}_image.tif',
        dtm_path=f'data/{region}_dtm.tif',
        crater_file_path=f'data/{region}_craters.csv',
        output_dir=f'results/{region}/',
        projection='stereographic',
        center_lat=-90
    )
```

### Extract Data for Analysis

```python
results = process_crater_inputs(...)

# Access crater data
crater_df = results['crater_data']

# Statistics
print(f"Total craters: {results['crater_count']}")
print(f"Mean diameter: {crater_df['diameter'].mean():.1f} m")
print(f"Diameter range: {crater_df['diameter'].min():.1f} - "
      f"{crater_df['diameter'].max():.1f} m")

# Export to additional formats
crater_df.to_csv('craters_table.csv', index=False)
crater_df.to_excel('craters_table.xlsx', index=False)
```

---

## Troubleshooting

### Problem: "No module named 'crater_analysis'"

**Cause:** Python can't find the src directory

**Solution:**
```bash
# Run from project root
cd /path/to/documents/
python process_crater_inputs.py ...
```

### Problem: Craters not visible on image

**Cause:** Coordinate mismatch between craters and image

**Solution:**
1. Check image CRS: `gdalinfo image.tif | grep "Coordinate System"`
2. Check crater coordinates are in same system
3. Try different projection

### Problem: CSFD plot is empty

**Cause:** Area calculation failed

**Solution:**
```python
# Manually specify area
from crater_analysis.input_module import plot_csfd
plot_csfd(crater_df, 'csfd.png', area_km2=1000.0)
```

### Problem: Wrong units detected

**Cause:** Headers don't clearly indicate units

**Solution:** Use explicit headers:
```csv
lat,lon,diam_m     # Good
x_m,y_m,diam_km    # Good
col1,col2,col3     # Bad - unclear
```

---

## API Reference

### `process_crater_inputs()`

Main processing function.

**Parameters:**
- `image_path` (str): Path to contrast image
- `dtm_path` (str): Path to DTM file
- `crater_file_path` (str): Path to crater file
- `output_dir` (str): Output directory
- `projection` (str): Projection type
- `center_lon` (float): Center longitude
- `center_lat` (float): Center latitude

**Returns:**
- `dict`: Output paths and data

### `read_crater_file()`

Read crater location file.

**Parameters:**
- `filepath` (str): Path to file
- `delimiter` (str): Column delimiter

**Returns:**
- `DataFrame`: Crater data

### `CoordinateConverter`

Handle coordinate conversions.

**Methods:**
- `latlon_to_xy(lat, lon)`: Convert to projected coords
- `xy_to_latlon(x, y)`: Convert to geographic coords

---

## See Also

- [Main README](../README.md) - Project overview
- [Step 1 Guide](STEP1_PREPARE_GEOMETRIES_EXPLAINED.md) - Geometry preparation
- [Step 2 Guide](STEP2_REFINE_RIMS_EXPLAINED.md) - Rim refinement

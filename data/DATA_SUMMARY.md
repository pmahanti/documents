# Notebook Data Summary

## Data Files Used

The Jupyter notebook (`compute_depth_diameter.ipynb`) uses the following data files:

### Input Data Files

#### 1. Region Configurations

| Region | Shapefile Name | DEM File | Orthophoto File |
|--------|---------------|----------|-----------------|
| **faustini** | faustini | SHADOWCAM_FAUSTINI_P871S0839.tif | SHADOWCAM_FAUSTINI_M017670865SE_170cm.tif |
| **sverdrup** | sverdrup | SHADOWCAM_SVERDRUP.tif | SHADOWCAM_SVERDRUP_M016815246SE_210cm.tif |
| **test** (default) | tmp.test_craters | SHADOWCAM_FAUSTINI_P871S0839.tif | SHADOWCAM_FAUSTINI_M017670865SE_170cm.tif |
| **site1** | yolo_site1 | LDEM_80S_20MPP_ADJ.TIF | NAC_POLE_SOUTH_CM_AVG_P848S0337.TIF |

#### 2. File Paths (Original)

```
Images Directory:     /Users/asonke/Library/CloudStorage/OneDrive-ArizonaStateUniversity/SLC/images/
Shapefiles Directory: /Users/asonke/Library/CloudStorage/OneDrive-ArizonaStateUniversity/SLC/shapefiles/
Output Directory:     /Users/asonke/Library/CloudStorage/OneDrive-ArizonaStateUniversity/SLC/figs/
```

#### 3. Parameters

- **Default Region**: test
- **Minimum Diameter**: 60 meters

---

## Crater Dataset Structure

### Input Shapefile Fields

The notebook processes shapefiles with the following fields:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| **FeatureID** | Integer | Feature identifier | 2322 |
| **UFID** | String | Unique crater ID | p003c2322 |
| **lat** | Float | Latitude (degrees) | -87.031491 |
| **lon** | Float | Longitude (degrees) | 84.372185 |
| **x_km** | Float | X coordinate (km) | 89.601294 |
| **y_km** | Float | Y coordinate (km) | 8.829402 |
| **Diam_km** | Float | Diameter (km) | 0.169115 |
| **D_m** | Float | **Diameter (meters)** | 169.114867 |
| **x_m** | Float | X coordinate (meters) | 89601.29355 |
| **y_m** | Float | Y coordinate (meters) | 8829.402469 |
| **zp_slope** | Float | Slope measurement | 0.312395 |
| **davg** | Float | Average depth | 8.643539 |
| **davg_D** | Float | Depth/diameter ratio | 0.051110 |
| **dh_D** | Float | Horizontal depth ratio | 0.047490 |
| **dv_D** | Float | Vertical depth ratio | 0.039034 |
| **rim** | Float | Rim elevation | (varies) |
| **fl** | Float | Floor elevation | (varies) |
| **rim_sd** | Float | Rim standard deviation | 1.388385 |
| **scaled_sd** | Float | Scaled standard deviation | 0.160627 |
| **geometry** | Geometry | Spatial geometry | Point/Polygon |

---

## Sample Data

### Crater Examples from "test" Region

Here are 19 craters from the test dataset (tmp.test_craters):

| UFID | Lat | Lon | Diameter (m) | Avg Depth (m) | d/D Ratio |
|------|-----|-----|--------------|---------------|-----------|
| p003c2322 | -87.031 | 84.372 | 169.1 | 8.64 | 0.0511 |
| p003c2323 | -87.026 | 84.669 | 307.7 | 13.14 | 0.0427 |
| p003c2324 | -87.018 | 84.798 | 97.5 | 3.12 | 0.0320 |
| p003c2325 | -87.018 | 84.881 | 85.7 | 3.57 | 0.0417 |
| p003c2326 | -87.020 | 85.012 | 366.8 | 21.50 | 0.0586 |
| p003c2327 | -87.011 | 85.142 | 178.6 | 7.44 | 0.0417 |
| p003c2328 | -87.028 | 85.048 | 110.2 | 4.28 | 0.0388 |
| p003c2344 | -87.021 | 84.163 | 295.9 | 17.20 | 0.0508 |
| p003c2349 | -87.015 | 84.383 | 273.9 | 7.34 | 0.0147 |
| p003c2446 | -87.017 | 83.517 | 240.9 | 5.61 | 0.0149 |
| p003c2447 | -87.031 | 83.951 | 60.1 | 2.46 | 0.0293 |
| p003c2448 | -87.014 | 83.835 | 225.0 | 11.03 | 0.0475 |
| p003c2449 | -87.027 | 83.953 | 81.9 | 2.37 | 0.0252 |
| p003c2450 | -87.024 | 83.678 | 82.5 | 1.56 | 0.0117 |
| p003c2453 | -87.020 | 83.865 | 60.4 | 1.20 | 0.0184 |
| p003c2468 | -87.031 | 84.253 | 75.5 | 2.83 | 0.0229 |
| p003c2483 | -87.008 | 83.646 | 84.2 | 2.57 | 0.0180 |
| p003c2486 | -87.012 | 83.677 | 79.3 | 1.91 | 0.0219 |
| p003c2487 | -87.027 | 84.341 | 131.9 | 2.90 | 0.0208 |

**Dataset Characteristics:**
- Total craters in sample: 19
- Diameter range: 60.1 - 366.8 meters
- Depth range: 1.20 - 21.50 meters
- d/D ratio range: 0.0117 - 0.0586
- Location: Near South Pole (lat ~ -87°)

---

## Workflow Data Flow

### Step 1: Prepare Geometries
**Input**: `tmp.test_craters.shp`
- Original crater point locations with diameter information
- Filters craters with D_m > 60 meters

**Output**: `tmp.test_craters.geom.shp`
- Point geometries buffered to circles
- Circle radius = D_m / 2

### Step 2: Refine Crater Rims
**Input**:
- `tmp.test_craters.geom.shp` (buffered geometries)
- `SHADOWCAM_FAUSTINI_P871S0839.tif` (DEM)
- `SHADOWCAM_FAUSTINI_M017670865SE_170cm.tif` (Orthophoto)

**Output**: `tmp.test_craters.refined.shp`
- Refined crater geometries with topography-based rim positions
- Additional fields:
  - `err_x0`: X-position error estimate
  - `err_y0`: Y-position error estimate
  - `err_r`: Radius error estimate

### Step 3: Compute Morphometry
**Input**:
- `tmp.test_craters.refined.shp` (refined geometries)
- `SHADOWCAM_FAUSTINI_P871S0839.tif` (DEM)
- `SHADOWCAM_FAUSTINI_M017670865SE_170cm.tif` (Orthophoto)

**Output**: `tmp.test_craters.data.shp`
- Complete morphometry data
- Additional fields:
  - `diam`: Refined diameter
  - `d_D`: Depth-diameter ratio
  - `d_D_err`: d/D uncertainty
  - `depth`: Crater depth
  - `depth_err`: Depth uncertainty
  - `rim_JKA`: Rim height (this analysis)
  - `floor_JKA`: Floor elevation (this analysis)

---

## Data Export Formats

### Shapefile Output
- Format: ESRI Shapefile (.shp)
- Includes geometry and all attribute fields
- CRS: Preserved from input data

### CSV Export (in modular version)
- Format: Comma-separated values (.csv)
- Contains all numerical data (no geometry)
- Suitable for statistical analysis in Excel/R/Python

---

## Key Data Characteristics

### Spatial Coverage
- **Region**: Lunar South Pole
- **Coordinate System**: Polar stereographic projection
- **Extent**: x: ~89.5 - 90.3 km, y: ~7.6 - 10.2 km

### Crater Size Distribution
- **Minimum diameter**: 60 meters (configurable threshold)
- **Typical range**: 60 - 400 meters
- **Category**: Small impact craters

### DEM Data
- **Source**: SHADOWCAM (Lunar Reconnaissance Orbiter)
- **Format**: GeoTIFF
- **Application**: Topographic analysis, elevation profiles

### Orthophoto Data
- **Source**: SHADOWCAM or NAC (Narrow Angle Camera)
- **Format**: GeoTIFF
- **Resolution**: 170-210 cm/pixel
- **Application**: Visual context, visualization

---

## Using This Data

### With Original Notebook
```python
# Set region
region, dem, orthophoto = regions['test']

# Load shapefile
df = gpd.read_file(shapefile_path + ".shp")
```

### With Modular Code
```python
from crater_analysis.config import Config

config = Config()
dem_path = config.get_dem_path('test')
shapefile_path = config.get_shapefile_path('test')
```

### Required Data Format
To use your own data with this toolkit:

1. **DEM**: GeoTIFF format, metric units
2. **Orthophoto**: GeoTIFF format, same CRS as DEM
3. **Shapefile**: Must include:
   - `D_m` field (diameter in meters)
   - `UFID` field (unique identifier)
   - Geometry (points or polygons)

---

## Data Source

Based on file paths and naming conventions:
- **Institution**: Arizona State University (ASU)
- **Project**: South Lunar Crater (SLC) analysis
- **Investigator**: asonke (from file paths)
- **Data**: SHADOWCAM mission data products

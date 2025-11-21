# Usage Examples

## Quick Start

### Basic Analysis

```bash
# Run complete analysis on default region
python main.py

# Run on specific region
python main.py --region faustini --min-diameter 100

# Disable plotting for faster processing
python main.py --no-plot
```

### Step-by-Step Analysis

```bash
# Step 1: Prepare crater geometries
python main.py --steps prepare --region test

# Step 2: Refine crater rims
python main.py --steps refine --region test

# Step 3: Compute morphometry
python main.py --steps analyze --region test
```

## Python API Examples

### Example 1: Basic Workflow

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from crater_analysis.config import Config
from crater_analysis.refinement import prepare_crater_geometries, update_crater_rims
from crater_analysis.morphometry import compute_depth_diameter_ratios

# Initialize configuration
config = Config()

# Get paths for 'test' region
region = 'test'
dem_path = config.get_dem_path(region)
orthophoto_path = config.get_orthophoto_path(region)
shapefile_base = config.get_shapefile_path(region)

# Define file paths
input_shp = f"{shapefile_base}.shp"
geom_shp = f"{shapefile_base}.geom.shp"
refined_shp = f"{shapefile_base}.refined.shp"
data_shp = f"{shapefile_base}.data.shp"

# Run workflow
prepare_crater_geometries(input_shp, geom_shp, min_diameter=60)
update_crater_rims(geom_shp, refined_shp, dem_path, orthophoto_path)
compute_depth_diameter_ratios(refined_shp, data_shp, dem_path, orthophoto_path)
```

### Example 2: Configuration Management

```python
from crater_analysis.config import Config

# Load default configuration
config = Config()

# Or load custom configuration
config = Config('/path/to/custom/config.json')

# Get configuration values
regions = list(config.config['regions'].keys())
print(f"Available regions: {regions}")

min_diam = config.get_min_diameter()
print(f"Minimum diameter: {min_diam} m")

# Get paths
dem_path = config.get_dem_path('faustini')
print(f"DEM path: {dem_path}")

# Update paths programmatically
config.set_paths(
    images_dir='/new/path/to/images',
    output_dir='/new/path/to/output'
)
```

### Example 3: Custom Analysis

```python
from crater_analysis.refinement import update_crater_rims
from crater_analysis.morphometry import get_morphometry_summary, export_morphometry_to_csv

# Refine with custom parameters
update_crater_rims(
    input_shapefile='my_craters.shp',
    output_shapefile='my_craters_refined.shp',
    dem_path='my_dem.tif',
    orthophoto_path='my_ortho.tif',
    min_diameter=50,           # Custom minimum
    inner_radius=0.7,          # Custom search range
    outer_radius=1.3,
    plot=True,                 # Enable diagnostic plots
    remove_external_topo=True  # Remove regional slope
)

# Get summary statistics
summary = get_morphometry_summary('my_craters_data.shp')
print(f"Analyzed {summary['count']} craters")
print(f"Mean d/D: {summary['d_D_ratio']['mean']:.3f}")

# Export to CSV for further analysis
export_morphometry_to_csv('my_craters_data.shp', 'results.csv')
```

### Example 4: Using Core Algorithms Directly

```python
import sys
from pathlib import Path
import rasterio as rio
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from crater_analysis import cratools

# Read shapefile and DEM
gdf = gpd.read_file('craters.shp')
dem = rio.open('dem.tif')

# Process single crater
crater_geom = gdf.geometry.iloc[0]
crs = gdf.crs

# Refine rim
refined_geom, errors = cratools.fit_crater_rim(
    geom=crater_geom,
    dem_src=dem,
    crs=crs,
    orthophoto='orthophoto.tif',
    inner_radius=0.8,
    outer_radius=1.2,
    plot=True
)

print(f"Position error: ±{errors[0]:.2f}, ±{errors[1]:.2f} m")
print(f"Radius error: ±{errors[2]:.2f} m")

# Compute depth/diameter
ratio, depth, diam, rim, floor = cratools.compute_depth_diameter_ratio(
    geom=refined_geom,
    filename_dem='dem.tif',
    crs=crs,
    orthophoto='orthophoto.tif',
    plot=True
)

print(f"Diameter: {diam}")
print(f"Depth: {depth}")
print(f"d/D ratio: {ratio}")
```

## Data Preparation

### Converting Point Shapefiles

If you have crater locations as points with diameter information:

```python
from crater_analysis.refinement import prepare_crater_geometries

# This buffers points to circles
prepare_crater_geometries(
    input_shapefile='crater_points.shp',  # Points with D_m field
    output_shapefile='crater_circles.shp', # Output circles
    min_diameter=60
)
```

### Required Shapefile Fields

Your input shapefile should have these fields:

```python
# Required
- geometry: Point or Polygon
- D_m: Diameter in meters (float)
- UFID: Unique identifier (int or string)

# Optional (for comparison)
- davg: Average depth (float)
- davg_D: Depth/diameter ratio (float)
- rim: Rim elevation (float)
- fl: Floor elevation (float)
```

## Batch Processing Multiple Regions

```python
from crater_analysis.config import Config
from crater_analysis.refinement import update_crater_rims
from crater_analysis.morphometry import compute_depth_diameter_ratios

config = Config()

# Process all regions
for region_name in config.config['regions'].keys():
    print(f"Processing region: {region_name}")

    try:
        dem = config.get_dem_path(region_name)
        ortho = config.get_orthophoto_path(region_name)
        shp_base = config.get_shapefile_path(region_name)

        # Run analysis
        update_crater_rims(
            f"{shp_base}.geom.shp",
            f"{shp_base}.refined.shp",
            dem, ortho
        )

        compute_depth_diameter_ratios(
            f"{shp_base}.refined.shp",
            f"{shp_base}.data.shp",
            dem, ortho
        )

        print(f"✓ Completed {region_name}")
    except Exception as e:
        print(f"✗ Failed {region_name}: {e}")
```

## Tips and Tricks

### Performance

- Use `--no-plot` for faster processing
- Process smaller regions separately
- Adjust `min_diameter` to filter small craters

### Quality Control

- Always review diagnostic plots for a few craters
- Check error estimates in output shapefiles
- Compare with original measurements when available

### Troubleshooting

**Import errors**: Install dependencies with `pip install -r requirements.txt`

**File not found**: Check paths in `config/regions.json`

**Memory issues**: Process fewer craters at once or reduce DEM resolution

**Bad rim fits**: Adjust `inner_radius` and `outer_radius` parameters

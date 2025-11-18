# Crater Degradation Analysis and Age Estimation

This toolkit provides comprehensive analysis of lunar impact craters using diffusion-based degradation models to estimate crater ages.

## Features

- **Rim Refinement**: Uses computer vision and image processing to refine approximate crater rim positions from shapefiles
- **Center & Diameter Calculation**: Fits circles to refined rim points for accurate center and diameter determination
- **Tilt Correction**: Removes first-order planar tilt from topography data
- **Radial Profile Extraction**: Extracts 8 elevation profiles at 45° intervals from -1.5D to +1.5D
- **Age Estimation**: Uses diffusion-based degradation models via cratermaker library
- **Output Generation**: Creates labeled shapefiles and visualizations

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install cratermaker

```bash
pip install cratermaker
```

Or follow installation instructions at: https://cratermaker.readthedocs.io/

## Usage

### Command Line Interface

```bash
python crater_age_analysis.py \
    --topo path/to/topography.tif \
    --image path/to/lunar_image.tif \
    --shapefile path/to/crater_rims.shp \
    --output-shp output_crater_ages.shp \
    --output-img output_visualization.png \
    --pixel-size 5.0
```

### Parameters

- `--topo`: Path to GeoTIFF topography raster (required)
- `--image`: Path to GeoTIFF lunar image (required)
- `--shapefile`: Path to shapefile with approximate crater rims (required)
- `--output-shp`: Output shapefile path (default: crater_ages.shp)
- `--output-img`: Output visualization image path (default: crater_ages_visualization.png)
- `--pixel-size`: Pixel size in meters (optional, extracted from raster if not provided)

### Python API Usage

```python
from crater_age_analysis import CraterAgeAnalyzer

# Initialize analyzer
analyzer = CraterAgeAnalyzer(
    topo_path='topography.tif',
    image_path='lunar_image.tif',
    shapefile_path='crater_rims.shp',
    pixel_size_meters=5.0  # Optional
)

# Process all craters
results = analyzer.process_all_craters()

# Save results
analyzer.save_results(results, 'crater_ages.shp')

# Create visualization
analyzer.visualize_results(results, 'crater_ages_visualization.png')

# Clean up
analyzer.close()
```

## Input Data Requirements

### 1. Topography GeoTIFF
- Single-band raster with elevation values
- Must have proper georeferencing (coordinate system)
- Common sources: LRO LOLA DEM, SELENE Kaguya TC DEM

### 2. Lunar Image GeoTIFF
- Single-band grayscale image
- Same coordinate system as topography
- Ideally same resolution and extent
- Common sources: LRO NAC/WAC, SELENE TC ortho

### 3. Crater Rim Shapefile
- Polygon geometries representing approximate crater rims
- Can be circular or irregular polygons
- Must be in same coordinate system as rasters
- Each polygon represents one crater

## Methodology

### 1. Rim Refinement
The algorithm refines crater rim positions using:
- Sobel edge detection on topography (60% weight)
- Sobel edge detection on image (40% weight)
- Local search within specified radius (default 10 pixels)

### 2. Center and Diameter Calculation
- Fits optimal circle to refined rim points
- Minimizes variance of radii using Nelder-Mead optimization
- Returns center coordinates and radius

### 3. Tilt Correction
- Fits planar surface (z = ax + by + c) to topography
- Removes regional slope to isolate crater morphology
- Uses least-squares plane fitting

### 4. Radial Profile Extraction
- Extracts 8 profiles at 45° intervals (N, NE, E, SE, S, SW, W, NW)
- Each profile extends from -1.5D to +1.5D
- Samples 300 points per profile

### 5. Age Estimation
Uses the cratermaker library's diffusion-based model:
- Analyzes crater depth and rim degradation
- Compares to theoretical diffusion profiles
- Estimates age in billions of years (Ga)

Fallback method (if cratermaker unavailable):
- Calculates depth-to-diameter (d/D) ratio
- Classifies based on degradation state:
  - Fresh: d/D > 0.18 (<0.1 Ga)
  - Young: d/D > 0.12 (0.1-1 Ga)
  - Mature: d/D > 0.08 (1-3 Ga)
  - Old: d/D > 0.04 (3-4 Ga)
  - Very Old: d/D ≤ 0.04 (>4 Ga)

## Output Files

### 1. Output Shapefile (crater_ages.shp)
Contains all original attributes plus:
- `center_x`, `center_y`: Refined center coordinates
- `diameter_m`: Corrected diameter in meters
- `depth_m`: Crater depth in meters
- `age`: Estimated age (numerical or categorical)
- `degradation`: Degradation parameter from cratermaker
- `num_profiles`: Number of successfully extracted profiles

### 2. Visualization Image (crater_ages_visualization.png)
- Lunar image background
- Red circles showing refined crater positions
- Yellow text labels with age estimates
- High-resolution output (300 DPI default)

## Example Workflow

```python
# Example: Process lunar craters and analyze results

import geopandas as gpd
import matplotlib.pyplot as plt
from crater_age_analysis import CraterAgeAnalyzer

# Initialize
analyzer = CraterAgeAnalyzer(
    topo_path='data/lunar_dem.tif',
    image_path='data/lunar_ortho.tif',
    shapefile_path='data/crater_candidates.shp'
)

# Process
results = analyzer.process_all_craters()

# Save
analyzer.save_results(results, 'outputs/crater_ages.shp')
analyzer.visualize_results(results, 'outputs/crater_map.png')

# Analysis
print(f"Total craters analyzed: {len(results)}")
print(f"Mean diameter: {results['diameter_m'].mean():.2f} m")
print(f"Mean depth: {results['depth_m'].mean():.2f} m")

# Age distribution
age_counts = results['age'].value_counts()
print("\nAge Distribution:")
print(age_counts)

# Clean up
analyzer.close()
```

## Troubleshooting

### Issue: "cratermaker not installed"
**Solution**: Install cratermaker: `pip install cratermaker`
The script will use a fallback method if cratermaker is unavailable.

### Issue: "Raster and shapefile CRS mismatch"
**Solution**: Reproject shapefile to match raster CRS:
```python
import geopandas as gpd
import rasterio

with rasterio.open('topography.tif') as src:
    raster_crs = src.crs

gdf = gpd.read_file('craters.shp')
gdf = gdf.to_crs(raster_crs)
gdf.to_file('craters_reprojected.shp')
```

### Issue: "Memory error with large rasters"
**Solution**: Process craters in batches or use lower resolution data

### Issue: Poor rim refinement results
**Solution**: Adjust `search_radius` parameter in `refine_rim_position()` method

## References

1. **cratermaker Documentation**: https://cratermaker.readthedocs.io/en/latest/api/morphology.html
2. Fassett, C. I., & Thomson, B. J. (2014). Crater degradation on the lunar maria: Topographic diffusion and the rate of erosion on the Moon. JGR Planets.
3. Howard, A. D. (2007). Simulating the development of Martian highland landscapes through the interaction of impact cratering, fluvial erosion, and variable hydrologic forcing. Geomorphology.

## Citation

If you use this code in your research, please cite:
- The cratermaker library: https://github.com/profminton/cratermaker
- Relevant scientific papers on crater degradation modeling

## License

This code is provided for research and educational purposes.

## Contributing

For bug reports, feature requests, or contributions, please contact the development team.

## Version History

- **v1.0.0** (2025-11-18): Initial release
  - Rim refinement with edge detection
  - Circle fitting for center/diameter
  - Tilt correction
  - Radial profile extraction
  - Age estimation with cratermaker integration
  - Shapefile and visualization output

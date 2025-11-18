# Crater Degradation Analysis and Age Estimation

This toolkit provides comprehensive analysis of lunar impact craters using diffusion-based degradation models to estimate crater ages.

## Features

- **Rim Refinement**: Uses computer vision and image processing to refine approximate crater rim positions from shapefiles
- **Center & Diameter Calculation**: Fits circles to refined rim points for accurate center and diameter determination
- **Tilt Correction**: Removes first-order planar tilt from topography data
- **Radial Profile Extraction**: Extracts 8 elevation profiles at 45° intervals from -1.5D to +1.5D
- **Multiple Age Estimation Methods**:
  - **Topography Degradation Model** (Luo et al. 2025) - Primary method
  - **Cratermaker Diffusion Model** - Alternative method
  - **Depth-Diameter Ratio** - Fallback method
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

# Initialize analyzer with default (auto) age estimation method
analyzer = CraterAgeAnalyzer(
    topo_path='topography.tif',
    image_path='lunar_image.tif',
    shapefile_path='crater_rims.shp',
    pixel_size_meters=5.0,  # Optional
    age_method='auto',  # Options: 'auto', 'topography_degradation', 'cratermaker', 'both'
    diffusivity=5.0  # Diffusivity coefficient in m²/Myr (for topography_degradation)
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

### Age Estimation Methods

The toolkit supports multiple age estimation methods:

1. **`age_method='auto'`** (default): Automatically selects the best available method
   - Prefers topography_degradation if available
   - Falls back to cratermaker, then depth-diameter ratio

2. **`age_method='topography_degradation'`**: Uses Luo et al. (2025) model
   - Best for craters > 400m diameter
   - Uncertainty < 165 Ma for large craters
   - Based on diffusive degradation modeling

3. **`age_method='cratermaker'`**: Uses cratermaker library
   - Requires cratermaker installation
   - General diffusion-based approach

4. **`age_method='both'`**: Runs both methods for comparison
   - Returns ages from all available methods
   - Useful for validation and uncertainty quantification

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

The toolkit provides multiple age estimation methods:

#### Method A: Topography Degradation Model (Luo et al. 2025) - **Primary Method**
Based on DOI: 10.5281/zenodo.15168130

This method uses diffusive degradation modeling:
- Generates pristine (fresh) crater profile based on empirical morphology
  - Parabolic bowl interior for simple craters
  - Exponential rim decay
  - d/D ≈ 0.196 for fresh lunar simple craters
- Models degradation through diffusion equation
  - Smoothing length scale: σ = √(2κt) where κ is diffusivity, t is time
  - Default diffusivity: 5 m²/Myr (typical lunar value: 3-10 m²/Myr)
- Fits observed profiles to degraded model to estimate age
- Calculates age from 8 radial profiles and returns mean ± std
- **Best for craters > 400m diameter**
- **Typical uncertainty: < 165 Ma for large craters**
- Validates against isotopic ages from lunar samples

#### Method B: Cratermaker Diffusion Model
Uses the cratermaker library's diffusion-based model:
- Analyzes crater depth and rim degradation
- Compares to theoretical diffusion profiles
- Estimates age in billions of years (Ga)
- Requires cratermaker package installation

#### Method C: Depth-Diameter Ratio (Fallback)
Quick classification based on degradation state:
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

1. **Luo, F., Xiao, Z., Xie, M., Wang, Y., & Ma, Y. (2025).** Age Estimation of Individual Lunar Simple Craters Using the Topography Degradation Model. *Journal of Geophysical Research: Planets*. DOI: 10.1029/2025JE008937
   - Software/Data: https://doi.org/10.5281/zenodo.15168130

2. **Fassett, C. I., & Thomson, B. J. (2014).** Crater degradation on the lunar maria: Topographic diffusion and the rate of erosion on the Moon. *JGR Planets*, 119(10), 2255-2271.

3. **Pike, R. J. (1977).** Size-dependence in the shape of fresh impact craters on the moon. *Impact and Explosion Cratering*, 489-509.

4. **cratermaker Documentation**: https://cratermaker.readthedocs.io/en/latest/api/morphology.html

5. **Howard, A. D. (2007).** Simulating the development of Martian highland landscapes through the interaction of impact cratering, fluvial erosion, and variable hydrologic forcing. *Geomorphology*, 91(3-4), 332-363.

## Citation

If you use this code in your research, please cite:

- **For the topography degradation method:**
  ```
  Luo, F., Xiao, Z., Xie, M., Wang, Y., & Ma, Y. (2025). Age Estimation of
  Individual Lunar Simple Craters Using the Topography Degradation Model.
  Journal of Geophysical Research: Planets. https://doi.org/10.1029/2025JE008937
  ```

- **For the cratermaker method:**
  - The cratermaker library: https://github.com/profminton/cratermaker

- **General crater degradation modeling:**
  - Fassett & Thomson (2014) for diffusion-based approaches
  - Pike (1977) for pristine crater morphology relationships

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

# Crater Degradation Analysis and Age Estimation

This toolkit provides comprehensive analysis of lunar impact craters using diffusion-based degradation models to estimate crater ages.

## Features

- **Rim Refinement**: Uses computer vision and image processing to refine approximate crater rim positions from shapefiles
- **Center & Diameter Calculation**: Fits circles to refined rim points for accurate center and diameter determination
- **Tilt Correction**: Removes first-order planar tilt from topography data
- **Radial Profile Extraction**: Extracts 8 elevation profiles at 45° intervals from -1.5D to +1.5D
- **Chebyshev Coefficient Extraction**: Computes 17×8 matrix of Chebyshev coefficients (C0-C16) from radial profiles
  - Standardized morphological descriptors
  - Depth-to-diameter ratio inference (C2)
  - Central peak detection (C4, C8)
  - Asymmetry analysis (odd coefficients)
- **Multiple Age Estimation Methods**:
  - **Topography Degradation Model** (Luo et al. 2025) - Primary method
  - **Cratermaker Diffusion Model** - Alternative method
  - **Depth-Diameter Ratio** - Fallback method
- **Output Generation**: Creates labeled shapefiles, visualizations, and Chebyshev coefficient matrices

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

### 4.5. Chebyshev Coefficient Extraction
Chebyshev polynomials provide a standardized mathematical framework for quantitative crater characterization:

- **Polynomial Fitting**: Fits 17 Chebyshev polynomials (C0-C16) to each radial profile
- **Normalization**:
  - Radial distances normalized by crater diameter D: positions from -D to +D map to -1 to +1
  - Elevations centered at 0 (mean subtracted) and divided by diameter D
  - This ensures size-independent comparison across different crater diameters
- **Output**: 17×8 coefficient matrix (17 coefficients × 8 profiles)
- **Physical Interpretation**:
  - **C0**: Normalized mean elevation offset (dimensionless)
  - **C2**: Depth-to-diameter ratio indicator (curvature)
  - **C4, C8**: Central peak presence indicators
  - **Odd coefficients (C1, C3, C5, ...)**: Asymmetry indicators
  - **Even coefficients (C0, C2, C4, ...)**: Symmetric morphological features

**Research Basis**: Most crater elevation profiles can be well represented using 17 Chebyshev coefficients, as demonstrated by LROC team analysis of 765 craters ranging from 100m to 145km in diameter.

**Standardization**: By normalizing distances and elevations by diameter, coefficients become directly comparable across craters of different sizes, enabling:
- Standardized comparison between craters
- Degradation state quantification independent of size
- Central peak detection in complex craters
- Profile asymmetry assessment
- Automated crater classification

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
- `degradation`: Degradation parameter from age estimation
- `num_profiles`: Number of successfully extracted profiles
- `C0_mean` through `C16_mean`: Mean Chebyshev coefficients across 8 profiles
- `depth_indicator`: C2-based depth-to-diameter indicator
- `central_peak_idx`: Combined C4+C8 central peak indicator
- `asymmetry_idx`: Sum of odd coefficient magnitudes
- `profile_consistency`: Standard deviation of coefficients (lower = more symmetric)

### 2. Chebyshev Coefficient Matrices (chebyshev_coefficients/ directory)
Full 17×8 coefficient matrices for detailed morphological analysis:

- **Individual crater files**:
  - `crater_N_chebyshev_17x8.npy`: NumPy binary format (for Python analysis)
  - `crater_N_chebyshev_17x8.csv`: CSV format (for spreadsheet viewing)
  - Rows: C0-C16 coefficients
  - Columns: 8 profiles at 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

- **Summary file**:
  - `all_craters_chebyshev.npz`: All crater matrices in compressed NumPy archive

**Loading Chebyshev matrices in Python**:
```python
import numpy as np

# Load single crater matrix
matrix = np.load('chebyshev_coefficients/crater_0_chebyshev_17x8.npy')
print(f"Shape: {matrix.shape}")  # (17, 8)

# Load all craters
all_data = np.load('chebyshev_coefficients/all_craters_chebyshev.npz')
crater_0 = all_data['crater_0']
```

### 3. MATLAB Export (crater_analysis.mat)
Optional export format for MATLAB users:

```python
# Enable MATLAB export
analyzer.save_results(results, 'crater_ages.shp', save_matlab=True)
```

**MATLAB file contents**:
- `num_craters`: Number of craters analyzed
- `chebyshev_coefficients`: 3D array (17 × 8 × N) - all Chebyshev matrices
- `diameters`: Array of crater diameters (meters)
- `depths`: Array of crater depths (meters)
- `crater_info`: Struct array with detailed information

**Loading in MATLAB**:
```matlab
% Load data
data = load('crater_analysis.mat');

% Access Chebyshev coefficients for crater 1
cheb_crater1 = data.chebyshev_coefficients(:,:,1);  % 17x8 matrix

% Plot all crater diameters
histogram(data.diameters);
xlabel('Diameter (m)');
ylabel('Count');
```

### 4. Visualization Image (crater_ages_visualization.png)
- Lunar image background
- Red circles showing refined crater positions
- Yellow text labels with age estimates
- High-resolution output (300 DPI default)

## Synthesis and Degradation Testing

A comprehensive test suite is provided to validate the degradation models and Chebyshev analysis:

### Running the Synthesis Test

```bash
python crater_synthesis_degradation_test.py
```

This generates:
- 10 synthetic craters (800m to 5km diameter)
- Degraded profiles from 0.1 to 3.5 Ga at 0.1 Ga intervals
- Chebyshev coefficient evolution analysis
- Depth-to-diameter ratio tracking

### Test Outputs

**CSV File** (`degradation_chebyshev_results.csv`):
- Complete tabular data for all craters and ages
- Columns: crater_id, diameter, age, d/D ratio, 17 Chebyshev coefficients, derived indices

**MATLAB File** (`degradation_analysis.mat`):
- All pristine and degraded profiles
- Chebyshev coefficient matrices
- d/D ratio evolution arrays

**PDF Report** (`degradation_analysis_report.pdf`):
- Page 1: Methodology summary
- Pages 2-3: Elevation profile evolution
- Page 4: d/D ratio evolution plots
- Pages 5-6: Chebyshev coefficient evolution
- Page 7: Absolute coefficient magnitude trends

### Key Findings from Synthesis Test

The test demonstrates:
1. **d/D Degradation**: Exponential-like decrease from ~0.196 (fresh) to ~0.05 (ancient)
2. **Coefficient Evolution**:
   - C2 (depth indicator) decreases monotonically
   - Higher-order coefficients decay faster
   - C0 remains relatively stable (baseline)
3. **Size Dependence**: Larger craters show slower relative degradation
4. **Time Scales**: Significant morphological changes within first 1 Ga

## Degradation Animation

Generate animated quadchart visualizations showing crater evolution over time.

### Usage

```bash
# Basic usage - 2km crater
python crater_degradation_animation.py --diameter 2000 --output crater_2km.mp4

# Full parameter control
python crater_degradation_animation.py \
    --diameter 5000 \
    --age-min 0.1 \
    --age-max 3.9 \
    --frames 150 \
    --fps 15 \
    --output crater_degradation.mp4

# Generate GIF instead of MP4
python crater_degradation_animation.py --diameter 3000 --output crater.gif
```

### Animation Layout

The animation shows a quadchart with 4 panels evolving over time:

**Quadrant 1 (Top Left): 3D Topography**
- 3D surface plot of crater morphology
- Generated from radial profile using axisymmetric assumption
- Shows visual degradation of rim and floor
- Consistent viewing angle (30° elevation, 45° azimuth)

**Quadrant 2 (Top Right): 2D Elevation Profile**
- Normalized radial elevation profile (h/D vs r/D)
- Black dashed line: pristine crater (reference)
- Blue solid line: current degraded state
- Shows smoothing of rim and filling of crater floor

**Quadrant 3 (Bottom Left): d/D Ratio Evolution**
- Tracks depth-to-diameter ratio over time
- Horizontal dashed line: pristine value (0.196)
- Red curve: degradation trajectory
- Large red dot: current state
- Demonstrates exponential-like decay

**Quadrant 4 (Bottom Right): Chebyshev Coefficient Evolution**
- Tracks normalized values of key coefficients:
  - C0 (mean elevation) - Blue
  - C2 (depth indicator) - Red
  - C4 (central peak) - Green
  - C8 (central peak) - Orange
- Shows which morphological features change fastest
- All coefficients normalized to [-1, 1] range

### Parameters

- `--diameter FLOAT`: Crater diameter in meters (required)
- `--output STR`: Output filename (.mp4 or .gif)
- `--age-min FLOAT`: Minimum age in Ga (default: 0.1)
- `--age-max FLOAT`: Maximum age in Ga (default: 3.9)
- `--frames INT`: Number of frames (default: 100)
- `--fps INT`: Frames per second (default: 10)

### Output Formats

**MP4 (recommended)**:
- Requires ffmpeg
- Smaller file size
- Better quality
- Suitable for presentations

**GIF**:
- No external dependencies
- Larger file size
- Compatible with all platforms
- Good for web/email

### Technical Details

- **Degradation model**: Diffusion-based (κ = 5 m²/Myr)
- **Frame generation**: Pre-computed for smooth playback
- **3D surface**: 72 angular points × radial profile length
- **Coefficient normalization**: Per-coefficient global max
- **Resolution**: 1600×1200 pixels at 150 DPI

### Example Output Characteristics

For a 2km crater over 0.1-3.9 Ga:
- Initial d/D: ~0.196 (fresh)
- Final d/D: ~0.05 (heavily degraded)
- C2 coefficient: Decreases ~80%
- Higher-order coefficients: Approach zero faster
- Visual: Rim height reduces from ~80m to ~20m

### Python API Usage

```python
from crater_degradation_animation import generate_crater_animation

# Generate animation
animator = generate_crater_animation(
    diameter_m=2000,
    output_file='crater_2km.mp4',
    age_range=(0.1, 3.9),
    num_frames=100,
    fps=10
)

# Access computed data
print(f"Final d/D ratio: {animator.d_D_ratios[-1]:.4f}")
print(f"Coefficient matrix shape: {animator.chebyshev_matrices.shape}")
```

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

4. **LROC Team.** Chebyshev polynomial analysis of lunar crater morphology. *Lunar Reconnaissance Orbiter Camera*. http://lroc.sese.asu.edu/posts/864
   - Standardized approach for quantitative crater topography characterization
   - Analysis of 765 craters (100m to 145km diameter) using 17 Chebyshev coefficients

5. **cratermaker Documentation**: https://cratermaker.readthedocs.io/en/latest/api/morphology.html

6. **Howard, A. D. (2007).** Simulating the development of Martian highland landscapes through the interaction of impact cratering, fluvial erosion, and variable hydrologic forcing. *Geomorphology*, 91(3-4), 332-363.

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

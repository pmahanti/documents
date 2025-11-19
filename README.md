# Documents

References for projects

## Crater Degradation Analysis Tool

A comprehensive Python toolkit for analyzing lunar impact craters using diffusion-based degradation models to estimate crater ages.

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python crater_age_analysis.py \
    --topo topography.tif \
    --image lunar_image.tif \
    --shapefile crater_rims.shp \
    --output-shp crater_ages.shp \
    --output-img crater_visualization.png
```

### Features

- **Automated rim refinement** using computer vision and edge detection
- **Precise center and diameter calculation** via circle fitting
- **First-order tilt correction** for accurate morphology
- **8 radial elevation profiles** at 45° intervals
- **Chebyshev coefficient extraction**: 17×8 matrix for morphological analysis
- **Multiple age estimation methods**:
  - Topography degradation model (Luo et al. 2025)
  - Cratermaker diffusion models
  - Depth-diameter ratio fallback
- **Degradation Animation**: Quadchart visualization of crater evolution (3D topography, profiles, d/D, Chebyshev)
- **Shapefile output** with age labels, statistics, and Chebyshev coefficients
- **Visualization** with labeled crater map and animations

### Documentation

See [README_CRATER_ANALYSIS.md](README_CRATER_ANALYSIS.md) for detailed documentation.

### Files

- `crater_age_analysis.py` - Main analysis script with MATLAB export
- `topography_degradation_age.py` - Topography degradation model (Luo et al. 2025)
- `chebyshev_coefficients.py` - Chebyshev polynomial coefficient extraction
- `crater_synthesis_degradation_test.py` - Synthesis and degradation test suite
- `crater_degradation_animation.py` - Animated quadchart degradation visualization
- `example_usage.py` - Example usage and API demonstrations
- `validate_code.py` - Code validation script
- `test_chebyshev.py` - Chebyshev coefficient unit tests
- `requirements.txt` - Python dependencies
- `README_CRATER_ANALYSIS.md` - Comprehensive documentation

### Quick Animation

```bash
# Generate degradation animation for a 2km crater
python crater_degradation_animation.py --diameter 2000 --output crater_2km.mp4

# Custom parameters
python crater_degradation_animation.py \
    --diameter 5000 \
    --age-min 0.1 \
    --age-max 3.9 \
    --frames 150 \
    --fps 15 \
    --output crater_5km_degradation.mp4
```

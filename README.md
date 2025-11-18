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
- **Multiple age estimation methods**:
  - Topography degradation model (Luo et al. 2025)
  - Cratermaker diffusion models
  - Depth-diameter ratio fallback
- **Shapefile output** with age labels and statistics
- **Visualization** with labeled crater map

### Documentation

See [README_CRATER_ANALYSIS.md](README_CRATER_ANALYSIS.md) for detailed documentation.

### Files

- `crater_age_analysis.py` - Main analysis script
- `topography_degradation_age.py` - Topography degradation model (Luo et al. 2025)
- `example_usage.py` - Example usage and API demonstrations
- `validate_code.py` - Code validation script
- `requirements.txt` - Python dependencies
- `README_CRATER_ANALYSIS.md` - Comprehensive documentation

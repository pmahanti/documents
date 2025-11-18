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
- **Age estimation** using cratermaker diffusion models
- **Shapefile output** with age labels
- **Visualization** with labeled crater map

### Documentation

See [README_CRATER_ANALYSIS.md](README_CRATER_ANALYSIS.md) for detailed documentation.

### Files

- `crater_age_analysis.py` - Main analysis script
- `example_usage.py` - Example usage and API demonstrations
- `validate_code.py` - Code validation script
- `requirements.txt` - Python dependencies
- `README_CRATER_ANALYSIS.md` - Comprehensive documentation

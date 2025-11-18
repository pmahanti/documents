# documents
References for projects

## Crater Detection Training Data Generator

A comprehensive Python toolkit for generating labeled training datasets for crater detection in lunar images using deep learning models (YOLO, CNN, DETR).

### Features

- Support for GeoTiff and ISIS .cub files
- Handles multiple map projections (equidistant cylindrical, stereographic, orthographic)
- Shapefile-based crater annotations
- YOLO and COCO output formats
- Optional topography raster processing
- Image tiling for large datasets
- Visualization and verification tools

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate a dataset
python crater_training_data_generator.py \
    --image lunar_image.tif \
    --craters craters.shp \
    --output ./crater_dataset \
    --format yolo
```

### Documentation

See [README_CRATER_DETECTION.md](README_CRATER_DETECTION.md) for complete documentation.

### Files

- `crater_training_data_generator.py` - Main dataset generation script
- `crater_utils.py` - Utility functions for dataset management
- `example_usage.py` - Example scripts demonstrating different use cases
- `requirements.txt` - Python package dependencies
- `README_CRATER_DETECTION.md` - Detailed documentation

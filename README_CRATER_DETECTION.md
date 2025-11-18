# Crater Detection Training Data Generator

A Python toolkit for generating labeled training datasets for crater detection in lunar images using deep learning models (YOLO, CNN, DETR).

## Features

- **Multiple Input Formats**: Supports GeoTiff and ISIS .cub files
- **Flexible Projections**: Handles equidistant cylindrical, stereographic, and orthographic projections
- **Shapefile Integration**: Uses crater annotations from shapefiles
- **Multiple Output Formats**: YOLO and COCO formats for different model architectures
- **Topography Support**: Optional topography raster processing
- **Image Tiling**: Automatic tiling for large images with configurable overlap
- **Visualization**: Generates annotated images for verification
- **Dataset Utilities**: Train/val/test splitting, statistics, and verification tools

## Installation

### System Dependencies

For ISIS .cub file support, install GDAL with ISIS drivers:

```bash
# Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev python3-gdal

# macOS (using Homebrew)
brew install gdal

# Or install USGS ISIS3 for full ISIS support
# https://isis.astrogeology.usgs.gov/
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Generate YOLO format dataset from a single image:

```bash
python crater_training_data_generator.py \
    --image lunar_image.tif \
    --craters craters.shp \
    --output ./crater_dataset \
    --format yolo
```

### Advanced Usage

Generate tiled dataset with topography in both YOLO and COCO formats:

```bash
python crater_training_data_generator.py \
    --image lunar_mosaic.cub \
    --craters crater_database.shp \
    --output ./crater_dataset_tiled \
    --format both \
    --tile-size 512 \
    --overlap 64 \
    --topography lunar_dem.tif
```

### Command Line Arguments

- `--image`: Path to input image (GeoTiff or ISIS .cub file) **[Required]**
- `--craters`: Path to crater shapefile **[Required]**
- `--output`: Output directory for dataset **[Required]**
- `--format`: Output format - `yolo`, `coco`, or `both` (default: `yolo`)
- `--topography`: Optional path to topography raster file
- `--tile-size`: Tile size for splitting large images (e.g., 512, 1024)
- `--overlap`: Overlap between tiles in pixels (default: 0)

## Projection Support

The tool automatically handles different map projections commonly used in planetary science:

- **Equidistant Cylindrical (Equirectangular)**: Most common for global mosaics
- **Stereographic**: Often used for polar regions
- **Orthographic**: Used for hemisphere views

The projection information should be embedded in the GeoTiff metadata or ISIS label. The tool uses rasterio/GDAL which automatically reads and handles the projection transformations when matching crater coordinates to image pixels.

## Input Data Requirements

### Image Files

**Supported formats:**
- GeoTiff (.tif, .tiff) with embedded georeferencing
- ISIS cube files (.cub) with label information

**Requirements:**
- Must have valid coordinate reference system (CRS) information
- Can be single-band (grayscale) or multi-band (RGB)
- Any bit depth supported by GDAL/rasterio

### Crater Shapefile

**Format:** ESRI Shapefile (.shp)

**Geometry types:**
- Point: Center point of crater (will generate circular bounding box)
- Polygon: Crater outline (will generate bounding box from bounds)
- Circle/Ellipse: Crater perimeter

**Requirements:**
- Must have the same CRS as the image, or tool will automatically reproject
- Common attributes (optional but recommended):
  - `diameter` or `radius`: Crater size in map units
  - `name` or `id`: Crater identifier
  - `confidence`: Annotation confidence/quality

### Topography Raster (Optional)

**Format:** GeoTiff or any GDAL-readable raster

**Requirements:**
- Should cover the same spatial extent as the main image
- Same or compatible CRS
- Typically a Digital Elevation Model (DEM) or slope map

## Output Structure

```
output_directory/
├── images/                          # Training images
│   ├── image_000.png
│   ├── image_001.png
│   └── ...
├── labels/                          # YOLO format labels
│   ├── image_000.txt
│   ├── image_001.txt
│   └── ...
├── visualizations/                  # Annotated images for verification
│   ├── image_000_labeled.png
│   └── ...
├── topography_images/               # Topography images (if --topography used)
│   └── ...
├── topography_labels/               # Topography labels (if --topography used)
│   └── ...
├── annotations_coco.json            # COCO format labels (if format is 'coco' or 'both')
└── dataset.yaml                     # YOLO dataset configuration
```

## Label Formats

### YOLO Format (.txt files)

Each line represents one crater:
```
class_id x_center y_center width height
```

All values except class_id are normalized to [0, 1]:
```
0 0.5234 0.7123 0.0456 0.0389
```

### COCO Format (annotations_coco.json)

JSON file with three main sections:
```json
{
  "images": [...],
  "annotations": [...],
  "categories": [{"id": 0, "name": "crater"}]
}
```

## Dataset Utilities

The `crater_utils.py` module provides additional functionality:

### Split Dataset into Train/Val/Test

```bash
python crater_utils.py split --dataset ./crater_dataset \
                             --train-ratio 0.8 \
                             --val-ratio 0.1 \
                             --test-ratio 0.1
```

### View Dataset Statistics

```bash
python crater_utils.py stats --dataset ./crater_dataset
```

Output example:
```
Dataset Statistics
==============================================================
Total images: 150
Images with craters: 142
Images without craters: 8
Total craters: 1523
Average craters per image: 10.15

Crater size statistics (normalized):
  Average: 0.0523
  Median: 0.0445
  Min: 0.0089
  Max: 0.2341
==============================================================
```

### Verify Dataset Integrity

```bash
python crater_utils.py verify --dataset ./crater_dataset
```

### Convert YOLO to COCO Format

```bash
python crater_utils.py convert --dataset ./crater_dataset \
                               --output ./annotations_coco.json
```

## Training with Different Models

### YOLO (Ultralytics)

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='./crater_dataset/dataset.yaml',
    epochs=100,
    imgsz=512,
    batch=16
)
```

### DETR (Detection Transformer)

```python
from transformers import DetrForObjectDetection, DetrImageProcessor
import torch

# Load COCO format dataset
# Use annotations_coco.json for training

# Initialize model
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=1,  # One class: crater
    ignore_mismatched_sizes=True
)

# Training code here...
```

## Examples

### Example 1: Simple Dataset Generation

```bash
python crater_training_data_generator.py \
    --image lunar_region_M123456.tif \
    --craters manually_annotated_craters.shp \
    --output ./simple_dataset \
    --format yolo
```

### Example 2: Large Mosaic with Tiling

```bash
# For a large 10000x10000 pixel image
python crater_training_data_generator.py \
    --image large_lunar_mosaic.cub \
    --craters crater_catalog.shp \
    --output ./tiled_dataset \
    --format both \
    --tile-size 640 \
    --overlap 128
```

### Example 3: Multi-Modal Dataset (Image + Topography)

```bash
python crater_training_data_generator.py \
    --image lunar_image.tif \
    --craters craters.shp \
    --topography lunar_slope_map.tif \
    --output ./multimodal_dataset \
    --format yolo \
    --tile-size 512
```

Then split the dataset:

```bash
python crater_utils.py split --dataset ./multimodal_dataset
```

## Tips and Best Practices

### Image Tiling

- **Tile size**: Use 512, 640, or 1024 pixels (common for YOLO/DETR)
- **Overlap**: Use 10-20% overlap (64-128 pixels) to avoid missing craters on tile boundaries
- **Empty tiles**: The generator saves all tiles by default; filter out empty tiles if needed

### Handling Different Scales

Craters can vary widely in size. Consider:

- Creating separate datasets for different size ranges
- Using multi-scale training
- Adjusting tile size based on crater sizes in your dataset

### Quality Control

1. **Always check visualizations**: Review images in `visualizations/` folder
2. **Verify projections**: Ensure crater locations align correctly with image
3. **Check statistics**: Use `crater_utils.py stats` to understand your data distribution
4. **Validate dataset**: Run `crater_utils.py verify` before training

### Performance Considerations

- Large images (>10000px): Use tiling to reduce memory usage
- Many small craters: Smaller tile sizes work better
- Few large craters: Larger tile sizes or full image processing

## Troubleshooting

### Issue: "CRS mismatch" warning

**Solution**: The tool automatically reprojects shapefiles, but verify your data has valid CRS information:
```bash
# Check image CRS
gdalinfo your_image.tif | grep "Coordinate System"

# Check shapefile CRS
ogrinfo -al -so your_craters.shp
```

### Issue: Craters not appearing in correct locations

**Possible causes:**
1. Projection mismatch (check CRS)
2. Shapefile uses different coordinate system
3. Image lacks georeferencing

**Solution**: Ensure both image and shapefile have proper CRS defined

### Issue: ISIS .cub files not opening

**Solution**: Install GDAL with ISIS drivers or convert to GeoTiff:
```bash
gdal_translate input.cub output.tif
```

### Issue: Out of memory errors

**Solution**: Use tiling:
```bash
--tile-size 512 --overlap 64
```

## Citation

If you use this tool in your research, please cite appropriately and consider sharing your improvements!

## License

MIT License - feel free to use and modify for your research.

## Contributing

Contributions are welcome! Areas for improvement:

- Support for additional annotation formats (CSV, GeoJSON)
- Data augmentation pipeline
- Multi-class support (different crater types)
- Integration with planetary data archives
- Automated quality control

## Contact

For questions, issues, or contributions, please open an issue on the repository.

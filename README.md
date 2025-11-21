# Crater Morphometry Analysis Toolkit

A Python package for analyzing impact crater morphology from Digital Elevation Models (DEMs). This toolkit provides automated rim detection, depth-diameter ratio computation, and comprehensive morphometric analysis.

## Features

- **Crater Rim Refinement**: Automated topographic rim detection and circle fitting
- **Morphometry Analysis**: Depth, diameter, and d/D ratio calculations with uncertainty propagation
- **Regional Topography Removal**: Plane fitting to isolate crater-specific topography
- **Modular Architecture**: Separated concerns for configuration, refinement, and analysis
- **Export Capabilities**: Results in shapefile and CSV formats

## Project Structure

```
documents/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── main.py                           # Main orchestration script
├── config/
│   └── regions.json                  # Region and path configuration
├── src/
│   └── crater_analysis/
│       ├── __init__.py               # Package initialization
│       ├── config.py                 # Configuration management
│       ├── cratools.py               # Core crater analysis algorithms
│       ├── refinement.py             # Crater rim refinement
│       └── morphometry.py            # Depth-diameter analysis
├── tests/
│   ├── test_syntax.py                # Syntax validation
│   └── test_imports.py               # Import tests
├── data/                             # Data directory (user-provided)
├── compute_depth_diameter.ipynb      # Original Jupyter notebook
└── cratools.py                       # Legacy standalone version
```

## Installation

### Requirements

- Python 3.7+
- Scientific Python stack (numpy, scipy, pandas)
- Geospatial libraries (geopandas, rasterio, shapely)

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure paths in `config/regions.json` to match your data locations

## Configuration

Edit `config/regions.json` to define your regions and data paths:

```json
{
  "regions": {
    "my_region": {
      "name": "my_crater_dataset",
      "dem": "my_dem.tif",
      "orthophoto": "my_orthophoto.tif"
    }
  },
  "default_region": "my_region",
  "min_diameter": 60,
  "paths": {
    "images_dir": "/path/to/images",
    "shapefiles_dir": "/path/to/shapefiles",
    "output_dir": "/path/to/output"
  }
}
```

## Usage

### Command Line Interface

Run the complete workflow:

```bash
python main.py --region test --min-diameter 60
```

Run specific steps:

```bash
# Prepare geometries only
python main.py --steps prepare

# Refine rims only
python main.py --steps refine

# Analyze morphometry only
python main.py --steps analyze
```

Additional options:

```bash
python main.py --help
```

### Python API

Use individual modules in your own scripts:

```python
from crater_analysis.config import Config
from crater_analysis.refinement import update_crater_rims
from crater_analysis.morphometry import compute_depth_diameter_ratios

# Load configuration
config = Config()

# Get paths
dem_path = config.get_dem_path('test')
orthophoto_path = config.get_orthophoto_path('test')
shapefile_base = config.get_shapefile_path('test')

# Refine crater rims
update_crater_rims(
    input_shapefile=f"{shapefile_base}.geom.shp",
    output_shapefile=f"{shapefile_base}.refined.shp",
    dem_path=dem_path,
    orthophoto_path=orthophoto_path,
    min_diameter=60
)

# Compute morphometry
compute_depth_diameter_ratios(
    input_shapefile=f"{shapefile_base}.refined.shp",
    output_shapefile=f"{shapefile_base}.data.shp",
    dem_path=dem_path,
    orthophoto_path=orthophoto_path
)
```

## Workflow

The analysis pipeline consists of three main steps:

### 1. Prepare Geometries
Converts point crater locations to circular polygons based on diameter.

**Input**: Shapefile with crater points and diameter field (`D_m`)
**Output**: Shapefile with circular geometries

### 2. Refine Crater Rims
Uses topographic data to refine crater rim positions.

**Algorithm**:
- Samples elevation along azimuthal profiles
- Detects rim peaks using signal processing
- Fits optimal circle to rim positions
- Estimates uncertainties

**Input**: Shapefile with crater geometries, DEM, orthophoto
**Output**: Refined shapefile with error estimates

### 3. Analyze Morphometry
Computes depth-diameter ratios and other morphometric parameters.

**Metrics**:
- Crater diameter (with uncertainty)
- Depth (rim height - floor elevation)
- d/D ratio (depth-to-diameter ratio)
- Rim height
- Floor elevation

**Input**: Refined shapefile, DEM, orthophoto
**Output**: Shapefile and CSV with morphometry data

## Module Documentation

### cratools.py

Core algorithms for crater analysis:

- `fit_crater_rim()`: Refines crater rim using topographic profiles
- `remove_external_topography()`: Removes regional slope/tilt
- `compute_E_matrix()`: Creates elevation matrix in polar coordinates
- `compute_depth_diameter_ratio()`: Calculates d/D ratio with uncertainties

### config.py

Configuration management:

- `Config`: Loads and manages JSON configuration
- Methods for retrieving paths and parameters

### refinement.py

Crater rim refinement:

- `prepare_crater_geometries()`: Buffers points to circles
- `update_crater_rims()`: Refines rims using topography

### morphometry.py

Morphometric analysis:

- `compute_depth_diameter_ratios()`: Main morphometry computation
- `export_morphometry_to_csv()`: Exports results to CSV
- `get_morphometry_summary()`: Generates summary statistics

## Testing

Run syntax validation:

```bash
python tests/test_syntax.py
```

Test imports (requires dependencies):

```bash
python tests/test_imports.py
```

## Data Requirements

### Input Data

1. **DEM (Digital Elevation Model)**
   - Format: GeoTIFF (.tif)
   - Coordinate system: Same as shapefile

2. **Orthophoto**
   - Format: GeoTIFF (.tif)
   - Used for visualization and context

3. **Crater Shapefile**
   - Required fields:
     - `geometry`: Point or polygon geometry
     - `D_m`: Crater diameter in meters
     - `UFID`: Unique identifier
     - `davg`: Average depth (optional)
     - `davg_D`: d/D ratio (optional)
     - `rim`: Rim elevation (optional)
     - `fl`: Floor elevation (optional)

### Output Data

1. **Refined Shapefile** (`*.refined.shp`)
   - Refined crater geometries
   - Error estimates (`err_x0`, `err_y0`, `err_r`)

2. **Morphometry Shapefile** (`*.data.shp`)
   - All refined geometry fields
   - Morphometric measurements:
     - `diam`: Refined diameter
     - `d_D`: Depth-diameter ratio
     - `d_D_err`: d/D uncertainty
     - `depth`: Crater depth
     - `depth_err`: Depth uncertainty
     - `rim_JKA`: Rim height
     - `floor_JKA`: Floor elevation

3. **CSV Export** (`*.morphometry.csv`)
   - Tabular format of morphometry data
   - Suitable for statistical analysis

## Algorithm Details

### Rim Detection

1. Sample elevation along radial profiles at multiple azimuths
2. Apply 1D smoothing splines to each profile
3. Detect local maxima (rim peaks) using prominence analysis
4. For profiles without clear peaks, use second derivative minimum
5. Fit optimal circle to detected rim positions
6. Compute fit uncertainties from Jacobian

### Depth Calculation

1. Remove regional topography (optional)
2. Mask DEM to crater geometry
3. Extract perimeter elevations (rim)
4. Find minimum elevation (floor)
5. Compute depth = mean(rim) - floor
6. Propagate uncertainties using uncertainties package

## Citation

If you use this toolkit in your research, please cite:

```
[Your citation information here]
```

## License

[Specify license]

## Contributing

Contributions are welcome! Please submit issues and pull requests.

## Contact

[Your contact information]

## Version History

- **1.0.0** (2025): Initial modular reorganization
  - Separated notebook code into modules
  - Added configuration management
  - Created CLI interface
  - Added comprehensive documentation

# Input Module - Crater Analysis Preprocessing

## Quick Start

Process crater input data in 3 steps:

```bash
# 1. Prepare your data files:
#    - image.tif (contrast image)
#    - dtm.tif (elevation data)
#    - craters.csv (crater locations)

# 2. Run the processor:
python process_crater_inputs.py \
    --image data/image.tif \
    --dtm data/dtm.tif \
    --craters data/craters.csv \
    --output results/

# 3. Check outputs:
#    results/craters_initial.shp - Shapefile
#    results/craters_initial_locations.png - Map
#    results/craters_csfd.png - Size-frequency plot
```

## What It Does

The input module is **Step 0** of the crater analysis workflow. It:

1. ✓ Reads multiple data formats (GeoTIFF, ISIS cubes, CSV, .diam)
2. ✓ Handles lat/lon AND X/Y coordinates automatically
3. ✓ Converts between coordinate systems for lunar projections
4. ✓ Creates georeferenced shapefile output
5. ✓ Generates visualization PNGs
6. ✓ Performs CSFD analysis

## Mandatory Inputs

### 1. Contrast Image

**Format:** GeoTIFF (`.tif`) or ISIS Cube (`.cub`)

**Purpose:** Visual context for crater locations

**Example:** `image.tif`

### 2. DTM (Digital Terrain Model)

**Format:** GeoTIFF (`.tif`)

**Purpose:** Elevation data for morphometry

**Requirements:** Same size and pixel scale as image

**Example:** `dtm.tif`

### 3. Crater Locations

**Format:** CSV or .diam file

**Required Columns:**
- Column 1: Latitude OR X-coordinate
- Column 2: Longitude OR Y-coordinate
- Column 3: Diameter (in meters)

**Example (lat/lon):**
```csv
lat,lon,diameter
-87.031,84.372,169.1
-87.026,84.669,307.7
```

**Example (X/Y):**
```csv
x_m,y_m,diam_m
89601.29,8829.40,169.1
89814.57,8380.60,307.7
```

## Outputs

### 1. Shapefile: `craters_initial.shp`

ESRI shapefile with crater circles (polygons)

**Attributes:**
- lat, lon - Geographic coordinates
- x, y - Projected coordinates
- diameter - Crater diameter in meters

**Use with:**
- QGIS, ArcGIS (GIS software)
- Python (geopandas)
- Next analysis steps

### 2. Location Plot: `craters_initial_locations.png`

Visualization showing:
- Contrast image as background
- Crater circles in red
- Title: "Initial location of craters, N = XX"

### 3. CSFD Plot: `craters_csfd.png`

Crater Size-Frequency Distribution diagram:
- Log-log plot
- Cumulative frequency vs diameter
- Standard planetary science format

## Supported Projections

### Equirectangular (Default)

Simple cylindrical projection, good for global/equatorial regions.

```bash
python process_crater_inputs.py ... --projection equirectangular
```

### Stereographic

Polar stereographic, best for high-latitude regions.

```bash
python process_crater_inputs.py ... \
    --projection stereographic \
    --center-lon 0 --center-lat -90  # South Pole
```

### Orthographic

Hemisphere view, good for visible disk.

```bash
python process_crater_inputs.py ... --projection orthographic
```

## Example Files

The `data/` directory contains example crater files:

- `example_craters_latlon.csv` - Latitude/longitude format
- `example_craters_xy.csv` - X/Y projected coordinates

**Test with:**
```bash
python process_crater_inputs.py \
    --craters data/example_craters_latlon.csv \
    --image your_image.tif \
    --dtm your_dtm.tif \
    --output test_output/
```

## Coordinate System Handling

### Automatic Detection

The module automatically detects coordinate type from headers:

**Lat/Lon indicators:**
- `lat`, `latitude`, `lat_deg`, `lat_d`
- `lon`, `long`, `longitude`, `lon_deg`, `lon_d`

**X/Y indicators:**
- `x`, `x_m`, `x_km`, `easting`
- `y`, `y_m`, `y_km`, `northing`

### Automatic Conversion

**If your file has lat/lon:**
→ Module converts to X/Y for processing

**If your file has X/Y:**
→ Module converts to lat/lon for completeness

**Result:** Shapefile has BOTH coordinate systems!

## Integration with Main Workflow

### Complete Workflow

```bash
# Step 0: Input Processing (NEW)
python process_crater_inputs.py \
    --image data/image.tif \
    --dtm data/dtm.tif \
    --craters data/craters.csv \
    --output results/

# Step 1: Prepare Geometries
python main.py --steps prepare

# Step 2: Refine Rims
python main.py --steps refine

# Step 3: Analyze Morphometry
python main.py --steps analyze
```

### Use Input Module Output

The shapefile from Step 0 can be used directly in Step 1:

```python
from crater_analysis.refinement import prepare_crater_geometries

prepare_crater_geometries(
    input_shapefile='results/craters_initial.shp',
    output_shapefile='results/craters_circles.shp'
)
```

## Common Use Cases

### Case 1: South Polar Craters

```bash
python process_crater_inputs.py \
    --image south_pole_image.tif \
    --dtm south_pole_dtm.tif \
    --craters south_pole_craters.csv \
    --output south_pole_results/ \
    --projection stereographic \
    --center-lon 0 \
    --center-lat -90
```

### Case 2: Equatorial Region

```bash
python process_crater_inputs.py \
    --image equator_image.tif \
    --dtm equator_dtm.tif \
    --craters equator_craters.csv \
    --output equator_results/ \
    --projection equirectangular
```

### Case 3: ISIS Cube Input

```bash
# If you have ISIS cube files
python process_crater_inputs.py \
    --image observation.cub \
    --dtm dtm.tif \
    --craters craters.csv \
    --output results/
```

**Note:** Requires GDAL with ISIS driver. If not available, convert first:
```bash
gdal_translate observation.cub observation.tif
```

## Troubleshooting

### "Could not read ISIS cube"

**Solution:** Convert to GeoTIFF:
```bash
gdal_translate input.cub output.tif
```

### "Coordinate type not clearly identified"

**Solution:** Add clear headers to your CSV file:
```csv
lat,lon,diameter  # Good
x_m,y_m,diam_m   # Good
col1,col2,col3   # Bad
```

### Craters not visible on image

**Solution:** Check coordinate systems match:
```bash
gdalinfo image.tif | grep "Coordinate System"
```

### Wrong diameter units

**Solution:** Specify units in header:
```csv
lat,lon,diam_km  # Will be auto-converted to meters
```

## Python API

Use directly in Python scripts:

```python
from crater_analysis.input_module import process_crater_inputs

results = process_crater_inputs(
    image_path='data/image.tif',
    dtm_path='data/dtm.tif',
    crater_file_path='data/craters.csv',
    output_dir='results/',
    projection='equirectangular'
)

print(f"Shapefile: {results['shapefile']}")
print(f"Total craters: {results['crater_count']}")

# Access data
crater_df = results['crater_data']
print(crater_df.head())
```

## Documentation

- **Full Guide:** [docs/INPUT_MODULE_GUIDE.md](docs/INPUT_MODULE_GUIDE.md)
- **Main README:** [README.md](README.md)
- **Step 1 Guide:** [docs/STEP1_PREPARE_GEOMETRIES_EXPLAINED.md](docs/STEP1_PREPARE_GEOMETRIES_EXPLAINED.md)

## Requirements

- Python 3.7+
- numpy
- pandas
- geopandas
- rasterio
- shapely
- matplotlib

Install with:
```bash
pip install -r requirements.txt
```

## Features

✓ Multiple input formats
✓ Automatic coordinate detection
✓ Lunar projection support
✓ Coordinate conversion
✓ Shapefile creation
✓ Visualization generation
✓ CSFD analysis
✓ Error handling

## Next Steps

After running the input module:

1. **Review outputs:**
   - Check `craters_initial_locations.png`
   - Verify crater positions look correct
   - Examine `craters_csfd.png` for size distribution

2. **Proceed to Step 1:**
   ```bash
   python main.py --steps prepare
   ```

3. **Continue workflow:**
   - Step 2: Refine rims
   - Step 3: Analyze morphometry

## Support

For issues or questions:
1. Check [docs/INPUT_MODULE_GUIDE.md](docs/INPUT_MODULE_GUIDE.md)
2. Review example files in `data/`
3. Run test script: `python tests/test_input_module.py`

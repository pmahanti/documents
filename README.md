# PSR-SDC1 Geodatabase and Visualization Tools

This repository contains Python applications for working with Permanently Shadowed Region (PSR) data and Sentinel-2 Cloud Optimized GeoTIFF (COG) imagery from lunar polar regions.

## Overview

The toolkit consists of three main applications:

1. **create_psr_geodatabase.py** - Creates a compact geodatabase from LOLA PSR shapefiles
2. **extract_cog_footprints.py** - Extracts valid data footprints from COG imagery
3. **visualize_psr_cog.py** - Visualizes and queries spatial relationships between PSRs and COG images

## Data

### PSR Shapefiles
- **Northern Hemisphere**: `LOLA_PSR_75N_120M_82N_060M_1KM2_FINAL.shp` (75°N to 82°N)
- **Southern Hemisphere**: `LOLA_PSR_75S_120M_82S_060M_1KM2_FINAL.shp` (75°S to 82°S)
- **Directory**: `shapefiles_1km2/`

### COG Imagery
- **Format**: Cloud Optimized GeoTIFF (60m resolution)
- **Count**: 100 images
- **Directory**: `SDC60_COG/`
- **Naming**: `M0131XXXXXS.60m.COG.tif`

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies
- geopandas (geospatial data manipulation)
- rasterio (raster data I/O)
- shapely (geometric operations)
- matplotlib (visualization)
- numpy, pandas (data processing)
- tqdm (progress bars)

## Usage

### 1. Create PSR Geodatabase

Creates a single GeoPackage file containing all PSR polygons from both hemispheres.

```bash
python create_psr_geodatabase.py
```

**Output**: `psr_database.gpkg` (~2-3 MB)

**Features**:
- Combines northern and southern hemisphere PSRs
- Adds hemisphere identifier
- Single-file format for easy sharing
- Compatible with Python (geopandas) and MATLAB (readgeotable)

**Python Usage**:
```python
import geopandas as gpd
psr_data = gpd.read_file('psr_database.gpkg', layer='psr_polygons')

# Query by hemisphere
north_psrs = psr_data[psr_data['hemisphere'] == 'North']
south_psrs = psr_data[psr_data['hemisphere'] == 'South']
```

**MATLAB Usage** (R2019a+):
```matlab
psr_data = readgeotable('psr_database.gpkg', 'Layer', 'psr_polygons');

% Query by hemisphere
north_psrs = psr_data(strcmp(psr_data.hemisphere, 'North'), :);
south_psrs = psr_data(strcmp(psr_data.hemisphere, 'South'), :);
```

### 2. Extract COG Footprints

Extracts the true valid data footprint from each COG image and saves to a geodatabase.

```bash
python extract_cog_footprints.py
```

**Output**: `cog_footprints.gpkg` (~1-2 MB)

**Features**:
- Detects valid data pixels (excludes nodata areas)
- Vectorizes footprints as polygons
- Simplifies polygons for compact storage
- Stores metadata (area, valid fraction, resolution)
- Progress bar for batch processing

**Python Usage**:
```python
import geopandas as gpd
cog_footprints = gpd.read_file('cog_footprints.gpkg', layer='cog_footprints')

# Query by filename
footprint = cog_footprints[cog_footprints['filename'] == 'M012728826S.60m.COG.tif']

# Find COGs with high valid data fraction
high_quality = cog_footprints[cog_footprints['valid_fraction'] > 0.9]
```

**MATLAB Usage** (R2019a+):
```matlab
cog_data = readgeotable('cog_footprints.gpkg', 'Layer', 'cog_footprints');

% Query by filename
footprint = cog_data(strcmp(cog_data.filename, 'M012728826S.60m.COG.tif'), :);

% Find COGs with high valid data fraction
high_quality = cog_data(cog_data.valid_fraction > 0.9, :);
```

### 3. Visualize PSR-COG Relationships

Generates maps showing spatial relationships between PSRs and COG imagery.

#### Option A: Visualize a COG footprint on PSR outlines

```bash
python visualize_psr_cog.py --cog M012728826S.60m.COG.tif
```

**Options**:
- `--output FILENAME.png` - Custom output filename
- `--guard-band 2.0` - Guard band in kilometers (default: 1.0 km)

**Output**: PNG image showing:
- COG footprint (red dashed outline)
- Overlapping PSR polygons (blue filled)
- Polar stereographic projection
- Cropped to COG extent + guard band

#### Option B: Find all COGs overlapping a specific PSR

```bash
python visualize_psr_cog.py --psr-id 1234
```

**Output**:
- List of overlapping COG filenames (printed to console)
- PNG image showing:
  - Target PSR (blue filled)
  - All overlapping COG footprints (colored)
  - Nearby PSRs for context (gray)

#### Advanced Options

```bash
# Custom database paths
python visualize_psr_cog.py --cog M012728826S.60m.COG.tif \
    --psr-db custom_psr.gpkg \
    --cog-db custom_cog.gpkg

# Custom output and guard band
python visualize_psr_cog.py --cog M012728826S.60m.COG.tif \
    --output my_visualization.png \
    --guard-band 2.5
```

## Workflow Example

### Complete Pipeline

```bash
# Step 1: Create PSR geodatabase
python create_psr_geodatabase.py

# Step 2: Extract COG footprints
python extract_cog_footprints.py

# Step 3: Visualize a specific COG
python visualize_psr_cog.py --cog M012728826S.60m.COG.tif

# Step 4: Query PSR overlaps
python visualize_psr_cog.py --psr-id 1234
```

## Output Files

| File | Size | Description |
|------|------|-------------|
| `psr_database.gpkg` | ~2-3 MB | PSR polygons from both hemispheres |
| `cog_footprints.gpkg` | ~1-2 MB | Valid data footprints for all COG images |
| `*.png` | ~1-5 MB | Visualization outputs |

## File Format: GeoPackage

GeoPackage (.gpkg) is an open, standards-based, platform-independent, portable, self-describing format for geospatial information.

**Advantages**:
- Single file (easy to share)
- No size limitations
- Full spatial indexing
- Works with Python, MATLAB, QGIS, ArcGIS, etc.
- SQLite-based (queryable with SQL)

## Coordinate Reference System

All data uses the coordinate system from the original LOLA PSR shapefiles (typically a polar stereographic projection appropriate for lunar polar regions).

## Performance Notes

- **PSR geodatabase creation**: ~5-10 seconds
- **COG footprint extraction**: ~2-5 minutes for 100 images
- **Visualization**: ~5-15 seconds per map

## Troubleshooting

### Missing Dependencies
```bash
# If you encounter import errors, reinstall dependencies
pip install --upgrade -r requirements.txt
```

### File Not Found Errors
- Ensure `shapefiles_1km2/` directory contains the PSR shapefiles
- Ensure `SDC60_COG/` directory contains the COG images
- Run scripts from the repository root directory

### Memory Issues
For large datasets, you may need to increase available memory or process in batches.

### MATLAB Compatibility
GeoPackage support requires MATLAB R2019a or later. For older versions, export to Shapefile:

```python
import geopandas as gpd
psr_data = gpd.read_file('psr_database.gpkg', layer='psr_polygons')
psr_data.to_file('psr_data.shp')
```

## Technical Details

### PSR Database Schema
- `geometry`: Polygon geometry
- `hemisphere`: 'North' or 'South'
- Additional fields from original shapefiles (PSR ID, area, etc.)

### COG Footprints Database Schema
- `footprint_id`: Unique identifier
- `filename`: COG filename
- `geometry`: Footprint polygon
- `area_km2`: Footprint area in square kilometers
- `valid_fraction`: Fraction of pixels with valid data
- `width`, `height`: Image dimensions in pixels
- `resolution_m`: Pixel size in meters
- `bounds`: Bounding box coordinates
- `filepath`: Absolute path to COG file

## Citation

If you use this software in your research, please cite:
- LOLA PSR data source
- Sentinel-2 SDC1 data source

## License

[Specify license here]

## Contact

[Your contact information]

## Version History

- **v1.0** (2025-11-21): Initial release
  - PSR geodatabase creation
  - COG footprint extraction
  - Spatial visualization and queries

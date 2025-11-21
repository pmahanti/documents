# Visualization Examples

This document shows example outputs from the PSR-SDC1 visualization tools with polar stereographic projection and lat/lon grids.

## Example 1: South Pole - COG Footprint on PSRs

**Files:**
- `example1_south_pole.png` (554 KB)
- `example1_south_pole.tif` (674 KB) - Georeferenced GeoTIFF

**Command:**
```bash
python visualize_psr_cog.py --cog M012728826S.60m.COG.tif --output example1_south_pole
```

**Description:**
- COG file: M012728826S.60m.COG.tif
- Hemisphere: Southern
- Footprint area: 850.41 km²
- Valid data fraction: 6.8%
- Image dimensions: 1992 x 1731 pixels
- PSRs in region: 2,384 (all south pole PSRs)

**Features:**
- Full south pole view (70-90°S)
- Latitude gridlines: 75°, 80°, 85°, 90° (5° spacing)
- Longitude gridlines: 0°, 30°, 60°, 90°, 120°, 150°, 180°, 210°, 240°, 270°, 300°, 330° (30° spacing)
- Red dashed line shows COG footprint
- Blue filled polygons show all PSRs in the south polar region

---

## Example 2: South Pole - PSR Overlap Query

**Files:**
- `example2_psr_overlap.png` (517 KB)
- `example2_psr_overlap.tif` (633 KB) - Georeferenced GeoTIFF

**Command:**
```bash
python visualize_psr_cog.py --psr-id SP_816480_0652210 --output example2_psr_overlap
```

**Description:**
- PSR ID: SP_816480_0652210
- Hemisphere: Southern
- PSR area: 79.788 km²
- Location: 81.648°S, 65.222°E
- Overlapping COG images: 2
  - M013121460S.60m.COG.tif
  - M013128596S.60m.COG.tif

**Features:**
- Full south pole view (70-90°S)
- Target PSR highlighted in blue
- Context PSRs shown in gray
- Two overlapping COG footprints in different colors
- Complete lat/lon grid overlay

---

## Example 3: North Pole - COG Footprint on PSRs

**Files:**
- `example3_north_pole.png` (554 KB)
- `example3_north_pole.tif` (687 KB) - Georeferenced GeoTIFF

**Command:**
```bash
python visualize_psr_cog.py --cog M013049982S.60m.COG.tif --output example3_north_pole
```

**Description:**
- COG file: M013049982S.60m.COG.tif
- Hemisphere: Northern
- Footprint area: 870.45 km²
- Valid data fraction: 20.2%
- Image dimensions: 2569 x 462 pixels
- PSRs in region: 5,655 (all north pole PSRs)

**Features:**
- Full north pole view (70-90°N)
- Latitude gridlines: 75°, 80°, 85°, 90° (5° spacing)
- Longitude gridlines: 0°, 30°, 60°, 90°, 120°, 150°, 180°, 210°, 240°, 270°, 300°, 330° (30° spacing)
- Red dashed line shows COG footprint
- Blue filled polygons show all PSRs in the north polar region

---

## Map Features

### Polar Stereographic Projection

All visualizations use native polar stereographic projections:
- **Northern hemisphere:** North Pole Stereographic (latitude_of_origin = 90°)
- **Southern hemisphere:** South Pole Stereographic (latitude_of_origin = -90°)

### Coverage Area

Maps show **20 degrees from each pole** (70-90° latitude range):
- Covers the primary PSR distribution zones
- Includes all permanently shadowed regions
- Extent: ~600 km from pole center

### Grid System

**Latitude circles** (parallels):
- Spacing: 5 degrees
- North: 75°N, 80°N, 85°N, 90°N
- South: 75°S, 80°S, 85°S, 90°S
- Rendered as dotted gray circles

**Longitude meridians**:
- Spacing: 30 degrees
- Values: 0°, 30°, 60°, 90°, 120°, 150°, 180°, 210°, 240°, 270°, 300°, 330°
- Rendered as dotted gray lines radiating from pole

### Color Coding

**COG Footprint Visualizations (Examples 1, 3):**
- **Red dashed line:** COG image footprint boundary
- **Light blue filled:** All PSR polygons in polar region
- **Dark blue outline:** PSR polygon boundaries
- **Gray dotted lines:** Lat/lon grid

**PSR Overlap Visualizations (Example 2):**
- **Light blue filled:** Target PSR polygon
- **Light gray filled:** Other PSRs in the region (context)
- **Colored filled (red outline):** Overlapping COG footprints
- **Gray dotted lines:** Lat/lon grid

### Output Formats

Each visualization produces two files:

1. **PNG (Portable Network Graphics)**
   - Resolution: 150 DPI
   - Size: ~500-550 KB
   - Use: Display, presentations, documentation
   - Non-georeferenced raster image

2. **GeoTIFF (Georeferenced TIFF)**
   - Resolution: Matches PNG (150 DPI)
   - Size: ~630-690 KB
   - Use: GIS applications, spatial analysis
   - Includes:
     - Coordinate Reference System (CRS)
     - Geospatial transform
     - Proper georeferencing for pole
   - Compression: LZW
   - Compatible with: QGIS, ArcGIS, ENVI, GDAL

---

## Using GeoTIFF Files in GIS Software

### QGIS
```
1. Open QGIS
2. Layer → Add Layer → Add Raster Layer
3. Browse to example*.tif
4. CRS will be automatically detected as Polar Stereographic
```

### ArcGIS/ArcGIS Pro
```
1. Add Data → Raster Dataset
2. Browse to example*.tif
3. Projection is embedded in file
```

### Python (rasterio)
```python
import rasterio
from rasterio.plot import show

# Read georeferenced map
with rasterio.open('example1_south_pole.tif') as src:
    print(f"CRS: {src.crs}")
    print(f"Bounds: {src.bounds}")
    print(f"Transform: {src.transform}")
    show(src)
```

### GDAL Command Line
```bash
# Get geospatial info
gdalinfo example1_south_pole.tif

# Reproject to another CRS
gdalwarp -t_srs EPSG:4326 example1_south_pole.tif output_latlon.tif
```

---

## Generating Custom Visualizations

### List Available COG Files
```bash
ls SDC60_COG/*.tif
```

### List Available PSR IDs
```python
import geopandas as gpd

# Northern PSRs
psr_north = gpd.read_file('psr_database.gpkg', layer='psr_north')
print(psr_north[['PSR_ID', 'latitude', 'longitude', 'area']].head(10))

# Southern PSRs
psr_south = gpd.read_file('psr_database.gpkg', layer='psr_south')
print(psr_south[['PSR_ID', 'latitude', 'longitude', 'area']].head(10))
```

### Customize Visualizations

**COG visualization:**
```bash
python visualize_psr_cog.py --cog FILENAME.tif --output my_map
```

**PSR overlap query:**
```bash
python visualize_psr_cog.py --psr-id PSR_ID --output my_psr_analysis
```

**Use custom geodatabase paths:**
```bash
python visualize_psr_cog.py --cog FILENAME.tif \
    --psr-db /path/to/psr_database.gpkg \
    --cog-db /path/to/cog_footprints.gpkg \
    --output custom_output
```

---

## Performance Notes

- **COG footprint visualizations:** ~10-15 seconds
- **PSR overlap visualizations:** ~10-20 seconds (depends on number of COGs)
- **PNG file size:** ~500-550 KB (150 dpi)
- **GeoTIFF file size:** ~630-690 KB (LZW compressed)

---

## Coordinate System Details

### North Pole Stereographic
```
PROJCS["Stereographic_North_Pole",
  GEOGCS["GCS_Moon",
    DATUM["D_Moon",
      SPHEROID["Moon",1737400,0]],
    PRIMEM["Greenwich",0],
    UNIT["Degree",0.0174532925199433]],
  PROJECTION["Polar_Stereographic"],
  PARAMETER["latitude_of_origin",90],
  PARAMETER["central_meridian",0],
  PARAMETER["false_easting",0],
  PARAMETER["false_northing",0],
  UNIT["metre",1]]
```

### South Pole Stereographic
```
PROJCS["Stereographic_South_Pole",
  GEOGCS["GCS_Moon",
    DATUM["D_Moon",
      SPHEROID["Moon",1737400,0]],
    PRIMEM["Greenwich",0],
    UNIT["Degree",0.0174532925199433]],
  PROJECTION["Polar_Stereographic"],
  PARAMETER["latitude_of_origin",-90],
  PARAMETER["central_meridian",0],
  PARAMETER["false_easting",0],
  PARAMETER["false_northing",0],
  UNIT["metre",1]]
```

**Key parameters:**
- Spheroid: Moon (radius = 1,737,400 m)
- Projection: Polar Stereographic
- Origin: North or South pole
- Units: Meters

---

## Troubleshooting

### "PSR with ID X not found"
Ensure you're using the correct PSR_ID format:
- Northern: `NP_XXXXXX_YYYYYYY`
- Southern: `SP_XXXXXX_YYYYYYY`

Use the Python snippet above to list available PSR IDs.

### "COG file not found in database"
Run `python extract_cog_footprints.py` first to create the footprints database.

### Grid lines not visible
Grid lines are subtle (gray dotted lines). They may be less visible in regions with many PSRs. Zoom in to see them more clearly.

### GeoTIFF not opening correctly
Ensure your GIS software supports:
- Custom CRS definitions (Moon spheroid)
- Polar stereographic projections
- GDAL/PROJ version 6.0 or higher

### Large file sizes
GeoTIFF files are ~630-690 KB due to high resolution (150 DPI). To reduce:
- Lower DPI in code (change `dpi=150` to `dpi=100`)
- Use different compression (change `compress='lzw'` to `compress='deflate'`)

---

## Citation

If you use these visualizations in publications, please cite:
- LOLA PSR data source
- Sentinel-2 SDC1 data source
- This visualization toolkit

---

## Version

- **Version:** 2.0
- **Date:** 2025-11-21
- **Features:** Polar stereographic projection, lat/lon grids, dual PNG/GeoTIFF output

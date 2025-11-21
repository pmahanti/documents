# Visualization Examples

This document shows example outputs from the PSR-SDC1 visualization tools with polar stereographic projection, lat/lon grids, and night mode theme.

## Example 1: South Pole - COG Footprint on PSRs

**Files:**
- `example1_south_pole.png` (491 KB)
- `example1_south_pole.tif` (642 KB) - Georeferenced GeoTIFF

**Command:**
```bash
python visualize_psr_cog.py --cog M013067672S.60m.COG.tif --output example1_south_pole
```

**Description:**
- COG file: M013067672S.60m.COG.tif
- Hemisphere: Southern
- Footprint area: 1,049.46 km²
- Valid data fraction: 15.9%
- Image dimensions: 2811 x 642 pixels
- PSRs in region: 2,362 (within 15° of south pole)

**Features:**
- South pole view (75-90°S) - 15 degrees from pole
- Latitude gridlines: 75°, 80°, 85°, 90° (5° spacing)
- Longitude gridlines: 0°, 30°, 60°, 90°, 120°, 150°, 180°, 210°, 240°, 270°, 300°, 330° (30° spacing)
- **Night mode theme:** Dark background with light elements
- **Thin outlines:** Coral red continuous line (1.5px) for COG footprint
- Dark blue-gray filled PSR polygons with light blue thin outlines (0.3px)
- **Axes in kilometers:** X/Y axes show distance in km from pole center

---

## Example 2: North Pole - COG Footprint on PSRs

**Files:**
- `example2_north_pole.png` (524 KB)
- `example2_north_pole.tif` (658 KB) - Georeferenced GeoTIFF

**Command:**
```bash
python visualize_psr_cog.py --cog M013057217S.60m.COG.tif --output example2_north_pole
```

**Description:**
- COG file: M013057217S.60m.COG.tif
- Hemisphere: Northern
- Footprint area: 858.75 km²
- Valid data fraction: 18.0%
- Image dimensions: 2564 x 509 pixels
- PSRs in region: 5,625 (within 15° of north pole)

**Features:**
- North pole view (75-90°N) - 15 degrees from pole
- Latitude gridlines: 75°, 80°, 85°, 90° (5° spacing)
- Longitude gridlines: 0°, 30°, 60°, 90°, 120°, 150°, 180°, 210°, 240°, 270°, 300°, 330° (30° spacing)
- **Night mode theme:** Dark background with light elements
- **Thin outlines:** Coral red continuous line (1.5px) for COG footprint
- Dark blue-gray filled PSR polygons with light blue thin outlines (0.3px)
- **Axes in kilometers:** X/Y axes show distance in km from pole center

---

## Example 3: South Pole - PSR Overlap Query

**Files:**
- `example3_psr_overlap.png` (464 KB) - Full polar view
- `example3_psr_overlap.tif` (607 KB) - Georeferenced GeoTIFF (full view)
- `example3_psr_overlap_zoom.png` (227 KB) - Zoomed view of overlap region
- `example3_psr_overlap_zoom.tif` (498 KB) - Georeferenced GeoTIFF (zoomed)

**Command:**
```bash
python visualize_psr_cog.py --psr-id SP_845460_0695020 --output example3_psr_overlap
```

**Description:**
- PSR ID: SP_845460_0695020
- Hemisphere: Southern
- PSR area: 6.484 km²
- Location: 84.546°S, 69.502°E
- Overlapping COG images: 2
  - M013078591S.60m.COG.tif
  - M013085783S.60m.COG.tif

**Features:**
- South pole view (75-90°S) - 15 degrees from pole
- **Night mode theme:** Dark background with light elements
- **Bright cyan target PSR:** Target PSR highlighted in dark blue-gray fill with bright cyan outline (2.0px, #00ffff)
- Context PSRs shown in very dark gray with dark outlines (0.2px)
- Two overlapping COG footprints in different colors with thin coral red continuous outlines (0.8px)
- Complete lat/lon grid overlay
- **Axes in kilometers:** X/Y axes show distance in km from pole center
- **Zoomed view included:** Additional visualization focused on overlap region with 10% buffer

---

## Map Features

### Polar Stereographic Projection

All visualizations use native polar stereographic projections:
- **Northern hemisphere:** North Pole Stereographic (latitude_of_origin = 90°)
- **Southern hemisphere:** South Pole Stereographic (latitude_of_origin = -90°)

### Coverage Area

Maps show **15 degrees from each pole** (75-90° latitude range):
- Focuses on the densest PSR concentration zones
- Includes core permanently shadowed regions
- Extent: ~450 km from pole center

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

### Color Coding (Night Mode Theme)

**COG Footprint Visualizations (Examples 1, 2):**
- **Black background:** Night mode theme for reduced eye strain
- **Coral red continuous line (1.5px):** COG image footprint boundary (#ff6b6b)
- **Dark blue-gray filled (#1a3a4a):** PSR polygons in polar region
- **Light blue thin outlines (0.3px):** PSR polygon boundaries (#4a90c0)
- **Gray dotted lines (0.3px):** Lat/lon grid (#808080)
- **White text:** Labels, title, axis text
- **Axes in kilometers:** Distance from pole center in km

**PSR Overlap Visualizations (Example 3):**
- **Black background:** Night mode theme
- **Dark blue-gray filled (#1a3a4a):** Target PSR polygon
- **Bright cyan outline (2.0px):** Target PSR boundary for high visibility (#00ffff)
- **Very dark gray filled (#0a0a0a):** Other PSRs in the region (context)
- **Dark gray outlines (0.2px):** Context PSR boundaries (#303030)
- **Set3 colormap:** Overlapping COG footprints (various colors)
- **Coral red continuous outlines (0.8px):** COG footprint boundaries (#ff6b6b)
- **Gray dotted lines (0.3px):** Lat/lon grid (#808080)
- **White text:** Labels, title, axis text
- **Axes in kilometers:** Distance from pole center in km
- **Dual views:** Full polar view (15° extent) + zoomed view (overlap region with 10% buffer)

### Output Formats

Each visualization produces two files (or four for PSR overlap queries):

1. **PNG (Portable Network Graphics)**
   - Resolution: 150 DPI
   - Size: ~500-550 KB (full view), ~290 KB (zoomed view)
   - Use: Display, presentations, documentation
   - Non-georeferenced raster image

2. **GeoTIFF (Georeferenced TIFF)**
   - Resolution: Matches PNG (150 DPI)
   - Size: ~590-690 KB
   - Use: GIS applications, spatial analysis
   - Includes:
     - Coordinate Reference System (CRS)
     - Geospatial transform
     - Proper georeferencing for pole
   - Compression: LZW
   - Compatible with: QGIS, ArcGIS, ENVI, GDAL

3. **PSR Overlap Queries Only: Zoomed View Files**
   - Additional `_zoom.png` and `_zoom.tif` files
   - Focused on overlap region with 10% buffer
   - Shows target PSR and all overlapping COG footprints in detail
   - Same format and quality as full view files

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
- **PSR overlap visualizations:** ~15-25 seconds (includes zoomed view generation)
- **PNG file size:** ~500-550 KB (full view, 150 dpi), ~290 KB (zoomed view)
- **GeoTIFF file size:** ~590-690 KB (LZW compressed)

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

- **Version:** 4.0
- **Date:** 2025-11-21
- **Features:**
  - Polar stereographic projection (15° from pole)
  - Lat/lon grids (5° latitude, 30° longitude)
  - Dual PNG/GeoTIFF output
  - **Night mode theme** with dark backgrounds
  - **Thin continuous outlines** for clean visualization (0.2-2.0px)
  - **Bright cyan target PSR outline** (#00ffff) for high visibility in overlap queries
  - **Axes in kilometers** for easier distance interpretation
  - **Zoomed views** for PSR overlap queries showing detailed overlap regions
  - All visualizations show all PSRs in polar region for complete context
  - Optimized for PSR-dense polar regions

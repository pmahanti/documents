# Visualization Examples

This document shows example outputs from the PSR-SDC1 visualization tools.

## Example 1: COG Footprint (Northern Hemisphere, No PSR Overlap)

**File:** `example1_cog_on_psr.png`

**Command:**
```bash
python visualize_psr_cog.py --cog M013049982S.60m.COG.tif --output example1_cog_on_psr.png
```

**Description:**
- COG file: M013049982S.60m.COG.tif
- Hemisphere: Northern
- Footprint area: 870.45 km²
- Valid data fraction: 20.2%
- Image dimensions: 2569 x 462 pixels
- Overlapping PSRs: 0

This example shows a northern hemisphere COG image that does not overlap with any PSRs. The red dashed line shows the COG footprint boundary in polar stereographic projection with a 1 km guard band.

---

## Example 2: COG Footprint (Southern Hemisphere, High PSR Density)

**File:** `example2_cog_footprint_south.png`

**Command:**
```bash
python visualize_psr_cog.py --cog M012728826S.60m.COG.tif --output example2_cog_footprint_south.png --guard-band 2.0
```

**Description:**
- COG file: M012728826S.60m.COG.tif
- Hemisphere: Southern
- Footprint area: 850.41 km²
- Valid data fraction: 6.8%
- Image dimensions: 1992 x 1731 pixels
- Overlapping PSRs: 98
- Guard band: 2.0 km

This example demonstrates a southern hemisphere COG image overlapping many PSR regions (98 PSRs). The blue filled polygons represent PSRs within the extended view. This COG covers a high-density PSR area near the lunar south pole. The 2 km guard band provides additional context around the image extent.

---

## Example 3: PSR Overlap Query

**File:** `example3_psr_overlap.png`

**Command:**
```bash
python visualize_psr_cog.py --psr-id SP_816480_0652210 --output example3_psr_overlap.png
```

**Description:**
- PSR ID: SP_816480_0652210
- Hemisphere: Southern
- PSR area: 79.788 km²
- Location: 81.648°S, 65.222°E
- Overlapping COG images: 2
  - M013121460S.60m.COG.tif
  - M013128596S.60m.COG.tif

This example shows the reverse query: finding all COG images that overlap with a specific PSR. The target PSR is highlighted in blue, with context PSRs shown in gray. The two overlapping COG footprints are shown in different colors (from the Set3 colormap), demonstrating partial coverage of the PSR by different satellite passes.

---

## Interpreting the Visualizations

### Color Coding

**COG Footprint Visualizations (Examples 1-2):**
- **Red dashed line:** COG image footprint boundary
- **Light blue filled:** PSR polygons overlapping with COG extent
- **Dark blue outline:** PSR polygon boundaries

**PSR Overlap Visualizations (Example 3):**
- **Light blue filled:** Target PSR polygon
- **Light gray filled:** Other PSRs in the area (context)
- **Colored filled (red outline):** Overlapping COG footprints

### Projection

All maps use polar stereographic projection appropriate for the hemisphere:
- **Northern hemisphere:** North Pole Stereographic (latitude_of_origin = 90°)
- **Southern hemisphere:** South Pole Stereographic (latitude_of_origin = -90°)

### Coordinate System

The coordinate axes show:
- **Easting (m):** Horizontal distance in meters from the projection origin
- **Northing (m):** Vertical distance in meters from the projection origin

### Statistics Box

Each visualization includes a statistics box showing:
- **COG visualizations:** Area, valid data percentage, overlapping PSR count, guard band
- **PSR visualizations:** Number of overlapping COG images

---

## Generating Your Own Visualizations

### List Available COG Files
```bash
ls SDC60_COG/*.tif
```

### List Available PSR IDs
```python
import geopandas as gpd

# Northern PSRs
psr_north = gpd.read_file('psr_database.gpkg', layer='psr_north')
print(psr_north['PSR_ID'].tolist())

# Southern PSRs
psr_south = gpd.read_file('psr_database.gpkg', layer='psr_south')
print(psr_south['PSR_ID'].tolist())
```

### Customize Visualizations

**Adjust guard band:**
```bash
python visualize_psr_cog.py --cog FILENAME.tif --guard-band 5.0  # 5 km guard band
```

**Custom output filename:**
```bash
python visualize_psr_cog.py --cog FILENAME.tif --output my_custom_map.png
```

**Use custom geodatabase paths:**
```bash
python visualize_psr_cog.py --cog FILENAME.tif \
    --psr-db /path/to/psr_database.gpkg \
    --cog-db /path/to/cog_footprints.gpkg
```

---

## Performance Notes

- **COG footprint visualizations:** ~5-10 seconds
- **PSR overlap visualizations:** ~5-15 seconds (depends on number of COGs)
- **Output file size:** ~200-400 KB per PNG (150 dpi)

---

## Troubleshooting

### No PSRs shown in visualization
This is normal if the COG image is located away from PSR regions. PSRs are concentrated near the lunar poles (>75° latitude).

### "PSR with ID X not found"
Ensure you're using the correct PSR_ID format:
- Northern: `NP_XXXXXX_YYYYYYY`
- Southern: `SP_XXXXXX_YYYYYYY`

Use the Python snippet above to list available PSR IDs.

### "COG file not found in database"
Run `python extract_cog_footprints.py` first to create the footprints database.

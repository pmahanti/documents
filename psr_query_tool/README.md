# Lunar Permanently Shadowed Regions (PSR) Query Tool

This tool queries Permanently Shadowed Regions on the Moon within a specified radius from a given location.

## Data Download Instructions

Since automated downloads are blocked, please manually download the LPSR shapefile components:

1. Visit: https://pgda.gsfc.nasa.gov/products/90
2. Download all LPSR_80S_20MPP_ADJ files (approximately 6 files):
   - LPSR_80S_20MPP_ADJ.shp (shapefile - main geometry)
   - LPSR_80S_20MPP_ADJ.shx (shape index)
   - LPSR_80S_20MPP_ADJ.dbf (database file)
   - LPSR_80S_20MPP_ADJ.prj (projection file)
   - LPSR_80S_20MPP_ADJ.csv (CSV version - optional)
   - LPSR_80S_20MPP_ADJ.cpg (codepage file, if available)

3. Place all files in the `data/` directory

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Query PSRs within 50km of a location (South Pole example)
python psr_query.py --lat -89.5 --lon 45.0 --radius 50

# Convert shapefile to GeoParquet (space-efficient format)
python psr_query.py --convert --input data/LPSR_80S_20MPP_ADJ.shp

# Query using converted format
python psr_query.py --lat -89.5 --lon 45.0 --radius 50 --use-parquet

# Export results to file
python psr_query.py --lat -89.5 --lon 45.0 --radius 50 --output results.geojson

# Create PNG map visualization
python psr_query.py --lat -89.5 --lon 45.0 --radius 50 --map-png psr_map.png

# Combine data export and visualization
python psr_query.py --lat -89.5 --lon 45.0 --radius 50 --output results.geojson --map-png map.png
```

## Example Map

Try the example script to see the visualization in action (uses synthetic data):

```bash
python create_example_map.py
```

This generates several example maps showing PSRs near various lunar south pole locations.

## Output

The tool provides multiple output formats:

**Console Output:**
- Number of PSRs found within the radius
- Distance of each PSR from query location
- PSR attributes (name, confidence, area, type)

**Data Export:**
- GeoJSON (for web mapping and GIS)
- CSV (for spreadsheet analysis, includes centroid coordinates)
- Shapefile (for GIS software)

**Visualization:**
- PNG maps with customizable resolution (DPI)
- Shows query location, search radius, and found PSRs
- Dark space-themed background
- Statistics overlay with nearest/farthest distances
- Color-coded PSRs (blue for matches, gray for others)

## Features

- **Accurate Distance Calculation**: Uses Haversine formula for great-circle distances on lunar surface
- **Space Efficiency**: GeoParquet format reduces file size by 50-80% compared to shapefiles
- **Flexible Queries**: Search any lat/lon with custom radius
- **Multiple Export Formats**: GeoJSON, CSV, Shapefile, PNG
- **Professional Visualizations**: Publication-ready maps with statistics

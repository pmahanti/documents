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
```

## Output

The tool will display:
- Number of PSRs found within the radius
- Details of each PSR (name, distance, area, etc.)
- Optional: Export results to GeoJSON or CSV

## Space Efficiency

The GeoParquet format can reduce file size by 50-80% compared to shapefiles while maintaining full spatial functionality.

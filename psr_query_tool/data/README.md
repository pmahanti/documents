# PSR Data Directory

Place your downloaded LPSR shapefile components here.

## Required Files

Download these files from https://pgda.gsfc.nasa.gov/products/90:

- LPSR_80S_20MPP_ADJ.shp
- LPSR_80S_20MPP_ADJ.shx
- LPSR_80S_20MPP_ADJ.dbf
- LPSR_80S_20MPP_ADJ.prj
- LPSR_80S_20MPP_ADJ.cpg (if available)
- LPSR_80S_20MPP_ADJ.csv (optional)

## File Descriptions

- `.shp` - Main shapefile containing geometry data
- `.shx` - Shape index file for quick access
- `.dbf` - Database file with attribute data
- `.prj` - Projection/coordinate system information
- `.cpg` - Codepage file for character encoding
- `.csv` - CSV version of the data (alternative format)

## After Download

Once files are in this directory, you can:

1. Convert to space-efficient format:
   ```bash
   python ../psr_query.py --convert --input LPSR_80S_20MPP_ADJ.shp
   ```

2. Query PSRs:
   ```bash
   python ../psr_query.py --lat -89.5 --lon 45.0 --radius 50
   ```

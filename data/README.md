# Data Directory

This directory contains sample data extracted from the Jupyter notebook and documentation about the data used by the crater analysis toolkit.

## Files

### Documentation

**DATA_SUMMARY.md**
- Complete overview of data used in the notebook
- File format specifications
- Field descriptions
- Workflow data flow
- Data source information

### Sample Data

**sample_crater_data.csv**
- 19 sample craters from "test" region
- CSV format for easy viewing in Excel/spreadsheets
- Contains: positions, diameters, depths, d/D ratios

**sample_crater_data.json**
- Same 19 craters in JSON format
- Includes metadata and field descriptions
- Easy to use in Python/JavaScript applications

## Dataset Details

### Sample Data Overview

- **Region**: test (tmp.test_craters)
- **Location**: Lunar South Pole (~87°S)
- **Number of craters**: 19
- **Diameter range**: 60.1 - 366.8 meters
- **Depth range**: 1.2 - 21.5 meters
- **d/D ratio range**: 0.012 - 0.059

### Key Fields

| Field | Description |
|-------|-------------|
| UFID | Unique crater identifier |
| lat, lon | Geographic coordinates |
| D_m | Diameter in meters (primary) |
| x_m, y_m | Projected coordinates (meters) |
| davg | Average depth (meters) |
| davg_D | Depth-to-diameter ratio |

## Using the Sample Data

### In Python

```python
import pandas as pd
import json

# Load CSV
df = pd.read_csv('data/sample_crater_data.csv')
print(df.head())

# Load JSON
with open('data/sample_crater_data.json', 'r') as f:
    data = json.load(f)
    craters = data['craters']
    print(f"Found {len(craters)} craters")
```

### In Excel/Google Sheets

Simply open `sample_crater_data.csv` directly.

### Quick Statistics

```python
import pandas as pd

df = pd.read_csv('data/sample_crater_data.csv')

print("Diameter Statistics (m):")
print(df['D_m'].describe())

print("\nd/D Ratio Statistics:")
print(df['davg_D'].describe())
```

Expected output:
```
Diameter Statistics (m):
count     19.000000
mean     141.757416
std       95.629823
min       60.130971
max      366.794230

d/D Ratio Statistics:
count    19.000000
mean      0.036258
std       0.012989
min       0.018933
max       0.058605
```

## Data Source

- **Mission**: Lunar Reconnaissance Orbiter (LRO)
- **Instrument**: SHADOWCAM
- **Data Type**: High-resolution DEM and orthophotos
- **Institution**: Arizona State University
- **Project**: South Lunar Crater (SLC) analysis

## Full Dataset

The sample data here is a subset used in the notebook. For the complete crater analysis workflow, you would need:

1. **DEM Files**: Digital Elevation Models (GeoTIFF format)
2. **Orthophoto Files**: Imagery for context (GeoTIFF format)
3. **Shapefile**: Complete crater catalog with geometries

These files are referenced in `config/regions.json` but are not included in this repository due to size.

## Creating Your Own Data

To use the toolkit with your own crater data, create a shapefile with:

**Required Fields:**
- `geometry`: Point or Polygon
- `D_m`: Diameter in meters (float)
- `UFID`: Unique identifier (string)

**Optional Fields** (for comparison):
- `davg`: Average depth (float)
- `davg_D`: Depth/diameter ratio (float)
- `rim`: Rim elevation (float)
- `fl`: Floor elevation (float)

Then configure the paths in `config/regions.json` to point to your DEM and orthophoto files.

## Data License

[Specify license for the data - typically NASA LRO data is public domain]

## References

For more information about the SHADOWCAM instrument and LRO mission:
- https://www.nasa.gov/mission_pages/LRO/
- https://www.shadowcam.asu.edu/ (if applicable)

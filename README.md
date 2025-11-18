# Crater Marker Tool

A Python application for marking and measuring craters on GeoTIFF and ISIS .cub files.

## Features

- Load GeoTIFF or ISIS .cub files
- Mark craters by selecting 3 points on the rim
- Automatically fit and display circles representing craters
- Delete previous crater marks
- Auto-save functionality after each crater identification
- Export crater data to:
  - .diam file (crater center and diameter)
  - Shapefile format
- Projection independent

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python crater_marker.py
```

## Workflow

1. Click "Open Image" to load a GeoTIFF or ISIS .cub file
2. Click on 3 points on a crater rim
3. The circle will automatically appear
4. Data is auto-saved after each crater
5. Use "Delete Last Crater" to remove the most recent crater
6. Use "Export" to save data as .diam and shapefile

## Output Formats

### .diam File
Tab-delimited text file with columns:
- Crater ID
- Center X coordinate
- Center Y coordinate
- Diameter (in map units)
- CRS information

### Shapefile
Standard ESRI shapefile with crater polygons and attributes.

## Reference

Based on concepts from [OpenCraterTool](https://github.com/thomasheyer/OpenCraterTool)

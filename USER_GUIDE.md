# Crater Marker Tool - User Guide

## Overview

The Crater Marker Tool is a Python application designed for planetary scientists and researchers to mark and measure impact craters on GeoTIFF and ISIS cube (.cub) files. The tool provides a simple graphical interface for crater identification and exports data in standard formats for further analysis.

## Features

### Core Functionality
- **Multi-format Support**: Load GeoTIFF (.tif, .tiff) and ISIS cube (.cub) files
- **3-Point Crater Marking**: Define craters by clicking 3 points on the rim
- **Automatic Circle Fitting**: Algorithm automatically calculates best-fit circle
- **Visual Feedback**: See marked craters overlaid on the image
- **Projection Independent**: Works with any coordinate reference system

### Data Management
- **Auto-save**: Automatically saves progress after each crater
- **Delete Function**: Remove the most recent crater if needed
- **Session Recovery**: Resume previous work from auto-save

### Export Options
- **.diam Format**: Tab-delimited text file with crater centers and diameters
- **Shapefile**: ESRI shapefile format with crater polygons and attributes

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Installing GDAL can be challenging on some systems. If you encounter issues:

- **Ubuntu/Debian**:
  ```bash
  sudo apt-get install gdal-bin libgdal-dev
  pip install GDAL==$(gdal-config --version)
  ```

- **macOS** (with Homebrew):
  ```bash
  brew install gdal
  pip install GDAL==$(gdal-config --version)
  ```

- **Windows**:
  Download precompiled wheels from https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal

## Usage

### Starting the Application

```bash
python crater_marker.py
```

### Basic Workflow

1. **Open an Image**
   - Click "Open Image" button
   - Select a GeoTIFF or ISIS .cub file
   - The image will be displayed in the main window

2. **Mark Craters**
   - Click on 3 points around a crater rim
   - Points should be roughly evenly spaced
   - The fitted circle will appear automatically after the 3rd point
   - The crater is auto-saved immediately

3. **Delete Mistakes**
   - Click "Delete Last Crater" to remove the most recent crater
   - You can only delete one crater at a time (most recent first)

4. **Export Data**
   - Click "Export" button
   - Select a directory for output files
   - Two files will be created:
     - `<image_name>.diam` - Crater measurements
     - `<image_name>_craters.shp` - Shapefile with polygons

### Tips for Best Results

- **Point Selection**: Try to select points that are evenly distributed around the crater rim
- **Avoid Collinear Points**: Don't select 3 points in a straight line
- **Zoom**: Use your mouse to zoom in for more precise point selection
- **Save Often**: The tool auto-saves, but you can export periodically for backup

## File Formats

### .diam File Format

Tab-delimited text file with the following structure:

```
# Crater Marker Tool Export
# Image: /path/to/image.tif
# CRS: +proj=longlat +datum=WGS84 ...
# Columns: ID    Center_X    Center_Y    Diameter
1    250.5    750.3    160.2
2    600.1    400.8    240.5
...
```

**Columns:**
- `ID`: Unique crater identifier
- `Center_X`: X coordinate of crater center (in map units)
- `Center_Y`: Y coordinate of crater center (in map units)
- `Diameter`: Crater diameter (in map units)

### Shapefile Format

Standard ESRI shapefile with polygon geometries representing crater outlines.

**Attributes:**
- `crater_id`: Unique identifier
- `center_x`: X coordinate of center
- `center_y`: Y coordinate of center
- `radius`: Crater radius
- `diameter`: Crater diameter

## Auto-save Feature

The application automatically saves your progress to `craters_autosave.json` after each crater is marked. This file includes:
- Path to the loaded image
- Coordinate reference system information
- All marked craters with their properties

If you close and reopen the application, it will ask if you want to resume your previous session.

## Coordinate Systems

The tool is **projection independent**, meaning it works with any coordinate reference system (CRS) defined in your GeoTIFF or ISIS cube file. The exported data will maintain the same CRS as the input image.

**Supported projections include:**
- Geographic (lat/lon)
- Projected coordinate systems (UTM, State Plane, etc.)
- Planetary projections (Sinusoidal, Stereographic, etc.)

## Testing with Sample Data

Generate a test GeoTIFF with synthetic craters:

```bash
python generate_test_data.py
```

This creates `test_crater_image.tif` with several synthetic craters for practice.

## Troubleshooting

### Issue: "Failed to open image"
- **Solution**: Ensure GDAL is properly installed and supports your file format
- For ISIS cubes, you may need ISIS3 installed

### Issue: "Could not fit a circle"
- **Cause**: Selected points are collinear (in a straight line)
- **Solution**: Select points that better define a circle

### Issue: Points appear in wrong location
- **Cause**: Coordinate transform issue
- **Solution**: Check that the GeoTIFF has valid geospatial metadata

### Issue: Application crashes on export
- **Solution**: Ensure GeoPandas and dependencies are properly installed

## Keyboard Shortcuts

Currently, the application uses mouse interaction only. Future versions may include keyboard shortcuts.

## Advanced Usage

### Working with Large Images

For very large images:
1. The application may be slow to load
2. Consider creating pyramids/overviews in the GeoTIFF beforehand
3. Use lower resolution versions for initial marking

### Batch Processing

For marking many craters:
1. Work in sessions to avoid fatigue
2. Export periodically
3. Use the auto-save feature to preserve work

## Known Limitations

- Only grayscale (single-band) images are displayed
- No zoom controls (use OS-level zoom if needed)
- Cannot edit individual craters (only delete most recent)
- No undo/redo stack (only delete last)

## References

This tool was inspired by [OpenCraterTool](https://github.com/thomasheyer/OpenCraterTool), a QGIS plugin for crater analysis.

For crater counting methodology, see:
- Heyer, T., et al. (2023). "OpenCraterTool: An open source QGIS plugin for crater size-frequency measurements." Planetary and Space Science. https://doi.org/10.1016/j.pss.2023.105687

## Support

For issues, questions, or contributions, please refer to the project repository.

## License

This tool is provided as-is for research and educational purposes.

#!/usr/bin/env python3
"""
Extract true image footprints from COG files and save to geodatabase.

This script processes all Cloud Optimized GeoTIFF (COG) files in a directory,
extracts the true valid data footprint (excluding nodata areas), and saves
the footprints as polygons in a GeoPackage database.

Usage:
    python extract_cog_footprints.py

Output:
    cog_footprints.gpkg - Geodatabase containing footprint polygons for all COG images
"""

import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.transform import guard_transform
from shapely.geometry import shape, box, Polygon
from shapely.ops import unary_union
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import warnings


def get_valid_data_footprint(tif_path, simplify_tolerance=100):
    """
    Extract the footprint polygon of valid data from a GeoTIFF.

    Parameters
    ----------
    tif_path : str
        Path to the GeoTIFF file
    simplify_tolerance : float
        Tolerance for simplifying the polygon (in map units, meters)

    Returns
    -------
    dict
        Dictionary containing footprint information:
        - filename: name of the file
        - geometry: shapely Polygon of the footprint
        - crs: coordinate reference system
        - bounds: (minx, miny, maxx, maxy)
        - area_m2: area in square meters
        - valid_pixels: number of valid pixels
        - total_pixels: total number of pixels
    """

    with rasterio.open(tif_path) as src:
        # Get the first band (assume all bands have same nodata pattern)
        band = src.read(1)

        # Get nodata value
        nodata = src.nodata
        if nodata is None:
            nodata = 0  # Assume 0 is nodata if not specified

        # Create mask of valid data (not nodata)
        valid_mask = band != nodata

        # Count valid pixels
        valid_pixels = np.sum(valid_mask)
        total_pixels = band.size

        # If no valid data, return None
        if valid_pixels == 0:
            return None

        # Convert mask to uint8 for vectorization
        valid_mask = valid_mask.astype(np.uint8)

        # Extract shapes from the mask
        geoms = []
        for geom, value in shapes(valid_mask, transform=src.transform):
            if value == 1:  # Valid data
                geoms.append(shape(geom))

        if not geoms:
            return None

        # Combine all geometries into a single polygon
        if len(geoms) > 1:
            footprint = unary_union(geoms)
        else:
            footprint = geoms[0]

        # Simplify the polygon to reduce complexity
        if simplify_tolerance > 0:
            footprint = footprint.simplify(simplify_tolerance, preserve_topology=True)

        # Calculate area in square meters
        area_m2 = footprint.area

        return {
            'filename': os.path.basename(tif_path),
            'filepath': os.path.abspath(tif_path),
            'geometry': footprint,
            'crs': src.crs,
            'bounds': footprint.bounds,
            'area_m2': area_m2,
            'area_km2': area_m2 / 1e6,
            'valid_pixels': int(valid_pixels),
            'total_pixels': int(total_pixels),
            'valid_fraction': float(valid_pixels) / total_pixels,
            'width': src.width,
            'height': src.height,
            'resolution_m': abs(src.transform[0])  # Pixel size in meters
        }


def extract_all_cog_footprints(
    cog_dir="SDC60_COG",
    output_file="cog_footprints.gpkg",
    simplify_tolerance=100,
    pattern="*.tif"
):
    """
    Extract footprints from all COG files in a directory and save to geodatabase.

    Parameters
    ----------
    cog_dir : str
        Directory containing COG files
    output_file : str
        Output GeoPackage filename
    simplify_tolerance : float
        Tolerance for simplifying polygons (meters)
    pattern : str
        File pattern to match (default: "*.tif")

    Returns
    -------
    str
        Path to the created geodatabase
    """

    print(f"{'='*60}")
    print(f"COG Footprint Extractor")
    print(f"{'='*60}")

    # Find all COG files
    cog_path = Path(cog_dir)
    if not cog_path.exists():
        raise FileNotFoundError(f"Directory not found: {cog_dir}")

    cog_files = sorted(list(cog_path.glob(pattern)))
    print(f"\nFound {len(cog_files)} COG files in {cog_dir}")

    if len(cog_files) == 0:
        raise ValueError(f"No files matching '{pattern}' found in {cog_dir}")

    # Extract footprints
    footprints = []
    failed_files = []

    print(f"\nExtracting footprints...")
    for cog_file in tqdm(cog_files, desc="Processing COG files"):
        try:
            footprint = get_valid_data_footprint(str(cog_file), simplify_tolerance)
            if footprint:
                footprints.append(footprint)
            else:
                failed_files.append((str(cog_file), "No valid data"))
        except Exception as e:
            failed_files.append((str(cog_file), str(e)))
            warnings.warn(f"Failed to process {cog_file.name}: {e}")

    if len(footprints) == 0:
        raise ValueError("No valid footprints extracted from any files")

    print(f"\nSuccessfully extracted {len(footprints)} footprints")
    if failed_files:
        print(f"Failed to process {len(failed_files)} files:")
        for fname, error in failed_files[:5]:  # Show first 5 failures
            print(f"  - {os.path.basename(fname)}: {error}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")

    # Create GeoDataFrame
    print(f"\nCreating GeoDataFrame...")
    gdf = gpd.GeoDataFrame(footprints, geometry='geometry', crs=footprints[0]['crs'])

    # Add a unique ID for each footprint
    gdf['footprint_id'] = range(1, len(gdf) + 1)

    # Reorder columns to put important ones first
    cols = ['footprint_id', 'filename', 'geometry', 'area_km2', 'valid_fraction',
            'width', 'height', 'resolution_m', 'bounds', 'filepath',
            'area_m2', 'valid_pixels', 'total_pixels']
    # Only include columns that exist
    cols = [c for c in cols if c in gdf.columns]
    gdf = gdf[cols]

    # Save to GeoPackage
    print(f"\nSaving to GeoPackage: {output_file}")
    gdf.to_file(output_file, driver="GPKG", layer="cog_footprints")

    # Get file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert to MB
    print(f"  File size: {file_size:.2f} MB")

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"COG Footprints Geodatabase created successfully!")
    print(f"{'='*60}")
    print(f"Output file: {os.path.abspath(output_file)}")
    print(f"Total footprints: {len(gdf)}")
    print(f"Coordinate system: {gdf.crs}")
    print(f"\nFootprint statistics:")
    print(f"  Total coverage area: {gdf['area_km2'].sum():.2f} km²")
    print(f"  Mean footprint area: {gdf['area_km2'].mean():.2f} km²")
    print(f"  Min footprint area: {gdf['area_km2'].min():.2f} km²")
    print(f"  Max footprint area: {gdf['area_km2'].max():.2f} km²")
    print(f"  Mean valid data fraction: {gdf['valid_fraction'].mean():.1%}")
    print(f"{'='*60}")

    return os.path.abspath(output_file)


def query_example(gpkg_file="cog_footprints.gpkg"):
    """
    Example of how to query the COG footprints geodatabase.

    Parameters
    ----------
    gpkg_file : str
        Path to the GeoPackage file
    """
    print(f"\n{'='*60}")
    print(f"Example: Reading from geodatabase")
    print(f"{'='*60}")

    # Read the geodatabase
    gdf = gpd.read_file(gpkg_file, layer="cog_footprints")

    print(f"\nLoaded {len(gdf)} COG footprints from {gpkg_file}")
    print(f"\nFirst 3 records:")
    print(gdf[['footprint_id', 'filename', 'area_km2', 'valid_fraction']].head(3))

    # Example queries
    print(f"\nExample queries:")
    print(f"  Files with >90% valid data: {len(gdf[gdf['valid_fraction'] > 0.9])}")
    print(f"  Files with <50% valid data: {len(gdf[gdf['valid_fraction'] < 0.5])}")

    # Find largest footprint
    largest = gdf.loc[gdf['area_km2'].idxmax()]
    print(f"\nLargest footprint:")
    print(f"  File: {largest['filename']}")
    print(f"  Area: {largest['area_km2']:.2f} km²")


if __name__ == "__main__":
    import pandas as pd

    # Extract footprints
    output_path = extract_all_cog_footprints()

    # Show example usage
    query_example(output_path)

    print(f"\n{'='*60}")
    print(f"Usage in Python:")
    print(f"  import geopandas as gpd")
    print(f"  cog_footprints = gpd.read_file('{os.path.basename(output_path)}', layer='cog_footprints')")
    print(f"\nUsage in MATLAB (R2019a+):")
    print(f"  cog_data = readgeotable('{os.path.basename(output_path)}', 'Layer', 'cog_footprints');")
    print(f"{'='*60}")

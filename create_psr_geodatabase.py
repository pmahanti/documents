#!/usr/bin/env python3
"""
Create a compact geodatabase from LOLA PSR shapefiles.

This script reads the northern and southern hemisphere PSR shapefiles
and combines them into a single GeoPackage file for easy reference and sharing.
Works with both Python and MATLAB.

Usage:
    python create_psr_geodatabase.py

Output:
    psr_database.gpkg - Compact geodatabase containing all PSR polygons
"""

import geopandas as gpd
import os
from pathlib import Path


def create_psr_geodatabase(
    shapefile_dir="shapefiles_1km2",
    output_file="psr_database.gpkg"
):
    """
    Create a GeoPackage database from LOLA PSR shapefiles.

    Parameters
    ----------
    shapefile_dir : str
        Directory containing the PSR shapefiles
    output_file : str
        Output GeoPackage filename

    Returns
    -------
    str
        Path to the created geodatabase
    """

    # Define shapefile paths
    north_psr = os.path.join(shapefile_dir, "LOLA_PSR_75N_120M_82N_060M_1KM2_FINAL.shp")
    south_psr = os.path.join(shapefile_dir, "LOLA_PSR_75S_120M_82S_060M_1KM2_FINAL.shp")

    print(f"Reading northern hemisphere PSR shapefile...")
    print(f"  {north_psr}")

    # Check if files exist
    if not os.path.exists(north_psr):
        raise FileNotFoundError(f"Northern PSR shapefile not found: {north_psr}")
    if not os.path.exists(south_psr):
        raise FileNotFoundError(f"Southern PSR shapefile not found: {south_psr}")

    # Read northern hemisphere PSR
    gdf_north = gpd.read_file(north_psr)
    gdf_north['hemisphere'] = 'North'
    print(f"  Loaded {len(gdf_north)} PSR polygons")
    print(f"  CRS: {gdf_north.crs}")

    print(f"\nReading southern hemisphere PSR shapefile...")
    print(f"  {south_psr}")

    # Read southern hemisphere PSR
    gdf_south = gpd.read_file(south_psr)
    gdf_south['hemisphere'] = 'South'
    print(f"  Loaded {len(gdf_south)} PSR polygons")
    print(f"  CRS: {gdf_south.crs}")

    # Note: North and South have different CRS (polar stereographic projections)
    # We'll store them as separate layers to preserve their native projections
    print(f"\nNote: North and South PSRs use different coordinate systems.")
    print(f"  Storing as separate layers to preserve native projections.")

    # Display column information
    print(f"\nColumns in northern PSR data:")
    for col in gdf_north.columns:
        if col != 'geometry':
            print(f"  - {col}: {gdf_north[col].dtype}")

    # Save to GeoPackage as separate layers
    print(f"\nSaving to GeoPackage: {output_file}")
    print(f"  Layer 1: psr_north (North Pole Stereographic)")
    gdf_north.to_file(output_file, driver="GPKG", layer="psr_north")

    print(f"  Layer 2: psr_south (South Pole Stereographic)")
    gdf_south.to_file(output_file, driver="GPKG", layer="psr_south", mode='a')

    # Get file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert to MB
    print(f"  File size: {file_size:.2f} MB")

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"PSR Geodatabase created successfully!")
    print(f"{'='*60}")
    print(f"Output file: {os.path.abspath(output_file)}")
    print(f"Total PSRs: {len(gdf_north) + len(gdf_south)}")
    print(f"  Northern hemisphere: {len(gdf_north)} PSRs (layer: psr_north)")
    print(f"  Southern hemisphere: {len(gdf_south)} PSRs (layer: psr_south)")
    print(f"\nCoordinate systems:")
    print(f"  North: {gdf_north.crs.name if gdf_north.crs else 'Unknown'}")
    print(f"  South: {gdf_south.crs.name if gdf_south.crs else 'Unknown'}")
    print(f"{'='*60}")

    return os.path.abspath(output_file), gdf_north, gdf_south


def query_example(gpkg_file="psr_database.gpkg"):
    """
    Example of how to query the geodatabase.

    Parameters
    ----------
    gpkg_file : str
        Path to the GeoPackage file
    """
    print(f"\n{'='*60}")
    print(f"Example: Reading from geodatabase")
    print(f"{'='*60}")

    # Read the northern hemisphere layer
    gdf_north = gpd.read_file(gpkg_file, layer="psr_north")
    print(f"\nLoaded {len(gdf_north)} northern PSR polygons")

    # Read the southern hemisphere layer
    gdf_south = gpd.read_file(gpkg_file, layer="psr_south")
    print(f"Loaded {len(gdf_south)} southern PSR polygons")

    print(f"\nFirst 3 northern PSRs:")
    cols_to_show = [col for col in gdf_north.columns if col != 'geometry'][:5]
    print(gdf_north[cols_to_show].head(3))

    # Area statistics if available
    if 'AREA_KM2' in gdf_north.columns:
        print(f"\nNorthern PSR area statistics (km²):")
        print(f"  Total area: {gdf_north['AREA_KM2'].sum():.2f} km²")
        print(f"  Mean area: {gdf_north['AREA_KM2'].mean():.2f} km²")
        print(f"  Max area: {gdf_north['AREA_KM2'].max():.2f} km²")

    if 'AREA_KM2' in gdf_south.columns:
        print(f"\nSouthern PSR area statistics (km²):")
        print(f"  Total area: {gdf_south['AREA_KM2'].sum():.2f} km²")
        print(f"  Mean area: {gdf_south['AREA_KM2'].mean():.2f} km²")
        print(f"  Max area: {gdf_south['AREA_KM2'].max():.2f} km²")


if __name__ == "__main__":
    import pandas as pd

    print("="*60)
    print("LOLA PSR Geodatabase Creator")
    print("="*60)

    # Create the geodatabase
    output_path, gdf_north, gdf_south = create_psr_geodatabase()

    # Show example usage
    query_example(output_path)

    print(f"\n{'='*60}")
    print(f"Usage in Python:")
    print(f"  import geopandas as gpd")
    print(f"  # Read northern PSRs")
    print(f"  psr_north = gpd.read_file('{os.path.basename(output_path)}', layer='psr_north')")
    print(f"  # Read southern PSRs")
    print(f"  psr_south = gpd.read_file('{os.path.basename(output_path)}', layer='psr_south')")
    print(f"\nUsage in MATLAB (R2019a+):")
    print(f"  % Read northern PSRs")
    print(f"  psr_north = readgeotable('{os.path.basename(output_path)}', 'Layer', 'psr_north');")
    print(f"  % Read southern PSRs")
    print(f"  psr_south = readgeotable('{os.path.basename(output_path)}', 'Layer', 'psr_south');")
    print(f"{'='*60}")

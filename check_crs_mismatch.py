#!/usr/bin/env python3
"""Check CRS and bounds mismatch"""
import os
os.environ['PROJ_IGNORE_CELESTIAL_BODY'] = 'YES'

import geopandas as gpd
import rasterio

print("PSR North CRS:")
psr_north = gpd.read_file('psr_database.gpkg', layer='psr_north')
print(f"  {psr_north.crs}")
print(f"  Bounds: {psr_north.total_bounds}")

print("\nPSR South CRS:")
psr_south = gpd.read_file('psr_database.gpkg', layer='psr_south')
print(f"  {psr_south.crs}")
print(f"  Bounds: {psr_south.total_bounds}")

print("\nNorth Temperature Map CRS:")
with rasterio.open('polar_north_80_summer_max-float.tif') as src:
    print(f"  {src.crs}")
    print(f"  Bounds: {src.bounds}")
    print(f"  CRS WKT:")
    print(f"  {src.crs.wkt[:500]}")

print("\nSouth Temperature Map CRS:")
with rasterio.open('polar_south_80_summer_max-float.tif') as src:
    print(f"  {src.crs}")
    print(f"  Bounds: {src.bounds}")
    print(f"  CRS WKT:")
    print(f"  {src.crs.wkt[:500]}")

# Check area column type
print("\nPSR North 'area' column:")
print(f"  Type: {type(psr_north['area'].iloc[0])}")
print(f"  Sample values: {psr_north['area'].head()}")

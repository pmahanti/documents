#!/usr/bin/env python3
"""
Example demonstration of Step 1: Prepare Geometries

This script shows exactly what happens when we buffer point craters
to circular polygons.
"""

import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

print("=" * 70)
print("Step 1: Prepare Geometries - Interactive Example")
print("=" * 70)

# Create sample crater data (points with diameters)
craters_data = {
    'UFID': ['p001', 'p002', 'p003', 'p004'],
    'lat': [-87.031, -87.026, -87.018, -87.017],
    'lon': [84.372, 84.669, 84.798, 84.881],
    'D_m': [169.1, 307.7, 97.5, 45.0],  # Last one is < 60m
    'davg': [8.64, 13.14, 3.12, 1.5]
}

# Create points using projected coordinates
points = [
    Point(89601.29, 8829.40),
    Point(89814.57, 8380.60),
    Point(90063.78, 8199.25),
    Point(90089.44, 8070.37)
]

craters_data['geometry'] = points

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(craters_data, geometry='geometry')

print("\n" + "=" * 70)
print("STEP 1: Read Input Data")
print("=" * 70)
print(gdf)
print(f"\nNumber of craters: {len(gdf)}")
print(f"Geometry type: {gdf.geometry.type.unique()[0]}")

# Step 2: Filter by diameter
min_diameter = 60  # meters
print("\n" + "=" * 70)
print(f"STEP 2: Filter craters (D_m > {min_diameter})")
print("=" * 70)
print(f"Before filtering: {len(gdf)} craters")
gdf_filtered = gdf[gdf['D_m'] > min_diameter].copy()
print(f"After filtering: {len(gdf_filtered)} craters")
print(f"Removed: {gdf[gdf['D_m'] <= min_diameter]['UFID'].tolist()}")

# Step 3: Buffer to circles
print("\n" + "=" * 70)
print("STEP 3: Buffer Points to Circles")
print("=" * 70)

for idx, row in gdf_filtered.iterrows():
    print(f"\nCrater {row['UFID']}:")
    print(f"  Original: POINT ({row.geometry.x:.2f}, {row.geometry.y:.2f})")
    print(f"  Diameter: {row['D_m']:.1f} meters")
    print(f"  Radius: {row['D_m']/2:.2f} meters")

    # Show original geometry
    before = row.geometry
    print(f"  Before: {before.geom_type}")

    # Buffer
    after = before.buffer(row['D_m'] / 2)
    print(f"  After: {after.geom_type}")
    print(f"  Circle area: {after.area:.2f} m²")
    print(f"  Expected area (πr²): {3.14159 * (row['D_m']/2)**2:.2f} m²")
    print(f"  Number of points in circle: {len(after.exterior.coords)}")

# Apply buffering to all
gdf_buffered = gdf_filtered.copy()
gdf_buffered['geometry'] = gdf_buffered.geometry.buffer(gdf_buffered['D_m'] / 2)

print("\n" + "=" * 70)
print("STEP 4: Result")
print("=" * 70)
print(gdf_buffered[['UFID', 'D_m', 'geometry']])
print(f"\nAll geometries are now: {gdf_buffered.geometry.type.unique()[0]}")

# Calculate some statistics
print("\n" + "=" * 70)
print("Statistics")
print("=" * 70)
print(f"Total area covered by craters: {gdf_buffered.geometry.area.sum():.2f} m²")
print(f"Average crater area: {gdf_buffered.geometry.area.mean():.2f} m²")
print(f"Largest crater: {gdf_buffered.loc[gdf_buffered['D_m'].idxmax(), 'UFID']} "
      f"({gdf_buffered['D_m'].max():.1f}m)")
print(f"Smallest crater: {gdf_buffered.loc[gdf_buffered['D_m'].idxmin(), 'UFID']} "
      f"({gdf_buffered['D_m'].min():.1f}m)")

# Show coordinate bounds
print("\n" + "=" * 70)
print("Spatial Extent")
print("=" * 70)
bounds = gdf_buffered.total_bounds
print(f"Bounding box: [{bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f}]")
print(f"Width: {bounds[2] - bounds[0]:.2f} meters")
print(f"Height: {bounds[3] - bounds[1]:.2f} meters")

print("\n" + "=" * 70)
print("Complete! These circular geometries are ready for Step 2 (Rim Refinement)")
print("=" * 70)

# Example of saving (commented out since we don't have actual file paths)
# gdf_buffered.to_file('craters_circles.shp')
# print("\nSaved to: craters_circles.shp")

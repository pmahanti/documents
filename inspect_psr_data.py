#!/usr/bin/env python3
"""
Inspect PSR database and Diviner temperature maps
"""
import geopandas as gpd
import rasterio
from rasterio.plot import show
import numpy as np

print("="*70)
print("PSR DATABASE INSPECTION")
print("="*70)

# Load PSR geopkg
psr_gdf = gpd.read_file('psr_database.gpkg')
print(f"\nPSR Database loaded")
print(f"  Total PSRs: {len(psr_gdf)}")
print(f"  CRS: {psr_gdf.crs}")
print(f"\nColumns:")
for col in psr_gdf.columns:
    print(f"  - {col}")
print(f"\nFirst few rows:")
print(psr_gdf.head())
print(f"\nGeometry types: {psr_gdf.geometry.type.value_counts()}")
print(f"\nBounds:")
print(psr_gdf.total_bounds)

# Calculate areas
if psr_gdf.crs and psr_gdf.crs.is_geographic:
    # Convert to projected CRS for accurate area calculation
    psr_gdf_proj = psr_gdf.to_crs('ESRI:102027')  # Sphere Sinusoidal
    areas = psr_gdf_proj.geometry.area / 1e6  # Convert to km²
    print(f"\nTotal PSR area: {areas.sum():.2f} km²")
    print(f"Mean PSR area: {areas.mean():.6f} km²")
    print(f"Median PSR area: {areas.median():.6f} km²")

print("\n" + "="*70)
print("NORTH POLE TEMPERATURE MAP")
print("="*70)

with rasterio.open('polar_north_80_summer_max-float.tif') as src:
    print(f"\nDimensions: {src.width} x {src.height}")
    print(f"CRS: {src.crs}")
    print(f"Transform: {src.transform}")
    print(f"Bounds: {src.bounds}")
    print(f"Resolution: {src.res}")
    print(f"Data type: {src.dtypes[0]}")
    print(f"NoData value: {src.nodata}")

    # Read a sample
    data = src.read(1, masked=True)
    print(f"\nTemperature Statistics (K):")
    print(f"  Min: {np.nanmin(data):.2f}")
    print(f"  Max: {np.nanmax(data):.2f}")
    print(f"  Mean: {np.nanmean(data):.2f}")
    print(f"  Median: {np.nanmedian(data):.2f}")
    print(f"  Pixels < 110K (PSR threshold): {np.sum(data < 110)}")
    print(f"  Total valid pixels: {np.sum(~data.mask)}")

print("\n" + "="*70)
print("SOUTH POLE TEMPERATURE MAP")
print("="*70)

with rasterio.open('polar_south_80_summer_max-float.tif') as src:
    print(f"\nDimensions: {src.width} x {src.height}")
    print(f"CRS: {src.crs}")
    print(f"Transform: {src.transform}")
    print(f"Bounds: {src.bounds}")
    print(f"Resolution: {src.res}")
    print(f"Data type: {src.dtypes[0]}")
    print(f"NoData value: {src.nodata}")

    # Read a sample
    data = src.read(1, masked=True)
    print(f"\nTemperature Statistics (K):")
    print(f"  Min: {np.nanmin(data):.2f}")
    print(f"  Max: {np.nanmax(data):.2f}")
    print(f"  Mean: {np.nanmean(data):.2f}")
    print(f"  Median: {np.nanmedian(data):.2f}")
    print(f"  Pixels < 110K (PSR threshold): {np.sum(data < 110)}")
    print(f"  Total valid pixels: {np.sum(~data.mask)}")

print("\n" + "="*70)
print("INSPECTION COMPLETE")
print("="*70)

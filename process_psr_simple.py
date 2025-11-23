#!/usr/bin/env python3
"""
Simple PSR temperature processing - works around CRS metadata issues
"""
import os
os.environ['PROJ_IGNORE_CELESTIAL_BODY'] = 'YES'

import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
from shapely.geometry import mapping
import warnings
warnings.filterwarnings('ignore')

print("Processing PSR temperatures (simplified version)...")

# Load PSRs
psr_north = gpd.read_file('psr_database.gpkg', layer='psr_north')
psr_south = gpd.read_file('psr_database.gpkg', layer='psr_south')

# Convert area column to numeric (it's stored as string)
psr_north['area_numeric'] = pd.to_numeric(psr_north['area'], errors='coerce')
psr_south['area_numeric'] = pd.to_numeric(psr_south['area'], errors='coerce')

# Calculate area in km² (area is in pixels at ~240m resolution)
psr_north['area_km2'] = psr_north['area_numeric'] * 0.0576
psr_south['area_km2'] = psr_south['area_numeric'] * 0.0576

print(f"Loaded {len(psr_north)} North PSRs, {len(psr_south)} South PSRs")

# Simple temperature extraction function
def get_temp_stats(geometry, raster_src):
    try:
        out_image, _ = mask(raster_src, [mapping(geometry)], crop=True, filled=False)
        data = out_image[0]
        valid = data[~data.mask]

        if len(valid) == 0:
            return [0, np.nan, np.nan, np.nan, 0, 0.0]

        return [
            len(valid),                      # pixel_count
            float(np.min(valid)),           # temp_min_K
            float(np.max(valid)),           # temp_max_K
            float(np.mean(valid)),          # temp_mean_K
            int(np.sum(valid < 110)),       # pixels_lt_110K
            float(100 * np.sum(valid < 110) / len(valid))  # pct_coldtrap
        ]
    except:
        return [0, np.nan, np.nan, np.nan, 0, 0.0]

# Process North (use South temperature map CRS since both files have same CRS in metadata)
print("\nProcessing North PSRs...")
with rasterio.open('polar_north_80_summer_max-float.tif') as src:
    # The file is mislabeled as South Pole, so we reproject PSRs to match
    psr_north_reproj = psr_north.to_crs(src.crs)

    results_north = []
    for idx, row in psr_north_reproj.iterrows():
        if idx % 1000 == 0:
            print(f"  {idx}/{len(psr_north_reproj)}")
        stats = get_temp_stats(row.geometry, src)
        results_north.append(stats)

stats_cols = ['pixel_count', 'temp_min_K', 'temp_max_K', 'temp_mean_K', 'pixels_lt_110K', 'pct_coldtrap']
psr_north[stats_cols] = pd.DataFrame(results_north, index=psr_north.index)

print("\nProcessing South PSRs...")
with rasterio.open('polar_south_80_summer_max-float.tif') as src:
    psr_south_reproj = psr_south.to_crs(src.crs)

    results_south = []
    for idx, row in psr_south_reproj.iterrows():
        if idx % 1000 == 0:
            print(f"  {idx}/{len(psr_south_reproj)}")
        stats = get_temp_stats(row.geometry, src)
        results_south.append(stats)

psr_south[stats_cols] = pd.DataFrame(results_south, index=psr_south.index)

# Combine and save (convert to common CRS first)
psr_south_common = psr_south.to_crs(psr_north.crs)
psr_all = pd.concat([psr_north, psr_south_common], ignore_index=True)

print("\nSaving outputs...")
psr_all.to_file('psr_with_temperatures.gpkg', driver='GPKG')
psr_all.to_csv('psr_with_temperatures.csv', index=False)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
valid = psr_all[psr_all['pixel_count'] > 0]
print(f"Total PSRs: {len(psr_all)}")
print(f"PSRs with temperature data: {len(valid)}")
print(f"Total PSR area: {psr_all['area_km2'].sum():.2f} km²")
print(f"Mean PSR temperature: {valid['temp_mean_K'].mean():.2f} K")
print(f"Cold trap PSRs (<110K max): {len(psr_all[psr_all['temp_max_K'] < 110])}")
print(f"Total cold trap pixels: {psr_all['pixels_lt_110K'].sum():,}")
print("\nDone! Created: psr_with_temperatures.gpkg and .csv")

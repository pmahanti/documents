#!/usr/bin/env python3
"""
Process PSR database with Diviner temperature data

This script:
1. Loads PSR polygons from geopkg (north and south)
2. Extracts Diviner maximum temperature statistics for each PSR
3. Calculates PSR areas in km²
4. Creates comprehensive output geotable and shapefile
"""
import os
# Allow CRS transformations between Earth and Moon
os.environ['PROJ_IGNORE_CELESTIAL_BODY'] = 'YES'

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
import numpy as np
import pandas as pd
from shapely.geometry import mapping, box
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PSR TEMPERATURE ANALYSIS - Processing")
print("="*80)

# Load PSR databases (both hemispheres)
print("\n[1/5] Loading PSR databases...")
psr_north = gpd.read_file('psr_database.gpkg', layer='psr_north')
psr_south = gpd.read_file('psr_database.gpkg', layer='psr_south')

print(f"  North PSRs: {len(psr_north)}")
print(f"  South PSRs: {len(psr_south)}")

# Open temperature rasters
print("\n[2/5] Loading Diviner temperature maps...")
north_temp = rasterio.open('polar_north_80_summer_max-float.tif')
south_temp = rasterio.open('polar_south_80_summer_max-float.tif')

print(f"  North map: {north_temp.width}x{north_temp.height}, res={north_temp.res[0]:.1f}m")
print(f"  South map: {south_temp.width}x{south_temp.height}, res={south_temp.res[0]:.1f}m")

# Function to extract temperature statistics for a PSR polygon
def extract_psr_temps(geometry, raster_src):
    """Extract temperature statistics for a PSR polygon"""
    try:
        # Crop raster to polygon
        out_image, out_transform = mask(raster_src, [mapping(geometry)], crop=True, filled=False)
        data = out_image[0]

        # Remove masked/nodata values
        valid_temps = data[~data.mask]

        if len(valid_temps) == 0:
            return {
                'pixel_count': 0,
                'temp_min_K': np.nan,
                'temp_max_K': np.nan,
                'temp_mean_K': np.nan,
                'temp_std_K': np.nan,
                'temp_median_K': np.nan,
                'pixels_lt_110K': 0,
                'pct_coldtrap': 0.0
            }

        # Calculate statistics
        pixel_count = len(valid_temps)
        pixels_cold = np.sum(valid_temps < 110)

        return {
            'pixel_count': int(pixel_count),
            'temp_min_K': float(np.min(valid_temps)),
            'temp_max_K': float(np.max(valid_temps)),
            'temp_mean_K': float(np.mean(valid_temps)),
            'temp_std_K': float(np.std(valid_temps)),
            'temp_median_K': float(np.median(valid_temps)),
            'pixels_lt_110K': int(pixels_cold),
            'pct_coldtrap': float(100 * pixels_cold / pixel_count)
        }
    except Exception as e:
        print(f"    Warning: Failed to process PSR - {e}")
        return {
            'pixel_count': 0,
            'temp_min_K': np.nan,
            'temp_max_K': np.nan,
            'temp_mean_K': np.nan,
            'temp_std_K': np.nan,
            'temp_median_K': np.nan,
            'pixels_lt_110K': 0,
            'pct_coldtrap': 0.0
        }

# Process North Pole PSRs
print("\n[3/5] Processing North Pole PSRs...")
print(f"  Reprojecting to temperature map CRS...")
psr_north_proj = psr_north.to_crs(north_temp.crs)

# Filter PSRs that overlap with temperature raster bounds
temp_bounds = box(*north_temp.bounds)
psr_north_overlap = psr_north_proj[psr_north_proj.intersects(temp_bounds)]
print(f"  PSRs overlapping temperature map: {len(psr_north_overlap)}/{len(psr_north_proj)}")

print(f"  Extracting temperatures for {len(psr_north_overlap)} PSRs...")
north_stats = []
north_indices = []
for idx, row in psr_north_overlap.iterrows():
    if len(north_stats) % 500 == 0:
        print(f"    Progress: {len(north_stats)}/{len(psr_north_overlap)}")
    stats = extract_psr_temps(row.geometry, north_temp)
    north_stats.append(stats)
    north_indices.append(idx)

# Add temperature stats to north dataframe (for overlapping PSRs only)
north_stats_df = pd.DataFrame(north_stats, index=north_indices)
psr_north_result = pd.concat([psr_north, north_stats_df], axis=1)

# Fill NaN for non-overlapping PSRs
for col in north_stats_df.columns:
    if col not in psr_north.columns:
        psr_north_result[col] = psr_north_result[col].fillna(0 if col == 'pixel_count' or col == 'pixels_lt_110K' else np.nan)

# Calculate area in km² (using native projection, area column is in pixels, convert to km²)
# Pixel size is ~240m, so 1 pixel = 0.0576 km²
psr_north_result['area_km2'] = psr_north_result['area'] * 0.0576

print(f"  ✓ North PSRs processed")

# Process South Pole PSRs
print("\n[4/5] Processing South Pole PSRs...")
print(f"  Reprojecting to temperature map CRS...")
psr_south_proj = psr_south.to_crs(south_temp.crs)

# Filter PSRs that overlap with temperature raster bounds
temp_bounds_south = box(*south_temp.bounds)
psr_south_overlap = psr_south_proj[psr_south_proj.intersects(temp_bounds_south)]
print(f"  PSRs overlapping temperature map: {len(psr_south_overlap)}/{len(psr_south_proj)}")

print(f"  Extracting temperatures for {len(psr_south_overlap)} PSRs...")
south_stats = []
south_indices = []
for idx, row in psr_south_overlap.iterrows():
    if len(south_stats) % 500 == 0:
        print(f"    Progress: {len(south_stats)}/{len(psr_south_overlap)}")
    stats = extract_psr_temps(row.geometry, south_temp)
    south_stats.append(stats)
    south_indices.append(idx)

# Add temperature stats to south dataframe (for overlapping PSRs only)
south_stats_df = pd.DataFrame(south_stats, index=south_indices)
psr_south_result = pd.concat([psr_south, south_stats_df], axis=1)

# Fill NaN for non-overlapping PSRs
for col in south_stats_df.columns:
    if col not in psr_south.columns:
        psr_south_result[col] = psr_south_result[col].fillna(0 if col == 'pixel_count' or col == 'pixels_lt_110K' else np.nan)

# Calculate area in km² (using native projection, area column is in pixels, convert to km²)
# Pixel size is ~240m, so 1 pixel = 0.0576 km²
psr_south_result['area_km2'] = psr_south_result['area'] * 0.0576

print(f"  ✓ South PSRs processed")

# Combine both hemispheres
print("\n[5/5] Creating output files...")
psr_combined = pd.concat([psr_north_result, psr_south_result], ignore_index=True)

# Save as GeoPackage
output_gpkg = 'psr_with_temperatures.gpkg'
psr_combined.to_file(output_gpkg, driver='GPKG', layer='psr_all')
print(f"  ✓ Saved: {output_gpkg}")

# Save as Shapefile (for compatibility)
output_shp = 'psr_with_temperatures.shp'
# Shapefiles have column name length limits, so abbreviate
psr_combined_shp = psr_combined.copy()
psr_combined_shp.columns = [col[:10] if len(col) > 10 else col for col in psr_combined_shp.columns]
psr_combined_shp.to_file(output_shp, driver='ESRI Shapefile')
print(f"  ✓ Saved: {output_shp}")

# Save summary statistics as CSV
summary_cols = ['PSR_ID', 'hemisphere', 'latitude', 'longitude', 'area_km2',
                'pixel_count', 'temp_min_K', 'temp_max_K', 'temp_mean_K',
                'temp_median_K', 'pixels_lt_110K', 'pct_coldtrap']
psr_combined[summary_cols].to_csv('psr_temperature_summary.csv', index=False)
print(f"  ✓ Saved: psr_temperature_summary.csv")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nTotal PSRs processed: {len(psr_combined)}")
print(f"  North: {len(psr_north_result)}")
print(f"  South: {len(psr_south_result)}")

print(f"\nTotal PSR Area: {psr_combined['area_km2'].sum():.2f} km²")
print(f"  North: {psr_north_result['area_km2'].sum():.2f} km²")
print(f"  South: {psr_south_result['area_km2'].sum():.2f} km²")

valid_psrs = psr_combined[psr_combined['pixel_count'] > 0]
print(f"\nPSRs with temperature data: {len(valid_psrs)} ({100*len(valid_psrs)/len(psr_combined):.1f}%)")

print(f"\nTemperature Statistics (all PSRs):")
print(f"  Min temperature: {psr_combined['temp_min_K'].min():.2f} K")
print(f"  Max temperature: {psr_combined['temp_max_K'].max():.2f} K")
print(f"  Mean of means: {psr_combined['temp_mean_K'].mean():.2f} K")
print(f"  Median of medians: {psr_combined['temp_median_K'].median():.2f} K")

cold_trap_psrs = psr_combined[psr_combined['temp_max_K'] < 110]
print(f"\nCold Trap PSRs (max temp < 110K): {len(cold_trap_psrs)}")
print(f"  Total area: {cold_trap_psrs['area_km2'].sum():.2f} km²")

total_cold_pixels = psr_combined['pixels_lt_110K'].sum()
total_pixels = psr_combined['pixel_count'].sum()
print(f"\nCold trap pixels (<110K): {total_cold_pixels:,} / {total_pixels:,} ({100*total_cold_pixels/total_pixels:.2f}%)")

print("\n" + "="*80)
print("PROCESSING COMPLETE")
print("="*80)
print("\nOutput files created:")
print(f"  1. {output_gpkg} - Full PSR database with temperatures")
print(f"  2. {output_shp} - Shapefile version")
print(f"  3. psr_temperature_summary.csv - Tabular summary")
print("\n")

# Close rasters
north_temp.close()
south_temp.close()

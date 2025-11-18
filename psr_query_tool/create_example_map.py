#!/usr/bin/env python3
"""
Create example PNG map using synthetic PSR data.

This demonstrates the visualization capabilities without requiring
the actual NASA data files.
"""

import sys
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np


def create_synthetic_psr_data():
    """Create realistic synthetic PSR data for the lunar south pole."""
    print("Creating synthetic lunar PSR data...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate PSRs in a realistic distribution around the south pole
    psrs = []

    # Large PSRs near the pole
    for i in range(15):
        # Center point (latitude around -85 to -89.5)
        lat = -85 - np.random.random() * 4.5
        lon = np.random.random() * 360 - 180

        # Size in degrees (larger near pole)
        size = 0.1 + np.random.random() * 0.5

        # Create a rough polygon
        angles = np.linspace(0, 2*np.pi, 8)
        r_variation = 0.8 + np.random.random(8) * 0.4  # Random radius variation

        points = []
        for angle, r_var in zip(angles, r_variation):
            dx = size * r_var * np.cos(angle)
            dy = size * r_var * np.sin(angle)
            points.append((lon + dx, lat + dy))

        psrs.append({
            'name': f'Synthetic_PSR_{i+1:03d}',
            'geometry': Polygon(points),
            'confidence': 0.85 + np.random.random() * 0.14,
            'area_km2': size * size * 100,  # Rough estimate
            'type': 'permanent' if np.random.random() > 0.2 else 'seasonal'
        })

    # Medium PSRs
    for i in range(25):
        lat = -82 - np.random.random() * 7
        lon = np.random.random() * 360 - 180
        size = 0.05 + np.random.random() * 0.2

        angles = np.linspace(0, 2*np.pi, 6)
        r_variation = 0.8 + np.random.random(6) * 0.4

        points = []
        for angle, r_var in zip(angles, r_variation):
            dx = size * r_var * np.cos(angle)
            dy = size * r_var * np.sin(angle)
            points.append((lon + dx, lat + dy))

        psrs.append({
            'name': f'Synthetic_PSR_{i+16:03d}',
            'geometry': Polygon(points),
            'confidence': 0.75 + np.random.random() * 0.24,
            'area_km2': size * size * 100,
            'type': 'permanent' if np.random.random() > 0.3 else 'seasonal'
        })

    # Small PSRs (scattered)
    for i in range(40):
        lat = -80 - np.random.random() * 9.5
        lon = np.random.random() * 360 - 180
        size = 0.02 + np.random.random() * 0.08

        angles = np.linspace(0, 2*np.pi, 5)
        r_variation = 0.7 + np.random.random(5) * 0.5

        points = []
        for angle, r_var in zip(angles, r_variation):
            dx = size * r_var * np.cos(angle)
            dy = size * r_var * np.sin(angle)
            points.append((lon + dx, lat + dy))

        psrs.append({
            'name': f'Synthetic_PSR_{i+41:03d}',
            'geometry': Polygon(points),
            'confidence': 0.65 + np.random.random() * 0.34,
            'area_km2': size * size * 100,
            'type': 'permanent' if np.random.random() > 0.5 else 'seasonal'
        })

    gdf = gpd.GeoDataFrame(psrs, crs='EPSG:4326')
    print(f"Created {len(gdf)} synthetic PSRs")

    return gdf


def main():
    """Generate example map with synthetic data."""
    from psr_query import PSRQuery

    print("=" * 80)
    print("PSR Query Tool - Example Map Generator")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path('examples')
    output_dir.mkdir(exist_ok=True)

    # Create synthetic data directory
    data_dir = Path('data/synthetic')
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic PSR data
    synthetic_file = data_dir / 'synthetic_psr.shp'
    if not synthetic_file.exists():
        synthetic_data = create_synthetic_psr_data()
        synthetic_data.to_file(synthetic_file)
        print(f"Saved synthetic data to: {synthetic_file}")
    else:
        print(f"Using existing synthetic data: {synthetic_file}")

    print()

    # Example 1: Query near Shackleton Crater region
    print("Example 1: Shackleton Crater Region (-89.9°, 0°, 50km radius)")
    print("-" * 80)

    psr = PSRQuery(str(synthetic_file))
    results = psr.query_psrs(-89.9, 0.0, 50)

    print(f"\nFound {len(results)} PSR(s)")
    if len(results) > 0:
        print(f"Nearest PSR: {results.iloc[0]['distance_km']:.2f} km")
        print(f"Farthest PSR: {results.iloc[-1]['distance_km']:.2f} km")

    # Create PNG map
    map_file = output_dir / 'example_shackleton_region.png'
    psr.create_map_png(
        results,
        -89.9,
        0.0,
        50,
        str(map_file),
        dpi=150
    )

    print(f"\n✓ Created map: {map_file}")
    print()

    # Example 2: Wider area search
    print("Example 2: South Pole Wide Area (-87°, 45°, 100km radius)")
    print("-" * 80)

    results2 = psr.query_psrs(-87.0, 45.0, 100)

    print(f"\nFound {len(results2)} PSR(s)")
    if len(results2) > 0:
        print(f"Nearest PSR: {results2.iloc[0]['distance_km']:.2f} km")
        print(f"Farthest PSR: {results2.iloc[-1]['distance_km']:.2f} km")

    # Create PNG map
    map_file2 = output_dir / 'example_wide_area.png'
    psr.create_map_png(
        results2,
        -87.0,
        45.0,
        100,
        str(map_file2),
        dpi=150
    )

    print(f"\n✓ Created map: {map_file2}")
    print()

    # Example 3: Multiple queries comparison
    print("Example 3: Multiple Location Comparison")
    print("-" * 80)

    locations = [
        ("Shackleton", -89.9, 0.0, 30),
        ("Haworth", -87.4, -4.5, 40),
        ("Cabeus", -84.9, -35.5, 50),
    ]

    for name, lat, lon, radius in locations:
        results_i = psr.query_psrs(lat, lon, radius)
        map_file_i = output_dir / f'example_{name.lower()}.png'

        psr.create_map_png(
            results_i,
            lat,
            lon,
            radius,
            str(map_file_i),
            dpi=120  # Lower DPI for smaller file size
        )

        print(f"✓ {name:12s}: {len(results_i):2d} PSRs found → {map_file_i.name}")

    print()
    print("=" * 80)
    print("All example maps created successfully!")
    print("=" * 80)
    print(f"\nView the maps in the '{output_dir}' directory")
    print()
    print("Note: These examples use synthetic data for demonstration.")
    print("Download real NASA PGDA data for actual lunar PSR analysis.")

    return 0


if __name__ == '__main__':
    sys.exit(main())

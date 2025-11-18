#!/usr/bin/env python3
"""
Example usage of the PSR Query Tool

This script demonstrates various use cases for querying Lunar PSRs.
"""

from psr_query import PSRQuery
from pathlib import Path


def example_basic_query():
    """Example: Basic query for PSRs near the South Pole."""
    print("=" * 80)
    print("Example 1: Basic Query")
    print("=" * 80)

    # Initialize with shapefile
    psr = PSRQuery('data/LPSR_80S_20MPP_ADJ.shp')

    # Query PSRs within 50km of Shackleton Crater (approximately -89.9°S, 0°E)
    results = psr.query_psrs(
        latitude=-89.9,
        longitude=0.0,
        radius_km=50
    )

    # Display results
    psr.display_results(results)

    # Export to GeoJSON
    if len(results) > 0:
        psr.export_results(results, 'shackleton_psrs.geojson')


def example_convert_and_query():
    """Example: Convert to parquet first, then query."""
    print("\n" + "=" * 80)
    print("Example 2: Convert to Parquet and Query")
    print("=" * 80)

    # Load and convert
    psr = PSRQuery('data/LPSR_80S_20MPP_ADJ.shp')
    parquet_path = psr.convert_to_parquet()

    # Query using parquet
    psr_parquet = PSRQuery(parquet_path, use_parquet=True)
    results = psr_parquet.query_psrs(
        latitude=-85.0,
        longitude=90.0,
        radius_km=100
    )

    psr_parquet.display_results(results)


def example_multiple_locations():
    """Example: Query multiple locations."""
    print("\n" + "=" * 80)
    print("Example 3: Multiple Location Queries")
    print("=" * 80)

    psr = PSRQuery('data/LPSR_80S_20MPP_ADJ.shp')

    # Interesting lunar south pole locations
    locations = [
        ("Shackleton Crater", -89.9, 0.0),
        ("Haworth Crater", -87.4, -4.5),
        ("Cabeus Crater", -84.9, -35.5),
        ("de Gerlache Crater", -88.5, -67.5),
    ]

    radius_km = 30

    for name, lat, lon in locations:
        print(f"\n{name} ({lat}°, {lon}°)")
        print("-" * 40)
        results = psr.query_psrs(lat, lon, radius_km)
        print(f"Found {len(results)} PSR(s) within {radius_km} km")

        if len(results) > 0:
            total_area = 0
            for _, row in results.iterrows():
                if row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                    # Rough area calculation
                    import numpy as np
                    area_deg2 = row.geometry.area
                    lat_centroid = row.geometry.centroid.y
                    area_km2 = area_deg2 * (111.32 * np.cos(np.radians(lat_centroid)) * 111.32)
                    total_area += area_km2

            print(f"Total PSR area: ~{total_area:.2f} km²")


def example_export_formats():
    """Example: Export results in different formats."""
    print("\n" + "=" * 80)
    print("Example 4: Export in Different Formats")
    print("=" * 80)

    psr = PSRQuery('data/LPSR_80S_20MPP_ADJ.shp')
    results = psr.query_psrs(-89.0, 0.0, 75)

    if len(results) > 0:
        # Export as GeoJSON
        psr.export_results(results, 'results.geojson', 'geojson')

        # Export as CSV
        psr.export_results(results, 'results.csv', 'csv')

        # Export as Shapefile
        psr.export_results(results, 'results.shp', 'shapefile')

        print("\nExported results in 3 formats!")
    else:
        print("No results to export.")


def main():
    """Run all examples."""
    data_file = Path('data/LPSR_80S_20MPP_ADJ.shp')

    if not data_file.exists():
        print("ERROR: Data file not found!")
        print(f"Please download the PSR shapefile to: {data_file}")
        print("See data/README.md for instructions.")
        return

    try:
        # Run example 1
        example_basic_query()

        # Uncomment to run other examples:
        # example_convert_and_query()
        # example_multiple_locations()
        # example_export_formats()

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

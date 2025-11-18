#!/usr/bin/env python3
"""
Basic test script to verify the PSR query tool works correctly.

This uses synthetic test data if the real data is not available.
"""

import sys
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd


def create_test_data():
    """Create synthetic PSR data for testing."""
    print("Creating synthetic test data...")

    # Create some test PSRs near the lunar south pole
    psrs = [
        {
            'name': 'Test_PSR_1',
            'geometry': Polygon([
                (-1, -89), (1, -89), (1, -88.5), (-1, -88.5), (-1, -89)
            ]),
            'confidence': 0.95
        },
        {
            'name': 'Test_PSR_2',
            'geometry': Polygon([
                (44, -89.5), (46, -89.5), (46, -89.2), (44, -89.2), (44, -89.5)
            ]),
            'confidence': 0.90
        },
        {
            'name': 'Test_PSR_3',
            'geometry': Polygon([
                (89, -85), (91, -85), (91, -84.5), (89, -84.5), (89, -85)
            ]),
            'confidence': 0.88
        },
        {
            'name': 'Test_PSR_4',
            'geometry': Polygon([
                (-45, -87), (-43, -87), (-43, -86.5), (-45, -86.5), (-45, -87)
            ]),
            'confidence': 0.92
        },
    ]

    gdf = gpd.GeoDataFrame(psrs, crs='EPSG:4326')
    return gdf


def test_basic_functionality():
    """Test basic PSR query functionality."""
    from psr_query import PSRQuery

    print("=" * 80)
    print("Testing PSR Query Tool")
    print("=" * 80)

    # Create test shapefile
    test_dir = Path('data/test')
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / 'test_psr.shp'

    if not test_file.exists():
        test_data = create_test_data()
        test_data.to_file(test_file)
        print(f"Created test data: {test_file}")

    # Test 1: Load data
    print("\nTest 1: Loading data...")
    try:
        psr = PSRQuery(str(test_file))
        print("✓ Data loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return False

    # Test 2: Query near South Pole (should find Test_PSR_1 and Test_PSR_2)
    print("\nTest 2: Query near South Pole (-89°, 0°) with 50km radius...")
    try:
        results = psr.query_psrs(-89.0, 0.0, 50)
        print(f"✓ Found {len(results)} PSR(s)")
        if len(results) > 0:
            psr.display_results(results)
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False

    # Test 3: Query with small radius (should find fewer)
    print("\nTest 3: Query with smaller radius (10km)...")
    try:
        results = psr.query_psrs(-89.0, 0.0, 10)
        print(f"✓ Found {len(results)} PSR(s) within 10km")
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False

    # Test 4: Convert to parquet
    print("\nTest 4: Converting to GeoParquet...")
    try:
        parquet_file = psr.convert_to_parquet(test_dir / 'test_psr.parquet')
        print(f"✓ Converted to parquet: {parquet_file}")
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return False

    # Test 5: Query using parquet
    print("\nTest 5: Query using parquet format...")
    try:
        psr_parquet = PSRQuery(str(parquet_file), use_parquet=True)
        results = psr_parquet.query_psrs(-89.0, 0.0, 50)
        print(f"✓ Query successful, found {len(results)} PSR(s)")
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False

    # Test 6: Export results
    print("\nTest 6: Exporting results...")
    try:
        if len(results) > 0:
            psr.export_results(results, test_dir / 'test_results.geojson')
            psr.export_results(results, test_dir / 'test_results.csv', format='csv')
            print("✓ Export successful")
        else:
            print("⊘ No results to export (not an error)")
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    return True


def main():
    """Run tests."""
    try:
        success = test_basic_functionality()
        return 0 if success else 1
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

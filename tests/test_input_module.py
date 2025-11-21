#!/usr/bin/env python3
"""
Test script for input_module.py

Tests coordinate conversions, file reading, and basic functionality.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("=" * 70)
print("Testing Input Module")
print("=" * 70)

# Test 1: Import module
print("\n[Test 1] Importing input_module...")
try:
    from crater_analysis.input_module import (
        CoordinateConverter,
        read_crater_file,
        MOON_RADIUS
    )
    print("✓ Success - Module imported")
    print(f"  Moon radius: {MOON_RADIUS} m")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 2: Coordinate conversions
print("\n[Test 2] Testing coordinate conversions...")
try:
    # Test equirectangular
    converter = CoordinateConverter('equirectangular', center_lon=0, center_lat=0)

    # Test point
    lat, lon = -87.031, 84.372
    x, y = converter.latlon_to_xy(lat, lon)
    lat2, lon2 = converter.xy_to_latlon(x, y)

    error_lat = abs(lat2 - lat)
    error_lon = abs(lon2 - lon)

    print(f"  Original: lat={lat:.6f}, lon={lon:.6f}")
    print(f"  Converted to: x={x:.2f} m, y={y:.2f} m")
    print(f"  Converted back: lat={lat2:.6f}, lon={lon2:.6f}")
    print(f"  Round-trip error: {error_lat:.2e}° lat, {error_lon:.2e}° lon")

    if error_lat < 1e-6 and error_lon < 1e-6:
        print("✓ Success - Equirectangular conversion accurate")
    else:
        print("✗ Warning - High round-trip error")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Stereographic projection
print("\n[Test 3] Testing stereographic projection...")
try:
    converter_stereo = CoordinateConverter('stereographic', center_lon=0, center_lat=-90)

    lat, lon = -85.0, 0.0  # Near south pole
    x, y = converter_stereo.latlon_to_xy(lat, lon)
    lat2, lon2 = converter_stereo.xy_to_latlon(x, y)

    error_lat = abs(lat2 - lat)
    error_lon = abs(lon2 - lon)

    print(f"  Original: lat={lat:.6f}, lon={lon:.6f}")
    print(f"  Converted to: x={x:.2f} m, y={y:.2f} m")
    print(f"  Converted back: lat={lat2:.6f}, lon={lon2:.6f}")
    print(f"  Round-trip error: {error_lat:.2e}° lat, {error_lon:.2e}° lon")

    if error_lat < 1e-4 and error_lon < 1e-4:
        print("✓ Success - Stereographic conversion accurate")
    else:
        print("✗ Warning - High round-trip error")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 4: Read crater file (lat/lon)
print("\n[Test 4] Reading lat/lon crater file...")
try:
    crater_file = Path(__file__).parent.parent / 'data' / 'example_craters_latlon.csv'

    if crater_file.exists():
        df = read_crater_file(crater_file)
        print(f"✓ Success - Read {len(df)} craters")
        print(f"  Coordinate type: {df['coord_type'].iloc[0]}")
        print(f"  Diameter range: {df['diameter'].min():.1f} - {df['diameter'].max():.1f} m")
        print(f"  First crater: lat={df['coord1'].iloc[0]:.6f}, lon={df['coord2'].iloc[0]:.6f}")
    else:
        print(f"⊗ Skipped - Example file not found: {crater_file}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Read crater file (X/Y)
print("\n[Test 5] Reading X/Y crater file...")
try:
    crater_file = Path(__file__).parent.parent / 'data' / 'example_craters_xy.csv'

    if crater_file.exists():
        df = read_crater_file(crater_file)
        print(f"✓ Success - Read {len(df)} craters")
        print(f"  Coordinate type: {df['coord_type'].iloc[0]}")
        print(f"  X range: {df['coord1'].min():.1f} - {df['coord1'].max():.1f} m")
        print(f"  Y range: {df['coord2'].min():.1f} - {df['coord2'].max():.1f} m")
        print(f"  First crater: x={df['coord1'].iloc[0]:.2f}, y={df['coord2'].iloc[0]:.2f}")
    else:
        print(f"⊗ Skipped - Example file not found: {crater_file}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 6: Projection types
print("\n[Test 6] Testing all projection types...")
projections = ['equirectangular', 'stereographic', 'orthographic']
test_point = (-85.0, 45.0)  # lat, lon

for proj in projections:
    try:
        if proj == 'stereographic':
            conv = CoordinateConverter(proj, center_lon=0, center_lat=-90)
        else:
            conv = CoordinateConverter(proj, center_lon=0, center_lat=0)

        x, y = conv.latlon_to_xy(test_point[0], test_point[1])
        lat, lon = conv.xy_to_latlon(x, y)

        error = np.sqrt((lat - test_point[0])**2 + (lon - test_point[1])**2)

        print(f"  {proj:15s}: x={x:10.2f} m, y={y:10.2f} m, error={error:.2e}°")

    except Exception as e:
        print(f"  {proj:15s}: ✗ Failed - {e}")

print("\n" + "=" * 70)
print("Testing Complete!")
print("=" * 70)

print("\nModule functions available:")
print("  - CoordinateConverter (class)")
print("  - read_crater_file()")
print("  - read_isis_cube()")
print("  - create_crater_shapefile()")
print("  - plot_crater_locations()")
print("  - plot_csfd()")
print("  - process_crater_inputs() [main function]")

print("\nTo test full functionality, run:")
print("  python process_crater_inputs.py --help")

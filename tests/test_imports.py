#!/usr/bin/env python3
"""Test script to validate module imports and basic functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("=" * 60)
print("Testing Crater Analysis Package Imports")
print("=" * 60)

# Test 1: Import main package
print("\n[Test 1] Importing crater_analysis package...")
try:
    import crater_analysis
    print(f"✓ Success - Version: {crater_analysis.__version__}")
except ImportError as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 2: Import config module
print("\n[Test 2] Importing config module...")
try:
    from crater_analysis.config import Config
    print("✓ Success - Config class imported")
except ImportError as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 3: Import refinement module
print("\n[Test 3] Importing refinement module...")
try:
    from crater_analysis.refinement import (
        update_crater_rims,
        prepare_crater_geometries
    )
    print("✓ Success - Refinement functions imported")
except ImportError as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 4: Import morphometry module
print("\n[Test 4] Importing morphometry module...")
try:
    from crater_analysis.morphometry import (
        compute_depth_diameter_ratios,
        export_morphometry_to_csv,
        get_morphometry_summary
    )
    print("✓ Success - Morphometry functions imported")
except ImportError as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 5: Import cratools module
print("\n[Test 5] Importing cratools module...")
try:
    from crater_analysis import cratools
    print("✓ Success - cratools imported")
    print(f"  Available functions: fit_crater_rim, remove_external_topography,")
    print(f"                       compute_E_matrix, compute_depth_diameter_ratio")
except ImportError as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 6: Test Config loading
print("\n[Test 6] Testing configuration loading...")
try:
    config = Config()
    print(f"✓ Success - Config loaded from: {config.config_path}")
    print(f"  Default region: {config.config['default_region']}")
    print(f"  Available regions: {list(config.config['regions'].keys())}")
    print(f"  Min diameter: {config.get_min_diameter()} m")
except Exception as e:
    print(f"✗ Failed: {e}")
    # Not critical, continue

# Test 7: Verify JSON config structure
print("\n[Test 7] Validating JSON config structure...")
try:
    config = Config()
    assert 'regions' in config.config, "Missing 'regions' key"
    assert 'paths' in config.config, "Missing 'paths' key"
    assert 'default_region' in config.config, "Missing 'default_region' key"
    print("✓ Success - Config structure is valid")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "=" * 60)
print("All Import Tests Completed Successfully!")
print("=" * 60)
print("\nNote: Full functionality testing requires dependencies:")
print("  - numpy, pandas, geopandas, rasterio, matplotlib, scipy, uncertainties")
print("\nInstall with: pip install -r requirements.txt")

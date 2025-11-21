#!/usr/bin/env python3
"""
Test script for refine_crater_rim.py module.

Tests rim refinement functions and probability scoring.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("=" * 70)
print("Testing Rim Refinement Module")
print("=" * 70)

# Test 1: Import module
print("\n[Test 1] Importing refine_crater_rim module...")
try:
    from crater_analysis.refine_crater_rim import (
        compute_rim_probability,
        compute_topographic_quality,
        refine_crater_rims
    )
    print("✓ Success - Module imported")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Probability scoring
print("\n[Test 2] Testing probability scoring...")
try:
    # Test with good detection
    prob_high = compute_rim_probability(
        topo_quality=0.8,
        edge_strength=0.7,
        radius_agreement=0.9,
        diameter_error=2.0
    )
    print(f"  High quality detection: probability = {prob_high:.3f}")
    assert 0.7 < prob_high < 1.0, "High quality should give high probability"

    # Test with poor detection
    prob_low = compute_rim_probability(
        topo_quality=0.3,
        edge_strength=0.2,
        radius_agreement=0.4,
        diameter_error=10.0
    )
    print(f"  Low quality detection: probability = {prob_low:.3f}")
    assert 0.0 < prob_low < 0.5, "Low quality should give low probability"

    # Test with medium detection
    prob_med = compute_rim_probability(
        topo_quality=0.5,
        edge_strength=0.5,
        radius_agreement=0.6,
        diameter_error=5.0
    )
    print(f"  Medium quality detection: probability = {prob_med:.3f}")
    assert 0.4 < prob_med < 0.7, "Medium quality should give medium probability"

    print("✓ Success - Probability scoring works correctly")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Topographic quality assessment
print("\n[Test 3] Testing topographic quality assessment...")
try:
    # Create synthetic elevation profile with clear peaks
    n_radii = 41
    n_azimuths = 72
    radii = np.linspace(0.8, 1.2, n_radii)

    # Profile with clear rim peak at r=1.0
    elevation_good = np.zeros((n_radii, n_azimuths))
    for i in range(n_azimuths):
        # Gaussian peak around r=1.0
        peak_idx = np.argmin(np.abs(radii - 1.0))
        elevation_good[:, i] = 1450 + 5 * np.exp(-((np.arange(n_radii) - peak_idx) / 5)**2)

    quality_good = compute_topographic_quality(elevation_good, radius=100)
    print(f"  Clear peaks: quality = {quality_good:.3f}")
    assert quality_good > 0.6, "Clear peaks should give high quality"

    # Profile with no clear peaks (degraded)
    elevation_bad = np.random.randn(n_radii, n_azimuths) * 2 + 1450
    quality_bad = compute_topographic_quality(elevation_bad, radius=100)
    print(f"  No clear peaks: quality = {quality_bad:.3f}")
    assert quality_bad < 0.5, "No peaks should give low quality"

    print("✓ Success - Topographic quality assessment works")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Probability bounds
print("\n[Test 4] Testing probability bounds...")
try:
    # Test extreme values
    test_cases = [
        (1.0, 1.0, 1.0, 0.0),  # Perfect
        (0.0, 0.0, 0.0, 100.0),  # Worst
        (0.5, 0.5, 0.5, 5.0),  # Average
    ]

    all_valid = True
    for topo, edge, agree, err in test_cases:
        prob = compute_rim_probability(topo, edge, agree, err)
        if not (0.0 <= prob <= 1.0):
            print(f"  ✗ Invalid probability: {prob:.3f} for inputs ({topo}, {edge}, {agree}, {err})")
            all_valid = False
        else:
            print(f"  Inputs ({topo:.1f}, {edge:.1f}, {agree:.1f}, {err:.1f}) → probability = {prob:.3f}")

    if all_valid:
        print("✓ Success - All probabilities in valid range [0, 1]")
    else:
        print("✗ Some probabilities out of range")

except Exception as e:
    print(f"✗ Failed: {e}")

# Test 5: Module functions available
print("\n[Test 5] Checking available functions...")
try:
    from crater_analysis import refine_crater_rim
    functions = [
        'compute_edge_strength',
        'compute_topographic_quality',
        'compute_rim_probability',
        'refine_single_crater',
        'refine_crater_rims',
        'plot_refined_positions',
        'plot_csfd_refined',
        'plot_rim_differences'
    ]

    available = []
    missing = []

    for func_name in functions:
        if hasattr(refine_crater_rim, func_name):
            available.append(func_name)
        else:
            missing.append(func_name)

    print(f"  Available functions: {len(available)}/{len(functions)}")
    for func in available:
        print(f"    ✓ {func}")

    if missing:
        print(f"  Missing functions:")
        for func in missing:
            print(f"    ✗ {func}")

    if len(available) == len(functions):
        print("✓ Success - All functions available")

except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "=" * 70)
print("Testing Complete!")
print("=" * 70)

print("\nModule capabilities:")
print("  - Topographic rim refinement (from existing cratools)")
print("  - Computer vision edge detection (Canny, gradients)")
print("  - Combined probability scoring (0-1 scale)")
print("  - Quality metrics (topographic, edge strength, agreement)")
print("  - Four output products (shapefile, 3 PNGs)")

print("\nTo test full functionality with real data, run:")
print("  python refine_crater_rims.py --help")

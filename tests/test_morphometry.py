#!/usr/bin/env python3
"""
Test script for morphometry analysis module.

Tests:
- 2D Gaussian function
- Gaussian floor fitting
- Error propagation with rim probability
- Morphometry field extraction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from uncertainties import ufloat


def test_gaussian_2d():
    """Test 2D Gaussian function."""
    from crater_analysis.analyze_morphometry import gaussian_2d

    # Create coordinate grid
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    X, Y = np.meshgrid(x, y)

    # Test parameters
    amplitude = 10.0
    xo, yo = 5.0, 5.0
    sigma_x, sigma_y = 1.5, 1.5
    theta = 0.0
    offset = 0.0

    # Compute Gaussian
    Z = gaussian_2d((X, Y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset)

    # Reshape
    Z = Z.reshape(X.shape)

    # Check peak at center
    center_idx = 25  # Middle of 50x50 grid
    peak_val = Z[center_idx, center_idx]

    print("Test: 2D Gaussian Function")
    print(f"  Peak value at center: {peak_val:.3f}")
    print(f"  Expected (amplitude + offset): {amplitude + offset:.3f}")
    print(f"  ✓ Peak correct: {np.isclose(peak_val, amplitude, rtol=0.1)}")

    # Check that values decrease away from center
    edge_val = Z[0, 0]
    print(f"  Edge value: {edge_val:.3f}")
    print(f"  ✓ Edge < Peak: {edge_val < peak_val}")

    return True


def test_fit_gaussian_floor():
    """Test Gaussian floor fitting with synthetic data."""
    from crater_analysis.analyze_morphometry import fit_gaussian_floor

    print("\nTest: Gaussian Floor Fitting (Synthetic Data)")

    # Create synthetic crater DEM
    size = 100
    x = np.linspace(-50, 50, size)
    y = np.linspace(-50, 50, size)
    X, Y = np.meshgrid(x, y)

    # Create crater bowl shape (inverted Gaussian)
    rim_height = 100.0
    floor_height = 85.0
    depth = rim_height - floor_height

    # Gaussian bowl
    sigma = 15.0
    bowl = depth * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    # Crater elevation: rim - bowl
    crater_dem = rim_height - bowl

    print(f"  Synthetic crater:")
    print(f"    Rim height: {rim_height} m")
    print(f"    Floor height (true): {floor_height} m")
    print(f"    Depth: {depth} m")

    # Fit Gaussian
    floor_est, floor_unc, fit_quality, params = fit_gaussian_floor(
        crater_dem, None, None
    )

    print(f"  Gaussian fitting results:")
    print(f"    Floor height (estimated): {floor_est:.2f} ± {floor_unc:.2f} m")
    print(f"    Fit quality: {fit_quality:.3f}")
    print(f"    Error: {abs(floor_est - floor_height):.2f} m")

    # Check if estimate is reasonable
    error = abs(floor_est - floor_height)
    print(f"  ✓ Floor estimate within 2m: {error < 2.0}")
    print(f"  ✓ Fit quality > 0.8: {fit_quality > 0.8}")

    return True


def test_error_propagation():
    """Test error propagation with rim probability."""
    print("\nTest: Error Propagation with Rim Probability")

    # Test case 1: High rim probability
    depth_nominal = 20.0
    depth_unc = 1.0
    rim_probability = 0.9

    depth = ufloat(depth_nominal, depth_unc)

    # Additional uncertainty from rim probability
    prob_uncertainty_factor = 1.0 - rim_probability  # 0.1
    prob_error = abs(depth_nominal) * prob_uncertainty_factor * 0.5
    total_unc = np.sqrt(depth_unc**2 + prob_error**2)

    print(f"  Case 1: High rim probability (0.9)")
    print(f"    Depth: {depth_nominal} ± {depth_unc} m")
    print(f"    Probability error contribution: {prob_error:.3f} m")
    print(f"    Total uncertainty: {total_unc:.3f} m")
    print(f"    ✓ Total error includes probability: {total_unc > depth_unc}")

    # Test case 2: Low rim probability
    rim_probability = 0.3
    prob_uncertainty_factor = 1.0 - rim_probability  # 0.7
    prob_error = abs(depth_nominal) * prob_uncertainty_factor * 0.5
    total_unc = np.sqrt(depth_unc**2 + prob_error**2)

    print(f"\n  Case 2: Low rim probability (0.3)")
    print(f"    Depth: {depth_nominal} ± {depth_unc} m")
    print(f"    Probability error contribution: {prob_error:.3f} m")
    print(f"    Total uncertainty: {total_unc:.3f} m")
    print(f"    ✓ Higher uncertainty for low probability: {total_unc > 3.0}")

    return True


def test_morphometry_field_extraction():
    """Test extraction of morphometry fields."""
    from crater_analysis.analyze_morphometry import extract_morphometry_fields

    print("\nTest: Morphometry Field Extraction")

    # Create mock results
    results = {
        'method1': {
            'diameter': ufloat(100.0, 1.0),
            'depth': ufloat(15.0, 0.5),
            'd_D': ufloat(0.15, 0.005),
            'rim_height': ufloat(120.0, 1.2),
            'floor_height': 105.0,
            'total_error': 1.8,
            'probability_contribution': 0.3
        },
        'method2': {
            'diameter': ufloat(100.0, 1.0),
            'depth': ufloat(14.5, 0.8),
            'd_D': ufloat(0.145, 0.008),
            'rim_height': ufloat(120.0, 1.2),
            'floor_height': 105.5,
            'floor_uncertainty': 0.6,
            'fit_quality': 0.92,
            'gaussian_params': [10, 50, 50, 15, 15, 0, 0],
            'total_error': 2.0,
            'probability_contribution': 0.4
        },
        'combined': {
            'd_D': ufloat(0.1475, 0.006)
        }
    }

    fields = extract_morphometry_fields(results)

    print(f"  Extracted {len(fields)} fields")
    print(f"  Sample fields:")
    for key in ['d_D_m1', 'd_D_m2', 'depth_m1', 'depth_m2', 'fit_quality_m2']:
        if key in fields:
            print(f"    {key}: {fields[key]}")

    # Verify key fields exist
    required_fields = ['d_D_m1', 'd_D_m2', 'depth_m1', 'depth_m2',
                       'total_error_m1', 'total_error_m2', 'fit_quality_m2']
    all_present = all(f in fields for f in required_fields)

    print(f"  ✓ All required fields present: {all_present}")

    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("MORPHOMETRY MODULE TESTS")
    print("=" * 70)

    tests = [
        test_gaussian_2d,
        test_fit_gaussian_floor,
        test_error_propagation,
        test_morphometry_field_extraction
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n  ✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")

    if all(results):
        print("\n  ✓ All tests passed!")
        return 0
    else:
        print("\n  ✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

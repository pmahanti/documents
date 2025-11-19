#!/usr/bin/env python3
"""
Simple test script for Chebyshev coefficient extraction.

Tests that the normalization works correctly:
- Distance normalized by diameter: -D to +D maps to -1 to +1
- Elevation centered at 0 and normalized by diameter
"""

import numpy as np
import sys

def test_chebyshev_normalization():
    """Test Chebyshev coefficient normalization."""
    print("Testing Chebyshev Coefficient Normalization")
    print("=" * 60)

    try:
        from chebyshev_coefficients import ChebyshevProfileAnalyzer

        # Create test profile
        # Simulate a crater with diameter 100 pixels
        diameter = 100.0

        # Profile from -150 to +150 pixels (extends to 1.5D as expected)
        distance = np.linspace(-150, 150, 300)

        # Simple parabolic elevation profile (fresh crater)
        # Maximum depth at center, rim at ±50 pixels (±0.5D)
        elevation = -10 * (1 - (distance / 50)**2)  # Parabolic shape
        elevation = np.maximum(elevation, -10)  # Cap at -10m depth

        # Initialize analyzer
        analyzer = ChebyshevProfileAnalyzer(num_coefficients=17)

        # Test normalization
        print("\n1. Testing normalization function...")
        x_norm, y_norm, mean_elev, diameter_used = analyzer.normalize_profile(
            distance, elevation, diameter=diameter
        )

        print(f"   Original distance range: [{distance.min():.1f}, {distance.max():.1f}] pixels")
        print(f"   Normalized distance range: [{x_norm.min():.3f}, {x_norm.max():.3f}]")
        print(f"   Expected: [-1.500, 1.500] (extends to ±1.5D)")

        assert np.allclose(x_norm.min(), -1.5, atol=0.01), "Min normalized distance incorrect"
        assert np.allclose(x_norm.max(), 1.5, atol=0.01), "Max normalized distance incorrect"
        print("   ✓ Distance normalization correct")

        print(f"\n   Original elevation mean: {np.mean(elevation):.3f}")
        print(f"   Normalized elevation mean: {y_norm.mean():.6f}")
        print(f"   Expected: ~0.0 (centered)")

        assert np.abs(y_norm.mean()) < 0.1, "Elevation not properly centered"
        print("   ✓ Elevation centering correct")

        print(f"\n   Diameter used for normalization: {diameter_used:.1f} pixels")
        assert diameter_used == diameter, "Diameter not passed correctly"
        print("   ✓ Diameter parameter correct")

        # Test coefficient extraction
        print("\n2. Testing coefficient extraction...")
        test_profile = {
            'distance': distance,
            'elevation': elevation,
            'angle': 0
        }

        coeffs, metadata = analyzer.extract_coefficients_from_profile(
            distance, elevation, diameter=diameter
        )

        print(f"   Number of coefficients: {len(coeffs)}")
        print(f"   Valid points: {metadata['valid_points']}")
        print(f"   RMS error: {metadata['rms_error']:.6f}")

        assert len(coeffs) == 17, "Should extract 17 coefficients"
        print("   ✓ Coefficient count correct")

        # Test full profile list extraction
        print("\n3. Testing multiple profile extraction...")
        profiles = []
        for angle in range(0, 360, 45):  # 8 profiles
            profiles.append({
                'distance': distance,
                'elevation': elevation + np.random.randn(len(elevation)) * 0.5,  # Add noise
                'angle': angle
            })

        from chebyshev_coefficients import extract_chebyshev_coefficients

        coef_matrix, analysis, metadata_list = extract_chebyshev_coefficients(
            profiles, diameter=diameter, num_coefficients=17
        )

        print(f"   Coefficient matrix shape: {coef_matrix.shape}")
        print(f"   Expected shape: (17, 8)")

        assert coef_matrix.shape == (17, 8), "Matrix shape incorrect"
        print("   ✓ Matrix shape correct")

        print(f"\n   Mean coefficients (first 5):")
        for i in range(5):
            print(f"     C{i}: {analysis['mean_coefficients'][i]:.6f}")

        print(f"\n   Crater characteristics:")
        chars = analysis['crater_characteristics']
        print(f"     Depth indicator (C2): {chars['mean_depth_indicator']:.6f}")
        print(f"     Central peak index: {chars['central_peak_indicator']:.6f}")
        print(f"     Asymmetry index: {chars['asymmetry_index']:.6f}")
        print(f"     Profile consistency: {chars['profile_consistency']:.6f}")

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_chebyshev_normalization()
    sys.exit(0 if success else 1)

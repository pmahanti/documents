#!/usr/bin/env python3
"""
Verify specific claims from the paper with current cold trap model.

Paper claims to verify:
1. South has greater overall cold-trapping area (~23,000 km²) vs north (~17,000 km²)
2. North has more cold traps of size ~1 m–10 km
3. South has more cold traps of >10 km
4. About 2,500 km² of cold-trapping area exists in shadows smaller than 100 m
5. ~700 km² of cold-trapping area is contributed by shadows smaller than 1 m
"""

import numpy as np
import pandas as pd
from hayne_model_corrected import hayne_cold_trap_fraction_corrected

# Physical constants
LATERAL_CONDUCTION_LIMIT = 0.01  # 1 cm in meters
TRANSITION_SCALE = 1000.0  # 1 km
COLD_TRAP_THRESHOLD = 110.0  # K

# File path
PSR_CSV = '/home/user/documents/psr_with_temperatures.csv'


def load_psr_data():
    """Load PSR data with temperatures."""
    psr = pd.read_csv(PSR_CSV)
    psr['diameter_m'] = 2 * np.sqrt(psr['area_km2'] * 1e6 / np.pi)

    # Calculate cold trap area
    psr['coldtrap_fraction'] = 0.0
    mask = psr['pixel_count'] > 0
    psr.loc[mask, 'coldtrap_fraction'] = psr.loc[mask, 'pixels_lt_110K'] / psr.loc[mask, 'pixel_count']
    psr['coldtrap_area_km2'] = psr['coldtrap_fraction'] * psr['area_km2']

    # Calculate cold trap diameter
    psr['coldtrap_diameter_m'] = 0.0
    ct_mask = psr['coldtrap_area_km2'] > 0
    psr.loc[ct_mask, 'coldtrap_diameter_m'] = 2 * np.sqrt(
        psr.loc[ct_mask, 'coldtrap_area_km2'] * 1e6 / np.pi)

    return psr


def generate_synthetic_coldtraps(L_min=1e-4, L_max=TRANSITION_SCALE, n_bins=100):
    """Generate synthetic cold traps for small scales."""
    # Create logarithmic bins
    L_bins = np.logspace(np.log10(L_min), np.log10(L_max), n_bins)
    dL = np.diff(np.logspace(np.log10(L_min), np.log10(L_max), n_bins + 1))

    # Power-law distribution
    b = 1.8
    K = 2e11
    N_diff = K * L_bins**(-b - 1)
    N_per_bin = N_diff * dL

    # Hemisphere split
    N_north_bins = N_per_bin * 0.40
    N_south_bins = N_per_bin * 0.60

    # Representative latitudes
    lat_north = 85.0
    lat_south = -85.0

    # Terrain parameters
    sigma_s_plains = 5.7
    sigma_s_craters = 20.0
    f_craters = 0.20
    f_plains = 0.80

    # Generate cold traps
    coldtraps = []

    for i, L in enumerate(L_bins):
        # Skip if below lateral conduction limit
        if L < LATERAL_CONDUCTION_LIMIT:
            continue

        # Calculate cold trap fractions
        f_ct_crater_north = hayne_cold_trap_fraction_corrected(sigma_s_craters, lat_north)
        f_ct_crater_south = hayne_cold_trap_fraction_corrected(sigma_s_craters, lat_south)
        f_ct_plains_north = hayne_cold_trap_fraction_corrected(sigma_s_plains, lat_north)
        f_ct_plains_south = hayne_cold_trap_fraction_corrected(sigma_s_plains, lat_south)

        # Weighted average
        f_ct_north = f_craters * f_ct_crater_north + f_plains * f_ct_plains_north
        f_ct_south = f_craters * f_ct_crater_south + f_plains * f_ct_plains_south

        # Number of PSRs in this bin
        n_north = N_north_bins[i]
        n_south = N_south_bins[i]

        # Cold trap diameter and area
        D_ct_north = L * np.sqrt(f_ct_north)
        D_ct_south = L * np.sqrt(f_ct_south)

        A_ct_north = np.pi * (D_ct_north / 2.0)**2 * 1e-6  # km² per cold trap
        A_ct_south = np.pi * (D_ct_south / 2.0)**2 * 1e-6  # km² per cold trap

        # Add to list
        if n_north > 0 and A_ct_north > 0:
            coldtraps.append({
                'diameter_m': D_ct_north,
                'area_km2': A_ct_north * n_north,
                'hemisphere': 'North',
                'count': n_north
            })

        if n_south > 0 and A_ct_south > 0:
            coldtraps.append({
                'diameter_m': D_ct_south,
                'area_km2': A_ct_south * n_south,
                'hemisphere': 'South',
                'count': n_south
            })

    return coldtraps


def extract_observed_coldtraps(psr_data, min_diameter_m=TRANSITION_SCALE):
    """Extract observed cold traps from large PSRs."""
    large_psrs = psr_data[psr_data['diameter_m'] >= min_diameter_m].copy()
    large_psrs_with_ct = large_psrs[large_psrs['coldtrap_area_km2'] > 0].copy()

    coldtraps = []
    for _, row in large_psrs_with_ct.iterrows():
        coldtraps.append({
            'diameter_m': row['coldtrap_diameter_m'],
            'area_km2': row['coldtrap_area_km2'],
            'hemisphere': row['hemisphere'],
            'count': 1
        })

    return coldtraps


def analyze_by_size_range(coldtraps):
    """Analyze cold traps by size ranges mentioned in the paper."""

    # Convert to arrays for easier filtering
    north_cts = [ct for ct in coldtraps if ct['hemisphere'] == 'North']
    south_cts = [ct for ct in coldtraps if ct['hemisphere'] == 'South']

    # Define size ranges
    ranges = [
        ("< 1 m", 0, 1),
        ("1 m - 10 m", 1, 10),
        ("10 m - 100 m", 10, 100),
        ("100 m - 1 km", 100, 1000),
        ("1 km - 10 km", 1000, 10000),
        ("> 10 km", 10000, 1e10),
    ]

    print("\n" + "=" * 80)
    print("COLD TRAP AREA BY SIZE RANGE")
    print("=" * 80)

    for label, min_size, max_size in ranges:
        # Calculate areas for this range
        north_area = sum(ct['area_km2'] for ct in north_cts
                        if min_size <= ct['diameter_m'] < max_size)
        south_area = sum(ct['area_km2'] for ct in south_cts
                        if min_size <= ct['diameter_m'] < max_size)
        total_area = north_area + south_area

        # Count cold traps
        north_count = sum(ct['count'] for ct in north_cts
                         if min_size <= ct['diameter_m'] < max_size)
        south_count = sum(ct['count'] for ct in south_cts
                         if min_size <= ct['diameter_m'] < max_size)

        print(f"\n{label}:")
        print(f"  North: {north_area:,.2f} km² ({north_count:.2e} cold traps)")
        print(f"  South: {south_area:,.2f} km² ({south_count:.2e} cold traps)")
        print(f"  TOTAL: {total_area:,.2f} km²")
        if total_area > 0:
            print(f"  N/S ratio: {north_area/south_area:.3f}" if south_area > 0 else "  N/S ratio: N/A")

    # Overall totals
    total_north = sum(ct['area_km2'] for ct in north_cts)
    total_south = sum(ct['area_km2'] for ct in south_cts)

    print("\n" + "-" * 80)
    print("OVERALL TOTALS:")
    print(f"  North: {total_north:,.2f} km²")
    print(f"  South: {total_south:,.2f} km²")
    print(f"  TOTAL: {total_north + total_south:,.2f} km²")
    print(f"  S/N ratio: {total_south/total_north:.3f}")
    print("=" * 80)


def verify_paper_claims(coldtraps):
    """Check specific claims from the paper."""

    # Separate by hemisphere
    north_cts = [ct for ct in coldtraps if ct['hemisphere'] == 'North']
    south_cts = [ct for ct in coldtraps if ct['hemisphere'] == 'South']

    # Total areas
    total_north = sum(ct['area_km2'] for ct in north_cts)
    total_south = sum(ct['area_km2'] for ct in south_cts)

    # Area in shadows < 100m
    area_lt_100m = sum(ct['area_km2'] for ct in coldtraps if ct['diameter_m'] < 100)

    # Area in shadows < 1m
    area_lt_1m = sum(ct['area_km2'] for ct in coldtraps if ct['diameter_m'] < 1)

    # Count cold traps in different ranges
    north_1m_10km = sum(ct['count'] for ct in north_cts if 1 <= ct['diameter_m'] < 10000)
    south_1m_10km = sum(ct['count'] for ct in south_cts if 1 <= ct['diameter_m'] < 10000)

    north_gt_10km = sum(ct['count'] for ct in north_cts if ct['diameter_m'] >= 10000)
    south_gt_10km = sum(ct['count'] for ct in south_cts if ct['diameter_m'] >= 10000)

    # Find minimum cold trap size
    min_diameter = min(ct['diameter_m'] for ct in coldtraps)

    print("\n" + "=" * 80)
    print("VERIFICATION OF PAPER CLAIMS")
    print("=" * 80)

    print(f"\n1. OVERALL COLD TRAP AREAS:")
    print(f"   Paper claim: South ~23,000 km², North ~17,000 km²")
    print(f"   Model result: South {total_south:,.0f} km², North {total_north:,.0f} km²")
    print(f"   ✓ MATCH" if (20000 <= total_south <= 26000 and 15000 <= total_north <= 19000)
          else f"   ✗ MISMATCH")

    print(f"\n2. COLD TRAPS IN 1m-10km RANGE:")
    print(f"   Paper claim: North has more cold traps of size ~1m-10km")
    print(f"   Model result: North {north_1m_10km:.2e}, South {south_1m_10km:.2e}")
    print(f"   ✓ MATCH" if north_1m_10km > south_1m_10km else f"   ✗ MISMATCH")

    print(f"\n3. COLD TRAPS > 10km:")
    print(f"   Paper claim: South has more cold traps > 10km")
    print(f"   Model result: North {north_gt_10km:.2e}, South {south_gt_10km:.2e}")
    print(f"   ✓ MATCH" if south_gt_10km > north_gt_10km else f"   ✗ MISMATCH")

    print(f"\n4. AREA IN SHADOWS < 100m:")
    print(f"   Paper claim: About 2,500 km²")
    print(f"   Model result: {area_lt_100m:,.0f} km²")
    print(f"   ✓ MATCH" if 2000 <= area_lt_100m <= 3000 else f"   ✗ MISMATCH")

    print(f"\n5. AREA IN SHADOWS < 1m:")
    print(f"   Paper claim: ~700 km²")
    print(f"   Model result: {area_lt_1m:,.0f} km²")
    print(f"   ✓ MATCH" if 500 <= area_lt_1m <= 900 else f"   ✗ MISMATCH")

    print(f"\n6. MINIMUM COLD TRAP SIZE:")
    print(f"   Lateral conduction limit: {LATERAL_CONDUCTION_LIMIT*100:.1f} cm")
    print(f"   Smallest cold trap found: {min_diameter:.1f} m")
    print(f"   Note: Cannot validate claims about <1m shadows with {LATERAL_CONDUCTION_LIMIT*100:.0f}cm limit")

    print("\n" + "=" * 80)


def main():
    """Main analysis."""
    print("\n" + "=" * 80)
    print("VERIFYING PAPER CLAIMS WITH CURRENT MODEL")
    print(f"Lateral conduction limit: {LATERAL_CONDUCTION_LIMIT*100:.1f} cm")
    print("=" * 80)

    # Load data
    print("\n[Loading PSR data...]")
    psr_data = load_psr_data()

    # Generate synthetic cold traps
    print("\n[Generating synthetic cold traps...]")
    synthetic_cts = generate_synthetic_coldtraps()

    # Extract observed cold traps
    print("\n[Extracting observed cold traps...]")
    observed_cts = extract_observed_coldtraps(psr_data)

    # Combine
    all_coldtraps = synthetic_cts + observed_cts

    # Analyze by size range
    analyze_by_size_range(all_coldtraps)

    # Verify specific claims
    verify_paper_claims(all_coldtraps)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

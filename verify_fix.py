#!/usr/bin/env python3
"""
Verify the fix to diviner_direct.py

This script checks if the 941 km² discrepancy has been resolved.
"""

import numpy as np
from hayne_model_corrected import hayne_cold_trap_fraction_corrected

# Constants
LATERAL_CONDUCTION_LIMIT = 0.01
TRANSITION_SCALE = 1000.0

def generate_synthetic_area_method1():
    """Method 1: by_coldtrap_size.py approach - integrate to exactly 1000m"""
    L_bins = np.logspace(np.log10(1e-4), np.log10(1000), 100)
    dL = np.diff(np.logspace(np.log10(1e-4), np.log10(1000), 100 + 1))

    K = 2e11
    b = 1.8

    N_diff = K * L_bins**(-b - 1)
    N_per_bin = N_diff * dL

    N_north = N_per_bin * 0.40
    N_south = N_per_bin * 0.60

    A_north = np.zeros_like(L_bins)
    A_south = np.zeros_like(L_bins)

    lat_north = 85.0
    lat_south = -85.0
    sigma_s_plains = 5.7
    sigma_s_craters = 20.0
    f_craters = 0.20
    f_plains = 0.80

    for i, (L, n_n, n_s) in enumerate(zip(L_bins, N_north, N_south)):
        if L < LATERAL_CONDUCTION_LIMIT:
            continue

        f_ct_crater_north = hayne_cold_trap_fraction_corrected(sigma_s_craters, lat_north)
        f_ct_crater_south = hayne_cold_trap_fraction_corrected(sigma_s_craters, lat_south)
        f_ct_plains_north = hayne_cold_trap_fraction_corrected(sigma_s_plains, lat_north)
        f_ct_plains_south = hayne_cold_trap_fraction_corrected(sigma_s_plains, lat_south)

        f_ct_north = f_craters * f_ct_crater_north + f_plains * f_ct_plains_north
        f_ct_south = f_craters * f_ct_crater_south + f_plains * f_ct_plains_south

        area_per_feature = np.pi * (L / 2.0)**2
        A_north[i] = n_n * area_per_feature * f_ct_north * 1e-6
        A_south[i] = n_s * area_per_feature * f_ct_south * 1e-6

    total = A_north.sum() + A_south.sum()
    print(f"Method 1 (to 1000m exactly): {total:.2f} km²")
    print(f"  L_bins range: {L_bins[0]:.6f} to {L_bins[-1]:.2f} m")
    print(f"  Number of bins: {len(L_bins)}")
    return total


def generate_synthetic_area_method2_old():
    """Method 2 OLD: diviner_direct.py approach - integrate to L_bins[76] (~811m)"""
    L_bins_full = np.logspace(np.log10(1e-4), np.log10(100000), 100)
    transition_idx = np.searchsorted(L_bins_full, 1000)

    # OLD WAY: Use L_bins[transition_idx-1]
    L_max_old = L_bins_full[transition_idx-1]

    L_bins = np.logspace(np.log10(1e-4), np.log10(L_max_old), transition_idx)
    dL = np.diff(np.logspace(np.log10(1e-4), np.log10(L_max_old), transition_idx + 1))

    K = 2e11
    b = 1.8

    N_diff = K * L_bins**(-b - 1)
    N_per_bin = N_diff * dL

    N_north = N_per_bin * 0.40
    N_south = N_per_bin * 0.60

    A_north = np.zeros_like(L_bins)
    A_south = np.zeros_like(L_bins)

    lat_north = 85.0
    lat_south = -85.0
    sigma_s_plains = 5.7
    sigma_s_craters = 20.0
    f_craters = 0.20
    f_plains = 0.80

    for i, (L, n_n, n_s) in enumerate(zip(L_bins, N_north, N_south)):
        if L < LATERAL_CONDUCTION_LIMIT:
            continue

        f_ct_crater_north = hayne_cold_trap_fraction_corrected(sigma_s_craters, lat_north)
        f_ct_crater_south = hayne_cold_trap_fraction_corrected(sigma_s_craters, lat_south)
        f_ct_plains_north = hayne_cold_trap_fraction_corrected(sigma_s_plains, lat_north)
        f_ct_plains_south = hayne_cold_trap_fraction_corrected(sigma_s_plains, lat_south)

        f_ct_north = f_craters * f_ct_crater_north + f_plains * f_ct_plains_north
        f_ct_south = f_craters * f_ct_crater_south + f_plains * f_ct_plains_south

        area_per_feature = np.pi * (L / 2.0)**2
        A_north[i] = n_n * area_per_feature * f_ct_north * 1e-6
        A_south[i] = n_s * area_per_feature * f_ct_south * 1e-6

    total = A_north.sum() + A_south.sum()
    print(f"\nMethod 2 OLD (to {L_max_old:.2f}m): {total:.2f} km²")
    print(f"  L_bins range: {L_bins[0]:.6f} to {L_bins[-1]:.2f} m")
    print(f"  Number of bins: {len(L_bins)}")
    return total


def generate_synthetic_area_method2_new():
    """Method 2 NEW: diviner_direct.py approach - integrate to TRANSITION_SCALE (1000m)"""
    L_bins_full = np.logspace(np.log10(1e-4), np.log10(100000), 100)
    transition_idx = np.searchsorted(L_bins_full, 1000)

    # NEW WAY: Use TRANSITION_SCALE directly
    L_bins = np.logspace(np.log10(1e-4), np.log10(TRANSITION_SCALE), transition_idx)
    dL = np.diff(np.logspace(np.log10(1e-4), np.log10(TRANSITION_SCALE), transition_idx + 1))

    K = 2e11
    b = 1.8

    N_diff = K * L_bins**(-b - 1)
    N_per_bin = N_diff * dL

    N_north = N_per_bin * 0.40
    N_south = N_per_bin * 0.60

    A_north = np.zeros_like(L_bins)
    A_south = np.zeros_like(L_bins)

    lat_north = 85.0
    lat_south = -85.0
    sigma_s_plains = 5.7
    sigma_s_craters = 20.0
    f_craters = 0.20
    f_plains = 0.80

    for i, (L, n_n, n_s) in enumerate(zip(L_bins, N_north, N_south)):
        if L < LATERAL_CONDUCTION_LIMIT:
            continue

        f_ct_crater_north = hayne_cold_trap_fraction_corrected(sigma_s_craters, lat_north)
        f_ct_crater_south = hayne_cold_trap_fraction_corrected(sigma_s_craters, lat_south)
        f_ct_plains_north = hayne_cold_trap_fraction_corrected(sigma_s_plains, lat_north)
        f_ct_plains_south = hayne_cold_trap_fraction_corrected(sigma_s_plains, lat_south)

        f_ct_north = f_craters * f_ct_crater_north + f_plains * f_ct_plains_north
        f_ct_south = f_craters * f_ct_crater_south + f_plains * f_ct_plains_south

        area_per_feature = np.pi * (L / 2.0)**2
        A_north[i] = n_n * area_per_feature * f_ct_north * 1e-6
        A_south[i] = n_s * area_per_feature * f_ct_south * 1e-6

    total = A_north.sum() + A_south.sum()
    print(f"\nMethod 2 NEW (to 1000m exactly): {total:.2f} km²")
    print(f"  L_bins range: {L_bins[0]:.6f} to {L_bins[-1]:.2f} m")
    print(f"  Number of bins: {len(L_bins)}")
    print(f"  transition_idx: {transition_idx}")
    return total


def main():
    print("=" * 80)
    print("VERIFICATION: diviner_direct.py FIX")
    print("=" * 80)

    area1 = generate_synthetic_area_method1()
    area2_old = generate_synthetic_area_method2_old()
    area2_new = generate_synthetic_area_method2_new()

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    old_diff = abs(area1 - area2_old)
    new_diff = abs(area1 - area2_new)
    improvement = old_diff - new_diff

    print(f"\nOLD discrepancy: {old_diff:.2f} km²")
    print(f"NEW discrepancy: {new_diff:.2f} km²")
    print(f"Improvement: {improvement:.2f} km² ({improvement/old_diff*100:.1f}% reduction)")

    if new_diff < 5:
        print("\n✓ FIX SUCCESSFUL: Discrepancy < 5 km² (numerical precision)")
    else:
        print(f"\n⚠ REMAINING DISCREPANCY: {new_diff:.2f} km²")
        print("  This may be due to different bin counts (100 vs 77)")

    print("=" * 80)


if __name__ == "__main__":
    main()

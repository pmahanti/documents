#!/usr/bin/env python3
"""
Debug the discrepancy between the two approaches.
"""

import numpy as np
from hayne_model_corrected import hayne_cold_trap_fraction_corrected

LATERAL_CONDUCTION_LIMIT = 0.01  # 1 cm
TRANSITION_SCALE = 1000.0  # 1 km
K = 2e11
b = 1.8

# Terrain parameters
lat_north = 85.0
lat_south = -85.0
sigma_s_plains = 5.7
sigma_s_craters = 20.0
f_craters = 0.20
f_plains = 0.80

# Calculate cold trap fractions (constant for all bins at same latitude)
f_ct_crater_north = hayne_cold_trap_fraction_corrected(sigma_s_craters, lat_north)
f_ct_crater_south = hayne_cold_trap_fraction_corrected(sigma_s_craters, lat_south)
f_ct_plains_north = hayne_cold_trap_fraction_corrected(sigma_s_plains, lat_north)
f_ct_plains_south = hayne_cold_trap_fraction_corrected(sigma_s_plains, lat_south)

f_ct_north = f_craters * f_ct_crater_north + f_plains * f_ct_plains_north
f_ct_south = f_craters * f_ct_crater_south + f_plains * f_ct_plains_south

print("=" * 80)
print("COMPARING TWO APPROACHES")
print("=" * 80)

print(f"\nCold trap fractions:")
print(f"  North (85°N): {f_ct_north:.6f} ({f_ct_north*100:.4f}%)")
print(f"  South (85°S): {f_ct_south:.6f} ({f_ct_south*100:.4f}%)")

# APPROACH 1: by_coldtrap_size (100 bins from 1e-4 to 1000m)
print("\n" + "=" * 80)
print("APPROACH 1: by_coldtrap_size")
print("=" * 80)

L_bins_1 = np.logspace(np.log10(1e-4), np.log10(1000), 100)
dL_1 = np.diff(np.logspace(np.log10(1e-4), np.log10(1000), 101))

N_diff_1 = K * L_bins_1**(-b - 1)
N_per_bin_1 = N_diff_1 * dL_1
N_north_1 = N_per_bin_1 * 0.40
N_south_1 = N_per_bin_1 * 0.60

# Calculate areas (only for bins >= conduction limit)
A_north_1 = np.zeros_like(L_bins_1)
A_south_1 = np.zeros_like(L_bins_1)

for i, L in enumerate(L_bins_1):
    if L >= LATERAL_CONDUCTION_LIMIT:
        area_per_feature = np.pi * (L / 2.0)**2  # m²
        A_north_1[i] = N_north_1[i] * area_per_feature * f_ct_north * 1e-6  # km²
        A_south_1[i] = N_south_1[i] * area_per_feature * f_ct_south * 1e-6  # km²

print(f"Total bins: {len(L_bins_1)}")
print(f"Bins >= {LATERAL_CONDUCTION_LIMIT}m: {(L_bins_1 >= LATERAL_CONDUCTION_LIMIT).sum()}")
print(f"Range: {L_bins_1[0]:.6f} to {L_bins_1[-1]:.2f} m")
print(f"Total synthetic cold trap area North: {A_north_1.sum():.2f} km²")
print(f"Total synthetic cold trap area South: {A_south_1.sum():.2f} km²")
print(f"TOTAL: {A_north_1.sum() + A_south_1.sum():.2f} km²")

# APPROACH 2: diviner_direct (100 bins from 1e-4 to 100000m, but only use first 77)
print("\n" + "=" * 80)
print("APPROACH 2: diviner_direct")
print("=" * 80)

# Create full bin array
L_bins_full = np.logspace(np.log10(1e-4), np.log10(100000), 100)
transition_idx = np.searchsorted(L_bins_full, TRANSITION_SCALE)

print(f"Full range bins: {len(L_bins_full)}")
print(f"Transition index: {transition_idx}")
print(f"Bins < {TRANSITION_SCALE}m: {transition_idx}")

# Generate synthetic distribution for small bins
L_bins_2 = np.logspace(np.log10(L_bins_full[0]), np.log10(L_bins_full[transition_idx-1]), transition_idx)
dL_2 = np.diff(np.logspace(np.log10(L_bins_full[0]), np.log10(L_bins_full[transition_idx-1]), transition_idx + 1))

N_diff_2 = K * L_bins_2**(-b - 1)
N_per_bin_2 = N_diff_2 * dL_2
N_north_2 = N_per_bin_2 * 0.40
N_south_2 = N_per_bin_2 * 0.60

# Calculate areas
A_north_2 = np.zeros_like(L_bins_2)
A_south_2 = np.zeros_like(L_bins_2)

for i, L in enumerate(L_bins_2):
    if L >= LATERAL_CONDUCTION_LIMIT:
        area_per_feature = np.pi * (L / 2.0)**2  # m²
        A_north_2[i] = N_north_2[i] * area_per_feature * f_ct_north * 1e-6  # km²
        A_south_2[i] = N_south_2[i] * area_per_feature * f_ct_south * 1e-6  # km²

print(f"Synthetic bins: {len(L_bins_2)}")
print(f"Bins >= {LATERAL_CONDUCTION_LIMIT}m: {(L_bins_2 >= LATERAL_CONDUCTION_LIMIT).sum()}")
print(f"Range: {L_bins_2[0]:.6f} to {L_bins_2[-1]:.2f} m")
print(f"Total synthetic cold trap area North: {A_north_2.sum():.2f} km²")
print(f"Total synthetic cold trap area South: {A_south_2.sum():.2f} km²")
print(f"TOTAL: {A_north_2.sum() + A_south_2.sum():.2f} km²")

# COMPARISON
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

diff_north = A_north_1.sum() - A_north_2.sum()
diff_south = A_south_1.sum() - A_south_2.sum()
diff_total = diff_north + diff_south

print(f"\nDifference (Approach 1 - Approach 2):")
print(f"  North: {diff_north:+.2f} km² ({diff_north/A_north_1.sum()*100:+.2f}%)")
print(f"  South: {diff_south:+.2f} km² ({diff_south/A_south_1.sum()*100:+.2f}%)")
print(f"  TOTAL: {diff_total:+.2f} km² ({diff_total/(A_north_1.sum() + A_south_1.sum())*100:+.2f}%)")

# Check bin alignment
print("\n" + "=" * 80)
print("BIN ALIGNMENT CHECK")
print("=" * 80)

print(f"\nApproach 1 bins near transition ({TRANSITION_SCALE}m):")
idx_near = np.abs(L_bins_1 - TRANSITION_SCALE).argmin()
for i in range(max(0, idx_near-2), min(len(L_bins_1), idx_near+3)):
    print(f"  Bin {i}: L = {L_bins_1[i]:.2f} m")

print(f"\nApproach 2 max bin:")
print(f"  Bin {len(L_bins_2)-1}: L = {L_bins_2[-1]:.2f} m")
print(f"  (Should integrate up to ~{L_bins_full[transition_idx-1]:.2f} m)")

# Numerical integration check
print("\n" + "=" * 80)
print("ANALYTICAL CHECK")
print("=" * 80)

# The integral of cold trap area from L_min to L_max should be:
# ∫ K * L^(-b-1) * π*(L/2)^2 * f_ct * dL
# = K * π/4 * f_ct * ∫ L^(-b+1) dL
# = K * π/4 * f_ct * [L^(-b+2) / (-b+2)] from L_min to L_max
# = K * π/4 * f_ct / (-b+2) * (L_max^(-b+2) - L_min^(-b+2))

def analytical_integral(L_min, L_max, f_ct):
    """Calculate analytical integral of cold trap area."""
    exponent = -b + 2
    factor = K * np.pi / 4 * f_ct / exponent
    return factor * (L_max**exponent - L_min**exponent) * 1e-6  # km²

# Approach 1
A1_north_analytical = analytical_integral(LATERAL_CONDUCTION_LIMIT, 1000, f_ct_north)
A1_south_analytical = analytical_integral(LATERAL_CONDUCTION_LIMIT, 1000, f_ct_south)

# Approach 2
L_max_2 = L_bins_full[transition_idx-1]
A2_north_analytical = analytical_integral(LATERAL_CONDUCTION_LIMIT, L_max_2, f_ct_north)
A2_south_analytical = analytical_integral(LATERAL_CONDUCTION_LIMIT, L_max_2, f_ct_south)

print(f"\nAnalytical integrals:")
print(f"\nApproach 1 ({LATERAL_CONDUCTION_LIMIT} to {1000}m):")
print(f"  North: {A1_north_analytical:.2f} km²")
print(f"  South: {A1_south_analytical:.2f} km²")
print(f"  Total: {A1_north_analytical + A1_south_analytical:.2f} km²")

print(f"\nApproach 2 ({LATERAL_CONDUCTION_LIMIT} to {L_max_2:.2f}m):")
print(f"  North: {A2_north_analytical:.2f} km²")
print(f"  South: {A2_south_analytical:.2f} km²")
print(f"  Total: {A2_north_analytical + A2_south_analytical:.2f} km²")

print(f"\nDifference (Approach 1 - Approach 2):")
diff_analytical = (A1_north_analytical + A1_south_analytical) - (A2_north_analytical + A2_south_analytical)
print(f"  {diff_analytical:+.2f} km²")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print(f"\nThe discrepancy is due to different upper integration limits:")
print(f"  Approach 1 integrates to exactly {1000}m")
print(f"  Approach 2 integrates to {L_max_2:.2f}m (bin {transition_idx-1} of 100-bin grid)")
print(f"\nThis missing range ({L_max_2:.2f}m to {1000}m) accounts for ~{diff_analytical:.0f} km²")
print("=" * 80)

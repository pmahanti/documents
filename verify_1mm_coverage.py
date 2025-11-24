#!/usr/bin/env python3
"""
Verify that simulation properly covers 1 mm PSRs.
"""

import numpy as np

# Current parameters
L_min = 1e-4  # 0.1 mm
L_max = 100000  # 100 km
n_bins = 100

# Generate bins
L_bins = np.logspace(np.log10(L_min), np.log10(L_max), n_bins)

# Find bins around 1 mm (0.001 m)
target = 0.001  # 1 mm in meters

print("=" * 80)
print("VERIFICATION: Coverage around 1 mm scale")
print("=" * 80)

# Find closest bins to 1 mm
idx_1mm = np.argmin(np.abs(L_bins - target))

print(f"\nTarget: 1 mm = {target} m")
print(f"\nBins around 1 mm:")
print(f"  Index | L [m]        | L [mm]     | L [cm]")
print(f"  ------|--------------|------------|--------")

for i in range(max(0, idx_1mm-5), min(len(L_bins), idx_1mm+6)):
    L = L_bins[i]
    marker = " <-- 1mm" if i == idx_1mm else ""
    print(f"  {i:5d} | {L:12.6f} | {L*1000:10.4f} | {L*100:8.4f}{marker}")

print(f"\nClosest bin to 1 mm:")
print(f"  L = {L_bins[idx_1mm]:.6f} m = {L_bins[idx_1mm]*1000:.4f} mm")

# Check lateral conduction limit
lateral_limit = 0.01  # 1 cm
idx_lateral = np.argmin(np.abs(L_bins - lateral_limit))

print(f"\nLateral conduction limit: {lateral_limit} m = {lateral_limit*100} cm")
print(f"  Closest bin: {L_bins[idx_lateral]:.6f} m")
print(f"  Index: {idx_lateral}")

print(f"\n1 mm is at index {idx_1mm}, lateral limit is at index {idx_lateral}")
print(f"Bins between 1 mm and lateral limit: {idx_lateral - idx_1mm}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)

if L_bins[idx_1mm] < lateral_limit:
    print(f"✓ 1 mm ({L_bins[idx_1mm]*1000:.4f} mm) is BELOW lateral conduction limit ({lateral_limit*100} cm)")
    print(f"  PSRs at 1 mm scale are simulated but NO cold traps form (physics constraint)")
    print(f"  {idx_lateral - idx_1mm} bins between 1mm and lateral limit (all zero cold trap area)")
else:
    print(f"✓ 1 mm is ABOVE lateral conduction limit - cold traps CAN form")

print("\n" + "=" * 80)

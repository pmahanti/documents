#!/usr/bin/env python3
"""
Test script to find projectile sizes for 100-500m craters.
"""

from lunar_impact_simulation import *

# Target parameters
target = TargetParameters()
scaling = CraterScalingLaws(target)

print("Finding projectile sizes for 100-500m lunar craters")
print("="*60)

# Test different projectile diameters
for proj_diameter in [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
    proj = ProjectileParameters(
        diameter=proj_diameter,
        velocity=20000,  # 20 km/s typical
        angle=90,
        density=2800,
        material_type='rocky'
    )

    D = scaling.final_crater_diameter(proj)
    d = scaling.crater_depth(proj)

    pi2 = scaling.pi_2_gravity(proj)
    pi3 = scaling.pi_3_strength(proj)

    if pi3 < pi2:
        regime = "Strength"
    elif pi3 > 10 * pi2:
        regime = "Gravity"
    else:
        regime = "Transitional"

    print(f"Projectile {proj_diameter:4.1f}m → Crater D={D:6.1f}m, d={d:5.1f}m, d/D={d/D:.3f} [{regime}]")

print("\n" + "="*60)
print("Target range: 100-500m craters")
print("Recommended projectile sizes: 2-5m diameter at 20 km/s")

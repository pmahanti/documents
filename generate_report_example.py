"""
Example script demonstrating how to generate impact crater analysis reports.

This script shows various ways to use the impact_report_generator module
to create comprehensive PDF reports for lunar (and other planetary) craters.
"""

from impact_report_generator import generate_impact_report

# Example 1: Simple lunar crater with minimal inputs
print("=" * 70)
print("EXAMPLE 1: Small Fresh Lunar Crater")
print("=" * 70)

report1 = generate_impact_report(
    diameter=500,                    # Crater diameter in meters
    depth=100,                       # Crater depth in meters
    velocity=15000,                  # Impact velocity in m/s (15 km/s)
    target_material='lunar_regolith', # Target material
    impactor_material='asteroid_rock', # Impactor type
    latitude=-89.5,                  # Latitude in decimal degrees
    longitude=45.2,                  # Longitude in decimal degrees
    crater_name="Fresh Crater A",    # Optional name
    output_filename='lunar_crater_500m.pdf'
)

print(f"Report generated: {report1}\n")


# Example 2: Large complex crater with uncertainties
print("=" * 70)
print("EXAMPLE 2: Large Complex Lunar Crater with Uncertainties")
print("=" * 70)

report2 = generate_impact_report(
    diameter=10000,                   # 10 km diameter
    depth=1000,                       # 1 km depth
    velocity=20000,                   # 20 km/s
    target_material='lunar_mare',     # Mare basalt target
    impactor_material='asteroid_rock',
    latitude=-85.3,
    longitude=120.5,
    diameter_uncertainty=100,         # ±100 m uncertainty
    depth_uncertainty=50,             # ±50 m uncertainty
    velocity_uncertainty=3000,        # ±3 km/s uncertainty
    crater_type='complex',            # Complex crater
    crater_name="Large Impact Basin",
    output_filename='lunar_basin_10km.pdf'
)

print(f"Report generated: {report2}\n")


# Example 3: Small crater with metallic impactor
print("=" * 70)
print("EXAMPLE 3: Small Crater from Iron Meteorite")
print("=" * 70)

report3 = generate_impact_report(
    diameter=150,
    depth=30,
    velocity=12000,                   # Lower velocity
    target_material='lunar_highland',  # Highland anorthosite
    impactor_material='asteroid_metal', # Iron meteorite
    latitude=-82.1,
    longitude=15.7,
    diameter_uncertainty=5,
    depth_uncertainty=2,
    velocity_uncertainty=1000,
    crater_name="Metallic Impactor Crater",
    output_filename='iron_meteorite_crater.pdf'
)

print(f"Report generated: {report3}\n")


# Example 4: Polar crater (possible ice target)
print("=" * 70)
print("EXAMPLE 4: Polar Crater in Ice-Rich Target")
print("=" * 70)

report4 = generate_impact_report(
    diameter=800,
    depth=160,
    velocity=18000,
    target_material='ice_regolith_mix',  # Ice-regolith mixture
    impactor_material='comet_ice',       # Cometary impactor
    latitude=-88.9,                      # Near south pole
    longitude=0.0,
    diameter_uncertainty=20,
    depth_uncertainty=10,
    velocity_uncertainty=2000,
    crater_name="South Pole Crater",
    output_filename='polar_ice_crater.pdf'
)

print(f"Report generated: {report4}\n")


# Example 5: Using coordinates from your dataset
print("=" * 70)
print("EXAMPLE 5: Crater from Your Lunar Dataset")
print("=" * 70)

# Example coordinates from a Shadowcam observation
# (Replace these with actual coordinates from your fnnames_sslatlon_time.xlsx)

report5 = generate_impact_report(
    diameter=250,                     # Measured from imagery
    depth=50,                         # Measured from DEM
    velocity=16000,                   # Typical lunar impact
    target_material='lunar_regolith',
    impactor_material='asteroid_rock',
    latitude=-89.2,                   # From your dataset
    longitude=33.8,                   # From your dataset
    diameter_uncertainty=10,          # Based on image resolution
    depth_uncertainty=5,              # Based on DEM accuracy
    velocity_uncertainty=2000,        # Standard assumption
    crater_name="Shadowcam Observation M012345678",
    output_filename='shadowcam_crater_report.pdf'
)

print(f"Report generated: {report5}\n")


# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nAll reports generated successfully!")
print("\nGenerated files:")
print("  1. lunar_crater_500m.pdf")
print("  2. lunar_basin_10km.pdf")
print("  3. iron_meteorite_crater.pdf")
print("  4. polar_ice_crater.pdf")
print("  5. shadowcam_crater_report.pdf")
print("\nEach report contains:")
print("  • Page 1: Summary with impact properties and uncertainties")
print("  • Page 2+: Detailed theoretical explanations")
print("  • All equations with substituted values")
print("  • Excavation depth and ejecta calculations")
print("  • Assumptions and limitations")
print("\nCustomize the parameters above for your specific craters!")
print("=" * 70)

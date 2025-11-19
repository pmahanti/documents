"""
Geological timescale simulation of elephant hide formation.

This example demonstrates how elephant hide textures form over millions
of years through thermal cycling and seismic shaking, starting from a
fresh crater with smooth regolith.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lunar_regolith_sim import (
    RegolithPhysics,
    SlopeGeometry,
    LunarThermalCycle,
    MoonquakeSimulator,
    GeologicalRegolithSimulation,
    SimulationVisualizer
)
import matplotlib.pyplot as plt
import numpy as np


def main():
    print("=" * 70)
    print("Geological Timescale Simulation: Elephant Hide Formation")
    print("=" * 70)
    print("\nModeling slow, cumulative downslope creep over millions of years")
    print("driven by thermal cycling and moonquakes.\n")

    # === CREATE CRATER WALL GEOMETRY ===
    print("1. Creating crater wall geometry...")
    slope = SlopeGeometry(width=200, height=200, resolution=1.0)

    # Create crater with appropriate slopes for texture formation
    slope.create_crater_wall(
        crater_x=100,
        crater_y=100,
        inner_radius=40,      # Crater floor
        outer_radius=85,      # Crater rim
        rim_height=20,        # Rim elevation
        floor_depth=10        # Floor depression
    )

    # Add initial surface roughness (impact-generated)
    slope.add_roughness(amplitude=0.5, wavelength=4.0, smoothing=1.5)

    print(f"   Grid size: {slope.nx} x {slope.ny} cells")
    print(f"   Resolution: {slope.resolution} m/cell")

    # === INITIALIZE PHYSICS ===
    print("\n2. Initializing physics with realistic parameters...")
    physics = RegolithPhysics(
        porosity=0.45,                 # 45% porosity (loose regolith)
        cohesion=0.5,                  # 0.5 kPa (very low cohesion)
        internal_friction_angle=37.5,  # 35-40° range
        grain_size=60e-6              # 60 μm median grain size
    )

    print(f"   Porosity: {physics.porosity*100:.0f}%")
    print(f"   Cohesion: {physics.cohesion} kPa")
    print(f"   Internal friction: {physics.internal_friction_angle}°")
    print(f"   Grain size: {physics.median_grain_size*1e6:.0f} μm")

    # === INITIALIZE THERMAL CYCLE ===
    print("\n3. Initializing lunar thermal cycle...")
    thermal = LunarThermalCycle(
        temp_max=400,   # 400 K daytime
        temp_min=100,   # 100 K nighttime
        latitude=0.0    # Equatorial (maximum temperature variation)
    )

    print(f"   Day temperature: {thermal.temp_max} K")
    print(f"   Night temperature: {thermal.temp_min} K")
    print(f"   Cycle period: {thermal.period/(24*3600):.1f} Earth days")

    # === INITIALIZE MOONQUAKE SIMULATOR ===
    print("\n4. Initializing moonquake simulator...")
    moonquakes = MoonquakeSimulator(
        quake_rate_multiplier=1.0,  # Typical rate
        seed=42                      # Reproducible
    )

    print(f"   Deep quake rate: {moonquakes.DEEP_QUAKE_RATE} events/year")
    print(f"   Shallow quake rate: {moonquakes.SHALLOW_QUAKE_RATE} events/year")
    print(f"   Magnitude range: {moonquakes.TYPICAL_MAGNITUDE_RANGE}")

    # === CREATE GEOLOGICAL SIMULATION ===
    print("\n5. Creating geological simulation...")
    geo_sim = GeologicalRegolithSimulation(
        slope_geometry=slope,
        physics=physics,
        thermal_cycle=thermal,
        moonquake_sim=moonquakes,
        initial_thickness=2.0  # 2 m initial regolith
    )

    # === SIMULATE FRESH CRATER ===
    print("\n6. Simulating fresh crater (t=0)...")
    fresh_state = geo_sim.simulate_fresh_crater(show_initial=True)

    # === ADVANCE GEOLOGICAL TIME ===
    print("\n7. Advancing geological time...")
    duration_million_years = 3.0  # 3 million years
    duration_years = duration_million_years * 1e6

    final_state = geo_sim.advance_geological_time(
        duration_years,
        progress_interval_years=duration_years/10
    )

    # === COMPARE FRESH VS AGED ===
    print("\n8. Comparing fresh vs. aged crater...")
    comparison = geo_sim.compare_fresh_vs_aged()

    print(f"\n   Fresh crater:")
    print(f"     Mean texture: {comparison['fresh_mean_texture']:.3f}")
    print(f"\n   Aged crater ({duration_million_years} Myr):")
    print(f"     Mean texture: {comparison['aged_mean_texture']:.3f}")
    print(f"     Max texture: {comparison['aged_max_texture']:.3f}")
    print(f"     Cells with texture: {comparison['slopes_with_texture']}")
    print(f"     Total displacement: {comparison['total_displacement']:.3e} m")

    # === VISUALIZATION ===
    print("\n9. Creating visualizations...")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'geological_timescale')
    os.makedirs(output_dir, exist_ok=True)

    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Elevation
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(slope.elevation, cmap='terrain', extent=[0, slope.width, 0, slope.height])
    ax1.set_title('Crater Topography')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Distance (m)')
    plt.colorbar(im1, ax=ax1, label='Elevation (m)')

    # 2. Slope angles
    ax2 = fig.add_subplot(gs[0, 1])
    slope_angles = fresh_state['slope_angles']
    im2 = ax2.imshow(slope_angles, cmap='hot', extent=[0, slope.width, 0, slope.height])
    ax2.contour(slope_angles, levels=[8, 15, 25], colors=['cyan', 'lime', 'yellow'],
               linewidths=2, linestyles='--', extent=[0, slope.width, 0, slope.height])
    ax2.set_title('Slope Angles (8°, 15°, 25° contours)')
    ax2.set_xlabel('Distance (m)')
    plt.colorbar(im2, ax=ax2, label='Slope (degrees)')

    # 3. Texture formation potential
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(fresh_state['texture_potential'], cmap='YlOrRd',
                    extent=[0, slope.width, 0, slope.height])
    ax3.set_title('Texture Formation Potential')
    ax3.set_xlabel('Distance (m)')
    plt.colorbar(im3, ax=ax3, label='Potential (0-1)')

    # 4. Fresh crater (smooth)
    ax4 = fig.add_subplot(gs[1, 0])
    fresh_texture = np.zeros_like(slope_angles)
    im4 = ax4.imshow(fresh_texture, cmap='bone', extent=[0, slope.width, 0, slope.height])
    ax4.set_title('Fresh Crater (t=0)')
    ax4.set_xlabel('Distance (m)')
    ax4.set_ylabel('Distance (m)')
    plt.colorbar(im4, ax=ax4, label='Texture Intensity')

    # 5. Aged crater (textured)
    ax5 = fig.add_subplot(gs[1, 1])
    aged_texture = final_state['texture_intensity']
    im5 = ax5.imshow(aged_texture, cmap='bone', extent=[0, slope.width, 0, slope.height])
    ax5.set_title(f'Aged Crater (t={duration_million_years:.1f} Myr)')
    ax5.set_xlabel('Distance (m)')
    plt.colorbar(im5, ax=ax5, label='Texture Intensity')

    # 6. Texture difference
    ax6 = fig.add_subplot(gs[1, 2])
    texture_diff = aged_texture - fresh_texture
    im6 = ax6.imshow(texture_diff, cmap='viridis', extent=[0, slope.width, 0, slope.height])
    ax6.set_title('Texture Development')
    ax6.set_xlabel('Distance (m)')
    plt.colorbar(im6, ax=ax6, label='Intensity Change')

    # 7. Cumulative displacement
    ax7 = fig.add_subplot(gs[2, 0])
    im7 = ax7.imshow(final_state['displacement'], cmap='plasma',
                    extent=[0, slope.width, 0, slope.height])
    ax7.set_title('Cumulative Displacement')
    ax7.set_xlabel('Distance (m)')
    ax7.set_ylabel('Distance (m)')
    plt.colorbar(im7, ax=ax7, label='Displacement (m)')

    # 8. Slope vs texture scatter plot
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.scatter(slope_angles.flatten(), aged_texture.flatten(),
               alpha=0.1, s=1, c='steelblue')
    ax8.axvline(8, color='cyan', linestyle='--', label='8° threshold')
    ax8.axvline(15, color='lime', linestyle='--', label='15° optimal min')
    ax8.axvline(25, color='yellow', linestyle='--', label='25° optimal max')
    ax8.set_xlabel('Slope Angle (degrees)')
    ax8.set_ylabel('Texture Intensity')
    ax8.set_title('Slope-Texture Relationship')
    ax8.legend()
    ax8.grid(alpha=0.3)

    # 9. Statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    stats_text = f"""
    SIMULATION SUMMARY

    Duration: {duration_million_years:.1f} Myr
    Thermal cycles: {final_state['num_thermal_cycles']:.2e}
    Seismic events: {final_state['num_seismic_events']}

    REGOLITH PROPERTIES
    Porosity: {physics.porosity*100:.0f}%
    Cohesion: {physics.cohesion} kPa
    Grain size: {physics.median_grain_size*1e6:.0f} μm

    THERMAL CYCLING
    Day temp: {thermal.temp_max} K
    Night temp: {thermal.temp_min} K
    ΔT: {thermal.temp_max - thermal.temp_min} K

    RESULTS
    Max displacement: {np.max(final_state['displacement']):.2e} m
    Mean texture: {aged_texture.mean():.3f}
    Textured area: {100*np.sum(aged_texture>0.3)/aged_texture.size:.1f}%
    """
    ax9.text(0.1, 0.5, stats_text, transform=ax9.transAxes,
            fontsize=9, verticalalignment='center', fontfamily='monospace')

    fig.suptitle(f'Elephant Hide Formation: Fresh vs. {duration_million_years:.1f} Myr Aged Crater',
                fontsize=14, fontweight='bold')

    plt.savefig(os.path.join(output_dir, 'geological_timescale.png'), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/geological_timescale.png")

    plt.close()

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

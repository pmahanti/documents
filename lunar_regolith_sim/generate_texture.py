#!/usr/bin/env python3
"""
Standalone script to generate elephant hide textures with custom parameters.

Usage:
    python generate_texture.py --slope 20 --duration 3.0 --porosity 0.45
    python generate_texture.py --help
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from lunar_regolith_sim import (
    RegolithPhysics,
    SlopeGeometry,
    LunarThermalCycle,
    MoonquakeSimulator,
    GeologicalRegolithSimulation
)
import matplotlib.pyplot as plt
import numpy as np


def generate_texture(
    slope_angle=20.0,
    porosity=0.45,
    cohesion=0.5,
    grain_size_um=60,
    temp_max=400,
    temp_min=100,
    duration_myr=3.0,
    output_path='texture_output.png',
    geometry='linear',
    domain_size=100,
    resolution=1.0,
    show_plot=False
):
    """
    Generate elephant hide texture with specified parameters.

    Args:
        slope_angle: Slope angle in degrees (optimal: 15-25°)
        porosity: Regolith porosity 0-1 (typical: 0.45)
        cohesion: Cohesion in kPa (typical: 0.1-1.0)
        grain_size_um: Grain size in micrometers (typical: 40-800)
        temp_max: Max temperature in K (typical: 400)
        temp_min: Min temperature in K (typical: 100)
        duration_myr: Evolution time in million years
        output_path: Output image path
        geometry: 'linear' or 'crater'
        domain_size: Domain size in meters
        resolution: Grid resolution in meters/cell
        show_plot: Whether to display plot

    Returns:
        dict: Simulation results
    """
    print("=" * 70)
    print("Elephant Hide Texture Generator")
    print("=" * 70)
    print("\nParameters:")
    print(f"  Slope angle: {slope_angle}°")
    print(f"  Porosity: {porosity*100:.0f}%")
    print(f"  Cohesion: {cohesion} kPa")
    print(f"  Grain size: {grain_size_um} μm")
    print(f"  Temperature range: {temp_min}-{temp_max} K (ΔT={temp_max-temp_min} K)")
    print(f"  Duration: {duration_myr} Myr")
    print(f"  Geometry: {geometry}")

    # Create slope geometry
    print("\n1. Creating geometry...")
    slope = SlopeGeometry(width=domain_size, height=domain_size, resolution=resolution)

    if geometry == 'linear':
        slope.create_linear_slope(angle=slope_angle, direction='y')
        slope.add_roughness(amplitude=0.3, wavelength=5.0)
    elif geometry == 'crater':
        slope.create_crater_wall(
            crater_x=domain_size/2,
            crater_y=domain_size/2,
            inner_radius=domain_size*0.2,
            outer_radius=domain_size*0.42,
            rim_height=domain_size*0.1,
            floor_depth=domain_size*0.05
        )
        slope.add_roughness(amplitude=0.5, wavelength=4.0)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    print(f"   Grid: {slope.nx}x{slope.ny} cells")

    # Initialize physics
    print("\n2. Initializing physics...")
    physics = RegolithPhysics(
        porosity=porosity,
        cohesion=cohesion,
        internal_friction_angle=37.5,
        grain_size=grain_size_um * 1e-6
    )

    # Thermal cycle
    print("\n3. Initializing thermal cycle...")
    thermal = LunarThermalCycle(temp_max=temp_max, temp_min=temp_min)

    # Moonquakes
    print("\n4. Initializing moonquakes...")
    moonquakes = MoonquakeSimulator(seed=42)

    # Create simulation
    print("\n5. Creating simulation...")
    geo_sim = GeologicalRegolithSimulation(
        slope_geometry=slope,
        physics=physics,
        thermal_cycle=thermal,
        moonquake_sim=moonquakes,
        initial_thickness=2.0
    )

    # Simulate fresh crater
    print("\n6. Simulating fresh crater...")
    fresh = geo_sim.simulate_fresh_crater(show_initial=False)

    # Evolve over time
    print(f"\n7. Evolving over {duration_myr} Myr...")
    result = geo_sim.advance_geological_time(duration_myr * 1e6)

    # Get texture
    texture = result['texture_intensity']
    slope_angles = result['slope_angles']

    # Create visualization
    print("\n8. Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    extent = [0, domain_size, 0, domain_size]

    # 1. Topography
    ax = axes[0, 0]
    im = ax.imshow(slope.elevation, cmap='terrain', extent=extent)
    ax.set_title('Topography')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance (m)')
    plt.colorbar(im, ax=ax, label='Elevation (m)')

    # 2. Slope angles
    ax = axes[0, 1]
    im = ax.imshow(slope_angles, cmap='hot', extent=extent)
    ax.contour(slope_angles, levels=[8, 15, 25], colors=['cyan', 'lime', 'yellow'],
              linewidths=2, linestyles='--', extent=extent)
    ax.set_title('Slope Angles')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance (m)')
    plt.colorbar(im, ax=ax, label='Angle (degrees)')

    # 3. Texture potential
    ax = axes[0, 2]
    potential = physics.get_texture_formation_intensity(slope_angles)
    im = ax.imshow(potential, cmap='YlOrRd', extent=extent, vmin=0, vmax=1)
    ax.set_title('Formation Potential')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance (m)')
    plt.colorbar(im, ax=ax, label='Potential')

    # 4. Fresh (smooth)
    ax = axes[1, 0]
    im = ax.imshow(np.zeros_like(texture), cmap='bone', extent=extent, vmin=0, vmax=1)
    ax.set_title('Fresh (t=0)')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance (m)')
    plt.colorbar(im, ax=ax, label='Texture')

    # 5. Aged (textured)
    ax = axes[1, 1]
    im = ax.imshow(texture, cmap='bone', extent=extent, vmin=0, vmax=1)
    ax.set_title(f'Aged (t={duration_myr} Myr)')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance (m)')
    plt.colorbar(im, ax=ax, label='Texture')

    # 6. Statistics
    ax = axes[1, 2]
    ax.axis('off')
    stats = f"""
SIMULATION RESULTS

Duration: {duration_myr} Myr
Thermal cycles: {result['num_thermal_cycles']:.2e}
Moonquakes: {result['num_seismic_events']:,}

REGOLITH PROPERTIES
Porosity: {porosity*100:.0f}%
Cohesion: {cohesion} kPa
Grain size: {grain_size_um} μm
Internal friction: {physics.internal_friction_angle}°

THERMAL CYCLING
Max temp: {temp_max} K
Min temp: {temp_min} K
ΔT: {temp_max - temp_min} K

TEXTURE METRICS
Mean intensity: {texture.mean():.3f}
Max intensity: {texture.max():.3f}
Textured area: {100*np.sum(texture>0.3)/texture.size:.1f}%

DISPLACEMENT
Max: {result['displacement'].max():.2e} m
Mean: {result['displacement'].mean():.2e} m
    """
    ax.text(0.05, 0.5, stats, transform=ax.transAxes, fontsize=9,
           verticalalignment='center', fontfamily='monospace')

    fig.suptitle('Elephant Hide Texture Formation Simulation',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    print(f"\n9. Saving to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Mean texture intensity: {texture.mean():.3f}")
    print(f"Max texture intensity: {texture.max():.3f}")
    print(f"Textured area (>0.3): {100*np.sum(texture>0.3)/texture.size:.1f}%")
    print(f"Max displacement: {result['displacement'].max():.2e} m")
    print(f"Output saved to: {output_path}")
    print("=" * 70)

    return {
        'texture': texture,
        'slope_angles': slope_angles,
        'displacement': result['displacement'],
        'result': result,
        'physics': physics
    }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Generate elephant hide texture simulations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Slope parameters
    parser.add_argument('--slope', type=float, default=20.0,
                       help='Slope angle in degrees (optimal: 15-25)')

    # Regolith parameters
    parser.add_argument('--porosity', type=float, default=0.45,
                       help='Porosity 0-1 (typical: 0.45)')
    parser.add_argument('--cohesion', type=float, default=0.5,
                       help='Cohesion in kPa (typical: 0.1-1.0)')
    parser.add_argument('--grain-size', type=float, default=60,
                       help='Grain size in micrometers (typical: 40-800)')

    # Thermal parameters
    parser.add_argument('--temp-max', type=float, default=400,
                       help='Maximum temperature in K')
    parser.add_argument('--temp-min', type=float, default=100,
                       help='Minimum temperature in K')

    # Simulation parameters
    parser.add_argument('--duration', type=float, default=3.0,
                       help='Evolution time in million years')
    parser.add_argument('--geometry', choices=['linear', 'crater'], default='linear',
                       help='Slope geometry type')
    parser.add_argument('--domain-size', type=float, default=100,
                       help='Domain size in meters')
    parser.add_argument('--resolution', type=float, default=1.0,
                       help='Grid resolution in meters/cell')

    # Output parameters
    parser.add_argument('--output', type=str, default='texture_output.png',
                       help='Output image path')
    parser.add_argument('--show', action='store_true',
                       help='Display plot window')

    args = parser.parse_args()

    # Generate texture
    result = generate_texture(
        slope_angle=args.slope,
        porosity=args.porosity,
        cohesion=args.cohesion,
        grain_size_um=args.grain_size,
        temp_max=args.temp_max,
        temp_min=args.temp_min,
        duration_myr=args.duration,
        output_path=args.output,
        geometry=args.geometry,
        domain_size=args.domain_size,
        resolution=args.resolution,
        show_plot=args.show
    )

    return result


if __name__ == "__main__":
    main()

"""
Generate animated elephant hide texture evolution over geological timescales.

This script creates an animation showing how elephant hide textures develop
from a fresh, smooth crater to a mature, textured surface over millions of years.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lunar_regolith_sim import (
    RegolithPhysics,
    SlopeGeometry,
    LunarThermalCycle,
    MoonquakeSimulator,
    GeologicalRegolithSimulation
)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np


class TextureEvolutionAnimator:
    """
    Create animations of elephant hide texture evolution.
    """

    def __init__(self, slope_geometry, physics, thermal_cycle, moonquake_sim):
        """
        Initialize animator.

        Args:
            slope_geometry: SlopeGeometry object
            physics: RegolithPhysics object
            thermal_cycle: LunarThermalCycle object
            moonquake_sim: MoonquakeSimulator object
        """
        self.slope = slope_geometry
        self.physics = physics
        self.thermal = thermal_cycle
        self.moonquakes = moonquake_sim

        # Create simulation
        self.sim = GeologicalRegolithSimulation(
            slope_geometry=slope_geometry,
            physics=physics,
            thermal_cycle=thermal_cycle,
            moonquake_sim=moonquake_sim,
            initial_thickness=2.0
        )

        # Storage for animation frames
        self.frames = []
        self.times = []

    def generate_evolution_sequence(self, total_duration_myr, num_snapshots=20):
        """
        Generate sequence of texture states over time.

        Args:
            total_duration_myr: Total duration in million years
            num_snapshots: Number of snapshots to generate

        Returns:
            list: List of texture states at different times
        """
        print(f"\nGenerating {num_snapshots} snapshots over {total_duration_myr} Myr...")

        # Time points (logarithmic spacing for better coverage)
        time_points_myr = np.logspace(-2, np.log10(total_duration_myr), num_snapshots)

        # Initial fresh crater
        print("\n  t = 0.00 Myr (Fresh crater)")
        fresh_state = self.sim.simulate_fresh_crater(show_initial=False)
        self.frames.append({
            'time_myr': 0.0,
            'texture': np.zeros_like(fresh_state['slope_angles']),
            'slope_angles': fresh_state['slope_angles'],
            'displacement': np.zeros_like(fresh_state['slope_angles']),
            'num_cycles': 0,
            'num_quakes': 0
        })
        self.times.append(0.0)

        # Generate snapshots at each time point
        prev_time = 0.0
        for i, time_myr in enumerate(time_points_myr):
            duration_years = (time_myr - prev_time) * 1e6

            print(f"  t = {time_myr:.2f} Myr (simulating {duration_years:.2e} years)")

            # Advance simulation
            result = self.sim.advance_geological_time(duration_years)

            # Store frame
            self.frames.append({
                'time_myr': time_myr,
                'texture': result['texture_intensity'].copy(),
                'slope_angles': result['slope_angles'].copy(),
                'displacement': result['displacement'].copy(),
                'num_cycles': result['num_thermal_cycles'],
                'num_quakes': result['num_seismic_events']
            })
            self.times.append(time_myr)

            prev_time = time_myr

        print(f"\n  Generated {len(self.frames)} frames total")
        return self.frames

    def create_animation(self, output_path, fps=2, dpi=150):
        """
        Create animation from generated frames.

        Args:
            output_path: Path to save animation
            fps: Frames per second
            dpi: Resolution

        Returns:
            matplotlib.animation.FuncAnimation: The animation object
        """
        print(f"\nCreating animation...")
        print(f"  Output: {output_path}")
        print(f"  FPS: {fps}, DPI: {dpi}")

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Subplots
        ax_topo = fig.add_subplot(gs[0, 0])
        ax_slope = fig.add_subplot(gs[0, 1])
        ax_texture = fig.add_subplot(gs[0, 2])
        ax_displacement = fig.add_subplot(gs[1, 0])
        ax_evolution = fig.add_subplot(gs[1, 1])
        ax_stats = fig.add_subplot(gs[1, 2])

        # Initialize plots
        extent = [0, self.slope.width, 0, self.slope.height]

        # Topography
        im_topo = ax_topo.imshow(self.slope.elevation, cmap='terrain',
                                 extent=extent, animated=True)
        ax_topo.set_title('Crater Topography')
        ax_topo.set_xlabel('Distance (m)')
        ax_topo.set_ylabel('Distance (m)')
        plt.colorbar(im_topo, ax=ax_topo, label='Elevation (m)')

        # Slope angles
        im_slope = ax_slope.imshow(self.frames[0]['slope_angles'], cmap='hot',
                                   extent=extent, animated=True, vmin=0, vmax=45)
        ax_slope.set_title('Slope Angles')
        ax_slope.set_xlabel('Distance (m)')
        ax_slope.set_ylabel('Distance (m)')
        ax_slope.contour(self.frames[0]['slope_angles'], levels=[8, 15, 25],
                        colors=['cyan', 'lime', 'yellow'], linewidths=2,
                        linestyles='--', extent=extent)
        plt.colorbar(im_slope, ax=ax_slope, label='Angle (degrees)')

        # Texture
        im_texture = ax_texture.imshow(self.frames[0]['texture'], cmap='bone',
                                       extent=extent, animated=True, vmin=0, vmax=1)
        ax_texture.set_title('Elephant Hide Texture')
        ax_texture.set_xlabel('Distance (m)')
        ax_texture.set_ylabel('Distance (m)')
        plt.colorbar(im_texture, ax=ax_texture, label='Intensity')

        # Displacement
        im_disp = ax_displacement.imshow(self.frames[0]['displacement'], cmap='plasma',
                                        extent=extent, animated=True)
        ax_displacement.set_title('Cumulative Displacement')
        ax_displacement.set_xlabel('Distance (m)')
        ax_displacement.set_ylabel('Distance (m)')
        cbar_disp = plt.colorbar(im_disp, ax=ax_displacement, label='Displacement (m)')

        # Evolution curve
        line_evolution, = ax_evolution.plot([], [], 'b-', linewidth=2)
        point_evolution, = ax_evolution.plot([], [], 'ro', markersize=10)
        ax_evolution.set_xlim(0, max(self.times))
        ax_evolution.set_ylim(0, 1)
        ax_evolution.set_xlabel('Time (Myr)')
        ax_evolution.set_ylabel('Mean Texture Intensity')
        ax_evolution.set_title('Texture Development Over Time')
        ax_evolution.grid(alpha=0.3)

        # Statistics text
        ax_stats.axis('off')
        stats_text = ax_stats.text(0.1, 0.5, '', transform=ax_stats.transAxes,
                                   fontsize=10, verticalalignment='center',
                                   fontfamily='monospace')

        # Title
        title = fig.suptitle('', fontsize=14, fontweight='bold')

        def init():
            """Initialize animation."""
            return [im_texture, im_disp, line_evolution, point_evolution, stats_text, title]

        def animate(frame_idx):
            """Update animation frame."""
            frame = self.frames[frame_idx]
            time_myr = frame['time_myr']

            # Update texture
            im_texture.set_array(frame['texture'])

            # Update displacement
            im_disp.set_array(frame['displacement'])
            im_disp.set_clim(vmin=0, vmax=np.max(frame['displacement']))

            # Update evolution curve
            mean_textures = [f['texture'].mean() for f in self.frames[:frame_idx+1]]
            times_so_far = self.times[:frame_idx+1]
            line_evolution.set_data(times_so_far, mean_textures)
            point_evolution.set_data([time_myr], [mean_textures[-1]])

            # Update statistics
            stats_str = f"""
TIME: {time_myr:.2f} Myr

CUMULATIVE EFFECTS:
  Thermal cycles: {frame['num_cycles']:.2e}
  Moonquakes: {frame['num_quakes']:,}

TEXTURE METRICS:
  Mean intensity: {frame['texture'].mean():.3f}
  Max intensity: {frame['texture'].max():.3f}
  Textured area: {100*np.sum(frame['texture']>0.3)/frame['texture'].size:.1f}%

DISPLACEMENT:
  Max: {frame['displacement'].max():.2e} m
  Mean: {frame['displacement'].mean():.2e} m

REGOLITH PROPERTIES:
  Porosity: {self.physics.porosity*100:.0f}%
  Cohesion: {self.physics.cohesion} kPa
  Grain size: {self.physics.median_grain_size*1e6:.0f} μm
            """
            stats_text.set_text(stats_str)

            # Update title
            if time_myr < 0.01:
                age_str = "Fresh Crater"
            elif time_myr < 1:
                age_str = f"{time_myr*1000:.0f} kyr old"
            else:
                age_str = f"{time_myr:.2f} Myr old"

            title.set_text(f'Elephant Hide Texture Evolution: {age_str}')

            return [im_texture, im_disp, line_evolution, point_evolution, stats_text, title]

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(self.frames),
            interval=1000/fps,
            blit=True,
            repeat=True
        )

        # Save animation
        print(f"  Saving animation (this may take a few minutes)...")
        Writer = animation.writers['pillow']
        writer = Writer(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)

        print(f"  Animation saved successfully!")

        return anim


def generate_texture_animation(
    # Geometry parameters
    crater_center=(100, 100),
    crater_inner_radius=40,
    crater_outer_radius=85,
    rim_height=20,
    floor_depth=10,
    domain_size=200,
    resolution=1.0,

    # Regolith parameters
    porosity=0.45,
    cohesion_kpa=0.5,
    internal_friction_deg=37.5,
    grain_size_um=60,

    # Thermal parameters
    temp_max_k=400,
    temp_min_k=100,
    latitude_deg=0.0,

    # Seismic parameters
    quake_rate_multiplier=1.0,

    # Simulation parameters
    total_duration_myr=3.0,
    num_snapshots=20,

    # Output parameters
    output_dir='output/texture_animation',
    animation_fps=2,
    animation_dpi=150
):
    """
    Generate animated elephant hide texture evolution.

    Args:
        crater_center: (x, y) center coordinates in meters
        crater_inner_radius: Inner radius in meters
        crater_outer_radius: Outer radius in meters
        rim_height: Rim height in meters
        floor_depth: Floor depth in meters
        domain_size: Domain size in meters
        resolution: Grid resolution in meters
        porosity: Regolith porosity (0-1)
        cohesion_kpa: Cohesion in kPa
        internal_friction_deg: Internal friction angle in degrees
        grain_size_um: Median grain size in micrometers
        temp_max_k: Maximum daytime temperature in Kelvin
        temp_min_k: Minimum nighttime temperature in Kelvin
        latitude_deg: Latitude in degrees (affects thermal cycling)
        quake_rate_multiplier: Moonquake rate multiplier (1.0 = typical)
        total_duration_myr: Total simulation duration in million years
        num_snapshots: Number of snapshots for animation
        output_dir: Output directory path
        animation_fps: Animation frames per second
        animation_dpi: Animation resolution

    Returns:
        dict: Results including animation path and frames
    """
    print("=" * 70)
    print("Elephant Hide Texture Animation Generator")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create slope geometry
    print("\n1. Creating crater geometry...")
    slope = SlopeGeometry(width=domain_size, height=domain_size, resolution=resolution)
    slope.create_crater_wall(
        crater_x=crater_center[0],
        crater_y=crater_center[1],
        inner_radius=crater_inner_radius,
        outer_radius=crater_outer_radius,
        rim_height=rim_height,
        floor_depth=floor_depth
    )
    slope.add_roughness(amplitude=0.5, wavelength=4.0, smoothing=1.5)
    print(f"   Domain: {domain_size}x{domain_size} m, Resolution: {resolution} m/cell")

    # 2. Initialize physics
    print("\n2. Initializing physics...")
    physics = RegolithPhysics(
        porosity=porosity,
        cohesion=cohesion_kpa,
        internal_friction_angle=internal_friction_deg,
        grain_size=grain_size_um * 1e-6  # Convert μm to m
    )
    print(f"   Porosity: {porosity*100:.0f}%")
    print(f"   Cohesion: {cohesion_kpa} kPa")
    print(f"   Grain size: {grain_size_um} μm")

    # 3. Initialize thermal cycle
    print("\n3. Initializing thermal cycle...")
    thermal = LunarThermalCycle(
        temp_max=temp_max_k,
        temp_min=temp_min_k,
        latitude=latitude_deg
    )
    print(f"   Day: {temp_max_k} K, Night: {temp_min_k} K")
    print(f"   ΔT: {temp_max_k - temp_min_k} K")

    # 4. Initialize moonquakes
    print("\n4. Initializing moonquake simulator...")
    moonquakes = MoonquakeSimulator(
        quake_rate_multiplier=quake_rate_multiplier,
        seed=42
    )
    print(f"   Rate multiplier: {quake_rate_multiplier}x")

    # 5. Create animator
    print("\n5. Creating animator...")
    animator = TextureEvolutionAnimator(slope, physics, thermal, moonquakes)

    # 6. Generate evolution sequence
    print("\n6. Generating texture evolution sequence...")
    frames = animator.generate_evolution_sequence(total_duration_myr, num_snapshots)

    # 7. Create animation
    print("\n7. Creating animation...")
    animation_path = os.path.join(output_dir, 'texture_evolution.gif')
    anim = animator.create_animation(animation_path, fps=animation_fps, dpi=animation_dpi)

    # 8. Save individual frames as images
    print("\n8. Saving individual frame images...")
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(frame['texture'], cmap='bone', vmin=0, vmax=1,
                      extent=[0, domain_size, 0, domain_size])
        ax.set_title(f"t = {frame['time_myr']:.2f} Myr")
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Distance (m)')
        plt.colorbar(im, ax=ax, label='Texture Intensity')

        frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"   Saved {len(frames)} frame images")

    # 9. Save summary figure
    print("\n9. Creating summary figure...")
    summary_fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Select 6 representative snapshots
    snapshot_indices = np.linspace(0, len(frames)-1, 6, dtype=int)

    for i, idx in enumerate(snapshot_indices):
        frame = frames[idx]
        ax = axes[i]
        im = ax.imshow(frame['texture'], cmap='bone', vmin=0, vmax=1,
                      extent=[0, domain_size, 0, domain_size])
        ax.set_title(f"t = {frame['time_myr']:.2f} Myr\nMean: {frame['texture'].mean():.3f}")
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Distance (m)')
        plt.colorbar(im, ax=ax, fraction=0.046)

    summary_fig.suptitle(f'Elephant Hide Texture Evolution Over {total_duration_myr} Myr',
                        fontsize=14, fontweight='bold')
    plt.tight_layout()

    summary_path = os.path.join(output_dir, 'evolution_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   Saved summary figure")

    # Results
    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Animation: {animation_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Frames: {frames_dir}/ ({len(frames)} images)")
    print(f"\nFinal state:")
    final_frame = frames[-1]
    print(f"  Time: {final_frame['time_myr']:.2f} Myr")
    print(f"  Mean texture: {final_frame['texture'].mean():.3f}")
    print(f"  Max texture: {final_frame['texture'].max():.3f}")
    print(f"  Textured area: {100*np.sum(final_frame['texture']>0.3)/final_frame['texture'].size:.1f}%")
    print("=" * 70)

    return {
        'animation_path': animation_path,
        'summary_path': summary_path,
        'frames_dir': frames_dir,
        'frames': frames,
        'animator': animator
    }


def main():
    """Main function for command-line execution."""

    # Default parameters (can be modified)
    result = generate_texture_animation(
        # Geometry
        crater_center=(100, 100),
        crater_inner_radius=40,
        crater_outer_radius=85,
        rim_height=20,
        floor_depth=10,
        domain_size=200,
        resolution=1.0,

        # Regolith properties
        porosity=0.45,              # 45% porosity (loose regolith)
        cohesion_kpa=0.5,          # 0.5 kPa cohesion
        internal_friction_deg=37.5, # 37.5° internal friction
        grain_size_um=60,          # 60 μm median grain size

        # Thermal cycling
        temp_max_k=400,            # 400 K daytime
        temp_min_k=100,            # 100 K nighttime
        latitude_deg=0.0,          # Equatorial (max variation)

        # Seismic activity
        quake_rate_multiplier=1.0, # Typical moonquake rate

        # Simulation duration
        total_duration_myr=3.0,    # 3 million years
        num_snapshots=25,          # 25 snapshots for animation

        # Output
        output_dir='../output/texture_animation',
        animation_fps=2,           # 2 frames per second
        animation_dpi=150          # 150 DPI resolution
    )

    print(f"\nTo view the animation, open:")
    print(f"  {result['animation_path']}")


if __name__ == "__main__":
    main()

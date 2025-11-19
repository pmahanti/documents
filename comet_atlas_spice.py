#!/usr/bin/env python3
"""
Comet C/2025 N1 (ATLAS) Trajectory Visualization
Uses real NAIF SPICE ephemeris data to plot the actual trajectory
of Comet ATLAS approaching the Earth-Moon system.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import spiceypy as spice
from datetime import datetime, timedelta
import os

# SPICE kernel file
COMET_KERNEL = "3latlas_C2025N1_1004083_2025-11-17_2025-12-17.bsp"

# NAIF ID codes (from the filename, the comet likely has ID 1004083)
COMET_ID = "1004083"  # Comet C/2025 N1
SUN_ID = "10"
EARTH_ID = "399"
MOON_ID = "301"

# Constants
AU = 1.496e8  # AU in km
EARTH_MOON_DISTANCE = 384400  # km


def download_standard_kernels():
    """
    Download standard SPICE kernels needed for Earth and Moon.
    These are small meta-kernels and leap seconds kernels.
    """
    import urllib.request

    kernels_needed = {
        'naif0012.tls': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls',
        'de440.bsp': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp'
    }

    print("Checking for required SPICE kernels...")
    for filename, url in kernels_needed.items():
        if not os.path.exists(filename):
            print(f"  Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"  ✓ Downloaded {filename}")
            except Exception as e:
                print(f"  ✗ Failed to download {filename}: {e}")
                return False
        else:
            print(f"  ✓ Found {filename}")
    return True


def load_spice_kernels():
    """Load all required SPICE kernels."""
    print("\nLoading SPICE kernels...")

    # Load leap seconds kernel
    if os.path.exists('naif0012.tls'):
        spice.furnsh('naif0012.tls')
        print("  ✓ Loaded leap seconds kernel")

    # Load planetary ephemeris
    if os.path.exists('de440.bsp'):
        spice.furnsh('de440.bsp')
        print("  ✓ Loaded planetary ephemeris (DE440)")

    # Load comet kernel
    if os.path.exists(COMET_KERNEL):
        spice.furnsh(COMET_KERNEL)
        print(f"  ✓ Loaded comet kernel: {COMET_KERNEL}")
    else:
        raise FileNotFoundError(f"Comet kernel not found: {COMET_KERNEL}")

    print("All kernels loaded successfully!\n")


def get_time_range():
    """
    Determine the time range covered by the comet kernel.
    Returns start and end times in ET (Ephemeris Time).
    """
    # From filename: 2025-11-17 to 2025-12-17
    start_date = "2025-11-17 00:00:00 UTC"
    end_date = "2025-12-17 00:00:00 UTC"

    start_et = spice.str2et(start_date)
    end_et = spice.str2et(end_date)

    return start_et, end_et, start_date, end_date


def get_trajectory_data(start_et, end_et, timesteps=1000):
    """
    Extract trajectory data for comet, Earth, and Moon.

    Parameters:
    - start_et: Start time in ET
    - end_et: End time in ET
    - timesteps: Number of time steps

    Returns:
    - times: Array of times (ET)
    - comet_pos: Comet positions (km)
    - earth_pos: Earth positions (km)
    - moon_pos: Moon positions (km)
    """
    print(f"Extracting trajectory data with {timesteps} timesteps...")

    times = np.linspace(start_et, end_et, timesteps)

    comet_pos = np.zeros((timesteps, 3))
    earth_pos = np.zeros((timesteps, 3))
    moon_pos = np.zeros((timesteps, 3))

    # Reference frame: J2000 (inertial)
    # Observer: Solar System Barycenter (SSB) or Sun

    for i, et in enumerate(times):
        try:
            # Get comet position relative to Sun
            state_comet, _ = spice.spkezr(COMET_ID, et, 'J2000', 'NONE', SUN_ID)
            comet_pos[i] = state_comet[:3]

            # Get Earth position relative to Sun
            state_earth, _ = spice.spkezr(EARTH_ID, et, 'J2000', 'NONE', SUN_ID)
            earth_pos[i] = state_earth[:3]

            # Get Moon position relative to Sun
            state_moon, _ = spice.spkezr(MOON_ID, et, 'J2000', 'NONE', SUN_ID)
            moon_pos[i] = state_moon[:3]

        except Exception as e:
            print(f"Warning at timestep {i}: {e}")
            # Use previous values or zeros
            if i > 0:
                comet_pos[i] = comet_pos[i-1]
                earth_pos[i] = earth_pos[i-1]
                moon_pos[i] = moon_pos[i-1]

    print("  ✓ Trajectory data extracted\n")
    return times, comet_pos, earth_pos, moon_pos


def calculate_distances(comet_pos, earth_pos, moon_pos):
    """Calculate distances from comet to Earth and Moon in km."""
    earth_distances = np.linalg.norm(comet_pos - earth_pos, axis=1)
    moon_distances = np.linalg.norm(comet_pos - moon_pos, axis=1)
    return earth_distances, moon_distances


def plot_3d_trajectory(times, comet_pos, earth_pos, moon_pos):
    """Create 3D plot of trajectories."""
    # Convert to AU for better visualization
    comet_pos_au = comet_pos / AU
    earth_pos_au = earth_pos / AU
    moon_pos_au = moon_pos / AU

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectories
    ax.plot(comet_pos_au[:, 0], comet_pos_au[:, 1], comet_pos_au[:, 2],
            'c-', linewidth=2.5, label='Comet C/2025 N1 (ATLAS)', alpha=0.8)
    ax.plot(earth_pos_au[:, 0], earth_pos_au[:, 1], earth_pos_au[:, 2],
            'b-', linewidth=2, label='Earth', alpha=0.7)
    ax.plot(moon_pos_au[:, 0], moon_pos_au[:, 1], moon_pos_au[:, 2],
            'gray', linewidth=1, label='Moon', alpha=0.5)

    # Plot starting positions
    ax.scatter(*comet_pos_au[0], color='cyan', s=150, marker='o',
               label='Comet Start', edgecolors='darkblue', linewidths=2)
    ax.scatter(*earth_pos_au[0], color='blue', s=250, marker='o',
               label='Earth Start', edgecolors='darkblue', linewidths=2)
    ax.scatter(*moon_pos_au[0], color='gray', s=80, marker='o', alpha=0.7)

    # Plot ending positions
    ax.scatter(*comet_pos_au[-1], color='red', s=200, marker='*',
               label='Comet End', edgecolors='darkred', linewidths=2)
    ax.scatter(*earth_pos_au[-1], color='darkblue', s=250, marker='o',
               edgecolors='navy', linewidths=2)
    ax.scatter(*moon_pos_au[-1], color='darkgray', s=80, marker='o', alpha=0.7)

    # Sun at origin
    ax.scatter(0, 0, 0, color='yellow', s=600, marker='o',
               label='Sun', edgecolors='orange', linewidths=3)

    ax.set_xlabel('X (AU)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (AU)', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z (AU)', fontsize=14, fontweight='bold')
    ax.set_title('Comet C/2025 N1 (ATLAS) Trajectory\nNov 17 - Dec 17, 2025',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio
    all_pos = np.vstack([comet_pos_au, earth_pos_au, moon_pos_au])
    max_range = np.array([
        all_pos[:, 0].max() - all_pos[:, 0].min(),
        all_pos[:, 1].max() - all_pos[:, 1].min(),
        all_pos[:, 2].max() - all_pos[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_pos[:, 0].max() + all_pos[:, 0].min()) * 0.5
    mid_y = (all_pos[:, 1].max() + all_pos[:, 1].min()) * 0.5
    mid_z = (all_pos[:, 2].max() + all_pos[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    return fig


def plot_close_approach(times, comet_pos, earth_pos, moon_pos):
    """Create detailed plot of close approach to Earth-Moon system."""
    earth_distances, moon_distances = calculate_distances(comet_pos, earth_pos, moon_pos)

    # Find closest approach to Earth
    closest_idx = np.argmin(earth_distances)
    closest_et = times[closest_idx]
    closest_date = spice.et2utc(closest_et, 'C', 0)

    # Convert times to days from start
    times_days = (times - times[0]) / 86400.0  # seconds to days

    fig = plt.figure(figsize=(18, 7))

    # 3D close approach view (Earth-centered)
    ax1 = fig.add_subplot(131, projection='3d')

    # Earth-centered coordinates
    comet_earth_centered = comet_pos - earth_pos
    moon_earth_centered = moon_pos - earth_pos

    ax1.plot(comet_earth_centered[:, 0], comet_earth_centered[:, 1],
             comet_earth_centered[:, 2], 'c-', linewidth=2.5,
             label='Comet C/2025 N1', alpha=0.8)
    ax1.plot(moon_earth_centered[:, 0], moon_earth_centered[:, 1],
             moon_earth_centered[:, 2], 'gray', linewidth=2,
             label='Moon', alpha=0.6)

    # Earth at origin
    ax1.scatter(0, 0, 0, color='blue', s=400, marker='o', label='Earth',
                edgecolors='darkblue', linewidths=2)

    # Closest approach point
    ax1.scatter(*comet_earth_centered[closest_idx], color='red', s=200,
                marker='*', label=f'Closest Approach\n{closest_date}',
                edgecolors='darkred', linewidths=2)

    ax1.set_xlabel('X (km)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y (km)', fontsize=11, fontweight='bold')
    ax1.set_zlabel('Z (km)', fontsize=11, fontweight='bold')
    ax1.set_title('Close Approach\n(Earth-Centered Frame)',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Distance vs time plot
    ax2 = fig.add_subplot(132)
    ax2.plot(times_days, earth_distances / 1e6, 'b-', linewidth=2.5,
             label='Distance to Earth')
    ax2.plot(times_days, moon_distances / 1e6, 'gray', linewidth=2.5,
             label='Distance to Moon', alpha=0.7)
    ax2.axvline(times_days[closest_idx], color='red', linestyle='--',
                linewidth=2, alpha=0.7, label=f'Closest Approach\n({times_days[closest_idx]:.1f} days)')
    ax2.axhline(EARTH_MOON_DISTANCE / 1e6, color='green', linestyle=':',
                linewidth=2, alpha=0.5, label='Earth-Moon Distance')

    ax2.set_xlabel('Time (days from Nov 17, 2025)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Distance (Million km)', fontsize=12, fontweight='bold')
    ax2.set_title('Comet Distance from Earth and Moon', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # XY plane view (ecliptic-like)
    ax3 = fig.add_subplot(133)
    ax3.plot(comet_earth_centered[:, 0] / 1e6, comet_earth_centered[:, 1] / 1e6,
             'c-', linewidth=2.5, label='Comet C/2025 N1', alpha=0.8)
    ax3.plot(moon_earth_centered[:, 0] / 1e6, moon_earth_centered[:, 1] / 1e6,
             'gray', linewidth=2, label='Moon Orbit', alpha=0.6)

    # Earth at origin
    ax3.scatter(0, 0, color='blue', s=400, marker='o', label='Earth',
                edgecolors='darkblue', linewidths=2)

    # Closest approach
    ax3.scatter(comet_earth_centered[closest_idx, 0] / 1e6,
                comet_earth_centered[closest_idx, 1] / 1e6,
                color='red', s=200, marker='*', label='Closest Approach',
                edgecolors='darkred', linewidths=2)

    # Draw Earth-Moon distance circle
    circle = plt.Circle((0, 0), EARTH_MOON_DISTANCE / 1e6,
                        color='green', fill=False, linestyle=':',
                        linewidth=2, alpha=0.4, label='Lunar Distance')
    ax3.add_patch(circle)

    ax3.set_xlabel('X (Million km)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Y (Million km)', fontsize=12, fontweight='bold')
    ax3.set_title('XY Plane View\n(Earth-Centered)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    plt.tight_layout()
    return fig, closest_idx, closest_date


def print_summary(times, comet_pos, earth_pos, moon_pos, start_date, end_date):
    """Print summary statistics of the encounter."""
    earth_distances, moon_distances = calculate_distances(comet_pos, earth_pos, moon_pos)

    closest_earth_idx = np.argmin(earth_distances)
    closest_moon_idx = np.argmin(moon_distances)

    closest_earth_et = times[closest_earth_idx]
    closest_moon_et = times[closest_moon_idx]

    closest_earth_date = spice.et2utc(closest_earth_et, 'C', 0)
    closest_moon_date = spice.et2utc(closest_moon_et, 'C', 0)

    print("\n" + "="*70)
    print("COMET C/2025 N1 (ATLAS) TRAJECTORY SUMMARY")
    print("="*70)
    print(f"\nData Source: NAIF SPICE Kernel")
    print(f"Kernel File: {COMET_KERNEL}")
    print(f"Time Range: {start_date} to {end_date}")

    print(f"\nClosest Approach to Earth:")
    print(f"  Date/Time: {closest_earth_date}")
    print(f"  Distance: {earth_distances[closest_earth_idx]:,.0f} km")
    print(f"  Distance: {earth_distances[closest_earth_idx] / 1e6:.3f} Million km")
    print(f"  Distance: {earth_distances[closest_earth_idx] / EARTH_MOON_DISTANCE:.2f} Lunar Distances")
    print(f"  Distance: {earth_distances[closest_earth_idx] / AU:.4f} AU")

    print(f"\nClosest Approach to Moon:")
    print(f"  Date/Time: {closest_moon_date}")
    print(f"  Distance: {moon_distances[closest_moon_idx]:,.0f} km")
    print(f"  Distance: {moon_distances[closest_moon_idx] / 1e6:.3f} Million km")
    print(f"  Distance: {moon_distances[closest_moon_idx] / EARTH_MOON_DISTANCE:.2f} Lunar Distances")

    # Minimum and maximum distances
    print(f"\nDistance Range to Earth:")
    print(f"  Minimum: {earth_distances.min() / 1e6:.3f} Million km ({earth_distances.min() / AU:.4f} AU)")
    print(f"  Maximum: {earth_distances.max() / 1e6:.3f} Million km ({earth_distances.max() / AU:.4f} AU)")

    print("="*70 + "\n")


def main():
    """Main execution function."""
    print("="*70)
    print("Comet C/2025 N1 (ATLAS) Real Trajectory Visualization")
    print("Using NAIF SPICE Ephemeris Data")
    print("="*70 + "\n")

    # Download standard kernels if needed
    if not download_standard_kernels():
        print("\n⚠ Warning: Could not download all kernels.")
        print("Attempting to continue with available kernels...\n")

    try:
        # Load SPICE kernels
        load_spice_kernels()

        # Get time range
        start_et, end_et, start_date, end_date = get_time_range()
        print(f"Time range: {start_date} to {end_date}\n")

        # Extract trajectory data
        times, comet_pos, earth_pos, moon_pos = get_trajectory_data(
            start_et, end_et, timesteps=1000
        )

        # Print summary
        print_summary(times, comet_pos, earth_pos, moon_pos, start_date, end_date)

        # Create visualizations
        print("Generating visualizations...")

        # 3D trajectory plot
        fig1 = plot_3d_trajectory(times, comet_pos, earth_pos, moon_pos)
        plt.savefig('comet_atlas_trajectory_3d.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: comet_atlas_trajectory_3d.png")

        # Close approach plot
        fig2, closest_idx, closest_date = plot_close_approach(
            times, comet_pos, earth_pos, moon_pos
        )
        plt.savefig('comet_atlas_close_approach.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: comet_atlas_close_approach.png")

        print("\n✓ Visualization complete!")
        print("  Close the plot windows to exit.\n")
        plt.show()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up SPICE kernels
        spice.kclear()
        print("\nSPICE kernels unloaded.")


if __name__ == "__main__":
    main()

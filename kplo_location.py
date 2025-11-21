#!/usr/bin/env python3
"""
KPLO Location Tracker

Displays the current position of the Korea Pathfinder Lunar Orbiter (KPLO)
relative to the Moon using SPICE kernels.

Outputs a PNG image with UTC and Arizona time stamps.
"""

import os
import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import pytz

def load_kernels(kernel_dir='kernels'):
    """
    Load all necessary SPICE kernels.

    Parameters
    ----------
    kernel_dir : str
        Directory containing SPICE kernel files
    """
    kernels = [
        'naif0012.tls',              # Leap seconds
        'pck00010.tpc',              # Planetary constants
        'gm_de431.tpc',              # Gravitational parameters
        'de430.bsp',                 # Planetary ephemeris
        'moon_080317.tf',            # Moon frames
        'moon_assoc_me.tf',          # Moon associations
        'moon_pa_de421_1900-2050.bpc', # Moon orientation
        'kplo_pm_20251116_20260111_v00.bsp',  # KPLO ephemeris (latest)
        'kplo_scp_20251118_20251129_v00.bc',  # KPLO orientation (latest)
        'kplo_sclkscet_v698.tsc',    # KPLO clock
        'kplo_shadowcam_v01.ti',     # ShadowCam instrument
        'kplo_v00_shc_a.tf',         # ShadowCam frames
        'SHC_REFERENCE.tf',          # ShadowCam reference
    ]

    loaded = []
    for kernel in kernels:
        kernel_path = os.path.join(kernel_dir, kernel)
        if os.path.exists(kernel_path):
            try:
                spice.furnsh(kernel_path)
                loaded.append(kernel)
                print(f"  Loaded: {kernel}")
            except Exception as e:
                print(f"  Warning: Could not load {kernel}: {e}")
        else:
            print(f"  Warning: Kernel not found: {kernel_path}")

    return loaded

def get_kplo_position(et):
    """
    Get KPLO position relative to the Moon.

    Parameters
    ----------
    et : float
        Ephemeris time (seconds past J2000)

    Returns
    -------
    position : np.ndarray
        Position vector [x, y, z] in km
    velocity : np.ndarray
        Velocity vector [vx, vy, vz] in km/s
    """
    try:
        # Get KPLO state (position and velocity) relative to Moon
        # KPLO ID: -155 (typical NAIF ID for KPLO)
        # Moon ID: 301
        state, lt = spice.spkezr('-155', et, 'MOON_ME', 'NONE', '301')
        position = state[:3]
        velocity = state[3:]
        return position, velocity
    except Exception as e:
        print(f"Error getting KPLO position: {e}")
        print("Trying alternative KPLO ID...")
        try:
            # Try with alternative naming
            state, lt = spice.spkezr('KPLO', et, 'MOON_ME', 'NONE', 'MOON')
            position = state[:3]
            velocity = state[3:]
            return position, velocity
        except Exception as e2:
            print(f"Alternative failed: {e2}")
            raise

def get_lat_lon_alt(position):
    """
    Convert Cartesian position to latitude, longitude, altitude.

    Parameters
    ----------
    position : np.ndarray
        Position vector [x, y, z] in km

    Returns
    -------
    lat : float
        Latitude in degrees
    lon : float
        Longitude in degrees
    alt : float
        Altitude in km
    """
    # Moon radius in km
    moon_radius = 1737.4

    # Convert to spherical coordinates
    r = np.linalg.norm(position)
    altitude = r - moon_radius

    # Latitude and longitude
    lat = np.degrees(np.arcsin(position[2] / r))
    lon = np.degrees(np.arctan2(position[1], position[0]))

    return lat, lon, altitude

def create_visualization(position, velocity, lat, lon, alt, utc_time, arizona_time, output_file='kplo_location.png'):
    """
    Create visualization of KPLO position.

    Parameters
    ----------
    position : np.ndarray
        Position vector in km
    velocity : np.ndarray
        Velocity vector in km/s
    lat, lon, alt : float
        Latitude, longitude, altitude
    utc_time : datetime
        Current UTC time
    arizona_time : datetime
        Current Arizona time
    output_file : str
        Output PNG filename
    """
    fig = plt.figure(figsize=(16, 10), facecolor='#0a0a0a')

    # Create main axis for Moon view
    ax_main = plt.subplot(1, 2, 1, projection='mollweide')
    ax_main.set_facecolor('#000000')

    # Plot KPLO position on Mollweide projection
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    ax_main.plot(lon_rad, lat_rad, 'o', color='#00ff00', markersize=20,
                 label='KPLO', zorder=10, markeredgecolor='white', markeredgewidth=2)

    # Add grid
    ax_main.grid(True, color='#404040', linestyle=':', linewidth=0.5, alpha=0.7)
    ax_main.set_xlabel('Longitude', fontsize=12, color='white', fontweight='bold')
    ax_main.set_ylabel('Latitude', fontsize=12, color='white', fontweight='bold')
    ax_main.set_title('KPLO Position - Lunar Surface Projection',
                      fontsize=14, color='white', fontweight='bold', pad=20)

    # Customize grid labels
    ax_main.tick_params(colors='white')

    # Add legend
    legend = ax_main.legend(loc='upper left', fontsize=12, framealpha=0.9)
    legend.get_frame().set_facecolor('#1a1a1a')
    legend.get_frame().set_edgecolor('#00ff00')
    for text in legend.get_texts():
        text.set_color('white')

    # Right panel - Information display
    ax_info = plt.subplot(1, 2, 2)
    ax_info.set_facecolor('#0a0a0a')
    ax_info.axis('off')

    # Create info text
    info_text = f"""
KOREA PATHFINDER LUNAR ORBITER (KPLO)
{'='*50}

POSITION INFORMATION:
  Latitude:      {lat:>10.4f}°
  Longitude:     {lon:>10.4f}°
  Altitude:      {alt:>10.2f} km

CARTESIAN COORDINATES (Moon-Centered):
  X:             {position[0]:>10.2f} km
  Y:             {position[1]:>10.2f} km
  Z:             {position[2]:>10.2f} km

  Distance:      {np.linalg.norm(position):>10.2f} km

VELOCITY:
  Vx:            {velocity[0]:>10.4f} km/s
  Vy:            {velocity[1]:>10.4f} km/s
  Vz:            {velocity[2]:>10.4f} km/s

  Speed:         {np.linalg.norm(velocity):>10.4f} km/s

TIME STAMPS:
  UTC:           {utc_time.strftime('%Y-%m-%d %H:%M:%S %Z')}
  Arizona:       {arizona_time.strftime('%Y-%m-%d %H:%M:%S %Z')}

REFERENCE FRAME: MOON_ME (Moon Mean Earth/Polar Axis)
MOON RADIUS:     1737.4 km
"""

    ax_info.text(0.05, 0.95, info_text,
                transform=ax_info.transAxes,
                fontsize=11,
                verticalalignment='top',
                color='#00ff00',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='#1a1a1a',
                         edgecolor='#00ff00', alpha=0.9, linewidth=2))

    # Add orbital indicator
    orbital_period = 2 * np.pi * np.linalg.norm(position) / np.linalg.norm(velocity) / 3600
    ax_info.text(0.05, 0.05,
                f"Approximate Orbital Period: {orbital_period:.2f} hours",
                transform=ax_info.transAxes,
                fontsize=10,
                color='#ffff00',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='#2a2a2a',
                         edgecolor='#ffff00', alpha=0.9))

    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    print(f"\nVisualization saved to: {os.path.abspath(output_file)}")
    plt.close()

def main():
    """Main execution function."""
    print("KPLO Location Tracker")
    print("=" * 60)

    # Load SPICE kernels
    print("\nLoading SPICE kernels...")
    loaded = load_kernels()
    print(f"\nSuccessfully loaded {len(loaded)} kernels")

    # Get current time
    utc_now = datetime.now(timezone.utc)
    arizona_tz = pytz.timezone('America/Phoenix')
    arizona_now = utc_now.astimezone(arizona_tz)

    print(f"\nCurrent Time:")
    print(f"  UTC:     {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Arizona: {arizona_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Convert to ephemeris time
    et = spice.str2et(utc_now.strftime('%Y-%m-%d %H:%M:%S UTC'))
    print(f"\nEphemeris Time: {et:.2f} seconds past J2000")

    # Get KPLO position
    print("\nQuerying KPLO position...")
    try:
        position, velocity = get_kplo_position(et)
        print(f"  Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}] km")
        print(f"  Velocity: [{velocity[0]:.4f}, {velocity[1]:.4f}, {velocity[2]:.4f}] km/s")

        # Convert to lat/lon/alt
        lat, lon, alt = get_lat_lon_alt(position)
        print(f"\nGeodetic Coordinates:")
        print(f"  Latitude:  {lat:.4f}°")
        print(f"  Longitude: {lon:.4f}°")
        print(f"  Altitude:  {alt:.2f} km")

        # Create visualization
        print("\nGenerating visualization...")
        create_visualization(position, velocity, lat, lon, alt,
                           utc_now, arizona_now, 'kplo_location.png')

        print("\nDone!")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nThis may be because:")
        print("  1. Current time is outside the kernel coverage period")
        print("  2. KPLO ID or naming convention differs")
        print("  3. Required kernels are not loaded")

        # Print kernel coverage info
        try:
            print("\nChecking kernel coverage...")
            # This would require iterating through loaded SPK kernels
            print("Please check that your KPLO ephemeris kernels cover the current date.")
        except:
            pass

    finally:
        # Unload kernels
        spice.kclear()

if __name__ == '__main__':
    main()

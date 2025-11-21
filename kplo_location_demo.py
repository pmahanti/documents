#!/usr/bin/env python3
"""
KPLO Location Tracker - Demonstration Version

Displays a demonstration of KPLO position visualization.
This version uses simulated orbital data since SPICE kernels require Git LFS.

For real-time data, the full SPICE kernels need to be downloaded from:
https://naif.jpl.nasa.gov/naif/data_kplo.html
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import pytz

def simulate_kplo_orbit(et):
    """
    Simulate KPLO orbital position based on typical lunar polar orbit parameters.

    KPLO typical orbit:
    - Altitude: ~100 km
    - Inclination: ~90° (polar orbit)
    - Period: ~2 hours

    Parameters
    ----------
    et : float
        Time in seconds (used to determine position in orbit)

    Returns
    -------
    position : np.ndarray
        Position vector [x, y, z] in km
    velocity : np.ndarray
        Velocity vector [vx, vy, vz] in km/s
    """
    # Moon radius in km
    moon_radius = 1737.4

    # Orbital parameters
    orbit_altitude = 100.0  # km
    orbit_radius = moon_radius + orbit_altitude

    # Orbital period (simplified circular orbit)
    mu = 4902.8  # km^3/s^2 (Moon's gravitational parameter)
    period = 2 * np.pi * np.sqrt(orbit_radius**3 / mu)

    # Orbital angular velocity
    omega = 2 * np.pi / period

    # Use time to determine position in orbit
    # Polar orbit: varies in Z (latitude) and rotates in XY (longitude)
    theta = (et * omega) % (2 * np.pi)  # Angle in orbit
    phi = (et * omega * 0.1) % (2 * np.pi)  # Slow precession

    # Position in polar orbit
    x = orbit_radius * np.cos(theta) * np.cos(phi)
    y = orbit_radius * np.cos(theta) * np.sin(phi)
    z = orbit_radius * np.sin(theta)

    position = np.array([x, y, z])

    # Velocity (perpendicular to position for circular orbit)
    v_mag = np.sqrt(mu / orbit_radius)
    # Velocity direction (perpendicular to radius vector)
    v_theta = -np.sin(theta) * np.cos(phi)
    v_phi = -np.sin(theta) * np.sin(phi) * orbit_radius / np.sqrt(x**2 + y**2 + 1e-10)
    v_z = np.cos(theta)

    velocity = v_mag * np.array([v_theta, v_phi, v_z])

    return position, velocity, period

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
    moon_radius = 1737.4

    # Convert to spherical coordinates
    r = np.linalg.norm(position)
    altitude = r - moon_radius

    # Latitude and longitude
    lat = np.degrees(np.arcsin(position[2] / r))
    lon = np.degrees(np.arctan2(position[1], position[0]))

    return lat, lon, altitude

def create_visualization(position, velocity, lat, lon, alt, utc_time, arizona_time,
                        orbital_period, output_file='kplo_location.png'):
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
    orbital_period : float
        Orbital period in seconds
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
                 label='KPLO (Simulated)', zorder=10, markeredgecolor='white', markeredgewidth=2)

    # Add ground track (simulate orbital path)
    orbit_points = 100
    track_lons = []
    track_lats = []
    for i in range(orbit_points):
        angle = 2 * np.pi * i / orbit_points
        track_lat = 90 * np.sin(angle)  # Polar orbit
        track_lon = lon + 360 * i / orbit_points - 180
        if track_lon > 180:
            track_lon -= 360
        track_lons.append(np.radians(track_lon))
        track_lats.append(np.radians(track_lat))

    ax_main.plot(track_lons, track_lats, '-', color='#00ff00', linewidth=1,
                alpha=0.3, label='Ground Track')

    # Add grid
    ax_main.grid(True, color='#404040', linestyle=':', linewidth=0.5, alpha=0.7)
    ax_main.set_xlabel('Longitude', fontsize=12, color='white', fontweight='bold')
    ax_main.set_ylabel('Latitude', fontsize=12, color='white', fontweight='bold')
    ax_main.set_title('KPLO Position - Lunar Surface Projection\n(DEMONSTRATION - Simulated Data)',
                      fontsize=14, color='#ffaa00', fontweight='bold', pad=20)

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

⚠️  DEMONSTRATION MODE - SIMULATED DATA
Real-time data requires SPICE kernels from Git LFS

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

ORBITAL PARAMETERS (Typical):
  Period:        {orbital_period/3600:>10.2f} hours
  Inclination:   ~90° (Polar orbit)
  Eccentricity:  ~0.0 (Circular)

TIME STAMPS:
  UTC:           {utc_time.strftime('%Y-%m-%d %H:%M:%S %Z')}
  Arizona:       {arizona_time.strftime('%Y-%m-%d %H:%M:%S %Z')}

REFERENCE FRAME: MOON_ME (Moon Mean Earth/Polar Axis)
MOON RADIUS:     1737.4 km
"""

    ax_info.text(0.05, 0.95, info_text,
                transform=ax_info.transAxes,
                fontsize=10.5,
                verticalalignment='top',
                color='#00ff00',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='#1a1a1a',
                         edgecolor='#ffaa00', alpha=0.9, linewidth=2))

    # Add note about real data
    note_text = """
NOTE: For real-time KPLO tracking data:
1. Download SPICE kernels from NAIF (NASA)
2. Install Git LFS to fetch binary kernel files
3. Run kplo_location.py (full version)

Current visualization uses typical polar orbit
parameters for demonstration purposes.
"""

    ax_info.text(0.05, 0.05,
                note_text,
                transform=ax_info.transAxes,
                fontsize=9,
                color='#ffaa00',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='#2a2a2a',
                         edgecolor='#ffaa00', alpha=0.9))

    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    print(f"\n✓ Visualization saved to: {os.path.abspath(output_file)}")
    plt.close()

def main():
    """Main execution function."""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 11 + "KPLO LOCATION TRACKER - DEMO" + " " * 19 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("⚠️  DEMONSTRATION MODE")
    print("Using simulated orbital data (typical KPLO polar orbit)")
    print()

    # Get current time
    utc_now = datetime.now(timezone.utc)
    arizona_tz = pytz.timezone('America/Phoenix')
    arizona_now = utc_now.astimezone(arizona_tz)

    print(f"Current Time:")
    print(f"  UTC:     {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Arizona: {arizona_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Simulate ephemeris time
    et = (utc_now - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)).total_seconds()
    print(f"\nSimulated Ephemeris Time: {et:.2f} seconds past J2000")

    # Get simulated KPLO position
    print("\nSimulating KPLO orbital position...")
    position, velocity, period = simulate_kplo_orbit(et)
    print(f"  Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}] km")
    print(f"  Velocity: [{velocity[0]:.4f}, {velocity[1]:.4f}, {velocity[2]:.4f}] km/s")

    # Convert to lat/lon/alt
    lat, lon, alt = get_lat_lon_alt(position)
    print(f"\nGeodetic Coordinates:")
    print(f"  Latitude:  {lat:>8.4f}°")
    print(f"  Longitude: {lon:>8.4f}°")
    print(f"  Altitude:  {alt:>8.2f} km")
    print(f"\nOrbital Period: {period/3600:.2f} hours")

    # Create visualization
    print("\nGenerating visualization...")
    create_visualization(position, velocity, lat, lon, alt,
                       utc_now, arizona_now, period, 'kplo_location.png')

    print("\n" + "─" * 60)
    print("✓ Done!")
    print()
    print("For real-time KPLO data:")
    print("  1. Install Git LFS: apt-get install git-lfs")
    print("  2. Pull kernel files: git lfs pull")
    print("  3. Run: python3 kplo_location.py")
    print()

if __name__ == '__main__':
    main()

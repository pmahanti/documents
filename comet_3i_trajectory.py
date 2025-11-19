#!/usr/bin/env python3
"""
Comet 3I Trajectory Simulation
Simulates and visualizes the path of an interstellar comet (3I)
approaching the Earth-Moon system.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import datetime

# Physical constants (SI units)
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_SUN = 1.989e30  # Mass of Sun (kg)
M_EARTH = 5.972e24  # Mass of Earth (kg)
M_MOON = 7.342e22  # Mass of Moon (kg)
AU = 1.496e11  # Astronomical Unit (m)
EARTH_MOON_DISTANCE = 3.844e8  # Average Earth-Moon distance (m)

# Convert to AU-based units for better numerical stability
# Mass unit: Solar mass
# Distance unit: AU
# Time unit: Years
M_EARTH_SOLAR = M_EARTH / M_SUN
M_MOON_SOLAR = M_MOON / M_SUN
EARTH_MOON_DIST_AU = EARTH_MOON_DISTANCE / AU

# Orbital parameters for a hypothetical interstellar comet 3I
# These represent a hyperbolic orbit with close approach to Earth
COMET_PARAMS = {
    'eccentricity': 3.5,  # Hyperbolic orbit (e > 1)
    'perihelion': 1.0,  # Perihelion distance (AU) - close to Earth orbit
    'inclination': 30.0,  # Orbital inclination (degrees)
    'longitude_ascending': 45.0,  # Longitude of ascending node (degrees)
    'argument_perihelion': 60.0,  # Argument of perihelion (degrees)
    'v_infinity': 26.0,  # Hyperbolic excess velocity (km/s)
}


def orbital_elements_to_state_vector(a, e, i, omega, w, nu, mu=1.0):
    """
    Convert orbital elements to state vector (position and velocity).

    Parameters:
    - a: semi-major axis (AU)
    - e: eccentricity
    - i: inclination (radians)
    - omega: longitude of ascending node (radians)
    - w: argument of perihelion (radians)
    - nu: true anomaly (radians)
    - mu: standard gravitational parameter (AU^3/year^2)

    Returns:
    - r: position vector [x, y, z] (AU)
    - v: velocity vector [vx, vy, vz] (AU/year)
    """
    # Calculate radius
    if e < 1.0:
        # Elliptical orbit
        r_mag = a * (1 - e**2) / (1 + e * np.cos(nu))
    else:
        # Hyperbolic orbit
        r_mag = a * (e**2 - 1) / (1 + e * np.cos(nu))

    # Position in orbital plane
    r_orbital = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0])

    # Velocity in orbital plane
    if e < 1.0:
        h = np.sqrt(mu * a * (1 - e**2))
    else:
        h = np.sqrt(mu * abs(a) * (e**2 - 1))

    v_orbital = np.array([
        -mu / h * np.sin(nu),
        mu / h * (e + np.cos(nu)),
        0
    ])

    # Rotation matrices
    R3_w = np.array([
        [np.cos(w), -np.sin(w), 0],
        [np.sin(w), np.cos(w), 0],
        [0, 0, 1]
    ])

    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])

    R3_omega = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])

    # Transform to inertial frame
    Q = R3_omega @ R1_i @ R3_w
    r = Q @ r_orbital
    v = Q @ v_orbital

    return r, v


def n_body_derivatives(state, t, masses, positions_func=None):
    """
    Calculate derivatives for N-body problem.

    Parameters:
    - state: [x1, y1, z1, vx1, vy1, vz1, ...] for all bodies
    - t: time
    - masses: list of masses in solar masses
    - positions_func: optional function to get positions (for time-varying systems)

    Returns:
    - derivatives: [vx1, vy1, vz1, ax1, ay1, az1, ...]
    """
    n = len(masses)
    state = state.reshape((n, 6))
    derivatives = np.zeros_like(state)

    # Velocities
    derivatives[:, 0:3] = state[:, 3:6]

    # Accelerations due to gravitational forces
    for i in range(n):
        for j in range(n):
            if i != j:
                r_vec = state[j, 0:3] - state[i, 0:3]
                r_mag = np.linalg.norm(r_vec)
                if r_mag > 0:
                    # G*M in AU^3/year^2 (with G_solar = 4*pi^2)
                    G_solar = 4 * np.pi**2
                    acc = G_solar * masses[j] * r_vec / r_mag**3
                    derivatives[i, 3:6] += acc

    return derivatives.flatten()


def simulate_comet_trajectory(duration_days=100, timesteps=1000):
    """
    Simulate the comet's trajectory through the Earth-Moon system.

    Parameters:
    - duration_days: simulation duration in days
    - timesteps: number of time steps

    Returns:
    - t: time array (days)
    - comet_pos: comet positions over time (AU)
    - earth_pos: Earth positions over time (AU)
    - moon_pos: Moon positions over time (AU)
    """
    # Convert duration to years
    duration_years = duration_days / 365.25
    t = np.linspace(0, duration_years, timesteps)

    # Initial conditions for comet (hyperbolic orbit)
    e = COMET_PARAMS['eccentricity']
    q = COMET_PARAMS['perihelion']  # perihelion distance
    a = -q / (e - 1)  # semi-major axis for hyperbolic orbit (negative)

    i = np.radians(COMET_PARAMS['inclination'])
    omega = np.radians(COMET_PARAMS['longitude_ascending'])
    w = np.radians(COMET_PARAMS['argument_perihelion'])

    # Start at a true anomaly that puts it approaching
    nu_0 = np.radians(-120)  # Approaching from distance

    r_comet_0, v_comet_0 = orbital_elements_to_state_vector(a, e, i, omega, w, nu_0)

    # Initial conditions for Earth (circular orbit at 1 AU)
    r_earth_0 = np.array([1.0, 0.0, 0.0])
    v_earth_0 = np.array([0.0, 2 * np.pi, 0.0])  # Circular orbit velocity

    # Initial conditions for Moon (relative to Earth)
    moon_angle_0 = 0.0
    r_moon_rel = EARTH_MOON_DIST_AU * np.array([np.cos(moon_angle_0), np.sin(moon_angle_0), 0.0])
    r_moon_0 = r_earth_0 + r_moon_rel

    # Moon orbital velocity around Earth (27.3 days period)
    moon_orbital_period = 27.3 / 365.25  # years
    moon_orbital_velocity = 2 * np.pi * EARTH_MOON_DIST_AU / moon_orbital_period
    v_moon_rel = moon_orbital_velocity * np.array([-np.sin(moon_angle_0), np.cos(moon_angle_0), 0.0])
    v_moon_0 = v_earth_0 + v_moon_rel

    # Combined initial state: [comet, earth, moon]
    initial_state = np.concatenate([
        r_comet_0, v_comet_0,
        r_earth_0, v_earth_0,
        r_moon_0, v_moon_0
    ])

    masses = [1e-15, M_EARTH_SOLAR, M_MOON_SOLAR]  # Negligible comet mass

    # Integrate
    print("Simulating trajectory...")
    solution = odeint(n_body_derivatives, initial_state, t, args=(masses,))

    # Extract positions
    comet_pos = solution[:, 0:3]
    earth_pos = solution[:, 6:9]
    moon_pos = solution[:, 12:15]

    return t * 365.25, comet_pos, earth_pos, moon_pos  # Convert time back to days


def calculate_distances(comet_pos, earth_pos, moon_pos):
    """Calculate distances from comet to Earth and Moon."""
    earth_distances = np.linalg.norm(comet_pos - earth_pos, axis=1) * AU / 1e6  # km
    moon_distances = np.linalg.norm(comet_pos - moon_pos, axis=1) * AU / 1e6  # km
    return earth_distances, moon_distances


def plot_3d_trajectory(t, comet_pos, earth_pos, moon_pos):
    """Create 3D plot of trajectories."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectories
    ax.plot(comet_pos[:, 0], comet_pos[:, 1], comet_pos[:, 2],
            'c-', linewidth=2, label='Comet 3I', alpha=0.7)
    ax.plot(earth_pos[:, 0], earth_pos[:, 1], earth_pos[:, 2],
            'b-', linewidth=1.5, label='Earth', alpha=0.6)
    ax.plot(moon_pos[:, 0], moon_pos[:, 1], moon_pos[:, 2],
            'gray', linewidth=1, label='Moon', alpha=0.5)

    # Plot starting positions
    ax.scatter(*comet_pos[0], color='cyan', s=100, marker='o', label='Comet Start')
    ax.scatter(*earth_pos[0], color='blue', s=200, marker='o', label='Earth Start')
    ax.scatter(*moon_pos[0], color='gray', s=50, marker='o')

    # Plot ending positions
    ax.scatter(*comet_pos[-1], color='red', s=100, marker='*', label='Comet End')
    ax.scatter(*earth_pos[-1], color='darkblue', s=200, marker='o')
    ax.scatter(*moon_pos[-1], color='darkgray', s=50, marker='o')

    # Sun at origin
    ax.scatter(0, 0, 0, color='yellow', s=500, marker='o', label='Sun', edgecolors='orange', linewidths=2)

    ax.set_xlabel('X (AU)', fontsize=12)
    ax.set_ylabel('Y (AU)', fontsize=12)
    ax.set_zlabel('Z (AU)', fontsize=12)
    ax.set_title('Comet 3I Trajectory through Earth-Moon System', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio
    max_range = np.array([
        comet_pos[:, 0].max() - comet_pos[:, 0].min(),
        comet_pos[:, 1].max() - comet_pos[:, 1].min(),
        comet_pos[:, 2].max() - comet_pos[:, 2].min()
    ]).max() / 2.0

    mid_x = (comet_pos[:, 0].max() + comet_pos[:, 0].min()) * 0.5
    mid_y = (comet_pos[:, 1].max() + comet_pos[:, 1].min()) * 0.5
    mid_z = (comet_pos[:, 2].max() + comet_pos[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    return fig


def plot_close_approach(t, comet_pos, earth_pos, moon_pos):
    """Create detailed plot of close approach to Earth-Moon system."""
    # Find closest approach to Earth
    earth_distances, moon_distances = calculate_distances(comet_pos, earth_pos, moon_pos)
    closest_idx = np.argmin(earth_distances)

    # Plot window around closest approach
    window = 200  # indices
    start_idx = max(0, closest_idx - window)
    end_idx = min(len(t), closest_idx + window)

    fig = plt.figure(figsize=(16, 6))

    # 3D close approach view
    ax1 = fig.add_subplot(121, projection='3d')

    # Convert to Earth-centered coordinates (in km)
    comet_earth_centered = (comet_pos[start_idx:end_idx] - earth_pos[start_idx:end_idx]) * AU / 1e6
    moon_earth_centered = (moon_pos[start_idx:end_idx] - earth_pos[start_idx:end_idx]) * AU / 1e6

    ax1.plot(comet_earth_centered[:, 0], comet_earth_centered[:, 1], comet_earth_centered[:, 2],
             'c-', linewidth=2, label='Comet 3I')
    ax1.plot(moon_earth_centered[:, 0], moon_earth_centered[:, 1], moon_earth_centered[:, 2],
             'gray', linewidth=1.5, label='Moon')

    # Earth at origin
    ax1.scatter(0, 0, 0, color='blue', s=300, marker='o', label='Earth')

    # Closest approach point
    ax1.scatter(*comet_earth_centered[closest_idx - start_idx],
                color='red', s=150, marker='*', label='Closest Approach')

    ax1.set_xlabel('X (km)', fontsize=10)
    ax1.set_ylabel('Y (km)', fontsize=10)
    ax1.set_zlabel('Z (km)', fontsize=10)
    ax1.set_title('Close Approach (Earth-Centered Frame)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Distance vs time plot
    ax2 = fig.add_subplot(122)
    ax2.plot(t, earth_distances / 1e6, 'b-', linewidth=2, label='Earth Distance')
    ax2.plot(t, moon_distances / 1e6, 'gray', linewidth=2, label='Moon Distance')
    ax2.axvline(t[closest_idx], color='red', linestyle='--', alpha=0.7, label='Closest Approach')
    ax2.axhline(EARTH_MOON_DISTANCE / 1e9, color='green', linestyle=':', alpha=0.5, label='Earth-Moon Distance')

    ax2.set_xlabel('Time (days)', fontsize=12)
    ax2.set_ylabel('Distance (Million km)', fontsize=12)
    ax2.set_title('Comet Distance from Earth and Moon', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    return fig, closest_idx


def print_summary(t, comet_pos, earth_pos, moon_pos):
    """Print summary statistics of the encounter."""
    earth_distances, moon_distances = calculate_distances(comet_pos, earth_pos, moon_pos)

    closest_earth_idx = np.argmin(earth_distances)
    closest_moon_idx = np.argmin(moon_distances)

    print("\n" + "="*60)
    print("COMET 3I TRAJECTORY SUMMARY")
    print("="*60)
    print(f"\nOrbital Parameters:")
    print(f"  Eccentricity: {COMET_PARAMS['eccentricity']:.2f} (hyperbolic)")
    print(f"  Perihelion: {COMET_PARAMS['perihelion']:.2f} AU")
    print(f"  Inclination: {COMET_PARAMS['inclination']:.1f}°")
    print(f"  Longitude of Ascending Node: {COMET_PARAMS['longitude_ascending']:.1f}°")
    print(f"  Argument of Perihelion: {COMET_PARAMS['argument_perihelion']:.1f}°")

    print(f"\nClosest Approach to Earth:")
    print(f"  Time: {t[closest_earth_idx]:.2f} days")
    print(f"  Distance: {earth_distances[closest_earth_idx]:,.0f} km")
    print(f"  Distance: {earth_distances[closest_earth_idx] / 384400:.2f} Lunar Distances")

    print(f"\nClosest Approach to Moon:")
    print(f"  Time: {t[closest_moon_idx]:.2f} days")
    print(f"  Distance: {moon_distances[closest_moon_idx]:,.0f} km")

    print(f"\nSimulation Duration: {t[-1]:.1f} days")
    print("="*60 + "\n")


def main():
    """Main execution function."""
    print("Comet 3I Trajectory Simulation")
    print("=" * 60)

    # Run simulation
    t, comet_pos, earth_pos, moon_pos = simulate_comet_trajectory(
        duration_days=100,
        timesteps=2000
    )

    # Print summary
    print_summary(t, comet_pos, earth_pos, moon_pos)

    # Create visualizations
    print("Generating visualizations...")

    # 3D trajectory plot
    fig1 = plot_3d_trajectory(t, comet_pos, earth_pos, moon_pos)
    plt.savefig('comet_3i_trajectory_3d.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: comet_3i_trajectory_3d.png")

    # Close approach plot
    fig2, closest_idx = plot_close_approach(t, comet_pos, earth_pos, moon_pos)
    plt.savefig('comet_3i_close_approach.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: comet_3i_close_approach.png")

    print("\nSimulation complete! Close the plot windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()

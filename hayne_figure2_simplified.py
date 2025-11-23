#!/usr/bin/env python3
"""
Simplified Implementation of Hayne et al. (2021) Figure 2

Generate temperature distributions for rough surfaces using a simplified
but physically accurate thermal model.

This version focuses on getting the physics right:
- Proper solar flux at polar latitudes
- Correct RMS slope generation
- Simplified but valid radiation balance
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from rough_surface_theory import generate_gaussian_surface


# Constants
STEFAN_BOLTZMANN = 5.670374419e-8  # W m⁻² K⁻⁴
SOLAR_CONSTANT = 1361.0  # W/m² at 1 AU
ALBEDO = 0.12  # Lunar regolith Bond albedo
EMISSIVITY = 0.95  # Lunar regolith infrared emissivity
T_SKY = 3.0  # Cosmic microwave background [K]


def generate_surface_with_target_slope(grid_size: int,
                                        target_slope_deg: float,
                                        pixel_scale: float = 1.0,
                                        random_seed: int = 42) -> np.ndarray:
    """
    Generate Gaussian surface with specified RMS slope.

    The RMS slope is controlled by the height variation and pixel scale:
        σ_slope ≈ σ_height / pixel_scale

    Parameters:
        grid_size: Grid size (NxN)
        target_slope_deg: Target RMS slope [degrees]
        pixel_scale: Pixel scale [m]
        random_seed: Random seed

    Returns:
        2D array of surface heights [m]
    """
    # Generate normalized Gaussian surface (zero mean, unit RMS)
    surface_norm = generate_gaussian_surface(grid_size, H=0.9, random_seed=random_seed)

    # Calculate gradients to check RMS slope
    grad_y, grad_x = np.gradient(surface_norm)  # This gives dz/dpixel

    # Convert to slope (dz/dm)
    slope_x = grad_x / pixel_scale
    slope_y = grad_y / pixel_scale
    slope = np.sqrt(slope_x**2 + slope_y**2)

    # Current RMS slope in radians
    current_rms_rad = np.sqrt(np.mean(slope**2))  # RMS of slope magnitude

    # Scale factor to achieve target slope
    target_slope_rad = np.radians(target_slope_deg)
    scale_factor = target_slope_rad / current_rms_rad

    # Scale the surface
    surface = surface_norm * scale_factor * pixel_scale

    # Verify
    grad_y, grad_x = np.gradient(surface)
    slope_x = grad_x / pixel_scale
    slope_y = grad_y / pixel_scale
    slope = np.sqrt(slope_x**2 + slope_y**2)
    actual_rms = np.degrees(np.sqrt(np.mean(slope**2)))

    print(f"  Target RMS slope: {target_slope_deg:.2f}°")
    print(f"  Actual RMS slope: {actual_rms:.2f}°")
    print(f"  Height range: {np.min(surface):.2f} to {np.max(surface):.2f} m")

    return surface


def calculate_simple_temperatures(surface: np.ndarray,
                                   pixel_scale: float,
                                   solar_elevation_deg: float,
                                   use_3d: bool = False) -> np.ndarray:
    """
    Calculate equilibrium temperatures using simplified radiation balance.

    For each pixel, solve:
        ε σ T⁴ = Q_solar + Q_sky

    where:
        Q_solar = (1-A) S cos(i) for illuminated pixels (direct + scattered)
        Q_solar = A × S × f_scattered for shadowed pixels (scattered only)
        Q_sky = ε σ T_sky⁴ (negligible)

    Parameters:
        surface: 2D surface heights [m]
        pixel_scale: Pixel scale [m]
        solar_elevation_deg: Solar elevation angle [degrees]
        use_3d: If True, account for local topography (slower)

    Returns:
        2D array of temperatures [K]
    """
    ny, nx = surface.shape

    # Solar flux at this latitude
    solar_elev_rad = np.radians(solar_elevation_deg)
    solar_flux_base = SOLAR_CONSTANT * np.sin(solar_elev_rad)  # W/m²

    print(f"  Solar elevation: {solar_elevation_deg:.1f}°")
    print(f"  Solar flux (base): {solar_flux_base:.2f} W/m²")

    # Calculate surface slopes
    grad_y, grad_x = np.gradient(surface)
    slope_x = grad_x / pixel_scale
    slope_y = grad_y / pixel_scale

    # Surface normals (assuming solar azimuth = 0, i.e., sun in +x direction)
    # Normal vector: n = [-∂z/∂x, -∂z/∂y, 1] (normalized)
    norm_x = -slope_x
    norm_y = -slope_y
    norm_z = np.ones_like(slope_x)
    norm_mag = np.sqrt(norm_x**2 + norm_y**2 + norm_z**2)
    norm_x /= norm_mag
    norm_y /= norm_mag
    norm_z /= norm_mag

    # Solar direction (elevation e, azimuth 0)
    sun_x = np.cos(solar_elev_rad)
    sun_y = 0.0
    sun_z = np.sin(solar_elev_rad)

    # Cosine of solar incidence angle
    cos_i = norm_x * sun_x + norm_y * sun_y + norm_z * sun_z

    # Only positive (illuminated facets)
    cos_i = np.maximum(cos_i, 0.0)

    # Direct solar flux on each facet
    Q_direct = (1.0 - ALBEDO) * solar_flux_base * cos_i

    if use_3d:
        # Add simple shadowing: pixels with cos_i < 0.1 are in shadow
        # (This is a very simplified shadow model)
        shadow_threshold = 0.1
        shadowed = cos_i < shadow_threshold

        # Shadowed pixels get only scattered light (rough approximation)
        fraction_illuminated = np.sum(~shadowed) / (ny * nx)
        Q_scattered = ALBEDO * solar_flux_base * 0.3  # 30% of illuminated flux

        Q_direct[shadowed] = Q_scattered
    else:
        # No shadowing in simple mode
        pass

    # Sky radiation (negligible)
    Q_sky = EMISSIVITY * STEFAN_BOLTZMANN * T_SKY**4  # ~0.0003 W/m²

    # Total absorbed
    Q_total = Q_direct + Q_sky

    # Solve for temperature: T = (Q / (ε σ))^(1/4)
    T = (Q_total / (EMISSIVITY * STEFAN_BOLTZMANN))**0.25

    return T


def generate_hayne_figure2_simplified():
    """
    Generate Hayne Figure 2 with simplified but accurate physics.
    """
    print("=" * 80)
    print("HAYNE ET AL. (2021) FIGURE 2 - SIMPLIFIED IMPLEMENTATION")
    print("=" * 80)

    # Parameters
    grid_size = 128
    pixel_scale = 1.0  # meters
    latitude = -85.0  # degrees South
    solar_elevation = 5.0  # degrees (colatitude at 85°S)

    # RMS slopes to test
    rms_slopes = [5.7, 26.6]  # degrees

    # Results storage
    surfaces = {}
    temperatures = {}

    # Generate surfaces and calculate temperatures
    for sigma_s in rms_slopes:
        print(f"\n{'='*70}")
        print(f"Surface with σs = {sigma_s}°")
        print(f"{'='*70}")

        # Generate surface
        seed = 42 if sigma_s < 10 else 123
        surface = generate_surface_with_target_slope(
            grid_size, sigma_s, pixel_scale, random_seed=seed
        )

        surfaces[sigma_s] = surface

        # Calculate temperatures
        print("\n  Calculating temperatures...")
        T = calculate_simple_temperatures(surface, pixel_scale, solar_elevation, use_3d=True)

        temperatures[sigma_s] = T

        # Statistics
        print(f"\n  Temperature Statistics:")
        print(f"    Mean: {np.mean(T):.2f} K")
        print(f"    Median: {np.median(T):.2f} K")
        print(f"    Min: {np.min(T):.2f} K")
        print(f"    Max: {np.max(T):.2f} K")
        print(f"    Std: {np.std(T):.2f} K")

        # Cold trap statistics
        cold_110 = np.sum(T < 110) / T.size * 100
        cold_100 = np.sum(T < 100) / T.size * 100
        cold_90 = np.sum(T < 90) / T.size * 100

        print(f"\n  Cold Trap Statistics:")
        print(f"    T < 110 K: {cold_110:.2f}%")
        print(f"    T < 100 K: {cold_100:.2f}%")
        print(f"    T < 90 K: {cold_90:.2f}%")

    # Create figure
    print(f"\n{'='*80}")
    print("Creating Figure 2...")
    print(f"{'='*80}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Hayne et al. (2021) Figure 2 - Simplified Implementation\\n' +
                 f'Latitude: {latitude}°, Solar elevation: {solar_elevation}°, Pixel: {pixel_scale}m',
                 fontsize=13, fontweight='bold')

    for row, sigma_s in enumerate(rms_slopes):
        surface = surfaces[sigma_s]
        T = temperatures[sigma_s]

        extent = [0, grid_size * pixel_scale, 0, grid_size * pixel_scale]

        # Column 0: Surface topography
        ax = axes[row, 0]
        im = ax.imshow(surface, cmap='terrain', extent=extent, origin='lower')
        ax.set_title(f'σs = {sigma_s}°: Topography', fontweight='bold')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Distance (m)')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Height (m)')

        # Column 1: Temperature map
        ax = axes[row, 1]
        im = ax.imshow(T, cmap='hot', vmin=0, vmax=200, extent=extent, origin='lower')
        ax.set_title(f'σs = {sigma_s}°: Temperature', fontweight='bold')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Distance (m)')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Temperature (K)')

        # Column 2: Temperature histogram
        ax = axes[row, 2]
        ax.hist(T.flatten(), bins=50, color='darkblue', alpha=0.7, edgecolor='black')
        ax.axvline(110, color='red', linestyle='--', linewidth=2, label='110 K')
        ax.axvline(np.mean(T), color='orange', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(T):.1f} K')
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Pixel count')
        ax.set_title(f'σs = {sigma_s}°: Distribution', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 200])

    plt.tight_layout()
    output_file = '/home/user/documents/hayne_figure2_simplified.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()

    # Summary table
    print(f"\n{'='*80}")
    print("COMPARISON WITH HAYNE ET AL. (2021)")
    print(f"{'='*80}")

    print(f"\n{'Surface':<20} {'σs':<8} {'T_mean':<10} {'T_min':<10} {'T_max':<10} {'%<110K':<10}")
    print("-" * 80)

    for sigma_s in rms_slopes:
        T = temperatures[sigma_s]
        cold_frac = np.sum(T < 110) / T.size * 100

        label = "Smooth (plains)" if sigma_s < 10 else "Rough (craters)"
        print(f"{label:<20} {sigma_s:<8.1f} {np.mean(T):<10.1f} {np.min(T):<10.1f} "
              f"{np.max(T):<10.1f} {cold_frac:<10.2f}")

    print("\nExpected from Hayne et al. (2021) Figure 2:")
    print("  Smooth (σs=5.7°):  T_mean ≈ 110 K, few pixels < 110 K")
    print("  Rough (σs=26.6°):  T_mean ≈ 88 K, many pixels < 110 K")

    print(f"\n{'='*80}")
    print("✓ FIGURE 2 COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    generate_hayne_figure2_simplified()

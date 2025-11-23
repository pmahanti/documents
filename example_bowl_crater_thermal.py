#!/usr/bin/env python3
"""
Example: Integrated Bowl-Shaped Crater Thermal Analysis

Demonstrates integration of:
1. Bowl-shaped crater geometry (Ingersoll et al. 1992)
2. Time-averaged thermal modeling (Hayne et al. 2017)
3. Sublimation rate calculations
4. Cold trap analysis with lateral conduction effects

This combines the analytical Ingersoll crater model with the
comprehensive thermal framework.
"""

import numpy as np

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, skipping plots")

from bowl_crater_thermal import (
    CraterGeometry,
    ingersoll_crater_temperature,
    crater_temperature_profile_radial,
    crater_cold_trap_area,
    crater_shadow_area_fraction
)
from thermal_model import (
    LunarThermalProperties,
    SimpleThermalModel,
    crater_cold_trap_fraction,
    lateral_conduction_scale
)
import vaporp_temp as vp


def get_volatile_species(name: str):
    """
    Get a volatile species from the database.

    Parameters:
    -----------
    name : str
        Species name (e.g., 'H2O', 'CO2')

    Returns:
    --------
    VolatileSpecies
        Species object
    """
    if name not in vp.VOLATILE_SPECIES:
        raise ValueError(f"Unknown species: {name}. Available: {list(vp.VOLATILE_SPECIES.keys())}")
    return vp.VOLATILE_SPECIES[name]


def hertz_knudsen_rate(species, temperature, alpha=1.0):
    """
    Calculate Hertz-Knudsen sublimation rate.

    Parameters:
    -----------
    species : VolatileSpecies
        Volatile species
    temperature : float
        Temperature [K]
    alpha : float
        Sticking coefficient

    Returns:
    --------
    float
        Sublimation rate [kg/m²/s]
    """
    result = species.sublimation_rate(temperature, alpha)
    return result['sublimation_rate_kg_m2_s']


def analyze_crater_volatile_retention(crater: CraterGeometry,
                                        volatile_species_name: str = 'H2O',
                                        T_sunlit: float = 200.0,
                                        solar_elevation_deg: float = 5.0,
                                        thermal_props: LunarThermalProperties = None):
    """
    Comprehensive crater analysis for volatile retention.

    Parameters:
    -----------
    crater : CraterGeometry
        Bowl-shaped crater parameters
    volatile_species_name : str
        Name of volatile species ('H2O', 'CO2', etc.)
    T_sunlit : float
        Sunlit terrain temperature [K]
    solar_elevation_deg : float
        Solar elevation angle [degrees]
    thermal_props : LunarThermalProperties
        Thermal properties, defaults to lunar values
    """
    if thermal_props is None:
        thermal_props = LunarThermalProperties()

    print("=" * 70)
    print(f"Comprehensive Crater Analysis: {volatile_species_name} Retention")
    print("=" * 70)

    # Crater geometry
    print(f"\n### Crater Geometry ###")
    print(f"Diameter: {crater.diameter:.1f} m ({crater.diameter/1000:.2f} km)")
    print(f"Depth: {crater.depth:.1f} m")
    print(f"Depth/Diameter ratio (γ): {crater.gamma:.4f}")
    print(f"Beta parameter: {crater.beta:.3f}")
    print(f"Latitude: {crater.latitude_deg:.1f}°")
    print(f"Crater area: {np.pi * crater.radius**2:.2e} m² ({np.pi * crater.radius**2 / 1e6:.3f} km²)")

    # Shadow analysis from Ingersoll theory
    print(f"\n### Shadow Analysis (Ingersoll et al. 1992) ###")
    shadow_info = crater_shadow_area_fraction(
        crater.gamma,
        crater.latitude_deg,
        solar_elevation_deg,
        solar_declination_deg=1.54  # Moon's obliquity
    )

    print(f"Solar elevation: {solar_elevation_deg}°")
    print(f"Instantaneous shadow fraction: {shadow_info['instantaneous_shadow_fraction']:.2%}")
    print(f"Permanent shadow fraction: {shadow_info['permanent_shadow_fraction']:.2%}")
    print(f"Permanent/Instantaneous ratio: {shadow_info['f_ratio']:.3f}")

    # Temperature analysis
    print(f"\n### Temperature Analysis ###")
    temps = ingersoll_crater_temperature(
        crater, T_sunlit, solar_elevation_deg,
        albedo=thermal_props.albedo,
        emissivity=thermal_props.emissivity
    )

    print(f"Sunlit terrain temperature: {T_sunlit:.1f} K")
    print(f"Shadow temperature: {temps['T_shadow']:.1f} K")
    print(f"Sunlit crater floor: {temps['T_sunlit_floor']:.1f} K")
    print(f"Average wall temperature: {temps['T_wall_avg']:.1f} K")
    print(f"\nRadiation in shadow:")
    print(f"  Scattered solar: {temps['irradiance_reflected']:.2f} W/m²")
    print(f"  Thermal IR from walls: {temps['irradiance_thermal']:.2f} W/m²")
    print(f"  Total irradiance: {temps['irradiance_total']:.2f} W/m²")

    # Lateral conduction analysis (Hayne et al. 2021)
    print(f"\n### Lateral Conduction Effects ###")
    k_avg = (thermal_props.ks + thermal_props.kd) / 2
    rho_avg = (thermal_props.rhos + thermal_props.rhod) / 2
    cp_avg = thermal_props.cp0
    kappa = k_avg / (rho_avg * cp_avg)

    L_crit = lateral_conduction_scale(kappa, thermal_props.day)
    print(f"Thermal diffusivity: {kappa:.2e} m²/s")
    print(f"Lateral conduction scale: {L_crit:.3f} m")
    print(f"  (cold traps smaller than this are partially warmed by conduction)")

    # Cold trap fraction with lateral conduction
    ct_result = crater_cold_trap_fraction(
        crater.gamma,
        crater.latitude_deg,
        length_scale=crater.diameter,
        T_threshold=110,
        thermal_props=thermal_props
    )

    print(f"\nCold trap fraction (accounting for lateral conduction):")
    print(f"  Geometric shadow fraction: {ct_result['permanent_shadow_fraction']:.2%}")
    print(f"  Conduction reduction factor: {ct_result['conduction_reduction_factor']:.3f}")
    print(f"  Effective cold trap fraction: {ct_result['cold_trap_fraction']:.2%}")

    # Sublimation rate analysis
    print(f"\n### Sublimation Rate Analysis for {volatile_species_name} ###")

    # Get volatile species
    species = get_volatile_species(volatile_species_name)

    # Calculate sublimation rates at different temperatures
    temps_test = [temps['T_shadow'], 110.0, 120.0, temps['T_sunlit_floor']]
    temp_labels = ['Shadow', '110K threshold', '120K', 'Sunlit floor']

    print(f"\nSublimation rates:")
    for T, label in zip(temps_test, temp_labels):
        if T > 0:
            result = species.sublimation_rate(T, alpha=1.0)
            # Convert mm/yr to m/Gyr
            E_m_per_Gyr = result['sublimation_rate_mm_yr'] / 1000 * 1e9
            print(f"  {label:20s} ({T:5.1f} K): {E_m_per_Gyr:.2e} m/Gyr")

    # Volatile stability assessment
    print(f"\n### Volatile Stability Assessment ###")

    # Typical delivery rate (Arnold 1979)
    delivery_rate_m_Gyr = 1.0

    # Calculate whether shadow is stable
    result_shadow = species.sublimation_rate(temps['T_shadow'], alpha=1.0)
    # Convert mm/yr to m/Gyr
    E_shadow_m_Gyr = result_shadow['sublimation_rate_mm_yr'] / 1000 * 1e9

    is_stable = E_shadow_m_Gyr < delivery_rate_m_Gyr

    print(f"Assumed delivery rate: {delivery_rate_m_Gyr} m/Gyr")
    print(f"Shadow sublimation rate: {E_shadow_m_Gyr:.2e} m/Gyr")
    print(f"Ice accumulation possible: {is_stable}")
    print(f"Aridity index (delivery/sublimation): {delivery_rate_m_Gyr / max(E_shadow_m_Gyr, 1e-10):.2e}")

    # Estimate accumulation over lunar history
    if is_stable:
        lunar_age_Gyr = 4.5
        net_accumulation = (delivery_rate_m_Gyr - E_shadow_m_Gyr) * lunar_age_Gyr
        cold_trap_area_m2 = np.pi * crater.radius**2 * ct_result['cold_trap_fraction']

        # Estimate density (kg/m³)
        density_map = {'H2O': 920, 'CO2': 1560, 'CO': 790, 'CH4': 420, 'NH3': 820, 'SO2': 1460}
        density = density_map.get(species.name, 1000)
        total_mass_kg = net_accumulation * cold_trap_area_m2 * density

        print(f"\nPotential accumulation over {lunar_age_Gyr} Gyr:")
        print(f"  Net ice thickness: {net_accumulation:.2f} m")
        print(f"  Cold trap area: {cold_trap_area_m2:.2e} m²")
        print(f"  Total ice mass: {total_mass_kg:.2e} kg ({total_mass_kg/1e9:.2f} million tonnes)")

    return {
        'crater': crater,
        'shadow_info': shadow_info,
        'temperatures': temps,
        'cold_trap_result': ct_result,
        'is_stable': is_stable,
        'species': species
    }


def compare_crater_depths():
    """
    Compare temperature and cold-trapping for craters with different d/D ratios.

    Demonstrates the effect of crater shape on volatile retention.
    """
    print("\n\n" + "=" * 70)
    print("Comparison: Effect of Crater Depth on Cold Trapping")
    print("=" * 70)

    # Fixed parameters
    diameter = 5000.0  # 5 km diameter
    latitude = -85.0
    solar_elev = 4.0

    # Range of d/D ratios (from shallow degraded to fresh deep craters)
    gamma_values = [0.05, 0.076, 0.10, 0.14, 0.18]
    depths = [gamma * diameter for gamma in gamma_values]

    print(f"\nCrater diameter: {diameter} m")
    print(f"Latitude: {latitude}°")
    print(f"Solar elevation: {solar_elev}°\n")

    print(f"{'γ (d/D)':>8} | {'Depth (m)':>10} | {'Shadow %':>10} | {'T_shadow (K)':>13} | {'Cold Trap':>10}")
    print("-" * 70)

    for gamma, depth in zip(gamma_values, depths):
        crater = CraterGeometry(diameter=diameter, depth=depth, latitude_deg=latitude)

        shadow_info = crater_shadow_area_fraction(gamma, latitude, solar_elev, 1.54)
        temps = ingersoll_crater_temperature(crater, T_sunlit=200.0, solar_elevation_deg=solar_elev)

        is_cold_trap = "Yes" if temps['T_shadow'] < 110 else "No"

        print(f"{gamma:8.3f} | {depth:10.1f} | {shadow_info['permanent_shadow_fraction']*100:9.1f}% | "
              f"{temps['T_shadow']:13.1f} | {is_cold_trap:>10}")

    print("\nKey insight: Shallower craters (smaller γ) have:")
    print("  - Smaller shadow fractions")
    print("  - BUT potentially colder shadows (see more sky, less warm walls)")
    print("  - Trade-off between shadow area and shadow temperature")


def plot_crater_temperature_map(crater: CraterGeometry, save_filename: str = None):
    """
    Create a 2D temperature map of a crater cross-section.

    Parameters:
    -----------
    crater : CraterGeometry
        Crater to visualize
    save_filename : str, optional
        If provided, save figure to this file
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, cannot create plot")
        return None

    # Calculate radial temperature profile
    r_vals, T_vals = crater_temperature_profile_radial(
        crater, T_sunlit=220.0, solar_elevation_deg=5.0, n_points=100
    )

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Temperature vs radius
    ax1.plot(r_vals, T_vals, 'b-', linewidth=2)
    ax1.axhline(y=110, color='r', linestyle='--', label='110 K threshold')
    ax1.set_xlabel('Radial distance from center (m)', fontsize=12)
    ax1.set_ylabel('Temperature (K)', fontsize=12)
    ax1.set_title(f'Radial Temperature Profile: D={crater.diameter}m, d={crater.depth}m, γ={crater.gamma:.3f}',
                  fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Crater cross-section with temperature colormap
    # Create crater shape
    R = crater.radius
    d = crater.depth
    R_sphere = crater.sphere_radius

    # Crater profile (simplified spherical bowl)
    theta = np.linspace(0, np.pi, 100)
    x_bowl = R_sphere * np.sin(theta)
    y_bowl = -R_sphere * np.cos(theta) + (R_sphere - d)

    # Mask to crater opening
    mask = x_bowl <= R
    x_bowl = x_bowl[mask]
    y_bowl = y_bowl[mask]

    # Create temperature field
    x_grid = np.linspace(-R, R, 100)
    y_grid = np.linspace(-d, 0, 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    T_grid = np.zeros_like(X)

    for i in range(len(y_grid)):
        for j in range(len(x_grid)):
            r = abs(x_grid[j])
            T_grid[i, j] = np.interp(r, r_vals, T_vals)

    # Plot temperature field
    im = ax2.contourf(X, Y, T_grid, levels=20, cmap='coolwarm')
    ax2.plot(x_bowl, y_bowl, 'k-', linewidth=2)
    ax2.plot(-x_bowl, y_bowl, 'k-', linewidth=2)
    ax2.set_xlabel('Horizontal distance (m)', fontsize=12)
    ax2.set_ylabel('Depth (m)', fontsize=12)
    ax2.set_title('Crater Cross-Section Temperature Map', fontsize=13)
    ax2.set_aspect('equal')
    plt.colorbar(im, ax=ax2, label='Temperature (K)')

    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_filename}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    # Example 1: Detailed analysis of a specific crater
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Detailed Analysis of Haworth Crater")
    print("=" * 70)

    # Haworth crater: ~51 km diameter, ~2.5 km deep (from Hayne paper)
    haworth = CraterGeometry(
        diameter=51000.0,
        depth=2500.0,
        latitude_deg=-87.5
    )

    result = analyze_crater_volatile_retention(
        haworth,
        volatile_species_name='H2O',
        T_sunlit=180.0,
        solar_elevation_deg=2.5
    )

    # Example 2: Compare different crater depths
    compare_crater_depths()

    # Example 3: Analyze multiple volatile species
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Multi-species Volatile Retention")
    print("=" * 70)

    test_crater = CraterGeometry(diameter=10000.0, depth=800.0, latitude_deg=-88.0)

    print(f"\nCrater: D={test_crater.diameter}m, γ={test_crater.gamma:.3f} at {test_crater.latitude_deg}°")

    temps = ingersoll_crater_temperature(test_crater, T_sunlit=170.0, solar_elevation_deg=2.0)
    T_shadow = temps['T_shadow']

    print(f"Shadow temperature: {T_shadow:.1f} K\n")

    volatiles = ['H2O', 'CO2', 'SO2', 'NH3']
    print(f"{'Species':>8} | {'Sublimation Rate (m/Gyr)':>25} | {'Stable?':>10}")
    print("-" * 50)

    for vol_name in volatiles:
        species = get_volatile_species(vol_name)
        result = species.sublimation_rate(T_shadow, alpha=1.0)
        # Convert mm/yr to m/Gyr
        E_m_Gyr = result['sublimation_rate_mm_yr'] / 1000 * 1e9

        stable = "Yes" if E_m_Gyr < 1.0 else "No"

        print(f"{vol_name:>8} | {E_m_Gyr:>25.2e} | {stable:>10}")

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)

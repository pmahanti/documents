#!/usr/bin/env python3
"""
Bowl-Shaped Crater Temperature Calculations

Based on:
- Ingersoll et al. (1992): Analytical theory for bowl-shaped craters
- Hayne et al. (2021): Micro cold traps with detailed crater geometry
- Schorghofer & Williams (2020): Time-dependent thermal modeling

This module provides comprehensive temperature calculations for bowl-shaped
(spherical) craters accounting for:
- Geometric shadow relationships
- View factors to sky and crater walls
- Scattered and thermal radiation from surroundings
- Depth/diameter ratio effects
- Latitude and solar declination dependencies
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


# Stefan-Boltzmann constant [W/(m²·K⁴)]
SIGMA_SB = 5.67051e-8


@dataclass
class CraterGeometry:
    """
    Bowl-shaped crater geometric parameters.

    Based on Ingersoll et al. (1992) and Hayne et al. (2021).
    """
    diameter: float  # Crater diameter D [m]
    depth: float  # Crater depth d [m]
    latitude_deg: float  # Latitude [degrees]

    @property
    def gamma(self) -> float:
        """Depth-to-diameter ratio γ = d/D"""
        return self.depth / self.diameter

    @property
    def beta(self) -> float:
        """Geometric parameter β = 1/(2γ) - 2γ"""
        return 1.0 / (2.0 * self.gamma) - 2.0 * self.gamma

    @property
    def radius(self) -> float:
        """Crater radius R = D/2 [m]"""
        return self.diameter / 2.0

    @property
    def sphere_radius(self) -> float:
        """
        Radius of curvature of spherical bowl.

        From geometry: R_sphere = (R² + d²) / (2d)
        """
        R = self.radius
        d = self.depth
        return (R**2 + d**2) / (2.0 * d)


def crater_shadow_area_fraction(gamma: float, latitude_deg: float,
                                  solar_elevation_deg: float,
                                  solar_declination_deg: float = 0.0) -> Dict[str, float]:
    """
    Calculate shadow area fractions in a bowl-shaped crater.

    Implements analytical relations from Hayne et al. (2021) Eqs. 2-9,
    based on Ingersoll et al. (1992).

    Parameters:
    -----------
    gamma : float
        Depth-to-diameter ratio d/D (typically 0.076-0.14 for lunar craters)
    latitude_deg : float
        Latitude [degrees]
    solar_elevation_deg : float
        Solar elevation angle [degrees]
    solar_declination_deg : float
        Solar declination [degrees], default 0

    Returns:
    --------
    dict containing:
        - 'instantaneous_shadow_fraction': Fraction in shadow at given time
        - 'permanent_shadow_fraction': Fraction in permanent shadow
        - 'f_ratio': Ratio of permanent to instantaneous shadow
        - 'beta': Geometric parameter
        - 'gamma': d/D ratio
    """
    beta = 1.0 / (2.0 * gamma) - 2.0 * gamma

    # Convert to radians
    e_rad = solar_elevation_deg * np.pi / 180.0
    delta_rad = solar_declination_deg * np.pi / 180.0
    lat_rad = latitude_deg * np.pi / 180.0

    # Colatitude (complement of latitude for polar regions)
    e0_rad = (90.0 - abs(latitude_deg)) * np.pi / 180.0

    # Instantaneous (noontime) shadow area fraction (Hayne Eq. 5, 23)
    if e_rad > 0:
        # Normalized shadow coordinate from Hayne Eq. 3
        x0_prime = np.cos(e_rad)**2 - np.sin(e_rad)**2 - beta * np.cos(e_rad) * np.sin(e_rad)
        # Shadow area fraction (Hayne Eq. 5)
        A_instant_frac = (1.0 + x0_prime) / 2.0
        A_instant_frac = max(0.0, min(1.0, A_instant_frac))
    else:
        A_instant_frac = 1.0

    # Permanent shadow area fraction (Hayne Eq. 22, 26)
    if e0_rad < 1e-6:  # At the pole
        A_perm_frac = max(0.0, 1.0 - 2.0 * beta * delta_rad)
    else:
        # Hayne Eq. 22 with declination correction (Eq. 26)
        A_perm_frac = max(0.0, 1.0 - (8.0 * beta * e0_rad) / (3.0 * np.pi) - 2.0 * beta * delta_rad)

    # Ratio of permanent to instantaneous shadow
    f_ratio = A_perm_frac / A_instant_frac if A_instant_frac > 0 else 0.0

    return {
        'instantaneous_shadow_fraction': A_instant_frac,
        'permanent_shadow_fraction': A_perm_frac,
        'f_ratio': f_ratio,
        'beta': beta,
        'gamma': gamma
    }


def crater_view_factors(gamma: float) -> Dict[str, float]:
    """
    Calculate view factors for radiation exchange in a bowl-shaped crater.

    Based on Ingersoll et al. (1992) geometric relations.

    Parameters:
    -----------
    gamma : float
        Depth-to-diameter ratio d/D

    Returns:
    --------
    dict containing:
        - 'f_sky': View factor to sky from crater floor
        - 'f_walls': View factor to crater walls from floor
        - 'f_floor_walls': Effective view factor accounting for geometry
    """
    # Exact view factor calculation from Ingersoll et al. (1992)
    # Based on solid angle subtended by the sky as seen from crater floor

    # For a spherical bowl:
    # - Sphere radius: R_s = (R² + d²) / (2d)
    # - For d/D = γ, R = D/2:
    # - R_s/D = (1/4 + γ²) / (2γ) = (1 + 4γ²) / (8γ)
    # - Opening half-angle: θ = arctan(R / (R_s - d))

    R_s_over_D = (0.25 + gamma**2) / (2.0 * gamma)  # R_s / D
    d_over_D = gamma                                  # d / D
    R_over_D = 0.5                                    # R / D

    # Height from floor to sphere center
    height = R_s_over_D - d_over_D

    # Opening half-angle from geometry
    cos_theta = height / np.sqrt(height**2 + R_over_D**2)

    # View factor from solid angle (Ingersoll 1992)
    # F_sky = (solid angle of sky) / (2π steradians)
    f_sky = (1.0 - cos_theta) / 2.0
    f_walls = 1.0 - f_sky

    return {
        'f_sky': f_sky,
        'f_walls': f_walls,
        'f_floor_walls': f_walls  # Simplified
    }


def ingersoll_crater_temperature(crater: CraterGeometry,
                                   T_sunlit: float,
                                   solar_elevation_deg: float,
                                   albedo: float = 0.12,
                                   emissivity: float = 0.95,
                                   T_sky: float = 3.0) -> Dict[str, float]:
    """
    Calculate temperature in a bowl-shaped crater using Ingersoll et al. (1992) theory.

    Accounts for:
    - Geometric shadowing based on d/D ratio and solar elevation
    - Scattered solar radiation from crater walls
    - Thermal infrared emission from crater walls
    - View factors to sky vs walls

    Parameters:
    -----------
    crater : CraterGeometry
        Crater geometric parameters
    T_sunlit : float
        Temperature of sunlit terrain [K]
    solar_elevation_deg : float
        Solar elevation angle [degrees]
    albedo : float
        Bond albedo, default 0.12
    emissivity : float
        Thermal emissivity, default 0.95
    T_sky : float
        Sky temperature (cosmic microwave background), default 3.0 K

    Returns:
    --------
    dict containing:
        - 'T_shadow': Temperature in permanent shadow [K]
        - 'T_sunlit_floor': Temperature on sunlit crater floor [K]
        - 'T_wall_avg': Average crater wall temperature [K]
        - 'irradiance_reflected': Scattered solar radiation [W/m²]
        - 'irradiance_thermal': Thermal radiation from walls [W/m²]
        - 'irradiance_total': Total irradiance in shadow [W/m²]
    """
    # Get shadow fractions
    shadow_info = crater_shadow_area_fraction(
        crater.gamma, crater.latitude_deg, solar_elevation_deg
    )

    # Get view factors
    view_factors = crater_view_factors(crater.gamma)
    f_sky = view_factors['f_sky']
    f_walls = view_factors['f_walls']

    # Estimate crater wall temperature
    # Walls receive both direct and scattered illumination
    # At high latitudes, surrounding terrain is also cold
    abs_lat = abs(crater.latitude_deg)
    if abs_lat > 85:
        T_walls = T_sunlit * 0.3
    elif abs_lat > 80:
        T_walls = T_sunlit * 0.5
    else:
        T_walls = T_sunlit * 0.7

    # Scattered solar radiation from walls (Lambertian scattering)
    # This depends on the solar flux and wall illumination
    solar_constant = 1361.0  # W/m² at 1 AU
    solar_flux = solar_constant * (1.0 - albedo) * max(0, np.sin(solar_elevation_deg * np.pi / 180.0))

    # Scattered radiation reaching shadow (simplified geometric factor)
    # This is a complex geometric integral - use approximate form
    geometric_factor = f_walls * albedo * 0.5  # Simplified
    irradiance_reflected = solar_flux * geometric_factor

    # Thermal infrared radiation from walls
    irradiance_thermal = f_walls * emissivity * SIGMA_SB * T_walls**4

    # Radiation from sky
    irradiance_sky = f_sky * emissivity * SIGMA_SB * T_sky**4

    # Total irradiance in permanent shadow
    irradiance_total = irradiance_reflected + irradiance_thermal + irradiance_sky

    # Temperature in permanent shadow from radiative equilibrium
    # emissivity * sigma * T^4 = irradiance_total
    T_shadow = (irradiance_total / (emissivity * SIGMA_SB))**0.25
    T_shadow = max(T_shadow, 30.0)  # Minimum temperature ~30K

    # Sunlit floor temperature (if any part is sunlit)
    # Receives direct solar + wall radiation
    if shadow_info['instantaneous_shadow_fraction'] < 1.0:
        direct_flux = solar_flux
        wall_thermal = f_walls * emissivity * SIGMA_SB * T_walls**4 * 0.3  # Reduced contribution
        T_sunlit_floor = ((direct_flux + wall_thermal) / (emissivity * SIGMA_SB))**0.25
    else:
        T_sunlit_floor = T_shadow

    return {
        'T_shadow': T_shadow,
        'T_sunlit_floor': T_sunlit_floor,
        'T_wall_avg': T_walls,
        'irradiance_reflected': irradiance_reflected,
        'irradiance_thermal': irradiance_thermal,
        'irradiance_total': irradiance_total,
        'shadow_fraction': shadow_info['permanent_shadow_fraction'],
        'view_factor_sky': f_sky,
        'view_factor_walls': f_walls
    }


def crater_temperature_profile_radial(crater: CraterGeometry,
                                       T_sunlit: float,
                                       solar_elevation_deg: float,
                                       n_points: int = 50,
                                       albedo: float = 0.12,
                                       emissivity: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate radial temperature profile across a bowl-shaped crater.

    Parameters:
    -----------
    crater : CraterGeometry
        Crater geometric parameters
    T_sunlit : float
        Temperature of sunlit terrain outside crater [K]
    solar_elevation_deg : float
        Solar elevation angle [degrees]
    n_points : int
        Number of points in radial profile
    albedo, emissivity : float
        Optical properties

    Returns:
    --------
    r_values : np.ndarray
        Radial distances from center [m]
    T_values : np.ndarray
        Temperatures at each radial position [K]
    """
    # Calculate shadow boundary
    shadow_info = crater_shadow_area_fraction(
        crater.gamma, crater.latitude_deg, solar_elevation_deg
    )

    # Shadow radius (simplified - assumes circular shadow)
    R = crater.radius
    r_shadow = R * np.sqrt(shadow_info['instantaneous_shadow_fraction'])

    # Base temperatures from Ingersoll model
    temps = ingersoll_crater_temperature(
        crater, T_sunlit, solar_elevation_deg, albedo, emissivity
    )

    # Create radial profile
    r_values = np.linspace(0, R, n_points)
    T_values = np.zeros(n_points)

    for i, r in enumerate(r_values):
        if r < r_shadow:
            # In shadow - use shadow temperature
            T_values[i] = temps['T_shadow']
        else:
            # Sunlit - interpolate between crater floor and rim
            # Transition from T_sunlit_floor to T_sunlit
            frac = (r - r_shadow) / (R - r_shadow) if R > r_shadow else 0
            T_values[i] = temps['T_sunlit_floor'] * (1 - frac) + T_sunlit * frac

    return r_values, T_values


def crater_cold_trap_area(crater: CraterGeometry,
                          T_threshold: float = 110.0,
                          solar_declination_deg: float = 1.54) -> Dict[str, float]:
    """
    Calculate cold trap area in a bowl-shaped crater.

    Integrates Ingersoll theory with thermal cold-trapping criteria.

    Parameters:
    -----------
    crater : CraterGeometry
        Crater geometric parameters
    T_threshold : float
        Temperature threshold for cold trapping [K], default 110 K
    solar_declination_deg : float
        Maximum solar declination [degrees], default 1.54 for Moon

    Returns:
    --------
    dict containing:
        - 'cold_trap_area': Area that remains below threshold [m²]
        - 'cold_trap_fraction': Fraction of crater area
        - 'permanent_shadow_area': Total permanent shadow area [m²]
        - 'is_cold_trap': Boolean, whether any cold trap exists
    """
    # Get permanent shadow area
    # Use minimum solar elevation (maximum zenith angle)
    e0_rad = (90.0 - abs(crater.latitude_deg)) * np.pi / 180.0
    min_elevation_deg = (e0_rad - solar_declination_deg * np.pi / 180.0) * 180.0 / np.pi

    shadow_info = crater_shadow_area_fraction(
        crater.gamma, crater.latitude_deg,
        max(0.1, min_elevation_deg),
        solar_declination_deg
    )

    # Total crater area
    A_crater = np.pi * crater.radius**2

    # Permanent shadow area
    A_perm_shadow = A_crater * shadow_info['permanent_shadow_fraction']

    # Estimate shadow temperature (simplified - use worst case)
    # For more accuracy, would integrate over time
    T_sunlit_avg = 250.0  # Typical high-latitude sunlit temperature

    temps = ingersoll_crater_temperature(
        crater, T_sunlit_avg, min_elevation_deg
    )

    # Determine if shadow temperature is below threshold
    is_cold_trap = temps['T_shadow'] < T_threshold

    # Cold trap area (conservative estimate)
    if is_cold_trap:
        A_cold_trap = A_perm_shadow
        cold_trap_frac = shadow_info['permanent_shadow_fraction']
    else:
        A_cold_trap = 0.0
        cold_trap_frac = 0.0

    return {
        'cold_trap_area': A_cold_trap,
        'cold_trap_fraction': cold_trap_frac,
        'permanent_shadow_area': A_perm_shadow,
        'permanent_shadow_fraction': shadow_info['permanent_shadow_fraction'],
        'is_cold_trap': is_cold_trap,
        'estimated_shadow_temp': temps['T_shadow'],
        'threshold_temp': T_threshold
    }


def typical_lunar_crater_examples():
    """
    Example calculations for typical lunar craters.

    Demonstrates usage with realistic crater parameters.
    """
    print("=" * 70)
    print("Bowl-Shaped Crater Temperature Calculations")
    print("Based on Ingersoll et al. (1992) and Hayne et al. (2021)")
    print("=" * 70)

    # Example 1: Small fresh crater at high latitude
    print("\n### Example 1: Small fresh crater (d/D ~ 0.14) at 85°S ###")
    crater1 = CraterGeometry(diameter=1000.0, depth=140.0, latitude_deg=-85.0)

    print(f"Diameter: {crater1.diameter} m")
    print(f"Depth: {crater1.depth} m")
    print(f"d/D ratio (γ): {crater1.gamma:.3f}")
    print(f"Beta parameter: {crater1.beta:.3f}")
    print(f"Latitude: {crater1.latitude_deg}°")

    # Calculate for noon conditions
    solar_elev = 5.0  # degrees
    temps1 = ingersoll_crater_temperature(crater1, T_sunlit=200.0, solar_elevation_deg=solar_elev)

    print(f"\nAt solar elevation {solar_elev}°:")
    print(f"  Shadow temperature: {temps1['T_shadow']:.1f} K")
    print(f"  Sunlit floor temperature: {temps1['T_sunlit_floor']:.1f} K")
    print(f"  Wall temperature: {temps1['T_wall_avg']:.1f} K")
    print(f"  Shadow fraction: {temps1['shadow_fraction']:.2%}")
    print(f"  View factor to sky: {temps1['view_factor_sky']:.2f}")
    print(f"  View factor to walls: {temps1['view_factor_walls']:.2f}")

    # Cold trap analysis
    cold_trap1 = crater_cold_trap_area(crater1)
    print(f"\nCold trap analysis:")
    print(f"  Is cold trap (< 110 K): {cold_trap1['is_cold_trap']}")
    print(f"  Estimated shadow temp: {cold_trap1['estimated_shadow_temp']:.1f} K")
    print(f"  Cold trap area: {cold_trap1['cold_trap_area']:.0f} m²")
    print(f"  Cold trap fraction: {cold_trap1['cold_trap_fraction']:.2%}")

    # Example 2: Larger degraded crater
    print("\n\n### Example 2: Larger degraded crater (d/D ~ 0.076) at 88°S ###")
    crater2 = CraterGeometry(diameter=10000.0, depth=760.0, latitude_deg=-88.0)

    print(f"Diameter: {crater2.diameter} m")
    print(f"Depth: {crater2.depth} m")
    print(f"d/D ratio (γ): {crater2.gamma:.3f}")
    print(f"Beta parameter: {crater2.beta:.3f}")

    temps2 = ingersoll_crater_temperature(crater2, T_sunlit=180.0, solar_elevation_deg=2.0)

    print(f"\nAt solar elevation 2°:")
    print(f"  Shadow temperature: {temps2['T_shadow']:.1f} K")
    print(f"  Shadow fraction: {temps2['shadow_fraction']:.2%}")

    cold_trap2 = crater_cold_trap_area(crater2)
    print(f"\nCold trap analysis:")
    print(f"  Is cold trap: {cold_trap2['is_cold_trap']}")
    print(f"  Cold trap area: {cold_trap2['cold_trap_area']:.0f} m² = {cold_trap2['cold_trap_area']/1e6:.3f} km²")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run examples
    typical_lunar_crater_examples()

    # Additional demonstration: radial temperature profile
    print("\n### Radial Temperature Profile Example ###")
    crater = CraterGeometry(diameter=5000.0, depth=400.0, latitude_deg=-85.0)
    r_vals, T_vals = crater_temperature_profile_radial(
        crater, T_sunlit=220.0, solar_elevation_deg=4.0, n_points=30
    )

    print(f"\nCrater: D={crater.diameter}m, d={crater.depth}m, γ={crater.gamma:.3f}")
    print(f"Radial position (m) | Temperature (K)")
    print("-" * 40)
    for r, T in zip(r_vals[::3], T_vals[::3]):  # Print every 3rd point
        print(f"{r:18.0f} | {T:14.1f}")

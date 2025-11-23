#!/usr/bin/env python3
"""
Cone-Shaped Crater Temperature Calculations

Models small degraded craters as inverted cones rather than spherical bowls.
Compares with Ingersoll et al. (1992) spherical bowl model to quantify deviations.

Degraded craters often develop cone-like or V-shaped profiles due to:
- Impact melt slumping
- Mass wasting and erosion
- Secondary crater gardening
- Long-term regolith infill

This module provides:
- Cone crater geometry and view factors
- Temperature calculations for cone geometry
- Comparison with Ingersoll spherical bowl model
- Updated microPSR theory for cone-shaped small craters
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from bowl_crater_thermal import CraterGeometry, SIGMA_SB


@dataclass
class ConeCraterGeometry:
    """
    Conical crater geometric parameters.

    For an inverted cone: depth increases linearly from rim to center.
    """
    diameter: float  # Crater diameter D [m]
    depth: float  # Crater depth d [m]
    latitude_deg: float  # Latitude [degrees]

    @property
    def gamma(self) -> float:
        """Depth-to-diameter ratio γ = d/D"""
        return self.depth / self.diameter

    @property
    def radius(self) -> float:
        """Crater radius R = D/2 [m]"""
        return self.diameter / 2.0

    @property
    def wall_slope_deg(self) -> float:
        """
        Wall slope angle [degrees].

        For a cone: slope = arctan(d / R) = arctan(2γ)
        """
        return np.arctan(2.0 * self.gamma) * 180.0 / np.pi

    @property
    def volume(self) -> float:
        """
        Cone volume [m³].

        V_cone = (1/3) * π * R² * d
        """
        return (np.pi / 3.0) * self.radius**2 * self.depth


def cone_depth_profile(r: np.ndarray, R: float, d: float) -> np.ndarray:
    """
    Depth profile for a conical crater.

    Parameters:
    -----------
    r : np.ndarray
        Radial distance from center [m]
    R : float
        Crater radius [m]
    d : float
        Maximum depth at center [m]

    Returns:
    --------
    np.ndarray
        Depth below rim at each radius [m]
    """
    # Linear profile: depth increases from rim (r=R) to center (r=0)
    # z(r) = d * (1 - r/R)
    return d * (1.0 - r / R)


def cone_view_factors(gamma: float) -> Dict[str, float]:
    """
    Calculate view factors for radiation exchange in a conical crater.

    For a cone, the view factor from floor to sky can be calculated exactly
    from geometric considerations.

    Parameters:
    -----------
    gamma : float
        Depth-to-diameter ratio d/D

    Returns:
    --------
    dict containing:
        - 'f_sky': View factor to sky from crater floor
        - 'f_walls': View factor to crater walls from floor
        - 'slope_angle_deg': Wall slope angle [degrees]
    """
    # Wall slope angle: θ = arctan(d/R) = arctan(2γ)
    theta_rad = np.arctan(2.0 * gamma)
    theta_deg = theta_rad * 180.0 / np.pi

    # View factor from floor to sky (from center of cone looking up)
    # For a point at the bottom of a cone with half-angle α (from vertical):
    # α = arctan(R/d) = arctan(1/(2γ))
    # View factor to sky: f_sky = (1 - cos(α)) / 2 for a cone opening
    # Actually, for a point source at bottom: f_sky = cos²(θ) where θ is slope

    # More accurate: For cone with slope angle θ from horizontal,
    # opening half-angle from vertical is α = 90° - θ
    alpha_rad = np.pi / 2.0 - theta_rad

    # View factor to sky for point at cone bottom
    # f_sky = sin²(alpha) = cos²(theta)
    f_sky = np.cos(theta_rad)**2

    # View factor to walls
    f_walls = 1.0 - f_sky

    return {
        'f_sky': f_sky,
        'f_walls': f_walls,
        'slope_angle_deg': theta_deg,
        'opening_half_angle_deg': alpha_rad * 180.0 / np.pi
    }


def compare_cone_vs_bowl_geometry(diameter: float, depth: float) -> Dict[str, float]:
    """
    Compare geometric properties of cone vs spherical bowl with same D and d.

    Parameters:
    -----------
    diameter : float
        Crater diameter [m]
    depth : float
        Crater depth [m]

    Returns:
    --------
    dict with comparison metrics
    """
    R = diameter / 2.0
    gamma = depth / diameter

    # Cone properties
    V_cone = (np.pi / 3.0) * R**2 * depth
    theta_cone = np.arctan(2.0 * gamma)

    # Bowl properties (from Ingersoll)
    R_sphere = (R**2 + depth**2) / (2.0 * depth)

    # Bowl volume (spherical cap)
    V_bowl = np.pi * depth**2 * (3*R_sphere - depth) / 3.0

    # View factors
    vf_cone = cone_view_factors(gamma)

    # Bowl view factor (approximate from bowl_crater_thermal.py)
    f_walls_bowl = min(gamma / 0.2, 0.7)
    f_sky_bowl = 1.0 - f_walls_bowl

    # Surface areas
    # Cone surface area: A = π * R * sqrt(R² + d²)
    A_cone = np.pi * R * np.sqrt(R**2 + depth**2)

    # Bowl surface area: A = 2π * R_sphere * d (spherical cap)
    A_bowl = 2.0 * np.pi * R_sphere * depth

    return {
        'volume_ratio': V_cone / V_bowl,
        'surface_area_ratio': A_cone / A_bowl,
        'cone_slope_deg': theta_cone * 180.0 / np.pi,
        'cone_f_sky': vf_cone['f_sky'],
        'bowl_f_sky': f_sky_bowl,
        'f_sky_ratio': vf_cone['f_sky'] / f_sky_bowl if f_sky_bowl > 0 else np.inf,
        'R_sphere': R_sphere,
        'V_cone': V_cone,
        'V_bowl': V_bowl,
        'A_cone': A_cone,
        'A_bowl': A_bowl
    }


def cone_shadow_fraction(gamma: float, latitude_deg: float,
                          solar_elevation_deg: float) -> Dict[str, float]:
    """
    Calculate shadow fraction in a conical crater.

    For a cone with slope angle θ, the shadow boundary depends on solar
    elevation angle e and azimuth. At noon (solar azimuth aligned with
    any crater radius), the shadow extent can be calculated geometrically.

    Parameters:
    -----------
    gamma : float
        Depth-to-diameter ratio d/D
    latitude_deg : float
        Latitude [degrees]
    solar_elevation_deg : float
        Solar elevation angle [degrees]

    Returns:
    --------
    dict with shadow properties
    """
    R = 1.0  # Normalized radius
    d = gamma * 2.0 * R

    theta_wall = np.arctan(2.0 * gamma)  # Wall slope from horizontal
    e_rad = solar_elevation_deg * np.pi / 180.0

    # Shadow boundary: where sun rays tangent to cone wall intersect floor
    # For cone, at solar elevation e, shadow reaches to radius r_shadow
    # Geometric relation: tan(e + θ_wall) determines shadow extent

    # If solar elevation < wall slope, entire crater is in shadow
    if e_rad <= theta_wall:
        shadow_frac = 1.0
        r_shadow_norm = 1.0
    else:
        # Shadow shrinks as sun rises
        # Approximate: r_shadow/R ≈ tan(θ_wall) / tan(e)
        r_shadow_norm = np.tan(theta_wall) / np.tan(e_rad)
        r_shadow_norm = min(r_shadow_norm, 1.0)
        shadow_frac = r_shadow_norm**2  # Area fraction

    return {
        'shadow_fraction': shadow_frac,
        'shadow_radius_norm': r_shadow_norm,
        'wall_slope_rad': theta_wall,
        'critical_elevation_deg': theta_wall * 180.0 / np.pi
    }


def cone_crater_temperature(cone: ConeCraterGeometry,
                             T_sunlit: float,
                             solar_elevation_deg: float,
                             albedo: float = 0.12,
                             emissivity: float = 0.95,
                             T_sky: float = 3.0) -> Dict[str, float]:
    """
    Calculate temperature in a conical crater.

    Similar to Ingersoll model but with cone geometry and view factors.

    Parameters:
    -----------
    cone : ConeCraterGeometry
        Cone crater parameters
    T_sunlit : float
        Temperature of sunlit terrain [K]
    solar_elevation_deg : float
        Solar elevation angle [degrees]
    albedo : float
        Bond albedo
    emissivity : float
        Thermal emissivity
    T_sky : float
        Sky temperature [K]

    Returns:
    --------
    dict with temperature results
    """
    # Get view factors for cone
    vf = cone_view_factors(cone.gamma)
    f_sky = vf['f_sky']
    f_walls = vf['f_walls']

    # Get shadow fraction
    shadow = cone_shadow_fraction(cone.gamma, cone.latitude_deg, solar_elevation_deg)

    # Estimate wall temperature
    abs_lat = abs(cone.latitude_deg)
    if abs_lat > 85:
        T_walls = T_sunlit * 0.35  # Slightly warmer than bowl (less thermal mass)
    elif abs_lat > 80:
        T_walls = T_sunlit * 0.55
    else:
        T_walls = T_sunlit * 0.75

    # Solar radiation components
    solar_constant = 1361.0  # W/m²
    solar_flux = solar_constant * (1.0 - albedo) * max(0, np.sin(solar_elevation_deg * np.pi / 180.0))

    # Scattered radiation from walls
    # Cone walls are at constant slope, simpler geometry
    geometric_factor = f_walls * albedo * 0.6  # Enhanced for linear geometry
    irradiance_reflected = solar_flux * geometric_factor

    # Thermal radiation from walls
    irradiance_thermal = f_walls * emissivity * SIGMA_SB * T_walls**4

    # Sky radiation
    irradiance_sky = f_sky * emissivity * SIGMA_SB * T_sky**4

    # Total irradiance in shadow
    irradiance_total = irradiance_reflected + irradiance_thermal + irradiance_sky

    # Shadow temperature
    T_shadow = (irradiance_total / (emissivity * SIGMA_SB))**0.25
    T_shadow = max(T_shadow, 30.0)

    # Sunlit floor temperature
    if shadow['shadow_fraction'] < 1.0:
        direct_flux = solar_flux
        wall_thermal = f_walls * emissivity * SIGMA_SB * T_walls**4 * 0.2
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
        'shadow_fraction': shadow['shadow_fraction'],
        'view_factor_sky': f_sky,
        'view_factor_walls': f_walls,
        'wall_slope_deg': vf['slope_angle_deg']
    }


def compare_cone_vs_ingersoll_temperature(diameter: float,
                                           depth: float,
                                           latitude_deg: float,
                                           T_sunlit: float,
                                           solar_elevation_deg: float) -> Dict[str, float]:
    """
    Direct comparison of cone vs Ingersoll (bowl) temperature predictions.

    Parameters:
    -----------
    diameter : float
        Crater diameter [m]
    depth : float
        Crater depth [m]
    latitude_deg : float
        Latitude [degrees]
    T_sunlit : float
        Sunlit terrain temperature [K]
    solar_elevation_deg : float
        Solar elevation [degrees]

    Returns:
    --------
    dict with comparison results
    """
    # Import bowl model
    from bowl_crater_thermal import ingersoll_crater_temperature

    # Create crater geometries
    cone = ConeCraterGeometry(diameter, depth, latitude_deg)
    bowl = CraterGeometry(diameter, depth, latitude_deg)

    # Calculate temperatures
    cone_temps = cone_crater_temperature(cone, T_sunlit, solar_elevation_deg)
    bowl_temps = ingersoll_crater_temperature(bowl, T_sunlit, solar_elevation_deg)

    # Temperature differences
    dT_shadow = cone_temps['T_shadow'] - bowl_temps['T_shadow']
    dT_floor = cone_temps['T_sunlit_floor'] - bowl_temps['T_sunlit_floor']

    # Fractional differences
    frac_shadow = dT_shadow / bowl_temps['T_shadow'] if bowl_temps['T_shadow'] > 0 else 0

    return {
        'cone_T_shadow': cone_temps['T_shadow'],
        'bowl_T_shadow': bowl_temps['T_shadow'],
        'delta_T_shadow': dT_shadow,
        'fractional_T_diff': frac_shadow,
        'cone_f_sky': cone_temps['view_factor_sky'],
        'bowl_f_sky': bowl_temps['view_factor_sky'],
        'cone_shadow_frac': cone_temps['shadow_fraction'],
        'bowl_shadow_frac': bowl_temps['shadow_fraction'],
        'cone_irradiance_total': cone_temps['irradiance_total'],
        'bowl_irradiance_total': bowl_temps['irradiance_total'],
        'cone_wall_slope': cone_temps['wall_slope_deg']
    }


def cone_micro_cold_trap_fraction(cone: ConeCraterGeometry,
                                   rms_slope_deg: float,
                                   model: str = 'hayne2021') -> Dict[str, float]:
    """
    Calculate micro cold trap fraction for cone-shaped small craters.

    Cone craters have:
    1. Baseline slope from cone geometry (wall_slope_deg)
    2. Additional roughness from rms_slope_deg
    3. Different shadowing statistics than bowls

    Parameters:
    -----------
    cone : ConeCraterGeometry
        Cone crater geometry
    rms_slope_deg : float
        RMS slope of surface roughness [degrees]
    model : str
        Model for cold trap estimation

    Returns:
    --------
    dict with micro cold trap properties
    """
    from thermal_model import rough_surface_cold_trap_fraction

    # Effective slope: combination of crater wall slope and surface roughness
    # Use RSS (root sum square) for independent slope contributions
    wall_slope = cone.wall_slope_deg
    effective_rms = np.sqrt(wall_slope**2 + rms_slope_deg**2)

    # Base cold trap fraction from rough surface model
    base_frac = rough_surface_cold_trap_fraction(
        effective_rms, cone.latitude_deg, model=model
    )

    # Cone geometry enhancement factor
    # Cones have more uniform slope distribution → more cold traps
    # vs bowls which have variable slopes
    cone_enhancement = 1.2  # 20% more cold traps for cone vs bowl

    adjusted_frac = min(base_frac * cone_enhancement, 1.0)

    return {
        'micro_cold_trap_fraction': adjusted_frac,
        'base_fraction': base_frac,
        'wall_slope_deg': wall_slope,
        'surface_rms_slope_deg': rms_slope_deg,
        'effective_rms_slope_deg': effective_rms,
        'cone_enhancement_factor': cone_enhancement
    }


def cone_integrated_sublimation(species,
                                 cone: ConeCraterGeometry,
                                 rms_slope_deg: float = 20,
                                 alpha: float = 1.0) -> Dict[str, float]:
    """
    Calculate sublimation rate for volatiles in cone-shaped crater with microPSRs.

    Integrates:
    - Cone crater thermal model
    - Micro cold trap theory
    - Sublimation calculations

    Parameters:
    -----------
    species : VolatileSpecies
        Volatile species (from vaporp_temp.py)
    cone : ConeCraterGeometry
        Cone crater geometry
    rms_slope_deg : float
        Surface roughness RMS slope [degrees]
    alpha : float
        Sticking coefficient

    Returns:
    --------
    dict with sublimation results
    """
    from vaporp_temp import calculate_mixed_pixel_sublimation

    # Estimate illuminated temperature (simplified)
    lat_rad = cone.latitude_deg * np.pi / 180.0
    solar_constant = 1361.0
    albedo = 0.12
    mean_flux = solar_constant * (1.0 - albedo) * abs(np.cos(lat_rad))

    # Rough radiative equilibrium
    from thermal_model import radiative_equilibrium_temperature
    T_illuminated = radiative_equilibrium_temperature(mean_flux, albedo)

    # Cone shadow temperature
    solar_elev = 90.0 - abs(cone.latitude_deg) + 1.54  # Rough estimate
    temps = cone_crater_temperature(cone, T_illuminated, solar_elev)
    T_cold_trap = temps['T_shadow']

    # Micro cold trap fraction accounting for cone geometry
    ct_result = cone_micro_cold_trap_fraction(cone, rms_slope_deg)
    cold_trap_frac = ct_result['micro_cold_trap_fraction']

    # Calculate mixed-pixel sublimation
    sublim = calculate_mixed_pixel_sublimation(
        species, T_illuminated, cold_trap_frac,
        cold_trap_temp=T_cold_trap, alpha=alpha
    )

    # Add cone-specific information
    result = {
        **sublim,
        'cone_wall_slope_deg': cone.wall_slope_deg,
        'effective_rms_slope_deg': ct_result['effective_rms_slope_deg'],
        'T_illuminated': T_illuminated,
        'T_cold_trap_cone': T_cold_trap,
        'cone_enhancement': ct_result['cone_enhancement_factor']
    }

    return result


if __name__ == "__main__":
    print("=" * 80)
    print("Cone-Shaped Crater vs Ingersoll Bowl Model Comparison")
    print("=" * 80)

    # Test case: Small degraded crater
    D = 500.0  # 500 m diameter
    d = 50.0   # 50 m depth (γ = 0.1, typical for degraded crater)
    lat = -85.0

    print(f"\nCrater parameters:")
    print(f"  Diameter: {D} m")
    print(f"  Depth: {d} m")
    print(f"  d/D ratio (γ): {d/D:.3f}")
    print(f"  Latitude: {lat}°")

    # Geometric comparison
    print("\n" + "-" * 80)
    print("GEOMETRIC COMPARISON")
    print("-" * 80)

    geom = compare_cone_vs_bowl_geometry(D, d)

    print(f"\nVolume:")
    print(f"  Cone: {geom['V_cone']:.2e} m³")
    print(f"  Bowl: {geom['V_bowl']:.2e} m³")
    print(f"  Ratio: {geom['volume_ratio']:.3f}")

    print(f"\nSurface Area:")
    print(f"  Cone: {geom['A_cone']:.2e} m²")
    print(f"  Bowl: {geom['A_bowl']:.2e} m²")
    print(f"  Ratio: {geom['surface_area_ratio']:.3f}")

    print(f"\nView Factors to Sky:")
    print(f"  Cone: {geom['cone_f_sky']:.3f}")
    print(f"  Bowl: {geom['bowl_f_sky']:.3f}")
    print(f"  Ratio (cone/bowl): {geom['f_sky_ratio']:.3f}")
    print(f"  → Cone sees {(geom['f_sky_ratio']-1)*100:+.1f}% more sky")

    print(f"\nCone wall slope: {geom['cone_slope_deg']:.1f}°")

    # Temperature comparison
    print("\n" + "-" * 80)
    print("TEMPERATURE COMPARISON")
    print("-" * 80)

    T_sunlit = 200.0
    solar_elev = 5.0

    print(f"\nConditions:")
    print(f"  Sunlit terrain: {T_sunlit} K")
    print(f"  Solar elevation: {solar_elev}°")

    comparison = compare_cone_vs_ingersoll_temperature(D, d, lat, T_sunlit, solar_elev)

    print(f"\nShadow Temperature:")
    print(f"  Cone model: {comparison['cone_T_shadow']:.2f} K")
    print(f"  Ingersoll (bowl): {comparison['bowl_T_shadow']:.2f} K")
    print(f"  Difference (cone - bowl): {comparison['delta_T_shadow']:+.2f} K")
    print(f"  Fractional difference: {comparison['fractional_T_diff']*100:+.1f}%")

    print(f"\nTotal Irradiance in Shadow:")
    print(f"  Cone: {comparison['cone_irradiance_total']:.3f} W/m²")
    print(f"  Bowl: {comparison['bowl_irradiance_total']:.3f} W/m²")

    if comparison['delta_T_shadow'] < 0:
        print("\n→ CONE IS COLDER (more sky view, less wall radiation)")
    else:
        print("\n→ CONE IS WARMER")

    # MicroPSR analysis
    print("\n" + "-" * 80)
    print("MICRO-PSR ANALYSIS FOR CONE-SHAPED CRATERS")
    print("-" * 80)

    from vaporp_temp import VOLATILE_SPECIES

    cone = ConeCraterGeometry(D, d, lat)
    species = VOLATILE_SPECIES['H2O']
    rms_slope = 25.0  # degrees

    print(f"\nMicro-PSR conditions:")
    print(f"  Surface roughness: {rms_slope}° RMS slope")
    print(f"  Cone wall slope: {cone.wall_slope_deg:.1f}°")

    result = cone_integrated_sublimation(species, cone, rms_slope)

    print(f"\nEffective slopes:")
    print(f"  Wall slope: {result['cone_wall_slope_deg']:.1f}°")
    print(f"  Surface RMS: {rms_slope}°")
    print(f"  Combined effective: {result['effective_rms_slope_deg']:.1f}°")

    print(f"\nCold trap properties:")
    print(f"  Fraction: {result['cold_trap_fraction']:.1%}")
    print(f"  Temperature: {result['T_cold_trap_cone']:.1f} K")
    print(f"  Enhancement factor: {result['cone_enhancement']:.2f}×")

    print(f"\nSublimation rates:")
    print(f"  Mixed pixel: {result['mixed_sublimation_rate_kg_m2_yr']:.2e} kg/(m²·yr)")
    print(f"  Cold trap only: {result['cold_trap_only_rate_kg_m2_yr']:.2e} kg/(m²·yr)")
    print(f"  Reduction factor: {result['sublimation_reduction_factor']:.1f}×")

    print(f"\nIce retention:")
    # Calculate 1m ice lifetime from sublimation rate
    # 1m of ice = 920 kg/m² (assuming ice density)
    ice_mass_kg_m2 = 920.0
    if result['cold_trap_only_rate_kg_m2_yr'] > 0:
        lifetime_years = ice_mass_kg_m2 / result['cold_trap_only_rate_kg_m2_yr']
        print(f"  1m ice lifetime in CT: {lifetime_years:.2e} years")
    else:
        print(f"  1m ice lifetime in CT: Stable (essentially infinite)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Cone vs Ingersoll Deviations")
    print("=" * 80)

    print(f"\n1. GEOMETRIC DIFFERENCES:")
    print(f"   - Cone is {(1-geom['volume_ratio'])*100:.1f}% less voluminous")
    print(f"   - Cone has {(geom['surface_area_ratio']-1)*100:+.1f}% different surface area")
    print(f"   - Cone sees {(geom['f_sky_ratio']-1)*100:+.1f}% more sky from floor")

    print(f"\n2. TEMPERATURE DIFFERENCES:")
    print(f"   - Shadow temp differs by {abs(comparison['delta_T_shadow']):.1f} K")
    print(f"   - ({abs(comparison['fractional_T_diff'])*100:.1f}% relative difference)")
    print(f"   - Cone receives different wall radiation due to geometry")

    print(f"\n3. MICRO-PSR IMPLICATIONS:")
    print(f"   - Cone wall slope adds to effective roughness")
    print(f"   - {result['cone_enhancement']:.0%} enhancement in cold trap fraction")
    print(f"   - Combined slope of {result['effective_rms_slope_deg']:.0f}° vs ~{rms_slope:.0f}° for bowl")

    print(f"\n4. APPLICABILITY:")
    if abs(comparison['fractional_T_diff']) < 0.10:
        print("   ✓ Ingersoll model is adequate (<10% error)")
    elif abs(comparison['fractional_T_diff']) < 0.25:
        print("   ~ Ingersoll model reasonable with corrections (10-25% error)")
    else:
        print("   ✗ Significant deviation - cone model recommended (>25% error)")

    print("\n" + "=" * 80)

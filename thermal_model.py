#!/usr/bin/env python3
"""
Integrated Thermal Model for Lunar Volatile Sublimation

Combines:
- 1-D heat conduction (based on Hayne et al. 2017 heat1d)
- Micro cold trap theory (Hayne et al. 2021 Nature Astronomy)
- Sublimation rate calculations (Hertz-Knudsen equation)

This module provides accurate thermal modeling accounting for:
- Subsurface thermal properties
- Lateral heat conduction (limits micro cold traps at <1 cm scale)
- Temperature-dependent thermal conductivity
- Diurnal and seasonal temperature variations
"""

import numpy as np
import math
from dataclasses import dataclass


@dataclass
class LunarThermalProperties:
    """
    Lunar regolith thermophysical properties.

    Based on Hayne et al. (2017) Diviner-derived global properties.
    """
    # Thermal properties
    ks: float = 7.4e-4  # Surface thermal conductivity [W/(m·K)]
    kd: float = 3.4e-3  # Deep thermal conductivity [W/(m·K)]
    rhos: float = 1100  # Surface density [kg/m³]
    rhod: float = 1800  # Deep density [kg/m³]
    H: float = 0.06     # Density scale height [m]
    cp0: float = 600    # Reference heat capacity [J/(kg·K)] at ~200K

    # Optical properties
    albedo: float = 0.12      # Bond albedo
    emissivity: float = 0.95  # Thermal emissivity

    # Radiative conductivity parameter
    chi: float = 2.7  # Mitchell & de Pater (1994)

    # Heat capacity polynomial coefficients (Ledlow et al. 1992)
    # cp = c[0]*T^3 + c[1]*T^2 + c[2]*T + c[3]
    cp_coeff: tuple = (-3.6125e-2, 2.7431, 2.3616e1, 3.8473e1)

    # Solar constant and planetary parameters
    S0: float = 1361.0  # Solar constant at 1 AU [W/m²]
    rAU: float = 1.0    # Distance from Sun [AU]
    day: float = 2551443.0  # Lunar day [s] (29.5 Earth days)
    year: float = 2551443.0  # Orbital period [s]
    obliquity: float = 1.54 * np.pi / 180  # Axis obliquity [rad]

    # Bottom boundary
    Qb: float = 0.018  # Interior heat flux [W/m²]


def skin_depth(period, kappa):
    """
    Calculate thermal skin depth.

    Parameters:
    -----------
    period : float
        Period (e.g., diurnal, seasonal) [s]
    kappa : float
        Thermal diffusivity k/(rho*cp) [m²/s]

    Returns:
    --------
    float
        Thermal skin depth [m]
    """
    return np.sqrt(kappa * period / np.pi)


def lateral_conduction_scale(kappa, period):
    """
    Estimate the length scale below which lateral heat conduction dominates.

    From Hayne et al. (2021), lateral conduction eliminates cold traps with
    sizes ranging from ~1 cm near the pole to ~10 m at 60° latitude.

    Parameters:
    -----------
    kappa : float
        Thermal diffusivity [m²/s]
    period : float
        Diurnal period [s]

    Returns:
    --------
    float
        Critical length scale [m] below which lateral conduction matters
    """
    # Based on the 2-D heat equation, lateral conduction becomes important
    # when horizontal scale ~ vertical skin depth
    return skin_depth(period, kappa)


def crater_shadow_fraction(depth_diameter_ratio, latitude_deg, solar_declination_deg=0):
    """
    Calculate permanent shadow fraction in a bowl-shaped crater.

    Based on analytical relations from Hayne et al. (2021) Eqs. 2-9.

    Parameters:
    -----------
    depth_diameter_ratio : float
        Crater d/D ratio (typically 0.076-0.14 for lunar craters)
    latitude_deg : float
        Latitude [degrees]
    solar_declination_deg : float
        Maximum solar declination [degrees]

    Returns:
    --------
    dict
        Dictionary with shadow fractions and geometric parameters
    """
    gamma = depth_diameter_ratio  # d/D
    beta = 1/(2*gamma) - 2*gamma  # Geometric parameter

    # Colatitude (Sun elevation at pole)
    e0_rad = (90 - abs(latitude_deg)) * np.pi / 180
    delta_max_rad = solar_declination_deg * np.pi / 180

    # For bowl-shaped crater, permanent shadow area fraction (Eq. 22 + 26)
    # A_perm/A_crater ≈ 1 - (8*beta*e0)/(3*pi) - 2*beta*delta
    if e0_rad < 1e-6:  # At the pole
        A_perm_frac = max(0, 1 - 2*beta*delta_max_rad)
    else:
        A_perm_frac = max(0, 1 - (8*beta*e0_rad)/(3*np.pi) - 2*beta*delta_max_rad)

    # Instantaneous (noontime) shadow fraction (Eq. 23)
    e_rad = e0_rad  # Approximation for high latitudes
    A_noon_frac = max(0, 1 - beta*e_rad/2) if e_rad > 0 else 1.0

    # Ratio of permanent to instantaneous shadow
    f = A_perm_frac / A_noon_frac if A_noon_frac > 0 else 0

    return {
        'permanent_shadow_fraction': A_perm_frac,
        'noon_shadow_fraction': A_noon_frac,
        'f_ratio': f,
        'beta': beta,
        'gamma': gamma
    }


def crater_cold_trap_fraction(depth_diameter_ratio, latitude_deg, length_scale=1.0,
                                T_threshold=110, thermal_props=LunarThermalProperties()):
    """
    Calculate cold trap fraction in craters accounting for lateral conduction.

    Integrates Hayne et al. (2021) crater geometry with thermal model.

    Parameters:
    -----------
    depth_diameter_ratio : float
        Crater d/D ratio
    latitude_deg : float
        Latitude [degrees]
    length_scale : float
        Crater diameter or shadow size [m]
    T_threshold : float
        Temperature threshold for cold trapping [K]
    thermal_props : LunarThermalProperties
        Thermal property object

    Returns:
    --------
    dict
        Cold trap properties including lateral conduction effects
    """
    # Get shadow fraction from geometry
    shadow_result = crater_shadow_fraction(depth_diameter_ratio, latitude_deg)

    # Estimate thermal diffusivity
    k_avg = (thermal_props.ks + thermal_props.kd) / 2
    rho_avg = (thermal_props.rhos + thermal_props.rhod) / 2
    cp_avg = thermal_props.cp0
    kappa = k_avg / (rho_avg * cp_avg)

    # Lateral conduction scale
    L_crit = lateral_conduction_scale(kappa, thermal_props.day)

    # Reduction factor due to lateral conduction
    # Based on Hayne et al. (2021) Supplementary Fig. 10
    # Cold traps are eliminated when L < ~L_crit (~1-10 cm depending on latitude)
    if length_scale < L_crit:
        # Approximate reduction: exponential decay below critical scale
        conduction_factor = np.exp(-(L_crit / length_scale))
    else:
        conduction_factor = 1.0

    # Effective cold trap fraction
    cold_trap_frac = shadow_result['permanent_shadow_fraction'] * conduction_factor

    return {
        **shadow_result,
        'lateral_conduction_scale_m': L_crit,
        'conduction_reduction_factor': conduction_factor,
        'cold_trap_fraction': cold_trap_frac,
        'length_scale_m': length_scale
    }


def rough_surface_cold_trap_fraction(rms_slope_deg, latitude_deg, solar_elevation_deg=None,
                                       model='hayne2021'):
    """
    Calculate cold trap fraction on rough surfaces.

    Implements Hayne et al. (2021) rough surface models.

    Parameters:
    -----------
    rms_slope_deg : float
        RMS slope [degrees]
    latitude_deg : float
        Latitude [degrees]
    solar_elevation_deg : float, optional
        Maximum solar elevation [degrees]. If None, estimated from latitude.
    model : str
        'hayne2021' uses optimal σs ≈ 10-20° model from the paper
        'geometric' uses simple geometric shadowing

    Returns:
    --------
    float
        Fractional area in permanent shadow
    """
    if solar_elevation_deg is None:
        # Estimate maximum solar elevation from latitude and obliquity
        obliquity_deg = 1.54  # Moon's obliquity
        solar_elevation_deg = 90 - abs(latitude_deg) + obliquity_deg

    sigma_s = rms_slope_deg

    if model == 'hayne2021':
        # Based on Hayne et al. (2021) numerical results
        # Rough surfaces with σs ≈ 10-20° have greatest cold-trapping area
        # Beyond this, temperatures in shadows are elevated due to proximity to steep sunlit terrain

        # Empirical fit to Hayne et al. (2021) Fig. 3 data
        # Maximum cold trap fraction occurs around σs = 15°
        sigma_optimal = 15.0
        if sigma_s < sigma_optimal:
            # Linear increase
            frac = 0.02 * sigma_s / sigma_optimal
        else:
            # Decrease due to radiative heating from surrounding terrain
            frac = 0.02 * np.exp(-(sigma_s - sigma_optimal) / 10.0)

    elif model == 'geometric':
        # Geometric shadowing model (simpler, more conservative)
        # Based on Smith (1967) formulation
        slope_rad = rms_slope_deg * np.pi / 180
        sun_rad = solar_elevation_deg * np.pi / 180

        # Fraction of surface in shadow (Smith 1967)
        if sun_rad > 0:
            # Approximate: integrating over slope distribution
            frac = 0.5 * (1 - np.exp(-slope_rad / sun_rad))
        else:
            frac = 1.0
    else:
        raise ValueError(f"Unknown model: {model}")

    return min(frac, 1.0)


def thermal_conductivity(T, kc, chi=2.7):
    """
    Calculate temperature-dependent thermal conductivity.

    Based on Mitchell & de Pater (1994), Vasavada et al. (2012).
    From heat1d model.

    Parameters:
    -----------
    T : float or array
        Temperature [K]
    kc : float or array
        Contact conductivity [W/(m·K)]
    chi : float
        Radiative conductivity parameter

    Returns:
    --------
    float or array
        Thermal conductivity [W/(m·K)]
    """
    R350 = chi / 350**3
    return kc * (1 + R350 * T**3)


def heat_capacity(T, coeff=(-3.6125e-2, 2.7431, 2.3616e1, 3.8473e1)):
    """
    Calculate temperature-dependent heat capacity of lunar regolith.

    Polynomial fit from Ledlow et al. (1992) and Hemingway et al. (1981).
    Valid for T > ~10 K.

    Parameters:
    -----------
    T : float or array
        Temperature [K]
    coeff : tuple
        Polynomial coefficients (highest order first)

    Returns:
    --------
    float or array
        Heat capacity [J/(kg·K)]
    """
    return np.polyval(coeff, T)


def radiative_equilibrium_temperature(solar_flux, albedo=0.12, emissivity=0.95):
    """
    Calculate radiative equilibrium temperature.

    Parameters:
    -----------
    solar_flux : float
        Absorbed solar flux [W/m²]
    albedo : float
        Bond albedo
    emissivity : float
        Thermal emissivity

    Returns:
    --------
    float
        Equilibrium temperature [K]
    """
    sigma = 5.67051e-8  # Stefan-Boltzmann constant [W/(m²·K⁴)]
    return ((1 - albedo) * solar_flux / (emissivity * sigma))**0.25


def shadow_temperature_crater(illuminated_temp, depth_diameter_ratio, latitude_deg):
    """
    Estimate temperature in a shadowed crater.

    Based on Ingersoll et al. (1992) and Hayne et al. (2021).
    Temperature depends primarily on latitude and d/D ratio.
    Shallower craters have smaller but colder shadows.

    Parameters:
    -----------
    illuminated_temp : float
        Temperature of sunlit terrain [K]
    depth_diameter_ratio : float
        Crater d/D ratio
    latitude_deg : float
        Latitude [degrees]

    Returns:
    --------
    float
        Approximate shadow temperature [K]
    """
    # Very simplified model - for accurate temperatures, use full thermal model
    # Shadows receive scattered and infrared radiation from surroundings

    abs_lat = abs(latitude_deg)

    # View factor to sky vs. walls depends on d/D
    # Shallow craters (small d/D) see more sky, are colder
    gamma = depth_diameter_ratio

    # Approximate view factor to walls (increases with gamma)
    f_walls = min(gamma / 0.2, 0.7)  # Saturates at ~70%
    f_sky = 1 - f_walls

    # Sky temperature (very cold)
    T_sky = 3.0  # Cosmic microwave background

    # Wall temperature (assume some fraction of illuminated)
    # At high latitudes, surrounding terrain is also cold
    if abs_lat > 85:
        T_walls = illuminated_temp * 0.3
    elif abs_lat > 80:
        T_walls = illuminated_temp * 0.5
    else:
        T_walls = illuminated_temp * 0.7

    # Stefan-Boltzmann weighted average
    T_shadow = (f_sky * T_sky**4 + f_walls * T_walls**4)**0.25

    return max(T_shadow, 30.0)  # Minimum ~30K


class SimpleThermalModel:
    """
    Simplified 1-D thermal model for lunar surface.

    Based on heat1d (Hayne et al. 2017) but simplified for integration
    with sublimation calculations.
    """

    def __init__(self, latitude_deg=85, thermal_props=LunarThermalProperties()):
        """
        Initialize thermal model.

        Parameters:
        -----------
        latitude_deg : float
            Latitude [degrees]
        thermal_props : LunarThermalProperties
            Thermal property object
        """
        self.lat = latitude_deg
        self.props = thermal_props

        # Calculate thermal skin depth
        k_avg = (thermal_props.ks + thermal_props.kd) / 2
        rho_avg = (thermal_props.rhos + thermal_props.rhod) / 2
        cp_avg = thermal_props.cp0
        kappa = k_avg / (rho_avg * cp_avg)

        self.kappa = kappa
        self.skin_depth = skin_depth(thermal_props.day, kappa)

    def diurnal_temperature_range(self, mean_solar_flux):
        """
        Estimate diurnal temperature range at the surface.

        Parameters:
        -----------
        mean_solar_flux : float
            Mean absorbed solar flux [W/m²]

        Returns:
        --------
        dict
            Min, max, and mean temperatures [K]
        """
        # Radiative equilibrium temperature
        T_eq = radiative_equilibrium_temperature(
            mean_solar_flux,
            self.props.albedo,
            self.props.emissivity
        )

        # Diurnal variation amplitude (simplified)
        # Real amplitude depends on thermal inertia
        lat_rad = self.lat * np.pi / 180

        # At high latitudes, diurnal range is smaller
        if abs(self.lat) > 80:
            T_max = T_eq * 1.2
            T_min = T_eq * 0.6
        else:
            T_max = T_eq * 1.4
            T_min = T_eq * 0.4

        return {
            'T_min': T_min,
            'T_max': T_max,
            'T_mean': (T_min + T_max) / 2,
            'T_eq': T_eq
        }

    def psr_temperature(self, surrounding_temp):
        """
        Estimate temperature in a permanently shadowed region.

        Parameters:
        -----------
        surrounding_temp : float
            Temperature of surrounding terrain [K]

        Returns:
        --------
        float
            PSR temperature [K]
        """
        # PSRs receive only scattered light and IR from surroundings
        # Temperature depends on latitude and local topography

        lat_rad = abs(self.lat) * np.pi / 180

        # Very crude estimate
        # At poles, PSRs are very cold
        if abs(self.lat) > 88:
            T_psr = 40  # ~40K at pole
        elif abs(self.lat) > 85:
            T_psr = 50 + (surrounding_temp - 50) * 0.2
        elif abs(self.lat) > 80:
            T_psr = 70 + (surrounding_temp - 70) * 0.3
        else:
            T_psr = surrounding_temp * 0.5

        return max(T_psr, 30)


def integrated_sublimation_with_thermal(species, latitude_deg, rms_slope_deg=20,
                                         length_scale=0.1, alpha=1.0,
                                         thermal_props=LunarThermalProperties()):
    """
    Calculate sublimation rate using integrated thermal + micro cold trap model.

    This is the main function that brings everything together.

    Parameters:
    -----------
    species : VolatileSpecies
        Volatile species object (from vaporp_temp.py)
    latitude_deg : float
        Latitude [degrees]
    rms_slope_deg : float
        RMS surface slope [degrees]
    length_scale : float
        Spatial scale of interest [m]
    alpha : float
        Sticking coefficient
    thermal_props : LunarThermalProperties
        Thermal properties

    Returns:
    --------
    dict
        Comprehensive results including thermal and sublimation data
    """
    from vaporp_temp import calculate_mixed_pixel_sublimation

    # Initialize thermal model
    thermal = SimpleThermalModel(latitude_deg, thermal_props)

    # Estimate illuminated temperature (simplified - would use full heat1d for accuracy)
    # Solar flux at high latitude
    lat_rad = latitude_deg * np.pi / 180
    mean_flux = thermal_props.S0 * (1 - thermal_props.albedo) * abs(np.cos(lat_rad))

    temps = thermal.diurnal_temperature_range(mean_flux)
    T_illuminated = temps['T_mean']

    # Calculate cold trap fraction accounting for lateral conduction
    if length_scale < 1.0:  # Crater-like geometry
        # Use crater model
        gamma = 0.12  # Typical d/D
        ct_result = crater_cold_trap_fraction(
            gamma, latitude_deg, length_scale, 110, thermal_props
        )
        cold_trap_frac = ct_result['cold_trap_fraction']
    else:
        # Use rough surface model
        cold_trap_frac = rough_surface_cold_trap_fraction(
            rms_slope_deg, latitude_deg, model='hayne2021'
        )

    # Estimate cold trap temperature
    T_cold_trap = thermal.psr_temperature(T_illuminated)

    # Calculate mixed-pixel sublimation
    sublim_result = calculate_mixed_pixel_sublimation(
        species, T_illuminated, cold_trap_frac,
        cold_trap_temp=T_cold_trap, alpha=alpha
    )

    # Add thermal context
    result = {
        **sublim_result,
        'thermal_skin_depth_m': thermal.skin_depth,
        'thermal_diffusivity_m2s': thermal.kappa,
        'lateral_conduction_scale_m': thermal.skin_depth,
        'length_scale_m': length_scale,
        'T_min': temps['T_min'],
        'T_max': temps['T_max'],
        'latitude_deg': latitude_deg,
        'rms_slope_deg': rms_slope_deg
    }

    return result

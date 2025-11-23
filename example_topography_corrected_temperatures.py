#!/usr/bin/env python3
"""
Example: Temperature Calculations with Topographic Deviation Corrections

Demonstrates how to use topographic deviation analysis to:
1. Assess validity of Ingersoll bowl-shaped model
2. Apply corrections to temperature calculations
3. Estimate uncertainties due to topographic irregularities

This bridges analytical Ingersoll theory with detailed topographic modeling.
"""

import numpy as np
from bowl_crater_thermal import (
    CraterGeometry,
    ingersoll_crater_temperature,
    crater_view_factors
)
from crater_topography_deviation import (
    generate_synthetic_irregular_crater,
    analyze_topographic_deviation,
    temperature_correction_factor,
    print_deviation_report
)


def corrected_ingersoll_temperature(crater: CraterGeometry,
                                    radii: np.ndarray,
                                    depths: np.ndarray,
                                    T_sunlit: float,
                                    solar_elevation_deg: float,
                                    albedo: float = 0.12,
                                    emissivity: float = 0.95) -> dict:
    """
    Calculate Ingersoll crater temperature with topographic corrections.

    Parameters:
    -----------
    crater : CraterGeometry
        Nominal crater geometry
    radii : np.ndarray
        Actual topographic profile radii [m]
    depths : np.ndarray
        Actual topographic profile depths [m]
    T_sunlit : float
        Sunlit terrain temperature [K]
    solar_elevation_deg : float
        Solar elevation [degrees]
    albedo, emissivity : float
        Optical properties

    Returns:
    --------
    dict containing:
        - All standard Ingersoll results
        - Corrected temperatures
        - Deviation analysis
        - Uncertainty estimates
    """
    # Calculate standard Ingersoll temperatures
    ingersoll_result = ingersoll_crater_temperature(
        crater, T_sunlit, solar_elevation_deg, albedo, emissivity
    )

    # Analyze topographic deviations
    deviation = analyze_topographic_deviation(radii, depths, crater.diameter)

    # Calculate correction factors
    corrections = temperature_correction_factor(deviation, crater)

    # Apply corrections
    # Corrected view factors
    view_factors_ideal = crater_view_factors(crater.gamma)
    f_sky_corrected = view_factors_ideal['f_sky'] * corrections['view_factor_correction']
    f_walls_corrected = 1.0 - f_sky_corrected

    # Corrected shadow temperature
    # More sky visibility → colder
    # Less sky visibility → warmer (more wall radiation)
    T_shadow_corrected = ingersoll_result['T_shadow'] + corrections['temperature_offset']

    # Corrected shadow area
    shadow_frac_corrected = (ingersoll_result['shadow_fraction'] *
                            corrections['shadow_area_correction'])

    return {
        # Original Ingersoll results
        'T_shadow_nominal': ingersoll_result['T_shadow'],
        'T_sunlit_floor_nominal': ingersoll_result['T_sunlit_floor'],
        'shadow_fraction_nominal': ingersoll_result['shadow_fraction'],

        # Corrected results
        'T_shadow_corrected': T_shadow_corrected,
        'shadow_fraction_corrected': shadow_frac_corrected,
        'f_sky_corrected': f_sky_corrected,
        'f_walls_corrected': f_walls_corrected,

        # Corrections applied
        'temperature_correction_K': corrections['temperature_offset'],
        'view_factor_correction': corrections['view_factor_correction'],
        'shadow_area_correction': corrections['shadow_area_correction'],

        # Uncertainty
        'temperature_uncertainty_K': corrections['uncertainty'],

        # Deviation analysis
        'deviation': deviation,
        'shape_factor': corrections['shape_factor'],

        # Full original result
        'ingersoll_full': ingersoll_result
    }


def compare_nominal_vs_corrected(crater: CraterGeometry,
                                  irregularity: float = 0.2,
                                  T_sunlit: float = 200.0,
                                  solar_elevation: float = 5.0):
    """
    Compare nominal Ingersoll vs topographically-corrected temperatures.

    Parameters:
    -----------
    crater : CraterGeometry
        Crater geometry
    irregularity : float
        Topographic irregularity factor (0-0.5)
    T_sunlit : float
        Sunlit terrain temperature [K]
    solar_elevation : float
        Solar elevation [degrees]
    """
    print("=" * 70)
    print(f"Comparison: Nominal vs Corrected Ingersoll Temperatures")
    print("=" * 70)

    print(f"\n### Crater Parameters ###")
    print(f"Diameter: {crater.diameter:.0f} m ({crater.diameter/1000:.1f} km)")
    print(f"Depth: {crater.depth:.0f} m")
    print(f"d/D ratio: {crater.gamma:.4f}")
    print(f"Latitude: {crater.latitude_deg:.1f}°")
    print(f"Irregularity factor: {irregularity:.2f}")

    # Generate irregular topography
    radii, depths = generate_synthetic_irregular_crater(
        crater.diameter, crater.depth, irregularity=irregularity
    )

    # Calculate with corrections
    result = corrected_ingersoll_temperature(
        crater, radii, depths, T_sunlit, solar_elevation
    )

    print(f"\n### Temperature Results ###")
    print(f"Solar elevation: {solar_elevation}°")
    print(f"Sunlit terrain: {T_sunlit:.1f} K")
    print()
    print(f"Shadow Temperature:")
    print(f"  Nominal (Ingersoll):  {result['T_shadow_nominal']:.1f} K")
    print(f"  Corrected:            {result['T_shadow_corrected']:.1f} K")
    print(f"  Correction:           {result['temperature_correction_K']:+.1f} K")
    print(f"  Uncertainty:          ±{result['temperature_uncertainty_K']:.1f} K")

    print(f"\nShadow Area Fraction:")
    print(f"  Nominal:              {result['shadow_fraction_nominal']:.2%}")
    print(f"  Corrected:            {result['shadow_fraction_corrected']:.2%}")
    print(f"  Change:               {(result['shadow_fraction_corrected'] - result['shadow_fraction_nominal'])*100:+.1f}%")

    print(f"\nView Factors:")
    print(f"  Sky (nominal):        {result['ingersoll_full']['view_factor_sky']:.3f}")
    print(f"  Sky (corrected):      {result['f_sky_corrected']:.3f}")
    print(f"  Walls (corrected):    {result['f_walls_corrected']:.3f}")

    print(f"\n### Topographic Deviation ###")
    dev = result['deviation']
    print(f"RMS deviation:          {dev.rms_deviation:.1f} m")
    print(f"Shape factor:           {dev.shape_factor:.3f}")
    print(f"Roughness exponent:     {dev.roughness_exponent:.3f}")

    # Assessment
    print(f"\n### Model Assessment ###")
    if dev.shape_factor < 0.15:
        print("✓ Nominal Ingersoll model is excellent")
        print(f"  Temperature error: <{result['temperature_uncertainty_K']:.1f} K")
    elif dev.shape_factor < 0.3:
        print("✓ Corrected Ingersoll model recommended")
        print(f"  Expected improvement: {abs(result['temperature_correction_K']):.1f} K")
    else:
        print("⚠ Detailed 3D topographic modeling recommended")
        print(f"  Ingersoll uncertainty: ±{result['temperature_uncertainty_K']:.1f} K")

    print("=" * 70)

    return result


def sensitivity_study():
    """
    Study how temperature corrections vary with irregularity.
    """
    print("\n\n" + "=" * 70)
    print("Sensitivity Study: Temperature Correction vs Irregularity")
    print("=" * 70)

    crater = CraterGeometry(diameter=5000, depth=400, latitude_deg=-85)

    irregularities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    print(f"\nCrater: D={crater.diameter}m, d={crater.depth}m, γ={crater.gamma:.3f}")
    print(f"\n{'Irreg.':>8} | {'Shape':>8} | {'T_corr (K)':>12} | {'Uncert. (K)':>12} | {'Assessment':>20}")
    print("-" * 75)

    for irreg in irregularities:
        if irreg == 0:
            # Perfect bowl - no correction needed
            print(f"{irreg:8.2f} | {'0.000':>8} | {0.0:>12.1f} | {0.0:>12.1f} | {'Perfect bowl':>20}")
            continue

        radii, depths = generate_synthetic_irregular_crater(
            crater.diameter, crater.depth, irregularity=irreg, random_seed=42
        )

        deviation = analyze_topographic_deviation(radii, depths, crater.diameter)
        corrections = temperature_correction_factor(deviation, crater)

        if corrections['uncertainty'] < 3:
            assessment = "Excellent"
        elif corrections['uncertainty'] < 7:
            assessment = "Good"
        elif corrections['uncertainty'] < 12:
            assessment = "Fair"
        else:
            assessment = "Poor (use 3D model)"

        print(f"{irreg:8.2f} | {deviation.shape_factor:8.3f} | "
              f"{corrections['temperature_offset']:>12.1f} | "
              f"{corrections['uncertainty']:>12.1f} | {assessment:>20}")

    print("\nKey Insights:")
    print("- Shape factor increases with irregularity")
    print("- Temperature corrections become significant above ~0.2 irregularity")
    print("- Uncertainty grows rapidly for highly irregular craters")
    print("- For shape factor > 0.3, detailed 3D modeling recommended")

    print("=" * 70)


def realistic_crater_example():
    """
    Example with realistic lunar crater parameters.
    """
    print("\n\n" + "=" * 70)
    print("Realistic Example: Degraded vs Fresh Crater")
    print("=" * 70)

    # Fresh crater: deeper, more regular (low irregularity)
    fresh_crater = CraterGeometry(diameter=8000, depth=1000, latitude_deg=-86)
    print("\n### Fresh Crater (recent impact) ###")
    result_fresh = compare_nominal_vs_corrected(
        fresh_crater, irregularity=0.08, T_sunlit=180, solar_elevation=3.0
    )

    # Degraded crater: shallower, more irregular
    degraded_crater = CraterGeometry(diameter=8000, depth=600, latitude_deg=-86)
    print("\n\n### Degraded Crater (ancient, eroded) ###")
    result_degraded = compare_nominal_vs_corrected(
        degraded_crater, irregularity=0.35, T_sunlit=180, solar_elevation=3.0
    )

    # Comparison
    print("\n\n" + "=" * 70)
    print("Summary Comparison")
    print("=" * 70)

    print(f"\n{'Parameter':30s} | {'Fresh':>12s} | {'Degraded':>12s}")
    print("-" * 60)
    print(f"{'Depth (m)':30s} | {fresh_crater.depth:>12.0f} | {degraded_crater.depth:>12.0f}")
    print(f"{'Shape factor':30s} | {result_fresh['shape_factor']:>12.3f} | {result_degraded['shape_factor']:>12.3f}")
    print(f"{'Shadow T (nominal, K)':30s} | {result_fresh['T_shadow_nominal']:>12.1f} | {result_degraded['T_shadow_nominal']:>12.1f}")
    print(f"{'Shadow T (corrected, K)':30s} | {result_fresh['T_shadow_corrected']:>12.1f} | {result_degraded['T_shadow_corrected']:>12.1f}")
    print(f"{'Temperature uncertainty (K)':30s} | {result_fresh['temperature_uncertainty_K']:>12.1f} | {result_degraded['temperature_uncertainty_K']:>12.1f}")

    print("\nConclusion:")
    print("- Fresh craters closely match Ingersoll model (low shape factor)")
    print("- Degraded craters require corrections (high shape factor)")
    print("- Temperature uncertainty increases with crater age/degradation")

    print("=" * 70)


if __name__ == "__main__":
    # Example 1: Basic comparison
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Nominal vs Corrected Comparison")
    print("=" * 70)

    crater1 = CraterGeometry(diameter=5000, depth=400, latitude_deg=-85)
    result1 = compare_nominal_vs_corrected(
        crater1, irregularity=0.15, T_sunlit=200, solar_elevation=5.0
    )

    # Example 2: Sensitivity study
    sensitivity_study()

    # Example 3: Realistic crater comparison
    realistic_crater_example()

    print("\n" + "=" * 70)
    print("All Examples Complete")
    print("=" * 70)

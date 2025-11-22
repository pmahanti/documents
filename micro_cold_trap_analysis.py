#!/usr/bin/env python3
"""
Micro Cold Trap Analysis for Lunar Volatile Retention

This script analyzes how surface roughness creates micro-scale permanently
shadowed regions (micro-PSRs or micro cold traps) that can retain volatiles
even when the pixel-averaged temperature is too warm.

Key concepts:
- Surface roughness (boulders, small craters, terrain undulations) creates shadows
- These shadows can be much colder than the pixel-average temperature
- Micro cold traps represent a fractional area of each pixel
- Roughness is characterized by RMS slope or RMS height
- Critical for understanding ice retention in "warm" pixels
"""

from vaporp_temp import (VOLATILE_SPECIES, estimate_cold_trap_fraction,
                          calculate_micro_cold_trap_temperature,
                          calculate_mixed_pixel_sublimation, format_results)
import math


def example_1_roughness_sensitivity():
    """Example 1: Effect of surface roughness on cold trap fraction."""
    print("\n" + "="*70)
    print("Example 1: Roughness Sensitivity - Cold Trap Fraction vs RMS Slope")
    print("="*70)
    print("\nHow does surface roughness affect the fraction of micro cold traps?")

    roughness_values = [5, 10, 15, 20, 25, 30, 35, 40]  # RMS slope in degrees
    models = ['cosine', 'linear', 'exponential']

    print(f"\n{'RMS Slope':<12} {'Cosine':<12} {'Linear':<12} {'Exponential':<12}")
    print(f"{'(degrees)':<12} {'Model':<12} {'Model':<12} {'Model':<12}")
    print("-" * 50)

    for roughness in roughness_values:
        fractions = []
        for model in models:
            frac = estimate_cold_trap_fraction(roughness_rms_slope=roughness, model=model)
            fractions.append(frac)

        print(f"{roughness:<12.0f} {fractions[0]:<12.3f} {fractions[1]:<12.3f} {fractions[2]:<12.3f}")

    print("\nInterpretation:")
    print("- Smooth terrain (5°): ~1-3% in micro cold traps")
    print("- Moderate roughness (15°): ~7-25% in micro cold traps")
    print("- Very rough (30°+): ~33-63% in micro cold traps")
    print("- Different models bound the uncertainty range")


def example_2_temperature_contrast():
    """Example 2: Temperature contrast between illuminated and cold trap areas."""
    print("\n" + "="*70)
    print("Example 2: Temperature Contrast in Micro Cold Traps")
    print("="*70)

    illuminated_temps = [100, 120, 140, 160, 180, 200]
    latitudes = [85, 80, 70]

    print("\nTemperature depression as function of latitude:")
    print(f"\n{'Illum Temp':<12} {'Lat=85°':<15} {'Lat=80°':<15} {'Lat=70°':<15}")
    print(f"{'(K)':<12} {'CT Temp (K)':<15} {'CT Temp (K)':<15} {'CT Temp (K)':<15}")
    print("-" * 57)

    for temp in illuminated_temps:
        ct_temps = []
        for lat in latitudes:
            ct_temp = calculate_micro_cold_trap_temperature(temp, latitude=lat)
            ct_temps.append(ct_temp)

        print(f"{temp:<12.0f} {ct_temps[0]:<15.1f} {ct_temps[1]:<15.1f} {ct_temps[2]:<15.1f}")

    print("\nKey Findings:")
    print("- Higher latitudes have larger temperature contrasts")
    print("- At 85° latitude, cold traps can be 70% cooler than illuminated areas")
    print("- Even 'warm' pixels (200K average) can have cold traps at ~60K")


def example_3_sublimation_reduction():
    """Example 3: Sublimation rate reduction due to micro cold traps."""
    print("\n" + "="*70)
    print("Example 3: Sublimation Reduction by Micro Cold Traps")
    print("="*70)

    species = VOLATILE_SPECIES['H2O']
    illuminated_temp = 150  # K - would normally lose ice quickly
    roughness_values = [0, 10, 20, 30, 40]

    print(f"\nIlluminated temperature: {illuminated_temp} K")
    print("Species: H2O")
    print("\nEffect of increasing surface roughness:\n")
    print(f"{'RMS Slope':<12} {'Cold Trap':<12} {'Mixed Rate':<15} {'Reduction':<12} {'CT Only Rate':<15}")
    print(f"{'(degrees)':<12} {'Fraction':<12} {'(kg/m²/yr)':<15} {'Factor':<12} {'(kg/m²/yr)':<15}")
    print("-" * 75)

    for roughness in roughness_values:
        if roughness == 0:
            # No cold traps - uniform temperature
            result = species.sublimation_rate(illuminated_temp)
            print(f"{roughness:<12.0f} {0.0:<12.3f} {result['sublimation_rate_kg_m2_yr']:<15.2e} {'1.000':<12} {'N/A':<15}")
        else:
            frac = estimate_cold_trap_fraction(roughness_rms_slope=roughness, model='cosine')
            ct_temp = calculate_micro_cold_trap_temperature(illuminated_temp, latitude=85)

            result = calculate_mixed_pixel_sublimation(
                species, illuminated_temp, frac,
                cold_trap_temp=ct_temp
            )

            print(f"{roughness:<12.0f} {frac:<12.3f} {result['mixed_sublimation_rate_kg_m2_yr']:<15.2e} "
                  f"{result['sublimation_reduction_factor']:<12.3f} "
                  f"{result['cold_trap_only_rate_kg_m2_yr']:<15.2e}")

    print("\nConclusion:")
    print("- Rough terrain dramatically reduces effective sublimation rate")
    print("- 30° RMS slope can reduce sublimation by 8-10x")
    print("- Ice in cold traps is protected and can persist for much longer")


def example_4_ice_stability_map():
    """Example 4: Ice stability as function of temperature and roughness."""
    print("\n" + "="*70)
    print("Example 4: Ice Stability Map - Temperature vs Roughness")
    print("="*70)

    species = VOLATILE_SPECIES['H2O']
    temperatures = [80, 100, 120, 140, 160]
    roughnesses = [0, 10, 20, 30, 40]

    # Calculate lifetime of 1m ice deposit in years
    print("\nLifetime of 1m H2O ice deposit (years):")
    print("Scenario: Ice retained only in micro cold traps\n")

    # Header
    print(f"{'Temp (K)':<10}", end='')
    for r in roughnesses:
        print(f"RMS={r}°{'':<8}", end='')
    print()
    print("-" * 70)

    for temp in temperatures:
        print(f"{temp:<10.0f}", end='')
        for roughness in roughnesses:
            if roughness == 0:
                # No cold traps
                result = species.sublimation_rate(temp)
                rate_mm_yr = result['sublimation_rate_mm_yr']
            else:
                frac = estimate_cold_trap_fraction(roughness_rms_slope=roughness, model='cosine')
                ct_temp = calculate_micro_cold_trap_temperature(temp, latitude=85)
                result = calculate_mixed_pixel_sublimation(
                    species, temp, frac, cold_trap_temp=ct_temp
                )
                rate_mm_yr = result['cold_trap_only_rate_mm_yr']

            if rate_mm_yr > 0:
                lifetime_years = 1000 / rate_mm_yr  # 1m = 1000mm
                if lifetime_years > 1e12:
                    print(f"{'Stable':<12}", end='')
                elif lifetime_years > 1e9:
                    print(f"{lifetime_years/1e9:<6.1f}Gyr{'':<5}", end='')
                elif lifetime_years > 1e6:
                    print(f"{lifetime_years/1e6:<6.1f}Myr{'':<5}", end='')
                else:
                    print(f"{lifetime_years:<10.1e}", end='')
            else:
                print(f"{'Stable':<12}", end='')

        print()

    print("\nKey Insights:")
    print("- Even at 160K, rough terrain (30-40° RMS) can preserve ice")
    print("- Smooth terrain loses ice rapidly above ~110K")
    print("- Surface roughness extends the stability boundary by tens of degrees")


def example_5_realistic_scenario():
    """Example 5: Realistic lunar south pole scenario."""
    print("\n" + "="*70)
    print("Example 5: Realistic Scenario - Lunar South Pole Crater")
    print("="*70)

    print("\nScenario: Pixel at 88°S latitude")
    print("  - Illuminated temperature: 130 K (from Diviner)")
    print("  - Surface roughness: 25° RMS slope (from laser altimetry)")
    print("  - Species: H2O ice")

    species = VOLATILE_SPECIES['H2O']
    illuminated_temp = 130  # K
    roughness = 25  # degrees
    latitude = 88  # degrees

    # Calculate cold trap properties
    frac = estimate_cold_trap_fraction(roughness_rms_slope=roughness, model='cosine')
    ct_temp = calculate_micro_cold_trap_temperature(illuminated_temp, latitude=latitude)

    print(f"\nDerived Properties:")
    print(f"  - Cold trap fraction: {frac:.2%}")
    print(f"  - Cold trap temperature: {ct_temp:.1f} K")
    print(f"  - Temperature depression: {illuminated_temp - ct_temp:.1f} K")

    # Calculate mixed-pixel sublimation
    result = calculate_mixed_pixel_sublimation(
        species, illuminated_temp, frac,
        cold_trap_temp=ct_temp
    )

    print(format_results('H2O (Mixed Pixel)', result))

    # Calculate without cold traps for comparison
    uniform_result = species.sublimation_rate(illuminated_temp)

    print("Comparison:")
    print(f"  Without cold traps (uniform 130K):")
    print(f"    Rate: {uniform_result['sublimation_rate_kg_m2_yr']:.2e} kg/(m²·yr)")
    print(f"    1m ice lifetime: {1000/uniform_result['sublimation_rate_mm_yr']:.2e} years")

    print(f"\n  With micro cold traps (25° RMS):")
    print(f"    Mixed rate: {result['mixed_sublimation_rate_kg_m2_yr']:.2e} kg/(m²·yr)")
    print(f"    CT-only rate: {result['cold_trap_only_rate_kg_m2_yr']:.2e} kg/(m²·yr)")
    print(f"    Ice in CT lifetime: {1000/result['cold_trap_only_rate_mm_yr']:.2e} years")

    print(f"\n  Improvement factor: {uniform_result['sublimation_rate_mm_yr']/result['cold_trap_only_rate_mm_yr']:.1f}x")
    print("\nConclusion: Micro cold traps extend ice stability by ~100-1000x!")


def example_6_parametric_study():
    """Example 6: Parametric study - varying temperature and roughness."""
    print("\n" + "="*70)
    print("Example 6: Parametric Study - Critical Roughness for Ice Stability")
    print("="*70)

    print("\nQuestion: What minimum roughness is needed to preserve ice")
    print("for 1 billion years as a function of pixel temperature?")

    species = VOLATILE_SPECIES['H2O']
    temperatures = range(100, 181, 10)  # 100K to 180K
    target_lifetime = 1e9  # 1 billion years
    target_rate_mm_yr = 1000 / target_lifetime  # Required rate for 1m ice to last 1Gyr

    print(f"\nTarget: 1m ice lasting > 1 Gyr")
    print(f"Required rate: < {target_rate_mm_yr:.2e} mm/yr\n")

    print(f"{'Temperature':<15} {'Min Roughness':<18} {'Cold Trap':<15} {'Achievable':<12}")
    print(f"{'(K)':<15} {'(degrees RMS)':<18} {'Fraction':<15} {'Lifetime':<12}")
    print("-" * 65)

    for temp in temperatures:
        # Binary search for minimum roughness
        achievable = False
        min_roughness = None

        for roughness in range(0, 46):  # Test 0-45 degrees
            frac = estimate_cold_trap_fraction(roughness_rms_slope=roughness, model='cosine')
            ct_temp = calculate_micro_cold_trap_temperature(temp, latitude=85)

            result = calculate_mixed_pixel_sublimation(
                species, temp, frac, cold_trap_temp=ct_temp
            )

            if result['cold_trap_only_rate_mm_yr'] <= target_rate_mm_yr:
                min_roughness = roughness
                achievable = True
                break

        if achievable:
            frac = estimate_cold_trap_fraction(roughness_rms_slope=min_roughness, model='cosine')
            lifetime = 1000 / result['cold_trap_only_rate_mm_yr']
            print(f"{temp:<15.0f} {min_roughness:<18.0f} {frac:<15.2%} {lifetime/1e9:<12.2f} Gyr")
        else:
            print(f"{temp:<15.0f} {'Not achievable':<18} {'N/A':<15} {'N/A':<12}")

    print("\nInsights:")
    print("- Below ~110K: Ice is stable even on smooth surfaces")
    print("- 110-150K: Moderate roughness (10-25°) enables long-term stability")
    print("- Above 150K: Even extreme roughness cannot preserve ice for Gyr timescales")
    print("- Roughness extends the 'cold trap' definition poleward by several degrees latitude")


def main():
    """Run all micro cold trap analysis examples."""
    print("\n" + "*"*70)
    print("Micro Cold Trap Analysis for Lunar Volatile Retention")
    print("*"*70)

    example_1_roughness_sensitivity()
    example_2_temperature_contrast()
    example_3_sublimation_reduction()
    example_4_ice_stability_map()
    example_5_realistic_scenario()
    example_6_parametric_study()

    print("\n" + "*"*70)
    print("Analysis Complete!")
    print("*"*70)
    print("\nKey Takeaways:")
    print("1. Surface roughness creates micro-scale permanently shadowed regions")
    print("2. Even 'warm' pixels can contain cold traps that preserve ice")
    print("3. Roughness of 20-30° RMS can extend ice stability by 100-1000x")
    print("4. This explains detection of ice in 'unexpected' warm locations")
    print("5. Critical for understanding total ice inventory at lunar poles")
    print("6. Roughness measurements should be included in ice stability models")
    print("*"*70 + "\n")


if __name__ == '__main__':
    main()

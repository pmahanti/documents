#!/usr/bin/env python3
"""
Example usage of vaporp_temp module for programmatic access.

This script demonstrates how to use the volatile sublimation calculator
as a Python module rather than command-line tool.
"""

import sys
from vaporp_temp import VOLATILE_SPECIES, format_results


def example_1_single_calculation():
    """Example 1: Single temperature and species calculation."""
    print("\n" + "="*70)
    print("Example 1: Calculate H2O sublimation at 110K")
    print("="*70)

    species = VOLATILE_SPECIES['H2O']
    temperature = 110  # Kelvin

    results = species.sublimation_rate(temperature)
    print(format_results('H2O', results))


def example_2_temperature_range():
    """Example 2: Calculate sublimation over a temperature range."""
    print("\n" + "="*70)
    print("Example 2: H2O sublimation across PSR temperature range")
    print("="*70)

    species = VOLATILE_SPECIES['H2O']
    temperatures = [40, 60, 80, 100, 120]

    print(f"\n{'Temp (K)':<12} {'Vapor P (Pa)':<15} {'Rate (kg/m²/yr)':<20} {'Depth (mm/yr)':<15}")
    print("-" * 70)

    for temp in temperatures:
        results = species.sublimation_rate(temp)
        print(f"{temp:<12.1f} {results['vapor_pressure_Pa']:<15.2e} "
              f"{results['sublimation_rate_kg_m2_yr']:<20.2e} "
              f"{results['sublimation_rate_mm_yr']:<15.2e}")


def example_3_species_comparison():
    """Example 3: Compare different volatile species at same temperature."""
    print("\n" + "="*70)
    print("Example 3: Compare all species at 100K")
    print("="*70)

    temperature = 100  # Kelvin

    print(f"\n{'Species':<10} {'Vapor P (Pa)':<15} {'Rate (kg/m²/yr)':<20} {'Depth (mm/yr)':<15}")
    print("-" * 70)

    for name, species in VOLATILE_SPECIES.items():
        try:
            results = species.sublimation_rate(temperature)
            print(f"{name:<10} {results['vapor_pressure_Pa']:<15.2e} "
                  f"{results['sublimation_rate_kg_m2_yr']:<20.2e} "
                  f"{results['sublimation_rate_mm_yr']:<15.2e}")
        except:
            print(f"{name:<10} {'N/A':<15} {'N/A':<20} {'N/A':<15}")


def example_4_stability_analysis():
    """Example 4: Ice stability analysis - find lifetime at different temps."""
    print("\n" + "="*70)
    print("Example 4: Water ice stability (time to sublimate 1m thickness)")
    print("="*70)

    species = VOLATILE_SPECIES['H2O']
    ice_thickness_m = 1.0  # 1 meter of ice
    ice_density = 920  # kg/m³

    temperatures = [40, 60, 80, 100, 110, 120]

    print(f"\n{'Temp (K)':<12} {'Rate (mm/yr)':<15} {'Lifetime (years)':<20}")
    print("-" * 60)

    for temp in temperatures:
        results = species.sublimation_rate(temp)
        rate_m_per_yr = results['sublimation_rate_mm_yr'] / 1000

        if rate_m_per_yr > 0:
            lifetime_years = ice_thickness_m / rate_m_per_yr
        else:
            lifetime_years = float('inf')

        print(f"{temp:<12.1f} {results['sublimation_rate_mm_yr']:<15.2e} "
              f"{lifetime_years:<20.2e}")


def example_5_custom_alpha():
    """Example 5: Effect of sticking coefficient on sublimation rate."""
    print("\n" + "="*70)
    print("Example 5: Effect of sticking coefficient (α) on H2O at 110K")
    print("="*70)

    species = VOLATILE_SPECIES['H2O']
    temperature = 110
    alphas = [0.1, 0.5, 1.0]

    print(f"\n{'Alpha':<12} {'Rate (kg/m²/yr)':<20} {'Depth (mm/yr)':<15}")
    print("-" * 50)

    for alpha in alphas:
        results = species.sublimation_rate(temperature, alpha=alpha)
        print(f"{alpha:<12.1f} {results['sublimation_rate_kg_m2_yr']:<20.2e} "
              f"{results['sublimation_rate_mm_yr']:<15.2e}")


def example_6_save_results():
    """Example 6: Generate data file for plotting or analysis."""
    print("\n" + "="*70)
    print("Example 6: Generate CSV data file")
    print("="*70)

    output_file = 'sublimation_data.csv'

    with open(output_file, 'w') as f:
        # Write header
        f.write("Species,Temperature_K,Vapor_Pressure_Pa,Sublimation_Rate_kg_m2_yr,Depth_Loss_mm_yr\n")

        # Generate data
        temperatures = range(40, 121, 10)  # 40K to 120K in 10K steps

        for temp in temperatures:
            for name, species in VOLATILE_SPECIES.items():
                try:
                    results = species.sublimation_rate(temp)
                    f.write(f"{name},{temp},"
                           f"{results['vapor_pressure_Pa']:.6e},"
                           f"{results['sublimation_rate_kg_m2_yr']:.6e},"
                           f"{results['sublimation_rate_mm_yr']:.6e}\n")
                except:
                    pass

    print(f"\nData saved to {output_file}")
    print("You can now plot this data with matplotlib, Excel, etc.")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 70)
    print("Lunar South Pole Volatile Sublimation Calculator - Examples")
    print("*" * 70)

    example_1_single_calculation()
    example_2_temperature_range()
    example_3_species_comparison()
    example_4_stability_analysis()
    example_5_custom_alpha()
    example_6_save_results()

    print("\n" + "*" * 70)
    print("All examples completed!")
    print("*" * 70 + "\n")


if __name__ == '__main__':
    main()

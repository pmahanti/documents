#!/usr/bin/env python3
"""
Example: Calculate time-averaged sublimation using subsolar position data.

This example demonstrates how to compute time-averaged sublimation rates
considering temperature variations due to solar illumination geometry.
"""

from vaporp_temp import VOLATILE_SPECIES, calculate_time_averaged_sublimation, format_results
import math


def calculate_illumination_temperature(subsolar_lat, subsolar_lon,
                                        location_lat, location_lon,
                                        max_temp=390, min_temp=40):
    """
    Estimate surface temperature based on solar illumination angle.

    This is a simplified model. For actual lunar temperatures, use thermal
    models or Diviner data.

    Parameters:
    -----------
    subsolar_lat, subsolar_lon : float
        Subsolar point coordinates (degrees)
    location_lat, location_lon : float
        Surface location coordinates (degrees)
    max_temp : float
        Maximum temperature when Sun is at zenith (K)
    min_temp : float
        Minimum temperature in shadow (K)

    Returns:
    --------
    float
        Estimated temperature in Kelvin
    """
    # Convert to radians
    ssl_rad = math.radians(subsolar_lat)
    sslon_rad = math.radians(subsolar_lon)
    loc_lat_rad = math.radians(location_lat)
    loc_lon_rad = math.radians(location_lon)

    # Calculate solar incidence angle using spherical trigonometry
    # cos(incidence) = sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon2-lon1)
    cos_incidence = (math.sin(ssl_rad) * math.sin(loc_lat_rad) +
                     math.cos(ssl_rad) * math.cos(loc_lat_rad) *
                     math.cos(sslon_rad - loc_lon_rad))

    # Clamp to valid range
    cos_incidence = max(-1.0, min(1.0, cos_incidence))

    # If in shadow (incidence angle > 90°), use minimum temperature
    if cos_incidence <= 0:
        return min_temp

    # Simple temperature model: T^4 ∝ cos(incidence_angle)
    # (Stefan-Boltzmann, equilibrium temperature)
    temperature = min_temp + (max_temp - min_temp) * (cos_incidence ** 0.25)

    return temperature


def example_1_diurnal_cycle():
    """Example 1: Sublimation over a simplified lunar diurnal cycle."""
    print("\n" + "="*70)
    print("Example 1: Sublimation over a simplified lunar diurnal cycle")
    print("="*70)
    print("\nSimulating temperature variation over one lunar day/night cycle")
    print("at a location near the south pole (-85° latitude)")

    # Location near lunar south pole
    location_lat = -85.0  # degrees
    location_lon = 0.0    # degrees

    # Simulate subsolar positions over ~1 lunar day (29.5 Earth days)
    # Using simplified model: subsolar longitude changes 360°/29.5days
    temperatures = []
    num_samples = 24  # 24 time steps over the lunar day

    print(f"\nTime step | Subsolar Lon | Temperature")
    print("-" * 45)

    for i in range(num_samples):
        subsolar_lon = (i / num_samples) * 360.0  # degrees
        subsolar_lat = 0.0  # Equator for simplicity

        temp = calculate_illumination_temperature(
            subsolar_lat, subsolar_lon,
            location_lat, location_lon,
            max_temp=250,  # Lower max temp at high latitude
            min_temp=40    # PSR shadow temperature
        )
        temperatures.append(temp)

        if i % 4 == 0:  # Print every 4th sample
            print(f"{i:9d} | {subsolar_lon:12.1f} | {temp:11.2f} K")

    # Calculate time-averaged sublimation for H2O
    species = VOLATILE_SPECIES['H2O']
    results = calculate_time_averaged_sublimation(species, temperatures)

    print(format_results('H2O', results))

    print("Interpretation:")
    print(f"  - Temperature varies from {min(temperatures):.1f}K to {max(temperatures):.1f}K")
    print(f"  - Time-averaged sublimation: {results['time_averaged_rate_mm_yr']:.2e} mm/yr")
    print(f"  - This represents average ice loss over complete lunar day/night cycle")


def example_2_psr_analysis():
    """Example 2: Permanently Shadowed Region (PSR) analysis."""
    print("\n" + "="*70)
    print("Example 2: PSR with small temperature fluctuations")
    print("="*70)
    print("\nPSRs have small temperature variations due to thermal emission")
    print("from surrounding terrain and seasonal solar position changes.")

    # PSR temperatures with small fluctuations (based on Diviner data)
    psr_base_temp = 40.0  # Very cold PSR
    temperature_variation = 15.0  # Small seasonal variation

    num_samples = 50
    temperatures = []

    print(f"\nBase temperature: {psr_base_temp} K")
    print(f"Variation range: ±{temperature_variation} K")
    print("\nGenerating {num_samples} temperature samples...")

    # Simulate small periodic variations
    for i in range(num_samples):
        # Simple sinusoidal variation
        phase = 2 * math.pi * i / num_samples
        temp = psr_base_temp + temperature_variation * math.sin(phase)
        temperatures.append(temp)

    print(f"Temperature range: {min(temperatures):.2f} - {max(temperatures):.2f} K")

    # Calculate for multiple species
    print("\nTime-averaged sublimation rates for different species:")
    print("=" * 70)

    for species_name in ['H2O', 'CO2', 'CO']:
        species = VOLATILE_SPECIES[species_name]
        results = calculate_time_averaged_sublimation(species, temperatures)

        print(f"\n{species_name}:")
        print(f"  Rate: {results['time_averaged_rate_kg_m2_yr']:.2e} kg/(m²·yr)")
        print(f"  Depth loss: {results['time_averaged_rate_mm_yr']:.2e} mm/yr")

        # Calculate lifetime of 1m ice deposit
        if results['time_averaged_rate_mm_yr'] > 0:
            lifetime_years = 1000 / results['time_averaged_rate_mm_yr']  # 1m = 1000mm
            print(f"  Lifetime (1m deposit): {lifetime_years:.2e} years")


def example_3_weighted_average():
    """Example 3: Weighted time average with different durations."""
    print("\n" + "="*70)
    print("Example 3: Weighted average - different time durations")
    print("="*70)
    print("\nExample: Location that spends different amounts of time at")
    print("various temperatures during a lunar month.")

    # Different temperatures and time spent at each (in hours)
    temperatures = [40, 60, 80, 100, 120, 140]  # Kelvin
    durations = [400, 200, 100, 50, 20, 10]     # Hours at each temperature

    total_hours = sum(durations)
    print(f"\nTemperature distribution over {total_hours} hours:")
    print(f"{'Temp (K)':<12} {'Duration (hrs)':<15} {'Fraction':<12}")
    print("-" * 40)
    for temp, dur in zip(temperatures, durations):
        fraction = dur / total_hours
        print(f"{temp:<12.0f} {dur:<15.0f} {fraction:<12.3f}")

    # Calculate weighted average
    species = VOLATILE_SPECIES['H2O']
    results = calculate_time_averaged_sublimation(
        species,
        temperatures,
        weights=durations
    )

    print(format_results('H2O', results))

    # Compare with unweighted average
    results_unweighted = calculate_time_averaged_sublimation(species, temperatures)

    print("Comparison:")
    print(f"  Weighted avg rate:   {results['time_averaged_rate_kg_m2_yr']:.2e} kg/(m²·yr)")
    print(f"  Unweighted avg rate: {results_unweighted['time_averaged_rate_kg_m2_yr']:.2e} kg/(m²·yr)")
    print(f"  Ratio: {results['time_averaged_rate_kg_m2_yr'] / results_unweighted['time_averaged_rate_kg_m2_yr']:.2f}")
    print("\n  Note: Weighted average accounts for more time spent at lower temperatures")


def main():
    """Run all examples."""
    print("\n" + "*"*70)
    print("Time-Averaged Sublimation Examples")
    print("*"*70)

    example_1_diurnal_cycle()
    example_2_psr_analysis()
    example_3_weighted_average()

    print("\n" + "*"*70)
    print("Examples complete!")
    print("*"*70)
    print("\nKey Takeaways:")
    print("1. Time-averaging is crucial for accurate volatile loss estimates")
    print("2. Even small temperature variations significantly affect sublimation")
    print("3. Weighting by time duration gives more accurate long-term rates")
    print("4. PSRs with T < 110K can preserve water ice over billions of years")
    print("*"*70 + "\n")


if __name__ == '__main__':
    main()

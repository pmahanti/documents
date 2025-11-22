#!/usr/bin/env python3
"""
Lunar South Pole Volatile Sublimation Rate Calculator

This application converts temperature values to sublimation rates for different
volatile species at the lunar south pole using the Hertz-Knudsen equation.
"""

import math
import argparse
import sys


class VolatileSpecies:
    """Class representing a volatile species with its thermodynamic properties."""

    def __init__(self, name, molecular_mass, A, B, C=0, T_triple=None):
        """
        Initialize a volatile species.

        Parameters:
        -----------
        name : str
            Name of the species
        molecular_mass : float
            Molecular mass in kg/mol
        A, B, C : float
            Antoine equation coefficients for log10(P) = A - B/(T+C)
            where P is in Pa and T is in K
        T_triple : float
            Triple point temperature in K (optional, for validation)
        """
        self.name = name
        self.molecular_mass = molecular_mass  # kg/mol
        self.A = A
        self.B = B
        self.C = C
        self.T_triple = T_triple

    def vapor_pressure(self, temperature):
        """
        Calculate vapor pressure using Antoine equation.

        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin

        Returns:
        --------
        float
            Vapor pressure in Pascals
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive (K)")

        # Antoine equation: log10(P) = A - B/(T+C)
        log_p = self.A - self.B / (temperature + self.C)
        return 10 ** log_p

    def sublimation_rate(self, temperature, alpha=1.0):
        """
        Calculate sublimation rate using Hertz-Knudsen equation.

        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin
        alpha : float
            Sticking coefficient (default 1.0 for free sublimation)

        Returns:
        --------
        dict
            Dictionary containing sublimation rates in different units
        """
        R = 8.314  # Gas constant in J/(mol·K)

        # Get vapor pressure
        P_vap = self.vapor_pressure(temperature)

        # Hertz-Knudsen equation
        # J = α * P_vap * sqrt(M / (2 * π * R * T))
        flux_molecules = alpha * P_vap * math.sqrt(
            self.molecular_mass / (2 * math.pi * R * temperature)
        )  # kg/(m²·s)

        # Convert to different useful units
        results = {
            'temperature_K': temperature,
            'temperature_C': temperature - 273.15,
            'vapor_pressure_Pa': P_vap,
            'sublimation_rate_kg_m2_s': flux_molecules,
            'sublimation_rate_g_m2_s': flux_molecules * 1000,
            'sublimation_rate_g_m2_hr': flux_molecules * 1000 * 3600,
            'sublimation_rate_kg_m2_yr': flux_molecules * 365.25 * 24 * 3600,
            'sublimation_rate_mm_yr': self._to_mm_per_year(flux_molecules)
        }

        return results

    def _to_mm_per_year(self, flux_kg_m2_s):
        """
        Convert mass flux to depth loss in mm/year.
        Assumes ice density of ~920 kg/m³ for H2O, adjusted for other species.
        """
        # Rough density estimates
        density_map = {
            'H2O': 920,
            'CO2': 1560,
            'CO': 790,
            'CH4': 420,
            'NH3': 820,
            'SO2': 1460
        }
        density = density_map.get(self.name, 1000)  # kg/m³

        # flux (kg/m²/s) / density (kg/m³) = m/s
        # Convert to mm/year
        m_per_s = flux_kg_m2_s / density
        mm_per_year = m_per_s * 1000 * 365.25 * 24 * 3600

        return mm_per_year


# Database of volatile species relevant to the lunar south pole
VOLATILE_SPECIES = {
    'H2O': VolatileSpecies(
        name='H2O',
        molecular_mass=0.018015,  # kg/mol
        A=10.196,  # Antoine coefficients for ice
        B=2332.5,
        C=-36.0,
        T_triple=273.16
    ),
    'CO2': VolatileSpecies(
        name='CO2',
        molecular_mass=0.04401,
        A=9.816,  # Coefficients for solid CO2
        B=1353.0,
        C=-17.0,
        T_triple=216.58
    ),
    'CO': VolatileSpecies(
        name='CO',
        molecular_mass=0.02801,
        A=8.205,
        B=530.2,
        C=-5.0,
        T_triple=68.15
    ),
    'CH4': VolatileSpecies(
        name='CH4',
        molecular_mass=0.01604,
        A=8.721,
        B=628.4,
        C=-8.0,
        T_triple=90.69
    ),
    'NH3': VolatileSpecies(
        name='NH3',
        molecular_mass=0.01703,
        A=9.950,
        B=1630.0,
        C=-21.0,
        T_triple=195.4
    ),
    'SO2': VolatileSpecies(
        name='SO2',
        molecular_mass=0.06407,
        A=9.814,
        B=1871.0,
        C=-25.0,
        T_triple=197.64
    )
}


def format_results(species_name, results):
    """Format sublimation rate results for display."""
    output = []
    output.append(f"\n{'='*70}")
    output.append(f"Species: {species_name}")
    output.append(f"{'='*70}")
    output.append(f"Temperature:           {results['temperature_K']:.2f} K ({results['temperature_C']:.2f} °C)")
    output.append(f"Vapor Pressure:        {results['vapor_pressure_Pa']:.2e} Pa")
    output.append(f"\nSublimation Rates:")
    output.append(f"  {results['sublimation_rate_kg_m2_s']:.2e} kg/(m²·s)")
    output.append(f"  {results['sublimation_rate_g_m2_s']:.2e} g/(m²·s)")
    output.append(f"  {results['sublimation_rate_g_m2_hr']:.2e} g/(m²·hr)")
    output.append(f"  {results['sublimation_rate_kg_m2_yr']:.2e} kg/(m²·yr)")
    output.append(f"  {results['sublimation_rate_mm_yr']:.2e} mm/yr (depth loss)")
    output.append(f"{'='*70}\n")

    return '\n'.join(output)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Calculate sublimation rates for lunar south pole volatiles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available species: {', '.join(VOLATILE_SPECIES.keys())}

Examples:
  # Calculate H2O sublimation at 110K
  python vaporp_temp.py -t 110 -s H2O

  # Calculate for multiple species
  python vaporp_temp.py -t 100 -s H2O CO2 CO

  # Temperature range for H2O
  python vaporp_temp.py -t 80 90 100 110 -s H2O

  # All species at 100K
  python vaporp_temp.py -t 100 --all
"""
    )

    parser.add_argument('-t', '--temperature',
                        type=float,
                        nargs='+',
                        required=True,
                        help='Temperature value(s) in Kelvin')

    parser.add_argument('-s', '--species',
                        nargs='+',
                        choices=list(VOLATILE_SPECIES.keys()),
                        help='Volatile species to calculate')

    parser.add_argument('--all',
                        action='store_true',
                        help='Calculate for all available species')

    parser.add_argument('-a', '--alpha',
                        type=float,
                        default=1.0,
                        help='Sticking coefficient (default: 1.0)')

    parser.add_argument('-o', '--output',
                        type=str,
                        help='Output file (optional, prints to stdout if not specified)')

    args = parser.parse_args()

    # Determine which species to process
    if args.all:
        species_list = list(VOLATILE_SPECIES.keys())
    elif args.species:
        species_list = args.species
    else:
        parser.error("Either --species or --all must be specified")

    # Process calculations
    results_text = []
    results_text.append(f"\nLunar South Pole Volatile Sublimation Calculator")
    results_text.append(f"Sticking coefficient (α): {args.alpha}")

    for temp in args.temperature:
        for species_name in species_list:
            try:
                species = VOLATILE_SPECIES[species_name]
                results = species.sublimation_rate(temp, alpha=args.alpha)
                results_text.append(format_results(species_name, results))
            except ValueError as e:
                results_text.append(f"\nError calculating {species_name} at {temp} K: {e}\n")
            except Exception as e:
                results_text.append(f"\nUnexpected error for {species_name} at {temp} K: {e}\n")

    # Output results
    output_str = '\n'.join(results_text)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_str)
        print(f"Results written to {args.output}")
    else:
        print(output_str)


if __name__ == '__main__':
    main()

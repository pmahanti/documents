#!/usr/bin/env python3
"""
Calculate time-averaged sublimation rates from temperature time series.

This script demonstrates time-averaged sublimation rate calculations, useful
for understanding volatile loss over lunar day/night cycles or seasonal variations.
"""

import argparse
import sys
from vaporp_temp import VOLATILE_SPECIES, calculate_time_averaged_sublimation, format_results


def parse_temperature_file(filename):
    """
    Parse a simple text file with temperature values (one per line).
    Lines starting with # are ignored as comments.
    """
    temperatures = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    temp = float(line)
                    temperatures.append(temp)
                except ValueError:
                    pass  # Skip invalid lines
    return temperatures


def main():
    parser = argparse.ArgumentParser(
        description='Calculate time-averaged sublimation rates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available species: {', '.join(VOLATILE_SPECIES.keys())}

Examples:
  # Simple example with temperature range
  python time_averaged_sublimation.py --temps 40 50 60 70 80 90 100 -s H2O

  # From temperature file
  python time_averaged_sublimation.py --file temperatures.txt -s H2O

  # With custom weights (e.g., hours at each temperature)
  python time_averaged_sublimation.py --temps 100 120 140 --weights 12 8 4 -s H2O

  # Multiple species
  python time_averaged_sublimation.py --temps 80 90 100 110 120 --all

Temperature file format (one temperature per line in Kelvin):
  # Example temperature file
  40.0
  60.0
  80.0
  100.0
"""
    )

    parser.add_argument('--temps',
                        type=float,
                        nargs='+',
                        help='Temperature values in Kelvin')

    parser.add_argument('--file',
                        type=str,
                        help='File with temperature values (one per line)')

    parser.add_argument('--weights',
                        type=float,
                        nargs='+',
                        help='Weights for each temperature (e.g., time duration)')

    parser.add_argument('-s', '--species',
                        nargs='+',
                        choices=list(VOLATILE_SPECIES.keys()),
                        help='Volatile species to calculate')

    parser.add_argument('--all',
                        action='store_true',
                        help='Calculate for all available species')

    parser.add_argument('--alpha',
                        type=float,
                        default=1.0,
                        help='Sticking coefficient (default: 1.0)')

    parser.add_argument('-o', '--output',
                        type=str,
                        help='Output file (optional)')

    args = parser.parse_args()

    # Get temperature array
    if args.file:
        try:
            temperatures = parse_temperature_file(args.file)
            print(f"Loaded {len(temperatures)} temperature values from {args.file}")
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    elif args.temps:
        temperatures = args.temps
    else:
        parser.error("Either --temps or --file must be specified")

    if len(temperatures) == 0:
        print("Error: No valid temperature values provided")
        sys.exit(1)

    # Get weights
    weights = args.weights if args.weights else None

    if weights and len(weights) != len(temperatures):
        print(f"Error: Number of weights ({len(weights)}) must match number of temperatures ({len(temperatures)})")
        sys.exit(1)

    # Determine species
    if args.all:
        species_list = list(VOLATILE_SPECIES.keys())
    elif args.species:
        species_list = args.species
    else:
        parser.error("Either --species or --all must be specified")

    # Calculate time-averaged sublimation
    results_text = []
    results_text.append(f"\nTime-Averaged Sublimation Rate Calculator")
    results_text.append(f"Sticking coefficient (α): {args.alpha}")
    results_text.append(f"Number of temperature samples: {len(temperatures)}")

    if weights:
        results_text.append(f"Using weighted average (total weight: {sum(weights)})")
    else:
        results_text.append(f"Using equal weights")

    for species_name in species_list:
        try:
            species = VOLATILE_SPECIES[species_name]
            results = calculate_time_averaged_sublimation(
                species,
                temperatures,
                weights=weights,
                alpha=args.alpha
            )
            results_text.append(format_results(species_name, results))
        except Exception as e:
            results_text.append(f"\nError calculating {species_name}: {e}\n")

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

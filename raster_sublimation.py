#!/usr/bin/env python3
"""
Convert temperature raster to sublimation rate raster for lunar volatiles.

This script takes a GeoTIFF temperature raster and produces a sublimation rate
raster using the Hertz-Knudsen equation.
"""

import argparse
import sys
from vaporp_temp import process_temperature_raster, VOLATILE_SPECIES


def main():
    parser = argparse.ArgumentParser(
        description='Convert temperature raster to sublimation rate raster',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available species: {', '.join(VOLATILE_SPECIES.keys())}

Examples:
  # Convert temperature raster to H2O sublimation rate
  python raster_sublimation.py -i temp.tif -o h2o_sublim.tif -s H2O

  # With custom scale/offset (if input is in Celsius, for example)
  python raster_sublimation.py -i temp_celsius.tif -o sublim.tif -s H2O --offset 273.15

  # Process with custom sticking coefficient
  python raster_sublimation.py -i temp.tif -o sublim.tif -s CO2 --alpha 0.8

Notes:
  - Input temperature should be in Kelvin
  - Output sublimation rate is in kg/(m²·yr)
  - Requires GDAL: pip install gdal
"""
    )

    parser.add_argument('-i', '--input',
                        required=True,
                        help='Input temperature raster (GeoTIFF, in Kelvin)')

    parser.add_argument('-o', '--output',
                        required=True,
                        help='Output sublimation rate raster (GeoTIFF)')

    parser.add_argument('-s', '--species',
                        required=True,
                        choices=list(VOLATILE_SPECIES.keys()),
                        help='Volatile species')

    parser.add_argument('--alpha',
                        type=float,
                        default=1.0,
                        help='Sticking coefficient (default: 1.0)')

    parser.add_argument('--scale',
                        type=float,
                        default=1.0,
                        help='Scale factor for input temperature (default: 1.0)')

    parser.add_argument('--offset',
                        type=float,
                        default=0.0,
                        help='Offset for input temperature (default: 0.0)')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Print detailed statistics')

    args = parser.parse_args()

    print(f"\nProcessing temperature raster to sublimation rate...")
    print(f"  Input:   {args.input}")
    print(f"  Output:  {args.output}")
    print(f"  Species: {args.species}")
    print(f"  Alpha:   {args.alpha}")

    if args.scale != 1.0 or args.offset != 0.0:
        print(f"  Scale:   {args.scale}")
        print(f"  Offset:  {args.offset}")
        print(f"  (Temperature_K = input * {args.scale} + {args.offset})")

    try:
        stats = process_temperature_raster(
            args.input,
            args.output,
            args.species,
            alpha=args.alpha,
            input_scale=args.scale,
            input_offset=args.offset
        )

        print(f"\n{'='*70}")
        print(f"Processing Complete!")
        print(f"{'='*70}")
        print(f"\nStatistics:")
        print(f"  Valid pixels:    {stats['valid_pixels']:,} / {stats['total_pixels']:,}")

        if stats['min_temperature_K'] is not None:
            print(f"  Temperature range: {stats['min_temperature_K']:.2f} - {stats['max_temperature_K']:.2f} K")
            print(f"  Mean temperature:  {stats['mean_temperature_K']:.2f} K")
            print(f"\n  Sublimation rate range: {stats['min_sublimation_rate']:.2e} - {stats['max_sublimation_rate']:.2e} kg/(m²·yr)")
            print(f"  Mean sublimation rate:  {stats['mean_sublimation_rate']:.2e} kg/(m²·yr)")

        print(f"\nOutput saved to: {args.output}")
        print(f"{'='*70}\n")

        if args.verbose:
            print("\nFull statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nGDAL is required for raster processing.")
        print("Install with: pip install gdal")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

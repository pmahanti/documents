#!/usr/bin/env python3
"""
Process temperature and roughness rasters to calculate sublimation with micro cold traps.

This tool takes temperature and surface roughness GeoTIFFs and computes sublimation
rates accounting for micro-scale permanently shadowed regions (micro cold traps).
"""

import argparse
import sys
from vaporp_temp import process_roughness_raster, VOLATILE_SPECIES


def main():
    parser = argparse.ArgumentParser(
        description='Calculate sublimation rates with micro cold trap effects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available species: {', '.join(VOLATILE_SPECIES.keys())}
Available roughness models: cosine, linear, exponential

Examples:
  # Basic usage with temperature and RMS slope rasters
  python raster_micro_coldtrap.py -t temp.tif -r slope_rms.tif -o sublim.tif -s H2O

  # Specify roughness type (slope vs height)
  python raster_micro_coldtrap.py -t temp.tif -r height_rms.tif -o sublim.tif -s H2O --rtype height

  # Use different roughness model
  python raster_micro_coldtrap.py -t temp.tif -r slope.tif -o sublim.tif -s H2O --model exponential

  # Fixed temperature depression instead of auto-estimation
  python raster_micro_coldtrap.py -t temp.tif -r slope.tif -o sublim.tif -s H2O --depression 50

Notes:
  - Temperature raster should be in Kelvin
  - Roughness raster: RMS slope (degrees) or RMS height (meters)
  - Output sublimation rate in kg/(m²·yr)
  - Requires GDAL and numpy: conda install -c conda-forge gdal numpy

Roughness Models:
  - cosine: f_cold = (1 - cos(slope)) / 2
    Conservative, based on geometric shadowing

  - linear: f_cold = slope / 90
    Simple linear relationship

  - exponential: f_cold = 1 - exp(-slope/15°)
    More realistic for very rough terrain

Roughness Calculation:
  From DEM, calculate RMS slope using window statistics:
  - gdaldem slope dem.tif slope.tif
  - Use focal statistics to get RMS (root-mean-square) slope
"""
    )

    parser.add_argument('-t', '--temperature',
                        required=True,
                        help='Input temperature raster (GeoTIFF, Kelvin)')

    parser.add_argument('-r', '--roughness',
                        required=True,
                        help='Input roughness raster (RMS slope in degrees or RMS height in meters)')

    parser.add_argument('-o', '--output',
                        required=True,
                        help='Output sublimation rate raster (GeoTIFF)')

    parser.add_argument('-s', '--species',
                        required=True,
                        choices=list(VOLATILE_SPECIES.keys()),
                        help='Volatile species')

    parser.add_argument('--rtype',
                        choices=['slope', 'height'],
                        default='slope',
                        help='Roughness type: slope (degrees) or height (meters) (default: slope)')

    parser.add_argument('--model',
                        choices=['cosine', 'linear', 'exponential'],
                        default='cosine',
                        help='Roughness model (default: cosine)')

    parser.add_argument('--depression',
                        type=float,
                        help='Fixed cold trap temperature depression (K). If not specified, auto-estimated.')

    parser.add_argument('--alpha',
                        type=float,
                        default=1.0,
                        help='Sticking coefficient (default: 1.0)')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Print detailed statistics')

    args = parser.parse_args()

    print(f"\nProcessing rasters with micro cold trap effects...")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Roughness:     {args.roughness} ({args.rtype})")
    print(f"  Output:        {args.output}")
    print(f"  Species:       {args.species}")
    print(f"  Roughness model: {args.model}")
    print(f"  Alpha:         {args.alpha}")

    if args.depression:
        print(f"  Cold trap depression: {args.depression} K (fixed)")
    else:
        print(f"  Cold trap depression: Auto-estimated (40% of illuminated temp)")

    try:
        stats = process_roughness_raster(
            args.temperature,
            args.roughness,
            args.output,
            args.species,
            alpha=args.alpha,
            roughness_type=args.rtype,
            roughness_model=args.model,
            cold_trap_depression_K=args.depression
        )

        print(f"\n{'='*70}")
        print(f"Processing Complete!")
        print(f"{'='*70}")
        print(f"\nStatistics:")
        print(f"  Species:              {stats['species']}")
        print(f"  Roughness type:       {stats['roughness_type']}")
        print(f"  Roughness model:      {stats['roughness_model']}")
        print(f"  Valid pixels:         {stats['valid_pixels']:,}")

        if stats['mean_cold_trap_fraction'] is not None:
            print(f"  Mean cold trap fraction: {stats['mean_cold_trap_fraction']:.3f} ({stats['mean_cold_trap_fraction']*100:.1f}%)")

        if stats['mean_sublimation_rate'] is not None:
            print(f"  Mean sublimation rate:   {stats['mean_sublimation_rate']:.2e} kg/(m²·yr)")

        print(f"\nOutput saved to: {args.output}")
        print(f"{'='*70}\n")

        if args.verbose:
            print("\nFull statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nGDAL and numpy are required for raster processing.")
        print("Install with: conda install -c conda-forge gdal numpy")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

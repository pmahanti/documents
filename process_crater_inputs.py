#!/usr/bin/env python3
"""
Command-line interface for crater input processing.

Usage:
    python process_crater_inputs.py \
        --image path/to/image.tif \
        --dtm path/to/dtm.tif \
        --craters path/to/craters.csv \
        --output output_directory \
        --projection equirectangular
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from crater_analysis.input_module import process_crater_inputs


def main():
    parser = argparse.ArgumentParser(
        description='Process crater input data and create initial outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with lat/lon crater file:
  python process_crater_inputs.py \\
      --image data/image.tif \\
      --dtm data/dtm.tif \\
      --craters data/craters.csv \\
      --output results/

  # With stereographic projection:
  python process_crater_inputs.py \\
      --image data/image.tif \\
      --dtm data/dtm.tif \\
      --craters data/craters.csv \\
      --output results/ \\
      --projection stereographic \\
      --center-lon 0 --center-lat -90

Supported projections:
  - equirectangular (simple cylindrical)
  - stereographic (polar)
  - orthographic

Input file formats:
  - Image: GeoTIFF (.tif, .tiff) or ISIS cube (.cub)
  - DTM: GeoTIFF (.tif, .tiff)
  - Craters: CSV or .diam file with headers
    Required columns: lat/lon OR x/y, plus diameter
        """
    )

    parser.add_argument(
        '--image',
        required=True,
        help='Path to contrast image (GeoTIFF or ISIS cube)'
    )

    parser.add_argument(
        '--dtm',
        required=True,
        help='Path to DTM/DEM file (GeoTIFF)'
    )

    parser.add_argument(
        '--craters',
        required=True,
        help='Path to crater location file (CSV or .diam)'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for results'
    )

    parser.add_argument(
        '--projection',
        default='equirectangular',
        choices=['equirectangular', 'stereographic', 'orthographic'],
        help='Map projection type (default: equirectangular)'
    )

    parser.add_argument(
        '--center-lon',
        type=float,
        default=0.0,
        help='Center longitude for projection (degrees, default: 0)'
    )

    parser.add_argument(
        '--center-lat',
        type=float,
        default=0.0,
        help='Center latitude for projection (degrees, default: 0)'
    )

    args = parser.parse_args()

    # Process inputs
    try:
        results = process_crater_inputs(
            image_path=args.image,
            dtm_path=args.dtm,
            crater_file_path=args.craters,
            output_dir=args.output,
            projection=args.projection,
            center_lon=args.center_lon,
            center_lat=args.center_lat
        )

        print("\n✓ Success!")
        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

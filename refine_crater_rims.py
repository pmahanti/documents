#!/usr/bin/env python3
"""
Command-line interface for crater rim refinement.

Usage:
    python refine_crater_rims.py \
        --shapefile path/to/craters_initial.shp \
        --image path/to/image.tif \
        --dtm path/to/dtm.tif \
        --output output_directory
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from crater_analysis.refine_crater_rim import refine_crater_rims


def main():
    parser = argparse.ArgumentParser(
        description='Refine crater rims using topography and computer vision',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage:
  python refine_crater_rims.py \\
      --shapefile results/craters_initial.shp \\
      --image data/image.tif \\
      --dtm data/dtm.tif \\
      --output results/

  # With custom parameters:
  python refine_crater_rims.py \\
      --shapefile craters.shp \\
      --image image.tif \\
      --dtm dtm.tif \\
      --output refined/ \\
      --min-diameter 50 \\
      --inner-radius 0.75 \\
      --outer-radius 1.25

Outputs:
  1. craters_refined.shp - Shapefile with refined rims + probability scores
  2. craters_refined_positions.png - Map of refined rim positions
  3. craters_refined_csfd.png - Crater size-frequency distribution
  4. craters_rim_differences.png - Before/after comparison

Rim Detection Probability:
  The shapefile includes a 'rim_probability' field (0-1) indicating
  detection confidence based on:
  - Topographic rim clarity
  - Edge detection strength from contrast image
  - Agreement between methods
  - Fitting error

  Probability ranges:
    > 0.7: High confidence (strong rim signal)
    0.5-0.7: Medium confidence (moderate signal)
    < 0.5: Low confidence (weak/degraded rim)
        """
    )

    parser.add_argument(
        '--shapefile',
        required=True,
        help='Path to input shapefile (from Step 0: input_module)'
    )

    parser.add_argument(
        '--image',
        required=True,
        help='Path to contrast image (GeoTIFF)'
    )

    parser.add_argument(
        '--dtm',
        required=True,
        help='Path to DTM/DEM file (GeoTIFF)'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for results'
    )

    parser.add_argument(
        '--min-diameter',
        type=float,
        default=60.0,
        help='Minimum crater diameter in meters (default: 60)'
    )

    parser.add_argument(
        '--inner-radius',
        type=float,
        default=0.8,
        help='Inner search radius as fraction of R (default: 0.8)'
    )

    parser.add_argument(
        '--outer-radius',
        type=float,
        default=1.2,
        help='Outer search radius as fraction of R (default: 1.2)'
    )

    parser.add_argument(
        '--keep-regional-topo',
        action='store_true',
        help='Keep regional topography (do not remove slope)'
    )

    parser.add_argument(
        '--plot-individual',
        action='store_true',
        help='Generate diagnostic plots for each crater (slow)'
    )

    args = parser.parse_args()

    # Process rim refinement
    try:
        results = refine_crater_rims(
            input_shapefile=args.shapefile,
            dem_path=args.dtm,
            image_path=args.image,
            output_dir=args.output,
            min_diameter=args.min_diameter,
            inner_radius=args.inner_radius,
            outer_radius=args.outer_radius,
            remove_external_topo=not args.keep_regional_topo,
            plot_individual=args.plot_individual
        )

        print("\n✓ Success!")
        print("\nQuality Summary:")
        stats = results['statistics']
        print(f"  High confidence craters: {stats['high_confidence']} "
              f"({stats['high_confidence']/stats['total_craters']*100:.1f}%)")
        print(f"  Medium confidence: {stats['medium_confidence']} "
              f"({stats['medium_confidence']/stats['total_craters']*100:.1f}%)")
        print(f"  Low confidence: {stats['low_confidence']} "
              f"({stats['low_confidence']/stats['total_craters']*100:.1f}%)")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

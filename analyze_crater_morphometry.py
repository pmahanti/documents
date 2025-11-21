#!/usr/bin/env python3
"""
Command-line interface for crater morphometry analysis (Step 3).

This script performs dual-method morphometry analysis:
- Method 1: Rim perimeter-based depth estimation
- Method 2: 2D Gaussian floor fitting

Outputs:
1. Shapefile with morphometry measurements
2. Scatter plots (depth vs diameter, d/D vs diameter)
3. Probability distribution plots
4. CSV with morphometry data
5. CSV with conditional probabilities
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from crater_analysis.analyze_morphometry import analyze_crater_morphometry


def main():
    """Main entry point for morphometry analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze crater morphometry using dual methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python analyze_crater_morphometry.py \\
      --shapefile results/craters_refined.shp \\
      --dtm data/dtm.tif \\
      --image data/image.tif \\
      --output results/morphometry/

  # With custom parameters
  python analyze_crater_morphometry.py \\
      --shapefile refined.shp \\
      --dtm dtm.tif \\
      --image image.tif \\
      --output morphometry/ \\
      --min-diameter 100 \\
      --plot-individual

Methods:
  Method 1: Rim perimeter analysis (existing algorithm)
    - Detects rim as perimeter pixels
    - Computes mean rim height
    - Finds floor minimum elevation
    - Depth = rim - floor

  Method 2: 2D Gaussian floor fitting (NEW)
    - Fits Gaussian to inverted crater
    - Gaussian peak = floor elevation
    - Constrained to avoid overshoot/undershoot
    - Provides fit quality metric

Outputs:
  1. craters_morphometry.shp - Shapefile with all measurements
  2. morphometry_scatter_plots.png - Depth and d/D vs diameter
  3. probability_distributions.png - Joint and marginal distributions
  4. morphometry_data.csv - All measurements (no geometry)
  5. conditional_probability.csv - P(d|D) and P(D|d)
        """
    )

    parser.add_argument(
        '--shapefile',
        required=True,
        help='Input shapefile from Step 2 (rim refinement output)'
    )

    parser.add_argument(
        '--dtm',
        required=True,
        help='Path to DTM/DEM file (GeoTIFF format)'
    )

    parser.add_argument(
        '--image',
        required=True,
        help='Path to contrast image/orthophoto (GeoTIFF format)'
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
        help='Minimum crater diameter to process in meters (default: 60.0)'
    )

    parser.add_argument(
        '--keep-regional-topo',
        action='store_true',
        help='Do not remove regional topography (default: remove it)'
    )

    parser.add_argument(
        '--plot-individual',
        action='store_true',
        help='Generate diagnostic plots for each crater (slow for many craters)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CRATER MORPHOMETRY ANALYSIS (STEP 3)")
    print("=" * 70)
    print("\nDual-Method Approach:")
    print("  Method 1: Rim perimeter analysis")
    print("  Method 2: 2D Gaussian floor fitting")
    print("\nInputs:")
    print(f"  Shapefile: {args.shapefile}")
    print(f"  DTM: {args.dtm}")
    print(f"  Image: {args.image}")
    print(f"  Output directory: {args.output}")
    print("\nParameters:")
    print(f"  Min diameter: {args.min_diameter} m")
    print(f"  Remove regional topo: {not args.keep_regional_topo}")
    print(f"  Plot individual craters: {args.plot_individual}")
    print("\n" + "=" * 70)

    # Validate inputs
    from pathlib import Path
    shapefile_path = Path(args.shapefile)
    dtm_path = Path(args.dtm)
    image_path = Path(args.image)

    if not shapefile_path.exists():
        print(f"Error: Shapefile not found: {args.shapefile}")
        return 1

    if not dtm_path.exists():
        print(f"Error: DTM file not found: {args.dtm}")
        return 1

    if not image_path.exists():
        print(f"Error: Image file not found: {args.image}")
        return 1

    # Define output shapefile
    output_shapefile = str(Path(args.output) / 'craters_morphometry.shp')

    # Run analysis
    try:
        results = analyze_crater_morphometry(
            input_shapefile=args.shapefile,
            output_shapefile=output_shapefile,
            dem_path=args.dtm,
            orthophoto_path=args.image,
            output_dir=args.output,
            min_diameter=args.min_diameter,
            remove_external_topo=not args.keep_regional_topo,
            plot_individual=args.plot_individual
        )

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print("\nOutputs:")
        print(f"  1. Shapefile: {results['shapefile']}")
        print(f"  2. Scatter plots: {results['scatter_plots']}")
        print(f"  3. Probability plots: {results['probability_plots']}")
        print(f"  4. Morphometry CSV: {results['csv_morphometry']}")
        print(f"  5. Conditional probability CSV: {results['conditional_probability']}")

        print("\nSummary Statistics:")
        stats = results['statistics']
        print(f"  Total craters analyzed: {stats['total_craters']}")

        if 'method1' in stats and stats['method1']:
            m1 = stats['method1']
            print(f"\n  Method 1 (Rim perimeter):")
            print(f"    Successful: {m1['count']}/{stats['total_craters']}")
            print(f"    Mean d/D: {m1['mean_d_D']:.4f} ± {m1['std_d_D']:.4f}")
            print(f"    Mean depth: {m1['mean_depth']:.2f} ± {m1['std_depth']:.2f} m")

        if 'method2' in stats and stats['method2']:
            m2 = stats['method2']
            print(f"\n  Method 2 (Gaussian fitting):")
            print(f"    Successful: {m2['count']}/{stats['total_craters']}")
            print(f"    Mean d/D: {m2['mean_d_D']:.4f} ± {m2['std_d_D']:.4f}")
            print(f"    Mean depth: {m2['mean_depth']:.2f} ± {m2['std_depth']:.2f} m")
            print(f"    Mean fit quality: {m2['mean_fit_quality']:.3f}")

        print("\n" + "=" * 70)
        print("✓ Morphometry analysis complete!")
        print("=" * 70)

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)
        print(f"\nAnalysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

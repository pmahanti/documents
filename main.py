#!/usr/bin/env python3
"""
Main script for crater morphometry analysis.

This script orchestrates the complete workflow:
1. Load configuration
2. Prepare crater geometries
3. Refine crater rims using topography
4. Compute depth-diameter ratios
5. Export results
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from crater_analysis.config import Config
from crater_analysis.refinement import prepare_crater_geometries, update_crater_rims
from crater_analysis.morphometry import (compute_depth_diameter_ratios,
                                         export_morphometry_to_csv,
                                         get_morphometry_summary)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze crater morphometry from DEM data'
    )
    parser.add_argument(
        '--region',
        type=str,
        default=None,
        help='Region name (default: use config default)'
    )
    parser.add_argument(
        '--min-diameter',
        type=float,
        default=60,
        help='Minimum crater diameter in meters (default: 60)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable plotting'
    )
    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help='Steps to run: prepare, refine, analyze, or all (default: all)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Crater Morphometry Analysis")
    print("=" * 60)

    # Load configuration
    try:
        config = Config(args.config)
        print(f"\nLoaded configuration from: {config.config_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease create a configuration file or specify --config")
        return 1

    # Get paths for region
    region_name = args.region or config.config['default_region']
    print(f"Region: {region_name}")

    try:
        dem_path = config.get_dem_path(region_name)
        orthophoto_path = config.get_orthophoto_path(region_name)
        shapefile_base = config.get_shapefile_path(region_name)
    except (ValueError, KeyError) as e:
        print(f"Error: {e}")
        return 1

    min_diameter = args.min_diameter
    plot = not args.no_plot

    print(f"\nPaths:")
    print(f"  DEM: {dem_path}")
    print(f"  Orthophoto: {orthophoto_path}")
    print(f"  Shapefile: {shapefile_base}.shp")
    print(f"\nParameters:")
    print(f"  Min diameter: {min_diameter} m")
    print(f"  Plotting: {plot}")

    # Define output paths
    geom_shapefile = f"{shapefile_base}.geom.shp"
    refined_shapefile = f"{shapefile_base}.refined.shp"
    data_shapefile = f"{shapefile_base}.data.shp"
    csv_output = f"{shapefile_base}.morphometry.csv"

    steps = args.steps.lower()

    # Step 1: Prepare geometries
    if steps in ['all', 'prepare']:
        print("\n" + "=" * 60)
        print("Step 1: Preparing crater geometries")
        print("=" * 60)

        try:
            prepare_crater_geometries(
                input_shapefile=f"{shapefile_base}.shp",
                output_shapefile=geom_shapefile,
                min_diameter=min_diameter
            )
        except Exception as e:
            print(f"Error in prepare step: {e}")
            if steps != 'all':
                return 1
            print("Skipping to next step...")

    # Step 2: Refine crater rims
    if steps in ['all', 'refine']:
        print("\n" + "=" * 60)
        print("Step 2: Refining crater rims")
        print("=" * 60)

        try:
            update_crater_rims(
                input_shapefile=geom_shapefile,
                output_shapefile=refined_shapefile,
                dem_path=dem_path,
                orthophoto_path=orthophoto_path,
                min_diameter=min_diameter,
                plot=plot,
                remove_external_topo=True
            )
        except Exception as e:
            print(f"Error in refine step: {e}")
            if steps != 'all':
                return 1
            print("Skipping to next step...")

    # Step 3: Compute morphometry
    if steps in ['all', 'analyze']:
        print("\n" + "=" * 60)
        print("Step 3: Computing crater morphometry")
        print("=" * 60)

        try:
            compute_depth_diameter_ratios(
                input_shapefile=refined_shapefile,
                output_shapefile=data_shapefile,
                dem_path=dem_path,
                orthophoto_path=orthophoto_path,
                min_diameter=min_diameter,
                plot=plot,
                remove_external_topo=True
            )

            # Export to CSV
            print("\n" + "-" * 60)
            print("Exporting to CSV")
            print("-" * 60)
            export_morphometry_to_csv(data_shapefile, csv_output)

            # Print summary
            print("\n" + "-" * 60)
            print("Summary Statistics")
            print("-" * 60)
            summary = get_morphometry_summary(data_shapefile)
            print(f"Total craters: {summary['count']}")
            print(f"\nDiameter (m):")
            print(f"  Mean: {summary['diameter']['mean']:.1f}")
            print(f"  Range: {summary['diameter']['min']:.1f} - {summary['diameter']['max']:.1f}")
            print(f"\nd/D ratio:")
            print(f"  Mean: {summary['d_D_ratio']['mean']:.3f}")
            print(f"  Range: {summary['d_D_ratio']['min']:.3f} - {summary['d_D_ratio']['max']:.3f}")
            print(f"\nDepth (m):")
            print(f"  Mean: {summary['depth']['mean']:.1f}")
            print(f"  Range: {summary['depth']['min']:.1f} - {summary['depth']['max']:.1f}")

        except Exception as e:
            print(f"Error in analyze step: {e}")
            if steps != 'all':
                return 1

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())

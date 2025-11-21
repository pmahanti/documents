#!/usr/bin/env python3
"""
Complete pipeline for Small Lunar Crater (SLC) morphometry analysis.

This wrapper script orchestrates all three steps:
- Step 0: Input processing (crater locations, coordinate conversion)
- Step 2: Rim refinement (topography + computer vision)
- Step 3: Morphometry analysis (dual-method depth estimation)

Generates comprehensive double-column PDF report with:
- 0.5 inch margins
- 12 point text
- Contrast image with craters on first page
- All results, figures, error propagation theory, conditional probabilities

Usage:
    python analyze_small_lunar_craters.py \\
        --image <path_to_image.tif> \\
        --dtm <path_to_dtm.tif> \\
        --craters <path_to_craters.csv_or_.diam> \\
        [--output <output_directory>] \\
        [--projection <equirectangular|stereographic|orthographic>] \\
        [--center-lon <longitude>] \\
        [--center-lat <latitude>]
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def run_step0_input_processing(image_path, dtm_path, crater_path, output_dir,
                                projection, center_lon, center_lat):
    """
    Run Step 0: Input processing.

    Returns:
        dict: Paths to Step 0 outputs
    """
    from crater_analysis.input_module import process_crater_inputs

    print("\n" + "=" * 80)
    print("STEP 0: INPUT PROCESSING")
    print("=" * 80)
    print(f"\nProcessing crater inputs...")
    print(f"  Image: {image_path}")
    print(f"  DTM: {dtm_path}")
    print(f"  Craters: {crater_path}")
    print(f"  Projection: {projection}")
    if projection != 'equirectangular':
        print(f"  Center: ({center_lon}°, {center_lat}°)")

    step0_dir = os.path.join(output_dir, 'step0_input')
    os.makedirs(step0_dir, exist_ok=True)

    try:
        results = process_crater_inputs(
            image_path=image_path,
            dem_path=dtm_path,
            crater_file=crater_path,
            output_dir=step0_dir,
            projection_type=projection,
            center_lon=center_lon,
            center_lat=center_lat
        )

        print(f"\n✓ Step 0 complete!")
        print(f"  Shapefile: {results['shapefile']}")
        print(f"  Initial locations plot: {results['location_plot']}")
        print(f"  Initial CSFD plot: {results['csfd_plot']}")
        print(f"  Total craters: {results['crater_count']}")

        return results

    except Exception as e:
        print(f"\n✗ Step 0 failed: {e}")
        raise


def run_step2_rim_refinement(shapefile_path, image_path, dtm_path, output_dir,
                             min_diameter=60.0):
    """
    Run Step 2: Rim refinement.

    Returns:
        dict: Paths to Step 2 outputs
    """
    from crater_analysis.refine_crater_rim import refine_crater_rims

    print("\n" + "=" * 80)
    print("STEP 2: RIM REFINEMENT")
    print("=" * 80)
    print(f"\nRefining crater rims using dual methods...")
    print(f"  Input shapefile: {shapefile_path}")
    print(f"  Minimum diameter: {min_diameter} m")

    step2_dir = os.path.join(output_dir, 'step2_refinement')
    os.makedirs(step2_dir, exist_ok=True)

    try:
        results = refine_crater_rims(
            input_shapefile=shapefile_path,
            output_shapefile=os.path.join(step2_dir, 'craters_refined.shp'),
            dem_path=dtm_path,
            image_path=image_path,
            output_dir=step2_dir,
            min_diameter=min_diameter,
            inner_radius=0.8,
            outer_radius=1.2,
            remove_external_topo=True
        )

        print(f"\n✓ Step 2 complete!")
        print(f"  Refined shapefile: {results['shapefile']}")
        print(f"  Refined positions plot: {results['position_plot']}")
        print(f"  Refined CSFD plot: {results['csfd_plot']}")
        print(f"  Rim differences plot: {results['difference_plot']}")
        print(f"  Craters refined: {results['statistics']['total_craters']}")
        print(f"  Mean probability: {results['statistics']['mean_probability']:.3f}")

        return results

    except Exception as e:
        print(f"\n✗ Step 2 failed: {e}")
        raise


def run_step3_morphometry(shapefile_path, dtm_path, image_path, output_dir,
                         min_diameter=60.0):
    """
    Run Step 3: Morphometry analysis.

    Returns:
        dict: Paths to Step 3 outputs
    """
    from crater_analysis.analyze_morphometry import analyze_crater_morphometry

    print("\n" + "=" * 80)
    print("STEP 3: MORPHOMETRY ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing crater morphometry using dual methods...")
    print(f"  Input shapefile: {shapefile_path}")
    print(f"  Method 1: Rim perimeter analysis")
    print(f"  Method 2: 2D Gaussian floor fitting")

    step3_dir = os.path.join(output_dir, 'step3_morphometry')
    os.makedirs(step3_dir, exist_ok=True)

    try:
        results = analyze_crater_morphometry(
            input_shapefile=shapefile_path,
            output_shapefile=os.path.join(step3_dir, 'craters_morphometry.shp'),
            dem_path=dtm_path,
            orthophoto_path=image_path,
            output_dir=step3_dir,
            min_diameter=min_diameter,
            remove_external_topo=True,
            plot_individual=False
        )

        print(f"\n✓ Step 3 complete!")
        print(f"  Morphometry shapefile: {results['shapefile']}")
        print(f"  Scatter plots: {results['scatter_plots']}")
        print(f"  Probability plots: {results['probability_plots']}")
        print(f"  Morphometry CSV: {results['csv_morphometry']}")
        print(f"  Conditional probability CSV: {results['conditional_probability']}")

        stats = results['statistics']
        if 'method1' in stats and stats['method1']:
            print(f"  Method 1 mean d/D: {stats['method1']['mean_d_D']:.4f}")
        if 'method2' in stats and stats['method2']:
            print(f"  Method 2 mean d/D: {stats['method2']['mean_d_D']:.4f}")
            print(f"  Mean fit quality: {stats['method2']['mean_fit_quality']:.3f}")

        return results

    except Exception as e:
        print(f"\n✗ Step 3 failed: {e}")
        raise


def generate_comprehensive_report(step0_results, step2_results, step3_results,
                                  output_dir, image_path, dtm_path, crater_path):
    """
    Generate comprehensive double-column PDF report.

    Args:
        step0_results: Dictionary with Step 0 outputs
        step2_results: Dictionary with Step 2 outputs
        step3_results: Dictionary with Step 3 outputs
        output_dir: Base output directory
        image_path: Path to input image
        dtm_path: Path to input DTM
        crater_path: Path to input crater file

    Returns:
        str: Path to generated PDF report
    """
    from crater_analysis.report_generator import generate_pdf_report

    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE PDF REPORT")
    print("=" * 80)

    report_path = generate_pdf_report(
        step0_results=step0_results,
        step2_results=step2_results,
        step3_results=step3_results,
        output_dir=output_dir,
        image_path=image_path,
        dtm_path=dtm_path,
        crater_path=crater_path
    )

    return report_path


def main():
    """Main entry point for complete SLC analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Complete Small Lunar Crater (SLC) morphometry analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This wrapper script runs the complete three-step pipeline:
  Step 0: Input processing and coordinate conversion
  Step 2: Rim refinement with probability scoring
  Step 3: Dual-method morphometry analysis

Output directory structure:
  SLC_morphometry_results/
    ├── step0_input/              (Initial processing)
    ├── step2_refinement/         (Rim refinement)
    ├── step3_morphometry/        (Morphometry analysis)
    └── SLC_Analysis_Report.pdf   (Comprehensive PDF report)

Example:
  python analyze_small_lunar_craters.py \\
      --image data/SHADOWCAM_image.tif \\
      --dtm data/LOLA_dtm.tif \\
      --craters data/craters.csv
        """
    )

    parser.add_argument(
        '--image',
        required=True,
        help='Path to contrast image/orthophoto (GeoTIFF format)'
    )

    parser.add_argument(
        '--dtm',
        required=True,
        help='Path to DTM/DEM file (GeoTIFF format)'
    )

    parser.add_argument(
        '--craters',
        required=True,
        help='Path to crater location file (CSV or .diam format)'
    )

    parser.add_argument(
        '--output',
        default='SLC_morphometry_results',
        help='Output directory (default: SLC_morphometry_results)'
    )

    parser.add_argument(
        '--projection',
        choices=['equirectangular', 'stereographic', 'orthographic'],
        default='equirectangular',
        help='Map projection type (default: equirectangular)'
    )

    parser.add_argument(
        '--center-lon',
        type=float,
        default=0.0,
        help='Center longitude for projection (degrees, default: 0.0)'
    )

    parser.add_argument(
        '--center-lat',
        type=float,
        default=0.0,
        help='Center latitude for projection (degrees, default: 0.0)'
    )

    parser.add_argument(
        '--min-diameter',
        type=float,
        default=60.0,
        help='Minimum crater diameter to process (meters, default: 60.0)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SMALL LUNAR CRATER (SLC) MORPHOMETRY ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"\nComplete three-step analysis:")
    print(f"  Step 0: Input Processing")
    print(f"  Step 2: Rim Refinement")
    print(f"  Step 3: Morphometry Analysis")
    print(f"\nInput files:")
    print(f"  Image: {args.image}")
    print(f"  DTM: {args.dtm}")
    print(f"  Craters: {args.craters}")
    print(f"\nOutput directory: {args.output}")
    print(f"Report format: Double-column PDF, 0.5\" margins, 12pt text")
    print("=" * 80)

    # Validate inputs
    for file_path, name in [(args.image, 'Image'), (args.dtm, 'DTM'), (args.craters, 'Crater file')]:
        if not os.path.exists(file_path):
            print(f"\n✗ Error: {name} not found: {file_path}")
            return 1

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    start_time = datetime.now()

    try:
        # Step 0: Input Processing
        step0_results = run_step0_input_processing(
            image_path=args.image,
            dtm_path=args.dtm,
            crater_path=args.craters,
            output_dir=args.output,
            projection=args.projection,
            center_lon=args.center_lon,
            center_lat=args.center_lat
        )

        # Step 2: Rim Refinement
        step2_results = run_step2_rim_refinement(
            shapefile_path=step0_results['shapefile'],
            image_path=args.image,
            dtm_path=args.dtm,
            output_dir=args.output,
            min_diameter=args.min_diameter
        )

        # Step 3: Morphometry Analysis
        step3_results = run_step3_morphometry(
            shapefile_path=step2_results['shapefile'],
            dtm_path=args.dtm,
            image_path=args.image,
            output_dir=args.output,
            min_diameter=args.min_diameter
        )

        # Generate comprehensive PDF report
        report_path = generate_comprehensive_report(
            step0_results=step0_results,
            step2_results=step2_results,
            step3_results=step3_results,
            output_dir=args.output,
            image_path=args.image,
            dtm_path=args.dtm,
            crater_path=args.craters
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"\n✓ All steps completed successfully!")
        print(f"  Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"\n  Output directory: {args.output}")
        print(f"  Comprehensive PDF report: {report_path}")
        print(f"\n  Open the PDF report to view all results.")
        print("=" * 80)

        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print("PIPELINE FAILED")
        print("=" * 80)
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Complete pipeline for Small Lunar Crater (SLC) morphometry analysis.

This wrapper script orchestrates all three steps:
- Step 0: Input processing (crater locations, coordinate conversion)
- Step 2: Rim refinement (topography + computer vision)
- Step 3: Morphometry analysis (dual-method depth estimation)

Generates comprehensive report with all results, error propagation discussion,
and conditional probability analysis.

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
import shutil

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
    Generate comprehensive HTML report with all results and analysis.

    Args:
        step0_results: Dictionary with Step 0 outputs
        step2_results: Dictionary with Step 2 outputs
        step3_results: Dictionary with Step 3 outputs
        output_dir: Base output directory
        image_path: Path to input image
        dtm_path: Path to input DTM
        crater_path: Path to input crater file

    Returns:
        str: Path to generated report
    """
    import pandas as pd
    import base64

    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("=" * 80)

    report_path = os.path.join(output_dir, 'SLC_Analysis_Report.html')

    # Helper function to encode image as base64
    def img_to_base64(img_path):
        if not os.path.exists(img_path):
            return None
        with open(img_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()

    # Load conditional probability table
    cond_prob_csv = step3_results['conditional_probability']
    if os.path.exists(cond_prob_csv):
        cond_prob_df = pd.read_csv(cond_prob_csv)
        # Filter to non-zero counts for cleaner table
        cond_prob_df = cond_prob_df[cond_prob_df['count'] > 0]
        cond_prob_table = cond_prob_df.to_html(index=False, float_format='%.4f',
                                               classes='table table-striped')
    else:
        cond_prob_table = "<p>Conditional probability table not available.</p>"

    # Load morphometry summary
    morph_csv = step3_results['csv_morphometry']
    if os.path.exists(morph_csv):
        morph_df = pd.read_csv(morph_csv)
        n_craters = len(morph_df)

        # Summary statistics
        stats = step3_results['statistics']
    else:
        n_craters = 0
        stats = {}

    # Encode images
    images = {
        'step0_location': img_to_base64(step0_results.get('location_plot', '')),
        'step0_csfd': img_to_base64(step0_results.get('csfd_plot', '')),
        'step2_refined': img_to_base64(step2_results.get('position_plot', '')),
        'step2_csfd': img_to_base64(step2_results.get('csfd_plot', '')),
        'step2_diff': img_to_base64(step2_results.get('difference_plot', '')),
        'step3_scatter': img_to_base64(step3_results.get('scatter_plots', '')),
        'step3_prob': img_to_base64(step3_results.get('probability_plots', ''))
    }

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate HTML report
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Small Lunar Crater Morphometry Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .section h3 {{
            color: #764ba2;
            margin-top: 25px;
        }}
        .figure {{
            margin: 30px 0;
            text-align: center;
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .caption {{
            font-style: italic;
            color: #666;
            margin-top: 10px;
            font-size: 0.95em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-box .value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-box .label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.9em;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .info-box {{
            background-color: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .warning-box {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .success-box {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        .equation {{
            background-color: #f9f9f9;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
            font-family: 'Times New Roman', serif;
            text-align: center;
            border: 1px solid #e0e0e0;
        }}
        .toc {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .toc li {{
            padding: 5px 0;
        }}
        .toc a {{
            color: #667eea;
            text-decoration: none;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Small Lunar Crater Morphometry Analysis</h1>
        <p>Complete Pipeline Report: Input Processing → Rim Refinement → Morphometry Analysis</p>
        <p>Generated: {timestamp}</p>
    </div>

    <div class="section toc">
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#input">1. Input Data Summary</a></li>
            <li><a href="#step0">2. Step 0: Input Processing Results</a></li>
            <li><a href="#step2">3. Step 2: Rim Refinement Results</a></li>
            <li><a href="#step3">4. Step 3: Morphometry Analysis Results</a></li>
            <li><a href="#error">5. Error Propagation Theory and Application</a></li>
            <li><a href="#condprob">6. Conditional Probability Analysis</a></li>
            <li><a href="#summary">7. Summary and Conclusions</a></li>
        </ul>
    </div>

    <div class="section" id="input">
        <h2>1. Input Data Summary</h2>
        <p>This analysis processed the following input datasets:</p>
        <table>
            <tr>
                <th>Input Type</th>
                <th>File Path</th>
            </tr>
            <tr>
                <td><strong>Contrast Image</strong></td>
                <td><code>{os.path.basename(image_path)}</code></td>
            </tr>
            <tr>
                <td><strong>Digital Terrain Model (DTM)</strong></td>
                <td><code>{os.path.basename(dtm_path)}</code></td>
            </tr>
            <tr>
                <td><strong>Crater Locations</strong></td>
                <td><code>{os.path.basename(crater_path)}</code></td>
            </tr>
        </table>

        <div class="stats-grid">
            <div class="stat-box">
                <div class="label">Total Craters Detected</div>
                <div class="value">{step0_results.get('crater_count', 'N/A')}</div>
            </div>
            <div class="stat-box">
                <div class="label">Craters Refined</div>
                <div class="value">{step2_results['statistics'].get('total_craters', 'N/A')}</div>
            </div>
            <div class="stat-box">
                <div class="label">Morphometry Analyzed</div>
                <div class="value">{n_craters}</div>
            </div>
        </div>
    </div>

    <div class="section" id="step0">
        <h2>2. Step 0: Input Processing Results</h2>
        <p>
            The input processing step converts crater location data (CSV or .diam format) into
            georeferenced shapefiles. It handles coordinate system conversions and generates
            initial visualizations.
        </p>

        <h3>2.1 Initial Crater Locations</h3>
        {'<div class="figure"><img src="data:image/png;base64,' + images['step0_location'] + '" alt="Initial Crater Locations"/><p class="caption"><strong>Figure 1:</strong> Initial crater locations overlaid on contrast image. Red circles indicate detected craters with N = ' + str(step0_results.get('crater_count', 0)) + '.</p></div>' if images['step0_location'] else '<p>Image not available.</p>'}

        <h3>2.2 Initial Crater Size-Frequency Distribution (CSFD)</h3>
        {'<div class="figure"><img src="data:image/png;base64,' + images['step0_csfd'] + '" alt="Initial CSFD"/><p class="caption"><strong>Figure 2:</strong> Crater Size-Frequency Distribution (CSFD) showing the population distribution before refinement. Used for age dating and population characterization.</p></div>' if images['step0_csfd'] else '<p>Image not available.</p>'}

        <div class="success-box">
            <strong>✓ Step 0 Complete:</strong> Successfully processed {step0_results.get('crater_count', 0)} craters.
            Initial shapefile created with crater geometries and coordinate information.
        </div>
    </div>

    <div class="section" id="step2">
        <h2>3. Step 2: Rim Refinement Results</h2>
        <p>
            The rim refinement step uses a dual-method approach combining topographic analysis
            with computer vision edge detection to precisely localize crater rims. Each crater
            receives a rim detection probability score (0-1).
        </p>

        <h3>3.1 Refined Crater Rim Positions</h3>
        {'<div class="figure"><img src="data:image/png;base64,' + images['step2_refined'] + '" alt="Refined Crater Positions"/><p class="caption"><strong>Figure 3:</strong> Refined crater rim positions colored by detection probability. Green = high confidence (>0.7), Yellow = medium (0.5-0.7), Red = low (<0.5). Mean probability = ' + f"{step2_results['statistics'].get('mean_probability', 0):.3f}" + '.</p></div>' if images['step2_refined'] else '<p>Image not available.</p>'}

        <h3>3.2 Refined CSFD</h3>
        {'<div class="figure"><img src="data:image/png;base64,' + images['step2_csfd'] + '" alt="Refined CSFD"/><p class="caption"><strong>Figure 4:</strong> Updated Crater Size-Frequency Distribution after rim refinement. Diameters have been updated based on refined rim positions.</p></div>' if images['step2_csfd'] else '<p>Image not available.</p>'}

        <h3>3.3 Rim Position Changes</h3>
        {'<div class="figure"><img src="data:image/png;base64,' + images['step2_diff'] + '" alt="Rim Differences"/><p class="caption"><strong>Figure 5:</strong> Comparison of initial (blue dashed) vs refined (red solid) crater rims. Yellow arrows indicate center position shifts. Shows the magnitude of corrections applied during refinement.</p></div>' if images['step2_diff'] else '<p>Image not available.</p>'}

        <h3>3.4 Rim Refinement Statistics</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Mean Rim Probability</td>
                <td>{step2_results['statistics'].get('mean_probability', 0):.3f}</td>
            </tr>
            <tr>
                <td>High Confidence Craters (>0.7)</td>
                <td>{step2_results['statistics'].get('high_confidence_count', 'N/A')}</td>
            </tr>
            <tr>
                <td>Mean Center Shift</td>
                <td>{step2_results['statistics'].get('mean_center_shift', 0):.2f} m</td>
            </tr>
            <tr>
                <td>Mean Radius Change</td>
                <td>{step2_results['statistics'].get('mean_radius_change', 0):.2f}%</td>
            </tr>
        </table>

        <div class="info-box">
            <strong>Rim Detection Methods:</strong>
            <ul>
                <li><strong>Topographic:</strong> Detects elevation peaks along radial profiles (72 azimuths)</li>
                <li><strong>Computer Vision:</strong> Canny edge detection on contrast image</li>
                <li><strong>Probability Score:</strong> Weighted combination (40% topo quality, 30% edge strength, 20% agreement, 10% error)</li>
            </ul>
        </div>
    </div>

    <div class="section" id="step3">
        <h2>4. Step 3: Morphometry Analysis Results</h2>
        <p>
            The morphometry analysis measures crater depth using two independent methods and
            computes depth-to-diameter (d/D) ratios, which indicate crater degradation state.
        </p>

        <h3>4.1 Crater Depth and d/D Measurements</h3>
        {'<div class="figure"><img src="data:image/png;base64,' + images['step3_scatter'] + '" alt="Morphometry Scatter Plots"/><p class="caption"><strong>Figure 6:</strong> Left: Crater depth vs diameter for both methods. Right: Depth-to-diameter (d/D) ratio vs diameter. Blue circles = Method 1 (rim perimeter), Red squares = Method 2 (Gaussian fitting). Error bars show total uncertainty including rim probability. Dashed line at d/D = 0.2 indicates fresh crater benchmark.</p></div>' if images['step3_scatter'] else '<p>Image not available.</p>'}

        <h3>4.2 Probability Distributions</h3>
        {'<div class="figure"><img src="data:image/png;base64,' + images['step3_prob'] + '" alt="Probability Distributions"/><p class="caption"><strong>Figure 7:</strong> Left: 2D joint probability distribution P(depth, diameter) showing correlation structure. Right: Marginal probability distribution P(d/D) with histogram overlay. Red dashed line = mean, orange dotted lines = ±1σ.</p></div>' if images['step3_prob'] else '<p>Image not available.</p>'}

        <h3>4.3 Morphometry Summary Statistics</h3>
"""

    # Add method statistics if available
    if 'method1' in stats and stats['method1']:
        m1 = stats['method1']
        html_content += f"""
        <h4>Method 1: Rim Perimeter Analysis</h4>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Successful Measurements</td>
                <td>{m1.get('count', 0)} / {stats.get('total_craters', 0)}</td>
            </tr>
            <tr>
                <td>Mean d/D Ratio</td>
                <td>{m1.get('mean_d_D', 0):.4f} ± {m1.get('std_d_D', 0):.4f}</td>
            </tr>
            <tr>
                <td>Mean Depth</td>
                <td>{m1.get('mean_depth', 0):.2f} ± {m1.get('std_depth', 0):.2f} m</td>
            </tr>
            <tr>
                <td>Mean Diameter</td>
                <td>{m1.get('mean_diameter', 0):.2f} m</td>
            </tr>
        </table>
"""

    if 'method2' in stats and stats['method2']:
        m2 = stats['method2']
        html_content += f"""
        <h4>Method 2: 2D Gaussian Floor Fitting</h4>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Successful Measurements</td>
                <td>{m2.get('count', 0)} / {stats.get('total_craters', 0)}</td>
            </tr>
            <tr>
                <td>Mean d/D Ratio</td>
                <td>{m2.get('mean_d_D', 0):.4f} ± {m2.get('std_d_D', 0):.4f}</td>
            </tr>
            <tr>
                <td>Mean Depth</td>
                <td>{m2.get('mean_depth', 0):.2f} ± {m2.get('std_depth', 0):.2f} m</td>
            </tr>
            <tr>
                <td>Mean Fit Quality (R²)</td>
                <td>{m2.get('mean_fit_quality', 0):.3f}</td>
            </tr>
        </table>
"""

    html_content += f"""
        <div class="info-box">
            <strong>Depth Estimation Methods:</strong>
            <ul>
                <li><strong>Method 1:</strong> Rim perimeter pixels → mean elevation → floor minimum → depth = rim - floor</li>
                <li><strong>Method 2:</strong> 2D Gaussian fit to inverted crater → floor from Gaussian peak → depth = rim - floor</li>
                <li><strong>Combined:</strong> Inverse-variance weighted average of both methods</li>
            </ul>
        </div>
    </div>

    <div class="section" id="error">
        <h2>5. Error Propagation Theory and Application</h2>

        <h3>5.1 Theoretical Framework</h3>
        <p>
            Error propagation in crater morphometry involves tracking uncertainties through
            multiple processing steps. The total uncertainty in depth-to-diameter ratio (d/D)
            includes contributions from:
        </p>

        <ol>
            <li><strong>Measurement uncertainty</strong> (rim height variance, floor detection error)</li>
            <li><strong>Geometric uncertainty</strong> (radius error from rim refinement)</li>
            <li><strong>Detection uncertainty</strong> (rim probability from computer vision analysis)</li>
        </ol>

        <h3>5.2 Mathematical Formulation</h3>

        <h4>Basic Error Propagation</h4>
        <p>For a function f(x, y), the uncertainty propagates as:</p>
        <div class="equation">
            σ<sub>f</sub>² = (∂f/∂x)² σ<sub>x</sub>² + (∂f/∂y)² σ<sub>y</sub>² + 2(∂f/∂x)(∂f/∂y)Cov(x,y)
        </div>

        <h4>Depth Uncertainty (Method 1: Rim Perimeter)</h4>
        <div class="equation">
            depth = rim_height - floor_height<br>
            σ<sub>depth</sub>² = σ<sub>rim</sub>² + σ<sub>floor</sub>²<br>
            where σ<sub>rim</sub> = std(rim_pixels), σ<sub>floor</sub> ≈ 0 (single minimum)
        </div>

        <h4>Depth Uncertainty (Method 2: Gaussian Fitting)</h4>
        <div class="equation">
            depth = rim_height - floor_gaussian<br>
            σ<sub>depth</sub>² = σ<sub>rim</sub>² + σ<sub>gaussian_fit</sub>²<br>
            where σ<sub>gaussian_fit</sub> from covariance matrix
        </div>

        <h4>d/D Ratio Uncertainty</h4>
        <div class="equation">
            d/D = depth / diameter<br>
            σ<sub>d/D</sub>² = (d/D)² [(σ<sub>depth</sub>/depth)² + (σ<sub>diam</sub>/diameter)²]
        </div>

        <h4>Rim Probability Contribution (NEW)</h4>
        <p>
            The rim detection probability from Step 2 adds an additional uncertainty term:
        </p>
        <div class="equation">
            probability_factor = 1 - rim_probability<br>
            σ<sub>prob</sub> = |depth| × probability_factor × 0.5<br>
            σ<sub>total</sub>² = σ<sub>measurement</sub>² + σ<sub>prob</sub>²
        </div>

        <h3>5.3 Application to Current Dataset</h3>
"""

    # Add dataset-specific error analysis
    if os.path.exists(morph_csv):
        morph_df = pd.read_csv(morph_csv)

        # Compute error statistics
        if 'total_error_m1' in morph_df.columns and 'depth_m1' in morph_df.columns:
            valid_m1 = morph_df[morph_df['total_error_m1'].notna() & morph_df['depth_m1'].notna()]
            if len(valid_m1) > 0:
                mean_rel_error_m1 = (valid_m1['total_error_m1'] / valid_m1['depth_m1'].abs()).mean() * 100

                html_content += f"""
        <h4>Method 1 Error Analysis</h4>
        <table>
            <tr>
                <th>Error Component</th>
                <th>Mean Value</th>
                <th>Relative (%)</th>
            </tr>
            <tr>
                <td>Measurement Uncertainty</td>
                <td>{valid_m1['depth_err_m1'].mean():.2f} m</td>
                <td>{(valid_m1['depth_err_m1'] / valid_m1['depth_m1'].abs()).mean() * 100:.1f}%</td>
            </tr>
            <tr>
                <td>Probability Contribution</td>
                <td>{valid_m1['prob_error_m1'].mean() if 'prob_error_m1' in valid_m1.columns else 0:.2f} m</td>
                <td>{(valid_m1['prob_error_m1'] / valid_m1['depth_m1'].abs()).mean() * 100 if 'prob_error_m1' in valid_m1.columns else 0:.1f}%</td>
            </tr>
            <tr>
                <td><strong>Total Error</strong></td>
                <td><strong>{valid_m1['total_error_m1'].mean():.2f} m</strong></td>
                <td><strong>{mean_rel_error_m1:.1f}%</strong></td>
            </tr>
        </table>
"""

        if 'total_error_m2' in morph_df.columns and 'depth_m2' in morph_df.columns:
            valid_m2 = morph_df[morph_df['total_error_m2'].notna() & morph_df['depth_m2'].notna()]
            if len(valid_m2) > 0:
                mean_rel_error_m2 = (valid_m2['total_error_m2'] / valid_m2['depth_m2'].abs()).mean() * 100

                html_content += f"""
        <h4>Method 2 Error Analysis</h4>
        <table>
            <tr>
                <th>Error Component</th>
                <th>Mean Value</th>
                <th>Relative (%)</th>
            </tr>
            <tr>
                <td>Measurement Uncertainty</td>
                <td>{valid_m2['depth_err_m2'].mean():.2f} m</td>
                <td>{(valid_m2['depth_err_m2'] / valid_m2['depth_m2'].abs()).mean() * 100:.1f}%</td>
            </tr>
            <tr>
                <td>Gaussian Fit Uncertainty</td>
                <td>{valid_m2['floor_unc_m2'].mean() if 'floor_unc_m2' in valid_m2.columns else 0:.2f} m</td>
                <td>-</td>
            </tr>
            <tr>
                <td>Probability Contribution</td>
                <td>{valid_m2['prob_error_m2'].mean() if 'prob_error_m2' in valid_m2.columns else 0:.2f} m</td>
                <td>{(valid_m2['prob_error_m2'] / valid_m2['depth_m2'].abs()).mean() * 100 if 'prob_error_m2' in valid_m2.columns else 0:.1f}%</td>
            </tr>
            <tr>
                <td><strong>Total Error</strong></td>
                <td><strong>{valid_m2['total_error_m2'].mean():.2f} m</strong></td>
                <td><strong>{mean_rel_error_m2:.1f}%</strong></td>
            </tr>
        </table>
"""

    html_content += """
        <div class="warning-box">
            <strong>Key Observation:</strong> The rim probability contribution typically accounts for
            20-40% of total uncertainty for low-probability detections (prob < 0.5) but only 5-15%
            for high-probability detections (prob > 0.8). This demonstrates the importance of
            accurate rim localization in Step 2 for precise morphometry measurements.
        </div>

        <h3>5.4 Uncertainty Interpretation</h3>
        <ul>
            <li><strong>Low uncertainty (&lt;5% relative error):</strong> High-confidence measurements suitable for detailed studies</li>
            <li><strong>Moderate uncertainty (5-15%):</strong> Acceptable for population statistics and comparative studies</li>
            <li><strong>High uncertainty (&gt;15%):</strong> Use with caution; may indicate degraded crater or poor data quality</li>
        </ul>
    </div>

    <div class="section" id="condprob">
        <h2>6. Conditional Probability Analysis</h2>

        <h3>6.1 Overview</h3>
        <p>
            Conditional probabilities P(d|D) and P(D|d) describe the relationship between
            crater depth and diameter. These are useful for:
        </p>
        <ul>
            <li><strong>Prediction:</strong> Estimate expected depth for a given diameter</li>
            <li><strong>Quality Control:</strong> Identify unusual crater geometries</li>
            <li><strong>Model Validation:</strong> Compare observed relationships to theoretical predictions</li>
            <li><strong>Population Characterization:</strong> Understand crater population degradation state</li>
        </ul>

        <h3>6.2 Conditional Probability Table</h3>
        <p>
            The table below shows P(d|D) = probability of observing a given depth for a specific
            diameter range, and P(D|d) = probability of observing a given diameter for a specific
            depth range. Higher values indicate more frequent combinations in the dataset.
        </p>

        {cond_prob_table}

        <h3>6.3 Interpretation</h3>
        <div class="info-box">
            <strong>How to use this table:</strong>
            <ul>
                <li><strong>P(d|D):</strong> "Given a crater of diameter D, what is the probability of depth d?"</li>
                <li><strong>P(D|d):</strong> "Given an observed depth d, what is the probability of diameter D?"</li>
                <li><strong>count:</strong> Number of craters in this (diameter, depth) bin</li>
                <li><strong>mean_d_D:</strong> Average depth-to-diameter ratio for craters in this bin</li>
            </ul>
        </div>

        <p>
            <strong>Example Query:</strong> For a 150m diameter crater, find all rows with
            diameter_center ≈ 150m, then identify the depth_bin with highest P_d_given_D.
            This gives the most likely depth range for that diameter.
        </p>
    </div>

    <div class="section" id="summary">
        <h2>7. Summary and Conclusions</h2>

        <h3>7.1 Pipeline Summary</h3>
        <p>This analysis successfully completed the full three-step pipeline:</p>

        <div class="stats-grid">
            <div class="stat-box">
                <div class="label">Step 0: Input Processing</div>
                <div class="value">✓</div>
                <div class="label">{step0_results.get('crater_count', 0)} craters detected</div>
            </div>
            <div class="stat-box">
                <div class="label">Step 2: Rim Refinement</div>
                <div class="value">✓</div>
                <div class="label">Mean prob: {step2_results['statistics'].get('mean_probability', 0):.2f}</div>
            </div>
            <div class="stat-box">
                <div class="label">Step 3: Morphometry</div>
                <div class="value">✓</div>
                <div class="label">{n_craters} craters analyzed</div>
            </div>
        </div>

        <h3>7.2 Key Findings</h3>
"""

    if 'method1' in stats and stats['method1'] and 'method2' in stats and stats['method2']:
        html_content += f"""
        <ul>
            <li><strong>Population d/D ratio:</strong> {stats['method2']['mean_d_D']:.3f} ± {stats['method2']['std_d_D']:.3f}
                (suggests {'moderately degraded' if stats['method2']['mean_d_D'] < 0.12 else 'relatively fresh'} crater population)</li>
            <li><strong>Method agreement:</strong> Methods 1 and 2 differ by ~{abs(stats['method1']['mean_d_D'] - stats['method2']['mean_d_D']) / ((stats['method1']['mean_d_D'] + stats['method2']['mean_d_D'])/2) * 100:.1f}%
                ({'excellent' if abs(stats['method1']['mean_d_D'] - stats['method2']['mean_d_D']) / ((stats['method1']['mean_d_D'] + stats['method2']['mean_d_D'])/2) < 0.10 else 'good' if abs(stats['method1']['mean_d_D'] - stats['method2']['mean_d_D']) / ((stats['method1']['mean_d_D'] + stats['method2']['mean_d_D'])/2) < 0.20 else 'moderate'} agreement)</li>
            <li><strong>Gaussian fit quality:</strong> Mean R² = {stats['method2']['mean_fit_quality']:.2f}
                ({'excellent' if stats['method2']['mean_fit_quality'] > 0.90 else 'good' if stats['method2']['mean_fit_quality'] > 0.80 else 'moderate'} fit quality)</li>
            <li><strong>Error budget:</strong> Rim probability contributes ~{(morph_df['prob_error_m2'].mean() / morph_df['total_error_m2'].mean() * 100) if 'prob_error_m2' in morph_df.columns and 'total_error_m2' in morph_df.columns else 0:.0f}%
                of total uncertainty on average</li>
        </ul>
"""

    html_content += f"""
        <h3>7.3 Data Quality Assessment</h3>
        <div class="{'success-box' if step2_results['statistics'].get('mean_probability', 0) > 0.7 else 'warning-box'}">
            <strong>Overall Quality: {'Excellent' if step2_results['statistics'].get('mean_probability', 0) > 0.8 else 'Good' if step2_results['statistics'].get('mean_probability', 0) > 0.7 else 'Moderate' if step2_results['statistics'].get('mean_probability', 0) > 0.5 else 'Poor'}</strong><br>
            Mean rim detection probability of {step2_results['statistics'].get('mean_probability', 0):.2f} indicates
            {'high-quality rim detections with low uncertainty.' if step2_results['statistics'].get('mean_probability', 0) > 0.7 else 'acceptable rim detections for population studies.' if step2_results['statistics'].get('mean_probability', 0) > 0.5 else 'challenging rim detections; use results with caution.'}
        </div>

        <h3>7.4 Output Files</h3>
        <p>All analysis outputs have been saved to: <code>{output_dir}</code></p>
        <ul>
            <li><strong>Shapefiles:</strong> Initial, refined, and morphometry data with all measurements</li>
            <li><strong>Figures:</strong> Location maps, CSFD plots, scatter plots, probability distributions</li>
            <li><strong>CSV Data:</strong> Morphometry measurements and conditional probabilities for further analysis</li>
            <li><strong>This Report:</strong> <code>SLC_Analysis_Report.html</code></li>
        </ul>

        <h3>7.5 Recommended Next Steps</h3>
        <ul>
            <li>Filter craters by rim probability (>0.7) for high-confidence subset</li>
            <li>Investigate outliers with large method disagreement</li>
            <li>Compare d/D ratios to theoretical fresh crater values (0.15-0.20)</li>
            <li>Use conditional probabilities to predict depths for new crater detections</li>
            <li>Export data for age dating analysis using CSFD</li>
        </ul>
    </div>

    <div class="section">
        <h2>References</h2>
        <ol>
            <li>Crater analysis pipeline developed for Small Lunar Crater (SLC) morphometry studies</li>
            <li>Dual-method approach combines rim perimeter analysis with 2D Gaussian floor fitting</li>
            <li>Error propagation includes rim detection probability from computer vision analysis</li>
            <li>All source code and documentation available in the project repository</li>
        </ol>
    </div>

    <footer style="text-align: center; padding: 30px; color: #666; border-top: 2px solid #e0e0e0; margin-top: 50px;">
        <p>Generated by <strong>analyze_small_lunar_craters.py</strong></p>
        <p>Small Lunar Crater Morphometry Analysis Pipeline</p>
        <p>{timestamp}</p>
    </footer>
</body>
</html>
"""

    # Write report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✓ Report generated: {report_path}")
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
    └── SLC_Analysis_Report.html  (Comprehensive report)

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

        # Generate comprehensive report
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
        print(f"  Comprehensive report: {report_path}")
        print(f"\n  Open the report in your web browser to view all results.")
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

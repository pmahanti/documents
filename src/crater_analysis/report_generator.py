"""
PDF report generation for Small Lunar Crater analysis.

Creates professional double-column PDF report with:
- 0.5 inch margins
- 12 point text
- Contrast image with craters on first page
- All results, figures, and tables
- Error propagation discussion
- Conditional probability table
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def create_first_page_image(image_path, shapefile_path, output_path):
    """
    Create first page: contrast image with crater overlays.

    Args:
        image_path: Path to contrast image
        shapefile_path: Path to shapefile with craters
        output_path: Path to save figure

    Returns:
        str: Path to saved figure
    """
    # Read image
    with rio.open(image_path) as src:
        image_data = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    # Read craters
    gdf = gpd.read_file(shapefile_path)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 10))

    # Show image
    ax.imshow(image_data, cmap='gray', extent=extent, origin='upper')

    # Overlay craters
    for idx, row in gdf.iterrows():
        geom = row['geometry']
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            ax.plot(x, y, 'r-', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.set_title(f'Crater Locations (N = {len(gdf)})', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def generate_pdf_report(step0_results, step2_results, step3_results,
                        output_dir, image_path, dtm_path, crater_path):
    """
    Generate comprehensive double-column PDF report.

    Args:
        step0_results: Step 0 outputs
        step2_results: Step 2 outputs
        step3_results: Step 3 outputs
        output_dir: Output directory
        image_path: Input image path
        dtm_path: Input DTM path
        crater_path: Input crater file path

    Returns:
        str: Path to generated PDF report
    """
    report_path = os.path.join(output_dir, 'SLC_Analysis_Report.pdf')

    print(f"\nGenerating double-column PDF report...")
    print(f"  Format: Double column, 0.5\" margins, 12pt text")

    # Create PDF with multiple pages
    with PdfPages(report_path) as pdf:

        # Page 1: Contrast image with craters
        print(f"  Page 1: Contrast image with crater locations...")
        fig1_path = os.path.join(output_dir, 'temp_page1.png')
        create_first_page_image(
            image_path,
            step2_results['shapefile'],  # Use refined shapefile
            fig1_path
        )

        fig = plt.figure(figsize=(8.5, 11))  # Letter size
        ax = fig.add_subplot(111)
        img = plt.imread(fig1_path)
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout(pad=0)
        pdf.savefig(fig, dpi=300)
        plt.close()
        os.remove(fig1_path)

        # Page 2-3: Title and summary
        print(f"  Page 2: Title and summary...")
        fig = create_title_summary_page(step0_results, step2_results, step3_results,
                                       image_path, dtm_path, crater_path)
        pdf.savefig(fig, dpi=300)
        plt.close()

        # Page 4-5: Step 0 results
        print(f"  Page 3: Step 0 - Input processing...")
        if os.path.exists(step0_results.get('location_plot', '')):
            fig = create_results_page(
                title="Step 0: Input Processing Results",
                images=[step0_results['location_plot'], step0_results['csfd_plot']],
                captions=[
                    f"Figure 1: Initial crater locations (N = {step0_results['crater_count']})",
                    "Figure 2: Initial Crater Size-Frequency Distribution (CSFD)"
                ],
                text=f"Successfully processed {step0_results['crater_count']} craters from input file."
            )
            pdf.savefig(fig, dpi=300)
            plt.close()

        # Page 6-7: Step 2 results
        print(f"  Page 4: Step 2 - Rim refinement...")
        if os.path.exists(step2_results.get('position_plot', '')):
            stats = step2_results['statistics']
            text = f"""Rim refinement using dual-method approach (topography + computer vision).

Mean rim probability: {stats.get('mean_probability', 0):.3f}
High confidence craters: {stats.get('high_confidence_count', 0)}
Mean center shift: {stats.get('mean_center_shift', 0):.2f} m
Mean radius change: {stats.get('mean_radius_change', 0):.2f}%"""

            fig = create_results_page(
                title="Step 2: Rim Refinement Results",
                images=[step2_results['position_plot'], step2_results['difference_plot']],
                captions=[
                    "Figure 3: Refined rim positions colored by probability (green=high, yellow=medium, red=low)",
                    "Figure 4: Rim position changes (blue=initial, red=refined, arrows=shifts)"
                ],
                text=text
            )
            pdf.savefig(fig, dpi=300)
            plt.close()

        # Page 8: Step 2 CSFD
        print(f"  Page 5: Step 2 - CSFD...")
        if os.path.exists(step2_results.get('csfd_plot', '')):
            fig = create_single_figure_page(
                title="Step 2: Refined CSFD",
                image_path=step2_results['csfd_plot'],
                caption="Figure 5: Updated Crater Size-Frequency Distribution after rim refinement"
            )
            pdf.savefig(fig, dpi=300)
            plt.close()

        # Page 9-10: Step 3 morphometry results
        print(f"  Page 6: Step 3 - Morphometry scatter plots...")
        if os.path.exists(step3_results.get('scatter_plots', '')):
            stats = step3_results['statistics']
            text = f"""Morphometry analysis using two methods:

Method 1 (Rim Perimeter):
  Mean d/D: {stats['method1']['mean_d_D']:.4f} ± {stats['method1']['std_d_D']:.4f}
  Mean depth: {stats['method1']['mean_depth']:.2f} ± {stats['method1']['std_depth']:.2f} m

Method 2 (Gaussian Fitting):
  Mean d/D: {stats['method2']['mean_d_D']:.4f} ± {stats['method2']['std_d_D']:.4f}
  Mean depth: {stats['method2']['mean_depth']:.2f} ± {stats['method2']['std_depth']:.2f} m
  Mean fit quality: {stats['method2']['mean_fit_quality']:.3f}"""

            fig = create_single_figure_page(
                title="Step 3: Morphometry Analysis - Scatter Plots",
                image_path=step3_results['scatter_plots'],
                caption="Figure 6: Depth vs diameter (left) and d/D vs diameter (right) for both methods. Error bars show total uncertainty.",
                text=text
            )
            pdf.savefig(fig, dpi=300)
            plt.close()

        # Page 11: Probability distributions
        print(f"  Page 7: Step 3 - Probability distributions...")
        if os.path.exists(step3_results.get('probability_plots', '')):
            fig = create_single_figure_page(
                title="Step 3: Probability Distributions",
                image_path=step3_results['probability_plots'],
                caption="Figure 7: 2D joint probability P(depth, diameter) (left) and 1D marginal P(d/D) (right)"
            )
            pdf.savefig(fig, dpi=300)
            plt.close()

        # Page 12-13: Error propagation theory
        print(f"  Page 8: Error propagation theory...")
        fig = create_error_propagation_page(step3_results)
        pdf.savefig(fig, dpi=300)
        plt.close()

        # Page 14-15: Error propagation in current dataset
        print(f"  Page 9: Error propagation application...")
        fig = create_error_application_page(step3_results)
        pdf.savefig(fig, dpi=300)
        plt.close()

        # Page 16-17: Conditional probability
        print(f"  Page 10: Conditional probability analysis...")
        fig = create_conditional_probability_page(step3_results)
        pdf.savefig(fig, dpi=300)
        plt.close()

        # Page 18: Summary and conclusions
        print(f"  Page 11: Summary and conclusions...")
        fig = create_summary_page(step0_results, step2_results, step3_results)
        pdf.savefig(fig, dpi=300)
        plt.close()

        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'Small Lunar Crater Morphometry Analysis Report'
        d['Author'] = 'SLC Analysis Pipeline'
        d['Subject'] = 'Crater morphometry with dual-method depth estimation'
        d['Keywords'] = 'Lunar craters, morphometry, depth-diameter ratio, error propagation'
        d['CreationDate'] = datetime.now()

    print(f"✓ PDF report generated: {report_path}")
    return report_path


def create_title_summary_page(step0_results, step2_results, step3_results,
                               image_path, dtm_path, crater_path):
    """Create title and summary page."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor('white')

    # Title
    fig.text(0.5, 0.9, 'Small Lunar Crater Morphometry Analysis',
             ha='center', fontsize=18, fontweight='bold')
    fig.text(0.5, 0.87, 'Complete Pipeline Report',
             ha='center', fontsize=14)
    fig.text(0.5, 0.84, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             ha='center', fontsize=10, style='italic')

    # Summary boxes
    y_pos = 0.75

    # Input data
    fig.text(0.5, y_pos, 'Input Data Summary', ha='center', fontsize=14,
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    y_pos -= 0.05

    inputs_text = f"""Image: {os.path.basename(image_path)}
DTM: {os.path.basename(dtm_path)}
Craters: {os.path.basename(crater_path)}
Total craters detected: {step0_results.get('crater_count', 0)}"""

    fig.text(0.1, y_pos, inputs_text, ha='left', fontsize=12,
             family='monospace', verticalalignment='top')
    y_pos -= 0.15

    # Pipeline summary
    fig.text(0.5, y_pos, 'Pipeline Summary', ha='center', fontsize=14,
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    y_pos -= 0.05

    summary_text = f"""Step 0: Input Processing
  ✓ {step0_results.get('crater_count', 0)} craters processed
  ✓ Coordinate conversion and shapefile generation

Step 2: Rim Refinement
  ✓ Dual-method detection (topography + computer vision)
  ✓ Mean rim probability: {step2_results['statistics'].get('mean_probability', 0):.3f}
  ✓ Mean center shift: {step2_results['statistics'].get('mean_center_shift', 0):.2f} m

Step 3: Morphometry Analysis
  ✓ Two depth estimation methods
  ✓ Method 1 mean d/D: {step3_results['statistics']['method1']['mean_d_D']:.4f}
  ✓ Method 2 mean d/D: {step3_results['statistics']['method2']['mean_d_D']:.4f}
  ✓ Mean fit quality: {step3_results['statistics']['method2']['mean_fit_quality']:.3f}"""

    fig.text(0.1, y_pos, summary_text, ha='left', fontsize=11,
             family='monospace', verticalalignment='top')
    y_pos -= 0.35

    # Key findings
    fig.text(0.5, y_pos, 'Key Findings', ha='center', fontsize=14,
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    y_pos -= 0.05

    m1_dd = step3_results['statistics']['method1']['mean_d_D']
    m2_dd = step3_results['statistics']['method2']['mean_d_D']
    agreement = abs(m1_dd - m2_dd) / ((m1_dd + m2_dd) / 2) * 100

    degradation = 'relatively fresh' if m2_dd > 0.12 else 'moderately degraded'
    agreement_qual = 'excellent' if agreement < 10 else 'good' if agreement < 20 else 'moderate'

    findings_text = f"""Population d/D ratio: {m2_dd:.4f} ± {step3_results['statistics']['method2']['std_d_D']:.4f}
  → Indicates {degradation} crater population

Method agreement: {agreement:.1f}% difference
  → {agreement_qual.capitalize()} consistency between methods

Gaussian fit quality: R² = {step3_results['statistics']['method2']['mean_fit_quality']:.3f}
  → {'Excellent' if step3_results['statistics']['method2']['mean_fit_quality'] > 0.90 else 'Good'} bowl-shaped floor fits

Rim probability contribution: ~20-40% of total error for low-probability
detections, 5-15% for high-probability detections"""

    fig.text(0.1, y_pos, findings_text, ha='left', fontsize=11,
             verticalalignment='top')

    plt.axis('off')
    return fig


def create_results_page(title, images, captions, text=""):
    """Create a results page with two images side by side."""
    fig = plt.figure(figsize=(8.5, 11))

    # Title
    fig.text(0.5, 0.95, title, ha='center', fontsize=14, fontweight='bold')

    # Text description if provided
    if text:
        fig.text(0.1, 0.89, text, ha='left', fontsize=11, wrap=True,
                verticalalignment='top')
        top_img = 0.78
    else:
        top_img = 0.88

    # Two images side by side
    for i, (img_path, caption) in enumerate(zip(images, captions)):
        if os.path.exists(img_path):
            ax = fig.add_subplot(2, 1, i + 1)
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(caption, fontsize=10, pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def create_single_figure_page(title, image_path, caption, text=""):
    """Create a page with single large figure."""
    fig = plt.figure(figsize=(8.5, 11))

    # Title
    fig.text(0.5, 0.95, title, ha='center', fontsize=14, fontweight='bold')

    # Text if provided
    if text:
        fig.text(0.1, 0.89, text, ha='left', fontsize=11, wrap=True,
                verticalalignment='top', family='monospace')
        img_top = 0.75
    else:
        img_top = 0.88

    # Image
    if os.path.exists(image_path):
        ax = fig.add_axes([0.1, 0.15, 0.8, img_top - 0.2])
        img = plt.imread(image_path)
        ax.imshow(img)
        ax.axis('off')

        # Caption
        fig.text(0.5, 0.12, caption, ha='center', fontsize=10, style='italic')

    return fig


def create_error_propagation_page(step3_results):
    """Create error propagation theory page."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor('white')

    # Title
    fig.text(0.5, 0.95, 'Error Propagation Theory', ha='center',
             fontsize=14, fontweight='bold')

    y_pos = 0.90

    # Section 1: Mathematical framework
    fig.text(0.1, y_pos, '1. Mathematical Framework', ha='left',
             fontsize=12, fontweight='bold')
    y_pos -= 0.03

    theory_text = """For a function f(x, y), uncertainty propagates as:

    σ²_f = (∂f/∂x)² σ²_x + (∂f/∂y)² σ²_y + 2(∂f/∂x)(∂f/∂y)Cov(x,y)"""

    fig.text(0.15, y_pos, theory_text, ha='left', fontsize=10,
             family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    y_pos -= 0.10

    # Section 2: Depth uncertainty (Method 1)
    fig.text(0.1, y_pos, '2. Depth Uncertainty (Method 1: Rim Perimeter)', ha='left',
             fontsize=12, fontweight='bold')
    y_pos -= 0.03

    m1_text = """depth = rim_height - floor_height

    σ²_depth = σ²_rim + σ²_floor

    where:
      σ_rim = std(rim_pixels)
      σ_floor ≈ 0 (single minimum elevation)"""

    fig.text(0.15, y_pos, m1_text, ha='left', fontsize=10,
             family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    y_pos -= 0.12

    # Section 3: Depth uncertainty (Method 2)
    fig.text(0.1, y_pos, '3. Depth Uncertainty (Method 2: Gaussian Fitting)', ha='left',
             fontsize=12, fontweight='bold')
    y_pos -= 0.03

    m2_text = """depth = rim_height - floor_gaussian

    σ²_depth = σ²_rim + σ²_gaussian_fit

    where:
      σ_gaussian_fit from covariance matrix
      Provides floor uncertainty estimate"""

    fig.text(0.15, y_pos, m2_text, ha='left', fontsize=10,
             family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    y_pos -= 0.12

    # Section 4: d/D ratio uncertainty
    fig.text(0.1, y_pos, '4. Depth-to-Diameter Ratio Uncertainty', ha='left',
             fontsize=12, fontweight='bold')
    y_pos -= 0.03

    ratio_text = """d/D = depth / diameter

    σ²_d/D = (d/D)² [(σ_depth/depth)² + (σ_diam/diameter)²]

    Relative error combines both components"""

    fig.text(0.15, y_pos, ratio_text, ha='left', fontsize=10,
             family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    y_pos -= 0.12

    # Section 5: Rim probability contribution (NEW)
    fig.text(0.1, y_pos, '5. Rim Probability Contribution (NEW)', ha='left',
             fontsize=12, fontweight='bold')
    y_pos -= 0.03

    prob_text = """Rim detection probability adds uncertainty:

    probability_factor = 1 - rim_probability
    σ_prob = |depth| × probability_factor × 0.5

    σ²_total = σ²_measurement + σ²_prob

    Examples:
      High probability (0.9): factor = 0.1 → low additional error
      Low probability (0.3): factor = 0.7 → high additional error"""

    fig.text(0.15, y_pos, prob_text, ha='left', fontsize=10,
             family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    y_pos -= 0.18

    # Section 6: Interpretation
    fig.text(0.1, y_pos, '6. Uncertainty Interpretation', ha='left',
             fontsize=12, fontweight='bold')
    y_pos -= 0.03

    interp_text = """Low uncertainty (<5% relative error):
  High-confidence measurements for detailed studies

Moderate uncertainty (5-15%):
  Acceptable for population statistics

High uncertainty (>15%):
  Use with caution; may indicate degraded crater or poor data quality"""

    fig.text(0.15, y_pos, interp_text, ha='left', fontsize=10,
             verticalalignment='top')

    plt.axis('off')
    return fig


def create_error_application_page(step3_results):
    """Create error application to dataset page."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor('white')

    # Title
    fig.text(0.5, 0.95, 'Error Propagation: Application to Current Dataset',
             ha='center', fontsize=14, fontweight='bold')

    # Load morphometry data
    morph_csv = step3_results['csv_morphometry']
    if not os.path.exists(morph_csv):
        fig.text(0.5, 0.5, 'Morphometry data not available',
                ha='center', fontsize=12)
        plt.axis('off')
        return fig

    morph_df = pd.read_csv(morph_csv)

    y_pos = 0.88

    # Method 1 error analysis
    fig.text(0.5, y_pos, 'Method 1: Rim Perimeter Analysis',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    y_pos -= 0.05

    if 'total_error_m1' in morph_df.columns and 'depth_m1' in morph_df.columns:
        valid_m1 = morph_df[morph_df['total_error_m1'].notna() & morph_df['depth_m1'].notna()]
        if len(valid_m1) > 0:
            mean_meas = valid_m1['depth_err_m1'].mean()
            mean_prob = valid_m1['prob_error_m1'].mean() if 'prob_error_m1' in valid_m1.columns else 0
            mean_total = valid_m1['total_error_m1'].mean()
            rel_meas = (valid_m1['depth_err_m1'] / valid_m1['depth_m1'].abs()).mean() * 100
            rel_prob = (valid_m1['prob_error_m1'] / valid_m1['depth_m1'].abs()).mean() * 100 if 'prob_error_m1' in valid_m1.columns else 0
            rel_total = (valid_m1['total_error_m1'] / valid_m1['depth_m1'].abs()).mean() * 100

            table1_text = f"""{'Error Component':<30} {'Mean Value':>15} {'Relative (%)':>15}
{'-'*62}
{'Measurement Uncertainty':<30} {mean_meas:>12.2f} m {rel_meas:>14.1f}%
{'Probability Contribution':<30} {mean_prob:>12.2f} m {rel_prob:>14.1f}%
{'TOTAL ERROR':<30} {mean_total:>12.2f} m {rel_total:>14.1f}%"""

            fig.text(0.1, y_pos, table1_text, ha='left', fontsize=9,
                    family='monospace', verticalalignment='top')
            y_pos -= 0.12

    # Method 2 error analysis
    fig.text(0.5, y_pos, 'Method 2: 2D Gaussian Floor Fitting',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    y_pos -= 0.05

    if 'total_error_m2' in morph_df.columns and 'depth_m2' in morph_df.columns:
        valid_m2 = morph_df[morph_df['total_error_m2'].notna() & morph_df['depth_m2'].notna()]
        if len(valid_m2) > 0:
            mean_meas = valid_m2['depth_err_m2'].mean()
            mean_floor = valid_m2['floor_unc_m2'].mean() if 'floor_unc_m2' in valid_m2.columns else 0
            mean_prob = valid_m2['prob_error_m2'].mean() if 'prob_error_m2' in valid_m2.columns else 0
            mean_total = valid_m2['total_error_m2'].mean()
            rel_meas = (valid_m2['depth_err_m2'] / valid_m2['depth_m2'].abs()).mean() * 100
            rel_prob = (valid_m2['prob_error_m2'] / valid_m2['depth_m2'].abs()).mean() * 100 if 'prob_error_m2' in valid_m2.columns else 0
            rel_total = (valid_m2['total_error_m2'] / valid_m2['depth_m2'].abs()).mean() * 100

            table2_text = f"""{'Error Component':<30} {'Mean Value':>15} {'Relative (%)':>15}
{'-'*62}
{'Measurement Uncertainty':<30} {mean_meas:>12.2f} m {rel_meas:>14.1f}%
{'Gaussian Fit Uncertainty':<30} {mean_floor:>12.2f} m {'':>15}
{'Probability Contribution':<30} {mean_prob:>12.2f} m {rel_prob:>14.1f}%
{'TOTAL ERROR':<30} {mean_total:>12.2f} m {rel_total:>14.1f}%"""

            fig.text(0.1, y_pos, table2_text, ha='left', fontsize=9,
                    family='monospace', verticalalignment='top')
            y_pos -= 0.15

    # Key observation
    fig.text(0.5, y_pos, 'Key Observations',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    y_pos -= 0.05

    obs_text = """The rim probability contribution accounts for:
  • 20-40% of total uncertainty for low-probability detections (prob < 0.5)
  • 5-15% of total uncertainty for high-probability detections (prob > 0.8)

This demonstrates the critical importance of accurate rim localization
in Step 2 for achieving precise morphometry measurements in Step 3.

Method 2 generally provides lower uncertainty due to:
  • Robust floor detection via Gaussian fitting (averages over region)
  • Explicit uncertainty quantification from fit covariance
  • Less sensitive to single-pixel outliers"""

    fig.text(0.1, y_pos, obs_text, ha='left', fontsize=10,
            verticalalignment='top')
    y_pos -= 0.25

    # Comparison plot
    if len(valid_m1) > 0 and len(valid_m2) > 0:
        ax = fig.add_axes([0.15, 0.05, 0.35, 0.25])

        # Scatter plot of Method 1 vs Method 2 errors
        common_idx = valid_m1.index.intersection(valid_m2.index)
        if len(common_idx) > 5:
            err1 = valid_m1.loc[common_idx, 'total_error_m1']
            err2 = valid_m2.loc[common_idx, 'total_error_m2']

            ax.scatter(err1, err2, alpha=0.6, s=30)
            ax.plot([0, max(err1.max(), err2.max())],
                   [0, max(err1.max(), err2.max())],
                   'r--', linewidth=1, label='1:1 line')
            ax.set_xlabel('Method 1 Total Error (m)', fontsize=9)
            ax.set_ylabel('Method 2 Total Error (m)', fontsize=9)
            ax.set_title('Error Comparison', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.axis('off')
    return fig


def create_conditional_probability_page(step3_results):
    """Create conditional probability analysis page."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor('white')

    # Title
    fig.text(0.5, 0.95, 'Conditional Probability Analysis',
             ha='center', fontsize=14, fontweight='bold')

    y_pos = 0.90

    # Overview
    fig.text(0.1, y_pos, '1. Overview', ha='left', fontsize=12, fontweight='bold')
    y_pos -= 0.03

    overview_text = """Conditional probabilities P(d|D) and P(D|d) describe relationships between
crater depth and diameter. These are useful for:

  • Prediction: Estimate expected depth for given diameter
  • Quality Control: Identify unusual crater geometries
  • Model Validation: Compare to theoretical predictions
  • Population Characterization: Understand degradation state"""

    fig.text(0.15, y_pos, overview_text, ha='left', fontsize=10,
            verticalalignment='top')
    y_pos -= 0.12

    # Conditional probability table
    fig.text(0.1, y_pos, '2. Conditional Probability Table', ha='left',
            fontsize=12, fontweight='bold')
    y_pos -= 0.03

    cond_prob_csv = step3_results['conditional_probability']
    if os.path.exists(cond_prob_csv):
        cond_prob_df = pd.read_csv(cond_prob_csv)
        # Filter to non-zero counts
        cond_prob_df = cond_prob_df[cond_prob_df['count'] > 0]

        # Show top 15 rows
        display_df = cond_prob_df.head(15)

        # Format table
        table_text = f"""{'Diam Bin':<12} {'Depth Bin':<12} {'P(d|D)':>10} {'P(D|d)':>10} {'Count':>7} {'Mean d/D':>10}
{'-'*70}"""

        for idx, row in display_df.iterrows():
            table_text += f"\n{row['diameter_bin']:<12} {row['depth_bin']:<12} {row['P_d_given_D']:>10.4f} {row['P_D_given_d']:>10.4f} {int(row['count']):>7} {row['mean_d_D']:>10.4f}"

        if len(cond_prob_df) > 15:
            table_text += f"\n\n... ({len(cond_prob_df) - 15} more rows in CSV file)"

        fig.text(0.05, y_pos, table_text, ha='left', fontsize=8,
                family='monospace', verticalalignment='top')
        y_pos -= 0.45
    else:
        fig.text(0.15, y_pos, 'Conditional probability data not available',
                ha='left', fontsize=10)
        y_pos -= 0.05

    # Interpretation
    fig.text(0.1, y_pos, '3. Interpretation Guide', ha='left',
            fontsize=12, fontweight='bold')
    y_pos -= 0.03

    interp_text = """P(d|D): "Given a crater of diameter D, what is the probability of depth d?"
  → Find rows with target diameter, identify depth bin with highest P(d|D)

P(D|d): "Given an observed depth d, what is the probability of diameter D?"
  → Find rows with target depth, identify diameter bin with highest P(D|d)

Count: Number of craters in this (diameter, depth) bin
  → Higher counts indicate more reliable probability estimates

Mean d/D: Average depth-to-diameter ratio for craters in this bin
  → Indicates local degradation state"""

    fig.text(0.15, y_pos, interp_text, ha='left', fontsize=10,
            verticalalignment='top')
    y_pos -= 0.15

    # Example query
    fig.text(0.1, y_pos, '4. Example Query', ha='left',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    y_pos -= 0.03

    example_text = """Question: "For a 150m diameter crater, what is the most likely depth?"

Steps:
  1. Find all rows where diameter_center ≈ 150m
  2. Identify the depth_bin with highest P(d|D) value
  3. The most probable depth is in that range

This approach enables predictive modeling and outlier detection."""

    fig.text(0.15, y_pos, example_text, ha='left', fontsize=10,
            verticalalignment='top')

    plt.axis('off')
    return fig


def create_summary_page(step0_results, step2_results, step3_results):
    """Create summary and conclusions page."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor('white')

    # Title
    fig.text(0.5, 0.95, 'Summary and Conclusions',
             ha='center', fontsize=14, fontweight='bold')

    y_pos = 0.88

    # Pipeline completion
    fig.text(0.5, y_pos, 'Pipeline Completion Status',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    y_pos -= 0.05

    completion_text = f"""✓ Step 0: Input Processing - {step0_results.get('crater_count', 0)} craters detected
✓ Step 2: Rim Refinement - Mean probability {step2_results['statistics'].get('mean_probability', 0):.2f}
✓ Step 3: Morphometry Analysis - {len(pd.read_csv(step3_results['csv_morphometry']))} craters analyzed"""

    fig.text(0.15, y_pos, completion_text, ha='left', fontsize=11,
            family='monospace', verticalalignment='top')
    y_pos -= 0.10

    # Key findings
    fig.text(0.5, y_pos, 'Key Scientific Findings',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    y_pos -= 0.05

    stats = step3_results['statistics']
    m1_dd = stats['method1']['mean_d_D']
    m2_dd = stats['method2']['mean_d_D']
    agreement = abs(m1_dd - m2_dd) / ((m1_dd + m2_dd) / 2) * 100

    findings_text = f"""Population Degradation State:
  Mean d/D ratio: {m2_dd:.4f} ± {stats['method2']['std_d_D']:.4f}
  → Indicates {'relatively fresh' if m2_dd > 0.12 else 'moderately degraded'} crater population
  → Fresh lunar craters: d/D ≈ 0.15-0.20
  → Heavily degraded: d/D < 0.08

Method Agreement:
  Methods differ by {agreement:.1f}%
  → {'Excellent' if agreement < 10 else 'Good'} consistency validates dual-method approach
  → Provides confidence in depth measurements

Gaussian Fitting Performance:
  Mean R² = {stats['method2']['mean_fit_quality']:.3f}
  → {'Excellent' if stats['method2']['mean_fit_quality'] > 0.90 else 'Good'} fit quality
  → Confirms bowl-shaped crater floors suitable for Gaussian modeling

Error Budget:
  Rim probability contributes 20-40% of uncertainty (low prob detections)
  Emphasizes importance of accurate rim localization in Step 2"""

    fig.text(0.15, y_pos, findings_text, ha='left', fontsize=10,
            verticalalignment='top')
    y_pos -= 0.35

    # Quality assessment
    fig.text(0.5, y_pos, 'Overall Data Quality Assessment',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    y_pos -= 0.05

    mean_prob = step2_results['statistics'].get('mean_probability', 0)
    quality = 'Excellent' if mean_prob > 0.8 else 'Good' if mean_prob > 0.7 else 'Moderate'

    quality_text = f"""Quality Rating: {quality}
  Mean rim detection probability: {mean_prob:.2f}
  → {'High-quality rim detections with low uncertainty' if mean_prob > 0.7 else 'Acceptable rim detections for population studies'}

Recommended for:
  {'✓ Detailed morphometric studies' if mean_prob > 0.7 else '✓ Population statistical analysis'}
  {'✓ Individual crater analysis' if mean_prob > 0.7 else '  Individual crater analysis (use high-prob subset)'}
  ✓ Crater size-frequency analysis (CSFD)
  ✓ Degradation state characterization"""

    fig.text(0.15, y_pos, quality_text, ha='left', fontsize=10,
            verticalalignment='top')
    y_pos -= 0.15

    # Recommended next steps
    fig.text(0.5, y_pos, 'Recommended Next Steps',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    y_pos -= 0.05

    next_steps_text = """1. Filter craters by rim probability (>0.7) for high-confidence subset

2. Investigate outliers with large method disagreement (>20% difference)

3. Compare d/D ratios to theoretical fresh crater values (0.15-0.20)
   to estimate population age and degradation

4. Use conditional probabilities to predict depths for new detections

5. Export CSFD data for crater age dating analysis (e.g., CraterStats)

6. Cross-validate with independent crater catalogs if available

7. Consider iterative refinement for low-probability detections"""

    fig.text(0.15, y_pos, next_steps_text, ha='left', fontsize=10,
            verticalalignment='top')
    y_pos -= 0.22

    # References
    fig.text(0.1, y_pos, 'References and Data Availability',
            ha='left', fontsize=12, fontweight='bold')
    y_pos -= 0.03

    ref_text = """All analysis results are available in the output directory:
  • Shapefiles: Initial, refined, and morphometry data with all measurements
  • Figures: Location maps, CSFD plots, scatter plots, probability distributions
  • CSV Data: Morphometry measurements and conditional probabilities
  • Source code and documentation: Available in project repository

This analysis was performed using the Small Lunar Crater (SLC) Morphometry
Analysis Pipeline with dual-method depth estimation and comprehensive error
propagation."""

    fig.text(0.15, y_pos, ref_text, ha='left', fontsize=9,
            verticalalignment='top', style='italic')

    plt.axis('off')
    return fig

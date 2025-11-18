#!/usr/bin/env python3
"""
Crater Synthesis, Degradation, and Chebyshev Analysis Test

This script synthesizes pristine crater profiles, applies topographic degradation
over time, and analyzes the evolution of Chebyshev coefficients with age.

Generates:
- 10 synthetic craters (800m to 5km diameter)
- Degradation at 0.1 to 3.5 Ga intervals
- Chebyshev coefficient evolution
- CSV data export
- PDF visualization report

Author: Crater Analysis Toolkit
Date: 2025-11-18
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.io import savemat
import os

# Import our modules
try:
    from topography_degradation_age import TopographyDegradationAgeEstimator
    from chebyshev_coefficients import extract_chebyshev_coefficients
    MODULES_AVAILABLE = True
except ImportError:
    print("Warning: Required modules not found. Run from repository root.")
    MODULES_AVAILABLE = False


class CraterSynthesizer:
    """Synthesize pristine and degraded crater profiles."""

    def __init__(self, diffusivity=5.0):
        """
        Initialize crater synthesizer.

        Parameters:
        -----------
        diffusivity : float
            Diffusivity coefficient in m²/Myr (default 5.0)
        """
        self.diffusivity = diffusivity

    def generate_pristine_profile(self, diameter, num_points=300):
        """
        Generate a pristine (fresh) crater elevation profile.

        Uses empirical morphology for lunar simple craters:
        - Parabolic bowl interior
        - Exponential rim decay
        - d/D ≈ 0.196 for fresh craters

        Parameters:
        -----------
        diameter : float
            Crater diameter in meters
        num_points : int
            Number of points in profile (default 300)

        Returns:
        --------
        distance : ndarray
            Radial distances from center (-1.5D to +1.5D)
        elevation : ndarray
            Elevation values (meters)
        """
        radius = diameter / 2.0

        # Profile extends from -1.5D to +1.5D
        distance = np.linspace(-1.5 * diameter, 1.5 * diameter, num_points)

        # Pristine depth/diameter ratio for lunar simple craters
        depth = 0.196 * diameter

        # Rim height (typically ~4% of diameter)
        rim_height = 0.04 * diameter

        # Initialize elevation
        elevation = np.zeros(num_points)

        # For each point, determine elevation based on distance
        for i, r in enumerate(distance):
            r_norm = abs(r) / radius  # Normalize by radius

            if r_norm <= 1.0:
                # Interior: parabolic bowl
                elevation[i] = -depth * (1 - r_norm**2)
            elif r_norm <= 1.3:
                # Rim: exponential decay
                elevation[i] = rim_height * np.exp(-5 * (r_norm - 1.0))
            else:
                # Far field: zero elevation
                elevation[i] = 0.0

        return distance, elevation

    def apply_degradation(self, distance, elevation, age_myr, diameter):
        """
        Apply topographic degradation using diffusion model.

        Parameters:
        -----------
        distance : ndarray
            Radial distances
        elevation : ndarray
            Initial elevation
        age_myr : float
            Time since formation in Myr
        diameter : float
            Crater diameter in meters

        Returns:
        --------
        degraded_elevation : ndarray
            Degraded elevation profile
        """
        if age_myr <= 0:
            return elevation.copy()

        # Diffusion smoothing length scale: sigma = sqrt(2 * kappa * t)
        sigma_meters = np.sqrt(2 * self.diffusivity * age_myr)

        # Convert to pixel units
        dr = np.mean(np.diff(distance))
        sigma_pixels = sigma_meters / abs(dr)

        # Apply Gaussian smoothing as approximation of diffusion
        if sigma_pixels > 0:
            degraded = gaussian_filter1d(elevation, sigma_pixels, mode='nearest')
        else:
            degraded = elevation.copy()

        return degraded

    def compute_depth_diameter_ratio(self, distance, elevation, diameter):
        """
        Compute depth-to-diameter ratio from profile.

        Parameters:
        -----------
        distance : ndarray
            Radial distances
        elevation : ndarray
            Elevation values
        diameter : float
            Crater diameter

        Returns:
        --------
        d_D_ratio : float
            Depth-to-diameter ratio
        depth : float
            Crater depth in meters
        """
        radius = diameter / 2.0

        # Find center region (crater floor)
        center_mask = np.abs(distance) < 0.3 * radius
        floor_elev = np.min(elevation[center_mask]) if np.sum(center_mask) > 0 else np.min(elevation)

        # Find rim region
        rim_mask = np.abs(np.abs(distance) - radius) < 0.1 * radius
        rim_elev = np.max(elevation[rim_mask]) if np.sum(rim_mask) > 0 else np.max(elevation)

        depth = rim_elev - floor_elev
        d_D_ratio = depth / diameter if diameter > 0 else 0

        return d_D_ratio, depth


def run_synthesis_degradation_test(output_dir='synthesis_test_results'):
    """
    Run comprehensive synthesis and degradation test.

    Generates 10 craters, applies degradation over time, computes Chebyshev
    coefficients, and creates visualization report.

    Parameters:
    -----------
    output_dir : str
        Output directory for results
    """
    if not MODULES_AVAILABLE:
        print("Error: Required modules not available.")
        return

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("="*60)
    print("CRATER SYNTHESIS, DEGRADATION, AND CHEBYSHEV ANALYSIS TEST")
    print("="*60)

    # Parameters
    num_craters = 10
    diameter_range = (800, 5000)  # meters
    age_range = (0.1, 3.5)  # Ga
    age_step = 0.1  # Ga
    diffusivity = 5.0  # m²/Myr

    # Generate random crater diameters
    np.random.seed(42)  # For reproducibility
    diameters = np.random.uniform(diameter_range[0], diameter_range[1], num_craters)
    diameters = np.sort(diameters)  # Sort for easier visualization

    # Age points (in Ga, convert to Myr for calculations)
    ages_ga = np.arange(age_range[0], age_range[1] + age_step, age_step)
    ages_myr = ages_ga * 1000  # Convert to Myr

    print(f"\nGenerating {num_craters} craters:")
    print(f"  Diameter range: {diameter_range[0]}-{diameter_range[1]} m")
    print(f"  Age range: {age_range[0]}-{age_range[1]} Ga")
    print(f"  Age step: {age_step} Ga")
    print(f"  Number of age points: {len(ages_ga)}")
    print(f"  Diffusivity: {diffusivity} m²/Myr")

    # Initialize synthesizer
    synthesizer = CraterSynthesizer(diffusivity=diffusivity)

    # Storage for results
    all_results = []
    crater_data = {}

    # Process each crater
    for crater_idx, diameter in enumerate(diameters):
        print(f"\n  Crater {crater_idx + 1}/{num_craters}: D = {diameter:.0f} m")

        # Generate pristine profile
        distance, elev_pristine = synthesizer.generate_pristine_profile(diameter)

        # Storage for this crater's degradation evolution
        crater_data[crater_idx] = {
            'diameter': diameter,
            'distance': distance,
            'pristine_elevation': elev_pristine,
            'degraded_profiles': {},
            'chebyshev_coeffs': {},
            'd_D_ratios': {},
            'depths': {}
        }

        # Compute pristine d/D ratio
        d_D_pristine, depth_pristine = synthesizer.compute_depth_diameter_ratio(
            distance, elev_pristine, diameter
        )

        # Process each age
        for age_idx, (age_ga, age_myr) in enumerate(zip(ages_ga, ages_myr)):
            # Apply degradation
            elev_degraded = synthesizer.apply_degradation(
                distance, elev_pristine, age_myr, diameter
            )

            # Store degraded profile
            crater_data[crater_idx]['degraded_profiles'][age_ga] = elev_degraded

            # Compute d/D ratio
            d_D_ratio, depth = synthesizer.compute_depth_diameter_ratio(
                distance, elev_degraded, diameter
            )
            crater_data[crater_idx]['d_D_ratios'][age_ga] = d_D_ratio
            crater_data[crater_idx]['depths'][age_ga] = depth

            # Create profile dict for Chebyshev analysis
            # We'll use a single profile (azimuthally symmetric)
            profiles = []
            for angle in range(0, 360, 45):  # 8 profiles
                profiles.append({
                    'distance': distance / (diameter / 2.0) * (diameter / 2.0),  # Keep in meters
                    'elevation': elev_degraded,
                    'angle': angle
                })

            # Extract Chebyshev coefficients
            try:
                coef_matrix, analysis, metadata = extract_chebyshev_coefficients(
                    profiles, diameter=diameter, num_coefficients=17
                )

                # Store mean coefficients
                mean_coeffs = analysis['mean_coefficients']
                crater_data[crater_idx]['chebyshev_coeffs'][age_ga] = mean_coeffs

                # Record for CSV
                result_row = {
                    'crater_id': crater_idx,
                    'diameter_m': diameter,
                    'age_ga': age_ga,
                    'd_D_ratio': d_D_ratio,
                    'depth_m': depth,
                    'd_D_pristine': d_D_pristine if age_ga == ages_ga[0] else np.nan,
                    'degradation_factor': d_D_ratio / d_D_pristine if d_D_pristine > 0 else 0
                }

                # Add individual coefficients
                for i in range(17):
                    result_row[f'C{i}'] = mean_coeffs[i]

                # Add derived indices
                result_row['central_peak_idx'] = abs(mean_coeffs[4]) + abs(mean_coeffs[8])
                result_row['asymmetry_idx'] = np.sum(np.abs(mean_coeffs[1::2]))

                all_results.append(result_row)

            except Exception as e:
                print(f"    Warning: Chebyshev extraction failed for age {age_ga} Ga: {e}")

        print(f"    Pristine d/D: {d_D_pristine:.4f}")
        print(f"    Final d/D ({ages_ga[-1]} Ga): {crater_data[crater_idx]['d_D_ratios'][ages_ga[-1]]:.4f}")

    # Create DataFrame and save to CSV
    df_results = pd.DataFrame(all_results)
    csv_file = os.path.join(output_dir, 'degradation_chebyshev_results.csv')
    df_results.to_csv(csv_file, index=False)
    print(f"\n✓ Results saved to: {csv_file}")

    # Save to MATLAB format
    try:
        matlab_data = {
            'num_craters': num_craters,
            'diameters': diameters,
            'ages_ga': ages_ga,
            'diffusivity': diffusivity
        }

        # Add crater-specific data
        for crater_idx in range(num_craters):
            matlab_data[f'crater{crater_idx}_diameter'] = crater_data[crater_idx]['diameter']
            matlab_data[f'crater{crater_idx}_distance'] = crater_data[crater_idx]['distance']
            matlab_data[f'crater{crater_idx}_pristine_elev'] = crater_data[crater_idx]['pristine_elevation']

            # Stack degraded profiles
            degraded_stack = np.array([
                crater_data[crater_idx]['degraded_profiles'][age]
                for age in ages_ga
            ]).T  # Shape: (num_points, num_ages)
            matlab_data[f'crater{crater_idx}_degraded_profiles'] = degraded_stack

            # Stack Chebyshev coefficients
            cheb_stack = np.array([
                crater_data[crater_idx]['chebyshev_coeffs'][age]
                for age in ages_ga
            ]).T  # Shape: (17, num_ages)
            matlab_data[f'crater{crater_idx}_chebyshev'] = cheb_stack

            # d/D ratios
            d_D_array = np.array([
                crater_data[crater_idx]['d_D_ratios'][age]
                for age in ages_ga
            ])
            matlab_data[f'crater{crater_idx}_d_D_ratios'] = d_D_array

        matlab_file = os.path.join(output_dir, 'degradation_analysis.mat')
        savemat(matlab_file, matlab_data, do_compression=True)
        print(f"✓ MATLAB file saved to: {matlab_file}")

    except Exception as e:
        print(f"  Warning: MATLAB export failed: {e}")

    # Create PDF visualization
    print(f"\nGenerating PDF visualization...")
    create_visualization_pdf(crater_data, diameters, ages_ga, output_dir)

    print("\n" + "="*60)
    print("SYNTHESIS AND DEGRADATION TEST COMPLETE")
    print("="*60)


def create_visualization_pdf(crater_data, diameters, ages_ga, output_dir):
    """
    Create comprehensive PDF visualization of results.

    Parameters:
    -----------
    crater_data : dict
        Dictionary containing all crater data
    diameters : ndarray
        Array of crater diameters
    ages_ga : ndarray
        Array of ages in Ga
    output_dir : str
        Output directory
    """
    pdf_file = os.path.join(output_dir, 'degradation_analysis_report.pdf')

    with PdfPages(pdf_file) as pdf:
        # Page 1: Overview and methodology
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Crater Degradation Analysis: Synthesis Test Report', fontsize=16, fontweight='bold')

        ax = fig.add_subplot(111)
        ax.axis('off')

        summary_text = f"""
CRATER SYNTHESIS AND DEGRADATION ANALYSIS
{'='*60}

PARAMETERS:
• Number of craters: {len(diameters)}
• Diameter range: {diameters.min():.0f} - {diameters.max():.0f} m
• Age range: {ages_ga.min():.1f} - {ages_ga.max():.1f} Ga
• Age step: {ages_ga[1] - ages_ga[0]:.1f} Ga
• Diffusivity: 5.0 m²/Myr

METHODOLOGY:
1. Pristine Profile Generation
   - Parabolic bowl interior
   - Exponential rim decay
   - Fresh d/D ≈ 0.196 (lunar simple craters)
   - Rim height ≈ 4% of diameter

2. Degradation Modeling
   - Diffusion-based degradation
   - Smoothing scale: σ = √(2κt)
   - Applied at {len(ages_ga)} age steps

3. Chebyshev Analysis
   - 17 coefficients (C0-C16) extracted
   - 8 radial profiles (45° intervals)
   - Normalized by diameter

4. Depth-to-Diameter Tracking
   - Measured at each age step
   - Compared to pristine value

OUTPUT FILES:
• degradation_chebyshev_results.csv
• degradation_analysis.mat
• degradation_analysis_report.pdf (this file)
        """

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2-3: Profile evolution for selected craters
        for page in range(2):
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle(f'Elevation Profile Evolution (Craters {page*4+1}-{page*4+4})',
                        fontsize=14, fontweight='bold')

            for idx in range(4):
                crater_idx = page * 4 + idx
                if crater_idx >= len(diameters):
                    break

                ax = axes[idx//2, idx%2]
                diameter = crater_data[crater_idx]['diameter']
                distance = crater_data[crater_idx]['distance']

                # Plot pristine
                ax.plot(distance/diameter, crater_data[crater_idx]['pristine_elevation']/diameter,
                       'k-', linewidth=2, label='Pristine', zorder=10)

                # Plot degraded profiles at selected ages
                age_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(ages_ga)))
                for age_idx, age in enumerate([0.5, 1.0, 2.0, 3.0, 3.5]):
                    if age in crater_data[crater_idx]['degraded_profiles']:
                        elev = crater_data[crater_idx]['degraded_profiles'][age]
                        ax.plot(distance/diameter, elev/diameter,
                               color=age_colors[int(age/0.1)], alpha=0.7,
                               label=f'{age} Ga')

                ax.set_xlabel('Normalized Distance (r/D)', fontsize=9)
                ax.set_ylabel('Normalized Elevation (h/D)', fontsize=9)
                ax.set_title(f'Crater {crater_idx+1}: D = {diameter:.0f} m', fontsize=10)
                ax.legend(fontsize=7, loc='best')
                ax.grid(True, alpha=0.3)
                ax.axhline(0, color='k', linewidth=0.5)
                ax.axvline(0, color='k', linewidth=0.5)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Page 4: d/D ratio evolution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5))
        fig.suptitle('Depth-to-Diameter Ratio Evolution', fontsize=14, fontweight='bold')

        # Plot d/D vs age for all craters
        colors = plt.cm.viridis(np.linspace(0, 1, len(diameters)))
        for crater_idx, diameter in enumerate(diameters):
            d_D_values = [crater_data[crater_idx]['d_D_ratios'][age] for age in ages_ga]
            ax1.plot(ages_ga, d_D_values, 'o-', color=colors[crater_idx],
                    label=f'D={diameter:.0f}m', linewidth=2, markersize=4)

        ax1.set_xlabel('Age (Ga)', fontsize=11)
        ax1.set_ylabel('d/D Ratio', fontsize=11)
        ax1.set_title('d/D Evolution with Age', fontsize=12)
        ax1.legend(fontsize=7, loc='best', ncol=2)
        ax1.grid(True, alpha=0.3)

        # Plot normalized d/D (relative to pristine)
        for crater_idx, diameter in enumerate(diameters):
            d_D_values = np.array([crater_data[crater_idx]['d_D_ratios'][age] for age in ages_ga])
            d_D_pristine = d_D_values[0]
            normalized = d_D_values / d_D_pristine if d_D_pristine > 0 else d_D_values
            ax2.plot(ages_ga, normalized, 'o-', color=colors[crater_idx],
                    linewidth=2, markersize=4)

        ax2.set_xlabel('Age (Ga)', fontsize=11)
        ax2.set_ylabel('Normalized d/D (relative to pristine)', fontsize=11)
        ax2.set_title('Normalized Degradation', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Pristine')
        ax2.legend(fontsize=9)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 5-6: Chebyshev coefficient evolution
        # Select key coefficients to plot
        key_coeffs = [0, 2, 4, 8]  # C0, C2 (depth), C4, C8 (central peaks)
        coeff_names = ['C0 (Mean)', 'C2 (Depth)', 'C4 (Peak)', 'C8 (Peak)']

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Key Chebyshev Coefficient Evolution', fontsize=14, fontweight='bold')

        for plot_idx, (coeff_idx, coeff_name) in enumerate(zip(key_coeffs, coeff_names)):
            ax = axes[plot_idx//2, plot_idx%2]

            for crater_idx, diameter in enumerate(diameters):
                coeff_values = np.array([
                    crater_data[crater_idx]['chebyshev_coeffs'][age][coeff_idx]
                    for age in ages_ga
                ])
                ax.plot(ages_ga, coeff_values, 'o-', color=colors[crater_idx],
                       linewidth=2, markersize=4, alpha=0.7)

            ax.set_xlabel('Age (Ga)', fontsize=10)
            ax.set_ylabel(f'{coeff_name} Value', fontsize=10)
            ax.set_title(f'{coeff_name} vs Age', fontsize=11)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 7: Absolute magnitude of all coefficients
        fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
        fig.suptitle('Chebyshev Coefficient Magnitude Evolution', fontsize=14, fontweight='bold')

        # Average across all craters
        for coeff_idx in range(17):
            avg_magnitudes = []
            for age in ages_ga:
                magnitudes = [
                    abs(crater_data[crater_idx]['chebyshev_coeffs'][age][coeff_idx])
                    for crater_idx in range(len(diameters))
                ]
                avg_magnitudes.append(np.mean(magnitudes))

            ax.plot(ages_ga, avg_magnitudes, 'o-', label=f'|C{coeff_idx}|',
                   linewidth=2, markersize=4, alpha=0.7)

        ax.set_xlabel('Age (Ga)', fontsize=12)
        ax.set_ylabel('Average |Coefficient| Magnitude', fontsize=12)
        ax.set_title('Average Absolute Chebyshev Coefficient Magnitudes vs Age', fontsize=13)
        ax.legend(fontsize=8, loc='best', ncol=3)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"✓ PDF report saved to: {pdf_file}")


if __name__ == '__main__':
    print("\nCrater Synthesis and Degradation Test")
    print("This will generate synthetic craters, apply degradation, and analyze evolution.\n")

    run_synthesis_degradation_test(output_dir='synthesis_test_results')

    print("\nTest complete! Check 'synthesis_test_results/' directory for outputs.")

"""
Enhanced crater morphometry analysis module with multiple depth estimation methods.

This module provides:
- Method 1: Existing rim perimeter-based depth estimation
- Method 2: 2D Gaussian fitting for floor depth estimation
- Error propagation including rim probability uncertainty
- Comprehensive visualization outputs
- Statistical analysis and probability distributions
"""

import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask as rio_mask
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import shapely.geometry
import copy
from uncertainties import ufloat, unumpy as unp
import warnings

from . import cratools


def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    2D Gaussian function for fitting.

    Args:
        xy: Tuple of (x, y) coordinate arrays
        amplitude: Peak amplitude
        xo, yo: Center coordinates
        sigma_x, sigma_y: Standard deviations in x and y
        theta: Rotation angle
        offset: Baseline offset

    Returns:
        Flattened Gaussian values
    """
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    g = offset + amplitude * np.exp(-(a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
    return g.ravel()


def fit_gaussian_floor(crater_dem, crater_geom, transform, max_iterations=10, tolerance=0.1):
    """
    Fit 2D Gaussian to inverted crater to estimate floor depth.

    Constrains the fit so the Gaussian just touches the lowest portion inside
    the crater with no overshoot or undershoot.

    Args:
        crater_dem: Elevation array for crater region
        crater_geom: Crater geometry (polygon)
        transform: Raster transform
        max_iterations: Maximum fitting iterations
        tolerance: Convergence tolerance (meters)

    Returns:
        floor_elevation: Estimated floor elevation (Gaussian peak)
        floor_uncertainty: Uncertainty in floor estimation
        fit_quality: Quality metric (0-1, higher is better)
        gaussian_params: Fitted Gaussian parameters
    """
    # Get valid (non-NaN) elevation data
    valid_mask = ~np.isnan(crater_dem)

    if not np.any(valid_mask):
        return np.nan, np.nan, 0.0, None

    # Create coordinate grids
    rows, cols = np.where(valid_mask)
    elevations = crater_dem[valid_mask]

    # Invert elevations so crater floor becomes peak
    min_elev = np.nanmin(crater_dem)
    max_elev = np.nanmax(crater_dem)
    inverted = max_elev - crater_dem
    inverted_valid = inverted[valid_mask]

    # Create x, y coordinates
    y_coords, x_coords = np.indices(crater_dem.shape)
    x_data = x_coords[valid_mask].ravel()
    y_data = y_coords[valid_mask].ravel()
    z_data = inverted_valid.ravel()

    # Initial parameter guesses
    amplitude_guess = np.nanmax(inverted)

    # Find deepest point as initial center
    deepest_idx = np.argmin(elevations)
    xo_guess = x_data[deepest_idx]
    yo_guess = y_data[deepest_idx]

    # Estimate sigma from crater radius
    crater_radius_pixels = np.sqrt(np.sum(valid_mask) / np.pi)
    sigma_guess = crater_radius_pixels / 3.0  # Crater floor typically ~1/3 diameter

    theta_guess = 0.0
    offset_guess = 0.0

    initial_guess = [amplitude_guess, xo_guess, yo_guess,
                     sigma_guess, sigma_guess, theta_guess, offset_guess]

    # Set bounds to prevent overshoot
    # Amplitude must be positive and less than max inverted elevation
    # Offset must be >= 0 (no undershoot)
    lower_bounds = [0, 0, 0, crater_radius_pixels/10, crater_radius_pixels/10, -np.pi, 0]
    upper_bounds = [amplitude_guess * 1.2, crater_dem.shape[1], crater_dem.shape[0],
                    crater_radius_pixels, crater_radius_pixels, np.pi, amplitude_guess * 0.1]

    try:
        # Fit the Gaussian
        popt, pcov = curve_fit(
            gaussian_2d,
            (x_data, y_data),
            z_data,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=max_iterations * 1000,
            method='trf'
        )

        # Extract parameters
        amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt

        # Calculate floor elevation (inverse of Gaussian peak)
        gaussian_peak_inverted = amplitude + offset
        floor_elevation = max_elev - gaussian_peak_inverted

        # Ensure floor is not below actual minimum
        if floor_elevation < min_elev - tolerance:
            floor_elevation = min_elev

        # Estimate uncertainty from covariance
        perr = np.sqrt(np.diag(pcov))
        amplitude_err = perr[0]
        floor_uncertainty = amplitude_err  # Uncertainty in peak amplitude

        # Compute fit quality (R-squared)
        fitted_z = gaussian_2d((x_data, y_data), *popt)
        residuals = z_data - fitted_z
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((z_data - np.mean(z_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        fit_quality = max(0.0, min(1.0, r_squared))

        return floor_elevation, floor_uncertainty, fit_quality, popt

    except (RuntimeError, ValueError) as e:
        warnings.warn(f"Gaussian fitting failed: {e}. Using minimum elevation.")
        # Fallback to minimum elevation
        floor_uncertainty = np.nanstd(elevations) * 0.5
        return min_elev, floor_uncertainty, 0.3, None


def compute_morphometry_dual_method(geom, crater_id, dem_path, orthophoto_path, crs,
                                     rim_probability=1.0, radius_error=0.1,
                                     remove_external_topo=True, plot=False):
    """
    Compute crater morphometry using two methods.

    Method 1: Existing rim perimeter method (cratools)
    Method 2: 2D Gaussian floor fitting

    Args:
        geom: Crater geometry (polygon)
        crater_id: Crater identifier
        dem_path: Path to DEM file
        orthophoto_path: Path to orthophoto
        crs: Coordinate reference system
        rim_probability: Probability of rim detection (0-1)
        radius_error: Radius error from refinement (meters)
        remove_external_topo: Remove regional slope
        plot: Generate diagnostic plots

    Returns:
        dict: Morphometry measurements with both methods
    """
    results = {
        'crater_id': crater_id,
        'method1': {},
        'method2': {},
        'combined': {}
    }

    # Get basic crater properties
    radius = np.sqrt(geom.area / np.pi)
    diameter = 2 * radius

    # === Method 1: Existing rim perimeter method ===
    try:
        with rio.open(dem_path, 'r') as dem_src:
            ratio_m1, depth_m1, diam_m1, rim_m1, floor_m1 = cratools.compute_depth_diameter_ratio(
                geom=geom,
                filename_dem=dem_path,
                crs=crs,
                orthophoto=orthophoto_path,
                diam_err=2 * radius_error,
                remove_ext=remove_external_topo,
                plot=plot
            )

        results['method1'] = {
            'd_D': ratio_m1,
            'depth': depth_m1,
            'diameter': diam_m1,
            'rim_height': rim_m1,
            'floor_height': floor_m1
        }
    except Exception as e:
        warnings.warn(f"Method 1 failed for crater {crater_id}: {e}")
        results['method1'] = {
            'd_D': ufloat(np.nan, np.nan),
            'depth': ufloat(np.nan, np.nan),
            'diameter': ufloat(diameter, radius_error * 2),
            'rim_height': ufloat(np.nan, np.nan),
            'floor_height': np.nan
        }

    # === Method 2: Gaussian floor fitting ===
    try:
        with rio.open(dem_path, 'r') as dem_src:
            # Extract crater region
            if remove_external_topo:
                (out_image, out_transform), _ = cratools.remove_external_topography(
                    geom=geom,
                    dem_src=dem_src,
                    orthophoto=rio.open(orthophoto_path, 'r'),
                    crs=crs
                )
            else:
                try:
                    mask_geom = shapely.geometry.MultiPolygon([geom])
                except:
                    mask_geom = geom
                (out_image, out_transform) = rio_mask(dem_src, mask_geom.geoms,
                                                      crop=True, all_touched=True, nodata=np.nan)

            crater_dem = out_image[0, :, :]

            # Fit Gaussian to find floor
            floor_m2, floor_unc_m2, fit_quality, gaussian_params = fit_gaussian_floor(
                crater_dem, geom, out_transform
            )

            # Get rim height from Method 1 (more reliable for rim)
            if not np.isnan(unp.nominal_values(results['method1']['rim_height'])[0]):
                rim_m2 = results['method1']['rim_height']
            else:
                # Fallback: compute rim from perimeter pixels
                image_perimeter = copy.deepcopy(crater_dem)
                test_image = np.pad(crater_dem, 1, constant_values=np.nan)

                for i in range(1, test_image.shape[0] - 1):
                    for j in range(1, test_image.shape[1] - 1):
                        subimage = test_image[i-1:i+2, j-1:j+2]
                        if not np.any(np.isnan(subimage)):
                            image_perimeter[i-1, j-1] = np.nan

                perim_vals = image_perimeter.flatten()
                perim_vals = perim_vals[~np.isnan(perim_vals)]

                if len(perim_vals) > 0:
                    rim_height_m2 = np.mean(perim_vals)
                    rim_unc_m2 = np.std(perim_vals)
                    rim_m2 = ufloat(rim_height_m2, rim_unc_m2)
                else:
                    rim_m2 = ufloat(np.nan, np.nan)

            # Compute depth (Method 2)
            depth_m2 = rim_m2 - floor_m2

            # Add Gaussian uncertainty to depth
            if not np.isnan(floor_unc_m2):
                # Combine rim uncertainty and floor uncertainty
                depth_total_unc = np.sqrt(unp.std_devs(rim_m2)**2 + floor_unc_m2**2)
                depth_m2 = ufloat(unp.nominal_values(depth_m2), depth_total_unc)

            # Compute d/D ratio (Method 2)
            diam_m2 = ufloat(diameter, radius_error * 2)
            ratio_m2 = depth_m2 / diam_m2

            results['method2'] = {
                'd_D': ratio_m2,
                'depth': depth_m2,
                'diameter': diam_m2,
                'rim_height': rim_m2,
                'floor_height': floor_m2,
                'floor_uncertainty': floor_unc_m2,
                'fit_quality': fit_quality,
                'gaussian_params': gaussian_params
            }
    except Exception as e:
        warnings.warn(f"Method 2 failed for crater {crater_id}: {e}")
        results['method2'] = {
            'd_D': ufloat(np.nan, np.nan),
            'depth': ufloat(np.nan, np.nan),
            'diameter': ufloat(diameter, radius_error * 2),
            'rim_height': ufloat(np.nan, np.nan),
            'floor_height': np.nan,
            'floor_uncertainty': np.nan,
            'fit_quality': 0.0,
            'gaussian_params': None
        }

    # === Propagate rim probability error ===
    # Convert rim probability to uncertainty factor
    # High probability (0.9) → low additional uncertainty
    # Low probability (0.3) → high additional uncertainty
    prob_uncertainty_factor = 1.0 - rim_probability  # 0.1 for prob=0.9, 0.7 for prob=0.3

    # Additional uncertainty as fraction of depth
    for method_name in ['method1', 'method2']:
        if method_name in results:
            depth_val = results[method_name]['depth']
            if not np.isnan(unp.nominal_values(depth_val)):
                depth_nominal = unp.nominal_values(depth_val)
                depth_unc = unp.std_devs(depth_val)

                # Add rim probability uncertainty
                prob_error = abs(depth_nominal) * prob_uncertainty_factor * 0.5
                total_unc = np.sqrt(depth_unc**2 + prob_error**2)

                results[method_name]['depth_with_prob_error'] = ufloat(depth_nominal, total_unc)
                results[method_name]['total_error'] = total_unc
                results[method_name]['probability_contribution'] = prob_error

                # Update d/D with total error
                diam_val = results[method_name]['diameter']
                ratio_with_error = ufloat(depth_nominal, total_unc) / diam_val
                results[method_name]['d_D_total_error'] = ratio_with_error

    # === Combined estimate (average of both methods) ===
    try:
        d_D_m1 = results['method1']['d_D']
        d_D_m2 = results['method2']['d_D']

        if not (np.isnan(unp.nominal_values(d_D_m1)) or np.isnan(unp.nominal_values(d_D_m2))):
            # Weighted average (weight by inverse variance)
            var1 = unp.std_devs(d_D_m1)**2
            var2 = unp.std_devs(d_D_m2)**2

            if var1 + var2 > 0:
                w1 = (1 / var1) if var1 > 0 else 1.0
                w2 = (1 / var2) if var2 > 0 else 1.0
                w_total = w1 + w2

                combined_d_D = (w1 * unp.nominal_values(d_D_m1) + w2 * unp.nominal_values(d_D_m2)) / w_total
                combined_unc = np.sqrt(1 / w_total)

                results['combined']['d_D'] = ufloat(combined_d_D, combined_unc)
            else:
                results['combined']['d_D'] = d_D_m1
        elif not np.isnan(unp.nominal_values(d_D_m1)):
            results['combined']['d_D'] = d_D_m1
        elif not np.isnan(unp.nominal_values(d_D_m2)):
            results['combined']['d_D'] = d_D_m2
        else:
            results['combined']['d_D'] = ufloat(np.nan, np.nan)
    except:
        results['combined']['d_D'] = ufloat(np.nan, np.nan)

    return results


def analyze_crater_morphometry(input_shapefile, output_shapefile, dem_path, orthophoto_path,
                                output_dir, min_diameter=60.0, remove_external_topo=True,
                                plot_individual=False):
    """
    Analyze crater morphometry for all craters in shapefile.

    Args:
        input_shapefile: Path to shapefile from Step 2 (rim refinement)
        output_shapefile: Path to save morphometry results
        dem_path: Path to DEM file
        orthophoto_path: Path to orthophoto
        output_dir: Directory for output plots and CSVs
        min_diameter: Minimum diameter to process (meters)
        remove_external_topo: Remove regional topography
        plot_individual: Plot each crater individually

    Returns:
        dict: Results summary with paths to outputs
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load shapefile
    gdf = gpd.read_file(input_shapefile)
    crs = gdf.crs

    print(f"Analyzing morphometry for {len(gdf)} craters...")
    print(f"Using dual-method approach:")
    print(f"  Method 1: Rim perimeter analysis")
    print(f"  Method 2: 2D Gaussian floor fitting")

    # Initialize result lists
    morphometry_data = []

    # Process each crater
    for idx, row in gdf.iterrows():
        crater_geom = row['geometry']
        crater_id = row.get('UFID', f'crater_{idx}')

        # Get diameter
        radius = np.sqrt(crater_geom.area / np.pi)
        diameter = radius * 2

        if diameter < min_diameter:
            continue

        print(f"\nProcessing {crater_id}: D={diameter:.1f}m")

        # Get rim probability and radius error from refinement
        rim_probability = row.get('rim_probability', 1.0)
        radius_error = row.get('err_r', 0.1)

        # Compute morphometry
        results = compute_morphometry_dual_method(
            geom=crater_geom,
            crater_id=crater_id,
            dem_path=dem_path,
            orthophoto_path=orthophoto_path,
            crs=crs,
            rim_probability=rim_probability,
            radius_error=radius_error,
            remove_external_topo=remove_external_topo,
            plot=plot_individual
        )

        # Store results
        morphometry_data.append({
            'crater_id': crater_id,
            'geometry': crater_geom,
            **{k: v for k, v in row.items() if k != 'geometry'},
            **extract_morphometry_fields(results)
        })

        print(f"  Method 1: d/D = {results['method1']['d_D']:.3f}")
        print(f"  Method 2: d/D = {results['method2']['d_D']:.3f}, fit_quality = {results['method2']['fit_quality']:.2f}")
        print(f"  Rim probability: {rim_probability:.2f}")

    # Create output GeoDataFrame
    output_gdf = gpd.GeoDataFrame(morphometry_data, crs=crs)
    output_gdf.to_file(output_shapefile)
    print(f"\n✓ Saved morphometry shapefile: {output_shapefile}")

    # Generate outputs
    outputs = {}

    # Output 4: CSV with morphometry data (no geometry)
    csv_path = os.path.join(output_dir, 'morphometry_data.csv')
    df_no_geom = output_gdf.drop(columns=['geometry'])
    df_no_geom.to_csv(csv_path, index=False)
    outputs['csv_morphometry'] = csv_path
    print(f"✓ Saved morphometry CSV: {csv_path}")

    # Output 2: Scatter plots (depth vs diameter, d/D vs diameter)
    scatter_path = os.path.join(output_dir, 'morphometry_scatter_plots.png')
    plot_morphometry_scatter(output_gdf, scatter_path)
    outputs['scatter_plots'] = scatter_path
    print(f"✓ Saved scatter plots: {scatter_path}")

    # Output 3: Probability distributions
    prob_path = os.path.join(output_dir, 'probability_distributions.png')
    plot_probability_distributions(output_gdf, prob_path)
    outputs['probability_plots'] = prob_path
    print(f"✓ Saved probability plots: {prob_path}")

    # Output 5: Conditional probability CSV
    cond_prob_path = os.path.join(output_dir, 'conditional_probability.csv')
    compute_conditional_probabilities(output_gdf, cond_prob_path)
    outputs['conditional_probability'] = cond_prob_path
    print(f"✓ Saved conditional probability CSV: {cond_prob_path}")

    # Summary statistics
    outputs['shapefile'] = output_shapefile
    outputs['statistics'] = compute_summary_statistics(output_gdf)

    return outputs


def extract_morphometry_fields(results):
    """Extract morphometry measurements into flat dictionary."""
    fields = {}

    # Method 1 fields
    if 'method1' in results:
        m1 = results['method1']
        fields['diam_m1'] = unp.nominal_values(m1['diameter']) if hasattr(m1['diameter'], 'n') else m1['diameter']
        fields['depth_m1'] = unp.nominal_values(m1['depth']) if hasattr(m1['depth'], 'n') else m1['depth']
        fields['depth_err_m1'] = unp.std_devs(m1['depth']) if hasattr(m1['depth'], 'n') else 0.0
        fields['d_D_m1'] = unp.nominal_values(m1['d_D']) if hasattr(m1['d_D'], 'n') else m1['d_D']
        fields['d_D_err_m1'] = unp.std_devs(m1['d_D']) if hasattr(m1['d_D'], 'n') else 0.0
        fields['rim_height_m1'] = unp.nominal_values(m1['rim_height']) if hasattr(m1['rim_height'], 'n') else m1['rim_height']
        fields['floor_height_m1'] = m1['floor_height']

        if 'total_error' in m1:
            fields['total_error_m1'] = m1['total_error']
            fields['prob_error_m1'] = m1['probability_contribution']

    # Method 2 fields
    if 'method2' in results:
        m2 = results['method2']
        fields['diam_m2'] = unp.nominal_values(m2['diameter']) if hasattr(m2['diameter'], 'n') else m2['diameter']
        fields['depth_m2'] = unp.nominal_values(m2['depth']) if hasattr(m2['depth'], 'n') else m2['depth']
        fields['depth_err_m2'] = unp.std_devs(m2['depth']) if hasattr(m2['depth'], 'n') else 0.0
        fields['d_D_m2'] = unp.nominal_values(m2['d_D']) if hasattr(m2['d_D'], 'n') else m2['d_D']
        fields['d_D_err_m2'] = unp.std_devs(m2['d_D']) if hasattr(m2['d_D'], 'n') else 0.0
        fields['rim_height_m2'] = unp.nominal_values(m2['rim_height']) if hasattr(m2['rim_height'], 'n') else m2['rim_height']
        fields['floor_height_m2'] = m2['floor_height']
        fields['floor_unc_m2'] = m2['floor_uncertainty']
        fields['fit_quality_m2'] = m2['fit_quality']

        if 'total_error' in m2:
            fields['total_error_m2'] = m2['total_error']
            fields['prob_error_m2'] = m2['probability_contribution']

    # Combined fields
    if 'combined' in results and 'd_D' in results['combined']:
        combined_d_D = results['combined']['d_D']
        fields['d_D_combined'] = unp.nominal_values(combined_d_D) if hasattr(combined_d_D, 'n') else combined_d_D
        fields['d_D_err_combined'] = unp.std_devs(combined_d_D) if hasattr(combined_d_D, 'n') else 0.0

    return fields


def plot_morphometry_scatter(gdf, output_path):
    """
    Create scatter plots: depth vs diameter, d/D vs diameter.

    Output 2: Two-panel figure with error bars.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Filter valid data
    valid_m1 = gdf[gdf['d_D_m1'].notna() & (gdf['d_D_m1'] > 0)]
    valid_m2 = gdf[gdf['d_D_m2'].notna() & (gdf['d_D_m2'] > 0)]

    # Panel 1: Depth vs Diameter
    if len(valid_m1) > 0:
        ax1.errorbar(valid_m1['diam_m1'], valid_m1['depth_m1'],
                     yerr=valid_m1['depth_err_m1'],
                     fmt='o', color='blue', alpha=0.6, label='Method 1 (Rim perimeter)',
                     markersize=6, capsize=3)

    if len(valid_m2) > 0:
        ax1.errorbar(valid_m2['diam_m2'], valid_m2['depth_m2'],
                     yerr=valid_m2['depth_err_m2'],
                     fmt='s', color='red', alpha=0.6, label='Method 2 (Gaussian fit)',
                     markersize=5, capsize=3)

    ax1.set_xlabel('Diameter (m)', fontsize=12)
    ax1.set_ylabel('Depth (m)', fontsize=12)
    ax1.set_title('Crater Depth vs Diameter', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: d/D vs Diameter
    if len(valid_m1) > 0:
        ax2.errorbar(valid_m1['diam_m1'], valid_m1['d_D_m1'],
                     yerr=valid_m1['d_D_err_m1'],
                     fmt='o', color='blue', alpha=0.6, label='Method 1',
                     markersize=6, capsize=3)

    if len(valid_m2) > 0:
        ax2.errorbar(valid_m2['diam_m2'], valid_m2['d_D_m2'],
                     yerr=valid_m2['d_D_err_m2'],
                     fmt='s', color='red', alpha=0.6, label='Method 2',
                     markersize=5, capsize=3)

    ax2.set_xlabel('Diameter (m)', fontsize=12)
    ax2.set_ylabel('d/D Ratio', fontsize=12)
    ax2.set_title('Depth-to-Diameter Ratio vs Diameter', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Fresh crater (d/D~0.2)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_probability_distributions(gdf, output_path):
    """
    Create probability distribution plots.

    Output 3:
    - Panel 1: 2D joint probability (d, D) filled contour
    - Panel 2: Probability distribution for d/D
    """
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Use Method 2 (Gaussian) for analysis
    valid = gdf[(gdf['depth_m2'].notna()) & (gdf['diam_m2'].notna()) &
                (gdf['depth_m2'] > 0) & (gdf['diam_m2'] > 0)]

    if len(valid) < 3:
        # Not enough data for KDE
        ax1.text(0.5, 0.5, 'Insufficient data for probability distributions\n(need >= 3 valid craters)',
                ha='center', va='center', fontsize=12)
        ax2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    depth_data = valid['depth_m2'].values
    diam_data = valid['diam_m2'].values
    d_D_data = valid['d_D_m2'].values

    # Panel 1: 2D joint probability (depth, diameter)
    try:
        # Create KDE
        values = np.vstack([depth_data, diam_data])
        kernel = gaussian_kde(values)

        # Create grid
        depth_min, depth_max = depth_data.min(), depth_data.max()
        diam_min, diam_max = diam_data.min(), diam_data.max()

        depth_range = depth_max - depth_min
        diam_range = diam_max - diam_min

        depth_grid = np.linspace(depth_min - 0.1*depth_range, depth_max + 0.1*depth_range, 100)
        diam_grid = np.linspace(diam_min - 0.1*diam_range, diam_max + 0.1*diam_range, 100)

        D_grid, Dep_grid = np.meshgrid(diam_grid, depth_grid)
        positions = np.vstack([Dep_grid.ravel(), D_grid.ravel()])
        Z = np.reshape(kernel(positions).T, Dep_grid.shape)

        # Plot filled contour
        contour = ax1.contourf(D_grid, Dep_grid, Z, levels=15, cmap='viridis', alpha=0.8)
        plt.colorbar(contour, ax=ax1, label='Probability Density')

        # Overlay data points
        ax1.scatter(diam_data, depth_data, c='white', s=30, alpha=0.7, edgecolors='black', linewidths=0.5)

        ax1.set_xlabel('Diameter (m)', fontsize=12)
        ax1.set_ylabel('Depth (m)', fontsize=12)
        ax1.set_title('Joint Probability Distribution: P(depth, diameter)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)

    except Exception as e:
        ax1.text(0.5, 0.5, f'2D KDE failed:\n{str(e)[:50]}', ha='center', va='center', fontsize=10)

    # Panel 2: Probability distribution for d/D
    try:
        # Create KDE for d/D
        kernel_dD = gaussian_kde(d_D_data)

        d_D_min, d_D_max = d_D_data.min(), d_D_data.max()
        d_D_range = d_D_max - d_D_min
        d_D_grid = np.linspace(max(0, d_D_min - 0.2*d_D_range), d_D_max + 0.2*d_D_range, 500)
        pdf = kernel_dD(d_D_grid)

        # Plot PDF
        ax2.fill_between(d_D_grid, pdf, alpha=0.3, color='blue', label='Probability density')
        ax2.plot(d_D_grid, pdf, color='blue', linewidth=2)

        # Overlay histogram
        ax2.hist(d_D_data, bins=15, density=True, alpha=0.5, color='gray',
                edgecolor='black', label='Histogram')

        # Add mean and std
        mean_dD = np.mean(d_D_data)
        std_dD = np.std(d_D_data)
        ax2.axvline(mean_dD, color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {mean_dD:.3f}')
        ax2.axvline(mean_dD - std_dD, color='orange', linestyle=':', linewidth=1.5)
        ax2.axvline(mean_dD + std_dD, color='orange', linestyle=':', linewidth=1.5,
                   label=f'Std = {std_dD:.3f}')

        ax2.set_xlabel('d/D Ratio', fontsize=12)
        ax2.set_ylabel('Probability Density', fontsize=12)
        ax2.set_title('Probability Distribution: P(d/D)', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)

    except Exception as e:
        ax2.text(0.5, 0.5, f'1D KDE failed:\n{str(e)[:50]}', ha='center', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_conditional_probabilities(gdf, output_path):
    """
    Compute conditional probabilities P(d|D) and P(D|d).

    Output 5: CSV with conditional probability estimates.
    """
    # Use Method 2 data
    valid = gdf[(gdf['depth_m2'].notna()) & (gdf['diam_m2'].notna()) &
                (gdf['depth_m2'] > 0) & (gdf['diam_m2'] > 0)]

    if len(valid) < 3:
        # Write empty CSV with header
        pd.DataFrame(columns=['diameter_bin', 'depth_bin', 'P_d_given_D', 'P_D_given_d',
                              'count', 'mean_d_D', 'std_d_D']).to_csv(output_path, index=False)
        return

    depth_data = valid['depth_m2'].values
    diam_data = valid['diam_m2'].values

    # Create bins
    n_bins = min(10, len(valid) // 2)

    diam_bins = np.linspace(diam_data.min(), diam_data.max(), n_bins + 1)
    depth_bins = np.linspace(depth_data.min(), depth_data.max(), n_bins + 1)

    # Compute 2D histogram
    H, diam_edges, depth_edges = np.histogram2d(diam_data, depth_data, bins=[diam_bins, depth_bins])

    # Conditional probabilities
    # P(d|D) = P(d,D) / P(D)
    P_D = H.sum(axis=1, keepdims=True)  # Marginal P(D)
    P_d_given_D = np.divide(H, P_D, where=P_D > 0, out=np.zeros_like(H))

    # P(D|d) = P(d,D) / P(d)
    P_d = H.sum(axis=0, keepdims=True)  # Marginal P(d)
    P_D_given_d = np.divide(H.T, P_d, where=P_d > 0, out=np.zeros_like(H.T)).T

    # Create output dataframe
    cond_prob_data = []

    for i in range(len(diam_bins) - 1):
        for j in range(len(depth_bins) - 1):
            diam_center = (diam_bins[i] + diam_bins[i+1]) / 2
            depth_center = (depth_bins[j] + depth_bins[j+1]) / 2

            # Find craters in this bin
            in_bin = ((diam_data >= diam_bins[i]) & (diam_data < diam_bins[i+1]) &
                     (depth_data >= depth_bins[j]) & (depth_data < depth_bins[j+1]))

            count = np.sum(in_bin)

            if count > 0:
                d_D_in_bin = valid[in_bin]['d_D_m2'].values
                mean_d_D = np.mean(d_D_in_bin)
                std_d_D = np.std(d_D_in_bin) if count > 1 else 0.0
            else:
                mean_d_D = np.nan
                std_d_D = np.nan

            cond_prob_data.append({
                'diameter_bin': f'{diam_bins[i]:.1f}-{diam_bins[i+1]:.1f}',
                'depth_bin': f'{depth_bins[j]:.1f}-{depth_bins[j+1]:.1f}',
                'diameter_center': diam_center,
                'depth_center': depth_center,
                'P_d_given_D': P_d_given_D[i, j],
                'P_D_given_d': P_D_given_d[i, j],
                'count': int(count),
                'mean_d_D': mean_d_D,
                'std_d_D': std_d_D
            })

    df = pd.DataFrame(cond_prob_data)
    df.to_csv(output_path, index=False)


def compute_summary_statistics(gdf):
    """Compute summary statistics for morphometry results."""
    summary = {
        'total_craters': len(gdf),
        'method1': {},
        'method2': {}
    }

    # Method 1 statistics
    valid_m1 = gdf[gdf['d_D_m1'].notna()]
    if len(valid_m1) > 0:
        summary['method1'] = {
            'count': len(valid_m1),
            'mean_d_D': valid_m1['d_D_m1'].mean(),
            'std_d_D': valid_m1['d_D_m1'].std(),
            'mean_depth': valid_m1['depth_m1'].mean(),
            'std_depth': valid_m1['depth_m1'].std(),
            'mean_diameter': valid_m1['diam_m1'].mean(),
        }

    # Method 2 statistics
    valid_m2 = gdf[gdf['d_D_m2'].notna()]
    if len(valid_m2) > 0:
        summary['method2'] = {
            'count': len(valid_m2),
            'mean_d_D': valid_m2['d_D_m2'].mean(),
            'std_d_D': valid_m2['d_D_m2'].std(),
            'mean_depth': valid_m2['depth_m2'].mean(),
            'std_depth': valid_m2['depth_m2'].std(),
            'mean_diameter': valid_m2['diam_m2'].mean(),
            'mean_fit_quality': valid_m2['fit_quality_m2'].mean(),
        }

    return summary

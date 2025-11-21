"""
Crater rim refinement module with computer vision enhancements.

This module refines crater rim positions using:
- Topographic analysis (existing algorithm)
- Computer vision techniques on contrast images
- Combined probability scoring

Outputs:
- Refined shapefile with probability scores
- PNG of new rim positions
- CSFD plot
- PNG showing rim differences
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import rasterio as rio
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
from scipy import ndimage
from scipy.signal import find_peaks
import cv2

from . import cratools


def compute_edge_strength(image, crater_geom, transform):
    """
    Compute edge strength from contrast image around crater.

    Uses Canny edge detection and circular Hough transform to find
    crater rim in the contrast image.

    Args:
        image: Contrast image array
        crater_geom: Crater geometry (circle)
        transform: Rasterio transform

    Returns:
        edge_strength: Score 0-1 indicating edge clarity
        edge_rim_radius: Detected rim radius from edges (or None)
    """
    # Get crater parameters
    center = crater_geom.centroid
    radius = np.sqrt(crater_geom.area / np.pi)

    # Convert center to pixel coordinates
    try:
        row, col = ~transform * (center.x, center.y)
        row, col = int(row), int(col)
    except Exception:
        return 0.0, None

    # Extract region around crater (3R box)
    box_size = int(3 * radius / abs(transform[0]))  # Convert to pixels

    if box_size < 10:  # Too small
        return 0.0, None

    # Bounds check
    r_min = max(0, row - box_size)
    r_max = min(image.shape[0], row + box_size)
    c_min = max(0, col - box_size)
    c_max = min(image.shape[1], col + box_size)

    if r_max - r_min < 20 or c_max - c_min < 20:  # Region too small
        return 0.0, None

    # Extract and normalize region
    region = image[r_min:r_max, c_min:c_max]

    # Handle different data types
    if region.dtype == np.float32 or region.dtype == np.float64:
        # Normalize to 0-255
        region_norm = ((region - region.min()) / (region.max() - region.min() + 1e-10) * 255).astype(np.uint8)
    else:
        region_norm = region.astype(np.uint8)

    # Apply Canny edge detection
    try:
        edges = cv2.Canny(region_norm, threshold1=50, threshold2=150)
    except Exception:
        # Fallback: use scipy
        from scipy import ndimage
        edges = ndimage.sobel(region_norm.astype(float))
        edges = ((edges - edges.min()) / (edges.max() - edges.min() + 1e-10) * 255).astype(np.uint8)

    # Compute edge density in annulus around expected rim
    center_local = np.array([box_size, box_size])  # Center in extracted region
    y_grid, x_grid = np.meshgrid(np.arange(edges.shape[0]), np.arange(edges.shape[1]), indexing='ij')

    # Distance from center
    dist = np.sqrt((x_grid - center_local[1])**2 + (y_grid - center_local[0])**2)

    # Expected rim radius in pixels
    rim_radius_pix = radius / abs(transform[0])

    # Define annulus (0.8R to 1.2R)
    annulus_mask = (dist >= 0.8 * rim_radius_pix) & (dist <= 1.2 * rim_radius_pix)

    if annulus_mask.sum() == 0:
        return 0.0, None

    # Edge strength = fraction of annulus pixels that are edges
    edge_strength = (edges[annulus_mask] > 0).sum() / annulus_mask.sum()

    # Try to detect rim radius from edges using radial profile
    angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
    edge_radii = []

    for angle in angles:
        # Sample along radial line
        max_r = min(box_size * 0.9, 1.5 * rim_radius_pix)
        radii = np.linspace(0, max_r, 100)
        x_samples = center_local[1] + radii * np.cos(angle)
        y_samples = center_local[0] + radii * np.sin(angle)

        # Check bounds
        valid = (x_samples >= 0) & (x_samples < edges.shape[1]) & \
                (y_samples >= 0) & (y_samples < edges.shape[0])

        if valid.sum() < 10:
            continue

        x_samples = x_samples[valid].astype(int)
        y_samples = y_samples[valid].astype(int)
        radii = radii[valid]

        # Get edge values along this radial
        edge_profile = edges[y_samples, x_samples]

        # Find peaks
        peaks, properties = find_peaks(edge_profile, prominence=10, distance=5)

        if len(peaks) > 0:
            # Use strongest peak
            strongest_idx = peaks[np.argmax(properties['prominences'])]
            edge_radii.append(radii[strongest_idx])

    if len(edge_radii) > 5:
        # Detected rim radius in pixels
        detected_radius_pix = np.median(edge_radii)
        # Convert back to meters
        edge_rim_radius = detected_radius_pix * abs(transform[0])
    else:
        edge_rim_radius = None

    return float(edge_strength), edge_rim_radius


def compute_topographic_quality(elevation_profile, radius):
    """
    Assess quality of topographic rim detection.

    Args:
        elevation_profile: 2D array (radius x azimuth) of elevations
        radius: Crater radius

    Returns:
        quality_score: 0-1 score for topographic rim clarity
    """
    if elevation_profile.size == 0:
        return 0.0

    # For each azimuth, check if there's a clear peak
    n_azimuths = elevation_profile.shape[1]
    peak_count = 0
    peak_prominences = []

    for i in range(n_azimuths):
        profile = elevation_profile[:, i]

        if len(profile) < 5:
            continue

        # Find peaks
        peaks, properties = find_peaks(profile, prominence=1e-5 * radius)

        if len(peaks) > 0:
            peak_count += 1
            peak_prominences.append(properties['prominences'].max())

    # Quality based on: fraction of azimuths with peaks, avg prominence
    azimuth_fraction = peak_count / max(n_azimuths, 1)

    if len(peak_prominences) > 0:
        avg_prominence = np.mean(peak_prominences)
        # Normalize prominence (typical values 1-10m for small craters)
        prominence_score = min(avg_prominence / (0.1 * radius), 1.0)
    else:
        prominence_score = 0.0

    # Combined quality
    quality_score = 0.6 * azimuth_fraction + 0.4 * prominence_score

    return float(np.clip(quality_score, 0, 1))


def compute_rim_probability(topo_quality, edge_strength, radius_agreement,
                            diameter_error):
    """
    Compute overall rim detection probability.

    Combines multiple quality metrics into a single probability score.

    Args:
        topo_quality: Quality of topographic rim detection (0-1)
        edge_strength: Strength of edges in contrast image (0-1)
        radius_agreement: Agreement between topo and edge radii (0-1)
        diameter_error: Relative error in diameter (meters)

    Returns:
        probability: Overall detection probability (0-1)
    """
    # Weight factors
    w_topo = 0.4      # Topography is most reliable
    w_edge = 0.3      # Edge detection supplements
    w_agree = 0.2     # Agreement between methods
    w_error = 0.1     # Low fitting error is good

    # Error score (lower error = higher score)
    if diameter_error is not None and diameter_error > 0:
        error_score = np.exp(-diameter_error / 5.0)  # Decays with error
    else:
        error_score = 0.5  # Neutral if no error available

    # Combined probability
    probability = (w_topo * topo_quality +
                  w_edge * edge_strength +
                  w_agree * radius_agreement +
                  w_error * error_score)

    return float(np.clip(probability, 0, 1))


def refine_single_crater(crater_geom, crater_id, dem_src, image_src, crs,
                        inner_radius=0.8, outer_radius=1.2,
                        remove_external_topo=True, plot=False):
    """
    Refine a single crater rim using topography and computer vision.

    Args:
        crater_geom: Input crater geometry (circle)
        crater_id: Crater identifier
        dem_src: Rasterio dataset for DEM
        image_src: Rasterio dataset for contrast image
        crs: Coordinate reference system
        inner_radius: Inner search radius (fraction of R)
        outer_radius: Outer search radius (fraction of R)
        remove_external_topo: Remove regional topography
        plot: Generate diagnostic plots

    Returns:
        refined_geom: Refined crater geometry
        metadata: Dict with errors, quality scores, probability
    """
    # Original parameters
    radius_orig = np.sqrt(crater_geom.area / np.pi)
    center_orig = crater_geom.centroid

    # Step 1: Topographic rim refinement (existing algorithm)
    try:
        refined_geom_topo, err = cratools.fit_crater_rim(
            geom=crater_geom,
            dem_src=dem_src,
            crs=crs,
            orthophoto=image_src.name,  # Pass filename
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            remove_ext=remove_external_topo,
            plot=False  # Don't plot individual craters
        )

        # Extract topographic elevation matrix for quality assessment
        center = np.array([center_orig.x, center_orig.y])

        # Get DEM region
        if remove_external_topo:
            from .cratools import remove_external_topography
            (out_image, out_transform), _ = remove_external_topography(
                geom=crater_geom,
                dem_src=dem_src,
                orthophoto=image_src,
                crs=crs
            )
        else:
            import shapely
            buffered = shapely.geometry.box(
                center[0] - radius_orig * 3, center[1] - radius_orig * 3,
                center[0] + radius_orig * 3, center[1] + radius_orig * 3
            )
            mask = shapely.geometry.MultiPolygon([buffered])
            (out_image, out_transform) = rio.mask.mask(dem_src, mask, crop=True, all_touched=True)

        # Get elevation matrix
        azimuths = np.arange(0., 359., 5)
        r_dist = np.arange(inner_radius, outer_radius, 0.01)
        E_rim = cratools.compute_E_matrix(
            image=out_image,
            transform=out_transform,
            center=center,
            azimuths=azimuths,
            r_dist=r_dist
        )

        # Assess topographic quality
        topo_quality = compute_topographic_quality(E_rim, radius_orig)

        radius_topo = np.sqrt(refined_geom_topo.area / np.pi)
        center_topo = refined_geom_topo.centroid

    except Exception as e:
        print(f"  Warning: Topographic refinement failed for {crater_id}: {e}")
        refined_geom_topo = crater_geom
        err = (np.nan, np.nan, np.nan)
        topo_quality = 0.0
        radius_topo = radius_orig
        center_topo = center_orig
        E_rim = np.array([])

    # Step 2: Edge detection from contrast image
    try:
        # Read image data
        image_data = image_src.read(1)
        edge_strength, edge_rim_radius = compute_edge_strength(
            image_data, crater_geom, image_src.transform
        )
    except Exception as e:
        print(f"  Warning: Edge detection failed for {crater_id}: {e}")
        edge_strength = 0.0
        edge_rim_radius = None

    # Step 3: Combine results and compute probability

    # Radius agreement
    if edge_rim_radius is not None:
        radius_diff = abs(radius_topo - edge_rim_radius)
        radius_agreement = np.exp(-radius_diff / (0.2 * radius_orig))

        # Weighted combination of radii
        # Trust topography more, but incorporate edge detection
        radius_final = 0.7 * radius_topo + 0.3 * edge_rim_radius
    else:
        radius_agreement = 0.5  # Neutral
        radius_final = radius_topo

    # Overall probability
    probability = compute_rim_probability(
        topo_quality=topo_quality,
        edge_strength=edge_strength,
        radius_agreement=radius_agreement,
        diameter_error=err[2] if len(err) > 2 else None
    )

    # Create final refined geometry
    # Use topographic center (most reliable)
    refined_geom = Point(center_topo.x, center_topo.y).buffer(radius_final)

    # Metadata
    metadata = {
        'err_x0': err[0] if len(err) > 0 else np.nan,
        'err_y0': err[1] if len(err) > 1 else np.nan,
        'err_r': err[2] if len(err) > 2 else np.nan,
        'topo_quality': topo_quality,
        'edge_strength': edge_strength,
        'radius_agreement': radius_agreement,
        'rim_probability': probability,
        'radius_orig': radius_orig,
        'radius_refined': radius_final,
        'center_shift': np.sqrt((center_topo.x - center_orig.x)**2 +
                               (center_topo.y - center_orig.y)**2)
    }

    return refined_geom, metadata


def refine_crater_rims(input_shapefile, dem_path, image_path, output_dir,
                      min_diameter=60, inner_radius=0.8, outer_radius=1.2,
                      remove_external_topo=True, plot_individual=False):
    """
    Refine crater rims using topography and computer vision.

    Args:
        input_shapefile: Path to input shapefile from Step 0
        dem_path: Path to DTM/DEM file
        image_path: Path to contrast image
        output_dir: Output directory for results
        min_diameter: Minimum crater diameter (meters)
        inner_radius: Inner search radius (fraction of R)
        outer_radius: Outer search radius (fraction of R)
        remove_external_topo: Remove regional topography
        plot_individual: Plot each crater individually

    Returns:
        dict: Output paths and statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Crater Rim Refinement Module")
    print("=" * 70)

    # Step 1: Read inputs
    print("\n[Step 1] Reading input data...")
    gdf_input = gpd.read_file(input_shapefile)
    print(f"  Loaded {len(gdf_input)} craters from shapefile")

    # Filter by diameter
    if 'diameter' in gdf_input.columns:
        gdf_input = gdf_input[gdf_input['diameter'] > min_diameter]
        print(f"  Filtered to {len(gdf_input)} craters > {min_diameter}m")

    dem_src = rio.open(dem_path)
    print(f"  DEM: {dem_src.width} x {dem_src.height} pixels")

    image_src = rio.open(image_path)
    print(f"  Image: {image_src.width} x {image_src.height} pixels")

    crs = gdf_input.crs

    # Step 2: Refine each crater
    print("\n[Step 2] Refining crater rims...")

    refined_geoms = []
    metadata_list = []

    for idx, row in gdf_input.iterrows():
        crater_id = row.get('UFID', f'crater_{idx}')
        geom = row.geometry

        print(f"  Processing {idx+1}/{len(gdf_input)}: {crater_id}")

        refined_geom, metadata = refine_single_crater(
            crater_geom=geom,
            crater_id=crater_id,
            dem_src=dem_src,
            image_src=image_src,
            crs=crs,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            remove_external_topo=remove_external_topo,
            plot=plot_individual
        )

        refined_geoms.append(refined_geom)
        metadata_list.append(metadata)

    # Step 3: Create output GeoDataFrame
    print("\n[Step 3] Creating output shapefile...")

    # Combine original data with new metadata
    gdf_refined = gdf_input.copy()
    gdf_refined['geometry'] = refined_geoms

    # Add new columns
    for key in metadata_list[0].keys():
        gdf_refined[key] = [m[key] for m in metadata_list]

    # Save shapefile
    shapefile_path = output_dir / 'craters_refined.shp'
    gdf_refined.to_file(shapefile_path)
    print(f"  Saved: {shapefile_path}")

    # Step 4: Generate visualizations
    print("\n[Step 4] Generating visualizations...")

    # Plot 1: Refined rim positions
    plot1_path = output_dir / 'craters_refined_positions.png'
    plot_refined_positions(image_path, gdf_refined, plot1_path)

    # Plot 2: CSFD
    plot2_path = output_dir / 'craters_refined_csfd.png'
    plot_csfd_refined(gdf_refined, plot2_path, area_km2=None)

    # Plot 3: Rim differences
    plot3_path = output_dir / 'craters_rim_differences.png'
    plot_rim_differences(image_path, gdf_input, gdf_refined, plot3_path)

    # Cleanup
    dem_src.close()
    image_src.close()

    # Statistics
    print("\n[Step 5] Computing statistics...")
    stats = {
        'total_craters': len(gdf_refined),
        'mean_probability': gdf_refined['rim_probability'].mean(),
        'high_confidence': (gdf_refined['rim_probability'] > 0.7).sum(),
        'medium_confidence': ((gdf_refined['rim_probability'] >= 0.5) &
                             (gdf_refined['rim_probability'] <= 0.7)).sum(),
        'low_confidence': (gdf_refined['rim_probability'] < 0.5).sum(),
        'mean_topo_quality': gdf_refined['topo_quality'].mean(),
        'mean_edge_strength': gdf_refined['edge_strength'].mean(),
        'mean_radius_change': (gdf_refined['radius_refined'] /
                              gdf_refined['radius_orig']).mean()
    }

    print(f"\n  Total craters: {stats['total_craters']}")
    print(f"  Mean rim probability: {stats['mean_probability']:.3f}")
    print(f"  High confidence (>0.7): {stats['high_confidence']}")
    print(f"  Medium confidence (0.5-0.7): {stats['medium_confidence']}")
    print(f"  Low confidence (<0.5): {stats['low_confidence']}")
    print(f"  Mean topographic quality: {stats['mean_topo_quality']:.3f}")
    print(f"  Mean edge strength: {stats['mean_edge_strength']:.3f}")
    print(f"  Mean radius change: {stats['mean_radius_change']:.3f}")

    print("\n" + "=" * 70)
    print("Rim Refinement Complete!")
    print("=" * 70)

    results = {
        'shapefile': shapefile_path,
        'refined_positions_plot': plot1_path,
        'csfd_plot': plot2_path,
        'differences_plot': plot3_path,
        'statistics': stats,
        'refined_data': gdf_refined
    }

    print(f"\nOutputs:")
    print(f"  Shapefile: {shapefile_path}")
    print(f"  Refined positions: {plot1_path}")
    print(f"  CSFD plot: {plot2_path}")
    print(f"  Differences plot: {plot3_path}")

    return results


def plot_refined_positions(image_path, gdf_refined, output_path):
    """
    Plot refined crater rim positions on contrast image.

    Args:
        image_path: Path to contrast image
        gdf_refined: GeoDataFrame with refined craters
        output_path: Path for output PNG
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # Read and display image
    with rio.open(image_path) as src:
        show(src, ax=ax, cmap='gray')

    # Plot craters colored by probability
    gdf_refined.plot(
        ax=ax,
        column='rim_probability',
        facecolor='none',
        edgecolor='red',
        linewidth=1.5,
        alpha=0.8,
        legend=True,
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        legend_kwds={'label': 'Rim Detection Probability',
                    'orientation': 'horizontal',
                    'pad': 0.05}
    )

    n_craters = len(gdf_refined)
    mean_prob = gdf_refined['rim_probability'].mean()

    ax.set_title(f'Refined Crater Rims, N = {n_craters}\n'
                f'Mean Probability = {mean_prob:.3f}',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_csfd_refined(gdf_refined, output_path, area_km2=None):
    """
    Create CSFD plot for refined craters.

    Args:
        gdf_refined: GeoDataFrame with refined crater data
        output_path: Path for output PNG
        area_km2: Survey area in km²
    """
    from .input_module import plot_csfd

    # Extract diameters from refined geometries
    diameters = 2 * np.sqrt(gdf_refined.geometry.area / np.pi)

    # Create temporary dataframe
    df_temp = pd.DataFrame({'diameter': diameters})

    plot_csfd(df_temp, output_path, area_km2=area_km2)


def plot_rim_differences(image_path, gdf_original, gdf_refined, output_path):
    """
    Plot showing differences between original and refined crater rims.

    Args:
        image_path: Path to contrast image
        gdf_original: GeoDataFrame with original craters
        gdf_refined: GeoDataFrame with refined craters
        output_path: Path for output PNG
    """
    fig, ax = plt.subplots(figsize=(14, 12))

    # Read and display image
    with rio.open(image_path) as src:
        show(src, ax=ax, cmap='gray', alpha=0.5)

    # Plot original rims (blue, dashed)
    gdf_original.plot(
        ax=ax,
        facecolor='none',
        edgecolor='blue',
        linewidth=1.5,
        linestyle='--',
        alpha=0.6,
        label='Original'
    )

    # Plot refined rims (red, solid)
    gdf_refined.plot(
        ax=ax,
        facecolor='none',
        edgecolor='red',
        linewidth=1.5,
        linestyle='-',
        alpha=0.8,
        label='Refined'
    )

    # Add arrows showing movement of centers
    for idx in range(min(len(gdf_original), len(gdf_refined))):
        orig_center = gdf_original.iloc[idx].geometry.centroid
        refined_center = gdf_refined.iloc[idx].geometry.centroid

        # Only draw arrow if movement is significant
        shift = np.sqrt((refined_center.x - orig_center.x)**2 +
                       (refined_center.y - orig_center.y)**2)

        if shift > 5:  # More than 5 meters
            ax.annotate('',
                       xy=(refined_center.x, refined_center.y),
                       xytext=(orig_center.x, orig_center.y),
                       arrowprops=dict(arrowstyle='->', color='yellow',
                                     lw=1.5, alpha=0.7))

    # Statistics text
    mean_shift = gdf_refined['center_shift'].mean()
    max_shift = gdf_refined['center_shift'].max()
    mean_radius_change = ((gdf_refined['radius_refined'] - gdf_refined['radius_orig']) /
                          gdf_refined['radius_orig'] * 100).mean()

    stats_text = (f'Mean center shift: {mean_shift:.1f} m\n'
                 f'Max center shift: {max_shift:.1f} m\n'
                 f'Mean radius change: {mean_radius_change:+.1f}%')

    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=10)

    ax.set_title('Crater Rim Refinement Comparison\n'
                'Blue (dashed) = Original, Red (solid) = Refined, '
                'Yellow arrows = Center shifts',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")

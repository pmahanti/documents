#!/usr/bin/env python3
"""
Crater Degradation Analysis and Age Estimation

This script analyzes lunar impact craters from GeoTIFF topography and image data,
refines crater rim positions, and estimates ages using diffusion-based degradation models.

Author: Generated for crater morphology analysis
Date: 2025-11-18
"""

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.transform import rowcol, xy
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from skimage import filters, feature, morphology
from skimage.transform import hough_circle, hough_circle_peaks
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

try:
    from cratermaker.morphology import diffusion_age
except ImportError:
    print("Warning: cratermaker not installed. Age estimation will use placeholder.")
    diffusion_age = None

# Import alternate topography degradation model
try:
    from topography_degradation_age import (
        TopographyDegradationAgeEstimator,
        estimate_age_topography_degradation
    )
    TOPO_DEGRADATION_AVAILABLE = True
except ImportError:
    print("Warning: topography_degradation_age module not found.")
    TOPO_DEGRADATION_AVAILABLE = False

# Import Chebyshev coefficient extraction
try:
    from chebyshev_coefficients import (
        ChebyshevProfileAnalyzer,
        extract_chebyshev_coefficients
    )
    CHEBYSHEV_AVAILABLE = True
except ImportError:
    print("Warning: chebyshev_coefficients module not found.")
    CHEBYSHEV_AVAILABLE = False


class CraterAgeAnalyzer:
    """Main class for crater age analysis using diffusion-based degradation."""

    def __init__(self, topo_path, image_path, shapefile_path, pixel_size_meters=None,
                 age_method='auto', diffusivity=5.0):
        """
        Initialize the crater age analyzer.

        Parameters:
        -----------
        topo_path : str
            Path to GeoTIFF topography raster
        image_path : str
            Path to GeoTIFF lunar image
        shapefile_path : str
            Path to shapefile with approximate crater rims
        pixel_size_meters : float, optional
            Pixel size in meters (if None, extracted from raster)
        age_method : str, optional
            Age estimation method: 'auto', 'cratermaker', 'topography_degradation', or 'both'
            Default: 'auto' (uses topography_degradation if available, else cratermaker, else fallback)
        diffusivity : float, optional
            Topographic diffusivity coefficient in m²/Myr (default 5.0)
            Used for topography_degradation method
        """
        self.topo_path = topo_path
        self.image_path = image_path
        self.shapefile_path = shapefile_path
        self.age_method = age_method
        self.diffusivity = diffusivity

        # Load data
        self.topo_src = rasterio.open(topo_path)
        self.image_src = rasterio.open(image_path)
        self.craters_gdf = gpd.read_file(shapefile_path)

        # Get pixel size
        if pixel_size_meters is None:
            self.pixel_size = abs(self.topo_src.transform[0])
        else:
            self.pixel_size = pixel_size_meters

        # Initialize topography degradation estimator if requested
        if age_method in ['topography_degradation', 'both', 'auto'] and TOPO_DEGRADATION_AVAILABLE:
            self.topo_deg_estimator = TopographyDegradationAgeEstimator(
                diffusivity=diffusivity
            )
        else:
            self.topo_deg_estimator = None

        # Initialize Chebyshev coefficient analyzer
        if CHEBYSHEV_AVAILABLE:
            self.chebyshev_analyzer = ChebyshevProfileAnalyzer(num_coefficients=17)
        else:
            self.chebyshev_analyzer = None

        print(f"Loaded {len(self.craters_gdf)} craters")
        print(f"Pixel size: {self.pixel_size} meters")
        print(f"Age method: {age_method}")
        print(f"Chebyshev analysis: {'enabled' if CHEBYSHEV_AVAILABLE else 'disabled'}")

    def refine_rim_position(self, geometry, topo_array, image_array, transform, search_radius=10):
        """
        Refine crater rim position using edge detection on image and topography.

        Parameters:
        -----------
        geometry : shapely.Polygon
            Approximate crater rim polygon
        topo_array : numpy.ndarray
            Topography data array
        image_array : numpy.ndarray
            Image data array
        transform : affine.Affine
            Raster transform
        search_radius : int
            Search radius in pixels for rim refinement

        Returns:
        --------
        refined_points : list of tuples
            Refined (x, y) rim coordinates
        """
        # Get approximate rim coordinates
        coords = np.array(geometry.exterior.coords[:-1])

        refined_points = []

        for coord in coords:
            # Convert geographic to pixel coordinates
            row, col = rowcol(transform, coord[0], coord[1])

            # Extract local window
            r_min = max(0, row - search_radius)
            r_max = min(topo_array.shape[0], row + search_radius)
            c_min = max(0, col - search_radius)
            c_max = min(topo_array.shape[1], col + search_radius)

            topo_window = topo_array[r_min:r_max, c_min:c_max]
            image_window = image_array[r_min:r_max, c_min:c_max]

            if topo_window.size == 0 or image_window.size == 0:
                refined_points.append(coord)
                continue

            # Edge detection on both topography and image
            topo_edges = filters.sobel(topo_window)
            image_edges = filters.sobel(image_window)

            # Combine edges (weighted)
            combined_edges = 0.6 * topo_edges + 0.4 * image_edges

            # Find maximum edge response
            edge_max_idx = np.unravel_index(np.argmax(combined_edges), combined_edges.shape)

            # Convert back to pixel coordinates
            refined_row = r_min + edge_max_idx[0]
            refined_col = c_min + edge_max_idx[1]

            # Convert to geographic coordinates
            refined_x, refined_y = xy(transform, refined_row, refined_col)
            refined_points.append((refined_x, refined_y))

        return refined_points

    def fit_circle_to_points(self, points):
        """
        Fit a circle to rim points to find center and diameter.

        Parameters:
        -----------
        points : list of tuples
            (x, y) coordinates of rim points

        Returns:
        --------
        center : tuple
            (cx, cy) center coordinates
        radius : float
            Circle radius
        """
        points = np.array(points)

        def calc_R(xc, yc):
            """Calculate distance of each point from center (xc, yc)"""
            return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)

        def objective(c):
            """Minimize variance of radii"""
            Ri = calc_R(*c)
            return np.sum((Ri - Ri.mean())**2)

        # Initial guess: centroid
        center_init = points.mean(axis=0)

        # Optimize
        result = minimize(objective, center_init, method='Nelder-Mead')

        cx, cy = result.x
        radii = calc_R(cx, cy)
        radius = radii.mean()

        return (cx, cy), radius

    def extract_crater_region(self, center, radius, buffer_factor=1.5):
        """
        Extract crater region from rasters with buffer.

        Parameters:
        -----------
        center : tuple
            (cx, cy) center coordinates
        radius : float
            Crater radius
        buffer_factor : float
            Multiplier for extraction region (default 1.5 for -1.5D to +1.5D)

        Returns:
        --------
        topo_crop : numpy.ndarray
            Cropped topography
        image_crop : numpy.ndarray
            Cropped image
        crop_transform : affine.Affine
            Transform for cropped region
        """
        # Create circular mask geometry
        buffer_radius = radius * buffer_factor
        circle = Point(center).buffer(buffer_radius)

        # Extract from topography
        topo_crop, topo_transform = mask(self.topo_src, [circle], crop=True)
        topo_crop = topo_crop[0]  # First band

        # Extract from image
        image_crop, image_transform = mask(self.image_src, [circle], crop=True)
        image_crop = image_crop[0]  # First band

        return topo_crop, image_crop, topo_transform

    def correct_tilt(self, topo_array):
        """
        Remove first-order (planar) tilt from topography.

        Parameters:
        -----------
        topo_array : numpy.ndarray
            Topography data

        Returns:
        --------
        corrected : numpy.ndarray
            Tilt-corrected topography
        plane_params : tuple
            (a, b, c) plane parameters z = ax + by + c
        """
        rows, cols = topo_array.shape

        # Create coordinate grids
        x = np.arange(cols)
        y = np.arange(rows)
        X, Y = np.meshgrid(x, y)

        # Flatten arrays
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = topo_array.flatten()

        # Remove NaN values
        valid_mask = ~np.isnan(Z_flat)
        X_valid = X_flat[valid_mask]
        Y_valid = Y_flat[valid_mask]
        Z_valid = Z_flat[valid_mask]

        if len(Z_valid) == 0:
            return topo_array, (0, 0, 0)

        # Fit plane: z = ax + by + c
        A = np.column_stack([X_valid, Y_valid, np.ones(len(X_valid))])
        plane_params, residuals, rank, s = np.linalg.lstsq(A, Z_valid, rcond=None)

        a, b, c = plane_params

        # Subtract plane
        plane = a * X + b * Y + c
        corrected = topo_array - plane

        return corrected, (a, b, c)

    def extract_radial_profiles(self, topo_array, center_row, center_col, diameter_pixels, num_profiles=8):
        """
        Extract radial elevation profiles at equally spaced angles.

        Parameters:
        -----------
        topo_array : numpy.ndarray
            Tilt-corrected topography
        center_row, center_col : int
            Center position in pixels
        diameter_pixels : float
            Crater diameter in pixels
        num_profiles : int
            Number of radial profiles (default 8)

        Returns:
        --------
        profiles : list of dict
            Each dict contains 'angle', 'distance', 'elevation'
        """
        radius_pixels = diameter_pixels / 2.0

        # Profile extends from -1.5D to +1.5D
        max_distance = 1.5 * diameter_pixels

        profiles = []

        for i in range(num_profiles):
            angle = i * 45  # Degrees
            angle_rad = np.deg2rad(angle)

            # Sample points along this angle
            distances = np.linspace(-max_distance, max_distance, 300)

            elevations = []
            valid_distances = []

            for dist in distances:
                # Calculate pixel position
                row = int(center_row + dist * np.sin(angle_rad))
                col = int(center_col + dist * np.cos(angle_rad))

                # Check bounds
                if 0 <= row < topo_array.shape[0] and 0 <= col < topo_array.shape[1]:
                    elev = topo_array[row, col]
                    if not np.isnan(elev):
                        elevations.append(elev)
                        valid_distances.append(dist)

            if len(elevations) > 0:
                profiles.append({
                    'angle': angle,
                    'distance': np.array(valid_distances),
                    'elevation': np.array(elevations)
                })

        return profiles

    def estimate_age_from_profiles(self, profiles, diameter_meters, pixel_size):
        """
        Estimate crater age using diffusion-based degradation model.

        Parameters:
        -----------
        profiles : list of dict
            Radial elevation profiles
        diameter_meters : float
            Crater diameter in meters
        pixel_size : float
            Pixel size in meters

        Returns:
        --------
        age : float or dict
            Estimated age in years (Ga if applicable), or dict with multiple estimates
        depth : float
            Current depth in meters
        degradation_param : float or dict
            Degradation parameter(s)
        """
        if len(profiles) == 0:
            return None, None, None

        # Calculate average depth and rim height
        depths = []
        rim_heights = []

        for profile in profiles:
            dist = profile['distance']
            elev = profile['elevation']

            # Find center region (crater floor)
            center_mask = np.abs(dist) < 0.3 * (diameter_meters / pixel_size)
            if np.sum(center_mask) > 0:
                floor_elev = np.median(elev[center_mask])
            else:
                floor_elev = np.min(elev)

            # Find rim region
            rim_mask = np.abs(np.abs(dist) - (diameter_meters / pixel_size / 2)) < 0.1 * (diameter_meters / pixel_size)
            if np.sum(rim_mask) > 0:
                rim_elev = np.max(elev[rim_mask])
            else:
                rim_elev = np.max(elev)

            depth = rim_elev - floor_elev
            depths.append(depth)
            rim_heights.append(rim_elev)

        avg_depth = np.median(depths) * pixel_size  # Convert to meters
        avg_rim_height = np.median(rim_heights) * pixel_size if rim_heights else None

        # Determine which method(s) to use
        use_topo_deg = False
        use_cratermaker = False

        if self.age_method == 'auto':
            # Prefer topography degradation, fall back to others
            use_topo_deg = TOPO_DEGRADATION_AVAILABLE and self.topo_deg_estimator is not None
            use_cratermaker = (not use_topo_deg) and (diffusion_age is not None)
        elif self.age_method == 'topography_degradation':
            use_topo_deg = TOPO_DEGRADATION_AVAILABLE and self.topo_deg_estimator is not None
        elif self.age_method == 'cratermaker':
            use_cratermaker = diffusion_age is not None
        elif self.age_method == 'both':
            use_topo_deg = TOPO_DEGRADATION_AVAILABLE and self.topo_deg_estimator is not None
            use_cratermaker = diffusion_age is not None

        results = {}

        # Try topography degradation method
        if use_topo_deg:
            try:
                topo_result = self.topo_deg_estimator.estimate_age_from_profiles(
                    profiles, diameter_meters, pixel_size,
                    observed_depth=avg_depth,
                    observed_rim_height=avg_rim_height
                )
                if topo_result is not None:
                    results['topography_degradation'] = topo_result
            except Exception as e:
                print(f"Warning: Topography degradation estimation failed: {e}")

        # Try cratermaker method
        if use_cratermaker:
            try:
                age_result = diffusion_age(
                    diameter=diameter_meters,
                    depth=avg_depth,
                    profiles=profiles
                )

                if isinstance(age_result, dict):
                    age = age_result.get('age', None)
                    degradation = age_result.get('degradation', None)
                else:
                    age = age_result
                    degradation = None

                results['cratermaker'] = {
                    'age_ga': age,
                    'degradation': degradation,
                    'method': 'cratermaker'
                }
            except Exception as e:
                print(f"Warning: cratermaker estimation failed: {e}")

        # Fallback: simple depth-diameter ratio method
        if len(results) == 0:
            age = self._simple_age_estimate(diameter_meters, avg_depth)
            return age, avg_depth, None

        # Return results based on what was requested
        if self.age_method == 'both' and len(results) > 1:
            # Return both methods
            return results, avg_depth, results
        elif len(results) > 0:
            # Return the first available result
            method_name = list(results.keys())[0]
            result = results[method_name]

            if 'age_ga' in result:
                age = f"{result['age_ga']:.2f} Ga ({method_name})"
            else:
                age = f"{result.get('degradation_state', 'Unknown')} ({method_name})"

            return age, avg_depth, result
        else:
            age = self._simple_age_estimate(diameter_meters, avg_depth)
            return age, avg_depth, None

    def _simple_age_estimate(self, diameter, depth):
        """
        Simple age estimation based on depth-diameter ratio.

        Fresh lunar craters: d/D ≈ 0.2
        As craters age, d/D decreases due to erosion/infilling

        Parameters:
        -----------
        diameter : float
            Crater diameter in meters
        depth : float
            Crater depth in meters

        Returns:
        --------
        age_category : str
            Qualitative age estimate
        """
        if diameter <= 0:
            return "Unknown"

        d_D_ratio = depth / diameter

        # Classify based on depth/diameter ratio
        # These are approximate thresholds for lunar craters
        if d_D_ratio > 0.18:
            return "Fresh (<0.1 Ga)"
        elif d_D_ratio > 0.12:
            return "Young (0.1-1 Ga)"
        elif d_D_ratio > 0.08:
            return "Mature (1-3 Ga)"
        elif d_D_ratio > 0.04:
            return "Old (3-4 Ga)"
        else:
            return "Very Old (>4 Ga)"

    def process_all_craters(self):
        """
        Process all craters in the shapefile.

        Returns:
        --------
        results_gdf : GeoDataFrame
            GeoDataFrame with age labels and refined parameters
        """
        results = []

        for idx, row in self.craters_gdf.iterrows():
            print(f"\nProcessing crater {idx + 1}/{len(self.craters_gdf)}...")

            try:
                geometry = row.geometry

                # Get bounding box for extraction
                bounds = geometry.bounds
                minx, miny, maxx, maxy = bounds

                # Extract region
                box = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])

                topo_data, topo_transform = mask(self.topo_src, [box], crop=True)
                image_data, image_transform = mask(self.image_src, [box], crop=True)

                topo_data = topo_data[0]
                image_data = image_data[0]

                # Refine rim position
                refined_points = self.refine_rim_position(
                    geometry, topo_data, image_data, topo_transform
                )

                # Fit circle to refined points
                center, radius = self.fit_circle_to_points(refined_points)
                diameter_meters = 2 * radius

                # Extract crater region with buffer
                topo_crop, image_crop, crop_transform = self.extract_crater_region(
                    center, radius, buffer_factor=1.5
                )

                # Correct tilt
                topo_corrected, plane_params = self.correct_tilt(topo_crop)

                # Get center in pixel coordinates of cropped array
                center_row, center_col = rowcol(crop_transform, center[0], center[1])
                diameter_pixels = diameter_meters / self.pixel_size

                # Extract radial profiles
                profiles = self.extract_radial_profiles(
                    topo_corrected, center_row, center_col, diameter_pixels, num_profiles=8
                )

                # Extract Chebyshev coefficients from profiles
                # Profiles are normalized by diameter: -D to +D maps to -1 to +1
                chebyshev_matrix = None
                chebyshev_analysis = None
                if self.chebyshev_analyzer is not None and len(profiles) > 0:
                    try:
                        chebyshev_matrix, chebyshev_analysis, cheb_metadata = \
                            extract_chebyshev_coefficients(
                                profiles,
                                diameter=diameter_pixels,  # Normalize by crater diameter
                                num_coefficients=17
                            )
                        print(f"  Chebyshev: {chebyshev_matrix.shape[0]}×{chebyshev_matrix.shape[1]} coefficient matrix")
                    except Exception as e:
                        print(f"  Warning: Chebyshev extraction failed: {e}")

                # Estimate age
                age, depth, degradation = self.estimate_age_from_profiles(
                    profiles, diameter_meters, self.pixel_size
                )

                # Store results
                result = {
                    'geometry': Point(center),
                    'original_geom': geometry,
                    'center_x': center[0],
                    'center_y': center[1],
                    'diameter_m': diameter_meters,
                    'depth_m': depth,
                    'age': age,
                    'degradation': degradation,
                    'num_profiles': len(profiles),
                    'chebyshev_matrix': chebyshev_matrix,
                    'chebyshev_analysis': chebyshev_analysis
                }

                # Add individual Chebyshev coefficient columns for shapefile
                if chebyshev_matrix is not None and chebyshev_analysis is not None:
                    # Add mean coefficients (easier to store in shapefile)
                    mean_coeffs = chebyshev_analysis['mean_coefficients']
                    for i in range(min(17, len(mean_coeffs))):
                        result[f'C{i}_mean'] = float(mean_coeffs[i])

                    # Add key crater characteristics
                    crater_chars = chebyshev_analysis.get('crater_characteristics', {})
                    result['depth_indicator'] = float(crater_chars.get('mean_depth_indicator', 0))
                    result['central_peak_idx'] = float(crater_chars.get('central_peak_indicator', 0))
                    result['asymmetry_idx'] = float(crater_chars.get('asymmetry_index', 0))
                    result['profile_consistency'] = float(crater_chars.get('profile_consistency', 0))

                # Copy original attributes
                for col in self.craters_gdf.columns:
                    if col != 'geometry':
                        result[col] = row[col]

                results.append(result)

                print(f"  Center: ({center[0]:.2f}, {center[1]:.2f})")
                print(f"  Diameter: {diameter_meters:.2f} m")
                print(f"  Depth: {depth:.2f} m" if depth else "  Depth: N/A")
                print(f"  Age: {age}")

            except Exception as e:
                print(f"  Error processing crater {idx}: {e}")
                # Store minimal result
                result = {
                    'geometry': row.geometry.centroid,
                    'original_geom': row.geometry,
                    'age': 'Error',
                    'error': str(e)
                }
                results.append(result)

        # Create GeoDataFrame
        results_gdf = gpd.GeoDataFrame(results, crs=self.craters_gdf.crs)

        return results_gdf

    def visualize_results(self, results_gdf, output_path='crater_ages_visualization.png', dpi=300):
        """
        Create visualization with age labels marked on image.

        Parameters:
        -----------
        results_gdf : GeoDataFrame
            Results from process_all_craters
        output_path : str
            Output image path
        dpi : int
            Image resolution
        """
        fig, ax = plt.subplots(figsize=(16, 12))

        # Read full image
        image_data = self.image_src.read(1)

        # Display image
        extent = [
            self.image_src.bounds.left,
            self.image_src.bounds.right,
            self.image_src.bounds.bottom,
            self.image_src.bounds.top
        ]

        ax.imshow(image_data, cmap='gray', extent=extent, origin='upper')

        # Plot craters
        for idx, row in results_gdf.iterrows():
            if 'center_x' in row and 'center_y' in row and 'diameter_m' in row:
                center = (row['center_x'], row['center_y'])
                radius = row['diameter_m'] / 2

                # Draw circle
                circle = Circle(center, radius, fill=False, color='red', linewidth=2)
                ax.add_patch(circle)

                # Add age label
                age_text = str(row['age'])
                ax.text(
                    center[0], center[1], age_text,
                    color='yellow', fontsize=10, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
                )

        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        ax.set_title('Lunar Crater Age Estimation', fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"\nVisualization saved to: {output_path}")

    def save_chebyshev_matrices(self, results_gdf, output_dir='output'):
        """
        Save Chebyshev coefficient matrices to separate files.

        Parameters:
        -----------
        results_gdf : GeoDataFrame
            Results containing Chebyshev matrices
        output_dir : str
            Output directory for matrix files
        """
        import os
        import json

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save individual matrices as numpy files
        matrices_saved = 0
        for idx, row in results_gdf.iterrows():
            if 'chebyshev_matrix' in row and row['chebyshev_matrix'] is not None:
                matrix = row['chebyshev_matrix']

                # Save as numpy file
                matrix_file = os.path.join(output_dir, f'crater_{idx}_chebyshev_17x8.npy')
                np.save(matrix_file, matrix)

                # Also save as CSV for easier viewing
                csv_file = os.path.join(output_dir, f'crater_{idx}_chebyshev_17x8.csv')
                np.savetxt(csv_file, matrix, delimiter=',',
                          header='Profile angles: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°\n' +
                                 'Rows: C0-C16, Columns: 8 profiles',
                          fmt='%.6f')

                matrices_saved += 1

        # Save summary with all matrices in one file
        if matrices_saved > 0:
            summary_file = os.path.join(output_dir, 'all_craters_chebyshev.npz')
            matrices_dict = {}
            for idx, row in results_gdf.iterrows():
                if 'chebyshev_matrix' in row and row['chebyshev_matrix'] is not None:
                    matrices_dict[f'crater_{idx}'] = row['chebyshev_matrix']

            np.savez(summary_file, **matrices_dict)
            print(f"  Chebyshev matrices saved: {matrices_saved} craters")
            print(f"  Summary file: {summary_file}")
            print(f"  Individual files: {output_dir}/crater_*_chebyshev_17x8.*")

    def export_to_matlab(self, results_gdf, output_file='crater_analysis.mat'):
        """
        Export profiles and Chebyshev coefficients to MATLAB .mat file.

        Parameters:
        -----------
        results_gdf : GeoDataFrame
            Results containing profiles and Chebyshev matrices
        output_file : str
            Output .mat file path
        """
        try:
            from scipy.io import savemat
        except ImportError:
            print("Warning: scipy.io not available. Cannot export to MATLAB format.")
            return

        # Prepare data structure for MATLAB
        matlab_data = {
            'num_craters': len(results_gdf),
            'crater_info': []
        }

        # Arrays for all craters
        all_chebyshev = []
        all_diameters = []
        all_depths = []
        all_ages = []

        for idx, row in results_gdf.iterrows():
            crater_data = {
                'crater_id': idx,
                'center_x': row.get('center_x', np.nan),
                'center_y': row.get('center_y', np.nan),
                'diameter_m': row.get('diameter_m', np.nan),
                'depth_m': row.get('depth_m', np.nan),
                'age': str(row.get('age', 'Unknown'))
            }

            # Add Chebyshev matrix if available
            if 'chebyshev_matrix' in row and row['chebyshev_matrix'] is not None:
                all_chebyshev.append(row['chebyshev_matrix'])
                crater_data['has_chebyshev'] = True
            else:
                all_chebyshev.append(np.zeros((17, 8)))
                crater_data['has_chebyshev'] = False

            all_diameters.append(row.get('diameter_m', np.nan))
            all_depths.append(row.get('depth_m', np.nan))

            matlab_data['crater_info'].append(crater_data)

        # Stack all Chebyshev matrices into a 3D array
        if all_chebyshev:
            matlab_data['chebyshev_coefficients'] = np.stack(all_chebyshev, axis=2)  # 17 x 8 x N

        matlab_data['diameters'] = np.array(all_diameters)
        matlab_data['depths'] = np.array(all_depths)

        # Save to .mat file
        savemat(output_file, matlab_data, do_compression=True)
        print(f"\nMATLAB export saved to: {output_file}")
        print(f"  Chebyshev matrix shape: {matlab_data['chebyshev_coefficients'].shape if 'chebyshev_coefficients' in matlab_data else 'N/A'}")

    def save_results(self, results_gdf, output_shapefile='crater_ages.shp',
                    save_chebyshev=True, save_matlab=False):
        """
        Save results to shapefile, Chebyshev matrices, and optionally MATLAB format.

        Parameters:
        -----------
        results_gdf : GeoDataFrame
            Results to save
        output_shapefile : str
            Output shapefile path
        save_chebyshev : bool
            Whether to save Chebyshev matrices to separate files (default True)
        save_matlab : bool
            Whether to export to MATLAB .mat format (default False)
        """
        # Create a copy for shapefile (remove non-serializable columns)
        gdf_copy = results_gdf.copy()

        # Remove matrix columns (they'll be saved separately)
        cols_to_remove = ['chebyshev_matrix', 'chebyshev_analysis']
        for col in cols_to_remove:
            if col in gdf_copy.columns:
                gdf_copy = gdf_copy.drop(columns=[col])

        # Ensure all remaining columns are serializable
        for col in gdf_copy.columns:
            if gdf_copy[col].dtype == 'object':
                gdf_copy[col] = gdf_copy[col].astype(str)

        gdf_copy.to_file(output_shapefile)
        print(f"\nResults saved to: {output_shapefile}")

        # Save Chebyshev matrices separately
        if save_chebyshev:
            import os
            output_dir = os.path.join(os.path.dirname(output_shapefile) or '.', 'chebyshev_coefficients')
            self.save_chebyshev_matrices(results_gdf, output_dir)

        # Export to MATLAB if requested
        if save_matlab:
            import os
            matlab_file = os.path.join(os.path.dirname(output_shapefile) or '.', 'crater_analysis.mat')
            self.export_to_matlab(results_gdf, matlab_file)

    def close(self):
        """Close raster datasets."""
        self.topo_src.close()
        self.image_src.close()


def main():
    """Main execution function with example usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Crater Age Analysis')
    parser.add_argument('--topo', required=True, help='Path to topography GeoTIFF')
    parser.add_argument('--image', required=True, help='Path to image GeoTIFF')
    parser.add_argument('--shapefile', required=True, help='Path to crater rim shapefile')
    parser.add_argument('--output-shp', default='crater_ages.shp', help='Output shapefile path')
    parser.add_argument('--output-img', default='crater_ages_visualization.png', help='Output image path')
    parser.add_argument('--pixel-size', type=float, default=None, help='Pixel size in meters')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = CraterAgeAnalyzer(
        topo_path=args.topo,
        image_path=args.image,
        shapefile_path=args.shapefile,
        pixel_size_meters=args.pixel_size
    )

    # Process all craters
    results = analyzer.process_all_craters()

    # Save results
    analyzer.save_results(results, args.output_shp)

    # Create visualization
    analyzer.visualize_results(results, args.output_img)

    # Close
    analyzer.close()

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()

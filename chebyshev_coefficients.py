#!/usr/bin/env python3
"""
Chebyshev Coefficient Extraction for Crater Profiles

This module implements Chebyshev polynomial fitting to crater radial profiles
to extract morphological descriptors. Chebyshev coefficients provide a standardized
mathematical framework for quantitative crater characterization.

Based on research showing that most crater elevation profiles can be well
represented using 17 Chebyshev coefficients (C0-C16), enabling:
- Depth-to-diameter ratio inference (primarily C2)
- Central peak detection (C4, C8)
- Asymmetry analysis (odd-numbered coefficients)
- Degradation state assessment

References:
- LROC team findings on crater morphology quantification
- "A standardized approach for quantitative characterization of impact
  crater topography" (ScienceDirect)
"""

import numpy as np
from numpy.polynomial import chebyshev as cheb
import warnings


class ChebyshevProfileAnalyzer:
    """
    Extracts Chebyshev coefficients from crater elevation profiles.

    Provides standardized mathematical representation of crater morphology
    using Chebyshev polynomial expansion.
    """

    def __init__(self, num_coefficients=17):
        """
        Initialize Chebyshev profile analyzer.

        Parameters:
        -----------
        num_coefficients : int
            Number of Chebyshev coefficients to extract (default 17, C0-C16)
        """
        self.num_coefficients = num_coefficients

    def normalize_profile(self, distance, elevation, diameter=None):
        """
        Normalize profile for Chebyshev fitting.

        Normalizes distance by crater diameter D so profiles extend from -1 to +1
        (representing -D to +D). Elevation is centered at 0.

        Parameters:
        -----------
        distance : ndarray
            Radial distances from crater center (in pixels or meters)
        elevation : ndarray
            Elevation values
        diameter : float, optional
            Crater diameter (in same units as distance)
            If None, uses max extent of distance data

        Returns:
        --------
        x_norm : ndarray
            Normalized distances in [-1, 1], where ±1 corresponds to ±D
        y_norm : ndarray
            Normalized elevations (divided by diameter)
        mean_elev : float
            Mean elevation (for denormalization)
        diameter_used : float
            Diameter value used for normalization
        """
        # Determine diameter for normalization
        if diameter is None:
            # Use the full extent of the distance data
            diameter_used = np.max(np.abs(distance))
            if diameter_used < 1e-10:
                diameter_used = 1.0
        else:
            diameter_used = diameter

        # Normalize distance by diameter: x_norm = distance / D
        # This maps -D to -1 and +D to +1
        x_norm = distance / diameter_used

        # Clip to [-1, 1] range (in case profiles extend slightly beyond D)
        x_norm = np.clip(x_norm, -1.0, 1.0)

        # Center elevation at 0 and normalize by diameter
        mean_elev = np.mean(elevation)
        y_norm = (elevation - mean_elev) / diameter_used

        return x_norm, y_norm, mean_elev, diameter_used

    def fit_chebyshev(self, x, y, degree=16):
        """
        Fit Chebyshev polynomial to profile data.

        Parameters:
        -----------
        x : ndarray
            Normalized x coordinates (should be in [-1, 1])
        y : ndarray
            Normalized y values
        degree : int
            Polynomial degree (default 16 for 17 coefficients)

        Returns:
        --------
        coefficients : ndarray
            Chebyshev coefficients [C0, C1, ..., C16]
        """
        if len(x) < degree + 1:
            # Not enough points, pad with zeros
            coefficients = np.zeros(degree + 1)
            if len(x) > 0:
                # Fit what we can
                actual_degree = min(degree, len(x) - 1)
                coef = cheb.chebfit(x, y, actual_degree)
                coefficients[:len(coef)] = coef
            return coefficients

        # Fit Chebyshev polynomial
        try:
            coefficients = cheb.chebfit(x, y, degree)
        except Exception as e:
            warnings.warn(f"Chebyshev fitting failed: {e}. Returning zeros.")
            coefficients = np.zeros(degree + 1)

        return coefficients

    def extract_coefficients_from_profile(self, distance, elevation, diameter=None, normalize=True):
        """
        Extract Chebyshev coefficients from a single radial profile.

        Parameters:
        -----------
        distance : ndarray
            Radial distance values from crater center
        elevation : ndarray
            Elevation values
        diameter : float, optional
            Crater diameter for normalization (in same units as distance)
        normalize : bool
            Whether to normalize the profile first (default True)

        Returns:
        --------
        coefficients : ndarray
            Array of Chebyshev coefficients [C0, C1, ..., C16]
        metadata : dict
            Dictionary with normalization parameters and fit quality
        """
        # Remove NaN values
        valid_mask = ~(np.isnan(distance) | np.isnan(elevation))
        dist_clean = distance[valid_mask]
        elev_clean = elevation[valid_mask]

        if len(dist_clean) < 5:
            # Not enough valid points
            return np.zeros(self.num_coefficients), {
                'valid_points': len(dist_clean),
                'error': 'insufficient_data'
            }

        # Normalize if requested
        if normalize:
            x_norm, y_norm, mean_elev, diameter_used = self.normalize_profile(
                dist_clean, elev_clean, diameter=diameter
            )
        else:
            x_norm = dist_clean
            y_norm = elev_clean
            mean_elev = 0.0
            diameter_used = diameter if diameter is not None else 1.0

        # Fit Chebyshev polynomial
        coefficients = self.fit_chebyshev(x_norm, y_norm, degree=self.num_coefficients-1)

        # Calculate fit quality (RMS error)
        y_fit = cheb.chebval(x_norm, coefficients)
        rms_error = np.sqrt(np.mean((y_norm - y_fit)**2))

        metadata = {
            'valid_points': len(dist_clean),
            'mean_elevation': mean_elev,
            'diameter_used': diameter_used,
            'rms_error': rms_error,
            'normalized': normalize
        }

        return coefficients, metadata

    def extract_coefficients_from_profiles(self, profiles, diameter=None):
        """
        Extract Chebyshev coefficients from multiple radial profiles.

        Parameters:
        -----------
        profiles : list of dict
            List of profile dictionaries with 'distance' and 'elevation' keys
        diameter : float, optional
            Crater diameter for normalization (in pixels or meters)
            If None, will use the maximum distance extent

        Returns:
        --------
        coef_matrix : ndarray
            Matrix of shape (num_coefficients, num_profiles)
            Each column contains coefficients for one profile
        metadata_list : list of dict
            List of metadata dictionaries for each profile
        """
        num_profiles = len(profiles)
        coef_matrix = np.zeros((self.num_coefficients, num_profiles))
        metadata_list = []

        for i, profile in enumerate(profiles):
            distance = profile['distance']
            elevation = profile['elevation']

            coeffs, metadata = self.extract_coefficients_from_profile(
                distance, elevation, diameter=diameter, normalize=True
            )

            coef_matrix[:, i] = coeffs
            metadata['profile_index'] = i
            metadata['profile_angle'] = profile.get('angle', i * 45)
            metadata_list.append(metadata)

        return coef_matrix, metadata_list

    def interpret_coefficients(self, coefficients):
        """
        Provide physical interpretation of Chebyshev coefficients.

        Parameters:
        -----------
        coefficients : ndarray
            Chebyshev coefficients (C0-C16)

        Returns:
        --------
        interpretation : dict
            Dictionary with physical interpretations
        """
        if len(coefficients) < 17:
            return {'error': 'Need at least 17 coefficients'}

        interpretation = {
            'C0': {
                'value': coefficients[0],
                'meaning': 'Mean elevation (baseline offset)'
            },
            'C1': {
                'value': coefficients[1],
                'meaning': 'Linear trend (asymmetry in slope)'
            },
            'C2': {
                'value': coefficients[2],
                'meaning': 'Depth-to-diameter related (curvature)'
            },
            'C3': {
                'value': coefficients[3],
                'meaning': 'Asymmetry in bowl shape'
            },
            'C4': {
                'value': coefficients[4],
                'meaning': 'Central peak indicator (with C8)'
            },
            'C8': {
                'value': coefficients[8],
                'meaning': 'Central peak indicator (with C4)'
            },
            'odd_sum': {
                'value': np.sum(np.abs(coefficients[1::2])),
                'meaning': 'Total asymmetry indicator'
            },
            'even_sum': {
                'value': np.sum(np.abs(coefficients[0::2])),
                'meaning': 'Symmetric features'
            },
            'central_peak_index': {
                'value': abs(coefficients[4]) + abs(coefficients[8]),
                'meaning': 'Combined central peak indicator'
            }
        }

        return interpretation

    def analyze_coefficient_matrix(self, coef_matrix):
        """
        Analyze matrix of Chebyshev coefficients from multiple profiles.

        Parameters:
        -----------
        coef_matrix : ndarray
            Matrix of shape (num_coefficients, num_profiles)

        Returns:
        --------
        analysis : dict
            Statistical analysis of coefficients across profiles
        """
        num_coef, num_prof = coef_matrix.shape

        analysis = {
            'mean_coefficients': np.mean(coef_matrix, axis=1),
            'std_coefficients': np.std(coef_matrix, axis=1),
            'median_coefficients': np.median(coef_matrix, axis=1),
            'coefficient_range': np.ptp(coef_matrix, axis=1),  # peak-to-peak
            'asymmetry_variance': np.var(coef_matrix[1::2, :], axis=1),  # variance of odd coeffs
            'num_profiles': num_prof,
            'num_coefficients': num_coef
        }

        # Overall crater characteristics
        mean_coefs = analysis['mean_coefficients']

        analysis['crater_characteristics'] = {
            'mean_depth_indicator': mean_coefs[2] if len(mean_coefs) > 2 else 0,
            'central_peak_indicator': (abs(mean_coefs[4]) + abs(mean_coefs[8]))
                                       if len(mean_coefs) > 8 else 0,
            'asymmetry_index': np.sum(np.abs(mean_coefs[1::2])),
            'profile_consistency': np.mean(analysis['std_coefficients'])
        }

        return analysis


def extract_chebyshev_coefficients(profiles, diameter=None, num_coefficients=17):
    """
    Convenience function to extract Chebyshev coefficients from profiles.

    Parameters:
    -----------
    profiles : list of dict
        List of profile dictionaries with 'distance' and 'elevation' keys
    diameter : float, optional
        Crater diameter for normalization (in pixels or meters)
        Profiles are normalized so -D to +D maps to -1 to +1
    num_coefficients : int
        Number of coefficients to extract (default 17)

    Returns:
    --------
    coef_matrix : ndarray
        Matrix of shape (num_coefficients, num_profiles)
    analysis : dict
        Statistical analysis of coefficients
    metadata_list : list of dict
        Metadata for each profile
    """
    analyzer = ChebyshevProfileAnalyzer(num_coefficients=num_coefficients)

    coef_matrix, metadata_list = analyzer.extract_coefficients_from_profiles(
        profiles, diameter=diameter
    )
    analysis = analyzer.analyze_coefficient_matrix(coef_matrix)

    return coef_matrix, analysis, metadata_list


# Example usage
if __name__ == '__main__':
    print("Chebyshev Coefficient Extraction Module")
    print("=" * 60)
    print("\nThis module provides standardized Chebyshev polynomial fitting")
    print("for crater profile analysis.")
    print("\nKey features:")
    print("  - Extracts 17 Chebyshev coefficients (C0-C16)")
    print("  - Returns 17×8 matrix for 8 radial profiles")
    print("  - Provides morphological interpretation")
    print("  - Analyzes profile consistency and asymmetry")
    print("\nCoefficient meanings:")
    print("  C0: Mean elevation")
    print("  C2: Depth-to-diameter ratio")
    print("  C4, C8: Central peak indicators")
    print("  Odd coefficients: Asymmetry")
    print("  Even coefficients: Symmetric features")

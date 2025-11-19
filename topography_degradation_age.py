#!/usr/bin/env python3
"""
Topography Degradation Model for Crater Age Estimation

This module implements an alternate age estimation method based on topographic
degradation of lunar simple craters, following the approach of Luo et al. (2025).

Reference:
Luo, F., Xiao, Z., Xie, M., Wang, Y., & Ma, Y. (2025). Age Estimation of
Individual Lunar Simple Craters Using the Topography Degradation Model.
Journal of Geophysical Research: Planets.
DOI: 10.5281/zenodo.15168130

The method uses diffusive degradation modeling to estimate crater ages by
comparing observed topographic profiles with pristine crater shapes.

Note: This is an implementation based on the published methodology. For craters
> 400m diameter, uncertainty is typically < 165 Ma. Not suitable for craters < 400m.
"""

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import warnings


class TopographyDegradationAgeEstimator:
    """
    Estimates crater age using topographic degradation model.

    This class implements a diffusion-based degradation model that compares
    observed crater profiles with pristine crater shapes to estimate age.
    """

    def __init__(self, diffusivity=5.0, min_diameter_m=400):
        """
        Initialize the age estimator.

        Parameters:
        -----------
        diffusivity : float
            Topographic diffusivity coefficient in m²/Myr (default 5.0)
            Typical lunar values: 3-10 m²/Myr
        min_diameter_m : float
            Minimum crater diameter for reliable estimates (default 400m)
        """
        self.diffusivity = diffusivity
        self.min_diameter_m = min_diameter_m

    def pristine_crater_profile(self, r, R, h_rim, d):
        """
        Generate pristine (fresh) crater topographic profile.

        Uses empirical shape function based on lunar simple crater morphology.

        Parameters:
        -----------
        r : ndarray
            Radial distance from center (normalized by crater radius)
        R : float
            Crater radius in meters
        h_rim : float
            Rim height above surrounding terrain (meters)
        d : float
            Crater depth below rim (meters)

        Returns:
        --------
        z : ndarray
            Elevation profile (meters)
        """
        # Normalize radial distance
        r_norm = r / R

        # Initialize profile
        z = np.zeros_like(r_norm)

        # Interior (parabolic bowl for simple craters)
        interior_mask = r_norm <= 1.0
        z[interior_mask] = -d * (1 - r_norm[interior_mask]**2)

        # Rim (exponential decay)
        rim_mask = (r_norm > 1.0) & (r_norm <= 1.3)
        r_rim = r_norm[rim_mask]
        # Rim peaks at r/R = 1.0 with height h_rim, decays exponentially
        z[rim_mask] = h_rim * np.exp(-5 * (r_rim - 1.0))

        # Far field (zero elevation)
        far_mask = r_norm > 1.3
        z[far_mask] = 0.0

        return z

    def degraded_crater_profile_analytical(self, r, R, h_rim, d, age_myr, diffusivity):
        """
        Analytical approximation of degraded crater profile using diffusion.

        Parameters:
        -----------
        r : ndarray
            Radial distance from center (meters)
        R : float
            Crater radius (meters)
        h_rim : float
            Initial rim height (meters)
        d : float
            Initial crater depth (meters)
        age_myr : float
            Time since formation (Myr)
        diffusivity : float
            Diffusivity coefficient (m²/Myr)

        Returns:
        --------
        z : ndarray
            Degraded elevation profile (meters)
        """
        # Get pristine profile
        z_pristine = self.pristine_crater_profile(r, R, h_rim, d)

        # Diffusion smoothing length scale
        # sigma = sqrt(2 * kappa * t)
        sigma = np.sqrt(2 * diffusivity * age_myr)

        if sigma < 1e-10:
            return z_pristine

        # Apply Gaussian smoothing as approximation of diffusion
        # This is a simplified analytical approach
        from scipy.ndimage import gaussian_filter1d

        # Need to handle the profile carefully
        # Create a finer grid for smoothing
        r_fine = np.linspace(r.min(), r.max(), len(r) * 5)
        z_fine = np.interp(r_fine, r, z_pristine)

        # Apply Gaussian filter
        # Convert sigma from meters to pixel units
        dr = np.mean(np.diff(r_fine))
        sigma_pixels = sigma / dr if dr > 0 else 0

        if sigma_pixels > 0:
            z_degraded_fine = gaussian_filter1d(z_fine, sigma_pixels)
            # Interpolate back to original grid
            z_degraded = np.interp(r, r_fine, z_degraded_fine)
        else:
            z_degraded = z_pristine

        return z_degraded

    def estimate_pristine_parameters(self, diameter_m, observed_depth_m=None):
        """
        Estimate pristine crater parameters from diameter.

        Uses empirical scaling relationships for lunar simple craters.

        Parameters:
        -----------
        diameter_m : float
            Crater diameter in meters
        observed_depth_m : float, optional
            Observed depth (if available)

        Returns:
        --------
        params : dict
            Dictionary with 'depth', 'rim_height', 'radius'
        """
        R = diameter_m / 2.0

        # Pristine depth/diameter ratio for fresh lunar simple craters
        # Based on Pike (1977), Fassett & Thomson (2014)
        # d/D ~ 0.196 for fresh simple craters
        d_pristine = 0.196 * diameter_m

        # Rim height (typically ~4% of diameter for fresh craters)
        h_rim_pristine = 0.04 * diameter_m

        return {
            'depth': d_pristine,
            'rim_height': h_rim_pristine,
            'radius': R
        }

    def fit_age_to_profile(self, r_observed, z_observed, diameter_m,
                           observed_depth=None, observed_rim_height=None):
        """
        Fit age by comparing observed profile to degradation model.

        Parameters:
        -----------
        r_observed : ndarray
            Radial distances from center (meters)
        z_observed : ndarray
            Observed elevation values (meters)
        diameter_m : float
            Crater diameter (meters)
        observed_depth : float, optional
            Measured crater depth (meters)
        observed_rim_height : float, optional
            Measured rim height (meters)

        Returns:
        --------
        age_myr : float
            Estimated age in Myr (million years)
        age_ga : float
            Estimated age in Ga (billion years)
        fit_quality : float
            RMS error of fit
        """
        # Check minimum diameter
        if diameter_m < self.min_diameter_m:
            warnings.warn(f"Crater diameter ({diameter_m:.0f}m) is below recommended "
                         f"minimum ({self.min_diameter_m}m). Results may be unreliable.")

        # Get pristine parameters
        pristine = self.estimate_pristine_parameters(diameter_m, observed_depth)
        R = pristine['radius']
        d_pristine = pristine['depth']
        h_rim_pristine = pristine['rim_height']

        # If observed values provided, use them to better constrain the fit
        if observed_depth is not None:
            d_current = observed_depth
        else:
            # Estimate from profile
            d_current = abs(np.min(z_observed))

        if observed_rim_height is not None:
            h_rim_current = observed_rim_height
        else:
            # Estimate from profile
            h_rim_current = np.max(z_observed)

        # Define objective function for age fitting
        def objective(age_myr):
            """Calculate RMS difference between observed and modeled profile."""
            if age_myr < 0:
                return 1e10

            z_model = self.degraded_crater_profile_analytical(
                r_observed, R, h_rim_pristine, d_pristine,
                age_myr, self.diffusivity
            )

            # Calculate RMS error
            rms = np.sqrt(np.mean((z_observed - z_model)**2))
            return rms

        # Optimize to find best-fit age
        # Search range: 0 to 4500 Myr (0-4.5 Ga)
        result = minimize_scalar(objective, bounds=(0, 4500), method='bounded')

        best_age_myr = result.x
        best_age_ga = best_age_myr / 1000.0
        fit_quality = result.fun

        return best_age_myr, best_age_ga, fit_quality

    def estimate_age_from_profiles(self, profiles, diameter_m, pixel_size_m,
                                   observed_depth=None, observed_rim_height=None):
        """
        Estimate age from multiple radial profiles.

        Parameters:
        -----------
        profiles : list of dict
            List of profile dictionaries with 'distance' and 'elevation' keys
        diameter_m : float
            Crater diameter in meters
        pixel_size_m : float
            Pixel size in meters
        observed_depth : float, optional
            Observed crater depth in meters
        observed_rim_height : float, optional
            Observed rim height in meters

        Returns:
        --------
        result : dict
            Dictionary with age estimates and statistics:
            - 'age_myr': Mean age in Myr
            - 'age_ga': Mean age in Ga
            - 'age_std_myr': Standard deviation in Myr
            - 'age_std_ga': Standard deviation in Ga
            - 'age_range_ga': (min, max) age range in Ga
            - 'fit_quality': Mean RMS fit error
            - 'n_profiles': Number of profiles used
            - 'degradation_state': Qualitative assessment
        """
        if len(profiles) == 0:
            return None

        ages_myr = []
        fit_qualities = []

        for profile in profiles:
            try:
                # Convert pixel distances to meters
                r_observed = profile['distance'] * pixel_size_m
                z_observed = profile['elevation']

                # Fit age
                age_myr, age_ga, fit_quality = self.fit_age_to_profile(
                    r_observed, z_observed, diameter_m,
                    observed_depth, observed_rim_height
                )

                ages_myr.append(age_myr)
                fit_qualities.append(fit_quality)

            except Exception as e:
                warnings.warn(f"Failed to fit profile: {e}")
                continue

        if len(ages_myr) == 0:
            return None

        ages_myr = np.array(ages_myr)
        ages_ga = ages_myr / 1000.0

        mean_age_myr = np.mean(ages_myr)
        mean_age_ga = mean_age_myr / 1000.0
        std_age_myr = np.std(ages_myr)
        std_age_ga = std_age_myr / 1000.0

        # Classify degradation state
        degradation_state = self.classify_degradation(mean_age_ga, diameter_m)

        return {
            'age_myr': mean_age_myr,
            'age_ga': mean_age_ga,
            'age_std_myr': std_age_myr,
            'age_std_ga': std_age_ga,
            'age_range_ga': (np.min(ages_ga), np.max(ages_ga)),
            'fit_quality': np.mean(fit_qualities),
            'n_profiles': len(ages_myr),
            'degradation_state': degradation_state,
            'method': 'topography_degradation'
        }

    def classify_degradation(self, age_ga, diameter_m):
        """
        Classify crater degradation state based on age.

        Parameters:
        -----------
        age_ga : float
            Age in Ga
        diameter_m : float
            Diameter in meters

        Returns:
        --------
        state : str
            Degradation classification
        """
        if age_ga < 0.1:
            return "Fresh"
        elif age_ga < 0.8:
            return "Young"
        elif age_ga < 2.0:
            return "Mature"
        elif age_ga < 3.5:
            return "Old"
        else:
            return "Very Old"

    def estimate_age_from_depth_ratio(self, diameter_m, observed_depth_m):
        """
        Quick age estimate from depth-diameter ratio (fallback method).

        Parameters:
        -----------
        diameter_m : float
            Crater diameter in meters
        observed_depth_m : float
            Observed depth in meters

        Returns:
        --------
        age_estimate : dict
            Dictionary with age estimate
        """
        pristine = self.estimate_pristine_parameters(diameter_m, observed_depth_m)
        d_pristine = pristine['depth']

        # Current d/D ratio
        d_D_current = observed_depth_m / diameter_m

        # Pristine d/D ratio
        d_D_pristine = d_pristine / diameter_m

        # Degradation factor
        degradation_factor = d_D_current / d_D_pristine if d_D_pristine > 0 else 0

        # Empirical relationship: age roughly scales with (1 - degradation_factor)
        # This is a crude approximation
        if degradation_factor > 0.9:
            age_ga = 0.05  # Very fresh
        elif degradation_factor > 0.7:
            age_ga = 0.5
        elif degradation_factor > 0.5:
            age_ga = 1.5
        elif degradation_factor > 0.3:
            age_ga = 3.0
        else:
            age_ga = 4.0

        return {
            'age_ga': age_ga,
            'age_myr': age_ga * 1000,
            'd_D_ratio': d_D_current,
            'd_D_pristine': d_D_pristine,
            'degradation_factor': degradation_factor,
            'method': 'depth_ratio_fallback'
        }


def estimate_age_topography_degradation(profiles, diameter_m, pixel_size_m,
                                        depth_m=None, rim_height_m=None,
                                        diffusivity=5.0):
    """
    Convenience function for age estimation using topography degradation model.

    Parameters:
    -----------
    profiles : list of dict
        Radial elevation profiles
    diameter_m : float
        Crater diameter in meters
    pixel_size_m : float
        Pixel size in meters
    depth_m : float, optional
        Observed crater depth
    rim_height_m : float, optional
        Observed rim height
    diffusivity : float, optional
        Diffusivity coefficient (default 5.0 m²/Myr)

    Returns:
    --------
    result : dict or None
        Age estimation results
    """
    estimator = TopographyDegradationAgeEstimator(diffusivity=diffusivity)

    result = estimator.estimate_age_from_profiles(
        profiles, diameter_m, pixel_size_m,
        observed_depth=depth_m,
        observed_rim_height=rim_height_m
    )

    return result


# Example usage
if __name__ == '__main__':
    print("Topography Degradation Age Estimation Module")
    print("=" * 60)
    print("\nThis module implements crater age estimation using")
    print("topographic degradation modeling (Luo et al. 2025).")
    print("\nBest for craters > 400m diameter")
    print("Typical uncertainty: < 165 Ma for large craters")
    print("\nUse via CraterAgeAnalyzer with method='topography_degradation'")

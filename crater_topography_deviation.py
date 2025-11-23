#!/usr/bin/env python3
"""
Crater Topography Deviation Analysis

Analyzes deviations from Ingersoll et al. (1992) idealized spherical bowl model
when actual crater topography is provided.

Computes:
- Shape deviation metrics (RMS, maximum deviation)
- Shape factors quantifying departure from ideal geometry
- Temperature corrections due to topographic deviations
- View factor corrections
- Enhanced shadow calculations

This module bridges the gap between analytical Ingersoll theory and
detailed topographic modeling.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from bowl_crater_thermal import CraterGeometry


@dataclass
class TopographicDeviation:
    """
    Quantifies deviation of actual topography from ideal spherical bowl.
    """
    rms_deviation: float  # RMS height deviation [m]
    max_deviation: float  # Maximum deviation [m]
    min_deviation: float  # Minimum deviation [m]
    shape_factor: float  # Dimensionless shape factor (0=perfect, 1=highly irregular)
    roughness_exponent: float  # Hurst exponent for surface roughness

    # Detailed statistics
    mean_deviation: float  # Mean signed deviation [m]
    std_deviation: float  # Standard deviation [m]
    skewness: float  # Skewness of deviation distribution

    # Geometric corrections
    effective_gamma: float  # Effective depth/diameter ratio
    volume_ratio: float  # Actual volume / ideal volume
    surface_area_ratio: float  # Actual area / ideal area


def fit_ideal_bowl(radii: np.ndarray, depths: np.ndarray,
                    diameter: float) -> Dict[str, float]:
    """
    Fit ideal spherical bowl to topographic profile.

    Uses least-squares fitting to find best-fit bowl parameters.

    Parameters:
    -----------
    radii : np.ndarray
        Radial distances from crater center [m]
    depths : np.ndarray
        Depths below rim at each radius [m] (positive downward)
    diameter : float
        Crater rim-to-rim diameter [m]

    Returns:
    --------
    dict containing:
        - 'depth': Best-fit crater depth [m]
        - 'gamma': Best-fit d/D ratio
        - 'R_sphere': Radius of curvature [m]
        - 'residuals': Fit residuals [m]
        - 'rms_error': RMS fit error [m]
    """
    R_rim = diameter / 2.0

    # For a spherical bowl: z(r) = R_sphere - sqrt(R_sphere^2 - r^2)
    # At rim: d = R_sphere - sqrt(R_sphere^2 - R_rim^2)
    # Solve for R_sphere given d: R_sphere = (R_rim^2 + d^2) / (2*d)

    # Fit by minimizing residuals for different depths
    def bowl_profile(r, d):
        """Calculate ideal bowl depth at radius r for crater depth d."""
        if d <= 0:
            return np.full_like(r, np.inf)
        R_sphere = (R_rim**2 + d**2) / (2.0 * d)
        # Ensure we don't take sqrt of negative number
        under_sqrt = R_sphere**2 - r**2
        if np.any(under_sqrt < 0):
            return np.full_like(r, np.inf)
        return R_sphere - np.sqrt(under_sqrt)

    # Grid search for best-fit depth
    d_test = np.linspace(depths.min() * 0.5, depths.max() * 1.5, 100)
    rms_errors = []

    for d in d_test:
        z_ideal = bowl_profile(radii, d)
        if np.all(np.isfinite(z_ideal)):
            rms_error = np.sqrt(np.mean((depths - z_ideal)**2))
            rms_errors.append(rms_error)
        else:
            rms_errors.append(np.inf)

    # Find minimum RMS error
    best_idx = np.argmin(rms_errors)
    best_depth = d_test[best_idx]
    best_gamma = best_depth / diameter

    # Calculate best-fit profile
    R_sphere = (R_rim**2 + best_depth**2) / (2.0 * best_depth)
    z_fit = bowl_profile(radii, best_depth)
    residuals = depths - z_fit

    return {
        'depth': best_depth,
        'gamma': best_gamma,
        'R_sphere': R_sphere,
        'residuals': residuals,
        'rms_error': rms_errors[best_idx],
        'z_ideal': z_fit
    }


def calculate_shape_factor(deviation_stats: Dict[str, float],
                           crater_diameter: float) -> float:
    """
    Calculate dimensionless shape factor quantifying irregularity.

    Shape factor = 0: Perfect spherical bowl
    Shape factor → 1: Highly irregular

    Based on ratio of topographic deviation to crater scale.

    Parameters:
    -----------
    deviation_stats : dict
        Statistics from topographic deviation analysis
    crater_diameter : float
        Crater diameter [m]

    Returns:
    --------
    float
        Shape factor (0-1)
    """
    # Normalize RMS deviation by crater diameter
    rms_normalized = deviation_stats['rms_error'] / crater_diameter

    # Also consider variance in local slopes
    # Higher variance = more irregular
    if 'slope_variance' in deviation_stats:
        slope_factor = min(deviation_stats['slope_variance'] / 0.1, 1.0)
    else:
        slope_factor = 0.0

    # Combined shape factor (weighted average)
    shape_factor = 0.7 * min(rms_normalized * 100, 1.0) + 0.3 * slope_factor

    return min(max(shape_factor, 0.0), 1.0)


def analyze_topographic_deviation(radii: np.ndarray,
                                   depths: np.ndarray,
                                   diameter: float,
                                   compute_roughness: bool = True) -> TopographicDeviation:
    """
    Comprehensive analysis of topographic deviation from ideal bowl.

    Parameters:
    -----------
    radii : np.ndarray
        Radial distances from crater center [m]
    depths : np.ndarray
        Depths below rim [m]
    diameter : float
        Crater diameter [m]
    compute_roughness : bool
        Whether to compute roughness exponent (computationally expensive)

    Returns:
    --------
    TopographicDeviation
        Complete deviation statistics
    """
    # Fit ideal bowl
    fit_result = fit_ideal_bowl(radii, depths, diameter)
    residuals = fit_result['residuals']

    # Basic statistics
    rms_dev = np.sqrt(np.mean(residuals**2))
    max_dev = np.max(np.abs(residuals))
    min_dev = np.min(residuals)
    mean_dev = np.mean(residuals)
    std_dev = np.std(residuals)

    # Skewness (asymmetry of deviations)
    if std_dev > 0:
        skewness = np.mean((residuals - mean_dev)**3) / std_dev**3
    else:
        skewness = 0.0

    # Calculate shape factor
    shape_factor = calculate_shape_factor(fit_result, diameter)

    # Roughness exponent (Hurst exponent) - simplified estimate
    if compute_roughness and len(radii) > 10:
        # Use structure function method
        roughness_exp = estimate_hurst_exponent(radii, residuals)
    else:
        roughness_exp = 0.5  # Default to white noise

    # Volume and surface area ratios
    # Simplified calculation assuming azimuthal symmetry
    R_rim = diameter / 2.0

    # Ideal volume (spherical cap)
    d_ideal = fit_result['depth']
    R_sphere = fit_result['R_sphere']
    V_ideal = np.pi * d_ideal**2 * (3*R_sphere - d_ideal) / 3

    # Actual volume (numerical integration)
    if len(radii) > 1:
        # Trapezoidal integration
        dr = np.diff(radii)
        r_mid = (radii[:-1] + radii[1:]) / 2
        z_mid = (depths[:-1] + depths[1:]) / 2
        V_actual = 2 * np.pi * np.sum(r_mid * z_mid * dr)
        volume_ratio = V_actual / V_ideal if V_ideal > 0 else 1.0
    else:
        volume_ratio = 1.0

    # Surface area ratio (approximate)
    # Account for additional area due to roughness
    if len(radii) > 1:
        # Arc length element: ds = sqrt(dr^2 + dz^2)
        dz = np.diff(depths)
        ds = np.sqrt(dr**2 + dz**2)
        A_actual = 2 * np.pi * np.sum(r_mid * ds)

        # Ideal surface area (spherical cap)
        A_ideal = 2 * np.pi * R_sphere * d_ideal
        surface_area_ratio = A_actual / A_ideal if A_ideal > 0 else 1.0
    else:
        surface_area_ratio = 1.0

    return TopographicDeviation(
        rms_deviation=rms_dev,
        max_deviation=max_dev,
        min_deviation=min_dev,
        shape_factor=shape_factor,
        roughness_exponent=roughness_exp,
        mean_deviation=mean_dev,
        std_deviation=std_dev,
        skewness=skewness,
        effective_gamma=fit_result['gamma'],
        volume_ratio=volume_ratio,
        surface_area_ratio=surface_area_ratio
    )


def estimate_hurst_exponent(x: np.ndarray, z: np.ndarray) -> float:
    """
    Estimate Hurst exponent (roughness exponent) using structure function.

    H ≈ 0.9-1.0: Very smooth (like ideal bowl)
    H ≈ 0.5: Random (white noise)
    H ≈ 0.0-0.3: Very rough

    Parameters:
    -----------
    x : np.ndarray
        Spatial positions
    z : np.ndarray
        Heights/depths

    Returns:
    --------
    float
        Hurst exponent (0-1)
    """
    # Structure function: S(Δx) = <|z(x+Δx) - z(x)|^2>
    # For self-affine surfaces: S(Δx) ∝ Δx^(2H)

    # Calculate lags
    n = len(x)
    max_lag = min(n // 3, 50)  # Don't use more than 1/3 of points

    lags = np.arange(1, max_lag)
    structure_func = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        dz = z[lag:] - z[:-lag]
        structure_func[i] = np.mean(dz**2)

    # Fit power law: log(S) = 2H * log(Δx) + const
    dx = np.abs(x[lags] - x[0])

    # Remove any zeros or negatives
    valid = (dx > 0) & (structure_func > 0)
    if np.sum(valid) < 3:
        return 0.5  # Not enough data

    log_dx = np.log(dx[valid])
    log_S = np.log(structure_func[valid])

    # Linear regression
    coeffs = np.polyfit(log_dx, log_S, 1)
    H = coeffs[0] / 2.0

    # Constrain to reasonable range
    return max(0.0, min(1.0, H))


def temperature_correction_factor(deviation: TopographicDeviation,
                                   crater: CraterGeometry) -> Dict[str, float]:
    """
    Estimate temperature corrections due to topographic deviations.

    Deviations from ideal bowl affect:
    1. View factors (more/less sky visibility)
    2. Shadow areas (irregularities create micro-shadows)
    3. Thermal radiation patterns

    Parameters:
    -----------
    deviation : TopographicDeviation
        Deviation statistics
    crater : CraterGeometry
        Nominal crater geometry

    Returns:
    --------
    dict containing correction factors:
        - 'view_factor_correction': Multiplier for sky view factor
        - 'shadow_area_correction': Multiplier for shadow area
        - 'temperature_offset': Estimated temperature offset [K]
        - 'uncertainty': Estimated uncertainty [K]
    """
    # Shape factor effects
    sf = deviation.shape_factor

    # View factor correction
    # Irregular craters have more surface area, potentially different view factors
    # Positive deviations (bumps) → see more sky
    # Negative deviations (extra depth) → see less sky

    # Use skewness to determine tendency
    if deviation.skewness > 0:
        # Tends toward bumps/ridges
        view_factor_mult = 1.0 + 0.1 * sf
    else:
        # Tends toward extra depressions
        view_factor_mult = 1.0 - 0.1 * sf

    # Shadow area correction
    # Rougher surfaces create more micro-shadows
    # Based on Hayne et al. (2021) rough surface theory
    shadow_area_mult = 1.0 + 0.15 * sf

    # Temperature offset estimate
    # Irregularities generally increase thermal heterogeneity
    # More surface area → more radiation exchange → warmer shadows
    T_offset = 2.0 * sf * deviation.surface_area_ratio

    # Uncertainty scales with deviation magnitude
    uncertainty = deviation.rms_deviation / crater.depth * 5.0  # ~5K per 10% depth variation

    return {
        'view_factor_correction': view_factor_mult,
        'shadow_area_correction': shadow_area_mult,
        'temperature_offset': T_offset,
        'uncertainty': uncertainty,
        'shape_factor': sf
    }


def generate_synthetic_irregular_crater(diameter: float,
                                        depth: float,
                                        irregularity: float = 0.1,
                                        n_points: int = 100,
                                        random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic crater profile with controlled irregularity.

    Useful for testing and demonstration.

    Parameters:
    -----------
    diameter : float
        Crater diameter [m]
    depth : float
        Mean crater depth [m]
    irregularity : float
        Irregularity factor (0=perfect bowl, 0.5=very irregular)
    n_points : int
        Number of radial points
    random_seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    radii : np.ndarray
        Radial distances [m]
    depths : np.ndarray
        Depths with irregularities [m]
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    R = diameter / 2.0
    radii = np.linspace(0, R, n_points)

    # Ideal spherical bowl
    R_sphere = (R**2 + depth**2) / (2.0 * depth)
    depths_ideal = R_sphere - np.sqrt(R_sphere**2 - radii**2)

    # Add irregularities with multiple spatial scales
    perturbation = np.zeros_like(radii)

    # Large-scale variations (asymmetry)
    if irregularity > 0:
        k_large = 2.0 * np.pi / R
        perturbation += irregularity * depth * 0.3 * np.sin(k_large * radii + np.random.random())

        # Medium-scale variations (wall features)
        k_medium = 10.0 * np.pi / R
        perturbation += irregularity * depth * 0.15 * np.sin(k_medium * radii + np.random.random())

        # Small-scale roughness
        roughness = np.random.normal(0, irregularity * depth * 0.05, len(radii))
        # Smooth it slightly
        try:
            from scipy.ndimage import gaussian_filter1d
            roughness = gaussian_filter1d(roughness, sigma=2)
        except ImportError:
            # If scipy not available, use simple moving average
            window = 3
            roughness_smooth = np.copy(roughness)
            for i in range(window, len(roughness) - window):
                roughness_smooth[i] = np.mean(roughness[i-window:i+window+1])
            roughness = roughness_smooth

        perturbation += roughness

    depths_irregular = depths_ideal + perturbation

    # Ensure depths are positive and smooth at rim
    depths_irregular = np.maximum(depths_irregular, 0)
    depths_irregular[0] = 0  # Rim
    depths_irregular[-1] = 0  # Rim

    return radii, depths_irregular


def print_deviation_report(deviation: TopographicDeviation,
                           crater: CraterGeometry,
                           corrections: Dict[str, float]):
    """
    Print comprehensive deviation analysis report.

    Parameters:
    -----------
    deviation : TopographicDeviation
        Deviation statistics
    crater : CraterGeometry
        Crater geometry
    corrections : dict
        Temperature correction factors
    """
    print("=" * 70)
    print("Topographic Deviation Analysis")
    print("=" * 70)

    print(f"\n### Crater Geometry ###")
    print(f"Nominal diameter: {crater.diameter:.1f} m")
    print(f"Nominal depth: {crater.depth:.1f} m")
    print(f"Nominal γ (d/D): {crater.gamma:.4f}")

    print(f"\n### Best-Fit Ideal Bowl ###")
    print(f"Fitted depth: {deviation.effective_gamma * crater.diameter:.1f} m")
    print(f"Fitted γ: {deviation.effective_gamma:.4f}")
    print(f"Difference: {(deviation.effective_gamma - crater.gamma) / crater.gamma * 100:+.1f}%")

    print(f"\n### Deviation Statistics ###")
    print(f"RMS deviation: {deviation.rms_deviation:.2f} m ({deviation.rms_deviation/crater.depth*100:.1f}% of depth)")
    print(f"Maximum deviation: {deviation.max_deviation:.2f} m")
    print(f"Mean deviation: {deviation.mean_deviation:+.2f} m")
    print(f"Std deviation: {deviation.std_deviation:.2f} m")
    print(f"Skewness: {deviation.skewness:+.3f}")
    if abs(deviation.skewness) > 0.5:
        skew_desc = "bumpy/ridged" if deviation.skewness > 0 else "with extra depressions"
        print(f"  → Tends toward {skew_desc} topography")

    print(f"\n### Shape Characterization ###")
    print(f"Shape factor: {deviation.shape_factor:.3f}")
    if deviation.shape_factor < 0.1:
        print("  → Nearly perfect spherical bowl")
    elif deviation.shape_factor < 0.3:
        print("  → Moderately irregular")
    else:
        print("  → Highly irregular")

    print(f"Roughness exponent (H): {deviation.roughness_exponent:.3f}")
    if deviation.roughness_exponent > 0.7:
        print("  → Very smooth surface")
    elif deviation.roughness_exponent > 0.4:
        print("  → Moderate roughness")
    else:
        print("  → High roughness")

    print(f"\n### Geometric Corrections ###")
    print(f"Volume ratio (actual/ideal): {deviation.volume_ratio:.3f}")
    print(f"Surface area ratio: {deviation.surface_area_ratio:.3f}")

    print(f"\n### Temperature Corrections ###")
    print(f"View factor correction: {corrections['view_factor_correction']:.3f}×")
    print(f"Shadow area correction: {corrections['shadow_area_correction']:.3f}×")
    print(f"Estimated temperature offset: {corrections['temperature_offset']:+.1f} K")
    print(f"Uncertainty: ±{corrections['uncertainty']:.1f} K")

    print(f"\n### Applicability of Ingersoll Model ###")
    if deviation.shape_factor < 0.15 and corrections['uncertainty'] < 5:
        print("✓ Ingersoll model is excellent approximation")
    elif deviation.shape_factor < 0.3 and corrections['uncertainty'] < 10:
        print("✓ Ingersoll model is good approximation with corrections")
    else:
        print("⚠ Significant deviations - detailed topographic modeling recommended")

    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Crater Topography Deviation Analysis - Examples")
    print("=" * 70)

    # Example 1: Nearly perfect bowl
    print("\n### Example 1: Nearly Perfect Spherical Bowl ###")
    crater1 = CraterGeometry(diameter=5000, depth=400, latitude_deg=-85)
    radii1, depths1 = generate_synthetic_irregular_crater(
        crater1.diameter, crater1.depth, irregularity=0.05, random_seed=42
    )

    deviation1 = analyze_topographic_deviation(radii1, depths1, crater1.diameter)
    corrections1 = temperature_correction_factor(deviation1, crater1)
    print_deviation_report(deviation1, crater1, corrections1)

    # Example 2: Moderately irregular crater
    print("\n\n### Example 2: Moderately Irregular Crater ###")
    crater2 = CraterGeometry(diameter=10000, depth=800, latitude_deg=-87)
    radii2, depths2 = generate_synthetic_irregular_crater(
        crater2.diameter, crater2.depth, irregularity=0.25, random_seed=123
    )

    deviation2 = analyze_topographic_deviation(radii2, depths2, crater2.diameter)
    corrections2 = temperature_correction_factor(deviation2, crater2)
    print_deviation_report(deviation2, crater2, corrections2)

    # Example 3: Highly irregular crater
    print("\n\n### Example 3: Highly Irregular Crater ###")
    crater3 = CraterGeometry(diameter=3000, depth=200, latitude_deg=-86)
    radii3, depths3 = generate_synthetic_irregular_crater(
        crater3.diameter, crater3.depth, irregularity=0.5, random_seed=456
    )

    deviation3 = analyze_topographic_deviation(radii3, depths3, crater3.diameter)
    corrections3 = temperature_correction_factor(deviation3, crater3)
    print_deviation_report(deviation3, crater3, corrections3)

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)

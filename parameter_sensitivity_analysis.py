#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis for Cold Trap Model

This script performs sensitivity scans for three key parameters:
1. K (power-law scale factor): Tests range to find value matching paper's 40,000 km²
2. b (power-law exponent): Tests range to match paper's size distribution
3. Lateral conduction limit: Tests range to match paper's <1m area claim

Goal: Identify parameter combinations that reconcile model with paper claims.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from hayne_model_corrected import hayne_cold_trap_fraction_corrected

# Physical constants
COLD_TRAP_THRESHOLD = 110.0  # K
LATERAL_CONDUCTION_LIMIT = 0.01  # 1 cm in meters (baseline)
TRANSITION_SCALE = 1000.0  # 1 km
LUNAR_SURFACE_AREA = 3.793e7  # km²

# Paper claims for validation
PAPER_TOTAL_AREA = 40000  # km²
PAPER_AREA_LT_1M = 700  # km²
PAPER_AREA_LT_100M = 2500  # km²

# File path
PSR_CSV = '/home/user/documents/psr_with_temperatures.csv'


def load_observed_cold_trap_area():
    """Load total cold trap area from observed large PSRs (≥1km)."""
    psr = pd.read_csv(PSR_CSV)
    psr['diameter_m'] = 2 * np.sqrt(psr['area_km2'] * 1e6 / np.pi)

    # Calculate cold trap area
    psr['coldtrap_fraction'] = 0.0
    mask = psr['pixel_count'] > 0
    psr.loc[mask, 'coldtrap_fraction'] = psr.loc[mask, 'pixels_lt_110K'] / psr.loc[mask, 'pixel_count']
    psr['coldtrap_area_km2'] = psr['coldtrap_fraction'] * psr['area_km2']

    # Filter for large PSRs
    large_psrs = psr[psr['diameter_m'] >= TRANSITION_SCALE]
    observed_area = large_psrs['coldtrap_area_km2'].sum()

    return observed_area


def calculate_synthetic_area(K, b, conduction_limit_m, L_min=1e-4, L_max=1000.0, n_bins=100):
    """
    Calculate total synthetic cold trap area for given parameters.

    Args:
        K: Power-law scale factor
        b: Power-law exponent
        conduction_limit_m: Minimum cold trap size [m]
        L_min: Minimum integration scale [m]
        L_max: Maximum integration scale [m]
        n_bins: Number of bins

    Returns:
        Dictionary with total area and breakdown by size
    """
    # Create bins
    L_bins = np.logspace(np.log10(L_min), np.log10(L_max), n_bins)
    dL = np.diff(np.logspace(np.log10(L_min), np.log10(L_max), n_bins + 1))

    # Differential number density
    N_diff = K * L_bins**(-b - 1)
    N_per_bin = N_diff * dL

    # Hemisphere asymmetry (60% south, 40% north)
    N_north = N_per_bin * 0.40
    N_south = N_per_bin * 0.60

    # Calculate cold trap areas using Hayne model
    A_north = np.zeros_like(L_bins)
    A_south = np.zeros_like(L_bins)

    # Representative polar latitudes
    lat_north = 85.0
    lat_south = -85.0

    # RMS slopes
    sigma_s_plains = 5.7
    sigma_s_craters = 20.0
    f_craters = 0.20
    f_plains = 0.80

    for i, (L, n_n, n_s) in enumerate(zip(L_bins, N_north, N_south)):
        # Apply conduction limit
        if L < conduction_limit_m:
            continue

        # Calculate cold trap fractions
        f_ct_crater_north = hayne_cold_trap_fraction_corrected(sigma_s_craters, lat_north)
        f_ct_crater_south = hayne_cold_trap_fraction_corrected(sigma_s_craters, lat_south)
        f_ct_plains_north = hayne_cold_trap_fraction_corrected(sigma_s_plains, lat_north)
        f_ct_plains_south = hayne_cold_trap_fraction_corrected(sigma_s_plains, lat_south)

        # Weighted average
        f_ct_north = f_craters * f_ct_crater_north + f_plains * f_ct_plains_north
        f_ct_south = f_craters * f_ct_crater_south + f_plains * f_ct_plains_south

        # Area per feature
        area_per_feature = np.pi * (L / 2.0)**2  # m²

        # Total cold trap area
        A_north[i] = n_n * area_per_feature * f_ct_north * 1e-6  # km²
        A_south[i] = n_s * area_per_feature * f_ct_south * 1e-6  # km²

    # Calculate totals and breakdowns
    total_area = A_north.sum() + A_south.sum()

    # Area in different size ranges
    mask_lt_1m = L_bins < 1.0
    area_lt_1m = (A_north[mask_lt_1m].sum() + A_south[mask_lt_1m].sum())

    mask_lt_100m = L_bins < 100.0
    area_lt_100m = (A_north[mask_lt_100m].sum() + A_south[mask_lt_100m].sum())

    return {
        'total_area': total_area,
        'area_lt_1m': area_lt_1m,
        'area_lt_100m': area_lt_100m,
        'L_bins': L_bins,
        'A_north': A_north,
        'A_south': A_south
    }


def scan_K_parameter(K_values, b=1.8, conduction_limit=0.01, observed_area=0):
    """Scan K parameter to find value matching paper's total area."""
    print("\n" + "=" * 80)
    print("SCANNING PARAMETER: K (Power-law scale factor)")
    print("=" * 80)
    print(f"Fixed parameters: b={b}, conduction_limit={conduction_limit}m")
    print(f"Observed area (≥1km): {observed_area:.2f} km²")
    print(f"Target total area: {PAPER_TOTAL_AREA} km²")
    print(f"Target synthetic area: {PAPER_TOTAL_AREA - observed_area:.2f} km²")

    results = []

    for K in K_values:
        result = calculate_synthetic_area(K, b, conduction_limit)
        total_with_observed = result['total_area'] + observed_area

        results.append({
            'K': K,
            'synthetic_area': result['total_area'],
            'total_area': total_with_observed,
            'area_lt_1m': result['area_lt_1m'],
            'area_lt_100m': result['area_lt_100m']
        })

        print(f"\nK = {K:.2e}")
        print(f"  Synthetic area: {result['total_area']:,.0f} km²")
        print(f"  Total area: {total_with_observed:,.0f} km²")
        print(f"  Area <1m: {result['area_lt_1m']:,.0f} km² (paper: {PAPER_AREA_LT_1M} km²)")
        print(f"  Area <100m: {result['area_lt_100m']:,.0f} km² (paper: {PAPER_AREA_LT_100M} km²)")
        print(f"  Error from paper total: {total_with_observed - PAPER_TOTAL_AREA:+,.0f} km²")

    return pd.DataFrame(results)


def scan_b_parameter(b_values, K=2e11, conduction_limit=0.01, observed_area=0):
    """Scan b parameter to find value matching paper's size distribution."""
    print("\n" + "=" * 80)
    print("SCANNING PARAMETER: b (Power-law exponent)")
    print("=" * 80)
    print(f"Fixed parameters: K={K:.2e}, conduction_limit={conduction_limit}m")
    print(f"Observed area (≥1km): {observed_area:.2f} km²")
    print(f"Target: ~6% of total area in <100m features (paper's distribution)")

    results = []

    for b in b_values:
        result = calculate_synthetic_area(K, b, conduction_limit)
        total_with_observed = result['total_area'] + observed_area
        fraction_lt_100m = result['area_lt_100m'] / total_with_observed * 100

        results.append({
            'b': b,
            'synthetic_area': result['total_area'],
            'total_area': total_with_observed,
            'area_lt_100m': result['area_lt_100m'],
            'fraction_lt_100m': fraction_lt_100m
        })

        print(f"\nb = {b:.2f}")
        print(f"  Synthetic area: {result['total_area']:,.0f} km²")
        print(f"  Total area: {total_with_observed:,.0f} km²")
        print(f"  Area <100m: {result['area_lt_100m']:,.0f} km²")
        print(f"  Fraction <100m: {fraction_lt_100m:.1f}% (paper: ~6%)")

    return pd.DataFrame(results)


def scan_conduction_limit(limit_values, K=2e11, b=1.8, observed_area=0):
    """Scan lateral conduction limit to find value matching paper's <1m area."""
    print("\n" + "=" * 80)
    print("SCANNING PARAMETER: Lateral Conduction Limit")
    print("=" * 80)
    print(f"Fixed parameters: K={K:.2e}, b={b}")
    print(f"Observed area (≥1km): {observed_area:.2f} km²")
    print(f"Target area <1m: {PAPER_AREA_LT_1M} km²")

    results = []

    for limit in limit_values:
        result = calculate_synthetic_area(K, b, limit)
        total_with_observed = result['total_area'] + observed_area

        results.append({
            'limit_m': limit,
            'limit_cm': limit * 100,
            'synthetic_area': result['total_area'],
            'total_area': total_with_observed,
            'area_lt_1m': result['area_lt_1m']
        })

        print(f"\nConduction limit = {limit}m ({limit*100:.1f} cm)")
        print(f"  Synthetic area: {result['total_area']:,.0f} km²")
        print(f"  Total area: {total_with_observed:,.0f} km²")
        print(f"  Area <1m: {result['area_lt_1m']:,.0f} km² (paper: {PAPER_AREA_LT_1M} km²)")
        print(f"  Error from paper <1m: {result['area_lt_1m'] - PAPER_AREA_LT_1M:+,.0f} km²")

    return pd.DataFrame(results)


def plot_sensitivity_results(df_K, df_b, df_limit, observed_area):
    """Create visualization of sensitivity analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Panel 1: K vs Total Area
    ax = axes[0, 0]
    ax.plot(df_K['K'], df_K['total_area'], 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=PAPER_TOTAL_AREA, color='r', linestyle='--', linewidth=2, label='Paper claim')
    ax.set_xlabel('K (scale factor)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Area [km²]', fontsize=12, fontweight='bold')
    ax.set_title('K Parameter Scan\nTotal Cold Trap Area', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Panel 2: K vs Area <1m
    ax = axes[0, 1]
    ax.plot(df_K['K'], df_K['area_lt_1m'], 'g-o', linewidth=2, markersize=8)
    ax.axhline(y=PAPER_AREA_LT_1M, color='r', linestyle='--', linewidth=2, label='Paper claim')
    ax.set_xlabel('K (scale factor)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Area <1m [km²]', fontsize=12, fontweight='bold')
    ax.set_title('K Parameter Scan\nSmall Features (<1m)', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Panel 3: K vs Area <100m
    ax = axes[0, 2]
    ax.plot(df_K['K'], df_K['area_lt_100m'], 'purple', marker='o', linewidth=2, markersize=8)
    ax.axhline(y=PAPER_AREA_LT_100M, color='r', linestyle='--', linewidth=2, label='Paper claim')
    ax.set_xlabel('K (scale factor)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Area <100m [km²]', fontsize=12, fontweight='bold')
    ax.set_title('K Parameter Scan\nMedium Features (<100m)', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Panel 4: b vs Total Area
    ax = axes[1, 0]
    ax.plot(df_b['b'], df_b['total_area'], 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=PAPER_TOTAL_AREA, color='r', linestyle='--', linewidth=2, label='Paper claim')
    ax.set_xlabel('b (power-law exponent)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Area [km²]', fontsize=12, fontweight='bold')
    ax.set_title('b Parameter Scan\nTotal Cold Trap Area', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Panel 5: b vs Fraction <100m
    ax = axes[1, 1]
    ax.plot(df_b['b'], df_b['fraction_lt_100m'], 'orange', marker='o', linewidth=2, markersize=8)
    ax.axhline(y=6.0, color='r', linestyle='--', linewidth=2, label='Paper (~6%)')
    ax.set_xlabel('b (power-law exponent)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction <100m [%]', fontsize=12, fontweight='bold')
    ax.set_title('b Parameter Scan\nSize Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Panel 6: Conduction Limit vs Area <1m
    ax = axes[1, 2]
    ax.plot(df_limit['limit_cm'], df_limit['area_lt_1m'], 'g-o', linewidth=2, markersize=8)
    ax.axhline(y=PAPER_AREA_LT_1M, color='r', linestyle='--', linewidth=2, label='Paper claim')
    ax.set_xlabel('Lateral Conduction Limit [cm]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Area <1m [km²]', fontsize=12, fontweight='bold')
    ax.set_title('Conduction Limit Scan\nVery Small Features (<1m)', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    output_path = '/home/user/documents/parameter_sensitivity_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot: {output_path}")
    plt.close()


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("Cold Trap Model - Paper vs Implementation")
    print("=" * 80)

    # Load observed cold trap area
    observed_area = load_observed_cold_trap_area()
    print(f"\nObserved cold trap area (PSRs ≥1km): {observed_area:.2f} km²")
    print(f"Target synthetic area: {PAPER_TOTAL_AREA - observed_area:.2f} km²")

    # 1. Scan K parameter
    K_values = np.logspace(10, 12, 15)  # 1e10 to 1e12
    df_K = scan_K_parameter(K_values, observed_area=observed_area)

    # Find K that best matches paper total
    best_K_idx = (df_K['total_area'] - PAPER_TOTAL_AREA).abs().idxmin()
    best_K = df_K.loc[best_K_idx, 'K']
    print(f"\n>>> Best K for matching paper total: {best_K:.2e}")
    print(f"    Gives total area: {df_K.loc[best_K_idx, 'total_area']:,.0f} km²")
    print(f"    But area <1m: {df_K.loc[best_K_idx, 'area_lt_1m']:,.0f} km² (vs paper: {PAPER_AREA_LT_1M} km²)")

    # 2. Scan b parameter
    b_values = np.linspace(1.0, 2.5, 16)
    df_b = scan_b_parameter(b_values, observed_area=observed_area)

    # Find b that best matches paper's size distribution
    best_b_idx = (df_b['fraction_lt_100m'] - 6.0).abs().idxmin()
    best_b = df_b.loc[best_b_idx, 'b']
    print(f"\n>>> Best b for matching paper size distribution: {best_b:.2f}")
    print(f"    Gives fraction <100m: {df_b.loc[best_b_idx, 'fraction_lt_100m']:.1f}%")
    print(f"    Total area: {df_b.loc[best_b_idx, 'total_area']:,.0f} km²")

    # 3. Scan conduction limit
    limit_values = np.logspace(-2, 0, 15)  # 0.01m to 1m (1cm to 100cm)
    df_limit = scan_conduction_limit(limit_values, observed_area=observed_area)

    # Find limit that best matches paper's <1m area
    best_limit_idx = (df_limit['area_lt_1m'] - PAPER_AREA_LT_1M).abs().idxmin()
    best_limit = df_limit.loc[best_limit_idx, 'limit_m']
    print(f"\n>>> Best conduction limit for matching paper <1m area: {best_limit:.3f}m ({best_limit*100:.1f} cm)")
    print(f"    Gives area <1m: {df_limit.loc[best_limit_idx, 'area_lt_1m']:,.0f} km²")
    print(f"    Total area: {df_limit.loc[best_limit_idx, 'total_area']:,.0f} km²")

    # Create visualization
    plot_sensitivity_results(df_K, df_b, df_limit, observed_area)

    # Save results to CSV
    df_K.to_csv('/home/user/documents/sensitivity_K_scan.csv', index=False)
    df_b.to_csv('/home/user/documents/sensitivity_b_scan.csv', index=False)
    df_limit.to_csv('/home/user/documents/sensitivity_limit_scan.csv', index=False)
    print("\n✓ Saved CSV files: sensitivity_K_scan.csv, sensitivity_b_scan.csv, sensitivity_limit_scan.csv")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY: Key Findings")
    print("=" * 80)
    print(f"\n1. K Parameter:")
    print(f"   - Current value: 2.00e+11")
    print(f"   - To match paper total (40,000 km²): {best_K:.2e}")
    print(f"   - But this makes <1m area WORSE (9× → {df_K.loc[best_K_idx, 'area_lt_1m']/PAPER_AREA_LT_1M:.1f}×)")
    print(f"\n2. b Parameter:")
    print(f"   - Current value: 1.8")
    print(f"   - To match paper size distribution: {best_b:.2f}")
    print(f"   - This shifts dominance from small to large features")
    print(f"\n3. Lateral Conduction Limit:")
    print(f"   - Current value: 0.01m (1 cm)")
    print(f"   - To match paper <1m area: {best_limit:.3f}m ({best_limit*100:.1f} cm)")
    print(f"   - Larger limit reduces very small cold trap contribution")

    print("\n" + "=" * 80)
    print("\n✓ COMPLETE: Parameter sensitivity analysis")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

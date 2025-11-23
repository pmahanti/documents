#!/usr/bin/env python3
"""
Remake Figure 4 Top Panel: Cumulative Cold Trap Area

This version directly uses cold trap areas from Diviner temperature data:
1. For PSRs >= 1 km²: Use area < 110K from Diviner (pixels_lt_110K from CSV)
2. For PSRs < 1 km²: Use synthetic power-law distribution + Hayne model
3. Plot: Cumulative cold trap area < L (km²) vs L
4. Report minimum L value

Note: Diviner pixels are 240m x 240m = 0.0576 km² per pixel
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from hayne_model_corrected import hayne_cold_trap_fraction_corrected

# Physical constants
COLD_TRAP_THRESHOLD = 110.0  # K
DIVINER_PIXEL_SIZE = 0.24  # km (240m)
DIVINER_PIXEL_AREA = DIVINER_PIXEL_SIZE ** 2  # km² = 0.0576 km²/pixel
LATERAL_CONDUCTION_LIMIT = 0.01  # 1 cm in meters
TRANSITION_SCALE = 1000.0  # 1 km - transition from synthetic to observed
LUNAR_SURFACE_AREA = 3.793e7  # km²

# File path
PSR_CSV = '/home/user/documents/psr_with_temperatures.csv'


def load_psr_data_with_temperatures():
    """
    Load PSR data with Diviner temperatures from CSV.

    The CSV contains pre-computed temperature statistics from Diviner geotiff files,
    including pixels_lt_110K which counts pixels below the cold trap threshold.

    Returns:
        DataFrame with PSR data including cold trap information
    """
    print("\n[Loading PSR data with Diviner temperatures]")

    psr = pd.read_csv(PSR_CSV)

    # Calculate diameter from area (assuming circular)
    psr['diameter_m'] = 2 * np.sqrt(psr['area_km2'] * 1e6 / np.pi)

    # Calculate cold trap area from pixel count
    # pixels_lt_110K is the number of Diviner pixels < 110K within each PSR
    # Cold trap area = (fraction of cold pixels) * total PSR area
    # Avoid division by zero
    psr['coldtrap_fraction'] = 0.0
    mask = psr['pixel_count'] > 0
    psr.loc[mask, 'coldtrap_fraction'] = psr.loc[mask, 'pixels_lt_110K'] / psr.loc[mask, 'pixel_count']
    psr['coldtrap_area_km2'] = psr['coldtrap_fraction'] * psr['area_km2']

    print(f"✓ Loaded {len(psr)} PSRs with Diviner temperature data")
    print(f"  Total PSR area: {psr['area_km2'].sum():.2f} km²")
    print(f"  Total cold trap area (< 110K): {psr['coldtrap_area_km2'].sum():.2f} km²")
    print(f"  Diameter range: {psr['diameter_m'].min():.1f} - {psr['diameter_m'].max():.1f} m")

    # Split by hemisphere
    north = psr[psr['hemisphere'] == 'North']
    south = psr[psr['hemisphere'] == 'South']

    print(f"\n  North: {len(north)} PSRs, {north['area_km2'].sum():.2f} km² total")
    print(f"    Cold trap area: {north['coldtrap_area_km2'].sum():.2f} km²")
    print(f"  South: {len(south)} PSRs, {south['area_km2'].sum():.2f} km² total")
    print(f"    Cold trap area: {south['coldtrap_area_km2'].sum():.2f} km²")

    return psr


def filter_large_psrs(psr_data, min_diameter_m=1000.0):
    """
    Filter for large PSRs (>= min_diameter_m).

    Args:
        psr_data: DataFrame with all PSRs
        min_diameter_m: Minimum diameter in meters

    Returns:
        DataFrame with large PSRs only
    """
    print(f"\n[Filtering for large PSRs (D >= {min_diameter_m}m = {min_diameter_m/1000:.0f} km)]")

    # Filter for large PSRs
    large_psrs = psr_data[psr_data['diameter_m'] >= min_diameter_m].copy()

    # For large PSRs, also filter for area >= 1 km² to ensure consistency
    large_psrs = large_psrs[large_psrs['area_km2'] >= 1.0].copy()

    print(f"  Found {len(large_psrs)} large PSRs (area >= 1 km²)")
    print(f"  Total area: {large_psrs['area_km2'].sum():.2f} km²")
    print(f"  Total cold trap area: {large_psrs['coldtrap_area_km2'].sum():.2f} km²")

    # Split by hemisphere
    north = large_psrs[large_psrs['hemisphere'] == 'North']
    south = large_psrs[large_psrs['hemisphere'] == 'South']

    print(f"\n  North: {len(north)} PSRs")
    print(f"    Total area: {north['area_km2'].sum():.2f} km²")
    print(f"    Cold trap area: {north['coldtrap_area_km2'].sum():.2f} km²")
    print(f"    PSRs with cold traps: {(north['coldtrap_area_km2'] > 0).sum()}")

    print(f"\n  South: {len(south)} PSRs")
    print(f"    Total area: {south['area_km2'].sum():.2f} km²")
    print(f"    Cold trap area: {south['coldtrap_area_km2'].sum():.2f} km²")
    print(f"    PSRs with cold traps: {(south['coldtrap_area_km2'] > 0).sum()}")

    # Report minimum diameter/area in the large PSR dataset
    print(f"\n  Minimum diameter: {large_psrs['diameter_m'].min():.1f} m")
    print(f"  Minimum area: {large_psrs['area_km2'].min():.4f} km²")
    print(f"  Maximum diameter: {large_psrs['diameter_m'].max():.1f} m")
    print(f"  Maximum area: {large_psrs['area_km2'].max():.2f} km²")

    return large_psrs[['diameter_m', 'coldtrap_area_km2', 'hemisphere', 'area_km2']]


def bin_large_psrs(large_psrs_data, L_bins):
    """
    Bin large PSRs into logarithmic size bins.

    Args:
        large_psrs_data: DataFrame with diameter_m and coldtrap_area_km2
        L_bins: Length scale bin centers [m]

    Returns:
        N_north, N_south, A_north, A_south: Arrays for each bin
    """
    N_north = np.zeros(len(L_bins))
    N_south = np.zeros(len(L_bins))
    A_north = np.zeros(len(L_bins))
    A_south = np.zeros(len(L_bins))

    # Create bin edges (geometric mean between adjacent bins)
    L_edges = np.zeros(len(L_bins) + 1)
    L_edges[0] = L_bins[0] / (L_bins[1]/L_bins[0])**0.5
    L_edges[-1] = L_bins[-1] * (L_bins[-1]/L_bins[-2])**0.5
    for i in range(1, len(L_bins)):
        L_edges[i] = np.sqrt(L_bins[i-1] * L_bins[i])

    # Bin the PSRs
    for _, row in large_psrs_data.iterrows():
        diameter = row['diameter_m']

        # Find which bin this PSR belongs to
        bin_idx = np.searchsorted(L_edges, diameter) - 1

        # Check if within bin range
        if 0 <= bin_idx < len(L_bins):
            if row['hemisphere'] == 'North':
                N_north[bin_idx] += 1
                A_north[bin_idx] += row['coldtrap_area_km2']
            else:  # South
                N_south[bin_idx] += 1
                A_south[bin_idx] += row['coldtrap_area_km2']

    return N_north, N_south, A_north, A_south


def generate_synthetic_distribution(L_min=1e-4, L_max=TRANSITION_SCALE, n_bins=50):
    """
    Generate synthetic size-frequency distribution for small PSRs.

    Args:
        L_min: Minimum length scale [m]
        L_max: Maximum length scale [m]
        n_bins: Number of logarithmic bins

    Returns:
        Dictionary with L_bins, N_north, N_south
    """
    L_bins = np.logspace(np.log10(L_min), np.log10(L_max), n_bins)
    dL = np.diff(np.logspace(np.log10(L_min), np.log10(L_max), n_bins + 1))

    # Power-law exponent (from Hayne model)
    b = 1.8

    # Scale factor (calibrated to connect smoothly with observed PSRs)
    K = 2e11

    # Differential number density
    N_diff = K * L_bins**(-b - 1)
    N_per_bin = N_diff * dL

    # Hemisphere asymmetry (60% south, 40% north)
    N_north = N_per_bin * 0.40
    N_south = N_per_bin * 0.60

    return {
        'L_bins': L_bins,
        'N_north': N_north,
        'N_south': N_south,
        'dL': dL
    }


def calculate_synthetic_cold_trap_areas(L_bins, N_north, N_south):
    """
    Calculate cold trap areas for synthetic (small) PSRs using Hayne model.

    Args:
        L_bins: Length scale bin centers [m]
        N_north: Number of features per bin (northern hemisphere)
        N_south: Number of features per bin (southern hemisphere)

    Returns:
        A_north, A_south: Total cold trap area per bin [km²]
    """
    A_north = np.zeros_like(L_bins)
    A_south = np.zeros_like(L_bins)

    # Representative polar latitudes
    lat_north = 85.0
    lat_south = -85.0

    # RMS slopes for different terrain types
    sigma_s_plains = 5.7
    sigma_s_craters = 20.0

    # Landscape mixture
    f_craters = 0.20
    f_plains = 0.80

    for i, (L, n_n, n_s) in enumerate(zip(L_bins, N_north, N_south)):
        # Skip if below lateral conduction limit
        if L < LATERAL_CONDUCTION_LIMIT:
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

    return A_north, A_south


def create_hybrid_distribution(large_psrs_data, L_min=1e-4, L_max=100000, n_bins=100):
    """
    Create hybrid size distribution combining synthetic and observed data.

    Args:
        large_psrs_data: DataFrame with large PSR data
        L_min: Minimum length scale [m]
        L_max: Maximum length scale [m]
        n_bins: Total number of bins

    Returns:
        L_bins, N_north, N_south, A_north, A_south
    """
    print("\n[Creating hybrid distribution]")

    # Create logarithmic bins spanning full range
    L_bins = np.logspace(np.log10(L_min), np.log10(L_max), n_bins)

    # Initialize arrays
    N_north_total = np.zeros(len(L_bins))
    N_south_total = np.zeros(len(L_bins))
    A_north_total = np.zeros(len(L_bins))
    A_south_total = np.zeros(len(L_bins))

    # Find transition index
    transition_idx = np.searchsorted(L_bins, TRANSITION_SCALE)

    print(f"  Total bins: {len(L_bins)}")
    print(f"  Bins < {TRANSITION_SCALE}m (synthetic): {transition_idx}")
    print(f"  Bins ≥ {TRANSITION_SCALE}m (observed): {len(L_bins) - transition_idx}")

    # 1. SMALL SCALES: Use synthetic distribution
    if transition_idx > 0:
        print("\n  [Small scales: Synthetic power-law + Hayne model]")

        # Generate synthetic distribution only for small bins
        synth = generate_synthetic_distribution(L_min=L_bins[0], L_max=L_bins[transition_idx-1],
                                                n_bins=transition_idx)

        # Calculate cold trap areas using Hayne model
        A_north_synth, A_south_synth = calculate_synthetic_cold_trap_areas(
            synth['L_bins'], synth['N_north'], synth['N_south'])

        # Assign to total arrays
        N_north_total[:transition_idx] = synth['N_north']
        N_south_total[:transition_idx] = synth['N_south']
        A_north_total[:transition_idx] = A_north_synth
        A_south_total[:transition_idx] = A_south_synth

        print(f"    Synthetic cold trap area North: {A_north_synth.sum():.2f} km²")
        print(f"    Synthetic cold trap area South: {A_south_synth.sum():.2f} km²")

    # 2. LARGE SCALES: Use observed PSRs from Diviner
    if transition_idx < len(L_bins):
        print("\n  [Large scales: Observed PSRs from Diviner temperature data]")

        # Bin observed PSRs into large-scale bins
        N_north_obs, N_south_obs, A_north_obs, A_south_obs = bin_large_psrs(
            large_psrs_data, L_bins)

        # Assign to total arrays (only for large bins)
        N_north_total[transition_idx:] = N_north_obs[transition_idx:]
        N_south_total[transition_idx:] = N_south_obs[transition_idx:]
        A_north_total[transition_idx:] = A_north_obs[transition_idx:]
        A_south_total[transition_idx:] = A_south_obs[transition_idx:]

        print(f"    Observed PSRs North: {N_north_obs[transition_idx:].sum():.0f}")
        print(f"    Observed PSRs South: {N_south_obs[transition_idx:].sum():.0f}")
        print(f"    Observed cold trap area North: {A_north_obs[transition_idx:].sum():.2f} km²")
        print(f"    Observed cold trap area South: {A_south_obs[transition_idx:].sum():.2f} km²")

    return L_bins, N_north_total, N_south_total, A_north_total, A_south_total


def plot_figure4_top_panel(L_bins, A_north, A_south,
                           output_path='/home/user/documents/figure4_top_panel.png'):
    """
    Create Figure 4 top panel: Cumulative cold trap area vs length scale.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Calculate cumulative areas
    A_north_cum = np.cumsum(A_north)
    A_south_cum = np.cumsum(A_south)

    # Plot cumulative areas
    ax.loglog(L_bins, A_north_cum, 'b-', linewidth=2.5,
              label='Northern Hemisphere', marker='o', markersize=5, markevery=10)
    ax.loglog(L_bins, A_south_cum, 'r-', linewidth=2.5,
              label='Southern Hemisphere', marker='s', markersize=5, markevery=10)

    # Add reference lines
    ax.axvline(x=LATERAL_CONDUCTION_LIMIT, color='gray', linestyle='--',
               alpha=0.7, linewidth=2, label='Lateral conduction limit (1 cm)')
    ax.axvline(x=TRANSITION_SCALE, color='purple', linestyle=':',
               alpha=0.6, linewidth=2, label=f'Transition to observed PSRs ({TRANSITION_SCALE/1000:.0f} km)')

    # Mark smallest PSR from geodata (diameter = 270.81 m)
    SMALLEST_PSR_DIAMETER = 270.81  # m
    ax.axvline(x=SMALLEST_PSR_DIAMETER, color='green', linestyle=':',
               alpha=0.8, linewidth=1.5, label=f'Smallest PSR ({SMALLEST_PSR_DIAMETER:.1f} m)')

    ax.set_xlabel('Length Scale L [m]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Cold Trap Area < L [km²]', fontsize=14, fontweight='bold')
    ax.set_title('Figure 4 Top Panel: Cumulative Cold Trap Area\n' +
                 '(Synthetic <1km + Observed ≥1km from Diviner)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, which='both', linestyle=':')
    ax.set_xlim([L_bins[0], L_bins[-1]])
    ax.tick_params(labelsize=12)

    # Add annotations
    total_north = A_north_cum[-1]
    total_south = A_south_cum[-1]
    total_both = total_north + total_south

    ax.text(0.95, 0.35, f'Total North: {total_north:,.0f} km²',
            transform=ax.transAxes, fontsize=11, ha='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.text(0.95, 0.25, f'Total South: {total_south:,.0f} km²',
            transform=ax.transAxes, fontsize=11, ha='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    ax.text(0.95, 0.15, f'TOTAL: {total_both:,.0f} km²',
            transform=ax.transAxes, fontsize=12, fontweight='bold', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def report_minimum_L(L_bins, A_north, A_south):
    """
    Report the minimum length scale L.
    """
    print("\n" + "=" * 80)
    print("MINIMUM LENGTH SCALE L")
    print("=" * 80)

    # Find first bin with non-zero area
    total_area = A_north + A_south
    nonzero_idx = np.where(total_area > 0)[0]

    if len(nonzero_idx) > 0:
        min_L = L_bins[nonzero_idx[0]]
        print(f"\nMinimum L with cold traps: {min_L:.6f} m ({min_L*1e6:.2f} µm)")
        print(f"  Cold trap area at min L: {total_area[nonzero_idx[0]]:.6f} km²")
    else:
        print("\nNo cold traps found in distribution")

    # Overall range
    print(f"\nLength scale range:")
    print(f"  Minimum L (bin): {L_bins[0]:.6f} m ({L_bins[0]*1e6:.2f} µm)")
    print(f"  Maximum L (bin): {L_bins[-1]:.2f} m ({L_bins[-1]/1000:.2f} km)")

    print("\n" + "=" * 80)


def print_summary(L_bins, N_north, N_south, A_north, A_south):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("FIGURE 4 TOP PANEL SUMMARY")
    print("=" * 80)

    # Find transition index
    transition_idx = np.searchsorted(L_bins, TRANSITION_SCALE)

    # Small scale totals
    A_north_small = A_north[:transition_idx].sum()
    A_south_small = A_south[:transition_idx].sum()

    # Large scale totals
    A_north_large = A_north[transition_idx:].sum()
    A_south_large = A_south[transition_idx:].sum()

    # Overall totals
    total_north = A_north.sum()
    total_south = A_south.sum()
    total_both = total_north + total_south

    print(f"\nSMALL SCALES (<{TRANSITION_SCALE}m) - Synthetic Model:")
    print(f"  North: {A_north_small:,.2f} km²")
    print(f"  South: {A_south_small:,.2f} km²")
    print(f"  Total: {A_north_small + A_south_small:,.2f} km²")

    print(f"\nLARGE SCALES (≥{TRANSITION_SCALE}m) - Observed PSRs from Diviner:")
    print(f"  North: {A_north_large:,.2f} km²")
    print(f"  South: {A_south_large:,.2f} km²")
    print(f"  Total: {A_north_large + A_south_large:,.2f} km²")
    print(f"  North PSR count: {N_north[transition_idx:].sum():.0f}")
    print(f"  South PSR count: {N_south[transition_idx:].sum():.0f}")

    print(f"\nOVERALL TOTALS:")
    print(f"  Northern Hemisphere: {total_north:,.2f} km²")
    print(f"  Southern Hemisphere: {total_south:,.2f} km²")
    print(f"  TOTAL COLD TRAP AREA: {total_both:,.2f} km²")
    print(f"  Fraction of lunar surface: {total_both/LUNAR_SURFACE_AREA*100:.4f}%")
    print(f"  South/North ratio: {total_south/total_north:.2f}")

    print("\n" + "=" * 80)


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("FIGURE 4 TOP PANEL: CUMULATIVE COLD TRAP AREA")
    print("Combining Diviner Temperature Data (≥1km) + Synthetic Model (<1km)")
    print("=" * 80)

    # Load PSR data with Diviner temperatures
    all_psrs = load_psr_data_with_temperatures()

    # Filter for large PSRs (>= 1 km²)
    large_psrs_data = filter_large_psrs(all_psrs, min_diameter_m=TRANSITION_SCALE)

    # Create hybrid distribution
    L_bins, N_north, N_south, A_north, A_south = create_hybrid_distribution(
        large_psrs_data, L_min=1e-4, L_max=100000, n_bins=100)

    # Plot Figure 4 top panel
    plot_figure4_top_panel(L_bins, A_north, A_south)

    # Print summary
    print_summary(L_bins, N_north, N_south, A_north, A_south)

    # Report minimum L
    report_minimum_L(L_bins, A_north, A_south)

    print("\n✓ COMPLETE: Figure 4 top panel with Diviner temperature data")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Remake Figure 4 Top Panel: Cumulative Cold Trap Area by Cold Trap Size

This version plots cumulative area of cold traps whose size < L:
1. For each cold trap, calculate its equivalent diameter D_ct
2. For synthetic PSRs: D_ct = D_psr * sqrt(f_ct) where f_ct is Hayne cold trap fraction
3. For observed PSRs: D_ct = 2*sqrt(A_ct/π) where A_ct is measured cold trap area
4. Plot: Cumulative area of cold traps with D_ct < L vs L
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
LATERAL_CONDUCTION_LIMIT = 0.001  # 1 mm in meters (extended from 1 cm to enable sub-cm cold traps)
TRANSITION_SCALE = 1000.0  # 1 km - transition from synthetic to observed
LUNAR_SURFACE_AREA = 3.793e7  # km²

# File path
PSR_CSV = '/home/user/documents/psr_with_temperatures.csv'


def load_psr_data_with_temperatures():
    """Load PSR data with Diviner temperatures from CSV."""
    print("\n[Loading PSR data with Diviner temperatures]")

    psr = pd.read_csv(PSR_CSV)

    # Calculate diameter from area (assuming circular)
    psr['diameter_m'] = 2 * np.sqrt(psr['area_km2'] * 1e6 / np.pi)

    # Calculate cold trap area from pixel count
    psr['coldtrap_fraction'] = 0.0
    mask = psr['pixel_count'] > 0
    psr.loc[mask, 'coldtrap_fraction'] = psr.loc[mask, 'pixels_lt_110K'] / psr.loc[mask, 'pixel_count']
    psr['coldtrap_area_km2'] = psr['coldtrap_fraction'] * psr['area_km2']

    # Calculate cold trap equivalent diameter
    # D_ct = 2*sqrt(A_ct/π)
    psr['coldtrap_diameter_m'] = 0.0
    ct_mask = psr['coldtrap_area_km2'] > 0
    psr.loc[ct_mask, 'coldtrap_diameter_m'] = 2 * np.sqrt(
        psr.loc[ct_mask, 'coldtrap_area_km2'] * 1e6 / np.pi)

    print(f"✓ Loaded {len(psr)} PSRs with Diviner temperature data")
    print(f"  Total PSR area: {psr['area_km2'].sum():.2f} km²")
    print(f"  Total cold trap area (< 110K): {psr['coldtrap_area_km2'].sum():.2f} km²")
    print(f"  PSRs with cold traps: {(psr['coldtrap_area_km2'] > 0).sum()}")
    print(f"  Cold trap diameter range: {psr[ct_mask]['coldtrap_diameter_m'].min():.1f} - {psr[ct_mask]['coldtrap_diameter_m'].max():.1f} m")

    return psr


def generate_synthetic_coldtraps(L_min=1e-4, L_max=TRANSITION_SCALE, n_bins=100):
    """
    Generate synthetic cold traps for small scales.

    Returns list of individual cold traps with (diameter_m, area_km2, hemisphere).
    """
    print("\n[Generating synthetic cold traps]")

    # Create logarithmic bins for PSR diameters
    L_bins = np.logspace(np.log10(L_min), np.log10(L_max), n_bins)
    dL = np.diff(np.logspace(np.log10(L_min), np.log10(L_max), n_bins + 1))

    # Power-law exponent
    b = 1.8

    # Scale factor
    K = 2e11

    # Differential number density
    N_diff = K * L_bins**(-b - 1)
    N_per_bin = N_diff * dL

    # Hemisphere asymmetry
    N_north_bins = N_per_bin * 0.40
    N_south_bins = N_per_bin * 0.60

    # Representative latitudes
    lat_north = 85.0
    lat_south = -85.0

    # Terrain parameters
    sigma_s_plains = 5.7
    sigma_s_craters = 20.0
    f_craters = 0.20
    f_plains = 0.80

    # Generate individual cold traps
    coldtraps = []

    for i, L in enumerate(L_bins):
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

        # Number of PSRs in this bin
        n_north = N_north_bins[i]
        n_south = N_south_bins[i]

        # For each PSR of diameter L with cold trap fraction f_ct:
        # Cold trap has effective diameter D_ct = L * sqrt(f_ct)
        # Cold trap area = π(D_ct/2)² = π(L/2)² * f_ct

        D_ct_north = L * np.sqrt(f_ct_north)
        D_ct_south = L * np.sqrt(f_ct_south)

        A_ct_north = np.pi * (D_ct_north / 2.0)**2 * 1e-6  # km² per cold trap
        A_ct_south = np.pi * (D_ct_south / 2.0)**2 * 1e-6  # km² per cold trap

        # Add individual cold traps
        # Note: We add fractional counts since N can be large
        if n_north > 0 and A_ct_north > 0:
            coldtraps.append({
                'diameter_m': D_ct_north,
                'area_km2': A_ct_north * n_north,  # Total area from all n_north cold traps
                'hemisphere': 'North',
                'count': n_north  # Number of cold traps at this size
            })

        if n_south > 0 and A_ct_south > 0:
            coldtraps.append({
                'diameter_m': D_ct_south,
                'area_km2': A_ct_south * n_south,  # Total area from all n_south cold traps
                'hemisphere': 'South',
                'count': n_south
            })

    print(f"  Generated {len(coldtraps)} synthetic cold trap size bins")
    total_area = sum(ct['area_km2'] for ct in coldtraps)
    print(f"  Total synthetic cold trap area: {total_area:.2f} km²")

    return coldtraps


def extract_observed_coldtraps(psr_data, min_diameter_m=TRANSITION_SCALE):
    """
    Extract observed cold traps from large PSRs.

    Returns list of cold traps with (diameter_m, area_km2, hemisphere).
    """
    print("\n[Extracting observed cold traps from large PSRs]")

    # Filter for large PSRs with cold traps
    large_psrs = psr_data[psr_data['diameter_m'] >= min_diameter_m].copy()
    large_psrs_with_ct = large_psrs[large_psrs['coldtrap_area_km2'] > 0].copy()

    print(f"  Large PSRs (D >= {min_diameter_m}m): {len(large_psrs)}")
    print(f"  Large PSRs with cold traps: {len(large_psrs_with_ct)}")

    coldtraps = []
    for _, row in large_psrs_with_ct.iterrows():
        coldtraps.append({
            'diameter_m': row['coldtrap_diameter_m'],
            'area_km2': row['coldtrap_area_km2'],
            'hemisphere': row['hemisphere'],
            'count': 1  # Each PSR contributes one cold trap
        })

    total_area = sum(ct['area_km2'] for ct in coldtraps)
    print(f"  Total observed cold trap area: {total_area:.2f} km²")

    return coldtraps


def calculate_cumulative_by_size(coldtraps, L_values):
    """
    Calculate cumulative cold trap area for cold traps with diameter < L.

    Args:
        coldtraps: List of dicts with 'diameter_m', 'area_km2', 'hemisphere'
        L_values: Array of diameter thresholds [m]

    Returns:
        cum_area_north, cum_area_south: Cumulative areas at each L threshold
    """
    print("\n[Calculating cumulative areas by cold trap size]")

    # Separate by hemisphere
    north_cts = [ct for ct in coldtraps if ct['hemisphere'] == 'North']
    south_cts = [ct for ct in coldtraps if ct['hemisphere'] == 'South']

    # Sort by diameter
    north_cts.sort(key=lambda x: x['diameter_m'])
    south_cts.sort(key=lambda x: x['diameter_m'])

    cum_area_north = np.zeros(len(L_values))
    cum_area_south = np.zeros(len(L_values))

    # For each L threshold, sum area of all cold traps with diameter < L
    for i, L in enumerate(L_values):
        # North
        for ct in north_cts:
            if ct['diameter_m'] < L:
                cum_area_north[i] += ct['area_km2']
            else:
                break  # Sorted, so we can stop

        # South
        for ct in south_cts:
            if ct['diameter_m'] < L:
                cum_area_south[i] += ct['area_km2']
            else:
                break

    print(f"  Total cold trap area (North): {cum_area_north[-1]:.2f} km²")
    print(f"  Total cold trap area (South): {cum_area_south[-1]:.2f} km²")
    print(f"  Total: {cum_area_north[-1] + cum_area_south[-1]:.2f} km²")

    return cum_area_north, cum_area_south


def plot_cumulative_coldtrap_area(L_values, cum_area_north, cum_area_south,
                                   output_path='/home/user/documents/figure4_top_panel.png'):
    """
    Create Figure 4 top panel: Cumulative cold trap area vs cold trap diameter.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot cumulative areas
    ax.loglog(L_values, cum_area_north, 'b-', linewidth=2.5,
              label='Northern Hemisphere', marker='o', markersize=5, markevery=10)
    ax.loglog(L_values, cum_area_south, 'r-', linewidth=2.5,
              label='Southern Hemisphere', marker='s', markersize=5, markevery=10)

    # Add reference lines
    ax.axvline(x=LATERAL_CONDUCTION_LIMIT, color='gray', linestyle='--',
               alpha=0.7, linewidth=2, label='Lateral conduction limit (1 cm)')
    ax.axvline(x=TRANSITION_SCALE, color='purple', linestyle=':',
               alpha=0.6, linewidth=2, label=f'Transition to observed data ({TRANSITION_SCALE/1000:.0f} km)')

    ax.set_xlabel('Cold Trap Diameter L [m]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Cold Trap Area < L [km²]', fontsize=14, fontweight='bold')
    ax.set_title('Figure 4 Top Panel: Cumulative Area of Cold Traps < L\n' +
                 '(Binned by cold trap size, not PSR size)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, which='both', linestyle=':')
    ax.set_xlim([L_values[0], L_values[-1]])
    ax.set_ylim([1e-2, 1e5])
    ax.tick_params(labelsize=12)

    # Add annotations
    total_north = cum_area_north[-1]
    total_south = cum_area_south[-1]
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


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("FIGURE 4 TOP PANEL: CUMULATIVE COLD TRAP AREA BY COLD TRAP SIZE")
    print("Plots cumulative area of cold traps with diameter < L")
    print("=" * 80)

    # Load PSR data with Diviner temperatures
    all_psrs = load_psr_data_with_temperatures()

    # Generate synthetic cold traps for small scales
    synthetic_cts = generate_synthetic_coldtraps(L_min=1e-4, L_max=TRANSITION_SCALE, n_bins=100)

    # Extract observed cold traps from large PSRs
    observed_cts = extract_observed_coldtraps(all_psrs, min_diameter_m=TRANSITION_SCALE)

    # Combine all cold traps
    all_coldtraps = synthetic_cts + observed_cts

    print(f"\n[Combined cold trap inventory]")
    print(f"  Synthetic cold trap bins: {len(synthetic_cts)}")
    print(f"  Observed cold traps: {len(observed_cts)}")
    print(f"  Total entries: {len(all_coldtraps)}")

    # Define L values for cumulative calculation
    L_values = np.logspace(np.log10(1e-4), np.log10(100000), 200)

    # Calculate cumulative areas
    cum_area_north, cum_area_south = calculate_cumulative_by_size(all_coldtraps, L_values)

    # Plot
    plot_cumulative_coldtrap_area(L_values, cum_area_north, cum_area_south)

    # Summary
    total_area = cum_area_north[-1] + cum_area_south[-1]
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total cumulative cold trap area: {total_area:,.2f} km²")
    print(f"Fraction of lunar surface: {total_area/LUNAR_SURFACE_AREA*100:.4f}%")
    print(f"South/North ratio: {cum_area_south[-1]/cum_area_north[-1]:.2f}")
    print("\n✓ COMPLETE: Figure 4 top panel by cold trap size")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

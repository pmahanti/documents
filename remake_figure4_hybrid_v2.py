#!/usr/bin/env python3
"""
Remake Figure 4: Hybrid PSR and Cold Trap Size Distribution (v2)

This version properly combines:
1. ACTUAL large PSRs (≥300m) from geodata, but applies HAYNE MODEL
   to estimate cold trap fraction within each PSR due to micro-roughness
2. Synthetic power-law distribution for small scales (<300m)

Key insight: Large PSRs may have mean temp >110K (from Diviner at ~240m resolution),
but still contain micro-scale cold traps due to sub-pixel roughness. We apply the
Hayne model to account for this multi-scale physics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from hayne_model_corrected import hayne_cold_trap_fraction_corrected

# Physical constants
SIGMA_SB = 5.67051e-8
LUNAR_SURFACE_AREA = 3.793e7  # km²
LUNAR_RADIUS = 1737.4  # km
LATERAL_CONDUCTION_LIMIT = 0.01  # 1 cm
COLD_TRAP_THRESHOLD = 110.0  # K

# Transition scale between synthetic and observed data
TRANSITION_SCALE = 300.0  # m


def load_observed_psrs(filepath='/home/user/documents/psr_with_temperatures.csv'):
    """Load actual PSR data from geodata package."""
    print("\n[Loading observed PSR data from geodata package]")
    psr = pd.read_csv(filepath)

    # Calculate diameter from area
    psr['diameter_m'] = 2 * np.sqrt(psr['area_km2'] * 1e6 / np.pi)

    print(f"✓ Loaded {len(psr)} observed PSRs")
    print(f"  Total PSR area: {psr['area_km2'].sum():.2f} km²")
    print(f"  Diameter range: {psr['diameter_m'].min():.1f} - {psr['diameter_m'].max():.1f} m")

    north = psr[psr['hemisphere'] == 'North']
    south = psr[psr['hemisphere'] == 'South']

    print(f"\n  North: {len(north)} PSRs, {north['area_km2'].sum():.2f} km²")
    print(f"  South: {len(south)} PSRs, {south['area_km2'].sum():.2f} km²")

    return psr


def bin_observed_psrs_with_hayne_model(psr, L_bins):
    """
    Bin observed PSRs and apply Hayne model for cold trap fraction.

    For each large PSR, we apply the Hayne model to estimate what fraction
    is cold trapped due to micro-scale roughness, even if the Diviner mean
    temperature exceeds 110K.

    Args:
        psr: DataFrame with PSR data
        L_bins: Length scale bin centers [m]

    Returns:
        N_north, N_south, A_north, A_south
    """
    N_north = np.zeros(len(L_bins))
    N_south = np.zeros(len(L_bins))
    A_north = np.zeros(len(L_bins))  # Cold trap area
    A_south = np.zeros(len(L_bins))

    # Create bin edges
    L_edges = np.zeros(len(L_bins) + 1)
    L_edges[0] = L_bins[0] / (L_bins[1]/L_bins[0])**0.5
    L_edges[-1] = L_bins[-1] * (L_bins[-1]/L_bins[-2])**0.5
    for i in range(1, len(L_bins)):
        L_edges[i] = np.sqrt(L_bins[i-1] * L_bins[i])

    # RMS slopes for cold trap fraction calculation
    sigma_s_plains = 5.7
    sigma_s_craters = 20.0
    f_craters = 0.20
    f_plains = 0.80

    # Bin the PSRs
    for _, row in psr.iterrows():
        diameter = row['diameter_m']
        latitude = row['latitude']
        psr_area = row['area_km2']

        # Find bin
        bin_idx = np.searchsorted(L_edges, diameter) - 1

        if 0 <= bin_idx < len(L_bins):
            # Calculate cold trap fraction using Hayne model
            # Even if Diviner shows temp > 110K, there may be micro-cold-traps
            f_ct_crater = hayne_cold_trap_fraction_corrected(sigma_s_craters, latitude)
            f_ct_plains = hayne_cold_trap_fraction_corrected(sigma_s_plains, latitude)
            f_ct = f_craters * f_ct_crater + f_plains * f_ct_plains

            # Cold trap area within this PSR
            cold_trap_area = psr_area * f_ct

            if row['hemisphere'] == 'North':
                N_north[bin_idx] += 1
                A_north[bin_idx] += cold_trap_area
            else:  # South
                N_south[bin_idx] += 1
                A_south[bin_idx] += cold_trap_area

    return N_north, N_south, A_north, A_south


def generate_synthetic_distribution(L_min=1e-4, L_max=TRANSITION_SCALE, n_bins=50):
    """Generate synthetic size-frequency distribution for small PSRs."""
    L_bins = np.logspace(np.log10(L_min), np.log10(L_max), n_bins)
    dL = np.diff(np.logspace(np.log10(L_min), np.log10(L_max), n_bins + 1))

    b = 1.8
    K = 2e11

    N_diff = K * L_bins**(-b - 1)
    N_per_bin = N_diff * dL

    N_north = N_per_bin * 0.40
    N_south = N_per_bin * 0.60

    return {
        'L_bins': L_bins,
        'N_north': N_north,
        'N_south': N_south,
        'dL': dL
    }


def calculate_synthetic_cold_trap_areas(L_bins, N_north, N_south):
    """Calculate cold trap areas for synthetic PSRs using Hayne model."""
    A_north = np.zeros_like(L_bins)
    A_south = np.zeros_like(L_bins)

    lat_north = 85.0
    lat_south = -85.0

    sigma_s_plains = 5.7
    sigma_s_craters = 20.0
    f_craters = 0.20
    f_plains = 0.80

    for i, (L, n_n, n_s) in enumerate(zip(L_bins, N_north, N_south)):
        if L < LATERAL_CONDUCTION_LIMIT:
            continue

        f_ct_crater_north = hayne_cold_trap_fraction_corrected(sigma_s_craters, lat_north)
        f_ct_crater_south = hayne_cold_trap_fraction_corrected(sigma_s_craters, lat_south)
        f_ct_plains_north = hayne_cold_trap_fraction_corrected(sigma_s_plains, lat_north)
        f_ct_plains_south = hayne_cold_trap_fraction_corrected(sigma_s_plains, lat_south)

        f_ct_north = f_craters * f_ct_crater_north + f_plains * f_ct_plains_north
        f_ct_south = f_craters * f_ct_crater_south + f_plains * f_ct_plains_south

        area_per_feature = np.pi * (L / 2.0)**2  # m²

        A_north[i] = n_n * area_per_feature * f_ct_north * 1e-6  # km²
        A_south[i] = n_s * area_per_feature * f_ct_south * 1e-6  # km²

    return A_north, A_south


def create_hybrid_distribution_v2(psr_data, L_min=1e-4, L_max=100000, n_bins=100):
    """
    Create hybrid distribution combining synthetic and observed data.

    Key difference from v1: Apply Hayne model to observed large PSRs to account
    for micro-scale cold traps within them.
    """
    print("\n[Creating hybrid distribution v2]")
    print("  Using Hayne model for cold trap fraction on ALL PSRs")

    L_bins = np.logspace(np.log10(L_min), np.log10(L_max), n_bins)

    N_north_total = np.zeros(len(L_bins))
    N_south_total = np.zeros(len(L_bins))
    A_north_total = np.zeros(len(L_bins))
    A_south_total = np.zeros(len(L_bins))

    transition_idx = np.searchsorted(L_bins, TRANSITION_SCALE)

    print(f"  Total bins: {len(L_bins)}")
    print(f"  Bins < {TRANSITION_SCALE}m (synthetic): {transition_idx}")
    print(f"  Bins ≥ {TRANSITION_SCALE}m (observed): {len(L_bins) - transition_idx}")

    # 1. SMALL SCALES: Synthetic distribution
    if transition_idx > 0:
        print("\n  [Small scales: Synthetic power-law + Hayne model]")

        synth = generate_synthetic_distribution(L_min=L_bins[0], L_max=L_bins[transition_idx-1],
                                                n_bins=transition_idx)

        A_north_synth, A_south_synth = calculate_synthetic_cold_trap_areas(
            synth['L_bins'], synth['N_north'], synth['N_south'])

        N_north_total[:transition_idx] = synth['N_north']
        N_south_total[:transition_idx] = synth['N_south']
        A_north_total[:transition_idx] = A_north_synth
        A_south_total[:transition_idx] = A_south_synth

        print(f"    Synthetic cold trap area North: {A_north_synth.sum():.2f} km²")
        print(f"    Synthetic cold trap area South: {A_south_synth.sum():.2f} km²")

    # 2. LARGE SCALES: Observed PSRs with Hayne model applied
    if transition_idx < len(L_bins):
        print("\n  [Large scales: Observed PSRs + Hayne model for micro-cold-traps]")

        N_north_obs, N_south_obs, A_north_obs, A_south_obs = \
            bin_observed_psrs_with_hayne_model(psr_data, L_bins)

        N_north_total[transition_idx:] = N_north_obs[transition_idx:]
        N_south_total[transition_idx:] = N_south_obs[transition_idx:]
        A_north_total[transition_idx:] = A_north_obs[transition_idx:]
        A_south_total[transition_idx:] = A_south_obs[transition_idx:]

        print(f"    Observed PSRs North: {N_north_obs[transition_idx:].sum():.0f}")
        print(f"    Observed PSRs South: {N_south_obs[transition_idx:].sum():.0f}")
        print(f"    Estimated cold trap area North: {A_north_obs[transition_idx:].sum():.2f} km²")
        print(f"    Estimated cold trap area South: {A_south_obs[transition_idx:].sum():.2f} km²")

    return L_bins, N_north_total, N_south_total, A_north_total, A_south_total


def plot_figure4_hybrid(L_bins, N_north, N_south, A_north, A_south,
                        output_path='/home/user/documents/figure4_hybrid_v2.png'):
    """Create Figure 4 with hybrid distribution."""
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 11))

    # TOP PANEL: Cumulative Cold Trap Area
    A_north_cum = np.cumsum(A_north)
    A_south_cum = np.cumsum(A_south)

    ax_top.loglog(L_bins, A_north_cum, 'b-', linewidth=2.5,
                  label='Northern Hemisphere', marker='o', markersize=5, markevery=10)
    ax_top.loglog(L_bins, A_south_cum, 'r-', linewidth=2.5,
                  label='Southern Hemisphere', marker='s', markersize=5, markevery=10)

    ax_top.axvline(x=LATERAL_CONDUCTION_LIMIT, color='gray', linestyle='--',
                   alpha=0.7, linewidth=2, label='Lateral conduction limit (1 cm)')
    ax_top.axvline(x=TRANSITION_SCALE, color='purple', linestyle=':',
                   alpha=0.6, linewidth=2, label=f'Observed PSRs ≥{TRANSITION_SCALE}m')
    ax_top.axvline(x=1000.0, color='orange', linestyle=':', alpha=0.6, linewidth=1.5,
                   label='1 km scale')

    ax_top.set_xlabel('Length Scale L [m]', fontsize=14, fontweight='bold')
    ax_top.set_ylabel('Cumulative Cold Trap Area [km²]', fontsize=14, fontweight='bold')
    ax_top.set_title('Cumulative area of cold traps (<110 K) - HYBRID MODEL v2\n' +
                     '(Observed large PSRs + Hayne model for micro-scale cold traps)',
                     fontsize=12, fontweight='bold', pad=10)
    ax_top.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax_top.grid(True, alpha=0.3, which='both', linestyle=':')
    ax_top.set_xlim([L_bins[0], L_bins[-1]])
    ax_top.tick_params(labelsize=12)

    total_north = A_north_cum[-1]
    total_south = A_south_cum[-1]
    total_both = total_north + total_south

    ax_top.text(0.95, 0.35, f'Total North: {total_north:,.0f} km²',
                transform=ax_top.transAxes, fontsize=11, ha='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax_top.text(0.95, 0.25, f'Total South: {total_south:,.0f} km²',
                transform=ax_top.transAxes, fontsize=11, ha='right',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    ax_top.text(0.95, 0.15, f'TOTAL: {total_both:,.0f} km²',
                transform=ax_top.transAxes, fontsize=12, fontweight='bold', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # BOTTOM PANEL: Number of PSRs
    ax_bottom.loglog(L_bins, N_north, 'b-', linewidth=2.5,
                     label='Northern Hemisphere', marker='o', markersize=5, markevery=10)
    ax_bottom.loglog(L_bins, N_south, 'r-', linewidth=2.5,
                     label='Southern Hemisphere', marker='s', markersize=5, markevery=10)

    ax_bottom.axvline(x=LATERAL_CONDUCTION_LIMIT, color='gray', linestyle='--',
                      alpha=0.7, linewidth=2, label='Lateral conduction limit (1 cm)')
    ax_bottom.axvline(x=TRANSITION_SCALE, color='purple', linestyle=':',
                      alpha=0.6, linewidth=2, label=f'Observed PSRs ≥{TRANSITION_SCALE}m')

    ax_bottom.set_xlabel('Length Scale L [m]', fontsize=14, fontweight='bold')
    ax_bottom.set_ylabel('Number of PSRs/Cold Traps', fontsize=14, fontweight='bold')
    ax_bottom.set_title('Number of individual PSRs - HYBRID MODEL v2',
                        fontsize=13, fontweight='bold', pad=10)
    ax_bottom.legend(fontsize=10, loc='upper right', framealpha=0.95)
    ax_bottom.grid(True, alpha=0.3, which='both', linestyle=':')
    ax_bottom.set_xlim([L_bins[0], L_bins[-1]])
    ax_bottom.tick_params(labelsize=12)

    plt.suptitle('Figure 4: PSR and Cold-Trapping Areas\n' +
                 '(Geodata large PSRs + Hayne Model Cold Trap Fractions)',
                 fontsize=15, fontweight='bold', y=0.996)

    plt.tight_layout(rect=[0, 0, 1, 0.988])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def print_summary(L_bins, N_north, N_south, A_north, A_south):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("HYBRID FIGURE 4 v2 SUMMARY")
    print("=" * 80)

    transition_idx = np.searchsorted(L_bins, TRANSITION_SCALE)

    A_north_small = A_north[:transition_idx].sum()
    A_south_small = A_south[:transition_idx].sum()
    A_north_large = A_north[transition_idx:].sum()
    A_south_large = A_south[transition_idx:].sum()

    total_north = A_north.sum()
    total_south = A_south.sum()
    total_both = total_north + total_south

    print(f"\nSMALL SCALES (<{TRANSITION_SCALE}m) - Synthetic + Hayne Model:")
    print(f"  North: {A_north_small:,.2f} km²")
    print(f"  South: {A_south_small:,.2f} km²")
    print(f"  Total: {A_north_small + A_south_small:,.2f} km²")

    print(f"\nLARGE SCALES (≥{TRANSITION_SCALE}m) - Observed PSRs + Hayne Model:")
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

    # Check large scales specifically (≥1km)
    idx_1km = np.searchsorted(L_bins, 1000)
    A_north_1km = A_north[idx_1km:].sum()
    A_south_1km = A_south[idx_1km:].sum()
    N_north_1km = N_north[idx_1km:].sum()
    N_south_1km = N_south[idx_1km:].sum()

    print(f"\nLARGE SCALE (≥1km) DETAILS:")
    print(f"  North PSRs ≥1km: {N_north_1km:.0f}, Cold trap area: {A_north_1km:.2f} km²")
    print(f"  South PSRs ≥1km: {N_south_1km:.0f}, Cold trap area: {A_south_1km:.2f} km²")
    print(f"  Total ≥1km: {N_north_1km + N_south_1km:.0f} PSRs, {A_north_1km + A_south_1km:.2f} km²")

    print("\n" + "=" * 80)


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("REMAKE FIGURE 4 WITH HYBRID APPROACH v2")
    print("Observed Large PSRs + Hayne Model for Micro-Scale Cold Traps")
    print("=" * 80)

    psr_data = load_observed_psrs()

    L_bins, N_north, N_south, A_north, A_south = create_hybrid_distribution_v2(
        psr_data, L_min=1e-4, L_max=100000, n_bins=100)

    plot_figure4_hybrid(L_bins, N_north, N_south, A_north, A_south)

    print_summary(L_bins, N_north, N_south, A_north, A_south)

    print("\n✓ COMPLETE: Figure 4 hybrid v2")
    print("  Key: Large PSRs use observed counts from geodata,")
    print("       but cold trap areas use Hayne model fractions")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

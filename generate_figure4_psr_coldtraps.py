#!/usr/bin/env python3
"""
Generate Figure 4: PSR and Cold Trap Size Distribution Analysis

Permanently shadowed and cold-trapping areas as a function of
size in the northern and southern hemispheres.

Top panel: Cumulative area of cold traps (<110 K) at all latitudes, as a function of L
Bottom panel: Modeled number of individual PSRs and cold traps on the Moon
Length-scale bins are logarithmically spaced.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, List, Tuple

# Import thermal models
from bowl_crater_thermal import CraterGeometry, crater_cold_trap_area

# Constants
SIGMA_SB = 5.67051e-8
LUNAR_SURFACE_AREA = 3.793e7  # km²


def generate_crater_size_distribution(L_min: float = 0.01, L_max: float = 100000,
                                     n_bins: int = 50) -> Dict[str, np.ndarray]:
    """
    Generate crater size-frequency distribution for PSRs and cold traps.

    Uses power-law size-frequency distribution similar to lunar crater populations.

    Parameters:
    -----------
    L_min : float
        Minimum length scale [m] (default: 1 cm)
    L_max : float
        Maximum length scale [m] (default: 100 km)
    n_bins : int
        Number of logarithmically-spaced bins

    Returns:
    --------
    dict containing:
        - 'L_bins': Length scale bin centers [m]
        - 'N_north': Number of features per bin (northern hemisphere)
        - 'N_south': Number of features per bin (southern hemisphere)
        - 'A_north': Cold trap area per bin [km²] (northern)
        - 'A_south': Cold trap area per bin [km²] (southern)
    """
    # Logarithmically spaced length scales
    L_bins = np.logspace(np.log10(L_min), np.log10(L_max), n_bins)

    # Power-law exponent for crater size-frequency distribution
    # N(>D) ∝ D^(-b) where b ≈ 2 for lunar craters
    b = 2.0

    # Differential number density: dN/dD ∝ D^(-b-1)
    # Scale by total lunar surface area
    scale_factor = 1e10  # Total number of small features

    N_differential = scale_factor * L_bins**(-b - 1)

    # Convert to number per bin (multiply by bin width)
    dL = np.diff(np.logspace(np.log10(L_min), np.log10(L_max), n_bins + 1))
    N_per_bin = N_differential * dL

    # Hemisphere asymmetry: South polar region has more PSRs due to topography
    # Based on observations showing ~60% of PSRs in south vs ~40% in north
    north_fraction = 0.40
    south_fraction = 0.60

    N_north = N_per_bin * north_fraction
    N_south = N_per_bin * south_fraction

    return {
        'L_bins': L_bins,
        'N_north': N_north,
        'N_south': N_south,
        'dL': dL
    }


def calculate_cold_trap_areas(L_bins: np.ndarray, N_north: np.ndarray,
                              N_south: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate total cold trap area per length scale bin for each hemisphere.

    Parameters:
    -----------
    L_bins : np.ndarray
        Length scale bin centers [m]
    N_north : np.ndarray
        Number of features per bin (northern hemisphere)
    N_south : np.ndarray
        Number of features per bin (southern hemisphere)

    Returns:
    --------
    A_north, A_south : tuple of np.ndarray
        Total cold trap area per bin [km²] for north and south hemispheres
    """
    # Depth-to-diameter ratios for degraded craters (where PSRs form)
    gamma_degraded = 0.076

    # Average latitudes for polar regions
    lat_north = 85.0  # degrees North
    lat_south = -85.0  # degrees South

    A_north = np.zeros_like(L_bins)
    A_south = np.zeros_like(L_bins)

    for i, (D, n_north, n_south) in enumerate(zip(L_bins, N_north, N_south)):
        # Skip if below lateral conduction limit (~1 cm)
        if D < 0.01:
            continue

        # Calculate depth from diameter
        d = gamma_degraded * D

        # Northern hemisphere
        try:
            crater_north = CraterGeometry(D, d, lat_north)
            cold_trap_north = crater_cold_trap_area(crater_north, T_threshold=110.0)
            area_per_crater_north = cold_trap_north['cold_trap_area']  # m²
            A_north[i] = n_north * area_per_crater_north * 1e-6  # Convert to km²
        except:
            A_north[i] = 0.0

        # Southern hemisphere
        try:
            crater_south = CraterGeometry(D, d, lat_south)
            cold_trap_south = crater_cold_trap_area(crater_south, T_threshold=110.0)
            area_per_crater_south = cold_trap_south['cold_trap_area']  # m²
            A_south[i] = n_south * area_per_crater_south * 1e-6  # Convert to km²
        except:
            A_south[i] = 0.0

    return A_north, A_south


def plot_figure4(L_bins: np.ndarray, N_north: np.ndarray, N_south: np.ndarray,
                A_north: np.ndarray, A_south: np.ndarray,
                output_path: str = '/home/user/documents/figure4_psr_coldtraps.png'):
    """
    Create Figure 4: PSR and cold trap size distributions.

    Top panel: Cumulative area of cold traps
    Bottom panel: Number of individual PSRs and cold traps
    """
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 10))

    # ========================================================================
    # TOP PANEL: Cumulative cold trap area
    # ========================================================================

    # Calculate cumulative areas (from small to large features)
    # For each L, show total area of all cold traps with size <= L
    A_north_cumulative = np.cumsum(A_north)
    A_south_cumulative = np.cumsum(A_south)

    ax_top.loglog(L_bins, A_north_cumulative, 'b-', linewidth=2.5,
                  label='Northern Hemisphere', marker='o', markersize=4, markevery=5)
    ax_top.loglog(L_bins, A_south_cumulative, 'r-', linewidth=2.5,
                  label='Southern Hemisphere', marker='s', markersize=4, markevery=5)

    # Add lateral conduction limit
    ax_top.axvline(x=0.01, color='gray', linestyle='--', alpha=0.7,
                   linewidth=1.5, label='Lateral conduction limit')

    ax_top.set_xlabel('Length Scale L [m]', fontsize=13, fontweight='bold')
    ax_top.set_ylabel('Cumulative Cold Trap Area [km²]', fontsize=13, fontweight='bold')
    ax_top.set_title('Cumulative area of cold traps (<110 K) at all latitudes',
                     fontsize=12, fontweight='bold', pad=10)
    ax_top.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax_top.grid(True, alpha=0.3, which='both', linestyle=':')
    ax_top.set_xlim([L_bins[0], L_bins[-1]])

    # Format ticks
    ax_top.tick_params(labelsize=11)

    # Add total area annotations
    total_north = A_north_cumulative[0]
    total_south = A_south_cumulative[0]
    ax_top.text(0.05, 0.25, f'Total North: {total_north:.1f} km²',
                transform=ax_top.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax_top.text(0.05, 0.15, f'Total South: {total_south:.1f} km²',
                transform=ax_top.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    # ========================================================================
    # BOTTOM PANEL: Number of individual PSRs and cold traps
    # ========================================================================

    ax_bottom.loglog(L_bins, N_north, 'b-', linewidth=2.5,
                     label='Northern Hemisphere', marker='o', markersize=4, markevery=5)
    ax_bottom.loglog(L_bins, N_south, 'r-', linewidth=2.5,
                     label='Southern Hemisphere', marker='s', markersize=4, markevery=5)

    # Add lateral conduction limit
    ax_bottom.axvline(x=0.01, color='gray', linestyle='--', alpha=0.7,
                      linewidth=1.5, label='Lateral conduction limit')

    ax_bottom.set_xlabel('Length Scale L [m]', fontsize=13, fontweight='bold')
    ax_bottom.set_ylabel('Number of PSRs/Cold Traps', fontsize=13, fontweight='bold')
    ax_bottom.set_title('Modeled number of individual PSRs and cold traps on the Moon',
                        fontsize=12, fontweight='bold', pad=10)
    ax_bottom.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax_bottom.grid(True, alpha=0.3, which='both', linestyle=':')
    ax_bottom.set_xlim([L_bins[0], L_bins[-1]])

    # Format ticks
    ax_bottom.tick_params(labelsize=11)

    # Add power-law slope annotation
    ax_bottom.text(0.05, 0.95, 'Size-frequency distribution: N(>L) ∝ L⁻²',
                   transform=ax_bottom.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # ========================================================================
    # Overall figure formatting
    # ========================================================================

    plt.suptitle('Figure 4: Permanently Shadowed and Cold-Trapping Areas\n' +
                 'as a Function of Size in Northern and Southern Hemispheres',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def print_summary_statistics(L_bins: np.ndarray, N_north: np.ndarray,
                            N_south: np.ndarray, A_north: np.ndarray,
                            A_south: np.ndarray):
    """Print summary statistics for the analysis."""
    print("\n" + "=" * 80)
    print("FIGURE 4 SUMMARY STATISTICS")
    print("=" * 80)

    # Cumulative totals
    total_N_north = np.sum(N_north)
    total_N_south = np.sum(N_south)
    total_A_north = np.sum(A_north)
    total_A_south = np.sum(A_south)

    print(f"\nTOTAL COUNTS:")
    print(f"  Northern Hemisphere: {total_N_north:.2e} features")
    print(f"  Southern Hemisphere: {total_N_south:.2e} features")
    print(f"  Total (both):        {total_N_north + total_N_south:.2e} features")

    print(f"\nTOTAL COLD TRAP AREA:")
    print(f"  Northern Hemisphere: {total_A_north:.1f} km²")
    print(f"  Southern Hemisphere: {total_A_south:.1f} km²")
    print(f"  Total (both):        {total_A_north + total_A_south:.1f} km²")
    print(f"  Fraction of lunar surface: {(total_A_north + total_A_south)/LUNAR_SURFACE_AREA*100:.4f}%")

    print(f"\nHEMISPHERIC ASYMMETRY:")
    print(f"  South/North ratio (count): {total_N_south/total_N_north:.2f}")
    print(f"  South/North ratio (area):  {total_A_south/total_A_north:.2f}")

    # Length scale ranges
    print(f"\nLENGTH SCALE RANGE:")
    print(f"  Minimum: {L_bins[0]:.4f} m ({L_bins[0]*100:.2f} cm)")
    print(f"  Maximum: {L_bins[-1]:.0f} m ({L_bins[-1]/1000:.1f} km)")
    print(f"  Number of bins: {len(L_bins)}")

    # Dominant contributors
    idx_max_A_north = np.argmax(A_north)
    idx_max_A_south = np.argmax(A_south)

    print(f"\nDOMINANT SIZE SCALES (by area contribution):")
    print(f"  Northern: L = {L_bins[idx_max_A_north]:.1f} m " +
          f"({A_north[idx_max_A_north]/total_A_north*100:.1f}% of total)")
    print(f"  Southern: L = {L_bins[idx_max_A_south]:.1f} m " +
          f"({A_south[idx_max_A_south]/total_A_south*100:.1f}% of total)")

    print("\n" + "=" * 80)


def main():
    """Main execution: Generate Figure 4."""
    print("\n" + "=" * 80)
    print("GENERATING FIGURE 4: PSR AND COLD TRAP SIZE DISTRIBUTIONS")
    print("=" * 80)

    print("\nGenerating crater size-frequency distribution...")
    size_dist = generate_crater_size_distribution(L_min=0.01, L_max=100000, n_bins=50)

    L_bins = size_dist['L_bins']
    N_north = size_dist['N_north']
    N_south = size_dist['N_south']

    print(f"✓ Generated {len(L_bins)} logarithmically-spaced bins")
    print(f"  Length scale range: {L_bins[0]:.4f} m to {L_bins[-1]:.0f} m")

    print("\nCalculating cold trap areas for each bin...")
    A_north, A_south = calculate_cold_trap_areas(L_bins, N_north, N_south)
    print("✓ Cold trap areas calculated")

    print("\nCreating Figure 4...")
    plot_figure4(L_bins, N_north, N_south, A_north, A_south)

    print_summary_statistics(L_bins, N_north, N_south, A_north, A_south)

    print("\n" + "=" * 80)
    print("FIGURE 4 GENERATION COMPLETE")
    print("=" * 80)
    print("\nFigure shows:")
    print("  • Top panel: Cumulative cold trap area vs length scale")
    print("  • Bottom panel: Number of PSRs/cold traps vs length scale")
    print("  • Both hemispheres shown separately")
    print("  • Length-scale bins are logarithmically spaced")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

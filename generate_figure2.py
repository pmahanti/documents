#!/usr/bin/env python3
"""
Generate Figure 2: PSR Temperature Distribution and Cold Trap Identification

This figure shows:
- Panel A: Histogram of PSR maximum temperatures from Diviner data
- Panel B: Cold trap area vs latitude for both hemispheres
- Panel C: Cold trap fraction vs PSR size
- Panel D: Cumulative distribution of cold trap and PSR areas

Demonstrates how cold traps (T < 110K) are identified from the PSR database.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd

# Constants
COLD_TRAP_THRESHOLD = 110.0  # K
PSR_CSV = '/home/user/documents/psr_with_temperatures.csv'


def load_psr_data():
    """Load PSR data with Diviner temperatures."""
    print("\n[Loading PSR data]")
    psr = pd.read_csv(PSR_CSV)

    # Calculate diameter and cold trap metrics
    psr['diameter_m'] = 2 * np.sqrt(psr['area_km2'] * 1e6 / np.pi)
    psr['coldtrap_fraction'] = 0.0
    mask = psr['pixel_count'] > 0
    psr.loc[mask, 'coldtrap_fraction'] = psr.loc[mask, 'pixels_lt_110K'] / psr.loc[mask, 'pixel_count']
    psr['coldtrap_area_km2'] = psr['coldtrap_fraction'] * psr['area_km2']
    psr['is_coldtrap'] = psr['coldtrap_area_km2'] > 0

    # Use absolute latitude
    psr['abs_latitude'] = psr['latitude'].abs()

    print(f"✓ Loaded {len(psr)} PSRs")
    print(f"  Cold traps (T < 110K): {psr['is_coldtrap'].sum()}")
    print(f"  Total PSR area: {psr['area_km2'].sum():.2f} km²")
    print(f"  Total cold trap area: {psr['coldtrap_area_km2'].sum():.2f} km²")

    return psr


def create_figure2(psr_data, output_path='/home/user/documents/figure2_psr_temperatures.png'):
    """
    Create Figure 2 with 4 panels showing PSR temperature distribution and cold trap identification.
    """
    fig = plt.figure(figsize=(14, 10))

    # Separate by hemisphere
    north = psr_data[psr_data['hemisphere'] == 'North']
    south = psr_data[psr_data['hemisphere'] == 'South']

    # Panel A: Temperature distribution histogram
    ax1 = plt.subplot(2, 2, 1)

    # Get temperature data (use temp_max_K as representative)
    temps_north = north['temp_max_K'].values
    temps_south = south['temp_max_K'].values
    temps_all = psr_data['temp_max_K'].values

    # Create histogram
    bins = np.linspace(40, 400, 50)
    ax1.hist(temps_all, bins=bins, alpha=0.5, label='All PSRs', color='gray', edgecolor='black')
    ax1.hist(temps_north, bins=bins, alpha=0.6, label='North', color='blue', edgecolor='black')
    ax1.hist(temps_south, bins=bins, alpha=0.6, label='South', color='red', edgecolor='black')

    # Mark cold trap threshold
    ax1.axvline(x=COLD_TRAP_THRESHOLD, color='green', linestyle='--', linewidth=2.5,
                label=f'Cold trap threshold ({COLD_TRAP_THRESHOLD}K)')

    ax1.set_xlabel('Maximum Temperature [K]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of PSRs', fontsize=12, fontweight='bold')
    ax1.set_title('A. PSR Temperature Distribution', fontsize=13, fontweight='bold', loc='left')
    ax1.legend(fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xlim([40, 400])

    # Add statistics annotation
    n_cold = (temps_all < COLD_TRAP_THRESHOLD).sum()
    pct_cold = n_cold / len(temps_all) * 100
    ax1.text(0.98, 0.97, f'Cold traps: {n_cold} ({pct_cold:.1f}%)\nPSRs: {len(temps_all)}',
             transform=ax1.transAxes, fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel B: Cold trap area vs latitude
    ax2 = plt.subplot(2, 2, 2)

    # Bin by latitude
    lat_bins = np.arange(70, 91, 2)
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

    area_north_bins = []
    area_south_bins = []

    for i in range(len(lat_bins) - 1):
        lat_min, lat_max = lat_bins[i], lat_bins[i+1]

        north_bin = north[(north['abs_latitude'] >= lat_min) & (north['abs_latitude'] < lat_max)]
        south_bin = south[(south['abs_latitude'] >= lat_min) & (south['abs_latitude'] < lat_max)]

        area_north_bins.append(north_bin['coldtrap_area_km2'].sum())
        area_south_bins.append(south_bin['coldtrap_area_km2'].sum())

    ax2.plot(lat_centers, area_north_bins, 'o-', linewidth=2, markersize=8,
             color='blue', label='North', markeredgecolor='black')
    ax2.plot(lat_centers, area_south_bins, 's-', linewidth=2, markersize=8,
             color='red', label='South', markeredgecolor='black')

    ax2.set_xlabel('Latitude [°]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cold Trap Area [km²]', fontsize=12, fontweight='bold')
    ax2.set_title('B. Cold Trap Area vs Latitude', fontsize=13, fontweight='bold', loc='left')
    ax2.legend(fontsize=10, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_xlim([70, 90])

    # Add annotation
    total_north = sum(area_north_bins)
    total_south = sum(area_south_bins)
    ax2.text(0.02, 0.97, f'Total North: {total_north:.0f} km²\nTotal South: {total_south:.0f} km²',
             transform=ax2.transAxes, fontsize=10, ha='left', va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Panel C: Cold trap fraction vs PSR size
    ax3 = plt.subplot(2, 2, 3)

    # Bin by size
    size_bins = np.logspace(np.log10(100), np.log10(20000), 15)  # 100m to 20km
    size_centers = np.sqrt(size_bins[:-1] * size_bins[1:])

    frac_north_bins = []
    frac_south_bins = []

    for i in range(len(size_bins) - 1):
        d_min, d_max = size_bins[i], size_bins[i+1]

        north_bin = north[(north['diameter_m'] >= d_min) & (north['diameter_m'] < d_max)]
        south_bin = south[(south['diameter_m'] >= d_min) & (south['diameter_m'] < d_max)]

        # Calculate fraction with cold traps
        if len(north_bin) > 0:
            frac_north_bins.append((north_bin['is_coldtrap'].sum() / len(north_bin)) * 100)
        else:
            frac_north_bins.append(0)

        if len(south_bin) > 0:
            frac_south_bins.append((south_bin['is_coldtrap'].sum() / len(south_bin)) * 100)
        else:
            frac_south_bins.append(0)

    ax3.semilogx(size_centers, frac_north_bins, 'o-', linewidth=2, markersize=8,
                 color='blue', label='North', markeredgecolor='black')
    ax3.semilogx(size_centers, frac_south_bins, 's-', linewidth=2, markersize=8,
                 color='red', label='South', markeredgecolor='black')

    ax3.set_xlabel('PSR Diameter [m]', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cold Trap Fraction [%]', fontsize=12, fontweight='bold')
    ax3.set_title('C. Cold Trap Fraction vs PSR Size', fontsize=13, fontweight='bold', loc='left')
    ax3.legend(fontsize=10, framealpha=0.95, loc='upper left')
    ax3.grid(True, alpha=0.3, linestyle=':', which='both')
    ax3.set_xlim([100, 20000])
    ax3.set_ylim([0, 50])

    # Panel D: Cumulative area distributions
    ax4 = plt.subplot(2, 2, 4)

    # Sort PSRs by size
    north_sorted = north.sort_values('diameter_m')
    south_sorted = south.sort_values('diameter_m')

    # Cumulative areas
    psr_area_north_cum = north_sorted['area_km2'].cumsum().values
    psr_area_south_cum = south_sorted['area_km2'].cumsum().values
    cold_area_north_cum = north_sorted['coldtrap_area_km2'].cumsum().values
    cold_area_south_cum = south_sorted['coldtrap_area_km2'].cumsum().values

    # Plot
    ax4.loglog(north_sorted['diameter_m'], psr_area_north_cum, '-',
               linewidth=2, color='blue', alpha=0.5, label='North PSR area')
    ax4.loglog(south_sorted['diameter_m'], psr_area_south_cum, '-',
               linewidth=2, color='red', alpha=0.5, label='South PSR area')
    ax4.loglog(north_sorted['diameter_m'], cold_area_north_cum, '-',
               linewidth=2.5, color='blue', label='North cold trap area')
    ax4.loglog(south_sorted['diameter_m'], cold_area_south_cum, '-',
               linewidth=2.5, color='red', label='South cold trap area')

    ax4.set_xlabel('PSR Diameter [m]', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cumulative Area [km²]', fontsize=12, fontweight='bold')
    ax4.set_title('D. Cumulative Area Distributions', fontsize=13, fontweight='bold', loc='left')
    ax4.legend(fontsize=9, framealpha=0.95, loc='upper left')
    ax4.grid(True, alpha=0.3, linestyle=':', which='both')

    # Overall title
    plt.suptitle('Figure 2: PSR Temperature Distribution and Cold Trap Identification\n' +
                 'Analysis of Diviner Temperature Data for Lunar Polar PSRs',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def print_summary(psr_data):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("FIGURE 2 SUMMARY: PSR TEMPERATURE DISTRIBUTION")
    print("=" * 80)

    cold_traps = psr_data[psr_data['is_coldtrap']]
    north = psr_data[psr_data['hemisphere'] == 'North']
    south = psr_data[psr_data['hemisphere'] == 'South']
    cold_north = cold_traps[cold_traps['hemisphere'] == 'North']
    cold_south = cold_traps[cold_traps['hemisphere'] == 'South']

    print(f"\nAll PSRs:")
    print(f"  Total: {len(psr_data)}")
    print(f"  Total area: {psr_data['area_km2'].sum():.2f} km²")
    print(f"  Temperature range: {psr_data['temp_max_K'].min():.1f} - {psr_data['temp_max_K'].max():.1f} K")

    print(f"\nCold Traps (T < {COLD_TRAP_THRESHOLD}K):")
    print(f"  Total: {len(cold_traps)} ({len(cold_traps)/len(psr_data)*100:.1f}% of PSRs)")
    print(f"  Total area: {cold_traps['coldtrap_area_km2'].sum():.2f} km²")
    print(f"  North: {len(cold_north)} PSRs, {cold_north['coldtrap_area_km2'].sum():.2f} km²")
    print(f"  South: {len(cold_south)} PSRs, {cold_south['coldtrap_area_km2'].sum():.2f} km²")

    print(f"\nDiameter ranges:")
    print(f"  PSRs: {psr_data['diameter_m'].min():.1f} - {psr_data['diameter_m'].max():.1f} m")
    print(f"  Cold traps: {cold_traps['diameter_m'].min():.1f} - {cold_traps['diameter_m'].max():.1f} m")

    print(f"\nLatitude ranges:")
    print(f"  PSRs: {psr_data['abs_latitude'].min():.1f}° - {psr_data['abs_latitude'].max():.1f}°")
    print(f"  Cold traps: {cold_traps['abs_latitude'].min():.1f}° - {cold_traps['abs_latitude'].max():.1f}°")

    print("\n" + "=" * 80)


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("GENERATING FIGURE 2: PSR TEMPERATURE DISTRIBUTION")
    print("=" * 80)

    # Load data
    psr_data = load_psr_data()

    # Create figure
    create_figure2(psr_data)

    # Print summary
    print_summary(psr_data)

    print("\n✓ COMPLETE: Figure 2")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

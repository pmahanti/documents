#!/usr/bin/env python3
"""
Integrated Lunar Communication Analysis

Combines surface-to-surface LTE coverage with:
- Direct-to-Earth (DTE) communication
- Surface asset tracking
- Multi-link analysis and scheduling
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os

from lunar_lte_simulator import LunarLTESimulator, TransmitterConfig
from lunar_comm_spice import LunarCommSPICE, SurfaceAsset, DSNStation


class IntegratedCommAnalysis:
    """Integrated analysis of lunar communication systems."""

    def __init__(self, dem_path: str, tx_config: TransmitterConfig,
                 kernel_dir: str = 'kernels'):
        """
        Initialize integrated analysis.

        Args:
            dem_path: Path to DEM GeoTIFF
            tx_config: Transmitter configuration
            kernel_dir: SPICE kernel directory
        """
        self.tx_config = tx_config
        self.dem_path = dem_path

        # Initialize LTE simulator for surface coverage
        print("Initializing LTE simulator...")
        self.lte_sim = LunarLTESimulator(dem_path, tx_config)

        # Initialize SPICE for DTE and asset tracking
        print("Initializing SPICE analysis...")
        self.spice_comm = LunarCommSPICE(kernel_dir)

        # Storage for results
        self.surface_assets = []
        self.asset_links = []
        self.dte_results = None
        self.visibility_data = None

    def add_surface_asset(self, asset: SurfaceAsset):
        """Add a surface asset to track."""
        self.surface_assets.append(asset)

    def analyze_surface_coverage(self, verbose: bool = True):
        """Analyze surface-to-surface LTE coverage."""
        print("\n" + "="*70)
        print("SURFACE COVERAGE ANALYSIS")
        print("="*70)

        self.lte_sim.run_analysis(verbose=verbose)

    def analyze_asset_links(self):
        """Analyze communication links to all surface assets."""
        print("\n" + "="*70)
        print("SURFACE ASSET LINK ANALYSIS")
        print("="*70)

        self.asset_links = []

        for asset in self.surface_assets:
            link = self.spice_comm.check_asset_link(
                tx_lat=self.tx_config.lat,
                tx_lon=self.tx_config.lon,
                tx_alt=self.tx_config.height_above_ground,
                asset=asset,
                tx_power_dbm=self.tx_config.transmit_power_dbm,
                tx_gain_dbi=self.tx_config.antenna_gain_dbi,
                frequency_mhz=self.tx_config.frequency_mhz
            )
            self.asset_links.append(link)

            print(f"\n{asset.name}:")
            print(f"  Location: ({asset.lat:.4f}°, {asset.lon:.4f}°)")
            print(f"  Distance: {link['distance_km']:.2f} km")
            print(f"  Path Loss: {link['fspl_db']:.1f} dB")
            print(f"  RX Power: {link['rx_power_dbm']:.1f} dBm")
            print(f"  Link Margin: {link['link_margin_db']:.1f} dB")
            print(f"  Status: {'✓ AVAILABLE' if link['link_available'] else '✗ UNAVAILABLE'}")

    def analyze_dte_windows(self, start_time: str, duration_hours: float = 720,
                           time_step_minutes: float = 30):
        """
        Analyze Direct-to-Earth visibility windows.

        Args:
            start_time: Start time (ISO format)
            duration_hours: Analysis duration (default 30 days)
            time_step_minutes: Time step (default 30 min)
        """
        print("\n" + "="*70)
        print("DIRECT-TO-EARTH VISIBILITY ANALYSIS")
        print("="*70)

        self.visibility_data = self.spice_comm.find_earth_visibility_windows(
            tx_lat=self.tx_config.lat,
            tx_lon=self.tx_config.lon,
            tx_alt=self.tx_config.height_above_ground,
            start_time=start_time,
            duration_hours=duration_hours,
            time_step_minutes=time_step_minutes
        )

        windows = self.visibility_data['windows']
        total_duration = sum([w['duration_hours'] for w in windows])
        coverage_percent = 100 * total_duration / duration_hours

        print(f"\nAnalysis Period: {duration_hours:.0f} hours ({duration_hours/24:.1f} days)")
        print(f"Visibility Windows: {len(windows)}")
        print(f"Total Visible Time: {total_duration:.1f} hours ({total_duration/24:.1f} days)")
        print(f"Visibility Coverage: {coverage_percent:.1f}%")

        if windows:
            print(f"\nLongest Window: {max(w['duration_hours'] for w in windows):.2f} hours")
            print(f"Shortest Window: {min(w['duration_hours'] for w in windows):.2f} hours")
            print(f"Average Window: {np.mean([w['duration_hours'] for w in windows]):.2f} hours")

            print("\nFirst 5 Visibility Windows:")
            for i, window in enumerate(windows[:5], 1):
                print(f"  {i}. Duration: {window['duration_hours']:.2f} hrs, "
                      f"Max Elevation: {window['max_elevation_deg']:.1f}°")

    def analyze_dte_link_budget(self, et_time: Optional[float] = None,
                                dsn_stations: Optional[List[str]] = None):
        """
        Analyze DTE link budget to DSN stations.

        Args:
            et_time: Ephemeris time (use middle of first visibility window if None)
            dsn_stations: List of station names (use all 70m if None)
        """
        print("\n" + "="*70)
        print("DIRECT-TO-EARTH LINK BUDGET")
        print("="*70)

        if et_time is None:
            if self.visibility_data and self.visibility_data['windows']:
                # Use middle of first visibility window
                first_window = self.visibility_data['windows'][0]
                et_time = (first_window['start_et'] + first_window['end_et']) / 2
            else:
                # Use current time approximation
                import spiceypy as spice
                try:
                    et_time = spice.str2et("2025-11-22T12:00:00")
                except:
                    et_time = 0.0

        if dsn_stations is None:
            # Use all 70m dishes
            dsn_stations = ['Goldstone', 'Canberra', 'Madrid']

        self.dte_results = {}

        # Use X-band for DTE (typical for deep space)
        dte_frequency = 8450.0  # MHz
        dte_power = 50.0  # dBm (100W)
        dte_gain = 30.0  # dBi (high-gain directional to Earth)

        print(f"\nTransmitter Configuration:")
        print(f"  Frequency: {dte_frequency} MHz (X-band)")
        print(f"  TX Power: {dte_power} dBm ({10**((dte_power-30)/10):.1f} W)")
        print(f"  TX Gain: {dte_gain} dBi (Earth-pointing)")

        for station_name in dsn_stations:
            station = self.spice_comm.DSN_STATIONS[station_name]
            link = self.spice_comm.calculate_dte_link_budget(
                tx_lat=self.tx_config.lat,
                tx_lon=self.tx_config.lon,
                tx_alt=self.tx_config.height_above_ground,
                tx_power_dbm=dte_power,
                tx_gain_dbi=dte_gain,
                frequency_mhz=dte_frequency,
                et=et_time,
                dsn_station=station
            )

            self.dte_results[station_name] = link

            print(f"\n{station.name} ({station.location}):")
            print(f"  Distance: {link['distance_km']:.0f} km")
            print(f"  Free Space Path Loss: {link['fspl_db']:.1f} dB")
            print(f"  RX Antenna Gain: {link['rx_gain_dbi']:.1f} dBi")
            print(f"  RX Power: {link['rx_power_dbm']:.1f} dBm")
            print(f"  RX Sensitivity: {link['rx_sensitivity_dbm']:.1f} dBm")
            print(f"  Link Margin: {link['link_margin_db']:.1f} dB")
            print(f"  TX Visible from Earth: {link['tx_visible']}")
            print(f"  Status: {'✓ LINK AVAILABLE' if link['link_available'] else '✗ LINK UNAVAILABLE'}")

    def plot_integrated_coverage(self, save_path: Optional[str] = None):
        """
        Create integrated coverage visualization.

        Args:
            save_path: Optional path to save figure
        """
        if self.lte_sim.received_power_dbm is None:
            print("Run surface coverage analysis first!")
            return

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Surface Coverage Map (large)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        rx_power_plot = np.copy(self.lte_sim.received_power_dbm)
        rx_power_plot[self.lte_sim.distances_km > self.tx_config.max_range_km] = np.nan
        rx_power_plot[np.isinf(rx_power_plot)] = np.nan

        vmin = max(self.tx_config.receiver_sensitivity_dbm - 20, np.nanmin(rx_power_plot))
        vmax = np.nanmax(rx_power_plot)

        im1 = ax1.contourf(self.lte_sim.X/1000, self.lte_sim.Y/1000, rx_power_plot,
                          levels=20, cmap='RdYlGn', vmin=vmin, vmax=vmax, alpha=0.8)

        # Plot transmitter
        ax1.plot(self.lte_sim.dem_x[self.lte_sim.tx_x_idx]/1000,
                self.lte_sim.dem_y[self.lte_sim.tx_y_idx]/1000,
                'r*', markersize=25, label='Base Station', markeredgecolor='black', linewidth=1.5)

        # Plot surface assets
        if self.surface_assets:
            asset_x = [self.spice_comm.geodetic_to_moon_fixed(a.lat, a.lon, a.altitude)[0]
                      for a in self.surface_assets]
            asset_y = [self.spice_comm.geodetic_to_moon_fixed(a.lat, a.lon, a.altitude)[1]
                      for a in self.surface_assets]

            for i, asset in enumerate(self.surface_assets):
                # Simple projection for plotting (not accurate for large areas)
                # Just use lat/lon as approximate x/y for visualization
                marker = 'o' if 'Rover' in asset.name else 's'
                color = 'green' if self.asset_links[i]['link_available'] else 'red'
                ax1.plot(asset_x[i], asset_y[i], marker, markersize=12,
                        color=color, label=asset.name, markeredgecolor='black', linewidth=1)

        ax1.set_xlabel('X (km)')
        ax1.set_ylabel('Y (km)')
        ax1.set_title(f'Surface Coverage Map\n{self.tx_config.frequency_mhz} MHz LTE')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)
        plt.colorbar(im1, ax=ax1, label='RX Power (dBm)')

        # Plot 2: Asset Link Summary
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        ax2.set_title('Surface Asset Links', fontweight='bold', fontsize=10)

        if self.asset_links:
            summary_text = []
            for link in self.asset_links:
                status = '✓' if link['link_available'] else '✗'
                summary_text.append(
                    f"{status} {link['asset_name']}\n"
                    f"  {link['distance_km']:.1f} km\n"
                    f"  {link['link_margin_db']:+.1f} dB\n"
                )

            ax2.text(0.05, 0.95, '\n'.join(summary_text),
                    transform=ax2.transAxes,
                    verticalalignment='top',
                    fontfamily='monospace',
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax2.text(0.5, 0.5, 'No assets defined',
                    transform=ax2.transAxes,
                    ha='center', va='center')

        # Plot 3: DTE Link Budget
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.axis('off')
        ax3.set_title('Direct-to-Earth Links', fontweight='bold', fontsize=10)

        if self.dte_results:
            dte_text = []
            for station_name, link in self.dte_results.items():
                status = '✓' if link['link_available'] else '✗'
                dte_text.append(
                    f"{status} {station_name}\n"
                    f"  {link['distance_km']:.0f} km\n"
                    f"  {link['link_margin_db']:+.1f} dB\n"
                )

            ax3.text(0.05, 0.95, '\n'.join(dte_text),
                    transform=ax3.transAxes,
                    verticalalignment='top',
                    fontfamily='monospace',
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        else:
            ax3.text(0.5, 0.5, 'Run DTE analysis',
                    transform=ax3.transAxes,
                    ha='center', va='center')

        # Plot 4: Earth Visibility
        if self.visibility_data:
            ax4 = fig.add_subplot(gs[2, :])

            et_array = self.visibility_data['et_array']
            visibility = self.visibility_data['visibility']

            # Convert to datetime
            try:
                import spiceypy as spice
                times = [spice.et2datetime(et) for et in et_array]
            except:
                times = [datetime(2025, 1, 1) + timedelta(seconds=float(et - et_array[0]))
                        for et in et_array]

            ax4.fill_between(times, 0, visibility, alpha=0.6, color='blue', label='Earth Visible')
            ax4.set_ylabel('Visibility')
            ax4.set_xlabel('Time')
            ax4.set_ylim(-0.1, 1.1)
            ax4.set_title('Earth Visibility Timeline')
            ax4.grid(True, alpha=0.3)
            ax4.legend()

            from matplotlib.dates import DateFormatter
            ax4.xaxis.set_major_formatter(DateFormatter('%m-%d'))
        else:
            ax4 = fig.add_subplot(gs[2, :])
            ax4.text(0.5, 0.5, 'Run Earth visibility analysis',
                    transform=ax4.transAxes,
                    ha='center', va='center', fontsize=12)
            ax4.axis('off')

        # Overall title
        fig.suptitle(f'Integrated Lunar Communication Analysis\n'
                    f'Base Station: ({self.tx_config.lat:.2f}°, {self.tx_config.lon:.2f}°)',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to {save_path}")

        plt.show()

    def generate_report(self, output_path: str = "comm_analysis_report.txt"):
        """Generate a text report of all analyses."""

        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("INTEGRATED LUNAR COMMUNICATION ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Transmitter Info
            f.write("BASE STATION CONFIGURATION\n")
            f.write("-"*70 + "\n")
            f.write(f"Location: ({self.tx_config.lat:.4f}°, {self.tx_config.lon:.4f}°)\n")
            f.write(f"Antenna Height: {self.tx_config.height_above_ground} m\n")
            f.write(f"Frequency: {self.tx_config.frequency_mhz} MHz\n")
            f.write(f"TX Power: {self.tx_config.transmit_power_dbm} dBm\n")
            f.write(f"TX Gain: {self.tx_config.antenna_gain_dbi} dBi\n")
            f.write(f"Analysis Range: {self.tx_config.max_range_km} km\n\n")

            # Surface Coverage
            if self.lte_sim.coverage_mask is not None:
                f.write("SURFACE COVERAGE SUMMARY\n")
                f.write("-"*70 + "\n")
                coverage_area = np.sum(self.lte_sim.coverage_mask) * (self.tx_config.resolution_m / 1000)**2
                f.write(f"Coverage Area: {coverage_area:.2f} km²\n")
                total_area = np.pi * self.tx_config.max_range_km**2
                coverage_pct = 100 * coverage_area / total_area
                f.write(f"Coverage Percentage: {coverage_pct:.1f}%\n\n")

            # Asset Links
            if self.asset_links:
                f.write("SURFACE ASSET LINKS\n")
                f.write("-"*70 + "\n")
                for link in self.asset_links:
                    f.write(f"\n{link['asset_name']}:\n")
                    f.write(f"  Distance: {link['distance_km']:.2f} km\n")
                    f.write(f"  RX Power: {link['rx_power_dbm']:.1f} dBm\n")
                    f.write(f"  Link Margin: {link['link_margin_db']:.1f} dB\n")
                    f.write(f"  Status: {'AVAILABLE' if link['link_available'] else 'UNAVAILABLE'}\n")
                f.write("\n")

            # DTE Links
            if self.dte_results:
                f.write("DIRECT-TO-EARTH LINKS\n")
                f.write("-"*70 + "\n")
                for station_name, link in self.dte_results.items():
                    f.write(f"\n{station_name}:\n")
                    f.write(f"  Distance: {link['distance_km']:.0f} km\n")
                    f.write(f"  Path Loss: {link['fspl_db']:.1f} dB\n")
                    f.write(f"  RX Power: {link['rx_power_dbm']:.1f} dBm\n")
                    f.write(f"  Link Margin: {link['link_margin_db']:.1f} dB\n")
                    f.write(f"  Status: {'AVAILABLE' if link['link_available'] else 'UNAVAILABLE'}\n")
                f.write("\n")

            # Earth Visibility
            if self.visibility_data:
                f.write("EARTH VISIBILITY WINDOWS\n")
                f.write("-"*70 + "\n")
                windows = self.visibility_data['windows']
                f.write(f"Total Windows: {len(windows)}\n")
                if windows:
                    total_duration = sum([w['duration_hours'] for w in windows])
                    f.write(f"Total Visible Time: {total_duration:.1f} hours\n")
                    f.write(f"Average Window Duration: {np.mean([w['duration_hours'] for w in windows]):.2f} hours\n")
                    f.write(f"\nFirst 10 Windows:\n")
                    for i, window in enumerate(windows[:10], 1):
                        f.write(f"  {i}. Duration: {window['duration_hours']:.2f} hrs, "
                               f"Max Elev: {window['max_elevation_deg']:.1f}°\n")

            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")

        print(f"\nReport saved to {output_path}")


def main():
    """Example integrated analysis."""

    print("\n" + "="*70)
    print("INTEGRATED LUNAR COMMUNICATION ANALYSIS")
    print("="*70 + "\n")

    # Configuration
    config = TransmitterConfig(
        lat=-89.5,  # Lunar south pole
        lon=0.0,
        height_above_ground=10.0,
        frequency_mhz=2600.0,  # S-band for surface comms
        transmit_power_dbm=46.0,  # 40W
        antenna_gain_dbi=18.0,
        max_range_km=20.0,
        resolution_m=200.0,  # Coarser for speed
        receiver_sensitivity_dbm=-110.0
    )

    # Select DEM
    dem_files = [f for f in os.listdir("SDC60_COG") if f.endswith(".tif")]
    if not dem_files:
        print("ERROR: No DEM files found!")
        return

    dem_path = os.path.join("SDC60_COG", dem_files[0])

    # Create analyzer
    analyzer = IntegratedCommAnalysis(dem_path, config, kernel_dir='kernels')

    # Add surface assets
    analyzer.add_surface_asset(SurfaceAsset(
        name="Rover-Alpha",
        lat=-89.3,
        lon=10.0,
        altitude=2.0
    ))
    analyzer.add_surface_asset(SurfaceAsset(
        name="Lander-Base",
        lat=-89.7,
        lon=-5.0,
        altitude=5.0
    ))
    analyzer.add_surface_asset(SurfaceAsset(
        name="Rover-Beta",
        lat=-88.8,
        lon=25.0,
        altitude=2.0
    ))

    # Run analyses
    analyzer.analyze_surface_coverage(verbose=True)
    analyzer.analyze_asset_links()
    analyzer.analyze_dte_windows(
        start_time="2025-11-22T00:00:00",
        duration_hours=240,  # 10 days
        time_step_minutes=30
    )
    analyzer.analyze_dte_link_budget()

    # Generate outputs
    analyzer.plot_integrated_coverage(save_path="integrated_coverage.png")
    analyzer.generate_report("integrated_analysis_report.txt")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()

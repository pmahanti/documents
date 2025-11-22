#!/usr/bin/env python3
"""
Example SPICE-based Lunar Communication Analysis

Demonstrates:
1. Earth visibility from lunar surface
2. Direct-to-Earth link budget to DSN stations
3. Surface asset communication tracking
4. Integrated multi-link analysis
"""

from lunar_comm_spice import LunarCommSPICE, SurfaceAsset
from integrated_comm_analysis import IntegratedCommAnalysis
from lunar_lte_simulator import TransmitterConfig
import os


def example_earth_visibility():
    """Example 1: Analyze Earth visibility windows."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Earth Visibility Analysis")
    print("="*70)

    comm = LunarCommSPICE(kernel_dir='kernels')

    # Transmitter at lunar south pole
    tx_lat = -89.5
    tx_lon = 0.0
    tx_alt = 10.0

    print(f"\nTransmitter Location: ({tx_lat}°, {tx_lon}°)")
    print(f"Analyzing 30-day period...\n")

    # Find visibility windows
    vis_data = comm.find_earth_visibility_windows(
        tx_lat, tx_lon, tx_alt,
        start_time="2025-11-22T00:00:00",
        duration_hours=720,  # 30 days
        time_step_minutes=30
    )

    windows = vis_data['windows']
    total_duration = sum([w['duration_hours'] for w in windows])

    print(f"Results:")
    print(f"  Total Windows: {len(windows)}")
    print(f"  Total Visibility: {total_duration:.1f} hours ({total_duration/24:.1f} days)")
    print(f"  Coverage: {100*total_duration/720:.1f}%")

    if windows:
        print(f"\n  Longest Window: {max(w['duration_hours'] for w in windows):.2f} hours")
        print(f"  Shortest Window: {min(w['duration_hours'] for w in windows):.2f} hours")

        print(f"\n  First 5 Windows:")
        for i, w in enumerate(windows[:5], 1):
            print(f"    {i}. Duration: {w['duration_hours']:.2f} hrs, "
                  f"Max Elev: {w['max_elevation_deg']:.1f}°, "
                  f"Mean Elev: {w['mean_elevation_deg']:.1f}°")

    # Plot visibility
    comm.plot_earth_visibility(vis_data, save_path="earth_visibility_example.png")

    return vis_data


def example_dte_link_budget(vis_data=None):
    """Example 2: Direct-to-Earth link budget."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Direct-to-Earth Link Budget")
    print("="*70)

    comm = LunarCommSPICE(kernel_dir='kernels')

    tx_lat = -89.5
    tx_lon = 0.0
    tx_alt = 10.0

    # Select a time during visibility
    if vis_data and vis_data['windows']:
        window = vis_data['windows'][0]
        et = (window['start_et'] + window['end_et']) / 2
        print(f"\nAnalyzing during visibility window (duration: {window['duration_hours']:.1f} hrs)")
    else:
        import spiceypy as spice
        et = spice.str2et("2025-11-22T12:00:00")
        print("\nAnalyzing at nominal time")

    # X-band DTE configuration
    tx_power_dbm = 50.0  # 100W
    tx_gain_dbi = 30.0   # High-gain dish pointed at Earth
    frequency_mhz = 8450.0  # X-band

    print(f"\nTransmitter Configuration:")
    print(f"  Location: ({tx_lat}°, {tx_lon}°), {tx_alt}m altitude")
    print(f"  Power: {tx_power_dbm} dBm (100W)")
    print(f"  Gain: {tx_gain_dbi} dBi (Earth-pointing)")
    print(f"  Frequency: {frequency_mhz} MHz (X-band)")

    print(f"\nDSN Link Budget Analysis:")
    print("-" * 70)

    # Analyze each major DSN station
    for station_name in ['Goldstone', 'Canberra', 'Madrid']:
        station = comm.DSN_STATIONS[station_name]

        link = comm.calculate_dte_link_budget(
            tx_lat, tx_lon, tx_alt,
            tx_power_dbm, tx_gain_dbi, frequency_mhz,
            et, station
        )

        print(f"\n{station.name}:")
        print(f"  Location: {station.location}")
        print(f"  Distance: {link['distance_km']:,.0f} km")
        print(f"  Free-Space Path Loss: {link['fspl_db']:.1f} dB")
        print(f"  TX Gain: {link['tx_gain_dbi']:.1f} dBi")
        print(f"  RX Gain: {link['rx_gain_dbi']:.1f} dBi")
        print(f"  RX Power: {link['rx_power_dbm']:.1f} dBm")
        print(f"  RX Sensitivity: {link['rx_sensitivity_dbm']:.1f} dBm")
        print(f"  Link Margin: {link['link_margin_db']:+.1f} dB")
        print(f"  Earth Visible from TX: {link['tx_visible']}")
        print(f"  Link Status: {'✓ AVAILABLE' if link['link_available'] else '✗ UNAVAILABLE'}")


def example_surface_assets():
    """Example 3: Surface asset communication."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Surface Asset Communication")
    print("="*70)

    comm = LunarCommSPICE(kernel_dir='kernels')

    # Base station
    tx_lat = -89.5
    tx_lon = 0.0
    tx_alt = 10.0

    print(f"\nBase Station: ({tx_lat}°, {tx_lon}°)")

    # Define surface assets
    assets = [
        SurfaceAsset(
            name="Rover-Alpha",
            lat=-89.3,
            lon=10.0,
            altitude=2.0,
            receiver_sensitivity_dbm=-115.0,
            antenna_gain_dbi=8.0
        ),
        SurfaceAsset(
            name="Lander-1",
            lat=-89.7,
            lon=-15.0,
            altitude=5.0,
            receiver_sensitivity_dbm=-120.0,
            antenna_gain_dbi=12.0
        ),
        SurfaceAsset(
            name="Rover-Beta",
            lat=-88.5,
            lon=45.0,
            altitude=2.0,
            receiver_sensitivity_dbm=-115.0,
            antenna_gain_dbi=8.0
        ),
        SurfaceAsset(
            name="Relay-Station",
            lat=-89.0,
            lon=90.0,
            altitude=10.0,
            receiver_sensitivity_dbm=-120.0,
            antenna_gain_dbi=15.0
        ),
    ]

    # S-band for surface communications
    tx_power_dbm = 40.0  # 10W
    tx_gain_dbi = 12.0   # Omnidirectional or sector antenna
    frequency_mhz = 2400.0  # S-band

    print(f"\nBase Station TX: {tx_power_dbm} dBm, {tx_gain_dbi} dBi, {frequency_mhz} MHz")
    print("\nAsset Link Analysis:")
    print("-" * 70)

    for asset in assets:
        link = comm.check_asset_link(
            tx_lat, tx_lon, tx_alt,
            asset,
            tx_power_dbm, tx_gain_dbi, frequency_mhz
        )

        print(f"\n{asset.name}:")
        print(f"  Location: ({asset.lat:.2f}°, {asset.lon:.2f}°)")
        print(f"  Distance: {link['distance_km']:.2f} km")
        print(f"  Path Loss: {link['fspl_db']:.1f} dB")
        print(f"  RX Power: {link['rx_power_dbm']:.1f} dBm")
        print(f"  RX Sensitivity: {asset.receiver_sensitivity_dbm:.1f} dBm")
        print(f"  Link Margin: {link['link_margin_db']:+.1f} dB")
        print(f"  Geometric LOS: {link['geometric_los']}")
        print(f"  Status: {'✓ LINK OK' if link['link_available'] else '✗ NO LINK'}")


def example_integrated_analysis():
    """Example 4: Full integrated analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Integrated Communication Analysis")
    print("="*70)

    # Check for DEM files
    if not os.path.exists("SDC60_COG"):
        print("\nERROR: SDC60_COG directory not found!")
        print("Integrated analysis requires DEM files.")
        return

    dem_files = [f for f in os.listdir("SDC60_COG") if f.endswith(".tif")]
    if not dem_files:
        print("\nERROR: No DEM files found in SDC60_COG/")
        return

    dem_path = os.path.join("SDC60_COG", dem_files[0])

    # Configure transmitter
    config = TransmitterConfig(
        lat=-89.5,
        lon=0.0,
        height_above_ground=10.0,
        frequency_mhz=2600.0,
        transmit_power_dbm=46.0,
        antenna_gain_dbi=18.0,
        max_range_km=15.0,
        resolution_m=150.0,
        receiver_sensitivity_dbm=-110.0
    )

    print(f"\nUsing DEM: {dem_path}")
    print(f"Transmitter: ({config.lat}°, {config.lon}°)")

    # Create integrated analyzer
    analyzer = IntegratedCommAnalysis(dem_path, config, kernel_dir='kernels')

    # Add surface assets
    print("\nAdding surface assets...")
    analyzer.add_surface_asset(SurfaceAsset(
        name="Rover-1", lat=-89.3, lon=10.0, altitude=2.0
    ))
    analyzer.add_surface_asset(SurfaceAsset(
        name="Lander-A", lat=-89.7, lon=-5.0, altitude=5.0
    ))
    analyzer.add_surface_asset(SurfaceAsset(
        name="Rover-2", lat=-88.8, lon=25.0, altitude=2.0
    ))

    # Run complete analysis
    print("\nRunning integrated analysis...")
    analyzer.analyze_surface_coverage(verbose=True)
    analyzer.analyze_asset_links()
    analyzer.analyze_dte_windows(
        start_time="2025-11-22T00:00:00",
        duration_hours=240,  # 10 days
        time_step_minutes=30
    )
    analyzer.analyze_dte_link_budget()

    # Generate outputs
    print("\nGenerating visualizations and reports...")
    analyzer.plot_integrated_coverage(save_path="integrated_example.png")
    analyzer.generate_report("integrated_example_report.txt")

    print("\n✓ Analysis complete!")
    print("  - Coverage map: integrated_example.png")
    print("  - Report: integrated_example_report.txt")


def print_menu():
    """Print example selection menu."""
    print("\n" + "="*70)
    print("LUNAR COMMUNICATION SPICE ANALYSIS - EXAMPLES")
    print("="*70)
    print("\nSelect an example:")
    print("  1 - Earth visibility windows")
    print("  2 - Direct-to-Earth link budget (DSN)")
    print("  3 - Surface asset communication")
    print("  4 - Full integrated analysis (requires DEM)")
    print("  5 - Run all examples")
    print("  0 - Exit")
    print("="*70)


def main():
    """Main function."""

    # Check for kernels directory
    if not os.path.exists('kernels'):
        print("\nWARNING: 'kernels/' directory not found!")
        print("SPICE analysis will use approximations.")
        print("For full functionality, add SPICE kernels to the 'kernels/' directory.")
        input("\nPress Enter to continue anyway...")

    vis_data = None

    while True:
        print_menu()
        choice = input("\nEnter your choice (0-5): ").strip()

        if choice == '0':
            print("\nExiting...")
            break
        elif choice == '1':
            vis_data = example_earth_visibility()
        elif choice == '2':
            example_dte_link_budget(vis_data)
        elif choice == '3':
            example_surface_assets()
        elif choice == '4':
            example_integrated_analysis()
        elif choice == '5':
            print("\nRunning all examples...")
            vis_data = example_earth_visibility()
            example_dte_link_budget(vis_data)
            example_surface_assets()
            example_integrated_analysis()
            print("\n" + "="*70)
            print("All examples completed!")
            print("="*70)
        else:
            print("\nInvalid choice. Please try again.")

        if choice in ['1', '2', '3', '4', '5']:
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()

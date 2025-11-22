#!/usr/bin/env python3
"""
Test Case: Rover Path DTE Coverage Analysis

Tests DTE communication coverage for a realistic multi-day rover traverse
mission on the lunar surface.

Mission Scenario:
- VIPER-like ice prospecting mission
- Start: Shackleton Crater rim
- Mission: Explore PSRs and collect ice samples
- Duration: 5 days (120 hours)
- Path: Multiple waypoints covering ~30 km traverse
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rover_path_dte_coverage import RoverPathDTEAnalyzer
import matplotlib.pyplot as plt
import numpy as np


def test_viper_mission_traverse():
    """
    Test Case: VIPER-style Ice Prospecting Mission

    Mission Profile:
    - 5-day traverse mission
    - Multiple science stops in PSRs
    - Return to landing site
    - DTE communication for telemetry and science data
    """

    print("\n" + "="*80)
    print("TEST CASE: VIPER ICE PROSPECTING MISSION")
    print("Rover Path DTE Coverage Analysis")
    print("="*80)

    # Mission waypoints (lat, lon in degrees)
    # Realistic traverse around Shackleton Crater south pole region
    waypoints = [
        # Start at landing site
        (-89.50, 0.00),      # Landing Site (Shackleton rim)

        # Phase 1: Initial exploration
        (-89.52, 3.00),      # Science Stop 1: Illuminated area
        (-89.55, 6.00),      # Science Stop 2: Crater wall

        # Phase 2: PSR investigation
        (-89.60, 8.00),      # Science Stop 3: PSR interior (ice prospect)
        (-89.62, 10.00),     # Science Stop 4: Deep PSR (core sample)
        (-89.65, 12.00),     # Science Stop 5: PSR boundary

        # Phase 3: Extended traverse
        (-89.63, 15.00),     # Science Stop 6: Ridge traverse
        (-89.58, 18.00),     # Science Stop 7: Boulder field
        (-89.55, 20.00),     # Science Stop 8: Crater ejecta

        # Phase 4: Northern excursion
        (-89.50, 22.00),     # Science Stop 9: Illuminated plateau
        (-89.45, 20.00),     # Science Stop 10: Slope analysis

        # Phase 5: Return traverse
        (-89.48, 15.00),     # Waypoint: Return path
        (-89.50, 10.00),     # Waypoint: Shortcut route
        (-89.52, 5.00),      # Waypoint: Near landing site
        (-89.50, 0.00),      # Return to landing site
    ]

    print("\nMission Waypoints (15 total):")
    print(f"{'#':>3} {'Latitude':>10} {'Longitude':>11} {'Description':<30}")
    print("-" * 80)

    descriptions = [
        "Landing Site (Shackleton rim)",
        "Science Stop 1: Illuminated area",
        "Science Stop 2: Crater wall",
        "Science Stop 3: PSR interior",
        "Science Stop 4: Deep PSR sample",
        "Science Stop 5: PSR boundary",
        "Science Stop 6: Ridge traverse",
        "Science Stop 7: Boulder field",
        "Science Stop 8: Crater ejecta",
        "Science Stop 9: Illuminated plateau",
        "Science Stop 10: Slope analysis",
        "Return Waypoint 1",
        "Return Waypoint 2",
        "Return Waypoint 3",
        "Landing Site (return)"
    ]

    for i, ((lat, lon), desc) in enumerate(zip(waypoints, descriptions)):
        print(f"{i:3d} {lat:10.4f}° {lon:10.4f}° {desc:<30}")

    # Mission parameters
    mission_start = "2026-03-15T06:00:00"  # March 15, 2026, 06:00 UTC
    mission_duration_hours = 120  # 5 days
    rover_speed_kmh = 0.8  # Conservative speed for science mission

    # VIPER-class rover DTE configuration
    rover_config = {
        'antenna_height': 2.5,    # meters (mast-mounted HGA)
        'tx_power_dbm': 43.0,     # 20W SSPA
        'tx_gain_dbi': 25.0,      # High-gain steerable dish
        'frequency_mhz': 8450.0,  # X-band downlink
    }

    print(f"\nMission Parameters:")
    print(f"  Start Time: {mission_start}")
    print(f"  Duration: {mission_duration_hours} hours ({mission_duration_hours/24} days)")
    print(f"  Rover Speed: {rover_speed_kmh} km/h (science pace)")

    print(f"\nRover DTE Configuration:")
    print(f"  Antenna Height: {rover_config['antenna_height']} m")
    print(f"  TX Power: {rover_config['tx_power_dbm']} dBm "
          f"({10**((rover_config['tx_power_dbm']-30)/10):.1f} W)")
    print(f"  TX Gain: {rover_config['tx_gain_dbi']} dBi (steerable HGA)")
    print(f"  Frequency: {rover_config['frequency_mhz']} MHz (X-band)")
    print(f"  Data Products: Telemetry, images, spectrometer data")

    # Create analyzer
    print("\n" + "="*80)
    analyzer = RoverPathDTEAnalyzer(kernel_dir='kernels')

    # Run analysis
    coverage_df = analyzer.analyze_coverage(
        waypoints=waypoints,
        start_time=mission_start,
        duration_hours=mission_duration_hours,
        rover_speed_kmh=rover_speed_kmh,
        rover_antenna_height=rover_config['antenna_height'],
        tx_power_dbm=rover_config['tx_power_dbm'],
        tx_gain_dbi=rover_config['tx_gain_dbi'],
        frequency_mhz=rover_config['frequency_mhz'],
        time_step_minutes=1.0,  # Minute-by-minute resolution
        interpolate_path=True,
        path_samples=100
    )

    # Print summary
    analyzer.print_summary()

    # Save CSV
    output_csv = "viper_mission_dte_coverage.csv"
    analyzer.save_csv(output_csv)

    # Generate visualization
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATION")
    print(f"{'='*80}\n")

    plot_coverage_timeline(coverage_df, "viper_mission_coverage_timeline.png")
    plot_coverage_map(coverage_df, waypoints, "viper_mission_coverage_map.png")

    # Detailed analysis
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS")
    print(f"{'='*80}\n")

    print_coverage_windows(coverage_df)
    print_station_handovers(coverage_df)

    # Sample data display
    print(f"\n{'='*80}")
    print("SAMPLE CSV DATA (First 20 Records)")
    print(f"{'='*80}\n")

    # Select key columns for display
    display_cols = [
        'timestamp', 'rover_lat', 'rover_lon', 'distance_traveled_km',
        'earth_visible', 'earth_elevation_deg', 'any_dsn_available',
        'best_station', 'best_margin_db'
    ]
    print(coverage_df[display_cols].head(20).to_string(index=False))

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")
    print(f"\nGenerated Files:")
    print(f"  1. {output_csv}")
    print(f"     - Minute-by-minute DTE coverage data")
    print(f"     - {len(coverage_df):,} records × {len(coverage_df.columns)} columns")
    print(f"  2. viper_mission_coverage_timeline.png")
    print(f"     - Coverage timeline visualization")
    print(f"  3. viper_mission_coverage_map.png")
    print(f"     - Rover path with coverage indicators")
    print()


def plot_coverage_timeline(df, save_path):
    """Plot coverage timeline."""

    import pandas as pd

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'])

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    # Plot 1: Earth visibility
    ax = axes[0]
    ax.fill_between(df['datetime'], 0, df['earth_visible'], alpha=0.6, color='blue')
    ax.set_ylabel('Earth\nVisible')
    ax.set_ylim(-0.1, 1.1)
    ax.set_title('DTE Coverage Timeline for VIPER Mission', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: DSN station availability
    ax = axes[1]
    ax.fill_between(df['datetime'], 0, df['goldstone_available'],
                    alpha=0.6, color='gold', label='Goldstone')
    ax.fill_between(df['datetime'], 0, df['canberra_available'],
                    alpha=0.6, color='green', label='Canberra')
    ax.fill_between(df['datetime'], 0, df['madrid_available'],
                    alpha=0.6, color='red', label='Madrid')
    ax.set_ylabel('DSN\nStations')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3)

    # Plot 3: Number of available stations
    ax = axes[2]
    ax.plot(df['datetime'], df['num_stations_available'], 'b-', linewidth=1)
    ax.fill_between(df['datetime'], 0, df['num_stations_available'], alpha=0.3)
    ax.set_ylabel('# Stations\nAvailable')
    ax.set_ylim(-0.1, 3.5)
    ax.grid(True, alpha=0.3)

    # Plot 4: Distance traveled
    ax = axes[3]
    ax.plot(df['datetime'], df['distance_traveled_km'], 'g-', linewidth=2)
    ax.set_ylabel('Distance\nTraveled (km)')
    ax.set_xlabel('Mission Time')
    ax.grid(True, alpha=0.3)

    # Format x-axis
    import matplotlib.dates as mdates
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Coverage timeline saved to: {save_path}")
    plt.close()


def plot_coverage_map(df, waypoints, save_path):
    """Plot rover path with coverage indicators."""

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot waypoints
    wp_lats = [w[0] for w in waypoints]
    wp_lons = [w[1] for w in waypoints]
    ax.plot(wp_lons, wp_lats, 'ko-', markersize=8, linewidth=2,
            label='Planned Path', alpha=0.5)

    # Mark start and end
    ax.plot(wp_lons[0], wp_lats[0], 'g*', markersize=20,
            label='Start/End', markeredgecolor='black', linewidth=1.5)

    # Color code path by DSN availability
    has_coverage = df['any_dsn_available'].values
    lats = df['rover_lat'].values
    lons = df['rover_lon'].values

    # Plot segments with coverage (green) and without (red)
    for i in range(len(df) - 1):
        color = 'green' if has_coverage[i] else 'red'
        alpha = 0.8 if has_coverage[i] else 0.3
        ax.plot([lons[i], lons[i+1]], [lats[i], lats[i+1]],
                color=color, alpha=alpha, linewidth=1)

    # Dummy plots for legend
    ax.plot([], [], 'g-', linewidth=3, label='DSN Available', alpha=0.8)
    ax.plot([], [], 'r-', linewidth=3, label='No DSN', alpha=0.5)

    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title('VIPER Mission Path with DTE Coverage', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Coverage map saved to: {save_path}")
    plt.close()


def print_coverage_windows(df):
    """Print continuous coverage windows."""

    print("Communication Windows (Continuous DSN Coverage):\n")

    # Find continuous coverage windows
    windows = []
    in_window = False
    window_start = None

    for i, row in df.iterrows():
        if row['any_dsn_available'] and not in_window:
            window_start = i
            in_window = True
        elif not row['any_dsn_available'] and in_window:
            windows.append((window_start, i - 1))
            in_window = False

    # Handle case where window extends to end
    if in_window:
        windows.append((window_start, len(df) - 1))

    if windows:
        print(f"Found {len(windows)} communication windows:\n")
        print(f"{'#':>3} {'Start Time':<20} {'End Time':<20} {'Duration':<12} {'Best Station':<15}")
        print("-" * 85)

        for i, (start_idx, end_idx) in enumerate(windows[:20], 1):  # Show first 20
            start_time = df.iloc[start_idx]['timestamp']
            end_time = df.iloc[end_idx]['timestamp']
            duration_min = end_idx - start_idx + 1

            # Find best station during window
            window_data = df.iloc[start_idx:end_idx+1]
            best_margins = {
                'Goldstone': window_data['goldstone_margin_db'].max(),
                'Canberra': window_data['canberra_margin_db'].max(),
                'Madrid': window_data['madrid_margin_db'].max()
            }
            best_station = max(best_margins.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -999)[0]

            print(f"{i:3d} {start_time:<20} {end_time:<20} {duration_min:5d} min   {best_station:<15}")

        if len(windows) > 20:
            print(f"\n... and {len(windows) - 20} more windows")
    else:
        print("No communication windows found!")

    print()


def print_station_handovers(df):
    """Print DSN station handover events."""

    print("DSN Station Handovers:\n")

    handovers = []
    prev_station = None

    for i, row in df.iterrows():
        curr_station = row['best_station']
        if curr_station != 'None' and curr_station != prev_station and prev_station is not None:
            handovers.append({
                'time': row['timestamp'],
                'from': prev_station,
                'to': curr_station,
                'position': f"({row['rover_lat']:.3f}°, {row['rover_lon']:.3f}°)"
            })
        prev_station = curr_station if curr_station != 'None' else prev_station

    if handovers:
        print(f"Found {len(handovers)} station handovers:\n")
        print(f"{'#':>3} {'Time':<20} {'From':<12} {'To':<12} {'Rover Position':<25}")
        print("-" * 75)

        for i, ho in enumerate(handovers[:15], 1):  # Show first 15
            print(f"{i:3d} {ho['time']:<20} {ho['from']:<12} {ho['to']:<12} {ho['position']:<25}")

        if len(handovers) > 15:
            print(f"\n... and {len(handovers) - 15} more handovers")
    else:
        print("No station handovers detected")

    print()


if __name__ == "__main__":
    test_viper_mission_traverse()

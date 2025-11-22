#!/usr/bin/env python3
"""
Test All ConOps Scenarios

Runs all 4 operational scenarios and generates outputs:
1. Surface TX → Surface RX
2. Surface TX → Earth RX (DTE)
3. Crater TX → Earth RX (DTE)
4. Rover Path → Earth RX (DTE)
"""

import sys
import os
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation_engine import (
    LunarCommSimulationEngine,
    SimulationConfig,
    create_example_config
)
from output_manager import OutputManager


def test_conops_1_surface_to_surface():
    """ConOps 1: Surface TX → Surface RX"""

    print("\n" + "="*80)
    print("ConOps 1: SURFACE TX → SURFACE RX")
    print("="*80)

    # Create configuration
    config = SimulationConfig(
        scenario='surface_to_surface',
        tx_lat=-89.5,
        tx_lon=0.0,
        tx_height_m=10.0,
        frequency_mhz=2600.0,
        tx_power_dbm=46.0,
        tx_gain_dbi=18.0,
        analysis_range_km=15.0,
        grid_resolution_m=150.0,  # Coarser for speed
        rx_sensitivity_dbm=-110.0,
        propagation_model='two_ray',
        include_multipath=True,
        include_diffraction=True,
        surface_roughness_m=0.1,
        surface_assets=[
            {
                'name': 'VIPER Rover',
                'lat': -89.35,
                'lon': 10.0,
                'altitude': 2.0,
                'rx_sensitivity': -115.0,
                'antenna_gain': 8.0
            },
            {
                'name': 'Artemis Lander',
                'lat': -89.65,
                'lon': -5.0,
                'altitude': 8.0,
                'rx_sensitivity': -120.0,
                'antenna_gain': 12.0
            },
            {
                'name': 'Science Platform',
                'lat': -89.4,
                'lon': 15.0,
                'altitude': 3.0,
                'rx_sensitivity': -110.0,
                'antenna_gain': 9.0
            }
        ],
        output_format=['png', 'csv'],
        output_dir='conops_outputs'
    )

    print(f"\nConfiguration:")
    print(f"  Location: ({config.tx_lat}°, {config.tx_lon}°)")
    print(f"  Frequency: {config.frequency_mhz} MHz (S-band LTE)")
    print(f"  TX Power: {config.tx_power_dbm} dBm ({10**((config.tx_power_dbm-30)/10):.1f} W)")
    print(f"  Propagation Model: {config.propagation_model}")
    print(f"  Multipath: {config.include_multipath}")
    print(f"  Surface Assets: {len(config.surface_assets)}")

    # Run simulation
    print(f"\nRunning simulation...")
    engine = LunarCommSimulationEngine(config)
    results = engine.run_simulation()

    print(f"Status: {engine.status}")

    # Display results
    stats = results.get('statistics', {})
    print(f"\nResults:")
    print(f"  Coverage: {stats.get('coverage_percentage', 0):.1f}%")
    print(f"  Max Range: {stats.get('max_covered_range_km', 0):.2f} km")

    asset_links = results.get('asset_links', [])
    print(f"\n  Asset Links:")
    for link in asset_links:
        status = '✓' if link['link_available'] else '✗'
        print(f"    {status} {link['name']}: {link['distance_km']:.2f} km, {link['link_margin_db']:+.1f} dB")

    # Generate outputs
    print(f"\nGenerating outputs...")
    output_mgr = OutputManager(output_dir='conops_outputs')
    files = output_mgr.generate_all_outputs(results, formats=['png', 'csv'])

    print(f"  Outputs saved:")
    for fmt, path in files.items():
        print(f"    {fmt}: {path}")

    return results


def test_conops_2_surface_to_earth():
    """ConOps 2: Surface TX → Earth RX (DTE)"""

    print("\n" + "="*80)
    print("ConOps 2: SURFACE TX → EARTH RX (DTE)")
    print("="*80)

    # Create configuration
    config = SimulationConfig(
        scenario='surface_to_earth',
        tx_lat=-89.5,
        tx_lon=0.0,
        tx_height_m=20.0,
        dte_frequency_mhz=8450.0,
        dte_tx_power_dbm=50.0,
        dte_tx_gain_dbi=30.0,
        dte_start_time="2026-03-15T00:00:00",
        dte_duration_hours=240.0,  # 10 days
        dte_time_step_minutes=30.0,
        propagation_model='free_space',
        output_format=['png', 'csv'],
        output_dir='conops_outputs'
    )

    print(f"\nConfiguration:")
    print(f"  Location: ({config.tx_lat}°, {config.tx_lon}°)")
    print(f"  Frequency: {config.dte_frequency_mhz} MHz (X-band)")
    print(f"  TX Power: {config.dte_tx_power_dbm} dBm ({10**((config.dte_tx_power_dbm-30)/10):.0f} W)")
    print(f"  TX Gain: {config.dte_tx_gain_dbi} dBi (Earth-pointing HGA)")
    print(f"  Duration: {config.dte_duration_hours} hours ({config.dte_duration_hours/24} days)")

    # Run simulation
    print(f"\nRunning simulation...")
    engine = LunarCommSimulationEngine(config)
    results = engine.run_simulation()

    print(f"Status: {engine.status}")

    # Display results
    visibility = results.get('visibility', {})
    windows = visibility.get('windows', [])

    print(f"\nResults:")
    print(f"  Visibility Windows: {len(windows)}")

    if windows:
        total_hours = sum([w['duration_hours'] for w in windows])
        print(f"  Total Visible Time: {total_hours:.1f} hours ({total_hours/24:.1f} days)")
        print(f"  Visibility Coverage: {100*total_hours/config.dte_duration_hours:.1f}%")

        print(f"\n  First 5 Windows:")
        for i, window in enumerate(windows[:5], 1):
            print(f"    {i}. Duration: {window['duration_hours']:.2f} hrs, Max Elev: {window['max_elevation_deg']:.2f}°")

    # DSN links
    dsn_links = results.get('dsn_links', {})
    print(f"\n  DSN Station Links:")
    for station, link in dsn_links.items():
        status = '✓' if link['link_available'] else '✗'
        print(f"    {status} {station}: {link['link_margin_db']:+.1f} dB")

    # Generate outputs
    print(f"\nGenerating outputs...")
    output_mgr = OutputManager(output_dir='conops_outputs')
    files = output_mgr.generate_all_outputs(results, formats=['png', 'csv'])

    print(f"  Outputs saved:")
    for fmt, path in files.items():
        print(f"    {fmt}: {path}")

    return results


def test_conops_3_crater_to_earth():
    """ConOps 3: Crater TX → Earth RX (DTE)"""

    print("\n" + "="*80)
    print("ConOps 3: CRATER TX → EARTH RX (DTE)")
    print("="*80)

    # Create configuration
    config = SimulationConfig(
        scenario='crater_to_earth',
        tx_lat=-89.5,
        tx_lon=0.0,
        tx_height_m=10.0,
        crater_depth_m=150.0,
        crater_radius_m=600.0,
        tx_inside_crater=True,
        dte_frequency_mhz=8450.0,
        dte_tx_power_dbm=50.0,
        dte_tx_gain_dbi=30.0,
        dte_start_time="2026-03-15T00:00:00",
        dte_duration_hours=240.0,
        dte_time_step_minutes=30.0,
        propagation_model='free_space',
        output_format=['png', 'csv'],
        output_dir='conops_outputs'
    )

    print(f"\nConfiguration:")
    print(f"  Crater: {config.crater_radius_m}m radius, {config.crater_depth_m}m depth")
    print(f"  TX Inside Crater: {config.tx_inside_crater}")
    print(f"  TX Height Above Floor: {config.tx_height_m}m")
    print(f"  Frequency: {config.dte_frequency_mhz} MHz (X-band)")
    print(f"  TX Power: {config.dte_tx_power_dbm} dBm")

    # Run simulation
    print(f"\nRunning simulation...")
    engine = LunarCommSimulationEngine(config)
    results = engine.run_simulation()

    print(f"Status: {engine.status}")

    # Display results
    crater_effects = results.get('crater_effects', {})
    print(f"\nCrater Effects:")
    print(f"  Additional Diffraction Loss: {crater_effects.get('additional_diffraction_loss_db', 0):.2f} dB")

    # DSN links with crater adjustment
    dsn_links = results.get('dsn_links', {})
    print(f"\n  DSN Station Links (with crater effects):")
    for station, link in dsn_links.items():
        normal_margin = link['link_margin_db']
        crater_margin = link.get('crater_adjusted_margin_db', normal_margin)
        status = '✓' if link.get('crater_link_available', False) else '✗'
        print(f"    {status} {station}: {normal_margin:+.1f} dB → {crater_margin:+.1f} dB (adjusted)")

    # Generate outputs
    print(f"\nGenerating outputs...")
    output_mgr = OutputManager(output_dir='conops_outputs')
    files = output_mgr.generate_all_outputs(results, formats=['png', 'csv'])

    print(f"  Outputs saved:")
    for fmt, path in files.items():
        print(f"    {fmt}: {path}")

    return results


def test_conops_4_rover_path_dte():
    """ConOps 4: Rover Path → Earth RX (DTE)"""

    print("\n" + "="*80)
    print("ConOps 4: ROVER PATH → EARTH RX (DTE)")
    print("="*80)

    # Create configuration
    config = SimulationConfig(
        scenario='rover_path_dte',
        rover_waypoints=[
            (-89.50, 0.00),   # Landing site
            (-89.52, 5.00),   # WP1
            (-89.55, 8.00),   # WP2
            (-89.58, 10.00),  # WP3
            (-89.55, 12.00),  # WP4
            (-89.52, 8.00),   # WP5
            (-89.50, 0.00),   # Return
        ],
        rover_speed_kmh=1.2,
        rover_mission_hours=48.0,  # 2 days for faster test
        tx_height_m=2.5,
        dte_frequency_mhz=8450.0,
        dte_tx_power_dbm=43.0,
        dte_tx_gain_dbi=25.0,
        dte_start_time="2026-03-20T06:00:00",
        dte_time_step_minutes=5.0,  # 5-minute resolution for faster test
        propagation_model='free_space',
        output_format=['png', 'csv'],
        output_dir='conops_outputs'
    )

    print(f"\nConfiguration:")
    print(f"  Waypoints: {len(config.rover_waypoints)}")
    print(f"  Speed: {config.rover_speed_kmh} km/h")
    print(f"  Duration: {config.rover_mission_hours} hours ({config.rover_mission_hours/24} days)")
    print(f"  Frequency: {config.dte_frequency_mhz} MHz (X-band)")
    print(f"  TX Power: {config.dte_tx_power_dbm} dBm ({10**((config.dte_tx_power_dbm-30)/10):.1f} W)")
    print(f"  Resolution: {config.dte_time_step_minutes} minutes")

    # Run simulation
    print(f"\nRunning simulation (may take 1-2 minutes)...")
    engine = LunarCommSimulationEngine(config)
    results = engine.run_simulation()

    print(f"Status: {engine.status}")

    # Display results
    summary = results.get('summary', {})

    print(f"\nResults:")
    print(f"  Mission Duration: {summary.get('total_minutes', 0)} minutes")
    print(f"  Earth Visible: {summary.get('earth_visible_percent', 0):.1f}%")
    print(f"  DSN Available: {summary.get('any_dsn_percent', 0):.1f}%")

    print(f"\n  Station Availability:")
    print(f"    Goldstone: {summary.get('goldstone_percent', 0):.1f}%")
    print(f"    Canberra: {summary.get('canberra_percent', 0):.1f}%")
    print(f"    Madrid: {summary.get('madrid_percent', 0):.1f}%")

    if not np.isnan(summary.get('best_margin_overall_db', np.nan)):
        print(f"\n  Best Link Margin: {summary.get('best_margin_overall_db', 0):.1f} dB")
        print(f"  Mean Link Margin: {summary.get('mean_margin_db', 0):.1f} dB")

    # Generate outputs
    print(f"\nGenerating outputs...")
    output_mgr = OutputManager(output_dir='conops_outputs')
    files = output_mgr.generate_all_outputs(results, formats=['png', 'csv'])

    print(f"  Outputs saved:")
    for fmt, path in files.items():
        print(f"    {fmt}: {path}")

    # Show sample of CSV data
    coverage_records = results.get('coverage_data', [])
    if coverage_records:
        import pandas as pd
        df = pd.DataFrame(coverage_records)

        print(f"\n  CSV Data Sample (first 5 rows):")
        sample_cols = ['timestamp', 'rover_lat', 'rover_lon', 'earth_visible',
                      'num_stations_available', 'best_station', 'best_margin_db']
        if all(col in df.columns for col in sample_cols):
            print(df[sample_cols].head().to_string(index=False))

    return results


def main():
    """Run all ConOps scenarios."""

    print("\n" + "="*80)
    print("LUNAR COMMUNICATION SIMULATOR - ALL CONOPS TEST")
    print("="*80)
    print(f"\nTest Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directory
    os.makedirs('conops_outputs', exist_ok=True)

    # Test all scenarios
    results_all = {}

    try:
        print("\n" + "█"*80)
        results_all['conops_1'] = test_conops_1_surface_to_surface()
    except Exception as e:
        print(f"\n❌ ConOps 1 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\n" + "█"*80)
        results_all['conops_2'] = test_conops_2_surface_to_earth()
    except Exception as e:
        print(f"\n❌ ConOps 2 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\n" + "█"*80)
        results_all['conops_3'] = test_conops_3_crater_to_earth()
    except Exception as e:
        print(f"\n❌ ConOps 3 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\n" + "█"*80)
        results_all['conops_4'] = test_conops_4_rover_path_dte()
    except Exception as e:
        print(f"\n❌ ConOps 4 failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    successful = len(results_all)
    print(f"\nCompleted {successful} out of 4 scenarios")

    if successful > 0:
        print(f"\nOutputs saved to: conops_outputs/")

        # List all generated files
        if os.path.exists('conops_outputs'):
            files = os.listdir('conops_outputs')
            if files:
                print(f"\nGenerated {len(files)} output files:")
                for f in sorted(files):
                    size = os.path.getsize(os.path.join('conops_outputs', f))
                    size_kb = size / 1024
                    print(f"  {f:50s} {size_kb:8.1f} KB")

    print(f"\nTest Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

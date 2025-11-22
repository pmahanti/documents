#!/usr/bin/env python3
"""
Example usage of the Lunar LTE Simulator

This script demonstrates different scenarios for simulating 4G LTE
communication on the lunar surface.
"""

from lunar_lte_simulator import LunarLTESimulator, TransmitterConfig
import os


def example_1_basic_coverage():
    """Example 1: Basic coverage analysis with default parameters."""

    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Coverage Analysis")
    print("="*70)

    config = TransmitterConfig(
        lat=-89.5,  # Lunar south pole region
        lon=0.0,
        height_above_ground=10.0,
        frequency_mhz=2600.0,
        transmit_power_dbm=46.0,
        antenna_gain_dbi=18.0,
        max_range_km=15.0,
        resolution_m=100.0
    )

    # Select a COG file from the SDC60_COG directory
    dem_files = [f for f in os.listdir("SDC60_COG") if f.endswith(".tif")]
    if not dem_files:
        print("No DEM files found in SDC60_COG/")
        return

    dem_path = os.path.join("SDC60_COG", dem_files[0])
    print(f"\nUsing DEM: {dem_path}")

    simulator = LunarLTESimulator(dem_path, config)
    simulator.run_analysis(verbose=True)
    simulator.plot_results(save_path="example1_basic_coverage.png")
    simulator.export_results("example1_results")


def example_2_high_power_long_range():
    """Example 2: High-power transmitter for extended range."""

    print("\n" + "="*70)
    print("EXAMPLE 2: High-Power Long-Range Communication")
    print("="*70)

    config = TransmitterConfig(
        lat=-89.5,
        lon=0.0,
        height_above_ground=20.0,  # Taller mast
        frequency_mhz=1800.0,  # Lower frequency for better range
        transmit_power_dbm=50.0,  # 100W transmitter
        antenna_gain_dbi=21.0,  # Higher gain antenna
        receiver_sensitivity_dbm=-115.0,  # More sensitive receiver
        max_range_km=50.0,  # Extended range
        resolution_m=200.0,  # Coarser resolution for speed
        include_diffraction=True
    )

    dem_files = [f for f in os.listdir("SDC60_COG") if f.endswith(".tif")]
    if not dem_files:
        print("No DEM files found in SDC60_COG/")
        return

    dem_path = os.path.join("SDC60_COG", dem_files[0])
    print(f"\nUsing DEM: {dem_path}")

    simulator = LunarLTESimulator(dem_path, config)
    simulator.run_analysis(verbose=True)
    simulator.plot_results(save_path="example2_long_range.png")
    simulator.export_results("example2_results")


def example_3_low_power_local():
    """Example 3: Low-power local area network."""

    print("\n" + "="*70)
    print("EXAMPLE 3: Low-Power Local Area Network")
    print("="*70)

    config = TransmitterConfig(
        lat=-89.5,
        lon=0.0,
        height_above_ground=5.0,  # Low mast
        frequency_mhz=2600.0,
        transmit_power_dbm=30.0,  # 1W transmitter
        antenna_gain_dbi=10.0,  # Moderate gain
        receiver_sensitivity_dbm=-100.0,
        max_range_km=5.0,  # Short range
        resolution_m=50.0,  # Fine resolution
        include_diffraction=True
    )

    dem_files = [f for f in os.listdir("SDC60_COG") if f.endswith(".tif")]
    if not dem_files:
        print("No DEM files found in SDC60_COG/")
        return

    dem_path = os.path.join("SDC60_COG", dem_files[0])
    print(f"\nUsing DEM: {dem_path}")

    simulator = LunarLTESimulator(dem_path, config)
    simulator.run_analysis(verbose=True)
    simulator.plot_results(save_path="example3_local_network.png")
    simulator.export_results("example3_results")


def example_4_custom_parameters():
    """Example 4: Custom parameters - user defined."""

    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Parameters")
    print("="*70)
    print("\nThis example shows how to customize all parameters")
    print("for your specific use case.")

    # Custom configuration
    config = TransmitterConfig(
        # Location (modify these based on your DEM)
        lat=-89.0,
        lon=15.0,
        height_above_ground=15.0,

        # RF parameters
        frequency_mhz=3500.0,  # 5G n78 band (also used for 4G)
        transmit_power_dbm=43.0,  # 20W
        antenna_gain_dbi=15.0,
        antenna_tilt_deg=0.0,

        # Receiver parameters
        receiver_sensitivity_dbm=-105.0,
        receiver_gain_dbi=2.0,

        # Analysis parameters
        max_range_km=25.0,
        resolution_m=120.0,

        # Propagation parameters
        polarization='vertical',
        include_diffraction=True
    )

    # You can specify a different DEM file
    dem_files = [f for f in os.listdir("SDC60_COG") if f.endswith(".tif")]
    if not dem_files:
        print("No DEM files found in SDC60_COG/")
        return

    # Select a specific DEM or use the first one
    dem_path = os.path.join("SDC60_COG", dem_files[5])  # 6th file
    print(f"\nUsing DEM: {dem_path}")

    simulator = LunarLTESimulator(dem_path, config)
    simulator.run_analysis(verbose=True)
    simulator.plot_results(save_path="example4_custom.png")
    simulator.export_results("example4_results")


def print_menu():
    """Print example selection menu."""
    print("\n" + "="*70)
    print("LUNAR LTE SIMULATOR - EXAMPLE SCENARIOS")
    print("="*70)
    print("\nSelect an example to run:")
    print("  1 - Basic coverage analysis (15 km range, standard parameters)")
    print("  2 - High-power long-range (50 km range, 100W transmitter)")
    print("  3 - Low-power local network (5 km range, 1W transmitter)")
    print("  4 - Custom parameters (user-defined configuration)")
    print("  5 - Run all examples")
    print("  0 - Exit")
    print("="*70)


def main():
    """Main function to run examples."""

    # Check if SDC60_COG directory exists
    if not os.path.exists("SDC60_COG"):
        print("ERROR: SDC60_COG directory not found!")
        print("Please ensure the DEM files are in the SDC60_COG/ directory.")
        return

    while True:
        print_menu()
        choice = input("\nEnter your choice (0-5): ").strip()

        if choice == '0':
            print("\nExiting...")
            break
        elif choice == '1':
            example_1_basic_coverage()
        elif choice == '2':
            example_2_high_power_long_range()
        elif choice == '3':
            example_3_low_power_local()
        elif choice == '4':
            example_4_custom_parameters()
        elif choice == '5':
            print("\nRunning all examples...")
            example_1_basic_coverage()
            example_2_high_power_long_range()
            example_3_low_power_local()
            example_4_custom_parameters()
            print("\n" + "="*70)
            print("All examples completed!")
            print("="*70)
        else:
            print("\nInvalid choice. Please try again.")

        if choice in ['1', '2', '3', '4', '5']:
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()

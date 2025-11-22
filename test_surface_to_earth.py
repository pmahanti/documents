#!/usr/bin/env python3
"""
Test Case 2: Surface-to-Earth (Direct-to-Earth) Communication

Tests DTE communication from lunar surface base station to Deep Space Network
(DSN) ground stations, analyzing Earth visibility and link performance.

Test Configuration:
- Base Station: Shackleton Crater rim (-89.5°, 0.0°)
- Test Period: 2025-12-15 00:00:00 UTC to 2025-12-25 00:00:00 UTC (10 days)
- Frequency: 8450 MHz (X-band downlink)
- Power: 50 dBm (100W)
- DSN Stations: Goldstone, Canberra, Madrid (70m and 34m)
"""

import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lunar_comm_spice import LunarCommSPICE
import spiceypy as spice


class SurfaceToEarthTest:
    """Test Direct-to-Earth communication links."""

    def __init__(self):
        """Initialize test configuration."""

        # Test metadata
        self.test_name = "Surface-to-Earth (DTE) Communication Test"
        self.test_start = "2025-12-15T00:00:00"
        self.test_duration_hours = 240  # 10 days
        self.test_location = "Shackleton Crater, Lunar South Pole"

        # Transmitter Configuration (Lunar Surface)
        # Location: Shackleton Crater rim (Artemis landing site)
        self.transmitter = {
            'name': 'Artemis Base DTE Terminal',
            'lat': -89.5,
            'lon': 0.0,
            'altitude': 20.0,  # 20m mast for Earth visibility
            'frequency_mhz': 8450.0,  # X-band downlink
            'tx_power_dbm': 50.0,  # 100W SSPA
            'tx_gain_dbi': 30.0,  # 1.2m dish pointed at Earth
            'modulation': 'QPSK',
            'data_rate_mbps': 10.0
        }

        # DSN Stations to test
        self.dsn_stations_to_test = [
            'Goldstone',      # DSS-14 (70m)
            'Goldstone_34m',  # DSS-24 (34m)
            'Canberra',       # DSS-43 (70m)
            'Canberra_34m',   # DSS-34 (34m)
            'Madrid',         # DSS-63 (70m)
            'Madrid_34m'      # DSS-54 (34m)
        ]

        # Analysis parameters
        self.time_step_minutes = 30  # Sample every 30 minutes

        # Initialize SPICE
        print(f"\n{'='*80}")
        print(f"INITIALIZING TEST: {self.test_name}")
        print(f"{'='*80}")
        print(f"\nTest Period: {self.test_start} to", end=" ")
        start_dt = datetime.fromisoformat(self.test_start.replace('T', ' '))
        end_dt = start_dt + timedelta(hours=self.test_duration_hours)
        print(f"{end_dt.isoformat().replace('T', ' ')}")
        print(f"Duration: {self.test_duration_hours} hours ({self.test_duration_hours/24:.1f} days)")
        print(f"Location: {self.test_location}")
        print("\nInitializing SPICE kernels...")

        self.comm = LunarCommSPICE(kernel_dir='kernels')

    def analyze_earth_visibility(self):
        """Analyze Earth visibility windows."""

        print(f"\n{'='*80}")
        print("EARTH VISIBILITY ANALYSIS")
        print(f"{'='*80}")

        print(f"\nTransmitter Location: ({self.transmitter['lat']:.4f}°, {self.transmitter['lon']:.4f}°)")
        print(f"Antenna Height: {self.transmitter['altitude']} m")
        print(f"\nAnalyzing visibility over {self.test_duration_hours/24:.1f} days...")

        # Find visibility windows
        self.visibility_data = self.comm.find_earth_visibility_windows(
            tx_lat=self.transmitter['lat'],
            tx_lon=self.transmitter['lon'],
            tx_alt=self.transmitter['altitude'],
            start_time=self.test_start,
            duration_hours=self.test_duration_hours,
            time_step_minutes=self.time_step_minutes
        )

        windows = self.visibility_data['windows']
        total_visible_hours = sum([w['duration_hours'] for w in windows])
        visibility_percentage = 100 * total_visible_hours / self.test_duration_hours

        print(f"\nVisibility Results:")
        print(f"  Total Windows: {len(windows)}")
        print(f"  Total Visible Time: {total_visible_hours:.2f} hours ({total_visible_hours/24:.2f} days)")
        print(f"  Visibility Coverage: {visibility_percentage:.1f}%")

        if windows:
            durations = [w['duration_hours'] for w in windows]
            max_elevations = [w['max_elevation_deg'] for w in windows]

            print(f"\nWindow Statistics:")
            print(f"  Longest Window: {max(durations):.2f} hours")
            print(f"  Shortest Window: {min(durations):.2f} hours")
            print(f"  Average Window: {np.mean(durations):.2f} hours")
            print(f"  Max Elevation Achieved: {max(max_elevations):.2f}°")
            print(f"  Average Max Elevation: {np.mean(max_elevations):.2f}°")

            print(f"\nFirst 10 Visibility Windows:")
            for i, window in enumerate(windows[:10], 1):
                # Convert ET to datetime for display
                try:
                    start_dt = spice.et2datetime(window['start_et'])
                    end_dt = spice.et2datetime(window['end_et'])
                    start_str = start_dt.strftime('%Y-%m-%d %H:%M')
                    end_str = end_dt.strftime('%H:%M UTC')
                except:
                    start_str = f"ET {window['start_et']:.0f}"
                    end_str = f"ET {window['end_et']:.0f}"

                print(f"  {i:2d}. {start_str} to {end_str}")
                print(f"      Duration: {window['duration_hours']:6.2f} hrs | "
                      f"Max Elev: {window['max_elevation_deg']:5.2f}° | "
                      f"Mean Elev: {window['mean_elevation_deg']:5.2f}°")

    def analyze_dsn_links(self):
        """Analyze DTE links to all DSN stations."""

        print(f"\n{'='*80}")
        print("DSN LINK BUDGET ANALYSIS")
        print(f"{'='*80}")

        print(f"\nTransmitter Configuration:")
        print(f"  Frequency: {self.transmitter['frequency_mhz']} MHz (X-band)")
        print(f"  TX Power: {self.transmitter['tx_power_dbm']} dBm "
              f"({10**((self.transmitter['tx_power_dbm']-30)/10):.0f} W)")
        print(f"  TX Gain: {self.transmitter['tx_gain_dbi']} dBi (Earth-pointing dish)")
        print(f"  Modulation: {self.transmitter['modulation']}")
        print(f"  Data Rate: {self.transmitter['data_rate_mbps']} Mbps")

        # Use middle of first visibility window for analysis, or approximate time
        if self.visibility_data['windows']:
            first_window = self.visibility_data['windows'][0]
            analysis_et = (first_window['start_et'] + first_window['end_et']) / 2
            print(f"\nAnalyzing at mid-point of first visibility window")
        else:
            try:
                analysis_et = spice.str2et(self.test_start) + 12*3600  # 12 hours in
            except:
                analysis_et = 0.0
            print(f"\nAnalyzing at nominal time")

        print(f"\n{'='*80}")
        print("DSN Station Link Budgets:")
        print(f"{'='*80}\n")

        self.dsn_results = {}

        for station_name in self.dsn_stations_to_test:
            station = self.comm.DSN_STATIONS[station_name]

            link = self.comm.calculate_dte_link_budget(
                tx_lat=self.transmitter['lat'],
                tx_lon=self.transmitter['lon'],
                tx_alt=self.transmitter['altitude'],
                tx_power_dbm=self.transmitter['tx_power_dbm'],
                tx_gain_dbi=self.transmitter['tx_gain_dbi'],
                frequency_mhz=self.transmitter['frequency_mhz'],
                et=analysis_et,
                dsn_station=station
            )

            self.dsn_results[station_name] = link

            # Display results
            dish_size = "70m" if "70" in str(station.dish_diameter) else "34m"
            status_symbol = '✓' if link['link_available'] else '✗'

            print(f"{status_symbol} {station.name}")
            print(f"   Location: {station.location}")
            print(f"   Dish Size: {dish_size} ({station.dish_diameter}m)")
            print(f"   Distance: {link['distance_km']:,.0f} km")
            print(f"   Free-Space Path Loss: {link['fspl_db']:.2f} dB")
            print(f"   RX Antenna Gain: {link['rx_gain_dbi']:.1f} dBi")
            print(f"   RX Power: {link['rx_power_dbm']:.2f} dBm")
            print(f"   RX Sensitivity: {link['rx_sensitivity_dbm']:.1f} dBm")
            print(f"   Link Margin: {link['link_margin_db']:+.2f} dB")
            print(f"   Moon Visible from Earth: {link['tx_visible']}")

            if link['tx_visible']:
                print(f"   Elevation from TX: {link['tx_elevation_deg']:.2f}°")
                print(f"   Azimuth from TX: {link['tx_azimuth_deg']:.2f}°")

            status_text = "LINK AVAILABLE" if link['link_available'] else "LINK UNAVAILABLE"
            print(f"   Status: {status_text}")

            # Performance assessment
            if link['link_available']:
                if link['link_margin_db'] >= 20:
                    quality = "EXCELLENT"
                elif link['link_margin_db'] >= 15:
                    quality = "VERY GOOD"
                elif link['link_margin_db'] >= 10:
                    quality = "GOOD"
                elif link['link_margin_db'] >= 6:
                    quality = "FAIR"
                else:
                    quality = "MARGINAL"
                print(f"   Link Quality: {quality}")

                # Estimated data rate (simplified)
                # Assuming Shannon limit: C = B * log2(1 + SNR)
                # For estimation, use link margin as proxy for SNR
                snr_linear = 10**(link['link_margin_db']/10)
                # Assume 10 MHz bandwidth for X-band
                bandwidth_mhz = 10.0
                theoretical_mbps = bandwidth_mhz * np.log2(1 + snr_linear)
                print(f"   Theoretical Max Data Rate: {theoretical_mbps:.1f} Mbps")

            print()

    def analyze_dsn_coverage(self):
        """Analyze which DSN stations provide coverage over time."""

        print(f"\n{'='*80}")
        print("DSN COVERAGE TIMELINE ANALYSIS")
        print(f"{'='*80}\n")

        # Sample at key points during visibility windows
        if not self.visibility_data['windows']:
            print("No Earth visibility windows found - cannot analyze DSN coverage")
            return

        print(f"Sampling {len(self.visibility_data['windows'])} visibility windows...\n")

        # Track which stations are available during each window
        window_coverage = []

        for i, window in enumerate(self.visibility_data['windows'][:20], 1):  # First 20 windows
            # Sample at middle of window
            sample_et = (window['start_et'] + window['end_et']) / 2

            available_stations = []

            for station_name in self.dsn_stations_to_test:
                station = self.comm.DSN_STATIONS[station_name]

                link = self.comm.calculate_dte_link_budget(
                    tx_lat=self.transmitter['lat'],
                    tx_lon=self.transmitter['lon'],
                    tx_alt=self.transmitter['altitude'],
                    tx_power_dbm=self.transmitter['tx_power_dbm'],
                    tx_gain_dbi=self.transmitter['tx_gain_dbi'],
                    frequency_mhz=self.transmitter['frequency_mhz'],
                    et=sample_et,
                    dsn_station=station
                )

                if link['link_available']:
                    available_stations.append({
                        'name': station_name,
                        'margin': link['link_margin_db']
                    })

            # Convert time
            try:
                window_time = spice.et2datetime(sample_et).strftime('%Y-%m-%d %H:%M UTC')
            except:
                window_time = f"Window {i}"

            window_coverage.append({
                'time': window_time,
                'duration': window['duration_hours'],
                'elevation': window['max_elevation_deg'],
                'stations': available_stations
            })

            # Display
            station_names = [s['name'] for s in available_stations]
            if station_names:
                print(f"Window {i:2d} @ {window_time}")
                print(f"  Duration: {window['duration_hours']:.2f} hrs | Elev: {window['max_elevation_deg']:.2f}°")
                print(f"  Available Stations ({len(available_stations)}):")
                for s in available_stations:
                    print(f"    • {s['name']:20s} Margin: {s['margin']:+.1f} dB")
            else:
                print(f"Window {i:2d} @ {window_time}: No stations available")

            print()

        # Summary statistics
        self.window_coverage = window_coverage
        self._print_coverage_summary()

    def _print_coverage_summary(self):
        """Print summary of DSN coverage."""

        print(f"\n{'='*80}")
        print("DSN COVERAGE SUMMARY")
        print(f"{'='*80}\n")

        # Count station availability
        station_counts = {name: 0 for name in self.dsn_stations_to_test}

        for window in self.window_coverage:
            for station in window['stations']:
                station_counts[station['name']] += 1

        total_windows = len(self.window_coverage)

        print(f"Station Availability (out of {total_windows} windows sampled):\n")

        # Group by complex
        for complex_name in ['Goldstone', 'Canberra', 'Madrid']:
            print(f"{complex_name} Complex:")
            for station_name in [complex_name, f"{complex_name}_34m"]:
                if station_name in station_counts:
                    count = station_counts[station_name]
                    pct = 100 * count / total_windows
                    station_obj = self.comm.DSN_STATIONS[station_name]
                    dish_size = f"{station_obj.dish_diameter:.0f}m"
                    print(f"  {station_name:20s} ({dish_size}): {count:3d} / {total_windows} ({pct:5.1f}%)")
            print()

        # Best station
        best_station = max(station_counts.items(), key=lambda x: x[1])
        print(f"Most Available Station: {best_station[0]} ({100*best_station[1]/total_windows:.1f}% coverage)")

        # Windows with no coverage
        no_coverage = sum(1 for w in self.window_coverage if len(w['stations']) == 0)
        if no_coverage > 0:
            print(f"\nWindows with NO DSN coverage: {no_coverage} ({100*no_coverage/total_windows:.1f}%)")
            print(f"  → Recommendation: Increase TX power or antenna gain")

    def generate_report(self, output_path: str = "surface_to_earth_test_report.txt"):
        """Generate detailed test report."""

        with open(output_path, 'w') as f:
            # Header
            f.write("="*80 + "\n")
            f.write(f"{self.test_name}\n")
            f.write("="*80 + "\n\n")

            start_dt = datetime.fromisoformat(self.test_start.replace('T', ' '))
            end_dt = start_dt + timedelta(hours=self.test_duration_hours)

            f.write(f"Test Period: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} to ")
            f.write(f"{end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
            f.write(f"Duration: {self.test_duration_hours} hours ({self.test_duration_hours/24:.1f} days)\n")
            f.write(f"Location: {self.test_location}\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Transmitter configuration
            f.write("="*80 + "\n")
            f.write("TRANSMITTER CONFIGURATION\n")
            f.write("="*80 + "\n\n")
            f.write(f"Name: {self.transmitter['name']}\n")
            f.write(f"Location: ({self.transmitter['lat']:.6f}°, {self.transmitter['lon']:.6f}°)\n")
            f.write(f"Antenna Height: {self.transmitter['altitude']} m above local terrain\n")
            f.write(f"Frequency: {self.transmitter['frequency_mhz']} MHz (X-band downlink)\n")
            f.write(f"TX Power: {self.transmitter['tx_power_dbm']} dBm ")
            f.write(f"({10**((self.transmitter['tx_power_dbm']-30)/10):.1f} W)\n")
            f.write(f"TX Antenna Gain: {self.transmitter['tx_gain_dbi']} dBi\n")
            f.write(f"Modulation: {self.transmitter['modulation']}\n")
            f.write(f"Target Data Rate: {self.transmitter['data_rate_mbps']} Mbps\n\n")

            # Earth visibility
            f.write("="*80 + "\n")
            f.write("EARTH VISIBILITY ANALYSIS\n")
            f.write("="*80 + "\n\n")

            windows = self.visibility_data['windows']
            total_visible_hours = sum([w['duration_hours'] for w in windows])
            visibility_percentage = 100 * total_visible_hours / self.test_duration_hours

            f.write(f"Total Visibility Windows: {len(windows)}\n")
            f.write(f"Total Visible Time: {total_visible_hours:.2f} hours ({total_visible_hours/24:.2f} days)\n")
            f.write(f"Visibility Coverage: {visibility_percentage:.2f}%\n\n")

            if windows:
                durations = [w['duration_hours'] for w in windows]
                elevations = [w['max_elevation_deg'] for w in windows]

                f.write(f"Window Duration Statistics:\n")
                f.write(f"  Longest: {max(durations):.2f} hours\n")
                f.write(f"  Shortest: {min(durations):.2f} hours\n")
                f.write(f"  Mean: {np.mean(durations):.2f} hours\n")
                f.write(f"  Median: {np.median(durations):.2f} hours\n")
                f.write(f"  Std Dev: {np.std(durations):.2f} hours\n\n")

                f.write(f"Elevation Angle Statistics:\n")
                f.write(f"  Maximum: {max(elevations):.2f}°\n")
                f.write(f"  Mean of Maxima: {np.mean(elevations):.2f}°\n\n")

                f.write("All Visibility Windows:\n\n")
                for i, window in enumerate(windows, 1):
                    try:
                        start_dt = spice.et2datetime(window['start_et'])
                        end_dt = spice.et2datetime(window['end_et'])
                        start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
                        end_str = end_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
                    except:
                        start_str = f"ET {window['start_et']}"
                        end_str = f"ET {window['end_et']}"

                    f.write(f"{i:3d}. Start: {start_str}\n")
                    f.write(f"     End:   {end_str}\n")
                    f.write(f"     Duration: {window['duration_hours']:.2f} hours | ")
                    f.write(f"Max Elev: {window['max_elevation_deg']:.2f}° | ")
                    f.write(f"Mean Elev: {window['mean_elevation_deg']:.2f}°\n\n")

            # DSN Link Budgets
            f.write("="*80 + "\n")
            f.write("DSN LINK BUDGET ANALYSIS\n")
            f.write("="*80 + "\n\n")

            for station_name, link in self.dsn_results.items():
                station = self.comm.DSN_STATIONS[station_name]

                f.write(f"\n{station.name}\n")
                f.write("-"*80 + "\n")
                f.write(f"Location: {station.location}\n")
                f.write(f"Coordinates: {station.latitude:.4f}°, {station.longitude:.4f}°\n")
                f.write(f"Altitude: {station.altitude} m\n")
                f.write(f"Dish Diameter: {station.dish_diameter} m\n")
                f.write(f"Min Elevation: {station.min_elevation}°\n\n")

                f.write(f"Link Budget:\n")
                f.write(f"  Distance: {link['distance_km']:,.2f} km\n")
                f.write(f"  Free-Space Path Loss: {link['fspl_db']:.2f} dB\n")
                f.write(f"  TX Power: {link['tx_power_dbm']:.2f} dBm\n")
                f.write(f"  TX Gain: {link['tx_gain_dbi']:.2f} dBi\n")
                f.write(f"  RX Gain: {link['rx_gain_dbi']:.2f} dBi\n")
                f.write(f"  RX Power: {link['rx_power_dbm']:.2f} dBm\n")
                f.write(f"  RX Sensitivity: {link['rx_sensitivity_dbm']:.2f} dBm\n")
                f.write(f"  Link Margin: {link['link_margin_db']:+.2f} dB\n\n")

                f.write(f"Geometry:\n")
                f.write(f"  Moon Visible from Earth: {link['tx_visible']}\n")
                if link['tx_visible']:
                    f.write(f"  Elevation from TX: {link['tx_elevation_deg']:.2f}°\n")
                    f.write(f"  Azimuth from TX: {link['tx_azimuth_deg']:.2f}°\n")

                f.write(f"\nStatus: {'LINK AVAILABLE' if link['link_available'] else 'LINK UNAVAILABLE'}\n")

                if link['link_available']:
                    if link['link_margin_db'] >= 20:
                        quality = "EXCELLENT"
                    elif link['link_margin_db'] >= 15:
                        quality = "VERY GOOD"
                    elif link['link_margin_db'] >= 10:
                        quality = "GOOD"
                    elif link['link_margin_db'] >= 6:
                        quality = "FAIR"
                    else:
                        quality = "MARGINAL"
                    f.write(f"Link Quality: {quality}\n")

                f.write("\n")

            # Coverage summary
            if hasattr(self, 'window_coverage'):
                f.write("="*80 + "\n")
                f.write("DSN COVERAGE SUMMARY\n")
                f.write("="*80 + "\n\n")

                station_counts = {name: 0 for name in self.dsn_stations_to_test}
                for window in self.window_coverage:
                    for station in window['stations']:
                        station_counts[station['name']] += 1

                total_sampled = len(self.window_coverage)

                f.write(f"Station Availability (out of {total_sampled} windows sampled):\n\n")
                for station_name, count in sorted(station_counts.items(), key=lambda x: -x[1]):
                    pct = 100 * count / total_sampled
                    station_obj = self.comm.DSN_STATIONS[station_name]
                    f.write(f"  {station_name:25s} ({station_obj.dish_diameter:.0f}m): ")
                    f.write(f"{count:3d} / {total_sampled} ({pct:5.1f}%)\n")

            # Recommendations
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")

            f.write("Earth Visibility:\n")
            if visibility_percentage < 50:
                f.write(f"  • Low visibility ({visibility_percentage:.1f}%) from this location\n")
                f.write(f"  • Consider relay satellite or different landing site\n")
            elif visibility_percentage < 80:
                f.write(f"  • Moderate visibility ({visibility_percentage:.1f}%)\n")
                f.write(f"  • Plan data buffering during occultation periods\n")
            else:
                f.write(f"  • Good visibility ({visibility_percentage:.1f}%)\n")

            f.write(f"\nDSN Station Selection:\n")
            available_70m = sum(1 for name, link in self.dsn_results.items()
                              if '70' in str(self.comm.DSN_STATIONS[name].dish_diameter)
                              and link['link_available'])
            available_34m = sum(1 for name, link in self.dsn_results.items()
                              if '34' in str(self.comm.DSN_STATIONS[name].dish_diameter)
                              and link['link_available'])

            f.write(f"  • 70m dishes available: {available_70m}/3 complexes\n")
            f.write(f"  • 34m dishes available: {available_34m}/3 complexes\n")

            if available_70m == 3:
                f.write(f"  • Excellent: All DSN complexes accessible\n")
            elif available_70m >= 2:
                f.write(f"  • Good: Multiple DSN complexes accessible\n")
            else:
                f.write(f"  • Limited: Few DSN stations accessible - increase margin\n")

            # Best practices
            f.write(f"\nOperational Recommendations:\n")
            if windows:
                avg_window = np.mean([w['duration_hours'] for w in windows])
                f.write(f"  • Average contact duration: {avg_window:.1f} hours\n")
                f.write(f"  • Schedule data dumps during visibility windows\n")
                f.write(f"  • Implement store-and-forward for continuous operations\n")

            margins = [link['link_margin_db'] for link in self.dsn_results.values()
                      if link['link_available']]
            if margins:
                avg_margin = np.mean(margins)
                if avg_margin >= 15:
                    f.write(f"  • Excellent link margins ({avg_margin:.1f} dB avg) - reliable operation\n")
                elif avg_margin >= 10:
                    f.write(f"  • Good link margins ({avg_margin:.1f} dB avg)\n")
                else:
                    f.write(f"  • Moderate link margins ({avg_margin:.1f} dB avg)\n")
                    f.write(f"  • Consider weather/atmospheric margins for Earth-based DSN\n")

            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        print(f"\n✓ Detailed report saved to: {output_path}")

    def plot_visibility(self):
        """Generate Earth visibility plot."""
        print(f"\nGenerating Earth visibility plot...")
        self.comm.plot_earth_visibility(
            self.visibility_data,
            save_path="earth_visibility_test.png"
        )


def main():
    """Run surface-to-Earth communication test."""

    print("\n" + "="*80)
    print("LUNAR SURFACE-TO-EARTH (DTE) COMMUNICATION TEST")
    print("="*80)

    # Initialize and run test
    test = SurfaceToEarthTest()
    test.analyze_earth_visibility()
    test.analyze_dsn_links()
    test.analyze_dsn_coverage()
    test.generate_report()
    test.plot_visibility()

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

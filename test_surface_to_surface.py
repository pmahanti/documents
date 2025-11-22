#!/usr/bin/env python3
"""
Test Case 1: Surface-to-Surface Communication

Tests LTE communication between a lunar base station and multiple
surface assets (rovers and landers) at specified locations and times.

Test Configuration:
- Base Station: Shackleton Crater rim (-89.5°, 0.0°)
- Test Date: 2025-12-15 12:00:00 UTC
- Frequency: 2600 MHz (S-band LTE)
- Power: 46 dBm (40W)
"""

import numpy as np
from datetime import datetime
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lunar_comm_spice import LunarCommSPICE, SurfaceAsset
from lunar_lte_simulator import LunarLTESimulator, TransmitterConfig


class SurfaceToSurfaceTest:
    """Test surface-to-surface communication links."""

    def __init__(self):
        """Initialize test configuration."""

        # Test metadata
        self.test_name = "Surface-to-Surface Communication Test"
        self.test_date = "2025-12-15 12:00:00 UTC"
        self.test_location = "Shackleton Crater, Lunar South Pole"

        # Base Station Configuration
        # Location: Shackleton Crater rim (proposed Artemis Base Camp site)
        self.base_station = {
            'name': 'Artemis Base Station Alpha',
            'lat': -89.5,  # Near Shackleton Crater rim
            'lon': 0.0,
            'altitude': 10.0,  # 10m antenna mast
            'frequency_mhz': 2600.0,  # S-band LTE (Band 7)
            'tx_power_dbm': 46.0,  # 40W transmitter
            'tx_gain_dbi': 18.0,  # Sector antenna
            'rx_sensitivity_dbm': -110.0
        }

        # Surface Assets Configuration
        self.surface_assets = [
            {
                'name': 'VIPER Rover',
                'type': 'Heavy Rover',
                'lat': -89.35,  # 16.7 km from base
                'lon': 12.0,
                'altitude': 2.0,  # Rover antenna height
                'rx_sensitivity_dbm': -115.0,
                'antenna_gain_dbi': 8.0,
                'mission': 'Ice prospecting in PSR'
            },
            {
                'name': 'Artemis Lander 1',
                'type': 'HLS Lander',
                'lat': -89.65,  # 16.7 km from base (opposite direction)
                'lon': -8.0,
                'altitude': 8.0,  # Lander communications array
                'rx_sensitivity_dbm': -120.0,
                'antenna_gain_dbi': 12.0,
                'mission': 'Crew habitat and operations'
            },
            {
                'name': 'Exploration Rover Alpha',
                'type': 'Light Rover',
                'lat': -89.25,  # 28.1 km from base
                'lon': 25.0,
                'altitude': 1.5,
                'rx_sensitivity_dbm': -110.0,
                'antenna_gain_dbi': 6.0,
                'mission': 'Geological survey'
            },
            {
                'name': 'Communications Relay 1',
                'type': 'Relay Station',
                'lat': -89.0,  # 56.6 km from base
                'lon': 45.0,
                'altitude': 15.0,  # Tall mast for extended coverage
                'rx_sensitivity_dbm': -125.0,
                'antenna_gain_dbi': 18.0,
                'mission': 'Extended range relay'
            },
            {
                'name': 'Resource Extractor 1',
                'type': 'ISRU Plant',
                'lat': -89.75,  # 27.8 km from base
                'lon': 3.0,
                'altitude': 5.0,
                'rx_sensitivity_dbm': -112.0,
                'antenna_gain_dbi': 10.0,
                'mission': 'Water ice extraction'
            },
            {
                'name': 'Science Lander Beta',
                'type': 'Science Station',
                'lat': -88.8,  # 80.2 km from base
                'lon': 60.0,
                'altitude': 3.0,
                'rx_sensitivity_dbm': -108.0,
                'antenna_gain_dbi': 9.0,
                'mission': 'Seismic monitoring'
            }
        ]

        # Initialize SPICE
        print(f"\n{'='*80}")
        print(f"INITIALIZING TEST: {self.test_name}")
        print(f"{'='*80}")
        print(f"\nTest Date: {self.test_date}")
        print(f"Test Location: {self.test_location}")
        print("\nInitializing SPICE kernels...")

        self.comm_spice = LunarCommSPICE(kernel_dir='kernels')

    def run_link_analysis(self):
        """Run communication link analysis for all assets."""

        print(f"\n{'='*80}")
        print("BASE STATION CONFIGURATION")
        print(f"{'='*80}")
        print(f"Name: {self.base_station['name']}")
        print(f"Location: ({self.base_station['lat']:.4f}°, {self.base_station['lon']:.4f}°)")
        print(f"Antenna Height: {self.base_station['altitude']} m")
        print(f"Frequency: {self.base_station['frequency_mhz']} MHz (S-band LTE)")
        print(f"TX Power: {self.base_station['tx_power_dbm']} dBm ({10**((self.base_station['tx_power_dbm']-30)/10):.1f} W)")
        print(f"TX Antenna Gain: {self.base_station['tx_gain_dbi']} dBi")

        print(f"\n{'='*80}")
        print("SURFACE ASSETS")
        print(f"{'='*80}")
        print(f"Total Assets: {len(self.surface_assets)}\n")

        for i, asset_info in enumerate(self.surface_assets, 1):
            print(f"{i}. {asset_info['name']} ({asset_info['type']})")
            print(f"   Location: ({asset_info['lat']:.4f}°, {asset_info['lon']:.4f}°)")
            print(f"   Mission: {asset_info['mission']}")

        # Analyze each link
        print(f"\n{'='*80}")
        print("COMMUNICATION LINK ANALYSIS")
        print(f"{'='*80}")
        print(f"Analysis Time: {self.test_date}\n")

        self.link_results = []

        for asset_info in self.surface_assets:
            # Create SurfaceAsset object
            asset = SurfaceAsset(
                name=asset_info['name'],
                lat=asset_info['lat'],
                lon=asset_info['lon'],
                altitude=asset_info['altitude'],
                receiver_sensitivity_dbm=asset_info['rx_sensitivity_dbm'],
                antenna_gain_dbi=asset_info['antenna_gain_dbi']
            )

            # Calculate link budget
            link = self.comm_spice.check_asset_link(
                tx_lat=self.base_station['lat'],
                tx_lon=self.base_station['lon'],
                tx_alt=self.base_station['altitude'],
                asset=asset,
                tx_power_dbm=self.base_station['tx_power_dbm'],
                tx_gain_dbi=self.base_station['tx_gain_dbi'],
                frequency_mhz=self.base_station['frequency_mhz']
            )

            # Store result with metadata
            result = {
                **link,
                'asset_type': asset_info['type'],
                'mission': asset_info['mission']
            }
            self.link_results.append(result)

            # Print summary
            status_symbol = '✓' if link['link_available'] else '✗'
            status_text = 'LINK OK' if link['link_available'] else 'NO LINK'

            print(f"\n{status_symbol} {asset_info['name']}")
            print(f"   Type: {asset_info['type']}")
            print(f"   Distance: {link['distance_km']:.2f} km")
            print(f"   Path Loss: {link['fspl_db']:.1f} dB")
            print(f"   RX Power: {link['rx_power_dbm']:.1f} dBm")
            print(f"   RX Sensitivity: {asset_info['rx_sensitivity_dbm']:.1f} dBm")
            print(f"   Link Margin: {link['link_margin_db']:+.1f} dB")
            print(f"   Geometric LOS: {link['geometric_los']}")
            print(f"   Status: {status_text}")

            # Add recommendations
            if link['link_available']:
                if link['link_margin_db'] < 6:
                    print(f"   ⚠ WARNING: Low margin - link may be unreliable")
                elif link['link_margin_db'] < 3:
                    print(f"   ⚠ CRITICAL: Marginal link - high packet loss expected")
            else:
                if link['geometric_los']:
                    print(f"   → Recommendation: Increase TX power or antenna gain")
                else:
                    print(f"   → Recommendation: Relay station needed (beyond horizon)")

    def generate_summary_statistics(self):
        """Generate summary statistics."""

        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}\n")

        total_assets = len(self.link_results)
        available_links = sum(1 for r in self.link_results if r['link_available'])
        unavailable_links = total_assets - available_links

        print(f"Total Assets: {total_assets}")
        print(f"Links Available: {available_links} ({100*available_links/total_assets:.1f}%)")
        print(f"Links Unavailable: {unavailable_links} ({100*unavailable_links/total_assets:.1f}%)")

        if available_links > 0:
            available_results = [r for r in self.link_results if r['link_available']]
            margins = [r['link_margin_db'] for r in available_results]
            distances = [r['distance_km'] for r in available_results]

            print(f"\nAvailable Links Performance:")
            print(f"  Average Link Margin: {np.mean(margins):.1f} dB")
            print(f"  Best Link Margin: {np.max(margins):.1f} dB ({available_results[np.argmax(margins)]['asset_name']})")
            print(f"  Worst Link Margin: {np.min(margins):.1f} dB ({available_results[np.argmin(margins)]['asset_name']})")
            print(f"  Average Distance: {np.mean(distances):.1f} km")
            print(f"  Maximum Distance: {np.max(distances):.1f} km ({available_results[np.argmax(distances)]['asset_name']})")

        if unavailable_links > 0:
            unavailable_results = [r for r in self.link_results if not r['link_available']]
            print(f"\nUnavailable Links:")
            for r in unavailable_results:
                reason = "Beyond horizon" if not r['geometric_los'] else "Insufficient power"
                print(f"  • {r['asset_name']}: {r['distance_km']:.1f} km - {reason}")

    def generate_report(self, output_path: str = "surface_to_surface_test_report.txt"):
        """Generate detailed test report."""

        with open(output_path, 'w') as f:
            # Header
            f.write("="*80 + "\n")
            f.write(f"{self.test_name}\n")
            f.write("="*80 + "\n\n")

            f.write(f"Test Date: {self.test_date}\n")
            f.write(f"Test Location: {self.test_location}\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Base Station
            f.write("="*80 + "\n")
            f.write("BASE STATION CONFIGURATION\n")
            f.write("="*80 + "\n\n")
            f.write(f"Name: {self.base_station['name']}\n")
            f.write(f"Location: ({self.base_station['lat']:.6f}°, {self.base_station['lon']:.6f}°)\n")
            f.write(f"Antenna Height: {self.base_station['altitude']} m above local terrain\n")
            f.write(f"Frequency: {self.base_station['frequency_mhz']} MHz\n")
            f.write(f"Technology: 4G LTE (S-band)\n")
            f.write(f"TX Power: {self.base_station['tx_power_dbm']} dBm ({10**((self.base_station['tx_power_dbm']-30)/10):.1f} W)\n")
            f.write(f"TX Antenna Gain: {self.base_station['tx_gain_dbi']} dBi\n")
            f.write(f"RX Sensitivity: {self.base_station['rx_sensitivity_dbm']} dBm\n\n")

            # Assets
            f.write("="*80 + "\n")
            f.write("SURFACE ASSETS\n")
            f.write("="*80 + "\n\n")

            for i, asset_info in enumerate(self.surface_assets, 1):
                f.write(f"{i}. {asset_info['name']}\n")
                f.write(f"   Type: {asset_info['type']}\n")
                f.write(f"   Location: ({asset_info['lat']:.6f}°, {asset_info['lon']:.6f}°)\n")
                f.write(f"   Antenna Height: {asset_info['altitude']} m\n")
                f.write(f"   RX Sensitivity: {asset_info['rx_sensitivity_dbm']} dBm\n")
                f.write(f"   Antenna Gain: {asset_info['antenna_gain_dbi']} dBi\n")
                f.write(f"   Mission: {asset_info['mission']}\n\n")

            # Link Analysis Results
            f.write("="*80 + "\n")
            f.write("COMMUNICATION LINK ANALYSIS RESULTS\n")
            f.write("="*80 + "\n\n")

            for result in self.link_results:
                f.write(f"\n{result['asset_name']}\n")
                f.write("-"*80 + "\n")
                f.write(f"Asset Type: {result['asset_type']}\n")
                f.write(f"Mission: {result['mission']}\n")
                f.write(f"Distance: {result['distance_km']:.3f} km\n")
                f.write(f"Free-Space Path Loss: {result['fspl_db']:.2f} dB\n")
                f.write(f"TX Power: {self.base_station['tx_power_dbm']:.1f} dBm\n")
                f.write(f"TX Gain: {self.base_station['tx_gain_dbi']:.1f} dBi\n")
                f.write(f"RX Gain: {result['asset_name']}: {[a for a in self.surface_assets if a['name']==result['asset_name']][0]['antenna_gain_dbi']:.1f} dBi\n")
                f.write(f"RX Power: {result['rx_power_dbm']:.2f} dBm\n")

                asset_data = [a for a in self.surface_assets if a['name']==result['asset_name']][0]
                f.write(f"RX Sensitivity: {asset_data['rx_sensitivity_dbm']:.1f} dBm\n")
                f.write(f"Link Margin: {result['link_margin_db']:+.2f} dB\n")
                f.write(f"Geometric LOS: {'YES' if result['geometric_los'] else 'NO'}\n")
                f.write(f"Link Status: {'AVAILABLE' if result['link_available'] else 'UNAVAILABLE'}\n")

                # Link quality assessment
                if result['link_available']:
                    if result['link_margin_db'] >= 15:
                        quality = "EXCELLENT"
                    elif result['link_margin_db'] >= 10:
                        quality = "GOOD"
                    elif result['link_margin_db'] >= 6:
                        quality = "FAIR"
                    elif result['link_margin_db'] >= 3:
                        quality = "MARGINAL"
                    else:
                        quality = "POOR"
                    f.write(f"Link Quality: {quality}\n")

                f.write("\n")

            # Summary Statistics
            f.write("="*80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("="*80 + "\n\n")

            total_assets = len(self.link_results)
            available_links = sum(1 for r in self.link_results if r['link_available'])
            unavailable_links = total_assets - available_links

            f.write(f"Total Assets: {total_assets}\n")
            f.write(f"Links Available: {available_links} ({100*available_links/total_assets:.1f}%)\n")
            f.write(f"Links Unavailable: {unavailable_links} ({100*unavailable_links/total_assets:.1f}%)\n\n")

            if available_links > 0:
                available_results = [r for r in self.link_results if r['link_available']]
                margins = [r['link_margin_db'] for r in available_results]
                distances = [r['distance_km'] for r in available_results]
                rx_powers = [r['rx_power_dbm'] for r in available_results]

                f.write("Performance Metrics (Available Links Only):\n")
                f.write(f"  Link Margin:\n")
                f.write(f"    Mean: {np.mean(margins):.2f} dB\n")
                f.write(f"    Min:  {np.min(margins):.2f} dB ({available_results[np.argmin(margins)]['asset_name']})\n")
                f.write(f"    Max:  {np.max(margins):.2f} dB ({available_results[np.argmax(margins)]['asset_name']})\n")
                f.write(f"    Std:  {np.std(margins):.2f} dB\n\n")

                f.write(f"  Distance:\n")
                f.write(f"    Mean: {np.mean(distances):.2f} km\n")
                f.write(f"    Min:  {np.min(distances):.2f} km ({available_results[np.argmin(distances)]['asset_name']})\n")
                f.write(f"    Max:  {np.max(distances):.2f} km ({available_results[np.argmax(distances)]['asset_name']})\n")
                f.write(f"    Std:  {np.std(distances):.2f} km\n\n")

                f.write(f"  Received Power:\n")
                f.write(f"    Mean: {np.mean(rx_powers):.2f} dBm\n")
                f.write(f"    Min:  {np.min(rx_powers):.2f} dBm\n")
                f.write(f"    Max:  {np.max(rx_powers):.2f} dBm\n\n")

            # Recommendations
            f.write("="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")

            if unavailable_links > 0:
                f.write("Assets Without Communication:\n")
                for r in [r for r in self.link_results if not r['link_available']]:
                    if not r['geometric_los']:
                        f.write(f"  • {r['asset_name']}: Beyond radio horizon - deploy relay station\n")
                    else:
                        margin_deficit = abs(r['link_margin_db'])
                        f.write(f"  • {r['asset_name']}: Need {margin_deficit:.1f} dB improvement\n")
                        f.write(f"    Options: Increase TX power, use higher gain antennas, or relay\n")
                f.write("\n")

            # Assets with marginal links
            marginal_links = [r for r in self.link_results if r['link_available'] and r['link_margin_db'] < 6]
            if marginal_links:
                f.write("Assets With Marginal Links (< 6 dB margin):\n")
                for r in marginal_links:
                    f.write(f"  • {r['asset_name']}: {r['link_margin_db']:.1f} dB margin\n")
                    f.write(f"    Recommendation: Increase margin by 3-6 dB for reliability\n")
                f.write("\n")

            # General recommendations
            f.write("General Recommendations:\n")
            f.write(f"  • Network coverage: {100*available_links/total_assets:.1f}%\n")
            if available_links/total_assets < 1.0:
                f.write(f"  • Consider additional relay stations for full coverage\n")
            if available_links > 0:
                avg_margin = np.mean([r['link_margin_db'] for r in self.link_results if r['link_available']])
                if avg_margin < 10:
                    f.write(f"  • Average margin ({avg_margin:.1f} dB) below optimal (10-15 dB)\n")
                    f.write(f"  • Recommend increasing TX power or antenna gains\n")

            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        print(f"\n✓ Detailed report saved to: {output_path}")


def main():
    """Run surface-to-surface communication test."""

    print("\n" + "="*80)
    print("LUNAR SURFACE-TO-SURFACE COMMUNICATION TEST")
    print("="*80)

    # Initialize and run test
    test = SurfaceToSurfaceTest()
    test.run_link_analysis()
    test.generate_summary_statistics()
    test.generate_report()

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

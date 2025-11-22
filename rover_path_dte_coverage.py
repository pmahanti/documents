#!/usr/bin/env python3
"""
Rover Path DTE Coverage Analysis

Analyzes Direct-to-Earth communication availability for a rover
traveling along a planned path on the lunar surface.

Outputs CSV with minute-by-minute DSN station availability at each
point along the rover's trajectory.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lunar_comm_spice import LunarCommSPICE
import spiceypy as spice


class RoverPathDTEAnalyzer:
    """Analyze DTE coverage along a rover path."""

    def __init__(self, kernel_dir: str = 'kernels'):
        """
        Initialize analyzer.

        Args:
            kernel_dir: Directory containing SPICE kernels
        """
        print("Initializing Rover Path DTE Analyzer...")
        self.comm = LunarCommSPICE(kernel_dir)
        self.coverage_data = []

    def interpolate_path(self, waypoints: List[Tuple[float, float]],
                        num_samples: int = 100) -> List[Tuple[float, float]]:
        """
        Interpolate waypoints to create smooth rover path.

        Args:
            waypoints: List of (lat, lon) tuples in degrees
            num_samples: Number of points to sample along path

        Returns:
            List of interpolated (lat, lon) points
        """
        if len(waypoints) < 2:
            return waypoints

        # Convert to arrays
        lats = np.array([w[0] for w in waypoints])
        lons = np.array([w[1] for w in waypoints])

        # Parameter for interpolation
        t_waypoints = np.linspace(0, 1, len(waypoints))
        t_samples = np.linspace(0, 1, num_samples)

        # Interpolate
        lats_interp = np.interp(t_samples, t_waypoints, lats)
        lons_interp = np.interp(t_samples, t_waypoints, lons)

        return list(zip(lats_interp, lons_interp))

    def calculate_path_distance(self, waypoints: List[Tuple[float, float]]) -> float:
        """
        Calculate total path distance in kilometers.

        Args:
            waypoints: List of (lat, lon) tuples

        Returns:
            Total distance in km
        """
        total_dist = 0.0
        R_MOON = 1737.4  # km

        for i in range(len(waypoints) - 1):
            lat1, lon1 = waypoints[i]
            lat2, lon2 = waypoints[i + 1]

            # Haversine formula
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)

            a = (np.sin(dlat/2)**2 +
                 np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
                 np.sin(dlon/2)**2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

            total_dist += R_MOON * c

        return total_dist

    def analyze_coverage(self,
                        waypoints: List[Tuple[float, float]],
                        start_time: str,
                        duration_hours: float,
                        rover_speed_kmh: float = 2.0,
                        rover_antenna_height: float = 2.0,
                        tx_power_dbm: float = 40.0,
                        tx_gain_dbi: float = 20.0,
                        frequency_mhz: float = 8450.0,
                        time_step_minutes: float = 1.0,
                        interpolate_path: bool = True,
                        path_samples: int = 100) -> pd.DataFrame:
        """
        Analyze DTE coverage along rover path.

        Args:
            waypoints: List of (lat, lon) waypoints in degrees
            start_time: Mission start time (ISO format)
            duration_hours: Mission duration (hours)
            rover_speed_kmh: Rover speed (km/h)
            rover_antenna_height: Antenna height above ground (m)
            tx_power_dbm: Transmit power (dBm)
            tx_gain_dbi: Transmit antenna gain (dBi)
            frequency_mhz: Frequency (MHz)
            time_step_minutes: Time resolution (minutes)
            interpolate_path: Interpolate between waypoints
            path_samples: Number of samples for interpolation

        Returns:
            DataFrame with coverage data
        """
        print(f"\n{'='*80}")
        print("ROVER PATH DTE COVERAGE ANALYSIS")
        print(f"{'='*80}\n")

        # Interpolate path if requested
        if interpolate_path and len(waypoints) > 2:
            print(f"Interpolating path with {path_samples} samples...")
            path_points = self.interpolate_path(waypoints, path_samples)
        else:
            path_points = waypoints

        # Calculate path distance
        total_distance = self.calculate_path_distance(path_points)
        print(f"Total Path Distance: {total_distance:.2f} km")
        print(f"Number of Path Points: {len(path_points)}")

        # Calculate mission timeline
        mission_duration = duration_hours * 60  # minutes
        travel_time = total_distance / rover_speed_kmh * 60  # minutes

        print(f"\nMission Parameters:")
        print(f"  Start Time: {start_time}")
        print(f"  Duration: {duration_hours:.1f} hours ({duration_hours*60:.0f} minutes)")
        print(f"  Rover Speed: {rover_speed_kmh:.1f} km/h")
        print(f"  Travel Time: {travel_time:.1f} minutes ({travel_time/60:.1f} hours)")

        if travel_time > mission_duration:
            print(f"\n  ⚠ WARNING: Travel time ({travel_time/60:.1f} hrs) exceeds mission duration!")
            print(f"             Rover will not complete path at {rover_speed_kmh} km/h")

        # Convert start time to ET
        try:
            et_start = spice.str2et(start_time)
        except:
            print(f"\n⚠ WARNING: Could not parse time, using approximation")
            et_start = 0.0

        # Generate time array (every minute)
        time_step_sec = time_step_minutes * 60
        num_time_steps = int(mission_duration / time_step_minutes)
        et_times = et_start + np.arange(num_time_steps) * time_step_sec

        print(f"\nAnalyzing coverage at {len(et_times)} time steps...")
        print(f"  Time Step: {time_step_minutes:.1f} minute(s)")
        print(f"  Total Samples: {len(et_times)} × {len(path_points)} = {len(et_times) * len(path_points):,}")

        # DSN stations to analyze
        dsn_stations = ['Goldstone', 'Canberra', 'Madrid']

        # Initialize results list
        results = []

        # For each time step
        for i, et in enumerate(et_times):
            # Calculate rover position along path based on elapsed time
            elapsed_minutes = i * time_step_minutes
            elapsed_hours = elapsed_minutes / 60.0
            distance_traveled = elapsed_hours * rover_speed_kmh

            # Find rover position on path
            cumulative_dist = 0.0
            rover_lat, rover_lon = path_points[0]

            for j in range(len(path_points) - 1):
                segment_start = path_points[j]
                segment_end = path_points[j + 1]

                # Distance of this segment
                segment_dist = self.calculate_path_distance([segment_start, segment_end])

                if cumulative_dist + segment_dist >= distance_traveled:
                    # Rover is on this segment
                    fraction = (distance_traveled - cumulative_dist) / segment_dist if segment_dist > 0 else 0
                    rover_lat = segment_start[0] + fraction * (segment_end[0] - segment_start[0])
                    rover_lon = segment_start[1] + fraction * (segment_end[1] - segment_start[1])
                    break

                cumulative_dist += segment_dist
            else:
                # Rover has reached end of path
                rover_lat, rover_lon = path_points[-1]

            # Convert ET to datetime
            try:
                timestamp = spice.et2datetime(et)
            except:
                timestamp = datetime.fromisoformat(start_time.replace('T', ' ')) + timedelta(seconds=float(et - et_start))

            # Check Earth visibility
            earth_visible, earth_elev, earth_az = self.comm.check_earth_visibility(
                rover_lat, rover_lon, rover_antenna_height, et
            )

            # Initialize result row
            result = {
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_minutes': elapsed_minutes,
                'rover_lat': rover_lat,
                'rover_lon': rover_lon,
                'distance_traveled_km': min(distance_traveled, total_distance),
                'earth_visible': earth_visible,
                'earth_elevation_deg': earth_elev if earth_visible else np.nan,
                'earth_azimuth_deg': earth_az if earth_visible else np.nan
            }

            # Check each DSN station
            best_margin = -999.0
            best_station = 'None'
            available_stations = []

            for station_name in dsn_stations:
                station = self.comm.DSN_STATIONS[station_name]

                # Calculate link budget
                link = self.comm.calculate_dte_link_budget(
                    tx_lat=rover_lat,
                    tx_lon=rover_lon,
                    tx_alt=rover_antenna_height,
                    tx_power_dbm=tx_power_dbm,
                    tx_gain_dbi=tx_gain_dbi,
                    frequency_mhz=frequency_mhz,
                    et=et,
                    dsn_station=station
                )

                # Store station-specific data
                result[f'{station_name.lower()}_available'] = link['link_available']
                result[f'{station_name.lower()}_margin_db'] = link['link_margin_db'] if link['link_available'] else np.nan
                result[f'{station_name.lower()}_distance_km'] = link['distance_km']

                if link['link_available']:
                    available_stations.append(station_name)
                    if link['link_margin_db'] > best_margin:
                        best_margin = link['link_margin_db']
                        best_station = station_name

            # Summary fields
            result['num_stations_available'] = len(available_stations)
            result['best_station'] = best_station
            result['best_margin_db'] = best_margin if best_margin > -999 else np.nan
            result['any_dsn_available'] = len(available_stations) > 0

            results.append(result)

            # Progress indicator
            if (i + 1) % 100 == 0 or i == 0 or i == len(et_times) - 1:
                progress = 100 * (i + 1) / len(et_times)
                print(f"  Progress: {i+1:5d}/{len(et_times)} ({progress:5.1f}%) - "
                      f"Pos: ({rover_lat:.3f}°, {rover_lon:.3f}°) - "
                      f"DSN: {len(available_stations)}/3")

        # Convert to DataFrame
        df = pd.DataFrame(results)
        self.coverage_data = df

        print(f"\n✓ Analysis complete!")
        print(f"  Total records: {len(df):,}")

        return df

    def generate_summary(self) -> Dict:
        """Generate summary statistics from coverage data."""

        if self.coverage_data is None or len(self.coverage_data) == 0:
            return {}

        df = self.coverage_data

        # Calculate statistics
        total_minutes = len(df)
        earth_visible_minutes = df['earth_visible'].sum()
        any_dsn_minutes = df['any_dsn_available'].sum()

        goldstone_minutes = df['goldstone_available'].sum()
        canberra_minutes = df['canberra_available'].sum()
        madrid_minutes = df['madrid_available'].sum()

        summary = {
            'total_minutes': total_minutes,
            'earth_visible_minutes': earth_visible_minutes,
            'earth_visible_percent': 100 * earth_visible_minutes / total_minutes,
            'any_dsn_minutes': any_dsn_minutes,
            'any_dsn_percent': 100 * any_dsn_minutes / total_minutes,
            'goldstone_minutes': goldstone_minutes,
            'goldstone_percent': 100 * goldstone_minutes / total_minutes,
            'canberra_minutes': canberra_minutes,
            'canberra_percent': 100 * canberra_minutes / total_minutes,
            'madrid_minutes': madrid_minutes,
            'madrid_percent': 100 * madrid_minutes / total_minutes,
            'max_elevation_deg': df['earth_elevation_deg'].max(),
            'mean_elevation_deg': df[df['earth_visible']]['earth_elevation_deg'].mean(),
            'best_margin_overall_db': df['best_margin_db'].max(),
            'mean_margin_db': df[df['any_dsn_available']]['best_margin_db'].mean()
        }

        return summary

    def print_summary(self):
        """Print summary statistics."""

        summary = self.generate_summary()

        if not summary:
            print("No coverage data available")
            return

        print(f"\n{'='*80}")
        print("COVERAGE SUMMARY")
        print(f"{'='*80}\n")

        print(f"Mission Duration: {summary['total_minutes']:.0f} minutes ({summary['total_minutes']/60:.1f} hours)")
        print(f"\nEarth Visibility:")
        print(f"  Visible Time: {summary['earth_visible_minutes']:.0f} min ({summary['earth_visible_percent']:.1f}%)")
        print(f"  Max Elevation: {summary['max_elevation_deg']:.2f}°")
        if not np.isnan(summary['mean_elevation_deg']):
            print(f"  Mean Elevation: {summary['mean_elevation_deg']:.2f}°")

        print(f"\nDSN Station Availability:")
        print(f"  Any Station: {summary['any_dsn_minutes']:.0f} min ({summary['any_dsn_percent']:.1f}%)")
        print(f"  Goldstone:   {summary['goldstone_minutes']:.0f} min ({summary['goldstone_percent']:.1f}%)")
        print(f"  Canberra:    {summary['canberra_minutes']:.0f} min ({summary['canberra_percent']:.1f}%)")
        print(f"  Madrid:      {summary['madrid_minutes']:.0f} min ({summary['madrid_percent']:.1f}%)")

        if not np.isnan(summary['best_margin_overall_db']):
            print(f"\nLink Quality:")
            print(f"  Best Margin: {summary['best_margin_overall_db']:.2f} dB")
            if not np.isnan(summary['mean_margin_db']):
                print(f"  Mean Margin: {summary['mean_margin_db']:.2f} dB")

        # Coverage gaps
        df = self.coverage_data
        no_coverage = df[~df['any_dsn_available']]
        if len(no_coverage) > 0:
            gap_pct = 100 * len(no_coverage) / len(df)
            print(f"\nCoverage Gaps:")
            print(f"  Time Without DSN: {len(no_coverage)} min ({gap_pct:.1f}%)")

    def save_csv(self, output_path: str):
        """
        Save coverage data to CSV file.

        Args:
            output_path: Path to output CSV file
        """
        if self.coverage_data is None or len(self.coverage_data) == 0:
            print("No coverage data to save")
            return

        self.coverage_data.to_csv(output_path, index=False, float_format='%.6f')
        print(f"\n✓ Coverage data saved to: {output_path}")
        print(f"  Rows: {len(self.coverage_data):,}")
        print(f"  Columns: {len(self.coverage_data.columns)}")


def example_rover_path():
    """Example: Analyze DTE coverage for a rover traverse."""

    print("\n" + "="*80)
    print("EXAMPLE: ROVER PATH DTE COVERAGE ANALYSIS")
    print("="*80)

    # Define rover path (waypoints)
    # Example: Traverse from Shackleton rim to interior and back
    waypoints = [
        (-89.50, 0.00),   # Start: Shackleton rim (landing site)
        (-89.55, 5.00),   # Waypoint 1: Move south and east
        (-89.60, 10.00),  # Waypoint 2: Deeper into crater
        (-89.65, 8.00),   # Waypoint 3: Exploration site
        (-89.60, 6.00),   # Waypoint 4: Return path
        (-89.52, 2.00),   # Waypoint 5: Near landing site
        (-89.50, 0.00),   # End: Return to landing site
    ]

    print("\nRover Path Waypoints:")
    for i, (lat, lon) in enumerate(waypoints):
        print(f"  {i}: ({lat:.2f}°, {lon:.2f}°)")

    # Mission parameters
    start_time = "2025-12-20T08:00:00"  # Mission start
    duration_hours = 72  # 3-day mission
    rover_speed_kmh = 1.5  # Slow, careful driving

    # Rover DTE configuration
    rover_antenna_height = 2.5  # meters (mast-mounted antenna)
    tx_power_dbm = 40.0  # 10W
    tx_gain_dbi = 20.0  # Medium-gain steerable antenna
    frequency_mhz = 8450.0  # X-band

    print(f"\nMission Configuration:")
    print(f"  Start: {start_time}")
    print(f"  Duration: {duration_hours} hours")
    print(f"  Rover Speed: {rover_speed_kmh} km/h")
    print(f"  TX Power: {tx_power_dbm} dBm ({10**((tx_power_dbm-30)/10):.1f} W)")
    print(f"  TX Gain: {tx_gain_dbi} dBi")
    print(f"  Frequency: {frequency_mhz} MHz (X-band)")

    # Create analyzer
    analyzer = RoverPathDTEAnalyzer(kernel_dir='kernels')

    # Run analysis
    coverage_df = analyzer.analyze_coverage(
        waypoints=waypoints,
        start_time=start_time,
        duration_hours=duration_hours,
        rover_speed_kmh=rover_speed_kmh,
        rover_antenna_height=rover_antenna_height,
        tx_power_dbm=tx_power_dbm,
        tx_gain_dbi=tx_gain_dbi,
        frequency_mhz=frequency_mhz,
        time_step_minutes=1.0,  # Every minute
        interpolate_path=True,
        path_samples=50
    )

    # Print summary
    analyzer.print_summary()

    # Save to CSV
    output_file = "rover_path_dte_coverage.csv"
    analyzer.save_csv(output_file)

    # Print sample of data
    print(f"\n{'='*80}")
    print("SAMPLE DATA (First 10 Records)")
    print(f"{'='*80}\n")

    print(coverage_df.head(10).to_string(index=False))

    print(f"\n{'='*80}")
    print(f"COMPLETE! See {output_file} for full data")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    example_rover_path()

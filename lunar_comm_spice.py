#!/usr/bin/env python3
"""
Lunar Communication SPICE Integration

This module provides SPICE-based analysis for:
- Direct-to-Earth (DTE) communication via DSN stations
- Surface asset tracking and communication links
- Earth visibility windows from lunar surface
"""

import numpy as np
import spiceypy as spice
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os


@dataclass
class DSNStation:
    """Deep Space Network station configuration."""
    name: str
    location: str  # Goldstone, Canberra, Madrid
    longitude: float  # degrees East
    latitude: float  # degrees North
    altitude: float  # meters above sea level
    dish_diameter: float = 34.0  # meters (default 34m antenna)
    min_elevation: float = 10.0  # degrees above horizon


@dataclass
class SurfaceAsset:
    """Lunar surface asset (rover, lander, etc.)."""
    name: str
    lat: float  # degrees
    lon: float  # degrees
    altitude: float = 0.0  # meters above local terrain
    receiver_sensitivity_dbm: float = -120.0
    antenna_gain_dbi: float = 10.0


class LunarCommSPICE:
    """SPICE-based lunar communication analysis."""

    # DSN Station definitions
    DSN_STATIONS = {
        'Goldstone': DSNStation(
            name='Goldstone DSS-14 (70m)',
            location='California, USA',
            longitude=-116.8895,
            latitude=35.4267,
            altitude=1001.0,
            dish_diameter=70.0,
            min_elevation=10.0
        ),
        'Goldstone_34m': DSNStation(
            name='Goldstone DSS-24 (34m)',
            location='California, USA',
            longitude=-116.8745,
            latitude=35.3393,
            altitude=951.0,
            dish_diameter=34.0,
            min_elevation=10.0
        ),
        'Canberra': DSNStation(
            name='Canberra DSS-43 (70m)',
            location='Australia',
            longitude=148.9812,
            latitude=-35.4019,
            altitude=691.0,
            dish_diameter=70.0,
            min_elevation=10.0
        ),
        'Canberra_34m': DSNStation(
            name='Canberra DSS-34 (34m)',
            location='Australia',
            longitude=148.9816,
            latitude=-35.3984,
            altitude=692.0,
            dish_diameter=34.0,
            min_elevation=10.0
        ),
        'Madrid': DSNStation(
            name='Madrid DSS-63 (70m)',
            location='Spain',
            longitude=-4.2480,
            latitude=40.4272,
            altitude=834.0,
            dish_diameter=70.0,
            min_elevation=10.0
        ),
        'Madrid_34m': DSNStation(
            name='Madrid DSS-54 (34m)',
            location='Spain',
            longitude=-4.2511,
            latitude=40.4278,
            altitude=837.0,
            dish_diameter=34.0,
            min_elevation=10.0
        ),
    }

    def __init__(self, kernel_dir: str = 'kernels'):
        """
        Initialize SPICE analysis.

        Args:
            kernel_dir: Directory containing SPICE kernels
        """
        self.kernel_dir = kernel_dir
        self.kernels_loaded = []
        self._load_kernels()

    def _load_kernels(self):
        """Load SPICE kernels."""
        if not os.path.exists(self.kernel_dir):
            raise FileNotFoundError(f"Kernel directory not found: {self.kernel_dir}")

        # Clear existing kernels
        spice.kclear()

        # Load standard kernels
        kernel_files = []

        # Look for common kernel types
        for root, dirs, files in os.walk(self.kernel_dir):
            for file in files:
                if file.endswith(('.bsp', '.tpc', '.tf', '.tls', '.tsc')):
                    kernel_files.append(os.path.join(root, file))

        # Load kernels
        for kernel in kernel_files:
            try:
                spice.furnsh(kernel)
                self.kernels_loaded.append(kernel)
                print(f"Loaded: {os.path.basename(kernel)}")
            except Exception as e:
                print(f"Warning: Could not load {kernel}: {e}")

        # Load leap seconds kernel if not present
        # Try to use a generic LSK
        try:
            # Create a minimal LSK if needed (simplified)
            self._ensure_lsk()
        except:
            pass

        if len(self.kernels_loaded) == 0:
            print("Warning: No SPICE kernels loaded. Some functionality may be limited.")

    def _ensure_lsk(self):
        """Ensure leap seconds kernel is available."""
        # Check if LSK is already loaded
        try:
            spice.str2et("2025-01-01T00:00:00")
            return  # LSK already loaded
        except:
            pass

        # Try to load from common locations or create basic one
        lsk_path = os.path.join(self.kernel_dir, "naif0012.tls")
        if os.path.exists(lsk_path):
            spice.furnsh(lsk_path)

    def geodetic_to_moon_fixed(self, lat: float, lon: float, alt: float = 0.0) -> np.ndarray:
        """
        Convert geodetic coordinates to Moon-fixed Cartesian.

        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)
            alt: Altitude above surface (meters)

        Returns:
            Position vector in Moon-fixed frame (km)
        """
        # Moon mean radius (km)
        R_MOON = 1737.4

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        r = (R_MOON + alt / 1000.0)  # km

        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)

        return np.array([x, y, z])

    def earth_position_from_moon(self, et: float) -> np.ndarray:
        """
        Get Earth position vector from Moon center.

        Args:
            et: Ephemeris time (seconds past J2000)

        Returns:
            Position vector (km) in Moon-fixed frame
        """
        try:
            # Get Earth position relative to Moon in J2000 frame
            state, lt = spice.spkezr('EARTH', et, 'J2000', 'LT+S', 'MOON')
            pos_j2000 = state[:3]

            # Transform to Moon-fixed frame (IAU_MOON or MOON_ME)
            try:
                rot_matrix = spice.pxform('J2000', 'IAU_MOON', et)
                pos_moon_fixed = spice.mxv(rot_matrix, pos_j2000)
            except:
                # If IAU_MOON not available, use J2000
                pos_moon_fixed = pos_j2000

            return pos_moon_fixed
        except Exception as e:
            # Fallback: use approximate Earth-Moon distance
            # Average Earth-Moon distance: 384,400 km
            # Simple approximation if SPICE fails
            print(f"Warning: SPICE calculation failed, using approximation: {e}")
            return np.array([384400.0, 0.0, 0.0])

    def check_earth_visibility(self, tx_lat: float, tx_lon: float, tx_alt: float,
                               et: float) -> Tuple[bool, float, float]:
        """
        Check if Earth is visible from transmitter location.

        Args:
            tx_lat: Transmitter latitude (degrees)
            tx_lon: Transmitter longitude (degrees)
            tx_alt: Transmitter altitude above surface (meters)
            et: Ephemeris time

        Returns:
            Tuple of (is_visible, elevation_deg, azimuth_deg)
        """
        # Get transmitter position in Moon-fixed frame
        tx_pos = self.geodetic_to_moon_fixed(tx_lat, tx_lon, tx_alt)

        # Get Earth position from Moon
        earth_pos = self.earth_position_from_moon(et)

        # Vector from transmitter to Earth
        to_earth = earth_pos - tx_pos
        distance = np.linalg.norm(to_earth)
        to_earth_unit = to_earth / distance

        # Local vertical at transmitter (radial outward)
        local_vertical = tx_pos / np.linalg.norm(tx_pos)

        # Calculate elevation angle
        elevation_rad = np.arcsin(np.dot(to_earth_unit, local_vertical))
        elevation_deg = np.degrees(elevation_rad)

        # Calculate azimuth (simplified - from local North)
        # Local East direction
        north = np.array([0, 0, 1])  # Lunar north pole
        east = np.cross(north, local_vertical)
        if np.linalg.norm(east) > 0:
            east = east / np.linalg.norm(east)
            local_north = np.cross(local_vertical, east)
            local_north = local_north / np.linalg.norm(local_north)

            # Project to_earth onto local horizontal plane
            horizontal_component = to_earth_unit - np.dot(to_earth_unit, local_vertical) * local_vertical
            if np.linalg.norm(horizontal_component) > 0:
                horizontal_component = horizontal_component / np.linalg.norm(horizontal_component)

                azimuth_rad = np.arctan2(np.dot(horizontal_component, east),
                                        np.dot(horizontal_component, local_north))
                azimuth_deg = np.degrees(azimuth_rad)
                if azimuth_deg < 0:
                    azimuth_deg += 360
            else:
                azimuth_deg = 0.0
        else:
            azimuth_deg = 0.0

        # Earth is visible if elevation > 0
        is_visible = elevation_deg > 0

        return is_visible, elevation_deg, azimuth_deg

    def calculate_dte_link_budget(self, tx_lat: float, tx_lon: float, tx_alt: float,
                                   tx_power_dbm: float, tx_gain_dbi: float,
                                   frequency_mhz: float, et: float,
                                   dsn_station: DSNStation) -> Dict:
        """
        Calculate Direct-to-Earth link budget.

        Args:
            tx_lat: Transmitter latitude (degrees)
            tx_lon: Transmitter longitude (degrees)
            tx_alt: Transmitter altitude (meters)
            tx_power_dbm: Transmit power (dBm)
            tx_gain_dbi: Transmit antenna gain (dBi)
            frequency_mhz: Frequency (MHz)
            et: Ephemeris time
            dsn_station: DSN station configuration

        Returns:
            Dictionary with link budget results
        """
        # Get transmitter and Earth positions
        tx_pos = self.geodetic_to_moon_fixed(tx_lat, tx_lon, tx_alt)
        earth_pos = self.earth_position_from_moon(et)

        # Calculate distance
        distance_km = np.linalg.norm(earth_pos - tx_pos)

        # Free-space path loss
        fspl_db = 20 * np.log10(distance_km) + 20 * np.log10(frequency_mhz) + 32.45

        # Check Earth visibility from TX
        tx_visible, tx_elevation, tx_azimuth = self.check_earth_visibility(
            tx_lat, tx_lon, tx_alt, et
        )

        # DSN antenna gain (approximate for 70m dish)
        # G ≈ 20*log10(D) + 20*log10(f) + 20*log10(π/c) + 10*log10(efficiency)
        # Simplified: assume 74 dBi for 70m at X-band
        if dsn_station.dish_diameter >= 70:
            if frequency_mhz >= 8000:  # X-band
                rx_gain_dbi = 74.0
            elif frequency_mhz >= 2000:  # S-band
                rx_gain_dbi = 65.0
            else:
                rx_gain_dbi = 60.0
        else:  # 34m dish
            if frequency_mhz >= 8000:
                rx_gain_dbi = 68.0
            elif frequency_mhz >= 2000:
                rx_gain_dbi = 56.0
            else:
                rx_gain_dbi = 50.0

        # Receiver sensitivity (70m DSN)
        if dsn_station.dish_diameter >= 70:
            rx_sensitivity_dbm = -160.0  # Very sensitive
        else:
            rx_sensitivity_dbm = -150.0

        # Calculate received power
        rx_power_dbm = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - fspl_db

        # Link margin
        link_margin_db = rx_power_dbm - rx_sensitivity_dbm

        return {
            'distance_km': distance_km,
            'fspl_db': fspl_db,
            'tx_power_dbm': tx_power_dbm,
            'tx_gain_dbi': tx_gain_dbi,
            'rx_gain_dbi': rx_gain_dbi,
            'rx_power_dbm': rx_power_dbm,
            'rx_sensitivity_dbm': rx_sensitivity_dbm,
            'link_margin_db': link_margin_db,
            'tx_visible': tx_visible,
            'tx_elevation_deg': tx_elevation,
            'tx_azimuth_deg': tx_azimuth,
            'link_available': tx_visible and link_margin_db > 0
        }

    def find_earth_visibility_windows(self, tx_lat: float, tx_lon: float, tx_alt: float,
                                      start_time: str, duration_hours: float,
                                      time_step_minutes: float = 10.0) -> List[Dict]:
        """
        Find Earth visibility windows from transmitter location.

        Args:
            tx_lat: Transmitter latitude (degrees)
            tx_lon: Transmitter longitude (degrees)
            tx_alt: Transmitter altitude (meters)
            start_time: Start time (ISO format or ET)
            duration_hours: Analysis duration (hours)
            time_step_minutes: Time step for analysis (minutes)

        Returns:
            List of visibility window dictionaries
        """
        # Convert start time to ET
        try:
            if isinstance(start_time, str):
                et_start = spice.str2et(start_time)
            else:
                et_start = start_time
        except:
            # Use current time as fallback
            et_start = spice.str2et("2025-11-22T00:00:00")

        # Generate time array
        time_step_sec = time_step_minutes * 60
        n_steps = int(duration_hours * 3600 / time_step_sec)
        et_array = et_start + np.arange(n_steps) * time_step_sec

        # Check visibility at each time step
        visibility = []
        elevations = []
        azimuths = []

        for et in et_array:
            visible, elev, az = self.check_earth_visibility(tx_lat, tx_lon, tx_alt, et)
            visibility.append(visible)
            elevations.append(elev)
            azimuths.append(az)

        # Find continuous visibility windows
        windows = []
        in_window = False
        window_start_idx = 0

        for i, vis in enumerate(visibility):
            if vis and not in_window:
                # Start of window
                in_window = True
                window_start_idx = i
            elif not vis and in_window:
                # End of window
                in_window = False
                windows.append({
                    'start_et': et_array[window_start_idx],
                    'end_et': et_array[i-1],
                    'duration_hours': (et_array[i-1] - et_array[window_start_idx]) / 3600,
                    'max_elevation_deg': max(elevations[window_start_idx:i]),
                    'mean_elevation_deg': np.mean(elevations[window_start_idx:i])
                })

        # Handle case where window extends to end of analysis period
        if in_window:
            windows.append({
                'start_et': et_array[window_start_idx],
                'end_et': et_array[-1],
                'duration_hours': (et_array[-1] - et_array[window_start_idx]) / 3600,
                'max_elevation_deg': max(elevations[window_start_idx:]),
                'mean_elevation_deg': np.mean(elevations[window_start_idx:])
            })

        return {
            'windows': windows,
            'et_array': et_array,
            'visibility': np.array(visibility),
            'elevations': np.array(elevations),
            'azimuths': np.array(azimuths)
        }

    def check_asset_link(self, tx_lat: float, tx_lon: float, tx_alt: float,
                        asset: SurfaceAsset, tx_power_dbm: float, tx_gain_dbi: float,
                        frequency_mhz: float, dem_data: Optional[np.ndarray] = None,
                        dem_transform=None) -> Dict:
        """
        Check communication link to a surface asset.

        Args:
            tx_lat: Transmitter latitude (degrees)
            tx_lon: Transmitter longitude (degrees)
            tx_alt: Transmitter altitude (meters)
            asset: Surface asset configuration
            tx_power_dbm: Transmit power (dBm)
            tx_gain_dbi: Transmit antenna gain (dBi)
            frequency_mhz: Frequency (MHz)
            dem_data: Optional DEM data for LOS check
            dem_transform: Optional DEM transform

        Returns:
            Dictionary with link analysis results
        """
        # Get positions
        tx_pos = self.geodetic_to_moon_fixed(tx_lat, tx_lon, tx_alt)
        asset_pos = self.geodetic_to_moon_fixed(asset.lat, asset.lon, asset.altitude)

        # Calculate distance (convert to km)
        distance_vector = asset_pos - tx_pos
        distance_km = np.linalg.norm(distance_vector)

        # Free-space path loss
        fspl_db = 20 * np.log10(distance_km) + 20 * np.log10(frequency_mhz) + 32.45

        # Calculate received power
        rx_power_dbm = (tx_power_dbm + tx_gain_dbi + asset.antenna_gain_dbi - fspl_db)

        # Link margin
        link_margin_db = rx_power_dbm - asset.receiver_sensitivity_dbm

        # Simple LOS check (geometric only, no terrain)
        # Both assets on surface, so check if horizon blocks
        R_MOON = 1737.4  # km

        # Distance to horizon from TX
        horizon_dist_tx = np.sqrt(2 * R_MOON * (tx_alt / 1000) + (tx_alt / 1000)**2)
        horizon_dist_asset = np.sqrt(2 * R_MOON * (asset.altitude / 1000) + (asset.altitude / 1000)**2)
        max_los_distance = horizon_dist_tx + horizon_dist_asset

        geometric_los = distance_km <= max_los_distance

        return {
            'asset_name': asset.name,
            'distance_km': distance_km,
            'fspl_db': fspl_db,
            'rx_power_dbm': rx_power_dbm,
            'link_margin_db': link_margin_db,
            'geometric_los': geometric_los,
            'link_available': geometric_los and link_margin_db > 0
        }

    def plot_earth_visibility(self, visibility_data: Dict, save_path: Optional[str] = None):
        """
        Plot Earth visibility windows.

        Args:
            visibility_data: Output from find_earth_visibility_windows()
            save_path: Optional path to save figure
        """
        et_array = visibility_data['et_array']
        visibility = visibility_data['visibility']
        elevations = visibility_data['elevations']
        azimuths = visibility_data['azimuths']
        windows = visibility_data['windows']

        # Convert ET to datetime for plotting
        try:
            times = [spice.et2datetime(et) for et in et_array]
        except:
            # Fallback if conversion fails
            times = [datetime(2025, 1, 1) + timedelta(seconds=float(et - et_array[0]))
                    for et in et_array]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Plot 1: Visibility
        ax = axes[0]
        ax.fill_between(times, 0, visibility, alpha=0.5, color='green', label='Earth Visible')
        ax.set_ylabel('Visibility')
        ax.set_ylim(-0.1, 1.1)
        ax.set_title('Earth Visibility from Lunar Surface Transmitter')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 2: Elevation
        ax = axes[1]
        # Only plot positive elevations
        elev_plot = np.copy(elevations)
        elev_plot[~visibility] = np.nan
        ax.plot(times, elev_plot, 'b-', linewidth=2, label='Elevation')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Horizon')
        ax.set_ylabel('Elevation (deg)')
        ax.set_title('Earth Elevation Angle')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 3: Azimuth
        ax = axes[2]
        az_plot = np.copy(azimuths)
        az_plot[~visibility] = np.nan
        ax.plot(times, az_plot, 'g-', linewidth=2, label='Azimuth')
        ax.set_ylabel('Azimuth (deg)')
        ax.set_xlabel('Time')
        ax.set_title('Earth Azimuth Angle')
        ax.set_ylim(0, 360)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # Add text summary
        total_windows = len(windows)
        if total_windows > 0:
            total_duration = sum([w['duration_hours'] for w in windows])
            fig.text(0.02, 0.02,
                    f"Visibility Windows: {total_windows} | Total Duration: {total_duration:.1f} hrs",
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

    def __del__(self):
        """Cleanup SPICE kernels."""
        try:
            spice.kclear()
        except:
            pass


def main():
    """Example usage."""
    print("Lunar Communication SPICE Analysis")
    print("=" * 70)

    # Initialize SPICE
    comm_spice = LunarCommSPICE(kernel_dir='kernels')

    # Example 1: Check Earth visibility windows
    print("\nExample 1: Earth Visibility Windows")
    print("-" * 70)

    tx_lat = -89.5  # Near south pole
    tx_lon = 0.0
    tx_alt = 10.0  # 10m mast

    visibility_data = comm_spice.find_earth_visibility_windows(
        tx_lat, tx_lon, tx_alt,
        start_time="2025-11-22T00:00:00",
        duration_hours=720,  # 30 days
        time_step_minutes=30
    )

    print(f"\nFound {len(visibility_data['windows'])} visibility windows:")
    for i, window in enumerate(visibility_data['windows'][:5], 1):  # Show first 5
        print(f"  Window {i}:")
        print(f"    Duration: {window['duration_hours']:.2f} hours")
        print(f"    Max elevation: {window['max_elevation_deg']:.1f}°")

    # Plot visibility
    comm_spice.plot_earth_visibility(visibility_data, save_path="earth_visibility.png")

    # Example 2: DTE Link Budget
    print("\n\nExample 2: Direct-to-Earth Link Budget")
    print("-" * 70)

    et_example = visibility_data['et_array'][100]  # Pick a time

    for station_name in ['Goldstone', 'Canberra', 'Madrid']:
        station = comm_spice.DSN_STATIONS[station_name]
        link = comm_spice.calculate_dte_link_budget(
            tx_lat, tx_lon, tx_alt,
            tx_power_dbm=50.0,  # 100W
            tx_gain_dbi=24.0,  # High-gain antenna
            frequency_mhz=8450.0,  # X-band
            et=et_example,
            dsn_station=station
        )

        print(f"\n{station.name}:")
        print(f"  Distance: {link['distance_km']:.0f} km")
        print(f"  Path Loss: {link['fspl_db']:.1f} dB")
        print(f"  RX Power: {link['rx_power_dbm']:.1f} dBm")
        print(f"  Link Margin: {link['link_margin_db']:.1f} dB")
        print(f"  Link Available: {link['link_available']}")

    # Example 3: Surface Asset Links
    print("\n\nExample 3: Surface Asset Communication")
    print("-" * 70)

    # Define some surface assets
    assets = [
        SurfaceAsset(name="Rover-1", lat=-89.3, lon=10.0, altitude=2.0),
        SurfaceAsset(name="Lander-A", lat=-89.7, lon=-15.0, altitude=5.0),
        SurfaceAsset(name="Rover-2", lat=-88.5, lon=45.0, altitude=2.0),
    ]

    for asset in assets:
        link = comm_spice.check_asset_link(
            tx_lat, tx_lon, tx_alt,
            asset=asset,
            tx_power_dbm=40.0,  # 10W
            tx_gain_dbi=12.0,
            frequency_mhz=2600.0  # S-band
        )

        print(f"\n{link['asset_name']}:")
        print(f"  Distance: {link['distance_km']:.2f} km")
        print(f"  RX Power: {link['rx_power_dbm']:.1f} dBm")
        print(f"  Link Margin: {link['link_margin_db']:.1f} dB")
        print(f"  Link Available: {link['link_available']}")


if __name__ == "__main__":
    main()

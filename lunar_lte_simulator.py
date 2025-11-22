#!/usr/bin/env python3
"""
Lunar Surface 4G LTE Communication Simulator

This application simulates 4G LTE communication on the lunar surface,
accounting for topography, line-of-sight, and RF propagation characteristics
in the lunar environment (vacuum, no atmosphere).
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass
from typing import Tuple, Optional, List
import warnings


@dataclass
class TransmitterConfig:
    """Configuration parameters for the LTE transmitter."""

    # Location
    lat: float  # Latitude in degrees
    lon: float  # Longitude in degrees
    height_above_ground: float = 10.0  # Height of antenna above local terrain (meters)

    # RF Parameters
    frequency_mhz: float = 2600.0  # 4G LTE Band 7 (2500-2690 MHz)
    transmit_power_dbm: float = 46.0  # Base station TX power (46 dBm = 40W)
    antenna_gain_dbi: float = 18.0  # Directional antenna gain
    antenna_tilt_deg: float = 0.0  # Antenna tilt from horizontal

    # Receiver parameters (for link budget)
    receiver_sensitivity_dbm: float = -110.0  # Typical UE sensitivity
    receiver_gain_dbi: float = 0.0  # User equipment antenna gain

    # Analysis parameters
    max_range_km: float = 50.0  # Maximum analysis radius
    resolution_m: float = 60.0  # Analysis grid resolution

    # Propagation parameters
    polarization: str = 'vertical'  # 'vertical' or 'horizontal'
    include_diffraction: bool = True  # Include knife-edge diffraction


class LunarLTESimulator:
    """Simulates 4G LTE communication on the lunar surface."""

    def __init__(self, dem_path: str, config: TransmitterConfig):
        """
        Initialize the simulator.

        Args:
            dem_path: Path to the Digital Elevation Model (GeoTIFF)
            config: Transmitter configuration
        """
        self.config = config
        self.dem_path = dem_path

        # Load DEM data
        self._load_dem()

        # Calculate transmitter position
        self._set_transmitter_position()

        # Results storage
        self.received_power_dbm = None
        self.path_loss_db = None
        self.los_mask = None
        self.coverage_mask = None

    def _load_dem(self):
        """Load the Digital Elevation Model."""
        with rasterio.open(self.dem_path) as src:
            # Read the elevation data
            self.dem_data = src.read(1)
            self.dem_transform = src.transform
            self.dem_crs = src.crs
            self.dem_bounds = src.bounds

            # Get spatial resolution
            self.dem_resolution = abs(src.transform[0])  # meters per pixel

            # Create coordinate arrays
            height, width = self.dem_data.shape
            self.dem_x = np.arange(width) * self.dem_transform[0] + self.dem_transform[2]
            self.dem_y = np.arange(height) * self.dem_transform[4] + self.dem_transform[5]

    def _set_transmitter_position(self):
        """Calculate transmitter position in DEM coordinates."""
        # For now, assume coordinates are in the same CRS as DEM
        # In practice, you might need coordinate transformation
        self.tx_lon = self.config.lon
        self.tx_lat = self.config.lat

        # Find nearest DEM pixel
        x_idx = np.argmin(np.abs(self.dem_x - self.tx_lon))
        y_idx = np.argmin(np.abs(self.dem_y - self.tx_lat))

        self.tx_x_idx = x_idx
        self.tx_y_idx = y_idx

        # Get elevation at transmitter
        self.tx_elevation = self.dem_data[y_idx, x_idx]
        self.tx_height_asl = self.tx_elevation + self.config.height_above_ground

    def _calculate_free_space_path_loss(self, distance_km: np.ndarray) -> np.ndarray:
        """
        Calculate free-space path loss in vacuum (lunar environment).

        FSPL(dB) = 20*log10(d) + 20*log10(f) + 32.45
        where d is in km and f is in MHz

        Args:
            distance_km: Distance array in kilometers

        Returns:
            Path loss in dB
        """
        freq_mhz = self.config.frequency_mhz

        # Avoid log(0)
        distance_km = np.maximum(distance_km, 0.001)

        fspl = (20 * np.log10(distance_km) +
                20 * np.log10(freq_mhz) +
                32.45)

        return fspl

    def _calculate_fresnel_zone_radius(self, d1: float, d2: float, wavelength: float) -> float:
        """
        Calculate the first Fresnel zone radius at a point.

        Args:
            d1: Distance from transmitter to point (m)
            d2: Distance from point to receiver (m)
            wavelength: Wavelength (m)

        Returns:
            Fresnel zone radius (m)
        """
        if d1 + d2 == 0:
            return 0.0

        return np.sqrt(wavelength * d1 * d2 / (d1 + d2))

    def _check_line_of_sight(self, rx_x: float, rx_y: float, rx_elevation: float,
                            num_samples: int = 100) -> Tuple[bool, float]:
        """
        Check if there is line-of-sight between transmitter and receiver.
        Uses terrain profiling and Fresnel zone clearance.

        Args:
            rx_x: Receiver X coordinate
            rx_y: Receiver Y coordinate
            rx_elevation: Receiver elevation (m)
            num_samples: Number of points to sample along path

        Returns:
            Tuple of (has_los, diffraction_loss_db)
        """
        # Create interpolator for DEM
        interpolator = RegularGridInterpolator(
            (self.dem_y, self.dem_x),
            self.dem_data,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        # Sample points along the path
        tx_x = self.dem_x[self.tx_x_idx]
        tx_y = self.dem_y[self.tx_y_idx]

        x_samples = np.linspace(tx_x, rx_x, num_samples)
        y_samples = np.linspace(tx_y, rx_y, num_samples)

        # Get terrain elevations along path
        points = np.column_stack([y_samples, x_samples])
        terrain_elevation = interpolator(points)

        # Calculate line-of-sight line elevation at each sample point
        distances = np.linspace(0, 1, num_samples)
        los_elevation = (self.tx_height_asl * (1 - distances) +
                        (rx_elevation + self.config.height_above_ground) * distances)

        # Calculate wavelength
        wavelength = 3e8 / (self.config.frequency_mhz * 1e6)  # meters

        # Calculate distance from TX to each sample point
        dx = x_samples - tx_x
        dy = y_samples - tx_y
        dist_from_tx = np.sqrt(dx**2 + dy**2)

        total_distance = dist_from_tx[-1]
        dist_to_rx = total_distance - dist_from_tx

        # Calculate Fresnel zone radius at each point
        fresnel_radius = np.array([
            self._calculate_fresnel_zone_radius(d1, d2, wavelength)
            for d1, d2 in zip(dist_from_tx, dist_to_rx)
        ])

        # Check clearance
        clearance = los_elevation - terrain_elevation

        # Check if path is obstructed
        has_los = np.all(clearance[1:-1] > 0)  # Exclude endpoints

        # Calculate diffraction loss if enabled
        diffraction_loss = 0.0
        if self.config.include_diffraction and not has_los:
            # Find worst obstruction (most negative clearance relative to Fresnel zone)
            normalized_clearance = clearance / (fresnel_radius + 1e-10)
            worst_idx = np.argmin(normalized_clearance)

            # Knife-edge diffraction parameter
            h = -clearance[worst_idx]  # Height of obstacle above LOS
            if h > 0:
                v = h * np.sqrt(2 * (dist_from_tx[worst_idx] + dist_to_rx[worst_idx]) /
                               (wavelength * dist_from_tx[worst_idx] * dist_to_rx[worst_idx]))

                # Knife-edge diffraction loss (ITU-R P.526)
                if v > -0.78:
                    diffraction_loss = 6.9 + 20 * np.log10(np.sqrt((v - 0.1)**2 + 1) + v - 0.1)
                else:
                    diffraction_loss = 0.0

                diffraction_loss = min(diffraction_loss, 40.0)  # Cap at 40 dB

        return has_los, diffraction_loss

    def _create_analysis_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a grid of analysis points around the transmitter.

        Returns:
            Tuple of (x_grid, y_grid, elevation_grid)
        """
        # Calculate grid extent
        max_range_m = self.config.max_range_km * 1000
        resolution = self.config.resolution_m

        # Center on transmitter
        tx_x = self.dem_x[self.tx_x_idx]
        tx_y = self.dem_y[self.tx_y_idx]

        # Create grid
        x_min = tx_x - max_range_m
        x_max = tx_x + max_range_m
        y_min = tx_y - max_range_m
        y_max = tx_y + max_range_m

        n_x = int((x_max - x_min) / resolution)
        n_y = int((y_max - y_min) / resolution)

        x_grid = np.linspace(x_min, x_max, n_x)
        y_grid = np.linspace(y_min, y_max, n_y)

        X, Y = np.meshgrid(x_grid, y_grid)

        # Interpolate elevations
        interpolator = RegularGridInterpolator(
            (self.dem_y, self.dem_x),
            self.dem_data,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        points = np.column_stack([Y.ravel(), X.ravel()])
        elevations = interpolator(points).reshape(Y.shape)

        return X, Y, elevations

    def run_analysis(self, verbose: bool = True):
        """
        Run the complete coverage analysis.

        Args:
            verbose: Print progress messages
        """
        if verbose:
            print("Creating analysis grid...")

        X, Y, elevations = self._create_analysis_grid()

        # Initialize result arrays
        path_loss = np.full_like(elevations, np.inf)
        received_power = np.full_like(elevations, -np.inf)
        los_mask = np.zeros_like(elevations, dtype=bool)

        # Get transmitter position
        tx_x = self.dem_x[self.tx_x_idx]
        tx_y = self.dem_y[self.tx_y_idx]

        # Calculate distances
        dx = X - tx_x
        dy = Y - tx_y
        distances_m = np.sqrt(dx**2 + dy**2)
        distances_km = distances_m / 1000.0

        # Filter points within max range
        mask = distances_km <= self.config.max_range_km

        if verbose:
            print(f"Analyzing {np.sum(mask)} grid points...")

        # Calculate free-space path loss
        fspl = self._calculate_free_space_path_loss(distances_km)

        # Check line-of-sight for each point
        total_points = np.sum(mask)
        analyzed_points = 0

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if not mask[i, j]:
                    continue

                if np.isnan(elevations[i, j]):
                    continue

                # Check LOS
                has_los, diff_loss = self._check_line_of_sight(
                    X[i, j], Y[i, j], elevations[i, j]
                )

                los_mask[i, j] = has_los

                # Calculate total path loss
                total_loss = fspl[i, j] + diff_loss

                # Add additional losses (polarization mismatch, etc.)
                # For lunar surface, assume vertical polarization alignment

                path_loss[i, j] = total_loss

                # Calculate received power
                # P_rx = P_tx + G_tx + G_rx - Path_Loss
                rx_power = (self.config.transmit_power_dbm +
                           self.config.antenna_gain_dbi +
                           self.config.receiver_gain_dbi -
                           total_loss)

                received_power[i, j] = rx_power

                analyzed_points += 1
                if verbose and analyzed_points % 1000 == 0:
                    print(f"  Progress: {analyzed_points}/{total_points} ({100*analyzed_points/total_points:.1f}%)")

        # Store results
        self.X = X
        self.Y = Y
        self.elevations = elevations
        self.path_loss_db = path_loss
        self.received_power_dbm = received_power
        self.los_mask = los_mask
        self.distances_km = distances_km

        # Calculate coverage mask (areas with sufficient signal strength)
        self.coverage_mask = received_power >= self.config.receiver_sensitivity_dbm

        if verbose:
            coverage_area_km2 = np.sum(self.coverage_mask) * (self.config.resolution_m / 1000)**2
            los_percentage = 100 * np.sum(los_mask[mask]) / total_points
            coverage_percentage = 100 * np.sum(self.coverage_mask) / total_points

            print(f"\nAnalysis complete!")
            print(f"  Line-of-sight coverage: {los_percentage:.1f}%")
            print(f"  Signal coverage: {coverage_percentage:.1f}%")
            print(f"  Coverage area: {coverage_area_km2:.2f} km²")

    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot the analysis results.

        Args:
            save_path: Optional path to save the figure
        """
        if self.received_power_dbm is None:
            raise ValueError("Run analysis first!")

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Plot 1: Topography with transmitter location
        ax = axes[0, 0]
        im1 = ax.contourf(self.X/1000, self.Y/1000, self.elevations,
                         levels=20, cmap='terrain', alpha=0.8)
        ax.plot(self.dem_x[self.tx_x_idx]/1000, self.dem_y[self.tx_y_idx]/1000,
               'r*', markersize=20, label='Transmitter')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title('Lunar Surface Topography')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(im1, ax=ax, label='Elevation (m)')

        # Plot 2: Line-of-Sight Coverage
        ax = axes[0, 1]
        los_plot = np.where(self.los_mask, 1, 0)
        los_plot = np.where(self.distances_km > self.config.max_range_km, np.nan, los_plot)
        im2 = ax.contourf(self.X/1000, self.Y/1000, los_plot,
                         levels=[0, 0.5, 1], colors=['red', 'green'], alpha=0.6)
        ax.plot(self.dem_x[self.tx_x_idx]/1000, self.dem_y[self.tx_y_idx]/1000,
               'w*', markersize=20, markeredgecolor='black', linewidth=1.5)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title('Line-of-Sight Coverage')
        ax.grid(True, alpha=0.3)

        # Plot 3: Received Signal Strength
        ax = axes[1, 0]
        rx_power_plot = np.copy(self.received_power_dbm)
        rx_power_plot[self.distances_km > self.config.max_range_km] = np.nan
        rx_power_plot[np.isinf(rx_power_plot)] = np.nan

        vmin = max(self.config.receiver_sensitivity_dbm - 20, np.nanmin(rx_power_plot))
        vmax = np.nanmax(rx_power_plot)

        im3 = ax.contourf(self.X/1000, self.Y/1000, rx_power_plot,
                         levels=20, cmap='RdYlGn', vmin=vmin, vmax=vmax)

        # Add sensitivity contour
        ax.contour(self.X/1000, self.Y/1000, rx_power_plot,
                  levels=[self.config.receiver_sensitivity_dbm],
                  colors='blue', linewidths=2, linestyles='--')

        ax.plot(self.dem_x[self.tx_x_idx]/1000, self.dem_y[self.tx_y_idx]/1000,
               'r*', markersize=20)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title(f'Received Signal Strength (dBm)\n4G LTE @ {self.config.frequency_mhz} MHz')
        ax.grid(True, alpha=0.3)
        plt.colorbar(im3, ax=ax, label='Power (dBm)')

        # Plot 4: Coverage Mask
        ax = axes[1, 1]
        coverage_plot = np.where(self.coverage_mask, 1, 0)
        coverage_plot = np.where(self.distances_km > self.config.max_range_km, np.nan, coverage_plot)

        im4 = ax.contourf(self.X/1000, self.Y/1000, coverage_plot,
                         levels=[0, 0.5, 1], colors=['darkred', 'lightgreen'], alpha=0.7)
        ax.plot(self.dem_x[self.tx_x_idx]/1000, self.dem_y[self.tx_y_idx]/1000,
               'r*', markersize=20)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title(f'Coverage Area\n(Rx > {self.config.receiver_sensitivity_dbm} dBm)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

    def export_results(self, output_path: str):
        """
        Export results to GeoTIFF files.

        Args:
            output_path: Base path for output files (without extension)
        """
        if self.received_power_dbm is None:
            raise ValueError("Run analysis first!")

        # Create transform for output grid
        x_min = self.X.min()
        y_max = self.Y.max()

        transform = rasterio.transform.from_bounds(
            x_min, self.Y.min(), self.X.max(), y_max,
            self.X.shape[1], self.X.shape[0]
        )

        # Export received power
        with rasterio.open(
            f"{output_path}_received_power.tif",
            'w',
            driver='GTiff',
            height=self.received_power_dbm.shape[0],
            width=self.received_power_dbm.shape[1],
            count=1,
            dtype=self.received_power_dbm.dtype,
            crs=self.dem_crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(self.received_power_dbm, 1)

        # Export coverage mask
        with rasterio.open(
            f"{output_path}_coverage.tif",
            'w',
            driver='GTiff',
            height=self.coverage_mask.shape[0],
            width=self.coverage_mask.shape[1],
            count=1,
            dtype='uint8',
            crs=self.dem_crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(self.coverage_mask.astype('uint8'), 1)

        print(f"Results exported to {output_path}_*.tif")


def main():
    """Example usage of the simulator."""

    # Example configuration for a transmitter at lunar south pole
    config = TransmitterConfig(
        lat=-89.9,  # Near south pole
        lon=0.0,
        height_above_ground=10.0,  # 10m mast
        frequency_mhz=2600.0,  # 4G LTE Band 7
        transmit_power_dbm=46.0,  # 40W
        antenna_gain_dbi=18.0,  # High-gain directional
        receiver_sensitivity_dbm=-110.0,
        max_range_km=20.0,  # Analyze 20km radius
        resolution_m=100.0,  # 100m grid resolution
        include_diffraction=True
    )

    # Path to DEM (use one of the COG files)
    dem_path = "SDC60_COG/M012728826S.60m.COG.tif"

    print("Lunar 4G LTE Communication Simulator")
    print("=" * 50)
    print(f"\nTransmitter Configuration:")
    print(f"  Location: ({config.lat:.4f}°, {config.lon:.4f}°)")
    print(f"  Frequency: {config.frequency_mhz} MHz")
    print(f"  TX Power: {config.transmit_power_dbm} dBm")
    print(f"  TX Gain: {config.antenna_gain_dbi} dBi")
    print(f"  Analysis Range: {config.max_range_km} km")
    print()

    # Create simulator
    simulator = LunarLTESimulator(dem_path, config)

    # Run analysis
    simulator.run_analysis(verbose=True)

    # Plot results
    simulator.plot_results(save_path="lunar_lte_coverage.png")

    # Export results
    simulator.export_results("lunar_lte_results")


if __name__ == "__main__":
    main()

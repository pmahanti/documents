"""
Output Manager for Lunar Communication Simulator

Handles generation of PNG and GeoTIFF outputs from simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio.transform import from_bounds
from typing import Dict, Optional, Tuple, List
import os
from datetime import datetime
import pandas as pd


class OutputManager:
    """Manages output generation for simulation results."""

    def __init__(self, output_dir: str = 'simulation_outputs'):
        """
        Initialize output manager.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_surface_coverage_png(self, results: Dict, filename: Optional[str] = None) -> str:
        """
        Save surface coverage map as PNG.

        Args:
            results: Simulation results dictionary
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"surface_coverage_{timestamp}.png"

        output_path = os.path.join(self.output_dir, filename)

        coverage_data = results.get('coverage_data', {})

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Plot 1: Received Power
        ax = axes[0, 0]
        X = coverage_data.get('X', np.array([[0]]))
        Y = coverage_data.get('Y', np.array([[0]]))
        rx_power = coverage_data.get('received_power_dbm', np.array([[0]]))

        # Filter infinite values
        rx_power_plot = np.copy(rx_power)
        rx_power_plot[np.isinf(rx_power_plot)] = np.nan

        vmin = np.nanmin(rx_power_plot)
        vmax = np.nanmax(rx_power_plot)

        im1 = ax.contourf(X/1000, Y/1000, rx_power_plot,
                         levels=20, cmap='RdYlGn', vmin=vmin, vmax=vmax)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title('Received Signal Strength')
        ax.grid(True, alpha=0.3)
        plt.colorbar(im1, ax=ax, label='Power (dBm)')

        # Plot 2: Path Loss
        ax = axes[0, 1]
        path_loss = coverage_data.get('path_loss_db', np.array([[0]]))
        path_loss_plot = np.copy(path_loss)
        path_loss_plot[np.isinf(path_loss_plot)] = np.nan

        im2 = ax.contourf(X/1000, Y/1000, path_loss_plot,
                         levels=20, cmap='RdYlGn_r')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title('Path Loss')
        ax.grid(True, alpha=0.3)
        plt.colorbar(im2, ax=ax, label='Loss (dB)')

        # Plot 3: Coverage Map
        ax = axes[1, 0]
        coverage_mask = coverage_data.get('coverage_mask', np.array([[False]]))
        im3 = ax.contourf(X/1000, Y/1000, coverage_mask.astype(float),
                         levels=[0, 0.5, 1], colors=['red', 'green'], alpha=0.7)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title('Coverage Map')
        ax.grid(True, alpha=0.3)

        # Plot 4: Statistics
        ax = axes[1, 1]
        ax.axis('off')

        stats = results.get('statistics', {})
        config = results.get('config', {})

        stats_text = [
            "SIMULATION PARAMETERS",
            "=" * 40,
            f"Frequency: {config.get('frequency_mhz', 0):.1f} MHz",
            f"TX Power: {config.get('tx_power_dbm', 0):.1f} dBm",
            f"TX Gain: {config.get('tx_gain_dbi', 0):.1f} dBi",
            f"TX Height: {config.get('tx_height_m', 0):.1f} m",
            f"Propagation Model: {config.get('propagation_model', 'N/A')}",
            "",
            "RESULTS",
            "=" * 40,
            f"Coverage: {stats.get('coverage_percentage', 0):.1f}%",
            f"Max Range: {stats.get('max_covered_range_km', 0):.2f} km",
            ""
        ]

        # Add asset info
        asset_links = results.get('asset_links', [])
        if asset_links:
            stats_text.append("ASSET LINKS")
            stats_text.append("=" * 40)
            for link in asset_links:
                status = "✓" if link.get('link_available', False) else "✗"
                stats_text.append(f"{status} {link.get('name', 'Unknown')}: {link.get('link_margin_db', 0):+.1f} dB")

        ax.text(0.05, 0.95, '\n'.join(stats_text),
               transform=ax.transAxes,
               verticalalignment='top',
               fontfamily='monospace',
               fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f"Lunar Surface Communication Coverage\n{results.get('scenario', 'Unknown').replace('_', ' ').title()}",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def save_dte_coverage_png(self, results: Dict, filename: Optional[str] = None) -> str:
        """
        Save DTE coverage timeline as PNG.

        Args:
            results: Simulation results dictionary
            filename: Output filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"dte_coverage_{timestamp}.png"

        output_path = os.path.join(self.output_dir, filename)

        visibility = results.get('visibility', {})

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        et_array = np.array(visibility.get('et_array', []))
        vis_array = np.array(visibility.get('visibility', []))
        elev_array = np.array(visibility.get('elevations', []))
        az_array = np.array(visibility.get('azimuths', []))

        # Convert ET to hours
        if len(et_array) > 0:
            hours = (et_array - et_array[0]) / 3600

            # Plot 1: Visibility
            ax = axes[0]
            ax.fill_between(hours, 0, vis_array, alpha=0.6, color='blue')
            ax.set_ylabel('Visibility')
            ax.set_ylim(-0.1, 1.1)
            ax.set_title('Earth Visibility from Lunar Surface')
            ax.grid(True, alpha=0.3)

            # Plot 2: Elevation
            ax = axes[1]
            elev_plot = np.copy(elev_array)
            elev_plot[~vis_array.astype(bool)] = np.nan
            ax.plot(hours, elev_plot, 'b-', linewidth=2)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_ylabel('Elevation (°)')
            ax.set_title('Earth Elevation Angle')
            ax.grid(True, alpha=0.3)

            # Plot 3: Azimuth
            ax = axes[2]
            az_plot = np.copy(az_array)
            az_plot[~vis_array.astype(bool)] = np.nan
            ax.plot(hours, az_plot, 'g-', linewidth=2)
            ax.set_ylabel('Azimuth (°)')
            ax.set_xlabel('Time (hours)')
            ax.set_title('Earth Azimuth Angle')
            ax.set_ylim(0, 360)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def save_rover_path_png(self, results: Dict, filename: Optional[str] = None) -> str:
        """
        Save rover path coverage as PNG.

        Args:
            results: Simulation results dictionary
            filename: Output filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"rover_path_{timestamp}.png"

        output_path = os.path.join(self.output_dir, filename)

        # Convert coverage data to DataFrame
        coverage_records = results.get('coverage_data', [])
        if not coverage_records:
            print("No coverage data available")
            return None

        df = pd.DataFrame(coverage_records)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Plot 1: Earth visibility
        ax = axes[0]
        ax.fill_between(range(len(df)), 0, df['earth_visible'], alpha=0.6, color='blue')
        ax.set_ylabel('Earth\nVisible')
        ax.set_ylim(-0.1, 1.1)
        ax.set_title('Rover Path DTE Coverage Timeline')
        ax.grid(True, alpha=0.3)

        # Plot 2: Number of available stations
        ax = axes[1]
        ax.plot(df['num_stations_available'], 'b-', linewidth=1)
        ax.fill_between(range(len(df)), 0, df['num_stations_available'], alpha=0.3)
        ax.set_ylabel('# DSN\nStations')
        ax.set_ylim(-0.1, 3.5)
        ax.grid(True, alpha=0.3)

        # Plot 3: Distance traveled
        ax = axes[2]
        ax.plot(df['distance_traveled_km'], 'g-', linewidth=2)
        ax.set_ylabel('Distance (km)')
        ax.set_xlabel('Time (minutes)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def save_geotiff(self, results: Dict, data_layer: str = 'received_power',
                    filename: Optional[str] = None) -> str:
        """
        Save simulation results as GeoTIFF.

        Args:
            results: Simulation results dictionary
            data_layer: Which data layer to export ('received_power', 'path_loss', 'coverage')
            filename: Output filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"coverage_{data_layer}_{timestamp}.tif"

        output_path = os.path.join(self.output_dir, filename)

        coverage_data = results.get('coverage_data', {})

        # Get the data array
        if data_layer == 'received_power':
            data = coverage_data.get('received_power_dbm', np.array([[0]]))
            dtype = rasterio.float32
        elif data_layer == 'path_loss':
            data = coverage_data.get('path_loss_db', np.array([[0]]))
            dtype = rasterio.float32
        elif data_layer == 'coverage':
            data = coverage_data.get('coverage_mask', np.array([[False]]))
            dtype = rasterio.uint8
            data = data.astype(np.uint8)
        else:
            raise ValueError(f"Unknown data layer: {data_layer}")

        # Get spatial extent
        X = coverage_data.get('X', np.array([[0]]))
        Y = coverage_data.get('Y', np.array([[0]]))

        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()

        # Create transform
        transform = from_bounds(x_min, y_min, x_max, y_max,
                               data.shape[1], data.shape[0])

        # Write GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=dtype,
            crs='+proj=longlat +datum=WGS84',  # Simple lat/lon
            transform=transform,
            compress='lzw',
            nodata=-9999 if data_layer != 'coverage' else 255
        ) as dst:
            # Replace inf with nodata
            if data_layer != 'coverage':
                data_out = np.copy(data)
                data_out[np.isinf(data_out)] = -9999
            else:
                data_out = data

            dst.write(data_out, 1)

            # Add metadata
            dst.update_tags(
                scenario=results.get('scenario', 'unknown'),
                timestamp=results.get('timestamp', ''),
                data_layer=data_layer
            )

        return output_path

    def save_csv_report(self, results: Dict, filename: Optional[str] = None) -> str:
        """
        Save simulation results as CSV report.

        Args:
            results: Simulation results dictionary
            filename: Output filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"report_{timestamp}.csv"

        output_path = os.path.join(self.output_dir, filename)

        scenario = results.get('scenario', 'unknown')

        if scenario == 'rover_path_dte':
            # Rover path has structured CSV data
            coverage_records = results.get('coverage_data', [])
            if coverage_records:
                df = pd.DataFrame(coverage_records)
                df.to_csv(output_path, index=False, float_format='%.6f')
        else:
            # Create summary CSV
            config = results.get('config', {})
            stats = results.get('statistics', {})

            summary_data = {
                'Parameter': [],
                'Value': []
            }

            # Add config
            for key, value in config.items():
                if not isinstance(value, (list, dict)):
                    summary_data['Parameter'].append(key)
                    summary_data['Value'].append(str(value))

            # Add stats
            for key, value in stats.items():
                summary_data['Parameter'].append(key)
                summary_data['Value'].append(str(value))

            df = pd.DataFrame(summary_data)
            df.to_csv(output_path, index=False)

        return output_path

    def generate_all_outputs(self, results: Dict, formats: List[str] = None) -> Dict[str, str]:
        """
        Generate all requested output formats.

        Args:
            results: Simulation results
            formats: List of formats to generate ('png', 'geotiff', 'csv')

        Returns:
            Dictionary mapping format to file path
        """
        if formats is None:
            formats = ['png']

        output_files = {}
        scenario = results.get('scenario', 'unknown')

        for fmt in formats:
            if fmt == 'png':
                if scenario in ['surface_to_surface']:
                    path = self.save_surface_coverage_png(results)
                elif scenario in ['surface_to_earth', 'crater_to_earth']:
                    path = self.save_dte_coverage_png(results)
                elif scenario == 'rover_path_dte':
                    path = self.save_rover_path_png(results)
                else:
                    path = None

                if path:
                    output_files['png'] = path

            elif fmt == 'geotiff':
                if scenario == 'surface_to_surface':
                    path = self.save_geotiff(results, data_layer='received_power')
                    output_files['geotiff_power'] = path

                    path = self.save_geotiff(results, data_layer='coverage')
                    output_files['geotiff_coverage'] = path

            elif fmt == 'csv':
                path = self.save_csv_report(results)
                output_files['csv'] = path

        return output_files


if __name__ == "__main__":
    print("Output Manager for Lunar Communication Simulator")
    print("Supports PNG, GeoTIFF, and CSV outputs")

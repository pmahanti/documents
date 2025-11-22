"""
Lunar Communication Simulation Engine

Comprehensive backend engine that integrates all communication analysis
capabilities for the GUI applications.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import os

# Import existing modules
from lunar_lte_simulator import LunarLTESimulator, TransmitterConfig
from lunar_comm_spice import LunarCommSPICE, SurfaceAsset, DSNStation
from rover_path_dte_coverage import RoverPathDTEAnalyzer
from propagation_models import (
    LunarPropagationModels,
    PropagationParameters,
    list_available_models,
    get_model_description
)


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""

    # Scenario selection
    scenario: str  # 'surface_to_surface', 'surface_to_earth', 'crater_to_earth', 'rover_path_dte'

    # Common parameters
    frequency_mhz: float = 2600.0
    tx_power_dbm: float = 40.0
    tx_gain_dbi: float = 12.0
    tx_lat: float = -89.5
    tx_lon: float = 0.0
    tx_height_m: float = 10.0

    # Propagation model selection
    propagation_model: str = 'two_ray'  # Default includes multipath
    include_multipath: bool = True
    include_diffraction: bool = True
    surface_roughness_m: float = 0.1

    # Surface-to-surface specific
    analysis_range_km: float = 20.0
    grid_resolution_m: float = 100.0
    rx_sensitivity_dbm: float = -110.0

    # Surface-to-Earth specific
    dte_start_time: str = "2026-01-01T00:00:00"
    dte_duration_hours: float = 240.0
    dte_time_step_minutes: float = 30.0
    dte_frequency_mhz: float = 8450.0
    dte_tx_power_dbm: float = 50.0
    dte_tx_gain_dbi: float = 30.0

    # Crater scenario specific
    crater_depth_m: float = 100.0
    crater_radius_m: float = 500.0
    tx_inside_crater: bool = True

    # Rover path specific
    rover_waypoints: List[Tuple[float, float]] = None
    rover_speed_kmh: float = 1.5
    rover_mission_hours: float = 72.0

    # Surface assets
    surface_assets: List[Dict] = None

    # Output options
    output_format: List[str] = None  # ['png', 'geotiff', 'csv']
    output_dir: str = 'simulation_outputs'

    # DEM path
    dem_path: Optional[str] = None

    def __post_init__(self):
        if self.output_format is None:
            self.output_format = ['png']
        if self.surface_assets is None:
            self.surface_assets = []
        if self.rover_waypoints is None:
            self.rover_waypoints = []


class LunarCommSimulationEngine:
    """Main simulation engine integrating all capabilities."""

    def __init__(self, config: SimulationConfig):
        """
        Initialize simulation engine.

        Args:
            config: Simulation configuration
        """
        self.config = config
        self.results = {}
        self.status = "initialized"

        # Initialize SPICE if needed
        if config.scenario in ['surface_to_earth', 'crater_to_earth', 'rover_path_dte']:
            try:
                self.spice_comm = LunarCommSPICE(kernel_dir='kernels')
            except Exception as e:
                print(f"Warning: SPICE initialization failed: {e}")
                self.spice_comm = None

        # Propagation models
        self.prop_models = LunarPropagationModels()

    def run_simulation(self) -> Dict[str, Any]:
        """
        Run the selected simulation scenario.

        Returns:
            Dictionary with simulation results
        """
        self.status = "running"

        try:
            if self.config.scenario == 'surface_to_surface':
                results = self._run_surface_to_surface()
            elif self.config.scenario == 'surface_to_earth':
                results = self._run_surface_to_earth()
            elif self.config.scenario == 'crater_to_earth':
                results = self._run_crater_to_earth()
            elif self.config.scenario == 'rover_path_dte':
                results = self._run_rover_path_dte()
            else:
                raise ValueError(f"Unknown scenario: {self.config.scenario}")

            self.results = results
            self.status = "completed"
            return results

        except Exception as e:
            self.status = f"error: {str(e)}"
            raise

    def _run_surface_to_surface(self) -> Dict[str, Any]:
        """Run surface-to-surface communication simulation."""

        results = {
            'scenario': 'surface_to_surface',
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat()
        }

        # Use LTE simulator if DEM available, otherwise use propagation models
        if self.config.dem_path and os.path.exists(self.config.dem_path):
            # Use full LTE simulator with terrain
            tx_config = TransmitterConfig(
                lat=self.config.tx_lat,
                lon=self.config.tx_lon,
                height_above_ground=self.config.tx_height_m,
                frequency_mhz=self.config.frequency_mhz,
                transmit_power_dbm=self.config.tx_power_dbm,
                antenna_gain_dbi=self.config.tx_gain_dbi,
                max_range_km=self.config.analysis_range_km,
                resolution_m=self.config.grid_resolution_m,
                receiver_sensitivity_dbm=self.config.rx_sensitivity_dbm,
                include_diffraction=self.config.include_diffraction
            )

            simulator = LunarLTESimulator(self.config.dem_path, tx_config)
            simulator.run_analysis(verbose=False)

            results['coverage_data'] = {
                'X': simulator.X,
                'Y': simulator.Y,
                'elevations': simulator.elevations,
                'received_power_dbm': simulator.received_power_dbm,
                'path_loss_db': simulator.path_loss_db,
                'los_mask': simulator.los_mask,
                'coverage_mask': simulator.coverage_mask,
                'distances_km': simulator.distances_km
            }

            # Calculate statistics
            coverage_area_km2 = np.sum(simulator.coverage_mask) * (self.config.grid_resolution_m / 1000)**2
            total_area_km2 = np.pi * self.config.analysis_range_km**2
            coverage_pct = 100 * coverage_area_km2 / total_area_km2

            results['statistics'] = {
                'coverage_area_km2': coverage_area_km2,
                'coverage_percentage': coverage_pct,
                'max_range_km': self.config.analysis_range_km
            }

        else:
            # Use analytical propagation models
            results['coverage_data'] = self._calculate_surface_coverage_analytical()
            results['statistics'] = self._calculate_coverage_statistics(results['coverage_data'])

        # Add asset link analysis
        if self.config.surface_assets:
            results['asset_links'] = self._analyze_surface_assets()

        return results

    def _calculate_surface_coverage_analytical(self) -> Dict:
        """Calculate surface coverage using analytical propagation models."""

        # Create grid
        max_range_m = self.config.analysis_range_km * 1000
        resolution = self.config.grid_resolution_m

        n_points = int(2 * max_range_m / resolution)
        x = np.linspace(-max_range_m, max_range_m, n_points)
        y = np.linspace(-max_range_m, max_range_m, n_points)
        X, Y = np.meshgrid(x, y)

        # Calculate distances
        distances_m = np.sqrt(X**2 + Y**2)
        distances_km = distances_m / 1000.0

        # Initialize arrays
        path_loss = np.zeros_like(distances_km)
        received_power = np.zeros_like(distances_km)

        # Calculate for each grid point
        for i in range(n_points):
            for j in range(n_points):
                d_km = distances_km[i, j]

                if d_km > self.config.analysis_range_km:
                    path_loss[i, j] = np.inf
                    received_power[i, j] = -np.inf
                    continue

                # Create propagation parameters
                params = PropagationParameters(
                    frequency_mhz=self.config.frequency_mhz,
                    tx_power_dbm=self.config.tx_power_dbm,
                    tx_gain_dbi=self.config.tx_gain_dbi,
                    tx_height_m=self.config.tx_height_m,
                    rx_gain_dbi=0.0,  # Will add later
                    rx_height_m=2.0,  # Assumed receiver height
                    distance_km=d_km
                )

                # Calculate path loss based on selected model
                if self.config.propagation_model == 'free_space':
                    pl = self.prop_models.free_space_path_loss(d_km, params.frequency_mhz)
                elif self.config.propagation_model == 'two_ray' and self.config.include_multipath:
                    pl, _ = self.prop_models.two_ray_ground_reflection(params)
                elif self.config.propagation_model == 'plane_earth':
                    pl = self.prop_models.plane_earth_loss(params)
                elif self.config.propagation_model == 'egli':
                    pl = self.prop_models.egli_model(params)
                else:
                    # Default to free space
                    pl = self.prop_models.free_space_path_loss(d_km, params.frequency_mhz)

                # Add scattering loss if enabled
                if self.config.surface_roughness_m > 0:
                    scatter = self.prop_models.scattering_loss(params, self.config.surface_roughness_m)
                    pl += scatter

                path_loss[i, j] = pl

                # Calculate received power
                rx_pwr = (self.config.tx_power_dbm +
                         self.config.tx_gain_dbi -
                         pl)
                received_power[i, j] = rx_pwr

        # Coverage mask
        coverage_mask = received_power >= self.config.rx_sensitivity_dbm

        return {
            'X': X,
            'Y': Y,
            'distances_km': distances_km,
            'path_loss_db': path_loss,
            'received_power_dbm': received_power,
            'coverage_mask': coverage_mask
        }

    def _analyze_surface_assets(self) -> List[Dict]:
        """Analyze links to surface assets."""

        asset_results = []

        for asset_config in self.config.surface_assets:
            # Create asset
            asset = SurfaceAsset(
                name=asset_config.get('name', 'Asset'),
                lat=asset_config.get('lat', -89.0),
                lon=asset_config.get('lon', 0.0),
                altitude=asset_config.get('altitude', 2.0),
                receiver_sensitivity_dbm=asset_config.get('rx_sensitivity', -110.0),
                antenna_gain_dbi=asset_config.get('antenna_gain', 8.0)
            )

            # Calculate distance
            # Simplified: use lat/lon difference
            dlat = asset.lat - self.config.tx_lat
            dlon = asset.lon - self.config.tx_lon
            distance_km = np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(asset.lat)))**2)

            # Create parameters
            params = PropagationParameters(
                frequency_mhz=self.config.frequency_mhz,
                tx_power_dbm=self.config.tx_power_dbm,
                tx_gain_dbi=self.config.tx_gain_dbi,
                tx_height_m=self.config.tx_height_m,
                rx_gain_dbi=asset.antenna_gain_dbi,
                rx_height_m=asset.altitude,
                distance_km=distance_km
            )

            # Calculate path loss
            if self.config.propagation_model == 'two_ray' and self.config.include_multipath:
                path_loss, details = self.prop_models.two_ray_ground_reflection(params)
            else:
                path_loss = self.prop_models.free_space_path_loss(distance_km, params.frequency_mhz)
                details = {}

            # Calculate received power
            rx_power = (self.config.tx_power_dbm +
                       self.config.tx_gain_dbi +
                       asset.antenna_gain_dbi -
                       path_loss)

            # Link margin
            link_margin = rx_power - asset.receiver_sensitivity_dbm

            asset_results.append({
                'name': asset.name,
                'distance_km': distance_km,
                'path_loss_db': path_loss,
                'rx_power_dbm': rx_power,
                'link_margin_db': link_margin,
                'link_available': link_margin > 0,
                'multipath_details': details
            })

        return asset_results

    def _run_surface_to_earth(self) -> Dict[str, Any]:
        """Run surface-to-Earth DTE simulation."""

        if self.spice_comm is None:
            raise RuntimeError("SPICE not initialized - cannot run DTE simulation")

        results = {
            'scenario': 'surface_to_earth',
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat()
        }

        # Find Earth visibility windows
        visibility_data = self.spice_comm.find_earth_visibility_windows(
            tx_lat=self.config.tx_lat,
            tx_lon=self.config.tx_lon,
            tx_alt=self.config.tx_height_m,
            start_time=self.config.dte_start_time,
            duration_hours=self.config.dte_duration_hours,
            time_step_minutes=self.config.dte_time_step_minutes
        )

        results['visibility'] = {
            'windows': visibility_data['windows'],
            'et_array': visibility_data['et_array'].tolist(),
            'visibility': visibility_data['visibility'].tolist(),
            'elevations': visibility_data['elevations'].tolist(),
            'azimuths': visibility_data['azimuths'].tolist()
        }

        # Analyze links to DSN stations
        dsn_results = {}
        if visibility_data['windows']:
            # Use middle of first window
            et = (visibility_data['windows'][0]['start_et'] + visibility_data['windows'][0]['end_et']) / 2
        else:
            import spiceypy as spice
            et = spice.str2et(self.config.dte_start_time)

        for station_name in ['Goldstone', 'Canberra', 'Madrid']:
            station = self.spice_comm.DSN_STATIONS[station_name]

            link = self.spice_comm.calculate_dte_link_budget(
                tx_lat=self.config.tx_lat,
                tx_lon=self.config.tx_lon,
                tx_alt=self.config.tx_height_m,
                tx_power_dbm=self.config.dte_tx_power_dbm,
                tx_gain_dbi=self.config.dte_tx_gain_dbi,
                frequency_mhz=self.config.dte_frequency_mhz,
                et=et,
                dsn_station=station
            )

            dsn_results[station_name] = link

        results['dsn_links'] = dsn_results

        return results

    def _run_crater_to_earth(self) -> Dict[str, Any]:
        """Run crater-to-Earth communication simulation."""

        # First run standard DTE analysis
        results = self._run_surface_to_earth()
        results['scenario'] = 'crater_to_earth'

        # Add crater diffraction effects
        crater_loss = self.prop_models.crater_diffraction_loss(
            crater_depth=self.config.crater_depth_m,
            crater_radius=self.config.crater_radius_m,
            tx_position=(0, self.config.tx_height_m) if self.config.tx_inside_crater else (self.config.crater_radius_m + 100, self.config.tx_height_m),
            rx_position=(self.config.analysis_range_km, 100.0),
            frequency_mhz=self.config.dte_frequency_mhz
        )

        results['crater_effects'] = {
            'crater_depth_m': self.config.crater_depth_m,
            'crater_radius_m': self.config.crater_radius_m,
            'tx_inside_crater': self.config.tx_inside_crater,
            'additional_diffraction_loss_db': crater_loss
        }

        # Adjust DSN link margins
        for station_name, link in results['dsn_links'].items():
            link['crater_adjusted_margin_db'] = link['link_margin_db'] - crater_loss
            link['crater_link_available'] = link['crater_adjusted_margin_db'] > 0

        return results

    def _run_rover_path_dte(self) -> Dict[str, Any]:
        """Run rover path DTE coverage analysis."""

        if not self.config.rover_waypoints:
            raise ValueError("No rover waypoints specified")

        analyzer = RoverPathDTEAnalyzer(kernel_dir='kernels')

        coverage_df = analyzer.analyze_coverage(
            waypoints=self.config.rover_waypoints,
            start_time=self.config.dte_start_time,
            duration_hours=self.config.rover_mission_hours,
            rover_speed_kmh=self.config.rover_speed_kmh,
            rover_antenna_height=self.config.tx_height_m,
            tx_power_dbm=self.config.dte_tx_power_dbm,
            tx_gain_dbi=self.config.dte_tx_gain_dbi,
            frequency_mhz=self.config.dte_frequency_mhz,
            time_step_minutes=self.config.dte_time_step_minutes
        )

        summary = analyzer.generate_summary()

        results = {
            'scenario': 'rover_path_dte',
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat(),
            'coverage_data': coverage_df.to_dict('records'),
            'summary': summary,
            'waypoints': self.config.rover_waypoints
        }

        return results

    def _calculate_coverage_statistics(self, coverage_data: Dict) -> Dict:
        """Calculate coverage statistics from coverage data."""

        coverage_mask = coverage_data['coverage_mask']
        distances = coverage_data['distances_km']

        # Calculate coverage area
        total_pixels = np.sum(distances <= self.config.analysis_range_km)
        covered_pixels = np.sum(coverage_mask)

        coverage_pct = 100 * covered_pixels / total_pixels if total_pixels > 0 else 0

        # Calculate covered distance range
        covered_distances = distances[coverage_mask]
        if len(covered_distances) > 0:
            max_covered_range = np.max(covered_distances)
        else:
            max_covered_range = 0

        return {
            'coverage_percentage': coverage_pct,
            'covered_pixels': int(covered_pixels),
            'total_pixels': int(total_pixels),
            'max_covered_range_km': float(max_covered_range)
        }

    def get_available_models(self) -> List[Tuple[str, str]]:
        """Get list of available propagation models."""
        models = list_available_models()
        return [(m, get_model_description(m)) for m in models]


def create_example_config(scenario: str) -> SimulationConfig:
    """Create example configuration for a scenario."""

    if scenario == 'surface_to_surface':
        return SimulationConfig(
            scenario='surface_to_surface',
            tx_lat=-89.5,
            tx_lon=0.0,
            tx_height_m=10.0,
            frequency_mhz=2600.0,
            tx_power_dbm=46.0,
            tx_gain_dbi=18.0,
            analysis_range_km=20.0,
            propagation_model='two_ray',
            include_multipath=True,
            surface_assets=[
                {'name': 'Rover-1', 'lat': -89.3, 'lon': 10.0, 'altitude': 2.0, 'rx_sensitivity': -115.0, 'antenna_gain': 8.0},
                {'name': 'Lander-1', 'lat': -89.7, 'lon': -5.0, 'altitude': 5.0, 'rx_sensitivity': -120.0, 'antenna_gain': 12.0}
            ]
        )

    elif scenario == 'surface_to_earth':
        return SimulationConfig(
            scenario='surface_to_earth',
            tx_lat=-89.5,
            tx_lon=0.0,
            tx_height_m=20.0,
            dte_frequency_mhz=8450.0,
            dte_tx_power_dbm=50.0,
            dte_tx_gain_dbi=30.0,
            dte_start_time="2026-01-15T00:00:00",
            dte_duration_hours=240.0
        )

    elif scenario == 'crater_to_earth':
        return SimulationConfig(
            scenario='crater_to_earth',
            tx_lat=-89.5,
            tx_lon=0.0,
            tx_height_m=10.0,
            crater_depth_m=100.0,
            crater_radius_m=500.0,
            tx_inside_crater=True,
            dte_frequency_mhz=8450.0,
            dte_tx_power_dbm=50.0,
            dte_tx_gain_dbi=30.0
        )

    elif scenario == 'rover_path_dte':
        return SimulationConfig(
            scenario='rover_path_dte',
            rover_waypoints=[
                (-89.50, 0.00),
                (-89.55, 5.00),
                (-89.60, 10.00),
                (-89.50, 0.00)
            ],
            rover_speed_kmh=1.5,
            rover_mission_hours=48.0,
            tx_height_m=2.5,
            dte_frequency_mhz=8450.0,
            dte_tx_power_dbm=43.0,
            dte_tx_gain_dbi=25.0
        )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")


if __name__ == "__main__":
    print("Lunar Communication Simulation Engine")
    print("=" * 60)

    # Example: Surface-to-surface
    print("\nExample: Surface-to-Surface Simulation")
    config = create_example_config('surface_to_surface')
    engine = LunarCommSimulationEngine(config)

    print(f"Running {config.scenario} simulation...")
    results = engine.run_simulation()

    print(f"Status: {engine.status}")
    if 'statistics' in results:
        stats = results['statistics']
        print(f"Coverage: {stats.get('coverage_percentage', 0):.1f}%")

    if 'asset_links' in results:
        print(f"\nAsset Links: {len(results['asset_links'])}")
        for link in results['asset_links']:
            print(f"  {link['name']}: {link['link_margin_db']:+.1f} dB margin")

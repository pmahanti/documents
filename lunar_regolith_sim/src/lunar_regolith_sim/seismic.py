"""
Seismic perturbation module for lunar regolith simulation.

Models moonquakes and impact-induced ground motion that contribute
to regolith mobilization and elephant hide texture formation.
"""

import numpy as np
from scipy.stats import expon, pareto


class MoonquakeSimulator:
    """
    Simulate moonquake activity and its effects on regolith.

    Moonquakes are generally less intense than earthquakes, typically
    ranging from magnitude 2-5, with rare larger events.
    """

    # Moonquake parameters (from Apollo seismic network data)
    TYPICAL_MAGNITUDE_RANGE = (2.0, 5.0)
    DEEP_QUAKE_DEPTH = 700  # km (deep moonquakes, most common)
    SHALLOW_QUAKE_DEPTH = 50  # km (shallow moonquakes, rare but stronger)
    THERMAL_QUAKE_DEPTH = 20  # km (thermal moonquakes near surface)

    # Frequency parameters
    DEEP_QUAKE_RATE = 600  # events per year (clustered in time)
    SHALLOW_QUAKE_RATE = 2  # events per year
    THERMAL_QUAKE_RATE = 100  # events per year
    IMPACT_QUAKE_RATE = 50  # events per year (meteorite impacts)

    def __init__(self, quake_rate_multiplier=1.0, seed=None):
        """
        Initialize moonquake simulator.

        Args:
            quake_rate_multiplier: Multiplier for quake rates (1.0 = typical)
            seed: Random seed for reproducibility
        """
        self.quake_rate_multiplier = quake_rate_multiplier
        self.rng = np.random.default_rng(seed)

    def generate_quake_sequence(self, duration_years):
        """
        Generate sequence of moonquakes over time period.

        Args:
            duration_years: Duration in years

        Returns:
            dict: Arrays of quake times, magnitudes, types, and depths
        """
        total_events = []

        # Deep moonquakes (most common, clustered)
        n_deep = int(self.DEEP_QUAKE_RATE * duration_years * self.quake_rate_multiplier)
        deep_times = np.sort(self.rng.uniform(0, duration_years, n_deep))
        deep_mags = self.rng.uniform(2.0, 3.5, n_deep)
        deep_types = ['deep'] * n_deep
        deep_depths = np.ones(n_deep) * self.DEEP_QUAKE_DEPTH

        # Shallow moonquakes (rare but stronger)
        n_shallow = int(self.SHALLOW_QUAKE_RATE * duration_years * self.quake_rate_multiplier)
        shallow_times = np.sort(self.rng.uniform(0, duration_years, n_shallow))
        shallow_mags = self.rng.uniform(3.0, 5.0, n_shallow)
        shallow_types = ['shallow'] * n_shallow
        shallow_depths = np.ones(n_shallow) * self.SHALLOW_QUAKE_DEPTH

        # Thermal moonquakes
        n_thermal = int(self.THERMAL_QUAKE_RATE * duration_years * self.quake_rate_multiplier)
        thermal_times = np.sort(self.rng.uniform(0, duration_years, n_thermal))
        thermal_mags = self.rng.uniform(1.5, 2.5, n_thermal)
        thermal_types = ['thermal'] * n_thermal
        thermal_depths = np.ones(n_thermal) * self.THERMAL_QUAKE_DEPTH

        # Impact-induced quakes
        n_impact = int(self.IMPACT_QUAKE_RATE * duration_years * self.quake_rate_multiplier)
        impact_times = np.sort(self.rng.uniform(0, duration_years, n_impact))
        impact_mags = self.rng.uniform(2.0, 4.0, n_impact)
        impact_types = ['impact'] * n_impact
        impact_depths = self.rng.uniform(0, 10, n_impact)  # Very shallow

        # Combine all events
        times = np.concatenate([deep_times, shallow_times, thermal_times, impact_times])
        mags = np.concatenate([deep_mags, shallow_mags, thermal_mags, impact_mags])
        types = deep_types + shallow_types + thermal_types + impact_types
        depths = np.concatenate([deep_depths, shallow_depths, thermal_depths, impact_depths])

        # Sort by time
        sort_idx = np.argsort(times)

        return {
            'times': times[sort_idx],
            'magnitudes': mags[sort_idx],
            'types': [types[i] for i in sort_idx],
            'depths': depths[sort_idx]
        }

    def calculate_ground_motion(self, magnitude, distance_km, depth_km):
        """
        Calculate peak ground acceleration from a moonquake.

        Args:
            magnitude: Moonquake magnitude
            distance_km: Distance from epicenter (km)
            depth_km: Depth of quake (km)

        Returns:
            float: Peak ground acceleration in m/s^2
        """
        # Hypocentral distance
        r = np.sqrt(distance_km**2 + depth_km**2)

        # Attenuation relationship (simplified)
        # PGA = A * 10^(B*M) / (r + C)^D
        A = 0.01  # Scaling factor
        B = 0.5   # Magnitude scaling
        C = 10    # Near-field saturation
        D = 1.5   # Geometric spreading + attenuation

        pga = A * 10**(B * magnitude) / (r + C)**D

        return pga

    def calculate_regolith_mobilization(self, ground_acceleration, slope_angle,
                                       cohesion_kpa=0.5):
        """
        Determine if ground motion is sufficient to mobilize regolith.

        Args:
            ground_acceleration: Peak ground acceleration (m/s^2)
            slope_angle: Slope angle in degrees
            cohesion_kpa: Regolith cohesion in kPa

        Returns:
            float: Mobilization factor (0-1, where >0.5 indicates significant motion)
        """
        slope_rad = np.radians(slope_angle)

        # Critical acceleration to overcome cohesion and trigger sliding
        lunar_g = 1.62  # m/s^2
        critical_accel = lunar_g * (np.tan(slope_rad) + cohesion_kpa / 10.0)

        # Mobilization factor
        mobilization = ground_acceleration / critical_accel

        # Saturate at 1.0
        mobilization = min(1.0, mobilization)

        return mobilization


class SeismicPerturbation:
    """
    Apply seismic perturbations to regolith simulation.
    """

    def __init__(self, moonquake_simulator):
        """
        Initialize seismic perturbation.

        Args:
            moonquake_simulator: MoonquakeSimulator object
        """
        self.simulator = moonquake_simulator

    def apply_to_grid(self, slope_angles, epicenter_x, epicenter_y,
                     magnitude, depth_km, grid_resolution):
        """
        Apply seismic perturbation to a 2D grid.

        Args:
            slope_angles: 2D array of slope angles (degrees)
            epicenter_x: Epicenter x coordinate (meters)
            epicenter_y: Epicenter y coordinate (meters)
            magnitude: Moonquake magnitude
            depth_km: Depth of quake (km)
            grid_resolution: Grid spacing in meters

        Returns:
            np.ndarray: Perturbation magnitude at each grid point (0-1 scale)
        """
        ny, nx = slope_angles.shape

        # Create coordinate grids
        x = np.arange(nx) * grid_resolution
        y = np.arange(ny) * grid_resolution
        X, Y = np.meshgrid(x, y)

        # Distance from epicenter
        distance = np.sqrt((X - epicenter_x)**2 + (Y - epicenter_y)**2) / 1000  # km

        # Calculate ground motion at each point
        perturbation = np.zeros_like(slope_angles)

        for i in range(ny):
            for j in range(nx):
                pga = self.simulator.calculate_ground_motion(
                    magnitude, distance[i, j], depth_km
                )
                mobilization = self.simulator.calculate_regolith_mobilization(
                    pga, slope_angles[i, j]
                )
                perturbation[i, j] = mobilization

        return perturbation

    def get_cumulative_seismic_activity(self, duration_years, location_x, location_y,
                                       grid_resolution=1.0, domain_size=100):
        """
        Calculate cumulative seismic activity over time period.

        Args:
            duration_years: Duration in years
            location_x: X coordinate of interest (meters)
            location_y: Y coordinate of interest (meters)
            grid_resolution: Grid spacing in meters
            domain_size: Domain size in meters

        Returns:
            float: Cumulative seismic activity level (0-1 scale)
        """
        # Generate quake sequence
        quakes = self.simulator.generate_quake_sequence(duration_years)

        # Random epicenter locations (uniform over domain)
        n_quakes = len(quakes['times'])
        epicenters_x = self.simulator.rng.uniform(0, domain_size, n_quakes)
        epicenters_y = self.simulator.rng.uniform(0, domain_size, n_quakes)

        # Accumulate effects
        cumulative_activity = 0.0

        for i in range(n_quakes):
            distance_km = np.sqrt(
                (location_x - epicenters_x[i])**2 +
                (location_y - epicenters_y[i])**2
            ) / 1000

            pga = self.simulator.calculate_ground_motion(
                quakes['magnitudes'][i],
                distance_km,
                quakes['depths'][i]
            )

            # Accumulate (with diminishing returns)
            cumulative_activity += pga / (1.0 + cumulative_activity)

        # Normalize to 0-1 scale
        cumulative_activity = min(1.0, cumulative_activity / 10.0)

        return cumulative_activity


class RandomPerturbation:
    """
    Apply random perturbations for regolith mobilization.

    Represents micrometeorite impacts and other stochastic processes.
    """

    def __init__(self, seed=None):
        """
        Initialize random perturbation generator.

        Args:
            seed: Random seed
        """
        self.rng = np.random.default_rng(seed)

    def generate_field(self, shape, intensity=0.1, correlation_length=5.0):
        """
        Generate spatially correlated random perturbation field.

        Args:
            shape: Shape of output array (ny, nx)
            intensity: Perturbation intensity (0-1)
            correlation_length: Correlation length in grid cells

        Returns:
            np.ndarray: Random perturbation field
        """
        from scipy.ndimage import gaussian_filter

        # Generate random field
        field = self.rng.normal(0, intensity, shape)

        # Apply smoothing for spatial correlation
        if correlation_length > 0:
            field = gaussian_filter(field, sigma=correlation_length)

        # Normalize to 0-1 range
        field = (field - field.min()) / (field.max() - field.min() + 1e-10)
        field *= intensity

        return field

    def apply_impact_perturbation(self, grid_shape, impact_x, impact_y,
                                  impact_energy):
        """
        Apply perturbation from a micrometeorite impact.

        Args:
            grid_shape: Shape of grid (ny, nx)
            impact_x: Impact x coordinate (grid cells)
            impact_y: Impact y coordinate (grid cells)
            impact_energy: Impact energy (arbitrary units)

        Returns:
            np.ndarray: Perturbation field
        """
        ny, nx = grid_shape
        perturbation = np.zeros((ny, nx))

        # Create coordinate grids
        y, x = np.ogrid[:ny, :nx]

        # Distance from impact
        distance = np.sqrt((x - impact_x)**2 + (y - impact_y)**2)

        # Gaussian-like decay
        radius = 10 * impact_energy**0.5
        perturbation = impact_energy * np.exp(-(distance / radius)**2)

        return perturbation

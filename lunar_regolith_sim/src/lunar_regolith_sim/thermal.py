"""
Thermal cycling module for lunar regolith simulation.

Models temperature variations over the lunar day/night cycle and their
effects on regolith behavior through thermal expansion and contraction.
"""

import numpy as np


class LunarThermalCycle:
    """
    Model thermal cycling on the lunar surface.

    The Moon experiences extreme temperature variations between day and night,
    causing thermal expansion/contraction that contributes to regolith creep.
    """

    # Lunar thermal constants
    LUNAR_DAY_PERIOD = 29.5 * 24 * 3600  # seconds (29.5 Earth days)
    DAYTIME_TEMP_MAX = 400  # K (near equator)
    NIGHTTIME_TEMP_MIN = 100  # K
    SUBSURFACE_SKIN_DEPTH = 0.5  # meters (thermal penetration depth)

    def __init__(self, period=None, temp_max=None, temp_min=None,
                 latitude=0.0, thermal_inertia=50):
        """
        Initialize lunar thermal cycle model.

        Args:
            period: Lunar day period in seconds
            temp_max: Maximum daytime temperature (K)
            temp_min: Minimum nighttime temperature (K)
            latitude: Latitude in degrees (affects temperature extremes)
            thermal_inertia: Thermal inertia of regolith (J m^-2 K^-1 s^-1/2)
        """
        self.period = period or self.LUNAR_DAY_PERIOD
        self.temp_max = temp_max or self.DAYTIME_TEMP_MAX
        self.temp_min = temp_min or self.NIGHTTIME_TEMP_MIN
        self.latitude = latitude
        self.thermal_inertia = thermal_inertia

        # Adjust temperature extremes based on latitude
        lat_factor = np.cos(np.radians(latitude))
        self.temp_max = self.temp_min + (self.temp_max - self.temp_min) * lat_factor

        # Mean temperature
        self.temp_mean = (self.temp_max + self.temp_min) / 2

    def get_surface_temperature(self, time):
        """
        Calculate surface temperature at given time.

        Args:
            time: Time in seconds (or array of times)

        Returns:
            float or np.ndarray: Surface temperature in Kelvin
        """
        # Simple sinusoidal model
        phase = 2 * np.pi * time / self.period
        amplitude = (self.temp_max - self.temp_min) / 2

        temperature = self.temp_mean + amplitude * np.cos(phase)

        return temperature

    def get_temperature_at_depth(self, time, depth):
        """
        Calculate temperature at depth in regolith.

        Uses simple thermal diffusion model with exponential decay.

        Args:
            time: Time in seconds
            depth: Depth below surface in meters

        Returns:
            float: Temperature in Kelvin
        """
        surface_temp = self.get_surface_temperature(time)

        # Amplitude decays exponentially with depth
        amplitude = (self.temp_max - self.temp_min) / 2
        depth_decay = np.exp(-depth / self.SUBSURFACE_SKIN_DEPTH)

        # Phase lag increases with depth
        phase = 2 * np.pi * time / self.period
        phase_lag = depth / self.SUBSURFACE_SKIN_DEPTH

        temperature = self.temp_mean + amplitude * depth_decay * np.cos(phase - phase_lag)

        return temperature

    def get_thermal_gradient(self, time, depth):
        """
        Calculate vertical thermal gradient.

        Args:
            time: Time in seconds
            depth: Depth in meters

        Returns:
            float: Temperature gradient in K/m
        """
        dz = 0.01  # Small depth increment
        temp_upper = self.get_temperature_at_depth(time, depth)
        temp_lower = self.get_temperature_at_depth(time, depth + dz)

        gradient = (temp_lower - temp_upper) / dz

        return gradient

    def get_thermal_stress(self, time, depth, thermal_expansion_coef=1e-5):
        """
        Calculate thermal stress in regolith.

        Args:
            time: Time in seconds
            depth: Depth in meters
            thermal_expansion_coef: Thermal expansion coefficient (K^-1)

        Returns:
            float: Thermal stress (dimensionless strain)
        """
        # Temperature deviation from mean
        temp = self.get_temperature_at_depth(time, depth)
        delta_temp = temp - self.temp_mean

        # Thermal strain
        thermal_strain = thermal_expansion_coef * delta_temp

        return thermal_strain

    def get_temperature_variation_amplitude(self, depth=0.0):
        """
        Get amplitude of temperature variation at given depth.

        Args:
            depth: Depth in meters

        Returns:
            float: Temperature variation amplitude in K
        """
        amplitude = (self.temp_max - self.temp_min) / 2
        depth_decay = np.exp(-depth / self.SUBSURFACE_SKIN_DEPTH)

        return amplitude * depth_decay

    def calculate_thermal_creep_factor(self, time, depth=0.0):
        """
        Calculate factor representing thermal contribution to creep.

        This varies sinusoidally with the lunar day, with maximum
        creep during periods of rapid temperature change.

        Args:
            time: Time in seconds
            depth: Depth in meters

        Returns:
            float: Creep factor (0-1 scale, with 1 = maximum thermal activity)
        """
        # Rate of temperature change drives creep
        dt = self.period / 1000  # Small time step
        temp_now = self.get_temperature_at_depth(time, depth)
        temp_next = self.get_temperature_at_depth(time + dt, depth)

        # Temperature rate of change
        dtemp_dt = abs(temp_next - temp_now) / dt

        # Normalize to 0-1 scale (max rate occurs at mean temperature)
        max_rate = 2 * np.pi * (self.temp_max - self.temp_min) / (2 * self.period)
        creep_factor = min(1.0, dtemp_dt / max_rate)

        return creep_factor

    def get_number_of_cycles(self, duration):
        """
        Calculate number of lunar day/night cycles in given duration.

        Args:
            duration: Duration in seconds

        Returns:
            float: Number of complete cycles
        """
        return duration / self.period


class ThermalCreepSimulator:
    """
    Simulate thermal creep of regolith over multiple lunar days.
    """

    def __init__(self, thermal_cycle, regolith_physics):
        """
        Initialize thermal creep simulator.

        Args:
            thermal_cycle: LunarThermalCycle object
            regolith_physics: RegolithPhysics object
        """
        self.thermal_cycle = thermal_cycle
        self.physics = regolith_physics

    def calculate_displacement(self, slope_angle, depth, num_cycles):
        """
        Calculate cumulative downslope displacement from thermal cycling.

        Args:
            slope_angle: Slope angle in degrees
            depth: Depth of regolith layer in meters
            num_cycles: Number of lunar day/night cycles

        Returns:
            float: Cumulative displacement in meters
        """
        # Temperature variation amplitude at this depth
        temp_variation = self.thermal_cycle.get_temperature_variation_amplitude(depth)

        # Creep rate per cycle
        creep_per_year = self.physics.calculate_creep_rate(
            slope_angle,
            temperature_variation=temp_variation,
            seismic_activity=0.1  # Background seismic activity
        )

        # Convert to per cycle
        years_per_cycle = self.thermal_cycle.period / (365.25 * 24 * 3600)
        creep_per_cycle = creep_per_year * years_per_cycle

        # Total displacement
        total_displacement = creep_per_cycle * num_cycles

        return total_displacement

    def simulate_texture_development(self, slope_angles, duration_years):
        """
        Simulate elephant hide texture development over geological time.

        Args:
            slope_angles: Array of slope angles in degrees
            duration_years: Simulation duration in years

        Returns:
            dict: Texture development metrics
        """
        # Convert duration to lunar cycles
        duration_seconds = duration_years * 365.25 * 24 * 3600
        num_cycles = self.thermal_cycle.get_number_of_cycles(duration_seconds)

        # Calculate texture intensity based on slope
        texture_intensity = self.physics.get_texture_formation_intensity(slope_angles)

        # Calculate cumulative displacement
        displacement = np.zeros_like(slope_angles)
        for i, slope in enumerate(slope_angles.flat):
            displacement.flat[i] = self.calculate_displacement(
                slope, depth=0.5, num_cycles=num_cycles
            )

        # Texture development = intensity * displacement
        texture_development = texture_intensity * displacement

        return {
            'texture_intensity': texture_intensity,
            'displacement': displacement,
            'texture_development': texture_development,
            'num_cycles': num_cycles,
            'duration_years': duration_years
        }

"""
Physics engine for regolith behavior on lunar slopes.

This module implements the physical properties and mechanics of lunar regolith,
including granular flow, friction, and gravitational effects.
"""

import numpy as np
from numba import jit


class RegolithPhysics:
    """
    Physical properties and mechanics of lunar regolith.

    The elephant hide texture forms due to granular flow instabilities
    on slopes, where regolith avalanches create characteristic patterns.
    """

    # Lunar physical constants
    LUNAR_GRAVITY = 1.62  # m/s^2 (approximately 1/6 of Earth's gravity)

    # Regolith properties (typical values for lunar soil)
    PARTICLE_DENSITY = 1800  # kg/m^3
    ANGLE_OF_REPOSE = 35.0  # degrees (typical for lunar regolith)
    COHESION = 0.1  # kPa (low cohesion for dry regolith)
    INTERNAL_FRICTION_ANGLE = 38.0  # degrees

    def __init__(self, gravity=None, particle_density=None,
                 angle_of_repose=None, cohesion=None):
        """
        Initialize regolith physics parameters.

        Args:
            gravity: Gravitational acceleration (m/s^2)
            particle_density: Density of regolith particles (kg/m^3)
            angle_of_repose: Critical angle for granular flow (degrees)
            cohesion: Cohesive strength of regolith (kPa)
        """
        self.gravity = gravity or self.LUNAR_GRAVITY
        self.particle_density = particle_density or self.PARTICLE_DENSITY
        self.angle_of_repose = angle_of_repose or self.ANGLE_OF_REPOSE
        self.cohesion = cohesion or self.COHESION
        self.internal_friction_angle = self.INTERNAL_FRICTION_ANGLE

        # Convert angles to radians for calculations
        self.angle_of_repose_rad = np.radians(self.angle_of_repose)
        self.internal_friction_rad = np.radians(self.internal_friction_angle)

    def is_stable(self, slope_angle, thickness=1.0):
        """
        Determine if regolith is stable on a given slope.

        Args:
            slope_angle: Slope angle in degrees
            thickness: Regolith layer thickness in meters

        Returns:
            bool: True if stable, False if unstable (will flow)
        """
        slope_rad = np.radians(slope_angle)

        # Calculate shear stress
        shear_stress = self.particle_density * self.gravity * thickness * np.sin(slope_rad)

        # Calculate normal stress
        normal_stress = self.particle_density * self.gravity * thickness * np.cos(slope_rad)

        # Calculate resistance (Mohr-Coulomb criterion)
        resistance = self.cohesion * 1000 + normal_stress * np.tan(self.internal_friction_rad)

        return shear_stress < resistance

    def calculate_flow_velocity(self, slope_angle, thickness, roughness=0.01):
        """
        Calculate granular flow velocity using Savage-Hutter model.

        Args:
            slope_angle: Slope angle in degrees
            thickness: Flow thickness in meters
            roughness: Surface roughness coefficient

        Returns:
            float: Flow velocity in m/s
        """
        slope_rad = np.radians(slope_angle)

        if self.is_stable(slope_angle, thickness):
            return 0.0

        # Savage-Hutter velocity for granular flows
        # v = sqrt(g * h * sin(theta) / (cos(theta) + roughness))
        numerator = self.gravity * thickness * np.sin(slope_rad)
        denominator = np.cos(slope_rad) + roughness

        velocity = np.sqrt(max(0, numerator / denominator))
        return velocity

    def calculate_runout_distance(self, slope_angle, initial_thickness,
                                  initial_velocity=0.0):
        """
        Estimate the runout distance of a regolith avalanche.

        Args:
            slope_angle: Slope angle in degrees
            initial_thickness: Initial flow thickness in meters
            initial_velocity: Initial flow velocity in m/s

        Returns:
            float: Estimated runout distance in meters
        """
        slope_rad = np.radians(slope_angle)

        # Energy dissipation coefficient
        friction_coef = np.tan(self.angle_of_repose_rad)

        # Kinetic energy
        if initial_velocity == 0:
            initial_velocity = self.calculate_flow_velocity(slope_angle, initial_thickness)

        kinetic_energy = 0.5 * initial_velocity**2

        # Potential energy loss
        potential_energy = self.gravity * initial_thickness * np.sin(slope_rad)

        # Runout distance (simplified energy balance)
        if friction_coef > 0:
            runout = (kinetic_energy + potential_energy) / (self.gravity * friction_coef)
        else:
            runout = 100.0  # Default large value

        return runout

    def calculate_deposition_rate(self, velocity, thickness, settling_velocity=0.01):
        """
        Calculate regolith deposition rate.

        Args:
            velocity: Flow velocity in m/s
            thickness: Flow thickness in meters
            settling_velocity: Particle settling velocity in m/s

        Returns:
            float: Deposition rate in m/s
        """
        # Higher velocity = less deposition
        # Thicker flows deposit more material
        deposition_rate = settling_velocity * thickness / (1.0 + velocity)
        return deposition_rate


@jit(nopython=True)
def calculate_stress_field(height_field, dx, dy, gravity, density):
    """
    Calculate stress distribution in regolith layer (JIT-compiled for speed).

    Args:
        height_field: 2D array of regolith heights
        dx: Grid spacing in x direction (meters)
        dy: Grid spacing in y direction (meters)
        gravity: Gravitational acceleration (m/s^2)
        density: Regolith density (kg/m^3)

    Returns:
        tuple: (shear_stress, normal_stress) 2D arrays
    """
    ny, nx = height_field.shape
    shear_stress = np.zeros_like(height_field)
    normal_stress = np.zeros_like(height_field)

    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # Calculate local slope
            dh_dx = (height_field[i, j+1] - height_field[i, j-1]) / (2 * dx)
            dh_dy = (height_field[i+1, j] - height_field[i-1, j]) / (2 * dy)

            slope_magnitude = np.sqrt(dh_dx**2 + dh_dy**2)
            slope_angle = np.arctan(slope_magnitude)

            # Calculate stresses
            h = height_field[i, j]
            shear_stress[i, j] = density * gravity * h * np.sin(slope_angle)
            normal_stress[i, j] = density * gravity * h * np.cos(slope_angle)

    return shear_stress, normal_stress

"""
Slope geometry and terrain generation for lunar simulations.

This module creates various slope configurations that are observed
on the Moon, including crater walls, terrace slopes, and linear slopes.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class SlopeGeometry:
    """
    Generate and manage slope geometries for regolith flow simulation.
    """

    def __init__(self, width, height, resolution=1.0):
        """
        Initialize slope geometry.

        Args:
            width: Width of simulation domain in meters
            height: Height of simulation domain in meters
            resolution: Grid resolution in meters per cell
        """
        self.width = width
        self.height = height
        self.resolution = resolution

        self.nx = int(width / resolution)
        self.ny = int(height / resolution)

        # Initialize elevation grid
        self.elevation = np.zeros((self.ny, self.nx))

        # Grid coordinates
        self.x = np.linspace(0, width, self.nx)
        self.y = np.linspace(0, height, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def create_linear_slope(self, angle, direction='x'):
        """
        Create a simple linear slope.

        Args:
            angle: Slope angle in degrees
            direction: 'x' for east-west slope, 'y' for north-south slope

        Returns:
            np.ndarray: Elevation grid
        """
        slope_rad = np.radians(angle)
        gradient = np.tan(slope_rad)

        if direction == 'x':
            self.elevation = self.X * gradient
        elif direction == 'y':
            self.elevation = self.Y * gradient
        else:
            raise ValueError("Direction must be 'x' or 'y'")

        return self.elevation

    def create_crater_wall(self, crater_x, crater_y, inner_radius,
                          outer_radius, rim_height, floor_depth=0):
        """
        Create a crater wall geometry.

        Elephant hide textures are commonly found on crater walls.

        Args:
            crater_x: X coordinate of crater center (meters)
            crater_y: Y coordinate of crater center (meters)
            inner_radius: Inner radius of crater floor (meters)
            outer_radius: Outer radius of crater rim (meters)
            rim_height: Height of crater rim above surroundings (meters)
            floor_depth: Depth of crater floor below surroundings (meters)

        Returns:
            np.ndarray: Elevation grid
        """
        # Distance from crater center
        dist = np.sqrt((self.X - crater_x)**2 + (self.Y - crater_y)**2)

        # Create crater profile
        self.elevation = np.zeros_like(dist)

        # Crater floor
        floor_mask = dist < inner_radius
        self.elevation[floor_mask] = -floor_depth

        # Crater wall (steep slope region)
        wall_mask = (dist >= inner_radius) & (dist < outer_radius)
        wall_dist = dist[wall_mask]

        # Smooth transition from floor to rim
        wall_fraction = (wall_dist - inner_radius) / (outer_radius - inner_radius)
        self.elevation[wall_mask] = -floor_depth + (rim_height + floor_depth) * wall_fraction**2

        # Outer rim with gentle slope
        rim_mask = dist >= outer_radius
        rim_dist = dist[rim_mask]
        decay_length = outer_radius * 0.5
        self.elevation[rim_mask] = rim_height * np.exp(-(rim_dist - outer_radius) / decay_length)

        return self.elevation

    def create_terrace(self, terrace_y, terrace_height, slope_angle_upper,
                      slope_angle_lower):
        """
        Create a terraced slope (common in impact crater walls).

        Args:
            terrace_y: Y coordinate of terrace position (meters)
            terrace_height: Height of terrace step (meters)
            slope_angle_upper: Slope angle above terrace (degrees)
            slope_angle_lower: Slope angle below terrace (degrees)

        Returns:
            np.ndarray: Elevation grid
        """
        upper_gradient = np.tan(np.radians(slope_angle_upper))
        lower_gradient = np.tan(np.radians(slope_angle_lower))

        self.elevation = np.zeros_like(self.Y)

        # Upper slope
        upper_mask = self.Y >= terrace_y
        self.elevation[upper_mask] = (self.Y[upper_mask] - terrace_y) * upper_gradient + terrace_height

        # Lower slope
        lower_mask = self.Y < terrace_y
        self.elevation[lower_mask] = (terrace_y - self.Y[lower_mask]) * lower_gradient

        return self.elevation

    def add_roughness(self, amplitude=0.1, wavelength=5.0, smoothing=1.0):
        """
        Add surface roughness to existing elevation.

        This simulates small-scale irregularities that influence flow patterns.

        Args:
            amplitude: Amplitude of roughness variations (meters)
            wavelength: Characteristic wavelength of roughness (meters)
            smoothing: Gaussian smoothing sigma (meters)

        Returns:
            np.ndarray: Modified elevation grid
        """
        # Generate random roughness
        np.random.seed(42)  # For reproducibility
        noise = np.random.randn(self.ny, self.nx) * amplitude

        # Apply smoothing to create correlated roughness
        if smoothing > 0:
            sigma_cells = smoothing / self.resolution
            noise = gaussian_filter(noise, sigma=sigma_cells)

        # Scale by wavelength
        scale_factor = wavelength / self.resolution
        if scale_factor > 1:
            from scipy.ndimage import zoom
            # Downsample, smooth, and upsample
            small_noise = zoom(noise, 1.0/scale_factor, order=1)
            small_noise = gaussian_filter(small_noise, sigma=1.0)
            noise = zoom(small_noise, scale_factor, order=1)[:self.ny, :self.nx]

        self.elevation += noise
        return self.elevation

    def add_perturbation(self, x, y, radius, amplitude):
        """
        Add a localized perturbation (e.g., boulder, small crater).

        Args:
            x: X coordinate of perturbation center (meters)
            y: Y coordinate of perturbation center (meters)
            radius: Radius of perturbation (meters)
            amplitude: Height/depth of perturbation (meters)

        Returns:
            np.ndarray: Modified elevation grid
        """
        dist = np.sqrt((self.X - x)**2 + (self.Y - y)**2)
        perturbation = amplitude * np.exp(-(dist / radius)**2)
        self.elevation += perturbation
        return self.elevation

    def get_slope_angle(self):
        """
        Calculate slope angle at each grid point.

        Returns:
            np.ndarray: Slope angles in degrees
        """
        # Calculate gradients
        dy_elev, dx_elev = np.gradient(self.elevation, self.resolution)

        # Calculate slope magnitude
        slope_magnitude = np.sqrt(dx_elev**2 + dy_elev**2)

        # Convert to angle in degrees
        slope_angle = np.degrees(np.arctan(slope_magnitude))

        return slope_angle

    def get_slope_direction(self):
        """
        Calculate slope direction (aspect) at each grid point.

        Returns:
            np.ndarray: Slope directions in degrees (0 = North, 90 = East)
        """
        dy_elev, dx_elev = np.gradient(self.elevation, self.resolution)

        # Calculate aspect (direction of steepest descent)
        aspect = np.degrees(np.arctan2(dy_elev, dx_elev))

        return aspect

"""
Main regolith flow simulation engine.

This module implements the time-stepping simulation of regolith flow
on lunar slopes, using cellular automata and continuum mechanics approaches.
"""

import numpy as np
from numba import jit
from .physics import RegolithPhysics, calculate_stress_field
from .slope import SlopeGeometry


class RegolithFlowSimulation:
    """
    Simulate regolith flow on lunar slopes to generate elephant hide textures.

    The simulation uses a hybrid approach combining:
    - Cellular automata for discrete avalanching events
    - Continuum mechanics for smooth flow
    - Stochastic processes for realistic texture formation
    """

    def __init__(self, slope_geometry, physics=None, initial_thickness=1.0):
        """
        Initialize regolith flow simulation.

        Args:
            slope_geometry: SlopeGeometry object defining the terrain
            physics: RegolithPhysics object (default: lunar conditions)
            initial_thickness: Initial regolith layer thickness in meters
        """
        self.slope = slope_geometry
        self.physics = physics or RegolithPhysics()

        # Simulation grids
        self.thickness = np.ones((self.slope.ny, self.slope.nx)) * initial_thickness
        self.velocity_x = np.zeros((self.slope.ny, self.slope.nx))
        self.velocity_y = np.zeros((self.slope.ny, self.slope.nx))

        # Accumulated deformation (tracks elephant hide formation)
        self.cumulative_deformation = np.zeros((self.slope.ny, self.slope.nx))
        self.flow_count = np.zeros((self.slope.ny, self.slope.nx))

        # Time tracking
        self.time = 0.0
        self.timestep = 0.1  # seconds

    def step(self, dt=None):
        """
        Advance simulation by one time step.

        Args:
            dt: Time step in seconds (default: use self.timestep)

        Returns:
            dict: Simulation state information
        """
        if dt is None:
            dt = self.timestep

        # Get current topography (bedrock + regolith)
        total_elevation = self.slope.elevation + self.thickness

        # Calculate slopes
        slope_angle = self._calculate_local_slopes(total_elevation)

        # Update velocities based on local slopes and physics
        self._update_velocities(slope_angle, dt)

        # Transport regolith
        self._transport_regolith(dt)

        # Apply avalanching (cellular automata)
        avalanche_occurred = self._apply_avalanching(slope_angle)

        # Update deformation tracking
        self._update_deformation(slope_angle)

        # Update time
        self.time += dt

        return {
            'time': self.time,
            'max_velocity': np.max(np.sqrt(self.velocity_x**2 + self.velocity_y**2)),
            'avalanche_occurred': avalanche_occurred,
            'total_deformation': np.sum(self.cumulative_deformation)
        }

    def run(self, duration, progress_interval=None):
        """
        Run simulation for specified duration.

        Args:
            duration: Total simulation time in seconds
            progress_interval: Interval for progress reporting (default: duration/10)

        Returns:
            list: History of simulation states
        """
        if progress_interval is None:
            progress_interval = duration / 10

        history = []
        next_report = progress_interval

        steps = int(duration / self.timestep)
        for i in range(steps):
            state = self.step()

            if self.time >= next_report:
                print(f"Time: {self.time:.1f}s, Max velocity: {state['max_velocity']:.3f} m/s")
                next_report += progress_interval

            # Store snapshot every 100 steps
            if i % 100 == 0:
                history.append({
                    'time': self.time,
                    'thickness': self.thickness.copy(),
                    'deformation': self.cumulative_deformation.copy(),
                    'elevation': self.slope.elevation + self.thickness
                })

        return history

    def _calculate_local_slopes(self, elevation):
        """
        Calculate local slope angles.

        Args:
            elevation: 2D elevation grid

        Returns:
            np.ndarray: Slope angles in degrees
        """
        dy, dx = np.gradient(elevation, self.slope.resolution)
        slope_magnitude = np.sqrt(dx**2 + dy**2)
        slope_angle = np.degrees(np.arctan(slope_magnitude))
        return slope_angle

    def _update_velocities(self, slope_angle, dt):
        """
        Update flow velocities based on local slopes.

        Args:
            slope_angle: 2D array of slope angles in degrees
            dt: Time step in seconds
        """
        # Calculate velocity magnitude from physics
        velocity_magnitude = np.zeros_like(slope_angle)

        for i in range(self.slope.ny):
            for j in range(self.slope.nx):
                velocity_magnitude[i, j] = self.physics.calculate_flow_velocity(
                    slope_angle[i, j],
                    self.thickness[i, j]
                )

        # Calculate slope direction
        elevation = self.slope.elevation + self.thickness
        dy, dx = np.gradient(elevation, self.slope.resolution)

        # Velocity components (downslope direction)
        slope_magnitude = np.sqrt(dx**2 + dy**2)
        slope_magnitude[slope_magnitude == 0] = 1e-10  # Avoid division by zero

        self.velocity_x = velocity_magnitude * dx / slope_magnitude
        self.velocity_y = velocity_magnitude * dy / slope_magnitude

    def _transport_regolith(self, dt):
        """
        Transport regolith according to flow velocities.

        Args:
            dt: Time step in seconds
        """
        # Simple upwind scheme for transport
        thickness_new = self.thickness.copy()

        flux_x = self.thickness * self.velocity_x
        flux_y = self.thickness * self.velocity_y

        # Calculate divergence of flux
        dflux_x_dx = np.zeros_like(flux_x)
        dflux_y_dy = np.zeros_like(flux_y)

        dflux_x_dx[:, 1:-1] = (flux_x[:, 2:] - flux_x[:, :-2]) / (2 * self.slope.resolution)
        dflux_y_dy[1:-1, :] = (flux_y[2:, :] - flux_y[:-2, :]) / (2 * self.slope.resolution)

        # Update thickness (conservation of mass)
        thickness_new -= dt * (dflux_x_dx + dflux_y_dy)

        # Ensure non-negative thickness
        thickness_new[thickness_new < 0] = 0

        self.thickness = thickness_new

    def _apply_avalanching(self, slope_angle):
        """
        Apply discrete avalanching using cellular automata.

        This creates the characteristic granular flow patterns.

        Args:
            slope_angle: 2D array of slope angles in degrees

        Returns:
            bool: True if any avalanche occurred
        """
        avalanche_occurred = False
        thickness_change = np.zeros_like(self.thickness)

        # Find unstable cells
        unstable = slope_angle > self.physics.angle_of_repose

        # Apply avalanching rules
        for i in range(1, self.slope.ny - 1):
            for j in range(1, self.slope.nx - 1):
                if unstable[i, j] and self.thickness[i, j] > 0.01:
                    # Distribute material to neighbors (downslope)
                    elevation = self.slope.elevation + self.thickness

                    # Find lowest neighbors
                    neighbors = [
                        (i-1, j), (i+1, j), (i, j-1), (i, j+1),
                        (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)
                    ]

                    current_elev = elevation[i, j]
                    lower_neighbors = []

                    for ni, nj in neighbors:
                        if elevation[ni, nj] < current_elev:
                            lower_neighbors.append((ni, nj, current_elev - elevation[ni, nj]))

                    if lower_neighbors:
                        # Distribute material proportional to elevation difference
                        total_diff = sum(diff for _, _, diff in lower_neighbors)
                        transfer_amount = self.thickness[i, j] * 0.1  # Transfer 10% per avalanche

                        thickness_change[i, j] -= transfer_amount

                        for ni, nj, diff in lower_neighbors:
                            fraction = diff / total_diff
                            thickness_change[ni, nj] += transfer_amount * fraction

                        avalanche_occurred = True
                        self.flow_count[i, j] += 1

        self.thickness += thickness_change
        return avalanche_occurred

    def _update_deformation(self, slope_angle):
        """
        Track cumulative deformation to identify elephant hide patterns.

        Args:
            slope_angle: 2D array of slope angles in degrees
        """
        # Deformation is higher where flow occurs
        velocity_magnitude = np.sqrt(self.velocity_x**2 + self.velocity_y**2)
        deformation_rate = velocity_magnitude * slope_angle / 100.0

        self.cumulative_deformation += deformation_rate * self.timestep

    def get_elephant_hide_texture(self):
        """
        Extract elephant hide texture pattern from simulation.

        The texture is characterized by:
        - Alternating ridges and troughs
        - Curved, anastomosing patterns
        - Oriented downslope

        Returns:
            np.ndarray: Texture intensity map
        """
        # Combine flow count and cumulative deformation
        texture = (
            0.6 * self.cumulative_deformation / (np.max(self.cumulative_deformation) + 1e-10) +
            0.4 * self.flow_count / (np.max(self.flow_count) + 1e-10)
        )

        # Normalize
        texture = (texture - np.min(texture)) / (np.max(texture) - np.min(texture) + 1e-10)

        return texture

    def reset(self, initial_thickness=1.0):
        """
        Reset simulation to initial state.

        Args:
            initial_thickness: Initial regolith thickness in meters
        """
        self.thickness = np.ones((self.slope.ny, self.slope.nx)) * initial_thickness
        self.velocity_x = np.zeros((self.slope.ny, self.slope.nx))
        self.velocity_y = np.zeros((self.slope.ny, self.slope.nx))
        self.cumulative_deformation = np.zeros((self.slope.ny, self.slope.nx))
        self.flow_count = np.zeros((self.slope.ny, self.slope.nx))
        self.time = 0.0

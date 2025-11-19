"""
Geological timescale simulation for elephant hide texture formation.

This module implements simulation of regolith creep over millions of years,
incorporating thermal cycling and seismic perturbations as the primary
driving forces.
"""

import numpy as np
from .physics import RegolithPhysics
from .slope import SlopeGeometry
from .thermal import LunarThermalCycle, ThermalCreepSimulator
from .seismic import MoonquakeSimulator, SeismicPerturbation, RandomPerturbation


class GeologicalRegolithSimulation:
    """
    Simulate elephant hide texture formation over geological timescales.

    This simulation models the slow, cumulative downslope creep driven by:
    1. Thermal cycling (lunar day/night temperature variations)
    2. Seismic shaking (moonquakes and impact-induced ground motion)

    The texture develops over millions of years on slopes >8°, with
    maximum development on 15-25° slopes.
    """

    def __init__(self, slope_geometry, physics=None, thermal_cycle=None,
                 moonquake_sim=None, initial_thickness=1.0):
        """
        Initialize geological simulation.

        Args:
            slope_geometry: SlopeGeometry object
            physics: RegolithPhysics object
            thermal_cycle: LunarThermalCycle object
            moonquake_sim: MoonquakeSimulator object
            initial_thickness: Initial regolith thickness (meters)
        """
        self.slope = slope_geometry
        self.physics = physics or RegolithPhysics()
        self.thermal_cycle = thermal_cycle or LunarThermalCycle()
        self.moonquake_sim = moonquake_sim or MoonquakeSimulator()

        # Initialize thermal creep simulator
        self.thermal_creep = ThermalCreepSimulator(self.thermal_cycle, self.physics)

        # Initialize seismic perturbation
        self.seismic_pert = SeismicPerturbation(self.moonquake_sim)

        # Random perturbations (micrometeorites, etc.)
        self.random_pert = RandomPerturbation()

        # Simulation grids
        self.thickness = np.ones((self.slope.ny, self.slope.nx)) * initial_thickness
        self.cumulative_displacement = np.zeros((self.slope.ny, self.slope.nx))
        self.texture_intensity = np.zeros((self.slope.ny, self.slope.nx))

        # Track number of thermal cycles and seismic events
        self.num_thermal_cycles = 0
        self.num_seismic_events = 0

        # Time tracking (in years)
        self.time_years = 0.0

    def simulate_fresh_crater(self, show_initial=True):
        """
        Simulate a fresh impact crater with smooth regolith.

        This represents the initial state before elephant hide develops.

        Returns:
            dict: Initial state information
        """
        # Fresh crater has smooth regolith
        self.cumulative_displacement[:, :] = 0.0
        self.texture_intensity[:, :] = 0.0

        # Calculate slope angles
        slope_angles = self.slope.get_slope_angle()

        # Potential for texture formation (based on slope)
        texture_potential = self.physics.get_texture_formation_intensity(slope_angles)

        if show_initial:
            print("Fresh Crater Initial State:")
            print(f"  Regolith thickness: {self.thickness.mean():.2f} m")
            print(f"  Slopes >8°: {np.sum(slope_angles > 8)}/{slope_angles.size} cells")
            print(f"  Optimal slopes (15-25°): {np.sum((slope_angles >= 15) & (slope_angles <= 25))}/{slope_angles.size} cells")
            print(f"  Mean texture potential: {texture_potential.mean():.3f}")

        return {
            'slope_angles': slope_angles,
            'texture_potential': texture_potential,
            'thickness': self.thickness.copy()
        }

    def advance_geological_time(self, duration_years, progress_interval_years=None):
        """
        Advance simulation over geological timescale.

        Args:
            duration_years: Duration in years (e.g., 1e6 for 1 million years)
            progress_interval_years: Interval for progress reporting

        Returns:
            dict: Final state and history
        """
        if progress_interval_years is None:
            progress_interval_years = duration_years / 10

        print(f"\nSimulating {duration_years:.2e} years of evolution...")
        print(f"Thermal cycles: ~{duration_years * 365.25 / 29.5:.0f}")
        print(f"Expected moonquakes: ~{self.moonquake_sim.DEEP_QUAKE_RATE * duration_years:.0f}\n")

        # Calculate slope angles (static for this simulation)
        slope_angles = self.slope.get_slope_angle()

        # Get texture formation potential
        texture_potential = self.physics.get_texture_formation_intensity(slope_angles)

        # === THERMAL CYCLING CONTRIBUTION ===
        # Simulate thermal creep over time
        thermal_result = self.thermal_creep.simulate_texture_development(
            slope_angles, duration_years
        )

        # Update cumulative displacement from thermal creep
        self.cumulative_displacement += thermal_result['displacement']
        self.num_thermal_cycles = thermal_result['num_cycles']

        # === SEISMIC CONTRIBUTION ===
        # Generate moonquake sequence
        quakes = self.moonquake_sim.generate_quake_sequence(duration_years)
        self.num_seismic_events = len(quakes['times'])

        # Calculate cumulative seismic effects
        seismic_displacement = self._calculate_seismic_displacement(
            quakes, slope_angles, duration_years
        )

        self.cumulative_displacement += seismic_displacement

        # === TEXTURE FORMATION ===
        # Texture intensity is proportional to displacement and slope potential
        displacement_normalized = self.cumulative_displacement / \
                                 (np.max(self.cumulative_displacement) + 1e-10)

        self.texture_intensity = texture_potential * displacement_normalized

        # Add stochastic variations (micrometeorites, local variations)
        random_variation = self.random_pert.generate_field(
            self.texture_intensity.shape,
            intensity=0.1,
            correlation_length=3.0
        )
        self.texture_intensity += random_variation
        self.texture_intensity = np.clip(self.texture_intensity, 0, 1)

        # Update time
        self.time_years = duration_years

        print(f"\nSimulation Complete!")
        print(f"  Total thermal cycles: {self.num_thermal_cycles:.0f}")
        print(f"  Total seismic events: {self.num_seismic_events}")
        print(f"  Max displacement: {np.max(self.cumulative_displacement):.3e} m")
        print(f"  Mean texture intensity: {self.texture_intensity.mean():.3f}")
        print(f"  Cells with texture >0.5: {np.sum(self.texture_intensity > 0.5)}/{self.texture_intensity.size}")

        return {
            'time_years': self.time_years,
            'num_thermal_cycles': self.num_thermal_cycles,
            'num_seismic_events': self.num_seismic_events,
            'displacement': self.cumulative_displacement,
            'texture_intensity': self.texture_intensity,
            'slope_angles': slope_angles,
            'texture_potential': texture_potential
        }

    def _calculate_seismic_displacement(self, quakes, slope_angles, duration_years):
        """
        Calculate cumulative displacement from seismic events.

        Args:
            quakes: Dictionary of quake parameters
            slope_angles: 2D array of slope angles
            duration_years: Duration in years

        Returns:
            np.ndarray: Cumulative seismic displacement
        """
        seismic_displacement = np.zeros_like(slope_angles)

        # Process each quake
        for i, (time, mag, qtype, depth) in enumerate(zip(
            quakes['times'], quakes['magnitudes'],
            quakes['types'], quakes['depths']
        )):
            # Random epicenter location
            epicenter_x = np.random.uniform(0, self.slope.width)
            epicenter_y = np.random.uniform(0, self.slope.height)

            # Calculate perturbation field
            perturbation = self.seismic_pert.apply_to_grid(
                slope_angles,
                epicenter_x,
                epicenter_y,
                mag,
                depth,
                self.slope.resolution
            )

            # Displacement from this event (proportional to perturbation strength)
            # Stronger shaking on steeper slopes causes more displacement
            event_displacement = perturbation * slope_angles / 100.0 * 0.01

            seismic_displacement += event_displacement

        return seismic_displacement

    def get_elephant_hide_texture(self):
        """
        Get the elephant hide texture pattern.

        Returns:
            np.ndarray: Texture intensity (0-1 scale)
        """
        return self.texture_intensity

    def compare_fresh_vs_aged(self):
        """
        Compare fresh crater (smooth) vs. aged crater (textured).

        Returns:
            dict: Comparison metrics
        """
        slope_angles = self.slope.get_slope_angle()

        # Fresh crater: smooth, no texture
        fresh_texture = np.zeros_like(self.texture_intensity)

        # Aged crater: developed texture
        aged_texture = self.texture_intensity

        # Calculate metrics
        comparison = {
            'fresh_mean_texture': fresh_texture.mean(),
            'fresh_max_texture': fresh_texture.max(),
            'aged_mean_texture': aged_texture.mean(),
            'aged_max_texture': aged_texture.max(),
            'texture_development': aged_texture.mean() / (fresh_texture.mean() + 0.01),
            'slopes_with_texture': np.sum(aged_texture > 0.3),
            'total_displacement': self.cumulative_displacement.sum(),
        }

        return comparison


class AcceleratedSimulation(GeologicalRegolithSimulation):
    """
    Accelerated simulation for testing and demonstration.

    Uses time compression to simulate geological processes in reasonable
    computation time while preserving the essential physics.
    """

    def __init__(self, *args, time_acceleration=1e6, **kwargs):
        """
        Initialize accelerated simulation.

        Args:
            time_acceleration: Factor to accelerate time (default: 1 million)
            *args, **kwargs: Passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.time_acceleration = time_acceleration

    def run_accelerated(self, simulation_duration_seconds=1000):
        """
        Run accelerated simulation.

        Args:
            simulation_duration_seconds: Real-time duration in seconds

        Returns:
            dict: Simulation results
        """
        # Convert simulation time to geological time
        geological_years = simulation_duration_seconds * self.time_acceleration / (365.25 * 24 * 3600)

        print(f"Running accelerated simulation:")
        print(f"  Simulation duration: {simulation_duration_seconds} s")
        print(f"  Geological time: {geological_years:.2e} years")
        print(f"  Time acceleration: {self.time_acceleration:.2e}x\n")

        # Run geological simulation
        result = self.advance_geological_time(geological_years)

        return result

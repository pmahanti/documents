#!/usr/bin/env python3
"""
Lunar Impact Cratering Simulation
==================================

Simulates 3D morphology progression and ejecta dynamics for lunar impact craters
using analytical scaling laws and ballistic ejecta models.

Crater size range: 100m - 500m (simple craters)
Physics: Strength-gravity regime transition, ballistic ejecta motion

Author: Generated for lunar impact cratering research
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ProjectileParameters:
    """Projectile parameters for impact simulation."""
    diameter: float  # meters
    velocity: float  # m/s (15-25 km/s typical)
    angle: float  # degrees from horizontal (45-90°)
    density: float  # kg/m³ (2500-3500 rocky, 7800 iron)
    material_type: str  # 'rocky' or 'metallic'

    @property
    def radius(self) -> float:
        return self.diameter / 2

    @property
    def mass(self) -> float:
        return (4/3) * np.pi * self.radius**3 * self.density

    @property
    def kinetic_energy(self) -> float:
        """Kinetic energy in Joules."""
        return 0.5 * self.mass * self.velocity**2


@dataclass
class TargetParameters:
    """Lunar surface target parameters."""
    regolith_density: float = 1650  # kg/m³ (loose regolith)
    rock_density: float = 3000  # kg/m³ (consolidated basalt/anorthosite)
    regolith_thickness: float = 5.0  # meters
    porosity: float = 0.45  # 40-50% for regolith
    cohesion: float = 1e4  # Pa (effective strength ~10 kPa for loose regolith)
    friction_angle: float = 35  # degrees
    gravity: float = 1.62  # m/s² (lunar gravity)

    @property
    def effective_density(self) -> float:
        """Effective density considering porosity."""
        return self.regolith_density * (1 - self.porosity) + self.rock_density * self.porosity


class CraterScalingLaws:
    """
    Pi-group scaling laws for crater formation.

    Based on Holsapple & Schmidt (1982, 1987) and subsequent work.
    Handles strength-gravity regime transition for simple craters.
    """

    def __init__(self, target: TargetParameters):
        self.target = target

        # Scaling law coefficients (calibrated for lunar conditions)
        # Based on Holsapple (1993), Collins et al. (2005), and Pike (1977)
        # Empirical calibration: 5m @ 20km/s → ~150m crater
        self.K1 = 0.94  # Strength regime coefficient
        self.K2 = 1.18  # Gravity regime coefficient
        self.mu = 0.41  # Velocity exponent (strength) - Holsapple (1993)
        self.nu = 0.41  # Velocity exponent (gravity) - Holsapple (1993)

    def pi_2_gravity(self, projectile: ProjectileParameters) -> float:
        """
        Pi-2: Gravity-scaled size parameter.
        π₂ = ga/v² where g=gravity, a=projectile radius, v=velocity
        """
        g = self.target.gravity
        a = projectile.radius
        v = projectile.velocity
        return (g * a) / (v**2)

    def pi_3_strength(self, projectile: ProjectileParameters) -> float:
        """
        Pi-3: Strength parameter.
        π₃ = Y/(ρv²) where Y=target strength, ρ=target density, v=velocity
        """
        Y = self.target.cohesion
        rho = self.target.effective_density
        v = projectile.velocity
        return Y / (rho * v**2)

    def pi_4_density_ratio(self, projectile: ProjectileParameters) -> float:
        """
        Pi-4: Density ratio.
        π₄ = ρₚ/ρₜ (projectile density / target density)
        """
        return projectile.density / self.target.effective_density

    def transient_crater_diameter_strength(self, projectile: ProjectileParameters) -> float:
        """
        Crater diameter in strength regime.
        D_tc = K1 * L * (ρₚ/ρₜ)^(1/3) * (ρₚv²/Y)^μ
        where L is projectile diameter
        """
        L = projectile.diameter  # projectile diameter
        pi4 = self.pi_4_density_ratio(projectile)
        pi3 = self.pi_3_strength(projectile)

        # Angle correction factor (sin²θ for oblique impacts, but saturates at vertical)
        angle_rad = np.radians(projectile.angle)
        f_angle = (np.sin(angle_rad))**(2/3)  # Shallower dependence

        # Transient crater diameter in strength regime
        D_strength = self.K1 * L * (pi4**(1/3)) * (1/pi3)**self.mu * f_angle
        return D_strength

    def transient_crater_diameter_gravity(self, projectile: ProjectileParameters) -> float:
        """
        Crater diameter in gravity regime.
        D_tc = K2 * L * (ρₚ/ρₜ)^(1/3) * (v²/gL)^ν
        """
        L = projectile.diameter
        pi4 = self.pi_4_density_ratio(projectile)
        pi2 = self.pi_2_gravity(projectile)

        # Angle correction factor
        angle_rad = np.radians(projectile.angle)
        f_angle = (np.sin(angle_rad))**(2/3)

        D_gravity = self.K2 * L * (pi4**(1/3)) * (1/pi2)**self.nu * f_angle
        return D_gravity

    def transient_crater_diameter(self, projectile: ProjectileParameters) -> float:
        """
        Simplified crater scaling based on empirical relations.
        Uses Collins et al. (2005) and Pierazzo/Melosh scaling.
        """
        L = projectile.diameter
        v = projectile.velocity
        rho_p = projectile.density
        rho_t = self.target.effective_density
        g = self.target.gravity
        Y = self.target.cohesion

        # Angle correction
        angle_rad = np.radians(projectile.angle)
        f_angle = (np.sin(angle_rad))**(1/3)

        # Simplified scaling (dimensionally correct):
        # D ~ L * (rho_p/rho_t)^(1/3) * (v^2 / (g*L + Y/rho_t))^0.3

        # Effective gravity term accounting for strength
        g_eff = g + Y / (rho_t * L)

        # Crater/projectile size ratio
        scaling_factor = (rho_p / rho_t)**(1/3) * (v**2 / (g_eff * L))**0.3

        D_transient = 1.25 * L * scaling_factor * f_angle

        return D_transient

    def final_crater_diameter(self, projectile: ProjectileParameters) -> float:
        """
        Final crater diameter after rim collapse (simple craters only).
        D_final ≈ 1.18-1.25 * D_transient for simple craters on Moon (Melosh 1989).
        """
        D_trans = self.transient_crater_diameter(projectile)
        return 1.2 * D_trans  # Conservative estimate for simple craters

    def crater_depth(self, projectile: ProjectileParameters) -> float:
        """
        Crater depth using depth/diameter ratio.
        d/D ≈ 0.196 for fresh simple craters (Pike 1977).
        """
        D = self.final_crater_diameter(projectile)
        return 0.196 * D

    def excavation_depth(self, projectile: ProjectileParameters) -> float:
        """
        Maximum excavation depth (deepest material excavated).
        d_ex ≈ 0.1 * D_transient
        """
        D_trans = self.transient_crater_diameter(projectile)
        return 0.1 * D_trans


class EjectaModel:
    """
    Ballistic ejecta model using Maxwell Z-model and streamline approach.

    Calculates ejecta velocities, trajectories, and blanket distribution.
    """

    def __init__(self, scaling: CraterScalingLaws, projectile: ProjectileParameters):
        self.scaling = scaling
        self.projectile = projectile
        self.target = scaling.target

        # Crater dimensions
        self.D_crater = scaling.final_crater_diameter(projectile)
        self.d_crater = scaling.crater_depth(projectile)
        self.R_crater = self.D_crater / 2

    def excavation_velocity(self, r: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate excavation velocity field using Z-model (Melosh 1989).

        V(r,z) = V₀ * (R/r)^Z * exp(-z/H)

        Parameters:
        -----------
        r : radial distance from impact point (m)
        z : depth below original surface (m)

        Returns:
        --------
        v_r : radial velocity component (m/s)
        v_z : vertical velocity component (m/s)
        """
        # Z-model parameters (Melosh 1989, Impact Cratering)
        Z = 2.7  # Exponent for velocity decay (2.5-3.0 typical)
        H = self.d_crater / 3  # Characteristic depth scale

        # Reference velocity at crater rim (empirical scaling from Melosh)
        # V_rim ~ sqrt(g * R) for gravity regime
        # V_rim ~ sqrt(Y/ρ) for strength regime
        # Use hybrid: V_rim ~ 0.5 * sqrt(g * D) for simple craters
        V0 = 0.5 * np.sqrt(self.target.gravity * self.D_crater)

        # Avoid division by zero and unrealistic velocities near center
        # Particles within 10% of crater radius have capped velocities
        r_min = 0.1 * self.R_crater
        r_safe = np.maximum(r, r_min)

        # Radial velocity component (Z-model)
        v_r = V0 * (self.R_crater / r_safe)**Z * np.exp(-z / H)

        # Cap maximum velocity at 10× V0 (physically realistic)
        v_r = np.minimum(v_r, 10 * V0)

        # Ejection angle: typically 45° at rim, shallower for distant ejecta
        # θ ≈ 45° - 10° * (r/R - 1) for r near R
        theta_ej = np.radians(45) - 0.2 * (r_safe / self.R_crater - 1)
        theta_ej = np.clip(theta_ej, np.radians(30), np.radians(60))

        # Vertical velocity from ejection angle
        v_z = v_r * np.tan(theta_ej)

        return v_r, v_z

    def ejecta_trajectories(self, n_particles: int = 1000,
                           time_steps: int = 100) -> dict:
        """
        Calculate ballistic trajectories for ejecta particles.

        Parameters:
        -----------
        n_particles : number of ejecta particles to track
        time_steps : number of time steps for trajectory integration

        Returns:
        --------
        trajectories : dict with particle positions over time
        """
        # Initialize particle positions (random within excavation zone)
        r0 = np.random.uniform(0, self.R_crater, n_particles)
        theta0 = np.random.uniform(0, 2*np.pi, n_particles)
        z0 = np.random.uniform(0, self.d_crater, n_particles)

        # Convert to Cartesian
        x0 = r0 * np.cos(theta0)
        y0 = r0 * np.sin(theta0)

        # Calculate initial velocities
        v_r, v_z = self.excavation_velocity(r0, z0)

        # Convert to Cartesian velocities
        vx0 = v_r * np.cos(theta0)
        vy0 = v_r * np.sin(theta0)
        vz0 = v_z

        # Time array (excavation phase ~ 1-10 seconds for 100-500m craters)
        t_ballistic = 2 * np.max(vz0) / self.target.gravity  # Ballistic time
        t_max = min(t_ballistic, 300.0)  # Cap at 5 minutes (reasonable for Moon)
        t_max = max(t_max, 10.0)  # Minimum 10 seconds
        t = np.linspace(0, t_max, time_steps)

        # Initialize trajectory arrays
        x = np.zeros((n_particles, time_steps))
        y = np.zeros((n_particles, time_steps))
        z = np.zeros((n_particles, time_steps))

        # Ballistic integration (no atmosphere on Moon)
        for i in range(time_steps):
            x[:, i] = x0 + vx0 * t[i]
            y[:, i] = y0 + vy0 * t[i]
            z[:, i] = z0 + vz0 * t[i] - 0.5 * self.target.gravity * t[i]**2

            # Particles hit ground when z <= 0
            z[:, i] = np.maximum(z[:, i], 0)

        return {
            'x': x, 'y': y, 'z': z,
            'time': t,
            'n_particles': n_particles,
            'landing_range': np.sqrt(x[:, -1]**2 + y[:, -1]**2)
        }

    def ejecta_blanket_thickness(self, r: np.ndarray) -> np.ndarray:
        """
        Ejecta blanket thickness as function of distance from crater rim.

        T(r) = T₀ * (R/r)^(-3) for r > R

        Parameters:
        -----------
        r : radial distance from crater center (m)

        Returns:
        --------
        thickness : ejecta thickness (m)
        """
        R = self.R_crater

        # Ejecta thickness at rim (empirical)
        T0 = 0.04 * R  # ~4% of crater radius

        # Power law decay
        thickness = np.zeros_like(r)
        mask = r > R
        thickness[mask] = T0 * (R / r[mask])**3

        return thickness


class CraterMorphology3D:
    """
    Generate 3D crater morphology using excavation flow field.
    """

    def __init__(self, scaling: CraterScalingLaws, projectile: ProjectileParameters):
        self.scaling = scaling
        self.projectile = projectile
        self.D = scaling.final_crater_diameter(projectile)
        self.d = scaling.crater_depth(projectile)
        self.R = self.D / 2

    def crater_profile(self, r: np.ndarray, time_fraction: float = 1.0) -> np.ndarray:
        """
        Generate crater elevation profile.

        Parameters:
        -----------
        r : radial distance from center (m)
        time_fraction : 0 to 1, representing excavation progress

        Returns:
        --------
        z : elevation (negative is below surface) (m)
        """
        # Fresh crater shape: parabolic bowl + raised rim
        z = np.zeros_like(r)

        # Interior (parabolic bowl)
        interior_mask = r <= self.R
        z[interior_mask] = -self.d * (1 - (r[interior_mask] / self.R)**2)

        # Rim (exponential decay)
        h_rim = 0.04 * self.D  # Rim height ~ 4% of diameter
        rim_mask = (r > self.R) & (r <= 1.5 * self.R)
        z[rim_mask] = h_rim * np.exp(-5 * (r[rim_mask] / self.R - 1))

        # Scale by time fraction (excavation progress)
        z = z * time_fraction

        return z

    def generate_3d_surface(self, resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 3D surface mesh for crater.

        Parameters:
        -----------
        resolution : grid resolution

        Returns:
        --------
        X, Y, Z : 3D coordinate arrays
        """
        # Create radial grid
        extent = 2 * self.D
        x = np.linspace(-extent, extent, resolution)
        y = np.linspace(-extent, extent, resolution)
        X, Y = np.meshgrid(x, y)

        # Calculate radius
        R_grid = np.sqrt(X**2 + Y**2)

        # Generate elevation
        Z = self.crater_profile(R_grid, time_fraction=1.0)

        return X, Y, Z


class ImpactSimulation:
    """
    Main impact crater simulation class coordinating all components.
    """

    def __init__(self, projectile: ProjectileParameters, target: TargetParameters):
        self.projectile = projectile
        self.target = target
        self.scaling = CraterScalingLaws(target)
        self.ejecta = EjectaModel(self.scaling, projectile)
        self.morphology = CraterMorphology3D(self.scaling, projectile)

        # Simulation results
        self.crater_diameter = self.scaling.final_crater_diameter(projectile)
        self.crater_depth = self.scaling.crater_depth(projectile)
        self.ejecta_trajectories_data = None

    def run(self, n_ejecta_particles: int = 1000):
        """
        Run the complete impact simulation.

        Parameters:
        -----------
        n_ejecta_particles : number of ejecta particles to simulate
        """
        print("=" * 60)
        print("LUNAR IMPACT CRATER SIMULATION")
        print("=" * 60)
        print(f"\nProjectile Parameters:")
        print(f"  Diameter: {self.projectile.diameter:.1f} m")
        print(f"  Velocity: {self.projectile.velocity/1000:.1f} km/s")
        print(f"  Angle: {self.projectile.angle:.1f}°")
        print(f"  Density: {self.projectile.density:.0f} kg/m³")
        print(f"  Material: {self.projectile.material_type}")
        print(f"  Kinetic Energy: {self.projectile.kinetic_energy:.2e} J")

        print(f"\nTarget Parameters:")
        print(f"  Surface: Lunar regolith/rock")
        print(f"  Gravity: {self.target.gravity:.2f} m/s²")
        print(f"  Density: {self.target.effective_density:.0f} kg/m³")
        print(f"  Porosity: {self.target.porosity:.1%}")

        # Regime analysis
        pi2 = self.scaling.pi_2_gravity(self.projectile)
        pi3 = self.scaling.pi_3_strength(self.projectile)

        print(f"\nRegime Analysis:")
        print(f"  π₂ (gravity): {pi2:.4e}")
        print(f"  π₃ (strength): {pi3:.4e}")

        if pi3 < pi2:
            regime = "Strength-dominated"
        elif pi3 > 10 * pi2:
            regime = "Gravity-dominated"
        else:
            regime = "Transitional (strength-gravity)"
        print(f"  Dominant regime: {regime}")

        print(f"\nCrater Results:")
        print(f"  Final diameter: {self.crater_diameter:.1f} m")
        print(f"  Depth: {self.crater_depth:.1f} m")
        print(f"  Depth/Diameter: {self.crater_depth/self.crater_diameter:.3f}")

        # Calculate ejecta trajectories
        print(f"\nCalculating ejecta trajectories ({n_ejecta_particles} particles)...")
        self.ejecta_trajectories_data = self.ejecta.ejecta_trajectories(
            n_particles=n_ejecta_particles,
            time_steps=100
        )

        max_range = np.max(self.ejecta_trajectories_data['landing_range'])
        print(f"  Maximum ejecta range: {max_range:.1f} m ({max_range/self.crater_diameter:.1f}× crater diameter)")

        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)

    def plot_2d_summary(self, output_file: str = 'crater_simulation_2d.png'):
        """
        Generate 2D summary plot with crater profile and ejecta.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Lunar Impact Simulation: {self.projectile.diameter}m Projectile → {self.crater_diameter:.0f}m Crater',
                     fontsize=14, fontweight='bold')

        # 1. Crater profile
        ax = axes[0, 0]
        r = np.linspace(0, 2*self.morphology.D, 500)
        z = self.morphology.crater_profile(r)
        ax.plot(r, z, 'k-', linewidth=2)
        ax.axhline(0, color='brown', linestyle='--', alpha=0.5, label='Original surface')
        ax.fill_between(r, z, 0, where=(z<0), alpha=0.3, color='tan', label='Excavation')
        ax.fill_between(r, z, 0, where=(z>0), alpha=0.3, color='gray', label='Rim')
        ax.set_xlabel('Radial Distance (m)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Crater Cross-Section Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')

        # 2. Ejecta range distribution
        ax = axes[0, 1]
        ranges = self.ejecta_trajectories_data['landing_range']
        ax.hist(ranges, bins=50, color='indianred', alpha=0.7, edgecolor='black')
        ax.axvline(self.morphology.R, color='blue', linestyle='--', label='Crater rim')
        ax.set_xlabel('Landing Distance from Center (m)')
        ax.set_ylabel('Number of Particles')
        ax.set_title('Ejecta Landing Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Ejecta blanket thickness
        ax = axes[1, 0]
        r = np.linspace(self.morphology.R, 10*self.morphology.R, 500)
        thickness = self.ejecta.ejecta_blanket_thickness(r)
        ax.semilogy(r/self.morphology.R, thickness, 'g-', linewidth=2)
        ax.set_xlabel('Distance from Center (crater radii)')
        ax.set_ylabel('Ejecta Thickness (m)')
        ax.set_title('Ejecta Blanket Thickness (r⁻³ decay)')
        ax.grid(True, alpha=0.3)
        ax.axvline(1, color='blue', linestyle='--', alpha=0.5, label='Crater rim')
        ax.legend()

        # 4. Parameters summary
        ax = axes[1, 1]
        ax.axis('off')

        params_text = f"""
SIMULATION PARAMETERS

Projectile:
  • Diameter: {self.projectile.diameter:.1f} m
  • Velocity: {self.projectile.velocity/1000:.1f} km/s
  • Angle: {self.projectile.angle:.1f}°
  • Density: {self.projectile.density:.0f} kg/m³
  • KE: {self.projectile.kinetic_energy:.2e} J

Crater:
  • Final diameter: {self.crater_diameter:.1f} m
  • Depth: {self.crater_depth:.1f} m
  • d/D ratio: {self.crater_depth/self.crater_diameter:.3f}

Ejecta:
  • Particles: {self.ejecta_trajectories_data['n_particles']}
  • Max range: {np.max(ranges):.1f} m
  • Range/D: {np.max(ranges)/self.crater_diameter:.1f}×

Target:
  • Body: Moon (g = {self.target.gravity:.2f} m/s²)
  • Surface: Regolith/rock
  • Density: {self.target.effective_density:.0f} kg/m³
        """

        ax.text(0.1, 0.9, params_text, transform=ax.transAxes,
                fontfamily='monospace', fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ 2D summary saved: {output_file}")

        return fig


def main():
    """Example simulation."""

    # Create projectile (10m diameter rocky asteroid at 20 km/s)
    projectile = ProjectileParameters(
        diameter=10.0,  # meters
        velocity=20000,  # m/s (20 km/s)
        angle=90,  # vertical impact
        density=2800,  # kg/m³ (rocky)
        material_type='rocky'
    )

    # Create target (lunar surface)
    target = TargetParameters()

    # Run simulation
    sim = ImpactSimulation(projectile, target)
    sim.run(n_ejecta_particles=1000)

    # Generate plots
    sim.plot_2d_summary()


if __name__ == "__main__":
    main()

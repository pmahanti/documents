#!/usr/bin/env python3
"""
Impact Crater 3D Animation
===========================

Generates 2D and 3D animations of crater formation and ejecta motion.

Features:
- 3D crater morphology evolution
- Ballistic ejecta trajectories
- Side-view and top-view animations
- Quadchart summary animation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from lunar_impact_simulation import *
import argparse


class ImpactAnimator:
    """
    Creates animations of impact crater formation and ejecta dynamics.
    """

    def __init__(self, simulation: ImpactSimulation):
        self.sim = simulation
        self.morphology = simulation.morphology
        self.ejecta_data = simulation.ejecta_trajectories_data

    def animate_3d_crater_formation(self, output_file: str = 'crater_3d.gif',
                                   frames: int = 60, fps: int = 15):
        """
        Animate 3D crater excavation process.

        Parameters:
        -----------
        output_file : output filename (.gif or .mp4)
        frames : number of animation frames
        fps : frames per second
        """
        print(f"\nGenerating 3D crater formation animation...")
        print(f"  Frames: {frames}")
        print(f"  Output: {output_file}")

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Generate surface mesh
        X, Y, Z_final = self.morphology.generate_3d_surface(resolution=80)

        def update(frame):
            ax.clear()

            # Time fraction (0 to 1)
            t_frac = frame / (frames - 1)

            # Excavation progress (non-linear for realistic dynamics)
            excavation_progress = 1 - (1 - t_frac)**2

            # Generate crater at current time
            R_grid = np.sqrt(X**2 + Y**2)
            Z = self.morphology.crater_profile(R_grid, time_fraction=excavation_progress)

            # Plot surface
            surf = ax.plot_surface(X, Y, Z, cmap='terrain',
                                  linewidth=0, antialiased=True,
                                  vmin=-self.morphology.d*1.2,
                                  vmax=self.morphology.d*0.3,
                                  alpha=0.9)

            # Original surface plane
            if excavation_progress < 0.99:
                ax.plot_surface(X, Y, np.zeros_like(Z), alpha=0.1,
                               color='brown', linewidth=0)

            # Styling
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_zlabel('Elevation (m)', fontsize=10)

            title_text = f'3D Crater Formation\n'
            title_text += f'D = {self.morphology.D:.0f}m, d = {self.morphology.d:.0f}m\n'
            title_text += f'Excavation: {excavation_progress*100:.0f}%'
            ax.set_title(title_text, fontsize=12, fontweight='bold')

            # Set consistent view
            extent = 1.5 * self.morphology.D
            ax.set_xlim(-extent, extent)
            ax.set_ylim(-extent, extent)
            ax.set_zlim(-self.morphology.d*1.2, self.morphology.d*0.3)

            ax.view_init(elev=25, azim=45 + frame)  # Rotating view

            return surf,

        anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)

        # Save animation
        if output_file.endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(output_file, writer=writer)
            print(f"✓ 3D animation saved as GIF: {output_file}")
        else:
            print("✗ MP4 export requires ffmpeg (using GIF instead)")
            output_file = output_file.replace('.mp4', '.gif')
            writer = PillowWriter(fps=fps)
            anim.save(output_file, writer=writer)
            print(f"✓ 3D animation saved as GIF: {output_file}")

        plt.close()

    def animate_ejecta_2d(self, output_file: str = 'ejecta_2d.gif',
                         frames: int = 80, fps: int = 20):
        """
        Animate ejecta trajectories in 2D side view.

        Parameters:
        -----------
        output_file : output filename
        frames : number of frames
        fps : frames per second
        """
        print(f"\nGenerating 2D ejecta animation...")
        print(f"  Particles: {self.ejecta_data['n_particles']}")
        print(f"  Frames: {frames}")
        print(f"  Output: {output_file}")

        fig, ax = plt.subplots(figsize=(14, 8))

        # Get trajectory data
        x = self.ejecta_data['x']
        y = self.ejecta_data['y']
        z = self.ejecta_data['z']
        time = self.ejecta_data['time']

        # Use radial distance for 2D view
        r = np.sqrt(x**2 + y**2)

        # Subsample particles for clarity
        n_show = min(200, self.ejecta_data['n_particles'])
        indices = np.random.choice(self.ejecta_data['n_particles'], n_show, replace=False)

        def update(frame):
            ax.clear()

            # Current time index
            t_idx = int(frame * (len(time) - 1) / (frames - 1))

            # Plot crater profile
            r_crater = np.linspace(0, 2*self.morphology.D, 300)
            z_crater = self.morphology.crater_profile(r_crater)
            ax.fill_between(r_crater, z_crater, -self.morphology.d*1.5,
                           color='tan', alpha=0.5, label='Crater')
            ax.plot(r_crater, z_crater, 'k-', linewidth=2)

            # Plot ejecta particles
            r_current = r[indices, t_idx]
            z_current = z[indices, t_idx]

            # Only show airborne particles
            airborne = z_current > 0.1
            ax.scatter(r_current[airborne], z_current[airborne],
                      c='red', s=20, alpha=0.6, label='Airborne ejecta')

            # Landed particles
            landed = z_current <= 0.1
            if np.any(landed):
                ax.scatter(r_current[landed], z_current[landed],
                          c='brown', s=10, alpha=0.4, marker='x', label='Landed ejecta')

            # Styling
            ax.set_xlabel('Radial Distance (m)', fontsize=12)
            ax.set_ylabel('Height (m)', fontsize=12)
            ax.set_title(f'Ejecta Ballistic Trajectories (Side View)\nTime: {time[t_idx]:.2f} s',
                        fontsize=14, fontweight='bold')

            ax.set_xlim(0, np.max(r) * 1.1)
            ax.set_ylim(-self.morphology.d*1.2, np.max(z)*1.1)
            ax.axhline(0, color='brown', linestyle='--', alpha=0.3, linewidth=1)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            # Add info text
            info_text = f"Projectile: {self.sim.projectile.diameter:.1f}m @ {self.sim.projectile.velocity/1000:.1f} km/s\n"
            info_text += f"Crater: D={self.morphology.D:.0f}m, d={self.morphology.d:.0f}m"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)

        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
        print(f"✓ 2D ejecta animation saved: {output_file}")
        plt.close()

    def animate_ejecta_3d(self, output_file: str = 'ejecta_3d.gif',
                         frames: int = 80, fps: int = 20):
        """
        Animate ejecta trajectories in 3D.

        Parameters:
        -----------
        output_file : output filename
        frames : number of frames
        fps : frames per second
        """
        print(f"\nGenerating 3D ejecta animation...")
        print(f"  Particles: {self.ejecta_data['n_particles']}")
        print(f"  Frames: {frames}")

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Get data
        x = self.ejecta_data['x']
        y = self.ejecta_data['y']
        z = self.ejecta_data['z']
        time = self.ejecta_data['time']

        # Subsample
        n_show = min(300, self.ejecta_data['n_particles'])
        indices = np.random.choice(self.ejecta_data['n_particles'], n_show, replace=False)

        # Generate crater surface
        X_crater, Y_crater, Z_crater = self.morphology.generate_3d_surface(resolution=60)

        def update(frame):
            ax.clear()

            t_idx = int(frame * (len(time) - 1) / (frames - 1))

            # Plot crater
            ax.plot_surface(X_crater, Y_crater, Z_crater, cmap='terrain',
                          alpha=0.6, linewidth=0, antialiased=True,
                          vmin=-self.morphology.d*1.2,
                          vmax=self.morphology.d*0.3)

            # Plot ejecta
            x_current = x[indices, t_idx]
            y_current = y[indices, t_idx]
            z_current = z[indices, t_idx]

            airborne = z_current > 0.1
            ax.scatter(x_current[airborne], y_current[airborne], z_current[airborne],
                      c='red', s=15, alpha=0.7, label='Ejecta')

            # Styling
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Height (m)')
            ax.set_title(f'3D Ejecta Cloud\nTime: {time[t_idx]:.2f} s',
                        fontsize=14, fontweight='bold')

            extent = np.max(np.abs(x)) * 1.1
            ax.set_xlim(-extent, extent)
            ax.set_ylim(-extent, extent)
            ax.set_zlim(-self.morphology.d*1.2, np.max(z)*1.1)

            ax.view_init(elev=20, azim=frame*2)

        anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)

        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
        print(f"✓ 3D ejecta animation saved: {output_file}")
        plt.close()

    def animate_quadchart(self, output_file: str = 'impact_quadchart.gif',
                         frames: int = 100, fps: int = 15):
        """
        Create comprehensive quadchart animation showing:
        Q1: 3D crater formation
        Q2: 2D crater profile evolution
        Q3: Ejecta side view
        Q4: Ejecta top view (plan)

        Parameters:
        -----------
        output_file : output filename
        frames : number of frames
        fps : frames per second
        """
        print(f"\nGenerating quadchart animation...")
        print(f"  Frames: {frames}, FPS: {fps}")
        print(f"  Output: {output_file}")

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0], projection='3d')  # 3D crater
        ax2 = fig.add_subplot(gs[0, 1])  # 2D profile
        ax3 = fig.add_subplot(gs[1, 0])  # Ejecta side view
        ax4 = fig.add_subplot(gs[1, 1])  # Ejecta plan view

        # Pre-generate data
        X_crater, Y_crater, Z_final_crater = self.morphology.generate_3d_surface(resolution=60)
        R_crater_grid = np.sqrt(X_crater**2 + Y_crater**2)

        x = self.ejecta_data['x']
        y = self.ejecta_data['y']
        z = self.ejecta_data['z']
        time = self.ejecta_data['time']
        r = np.sqrt(x**2 + y**2)

        # Subsample ejecta
        n_show = min(200, self.ejecta_data['n_particles'])
        indices = np.random.choice(self.ejecta_data['n_particles'], n_show, replace=False)

        def update(frame):
            # Time fractions
            t_frac = frame / (frames - 1)
            excavation_progress = min(0.3 + t_frac * 0.7, 1.0)  # Start at 30%
            t_idx = int(frame * (len(time) - 1) / (frames - 1))

            # === Q1: 3D Crater ===
            ax1.clear()
            Z_current = self.morphology.crater_profile(R_crater_grid, time_fraction=excavation_progress)
            ax1.plot_surface(X_crater, Y_crater, Z_current, cmap='terrain',
                           linewidth=0, antialiased=True,
                           vmin=-self.morphology.d*1.2, vmax=self.morphology.d*0.3,
                           alpha=0.9)
            ax1.set_xlabel('X (m)', fontsize=8)
            ax1.set_ylabel('Y (m)', fontsize=8)
            ax1.set_zlabel('Z (m)', fontsize=8)
            ax1.set_title(f'Q1: 3D Crater Formation ({excavation_progress*100:.0f}%)', fontweight='bold')
            extent = 1.5 * self.morphology.D
            ax1.set_xlim(-extent, extent)
            ax1.set_ylim(-extent, extent)
            ax1.set_zlim(-self.morphology.d*1.2, self.morphology.d*0.3)
            ax1.view_init(elev=25, azim=45)

            # === Q2: 2D Profile ===
            ax2.clear()
            r_profile = np.linspace(0, 2*self.morphology.D, 300)
            z_profile = self.morphology.crater_profile(r_profile, time_fraction=excavation_progress)
            ax2.plot(r_profile, z_profile, 'k-', linewidth=2)
            ax2.fill_between(r_profile, z_profile, -self.morphology.d*1.5,
                           where=(z_profile<0), color='tan', alpha=0.5)
            ax2.fill_between(r_profile, z_profile, 0,
                           where=(z_profile>0), color='gray', alpha=0.5)
            ax2.axhline(0, color='brown', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Radial Distance (m)')
            ax2.set_ylabel('Elevation (m)')
            ax2.set_title('Q2: Crater Profile Evolution', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 2*self.morphology.D)
            ax2.set_ylim(-self.morphology.d*1.2, self.morphology.d*0.3)

            # === Q3: Ejecta Side View ===
            ax3.clear()
            # Crater profile
            ax3.fill_between(r_profile, z_profile, -self.morphology.d*1.5,
                           color='tan', alpha=0.3)
            ax3.plot(r_profile, z_profile, 'k-', linewidth=1.5)

            # Ejecta particles
            r_current = r[indices, t_idx]
            z_current = z[indices, t_idx]
            airborne = z_current > 0.1
            landed = z_current <= 0.1

            if np.any(airborne):
                ax3.scatter(r_current[airborne], z_current[airborne],
                          c='red', s=15, alpha=0.6)
            if np.any(landed):
                ax3.scatter(r_current[landed], np.zeros(np.sum(landed)),
                          c='brown', s=5, alpha=0.4, marker='x')

            ax3.axhline(0, color='brown', linestyle='--', alpha=0.3)
            ax3.set_xlabel('Radial Distance (m)')
            ax3.set_ylabel('Height (m)')
            ax3.set_title(f'Q3: Ejecta Ballistics (t={time[t_idx]:.2f}s)', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, np.max(r)*1.1)
            ax3.set_ylim(-self.morphology.d*0.5, np.max(z)*0.6)

            # === Q4: Ejecta Plan View ===
            ax4.clear()
            x_current = x[indices, t_idx]
            y_current = y[indices, t_idx]
            z_current = z[indices, t_idx]

            # Crater outline
            theta = np.linspace(0, 2*np.pi, 100)
            crater_x = self.morphology.R * np.cos(theta)
            crater_y = self.morphology.R * np.sin(theta)
            ax4.fill(crater_x, crater_y, color='tan', alpha=0.3, label='Crater')
            ax4.plot(crater_x, crater_y, 'k-', linewidth=1.5)

            # Ejecta
            airborne_plan = z_current > 0.1
            landed_plan = z_current <= 0.1

            if np.any(airborne_plan):
                ax4.scatter(x_current[airborne_plan], y_current[airborne_plan],
                          c='red', s=15, alpha=0.6, label='Airborne')
            if np.any(landed_plan):
                ax4.scatter(x_current[landed_plan], y_current[landed_plan],
                          c='brown', s=5, alpha=0.4, marker='x', label='Landed')

            ax4.set_xlabel('X (m)')
            ax4.set_ylabel('Y (m)')
            ax4.set_title('Q4: Ejecta Distribution (Plan View)', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_aspect('equal')
            extent_plan = np.max(r) * 1.1
            ax4.set_xlim(-extent_plan, extent_plan)
            ax4.set_ylim(-extent_plan, extent_plan)
            if frame == 0:
                ax4.legend(loc='upper right', fontsize=8)

            # Overall title
            fig.suptitle(f'Lunar Impact Simulation: {self.sim.projectile.diameter:.1f}m → {self.morphology.D:.0f}m Crater\n'
                        f'Velocity: {self.sim.projectile.velocity/1000:.1f} km/s | Time: {time[t_idx]:.2f}s',
                        fontsize=14, fontweight='bold')

        anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)

        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
        print(f"✓ Quadchart animation saved: {output_file}")
        plt.close()


def main():
    """Example animation generation."""

    parser = argparse.ArgumentParser(description='Generate impact crater animations')
    parser.add_argument('--diameter', type=float, default=10.0,
                       help='Projectile diameter in meters (default: 10)')
    parser.add_argument('--velocity', type=float, default=20.0,
                       help='Impact velocity in km/s (default: 20)')
    parser.add_argument('--angle', type=float, default=90.0,
                       help='Impact angle in degrees (default: 90)')
    parser.add_argument('--density', type=float, default=2800.0,
                       help='Projectile density in kg/m³ (default: 2800)')
    parser.add_argument('--particles', type=int, default=1000,
                       help='Number of ejecta particles (default: 1000)')
    parser.add_argument('--frames', type=int, default=80,
                       help='Number of animation frames (default: 80)')
    parser.add_argument('--fps', type=int, default=15,
                       help='Frames per second (default: 15)')
    parser.add_argument('--output-prefix', type=str, default='impact',
                       help='Output file prefix (default: impact)')
    parser.add_argument('--all', action='store_true',
                       help='Generate all animation types')

    args = parser.parse_args()

    # Create projectile
    projectile = ProjectileParameters(
        diameter=args.diameter,
        velocity=args.velocity * 1000,  # Convert km/s to m/s
        angle=args.angle,
        density=args.density,
        material_type='rocky'
    )

    # Create target
    target = TargetParameters()

    # Run simulation
    print("\n" + "="*60)
    print("IMPACT ANIMATION GENERATOR")
    print("="*60)

    sim = ImpactSimulation(projectile, target)
    sim.run(n_ejecta_particles=args.particles)

    # Create animator
    animator = ImpactAnimator(sim)

    # Generate animations
    if args.all:
        animator.animate_quadchart(f'{args.output_prefix}_quadchart.gif',
                                  frames=args.frames, fps=args.fps)
        animator.animate_3d_crater_formation(f'{args.output_prefix}_crater3d.gif',
                                            frames=args.frames, fps=args.fps)
        animator.animate_ejecta_2d(f'{args.output_prefix}_ejecta2d.gif',
                                  frames=args.frames, fps=args.fps)
        animator.animate_ejecta_3d(f'{args.output_prefix}_ejecta3d.gif',
                                  frames=args.frames, fps=args.fps)
    else:
        # Just quadchart by default
        animator.animate_quadchart(f'{args.output_prefix}_quadchart.gif',
                                  frames=args.frames, fps=args.fps)

    print("\n" + "="*60)
    print("ANIMATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

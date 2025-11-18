#!/usr/bin/env python3
"""
Crater Degradation Animation Generator

Creates animated quadchart visualization showing crater degradation over time:
- Quadrant 1 (Top Left): 3D crater topography
- Quadrant 2 (Top Right): 2D elevation profile
- Quadrant 3 (Bottom Left): Depth-to-diameter ratio evolution
- Quadrant 4 (Bottom Right): Chebyshev coefficient evolution

The animation shows degradation from 0.1 to 3.9 Ga for a specified crater diameter.

Author: Crater Analysis Toolkit
Date: 2025-11-18
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

# Import our modules
try:
    from topography_degradation_age import TopographyDegradationAgeEstimator
    from chebyshev_coefficients import extract_chebyshev_coefficients
    MODULES_AVAILABLE = True
except ImportError:
    print("Warning: Required modules not found. Run from repository root.")
    MODULES_AVAILABLE = False


class CraterDegradationAnimator:
    """Generate animated quadchart of crater degradation evolution."""

    def __init__(self, diameter_m, diffusivity=5.0, age_range=(0.1, 3.9), num_frames=100):
        """
        Initialize crater degradation animator.

        Parameters:
        -----------
        diameter_m : float
            Crater diameter in meters
        diffusivity : float
            Diffusivity coefficient in m²/Myr (default 5.0)
        age_range : tuple
            (min_age_ga, max_age_ga) age range in Ga (default 0.1 to 3.9)
        num_frames : int
            Number of frames in animation (default 100)
        """
        self.diameter_m = diameter_m
        self.diffusivity = diffusivity
        self.age_range = age_range
        self.num_frames = num_frames

        # Generate age timeline
        self.ages_ga = np.linspace(age_range[0], age_range[1], num_frames)
        self.ages_myr = self.ages_ga * 1000

        # Generate pristine profile
        self.generate_pristine_crater()

        # Precompute all degraded states
        self.compute_degradation_sequence()

        print(f"Initialized animation for crater D={diameter_m:.0f}m")
        print(f"  Age range: {age_range[0]:.1f} - {age_range[1]:.1f} Ga")
        print(f"  Frames: {num_frames}")

    def generate_pristine_crater(self):
        """Generate pristine crater elevation profile."""
        num_points = 300
        radius = self.diameter_m / 2.0

        # Profile extends from -1.5D to +1.5D
        self.distance = np.linspace(-1.5 * self.diameter_m, 1.5 * self.diameter_m, num_points)

        # Pristine depth/diameter ratio
        depth = 0.196 * self.diameter_m

        # Rim height
        rim_height = 0.04 * self.diameter_m

        # Generate elevation profile
        elevation = np.zeros(num_points)

        for i, r in enumerate(self.distance):
            r_norm = abs(r) / radius

            if r_norm <= 1.0:
                # Interior: parabolic bowl
                elevation[i] = -depth * (1 - r_norm**2)
            elif r_norm <= 1.3:
                # Rim: exponential decay
                elevation[i] = rim_height * np.exp(-5 * (r_norm - 1.0))
            else:
                # Far field
                elevation[i] = 0.0

        self.pristine_elevation = elevation

    def compute_degradation_sequence(self):
        """Precompute degraded profiles, d/D ratios, and Chebyshev coefficients."""
        from scipy.ndimage import gaussian_filter1d

        print("Precomputing degradation sequence...")

        self.degraded_profiles = []
        self.d_D_ratios = []
        self.chebyshev_matrices = []

        radius = self.diameter_m / 2.0

        for age_idx, (age_ga, age_myr) in enumerate(zip(self.ages_ga, self.ages_myr)):
            # Apply diffusion degradation
            sigma_meters = np.sqrt(2 * self.diffusivity * age_myr)
            dr = np.mean(np.diff(self.distance))
            sigma_pixels = sigma_meters / abs(dr)

            if sigma_pixels > 0:
                degraded = gaussian_filter1d(self.pristine_elevation, sigma_pixels, mode='nearest')
            else:
                degraded = self.pristine_elevation.copy()

            self.degraded_profiles.append(degraded)

            # Compute d/D ratio
            center_mask = np.abs(self.distance) < 0.3 * radius
            floor_elev = np.min(degraded[center_mask]) if np.sum(center_mask) > 0 else np.min(degraded)

            rim_mask = np.abs(np.abs(self.distance) - radius) < 0.1 * radius
            rim_elev = np.max(degraded[rim_mask]) if np.sum(rim_mask) > 0 else np.max(degraded)

            depth = rim_elev - floor_elev
            d_D = depth / self.diameter_m if self.diameter_m > 0 else 0
            self.d_D_ratios.append(d_D)

            # Extract Chebyshev coefficients
            # Create 8 profiles (azimuthally symmetric, so all are the same)
            profiles = []
            for angle in range(0, 360, 45):
                profiles.append({
                    'distance': self.distance,
                    'elevation': degraded,
                    'angle': angle
                })

            try:
                coef_matrix, analysis, metadata = extract_chebyshev_coefficients(
                    profiles, diameter=self.diameter_m, num_coefficients=17
                )
                mean_coeffs = analysis['mean_coefficients']
                self.chebyshev_matrices.append(mean_coeffs)
            except Exception as e:
                print(f"  Warning: Chebyshev failed at age {age_ga:.1f} Ga: {e}")
                self.chebyshev_matrices.append(np.zeros(17))

            if age_idx % 20 == 0:
                print(f"  Processed frame {age_idx}/{self.num_frames}")

        self.chebyshev_matrices = np.array(self.chebyshev_matrices)  # Shape: (num_frames, 17)
        print("✓ Degradation sequence computed")

    def create_3d_surface(self, elevation_profile):
        """
        Create 3D crater surface from 1D radial profile (axisymmetric).

        Parameters:
        -----------
        elevation_profile : ndarray
            1D elevation profile

        Returns:
        --------
        X, Y, Z : ndarray
            3D mesh arrays for surface plotting
        """
        # Create polar grid
        num_radial = len(self.distance)
        num_angular = 72  # Number of angular points

        # Radial and angular coordinates
        r_vals = np.linspace(self.distance.min(), self.distance.max(), num_radial)
        theta_vals = np.linspace(0, 2*np.pi, num_angular)

        # Create meshgrid
        R, Theta = np.meshgrid(r_vals, theta_vals)

        # Convert to Cartesian
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)

        # Interpolate elevation onto radial grid
        # For axisymmetric crater, elevation only depends on radial distance
        r_abs = np.abs(R)

        # Interpolate elevation
        from scipy.interpolate import interp1d
        interp_func = interp1d(np.abs(self.distance), elevation_profile,
                              kind='linear', fill_value='extrapolate')
        Z = interp_func(r_abs)

        return X, Y, Z

    def normalize_coefficients(self):
        """Normalize Chebyshev coefficients for plotting."""
        # Find global max for each coefficient across all time steps
        coeff_max = np.max(np.abs(self.chebyshev_matrices), axis=0)
        coeff_max[coeff_max < 1e-10] = 1.0  # Avoid division by zero

        # Normalize
        self.normalized_coeffs = self.chebyshev_matrices / coeff_max[np.newaxis, :]

    def create_animation(self, output_file='crater_degradation_animation.mp4',
                        fps=10, dpi=150):
        """
        Create animated quadchart visualization.

        Parameters:
        -----------
        output_file : str
            Output filename (.mp4 or .gif)
        fps : int
            Frames per second (default 10)
        dpi : int
            Resolution (default 150)
        """
        print(f"\nGenerating animation: {output_file}")

        # Normalize coefficients
        self.normalize_coefficients()

        # Create figure with quadchart layout
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Create subplots
        ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
        ax_profile = fig.add_subplot(gs[0, 1])
        ax_dD = fig.add_subplot(gs[1, 0])
        ax_cheb = fig.add_subplot(gs[1, 1])

        # Title
        title = fig.suptitle(f'Crater Degradation Evolution | D = {self.diameter_m:.0f} m',
                           fontsize=16, fontweight='bold')

        # Initialize plots
        def init():
            """Initialize animation."""
            # Clear all axes
            ax_3d.clear()
            ax_profile.clear()
            ax_dD.clear()
            ax_cheb.clear()
            return []

        def update(frame):
            """Update function for animation."""
            # Clear previous plots
            ax_3d.clear()
            ax_profile.clear()
            ax_dD.clear()
            ax_cheb.clear()

            age_ga = self.ages_ga[frame]
            elevation = self.degraded_profiles[frame]

            # ===== QUADRANT 1: 3D Surface =====
            X, Y, Z = self.create_3d_surface(elevation)

            surf = ax_3d.plot_surface(X, Y, Z, cmap=cm.terrain,
                                     linewidth=0, antialiased=True, alpha=0.9)

            ax_3d.set_xlabel('X (m)', fontsize=9)
            ax_3d.set_ylabel('Y (m)', fontsize=9)
            ax_3d.set_zlabel('Elevation (m)', fontsize=9)
            ax_3d.set_title(f'3D Topography | Age: {age_ga:.2f} Ga', fontsize=11, fontweight='bold')

            # Set consistent viewing angle
            ax_3d.view_init(elev=30, azim=45)

            # Set axis limits
            ax_3d.set_xlim([self.distance.min(), self.distance.max()])
            ax_3d.set_ylim([self.distance.min(), self.distance.max()])
            ax_3d.set_zlim([self.pristine_elevation.min() * 1.2, self.pristine_elevation.max() * 1.5])

            # ===== QUADRANT 2: 2D Profile =====
            # Plot pristine (reference)
            ax_profile.plot(self.distance / self.diameter_m, self.pristine_elevation / self.diameter_m,
                          'k--', linewidth=2, alpha=0.4, label='Pristine')

            # Plot current degraded
            ax_profile.plot(self.distance / self.diameter_m, elevation / self.diameter_m,
                          'b-', linewidth=2.5, label=f'{age_ga:.2f} Ga')

            ax_profile.set_xlabel('Normalized Distance (r/D)', fontsize=10)
            ax_profile.set_ylabel('Normalized Elevation (h/D)', fontsize=10)
            ax_profile.set_title('Elevation Profile Evolution', fontsize=11, fontweight='bold')
            ax_profile.legend(fontsize=9, loc='upper right')
            ax_profile.grid(True, alpha=0.3)
            ax_profile.axhline(0, color='k', linewidth=0.5)
            ax_profile.axvline(0, color='k', linewidth=0.5)
            ax_profile.set_xlim([-1.5, 1.5])

            # ===== QUADRANT 3: d/D Ratio Evolution =====
            ages_so_far = self.ages_ga[:frame+1]
            dD_so_far = self.d_D_ratios[:frame+1]

            ax_dD.plot(ages_so_far, dD_so_far, 'o-', color='darkred',
                     linewidth=2, markersize=4, label='d/D Ratio')

            # Add pristine reference line
            ax_dD.axhline(0.196, color='k', linestyle='--', alpha=0.5, label='Pristine (0.196)')

            # Current point marker
            ax_dD.plot(age_ga, self.d_D_ratios[frame], 'ro', markersize=10, zorder=5)

            ax_dD.set_xlabel('Age (Ga)', fontsize=10)
            ax_dD.set_ylabel('Depth/Diameter Ratio', fontsize=10)
            ax_dD.set_title('Depth-to-Diameter Evolution', fontsize=11, fontweight='bold')
            ax_dD.legend(fontsize=9, loc='upper right')
            ax_dD.grid(True, alpha=0.3)
            ax_dD.set_xlim([0, self.age_range[1]])
            ax_dD.set_ylim([0, 0.25])

            # Add current value text
            ax_dD.text(0.98, 0.02, f'd/D = {self.d_D_ratios[frame]:.4f}',
                      transform=ax_dD.transAxes, fontsize=10,
                      verticalalignment='bottom', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # ===== QUADRANT 4: Chebyshev Coefficients =====
            # Plot evolution of key coefficients
            key_coeffs = [0, 2, 4, 8]  # C0, C2, C4, C8
            colors = ['blue', 'red', 'green', 'orange']
            labels = ['C0 (Mean)', 'C2 (Depth)', 'C4 (Peak)', 'C8 (Peak)']

            for coeff_idx, color, label in zip(key_coeffs, colors, labels):
                coeffs_so_far = self.normalized_coeffs[:frame+1, coeff_idx]
                ax_cheb.plot(ages_so_far, coeffs_so_far, 'o-',
                           color=color, linewidth=2, markersize=3,
                           label=label, alpha=0.8)

                # Current point
                ax_cheb.plot(age_ga, self.normalized_coeffs[frame, coeff_idx],
                           'o', color=color, markersize=8, zorder=5)

            ax_cheb.set_xlabel('Age (Ga)', fontsize=10)
            ax_cheb.set_ylabel('Normalized Coefficient Value', fontsize=10)
            ax_cheb.set_title('Chebyshev Coefficient Evolution', fontsize=11, fontweight='bold')
            ax_cheb.legend(fontsize=9, loc='best')
            ax_cheb.grid(True, alpha=0.3)
            ax_cheb.set_xlim([0, self.age_range[1]])
            ax_cheb.set_ylim([-1.1, 1.1])

            # Progress indicator
            progress = (frame + 1) / self.num_frames * 100
            fig.text(0.99, 0.01, f'Frame: {frame+1}/{self.num_frames} ({progress:.0f}%)',
                    ha='right', va='bottom', fontsize=9, alpha=0.6)

            return []

        # Create animation
        anim = FuncAnimation(fig, update, init_func=init,
                           frames=self.num_frames, interval=1000/fps,
                           blit=False, repeat=True)

        # Save animation
        if output_file.endswith('.mp4'):
            writer = FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(output_file, writer=writer, dpi=dpi)
            print(f"✓ Animation saved as MP4: {output_file}")
        elif output_file.endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(output_file, writer=writer, dpi=dpi)
            print(f"✓ Animation saved as GIF: {output_file}")
        else:
            print("Error: Output file must be .mp4 or .gif")
            return

        plt.close(fig)


def generate_crater_animation(diameter_m=2000, output_file='crater_degradation.mp4',
                              age_range=(0.1, 3.9), num_frames=100, fps=10):
    """
    Convenience function to generate crater degradation animation.

    Parameters:
    -----------
    diameter_m : float
        Crater diameter in meters (default 2000m)
    output_file : str
        Output filename (.mp4 or .gif)
    age_range : tuple
        (min_age_ga, max_age_ga) in Ga
    num_frames : int
        Number of frames in animation
    fps : int
        Frames per second

    Returns:
    --------
    animator : CraterDegradationAnimator
        The animator object
    """
    if not MODULES_AVAILABLE:
        print("Error: Required modules not available.")
        return None

    print("="*60)
    print("CRATER DEGRADATION ANIMATION GENERATOR")
    print("="*60)
    print(f"\nCrater diameter: {diameter_m:.0f} m")
    print(f"Age range: {age_range[0]:.1f} - {age_range[1]:.1f} Ga")
    print(f"Animation: {num_frames} frames at {fps} fps")
    print(f"Output: {output_file}")

    # Create animator
    animator = CraterDegradationAnimator(
        diameter_m=diameter_m,
        diffusivity=5.0,
        age_range=age_range,
        num_frames=num_frames
    )

    # Generate animation
    animator.create_animation(output_file=output_file, fps=fps, dpi=150)

    print("\n" + "="*60)
    print("ANIMATION COMPLETE")
    print("="*60)

    return animator


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate crater degradation animation')
    parser.add_argument('--diameter', type=float, default=2000,
                       help='Crater diameter in meters (default: 2000)')
    parser.add_argument('--output', type=str, default='crater_degradation.mp4',
                       help='Output file (.mp4 or .gif)')
    parser.add_argument('--age-min', type=float, default=0.1,
                       help='Minimum age in Ga (default: 0.1)')
    parser.add_argument('--age-max', type=float, default=3.9,
                       help='Maximum age in Ga (default: 3.9)')
    parser.add_argument('--frames', type=int, default=100,
                       help='Number of frames (default: 100)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second (default: 10)')

    args = parser.parse_args()

    # Generate animation
    animator = generate_crater_animation(
        diameter_m=args.diameter,
        output_file=args.output,
        age_range=(args.age_min, args.age_max),
        num_frames=args.frames,
        fps=args.fps
    )

    print(f"\nAnimation saved to: {args.output}")
    print("You can view it with any video player or import into presentations.")

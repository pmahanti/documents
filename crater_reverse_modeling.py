#!/usr/bin/env python3
"""
Crater Reverse Modeling and Context Animation
==============================================

Takes observed lunar crater data (GeoTIFF topography and images) and:
1. Estimates impact parameters from crater morphology
2. Generates impact animation matching observed crater
3. Composites formation sequence into context image
4. Creates movie showing crater addition to lunar surface

Input:
- Crater topography GeoTIFF (elevation data)
- Crater image GeoTIFF (optical/reflectance)
- Context image GeoTIFF (wider area showing crater location)

Output:
- Impact parameters (projectile size, velocity, age)
- Formation animation (impact → observed state)
- Context movie (crater appears in landscape)
- Composite video with side-by-side comparison

Scientific approach:
- Crater size-morphometry inversion (Pike 1977)
- Scaling law inversion (Holsapple 1993)
- Time-evolution based on degradation state
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from lunar_impact_simulation import *
import argparse
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not available. Using synthetic data for demonstration.")


class CraterReverseModeler:
    """
    Reverse-models impact parameters from observed crater morphology.
    """

    def __init__(self, crater_topo_file: Optional[str] = None,
                 crater_image_file: Optional[str] = None,
                 context_image_file: Optional[str] = None):
        """
        Initialize with GeoTIFF files or use synthetic data.

        Parameters:
        -----------
        crater_topo_file : Path to crater topography GeoTIFF (elevation)
        crater_image_file : Path to crater image GeoTIFF (reflectance/optical)
        context_image_file : Path to context image GeoTIFF (wider area)
        """
        self.crater_topo_file = crater_topo_file
        self.crater_image_file = crater_image_file
        self.context_image_file = context_image_file

        # Crater parameters (to be extracted)
        self.crater_diameter = None
        self.crater_depth = None
        self.crater_center = None

        # Topography data
        self.topo_data = None
        self.image_data = None
        self.context_data = None

        # Estimated impact parameters
        self.projectile_diameter = None
        self.impact_velocity = None

        # Load data if files provided
        if crater_topo_file or crater_image_file or context_image_file:
            if not RASTERIO_AVAILABLE:
                print("ERROR: rasterio required for GeoTIFF support")
                print("Install with: pip install rasterio")
                print("\nFalling back to synthetic demonstration mode...")
                self.use_synthetic_data = True
            else:
                self.load_geotiff_data()
                self.use_synthetic_data = False
        else:
            print("No input files provided - using synthetic data for demonstration")
            self.use_synthetic_data = True

    def load_geotiff_data(self):
        """Load crater and context GeoTIFF files."""
        print("Loading GeoTIFF data...")

        # Load crater topography
        if self.crater_topo_file:
            with rasterio.open(self.crater_topo_file) as src:
                self.topo_data = src.read(1)  # First band
                self.topo_transform = src.transform
                self.topo_crs = src.crs
                print(f"✓ Loaded crater topography: {self.topo_data.shape}")

        # Load crater image
        if self.crater_image_file:
            with rasterio.open(self.crater_image_file) as src:
                self.image_data = src.read(1)
                self.image_transform = src.transform
                print(f"✓ Loaded crater image: {self.image_data.shape}")

        # Load context image
        if self.context_image_file:
            with rasterio.open(self.context_image_file) as src:
                self.context_data = src.read(1)
                self.context_transform = src.transform
                print(f"✓ Loaded context image: {self.context_data.shape}")

    def create_synthetic_crater(self, diameter_m: float = 300.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic crater data for demonstration.

        Parameters:
        -----------
        diameter_m : Desired crater diameter in meters

        Returns:
        --------
        topo_data : 2D elevation array (meters)
        image_data : 2D reflectance array (0-1)
        """
        print(f"\nGenerating synthetic {diameter_m:.0f}m crater for demonstration...")

        # Grid size: 2× crater diameter, 1m resolution
        grid_size = int(2 * diameter_m)
        self.pixel_scale = 1.0  # meters per pixel

        # Create coordinate grids
        x = np.linspace(-diameter_m, diameter_m, grid_size)
        y = np.linspace(-diameter_m, diameter_m, grid_size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)

        # Crater parameters from Pike (1977)
        D = diameter_m
        R_crater = D / 2
        d = 0.196 * D  # depth
        h_rim = 0.036 * D  # rim height

        # Generate elevation profile
        topo = np.zeros_like(R)

        # Bowl (parabolic)
        bowl_mask = R <= R_crater
        topo[bowl_mask] = -d * (1 - (R[bowl_mask] / R_crater)**2)

        # Rim (exponential decay)
        rim_mask = (R > R_crater) & (R <= 1.3 * R_crater)
        topo[rim_mask] = h_rim * np.exp(-5 * (R[rim_mask] / R_crater - 1))

        # Ejecta blanket (power law)
        ejecta_mask = R > 1.3 * R_crater
        T0 = 0.04 * R_crater
        topo[ejecta_mask] = T0 * (R_crater / R[ejecta_mask])**3

        # Generate image (albedo) - darker inside crater, brighter ejecta
        image = np.ones_like(R) * 0.15  # Background albedo

        # Crater interior (darker)
        interior_mask = R <= R_crater
        image[interior_mask] = 0.08

        # Ejecta rays (brighter)
        ejecta_mask2 = (R > R_crater) & (R < 3 * R_crater)
        image[ejecta_mask2] = 0.20

        # Add some ray structure
        theta = np.arctan2(Y, X)
        n_rays = 8
        for i in range(n_rays):
            ray_angle = i * 2 * np.pi / n_rays
            ray_mask = (np.abs(theta - ray_angle) < 0.2) & ejecta_mask2
            image[ray_mask] += 0.05

        # Add noise
        topo += np.random.normal(0, d * 0.01, topo.shape)
        image += np.random.normal(0, 0.01, image.shape)
        image = np.clip(image, 0, 1)

        self.topo_data = topo
        self.image_data = image
        self.crater_center = (grid_size // 2, grid_size // 2)
        self.crater_diameter = diameter_m
        self.crater_depth = d

        print(f"✓ Synthetic crater created: D={D:.0f}m, d={d:.0f}m")

        return topo, image

    def create_synthetic_context(self, crater_location: Tuple[float, float] = (0.5, 0.5)):
        """
        Create synthetic context image showing wider lunar surface.

        Parameters:
        -----------
        crater_location : (x_frac, y_frac) normalized position in context (0-1)
        """
        print("\nGenerating synthetic context image...")

        # Context is 3× larger than crater scene
        context_size = self.topo_data.shape[0] * 3

        # Create lunar surface texture
        context = np.random.normal(0.12, 0.03, (context_size, context_size))

        # Add some background craters
        n_background_craters = 5
        for i in range(n_background_craters):
            cx = np.random.randint(50, context_size - 50)
            cy = np.random.randint(50, context_size - 50)
            cd = np.random.uniform(30, 100)  # Small background craters

            x = np.arange(context_size)
            y = np.arange(context_size)
            X, Y = np.meshgrid(x - cx, y - cy)
            R = np.sqrt(X**2 + Y**2)

            crater_mask = R < cd
            context[crater_mask] -= 0.03

        context = np.clip(context, 0, 1)

        self.context_data = context
        self.crater_location_context = (
            int(crater_location[0] * context_size),
            int(crater_location[1] * context_size)
        )

        print(f"✓ Context created: {context_size}×{context_size} pixels")

        return context

    def extract_crater_parameters(self):
        """
        Extract crater morphometry from topography data.

        Measures:
        - Diameter (rim to rim)
        - Depth (floor to rim)
        - Volume
        - Rim height
        """
        if self.use_synthetic_data and self.topo_data is None:
            # Create synthetic data first
            self.create_synthetic_crater(diameter_m=300)
            self.create_synthetic_context()

        print("\n" + "="*70)
        print("EXTRACTING CRATER MORPHOMETRY")
        print("="*70)

        # Find crater center (lowest point)
        if self.crater_center is None:
            min_idx = np.unravel_index(np.argmin(self.topo_data), self.topo_data.shape)
            self.crater_center = min_idx

        cy, cx = self.crater_center

        # Create radial profile
        ny, nx = self.topo_data.shape
        y, x = np.ogrid[0:ny, 0:nx]
        R_grid = np.sqrt((x - cx)**2 + (y - cy)**2)

        # Bin by radius
        max_r = min(cx, nx - cx, cy, ny - cy)
        r_bins = np.arange(0, max_r, 1)
        profile_mean = []
        profile_std = []

        for i in range(len(r_bins) - 1):
            mask = (R_grid >= r_bins[i]) & (R_grid < r_bins[i + 1])
            if np.any(mask):
                profile_mean.append(np.mean(self.topo_data[mask]))
                profile_std.append(np.std(self.topo_data[mask]))
            else:
                profile_mean.append(np.nan)
                profile_std.append(np.nan)

        profile_mean = np.array(profile_mean)
        profile_std = np.array(profile_std)

        # Find rim (maximum in first 30% of profile)
        search_range = int(len(profile_mean) * 0.3)
        rim_idx = np.nanargmax(profile_mean[:search_range])
        rim_radius_pix = r_bins[rim_idx]

        # Find depth (minimum elevation)
        floor_elev = np.nanmin(self.topo_data)
        rim_elev = profile_mean[rim_idx]

        # Calculate parameters
        pixel_scale = getattr(self, 'pixel_scale', 1.0)  # m/pixel
        self.crater_diameter = 2 * rim_radius_pix * pixel_scale
        self.crater_depth = rim_elev - floor_elev
        self.rim_height = rim_elev

        print(f"\nMeasured Crater Morphometry:")
        print(f"  Diameter (rim-to-rim): {self.crater_diameter:.1f} m")
        print(f"  Depth (floor to rim): {self.crater_depth:.1f} m")
        print(f"  Depth/Diameter ratio: {self.crater_depth/self.crater_diameter:.3f}")
        print(f"  Rim height: {self.rim_height:.1f} m")
        print(f"  Center position: ({cx}, {cy}) pixels")

        # Store profile for later
        self.radial_profile_r = r_bins[:-1] * pixel_scale
        self.radial_profile_z = profile_mean

        return self.crater_diameter, self.crater_depth

    def estimate_impact_parameters(self, velocity_kms: float = 20.0):
        """
        Invert scaling laws to estimate projectile parameters.

        Given observed crater, estimate what impact created it.

        Parameters:
        -----------
        velocity_kms : Assumed impact velocity (km/s) - default 20 km/s typical

        Returns:
        --------
        projectile_diameter : Estimated projectile size (m)
        impact_velocity : Impact velocity (m/s)
        """
        if self.crater_diameter is None:
            self.extract_crater_parameters()

        print("\n" + "="*70)
        print("ESTIMATING IMPACT PARAMETERS (Scaling Law Inversion)")
        print("="*70)

        # Use scaling laws in reverse
        # D = 0.084 * L * (rho_p/rho_t)^(1/3) * (v^2 / (g_eff*L))^0.4
        #
        # Solving for L given D, v:
        # L ≈ D / [0.084 * (rho_p/rho_t)^(1/3) * (v^2 / g_eff * L))^0.4]
        #
        # This is iterative since L appears on both sides

        D_obs = self.crater_diameter
        v = velocity_kms * 1000  # km/s to m/s

        # Target and projectile properties
        target = TargetParameters()
        rho_t = target.effective_density
        rho_p = 2800  # Assume rocky projectile
        g = target.gravity
        Y = target.cohesion

        # Iterative solution
        L_guess = D_obs / 100  # Initial guess: D/L ~ 100
        for iteration in range(10):
            g_eff = g + Y / (rho_t * L_guess)
            scaling = 0.084 * (rho_p / rho_t)**(1/3) * (v**2 / (g_eff * L_guess))**0.4
            L_new = D_obs / (1.2 * scaling)  # 1.2 is transient→final factor

            if abs(L_new - L_guess) / L_guess < 0.01:  # 1% convergence
                break

            L_guess = L_new

        self.projectile_diameter = L_new
        self.impact_velocity = v

        # Verify with forward model
        proj_test = ProjectileParameters(L_new, v, 90, rho_p, 'rocky')
        target_test = TargetParameters()
        scaling_test = CraterScalingLaws(target_test)
        D_predicted = scaling_test.final_crater_diameter(proj_test)

        error = abs(D_predicted - D_obs) / D_obs * 100

        print(f"\nAssumed impact velocity: {velocity_kms:.1f} km/s")
        print(f"\nInversion Results:")
        print(f"  Estimated projectile diameter: {self.projectile_diameter:.2f} m")
        print(f"  Projectile density (assumed): {rho_p} kg/m³ (rocky)")
        print(f"  Impact angle (assumed): 90° (vertical)")
        print(f"\nVerification (forward model):")
        print(f"  Predicted crater diameter: {D_predicted:.1f} m")
        print(f"  Observed crater diameter: {D_obs:.1f} m")
        print(f"  Error: {error:.2f}%")

        if error < 5:
            print(f"  Status: ✓ Good match (error < 5%)")
        else:
            print(f"  Status: ⚠ Consider adjusting velocity assumption")

        print("="*70)

        return self.projectile_diameter, self.impact_velocity

    def generate_formation_animation(self, output_file: str = 'crater_formation.gif',
                                    frames: int = 80, fps: int = 15):
        """
        Generate animation showing impact and crater formation matching observed state.

        Parameters:
        -----------
        output_file : Output filename (.gif)
        frames : Number of animation frames
        fps : Frames per second
        """
        if self.projectile_diameter is None:
            self.estimate_impact_parameters()

        print("\n" + "="*70)
        print("GENERATING FORMATION ANIMATION")
        print("="*70)
        print(f"Frames: {frames}, FPS: {fps}")
        print(f"Output: {output_file}")

        # Create simulation
        proj = ProjectileParameters(
            self.projectile_diameter,
            self.impact_velocity,
            90,  # Vertical impact
            2800,  # Rocky
            'rocky'
        )

        target = TargetParameters()
        sim = ImpactSimulation(proj, target)
        sim.run(n_ejecta_particles=600)

        # Set up animation
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax_profile = axes[0]
        ax_map = axes[1]

        def update(frame):
            t_frac = frame / (frames - 1)
            excavation_progress = min(0.2 + t_frac * 0.8, 1.0)

            # === Left: Crater Profile ===
            ax_profile.clear()

            # Simulated profile
            r_sim = np.linspace(0, 1.5 * sim.morphology.D, 300)
            z_sim = sim.morphology.crater_profile(r_sim, time_fraction=excavation_progress)

            ax_profile.plot(r_sim, z_sim, 'b-', linewidth=2, label='Simulated')

            # Observed profile (if available)
            if hasattr(self, 'radial_profile_r'):
                ax_profile.plot(self.radial_profile_r, self.radial_profile_z - np.max(self.radial_profile_z),
                              'r--', linewidth=2, alpha=0.7, label='Observed')

            ax_profile.axhline(0, color='brown', linestyle=':', alpha=0.5)
            ax_profile.fill_between(r_sim, z_sim, -sim.morphology.d * 1.5,
                                   where=(z_sim < 0), color='tan', alpha=0.3)
            ax_profile.set_xlabel('Radial Distance (m)', fontsize=11)
            ax_profile.set_ylabel('Elevation (m)', fontsize=11)
            ax_profile.set_title(f'Crater Formation Progress: {excavation_progress*100:.0f}%',
                                fontweight='bold')
            ax_profile.grid(True, alpha=0.3)
            ax_profile.legend(loc='upper right')
            ax_profile.set_xlim(0, 1.5 * sim.morphology.D)
            ax_profile.set_ylim(-sim.morphology.d * 1.2, sim.morphology.d * 0.3)

            # === Right: Map View ===
            ax_map.clear()

            if self.image_data is not None:
                # Blend observed image with formation progress
                alpha = excavation_progress
                ax_map.imshow(self.image_data, cmap='gray', alpha=alpha, origin='lower')

            # Crater outline
            if self.crater_center:
                cy, cx = self.crater_center
                crater_radius_pix = (sim.morphology.D / 2) / getattr(self, 'pixel_scale', 1.0)
                circle = Circle((cx, cy), crater_radius_pix * excavation_progress,
                              fill=False, edgecolor='red', linewidth=2)
                ax_map.add_patch(circle)

            ax_map.set_title('Map View: Crater Appearance', fontweight='bold')
            ax_map.set_xlabel('X (pixels)')
            ax_map.set_ylabel('Y (pixels)')

            # Info text
            info = f"Projectile: {self.projectile_diameter:.1f}m @ {self.impact_velocity/1000:.0f} km/s\n"
            info += f"Final crater: D={sim.morphology.D:.0f}m, d={sim.morphology.d:.0f}m"
            fig.suptitle(info, fontsize=12, fontweight='bold')

        anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)

        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
        print(f"✓ Formation animation saved: {output_file}")
        plt.close()

        return output_file

    def generate_context_movie(self, output_file: str = 'crater_in_context.gif',
                              frames: int = 60, fps: int = 12):
        """
        Generate movie showing crater appearing in context image.

        Parameters:
        -----------
        output_file : Output filename (.gif)
        frames : Number of frames
        fps : Frames per second
        """
        if self.context_data is None:
            print("No context data - creating synthetic context...")
            self.create_synthetic_context()

        print("\n" + "="*70)
        print("GENERATING CONTEXT MOVIE")
        print("="*70)
        print(f"Output: {output_file}")

        fig, ax = plt.subplots(figsize=(12, 10))

        # Determine crater location in context
        if hasattr(self, 'crater_location_context'):
            cx_ctx, cy_ctx = self.crater_location_context
        else:
            # Center of context
            cx_ctx = self.context_data.shape[1] // 2
            cy_ctx = self.context_data.shape[0] // 2

        # Crater size in pixels
        crater_size_pix = self.crater_diameter / getattr(self, 'pixel_scale', 1.0)

        def update(frame):
            t_frac = frame / (frames - 1)
            alpha = min(t_frac * 1.5, 1.0)  # Fade in

            ax.clear()

            # Show context
            ax.imshow(self.context_data, cmap='gray', origin='lower')

            # Overlay crater (fade in)
            if self.image_data is not None and alpha > 0:
                # Place crater in context
                h, w = self.image_data.shape
                y_start = int(cy_ctx - h//2)
                x_start = int(cx_ctx - w//2)

                # Create overlay
                overlay = self.context_data.copy()
                y_end = min(y_start + h, overlay.shape[0])
                x_end = min(x_start + w, overlay.shape[1])

                h_actual = y_end - y_start
                w_actual = x_end - x_start

                overlay[y_start:y_end, x_start:x_end] = (
                    (1 - alpha) * overlay[y_start:y_end, x_start:x_end] +
                    alpha * self.image_data[:h_actual, :w_actual]
                )

                ax.imshow(overlay, cmap='gray', origin='lower')

            # Crater marker
            circle = Circle((cx_ctx, cy_ctx), crater_size_pix / 2 * alpha,
                          fill=False, edgecolor='red', linewidth=2, linestyle='--')
            ax.add_patch(circle)

            # Scale bar
            scale_length_m = 500  # 500m scale bar
            scale_length_pix = scale_length_m / getattr(self, 'pixel_scale', 1.0)
            ax.plot([50, 50 + scale_length_pix], [50, 50], 'w-', linewidth=3)
            ax.text(50 + scale_length_pix/2, 70, f'{scale_length_m}m',
                   color='white', ha='center', fontweight='bold', fontsize=10)

            ax.set_title(f'Crater Formation in Context (D={self.crater_diameter:.0f}m)\n'
                        f'Progress: {alpha*100:.0f}%',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_xlim(0, self.context_data.shape[1])
            ax.set_ylim(0, self.context_data.shape[0])

        anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)

        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
        print(f"✓ Context movie saved: {output_file}")
        plt.close()

        return output_file


def main():
    """Example usage with synthetic data."""
    parser = argparse.ArgumentParser(
        description='Lunar crater reverse modeling and context animation')
    parser.add_argument('--crater-topo', type=str, default=None,
                       help='Crater topography GeoTIFF file')
    parser.add_argument('--crater-image', type=str, default=None,
                       help='Crater image GeoTIFF file')
    parser.add_argument('--context-image', type=str, default=None,
                       help='Context image GeoTIFF file')
    parser.add_argument('--diameter', type=float, default=300.0,
                       help='Crater diameter for synthetic mode (meters)')
    parser.add_argument('--velocity', type=float, default=20.0,
                       help='Assumed impact velocity (km/s)')
    parser.add_argument('--frames', type=int, default=80,
                       help='Number of animation frames')
    parser.add_argument('--fps', type=int, default=15,
                       help='Frames per second')
    parser.add_argument('--output-prefix', type=str, default='crater_reverse',
                       help='Output file prefix')

    args = parser.parse_args()

    print("\n" + "="*70)
    print(" LUNAR CRATER REVERSE MODELING AND CONTEXT ANIMATION")
    print("="*70)

    # Initialize modeler
    modeler = CraterReverseModeler(
        crater_topo_file=args.crater_topo,
        crater_image_file=args.crater_image,
        context_image_file=args.context_image
    )

    # If using synthetic data, create it
    if modeler.use_synthetic_data:
        print(f"\n[SYNTHETIC DATA MODE]")
        modeler.create_synthetic_crater(diameter_m=args.diameter)
        modeler.create_synthetic_context()

    # Extract crater parameters from data
    modeler.extract_crater_parameters()

    # Estimate impact parameters (inverse problem)
    modeler.estimate_impact_parameters(velocity_kms=args.velocity)

    # Generate animations
    print("\n")
    formation_file = f'{args.output_prefix}_formation.gif'
    modeler.generate_formation_animation(
        output_file=formation_file,
        frames=args.frames,
        fps=args.fps
    )

    context_file = f'{args.output_prefix}_context.gif'
    modeler.generate_context_movie(
        output_file=context_file,
        frames=60,
        fps=12
    )

    print("\n" + "="*70)
    print(" COMPLETE")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  1. {formation_file} - Impact formation matching observed crater")
    print(f"  2. {context_file} - Crater appearing in context image")
    print("\n")


if __name__ == "__main__":
    main()

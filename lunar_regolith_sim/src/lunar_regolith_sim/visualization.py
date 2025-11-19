"""
Visualization tools for regolith flow simulations.

This module provides functions to visualize simulation results,
including elevation maps, flow patterns, and elephant hide textures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import os


class SimulationVisualizer:
    """
    Visualize regolith flow simulation results.
    """

    def __init__(self, simulation):
        """
        Initialize visualizer.

        Args:
            simulation: RegolithFlowSimulation object
        """
        self.sim = simulation
        self.slope = simulation.slope
        self.figures = []

    def plot_elevation(self, include_regolith=True, save_path=None, show=True):
        """
        Plot elevation map.

        Args:
            include_regolith: Include regolith thickness in elevation
            save_path: Path to save figure (optional)
            show: Display the figure

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        if include_regolith:
            elevation = self.slope.elevation + self.sim.thickness
            title = "Total Elevation (Bedrock + Regolith)"
        else:
            elevation = self.slope.elevation
            title = "Bedrock Elevation"

        # Create hillshade for better visualization
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(elevation, vert_exag=2.0)

        # Plot with hillshade
        im = ax.imshow(elevation, cmap='terrain', extent=[0, self.slope.width, 0, self.slope.height])
        ax.imshow(hillshade, cmap='gray', alpha=0.3, extent=[0, self.slope.width, 0, self.slope.height])

        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Distance (m)')
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax, label='Elevation (m)')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        self.figures.append(fig)
        return fig

    def plot_slope_angle(self, save_path=None, show=True):
        """
        Plot slope angle distribution.

        Args:
            save_path: Path to save figure (optional)
            show: Display the figure

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        slope_angle = self.slope.get_slope_angle()

        # Map view
        im = ax1.imshow(slope_angle, cmap='hot', extent=[0, self.slope.width, 0, self.slope.height])
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Distance (m)')
        ax1.set_title('Slope Angle Distribution')
        plt.colorbar(im, ax=ax1, label='Slope Angle (degrees)')

        # Add angle of repose line
        ax1.contour(slope_angle, levels=[self.sim.physics.angle_of_repose],
                   colors='cyan', linewidths=2, linestyles='--',
                   extent=[0, self.slope.width, 0, self.slope.height])

        # Histogram
        ax2.hist(slope_angle.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(self.sim.physics.angle_of_repose, color='red', linestyle='--',
                   linewidth=2, label=f'Angle of Repose ({self.sim.physics.angle_of_repose:.1f}°)')
        ax2.set_xlabel('Slope Angle (degrees)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Slope Angle Histogram')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        self.figures.append(fig)
        return fig

    def plot_elephant_hide(self, save_path=None, show=True, enhance=True):
        """
        Plot elephant hide texture pattern.

        Args:
            save_path: Path to save figure (optional)
            show: Display the figure
            enhance: Apply enhancement to make texture more visible

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        texture = self.sim.get_elephant_hide_texture()

        if enhance:
            # Enhance texture visibility
            from scipy.ndimage import gaussian_filter, sobel
            texture_smooth = gaussian_filter(texture, sigma=2.0)
            texture_enhanced = sobel(texture_smooth)
            texture_enhanced = (texture_enhanced - texture_enhanced.min()) / (texture_enhanced.max() - texture_enhanced.min())
        else:
            texture_enhanced = texture

        # Plot 1: Raw elephant hide texture
        im1 = axes[0, 0].imshow(texture, cmap='bone', extent=[0, self.slope.width, 0, self.slope.height])
        axes[0, 0].set_xlabel('Distance (m)')
        axes[0, 0].set_ylabel('Distance (m)')
        axes[0, 0].set_title('Elephant Hide Texture (Raw)')
        plt.colorbar(im1, ax=axes[0, 0], label='Texture Intensity')

        # Plot 2: Enhanced texture
        im2 = axes[0, 1].imshow(texture_enhanced, cmap='bone', extent=[0, self.slope.width, 0, self.slope.height])
        axes[0, 1].set_xlabel('Distance (m)')
        axes[0, 1].set_ylabel('Distance (m)')
        axes[0, 1].set_title('Elephant Hide Texture (Enhanced)')
        plt.colorbar(im2, ax=axes[0, 1], label='Texture Intensity')

        # Plot 3: Cumulative deformation
        im3 = axes[1, 0].imshow(self.sim.cumulative_deformation, cmap='viridis',
                               extent=[0, self.slope.width, 0, self.slope.height])
        axes[1, 0].set_xlabel('Distance (m)')
        axes[1, 0].set_ylabel('Distance (m)')
        axes[1, 0].set_title('Cumulative Deformation')
        plt.colorbar(im3, ax=axes[1, 0], label='Deformation')

        # Plot 4: Flow event count
        im4 = axes[1, 1].imshow(self.sim.flow_count, cmap='plasma',
                               extent=[0, self.slope.width, 0, self.slope.height])
        axes[1, 1].set_xlabel('Distance (m)')
        axes[1, 1].set_ylabel('Distance (m)')
        axes[1, 1].set_title('Flow Event Count')
        plt.colorbar(im4, ax=axes[1, 1], label='Number of Events')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        self.figures.append(fig)
        return fig

    def plot_3d_surface(self, include_regolith=True, save_path=None, show=True):
        """
        Plot 3D surface visualization.

        Args:
            include_regolith: Include regolith thickness
            save_path: Path to save figure (optional)
            show: Display the figure

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        if include_regolith:
            elevation = self.slope.elevation + self.sim.thickness
        else:
            elevation = self.slope.elevation

        # Subsample for faster rendering
        stride = max(1, self.slope.nx // 100)
        X_sub = self.slope.X[::stride, ::stride]
        Y_sub = self.slope.Y[::stride, ::stride]
        Z_sub = elevation[::stride, ::stride]

        # Plot surface
        surf = ax.plot_surface(X_sub, Y_sub, Z_sub, cmap='terrain',
                              linewidth=0, antialiased=True, alpha=0.9)

        ax.set_xlabel('X Distance (m)')
        ax.set_ylabel('Y Distance (m)')
        ax.set_zlabel('Elevation (m)')
        ax.set_title('3D Surface View')

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (m)')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        self.figures.append(fig)
        return fig

    def plot_flow_velocity(self, save_path=None, show=True):
        """
        Plot flow velocity field.

        Args:
            save_path: Path to save figure (optional)
            show: Display the figure

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        velocity_magnitude = np.sqrt(self.sim.velocity_x**2 + self.sim.velocity_y**2)

        # Velocity magnitude map
        im = ax1.imshow(velocity_magnitude, cmap='plasma',
                       extent=[0, self.slope.width, 0, self.slope.height])
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Distance (m)')
        ax1.set_title('Flow Velocity Magnitude')
        plt.colorbar(im, ax=ax1, label='Velocity (m/s)')

        # Velocity vectors (subsampled)
        stride = max(1, self.slope.nx // 20)
        X_sub = self.slope.X[::stride, ::stride]
        Y_sub = self.slope.Y[::stride, ::stride]
        U_sub = self.sim.velocity_x[::stride, ::stride]
        V_sub = self.sim.velocity_y[::stride, ::stride]

        ax2.quiver(X_sub, Y_sub, U_sub, V_sub, velocity_magnitude[::stride, ::stride],
                  cmap='plasma', scale=5.0)
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Distance (m)')
        ax2.set_title('Flow Velocity Field')
        ax2.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        self.figures.append(fig)
        return fig

    def create_summary_figure(self, save_path=None, show=True):
        """
        Create comprehensive summary figure.

        Args:
            save_path: Path to save figure (optional)
            show: Display the figure

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Elevation with hillshade
        ax1 = fig.add_subplot(gs[0, 0])
        elevation = self.slope.elevation + self.sim.thickness
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(elevation, vert_exag=2.0)
        im1 = ax1.imshow(elevation, cmap='terrain', extent=[0, self.slope.width, 0, self.slope.height])
        ax1.imshow(hillshade, cmap='gray', alpha=0.3, extent=[0, self.slope.width, 0, self.slope.height])
        ax1.set_title('Elevation')
        plt.colorbar(im1, ax=ax1, label='m')

        # 2. Slope angle
        ax2 = fig.add_subplot(gs[0, 1])
        slope_angle = self.slope.get_slope_angle()
        im2 = ax2.imshow(slope_angle, cmap='hot', extent=[0, self.slope.width, 0, self.slope.height])
        ax2.set_title('Slope Angle')
        plt.colorbar(im2, ax=ax2, label='degrees')

        # 3. Elephant hide texture
        ax3 = fig.add_subplot(gs[0, 2])
        texture = self.sim.get_elephant_hide_texture()
        im3 = ax3.imshow(texture, cmap='bone', extent=[0, self.slope.width, 0, self.slope.height])
        ax3.set_title('Elephant Hide Texture')
        plt.colorbar(im3, ax=ax3)

        # 4. Regolith thickness
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(self.sim.thickness, cmap='YlOrBr', extent=[0, self.slope.width, 0, self.slope.height])
        ax4.set_title('Regolith Thickness')
        plt.colorbar(im4, ax=ax4, label='m')

        # 5. Velocity magnitude
        ax5 = fig.add_subplot(gs[1, 1])
        velocity_mag = np.sqrt(self.sim.velocity_x**2 + self.sim.velocity_y**2)
        im5 = ax5.imshow(velocity_mag, cmap='plasma', extent=[0, self.slope.width, 0, self.slope.height])
        ax5.set_title('Flow Velocity')
        plt.colorbar(im5, ax=ax5, label='m/s')

        # 6. Cumulative deformation
        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(self.sim.cumulative_deformation, cmap='viridis',
                        extent=[0, self.slope.width, 0, self.slope.height])
        ax6.set_title('Cumulative Deformation')
        plt.colorbar(im6, ax=ax6)

        # 7. Cross-section
        ax7 = fig.add_subplot(gs[2, :])
        mid_row = self.slope.ny // 2
        x = self.slope.x
        bedrock = self.slope.elevation[mid_row, :]
        total = bedrock + self.sim.thickness[mid_row, :]

        ax7.fill_between(x, 0, bedrock, color='gray', alpha=0.5, label='Bedrock')
        ax7.fill_between(x, bedrock, total, color='brown', alpha=0.5, label='Regolith')
        ax7.plot(x, total, 'k-', linewidth=1.5, label='Surface')
        ax7.set_xlabel('Distance (m)')
        ax7.set_ylabel('Elevation (m)')
        ax7.set_title('Cross-Section (mid-line)')
        ax7.legend()
        ax7.grid(alpha=0.3)

        fig.suptitle(f'Lunar Regolith Flow Simulation Summary (t = {self.sim.time:.1f} s)',
                    fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        self.figures.append(fig)
        return fig

    def save_all_figures(self, output_dir):
        """
        Save all generated figures to directory.

        Args:
            output_dir: Directory to save figures
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, fig in enumerate(self.figures):
            filepath = os.path.join(output_dir, f'figure_{i+1}.png')
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

    def close_all(self):
        """Close all figure windows."""
        plt.close('all')
        self.figures = []

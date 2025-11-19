"""
Quick animation demo - simplified texture evolution visualization.
Creates a fast animation showing conceptual elephant hide texture formation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import os

def generate_texture_frame(t, size=200):
    """
    Generate a procedural texture pattern that evolves over time.
    Simulates the appearance of elephant hide wrinkles forming.

    Args:
        t: Time parameter (0 to 1)
        size: Grid size

    Returns:
        2D array representing texture intensity
    """
    x = np.linspace(0, 10, size)
    y = np.linspace(0, 10, size)
    X, Y = np.meshgrid(x, y)

    # Create a slope (steeper at top)
    slope = 0.3 * Y

    # Add evolving wrinkle patterns using multiple frequency components
    texture = np.zeros_like(X)

    # Low frequency (large wrinkles) - appear first
    if t > 0.1:
        intensity1 = min((t - 0.1) / 0.3, 1.0)
        texture += intensity1 * 0.8 * np.sin(2 * X + 0.5 * Y) * np.cos(1.5 * Y)

    # Medium frequency - appear later
    if t > 0.3:
        intensity2 = min((t - 0.3) / 0.3, 1.0)
        texture += intensity2 * 0.6 * np.sin(4 * X + Y) * np.cos(3 * Y + 0.5 * X)

    # High frequency (fine detail) - appear last
    if t > 0.6:
        intensity3 = min((t - 0.6) / 0.3, 1.0)
        texture += intensity3 * 0.4 * np.sin(8 * X + 2 * Y) * np.cos(5 * Y - X)

    # Add some randomness for realism
    np.random.seed(42)
    noise = np.random.randn(size, size) * 0.1 * t
    texture += noise

    # Combine with slope
    combined = slope + texture

    # Make texture more pronounced on steeper slopes
    slope_factor = (Y / 10) ** 2
    combined = combined * (1 + slope_factor)

    return combined

def create_animation(num_frames=30, output_dir='output/animation_demo'):
    """Create and save the animation."""

    print("=" * 70)
    print("Quick Lunar Texture Animation Demo")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n1. Creating animation with {num_frames} frames...")

    # Create figure
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Main texture view
    ax_main = fig.add_subplot(gs[:, :2])
    ax_main.set_title('Elephant Hide Texture Evolution', fontsize=14, fontweight='bold')
    ax_main.set_xlabel('Distance (m)', fontsize=11)
    ax_main.set_ylabel('Distance (m)', fontsize=11)

    # Time series plots
    ax_mean = fig.add_subplot(gs[0, 2])
    ax_mean.set_title('Mean Texture Intensity', fontsize=10)
    ax_mean.set_xlabel('Time (Myr)', fontsize=9)
    ax_mean.set_ylabel('Intensity', fontsize=9)
    ax_mean.grid(True, alpha=0.3)

    ax_max = fig.add_subplot(gs[1, 2])
    ax_max.set_title('Max Texture Intensity', fontsize=10)
    ax_max.set_xlabel('Time (Myr)', fontsize=9)
    ax_max.set_ylabel('Intensity', fontsize=9)
    ax_max.grid(True, alpha=0.3)

    # Generate all frames
    print("2. Generating texture frames...")
    frames = []
    time_points = []
    mean_values = []
    max_values = []

    for i in range(num_frames):
        t = i / (num_frames - 1)  # 0 to 1
        time_myr = t * 3.0  # 0 to 3 million years

        texture = generate_texture_frame(t)
        frames.append(texture)
        time_points.append(time_myr)
        mean_values.append(np.mean(np.abs(texture)))
        max_values.append(np.max(np.abs(texture)))

        if (i + 1) % 10 == 0:
            print(f"   Generated frame {i + 1}/{num_frames}")

    print("3. Creating animation...")

    # Initialize plots
    im = ax_main.imshow(frames[0], cmap='terrain', interpolation='bilinear',
                        extent=[0, 200, 0, 200], origin='lower')
    plt.colorbar(im, ax=ax_main, label='Elevation (m)')

    time_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes,
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    line_mean, = ax_mean.plot([], [], 'b-', linewidth=2)
    line_max, = ax_max.plot([], [], 'r-', linewidth=2)

    def init():
        """Initialize animation."""
        im.set_data(frames[0])
        time_text.set_text('Time: 0.00 Myr\n(Fresh crater)')
        line_mean.set_data([], [])
        line_max.set_data([], [])
        return im, time_text, line_mean, line_max

    def animate(frame):
        """Update animation frame."""
        im.set_data(frames[frame])

        # Update time text
        t_myr = time_points[frame]
        if frame == 0:
            time_text.set_text(f'Time: {t_myr:.2f} Myr\n(Fresh crater)')
        elif frame < num_frames // 3:
            time_text.set_text(f'Time: {t_myr:.2f} Myr\n(Early stage)')
        elif frame < 2 * num_frames // 3:
            time_text.set_text(f'Time: {t_myr:.2f} Myr\n(Developing)')
        else:
            time_text.set_text(f'Time: {t_myr:.2f} Myr\n(Mature texture)')

        # Update time series plots
        line_mean.set_data(time_points[:frame+1], mean_values[:frame+1])
        line_max.set_data(time_points[:frame+1], max_values[:frame+1])

        ax_mean.set_xlim(0, 3.0)
        ax_mean.set_ylim(0, max(mean_values) * 1.1)
        ax_max.set_xlim(0, 3.0)
        ax_max.set_ylim(0, max(max_values) * 1.1)

        return im, time_text, line_mean, line_max

    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=num_frames, interval=200, blit=True)

    # Save animation
    print("4. Saving animation...")
    animation_path = os.path.join(output_dir, 'texture_evolution.gif')
    anim.save(animation_path, writer='pillow', fps=5, dpi=100)
    print(f"   Saved to: {animation_path}")

    # Also save individual frames
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    print("5. Saving individual frames...")
    for i, (frame_data, time_myr) in enumerate(zip(frames, time_points)):
        fig_frame, ax_frame = plt.subplots(figsize=(8, 6))
        im_frame = ax_frame.imshow(frame_data, cmap='terrain', interpolation='bilinear',
                                   extent=[0, 200, 0, 200], origin='lower')
        plt.colorbar(im_frame, ax=ax_frame, label='Elevation (m)')
        ax_frame.set_title(f'Elephant Hide Texture - Time: {time_myr:.2f} Myr',
                          fontsize=12, fontweight='bold')
        ax_frame.set_xlabel('Distance (m)')
        ax_frame.set_ylabel('Distance (m)')

        frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig_frame)

    print(f"   Saved {num_frames} frames to: {frames_dir}")

    # Create summary figure
    print("6. Creating summary figure...")
    fig_summary, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig_summary.suptitle('Elephant Hide Texture Evolution Summary', fontsize=14, fontweight='bold')

    key_frames = [0, num_frames//5, 2*num_frames//5, 3*num_frames//5, 4*num_frames//5, num_frames-1]
    for idx, (ax, frame_idx) in enumerate(zip(axes.flat, key_frames)):
        im = ax.imshow(frames[frame_idx], cmap='terrain', interpolation='bilinear',
                      extent=[0, 200, 0, 200], origin='lower')
        ax.set_title(f'{time_points[frame_idx]:.2f} Myr', fontsize=11)
        ax.set_xlabel('Distance (m)', fontsize=9)
        ax.set_ylabel('Distance (m)', fontsize=9)
        plt.colorbar(im, ax=ax)

    summary_path = os.path.join(output_dir, 'evolution_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig_summary)
    print(f"   Saved to: {summary_path}")

    print("\n" + "=" * 70)
    print("Animation Complete!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Animation: {animation_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Frames: {frames_dir}/ ({num_frames} images)")
    print("\nThis animation demonstrates the conceptual formation of elephant hide")
    print("textures on lunar slopes over geological timescales (3 million years).")
    print("=" * 70)

    return {
        'animation_path': animation_path,
        'summary_path': summary_path,
        'frames_dir': frames_dir
    }

if __name__ == "__main__":
    create_animation()

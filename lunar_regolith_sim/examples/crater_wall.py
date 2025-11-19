"""
Crater wall simulation.

This example simulates regolith flow on a crater wall, where
elephant hide textures are commonly observed on the Moon.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lunar_regolith_sim import (
    RegolithPhysics,
    SlopeGeometry,
    RegolithFlowSimulation,
    SimulationVisualizer
)


def main():
    print("=" * 60)
    print("Lunar Regolith Flow Simulation - Crater Wall")
    print("=" * 60)

    # Create crater wall geometry
    print("\n1. Creating crater wall geometry...")
    slope = SlopeGeometry(width=200, height=200, resolution=1.0)

    # Create a crater with steep walls
    slope.create_crater_wall(
        crater_x=100,       # Center X position
        crater_y=100,       # Center Y position
        inner_radius=30,    # Crater floor radius
        outer_radius=80,    # Crater rim radius
        rim_height=15,      # Rim elevation
        floor_depth=5       # Floor depression
    )

    # Add surface roughness
    slope.add_roughness(amplitude=0.3, wavelength=3.0, smoothing=1.5)

    print(f"   Grid size: {slope.nx} x {slope.ny}")
    print(f"   Crater inner radius: 30 m")
    print(f"   Crater outer radius: 80 m")

    # Initialize physics
    print("\n2. Initializing physics engine...")
    physics = RegolithPhysics()
    print(f"   Gravity: {physics.gravity} m/s²")
    print(f"   Angle of repose: {physics.angle_of_repose}°")

    # Create simulation with thicker regolith on crater walls
    print("\n3. Setting up simulation...")
    sim = RegolithFlowSimulation(
        slope_geometry=slope,
        physics=physics,
        initial_thickness=2.0
    )
    print(f"   Initial regolith thickness: 2.0 m")

    # Run simulation for longer duration (crater walls need time to develop textures)
    print("\n4. Running simulation...")
    duration = 2000  # seconds
    print(f"   Duration: {duration} s")
    history = sim.run(duration, progress_interval=200)
    print(f"   Simulation complete!")

    # Visualize results
    print("\n5. Creating visualizations...")
    viz = SimulationVisualizer(sim)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'crater_wall')
    os.makedirs(output_dir, exist_ok=True)

    # Generate all plots
    print("   - Elevation map...")
    viz.plot_elevation(save_path=os.path.join(output_dir, 'elevation.png'), show=False)

    print("   - Slope angle distribution...")
    viz.plot_slope_angle(save_path=os.path.join(output_dir, 'slope_angle.png'), show=False)

    print("   - Elephant hide texture...")
    viz.plot_elephant_hide(save_path=os.path.join(output_dir, 'elephant_hide.png'), show=False)

    print("   - Flow velocity field...")
    viz.plot_flow_velocity(save_path=os.path.join(output_dir, 'velocity.png'), show=False)

    print("   - 3D surface view...")
    viz.plot_3d_surface(save_path=os.path.join(output_dir, 'surface_3d.png'), show=False)

    print("   - Summary figure...")
    viz.create_summary_figure(save_path=os.path.join(output_dir, 'summary.png'), show=False)

    print(f"\n6. Results saved to: {output_dir}")

    # Print some statistics
    print("\n7. Simulation Statistics:")
    print(f"   Final simulation time: {sim.time:.1f} s")
    print(f"   Mean regolith thickness: {sim.thickness.mean():.3f} m")
    print(f"   Max flow events: {int(sim.flow_count.max())}")
    print(f"   Total deformation: {sim.cumulative_deformation.sum():.2f}")

    print("\nSimulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Simple linear slope simulation.

This example demonstrates basic regolith flow on a linear slope
and generates elephant hide texture patterns.
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
    print("Lunar Regolith Flow Simulation - Simple Slope")
    print("=" * 60)

    # Create slope geometry
    print("\n1. Creating slope geometry...")
    slope = SlopeGeometry(width=100, height=100, resolution=1.0)
    slope.create_linear_slope(angle=32, direction='y')
    slope.add_roughness(amplitude=0.2, wavelength=5.0)
    print(f"   Grid size: {slope.nx} x {slope.ny}")
    print(f"   Slope angle: 32 degrees")

    # Initialize physics (lunar conditions)
    print("\n2. Initializing physics engine...")
    physics = RegolithPhysics()
    print(f"   Gravity: {physics.gravity} m/s²")
    print(f"   Angle of repose: {physics.angle_of_repose}°")

    # Create simulation
    print("\n3. Setting up simulation...")
    sim = RegolithFlowSimulation(
        slope_geometry=slope,
        physics=physics,
        initial_thickness=1.5
    )
    print(f"   Initial regolith thickness: 1.5 m")
    print(f"   Time step: {sim.timestep} s")

    # Run simulation
    print("\n4. Running simulation...")
    duration = 1000  # seconds
    print(f"   Duration: {duration} s")
    history = sim.run(duration, progress_interval=100)
    print(f"   Simulation complete!")
    print(f"   Total steps: {int(duration / sim.timestep)}")

    # Visualize results
    print("\n5. Creating visualizations...")
    viz = SimulationVisualizer(sim)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'simple_slope')
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
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
    print("\nSimulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

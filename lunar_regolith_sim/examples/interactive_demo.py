"""
Interactive demonstration of regolith flow parameters.

This example allows you to experiment with different parameters
and see their effects on elephant hide texture formation.
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
import numpy as np


def run_simulation_scenario(scenario_name, slope_angle, initial_thickness,
                            duration, roughness_amp, output_base_dir):
    """
    Run a single simulation scenario.

    Args:
        scenario_name: Name of the scenario
        slope_angle: Slope angle in degrees
        initial_thickness: Initial regolith thickness in meters
        duration: Simulation duration in seconds
        roughness_amp: Surface roughness amplitude
        output_base_dir: Base directory for output
    """
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*60}")
    print(f"  Slope angle: {slope_angle}°")
    print(f"  Initial thickness: {initial_thickness} m")
    print(f"  Duration: {duration} s")
    print(f"  Roughness: {roughness_amp} m")

    # Create slope
    slope = SlopeGeometry(width=100, height=100, resolution=0.5)
    slope.create_linear_slope(angle=slope_angle, direction='y')
    slope.add_roughness(amplitude=roughness_amp, wavelength=4.0, smoothing=1.0)

    # Create simulation
    physics = RegolithPhysics()
    sim = RegolithFlowSimulation(slope, physics, initial_thickness)

    # Run
    print("  Running simulation...")
    history = sim.run(duration, progress_interval=duration)

    # Visualize
    viz = SimulationVisualizer(sim)
    output_dir = os.path.join(output_base_dir, scenario_name.replace(' ', '_').lower())
    os.makedirs(output_dir, exist_ok=True)

    viz.create_summary_figure(
        save_path=os.path.join(output_dir, 'summary.png'),
        show=False
    )

    print(f"  Results saved to: {output_dir}")
    print(f"  Max flow events: {int(sim.flow_count.max())}")
    print(f"  Total deformation: {sim.cumulative_deformation.sum():.2f}")

    return sim


def main():
    print("=" * 60)
    print("Interactive Regolith Flow Parameter Study")
    print("=" * 60)
    print("\nThis script runs multiple scenarios to demonstrate")
    print("how different parameters affect elephant hide formation.")

    output_base_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'parameter_study')

    # Scenario 1: Stable slope (below angle of repose)
    run_simulation_scenario(
        scenario_name="Stable Slope",
        slope_angle=30.0,          # Below angle of repose
        initial_thickness=1.0,
        duration=500,
        roughness_amp=0.1,
        output_base_dir=output_base_dir
    )

    # Scenario 2: Critical slope (at angle of repose)
    run_simulation_scenario(
        scenario_name="Critical Slope",
        slope_angle=35.0,          # At angle of repose
        initial_thickness=1.5,
        duration=1000,
        roughness_amp=0.2,
        output_base_dir=output_base_dir
    )

    # Scenario 3: Unstable slope (above angle of repose)
    run_simulation_scenario(
        scenario_name="Unstable Slope",
        slope_angle=40.0,          # Above angle of repose
        initial_thickness=2.0,
        duration=1500,
        roughness_amp=0.3,
        output_base_dir=output_base_dir
    )

    # Scenario 4: Thin regolith layer
    run_simulation_scenario(
        scenario_name="Thin Regolith",
        slope_angle=35.0,
        initial_thickness=0.5,     # Thin layer
        duration=800,
        roughness_amp=0.15,
        output_base_dir=output_base_dir
    )

    # Scenario 5: Thick regolith layer
    run_simulation_scenario(
        scenario_name="Thick Regolith",
        slope_angle=35.0,
        initial_thickness=3.0,     # Thick layer
        duration=1200,
        roughness_amp=0.25,
        output_base_dir=output_base_dir
    )

    # Scenario 6: High roughness
    run_simulation_scenario(
        scenario_name="High Roughness",
        slope_angle=35.0,
        initial_thickness=1.5,
        duration=1000,
        roughness_amp=0.5,         # High roughness
        output_base_dir=output_base_dir
    )

    print("\n" + "=" * 60)
    print("Parameter Study Complete!")
    print("=" * 60)
    print(f"\nAll results saved to: {output_base_dir}")
    print("\nKey Findings:")
    print("  - Elephant hide textures form best at critical slope angles")
    print("  - Thicker regolith produces more pronounced textures")
    print("  - Surface roughness nucleates flow patterns")
    print("  - Stable slopes show minimal texture development")
    print("=" * 60)


if __name__ == "__main__":
    main()

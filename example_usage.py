#!/usr/bin/env python3
"""
Example usage script for crater age analysis.

This script demonstrates how to use the CraterAgeAnalyzer class
with your lunar topography, image, and crater rim data.
"""

from crater_age_analysis import CraterAgeAnalyzer
import geopandas as gpd
import matplotlib.pyplot as plt
import os


def example_basic_usage():
    """Basic usage example."""
    print("="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)

    # Initialize analyzer
    analyzer = CraterAgeAnalyzer(
        topo_path='path/to/your/topography.tif',
        image_path='path/to/your/lunar_image.tif',
        shapefile_path='path/to/your/crater_rims.shp',
        pixel_size_meters=5.0  # Optional: specify if needed
    )

    # Process all craters
    print("\nProcessing craters...")
    results = analyzer.process_all_craters()

    # Save results
    print("\nSaving results...")
    analyzer.save_results(results, 'output/crater_ages.shp')

    # Create visualization
    print("\nCreating visualization...")
    analyzer.visualize_results(results, 'output/crater_ages_map.png')

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total craters analyzed: {len(results)}")
    print(f"Mean diameter: {results['diameter_m'].mean():.2f} m")
    print(f"Mean depth: {results['depth_m'].mean():.2f} m")
    print(f"Mean d/D ratio: {(results['depth_m'] / results['diameter_m']).mean():.4f}")

    # Clean up
    analyzer.close()

    return results


def example_batch_processing():
    """Example of processing craters in batches."""
    print("="*60)
    print("EXAMPLE 2: Batch Processing")
    print("="*60)

    analyzer = CraterAgeAnalyzer(
        topo_path='path/to/your/topography.tif',
        image_path='path/to/your/lunar_image.tif',
        shapefile_path='path/to/your/crater_rims.shp'
    )

    # Process craters one at a time with custom handling
    all_results = []

    for idx, row in analyzer.craters_gdf.iterrows():
        print(f"\nProcessing crater {idx + 1}/{len(analyzer.craters_gdf)}...")

        try:
            # Process single crater (you would implement this method)
            # For now, showing the concept
            print(f"  Crater ID: {idx}")
            print(f"  Bounds: {row.geometry.bounds}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    analyzer.close()


def example_custom_analysis():
    """Example with custom analysis and visualization."""
    print("="*60)
    print("EXAMPLE 3: Custom Analysis")
    print("="*60)

    analyzer = CraterAgeAnalyzer(
        topo_path='path/to/your/topography.tif',
        image_path='path/to/your/lunar_image.tif',
        shapefile_path='path/to/your/crater_rims.shp'
    )

    # Process craters
    results = analyzer.process_all_craters()

    # Custom analysis: age distribution
    print("\nAge Distribution:")
    age_counts = results['age'].value_counts()
    for age, count in age_counts.items():
        print(f"  {age}: {count} craters")

    # Custom analysis: size vs age
    print("\nSize-Age Analysis:")
    size_bins = [0, 100, 500, 1000, 5000, 100000]
    size_labels = ['<100m', '100-500m', '500-1000m', '1-5km', '>5km']

    results['size_bin'] = pd.cut(
        results['diameter_m'],
        bins=size_bins,
        labels=size_labels
    )

    for size_bin in size_labels:
        subset = results[results['size_bin'] == size_bin]
        if len(subset) > 0:
            print(f"  {size_bin}: {len(subset)} craters")

    # Create custom visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Diameter distribution
    axes[0, 0].hist(results['diameter_m'], bins=30, edgecolor='black')
    axes[0, 0].set_xlabel('Diameter (m)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Crater Diameter Distribution')

    # Plot 2: Depth vs Diameter
    axes[0, 1].scatter(results['diameter_m'], results['depth_m'], alpha=0.6)
    axes[0, 1].set_xlabel('Diameter (m)')
    axes[0, 1].set_ylabel('Depth (m)')
    axes[0, 1].set_title('Depth vs Diameter')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')

    # Plot 3: d/D ratio distribution
    d_D_ratio = results['depth_m'] / results['diameter_m']
    axes[1, 0].hist(d_D_ratio.dropna(), bins=30, edgecolor='black')
    axes[1, 0].set_xlabel('d/D Ratio')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Depth/Diameter Ratio Distribution')

    # Plot 4: Age distribution
    age_counts.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Age Category')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Age Distribution')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('output/custom_analysis.png', dpi=300)
    print("\nCustom analysis saved to: output/custom_analysis.png")

    # Save results
    analyzer.save_results(results, 'output/crater_ages_custom.shp')
    analyzer.close()

    return results


def example_single_crater_detailed():
    """Detailed analysis of a single crater."""
    print("="*60)
    print("EXAMPLE 4: Single Crater Detailed Analysis")
    print("="*60)

    analyzer = CraterAgeAnalyzer(
        topo_path='path/to/your/topography.tif',
        image_path='path/to/your/lunar_image.tif',
        shapefile_path='path/to/your/crater_rims.shp'
    )

    # Get first crater
    first_crater = analyzer.craters_gdf.iloc[0]
    geometry = first_crater.geometry

    print("\nAnalyzing single crater in detail...")

    # This demonstrates the step-by-step process
    # (You would need to extract relevant data first)

    print("Steps:")
    print("  1. Refining rim position...")
    print("  2. Fitting circle to rim...")
    print("  3. Extracting crater region...")
    print("  4. Correcting tilt...")
    print("  5. Extracting radial profiles...")
    print("  6. Estimating age...")

    analyzer.close()


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("CRATER AGE ANALYSIS - EXAMPLE USAGE")
    print("="*60 + "\n")

    # Check if output directory exists
    if not os.path.exists('output'):
        print("Creating output directory...")
        os.makedirs('output')

    # Note: These examples assume you have the data files
    # Replace the paths with your actual data files

    print("\nNOTE: Before running these examples, replace the file paths")
    print("in each example function with your actual data file paths.")
    print("\nExample paths to replace:")
    print("  - 'path/to/your/topography.tif'")
    print("  - 'path/to/your/lunar_image.tif'")
    print("  - 'path/to/your/crater_rims.shp'")

    # Uncomment the example you want to run:

    # results = example_basic_usage()
    # example_batch_processing()
    # results = example_custom_analysis()
    # example_single_crater_detailed()

    print("\n" + "="*60)
    print("Examples complete! Check the output directory for results.")
    print("="*60)


if __name__ == '__main__':
    # Import pandas for custom analysis example
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not installed. Some examples may not work.")

    main()

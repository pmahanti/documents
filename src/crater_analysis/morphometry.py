"""Crater morphometry analysis module."""

import numpy as np
import geopandas as gpd
from uncertainties import unumpy

from . import cratools


def compute_depth_diameter_ratios(input_shapefile, output_shapefile,
                                   dem_path, orthophoto_path,
                                   min_diameter=60, remove_external_topo=True,
                                   plot=True):
    """
    Compute depth-to-diameter ratios for craters.

    This function analyzes refined crater geometries and computes morphometric
    parameters including depth, diameter, d/D ratio, rim height, and floor elevation.

    Args:
        input_shapefile: Path to input shapefile with refined crater geometries
        output_shapefile: Path to save crater data with morphometry
        dem_path: Path to Digital Elevation Model file
        orthophoto_path: Path to orthophoto image file
        min_diameter: Minimum crater diameter to process (meters)
        remove_external_topo: Whether to remove regional topography
        plot: Whether to generate diagnostic plots

    Returns:
        GeoDataFrame: Crater data with morphometric measurements
    """
    df = gpd.read_file(input_shapefile)
    crs = df.crs

    # Initialize result containers
    dd_ratios = []
    diameters = []
    depths = []
    rim_heights = []
    floor_elevations = []

    print(f"Computing morphometry for {len(df)} craters...")

    # Process each crater
    for idx, crater_geom in enumerate(df['geometry']):
        radius = np.sqrt(crater_geom.area / np.pi)
        diameter = radius * 2

        print(f"Processing crater {idx + 1}/{len(df)}: D={diameter:.1f}m")

        # Get radius error from refinement
        r_err = df['err_r'].iloc[idx] if 'err_r' in df.columns else 0.1

        # Compute morphometry
        ratio, depth, diam, rim, floor = cratools.compute_depth_diameter_ratio(
            geom=crater_geom,
            filename_dem=dem_path,
            crs=crs,
            orthophoto=orthophoto_path,
            diam_err=2 * r_err,
            remove_ext=remove_external_topo,
            plot=plot
        )

        # Store results
        dd_ratios.append(ratio)
        diameters.append(diam)
        depths.append(depth)
        rim_heights.append(rim)
        floor_elevations.append(floor)

    # Convert uncertainty arrays to nominal and std dev
    dd_array = np.array(dd_ratios)
    dd_nominal = unumpy.nominal_values(dd_array)
    dd_errors = unumpy.std_devs(dd_array)

    depth_array = np.array(depths)
    depth_nominal = unumpy.nominal_values(depth_array)
    depth_errors = unumpy.std_devs(depth_array)

    diam_nominal = unumpy.nominal_values(np.array(diameters))
    rim_nominal = unumpy.nominal_values(np.array(rim_heights))
    floor_nominal = unumpy.nominal_values(np.array(floor_elevations))

    # Add morphometry data to dataframe
    df['diam'] = diam_nominal
    df['d_D'] = dd_nominal
    df['d_D_err'] = dd_errors
    df['depth'] = depth_nominal
    df['depth_err'] = depth_errors
    df['rim_JKA'] = rim_nominal
    df['floor_JKA'] = floor_nominal

    print("\nMorphometry results:")
    print(df[['diam', 'd_D', 'd_D_err', 'depth', 'depth_err']].head())

    # Save results
    df.to_file(output_shapefile)
    print(f"\nSaved morphometry data to: {output_shapefile}")

    return df


def export_morphometry_to_csv(shapefile_path, csv_path):
    """
    Export morphometry data from shapefile to CSV format.

    Args:
        shapefile_path: Path to shapefile with morphometry data
        csv_path: Path to save CSV file

    Returns:
        DataFrame: Morphometry data as pandas DataFrame
    """
    gdf = gpd.read_file(shapefile_path)

    # Select relevant columns (exclude geometry)
    columns = ['UFID', 'Diam_m', 'diam', 'd_D', 'd_D_err',
               'depth', 'depth_err', 'rim_JKA', 'floor_JKA']

    # Filter to available columns
    available_cols = [col for col in columns if col in gdf.columns]

    df = gdf[available_cols]
    df.to_csv(csv_path, index=False)

    print(f"Exported morphometry data to: {csv_path}")
    return df


def get_morphometry_summary(shapefile_path):
    """
    Generate summary statistics for crater morphometry.

    Args:
        shapefile_path: Path to shapefile with morphometry data

    Returns:
        dict: Summary statistics
    """
    gdf = gpd.read_file(shapefile_path)

    summary = {
        'count': len(gdf),
        'diameter': {
            'mean': gdf['diam'].mean() if 'diam' in gdf.columns else None,
            'std': gdf['diam'].std() if 'diam' in gdf.columns else None,
            'min': gdf['diam'].min() if 'diam' in gdf.columns else None,
            'max': gdf['diam'].max() if 'diam' in gdf.columns else None,
        },
        'd_D_ratio': {
            'mean': gdf['d_D'].mean() if 'd_D' in gdf.columns else None,
            'std': gdf['d_D'].std() if 'd_D' in gdf.columns else None,
            'min': gdf['d_D'].min() if 'd_D' in gdf.columns else None,
            'max': gdf['d_D'].max() if 'd_D' in gdf.columns else None,
        },
        'depth': {
            'mean': gdf['depth'].mean() if 'depth' in gdf.columns else None,
            'std': gdf['depth'].std() if 'depth' in gdf.columns else None,
            'min': gdf['depth'].min() if 'depth' in gdf.columns else None,
            'max': gdf['depth'].max() if 'depth' in gdf.columns else None,
        }
    }

    return summary

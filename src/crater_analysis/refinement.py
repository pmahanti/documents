"""Crater rim refinement module."""

import numpy as np
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt
import rasterio.plot

from . import cratools


def update_crater_rims(input_shapefile, output_shapefile, dem_path,
                       orthophoto_path, min_diameter=60,
                       inner_radius=0.8, outer_radius=1.2,
                       plot=True, remove_external_topo=True):
    """
    Refine crater rim positions using topographic data.

    This function reads a shapefile of crater locations, refines their rim
    positions by analyzing the DEM topography, and saves the updated craters
    with error estimates.

    Args:
        input_shapefile: Path to input shapefile with crater geometries
        output_shapefile: Path to save refined crater shapefile
        dem_path: Path to Digital Elevation Model file
        orthophoto_path: Path to orthophoto image file
        min_diameter: Minimum crater diameter to process (meters)
        inner_radius: Inner search radius for rim (fraction of crater radius)
        outer_radius: Outer search radius for rim (fraction of crater radius)
        plot: Whether to generate diagnostic plots
        remove_external_topo: Whether to remove regional topography

    Returns:
        GeoDataFrame: Refined crater data with error estimates
    """
    with rio.open(dem_path, 'r') as src:
        # Initialize result containers
        updated_craters = []
        metadata = {
            'UFID': [],
            'Diam_m': [],
            'davg': [],
            'old_dD': [],
            'rim_AJS': [],
            'fl_AJS': [],
            'err_x0': [],
            'err_y0': [],
            'err_r': []
        }

        # Read input craters
        df_crater = gpd.read_file(input_shapefile)
        df_crater = df_crater[df_crater['D_m'] > min_diameter]

        print(f"Processing {len(df_crater)} craters...")
        print(df_crater.head())

        crs = df_crater.crs

        # Process each crater
        for idx, geom in enumerate(df_crater['geometry']):
            radius = np.sqrt(geom.area / np.pi)
            diameter = radius * 2

            if diameter < min_diameter:
                continue

            print(f"Processing crater {idx + 1}/{len(df_crater)}: D={diameter:.1f}m")

            # Fit crater rim
            refined_geom, err = cratools.fit_crater_rim(
                geom=geom,
                dem_src=src,
                crs=crs,
                orthophoto=orthophoto_path,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                plot=plot,
                remove_ext=remove_external_topo
            )

            # Store results
            updated_craters.append(refined_geom)
            metadata['UFID'].append(df_crater['UFID'].iloc[idx])
            metadata['Diam_m'].append(df_crater['D_m'].iloc[idx])
            metadata['davg'].append(df_crater['davg'].iloc[idx])
            metadata['old_dD'].append(df_crater['davg_D'].iloc[idx])
            metadata['rim_AJS'].append(df_crater['rim'].iloc[idx])
            metadata['fl_AJS'].append(df_crater['fl'].iloc[idx])
            metadata['err_x0'].append(err[0])
            metadata['err_y0'].append(err[1])
            metadata['err_r'].append(err[2])

        # Create GeoDataFrame with results
        gdf_refined = gpd.GeoDataFrame(
            data=metadata,
            geometry=updated_craters,
            crs=crs
        )

        print("\nRefined crater data:")
        print(gdf_refined.head())

        # Generate overview plot
        if plot:
            fig, ax = plt.subplots(figsize=(12, 12))
            extent = [src.bounds[0], src.bounds[2], src.bounds[1], src.bounds[3]]
            rasterio.plot.show(src, extent=extent, ax=ax, cmap="pink")
            gdf_refined.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)
            ax.set_title("Refined Crater Rims")
            plt.tight_layout()
            plt.savefig(output_shapefile.replace('.shp', '_overview.png'))
            plt.show()

        # Save to file
        gdf_refined.to_file(output_shapefile)
        print(f"\nSaved refined craters to: {output_shapefile}")

        return gdf_refined


def prepare_crater_geometries(input_shapefile, output_shapefile, min_diameter=60):
    """
    Prepare crater geometries by buffering point locations to circles.

    Args:
        input_shapefile: Path to input shapefile with crater points
        output_shapefile: Path to save buffered geometries
        min_diameter: Minimum crater diameter threshold (meters)

    Returns:
        GeoDataFrame: Crater geometries as circles
    """
    df = gpd.read_file(input_shapefile)
    df = df[df['D_m'] > min_diameter]

    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'])

    # Buffer points to create circles with diameter D_m
    gdf['geometry'] = gdf.geometry.buffer(gdf['D_m'] / 2)

    gdf.to_file(output_shapefile)
    print(f"Prepared {len(gdf)} crater geometries")
    print(f"Saved to: {output_shapefile}")

    return gdf

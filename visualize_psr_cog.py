#!/usr/bin/env python3
"""
Visualize COG footprints and PSR polygons with spatial queries.

This script provides functionality to:
1. Visualize COG image footprints overlaid on PSR boundaries
2. Find all COG images that overlap with a specific PSR
3. Generate maps in polar stereographic projection

Usage:
    # Visualize a specific COG file
    python visualize_psr_cog.py --cog M012728826S.60m.COG.tif

    # Find all COGs overlapping a PSR
    python visualize_psr_cog.py --psr-id 1234

    # Custom output filename
    python visualize_psr_cog.py --cog M012728826S.60m.COG.tif --output my_map.png
"""

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os
import sys
from pathlib import Path
import argparse


class PSRCOGVisualizer:
    """Visualize and query PSR and COG footprint geodatabases."""

    def __init__(self, psr_db="psr_database.gpkg", cog_db="cog_footprints.gpkg"):
        """
        Initialize the visualizer with geodatabase paths.

        Parameters
        ----------
        psr_db : str
            Path to PSR geodatabase
        cog_db : str
            Path to COG footprints geodatabase
        """
        self.psr_db = psr_db
        self.cog_db = cog_db
        self.psr_data = None
        self.cog_data = None

        # Load databases
        self._load_databases()

    def _load_databases(self):
        """Load PSR and COG geodatabases."""
        print(f"Loading PSR database from {self.psr_db}...")
        if not os.path.exists(self.psr_db):
            raise FileNotFoundError(f"PSR database not found: {self.psr_db}")

        # Load both north and south PSR layers
        psr_north = gpd.read_file(self.psr_db, layer="psr_north")
        psr_south = gpd.read_file(self.psr_db, layer="psr_south")
        print(f"  Loaded {len(psr_north)} northern PSR polygons")
        print(f"  Loaded {len(psr_south)} southern PSR polygons")

        # Store both separately for later use
        self.psr_north = psr_north
        self.psr_south = psr_south

        print(f"Loading COG footprints database from {self.cog_db}...")
        if not os.path.exists(self.cog_db):
            raise FileNotFoundError(f"COG database not found: {self.cog_db}")

        self.cog_data = gpd.read_file(self.cog_db, layer="cog_footprints")
        print(f"  Loaded {len(self.cog_data)} COG footprints")

    def visualize_cog_on_psr(self, cog_filename, output_file=None, guard_band_km=1.0):
        """
        Visualize a COG footprint overlaid on PSR outlines.

        Parameters
        ----------
        cog_filename : str
            Name of the COG file (e.g., 'M012728826S.60m.COG.tif')
        output_file : str, optional
            Output PNG filename (default: auto-generated from COG name)
        guard_band_km : float
            Guard band around image extent in kilometers (default: 1.0)

        Returns
        -------
        str
            Path to the saved PNG file
        """
        print(f"\n{'='*60}")
        print(f"Visualizing COG: {cog_filename}")
        print(f"{'='*60}")

        # Find the COG footprint
        cog_row = self.cog_data[self.cog_data['filename'] == cog_filename]

        if len(cog_row) == 0:
            raise ValueError(f"COG file not found in database: {cog_filename}")

        cog_geom = cog_row.iloc[0].geometry
        cog_info = cog_row.iloc[0]

        # Determine hemisphere based on centroid Y coordinate
        centroid = cog_geom.centroid
        is_north = centroid.y > 0

        # Select appropriate PSR data and reproject COG if necessary
        if is_north:
            psr_data = self.psr_north
            hemisphere_name = "Northern"
        else:
            psr_data = self.psr_south
            hemisphere_name = "Southern"

        # Reproject COG footprint to match PSR CRS if needed
        if self.cog_data.crs != psr_data.crs:
            from shapely import geometry as shp_geom
            import pyproj
            from shapely.ops import transform

            # Create transformer
            project = pyproj.Transformer.from_crs(
                self.cog_data.crs, psr_data.crs, always_xy=True
            ).transform
            cog_geom = transform(project, cog_geom)

        # Calculate extent with guard band (in meters)
        guard_band_m = guard_band_km * 1000
        minx, miny, maxx, maxy = cog_geom.bounds
        extent = [
            minx - guard_band_m,
            maxx + guard_band_m,
            miny - guard_band_m,
            maxy + guard_band_m
        ]

        # Find PSRs that intersect with the extended extent
        from shapely.geometry import box
        extent_box = box(extent[0], extent[2], extent[1], extent[3])
        psr_subset = psr_data[psr_data.intersects(extent_box)]

        print(f"  Hemisphere: {hemisphere_name}")

        print(f"\nCOG Information:")
        print(f"  Area: {cog_info['area_km2']:.2f} km²")
        print(f"  Valid data fraction: {cog_info['valid_fraction']:.1%}")
        print(f"  Dimensions: {cog_info['width']} x {cog_info['height']} pixels")
        print(f"\nOverlapping PSRs: {len(psr_subset)}")

        # Create the visualization
        fig, ax = plt.subplots(figsize=(12, 12), dpi=150)

        # Plot PSRs
        if len(psr_subset) > 0:
            psr_subset.plot(
                ax=ax,
                facecolor='lightblue',
                edgecolor='darkblue',
                linewidth=1.5,
                alpha=0.6,
                label='PSR Regions'
            )

        # Plot COG footprint (now in the same CRS as PSR data)
        gpd.GeoSeries([cog_geom], crs=psr_data.crs).plot(
            ax=ax,
            facecolor='none',
            edgecolor='red',
            linewidth=2.5,
            linestyle='--',
            label='COG Footprint'
        )

        # Set extent
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        # Add grid
        ax.grid(True, linestyle=':', alpha=0.5)

        # Labels and title
        ax.set_xlabel('Easting (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Northing (m)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'COG Footprint on PSR Outlines\n{cog_filename}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        # Add legend
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

        # Add text box with statistics
        stats_text = (
            f"COG Area: {cog_info['area_km2']:.2f} km²\n"
            f"Valid Data: {cog_info['valid_fraction']:.1%}\n"
            f"Overlapping PSRs: {len(psr_subset)}\n"
            f"Guard Band: {guard_band_km} km"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        # Equal aspect ratio
        ax.set_aspect('equal')

        # Tight layout
        plt.tight_layout()

        # Save figure
        if output_file is None:
            base_name = os.path.splitext(cog_filename)[0]
            output_file = f"{base_name}_on_psr.png"

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {os.path.abspath(output_file)}")
        plt.close()

        return os.path.abspath(output_file)

    def find_cogs_for_psr(self, psr_id, output_file=None):
        """
        Find all COG images that overlap with a specific PSR.

        Parameters
        ----------
        psr_id : int or str
            PSR ID to query (column name depends on shapefile attributes)
        output_file : str, optional
            Output PNG filename (default: auto-generated from PSR ID)

        Returns
        -------
        tuple
            (list of COG filenames, path to saved PNG file)
        """
        print(f"\n{'='*60}")
        print(f"Finding COGs for PSR ID: {psr_id}")
        print(f"{'='*60}")

        # Find the PSR in both hemispheres
        # First, try to find common ID column names
        psr_id_cols = ['PSR_ID', 'ID', 'FID', 'OBJECTID', 'id', 'psr_id']
        psr_col = None

        for col in psr_id_cols:
            if col in self.psr_north.columns:
                psr_col = col
                break

        if psr_col is None:
            print(f"\nAvailable columns in PSR database:")
            for col in self.psr_north.columns:
                if col != 'geometry':
                    print(f"  - {col}")
            raise ValueError(
                f"Could not find PSR ID column. "
                f"Please specify the correct column name."
            )

        # Convert psr_id to appropriate type
        try:
            psr_id_value = int(psr_id)
        except:
            psr_id_value = psr_id

        # Search in both hemispheres
        psr_row_north = self.psr_north[self.psr_north[psr_col] == psr_id_value]
        psr_row_south = self.psr_south[self.psr_south[psr_col] == psr_id_value]

        if len(psr_row_north) > 0:
            psr_row = psr_row_north
            psr_data = self.psr_north
            hemisphere_name = "Northern"
        elif len(psr_row_south) > 0:
            psr_row = psr_row_south
            psr_data = self.psr_south
            hemisphere_name = "Southern"
        else:
            raise ValueError(f"PSR with ID {psr_id} not found in database")

        psr_geom = psr_row.iloc[0].geometry
        psr_info = psr_row.iloc[0]

        print(f"  Hemisphere: {hemisphere_name}")

        print(f"\nPSR Information:")
        for col in psr_row.columns:
            if col != 'geometry':
                print(f"  {col}: {psr_info[col]}")

        # Reproject COG footprints to match PSR CRS if needed
        cog_data_reprojected = self.cog_data.copy()
        if self.cog_data.crs != psr_data.crs:
            print(f"\n  Reprojecting COG footprints to match PSR CRS...")
            cog_data_reprojected = cog_data_reprojected.to_crs(psr_data.crs)

        # Find overlapping COGs
        overlapping_cogs = cog_data_reprojected[cog_data_reprojected.intersects(psr_geom)].copy()

        cog_filenames = overlapping_cogs['filename'].tolist()

        print(f"\nFound {len(cog_filenames)} overlapping COG images:")
        for i, fname in enumerate(cog_filenames, 1):
            print(f"  {i}. {fname}")

        if len(overlapping_cogs) == 0:
            print(f"\nNo COG images overlap with PSR ID {psr_id}")
            return [], None

        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 12), dpi=150)

        # Calculate extent with buffer
        minx, miny, maxx, maxy = psr_geom.bounds
        buffer = max(maxx - minx, maxy - miny) * 0.2  # 20% buffer
        extent = [minx - buffer, maxx + buffer, miny - buffer, maxy + buffer]

        # Find all PSRs in the extent for context (from the same hemisphere)
        from shapely.geometry import box
        extent_box = box(extent[0], extent[2], extent[1], extent[3])
        psr_context = psr_data[psr_data.intersects(extent_box)]

        # Plot context PSRs
        psr_context.plot(
            ax=ax,
            facecolor='lightgray',
            edgecolor='gray',
            linewidth=0.5,
            alpha=0.3,
            label='Other PSRs'
        )

        # Plot the target PSR
        gpd.GeoSeries([psr_geom], crs=psr_data.crs).plot(
            ax=ax,
            facecolor='lightblue',
            edgecolor='darkblue',
            linewidth=2,
            alpha=0.7,
            label=f'Target PSR (ID: {psr_id})'
        )

        # Plot overlapping COG footprints with different colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(overlapping_cogs)))
        for idx, (_, cog_row) in enumerate(overlapping_cogs.iterrows()):
            gpd.GeoSeries([cog_row.geometry], crs=cog_data_reprojected.crs).plot(
                ax=ax,
                facecolor=colors[idx],
                edgecolor='red',
                linewidth=1.5,
                alpha=0.5,
                label=f"COG: {cog_row['filename'][:15]}..."
            )

        # Set extent
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        # Add grid
        ax.grid(True, linestyle=':', alpha=0.5)

        # Labels and title
        ax.set_xlabel('Easting (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Northing (m)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'COG Images Overlapping PSR ID: {psr_id}\n'
            f'{len(overlapping_cogs)} COG images found',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        # Add legend (limit entries if too many)
        if len(overlapping_cogs) <= 10:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9, bbox_to_anchor=(1.02, 1))
        else:
            # Create custom legend with summary
            legend_elements = [
                Patch(facecolor='lightblue', edgecolor='darkblue', label=f'Target PSR (ID: {psr_id})'),
                Patch(facecolor='lightgray', edgecolor='gray', label='Other PSRs'),
                Patch(facecolor='red', alpha=0.5, label=f'{len(overlapping_cogs)} Overlapping COGs')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

        # Add text box with COG list (if not too many)
        if len(cog_filenames) <= 15:
            cog_list_text = "Overlapping COGs:\n" + "\n".join([f"{i}. {fn}" for i, fn in enumerate(cog_filenames, 1)])
            ax.text(
                0.02, 0.02, cog_list_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace'
            )

        # Equal aspect ratio
        ax.set_aspect('equal')

        # Tight layout
        plt.tight_layout()

        # Save figure
        if output_file is None:
            output_file = f"psr_{psr_id}_cog_overlap.png"

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {os.path.abspath(output_file)}")
        plt.close()

        return cog_filenames, os.path.abspath(output_file)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Visualize COG footprints and PSR polygons',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a specific COG file
  python visualize_psr_cog.py --cog M012728826S.60m.COG.tif

  # Find all COGs overlapping a PSR
  python visualize_psr_cog.py --psr-id 1234

  # Custom output filename and guard band
  python visualize_psr_cog.py --cog M012728826S.60m.COG.tif --output my_map.png --guard-band 2.0
        """
    )

    parser.add_argument('--cog', type=str, help='COG filename to visualize')
    parser.add_argument('--psr-id', type=str, help='PSR ID to query')
    parser.add_argument('--output', type=str, help='Output PNG filename')
    parser.add_argument('--guard-band', type=float, default=1.0,
                       help='Guard band around COG extent in km (default: 1.0)')
    parser.add_argument('--psr-db', type=str, default='psr_database.gpkg',
                       help='Path to PSR geodatabase')
    parser.add_argument('--cog-db', type=str, default='cog_footprints.gpkg',
                       help='Path to COG footprints geodatabase')

    args = parser.parse_args()

    # Check that at least one operation is specified
    if not args.cog and not args.psr_id:
        parser.print_help()
        print("\nError: Must specify either --cog or --psr-id")
        sys.exit(1)

    # Initialize visualizer
    try:
        viz = PSRCOGVisualizer(psr_db=args.psr_db, cog_db=args.cog_db)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the following scripts first:")
        print("  1. python create_psr_geodatabase.py")
        print("  2. python extract_cog_footprints.py")
        sys.exit(1)

    # Perform requested operation
    if args.cog:
        viz.visualize_cog_on_psr(args.cog, args.output, args.guard_band)

    if args.psr_id:
        cog_files, output_path = viz.find_cogs_for_psr(args.psr_id, args.output)


if __name__ == "__main__":
    main()

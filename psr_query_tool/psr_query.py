#!/usr/bin/env python3
"""
Lunar Permanently Shadowed Regions (PSR) Query Tool

This tool reads NASA's PSR shapefile data and queries PSRs within a specified
radius from a given lunar latitude and longitude.
"""

import argparse
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import transform
import pyproj
from typing import Optional, Tuple
import json


class PSRQuery:
    """Query tool for Lunar Permanently Shadowed Regions."""

    # Lunar radius in kilometers (mean radius)
    LUNAR_RADIUS_KM = 1737.4

    def __init__(self, data_path: str, use_parquet: bool = False):
        """
        Initialize the PSR query tool.

        Args:
            data_path: Path to the shapefile or parquet file
            use_parquet: Whether to use parquet format instead of shapefile
        """
        self.data_path = Path(data_path)
        self.use_parquet = use_parquet
        self.gdf = None
        self._load_data()

    def _load_data(self):
        """Load the PSR data from shapefile or parquet."""
        print(f"Loading PSR data from {self.data_path}...")

        if self.use_parquet:
            if not self.data_path.exists():
                raise FileNotFoundError(
                    f"Parquet file not found: {self.data_path}\n"
                    f"Run with --convert first to create the parquet file."
                )
            self.gdf = gpd.read_parquet(self.data_path)
        else:
            if not self.data_path.exists():
                raise FileNotFoundError(
                    f"Shapefile not found: {self.data_path}\n"
                    f"Please download the files from NASA PGDA."
                )
            self.gdf = gpd.read_file(self.data_path)

        print(f"Loaded {len(self.gdf)} PSR features")
        print(f"CRS: {self.gdf.crs}")
        print(f"Columns: {list(self.gdf.columns)}")

    def convert_to_parquet(self, output_path: Optional[str] = None):
        """
        Convert shapefile to GeoParquet format for space efficiency.

        Args:
            output_path: Output path for parquet file (optional)
        """
        if output_path is None:
            output_path = self.data_path.with_suffix('.parquet')
        else:
            output_path = Path(output_path)

        print(f"Converting to GeoParquet format...")
        print(f"Original size: {self._get_dir_size(self.data_path.parent)} MB")

        # Save as parquet
        self.gdf.to_parquet(output_path)

        output_size = output_path.stat().st_size / (1024 * 1024)
        print(f"Converted file saved to: {output_path}")
        print(f"Parquet size: {output_size:.2f} MB")
        print(f"Space savings: ~{self._calculate_savings(output_path)}%")

        return output_path

    def _get_dir_size(self, path: Path) -> float:
        """Get total size of shapefile components in MB."""
        total = 0
        base_name = self.data_path.stem
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            file_path = path / f"{base_name}{ext}"
            if file_path.exists():
                total += file_path.stat().st_size
        return total / (1024 * 1024)

    def _calculate_savings(self, parquet_path: Path) -> int:
        """Calculate space savings percentage."""
        original_size = self._get_dir_size(self.data_path.parent) * 1024 * 1024
        parquet_size = parquet_path.stat().st_size
        if original_size > 0:
            return int((1 - parquet_size / original_size) * 100)
        return 0

    def query_psrs(
        self,
        latitude: float,
        longitude: float,
        radius_km: float
    ) -> gpd.GeoDataFrame:
        """
        Query PSRs within a specified radius from a location.

        Args:
            latitude: Latitude in degrees (-90 to 90)
            longitude: Longitude in degrees (-180 to 180)
            radius_km: Search radius in kilometers

        Returns:
            GeoDataFrame containing PSRs within the radius
        """
        # Validate inputs
        if not -90 <= latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        if not -180 <= longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        if radius_km <= 0:
            raise ValueError("Radius must be positive")

        print(f"\nQuerying PSRs within {radius_km} km of ({latitude}°, {longitude}°)")

        # Create a point for the query location
        query_point = Point(longitude, latitude)

        # Convert the GeoDataFrame to a projected CRS if needed
        # For lunar data, we need to handle the coordinate system carefully
        if self.gdf.crs is None:
            print("Warning: No CRS defined, assuming EPSG:4326 (lat/lon)")
            self.gdf = self.gdf.set_crs("EPSG:4326")

        # Calculate distances using spherical geometry on the Moon
        # Create a GeoSeries with just the query point
        query_gdf = gpd.GeoDataFrame(
            geometry=[query_point],
            crs=self.gdf.crs
        )

        # Calculate great circle distances on lunar surface
        distances_km = self._calculate_lunar_distances(
            query_point,
            self.gdf.geometry
        )

        # Add distance column
        result_gdf = self.gdf.copy()
        result_gdf['distance_km'] = distances_km

        # Filter by radius
        result_gdf = result_gdf[result_gdf['distance_km'] <= radius_km]

        # Sort by distance
        result_gdf = result_gdf.sort_values('distance_km')

        return result_gdf

    def _calculate_lunar_distances(
        self,
        point: Point,
        geometries: gpd.GeoSeries
    ) -> pd.Series:
        """
        Calculate great circle distances on lunar surface.

        Args:
            point: Query point (lon, lat)
            geometries: GeoSeries of geometries to calculate distances to

        Returns:
            Series of distances in kilometers
        """
        # Extract lon/lat from query point
        lon1, lat1 = point.x, point.y

        # Get centroids of all PSR geometries
        centroids = geometries.centroid
        lon2 = centroids.x
        lat2 = centroids.y

        # Calculate great circle distance using Haversine formula
        # Convert to radians
        import numpy as np

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon = np.radians(lon2 - lon1)
        dlat = np.radians(lat2 - lat1)

        # Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # Distance in km
        distances = self.LUNAR_RADIUS_KM * c

        return pd.Series(distances, index=geometries.index)

    def display_results(self, results: gpd.GeoDataFrame):
        """
        Display query results in a readable format.

        Args:
            results: GeoDataFrame with query results
        """
        print(f"\n{'='*80}")
        print(f"Found {len(results)} PSR(s) within radius")
        print(f"{'='*80}\n")

        if len(results) == 0:
            print("No PSRs found in the specified area.")
            return

        # Display each PSR
        for idx, row in results.iterrows():
            print(f"PSR #{idx}")
            print(f"  Distance: {row['distance_km']:.2f} km")

            # Display available attributes
            for col in results.columns:
                if col not in ['geometry', 'distance_km']:
                    print(f"  {col}: {row[col]}")

            # Calculate area if geometry is polygon
            if row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                # Approximate area in km²
                # This is rough - for accurate area calculation we'd need proper projection
                area_deg2 = row.geometry.area
                # Very rough conversion for polar regions
                lat = row.geometry.centroid.y
                area_km2 = area_deg2 * (111.32 * np.cos(np.radians(lat)) * 111.32)
                print(f"  Approximate Area: {area_km2:.2f} km²")

            print()

    def export_results(
        self,
        results: gpd.GeoDataFrame,
        output_path: str,
        format: str = 'geojson'
    ):
        """
        Export query results to file.

        Args:
            results: GeoDataFrame with query results
            output_path: Output file path
            format: Output format ('geojson', 'csv', 'shapefile')
        """
        output_path = Path(output_path)

        if format == 'geojson':
            results.to_file(output_path, driver='GeoJSON')
        elif format == 'csv':
            # Drop geometry for CSV, include centroids as lat/lon
            df = pd.DataFrame(results.drop(columns='geometry'))
            centroids = results.geometry.centroid
            df['centroid_lon'] = centroids.x
            df['centroid_lat'] = centroids.y
            df.to_csv(output_path, index=False)
        elif format == 'shapefile':
            results.to_file(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Results exported to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Query Lunar Permanently Shadowed Regions (PSRs)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query PSRs within 50km of a location
  %(prog)s --lat -89.5 --lon 45.0 --radius 50

  # Convert shapefile to parquet
  %(prog)s --convert --input data/LPSR_80S_20MPP_ADJ.shp

  # Query using parquet format
  %(prog)s --lat -89.5 --lon 45.0 --radius 50 --use-parquet

  # Export results
  %(prog)s --lat -89.5 --lon 45.0 --radius 50 --output results.geojson
        """
    )

    parser.add_argument(
        '--input',
        default='data/LPSR_80S_20MPP_ADJ.shp',
        help='Input shapefile or parquet file path'
    )
    parser.add_argument(
        '--lat',
        type=float,
        help='Query latitude in degrees (-90 to 90)'
    )
    parser.add_argument(
        '--lon',
        type=float,
        help='Query longitude in degrees (-180 to 180)'
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=50.0,
        help='Search radius in kilometers (default: 50)'
    )
    parser.add_argument(
        '--convert',
        action='store_true',
        help='Convert shapefile to GeoParquet format'
    )
    parser.add_argument(
        '--use-parquet',
        action='store_true',
        help='Use parquet file instead of shapefile'
    )
    parser.add_argument(
        '--output',
        help='Output file path for results'
    )
    parser.add_argument(
        '--format',
        choices=['geojson', 'csv', 'shapefile'],
        default='geojson',
        help='Output format (default: geojson)'
    )

    args = parser.parse_args()

    # Adjust input path for parquet if needed
    input_path = args.input
    if args.use_parquet and not input_path.endswith('.parquet'):
        input_path = str(Path(input_path).with_suffix('.parquet'))

    try:
        # Initialize query tool
        psr = PSRQuery(input_path, use_parquet=args.use_parquet)

        # Handle conversion
        if args.convert:
            psr.convert_to_parquet()
            print("\nConversion complete!")
            return 0

        # Handle query
        if args.lat is not None and args.lon is not None:
            results = psr.query_psrs(args.lat, args.lon, args.radius)
            psr.display_results(results)

            # Export if requested
            if args.output and len(results) > 0:
                psr.export_results(results, args.output, args.format)
        else:
            parser.error("--lat and --lon are required for queries (or use --convert)")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

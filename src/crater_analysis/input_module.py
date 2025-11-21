"""
Input processing module for crater analysis.

This module handles:
- Reading contrast images (GeoTIFF, ISIS cube)
- Reading DTM/DEM data
- Reading crater location files (CSV, .diam)
- Coordinate conversions (lat/lon ↔ X/Y) for lunar projections
- Creating shapefile output
- Generating visualization PNGs
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio as rio
from rasterio.plot import show
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Moon radius in meters
MOON_RADIUS = 1737400.0


class CoordinateConverter:
    """Handle coordinate conversions for lunar projections."""

    def __init__(self, projection_type, center_lon=0.0, center_lat=0.0):
        """
        Initialize coordinate converter.

        Args:
            projection_type: 'equirectangular', 'stereographic', or 'orthographic'
            center_lon: Center longitude for projection (degrees)
            center_lat: Center latitude for projection (degrees)
        """
        self.projection_type = projection_type.lower()
        self.center_lon = np.radians(center_lon)
        self.center_lat = np.radians(center_lat)
        self.R = MOON_RADIUS

    def latlon_to_xy(self, lat, lon):
        """
        Convert latitude/longitude to projected X/Y coordinates.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            (x, y): Projected coordinates in meters
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        if self.projection_type == 'equirectangular':
            # Simple equidistant cylindrical
            x = self.R * (lon_rad - self.center_lon) * np.cos(self.center_lat)
            y = self.R * (lat_rad - self.center_lat)

        elif self.projection_type == 'stereographic':
            # Polar stereographic
            k = 2 * self.R / (1 + np.sin(self.center_lat) * np.sin(lat_rad) +
                              np.cos(self.center_lat) * np.cos(lat_rad) *
                              np.cos(lon_rad - self.center_lon))
            x = k * np.cos(lat_rad) * np.sin(lon_rad - self.center_lon)
            y = k * (np.cos(self.center_lat) * np.sin(lat_rad) -
                     np.sin(self.center_lat) * np.cos(lat_rad) *
                     np.cos(lon_rad - self.center_lon))

        elif self.projection_type == 'orthographic':
            # Orthographic projection
            x = self.R * np.cos(lat_rad) * np.sin(lon_rad - self.center_lon)
            y = self.R * (np.cos(self.center_lat) * np.sin(lat_rad) -
                         np.sin(self.center_lat) * np.cos(lat_rad) *
                         np.cos(lon_rad - self.center_lon))

        else:
            raise ValueError(f"Unknown projection: {self.projection_type}")

        return x, y

    def xy_to_latlon(self, x, y):
        """
        Convert projected X/Y coordinates to latitude/longitude.

        Args:
            x: X coordinate in meters
            y: Y coordinate in meters

        Returns:
            (lat, lon): Latitude and longitude in degrees
        """
        if self.projection_type == 'equirectangular':
            # Inverse equidistant cylindrical
            lon_rad = x / (self.R * np.cos(self.center_lat)) + self.center_lon
            lat_rad = y / self.R + self.center_lat

        elif self.projection_type == 'stereographic':
            # Inverse polar stereographic
            rho = np.sqrt(x**2 + y**2)
            c = 2 * np.arctan(rho / (2 * self.R))

            lat_rad = np.arcsin(np.cos(c) * np.sin(self.center_lat) +
                               (y * np.sin(c) * np.cos(self.center_lat)) / rho)
            lon_rad = self.center_lon + np.arctan2(
                x * np.sin(c),
                rho * np.cos(self.center_lat) * np.cos(c) -
                y * np.sin(self.center_lat) * np.sin(c)
            )

        elif self.projection_type == 'orthographic':
            # Inverse orthographic
            rho = np.sqrt(x**2 + y**2)
            c = np.arcsin(rho / self.R)

            lat_rad = np.arcsin(np.cos(c) * np.sin(self.center_lat) +
                               (y * np.sin(c) * np.cos(self.center_lat)) / rho)
            lon_rad = self.center_lon + np.arctan2(
                x * np.sin(c),
                rho * np.cos(self.center_lat) * np.cos(c) -
                y * np.sin(self.center_lat) * np.sin(c)
            )

        else:
            raise ValueError(f"Unknown projection: {self.projection_type}")

        lat = np.degrees(lat_rad)
        lon = np.degrees(lon_rad)

        return lat, lon


def read_crater_file(filepath, delimiter=','):
    """
    Read crater location file (CSV or .diam format).

    Args:
        filepath: Path to crater file
        delimiter: Column delimiter (default: ',')

    Returns:
        DataFrame with columns: coord1, coord2, diameter, coord_type
        coord_type is either 'latlon' or 'xy'
    """
    filepath = Path(filepath)

    # Read the file
    df = pd.read_csv(filepath, delimiter=delimiter)

    # Detect coordinate type from headers
    headers = [col.lower() for col in df.columns]

    # Check for lat/lon indicators
    has_lat = any(h in headers for h in ['lat', 'latitude', 'lat_deg', 'lat_d'])
    has_lon = any(h in headers for h in ['lon', 'long', 'longitude', 'lon_deg', 'lon_d'])

    # Check for x/y indicators
    has_x = any(h in headers for h in ['x', 'x_m', 'x_km', 'easting'])
    has_y = any(h in headers for h in ['y', 'y_m', 'y_km', 'northing'])

    # Check for diameter
    has_diam = any(h in headers for h in ['d', 'diam', 'diameter', 'd_m', 'd_km', 'diam_m', 'diam_km'])

    if not has_diam:
        # Assume third column is diameter
        print("Warning: Diameter column not clearly identified, using 3rd column")
        diameter_col = df.columns[2]
    else:
        # Find diameter column
        for i, h in enumerate(headers):
            if h in ['d', 'diam', 'diameter', 'd_m', 'd_km', 'diam_m', 'diam_km']:
                diameter_col = df.columns[i]
                break

    if has_lat and has_lon:
        coord_type = 'latlon'
        # Find lat/lon columns
        for i, h in enumerate(headers):
            if h in ['lat', 'latitude', 'lat_deg', 'lat_d']:
                coord1_col = df.columns[i]
            if h in ['lon', 'long', 'longitude', 'lon_deg', 'lon_d']:
                coord2_col = df.columns[i]
        print(f"Detected lat/lon coordinates: {coord1_col}, {coord2_col}")

    elif has_x and has_y:
        coord_type = 'xy'
        # Find x/y columns
        for i, h in enumerate(headers):
            if h in ['x', 'x_m', 'x_km', 'easting']:
                coord1_col = df.columns[i]
            if h in ['y', 'y_m', 'y_km', 'northing']:
                coord2_col = df.columns[i]
        print(f"Detected X/Y coordinates: {coord1_col}, {coord2_col}")

    else:
        # Assume first two columns are coordinates
        print("Warning: Coordinate type not clearly identified")
        print(f"Headers: {df.columns.tolist()}")
        coord1_col = df.columns[0]
        coord2_col = df.columns[1]

        # Guess based on values
        val1 = abs(df[coord1_col].iloc[0])
        val2 = abs(df[coord2_col].iloc[0])

        if val1 <= 90 and val2 <= 360:
            coord_type = 'latlon'
            print("Assuming lat/lon based on value ranges")
        else:
            coord_type = 'xy'
            print("Assuming X/Y based on value ranges")

    # Create standardized dataframe
    result = pd.DataFrame({
        'coord1': df[coord1_col],
        'coord2': df[coord2_col],
        'diameter': df[diameter_col],
        'coord_type': coord_type
    })

    # Convert diameter to meters if needed
    if 'km' in diameter_col.lower():
        print("Converting diameter from km to meters")
        result['diameter'] = result['diameter'] * 1000.0

    # Convert X/Y from km to meters if needed
    if coord_type == 'xy' and ('km' in coord1_col.lower() or 'km' in coord2_col.lower()):
        print("Converting X/Y from km to meters")
        result['coord1'] = result['coord1'] * 1000.0
        result['coord2'] = result['coord2'] * 1000.0

    return result


def read_isis_cube(filepath):
    """
    Read ISIS cube file.

    Note: This requires GDAL with ISIS driver or kalasiris/pysis packages.
    For now, returns error message if not GeoTIFF.

    Args:
        filepath: Path to ISIS cube file

    Returns:
        rasterio dataset or None
    """
    filepath = Path(filepath)

    if filepath.suffix.lower() == '.cub':
        # Try to open with rasterio (requires GDAL ISIS driver)
        try:
            src = rio.open(filepath)
            return src
        except Exception as e:
            print(f"Error reading ISIS cube: {e}")
            print("Note: ISIS cube support requires GDAL with ISIS driver")
            print("Consider converting to GeoTIFF using gdal_translate or isis2std")
            return None
    else:
        # Assume GeoTIFF
        return rio.open(filepath)


def create_crater_shapefile(crater_df, output_path, crs=None):
    """
    Create ESRI shapefile from crater data.

    Args:
        crater_df: DataFrame with lat, lon, x, y, diameter columns
        output_path: Path for output shapefile
        crs: Coordinate reference system (optional)

    Returns:
        GeoDataFrame
    """
    # Create point geometries
    if 'x' in crater_df.columns and 'y' in crater_df.columns:
        geometry = [Point(x, y) for x, y in zip(crater_df['x'], crater_df['y'])]
    else:
        raise ValueError("Crater dataframe must have 'x' and 'y' columns")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(crater_df, geometry=geometry, crs=crs)

    # Buffer to create circles
    gdf['geometry'] = gdf.geometry.buffer(gdf['diameter'] / 2)

    # Save shapefile
    gdf.to_file(output_path)
    print(f"Saved shapefile: {output_path}")

    return gdf


def plot_crater_locations(image_path, crater_gdf, output_path):
    """
    Create PNG showing crater locations on raster image.

    Args:
        image_path: Path to contrast image
        crater_gdf: GeoDataFrame with crater geometries
        output_path: Path for output PNG

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # Read and display image
    with rio.open(image_path) as src:
        show(src, ax=ax, cmap='gray')
        extent = [src.bounds.left, src.bounds.right,
                 src.bounds.bottom, src.bounds.top]

    # Plot crater circles
    crater_gdf.plot(ax=ax, facecolor='none', edgecolor='red',
                   linewidth=1, alpha=0.7)

    # Add title
    n_craters = len(crater_gdf)
    ax.set_title(f'Initial location of craters, N = {n_craters}',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved crater location plot: {output_path}")

    return fig


def plot_csfd(crater_df, output_path, area_km2=None):
    """
    Create CSFD (Crater Size-Frequency Distribution) plot.

    Args:
        crater_df: DataFrame with diameter column
        output_path: Path for output PNG
        area_km2: Survey area in km² (optional, for absolute frequency)

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get diameters in km
    diameters_km = crater_df['diameter'].values / 1000.0

    # Create diameter bins (log-spaced)
    min_d = np.floor(np.log10(diameters_km.min()))
    max_d = np.ceil(np.log10(diameters_km.max()))
    bins = np.logspace(min_d, max_d, 20)

    # Calculate cumulative counts
    counts, bin_edges = np.histogram(diameters_km, bins=bins)
    cumulative = np.cumsum(counts[::-1])[::-1]

    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate cumulative frequency
    if area_km2 is not None:
        # Absolute frequency (craters/km²)
        frequency = cumulative / area_km2
        ylabel = 'Cumulative Frequency (craters/km²)'
    else:
        # Relative frequency
        frequency = cumulative
        ylabel = 'Cumulative Count'

    # Plot
    ax.loglog(bin_centers, frequency, 'bo-', markersize=8, linewidth=2,
             label='Observed')

    # Reference isochrons (optional - would need craterstats integration)
    # This is a simplified version

    ax.set_xlabel('Diameter (km)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('Crater Size-Frequency Distribution (CSFD)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved CSFD plot: {output_path}")

    return fig


def process_crater_inputs(image_path, dtm_path, crater_file_path,
                          output_dir, projection='equirectangular',
                          center_lon=0.0, center_lat=0.0):
    """
    Main processing function for crater inputs.

    Args:
        image_path: Path to contrast image (GeoTIFF or ISIS cube)
        dtm_path: Path to DTM/DEM file
        crater_file_path: Path to crater location file (CSV or .diam)
        output_dir: Directory for outputs
        projection: Projection type ('equirectangular', 'stereographic', 'orthographic')
        center_lon: Center longitude for projection
        center_lat: Center latitude for projection

    Returns:
        dict with output paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Crater Input Processing Module")
    print("=" * 70)

    # Step 1: Read crater file
    print("\n[Step 1] Reading crater location file...")
    crater_df = read_crater_file(crater_file_path)
    print(f"  Loaded {len(crater_df)} craters")
    print(f"  Coordinate type: {crater_df['coord_type'].iloc[0]}")
    print(f"  Diameter range: {crater_df['diameter'].min():.1f} - "
          f"{crater_df['diameter'].max():.1f} m")

    # Step 2: Initialize coordinate converter
    print(f"\n[Step 2] Initializing {projection} projection...")
    converter = CoordinateConverter(projection, center_lon, center_lat)

    # Step 3: Convert coordinates if needed
    print("\n[Step 3] Processing coordinates...")
    coord_type = crater_df['coord_type'].iloc[0]

    if coord_type == 'latlon':
        print("  Converting lat/lon to X/Y...")
        crater_df['lat'] = crater_df['coord1']
        crater_df['lon'] = crater_df['coord2']
        crater_df['x'], crater_df['y'] = converter.latlon_to_xy(
            crater_df['lat'].values,
            crater_df['lon'].values
        )
    else:  # xy
        print("  Converting X/Y to lat/lon...")
        crater_df['x'] = crater_df['coord1']
        crater_df['y'] = crater_df['coord2']
        crater_df['lat'], crater_df['lon'] = converter.xy_to_latlon(
            crater_df['x'].values,
            crater_df['y'].values
        )

    print(f"  X range: {crater_df['x'].min():.1f} - {crater_df['x'].max():.1f} m")
    print(f"  Y range: {crater_df['y'].min():.1f} - {crater_df['y'].max():.1f} m")

    # Step 4: Read image and DTM
    print("\n[Step 4] Reading raster data...")
    image_src = read_isis_cube(image_path)
    dtm_src = rio.open(dtm_path)

    if image_src is None:
        print("  Warning: Could not read image file")
    else:
        print(f"  Image: {image_src.width} x {image_src.height} pixels")
        print(f"  Image CRS: {image_src.crs}")

    print(f"  DTM: {dtm_src.width} x {dtm_src.height} pixels")
    print(f"  DTM CRS: {dtm_src.crs}")

    # Step 5: Create shapefile
    print("\n[Step 5] Creating shapefile...")
    shapefile_path = output_dir / 'craters_initial.shp'
    crater_gdf = create_crater_shapefile(
        crater_df,
        shapefile_path,
        crs=dtm_src.crs if dtm_src else None
    )

    # Step 6: Create visualization plots
    print("\n[Step 6] Generating visualizations...")

    # Plot 1: Crater locations
    plot1_path = output_dir / 'craters_initial_locations.png'
    if image_src is not None:
        plot_crater_locations(image_path, crater_gdf, plot1_path)
    else:
        print("  Skipping location plot (no valid image)")

    # Plot 2: CSFD
    plot2_path = output_dir / 'craters_csfd.png'
    # Calculate area from image bounds
    if dtm_src:
        bounds = dtm_src.bounds
        area_m2 = (bounds.right - bounds.left) * (bounds.top - bounds.bottom)
        area_km2 = area_m2 / 1e6
    else:
        area_km2 = None

    plot_csfd(crater_df, plot2_path, area_km2=area_km2)

    # Cleanup
    if image_src:
        image_src.close()
    if dtm_src:
        dtm_src.close()

    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)

    results = {
        'shapefile': shapefile_path,
        'location_plot': plot1_path,
        'csfd_plot': plot2_path,
        'crater_count': len(crater_df),
        'crater_data': crater_df
    }

    print(f"\nOutputs:")
    print(f"  Shapefile: {shapefile_path}")
    print(f"  Location plot: {plot1_path}")
    print(f"  CSFD plot: {plot2_path}")
    print(f"  Total craters: {len(crater_df)}")

    return results

#!/usr/bin/env python3
"""
Generate a sample GeoTIFF with synthetic crater-like features for testing.
"""

import numpy as np
from osgeo import gdal, osr


def create_test_geotiff(output_path="test_crater_image.tif", width=1000, height=1000):
    """
    Create a test GeoTIFF with synthetic crater-like features.

    Args:
        output_path: Path to save the GeoTIFF
        width: Image width in pixels
        height: Image height in pixels
    """
    # Create a base image with random noise
    np.random.seed(42)
    image = np.random.randint(100, 150, (height, width), dtype=np.uint16)

    # Add some synthetic "craters" (dark circular features)
    craters = [
        (250, 250, 80),   # x, y, radius
        (600, 400, 120),
        (350, 700, 60),
        (800, 200, 90),
        (700, 750, 70),
    ]

    y_coords, x_coords = np.ogrid[:height, :width]

    for cx, cy, radius in craters:
        # Create circular depression
        distance = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)

        # Crater bowl (darker in center)
        crater_mask = distance <= radius
        crater_depth = (1 - distance[crater_mask] / radius) * 80
        image[crater_mask] = np.maximum(image[crater_mask] - crater_depth.astype(np.uint16), 50)

        # Crater rim (slightly brighter)
        rim_mask = (distance > radius) & (distance <= radius * 1.15)
        image[rim_mask] = np.minimum(image[rim_mask] + 30, 255)

    # Add some texture variation
    texture = np.random.randint(-20, 20, (height, width), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + texture, 0, 255).astype(np.uint16)

    # Create GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, width, height, 1, gdal.GDT_UInt16)

    if dataset is None:
        raise Exception(f"Failed to create {output_path}")

    # Set geotransform (simple equirectangular projection)
    # [top-left x, pixel width, rotation, top-left y, rotation, pixel height (negative)]
    geo_transform = [0.0, 1.0, 0.0, float(height), 0.0, -1.0]
    dataset.SetGeoTransform(geo_transform)

    # Set projection (simple Cartesian for testing)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')  # Using WGS84 geographic coordinates
    dataset.SetProjection(srs.ExportToWkt())

    # Write the data
    band = dataset.GetRasterBand(1)
    band.WriteArray(image)
    band.SetNoDataValue(0)

    # Close dataset
    dataset = None

    print(f"Created test GeoTIFF: {output_path}")
    print(f"Image size: {width} x {height}")
    print(f"Number of synthetic craters: {len(craters)}")
    print("\nSynthetic crater locations (for testing):")
    for i, (cx, cy, radius) in enumerate(craters, 1):
        print(f"  Crater {i}: center=({cx}, {cy}), radius={radius}, diameter={radius*2}")


if __name__ == "__main__":
    create_test_geotiff()

#!/usr/bin/env python3
"""
Crater Detection Training Data Generator

This script generates labeled training datasets for crater detection in lunar images
using YOLO, CNN, or DETR models. It processes GeoTiff or ISIS .cub files along with
shapefile annotations of known craters.

Usage:
    python crater_training_data_generator.py --image path/to/image.tif \
                                              --craters path/to/craters.shp \
                                              --output output_dir/ \
                                              --format yolo \
                                              --topography path/to/topo.tif (optional)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol, xy
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import warnings

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)


class CraterDatasetGenerator:
    """Generate training datasets for crater detection models."""

    def __init__(self, image_path: str, crater_shapefile: str, output_dir: str,
                 format_type: str = 'yolo', topography_path: Optional[str] = None,
                 tile_size: Optional[int] = None, overlap: int = 0):
        """
        Initialize the crater dataset generator.

        Args:
            image_path: Path to GeoTiff or ISIS .cub file
            crater_shapefile: Path to shapefile with crater annotations
            output_dir: Directory to save output files
            format_type: Output format ('yolo', 'coco', or 'both')
            topography_path: Optional path to topography raster
            tile_size: If specified, tile the image into smaller chunks
            overlap: Overlap between tiles in pixels (for tiling)
        """
        self.image_path = Path(image_path)
        self.crater_shapefile = Path(crater_shapefile)
        self.output_dir = Path(output_dir)
        self.format_type = format_type.lower()
        self.topography_path = Path(topography_path) if topography_path else None
        self.tile_size = tile_size
        self.overlap = overlap

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'labels').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)

        if self.topography_path:
            (self.output_dir / 'topography_images').mkdir(exist_ok=True)
            (self.output_dir / 'topography_labels').mkdir(exist_ok=True)

        # Load data
        self.image_src = None
        self.topo_src = None
        self.craters_gdf = None

    def load_data(self):
        """Load image and crater data."""
        print(f"Loading image from: {self.image_path}")

        try:
            self.image_src = rasterio.open(self.image_path)
            print(f"  Image size: {self.image_src.width} x {self.image_src.height}")
            print(f"  Bands: {self.image_src.count}")
            print(f"  CRS: {self.image_src.crs}")
        except Exception as e:
            print(f"Error loading image: {e}")
            sys.exit(1)

        print(f"\nLoading crater shapefile from: {self.crater_shapefile}")
        try:
            self.craters_gdf = gpd.read_file(self.crater_shapefile)
            print(f"  Total craters: {len(self.craters_gdf)}")
            print(f"  Shapefile CRS: {self.craters_gdf.crs}")

            # Reproject craters to image CRS if needed
            if self.craters_gdf.crs != self.image_src.crs:
                print(f"  Reprojecting craters to image CRS...")
                self.craters_gdf = self.craters_gdf.to_crs(self.image_src.crs)
        except Exception as e:
            print(f"Error loading shapefile: {e}")
            sys.exit(1)

        # Load topography if provided
        if self.topography_path:
            print(f"\nLoading topography from: {self.topography_path}")
            try:
                self.topo_src = rasterio.open(self.topography_path)
                print(f"  Topography size: {self.topo_src.width} x {self.topo_src.height}")
                print(f"  Topography CRS: {self.topo_src.crs}")
            except Exception as e:
                print(f"Error loading topography: {e}")
                self.topo_src = None

    def geometry_to_bbox(self, geometry, transform, img_width, img_height) -> Optional[Tuple[int, int, int, int]]:
        """
        Convert shapefile geometry to pixel bounding box.

        Args:
            geometry: Shapely geometry object
            transform: Rasterio transform
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Tuple of (xmin, ymin, xmax, ymax) in pixels, or None if out of bounds
        """
        bounds = geometry.bounds  # (minx, miny, maxx, maxy) in CRS coordinates

        # Convert to pixel coordinates
        row_min, col_min = rowcol(transform, bounds[0], bounds[3])  # top-left
        row_max, col_max = rowcol(transform, bounds[2], bounds[1])  # bottom-right

        xmin, ymin = col_min, row_min
        xmax, ymax = col_max, row_max

        # Ensure bbox is within image bounds
        xmin = max(0, min(xmin, img_width - 1))
        ymin = max(0, min(ymin, img_height - 1))
        xmax = max(0, min(xmax, img_width - 1))
        ymax = max(0, min(ymax, img_height - 1))

        # Check if bbox is valid
        if xmax <= xmin or ymax <= ymin:
            return None

        return (int(xmin), int(ymin), int(xmax), int(ymax))

    def bbox_to_yolo(self, bbox: Tuple[int, int, int, int],
                     img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert pixel bbox to YOLO format (normalized center coordinates and dimensions).

        Args:
            bbox: (xmin, ymin, xmax, ymax) in pixels
            img_width: Image width
            img_height: Image height

        Returns:
            (x_center, y_center, width, height) normalized to [0, 1]
        """
        xmin, ymin, xmax, ymax = bbox

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        return (x_center, y_center, width, height)

    def generate_full_image_dataset(self):
        """Generate dataset from the full image (no tiling)."""
        print("\n" + "="*60)
        print("Generating dataset from full image")
        print("="*60)

        # Read full image
        img_array = self.image_src.read()

        # Handle multi-band images (convert to RGB or grayscale)
        if img_array.shape[0] >= 3:
            # RGB
            img_array = np.stack([img_array[0], img_array[1], img_array[2]], axis=-1)
        else:
            # Grayscale - replicate to 3 channels for visualization
            img_array = np.stack([img_array[0], img_array[0], img_array[0]], axis=-1)

        # Normalize to 0-255 for saving
        img_array = self._normalize_image(img_array)

        img_height, img_width = img_array.shape[:2]

        # Save image
        img_name = self.image_path.stem
        img_save_path = self.output_dir / 'images' / f'{img_name}.png'
        Image.fromarray(img_array.astype(np.uint8)).save(img_save_path)
        print(f"Saved image: {img_save_path}")

        # Process craters
        annotations = []
        for idx, row in self.craters_gdf.iterrows():
            bbox = self.geometry_to_bbox(row.geometry, self.image_src.transform,
                                        img_width, img_height)
            if bbox:
                annotations.append(bbox)

        print(f"Found {len(annotations)} craters in full image")

        # Save labels
        if self.format_type in ['yolo', 'both']:
            self._save_yolo_labels(annotations, img_name, img_width, img_height)

        if self.format_type in ['coco', 'both']:
            self._save_coco_labels([(img_name, annotations, img_width, img_height)])

        # Generate visualization
        self._visualize_annotations(img_array, annotations, img_name)

        # Process topography if available
        if self.topo_src:
            self._process_topography_full_image(annotations, img_name)

    def generate_tiled_dataset(self):
        """Generate dataset by tiling the image into smaller chunks."""
        print("\n" + "="*60)
        print(f"Generating tiled dataset (tile size: {self.tile_size}x{self.tile_size}, overlap: {self.overlap})")
        print("="*60)

        img_width = self.image_src.width
        img_height = self.image_src.height
        stride = self.tile_size - self.overlap

        tile_info = []
        tile_count = 0

        for row_start in range(0, img_height, stride):
            for col_start in range(0, img_width, stride):
                # Calculate tile bounds
                row_end = min(row_start + self.tile_size, img_height)
                col_end = min(col_start + self.tile_size, img_width)

                # Adjust start if we're at the edge
                if row_end == img_height and row_end - row_start < self.tile_size:
                    row_start = max(0, img_height - self.tile_size)
                if col_end == img_width and col_end - col_start < self.tile_size:
                    col_start = max(0, img_width - self.tile_size)

                tile_width = col_end - col_start
                tile_height = row_end - row_start

                # Read tile
                window = Window(col_start, row_start, tile_width, tile_height)
                tile_array = self.image_src.read(window=window)

                # Handle multi-band
                if tile_array.shape[0] >= 3:
                    tile_array = np.stack([tile_array[0], tile_array[1], tile_array[2]], axis=-1)
                else:
                    tile_array = np.stack([tile_array[0], tile_array[0], tile_array[0]], axis=-1)

                tile_array = self._normalize_image(tile_array)

                # Get transform for this tile
                tile_transform = rasterio.windows.transform(window, self.image_src.transform)

                # Find craters in this tile
                tile_bbox = box(
                    *xy(tile_transform, 0, 0),
                    *xy(tile_transform, tile_height, tile_width)
                )

                craters_in_tile = self.craters_gdf[self.craters_gdf.intersects(tile_bbox)]

                annotations = []
                for idx, row in craters_in_tile.iterrows():
                    bbox = self.geometry_to_bbox(row.geometry, tile_transform,
                                                tile_width, tile_height)
                    if bbox:
                        annotations.append(bbox)

                # Only save tiles with craters (optional - can be changed)
                if len(annotations) > 0 or True:  # Set to True to save all tiles
                    tile_name = f"{self.image_path.stem}_tile_{tile_count:04d}"
                    tile_count += 1

                    # Save tile image
                    tile_save_path = self.output_dir / 'images' / f'{tile_name}.png'
                    Image.fromarray(tile_array.astype(np.uint8)).save(tile_save_path)

                    # Save labels
                    if self.format_type in ['yolo', 'both']:
                        self._save_yolo_labels(annotations, tile_name, tile_width, tile_height)

                    tile_info.append((tile_name, annotations, tile_width, tile_height))

                    # Visualize every 10th tile to avoid too many files
                    if tile_count % 10 == 0:
                        self._visualize_annotations(tile_array, annotations, tile_name)

                    print(f"Tile {tile_count}: {tile_name} - {len(annotations)} craters")

                    # Process topography tile
                    if self.topo_src:
                        self._process_topography_tile(window, annotations, tile_name,
                                                     tile_width, tile_height)

        if self.format_type in ['coco', 'both']:
            self._save_coco_labels(tile_info)

        print(f"\nGenerated {tile_count} tiles total")

    def _process_topography_full_image(self, annotations: List[Tuple], img_name: str):
        """Process full topography image."""
        print("\nProcessing topography data...")

        topo_array = self.topo_src.read(1)
        topo_array = self._normalize_image(topo_array)

        # Convert to RGB for visualization
        topo_rgb = np.stack([topo_array, topo_array, topo_array], axis=-1)

        # Save topography image
        topo_save_path = self.output_dir / 'topography_images' / f'{img_name}_topo.png'
        Image.fromarray(topo_rgb.astype(np.uint8)).save(topo_save_path)
        print(f"Saved topography image: {topo_save_path}")

        # Save labels (same annotations)
        if self.format_type in ['yolo', 'both']:
            label_path = self.output_dir / 'topography_labels' / f'{img_name}_topo.txt'
            with open(label_path, 'w') as f:
                for bbox in annotations:
                    yolo_bbox = self.bbox_to_yolo(bbox, self.topo_src.width,
                                                 self.topo_src.height)
                    f.write(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                           f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

    def _process_topography_tile(self, window: Window, annotations: List[Tuple],
                                 tile_name: str, tile_width: int, tile_height: int):
        """Process topography tile."""
        try:
            topo_tile = self.topo_src.read(1, window=window)
            topo_tile = self._normalize_image(topo_tile)

            topo_rgb = np.stack([topo_tile, topo_tile, topo_tile], axis=-1)

            topo_save_path = self.output_dir / 'topography_images' / f'{tile_name}_topo.png'
            Image.fromarray(topo_rgb.astype(np.uint8)).save(topo_save_path)

            if self.format_type in ['yolo', 'both']:
                self._save_yolo_labels(annotations, f'{tile_name}_topo',
                                      tile_width, tile_height,
                                      label_dir='topography_labels')
        except Exception as e:
            print(f"Warning: Could not process topography for {tile_name}: {e}")

    def _normalize_image(self, img_array: np.ndarray) -> np.ndarray:
        """Normalize image array to 0-255 range."""
        # Handle NaN and infinite values
        img_array = np.nan_to_num(img_array, nan=0.0, posinf=255.0, neginf=0.0)

        # Normalize to 0-255
        img_min = np.percentile(img_array, 2)  # Use percentile to handle outliers
        img_max = np.percentile(img_array, 98)

        if img_max - img_min > 0:
            img_array = (img_array - img_min) / (img_max - img_min) * 255
        else:
            img_array = np.zeros_like(img_array)

        img_array = np.clip(img_array, 0, 255)
        return img_array

    def _save_yolo_labels(self, annotations: List[Tuple], img_name: str,
                         img_width: int, img_height: int, label_dir: str = 'labels'):
        """Save annotations in YOLO format."""
        label_path = self.output_dir / label_dir / f'{img_name}.txt'
        with open(label_path, 'w') as f:
            for bbox in annotations:
                yolo_bbox = self.bbox_to_yolo(bbox, img_width, img_height)
                # Format: class_id x_center y_center width height (normalized)
                f.write(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                       f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

    def _save_coco_labels(self, tile_info: List[Tuple]):
        """Save annotations in COCO format."""
        coco_dict = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": "crater", "supercategory": "lunar_feature"}]
        }

        annotation_id = 0
        for img_id, (img_name, annotations, img_width, img_height) in enumerate(tile_info):
            coco_dict["images"].append({
                "id": img_id,
                "file_name": f"{img_name}.png",
                "width": img_width,
                "height": img_height
            })

            for bbox in annotations:
                xmin, ymin, xmax, ymax = bbox
                width = xmax - xmin
                height = ymax - ymin

                coco_dict["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": [xmin, ymin, width, height],  # COCO format: [x, y, w, h]
                    "area": width * height,
                    "iscrowd": 0
                })
                annotation_id += 1

        coco_path = self.output_dir / 'annotations_coco.json'
        with open(coco_path, 'w') as f:
            json.dump(coco_dict, f, indent=2)
        print(f"Saved COCO annotations: {coco_path}")

    def _visualize_annotations(self, img_array: np.ndarray,
                              annotations: List[Tuple], img_name: str):
        """Create visualization with bounding boxes."""
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(img_array.astype(np.uint8))

        for bbox in annotations:
            xmin, ymin, xmax, ymax = bbox
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        ax.set_title(f'{img_name} - {len(annotations)} craters', fontsize=14)
        ax.axis('off')

        vis_path = self.output_dir / 'visualizations' / f'{img_name}_labeled.png'
        plt.savefig(vis_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved visualization: {vis_path}")

    def generate_dataset_yaml(self):
        """Generate YAML configuration file for YOLO training."""
        yaml_content = f"""# Crater Detection Dataset Configuration
path: {self.output_dir.absolute()}
train: images
val: images  # You should split this into train/val sets

# Classes
names:
  0: crater

# Dataset info
nc: 1  # number of classes
"""
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"\nGenerated dataset.yaml: {yaml_path}")

    def run(self):
        """Run the dataset generation pipeline."""
        self.load_data()

        if self.tile_size:
            self.generate_tiled_dataset()
        else:
            self.generate_full_image_dataset()

        if self.format_type in ['yolo', 'both']:
            self.generate_dataset_yaml()

        self.close()

        print("\n" + "="*60)
        print("Dataset generation complete!")
        print("="*60)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"  - images/: Training images")
        print(f"  - labels/: YOLO format labels" if self.format_type in ['yolo', 'both'] else "")
        print(f"  - annotations_coco.json: COCO format labels" if self.format_type in ['coco', 'both'] else "")
        print(f"  - visualizations/: Labeled images for verification")
        if self.topo_src:
            print(f"  - topography_images/: Topography images")
            print(f"  - topography_labels/: Topography labels")

    def close(self):
        """Close opened file handles."""
        if self.image_src:
            self.image_src.close()
        if self.topo_src:
            self.topo_src.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate training data for crater detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate YOLO format dataset from full image
  python crater_training_data_generator.py \\
      --image lunar_image.tif \\
      --craters craters.shp \\
      --output ./crater_dataset \\
      --format yolo

  # Generate tiled dataset with topography
  python crater_training_data_generator.py \\
      --image lunar_image.cub \\
      --craters craters.shp \\
      --output ./crater_dataset \\
      --format both \\
      --tile-size 512 \\
      --overlap 64 \\
      --topography topography.tif
        """
    )

    parser.add_argument('--image', required=True,
                       help='Path to input image (GeoTiff or ISIS .cub file)')
    parser.add_argument('--craters', required=True,
                       help='Path to crater shapefile')
    parser.add_argument('--output', required=True,
                       help='Output directory for dataset')
    parser.add_argument('--format', choices=['yolo', 'coco', 'both'], default='yolo',
                       help='Output format (default: yolo)')
    parser.add_argument('--topography', default=None,
                       help='Optional topography raster file')
    parser.add_argument('--tile-size', type=int, default=None,
                       help='Tile size for splitting large images (e.g., 512)')
    parser.add_argument('--overlap', type=int, default=0,
                       help='Overlap between tiles in pixels (default: 0)')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    if not Path(args.craters).exists():
        print(f"Error: Crater shapefile not found: {args.craters}")
        sys.exit(1)

    if args.topography and not Path(args.topography).exists():
        print(f"Error: Topography file not found: {args.topography}")
        sys.exit(1)

    # Create generator and run
    generator = CraterDatasetGenerator(
        image_path=args.image,
        crater_shapefile=args.craters,
        output_dir=args.output,
        format_type=args.format,
        topography_path=args.topography,
        tile_size=args.tile_size,
        overlap=args.overlap
    )

    generator.run()


if __name__ == '__main__':
    main()

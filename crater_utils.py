#!/usr/bin/env python3
"""
Utility functions for crater detection dataset generation.

This module provides additional helper functions for image preprocessing,
data augmentation, and dataset splitting.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json
import shutil
from sklearn.model_selection import train_test_split


def split_dataset(dataset_dir: str, train_ratio: float = 0.8,
                 val_ratio: float = 0.1, test_ratio: float = 0.1,
                 seed: int = 42):
    """
    Split dataset into train/val/test sets.

    Args:
        dataset_dir: Directory containing images/ and labels/ folders
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed for reproducibility
    """
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'

    # Get all image files
    image_files = sorted(list(images_dir.glob('*.png')))
    image_names = [img.stem for img in image_files]

    # Split into train/val/test
    train_names, temp_names = train_test_split(
        image_names, test_size=(val_ratio + test_ratio), random_state=seed
    )

    if test_ratio > 0:
        val_names, test_names = train_test_split(
            temp_names, test_size=test_ratio/(val_ratio + test_ratio),
            random_state=seed
        )
    else:
        val_names = temp_names
        test_names = []

    # Create split directories
    splits = {
        'train': train_names,
        'val': val_names,
        'test': test_names
    }

    for split_name, names in splits.items():
        if len(names) == 0:
            continue

        split_img_dir = dataset_path / split_name / 'images'
        split_lbl_dir = dataset_path / split_name / 'labels'
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for name in names:
            # Copy image
            src_img = images_dir / f'{name}.png'
            dst_img = split_img_dir / f'{name}.png'
            if src_img.exists():
                shutil.copy(src_img, dst_img)

            # Copy label
            src_lbl = labels_dir / f'{name}.txt'
            dst_lbl = split_lbl_dir / f'{name}.txt'
            if src_lbl.exists():
                shutil.copy(src_lbl, dst_lbl)

    print(f"Dataset split complete:")
    print(f"  Train: {len(train_names)} images")
    print(f"  Val: {len(val_names)} images")
    print(f"  Test: {len(test_names)} images")

    # Update dataset.yaml
    update_dataset_yaml(dataset_path, splits)


def update_dataset_yaml(dataset_path: Path, splits: Dict[str, List[str]]):
    """Update dataset.yaml with train/val/test paths."""
    yaml_content = f"""# Crater Detection Dataset Configuration
path: {dataset_path.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: crater

# Dataset info
nc: 1  # number of classes

# Dataset statistics
train_count: {len(splits['train'])}
val_count: {len(splits['val'])}
test_count: {len(splits.get('test', []))}
"""
    yaml_path = dataset_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Updated {yaml_path}")


def calculate_dataset_statistics(dataset_dir: str) -> Dict:
    """
    Calculate statistics about the dataset.

    Args:
        dataset_dir: Directory containing labels/ folder

    Returns:
        Dictionary with dataset statistics
    """
    dataset_path = Path(dataset_dir)
    labels_dir = dataset_path / 'labels'

    label_files = list(labels_dir.glob('*.txt'))

    total_craters = 0
    crater_sizes = []
    images_with_craters = 0
    images_without_craters = 0

    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()

        if len(lines) > 0:
            images_with_craters += 1
            total_craters += len(lines)

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, _, _, w, h = map(float, parts)
                    # Size is the average of width and height (normalized)
                    crater_sizes.append((w + h) / 2)
        else:
            images_without_craters += 1

    stats = {
        'total_images': len(label_files),
        'images_with_craters': images_with_craters,
        'images_without_craters': images_without_craters,
        'total_craters': total_craters,
        'avg_craters_per_image': total_craters / len(label_files) if label_files else 0,
        'avg_crater_size': np.mean(crater_sizes) if crater_sizes else 0,
        'min_crater_size': np.min(crater_sizes) if crater_sizes else 0,
        'max_crater_size': np.max(crater_sizes) if crater_sizes else 0,
        'median_crater_size': np.median(crater_sizes) if crater_sizes else 0
    }

    return stats


def print_dataset_statistics(dataset_dir: str):
    """Print dataset statistics."""
    stats = calculate_dataset_statistics(dataset_dir)

    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"Total images: {stats['total_images']}")
    print(f"Images with craters: {stats['images_with_craters']}")
    print(f"Images without craters: {stats['images_without_craters']}")
    print(f"Total craters: {stats['total_craters']}")
    print(f"Average craters per image: {stats['avg_craters_per_image']:.2f}")
    print(f"\nCrater size statistics (normalized):")
    print(f"  Average: {stats['avg_crater_size']:.4f}")
    print(f"  Median: {stats['median_crater_size']:.4f}")
    print(f"  Min: {stats['min_crater_size']:.4f}")
    print(f"  Max: {stats['max_crater_size']:.4f}")
    print("="*60 + "\n")


def augment_image_and_labels(image: np.ndarray, bboxes: List[Tuple[float, float, float, float]],
                            flip_horizontal: bool = False, flip_vertical: bool = False,
                            rotate_90: int = 0) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
    """
    Apply augmentation to image and adjust bounding boxes accordingly.

    Args:
        image: Input image array
        bboxes: List of bounding boxes in YOLO format (x_center, y_center, width, height)
        flip_horizontal: Whether to flip horizontally
        flip_vertical: Whether to flip vertically
        rotate_90: Number of 90-degree rotations (0, 1, 2, 3)

    Returns:
        Augmented image and adjusted bounding boxes
    """
    aug_image = image.copy()
    aug_bboxes = []

    for bbox in bboxes:
        x_center, y_center, width, height = bbox

        # Horizontal flip
        if flip_horizontal:
            x_center = 1.0 - x_center

        # Vertical flip
        if flip_vertical:
            y_center = 1.0 - y_center

        # Rotation
        for _ in range(rotate_90 % 4):
            x_center, y_center = 1.0 - y_center, x_center
            width, height = height, width

        aug_bboxes.append((x_center, y_center, width, height))

    # Apply transformations to image
    if flip_horizontal:
        aug_image = np.fliplr(aug_image)

    if flip_vertical:
        aug_image = np.flipud(aug_image)

    if rotate_90 > 0:
        aug_image = np.rot90(aug_image, k=rotate_90)

    return aug_image, aug_bboxes


def verify_dataset(dataset_dir: str) -> bool:
    """
    Verify dataset integrity by checking if all images have corresponding labels.

    Args:
        dataset_dir: Directory containing images/ and labels/ folders

    Returns:
        True if dataset is valid, False otherwise
    """
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'

    if not images_dir.exists():
        print(f"Error: images/ directory not found in {dataset_dir}")
        return False

    if not labels_dir.exists():
        print(f"Error: labels/ directory not found in {dataset_dir}")
        return False

    image_files = set([img.stem for img in images_dir.glob('*.png')])
    label_files = set([lbl.stem for lbl in labels_dir.glob('*.txt')])

    missing_labels = image_files - label_files
    missing_images = label_files - image_files

    if missing_labels:
        print(f"Warning: {len(missing_labels)} images without labels:")
        for name in list(missing_labels)[:10]:
            print(f"  - {name}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more")

    if missing_images:
        print(f"Warning: {len(missing_images)} labels without images:")
        for name in list(missing_images)[:10]:
            print(f"  - {name}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")

    if not missing_labels and not missing_images:
        print(f"✓ Dataset verification passed: {len(image_files)} images with labels")
        return True
    else:
        return False


def convert_yolo_to_coco(dataset_dir: str, output_file: str = None):
    """
    Convert YOLO format labels to COCO format.

    Args:
        dataset_dir: Directory containing images/ and labels/ folders
        output_file: Output JSON file path (default: dataset_dir/annotations_coco.json)
    """
    from PIL import Image as PILImage

    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'

    if output_file is None:
        output_file = dataset_path / 'annotations_coco.json'

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "crater", "supercategory": "lunar_feature"}]
    }

    annotation_id = 0

    image_files = sorted(list(images_dir.glob('*.png')))

    for img_id, img_file in enumerate(image_files):
        # Get image dimensions
        with PILImage.open(img_file) as img:
            img_width, img_height = img.size

        coco_dict["images"].append({
            "id": img_id,
            "file_name": img_file.name,
            "width": img_width,
            "height": img_height
        })

        # Read corresponding label file
        label_file = labels_dir / f'{img_file.stem}.txt'
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, x_center, y_center, width, height = map(float, parts)

                        # Convert from YOLO to COCO format
                        x_center_px = x_center * img_width
                        y_center_px = y_center * img_height
                        width_px = width * img_width
                        height_px = height * img_height

                        xmin = x_center_px - width_px / 2
                        ymin = y_center_px - height_px / 2

                        coco_dict["annotations"].append({
                            "id": annotation_id,
                            "image_id": img_id,
                            "category_id": 0,
                            "bbox": [xmin, ymin, width_px, height_px],
                            "area": width_px * height_px,
                            "iscrowd": 0
                        })
                        annotation_id += 1

    with open(output_file, 'w') as f:
        json.dump(coco_dict, f, indent=2)

    print(f"Converted {len(image_files)} images to COCO format")
    print(f"Saved to: {output_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Utility functions for crater dataset')
    parser.add_argument('command', choices=['split', 'stats', 'verify', 'convert'],
                       help='Command to run')
    parser.add_argument('--dataset', required=True, help='Dataset directory')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output', help='Output file for convert command')

    args = parser.parse_args()

    if args.command == 'split':
        split_dataset(args.dataset, args.train_ratio, args.val_ratio,
                     args.test_ratio, args.seed)
    elif args.command == 'stats':
        print_dataset_statistics(args.dataset)
    elif args.command == 'verify':
        verify_dataset(args.dataset)
    elif args.command == 'convert':
        convert_yolo_to_coco(args.dataset, args.output)

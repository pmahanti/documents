#!/usr/bin/env python3
"""
Example usage scripts for crater detection training data generator.

This file demonstrates different use cases and workflows.
"""

from crater_training_data_generator import CraterDatasetGenerator
from crater_utils import (
    split_dataset,
    print_dataset_statistics,
    verify_dataset,
    convert_yolo_to_coco
)


def example_1_simple_dataset():
    """
    Example 1: Generate a simple YOLO dataset from a single image.
    """
    print("Example 1: Simple Dataset Generation")
    print("=" * 60)

    generator = CraterDatasetGenerator(
        image_path='data/lunar_image.tif',
        crater_shapefile='data/craters.shp',
        output_dir='output/simple_dataset',
        format_type='yolo'
    )

    generator.run()

    print("\nDataset generated successfully!")
    print("View results in: output/simple_dataset/")


def example_2_tiled_dataset():
    """
    Example 2: Generate a tiled dataset from a large image.
    """
    print("\nExample 2: Tiled Dataset Generation")
    print("=" * 60)

    generator = CraterDatasetGenerator(
        image_path='data/large_lunar_mosaic.cub',
        crater_shapefile='data/crater_database.shp',
        output_dir='output/tiled_dataset',
        format_type='both',  # Generate both YOLO and COCO formats
        tile_size=512,
        overlap=64
    )

    generator.run()

    print("\nTiled dataset generated successfully!")
    print("View results in: output/tiled_dataset/")


def example_3_multimodal_dataset():
    """
    Example 3: Generate dataset with both image and topography.
    """
    print("\nExample 3: Multi-Modal Dataset (Image + Topography)")
    print("=" * 60)

    generator = CraterDatasetGenerator(
        image_path='data/lunar_image.tif',
        crater_shapefile='data/craters.shp',
        output_dir='output/multimodal_dataset',
        format_type='yolo',
        topography_path='data/lunar_dem.tif',
        tile_size=640,
        overlap=128
    )

    generator.run()

    print("\nMulti-modal dataset generated successfully!")
    print("You now have both image and topography datasets!")


def example_4_complete_workflow():
    """
    Example 4: Complete workflow including dataset generation, splitting, and verification.
    """
    print("\nExample 4: Complete Workflow")
    print("=" * 60)

    # Step 1: Generate dataset
    print("\nStep 1: Generating dataset...")
    generator = CraterDatasetGenerator(
        image_path='data/lunar_region.tif',
        crater_shapefile='data/annotated_craters.shp',
        output_dir='output/complete_dataset',
        format_type='both',
        tile_size=512,
        overlap=64
    )
    generator.run()

    # Step 2: Verify dataset
    print("\nStep 2: Verifying dataset integrity...")
    verify_dataset('output/complete_dataset')

    # Step 3: View statistics
    print("\nStep 3: Viewing dataset statistics...")
    print_dataset_statistics('output/complete_dataset')

    # Step 4: Split into train/val/test
    print("\nStep 4: Splitting dataset into train/val/test...")
    split_dataset(
        dataset_dir='output/complete_dataset',
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )

    print("\nComplete workflow finished!")
    print("Dataset is ready for training!")
    print("  - Train set in: output/complete_dataset/train/")
    print("  - Val set in: output/complete_dataset/val/")
    print("  - Test set in: output/complete_dataset/test/")


def example_5_yolo_training():
    """
    Example 5: Training a YOLO model with the generated dataset.

    Note: Requires ultralytics package (pip install ultralytics)
    """
    print("\nExample 5: YOLO Model Training")
    print("=" * 60)

    try:
        from ultralytics import YOLO

        # Load a pretrained YOLOv8 model
        model = YOLO('yolov8n.pt')  # nano model (fastest)
        # Other options: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

        # Train the model
        results = model.train(
            data='output/complete_dataset/dataset.yaml',
            epochs=100,
            imgsz=512,
            batch=16,
            name='crater_detector',
            patience=20,  # Early stopping
            save=True,
            plots=True
        )

        print("\nTraining complete!")
        print("Model saved in: runs/detect/crater_detector/")

        # Validate the model
        metrics = model.val()
        print(f"\nValidation mAP50: {metrics.box.map50:.3f}")
        print(f"Validation mAP50-95: {metrics.box.map:.3f}")

    except ImportError:
        print("ultralytics package not installed.")
        print("Install with: pip install ultralytics")
        print("\nTraining code example:")
        print("""
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='output/complete_dataset/dataset.yaml',
    epochs=100,
    imgsz=512
)
        """)


def example_6_inference():
    """
    Example 6: Using a trained model for crater detection.

    Note: Requires a trained model from example 5
    """
    print("\nExample 6: Crater Detection Inference")
    print("=" * 60)

    try:
        from ultralytics import YOLO

        # Load the trained model
        model = YOLO('runs/detect/crater_detector/weights/best.pt')

        # Run inference on new images
        results = model.predict(
            source='data/test_images/',
            save=True,
            save_txt=True,  # Save labels
            conf=0.25,  # Confidence threshold
            iou=0.45,  # NMS IoU threshold
            name='crater_predictions'
        )

        print("\nInference complete!")
        print("Results saved in: runs/detect/crater_predictions/")

        # Process results
        for i, result in enumerate(results):
            boxes = result.boxes
            print(f"\nImage {i}: Detected {len(boxes)} craters")

            # Access predictions
            for box in boxes:
                conf = box.conf[0]
                cls = box.cls[0]
                xyxy = box.xyxy[0]  # Bounding box coordinates
                print(f"  Crater at {xyxy.tolist()} (confidence: {conf:.2f})")

    except ImportError:
        print("ultralytics package not installed.")
        print("Install with: pip install ultralytics")
    except FileNotFoundError:
        print("Trained model not found. Run example 5 first to train a model.")


def example_7_custom_preprocessing():
    """
    Example 7: Custom preprocessing with data augmentation.
    """
    print("\nExample 7: Custom Preprocessing")
    print("=" * 60)

    from crater_utils import augment_image_and_labels
    import numpy as np
    from PIL import Image

    # Load an image and its labels
    img_path = 'output/simple_dataset/images/image_000.png'
    label_path = 'output/simple_dataset/labels/image_000.txt'

    try:
        # Load image
        image = np.array(Image.open(img_path))

        # Load bounding boxes (YOLO format)
        bboxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, x, y, w, h = map(float, parts)
                    bboxes.append((x, y, w, h))

        print(f"Loaded image with {len(bboxes)} crater annotations")

        # Apply augmentations
        augmentations = [
            ('horizontal_flip', {'flip_horizontal': True}),
            ('vertical_flip', {'flip_vertical': True}),
            ('rotate_90', {'rotate_90': 1}),
            ('rotate_180', {'rotate_90': 2}),
        ]

        for aug_name, aug_params in augmentations:
            aug_img, aug_bboxes = augment_image_and_labels(image, bboxes, **aug_params)

            # Save augmented image
            output_path = f'output/augmented_{aug_name}.png'
            Image.fromarray(aug_img).save(output_path)
            print(f"Saved augmented image: {output_path}")

            # Save augmented labels
            label_output = f'output/augmented_{aug_name}.txt'
            with open(label_output, 'w') as f:
                for bbox in aug_bboxes:
                    f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        print("\nAugmentation complete! Generated 4 additional training samples.")

    except FileNotFoundError:
        print("Example files not found. Run example 1 first to generate a dataset.")


if __name__ == '__main__':
    import sys

    examples = {
        '1': ('Simple dataset generation', example_1_simple_dataset),
        '2': ('Tiled dataset generation', example_2_tiled_dataset),
        '3': ('Multi-modal dataset (image + topography)', example_3_multimodal_dataset),
        '4': ('Complete workflow (generate, split, verify)', example_4_complete_workflow),
        '5': ('YOLO model training', example_5_yolo_training),
        '6': ('Crater detection inference', example_6_inference),
        '7': ('Data augmentation', example_7_custom_preprocessing),
    }

    print("Crater Detection Training Data Generator - Examples")
    print("=" * 60)
    print("\nAvailable examples:")
    for key, (description, _) in examples.items():
        print(f"  {key}: {description}")

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            description, func = examples[example_num]
            func()
        else:
            print(f"\nInvalid example number: {example_num}")
            print("Use: python example_usage.py [1-7]")
    else:
        print("\nUsage: python example_usage.py [example_number]")
        print("Example: python example_usage.py 1")
        print("\nOr run interactively:")
        choice = input("\nEnter example number (1-7) or 'q' to quit: ")
        if choice in examples:
            description, func = examples[choice]
            func()
        elif choice.lower() != 'q':
            print("Invalid choice.")

#!/usr/bin/env python3
"""
Crater Marker Tool
A Python application for marking and measuring craters on GeoTIFF and ISIS .cub files.
"""

import sys
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QLabel, QScrollArea
)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from osgeo import gdal, osr
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pyproj


class Crater:
    """Represents a crater with center and radius."""

    def __init__(self, center_x: float, center_y: float, radius: float,
                 points: List[Tuple[float, float]], crater_id: int):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.diameter = radius * 2
        self.points = points  # The 3 points used to define the crater
        self.crater_id = crater_id

    def to_dict(self):
        """Convert crater to dictionary for saving."""
        return {
            'id': self.crater_id,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'radius': self.radius,
            'diameter': self.diameter,
            'points': self.points
        }

    @classmethod
    def from_dict(cls, data):
        """Create crater from dictionary."""
        return cls(
            data['center_x'],
            data['center_y'],
            data['radius'],
            data['points'],
            data['id']
        )


class ImageLabel(QLabel):
    """Custom QLabel for displaying and interacting with the image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.setMouseTracking(True)
        self.setScaledContents(False)
        self.pixmap_item = None

    def mousePressEvent(self, event):
        """Handle mouse clicks on the image."""
        if event.button() == Qt.LeftButton and self.pixmap_item is not None:
            # Convert widget coordinates to image coordinates
            label_pos = event.pos()
            if self.parent_widget:
                self.parent_widget.handle_image_click(label_pos.x(), label_pos.y())


class CraterMarkerApp(QMainWindow):
    """Main application window for crater marking."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crater Marker Tool")
        self.setGeometry(100, 100, 1200, 800)

        # Data attributes
        self.image_path = None
        self.gdal_dataset = None
        self.image_array = None
        self.geo_transform = None
        self.projection = None
        self.crs = None

        # Image display attributes
        self.display_image = None
        self.scale_factor = 1.0

        # Crater marking attributes
        self.current_points = []  # Points for current crater being marked
        self.craters = []  # List of Crater objects
        self.crater_counter = 0

        # Auto-save file
        self.autosave_file = "craters_autosave.json"

        # Setup UI
        self.init_ui()

        # Load autosave if exists
        self.load_autosave()

    def init_ui(self):
        """Initialize the user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Button panel
        button_layout = QHBoxLayout()

        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)
        button_layout.addWidget(self.open_button)

        self.delete_button = QPushButton("Delete Last Crater")
        self.delete_button.clicked.connect(self.delete_last_crater)
        self.delete_button.setEnabled(False)
        button_layout.addWidget(self.delete_button)

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_data)
        self.export_button.setEnabled(False)
        button_layout.addWidget(self.export_button)

        button_layout.addStretch()

        # Status label
        self.status_label = QLabel("No image loaded")
        button_layout.addWidget(self.status_label)

        main_layout.addLayout(button_layout)

        # Image display area with scroll
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.image_label = ImageLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        scroll_area.setWidget(self.image_label)

        main_layout.addWidget(scroll_area)

        # Info label
        self.info_label = QLabel(f"Craters marked: {len(self.craters)}")
        main_layout.addWidget(self.info_label)

    def open_image(self):
        """Open a GeoTIFF or ISIS .cub file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.tif *.tiff *.cub);;GeoTIFF (*.tif *.tiff);;ISIS Cube (*.cub);;All Files (*)"
        )

        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path: str):
        """Load and display the image using GDAL."""
        try:
            # Open with GDAL
            self.gdal_dataset = gdal.Open(file_path, gdal.GA_ReadOnly)

            if self.gdal_dataset is None:
                QMessageBox.critical(self, "Error", f"Failed to open {file_path}")
                return

            self.image_path = file_path

            # Get geospatial information
            self.geo_transform = self.gdal_dataset.GetGeoTransform()
            self.projection = self.gdal_dataset.GetProjection()

            # Get CRS
            if self.projection:
                srs = osr.SpatialReference(wkt=self.projection)
                self.crs = srs.ExportToProj4()
            else:
                self.crs = None

            # Read image data
            band = self.gdal_dataset.GetRasterBand(1)
            self.image_array = band.ReadAsArray()

            # Normalize for display
            self.create_display_image()

            # Update UI
            self.status_label.setText(f"Loaded: {Path(file_path).name}")
            self.export_button.setEnabled(True)
            self.update_info_label()

            # Redraw with existing craters if any
            self.update_display()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")

    def create_display_image(self):
        """Create a normalized image for display."""
        if self.image_array is None:
            return

        # Normalize to 0-255
        img_min = np.nanmin(self.image_array)
        img_max = np.nanmax(self.image_array)

        if img_max > img_min:
            normalized = ((self.image_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(self.image_array, dtype=np.uint8)

        # Convert to QImage
        height, width = normalized.shape
        bytes_per_line = width

        # Create RGB image (grayscale)
        rgb_array = np.stack([normalized, normalized, normalized], axis=2)

        self.display_image = QImage(
            rgb_array.data,
            width,
            height,
            width * 3,
            QImage.Format_RGB888
        )

        # Store a copy to avoid garbage collection
        self.rgb_array = rgb_array

    def update_display(self):
        """Update the display with current image and craters."""
        if self.display_image is None:
            return

        # Create a pixmap from the image
        pixmap = QPixmap.fromImage(self.display_image)

        # Draw craters and current points
        painter = QPainter(pixmap)

        # Draw completed craters
        pen_crater = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen_crater)

        for crater in self.craters:
            center_px, center_py = self.geo_to_pixel(crater.center_x, crater.center_y)

            # Calculate radius in pixels (approximate)
            edge_x = crater.center_x + crater.radius
            edge_y = crater.center_y
            edge_px, edge_py = self.geo_to_pixel(edge_x, edge_y)
            radius_px = abs(edge_px - center_px)

            # Draw circle
            painter.drawEllipse(
                int(center_px - radius_px),
                int(center_py - radius_px),
                int(radius_px * 2),
                int(radius_px * 2)
            )

            # Draw center point
            painter.drawLine(int(center_px - 5), int(center_py), int(center_px + 5), int(center_py))
            painter.drawLine(int(center_px), int(center_py - 5), int(center_px), int(center_py + 5))

        # Draw current points being selected
        pen_point = QPen(QColor(255, 0, 0), 6)
        painter.setPen(pen_point)

        for geo_x, geo_y in self.current_points:
            px, py = self.geo_to_pixel(geo_x, geo_y)
            painter.drawPoint(int(px), int(py))

        painter.end()

        # Display the pixmap
        self.image_label.setPixmap(pixmap)
        self.image_label.pixmap_item = pixmap

    def handle_image_click(self, x: int, y: int):
        """Handle a click on the image."""
        if self.gdal_dataset is None:
            return

        # Convert pixel coordinates to geo coordinates
        geo_x, geo_y = self.pixel_to_geo(x, y)

        # Add point to current selection
        self.current_points.append((geo_x, geo_y))

        # Update status
        points_needed = 3 - len(self.current_points)
        if points_needed > 0:
            self.status_label.setText(f"Select {points_needed} more point(s) on crater rim")

        # If we have 3 points, fit a circle
        if len(self.current_points) == 3:
            self.fit_crater()

        # Update display
        self.update_display()

    def fit_crater(self):
        """Fit a circle to the three selected points."""
        if len(self.current_points) != 3:
            return

        # Extract points
        p1, p2, p3 = self.current_points

        # Calculate circle from 3 points
        center, radius = self.circle_from_three_points(p1, p2, p3)

        if center is None:
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Could not fit a circle to the selected points. They may be collinear."
            )
            self.current_points = []
            self.update_display()
            return

        # Create crater object
        self.crater_counter += 1
        crater = Crater(
            center[0], center[1], radius,
            self.current_points.copy(),
            self.crater_counter
        )

        # Add to list
        self.craters.append(crater)

        # Clear current points
        self.current_points = []

        # Update UI
        self.update_info_label()
        self.delete_button.setEnabled(True)
        self.status_label.setText(f"Crater {self.crater_counter} marked")

        # Auto-save
        self.autosave()

        # Update display
        self.update_display()

    def circle_from_three_points(self, p1, p2, p3):
        """
        Calculate circle center and radius from three points.
        Returns (center, radius) or (None, None) if points are collinear.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Calculate the perpendicular bisectors
        # Midpoints
        ma_x, ma_y = (x1 + x2) / 2, (y1 + y2) / 2
        mb_x, mb_y = (x2 + x3) / 2, (y2 + y3) / 2

        # Slopes of the lines
        if abs(x2 - x1) < 1e-10:
            # Vertical line between p1 and p2
            slope_a = None
        else:
            slope_a = (y2 - y1) / (x2 - x1)

        if abs(x3 - x2) < 1e-10:
            # Vertical line between p2 and p3
            slope_b = None
        else:
            slope_b = (y3 - y2) / (x3 - x2)

        # Perpendicular slopes
        if slope_a is None:
            perp_a = 0
        elif abs(slope_a) < 1e-10:
            perp_a = None
        else:
            perp_a = -1 / slope_a

        if slope_b is None:
            perp_b = 0
        elif abs(slope_b) < 1e-10:
            perp_b = None
        else:
            perp_b = -1 / slope_b

        # Find intersection of perpendicular bisectors
        if perp_a is None and perp_b is None:
            # Both perpendicular lines are vertical - points are collinear
            return None, None
        elif perp_a is None:
            # First perpendicular is vertical
            cx = ma_x
            cy = perp_b * (cx - mb_x) + mb_y
        elif perp_b is None:
            # Second perpendicular is vertical
            cx = mb_x
            cy = perp_a * (cx - ma_x) + ma_y
        else:
            # Both are non-vertical
            if abs(perp_a - perp_b) < 1e-10:
                # Parallel lines - points are collinear
                return None, None

            # y - ma_y = perp_a * (x - ma_x)
            # y - mb_y = perp_b * (x - mb_x)
            # perp_a * (x - ma_x) - perp_b * (x - mb_x) = mb_y - ma_y
            # x * (perp_a - perp_b) = mb_y - ma_y + perp_a * ma_x - perp_b * mb_x
            cx = (mb_y - ma_y + perp_a * ma_x - perp_b * mb_x) / (perp_a - perp_b)
            cy = perp_a * (cx - ma_x) + ma_y

        # Calculate radius
        radius = np.sqrt((cx - x1)**2 + (cy - y1)**2)

        return (cx, cy), radius

    def pixel_to_geo(self, px: float, py: float) -> Tuple[float, float]:
        """Convert pixel coordinates to geo coordinates."""
        if self.geo_transform is None:
            return px, py

        geo_x = self.geo_transform[0] + px * self.geo_transform[1] + py * self.geo_transform[2]
        geo_y = self.geo_transform[3] + px * self.geo_transform[4] + py * self.geo_transform[5]

        return geo_x, geo_y

    def geo_to_pixel(self, geo_x: float, geo_y: float) -> Tuple[float, float]:
        """Convert geo coordinates to pixel coordinates."""
        if self.geo_transform is None:
            return geo_x, geo_y

        # Inverse of the geo_transform
        det = self.geo_transform[1] * self.geo_transform[5] - self.geo_transform[2] * self.geo_transform[4]

        if abs(det) < 1e-10:
            return 0, 0

        px = (self.geo_transform[5] * (geo_x - self.geo_transform[0]) -
              self.geo_transform[2] * (geo_y - self.geo_transform[3])) / det
        py = (-self.geo_transform[4] * (geo_x - self.geo_transform[0]) +
              self.geo_transform[1] * (geo_y - self.geo_transform[3])) / det

        return px, py

    def delete_last_crater(self):
        """Delete the most recently marked crater."""
        if self.craters:
            deleted = self.craters.pop()
            self.update_info_label()
            self.update_display()
            self.autosave()
            self.status_label.setText(f"Deleted crater {deleted.crater_id}")

            if not self.craters:
                self.delete_button.setEnabled(False)

    def autosave(self):
        """Auto-save current crater data."""
        data = {
            'image_path': self.image_path,
            'crs': self.crs,
            'geo_transform': self.geo_transform,
            'craters': [c.to_dict() for c in self.craters]
        }

        try:
            with open(self.autosave_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Autosave failed: {e}")

    def load_autosave(self):
        """Load autosave data if available."""
        if os.path.exists(self.autosave_file):
            try:
                with open(self.autosave_file, 'r') as f:
                    data = json.load(f)

                # Restore craters
                self.craters = [Crater.from_dict(c) for c in data.get('craters', [])]

                if self.craters:
                    self.crater_counter = max(c.crater_id for c in self.craters)
                    self.delete_button.setEnabled(True)

                self.update_info_label()

                # Ask user if they want to load the previous image
                if data.get('image_path') and os.path.exists(data['image_path']):
                    reply = QMessageBox.question(
                        self,
                        'Load Previous Session',
                        f"Found autosave with {len(self.craters)} crater(s).\nLoad previous image?",
                        QMessageBox.Yes | QMessageBox.No
                    )

                    if reply == QMessageBox.Yes:
                        self.load_image(data['image_path'])

            except Exception as e:
                print(f"Failed to load autosave: {e}")

    def export_data(self):
        """Export crater data to .diam and shapefile."""
        if not self.craters:
            QMessageBox.warning(self, "No Data", "No craters to export.")
            return

        # Get export directory
        export_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            ""
        )

        if not export_dir:
            return

        try:
            # Determine base filename
            if self.image_path:
                base_name = Path(self.image_path).stem
            else:
                base_name = "craters"

            # Export .diam file
            diam_path = os.path.join(export_dir, f"{base_name}.diam")
            self.export_diam(diam_path)

            # Export shapefile
            shp_path = os.path.join(export_dir, f"{base_name}_craters.shp")
            self.export_shapefile(shp_path)

            QMessageBox.information(
                self,
                "Export Successful",
                f"Exported {len(self.craters)} crater(s) to:\n{diam_path}\n{shp_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error during export: {str(e)}")

    def export_diam(self, file_path: str):
        """Export craters to .diam file format."""
        with open(file_path, 'w') as f:
            # Write header
            f.write("# Crater Marker Tool Export\n")
            f.write(f"# Image: {self.image_path}\n")
            f.write(f"# CRS: {self.crs}\n")
            f.write("# Columns: ID\tCenter_X\tCenter_Y\tDiameter\n")

            # Write crater data
            for crater in self.craters:
                f.write(f"{crater.crater_id}\t{crater.center_x}\t{crater.center_y}\t{crater.diameter}\n")

    def export_shapefile(self, file_path: str):
        """Export craters as shapefile with circular polygons."""
        # Create circle polygons for each crater
        geometries = []
        attributes = []

        for crater in self.craters:
            # Create a circle polygon
            center = Point(crater.center_x, crater.center_y)
            circle = center.buffer(crater.radius)

            geometries.append(circle)
            attributes.append({
                'crater_id': crater.crater_id,
                'center_x': crater.center_x,
                'center_y': crater.center_y,
                'radius': crater.radius,
                'diameter': crater.diameter
            })

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(attributes, geometry=geometries)

        # Set CRS if available
        if self.crs:
            try:
                gdf.crs = self.crs
            except:
                # If CRS setting fails, continue without it
                pass

        # Save to shapefile
        gdf.to_file(file_path)

    def update_info_label(self):
        """Update the information label."""
        self.info_label.setText(f"Craters marked: {len(self.craters)}")


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = CraterMarkerApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

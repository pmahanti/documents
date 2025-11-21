"""Configuration management for crater analysis."""

import json
import os
from pathlib import Path


class Config:
    """Configuration loader and manager."""

    def __init__(self, config_path=None):
        """
        Initialize configuration.

        Args:
            config_path: Path to configuration JSON file. If None, uses default location.
        """
        if config_path is None:
            # Default to config/regions.json relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "regions.json"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def get_region(self, region_name=None):
        """
        Get region configuration.

        Args:
            region_name: Name of the region. If None, uses default region.

        Returns:
            dict: Region configuration
        """
        if region_name is None:
            region_name = self.config.get('default_region', 'test')

        if region_name not in self.config['regions']:
            raise ValueError(f"Region '{region_name}' not found in configuration")

        return self.config['regions'][region_name]

    def get_dem_path(self, region_name=None):
        """Get full path to DEM file for a region."""
        region = self.get_region(region_name)
        images_dir = self.config['paths']['images_dir']
        return os.path.join(images_dir, region['dem'])

    def get_orthophoto_path(self, region_name=None):
        """Get full path to orthophoto file for a region."""
        region = self.get_region(region_name)
        images_dir = self.config['paths']['images_dir']
        return os.path.join(images_dir, region['orthophoto'])

    def get_shapefile_path(self, region_name=None):
        """Get full path to shapefile for a region."""
        region = self.get_region(region_name)
        shapefiles_dir = self.config['paths']['shapefiles_dir']
        return os.path.join(shapefiles_dir, region['name'])

    def get_output_dir(self):
        """Get output directory path."""
        return self.config['paths']['output_dir']

    def get_min_diameter(self):
        """Get minimum crater diameter threshold."""
        return self.config.get('min_diameter', 60)

    def set_paths(self, images_dir=None, shapefiles_dir=None, output_dir=None):
        """
        Update path configurations.

        Args:
            images_dir: Path to images directory
            shapefiles_dir: Path to shapefiles directory
            output_dir: Path to output directory
        """
        if images_dir:
            self.config['paths']['images_dir'] = images_dir
        if shapefiles_dir:
            self.config['paths']['shapefiles_dir'] = shapefiles_dir
        if output_dir:
            self.config['paths']['output_dir'] = output_dir

"""
Configuration file for geological timescale simulations.

This configuration uses realistic parameters for simulating
elephant hide texture formation over millions of years.
"""

# Simulation domain parameters
DOMAIN_CONFIG = {
    'width': 200,           # meters
    'height': 200,          # meters
    'resolution': 1.0,      # meters per grid cell
}

# Regolith physics parameters (based on Apollo samples and remote sensing)
PHYSICS_CONFIG = {
    # Basic properties
    'gravity': 1.62,                    # m/s² (lunar gravity)
    'grain_density': 3100,              # kg/m³ (solid grain density)

    # Porosity and density
    'porosity': 0.45,                   # 40-50% typical for loose regolith
    # bulk density automatically calculated from porosity

    # Strength parameters
    'angle_of_repose': 35.0,           # degrees
    'internal_friction_angle': 37.5,   # degrees (35-40° range)
    'cohesion': 0.5,                   # kPa (0.1-1 kPa range, very low)

    # Grain size
    'median_grain_size': 60e-6,        # meters (60 μm, within 40-800 μm range)

    # Slope thresholds for texture formation
    'min_slope_for_texture': 8.0,      # degrees (threshold)
    'optimal_slope_min': 15.0,         # degrees
    'optimal_slope_max': 25.0,         # degrees
}

# Thermal cycling parameters
THERMAL_CONFIG = {
    # Temperature range (equatorial)
    'temp_max': 400,                    # K (~127°C daytime)
    'temp_min': 100,                    # K (~-173°C nighttime)

    # Location
    'latitude': 0.0,                    # degrees (equatorial = max variation)

    # Thermal properties
    'thermal_inertia': 50,              # J m^-2 K^-1 s^-1/2
    'lunar_day_period': 29.5 * 24 * 3600,  # seconds (29.5 Earth days)
    'subsurface_skin_depth': 0.5,       # meters
}

# Seismic activity parameters
SEISMIC_CONFIG = {
    # Moonquake rates (events per year)
    'deep_quake_rate': 600,             # ~700 km depth, M 2-3.5
    'shallow_quake_rate': 2,            # ~50 km depth, M 3-5
    'thermal_quake_rate': 100,          # ~20 km depth, M 1.5-2.5
    'impact_quake_rate': 50,            # <10 km depth, M 2-4

    # Magnitude ranges
    'typical_magnitude_range': (2.0, 5.0),

    # Rate multiplier (1.0 = typical, >1.0 = more active)
    'quake_rate_multiplier': 1.0,

    # Random seed for reproducibility
    'seed': 42,
}

# Initial conditions
INITIAL_CONFIG = {
    'thickness': 2.0,       # meters (initial regolith thickness)
}

# Slope geometry parameters
SLOPE_CONFIG = {
    # Crater wall (most realistic for elephant hide)
    'crater': {
        'center_x': 100,        # meters (center of domain)
        'center_y': 100,        # meters
        'inner_radius': 40,     # meters (crater floor)
        'outer_radius': 85,     # meters (crater rim)
        'rim_height': 20,       # meters
        'floor_depth': 10,      # meters
    },

    # Surface roughness (impact-generated)
    'roughness': {
        'amplitude': 0.5,       # meters
        'wavelength': 4.0,      # meters
        'smoothing': 1.5,       # meters
    },

    # Alternative: linear slope
    'linear': {
        'angle': 20.0,          # degrees (in optimal range)
        'direction': 'y',       # 'x' or 'y'
    },
}

# Geological timescale parameters
GEOLOGICAL_CONFIG = {
    # Simulation duration
    'duration_years': 3e6,              # 3 million years

    # Progress reporting
    'progress_interval_years': 3e5,     # 300,000 years

    # Time acceleration (for accelerated simulations)
    'time_acceleration': 1e8,           # 100 million times faster
}

# Visualization parameters
VIZ_CONFIG = {
    'dpi': 300,                         # resolution for saved figures
    'figsize': (16, 12),               # figure size in inches

    # Colormaps
    'colormap_elevation': 'terrain',
    'colormap_texture': 'bone',
    'colormap_slope': 'hot',
    'colormap_displacement': 'plasma',

    # Hillshade
    'hillshade_azimuth': 315,          # degrees
    'hillshade_altitude': 45,          # degrees

    # Texture enhancement
    'enhance_texture': True,            # apply edge enhancement
    'edge_sigma': 2.0,                  # Gaussian smoothing for enhancement
}

# Output parameters
OUTPUT_CONFIG = {
    'base_dir': 'output',
    'save_figures': True,
    'save_data': True,                  # save numerical data as .npz files
    'figure_format': 'png',             # 'png', 'pdf', or 'both'
}

# Comparison scenarios for parameter studies
SCENARIOS = {
    'optimal': {
        'slope_angle': 20.0,
        'porosity': 0.45,
        'cohesion': 0.5,
        'thermal_range': (100, 400),
        'seismic_rate': 1.0,
        'duration_myr': 3.0,
    },

    'low_slope': {
        'slope_angle': 5.0,             # Below threshold
        'porosity': 0.45,
        'cohesion': 0.5,
        'thermal_range': (100, 400),
        'seismic_rate': 1.0,
        'duration_myr': 3.0,
    },

    'high_slope': {
        'slope_angle': 35.0,            # Above optimal, fresh avalanches
        'porosity': 0.45,
        'cohesion': 0.5,
        'thermal_range': (100, 400),
        'seismic_rate': 1.0,
        'duration_myr': 3.0,
    },

    'high_porosity': {
        'slope_angle': 20.0,
        'porosity': 0.50,               # Very loose
        'cohesion': 0.1,                # Lower cohesion
        'thermal_range': (100, 400),
        'seismic_rate': 1.0,
        'duration_myr': 3.0,
    },

    'high_latitude': {
        'slope_angle': 20.0,
        'porosity': 0.45,
        'cohesion': 0.5,
        'thermal_range': (150, 300),    # Reduced thermal cycling
        'seismic_rate': 1.0,
        'duration_myr': 3.0,
    },

    'high_seismic': {
        'slope_angle': 20.0,
        'porosity': 0.45,
        'cohesion': 0.5,
        'thermal_range': (100, 400),
        'seismic_rate': 2.0,            # 2x typical
        'duration_myr': 3.0,
    },

    'young_crater': {
        'slope_angle': 20.0,
        'porosity': 0.45,
        'cohesion': 0.5,
        'thermal_range': (100, 400),
        'seismic_rate': 1.0,
        'duration_myr': 0.5,            # Only 500,000 years
    },
}

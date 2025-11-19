"""
Example configuration file for lunar regolith simulations.

Copy this file and modify parameters as needed for your simulations.
"""

# Simulation domain parameters
DOMAIN_CONFIG = {
    'width': 100,           # meters
    'height': 100,          # meters
    'resolution': 1.0,      # meters per grid cell
}

# Physics parameters
PHYSICS_CONFIG = {
    'gravity': 1.62,                    # m/s² (lunar gravity)
    'particle_density': 1800,           # kg/m³
    'angle_of_repose': 35.0,           # degrees
    'cohesion': 0.1,                   # kPa
    'internal_friction_angle': 38.0,   # degrees
}

# Initial conditions
INITIAL_CONFIG = {
    'thickness': 1.5,       # meters (initial regolith thickness)
    'timestep': 0.1,        # seconds
}

# Slope geometry parameters
SLOPE_CONFIG = {
    # Linear slope
    'linear': {
        'angle': 35.0,      # degrees
        'direction': 'y',   # 'x' or 'y'
    },

    # Crater wall
    'crater': {
        'center_x': 50,         # meters
        'center_y': 50,         # meters
        'inner_radius': 20,     # meters
        'outer_radius': 40,     # meters
        'rim_height': 10,       # meters
        'floor_depth': 5,       # meters
    },

    # Terrace
    'terrace': {
        'terrace_y': 50,            # meters
        'terrace_height': 5,        # meters
        'slope_angle_upper': 30,    # degrees
        'slope_angle_lower': 35,    # degrees
    },

    # Surface roughness
    'roughness': {
        'amplitude': 0.2,       # meters
        'wavelength': 5.0,      # meters
        'smoothing': 1.0,       # meters
    },
}

# Simulation runtime parameters
RUNTIME_CONFIG = {
    'duration': 1000,           # seconds
    'progress_interval': 100,   # seconds (for progress reporting)
    'snapshot_interval': 100,   # steps (for saving history)
}

# Visualization parameters
VIZ_CONFIG = {
    'dpi': 300,                 # resolution for saved figures
    'figsize': (12, 9),        # figure size in inches
    'colormap_elevation': 'terrain',
    'colormap_texture': 'bone',
    'colormap_velocity': 'plasma',
    'hillshade_azimuth': 315,  # degrees
    'hillshade_altitude': 45,  # degrees
    'enhance_texture': True,    # apply edge enhancement to texture
}

# Output parameters
OUTPUT_CONFIG = {
    'base_dir': 'output',
    'save_figures': True,
    'save_data': False,         # save numerical data as .npz files
    'save_animations': False,   # create animations (requires ffmpeg)
}

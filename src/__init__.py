"""hash_flight_delay package

This package provides modular components for flight delay prediction.
Organized into logical subpackages: data, features, models, utils, visualization.
"""

# Package metadata
__version__ = "0.1.0"

# Public submodules (organized structure)
__all__ = [
    "data",
    "features", 
    "models",
    "utils",
    "visualization",
]

# Import subpackages for convenience
from . import data, features, models, utils, visualization

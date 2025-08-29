# This file marks the 'alignment' directory as a Python package.
# It can also be used to control what gets imported when someone does 'from alignment import *'
# For now, we'll keep it simple, but we'll add imports for our new modules here
# to make them easily accessible within the 'alignment' package.

# Import the main aligner class (will be the orchestrator)
from .star_aligner import StarAligner

# Import the new modular components
from .star_detector import StarDetector
from .centroid_refiner import CentroidRefiner
from .matcher import StarMatcher
from .transform_solver import TransformSolver

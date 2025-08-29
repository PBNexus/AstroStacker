import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = DATA_DIR / "temp"
OUTPUT_DIR = DATA_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
# STATIC_RESULTS_DIR is where web-accessible stacked image previews are stored.
# This should ideally be a sub-directory of STATIC_DIR if served directly,
# but for simplicity and to match the previous setup, it points to OUTPUT_DIR
# which is mounted as /output in main.py.
# The previous version had STATIC_RESULTS_DIR = STATIC_DIR / "results", but routes.py
# was changed to use OUTPUT_DIR. Let's keep it consistent with OUTPUT_DIR for serving.
# For v0.4, we will explicitly serve from OUTPUT_DIR.
# STATIC_RESULTS_DIR is no longer strictly needed as a separate path if OUTPUT_DIR is mounted.
# However, if there's a need to distinguish between raw output and web-ready previews,
# we might reintroduce it. For now, let's simplify and assume web previews
# are served directly from OUTPUT_DIR.

# Supported file formats for input
SUPPORTED_INPUT_FORMATS = {
    '.jpg', '.jpeg', '.png', '.tiff', '.tif',
    '.fits', '.fit', '.cr2', '.nef', '.dng', '.arw'
}

# Supported file formats for output
SUPPORTED_OUTPUT_FORMATS = {
    'fits', 'tiff', 'png', 'jpg'
}

# Limits
MAX_FILE_SIZE = 1024 * 1024 * 100  # 100MB per file
MAX_FILES = 50                     # Maximum number of files per upload type (light/dark)

# NEW in v0.4: Metadata validation and processing constants
# Supported EXIF/FITS tags to extract and validate.
# These keys should be standardized for internal use after extraction.
SUPPORTED_METADATA_TAGS = {
    "exposure_time": ["EXPOSURE", "EXPTIME", "ExposureTime", "Exif.Photo.ExposureTime"],
    "iso": ["ISO", "ISOSpeedRatings", "Exif.Photo.ISOSpeedRatings"],
    "temperature": ["CCD-TEMP", "TEMP", "Temperature"], # Example FITS/RAW temperature tags
    "camera_model": ["INSTRUME", "CAMERA", "Make", "Model", "Exif.Image.Make", "Exif.Image.Model"],
    "focal_length": ["FOCALLEN", "FocalLength", "Exif.Photo.FocalLength"],
    "aperture": ["APERTURE", "FNUM", "FNumber", "Exif.Photo.FNumber"]
}

# Threshold for warning about temperature differences between darks and lights (in Celsius)
MAX_TEMP_DIFF_DARK_LIGHT = 5.0 # Max 5 degrees Celsius difference allowed without warning

# Modes for background subtraction (e.g., 'median', 'gradient', 'none')
# This can be expanded based on implemented algorithms.
BACKGROUND_SUBTRACTION_MODES = ['none', 'median', 'gradient'] # 'none' means no background subtraction

# NEW in v0.4: Hot pixel removal settings
# Threshold for hot pixel detection (e.g., standard deviations above median)
HOT_PIXEL_THRESHOLD_SIGMA = 5.0
# Minimum number of pixels to consider for hot pixel detection
HOT_PIXEL_MIN_PIXELS = 1000

# Create directories if they don't exist
# Note: STATIC_RESULTS_DIR is removed from here as OUTPUT_DIR is mounted directly for serving.
for directory in [DATA_DIR, TEMP_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Ensure sensor_profiles directory exists for optional camera maps
SENSOR_PROFILES_DIR = BASE_DIR / "sensor_profiles"
SENSOR_PROFILES_DIR.mkdir(exist_ok=True)

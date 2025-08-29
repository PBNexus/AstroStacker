import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from logger.backend_logger import backend_logger
from config import SENSOR_PROFILES_DIR # Import the new SENSOR_PROFILES_DIR from config

class SensorProfile:
    """
    Manages loading and accessing camera sensor profiles,
    which can contain data like hot pixel maps.
    """

    def __init__(self, camera_model: str):
        """
        Initializes the SensorProfile for a given camera model.

        Args:
            camera_model (str): The model name of the camera (e.g., "Canon EOS 80D").
                                This will be used to find the corresponding JSON file.
        """
        self.camera_model = camera_model
        self.profile_data: Dict = {}
        self.profile_path: Path = SENSOR_PROFILES_DIR / f"{self._sanitize_model_name(camera_model)}.json"
        self._load_profile()

    def _sanitize_model_name(self, model_name: str) -> str:
        """
        Sanitizes the camera model name to create a valid filename.
        Replaces spaces and special characters with underscores and converts to lowercase.

        Args:
            model_name (str): The original camera model name.

        Returns:
            str: The sanitized filename.
        """
        return "".join(c if c.isalnum() else "_" for c in model_name).lower()

    def _load_profile(self) -> None:
        """
        Loads the sensor profile data from the corresponding JSON file.
        If the file does not exist, an empty profile is used.
        """
        if self.profile_path.exists():
            try:
                with open(self.profile_path, 'r') as f:
                    self.profile_data = json.load(f)
                backend_logger.info(f"Loaded sensor profile for {self.camera_model} from {self.profile_path}")
            except json.JSONDecodeError as e:
                backend_logger.error(f"Error decoding JSON for sensor profile {self.profile_path}: {e}")
                self.profile_data = {} # Reset to empty if file is corrupt
            except Exception as e:
                backend_logger.error(f"Error loading sensor profile {self.profile_path}: {e}")
                self.profile_data = {}
        else:
            backend_logger.info(f"No sensor profile found for {self.camera_model} at {self.profile_path}. Using empty profile.")
            self.profile_data = {}

    def get_hot_pixels(self) -> Optional[List[Tuple[int, int]]]:
        """
        Retrieves the list of known hot pixel coordinates from the profile.

        Returns:
            Optional[List[Tuple[int, int]]]: A list of (row, col) tuples for hot pixels,
                                             or None if not defined in the profile.
        """
        hot_pixels = self.profile_data.get("hot_pixels")
        if hot_pixels is not None:
            # Ensure hot pixels are in the correct format (list of lists/tuples)
            if isinstance(hot_pixels, list) and all(isinstance(p, list) and len(p) == 2 for p in hot_pixels):
                return [tuple(p) for p in hot_pixels] # Convert inner lists to tuples
            else:
                backend_logger.warning(f"Hot pixels data for {self.camera_model} is malformed. Expected list of [row, col].")
                return None
        return None

    def get_cold_pixels(self) -> Optional[List[Tuple[int, int]]]:
        """
        Retrieves the list of known cold pixel coordinates from the profile.

        Returns:
            Optional[List[Tuple[int, int]]]: A list of (row, col) tuples for cold pixels,
                                             or None if not defined in the profile.
        """
        cold_pixels = self.profile_data.get("cold_pixels")
        if cold_pixels is not None:
            if isinstance(cold_pixels, list) and all(isinstance(p, list) and len(p) == 2 for p in cold_pixels):
                return [tuple(p) for p in cold_pixels]
            else:
                backend_logger.warning(f"Cold pixels data for {self.camera_model} is malformed. Expected list of [row, col].")
                return None
        return None

    def get_profile_value(self, key: str, default=None):
        """
        Retrieves a generic value from the sensor profile by key.

        Args:
            key (str): The key for the desired value.
            default: The default value to return if the key is not found.

        Returns:
            Any: The value associated with the key, or the default value.
        """
        return self.profile_data.get(key, default)

    # Example of how a sensor profile JSON might look:
    # {
    #   "camera_model": "Canon EOS 80D",
    #   "resolution": [6000, 4000],
    #   "pixel_size_um": 3.72,
    #   "hot_pixels": [
    #     [123, 456],
    #     [789, 101]
    #   ],
    #   "cold_pixels": [
    #     [50, 60],
    #     [100, 110]
    #   ]
    # }

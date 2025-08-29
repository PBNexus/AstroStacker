import rawpy
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional
import exifread # For reading EXIF metadata from non-raw files
from logger.backend_logger import backend_logger
from config import SUPPORTED_INPUT_FORMATS

class FileLoader:
    """
    Handles loading various image formats (JPG, PNG, TIFF, FITS, RAW)
    and extracting relevant metadata.
    """

    def __init__(self):
        pass # No specific initialization needed for now

    def load_image(self, file_path: Path) -> np.ndarray:
        """
        Loads an image from the given file path into a NumPy array (float32, 0-1 range).
        Handles different file types including raw formats with debayering.

        Args:
            file_path (Path): The path to the image file.

        Returns:
            np.ndarray: The loaded image as a float32 NumPy array (0-1 range).

        Raises:
            ValueError: If the file format is unsupported or loading fails.
        """
        file_ext = file_path.suffix.lower()

        if file_ext not in SUPPORTED_INPUT_FORMATS:
            backend_logger.error(f"Unsupported file format: {file_path.name}")
            raise ValueError(f"Unsupported file format: {file_path.name}")

        try:
            if file_ext in ['.cr2', '.nef', '.dng', '.arw', '.orf', '.rw2', '.pef', '.srw']: # Common raw extensions
                return self._load_raw_image(file_path)
            elif file_ext in ['.fits', '.fit']:
                return self._load_fits_image(file_path)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                return self._load_standard_image(file_path)
            else:
                # Fallback for any other supported but unhandled types
                backend_logger.warning(f"File type {file_ext} not specifically handled, attempting standard load for {file_path.name}.")
                return self._load_standard_image(file_path)

        except Exception as e:
            backend_logger.error(f"Error loading image {file_path.name}: {e}", exc_info=True)
            raise ValueError(f"Failed to load image {file_path.name}: {e}")

    def _load_raw_image(self, file_path: Path) -> np.ndarray:
        """
        Loads and debayers a raw image using rawpy, preserving native color data.

        Args:
            file_path (Path): Path to the raw image file.

        Returns:
            np.ndarray: The loaded image as a float32 NumPy array (0-1 range).
        """
        backend_logger.info(f"Loading raw image: {file_path.name}")
        try:
            with rawpy.imread(str(file_path)) as raw:
                # Use postprocess for debayering, white balance, and color space conversion.
                # Removed 'demosaic' argument as it seems unsupported by your rawpy version.
                rgb = raw.postprocess(
                    gamma=(1,1), # Linear gamma for astrophotography
                    no_auto_bright=True, # Prevent auto brightness adjustment
                    output_bps=16, # 16-bit for better dynamic range
                    use_camera_wb=False, # Disable camera white balance
                    user_wb=[1, 1, 1, 1], # Neutral white balance for RGGB Bayer pattern
                    output_color=rawpy.ColorSpace.raw # Corrected to .raw
                )
                
                # Convert to float32 and normalize to 0-1 range
                if rgb.dtype == np.uint16:
                    image_float = rgb.astype(np.float32) / 65535.0
                elif rgb.dtype == np.uint8:
                    image_float = rgb.astype(np.float32) / 255.0
                else:
                    raise ValueError(f"Unexpected rawpy output dtype: {rgb.dtype}")

                backend_logger.info(f"Successfully loaded and debayered raw image: {file_path.name}")
                return image_float

        except rawpy.LibRawError as e:
            backend_logger.error(f"LibRaw error loading raw image {file_path.name}: {e}")
            raise ValueError(f"LibRaw error loading raw image {file_path.name}: {e}")
        except Exception as e:
            backend_logger.error(f"Unexpected error in _load_raw_image for {file_path.name}: {e}", exc_info=True)
            raise ValueError(f"Failed to load raw image {file_path.name}: {e}")

    def _load_fits_image(self, file_path: Path) -> np.ndarray:
        """
        Loads a FITS image using astropy.io.fits.
        """
        try:
            from astropy.io import fits
            backend_logger.info(f"Loading FITS image: {file_path.name}")
            with fits.open(file_path) as hdul:
                data = hdul[0].data
            
            # Normalize FITS data to 0-1 range
            if data.dtype == np.uint16:
                image_float = data.astype(np.float32) / 65535.0
            elif data.dtype == np.uint8:
                image_float = data.astype(np.float32) / 255.0
            elif data.dtype in [np.float32, np.float64]:
                min_val, max_val = data.min(), data.max()
                if max_val > min_val:
                    image_float = (data - min_val) / (max_val - min_val)
                else:
                    image_float = np.zeros_like(data, dtype=np.float32)
            else:
                info = np.iinfo(data.dtype)
                image_float = data.astype(np.float32) / info.max
            
            backend_logger.info(f"Successfully loaded FITS image: {file_path.name}")
            return image_float
        except ImportError:
            backend_logger.error("Astropy not installed. Cannot load FITS files.")
            raise ImportError("Astropy is not installed, cannot load FITS files.")
        except Exception as e:
            backend_logger.error(f"Error loading FITS image {file_path.name}: {e}", exc_info=True)
            raise ValueError(f"Failed to load FITS image {file_path.name}: {e}")

    def _load_standard_image(self, file_path: Path) -> np.ndarray:
        """
        Loads a standard image (JPG, PNG, TIFF) using OpenCV.
        """
        backend_logger.info(f"Loading standard image: {file_path.name}")
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)

        if img is None:
            raise ValueError(f"Could not read image file: {file_path.name}")

        # Convert to float32 and normalize to 0-1 range
        if img.dtype == np.uint16:
            image_float = img.astype(np.float32) / 65535.0
        elif img.dtype == np.uint8:
            image_float = img.astype(np.float32) / 255.0
        else:
            backend_logger.warning(f"Unexpected dtype for standard image {file_path.name}: {img.dtype}. Attempting direct conversion.")
            image_float = img.astype(np.float32)

        # Convert BGR to RGB if 3-channel
        if image_float.ndim == 3 and image_float.shape[2] == 3:
            image_float = cv2.cvtColor(image_float, cv2.COLOR_BGR2RGB)
        elif image_float.ndim == 3 and image_float.shape[2] == 4:
            image_float = cv2.cvtColor(image_float, cv2.COLOR_BGRA2RGB)
            backend_logger.info(f"Converted RGBA image {file_path.name} to RGB.")

        backend_logger.info(f"Successfully loaded standard image: {file_path.name}")
        return image_float

    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extracts relevant metadata (e.g., exposure, ISO, camera model, temperature)
        from an image file. Prioritizes rawpy for raw files, falls back to exifread for others.

        Args:
            file_path (Path): Path to the image file.

        Returns:
            Dict[str, Any]: Dictionary of extracted metadata.
        """
        metadata = {}
        file_ext = file_path.suffix.lower()

        try:
            if file_ext in ['.cr2', '.nef', '.dng', '.arw', '.orf', '.rw2', '.pef', '.srw']:
                with rawpy.imread(str(file_path)) as raw:
                    # Exposure time
                    if hasattr(raw, 'exposure') and raw.exposure is not None:
                        metadata['exposure_time'] = float(raw.exposure)
                    # ISO
                    if hasattr(raw, 'iso_speed') and raw.iso_speed is not None:
                        metadata['iso'] = int(raw.iso_speed)
                    # Camera model
                    # Check if 'camera_model' attribute exists before accessing
                    if hasattr(raw, 'camera_model') and raw.camera_model: # ADDED CHECK
                        metadata['camera_model'] = str(raw.camera_model)
                    else:
                        backend_logger.warning(f"Could not extract camera model from {file_path.name} using rawpy.camera_model.") # ADDED WARNING
                    
                    # Try to extract temperature from maker notes (camera-specific)
                    if hasattr(raw, 'makernotes') and hasattr(raw.makernotes, 'Temperature'):
                        metadata['temperature'] = float(raw.makernotes.Temperature)
            else: # For FITS, JPG, PNG, TIFF, etc.
                with open(file_path, 'rb') as f:
                    tags = exifread.process_file(f)
                    
                    # Common EXIF tags
                    if 'EXIF ExposureTime' in tags:
                        exposure = tags['EXIF ExposureTime'].values[0]
                        metadata['exposure_time'] = float(exposure.num) / exposure.den
                    if 'EXIF ISOSpeedRatings' in tags:
                        metadata['iso'] = int(tags['EXIF ISOSpeedRatings'].values[0])
                    if 'Image Model' in tags:
                        metadata['camera_model'] = str(tags['Image Model'])
                    
                    # Temperature (highly camera-specific, often in MakerNote)
                    if 'MakerNote Temperature' in tags:
                        metadata['temperature'] = float(tags['MakerNote Temperature'])
                    elif 'EXIF LensTemperature' in tags:
                        metadata['temperature'] = float(tags['EXIF LensTemperature'])

        except Exception as e:
            backend_logger.warning(f"Could not extract metadata from {file_path.name}: {e}")
        
        return metadata

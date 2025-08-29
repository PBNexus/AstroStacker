import numpy as np
from typing import Optional, List
from logger.backend_logger import backend_logger
from utils import ImageUtils
from calibration.hot_pixel_removal import HotPixelRemoval

class CalibrationPipeline:
    """
    Manages the application of calibration frames (bias, dark, flat)
    and hot/cold pixel correction to light frames.
    """

    @staticmethod
    def apply_calibration(
        light_frame: np.ndarray,
        master_bias: Optional[np.ndarray] = None,
        master_dark: Optional[np.ndarray] = None,
        master_flat: Optional[np.ndarray] = None,
        camera_model: Optional[str] = None
    ) -> np.ndarray:
        """
        Applies bias, dark, flat frame calibration and hot pixel removal to a light frame.

        Args:
            light_frame (np.ndarray): The input light frame (float32, 0-1 range).
            master_bias (Optional[np.ndarray]): The master bias frame.
            master_dark (Optional[np.ndarray]): The master dark frame.
            master_flat (Optional[np.ndarray]): The master flat frame.
            camera_model (Optional[str]): Camera model for hot pixel removal.

        Returns:
            np.ndarray: The calibrated light frame.
        """
        backend_logger.info("Starting calibration pipeline for light frame.")
        calibrated_frame = np.copy(light_frame)

        # Validate color space and shape consistency
        # If shapes don't match, it usually means a problem in master frame creation or file loading.
        # We set them to None to skip application rather than raising an error here.
        if master_bias is not None and master_bias.shape != light_frame.shape:
            backend_logger.warning(f"Master bias shape {master_bias.shape} does not match light frame shape {light_frame.shape}. Skipping bias subtraction.")
            master_bias = None
        if master_dark is not None and master_dark.shape != light_frame.shape:
            backend_logger.warning(f"Master dark shape {master_dark.shape} does not match light frame shape {light_frame.shape}. Skipping dark subtraction.")
            master_dark = None
        if master_flat is not None and master_flat.shape != light_frame.shape:
            backend_logger.warning(f"Master flat shape {master_flat.shape} does not match light frame shape {light_frame.shape}. Skipping flat correction.")
            master_flat = None

        # 1. Bias Subtraction
        if master_bias is not None:
            backend_logger.debug("Subtracting master bias.")
            calibrated_frame = ImageUtils.clamp_image(calibrated_frame - master_bias)

        # 2. Dark Subtraction
        # Dark frames should ideally be bias-subtracted *before* creating the master dark,
        # which is handled in stacker.py. Here we apply the master dark to the light frame.
        if master_dark is not None:
            backend_logger.debug("Subtracting master dark.")
            calibrated_frame = ImageUtils.clamp_image(calibrated_frame - master_dark)

        # 3. Flat Field Correction
        if master_flat is not None:
            backend_logger.debug("Dividing by master flat.")
            # Avoid division by zero by replacing 0s with a very small number (epsilon)
            master_flat_safe = np.where(master_flat == 0, np.finfo(master_flat.dtype).eps, master_flat)
            # Normalize the flat field by its mean to preserve overall brightness
            flat_mean = np.mean(master_flat_safe)
            # Ensure flat_mean is not zero, though unlikely with epsilon
            flat_mean = flat_mean if flat_mean != 0 else np.finfo(master_flat.dtype).eps
            
            calibrated_frame = calibrated_frame / (master_flat_safe / flat_mean)
            calibrated_frame = ImageUtils.clamp_image(calibrated_frame) # Clamp to 0-1 range

        # 4. Hot/Cold Pixel Removal
        backend_logger.debug("Applying hot/cold pixel removal.")
        calibrated_frame = HotPixelRemoval.remove_hot_pixels(calibrated_frame, camera_model)

        backend_logger.info("Calibration pipeline completed.")
        return calibrated_frame

    @staticmethod
    def create_master_bias(bias_frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """Creates a master bias frame by median stacking multiple bias frames."""
        if not bias_frames:
            backend_logger.warning("No bias frames provided. Cannot create master bias.")
            return None
        backend_logger.info(f"Creating master bias from {len(bias_frames)} frames.")
        master_bias = ImageUtils.calculate_median(bias_frames)
        master_bias = ImageUtils.clamp_image(master_bias) # Ensure 0-1 range after stacking
        backend_logger.debug(f"Master bias shape: {master_bias.shape}, dtype: {master_bias.dtype}")
        return master_bias

    @staticmethod
    def create_master_flat(flat_frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """Creates a master flat frame by median stacking multiple flat frames."""
        if not flat_frames:
            backend_logger.warning("No flat frames provided. Cannot create master flat.")
            return None
        backend_logger.info(f"Creating master flat from {len(flat_frames)} frames.")
        master_flat = ImageUtils.calculate_median(flat_frames)
        # Normalize the flat field by its average brightness after stacking
        master_flat_safe = np.where(master_flat == 0, np.finfo(master_flat.dtype).eps, master_flat)
        flat_mean = np.mean(master_flat_safe)
        flat_mean = flat_mean if flat_mean != 0 else np.finfo(master_flat.dtype).eps
        master_flat = ImageUtils.clamp_image(master_flat / flat_mean) # Clamp after normalization
        backend_logger.debug(f"Master flat shape: {master_flat.shape}, dtype: {master_flat.dtype}")
        return master_flat

    @staticmethod
    def create_master_dark(dark_frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """Creates a master dark frame by median stacking dark frames."""
        if not dark_frames:
            backend_logger.warning("No dark frames provided. Cannot create master dark.")
            return None
        backend_logger.info(f"Creating master dark from {len(dark_frames)} frames.")
        master_dark = ImageUtils.calculate_median(dark_frames)
        master_dark = ImageUtils.clamp_image(master_dark) # Ensure 0-1 range after stacking
        backend_logger.debug(f"Master dark shape: {master_dark.shape}, dtype: {master_dark.dtype}")
        return master_dark

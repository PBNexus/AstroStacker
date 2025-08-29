import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime

from file_loader import FileLoader
from logger.backend_logger import backend_logger
from logger.frontend_logger import FrontendLogBuffer
from config import OUTPUT_DIR, SUPPORTED_OUTPUT_FORMATS, BACKGROUND_SUBTRACTION_MODES

from utils import ImageUtils
from calibration.calibration_pipeline import CalibrationPipeline
from alignment.star_aligner import StarAligner # Ensure this is the ORB/Homography based one
from background.background_subtractor import BackgroundSubtractor
from validation.input_validator import InputValidator

try:
    from astropy.io import fits
except ImportError:
    fits = None
    backend_logger.warning("Astropy not found. FITS output format will not be supported.")


class AstroStacker:
    """
    Core engine for stacking astrophotography images.
    Orchestrates file loading, calibration, alignment, background subtraction, and stacking.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.log_buffer = FrontendLogBuffer(session_id)
        self.file_loader = FileLoader()
        self.calibration_pipeline = CalibrationPipeline()
        self.star_aligner = StarAligner()
        self.background_subtractor = BackgroundSubtractor()
        self.input_validator = InputValidator()


    def stack_images(
        self,
        light_file_paths: List[Path],
        dark_file_paths: Optional[List[Path]] = None,
        bias_file_paths: Optional[List[Path]] = None,
        flat_file_paths: Optional[List[Path]] = None,
        output_format: str = 'png',
        background_subtraction_method: str = 'median',
        camera_model: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Orchestrates the entire image stacking process.
        """
        self.log_buffer.add_log("Starting image stacking process...")
        backend_logger.info(f"Stacking session {self.session_id} initiated.")

        if not light_file_paths:
            self.log_buffer.add_log("No light frames provided. Stacking aborted.", "error")
            backend_logger.error("No light frames provided for stacking.")
            raise ValueError("No light frames provided.")

        if output_format.lower() not in SUPPORTED_OUTPUT_FORMATS:
            self.log_buffer.add_log(f"Unsupported output format: {output_format}. Supported: {', '.join(SUPPORTED_OUTPUT_FORMATS)}", "error")
            backend_logger.error(f"Unsupported output format: {output_format}")
            raise ValueError(f"Unsupported output format: {output_format}")

        # --- 1. Load Images ---
        self.log_buffer.add_log("Loading images...")
        light_frames = []
        for p in light_file_paths:
            frame = self.file_loader.load_image(p)
            light_frames.append(frame)
            backend_logger.debug(f"Loaded light frame: {p.name}, shape: {frame.shape}, dtype: {frame.dtype}")
        backend_logger.info(f"Loaded {len(light_frames)} light frames.")


        master_bias = None
        if bias_file_paths:
            self.log_buffer.add_log("Loading bias frames and creating master bias...")
            bias_frames = []
            for p in bias_file_paths:
                frame = self.file_loader.load_image(p)
                bias_frames.append(frame)
                backend_logger.debug(f"Loaded bias frame: {p.name}, shape: {frame.shape}, dtype: {frame.dtype}")
            master_bias = self.calibration_pipeline.create_master_bias(bias_frames)
            if master_bias is not None:
                self.log_buffer.add_log("Master bias created.")
            else:
                self.log_buffer.add_log("Failed to create master bias. Proceeding without it.", "warning")

        master_dark = None
        if dark_file_paths:
            self.log_buffer.add_log("Loading dark frames and creating master dark...")
            dark_frames = []
            for p in dark_file_paths:
                frame = self.file_loader.load_image(p)
                dark_frames.append(frame)
                backend_logger.debug(f"Loaded dark frame: {p.name}, shape: {frame.shape}, dtype: {frame.dtype}")
            # Dark frames should be bias-subtracted before stacking into a master dark
            if master_bias is not None:
                self.log_buffer.add_log("Subtracting master bias from dark frames...")
                dark_frames = [ImageUtils.clamp_image(d - master_bias) for d in dark_frames]
            master_dark = self.calibration_pipeline.create_master_dark(dark_frames)
            if master_dark is not None:
                self.log_buffer.add_log("Master dark created.")
            else:
                self.log_buffer.add_log("Failed to create master dark. Proceeding without it.", "warning")

        master_flat = None
        if flat_file_paths:
            self.log_buffer.add_log("Loading flat frames and creating master flat...")
            flat_frames = []
            for p in flat_file_paths:
                frame = self.file_loader.load_image(p)
                flat_frames.append(frame)
                backend_logger.debug(f"Loaded flat frame: {p.name}, shape: {frame.shape}, dtype: {frame.dtype}")
            # Flat frames should be bias and dark subtracted before stacking into a master flat
            if master_bias is not None:
                self.log_buffer.add_log("Subtracting master bias from flat frames...")
                flat_frames = [ImageUtils.clamp_image(f - master_bias) for f in flat_frames]
            if master_dark is not None:
                self.log_buffer.add_log("Subtracting master dark from flat frames...")
                flat_frames = [ImageUtils.clamp_image(f - master_dark) for f in flat_frames]
            master_flat = self.calibration_pipeline.create_master_flat(flat_frames)
            if master_flat is not None:
                self.log_buffer.add_log("Master flat created.")
            else:
                self.log_buffer.add_log("Failed to create master flat. Proceeding without it.", "warning")

        # --- 2. Apply Calibration to Light Frames ---
        self.log_buffer.add_log("Applying calibration to light frames...")
        calibrated_light_frames = []
        for i, frame in enumerate(light_frames):
            self.log_buffer.add_log(f"Calibrating light frame {i+1}/{len(light_frames)}...", "info")
            calibrated_frame = self.calibration_pipeline.apply_calibration(
                frame,
                master_bias=master_bias,
                master_dark=master_dark,
                master_flat=master_flat,
                camera_model=camera_model # Pass camera model for hot pixel removal
            )
            calibrated_light_frames.append(calibrated_frame)
        self.log_buffer.add_log("Calibration completed for all light frames.")

        # --- 3. Align Light Frames ---
        self.log_buffer.add_log("Aligning light frames...")
        aligned_light_frames = self.star_aligner.align_frames(calibrated_light_frames)
        self.log_buffer.add_log("Light frames alignment completed.")

        # --- 4. Background Subtraction/Normalization ---
        if background_subtraction_method != 'none':
            self.log_buffer.add_log(f"Applying background subtraction ({background_subtraction_method})...")
            # The subtract_background_batch method handles normalization across images
            aligned_light_frames = self.background_subtractor.subtract_background_batch(
                aligned_light_frames,
                method=background_subtraction_method
            )
            self.log_buffer.add_log("Background processing completed.")
        else:
            self.log_buffer.add_log("Skipping background subtraction as 'none' method selected.")


        # --- 5. Stacking ---
        self.log_buffer.add_log("Stacking aligned light frames (median stacking)...")
        # Use median stacking as it's robust to outliers (e.g., cosmic rays)
        stacked_image = ImageUtils.calculate_median(aligned_light_frames)
        self.log_buffer.add_log("Image stacking completed.")
        backend_logger.info(f"Stacked image shape: {stacked_image.shape}, dtype: {stacked_image.dtype}")

        # --- 6. Final Processing & Saving ---
        self.log_buffer.add_log("Saving final result and preview...")
        # Note: _auto_levels_and_color_balance is a post-stacking visual enhancement.
        # It is not part of the core stacking/calibration pipeline for linear data integrity.
        # It is intentionally omitted here to keep the stacked_image in its linear, debayered state
        # before saving, allowing for external post-processing tools to apply their own stretches.
        # The web preview will still get a basic stretch for display.
        output_filename, preview_filename = self._save_result(stacked_image, output_format)
        self.log_buffer.add_log("Stacking process finished successfully! ✨", "success")
        return output_filename, preview_filename


    def _auto_levels_and_color_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Applies auto levels and a basic color balance to the image.
        Assumes image is float32 (0-1 range) and RGB.
        This method is primarily for generating a visually appealing preview,
        not for the final scientific output.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            backend_logger.warning("Image is not a 3-channel RGB image. Skipping color balance and applying simple normalization.")
            return ImageUtils.normalize_image(image)

        backend_logger.info("Applying auto levels and basic color balance for preview.")

        # Convert to 8-bit for OpenCV processing, clamping to 0-1 range first
        img_8bit = (ImageUtils.clamp_image(image) * 255).astype(np.uint8)

        # Split channels
        b, g, r = cv2.split(img_8bit)

        # Apply CLAHE to each channel for adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        r_eq = clahe.apply(r)
        g_eq = clahe.apply(g)
        b_eq = clahe.apply(b)

        # Re-merge channels
        balanced_img_8bit = cv2.merge([b_eq, g_eq, r_eq])

        # Convert back to float32 (0-1 range)
        final_image_float = balanced_img_8bit.astype(np.float32) / 255.0

        # Apply a final normalization (stretch) to the entire image
        final_image_float = ImageUtils.normalize_image(final_image_float)
        
        backend_logger.info("Auto levels and basic color balance completed for preview.")
        return final_image_float


    def _save_result(self, stacked_image_float: np.ndarray, output_format: str) -> Tuple[str, str]:
        """
        Saves the final stacked image in the desired format and creates a web preview.
        """
        session_output_dir = OUTPUT_DIR / f"session_{self.session_id}"
        session_output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename_base = f"stacked_image_{timestamp}"
        preview_filename_base = f"stacked_image_preview_{timestamp}"

        output_path = session_output_dir / f"{output_filename_base}.{output_format.lower()}"
        preview_path = session_output_dir / f"{preview_filename_base}.png" # Always save preview as PNG

        try:
            # Create web preview from the final processed image (which will be stretched for display)
            # The stacked_image_float passed here is the *linear* stacked image.
            # _create_web_preview will apply a visual stretch for the preview.
            self._create_web_preview(stacked_image_float, preview_path)
            preview_filename = preview_path.name # Return just the filename for URL construction

            # Prepare image for saving based on output format
            # For the *main output*, we save the linear, un-stretched image
            # to preserve maximum data for post-processing in dedicated software.
            img_to_save_linear = ImageUtils.clamp_image(stacked_image_float)

            if output_format.lower() == 'png':
                # PNG can handle 16-bit, so scale to 0-65535. Convert RGB to BGR for OpenCV.
                if img_to_save_linear.ndim == 3:
                    img_to_save = cv2.cvtColor((img_to_save_linear * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR)
                else:
                    img_to_save = (img_to_save_linear * 65535).astype(np.uint16)
                cv2.imwrite(str(output_path), img_to_save)
                output_filename = output_path.name

            elif output_format.lower() == 'tiff' or output_format.lower() == 'tif':
                # TIFF can handle 16-bit, so scale to 0-65535. Convert RGB to BGR for OpenCV.
                if img_to_save_linear.ndim == 3:
                    img_to_save = cv2.cvtColor((img_to_save_linear * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR)
                else:
                    img_to_save = (img_to_save_linear * 65535).astype(np.uint16)
                cv2.imwrite(str(output_path), img_to_save)
                output_filename = output_path.name

            elif output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
                # JPG is 8-bit, so scale to 0-255. Convert RGB to BGR for OpenCV.
                # For JPG output, we *do* apply a basic stretch to make it viewable,
                # as JPG is not typically used for linear scientific data.
                img_to_save_for_jpg = self._auto_levels_and_color_balance(stacked_image_float) # Apply stretch for JPG
                img_to_save = (ImageUtils.clamp_image(img_to_save_for_jpg) * 255).astype(np.uint8)
                if img_to_save.ndim == 3:
                    img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), img_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
                output_filename = output_path.name

            elif output_format.lower() == 'fits':
                if fits:
                    # FITS files typically store float data directly (linear)
                    hdu = fits.PrimaryHDU(img_to_save_linear)
                    hdul = fits.HDUList([hdu])
                    hdul.writeto(output_path, overwrite=True)
                    output_filename = output_path.name
                else:
                    raise ImportError("Astropy is not installed, cannot save FITS format.")
            else:
                raise ValueError(f"Unsupported output format: {output_format}.")

            self.log_buffer.add_log(f"Result saved as {output_path.name} ({output_format.upper()}).")
            backend_logger.info(f"Saved result to {output_path}")
            return output_filename, preview_filename

        except Exception as e:
            self.log_buffer.add_log(f"Error saving result to {output_format.upper()}: {str(e)}", "error")
            backend_logger.error(f"Error saving result for session {self.session_id}: {e}", exc_info=True)
            raise

    def _create_web_preview(self, stacked_image_float: np.ndarray, preview_path: Path) -> None:
        """
        Creates a web-compatible (PNG) preview from the stacked float image.
        Applies a basic stretch for preview purposes.
        """
        try:
            # Apply a simple min-max stretch for preview
            min_val = np.min(stacked_image_float)
            max_val = np.max(stacked_image_float)
            
            if max_val - min_val < np.finfo(float).eps:
                stretched_image = np.zeros_like(stacked_image_float)
            else:
                stretched_image = (stacked_image_float - min_val) / (max_val - min_val)

            img_to_preview = (ImageUtils.clamp_image(stretched_image) * 255).astype(np.uint8)

            if img_to_preview.ndim == 3:
                bgr_image = cv2.cvtColor(img_to_preview, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = img_to_preview

            cv2.imwrite(str(preview_path), bgr_image)
            backend_logger.info(f"Saved web preview to {preview_path}")
        except Exception as e:
            backend_logger.error(f"Error creating web preview for session {self.session_id}: {e}", exc_info=True)
            self.log_buffer.add_log(f"Error creating web preview: {str(e)}", "error")
            raise

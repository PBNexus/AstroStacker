import numpy as np
import cv2
from typing import Optional, Tuple
from logger.backend_logger import backend_logger
from utils import ImageUtils # For clamp_image

# Import astropy/photutils components
try:
    from astropy.stats import sigma_clipped_stats
    from astropy.table import Table
    from photutils.detection import DAOStarFinder
    from photutils.background import Background2D, MedianBackground
except ImportError:
    DAOStarFinder = None
    Background2D = None
    MedianBackground = None
    Table = None
    sigma_clipped_stats = None
    backend_logger.error("Astropy or Photutils not found. Advanced star detection (DAOStarFinder) will be unavailable.")


# Configuration for star detection (can be moved to config.py if desired)
DEFAULT_FWHM = 4.0 # Default Full Width Half Maximum for stars in pixels
DAO_STAR_THRESHOLD_SIGMA = 3.0 # Sigma multiplier for DAOStarFinder threshold over background noise
MIN_STARS_FOR_DAO = 5 # Minimum stars DAOStarFinder must find before falling back
NL_MEANS_H_DENOISE = 10 # Denoising strength for cv2.fastNlMeansDenoising (higher = more denoising)
NL_MEANS_TEMPLATE_WINDOW = 7 # Size of the pixel neighborhood used to compute the weighted average
NL_MEANS_SEARCH_WINDOW = 21 # Size of the pixel neighborhood that is used to find similar patches

class StarDetector:
    """
    Detects stars in an image using either DAOStarFinder (preferred, professional)
    or falls back to OpenCV's SimpleBlobDetector. Includes denoising and background
    estimation for robust detection.
    """

    def __init__(self, fwhm: float = DEFAULT_FWHM):
        if DAOStarFinder is None:
            backend_logger.error("DAOStarFinder is not available. Only SimpleBlobDetector will be used for star detection.")
            self.dao_star_finder = None
        else:
            # Initialize DAOStarFinder with a scalar threshold (e.g., 0.0)
            # The actual threshold will be set dynamically per image in detect_stars.
            self.dao_star_finder = DAOStarFinder(fwhm=fwhm, threshold=0.0) # FIXED: Set threshold to a scalar value
            backend_logger.info(f"DAOStarFinder initialized with default FWHM={fwhm} and threshold=0.0.")


        # Initialize SimpleBlobDetector for fallback
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 5
        params.maxArea = 500
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.filterByInertia = True
        params.minInertiaRatio = 0.1
        self.blob_detector = cv2.SimpleBlobDetector_create(params)

        # Initialize CLAHE for adaptive contrast enhancement (used for both detectors)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def detect_stars(self, image: np.ndarray) -> Optional[Table]: # type: ignore
        """
        Detects stars in the input image. Prioritizes DAOStarFinder and falls back
        to SimpleBlobDetector if DAOStarFinder fails or finds too few stars.

        Args:
            image (np.ndarray): The input image (float32, 0-1 range), can be grayscale or RGB.

        Returns:
            Optional[Table]: An Astropy Table containing detected stars ('xcentroid', 'ycentroid' columns),
                             or None if no stars are found.
        """
        backend_logger.info("Starting star detection.")

        # --- Pre-processing: Denoising and Grayscale Conversion ---
        # Convert to 8-bit for OpenCV operations (denoising, CLAHE)
        img_uint8 = (ImageUtils.clamp_image(image) * 255).astype(np.uint8)

        # Convert to grayscale for denoising and feature detection.
        # Use mean for raw color space as per previous discussions.
        if img_uint8.ndim == 3:
            gray_img_uint8 = np.mean(img_uint8, axis=2).astype(np.uint8)
        else:
            gray_img_uint8 = img_uint8

        # Apply Non-Local Means Denoising
        # This helps suppress background noise while preserving star shapes
        denoised_img_uint8 = cv2.fastNlMeansDenoising(
            gray_img_uint8, 
            None, 
            NL_MEANS_H_DENOISE, 
            NL_MEANS_TEMPLATE_WINDOW, 
            NL_MEANS_SEARCH_WINDOW
        )
        backend_logger.debug("Applied Non-Local Means Denoising.")

        # Apply CLAHE for adaptive contrast enhancement
        processed_img_for_detection = self.clahe.apply(denoised_img_uint8)
        backend_logger.debug("Applied CLAHE for detection.")

        # Convert back to float for DAOStarFinder (which expects float data)
        processed_img_float = processed_img_for_detection.astype(np.float32) / 255.0

        detected_stars = None

        # --- Attempt DAOStarFinder ---
        if self.dao_star_finder:
            try:
                # Estimate background and noise for dynamic thresholding
                mean_bkg, median_bkg, std_bkg = sigma_clipped_stats(processed_img_float, sigma=3.0)
                
                # Set DAOStarFinder threshold dynamically based on background noise
                # This overrides the initial 0.0 threshold.
                self.dao_star_finder.threshold = median_bkg + (std_bkg * DAO_STAR_THRESHOLD_SIGMA)
                backend_logger.debug(f"DAOStarFinder dynamic threshold set to: {self.dao_star_finder.threshold:.4f}")

                # Detect stars
                # Subtract background before passing to DAOStarFinder for better results
                stars = self.dao_star_finder(processed_img_float - median_bkg)
                
                if stars is not None and len(stars) >= MIN_STARS_FOR_DAO:
                    detected_stars = stars
                    backend_logger.info(f"DAOStarFinder detected {len(detected_stars)} stars.")
                else:
                    backend_logger.warning(f"DAOStarFinder found {len(stars) if stars else 0} stars, which is less than {MIN_STARS_FOR_DAO}. Falling back to SimpleBlobDetector.")

            except Exception as e:
                backend_logger.error(f"Error during DAOStarFinder detection: {e}. Falling back to SimpleBlobDetector.", exc_info=True)
                self.dao_star_finder = None # Disable DAOStarFinder for this session if it fails
        
        # --- Fallback to SimpleBlobDetector ---
        if detected_stars is None:
            backend_logger.info("Using SimpleBlobDetector for star detection.")
            keypoints = self.blob_detector.detect(processed_img_for_detection) # Use 8-bit image for blob detector
            
            if keypoints:
                # Convert OpenCV KeyPoints to an Astropy Table format for consistency
                x_coords = [kp.pt[0] for kp in keypoints]
                y_coords = [kp.pt[1] for kp in keypoints]
                # Add a 'flux' or 'id' column if needed for later steps, though not strictly used by blob detector
                detected_stars = Table({'xcentroid': x_coords, 'ycentroid': y_coords, 'id': range(len(keypoints))})
                backend_logger.info(f"SimpleBlobDetector detected {len(detected_stars)} stars (fallback).")
            else:
                backend_logger.warning("SimpleBlobDetector found no stars.")

        if detected_stars is None or len(detected_stars) == 0:
            backend_logger.warning("No stars detected in the image.")
            return None
        
        return detected_stars

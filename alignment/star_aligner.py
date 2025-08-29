import cv2
import numpy as np
from typing import List, Optional, Tuple
from logger.backend_logger import backend_logger
from utils import ImageUtils # For clamping
from astropy.table import Table


# Import the new modular components
from alignment.star_detector import StarDetector
from alignment.centroid_refiner import CentroidRefiner
from alignment.matcher import StarMatcher
from alignment.transform_solver import TransformSolver

class StarAligner:
    """
    Aligns astronomical images using a professional pipeline:
    1. High-fidelity star detection (DAOStarFinder with fallback).
    2. Subpixel centroid refinement (2D Gaussian fitting).
    3. Robust star matching (KD-Trees with advanced filtering).
    4. Transformation estimation (Homography/Affine with RANSAC), with fallback to translation.
    """

    def __init__(self):
        # Initialize instances of the new modular components
        self.star_detector = StarDetector()
        self.centroid_refiner = CentroidRefiner()
        self.star_matcher = StarMatcher()
        self.transform_solver = TransformSolver()

    def align_frames(
        self,
        images: List[np.ndarray],
        transform_type: str = 'homography' # 'affine' or 'homography'
    ) -> List[np.ndarray]:
        """
        Aligns a list of images to the first image in the list using a professional
        star alignment pipeline, with a fallback to translation-only alignment.

        Args:
            images (List[np.ndarray]): A list of input images (float32, 0-1 range).
            transform_type (str): The type of transformation to estimate ('affine' or 'homography').

        Returns:
            List[np.ndarray]: A list of aligned images. Returns original images if alignment fails.
        """
        if not images:
            backend_logger.warning("No images provided for alignment. Returning empty list.")
            return []
        if len(images) == 1:
            backend_logger.info("Only one image provided, no alignment needed. Returning original image.")
            return images

        backend_logger.info(f"Starting professional star alignment for {len(images)} frames.")

        # --- 1. Detect and Refine Stars in Reference Image ---
        reference_image = images[0]
        backend_logger.info("Detecting and refining stars in the reference image...")
        ref_stars = self.star_detector.detect_stars(reference_image)
        if ref_stars is not None:
            # Pass output_dir for saving refined stars for debugging if needed
            ref_stars = self.centroid_refiner.refine_centroids(reference_image, ref_stars)

        if ref_stars is None or len(ref_stars) < self.transform_solver.TRANSFORM_MIN_MATCHES:
            backend_logger.warning(f"Not enough robust stars ({len(ref_stars) if ref_stars else 0}) found in reference image for robust alignment. Attempting translation-only alignment for all frames.")
            # If reference image itself has too few stars, we can't do robust alignment.
            # In this case, we'll try to align all frames using a simple translation based on a few common stars
            # or simply return originals if even translation is not possible.
            # For now, let's proceed with current logic, but this is a point for future enhancement.
            # If ref_stars is None, we cannot align at all.
            if ref_stars is None or len(ref_stars) == 0:
                backend_logger.error("No stars found in reference image. Cannot perform any alignment. Returning original images.")
                return images

        backend_logger.info(f"Successfully detected and refined {len(ref_stars)} stars in reference image.")
        ref_points = np.float32([[s['xcentroid'], s['ycentroid']] for s in ref_stars]).reshape(-1, 2)
        
        aligned_images = [np.copy(reference_image)] # Start with the reference image in the aligned list

        # --- 2. Align Subsequent Images ---
        for i, current_image in enumerate(images[1:]):
            backend_logger.info(f"Aligning image {i+2}/{len(images)}...")

            # Detect and Refine Stars in Current Image
            target_stars = self.star_detector.detect_stars(current_image)
            if target_stars is not None:
                target_stars = self.centroid_refiner.refine_centroids(current_image, target_stars)

            if target_stars is None or len(target_stars) < self.transform_solver.TRANSFORM_MIN_MATCHES:
                backend_logger.warning(f"Not enough robust stars ({len(target_stars) if target_stars else 0}) found in image {i+2} for robust alignment. Attempting translation-only fallback.")
                # If target image has too few stars, we cannot do robust alignment for this frame.
                # Fallback to translation if possible.
                aligned_frame = self._align_translation_fallback(reference_image, current_image, ref_stars, target_stars)
                aligned_images.append(aligned_frame)
                continue

            target_points_raw = np.float32([[s['xcentroid'], s['ycentroid']] for s in target_stars]).reshape(-1, 2)

            # Match Stars
            matches_indices = self.star_matcher.match_stars(ref_stars, target_stars)

            if matches_indices is None or len(matches_indices) < self.transform_solver.TRANSFORM_MIN_MATCHES:
                backend_logger.warning(f"Not enough robust matches ({len(matches_indices) if matches_indices else 0}) found between reference and image {i+2} for robust transform. Attempting translation-only fallback.")
                aligned_frame = self._align_translation_fallback(reference_image, current_image, ref_stars, target_stars)
                aligned_images.append(aligned_frame)
                continue

            # Extract matched points based on the indices
            matched_ref_points = ref_points[matches_indices[:, 0]]
            matched_target_points = target_points_raw[matches_indices[:, 1]]

            backend_logger.info(f"Found {len(matched_ref_points)} robust matches for image {i+2}.")

            # Estimate Transformation
            M, mask = self.transform_solver.estimate_transform(matched_ref_points, matched_target_points, transform_type)

            if M is None:
                backend_logger.warning(f"Could not estimate {transform_type} transformation for image {i+2}. Attempting translation-only fallback.")
                aligned_frame = self._align_translation_fallback(reference_image, current_image, ref_stars, target_stars)
                aligned_images.append(aligned_frame)
                continue

            # Apply Transformation
            h, w = images[0].shape[:2]
            
            # Convert current_image to 8-bit and BGR for OpenCV warpPerspective
            current_image_uint8 = (ImageUtils.clamp_image(current_image) * 255).astype(np.uint8)
            if current_image_uint8.ndim == 3:
                current_image_uint8 = cv2.cvtColor(current_image_uint8, cv2.COLOR_RGB2BGR)

            # Apply the warp
            aligned_frame_uint8 = cv2.warpPerspective(current_image_uint8, M, (w, h))
            
            # Convert back to float32 (0-1 range) and RGB if it was color
            aligned_frame = aligned_frame_uint8.astype(np.float32) / 255.0
            if aligned_frame.ndim == 3:
                aligned_frame = cv2.cvtColor(aligned_frame, cv2.COLOR_BGR2RGB)

            aligned_images.append(aligned_frame)
            backend_logger.info(f"Image {i+2} aligned successfully using {transform_type} transform.")

        backend_logger.info("Professional star alignment process completed.")
        return aligned_images

    def _align_translation_fallback(self, ref_image: np.ndarray, target_image: np.ndarray, ref_stars_table: Optional[Table], target_stars_table: Optional[Table]) -> np.ndarray: # type: ignore
        """
        Performs a simple translation-only alignment as a fallback.
        Calculates median shift between common stars or defaults to no shift if no stars.
        """
        backend_logger.info("Attempting translation-only fallback alignment.")
        
        # If either star table is None or empty, we cannot calculate a shift.
        # Return the original target image.
        if ref_stars_table is None or len(ref_stars_table) == 0 or \
           target_stars_table is None or len(target_stars_table) == 0:
            backend_logger.warning("Not enough stars for even translation fallback. Returning original image.")
            return target_image

        # Try to find some basic matches for translation
        # Use a more relaxed distance threshold for this simple fallback match
        temp_matcher = StarMatcher() # Create a temporary matcher for this fallback
        
        # We need to ensure ref_stars_table and target_stars_table are sorted by 'id' or a consistent order
        # for simple translation matching if we don't use the full matcher.
        # For simplicity, let's re-use the main StarMatcher's logic but acknowledge it might be very loose.
        
        # The previous match_stars call already returned None if not enough matches.
        # If we reach here, it means the robust transform failed, but we *might* still have some stars.
        # Let's try to find a simple translation by matching the brightest stars if possible.
        
        # For a quick and dirty translation, we can try to match the single brightest star
        # or a few brightest stars if their fluxes are similar.
        
        # A more robust translation fallback would be to use phase correlation
        # or a very simple centroid-of-brightest-stars approach.
        
        # For v0.4, let's calculate the median shift of the *available* stars from the matcher.
        # If the matcher returned None earlier, we've already returned the original image.
        
        # Let's assume for this fallback, we want to calculate a shift based on the overall
        # centroids of the brightest stars in each image.
        
        # Get top N brightest stars from each (already sorted by brightness in centroid_refiner)
        num_stars_for_translation = min(50, len(ref_stars_table), len(target_stars_table))
        
        if num_stars_for_translation < 4: # Need at least 4 for a decent median, even for translation
            backend_logger.warning(f"Too few stars ({num_stars_for_translation}) for translation fallback. Returning original image.")
            return target_image

        ref_coords_for_translation = np.float32([[s['xcentroid'], s['ycentroid']] for s in ref_stars_table[:num_stars_for_translation]])
        target_coords_for_translation = np.float32([[s['xcentroid'], s['ycentroid']] for s in target_stars_table[:num_stars_for_translation]])

        # Calculate the median difference in coordinates
        # This assumes the relative order of the brightest stars is mostly preserved.
        # A more robust approach would be to re-match these top stars.
        # For v0.4, let's use a simplified median shift.
        
        # Calculate the overall centroid of the brightest stars in each image
        ref_centroid = np.median(ref_coords_for_translation, axis=0)
        target_centroid = np.median(target_coords_for_translation, axis=0)
        
        # Calculate the shift needed to move target_centroid to ref_centroid
        shift_x = ref_centroid[0] - target_centroid[0]
        shift_y = ref_centroid[1] - target_centroid[1]

        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        h, w = ref_image.shape[:2]

        current_image_uint8 = (ImageUtils.clamp_image(target_image) * 255).astype(np.uint8)
        if current_image_uint8.ndim == 3:
            current_image_uint8 = cv2.cvtColor(current_image_uint8, cv2.COLOR_RGB2BGR)

        aligned_frame_uint8 = cv2.warpAffine(current_image_uint8, translation_matrix, (w, h))
        
        aligned_frame = aligned_frame_uint8.astype(np.float32) / 255.0
        if aligned_frame.ndim == 3:
            aligned_frame = cv2.cvtColor(aligned_frame, cv2.COLOR_BGR2RGB)

        backend_logger.info(f"Image aligned using translation fallback (shift: x={shift_x:.2f}, y={shift_y:.2f}).")
        return aligned_frame

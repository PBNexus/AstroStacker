import numpy as np
import cv2
from typing import Tuple, Optional
from logger.backend_logger import backend_logger

# Configuration for transformation estimation
# Moved these into the class for proper access as class attributes
# TRANSFORM_RANSAC_THRESHOLD = 5.0 # Max reprojection error for a point to be an inlier in RANSAC
# TRANSFORM_MIN_MATCHES = 4        # Minimum number of matches required to estimate a transform

class TransformSolver:
    """
    Estimates the geometric transformation (e.g., Affine, Homography)
    between two sets of matched points using robust methods like RANSAC.
    """
    # Define as class attributes
    TRANSFORM_RANSAC_THRESHOLD = 5.0 # Max reprojection error for a point to be an inlier in RANSAC
    TRANSFORM_MIN_MATCHES = 8        # Minimum number of matches required to estimate a transform (increased to 8 for more robustness)

    def __init__(self):
        pass

    def estimate_transform(
        self,
        ref_points: np.ndarray,    # Matched points from reference image (N, 2)
        target_points: np.ndarray, # Matched points from target image (N, 2)
        transform_type: str = 'homography' # 'affine' or 'homography'
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimates the transformation matrix (M) and an inlier mask between
        two sets of matched points.

        Args:
            ref_points (np.ndarray): NumPy array of shape (N, 2) containing
                                     (x, y) coordinates of matched stars in the reference image.
            target_points (np.ndarray): NumPy array of shape (N, 2) containing
                                        (x, y) coordinates of matched stars in the target image.
            transform_type (str): The type of transformation to estimate.
                                  'affine' for translation, rotation, scaling, shear.
                                  'homography' for perspective transformation (more general).

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: A tuple containing:
                - M (np.ndarray): The 2x3 (for affine) or 3x3 (for homography) transformation matrix.
                                  Returns None if transformation estimation fails.
                - mask (np.ndarray): A NumPy array of booleans indicating inliers (True)
                                     and outliers (False). Returns None if transformation
                                     estimation fails.
        """
        # Access as class attribute
        if ref_points.shape[0] < self.TRANSFORM_MIN_MATCHES or target_points.shape[0] < self.TRANSFORM_MIN_MATCHES:
            backend_logger.warning(f"Not enough matches ({ref_points.shape[0]}) to estimate a transform. Minimum required: {self.TRANSFORM_MIN_MATCHES}.")
            return None, None

        M = None
        mask = None

        if transform_type.lower() == 'affine':
            backend_logger.info("Estimating Affine transformation.")
            M, mask = cv2.estimateAffinePartial2D(target_points, ref_points, method=cv2.RANSAC, ransacReprojThreshold=self.TRANSFORM_RANSAC_THRESHOLD)
            if mask is None:
                mask = np.ones((ref_points.shape[0], 1), dtype=np.uint8) * 255

        elif transform_type.lower() == 'homography':
            backend_logger.info("Estimating Homography transformation.")
            M, mask = cv2.findHomography(target_points, ref_points, cv2.RANSAC, self.TRANSFORM_RANSAC_THRESHOLD)

        else:
            backend_logger.error(f"Unsupported transform type: {transform_type}. Supported: 'affine', 'homography'.")
            return None, None

        if M is None:
            backend_logger.warning(f"Failed to estimate {transform_type} transformation.")
            return None, None
        
        if mask is not None:
            num_inliers = np.sum(mask == 1)
            backend_logger.info(f"Estimated {transform_type} transform with {num_inliers} inliers out of {ref_points.shape[0]} matches.")
        else:
            backend_logger.info(f"Estimated {transform_type} transform.")

        return M, mask

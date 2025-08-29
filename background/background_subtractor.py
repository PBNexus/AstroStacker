import numpy as np
import cv2
from typing import List
from logger.backend_logger import backend_logger
from utils import ImageUtils # For normalization and clamping
from config import BACKGROUND_SUBTRACTION_MODES # Import modes from config

class BackgroundSubtractor:
    """
    Provides methods for subtracting or normalizing the background sky glow
    from astronomical images.
    """

    @staticmethod
    def subtract_background(
        image: np.ndarray,
        method: str = 'median',
        kernel_size: int = 50
    ) -> np.ndarray:
        """
        Subtracts or normalizes the background of a single image.
        Processes each channel independently for RGB images.

        Args:
            image (np.ndarray): The input image (float32, 0-1 range).
            method (str): The background subtraction method ('median', 'gradient', 'none').
                          'median': Uses a large median filter to estimate background.
                          'gradient': (Placeholder for more advanced gradient estimation)
                          'none': No background subtraction performed.
            kernel_size (int): Size of the kernel for median filtering (must be odd).
                               Larger values estimate a smoother, more global background.

        Returns:
            np.ndarray: The image with background subtracted/normalized.
        """
        if method not in BACKGROUND_SUBTRACTION_MODES:
            backend_logger.warning(f"Unsupported background subtraction method: {method}. Using 'none'.")
            return image

        if method == 'none':
            return image

        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            backend_logger.warning(f"Median blur kernel size must be odd. Adjusted to {kernel_size}.")

        corrected_image = np.copy(image)

        if image.ndim == 3: # Process RGB image channel by channel
            for channel_idx in range(image.shape[2]):
                current_channel = image[:, :, channel_idx]
                
                # Convert channel to 8-bit for median blur (OpenCV requirement)
                temp_channel_uint8 = (ImageUtils.clamp_image(current_channel) * 255).astype(np.uint8)
                
                # Apply median blur to the current channel
                background_estimate_uint8 = cv2.medianBlur(temp_channel_uint8, kernel_size)
                
                # Convert back to float32 (0-1 range)
                background_estimate_float = background_estimate_uint8.astype(np.float32) / 255.0
                
                # Subtract the estimated background from the original channel
                subtracted_channel = current_channel - background_estimate_float
                
                # Clamp and assign back to the corrected image
                corrected_image[:, :, channel_idx] = ImageUtils.clamp_image(subtracted_channel)
        else: # Grayscale image processing
            temp_image_uint8 = (ImageUtils.clamp_image(image) * 255).astype(np.uint8)
            background_estimate_uint8 = cv2.medianBlur(temp_image_uint8, kernel_size)
            background_estimate_float = background_estimate_uint8.astype(np.float32) / 255.0
            corrected_image = ImageUtils.clamp_image(image - background_estimate_float)
        
        backend_logger.info(f"Background subtraction ({method}) completed for single image.")
        return corrected_image

    @staticmethod
    def subtract_background_batch(
        images: List[np.ndarray],
        method: str = 'median',
        kernel_size: int = 101 # Default large kernel for batch normalization
    ) -> List[np.ndarray]:
        """
        Normalizes the background level across a batch of images to a common target.
        This is useful for evening out sky glow differences between frames.
        Processes each channel independently for RGB images.

        Args:
            images (List[np.ndarray]): A list of input images (float32, 0-1 range).
            method (str): The background normalization method ('median', 'none').
                          'median': Estimates background level using median blur and normalizes.
                          'none': No background normalization performed.
            kernel_size (int): Size of the kernel for median filtering (must be odd).
                               Larger values estimate a smoother, more global background.

        Returns:
            List[np.ndarray]: A list of images with backgrounds normalized.
        """
        if method not in BACKGROUND_SUBTRACTION_MODES:
            backend_logger.warning(f"Unsupported batch background normalization method: {method}. Using 'none'.")
            return images

        if method == 'none':
            backend_logger.info("Skipping batch background normalization as 'none' method selected.")
            return images

        if not images:
            backend_logger.warning("No images provided for batch background normalization. Returning empty list.")
            return []

        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            backend_logger.warning(f"Batch median blur kernel size must be odd. Adjusted to {kernel_size}.")

        normalized_images = []
        # Store background levels per image, per channel
        background_levels_per_channel = [] # List of lists: [[R1, G1, B1], [R2, G2, B2], ...]

        # Estimate background level for each image (e.g., a large median-filtered version)
        for img in images:
            current_image_background_levels = []
            if img.ndim == 3: # RGB image
                for channel_idx in range(img.shape[2]):
                    current_channel = img[:, :, channel_idx]
                    temp_channel_uint8 = (ImageUtils.clamp_image(current_channel) * 255).astype(np.uint8)
                    background_estimate_img_uint8 = cv2.medianBlur(temp_channel_uint8, kernel_size)
                    current_image_background_levels.append(np.median(background_estimate_img_uint8) / 255.0)
            else: # Grayscale image
                temp_image_uint8 = (ImageUtils.clamp_image(img) * 255).astype(np.uint8)
                background_estimate_img_uint8 = cv2.medianBlur(temp_image_uint8, kernel_size)
                current_image_background_levels.append(np.median(background_estimate_img_uint8) / 255.0)
            
            background_levels_per_channel.append(current_image_background_levels)

        # Calculate the target background level for each channel (median of all images' levels for that channel)
        if images[0].ndim == 3:
            num_channels = images[0].shape[2]
            target_background_levels = [
                np.median([img_levels[c] for img_levels in background_levels_per_channel])
                for c in range(num_channels)
            ]
        else: # Grayscale
            target_background_levels = [np.median([img_levels[0] for img_levels in background_levels_per_channel])]
        
        backend_logger.debug(f"Target background levels: {target_background_levels}")

        for i, img in enumerate(images):
            adjusted_img = np.copy(img)
            if img.ndim == 3: # RGB image
                for channel_idx in range(img.shape[2]):
                    current_background_level = background_levels_per_channel[i][channel_idx]
                    target_level = target_background_levels[channel_idx]
                    adjustment = target_level - current_background_level
                    adjusted_img[:, :, channel_idx] = img[:, :, channel_idx] + adjustment
            else: # Grayscale image
                current_background_level = background_levels_per_channel[i][0]
                target_level = target_background_levels[0]
                adjustment = target_level - current_background_level
                adjusted_img = img + adjustment
            
            normalized_images.append(ImageUtils.clamp_image(adjusted_img)) # Clamp after adjustment

        backend_logger.info("Batch background normalization completed.")
        return normalized_images

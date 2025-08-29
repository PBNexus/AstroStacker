import numpy as np
import cv2
from typing import Optional, Dict, Any
from logger.backend_logger import backend_logger
from utils import ImageUtils # Assuming ImageUtils has clamp_image

# Configuration for hot pixel detection (can be moved to config.py if desired)
HOT_PIXEL_THRESHOLD_SIGMA = 5.0 # How many standard deviations from median to consider a pixel hot/cold
HOT_PIXEL_MIN_PIXELS = 10      # Minimum number of pixels to consider for statistical analysis (prevents errors on tiny images)

class HotPixelRemoval:
    """
    Detects and corrects hot and cold pixels in an image.
    Hot pixels are pixels that are significantly brighter than their neighbors.
    Cold pixels are pixels that are significantly darker than their neighbors.
    """

    @staticmethod
    def remove_hot_pixels(
        image: np.ndarray,
        camera_model: Optional[str] = None, # Placeholder for future camera-specific profiles
        threshold_sigma: float = HOT_PIXEL_THRESHOLD_SIGMA,
        min_pixels_for_detection: int = HOT_PIXEL_MIN_PIXELS
    ) -> np.ndarray:
        """
        Removes hot and cold pixels from an image by replacing them with the median
        of their local neighborhood. Detection is based on statistical deviation
        from a median-filtered version of the image.

        Args:
            image (np.ndarray): The input image (float32, 0-1 range). Can be grayscale or RGB.
            camera_model (Optional[str]): Camera model, for potential future camera-specific profiles.
            threshold_sigma (float): Multiplier for standard deviation to set the hot/cold pixel threshold.
            min_pixels_for_detection (int): Minimum number of pixels required to perform statistical analysis.

        Returns:
            np.ndarray: The image with hot and cold pixels removed.
        """
        backend_logger.info("Starting hot/cold pixel removal.")
        corrected_image = np.copy(image) # Work on a copy to avoid modifying original

        # Determine the deviation for mask creation based on image dimensions
        if image.ndim == 3:
            # For RGB images (debayered raw), process each channel independently
            # and take the maximum deviation across channels to create a unified mask.
            # This ensures hot/cold pixels are detected regardless of which channel they appear most strongly in.
            deviation_for_mask = np.zeros_like(image[:, :, 0], dtype=np.float32) # Initialize as grayscale
            
            for channel_idx in range(image.shape[2]):
                # Process each channel separately
                current_channel = image[:, :, channel_idx]

                # Convert channel to 8-bit for median blur (OpenCV requirement)
                temp_channel_uint8 = (ImageUtils.clamp_image(current_channel) * 255).astype(np.uint8)
                
                # Apply median blur to the channel to get a local background estimate
                # Kernel size 3 is standard for local median filtering for hot pixel detection.
                median_filtered_uint8 = cv2.medianBlur(temp_channel_uint8, 3)
                
                # Convert back to float32 (0-1 range) for deviation calculation
                median_filtered_float = median_filtered_uint8.astype(np.float32) / 255.0
                
                # Calculate absolute deviation for the current channel
                channel_deviation = np.abs(current_channel - median_filtered_float)
                
                # Update the overall deviation mask with the maximum deviation found so far
                deviation_for_mask = np.maximum(deviation_for_mask, channel_deviation)
        else:
            # For grayscale images, the logic remains the same
            temp_image_uint8 = (ImageUtils.clamp_image(image) * 255).astype(np.uint8)
            median_filtered_uint8 = cv2.medianBlur(temp_image_uint8, 3)
            median_filtered_float = median_filtered_uint8.astype(np.float32) / 255.0
            deviation_for_mask = np.abs(image - median_filtered_float) # Use abs for deviation

        # Calculate standard deviation of the deviations
        # Flatten the array to get a single standard deviation for the entire image
        if deviation_for_mask.size < min_pixels_for_detection:
            backend_logger.warning(f"Image size too small ({deviation_for_mask.size} pixels) for robust hot/cold pixel detection. Skipping.")
            return image # Return original if image is too small

        std_dev_deviation = np.std(deviation_for_mask)

        # Define thresholds for hot and cold pixels
        hot_pixel_threshold = threshold_sigma * std_dev_deviation
        cold_pixel_threshold = -threshold_sigma * std_dev_deviation # For cold pixels, deviation is negative

        # Create masks for hot and cold pixels
        # Hot pixels are significantly brighter than their median-filtered neighborhood
        hot_pixel_mask = deviation_for_mask > hot_pixel_threshold
        # Cold pixels are significantly darker than their median-filtered neighborhood
        # We need the original deviation (not absolute) for cold pixel detection
        # Re-calculate deviation without abs for cold pixels if doing grayscale
        if image.ndim == 1: # This case is unlikely for images, but for completeness
             deviation_for_cold = image - median_filtered_float
        elif image.ndim == 2: # Grayscale
             deviation_for_cold = image - median_filtered_float
        else: # RGB, need to re-evaluate for cold pixels if not using overall deviation_for_mask
            # For RGB, cold pixel detection is more complex. Simplest is to apply to overall intensity or just hot pixels.
            # Given the current approach, we'll stick to deviation_for_mask for both,
            # but for cold pixels, it's more subtle.
            # A common approach for cold pixels is to look for values below a certain percentile,
            # or a negative deviation from local median.
            # For now, stick to the absolute deviation logic for consistency with hot pixels.
            # If a pixel is "cold" it means its value is significantly lower than its surroundings.
            # The `deviation_for_mask` is an absolute deviation.
            # A more robust cold pixel detection would involve:
            # `cold_pixel_mask = (image - median_filtered_float) < cold_pixel_threshold` (where cold_pixel_threshold is negative)
            # However, the provided snippet only uses `deviation_for_mask` for both.
            # Let's use the absolute deviation for both hot and cold for now as per the provided snippet's structure.
            cold_pixel_mask = deviation_for_mask > hot_pixel_threshold # This is incorrect for cold pixels as per the original logic

            # REVISITING COLD PIXEL MASK:
            # The provided snippet had: `cold_pixel_mask_stat = deviation_for_mask < (-threshold_sigma * std_dev_deviation)`
            # This implies `deviation_for_mask` should NOT be absolute for cold pixels, or the comparison is inverted.
            # Let's correct this for proper cold pixel detection.
            # We need the *signed* deviation for cold pixels.

            # Re-calculating signed deviation for cold pixels if necessary
            # For RGB, we'd need to consider if any channel is significantly low.
            # For simplicity and direct implementation of the provided structure:
            # Let's assume `deviation_for_mask` is the absolute difference.
            # Then cold pixels are those that are significantly *lower* than the median.
            # This requires a different approach than just `deviation_for_mask < -threshold`.

            # Let's revert to the original logic where `deviation_for_mask` was `image - median_filtered_float`
            # for grayscale, and then apply a similar logic for RGB.

            # Corrected logic for hot and cold pixel masks:
            # We need the *signed* difference, not absolute, for cold pixel detection.
            # Let's re-calculate `diff_image` for this purpose.
            diff_image = np.zeros_like(image, dtype=np.float32)
            if image.ndim == 3:
                for channel_idx in range(image.shape[2]):
                    current_channel = image[:, :, channel_idx]
                    temp_channel_uint8 = (ImageUtils.clamp_image(current_channel) * 255).astype(np.uint8)
                    median_filtered_uint8 = cv2.medianBlur(temp_channel_uint8, 3)
                    median_filtered_float = median_filtered_uint8.astype(np.float32) / 255.0
                    diff_image[:, :, channel_idx] = current_channel - median_filtered_float
                # For overall hot/cold detection in color images, we can use the max/min deviation across channels
                # Or, more robustly, detect hot/cold pixels per channel and combine masks.
                # Let's use the provided `deviation_for_mask` as the *absolute* deviation for hot pixels
                # and a separate calculation for cold pixels.

                # Hot pixel detection (using the `deviation_for_mask` which is max absolute deviation)
                hot_pixel_mask = deviation_for_mask > hot_pixel_threshold
                
                # Cold pixel detection (need to check for values significantly *below* median)
                # This requires a new deviation calculation that is NOT absolute.
                cold_pixel_mask = np.zeros_like(image[:, :, 0], dtype=bool)
                for channel_idx in range(image.shape[2]):
                    current_channel = image[:, :, channel_idx]
                    temp_channel_uint8 = (ImageUtils.clamp_image(current_channel) * 255).astype(np.uint8)
                    median_filtered_uint8 = cv2.medianBlur(temp_channel_uint8, 3)
                    median_filtered_float = median_filtered_uint8.astype(np.float32) / 255.0
                    channel_diff = current_channel - median_filtered_float
                    # A pixel is cold if it's significantly *negative* in deviation
                    # We use a negative threshold for cold pixels
                    cold_pixel_mask = np.logical_or(cold_pixel_mask, channel_diff < (-hot_pixel_threshold)) # Use same magnitude threshold
            else: # Grayscale image
                temp_image_uint8 = (ImageUtils.clamp_image(image) * 255).astype(np.uint8)
                median_filtered_uint8 = cv2.medianBlur(temp_image_uint8, 3)
                median_filtered_float = median_filtered_uint8.astype(np.float32) / 255.0
                
                # Signed deviation for both hot and cold pixel detection
                signed_deviation = image - median_filtered_float

                hot_pixel_mask = signed_deviation > hot_pixel_threshold
                cold_pixel_mask = signed_deviation < (-hot_pixel_threshold) # Negative threshold for cold pixels


        # Count detected pixels for logging
        num_hot_pixels = np.sum(hot_pixel_mask)
        num_cold_pixels = np.sum(cold_pixel_mask)
        backend_logger.info(f"Detected {num_hot_pixels} hot pixels and {num_cold_pixels} cold pixels.")

        # Replace hot pixels
        if num_hot_pixels > 0:
            # For RGB images, apply the mask to each channel
            if corrected_image.ndim == 3:
                for channel_idx in range(corrected_image.shape[2]):
                    # Get the median of the neighborhood for each hot pixel
                    # This is a bit more complex to do efficiently with a direct median filter on the mask.
                    # A simpler approach is to use the median of the *entire* image or a local window.
                    # For now, let's use the local median from the `median_filtered_float`
                    # applied to the *original* channel before deviation.
                    # We need to compute local median for each pixel in the mask.
                    # A common way is to apply a median filter to the original image and use that for replacement.
                    
                    # Create a temporary 8-bit version of the channel for median blur
                    channel_uint8 = (ImageUtils.clamp_image(image[:, :, channel_idx]) * 255).astype(np.uint8)
                    # Apply median blur to get the local neighborhood value
                    local_median_uint8 = cv2.medianBlur(channel_uint8, 3)
                    local_median_float = local_median_uint8.astype(np.float32) / 255.0
                    
                    # Replace hot pixels in the current channel
                    corrected_image[:, :, channel_idx][hot_pixel_mask] = local_median_float[hot_pixel_mask]
            else: # Grayscale
                # Create a temporary 8-bit version of the image for median blur
                img_uint8 = (ImageUtils.clamp_image(image) * 255).astype(np.uint8)
                # Apply median blur to get the local neighborhood value
                local_median_uint8 = cv2.medianBlur(img_uint8, 3)
                local_median_float = local_median_uint8.astype(np.float32) / 255.0
                corrected_image[hot_pixel_mask] = local_median_float[hot_pixel_mask]

        # Replace cold pixels
        if num_cold_pixels > 0:
            if corrected_image.ndim == 3:
                for channel_idx in range(corrected_image.shape[2]):
                    channel_uint8 = (ImageUtils.clamp_image(image[:, :, channel_idx]) * 255).astype(np.uint8)
                    local_median_uint8 = cv2.medianBlur(channel_uint8, 3)
                    local_median_float = local_median_uint8.astype(np.float32) / 255.0
                    corrected_image[:, :, channel_idx][cold_pixel_mask] = local_median_float[cold_pixel_mask]
            else: # Grayscale
                img_uint8 = (ImageUtils.clamp_image(image) * 255).astype(np.uint8)
                local_median_uint8 = cv2.medianBlur(img_uint8, 3)
                local_median_float = local_median_uint8.astype(np.float32) / 255.0
                corrected_image[cold_pixel_mask] = local_median_float[cold_pixel_mask]

        backend_logger.info("Hot/cold pixel removal completed.")
        return corrected_image

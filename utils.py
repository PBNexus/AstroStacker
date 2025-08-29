import numpy as np
import cv2

class ImageUtils:
    """
    A collection of static utility methods for common image processing tasks.
    These functions operate on NumPy arrays representing image data.
    """

    @staticmethod
    def normalize_image(img: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
        """
        Normalizes an image array to a specified float range (default 0.0 to 1.0).
        Handles various input data types by converting to float32 before scaling.

        Args:
            img (np.ndarray): The input image array. Can be uint8, uint16, float32, etc.
            min_val (float): The desired minimum value for the output range.
            max_val (float): The desired maximum value for the output range.

        Returns:
            np.ndarray: The normalized image array as float32, scaled to the [min_val, max_val] range.
        """
        if img.dtype == np.float32 or img.dtype == np.float64:
            # If already float, find its actual min/max to scale
            current_min = img.min()
            current_max = img.max()
        else:
            # For integer types, assume full range for normalization
            current_min = np.iinfo(img.dtype).min
            current_max = np.iinfo(img.dtype).max

        # Avoid division by zero if image is completely flat
        if current_max == current_min:
            return np.full(img.shape, (min_val + max_val) / 2.0, dtype=np.float32)

        # Scale to 0-1 first, then to desired min_val-max_val
        normalized_img = (img.astype(np.float32) - current_min) / (current_max - current_min)
        return normalized_img * (max_val - min_val) + min_val

    @staticmethod
    def convert_to_float32(img: np.ndarray) -> np.ndarray:
        """
        Converts an image array to float32 data type.
        If the image is an integer type, it is also normalized to 0-1 range.
        If it's already float, it's just cast to float32.

        Args:
            img (np.ndarray): The input image array.

        Returns:
            np.ndarray: The image array converted to float32.
        """
        if img.dtype == np.float32:
            return img
        elif img.dtype in [np.uint8, np.uint16, np.int16, np.int32]:
            # Normalize integer types to 0-1 float range
            return ImageUtils.normalize_image(img, 0.0, 1.0)
        else:
            # For other types, just cast (might not be ideal for all cases)
            return img.astype(np.float32)

    @staticmethod
    def clamp_image(img: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
        """
        Clamps the pixel values of an image array to a specified range.

        Args:
            img (np.ndarray): The input image array. Expected to be float.
            min_val (float): The minimum allowed pixel value.
            max_val (float): The maximum allowed pixel value.

        Returns:
            np.ndarray: The clamped image array.
        """
        return np.clip(img, min_val, max_val)

    @staticmethod
    def calculate_median(images: list[np.ndarray]) -> np.ndarray:
        """
        Calculates the median of a list of image arrays.
        Useful for creating master darks, flats, or bias frames.

        Args:
            images (list[np.ndarray]): A list of image arrays, assumed to be of the same shape and type.

        Returns:
            np.ndarray: The median image array.
        """
        if not images:
            raise ValueError("Input list of images cannot be empty.")
        # Stack images along a new axis and compute median
        stacked_images = np.stack(images, axis=0)
        return np.median(stacked_images, axis=0)

    @staticmethod
    def calculate_mean(images: list[np.ndarray]) -> np.ndarray:
        """
        Calculates the mean of a list of image arrays.
        Useful for creating master darks, flats, or bias frames.

        Args:
            images (list[np.ndarray]): A list of image arrays, assumed to be of the same shape and type.

        Returns:
            np.ndarray: The mean image array.
        """
        if not images:
            raise ValueError("Input list of images cannot be empty.")
        # Stack images along a new axis and compute mean
        stacked_images = np.stack(images, axis=0)
        return np.mean(stacked_images, axis=0)

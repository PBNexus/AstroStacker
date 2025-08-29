import numpy as np
from scipy.optimize import curve_fit
from astropy.table import Table
from typing import Optional, Tuple, List, Dict, Any
from logger.backend_logger import backend_logger
from utils import ImageUtils # For clamp_image
from concurrent.futures import ProcessPoolExecutor # Changed to ProcessPoolExecutor for CPU-bound tasks
import os # For managing processes
from pathlib import Path # For handling file paths

# Configuration for centroid refinement
STAR_STAMP_SIZE = 9 # Size of the square stamp (e.g., 9x9 pixels) around each star for fitting. Must be odd.
CENTROID_FIT_THRESHOLD = 0.01 # Minimum amplitude for a successful Gaussian fit (relative to max pixel value)
MAX_STARS_TO_REFINE = 2000 # Cap the number of stars to refine for performance.
MIN_FLUX_FOR_REFINEMENT = 0.001 # Skip stars below this flux/amplitude for refinement (relative to 0-1 image range)
BORDER_MARGIN = 10 # Pixels from image border to exclude stars from refinement
MAX_WORKERS = os.cpu_count() or 4 # Use all available CPU cores, or default to 4 if not detectable

class CentroidRefiner:
    """
    Refines star centroids to subpixel accuracy by fitting a 2D Gaussian
    to small image stamps around each detected star.
    Optimized for performance with parallel processing and filtering.
    """

    def __init__(self):
        pass

    @staticmethod
    def _gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        """
        2D Gaussian function for fitting.
        xy: tuple (x, y) coordinates
        amplitude: peak value of the Gaussian
        xo, yo: centroid coordinates
        sigma_x, sigma_y: standard deviations along x and y axes
        theta: rotation angle of the Gaussian (in radians)
        offset: background offset
        """
        x, y = xy
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
        c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
        g = offset + amplitude * np.exp( - (a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
        return g.ravel()

    @staticmethod
    def _fit_single_star_process(star_data: Dict[str, Any]) -> Optional[Tuple[float, float, int, float, float, float, float, float]]:
        """
        Helper function to fit a single star's centroid in a separate process.
        Takes a dictionary of data to avoid large object passing.
        Returns refined centroid and fit parameters, or None on failure.
        """
        stamp = star_data['stamp']
        star_id = star_data['id']
        initial_x = star_data['initial_x']
        initial_y = star_data['initial_y']
        
        # Ensure stamp is float32
        stamp = stamp.astype(np.float32)

        if stamp.ndim == 3:
            stamp_gray = np.mean(stamp, axis=2)
        else:
            stamp_gray = stamp

        if stamp_gray.size == 0 or np.all(stamp_gray == 0):
            return None # Empty or black stamp

        x_local = np.arange(0, stamp_gray.shape[1])
        y_local = np.arange(0, stamp_gray.shape[0])
        X, Y = np.meshgrid(x_local, y_local)

        # Initial guess for Gaussian parameters
        amplitude_guess = np.max(stamp_gray) - np.min(stamp_gray)
        offset_guess = np.min(stamp_gray)
        xo_guess = stamp_gray.shape[1] / 2.0
        yo_guess = stamp_gray.shape[0] / 2.0
        # Use a more robust sigma guess, e.g., based on stamp size or a fixed value
        sigma_guess = STAR_STAMP_SIZE / (2 * np.sqrt(2 * np.log(2))) # FWHM to sigma conversion
        if sigma_guess < 0.5: sigma_guess = 0.5 # Minimum sigma

        # Bounds for parameters to help fitting
        # (amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
        lower_bounds = [0, 0, 0, 0.1, 0.1, -np.pi, 0]
        upper_bounds = [1.0, stamp_gray.shape[1], stamp_gray.shape[0], STAR_STAMP_SIZE, STAR_STAMP_SIZE, np.pi, 1.0]
        p0 = [amplitude_guess, xo_guess, yo_guess, sigma_guess, sigma_guess, 0, offset_guess]

        try:
            popt, pcov = curve_fit(CentroidRefiner._gaussian_2d, (X.ravel(), Y.ravel()), stamp_gray.ravel(), p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=2000) # Increased maxfev
            
            amplitude, xo_fit, yo_fit, sigma_x_fit, sigma_y_fit, theta_fit, offset_fit = popt

            if amplitude < CENTROID_FIT_THRESHOLD:
                return None # Amplitude too low

            # Convert local centroid back to global image coordinates
            refined_x = initial_x + (xo_fit - xo_guess)
            refined_y = initial_y + (yo_fit - yo_guess)

            return (refined_x, refined_y, star_id, amplitude, sigma_x_fit, sigma_y_fit, theta_fit, offset_fit)

        except (RuntimeError, ValueError) as e:
            # Log these as debug, as they are expected for some noisy/bad stars
            # backend_logger.debug(f"Could not fit Gaussian for star {star_id} at ({initial_x:.2f}, {initial_y:.2f}): {e}")
            return None
        except Exception as e:
            backend_logger.error(f"Unexpected error during centroid refinement for star {star_id} at ({initial_x:.2f}, {initial_y:.2f}): {e}", exc_info=True)
            return None


    def refine_centroids(self, image: np.ndarray, stars: Table, save_to_disk: bool = False, output_dir: Optional[Path] = None) -> Optional[Table]:
        """
        Refines the centroids of detected stars using 2D Gaussian fitting.

        Args:
            image (np.ndarray): The original image (float32, 0-1 range) from which stars were detected.
                                This should be the *linear* image.
            stars (Table): An Astropy Table containing initial star detections,
                           at least with 'xcentroid', 'ycentroid', and 'flux' or 'amplitude' columns.
            save_to_disk (bool): If True, saves the refined stars table to a FITS file.
            output_dir (Optional[Path]): Directory to save the refined stars table if save_to_disk is True.

        Returns:
            Optional[Table]: A new Astropy Table with refined 'xcentroid' and 'ycentroid' columns,
                             or None if no stars could be refined.
        """
        if stars is None or len(stars) == 0:
            backend_logger.warning("No stars provided for centroid refinement.")
            return None

        backend_logger.info(f"Attempting to refine centroids for {len(stars)} stars.")
        
        img_height, img_width = image.shape[:2]
        half_stamp = STAR_STAMP_SIZE // 2

        stars_to_process_data = [] # List of dicts for parallel processing
        
        # Sort stars by brightness (descending)
        if 'amplitude' in stars.colnames:
            stars.sort('amplitude', reverse=True)
        elif 'flux' in stars.colnames:
            stars.sort('flux', reverse=True)

        # Cap the number of stars if MAX_STARS_TO_REFINE is set
        if MAX_STARS_TO_REFINE is not None and len(stars) > MAX_STARS_TO_REFINE:
            backend_logger.info(f"Capping star refinement to {MAX_STARS_TO_REFINE} brightest stars from {len(stars)} detected.")
            stars = stars[:MAX_STARS_TO_REFINE]

        # Prepare tasks for parallel processing, extracting stamps in main process
        for i, star in enumerate(stars):
            initial_x, initial_y = star['xcentroid'], star['ycentroid']
            
            # Skip stars too close to the border
            if (initial_x < BORDER_MARGIN or initial_x > img_width - BORDER_MARGIN or
                initial_y < BORDER_MARGIN or initial_y > img_height - BORDER_MARGIN):
                # backend_logger.debug(f"Skipping star {i} at ({initial_x:.2f}, {initial_y:.2f}) due to border proximity.")
                continue

            # Skip low-flux stars
            star_flux = star['amplitude'] if 'amplitude' in star.colnames else star['flux']
            if star_flux < MIN_FLUX_FOR_REFINEMENT:
                # backend_logger.debug(f"Skipping star {i} at ({initial_x:.2f}, {initial_y:.2f}) due to low flux ({star_flux:.4f}).")
                continue

            # Define stamp boundaries, ensuring they stay within image bounds
            x_min = max(0, int(initial_x - half_stamp))
            x_max = min(img_width, int(initial_x + half_stamp + 1))
            y_min = max(0, int(initial_y - half_stamp))
            y_max = min(img_height, int(initial_y + half_stamp + 1))

            # Skip if stamp is too small (e.g., at image edges, though border margin helps)
            if (x_max - x_min < STAR_STAMP_SIZE) or (y_max - y_min < STAR_STAMP_SIZE):
                # backend_logger.debug(f"Skipping star {i} at ({initial_x:.2f}, {initial_y:.2f}) due to small stamp size after border check.")
                continue

            stamp = image[y_min:y_max, x_min:x_max]
            
            # Store original star data for fallback
            original_star_data = {
                'id': star['id'] if 'id' in star else i,
                'xcentroid': initial_x,
                'ycentroid': initial_y,
                'amplitude': star_flux,
                'sigma_x': np.nan, 'sigma_y': np.nan, 'theta': np.nan, 'offset': np.nan # Placeholders
            }
            
            stars_to_process_data.append({
                'stamp': stamp,
                'id': original_star_data['id'],
                'initial_x': initial_x,
                'initial_y': initial_y,
                'original_star_data': original_star_data # Pass original data for fallback
            })

        if not stars_to_process_data:
            backend_logger.warning("No stars passed filtering for centroid refinement.")
            return None

        backend_logger.info(f"Processing {len(stars_to_process_data)} stars for refinement (after filtering).")

        refined_stars_results = []
        # Parallel processing of star fits using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Map _fit_single_star_process to each star's data
            # Use list() to force evaluation and collect results
            results = list(executor.map(self._fit_single_star_process, stars_to_process_data))

        # Collect refined stars, falling back to original if fit failed
        for i, result in enumerate(results):
            original_star_info = stars_to_process_data[i]['original_star_data']

            if result:
                # Fit was successful
                refined_stars_results.append(result)
            else:
                # Fit failed, use original centroid from DAOStarFinder
                backend_logger.debug(f"Falling back to original centroid for star {original_star_info['id']} at ({original_star_info['xcentroid']:.2f}, {original_star_info['ycentroid']:.2f}) due to failed fit.")
                # Add original centroid with placeholder fit parameters
                refined_stars_results.append((original_star_info['xcentroid'], original_star_info['ycentroid'], original_star_info['id'], 
                                              original_star_info['amplitude'], np.nan, np.nan, np.nan, np.nan)) # Use original amplitude/flux

        if len(refined_stars_results) == 0:
            backend_logger.warning("No stars could be successfully refined or fell back to original centroids.")
            return None
            
        # Create the final Astropy Table from the collected data
        refined_stars_table = Table(rows=refined_stars_results, names=('xcentroid', 'ycentroid', 'id', 'amplitude', 'sigma_x', 'sigma_y', 'theta', 'offset'))
        
        backend_logger.info(f"Successfully refined {len(refined_stars_table)} star centroids (including fallbacks).")

        # Optional: Save refined stars to disk
        if save_to_disk and output_dir:
            try:
                output_filename = output_dir / f"refined_stars_{os.getpid()}.fits" # Use PID to avoid filename conflicts
                refined_stars_table.write(output_filename, format='fits', overwrite=True)
                backend_logger.info(f"Refined stars table saved to {output_filename}")
            except ImportError:
                backend_logger.warning("Astropy FITS writer not available. Cannot save refined stars to disk.")
            except Exception as e:
                backend_logger.error(f"Error saving refined stars table to disk: {e}", exc_info=True)

        return refined_stars_table

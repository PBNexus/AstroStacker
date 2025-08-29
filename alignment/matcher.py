import numpy as np
from astropy.table import Table
from typing import List, Tuple, Optional
from logger.backend_logger import backend_logger

# Import scikit-learn for KDTree
try:
    from sklearn.neighbors import KDTree
except ImportError:
    KDTree = None
    backend_logger.error("scikit-learn not found. KDTree matching will be unavailable. Fallback to simpler methods may be needed.")


# Configuration for star matching (can be moved to config.py if desired)
MATCH_DISTANCE_THRESHOLD = 10.0  # Max pixel distance for a potential match
MATCH_BRIGHTNESS_RATIO_THRESHOLD = 0.5 # Max difference in brightness ratio (e.g., 0.5 means 50% difference)
# MATCH_ANGULAR_NEIGHBORS = 5      # Removed for v0.4 simplification
# MATCH_ANGULAR_TOLERANCE = 0.05   # Removed for v0.4 simplification

class StarMatcher:
    """
    Matches stars between a reference image and a target image using KD-Trees
    and applies robust filtering based on distance and brightness.
    Angular consistency check is removed for v0.4 to prioritize speed and stability.
    """

    def __init__(self):
        if KDTree is None:
            backend_logger.warning("KDTree (scikit-learn) is not available. Star matching will be less robust.")
        pass # No specific initialization needed for now

    def match_stars(self, ref_stars: Table, target_stars: Table) -> Optional[np.ndarray]:
        """
        Finds robust matches between two sets of stars.

        Args:
            ref_stars (Table): Astropy Table of stars from the reference image
                                (must have 'xcentroid', 'ycentroid', 'flux' or 'amplitude' columns).
            target_stars (Table): Astropy Table of stars from the target image
                                  (must have 'xcentroid', 'ycentroid', 'flux' or 'amplitude' columns).

        Returns:
            Optional[np.ndarray]: A NumPy array of shape (N, 2) where N is the number of matches.
                                  Each row contains (ref_star_index, target_star_index).
                                  Returns None if no robust matches are found.
        """
        if ref_stars is None or len(ref_stars) < 4:
            backend_logger.warning("Not enough reference stars for matching.")
            return None
        if target_stars is None or len(target_stars) < 4:
            backend_logger.warning("Not enough target stars for matching.")
            return None

        backend_logger.info(f"Attempting to match {len(ref_stars)} reference stars with {len(target_stars)} target stars.")

        # Extract coordinates and fluxes (prioritize 'amplitude' from centroid_refiner, fallback to 'flux')
        ref_coords = np.array([[s['xcentroid'], s['ycentroid']] for s in ref_stars])
        target_coords = np.array([[s['xcentroid'], s['ycentroid']] for s in target_stars])

        ref_fluxes = ref_stars['amplitude'] if 'amplitude' in ref_stars.colnames else ref_stars['flux']
        target_fluxes = target_stars['amplitude'] if 'amplitude' in target_stars.colnames else target_stars['flux']


        if KDTree is None:
            backend_logger.warning("KDTree not available. Falling back to simpler, less robust matching.")
            # Fallback to a basic brute-force nearest neighbor if KDTree is not available
            matches = []
            for i, target_coord in enumerate(target_coords):
                distances = np.linalg.norm(ref_coords - target_coord, axis=1)
                best_match_idx = np.argmin(distances)
                if distances[best_match_idx] < MATCH_DISTANCE_THRESHOLD:
                    # Apply brightness filter even in fallback
                    ref_flux = ref_fluxes[best_match_idx]
                    target_flux = target_fluxes[i]
                    if ref_flux > 0 and target_flux > 0:
                        brightness_ratio = min(ref_flux, target_flux) / max(ref_flux, target_flux)
                        if brightness_ratio >= MATCH_BRIGHTNESS_RATIO_THRESHOLD:
                            matches.append((best_match_idx, i))
            backend_logger.info(f"Found {len(matches)} matches using fallback (simple nearest neighbor with brightness filter).")
            return np.array(matches) if matches else None


        # Build KDTree for efficient nearest neighbor search on reference stars
        kdtree = KDTree(ref_coords)

        # Find the single nearest neighbor for each target star in the reference stars
        distances, indices = kdtree.query(target_coords, k=1) # Only need the closest one

        potential_matches = [] # List of (ref_idx, target_idx, distance, ref_flux, target_flux)

        for i in range(len(target_stars)): # Iterate through target stars
            target_star_coord = target_coords[i]
            target_star_flux = target_fluxes[i]

            # Consider the closest reference star as the primary candidate match
            closest_ref_idx = indices[i, 0]
            dist_to_closest = distances[i, 0]

            if dist_to_closest > MATCH_DISTANCE_THRESHOLD:
                continue # Too far to be a match

            # Apply Brightness Similarity Filter
            ref_flux = ref_fluxes[closest_ref_idx]
            if ref_flux > 0 and target_star_flux > 0: # Avoid division by zero
                brightness_ratio = min(ref_flux, target_star_flux) / max(ref_flux, target_star_flux)
                if brightness_ratio < MATCH_BRIGHTNESS_RATIO_THRESHOLD:
                    continue # Brightness too dissimilar

            # If all filters pass, add to potential matches
            potential_matches.append((closest_ref_idx, i)) # Store (ref_idx, target_idx)

        if not potential_matches:
            backend_logger.warning("No robust matches found after distance and brightness filtering.")
            return None

        backend_logger.info(f"Found {len(potential_matches)} robust matches after distance and brightness filtering.")
        return np.array(potential_matches, dtype=int)

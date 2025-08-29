from typing import List, Dict, Tuple
from logger.backend_logger import backend_logger
from config import MAX_TEMP_DIFF_DARK_LIGHT # Import temperature difference threshold
import numpy as np

class InputValidator:
    """
    Validates the consistency of metadata across input image files
    (e.g., exposure time, ISO, temperature).
    """

    @staticmethod
    def validate_metadata(
        light_frame_metadata: List[Dict],
        dark_frame_metadata: List[Dict]
    ) -> List[Dict]:
        """
        Validates metadata consistency between light and dark frames, and among light frames.
        Returns a list of warning messages if inconsistencies are found.

        Args:
            light_frame_metadata (List[Dict]): List of metadata dictionaries for light frames.
            dark_frame_metadata (List[Dict]): List of metadata dictionaries for dark frames.

        Returns:
            List[Dict]: A list of dictionaries, each representing a warning.
                        Example: [{'level': 'warning', 'message': '...'}]
        """
        warnings = []
        backend_logger.info("Starting input metadata validation.")

        # 1. Validate consistency among light frames
        if len(light_frame_metadata) > 1:
            ref_meta = light_frame_metadata[0]
            for i, meta in enumerate(light_frame_metadata[1:]):
                # Check Exposure Time
                if 'exposure_time' in ref_meta and 'exposure_time' in meta:
                    if not np.isclose(ref_meta['exposure_time'], meta['exposure_time'], rtol=1e-2): # 1% relative tolerance
                        warnings.append({
                            'level': 'warning',
                            'message': f"Exposure time mismatch: Light frame 1 ({ref_meta['exposure_time']:.3f}s) "
                                       f"and Light frame {i+2} ({meta['exposure_time']:.3f}s)."
                        })
                # Check ISO
                if 'iso' in ref_meta and 'iso' in meta and ref_meta['iso'] != meta['iso']:
                    warnings.append({
                        'level': 'warning',
                        'message': f"ISO mismatch: Light frame 1 ({ref_meta['iso']}) "
                                   f"and Light frame {i+2} ({meta['iso']})."
                    })
                # Check Camera Model
                if 'camera_model' in ref_meta and 'camera_model' in meta and ref_meta['camera_model'] != meta['camera_model']:
                    warnings.append({
                        'level': 'warning',
                        'message': f"Camera model mismatch: Light frame 1 ({ref_meta['camera_model']}) "
                                   f"and Light frame {i+2} ({meta['camera_model']})."
                    })

        # 2. Validate consistency between light frames and dark frames
        if light_frame_metadata and dark_frame_metadata:
            # Use the first light frame's metadata as reference for comparison with darks
            ref_light_meta = light_frame_metadata[0]

            # Check Dark Frame Exposure Time vs. Light Frame Exposure Time
            if 'exposure_time' in ref_light_meta:
                for i, dark_meta in enumerate(dark_frame_metadata):
                    if 'exposure_time' in dark_meta:
                        if not np.isclose(ref_light_meta['exposure_time'], dark_meta['exposure_time'], rtol=1e-2):
                            warnings.append({
                                'level': 'warning',
                                'message': f"Exposure time mismatch: Light frames ({ref_light_meta['exposure_time']:.3f}s) "
                                           f"and Dark frame {i+1} ({dark_meta['exposure_time']:.3f}s). "
                                           f"Dark frames should match light frame exposure time."
                            })
            else:
                warnings.append({
                    'level': 'info',
                    'message': "Exposure time not found in light frame metadata. Cannot validate dark frame exposure."
                })

            # Check Dark Frame Temperature vs. Light Frame Temperature
            if 'temperature' in ref_light_meta:
                for i, dark_meta in enumerate(dark_frame_metadata):
                    if 'temperature' in dark_meta:
                        temp_diff = abs(ref_light_meta['temperature'] - dark_meta['temperature'])
                        if temp_diff > MAX_TEMP_DIFF_DARK_LIGHT:
                            warnings.append({
                                'level': 'warning',
                                'message': f"Temperature mismatch: Light frames ({ref_light_meta['temperature']:.1f}°C) "
                                           f"and Dark frame {i+1} ({dark_meta['temperature']:.1f}°C). "
                                           f"Difference is {temp_diff:.1f}°C, exceeding {MAX_TEMP_DIFF_DARK_LIGHT}°C. "
                                           f"For best results, dark frames should be taken at a similar temperature to light frames."
                            })
            else:
                warnings.append({
                    'level': 'info',
                    'message': "Temperature not found in light frame metadata. Cannot validate dark frame temperature."
                })

            # Check Dark Frame ISO vs. Light Frame ISO
            if 'iso' in ref_light_meta:
                for i, dark_meta in enumerate(dark_frame_metadata):
                    if 'iso' in dark_meta and ref_light_meta['iso'] != dark_meta['iso']:
                        warnings.append({
                            'level': 'warning',
                            'message': f"ISO mismatch: Light frames ({ref_light_meta['iso']}) "
                                       f"and Dark frame {i+1} ({dark_meta['iso']}). "
                                       f"Dark frames should match light frame ISO."
                        })
            else:
                warnings.append({
                    'level': 'info',
                    'message': "ISO not found in light frame metadata. Cannot validate dark frame ISO."
                })

            # Check Dark Frame Camera Model vs. Light Frame Camera Model
            if 'camera_model' in ref_light_meta:
                for i, dark_meta in enumerate(dark_frame_metadata):
                    if 'camera_model' in dark_meta and ref_light_meta['camera_model'] != dark_meta['camera_model']:
                        warnings.append({
                            'level': 'warning',
                            'message': f"Camera model mismatch: Light frames ({ref_light_meta['camera_model']}) "
                                       f"and Dark frame {i+1} ({dark_meta['camera_model']}). "
                                       f"Dark frames should be taken with the same camera as light frames."
                        })
            else:
                warnings.append({
                    'level': 'info',
                    'message': "Camera model not found in light frame metadata. Cannot validate dark frame camera model."
                })

        backend_logger.info(f"Metadata validation completed with {len(warnings)} warnings.")
        return warnings


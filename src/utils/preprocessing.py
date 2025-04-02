"""
preprocessing.py

Performs optimized preprocessing on frames for video analysis.
Preprocessing techniques include gamma correction, CLAHE, and adaptive image resizing.

Author: fw7th
Date: 2025-03-31
Modified: 2025-04-02
"""

import cv2 as cv
import numpy as np
import functools

# Cache gamma correction tables for common values to avoid redundant computation
@functools.lru_cache(maxsize=16)
def _create_gamma_table(gamma):
    """
    Creates and caches a lookup table for gamma correction.
    
    Parameters
    ----------
    gamma : float
        Gamma value for correction
        
    Returns
    -------
    numpy.ndarray
        Lookup table for gamma correction
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255.0 for i in range(256)], np.uint8)
    return table

def gammaCorrection(src, gamma):
    """
    Performs Gamma Correction for enhancing brightness of dark pixels.
    Optimized with table caching for repeated values.
    
    Parameters
    ----------
    src : numpy.ndarray
        Frame to apply gamma correction on
    gamma : float
        Gamma value for correction

    Returns
    -------
    numpy.ndarray
        Gamma-corrected frame
    """
    # Get cached table or create new one
    table = _create_gamma_table(gamma)
    return cv.LUT(src, table)

# Cache CLAHE objects for reuse
_clahe_cache = {}
def get_clahe(clip_limit=2.0, tile_size=(8, 8)):
    """
    Returns a cached CLAHE object with the specified parameters.
    
    Parameters
    ----------
    clip_limit : float
        Threshold for contrast limiting
    tile_size : tuple
        Size of grid for histogram equalization
        
    Returns
    -------
    cv.CLAHE
        CLAHE object
    """
    key = (clip_limit, tile_size)
    if key not in _clahe_cache:
        _clahe_cache[key] = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return _clahe_cache[key]

def apply_clahe(frame, clip_limit=2.0, tile_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization to improve image contrast.
    Optimized with object caching.
    
    Parameters
    ----------
    frame : numpy.ndarray
        Input frame
    clip_limit : float, optional
        Threshold for contrast limiting
    tile_size : tuple, optional
        Size of grid for histogram equalization
        
    Returns
    -------
    numpy.ndarray
        Frame with enhanced contrast
    """
    # For small images, use smaller tile size
    if frame.shape[0] < 200 or frame.shape[1] < 200:
        tile_size = (4, 4)
    
    # Convert to LAB color space
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv.split(lab)
    
    # Apply CLAHE to L channel using cached object
    clahe = get_clahe(clip_limit, tile_size)
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel with the original A and B channels
    merged = cv.merge((cl, a, b))
    
    # Convert back to BGR color space
    return cv.cvtColor(merged, cv.COLOR_LAB2BGR)

# Target resolutions for common detection models
STANDARD_SIZES = {
    "tiny": (256, 256),
    "small": (384, 384),
    "medium": (512, 512),
    "large": (640, 640)
}

def adaptive_resize(frame, target_width=640, preserve_aspect=True, interpolation=None):
    """
    Resizes a frame optimally based on its content and target size.
    
    Parameters
    ----------
    frame : numpy.ndarray
        Input frame
    target_width : int or str, optional
        Target width or preset size name
    preserve_aspect : bool, optional
        Whether to preserve aspect ratio
    interpolation : int, optional
        OpenCV interpolation method, adaptively chosen if None
        
    Returns
    -------
    numpy.ndarray
        Resized frame
    """
    # Handle preset sizes
    if isinstance(target_width, str) and target_width in STANDARD_SIZES:
        target_size = STANDARD_SIZES[target_width]
        target_width, target_height = target_size
        preserve_aspect = False
    else:
        # Calculate height based on aspect ratio
        if preserve_aspect:
            aspect_ratio = frame.shape[1] / frame.shape[0]
            target_height = int(target_width / aspect_ratio)
        else:
            target_height = target_width
    
    # Skip resize if dimensions match
    if frame.shape[1] == target_width and frame.shape[0] == target_height:
        return frame
    
    # Choose optimal interpolation method if not specified
    if interpolation is None:
        # If downsampling, use AREA to prevent aliasing
        if frame.shape[1] > target_width or frame.shape[0] > target_height:
            interpolation = cv.INTER_AREA
        # If upsampling, use CUBIC for better quality
        else:
            interpolation = cv.INTER_CUBIC
    
    # Resize the frame
    return cv.resize(frame, (target_width, target_height), interpolation=interpolation)

def preprocess_frame(frame, brightness_threshold=100, resize_target="medium"):
    """
    Preprocesses frame for better detection performance with adaptive enhancement.
    Only applies corrections that are necessary based on frame content.
    
    Parameters
    ----------
    frame : numpy.ndarray
        Input frame
    brightness_threshold : int, optional
        Threshold to determine when to apply brightness enhancement
    resize_target : str or int, optional
        Target size for resizing
        
    Returns
    -------
    processed_frame : numpy.ndarray
        The frame after preprocessing
    """
    if frame is None or frame.size == 0:
        return None
    
    # Analyze frame brightness
    mean_brightness = frame.mean()
    
    # Apply enhancements only when needed
    processed_frame = frame.copy()
    
    # Apply CLAHE for contrast enhancement on low contrast images
    if mean_brightness < brightness_threshold + 30:
        processed_frame = apply_clahe(processed_frame)
    
    # Apply gamma correction only on dark frames
    if mean_brightness < brightness_threshold:
        # Adaptively choose gamma value based on brightness
        gamma_value = 1.5 if mean_brightness < 50 else 1.2
        processed_frame = gammaCorrection(processed_frame, gamma_value)
    
    # Resize frame to target size
    return adaptive_resize(processed_frame, resize_target)

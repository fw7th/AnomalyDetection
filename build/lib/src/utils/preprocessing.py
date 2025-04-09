"""
preprocessing.py

Performes preprocessing on frames.
Preprocessing techniques involved include GammaCorrection, CLAHE and image resizing.

Author: fw7th
Date: 2025-03-31
"""

import cv2 as cv
import numpy as np

def gammaCorrection(src, gamma):
    """
    Performes Gamma Correction for enhancing brightness of dark pixels.
    
    Parameters
    ----------
    src : numpy.ndarray
        Frame to preprocess.
    gamma : int or float
        Value which the new pixel value is calculated.
    Returns
    -------
    Lookup table for pixels values, and their values after gamma correction.
    """ 
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv.LUT(src, table)

def apply_clahe(frame):
    """
    Apply Contrast Limited Adaptive Histogram Equalization to improve image contrast.
    
    Parameters
    ----------
    frame : numpy.ndarray
        Input frame
        
    Returns
    -------
    numpy.ndarray
        Frame with enhanced contrast
    """
    # Convert to LAB color space
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv.split(lab)
    
    # Create a CLAHE object with desired parameters
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE to L channel
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel with the original A and B channels
    merged = cv.merge((cl, a, b))
    
    # Convert back to BGR color space
    return cv.cvtColor(merged, cv.COLOR_LAB2BGR)

def preprocess_frame(frame):
    """
    Preprocesses frame for better detection performance.
    
    Parameters
    ----------
    frame : numpy.ndarray
        Takes the generated frame.
        
    Returns
    -------
    processed_frame : numpy.ndarray
        The frame after preprocessing.
    """
    # Apply CLAHE for contrast enhancement
    contrast_enhanced = apply_clahe(frame)
    
    # Apply gamma correction
    gamma_corrected = gammaCorrection(contrast_enhanced, 2)
    
    # Resize frame
    target_width = 640  # Optimal size for most YOLO models
    aspect_ratio = frame.shape[1] / frame.shape[0]
    target_height = int(target_width / aspect_ratio)
    
    resized_frame = cv.resize(gamma_corrected, (target_width, target_height), 
                             interpolation=cv.INTER_AREA)
    
    return resized_frame

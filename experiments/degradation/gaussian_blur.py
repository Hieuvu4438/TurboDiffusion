"""
Gaussian Blur Degradation

Applies Gaussian blur to images to simulate various levels of defocus
or motion blur degradation.
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import Union
import cv2


def apply_gaussian_blur(
    image: Union[Image.Image, np.ndarray],
    sigma: float
) -> Image.Image:
    """
    Apply Gaussian blur to an image.
    
    Args:
        image: Input image (PIL Image or numpy array)
        sigma: Standard deviation of the Gaussian kernel in pixels
    
    Returns:
        Blurred image as PIL Image
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is in correct format
    if len(image.shape) == 2:
        # Grayscale image
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    else:
        # Color image
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    
    return Image.fromarray(blurred)


def apply_gaussian_blur_pil(
    image: Union[Image.Image, np.ndarray],
    sigma: float
) -> Image.Image:
    """
    Apply Gaussian blur using PIL (alternative implementation).
    
    Args:
        image: Input image (PIL Image or numpy array)
        sigma: Standard deviation of the Gaussian kernel in pixels
    
    Returns:
        Blurred image as PIL Image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # PIL's GaussianBlur uses radius, not sigma
    # radius ≈ 2 * sigma for a reasonable approximation
    radius = sigma * 2
    
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def get_blur_params():
    """
    Get the parameter list for Gaussian blur experiments.
    
    Returns:
        List of sigma values
    """
    return [2.5, 3.5, 4.5, 5.5, 6.5]

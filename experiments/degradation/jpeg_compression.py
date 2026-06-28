"""
Multi-Generation JPEG Compression Degradation

Applies multiple rounds of JPEG compression to simulate degradation
that occurs when images are repeatedly saved as JPEG.
"""

import io
import numpy as np
from PIL import Image
from typing import Union


def apply_jpeg_compression(image: Union[Image.Image, np.ndarray], quality: int) -> Image.Image:
    """
    Apply single JPEG compression to an image.
    
    Args:
        image: Input image (PIL Image or numpy array)
        quality: JPEG quality factor (1-100, lower = more compression)
    
    Returns:
        Compressed image as PIL Image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Compress to JPEG in memory
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    
    # Read back the compressed image
    compressed = Image.open(buffer)
    compressed.load()  # Force load before buffer is closed
    
    return compressed


def multi_generation_jpeg(
    image: Union[Image.Image, np.ndarray],
    num_cycles: int,
    quality: int
) -> Image.Image:
    """
    Apply multiple generations of JPEG compression.
    
    This simulates the degradation that occurs when an image is
    repeatedly saved as JPEG (e.g., through social media sharing).
    
    Args:
        image: Input image (PIL Image or numpy array)
        num_cycles: Number of compression cycles (k)
        quality: JPEG quality factor for each cycle (Q)
    
    Returns:
        Degraded image after k compression cycles
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply compression cycles
    result = image.copy()
    for _ in range(num_cycles):
        result = apply_jpeg_compression(result, quality)
    
    return result


def get_jpeg_params():
    """
    Get the parameter combinations for JPEG compression experiments.
    
    Returns:
        List of tuples (num_cycles, quality)
    """
    num_cycles_list = [4, 5, 6, 7, 8]
    quality_list = [8, 12, 16, 20, 25]
    
    params = []
    for k in num_cycles_list:
        for q in quality_list:
            params.append((k, q))
    
    return params

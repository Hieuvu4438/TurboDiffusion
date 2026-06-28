"""
Colour Channel Clipping Degradation

Applies per-channel offsets to simulate color degradation such as
exposure issues, white balance problems, or sensor saturation.
"""

import numpy as np
from PIL import Image
from typing import Union, Tuple


def apply_color_clipping(
    image: Union[Image.Image, np.ndarray],
    offset: int,
    channel: str = 'all'
) -> Image.Image:
    """
    Apply color channel clipping/offset to an image.
    
    Args:
        image: Input image (PIL Image or numpy array)
        offset: Per-channel offset value (can be negative or positive)
        channel: Which channel to apply offset to:
                 'r' - Red channel only
                 'g' - Green channel only
                 'b' - Blue channel only
                 'all' - All channels uniformly
    
    Returns:
        Color-clipped image as PIL Image
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is RGB
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    
    # Convert to float for processing
    result = image.astype(np.float32)
    
    if channel == 'all':
        result = result + offset
    elif channel == 'r':
        result[:, :, 0] = result[:, :, 0] + offset
    elif channel == 'g':
        result[:, :, 1] = result[:, :, 1] + offset
    elif channel == 'b':
        result[:, :, 2] = result[:, :, 2] + offset
    else:
        raise ValueError(f"Unknown channel: {channel}. Use 'r', 'g', 'b', or 'all'")
    
    # Clip to valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)


def apply_per_channel_clipping(
    image: Union[Image.Image, np.ndarray],
    r_offset: int = 0,
    g_offset: int = 0,
    b_offset: int = 0
) -> Image.Image:
    """
    Apply different offsets to each color channel.
    
    Args:
        image: Input image (PIL Image or numpy array)
        r_offset: Offset for red channel
        g_offset: Offset for green channel
        b_offset: Offset for blue channel
    
    Returns:
        Color-adjusted image as PIL Image
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is RGB
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    
    # Convert to float for processing
    result = image.astype(np.float32)
    
    # Apply offsets
    result[:, :, 0] = result[:, :, 0] + r_offset
    result[:, :, 1] = result[:, :, 1] + g_offset
    result[:, :, 2] = result[:, :, 2] + b_offset
    
    # Clip to valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)


def get_color_clip_params():
    """
    Get the parameter list for color clipping experiments.
    
    Returns:
        List of offset values
    """
    return [-35, -25, -15, 15, 25, 35]

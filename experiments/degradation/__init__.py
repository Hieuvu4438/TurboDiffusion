"""
Image Degradation Module
========================

This module provides various image degradation techniques:
- Multi-Generation JPEG Compression
- Gaussian Blur
- Colour Channel Clipping
"""

from .jpeg_compression import apply_jpeg_compression, multi_generation_jpeg
from .gaussian_blur import apply_gaussian_blur
from .color_clipping import apply_color_clipping

__all__ = [
    'apply_jpeg_compression',
    'multi_generation_jpeg',
    'apply_gaussian_blur',
    'apply_color_clipping',
]

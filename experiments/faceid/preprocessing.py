"""Image-preprocessing fallbacks for face detection on degraded inputs.

Extracted from ``evaluate_identity_preservation_v2.py`` (lines 44-103).
Detection on heavily degraded frames (low-light L3, heavy blur, salt-pepper)
frequently fails on the raw image; these helpers retry with targeted
enhancement and progressive padding until a face is found.
"""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_for_detection(img, noise_type=None):
    """Enhance an image for face detection based on the degradation type.

    - ``lowlight``       : CLAHE on the V channel of HSV (no colour shift).
    - ``motion``/``gaussian_blur`` : unsharp mask.
    - ``salt-pepper``    : 3x3 median filter.
    Returns the enhanced BGR image, or the original if ``noise_type`` is None.
    """
    if noise_type == "lowlight":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v_eq = clahe.apply(v)
        hsv_eq = cv2.merge([h, s, v_eq])
        return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    elif noise_type in ("motion", "gaussian_blur"):
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    elif noise_type == "salt-pepper":
        return cv2.medianBlur(img, 3)
    return img


def progressive_padding(img, pad_sizes=(100, 200, 300)):
    """Yield successively larger black-padded copies of ``img``.

    Downsampling-like effect from padding helps detectors recover faces from
    heavily degraded inputs (e.g. 200px padding lifted L3-lowlight detection
    from 0% to ~100%). Used as a fallback loop by :mod:`embeddings_insightface`.
    """
    for pad_size in pad_sizes:
        yield pad_size, cv2.copyMakeBorder(
            img, pad_size, pad_size, pad_size, pad_size,
            cv2.BORDER_CONSTANT, value=[0, 0, 0],
        )

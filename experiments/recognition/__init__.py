"""
MTCNN + InsightFace Face Recognition Module

This package provides face recognition functionality using:
- MTCNN for face detection and landmark extraction
- InsightFace (ArcFace) for feature extraction
- Cosine similarity for face matching

Based on paper methodology:
"Face Recognition: We employ ArcFace architecture with MTCNN for detection 
and alignment. Recognition uses cosine similarity with 75% threshold"
"""

from .face_recognition import FaceRecognizer, ARCFACE_DST

__version__ = '1.0.0'
__all__ = ['FaceRecognizer', 'ARCFACE_DST']

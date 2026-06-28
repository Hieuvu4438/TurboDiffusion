"""
Configuration file for Face Recognition Module

Modify the parameters below to customize the face recognition process.
Then simply run: python run.py
"""

# =============================================================================
# INPUT IMAGES
# =============================================================================

# Path to original image (left side in visualization)
ORIGINAL_IMAGE = "/home/haipd/TurboDiffusion/Experiment_Data/Ref/Condoleezza_Rice_ref.jpg"

# Path to reference image (right side in visualization)
REFERENCE_IMAGE = "/home/haipd/TurboDiffusion/Experiment_Data/Side/Condoleezza_Rice_side.jpg"


# =============================================================================
# FACE RECOGNITION SETTINGS
# =============================================================================

# Cosine similarity threshold for face matching
# - Paper default: 0.75
# - For side-view images: 0.5-0.6 recommended
# - Higher = stricter matching
THRESHOLD = 0.75

# InsightFace model name
# Options: 'buffalo_l' (large, accurate), 'buffalo_s' (small, faster)
MODEL_NAME = "buffalo_l"

# Face detection confidence threshold
# - Higher = fewer false positives but may miss some faces
# - Lower = detect more faces but may have false positives
# - Default: 0.5
DETECTION_THRESHOLD = 0.5

# Detection input size (width, height)
# Larger = more accurate but slower
DETECTION_SIZE = (640, 640)


# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

# Output directory for results
OUTPUT_DIR = "output"

# Show debug visualization (displays all detected faces)
DEBUG = False

# Show plot after comparison (set False for batch processing)
SHOW_PLOT = True

# Save comparison image
SAVE_IMAGE = True


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

# Minimum face area ratio to image area (to filter small faces)
MIN_FACE_RATIO = 0.05

# Use CPU only (set True if no GPU available)
USE_CPU = False

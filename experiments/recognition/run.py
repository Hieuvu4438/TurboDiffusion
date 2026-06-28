#!/usr/bin/env python3
"""
Run Face Recognition with config file

Simply edit config.py and run this script:
    python run.py
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ORIGINAL_IMAGE,
    REFERENCE_IMAGE,
    THRESHOLD,
    MODEL_NAME,
    DETECTION_THRESHOLD,
    DETECTION_SIZE,
    OUTPUT_DIR,
    DEBUG,
    SHOW_PLOT,
    SAVE_IMAGE,
    USE_CPU,
)
from face_recognition import FaceRecognizer


def main():
    # Validate inputs
    if not os.path.exists(ORIGINAL_IMAGE):
        print(f"❌ Error: Original image not found: {ORIGINAL_IMAGE}")
        return 1
    
    if not os.path.exists(REFERENCE_IMAGE):
        print(f"❌ Error: Reference image not found: {REFERENCE_IMAGE}")
        return 1
    
    print("="*60)
    print("🔍 Face Recognition - MTCNN + ArcFace")
    print("="*60)
    print(f"\n📋 Configuration:")
    print(f"   Original:  {ORIGINAL_IMAGE}")
    print(f"   Reference: {REFERENCE_IMAGE}")
    print(f"   Threshold: {THRESHOLD}")
    print(f"   Model:     {MODEL_NAME}")
    print(f"   Det Thresh: {DETECTION_THRESHOLD}")
    print(f"   Use CPU:   {USE_CPU}")
    print(f"   Output:    {OUTPUT_DIR}")
    print()
    
    # Initialize recognizer
    recognizer = FaceRecognizer(
        model_name=MODEL_NAME,
        threshold=THRESHOLD,
        det_size=DETECTION_SIZE,
        det_thresh=DETECTION_THRESHOLD,
        use_cpu=USE_CPU,
    )
    
    # Compare faces
    similarity, is_same = recognizer.compare_faces(
        original_path=ORIGINAL_IMAGE,
        reference_path=REFERENCE_IMAGE,
        output_dir=OUTPUT_DIR,
        debug=DEBUG,
        show_plot=SHOW_PLOT,
        save_image=SAVE_IMAGE,
    )
    
    if similarity is not None:
        print("\n" + "="*60)
        if is_same:
            print("✅ RESULT: SAME PERSON")
        else:
            print("❌ RESULT: DIFFERENT PERSONS")
        print("="*60)
        return 0
    
    return 1


if __name__ == "__main__":
    exit(main())

"""
Quick Demo Script for Face Recognition

Simple script to quickly test the face recognition module.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from face_recognition import FaceRecognizer


def demo():
    """
    Demo function showing how to use the FaceRecognizer.
    """
    # Example usage - modify these paths to your images
    # You can also provide paths as command line arguments
    
    if len(sys.argv) >= 3:
        original_path = sys.argv[1]
        reference_path = sys.argv[2]
    else:
        # Default example paths - change these to your images
        script_dir = Path(__file__).parent
        output_dir = script_dir / 'output'
        
        print("Usage: python demo.py <original_image> <reference_image>")
        print("\nNo images provided. Please provide two image paths.")
        print("\nExample:")
        print("  python demo.py /path/to/person1.jpg /path/to/person2.jpg")
        return
    
    # Create output directory
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Initialize recognizer
    print("\n" + "="*60)
    print("Face Recognition Demo")
    print("="*60)
    
    recognizer = FaceRecognizer(
        threshold=0.75,      # Paper threshold
        arcface_model='buffalo_l',  # High accuracy model
        device='cuda'        # Use GPU (change to 'cpu' if no GPU)
    )
    
    # Compare faces
    orig_name = Path(original_path).stem
    ref_name = Path(reference_path).stem
    output_path = output_dir / f'{orig_name}_vs_{ref_name}.png'
    
    similarity, is_same = recognizer.compare_faces(
        original_path,
        reference_path,
        output_path=str(output_path),
        show_plot=True
    )
    
    # Print result
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"Cosine Similarity: {similarity:.4f}")
    print(f"Threshold: 0.75")
    print(f"Verdict: {'SAME PERSON ✓' if is_same else 'DIFFERENT PERSONS ✗'}")
    print(f"Visualization saved: {output_path}")
    print("="*60)


if __name__ == '__main__':
    demo()

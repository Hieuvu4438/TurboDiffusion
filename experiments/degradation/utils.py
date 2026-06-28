"""
Utility functions for image degradation experiments.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import pandas as pd


def load_image(path: str) -> Image.Image:
    """Load an image from path."""
    return Image.open(path).convert('RGB')


def save_image(image: Image.Image, path: str, quality: int = 95):
    """Save an image to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Determine format from extension
    ext = Path(path).suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        image.save(path, 'JPEG', quality=quality)
    elif ext == '.png':
        image.save(path, 'PNG')
    else:
        image.save(path)


def get_person_name(filename: str) -> str:
    """Extract person name from filename."""
    # Remove _side.jpg or _ref.jpg suffix
    name = filename.replace('_side.jpg', '').replace('_ref.jpg', '')
    return name


def list_images(directory: str, suffix: str = '') -> List[Tuple[str, str]]:
    """
    List all images in a directory.
    
    Args:
        directory: Path to directory
        suffix: Optional suffix to filter by (e.g., '_side.jpg')
    
    Returns:
        List of tuples (person_name, full_path)
    """
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            if suffix and not filename.endswith(suffix):
                continue
            person_name = get_person_name(filename)
            full_path = os.path.join(directory, filename)
            images.append((person_name, full_path))
    return images


def create_output_structure(base_output_dir: str):
    """
    Create the output directory structure.
    
    Args:
        base_output_dir: Base output directory path
    """
    subdirs = [
        'jpeg_compression',
        'gaussian_blur',
        'color_clipping',
        'ref',
        'csv'
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(base_output_dir, subdir), exist_ok=True)


def copy_reference_images(ref_dir: str, output_ref_dir: str):
    """
    Copy reference images to output directory.
    
    Args:
        ref_dir: Source reference images directory
        output_ref_dir: Destination directory
    """
    os.makedirs(output_ref_dir, exist_ok=True)
    
    for filename in os.listdir(ref_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(ref_dir, filename)
            dst = os.path.join(output_ref_dir, filename)
            shutil.copy2(src, dst)
    
    print(f"Copied reference images to {output_ref_dir}")


def create_csv_record(
    person_name: str,
    ref_path: str,
    original_side_path: str,
    degraded_side_path: str,
    degradation_type: str,
    params: dict
) -> dict:
    """
    Create a CSV record for a degraded image.
    
    Args:
        person_name: Name of the person
        ref_path: Path to reference image
        original_side_path: Path to original side image
        degraded_side_path: Path to degraded side image
        degradation_type: Type of degradation applied
        params: Dictionary of degradation parameters
    
    Returns:
        Dictionary representing a CSV row
    """
    record = {
        'person_name': person_name,
        'reference_path': ref_path,
        'original_side_path': original_side_path,
        'degraded_side_path': degraded_side_path,
        'degradation_type': degradation_type,
    }
    record.update(params)
    return record


def save_csv(records: List[dict], output_path: str):
    """
    Save records to CSV file.
    
    Args:
        records: List of dictionaries representing rows
        output_path: Path to save CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Saved CSV to {output_path}")

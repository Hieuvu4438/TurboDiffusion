"""
Utility Functions for Face Recognition Module
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


def load_image(image_path: str, convert_rgb: bool = True) -> Optional[np.ndarray]:
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
        convert_rgb: Whether to convert BGR to RGB
        
    Returns:
        Image as numpy array, or None if loading fails
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return None
    
    if convert_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


def save_image(image: np.ndarray, output_path: str, is_rgb: bool = True) -> bool:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array
        output_path: Path to save the image
        is_rgb: Whether the image is in RGB format
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def get_image_files(folder: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image files from a folder.
    
    Args:
        folder: Path to the folder
        extensions: List of valid extensions (default: common image formats)
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    folder = Path(folder)
    image_files = []
    
    for ext in extensions:
        image_files.extend(folder.glob(f'*{ext}'))
        image_files.extend(folder.glob(f'*{ext.upper()}'))
    
    return sorted([str(f) for f in image_files])


def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum dimension (width or height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if max(h, w) <= max_size:
        return image
    
    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    L2-normalize an embedding vector.
    
    Args:
        embedding: Input embedding vector
        
    Returns:
        Normalized embedding vector
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def compute_similarity_matrix(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        Similarity matrix of shape (n, n)
    """
    n = len(embeddings)
    embeddings = np.array(embeddings)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_norm = embeddings / norms
    
    # Compute similarity matrix
    similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
    
    return similarity_matrix


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray, 
                   color: Tuple[int, int, int] = (0, 255, 0),
                   radius: int = 3) -> np.ndarray:
    """
    Draw facial landmarks on an image.
    
    Args:
        image: Input image
        landmarks: Array of landmark coordinates (N, 2)
        color: Color for the landmarks (RGB)
        radius: Radius of the landmark circles
        
    Returns:
        Image with landmarks drawn
    """
    image_copy = image.copy()
    
    for point in landmarks:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image_copy, (x, y), radius, color, -1)
    
    return image_copy


def draw_bounding_box(image: np.ndarray, box: Tuple[int, int, int, int],
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2,
                      label: str = None) -> np.ndarray:
    """
    Draw a bounding box on an image.
    
    Args:
        image: Input image
        box: Bounding box (x, y, width, height)
        color: Color for the box (RGB)
        thickness: Line thickness
        label: Optional label to display
        
    Returns:
        Image with bounding box drawn
    """
    image_copy = image.copy()
    x, y, w, h = box
    
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
    
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
        
        # Draw background for text
        cv2.rectangle(image_copy, (x, y - text_size[1] - 5), 
                     (x + text_size[0], y), color, -1)
        
        # Draw text
        cv2.putText(image_copy, label, (x, y - 5), font, font_scale, 
                   (255, 255, 255), 1, cv2.LINE_AA)
    
    return image_copy

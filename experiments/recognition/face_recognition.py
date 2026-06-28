"""
Face Recognition Module using MTCNN + InsightFace (ArcFace)

This module implements face detection, alignment, and recognition
based on the paper methodology:
- Face Detection & Alignment: MTCNN
- Feature Extraction: ArcFace (InsightFace)
- Similarity: Cosine similarity with 75% threshold
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import insightface
from insightface.app import FaceAnalysis
from skimage import transform as trans
import argparse
import warnings

warnings.filterwarnings('ignore')

# Standard face alignment template for 112x112
# 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


class FaceRecognizer:
    """Face Recognition class using MTCNN for detection and ArcFace for recognition"""
    
    def __init__(self, model_name='buffalo_l', threshold=0.75, det_size=(640, 640), det_thresh=0.5, use_cpu=True):
        """
        Initialize Face Recognizer
        
        Args:
            model_name: InsightFace model name ('buffalo_l', 'buffalo_s', etc.)
            threshold: Cosine similarity threshold for face matching (default: 0.75)
            det_size: Detection size for InsightFace
            det_thresh: Detection confidence threshold (default: 0.5)
            use_cpu: If True, use CPU only; if False, try to use GPU
        """
        self.threshold = threshold
        self.model_name = model_name
        self.det_thresh = det_thresh
        
        # Set providers based on CPU/GPU preference
        if use_cpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Initialize MTCNN detector
        print("Loading MTCNN detector...")
        self.mtcnn = MTCNN()
        
        # Initialize InsightFace model
        print(f"Loading InsightFace model: {model_name}...")
        self.face_app = FaceAnalysis(
            name=model_name, 
            providers=providers,
            allowed_modules=['detection', 'recognition']
        )
        self.face_app.prepare(ctx_id=0 if not use_cpu else -1, det_size=det_size, det_thresh=det_thresh)
        
        print("Models loaded successfully!\n")
    
    def detect_faces_mtcnn(self, image):
        """
        Detect faces using MTCNN
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            List of detected faces with bounding boxes and landmarks
        """
        # MTCNN expects RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        results = self.mtcnn.detect_faces(image)
        return results
    
    def align_face(self, image, landmarks):
        """
        Align face using 5 landmarks to standard 112x112 format
        
        Args:
            image: RGB image as numpy array
            landmarks: Dictionary with 5 facial landmarks from MTCNN
            
        Returns:
            Aligned face image (112x112)
        """
        # Extract 5 landmarks in correct order
        src_pts = np.array([
            landmarks['left_eye'],
            landmarks['right_eye'],
            landmarks['nose'],
            landmarks['mouth_left'],
            landmarks['mouth_right']
        ], dtype=np.float32)
        
        # Estimate similarity transform
        tform = trans.SimilarityTransform()
        tform.estimate(src_pts, ARCFACE_DST)
        
        # Apply transformation
        M = tform.params[0:2, :]
        aligned_face = cv2.warpAffine(image, M, (112, 112), borderValue=0)
        
        return aligned_face
    
    def align_face_from_kps(self, image, kps):
        """
        Align face using 5 keypoints from InsightFace
        
        Args:
            image: RGB image as numpy array
            kps: 5x2 array of keypoints
            
        Returns:
            Aligned face image (112x112)
        """
        src_pts = np.array(kps, dtype=np.float32)
        
        # Estimate similarity transform
        tform = trans.SimilarityTransform()
        tform.estimate(src_pts, ARCFACE_DST)
        
        # Apply transformation
        M = tform.params[0:2, :]
        aligned_face = cv2.warpAffine(image, M, (112, 112), borderValue=0)
        
        return aligned_face
    
    def select_best_face(self, faces, image_shape):
        """
        Select the best face from multiple detections.
        Prioritize: largest face that is closest to center
        
        Args:
            faces: List of detected faces from InsightFace
            image_shape: Shape of the image (H, W, C)
            
        Returns:
            Best face detection
        """
        if len(faces) == 1:
            return faces[0]
        
        h, w = image_shape[:2]
        center_x, center_y = w / 2, h / 2
        
        best_face = None
        best_score = -1
        
        for face in faces:
            bbox = face.bbox
            face_w = bbox[2] - bbox[0]
            face_h = bbox[3] - bbox[1]
            face_area = face_w * face_h
            
            # Face center
            face_cx = (bbox[0] + bbox[2]) / 2
            face_cy = (bbox[1] + bbox[3]) / 2
            
            # Distance from image center (normalized)
            dist = np.sqrt(((face_cx - center_x) / w) ** 2 + ((face_cy - center_y) / h) ** 2)
            
            # Score: larger area and closer to center is better
            # Normalize area by image size
            area_score = face_area / (w * h)
            center_score = 1 - dist  # closer = higher score
            
            # Combined score (weight area more)
            score = area_score * 0.7 + center_score * 0.3
            
            # Also consider detection confidence
            if hasattr(face, 'det_score'):
                score *= face.det_score
            
            if score > best_score:
                best_score = score
                best_face = face
        
        return best_face
    
    def extract_embedding_insightface(self, aligned_face):
        """
        Extract face embedding using InsightFace ArcFace model
        
        Args:
            aligned_face: Aligned face image (112x112, RGB)
            
        Returns:
            512-dimensional embedding vector
        """
        # InsightFace expects BGR
        face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
        
        # Use the recognition model directly
        for model in self.face_app.models.values():
            if hasattr(model, 'get_feat') or 'recognition' in str(type(model)).lower():
                # Prepare input
                face_input = cv2.resize(face_bgr, (112, 112))
                face_input = np.transpose(face_input, (2, 0, 1))  # HWC to CHW
                face_input = np.expand_dims(face_input, axis=0).astype(np.float32)
                face_input = (face_input - 127.5) / 127.5
                
                # Get embedding
                embedding = model.session.run(None, {model.session.get_inputs()[0].name: face_input})[0]
                return embedding.flatten()
        
        return None
    
    def get_embedding(self, image_path, debug=False):
        """
        Complete pipeline: Load image -> Detect face -> Align -> Extract embedding
        
        Args:
            image_path: Path to image file
            debug: If True, show debug visualization
            
        Returns:
            tuple: (embedding vector, aligned face image, original image) or (None, None, None) if failed
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return None, None, None
        
        # Convert BGR to RGB for MTCNN
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"  Image size: {image.shape}")
        
        # Method 1: Try InsightFace detection first (more robust)
        print("  Trying InsightFace detection...")
        faces = self.face_app.get(image)
        
        if len(faces) > 0:
            print(f"  InsightFace detected {len(faces)} face(s)")
            
            # Select the best face (largest + closest to center)
            face = self.select_best_face(faces, image.shape)
            
            bbox = face.bbox.astype(int)
            print(f"  Selected face bbox: {bbox}, score: {face.det_score:.3f}")
            
            embedding = face.embedding
            
            # Use landmarks to align
            if face.kps is not None:
                aligned_face = self.align_face_from_kps(image_rgb, face.kps)
            else:
                # Fallback: crop and resize
                x1, y1, x2, y2 = bbox
                h, w = image.shape[:2]
                pad = int(0.1 * max(x2-x1, y2-y1))
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(w, x2+pad), min(h, y2+pad)
                aligned_face = cv2.resize(image_rgb[y1:y2, x1:x2], (112, 112))
            
            if debug:
                self._debug_detection(image_rgb, faces, face)
            
            return embedding, aligned_face, image_rgb
        
        # Method 2: Try MTCNN detection
        print("  Trying MTCNN detection...")
        detections = self.detect_faces_mtcnn(image_rgb)
        
        if len(detections) > 0:
            print(f"  MTCNN detected {len(detections)} face(s)")
            # Get the largest face
            detection = max(detections, key=lambda x: x['box'][2] * x['box'][3])
            
            confidence = detection['confidence']
            print(f"  Detection confidence: {confidence:.3f}")
            
            if confidence < 0.9:
                print(f"  Warning: Low confidence detection")
            
            landmarks = detection['keypoints']
            
            # Align face
            aligned_face = self.align_face(image_rgb, landmarks)
            
            # Extract embedding using InsightFace recognition model
            embedding = self.extract_embedding_insightface(aligned_face)
            
            if embedding is not None:
                return embedding, aligned_face, image_rgb
        
        # Method 3: If no face detected, try with the whole image (for cropped face images)
        print("  No face detected, trying whole image as face...")
        
        # Resize to 112x112 and try to get embedding
        face_resized = cv2.resize(image_rgb, (112, 112))
        embedding = self.extract_embedding_insightface(face_resized)
        
        if embedding is not None:
            return embedding, face_resized, image_rgb
        
        print(f"  Warning: Could not extract embedding from {image_path}")
        return None, None, None
    
    def _debug_detection(self, image, all_faces, selected_face):
        """Show debug visualization of face detection"""
        img_debug = image.copy()
        
        # Draw all faces in blue
        for face in all_faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(img_debug, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        
        # Draw selected face in green
        bbox = selected_face.bbox.astype(int)
        cv2.rectangle(img_debug, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        
        # Draw keypoints
        if selected_face.kps is not None:
            for kp in selected_face.kps:
                cv2.circle(img_debug, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img_debug)
        plt.title(f"Detected {len(all_faces)} faces (green = selected)")
        plt.axis('off')
        plt.show()
    
    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        return similarity
    
    def compare_faces(self, original_path, reference_path, output_dir='output', debug=False, show_plot=True, save_image=True):
        """
        Compare two face images and visualize the result
        
        Args:
            original_path: Path to original image
            reference_path: Path to reference image
            output_dir: Directory to save output
            debug: If True, show debug visualizations
            show_plot: If True, display matplotlib plot
            save_image: If True, save comparison image to output_dir
            
        Returns:
            tuple: (similarity_score, is_same_person)
        """
        print(f"Processing original image: {original_path}")
        embedding1, aligned1, orig1 = self.get_embedding(original_path, debug=debug)
        
        print(f"Processing reference image: {reference_path}")
        embedding2, aligned2, orig2 = self.get_embedding(reference_path, debug=debug)
        
        if embedding1 is None or embedding2 is None:
            print("\nError: Could not extract embeddings from one or both images.")
            
            # Still try to visualize the images
            img1 = cv2.imread(original_path)
            img2 = cv2.imread(reference_path)
            
            if img1 is not None and img2 is not None:
                self._visualize_error(img1, img2, original_path, reference_path, output_dir)
            
            return None, None
        
        # Calculate similarity
        similarity = self.cosine_similarity(embedding1, embedding2)
        is_same = similarity >= self.threshold
        
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"{'='*60}")
        print(f"Cosine Similarity: {similarity:.4f}")
        print(f"Threshold: {self.threshold}")
        print(f"Same Person: {'Yes' if is_same else 'No'}")
        print(f"{'='*60}\n")
        
        # Visualize
        self._visualize_comparison(
            aligned1, aligned2,
            orig1, orig2,
            similarity, is_same,
            original_path, reference_path,
            output_dir,
            show_plot=show_plot,
            save_image=save_image
        )
        
        return similarity, is_same
    
    def _visualize_error(self, img1, img2, path1, path2, output_dir):
        """Visualize images when face detection fails"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Convert BGR to RGB for display
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(img1_rgb)
        axes[0].set_title(f'Original\n{os.path.basename(path1)}', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(img2_rgb)
        axes[1].set_title(f'Reference\n{os.path.basename(path2)}', fontsize=12)
        axes[1].axis('off')
        
        fig.suptitle('Face Detection Failed - Could not extract embeddings', 
                     fontsize=14, color='red', fontweight='bold')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'comparison_error.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Error visualization saved to: {output_path}")
        plt.show()
    
    def _visualize_comparison(self, face1, face2, orig1, orig2, similarity, is_same, 
                              path1, path2, output_dir, show_plot=True, save_image=True):
        """
        Visualize comparison result with both original and aligned faces
        
        Args:
            face1: First aligned face (RGB)
            face2: Second aligned face (RGB)
            orig1: First original image (RGB)
            orig2: Second original image (RGB)
            similarity: Cosine similarity score
            is_same: Boolean indicating if same person
            path1: Path to original image
            path2: Path to reference image
            output_dir: Output directory
            show_plot: If True, display matplotlib plot
            save_image: If True, save comparison image
        """
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Top row: Original images
        axes[0, 0].imshow(orig1)
        axes[0, 0].set_title(f'Original Image\n{os.path.basename(path1)}', fontsize=11)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(orig2)
        axes[0, 1].set_title(f'Reference Image\n{os.path.basename(path2)}', fontsize=11)
        axes[0, 1].axis('off')
        
        # Bottom row: Aligned faces
        axes[1, 0].imshow(face1)
        axes[1, 0].set_title('Aligned Face (Original)', fontsize=11)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(face2)
        axes[1, 1].set_title('Aligned Face (Reference)', fontsize=11)
        axes[1, 1].axis('off')
        
        # Set main title with similarity score
        match_status = "MATCH ✓" if is_same else "NO MATCH ✗"
        color = 'green' if is_same else 'red'
        
        fig.suptitle(
            f'Cosine Similarity: {similarity:.4f} | Threshold: {self.threshold} | {match_status}',
            fontsize=14,
            fontweight='bold',
            color=color
        )
        
        plt.tight_layout()
        
        # Save figure
        if save_image:
            output_filename = f"comparison_{os.path.splitext(os.path.basename(path1))[0]}_vs_{os.path.splitext(os.path.basename(path2))[0]}.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Comparison result saved to: {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Face Recognition using MTCNN + ArcFace')
    parser.add_argument('-o', '--original', required=True, help='Path to original image')
    parser.add_argument('-r', '--reference', required=True, help='Path to reference image')
    parser.add_argument('-t', '--threshold', type=float, default=0.75, help='Similarity threshold (default: 0.75)')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--model', default='buffalo_l', help='InsightFace model name')
    parser.add_argument('--debug', action='store_true', help='Show debug visualizations')
    
    args = parser.parse_args()
    
    # Initialize recognizer
    recognizer = FaceRecognizer(model_name=args.model, threshold=args.threshold)
    
    print(f"\n{'='*60}")
    print("Face Recognition Comparison")
    print(f"{'='*60}\n")
    
    # Compare faces
    similarity, is_same = recognizer.compare_faces(
        args.original,
        args.reference,
        args.output,
        debug=args.debug
    )
    
    if similarity is not None:
        return 0 if is_same else 1
    return 2


if __name__ == '__main__':
    exit(main())
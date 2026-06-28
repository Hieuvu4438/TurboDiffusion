# Face Recognition Module using MTCNN + ArcFace (InsightFace)

This module implements face recognition based on the paper methodology:
> "Face Recognition: We employ ArcFace architecture with MTCNN for detection and alignment. Recognition uses cosine similarity with 75% threshold"

## Features

- **Face Detection**: MTCNN for robust face detection
- **Face Alignment**: 5-point landmark alignment to 112x112 standard template
- **Feature Extraction**: ArcFace (InsightFace) for 512-dimensional embeddings
- **Similarity Calculation**: Cosine similarity with configurable threshold
- **Visualization**: Side-by-side comparison with matplotlib

## Installation

```bash
cd mtcnn_insightface
pip install -r requirements.txt
```

## Usage

### Single Image Pair Comparison

```bash
# Basic usage
python face_recognition.py -o /path/to/original.jpg -r /path/to/reference.jpg

# With custom threshold
python face_recognition.py -o /path/to/original.jpg -r /path/to/reference.jpg -t 0.80

# Save visualization to file
python face_recognition.py -o /path/to/original.jpg -r /path/to/reference.jpg --output result.png

# CPU-only mode
python face_recognition.py -o /path/to/original.jpg -r /path/to/reference.jpg -d cpu
```

### Batch Processing

```bash
# Using two folders with matching filenames
python batch_processing.py --folders /path/to/originals /path/to/references -o ./output

# Using a CSV file
python batch_processing.py --csv image_pairs.csv -o ./output

# Custom threshold and no visualization
python batch_processing.py --folders ./originals ./references -t 0.80 --no-vis
```

### Python API

```python
from face_recognition import FaceRecognizer

# Initialize
recognizer = FaceRecognizer(threshold=0.75, device='cuda')

# Compare two images
similarity, is_same = recognizer.compare_faces(
    'original.jpg',
    'reference.jpg',
    output_path='comparison.png',
    show_plot=True
)

print(f"Similarity: {similarity:.4f}")
print(f"Same person: {is_same}")
```

## Pipeline Details

### 1. Face Detection (MTCNN)
- Multi-task CNN for face detection
- Outputs bounding box and 5 facial landmarks:
  - Left eye, Right eye, Nose, Left mouth corner, Right mouth corner

### 2. Face Alignment
- Similarity transformation based on 5 landmarks
- Aligns face to ArcFace standard template (112x112)
- Ensures consistent pose for recognition

### 3. Feature Extraction (ArcFace/InsightFace)
- Uses pre-trained ArcFace model (buffalo_l by default)
- Extracts 512-dimensional embedding vector
- Embedding is L2-normalized

### 4. Similarity Calculation
- Cosine similarity between two embeddings
- Range: [-1, 1] (typically [0, 1] for face embeddings)
- Default threshold: 0.75

## Output

The visualization shows:
- **Left**: Original image
- **Right**: Reference image
- **Title**: Cosine similarity score, threshold, and match status

Results are saved in the `output/` directory by default.

## References

- [MTCNN Paper](https://arxiv.org/abs/1604.02878)
- [ArcFace Paper](https://arxiv.org/abs/1801.07698)
- [InsightFace](https://github.com/deepinsight/insightface)

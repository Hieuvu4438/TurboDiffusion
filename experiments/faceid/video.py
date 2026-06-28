"""Video / face-region utilities for best-frame selection.

Extracted from ``extract_face_from_video.py`` (lines 37-156). These scoring
helpers (sharpness, yaw, tail-frame extraction, multi-criteria face scoring)
were inlined in the standalone extractor; centralizing them lets other
pipeline stages reuse the same best-frontal-frame logic.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

# How much of the video tail to scan for the best frontal face.
DEFAULT_TAIL_RATIO = 0.25      # last 25% of frames
MIN_TAIL_FRAMES = 10           # at least 10 frames
MAX_TAIL_FRAMES = 40           # at most 40 frames (avoid wasting time)


def compute_sharpness(image: np.ndarray) -> float:
    """Laplacian variance – higher means sharper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def estimate_yaw(kps: np.ndarray) -> float:
    """Rough yaw estimation from 5-point landmarks.

    ``kps`` has shape (5, 2): left_eye, right_eye, nose, left_mouth, right_mouth.
    Returns absolute yaw angle in degrees (0 = perfect frontal).
    """
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_width = abs(right_eye[0] - left_eye[0])
    if eye_width < 1e-6:
        return 90.0
    # nose offset from eye center, normalized
    nose_offset = (nose[0] - eye_center_x) / eye_width
    # approximate: offset of 0 -> frontal, offset of +-0.5 -> ~90 deg
    yaw_deg = abs(nose_offset) * 180.0
    return min(yaw_deg, 90.0)


def extract_tail_frames(video_path: str,
                        tail_ratio: float = DEFAULT_TAIL_RATIO) -> List[Tuple[int, np.ndarray]]:
    """Read frames from the tail portion of a video. Returns list of (frame_index, bgr_image)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Video has 0 frames: {video_path}")

    n_tail = int(total_frames * tail_ratio)
    n_tail = max(min(n_tail, MAX_TAIL_FRAMES), min(MIN_TAIL_FRAMES, total_frames))
    start_frame = max(0, total_frames - n_tail)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    idx = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((idx, frame))
        idx += 1

    cap.release()
    return frames


def score_face(face, frame: np.ndarray, frame_idx: int, total_frames: int) -> Dict:
    """Score a detected face on multiple criteria; returns a dict with a combined score."""
    h, w = frame.shape[:2]
    bbox = face.bbox  # [x1, y1, x2, y2]
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]

    # 1. Detection confidence
    det_score = float(face.det_score) if hasattr(face, 'det_score') else 0.5
    # 2. Face size (larger = better, normalized by frame area)
    area_ratio = (face_w * face_h) / (w * h)
    # 3. Yaw angle (frontal = 0 -> higher score)
    yaw = 90.0
    if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 3:
        yaw = estimate_yaw(face.kps)
    frontal_score = max(0.0, 1.0 - yaw / 45.0)  # 0 deg -> 1.0, 45 deg+ -> 0.0
    # 4. Sharpness of face region
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
    face_crop = frame[y1:y2, x1:x2]
    sharpness = compute_sharpness(face_crop) if face_crop.size > 0 else 0.0
    # 5. Frame position bonus (later frames slightly preferred)
    position_score = frame_idx / total_frames if total_frames > 0 else 0.5

    combined = (
        det_score      * 0.25 +
        area_ratio     * 0.15 +
        frontal_score  * 0.35 +
        min(sharpness / 500.0, 1.0) * 0.15 +
        position_score * 0.10
    )

    return {
        "det_score": det_score,
        "area_ratio": area_ratio,
        "yaw": yaw,
        "frontal_score": frontal_score,
        "sharpness": sharpness,
        "position_score": position_score,
        "combined": combined,
        "frame_idx": frame_idx,
        "face": face,
        "bbox": (x1, y1, x2, y2),
    }

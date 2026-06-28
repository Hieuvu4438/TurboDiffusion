"""DeepFace face-embedding extraction (ArcFace + RetinaFace).

Extracted from ``evaluate_identity_preservation_v2_deepface.py`` (lines 64-159).
Requires the ``deepface`` package; the ``DeepFace`` symbol is imported lazily
inside the helpers so the module can be imported without the package installed.
"""

from __future__ import annotations

import os
import cv2
import numpy as np

from experiments.config import SKIP_START_PERCENT, SKIP_END_PERCENT, TOP_K_AVERAGE, DEEPFACE_DETECTOR
from .metrics import cosine_similarity


def _deepface_represent(img, model_name, detector_backend=DEEPFACE_DETECTOR):
    """Run DeepFace.represent on a BGR numpy image, return list of face dicts or []."""
    try:
        from deepface import DeepFace  # imported lazily
    except ImportError as e:
        raise ImportError(
            "DeepFace is required for experiments.faceid.embeddings_deepface. "
            "Install with `pip install deepface`."
        ) from e

    try:
        return DeepFace.represent(
            img_path=img,
            model_name=model_name,
            enforce_detection=False,
            detector_backend=detector_backend,
        )
    except Exception:
        return []


def _pick_best_face(faces, ref_emb=None):
    """Pick best face from DeepFace results: by similarity to ref_emb, else by area."""
    if ref_emb is not None:
        best_emb, best_sim = None, -1.0
        for r in faces:
            emb = np.array(r["embedding"], dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            sim = cosine_similarity(emb, ref_emb)
            if sim > best_sim:
                best_sim, best_emb = sim, emb
        return best_emb

    # No ref_emb: pick largest face
    best = max(faces, key=lambda r: r["facial_area"]["w"] * r["facial_area"]["h"])
    emb = np.array(best["embedding"], dtype=np.float32)
    return emb / (np.linalg.norm(emb) + 1e-10)


def get_embedding_robust(img, model_name, ref_emb=None):
    """Extract face embedding from image using DeepFace, with fallback padding."""
    if img is None:
        return None, None

    faces = _deepface_represent(img, model_name)

    if not faces:
        # Fallback: pad and retry
        img_pad = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        faces = _deepface_represent(img_pad, model_name)
        if not faces:
            return None, None

    emb = _pick_best_face(faces, ref_emb)
    return emb, faces


def get_image_embedding(path, model_name):
    """Extract embedding from static reference image."""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    emb, _ = get_embedding_robust(img, model_name)
    return emb


def get_best_embedding_from_video(video_path, model_name, ref_emb):
    """Scan video, get top-K frames with highest similarity to ref_emb,
    return mean of normalized embeddings (same approach as v2/test2.py).
    """
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    start_f = int(total_frames * SKIP_START_PERCENT)
    end_f = int(total_frames * (1 - SKIP_END_PERCENT))

    candidates = []
    for i in range(start_f, end_f, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        emb, _ = get_embedding_robust(frame, model_name, ref_emb)
        if emb is not None:
            sim = cosine_similarity(emb, ref_emb)
            candidates.append((sim, emb))
    cap.release()

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)

    top_embs = [c[1] for c in candidates[:TOP_K_AVERAGE]] if len(candidates) >= TOP_K_AVERAGE else [candidates[0][1]]
    avg_emb = np.mean(top_embs, axis=0)
    norm = np.linalg.norm(avg_emb)
    return avg_emb / norm if norm > 0 else avg_emb

"""InsightFace face-embedding extraction.

Extracted from ``evaluate_identity_preservation_v2.py`` (lines 69-166).
Requires the ``insightface`` package (imported lazily so importing this module
without the package only fails at call time, not at module import).
"""

from __future__ import annotations

import os
import cv2
import numpy as np

from experiments.config import SKIP_START_PERCENT, SKIP_END_PERCENT, TOP_K_AVERAGE
from .metrics import cosine_similarity
from .preprocessing import preprocess_for_detection, progressive_padding


def get_embedding_robust(img, app, ref_emb=None, noise_type=None):
    """Trích xuất face embedding từ ảnh, progressive fallback nếu không detect được.

    Strategies (in order):
      1. Raw image.
      2. Noise-specific preprocessing (CLAHE / unsharp / median).
      3. Progressive black padding (100 -> 200 -> 300 px).
      4. Preprocessing + padding combined.

    When ``ref_emb`` is given, the face most similar to it is returned;
    otherwise the largest face is returned.
    Returns ``(embedding, face)`` or ``(None, None)``.
    """
    if img is None:
        return None, None

    faces = app.get(img)

    # Strategy 2: preprocessing
    if not faces and noise_type is not None:
        faces = app.get(preprocess_for_detection(img, noise_type))

    # Strategy 3: progressive padding
    if not faces:
        for _, img_pad in progressive_padding(img):
            faces = app.get(img_pad)
            if faces:
                break

        # Strategy 4: preprocessing + padding
        if not faces and noise_type is not None:
            enhanced = preprocess_for_detection(img, noise_type)
            for pad_size, img_pad in progressive_padding(enhanced, (200, 300)):
                faces = app.get(img_pad)
                if faces:
                    break

    if not faces:
        return None, None

    if ref_emb is not None:
        best_face, max_sim = None, -1.0
        for face in faces:
            sim = cosine_similarity(face.normed_embedding, ref_emb)
            if sim > max_sim:
                max_sim, best_face = sim, face
        return best_face.embedding, best_face

    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest.embedding, largest


def get_image_embedding(path, app):
    """Trích xuất embedding từ ảnh tĩnh."""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    emb, _ = get_embedding_robust(img, app)
    return emb


def get_best_embedding_from_video(video_path, app, ref_emb):
    """Quét video, lấy top-K frame có similarity cao nhất với ref,
    trả về trung bình embedding đã normalize (theo test2.py).
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
        emb, _ = get_embedding_robust(frame, app, ref_emb)
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

"""Similarity and classification metrics for identity-preservation evaluation.

Extracted verbatim (logic preserved) from the ``evaluate_identity_preservation``
family of scripts, where the same ``cosine_similarity`` and the same
accuracy/precision/recall/F1 block were copy-pasted across nine files.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def cosine_similarity(emb1, emb2) -> float:
    """Cosine similarity between two embedding vectors.

    Returns ``0.0`` if either embedding is ``None`` (matches the original
    convention so downstream ``np.mean`` never propagates NaN).
    """
    if emb1 is None or emb2 is None:
        return 0.0
    emb1 = np.asarray(emb1, dtype=np.float32)
    emb2 = np.asarray(emb2, dtype=np.float32)
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def compute_classification_metrics(y_true, y_scores, threshold: float = 0.15) -> dict:
    """Accuracy / precision / recall / F1 at a fixed cosine-similarity threshold.

    ``y_scores`` are continuous cosine similarities; predictions are
    ``score >= threshold``. ``zero_division=0`` matches the originals.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mean_cosine": float(np.mean(y_scores)) if len(y_scores) else float("nan"),
    }


def find_optimal_threshold(y_true, y_scores, step: float = 0.01) -> tuple:
    """Sweep thresholds over [0, 1] and return ``(best_threshold, best_f1)``.

    Used by the reporting path to quote the F1-maximizing threshold alongside
    the fixed 0.15 operating point.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    best_thr, best_f1 = 0.0, -1.0
    for thr in np.arange(0.0, 1.0 + step, step):
        y_pred = (y_scores >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)

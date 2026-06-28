"""Shared face-identity utilities (metrics, preprocessing, video, IO, reporting).

The two embedding backends live in submodules and are imported explicitly so
that ``import experiments.faceid`` does not require ``insightface`` or
``deepface`` to be installed:

    from experiments.faceid import cosine_similarity, compute_classification_metrics
    from experiments.faceid.embeddings_insightface import get_embedding_robust   # needs insightface
    from experiments.faceid.embeddings_deepface import get_embedding_robust      # needs deepface
"""

from .metrics import (
    cosine_similarity,
    compute_classification_metrics,
    find_optimal_threshold,
)
from .preprocessing import preprocess_for_detection, progressive_padding
from .io import extract_person_name, person_from_ref_files, read_csv_rows, write_csv
from .video import compute_sharpness, estimate_yaw, score_face
from .reporting import format_summary_report, write_summary_report

__all__ = [
    # metrics
    "cosine_similarity",
    "compute_classification_metrics",
    "find_optimal_threshold",
    # preprocessing
    "preprocess_for_detection",
    "progressive_padding",
    # io
    "extract_person_name",
    "person_from_ref_files",
    "read_csv_rows",
    "write_csv",
    # video
    "compute_sharpness",
    "estimate_yaw",
    "score_face",
    # reporting
    "format_summary_report",
    "write_summary_report",
]

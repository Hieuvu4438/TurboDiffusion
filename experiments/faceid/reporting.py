"""Report formatting for identity-preservation evaluation.

Reproduces the ``summary.txt`` block written by the
``evaluate_identity_preservation*`` family, so all backends emit identical
report shapes for the downstream ``extract_metrics`` / ``generate_deepface_summary``
aggregators to parse.
"""

from __future__ import annotations

from pathlib import Path


def format_summary_report(view, noise_type, level, metrics, n_persons, n_pairs) -> str:
    """Build the canonical per-(view, noise, level) summary report text.

    ``metrics`` is the dict returned by
    :func:`experiments.faceid.metrics.compute_classification_metrics`.
    """
    pos = n_pairs // 2
    neg = n_pairs - pos
    m = metrics
    return (
        f"View: {view} | Noise: {noise_type} | Level: {level}\n"
        f"Total Persons: {n_persons}\n"
        f"Total Pairs: {n_pairs} (Pos: {pos}, Neg: {neg})\n"
        f"Threshold: {m['threshold']}\n"
        f"Mean Cosine Similarity: {m['mean_cosine']:.4f}\n"
        f"Accuracy:  {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)\n"
        f"Precision: {m['precision']:.4f}  ({m['precision']*100:.1f}%)\n"
        f"Recall:    {m['recall']:.4f}  ({m['recall']*100:.1f}%)\n"
        f"F1-Score:  {m['f1']:.4f}\n"
    )


def write_summary_report(out_dir, view, noise_type, level, metrics, n_persons, n_pairs) -> Path:
    """Format and write ``summary.txt`` into ``out_dir``; returns the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = format_summary_report(view, noise_type, level, metrics, n_persons, n_pairs)
    path = out_dir / "summary.txt"
    path.write_text(text)
    return path

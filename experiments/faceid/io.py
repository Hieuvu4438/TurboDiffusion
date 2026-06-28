"""CSV / filename helpers shared across the pipeline.

``extract_person_name`` is lifted verbatim from the evaluation scripts (it
appeared in every ``evaluate_identity_preservation*`` and
``run_cosine_similarity_for_degraded_input*`` file).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path


def extract_person_name(stem, known_persons):
    """Tìm person name trong stem bằng cách khớp prefix.

    Matches the longest known person name that is a prefix of ``stem``
    (e.g. stem ``"John_Smith_blurred_8"`` -> ``"John_Smith"``).
    """
    for person in sorted(known_persons, key=len, reverse=True):
        if stem.startswith(person):
            return person
    return None


def person_from_ref_files(ref_dir):
    """Return sorted (longest-first) person names from ``<person>_ref.jpg`` files."""
    persons = []
    for f in os.listdir(ref_dir):
        if f.endswith("_ref.jpg"):
            persons.append(f.replace("_ref.jpg", ""))
    return sorted(persons, key=len, reverse=True)


def read_csv_rows(csv_path):
    """Read a CSV with a ``person`` column into a list of dicts (empty rows skipped)."""
    csv_path = Path(csv_path)
    rows = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            if row.get("person", "").strip():
                rows.append(row)
    return rows


def write_csv(rows, csv_path, fieldnames=None):
    """Write a list of dict rows to ``csv_path`` (fieldnames inferred from row 0)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path

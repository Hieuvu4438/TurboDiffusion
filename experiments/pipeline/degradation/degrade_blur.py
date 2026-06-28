import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parents[3]) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
"""
Degradation script: Apply Gaussian blur (by sigma) to test images from CSV datasets.

Usage:
    python scripts/degrade_blur.py

This script reads lfw_side_test.csv and lfw_frontal_test.csv, applies Gaussian
blur with sigma values [3, 5, 8, 10, 12, 15] to the test images only, and saves
the results organized by experiment type (side/frontal) and sigma parameter.

The blur uses cv2.GaussianBlur(image, (0, 0), sigma) which lets OpenCV
automatically compute the appropriate kernel size from sigma.

Output structure:
    degradation_experiment/blurred_output/
    ├── side/
    │   ├── blurred3/
    │   │   ├── side_blurred3.csv
    │   │   └── {Person}_test_blurred_3.jpg
    │   ├── blurred5/
    │   │   └── ...
    │   └── ...
    └── frontal/
        ├── blurred3/
        │   ├── frontal_blurred3.csv
        │   └── {Person}_test_blurred_3.jpg
        └── ...
"""

import os
import csv
import cv2
import numpy as np
from pathlib import Path


# ─── Configuration ───────────────────────────────────────────────────────────
from experiments.config import PROJECT_ROOT as BASE_DIR
OUTPUT_DIR = BASE_DIR / "degradation_experiment" / "blurred_output_new"

SIGMA_PARAMS = [3, 5, 8, 10, 12, 15]

DATASETS = {
    "side": {
        "csv_path": BASE_DIR / "lfw_side_test_added.csv",
        "output_dir": OUTPUT_DIR / "side",
    },
    "frontal": {
        "csv_path": BASE_DIR / "lfw_frontal_test_added.csv",
        "output_dir": OUTPUT_DIR / "frontal",
    },
}


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur with a given sigma.

    Uses kernel size (0, 0) so OpenCV automatically determines the
    appropriate kernel size from sigma.  This matches the project's
    canonical ``apply_gaussian_blur`` in degradation/gaussian_blur.py.
    """
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def process_dataset(dataset_name: str, config: dict):
    """Process one dataset (side or frontal) with all sigma values."""

    csv_path = config["csv_path"]
    output_base = config["output_dir"]

    # Read source CSV
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["person"].strip():
                rows.append(row)

    print(f"\n{'='*60}")
    print(f"  Processing [{dataset_name.upper()}] dataset: {len(rows)} images")
    print(f"{'='*60}")

    for sigma in SIGMA_PARAMS:
        # Folder naming matches /home/haipd/TurboDiffusion/image/blurred{sigma}
        blur_folder = output_base / f"blurred{sigma}"
        blur_folder.mkdir(parents=True, exist_ok=True)

        csv_rows = []

        print(f"\n  ▸ sigma={sigma} → {blur_folder}")

        for row in rows:
            person = row["person"]
            test_path = row["test_path"]
            ref_path = row["ref_path"]

            # Read test image
            img_full_path = BASE_DIR / test_path
            img = cv2.imread(str(img_full_path))

            if img is None:
                print(f"    ⚠ WARNING: Cannot read {img_full_path}")
                continue

            # Apply Gaussian blur with sigma
            blurred = apply_gaussian_blur(img, sigma)

            # Save blurred image (naming matches reference: {Person}_test_blurred_{sigma}.jpg)
            out_filename = f"{person}_test_blurred_{sigma}.jpg"
            out_path = blur_folder / out_filename
            cv2.imwrite(str(out_path), blurred)

            # Record for CSV (relative path from BASE_DIR)
            blurred_rel_path = str(out_path.relative_to(BASE_DIR))

            csv_rows.append({
                "person": person,
                "test_path": blurred_rel_path,
                "ref_path": ref_path,
                f"blurred_{sigma}": sigma,
            })

        # Write CSV for this sigma value
        csv_output_path = blur_folder / f"{dataset_name}_blurred{sigma}.csv"
        fieldnames = ["person", "test_path", "ref_path", f"blurred_{sigma}"]

        with open(csv_output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"    ✓ {len(csv_rows)} images blurred (sigma={sigma}) & saved")
        print(f"    ✓ CSV saved: {csv_output_path.relative_to(BASE_DIR)}")


def main():
    print("=" * 60)
    print("  Gaussian Blur Degradation Script (sigma-based)")
    print(f"  Sigma values: {SIGMA_PARAMS}")
    print(f"  Output: {OUTPUT_DIR.relative_to(BASE_DIR)}")
    print("=" * 60)

    for dataset_name, config in DATASETS.items():
        process_dataset(dataset_name, config)

    print(f"\n{'='*60}")
    print("  ✅ All done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

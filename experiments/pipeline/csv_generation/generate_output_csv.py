"""
Generate CSV files for each blurred folder in output_full.

For each blurred level (blurred3, blurred5, ...) in output_full/{frontal,side},
creates a CSV file containing:
  - person: person name
  - degraded_image_path: path to the blurred image in blurred_output
  - output_video_path: path to the inference video in output_full
  - ref_image_path: path to the reference image in Experiment_Data_Split
  - test_image_path: path to the original test image in Experiment_Data_Split

CSV files are saved in output_full/{frontal,side}/{blurredN}/{view}_blurred{N}_full.csv
"""

import os
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ─── Configuration ──────────────────────────────────────────────────────────
OUTPUT_FULL_DIR = PROJECT_ROOT / "output_full"
BLURRED_OUTPUT_DIR = PROJECT_ROOT / "degradation_experiment" / "blurred_output"

EXPERIMENT_DATA = {
    "frontal": {
        "ref": PROJECT_ROOT / "Experiment_Data_Split" / "Frontal_Exp" / "Ref",
        "test": PROJECT_ROOT / "Experiment_Data_Split" / "Frontal_Exp" / "Test",
    },
    "side": {
        "ref": PROJECT_ROOT / "Experiment_Data_Split" / "Side_Exp" / "Ref",
        "test": PROJECT_ROOT / "Experiment_Data_Split" / "Side_Exp" / "Test",
    },
}

VIEWS = ["frontal", "side"]
BLUR_LEVELS = ["blurred3", "blurred5", "blurred8", "blurred10", "blurred12", "blurred15"]


def extract_person_from_video(video_filename: str, view: str, blur_name: str) -> str:
    """
    Extract person name from video filename.
    e.g. 'Bill_Clinton_side_blurred3.mp4' → 'Bill_Clinton'
    """
    # Remove .mp4 extension
    stem = Path(video_filename).stem  # e.g. 'Bill_Clinton_side_blurred3'
    # Remove the suffix '_{view}_{blur_name}'
    suffix = f"_{view}_{blur_name}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def extract_blur_number(blur_name: str) -> str:
    """Extract the numeric part from blur folder name. e.g. 'blurred3' → '3'"""
    return blur_name.replace("blurred", "")


def generate_csv_for_folder(view: str, blur_name: str):
    """Generate a CSV file for a specific view/blur combination."""
    blur_num = extract_blur_number(blur_name)

    # Directories
    video_dir = OUTPUT_FULL_DIR / view / blur_name
    blurred_img_dir = BLURRED_OUTPUT_DIR / view / blur_name
    ref_dir = EXPERIMENT_DATA[view]["ref"]
    test_dir = EXPERIMENT_DATA[view]["test"]

    if not video_dir.exists():
        print(f"  ⚠ Video dir not found: {video_dir}")
        return

    # List all video files
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

    if not video_files:
        print(f"  ⚠ No video files in: {video_dir}")
        return

    # Build CSV rows
    rows = []
    missing_count = 0
    for vf in video_files:
        person = extract_person_from_video(vf, view, blur_name)

        # Paths (absolute)
        video_path = video_dir / vf
        degraded_img_path = blurred_img_dir / f"{person}_test_blurred_{blur_num}.jpg"
        ref_img_path = ref_dir / f"{person}_ref.jpg"
        test_img_path = test_dir / f"{person}_test.jpg"

        # Verify files exist
        missing = []
        if not degraded_img_path.exists():
            missing.append(f"degraded: {degraded_img_path}")
        if not ref_img_path.exists():
            missing.append(f"ref: {ref_img_path}")
        if not test_img_path.exists():
            missing.append(f"test: {test_img_path}")

        if missing:
            missing_count += 1
            print(f"    ⚠ {person}: missing {', '.join(missing)}")

        rows.append({
            "person": person,
            "degraded_image_path": str(degraded_img_path),
            "output_video_path": str(video_path),
            "ref_image_path": str(ref_img_path),
            "test_image_path": str(test_img_path),
        })

    # Write CSV
    csv_filename = f"{view}_{blur_name}_full.csv"
    csv_path = video_dir / csv_filename

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["person", "degraded_image_path", "output_video_path", "ref_image_path", "test_image_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✓ {csv_path.relative_to(PROJECT_ROOT)}  ({len(rows)} rows, {missing_count} missing)")


def main():
    print("=" * 70)
    print("  Generating CSV files for output_full")
    print("=" * 70)

    for view in VIEWS:
        print(f"\n── {view.upper()} ──")
        for blur_name in BLUR_LEVELS:
            generate_csv_for_folder(view, blur_name)

    print(f"\n{'=' * 70}")
    print("  Done!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

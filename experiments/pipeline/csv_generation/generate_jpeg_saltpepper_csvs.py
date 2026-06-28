"""
Generate CSV files for jpeg and salt-pepper degradation experiments.

For each combination of (view, deg_type, level) in:
    views      : frontal, side
    deg_types  : jpeg, salt-pepper
    levels     : L1, L2, L3

Produces one CSV file at:
    degradation_experiment/<view>/<deg_type>/<level>/<view>_<deg_type>_<level_lower>.csv

CSV columns:
    person      - person name (derived from filename)
    test_path   - relative path to the degraded image (relative to PROJECT_ROOT)
    ref_path    - absolute path to the reference image
    <noise_col> - noise label column (value = 1), column name = e.g. jpeg_l1 / salt-pepper_l1

Usage:
    python scripts/generate_jpeg_saltpepper_csvs.py
    python scripts/generate_jpeg_saltpepper_csvs.py --dry_run
"""

import argparse
import csv
import re
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]

VIEWS             = ["frontal", "side"]
DEGRADATION_TYPES = ["jpeg", "salt-pepper"]
LEVELS            = ["L1", "L2", "L3"]


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_person(filename: str, deg_type: str, level: str) -> str:
    """
    Extract person name from a filename like:
        Aaron_Peirsol_0003_jpeg_L1.png
        Aaron_Peirsol_0003_salt-pepper_L1.png

    Strategy: strip the trailing _<index>_<deg_type>_<level>.png
    """
    stem = Path(filename).stem  # e.g. Aaron_Peirsol_0003_jpeg_L1
    # Build a suffix pattern: _<digits>_<deg_type>_<level>
    # deg_type may contain hyphens, so escape it
    suffix_pattern = rf"_\d+_{re.escape(deg_type)}_{re.escape(level)}$"
    person = re.sub(suffix_pattern, "", stem)
    return person


def ref_path_for(person: str, view: str) -> str:
    """Return absolute path string to the reference jpg for a person."""
    if view == "frontal":
        ref_dir = PROJECT_ROOT / "Experiment_Data_Split_Combined" / "Frontal_Exp" / "Ref"
    else:
        ref_dir = PROJECT_ROOT / "Experiment_Data_Split_Combined" / "Side_Exp" / "Ref"
    return str(ref_dir / f"{person}_ref.jpg")


def build_csv_for(view: str, deg_type: str, level: str, dry_run: bool = False) -> None:
    """Scan the folder and write the CSV."""
    folder   = PROJECT_ROOT / "degradation_experiment" / view / deg_type / level
    csv_name = f"{view}_{deg_type}_{level.lower()}.csv"
    csv_path = folder / csv_name

    # noise column name mirrors the existing convention, e.g. jpeg_l1, salt-pepper_l1
    noise_col = f"{deg_type}_{level.lower()}"

    if not folder.exists():
        print(f"  [SKIP] Folder not found: {folder}")
        return

    # Collect all .png / .jpg image files (sort for determinism)
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    if not images:
        print(f"  [SKIP] No images found in {folder}")
        return

    rows = []
    for img_path in images:
        person    = extract_person(img_path.name, deg_type, level)
        test_path = str(img_path.relative_to(PROJECT_ROOT))  # relative to project root
        ref_p     = ref_path_for(person, view)
        rows.append({
            "person":    person,
            "test_path": test_path,
            "ref_path":  ref_p,
            noise_col:   1,
        })

    if dry_run:
        print(f"  [DRY RUN] Would write {len(rows)} rows → {csv_path}")
        for r in rows[:3]:
            print(f"            {r}")
        if len(rows) > 3:
            print(f"            ... ({len(rows) - 3} more rows)")
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["person", "test_path", "ref_path", noise_col])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✓ Written {len(rows):>3} rows → {csv_path.relative_to(PROJECT_ROOT)}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CSV files for jpeg / salt-pepper degradation experiments",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Preview what would be written without creating any files",
    )
    parser.add_argument(
        "--views", nargs="+", default=VIEWS,
        help=f"Views to process (default: {VIEWS})",
    )
    parser.add_argument(
        "--deg_types", nargs="+", default=DEGRADATION_TYPES,
        help=f"Degradation types to process (default: {DEGRADATION_TYPES})",
    )
    parser.add_argument(
        "--levels", nargs="+", default=LEVELS,
        help=f"Levels to process (default: {LEVELS})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Generate Degradation CSVs — jpeg / salt-pepper")
    print("=" * 60)
    print(f"  Views             : {args.views}")
    print(f"  Degradation types : {args.deg_types}")
    print(f"  Levels            : {args.levels}")
    print(f"  Dry run           : {args.dry_run}")
    print(f"  Project root      : {PROJECT_ROOT}")
    print("=" * 60)

    total = 0
    for view in args.views:
        for deg_type in args.deg_types:
            for level in args.levels:
                print(f"\n  [{view.upper()}] [{deg_type.upper()}] {level}")
                build_csv_for(view, deg_type, level, dry_run=args.dry_run)
                total += 1

    print(f"\n{'=' * 60}")
    print(f"  Done — processed {total} CSV(s).")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""
Generate CSV files for side-view degradation experiments.

Creates CSV files for:
  - degradation_experiment/side/downup/{L1,L2,L3}/  →  side_downup_{l1,l2,l3}.csv
  - degradation_experiment/side/motion/{L1,L2,L3}/  →  side_motion_{l1,l2,l3}.csv
  - degradation_experiment/side/screen/              →  side_screen.csv  (no level sub-folders)

Each CSV has columns: person, test_path, ref_path, <noise_type>
  - test_path : relative path from project root  (no leading slash)
  - ref_path  : absolute path to Side_Exp/Ref/<person>_ref.jpg

jpeg and salt-pepper CSVs already exist — skipped by default.

Usage:
    python scripts/generate_side_csvs.py
    python scripts/generate_side_csvs.py --dry_run
    python scripts/generate_side_csvs.py --deg_types downup motion screen
    python scripts/generate_side_csvs.py --overwrite   # re-generate existing CSVs
"""

import re
import csv
import argparse
from pathlib import Path

# ── project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIDE_DEG_DIR = PROJECT_ROOT / "degradation_experiment" / "side"
REF_DIR      = PROJECT_ROOT / "Experiment_Data_Split_Combined" / "Side_Exp" / "Ref"

# Degradation types that have sub-level folders (L1 / L2 / L3)
LEVELED_DEG_TYPES = ["downup", "motion"]

# Degradation types WITHOUT sub-level folders
FLAT_DEG_TYPES = ["screen"]

LEVELS = ["L1", "L2", "L3"]


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_person(filename: str, deg_tag: str) -> str:
    """
    Extract person name from a filename like:
        Abdullah_0004_downup_L1.png
        George_W_Bush_0294_motion_L2.png
        Abdullah_0004_screen.png

    Strategy: strip the suffix that starts with the four-digit image-index.
    """
    stem = Path(filename).stem          # e.g. "Abdullah_0004_downup_L1"
    # Match the pattern  _NNNN_  followed by anything
    m = re.match(r"^(.+?)_\d{4,}_", stem)
    if m:
        return m.group(1)
    # Fallback: strip known deg_tag and everything after
    idx = stem.find(f"_{deg_tag}")
    if idx != -1:
        return stem[:idx]
    return stem


def generate_leveled_csv(
    deg_type: str,
    level: str,
    dry_run: bool,
    overwrite: bool,
) -> None:
    """Generate one CSV for a leveled degradation type (downup / motion)."""
    folder   = SIDE_DEG_DIR / deg_type / level
    csv_name = f"side_{deg_type}_{level.lower()}.csv"
    csv_path = folder / csv_name

    if not folder.exists():
        print(f"  [SKIP] Folder not found: {folder}")
        return

    if csv_path.exists() and not overwrite:
        print(f"  [SKIP] Already exists: {csv_path.relative_to(PROJECT_ROOT)}")
        return

    # Collect PNG files
    png_files = sorted(folder.glob("*.png"))
    if not png_files:
        print(f"  [WARN] No PNG files in {folder}")
        return

    noise_col = f"{deg_type}_{level.lower()}"   # e.g. "downup_l1"
    rows = []

    for png in png_files:
        person    = extract_person(png.name, deg_type)
        test_path = str(png.relative_to(PROJECT_ROOT))          # relative
        ref_path  = str(REF_DIR / f"{person}_ref.jpg")          # absolute
        rows.append([person, test_path, ref_path, 1])

    print(f"  {'[DRY-RUN] ' if dry_run else ''}Writing {len(rows)} rows → {csv_path.relative_to(PROJECT_ROOT)}")

    if not dry_run:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["person", "test_path", "ref_path", noise_col])
            writer.writerows(rows)


def generate_flat_csv(
    deg_type: str,
    dry_run: bool,
    overwrite: bool,
) -> None:
    """Generate one CSV for a flat (no-level) degradation type (screen)."""
    folder   = SIDE_DEG_DIR / deg_type
    csv_name = f"side_{deg_type}.csv"
    csv_path = folder / csv_name

    if not folder.exists():
        print(f"  [SKIP] Folder not found: {folder}")
        return

    if csv_path.exists() and not overwrite:
        print(f"  [SKIP] Already exists: {csv_path.relative_to(PROJECT_ROOT)}")
        return

    png_files = sorted(folder.glob("*.png"))
    if not png_files:
        print(f"  [WARN] No PNG files in {folder}")
        return

    noise_col = deg_type   # e.g. "screen"
    rows = []

    for png in png_files:
        person    = extract_person(png.name, deg_type)
        test_path = str(png.relative_to(PROJECT_ROOT))
        ref_path  = str(REF_DIR / f"{person}_ref.jpg")
        rows.append([person, test_path, ref_path, 1])

    print(f"  {'[DRY-RUN] ' if dry_run else ''}Writing {len(rows)} rows → {csv_path.relative_to(PROJECT_ROOT)}")

    if not dry_run:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["person", "test_path", "ref_path", noise_col])
            writer.writerows(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    all_types = LEVELED_DEG_TYPES + FLAT_DEG_TYPES

    parser = argparse.ArgumentParser(
        description="Generate CSV files for side-view degradation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--deg_types", nargs="+", default=None,
        help=f"Degradation types to process (default: {all_types})",
    )
    parser.add_argument(
        "--levels", nargs="+", default=LEVELS,
        help=f"Levels for leveled types (default: {LEVELS})",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be done without writing files",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate CSVs that already exist",
    )
    args = parser.parse_args()

    deg_types = args.deg_types if args.deg_types else all_types

    print("=" * 60)
    print("  📋 Side-View CSV Generator")
    print("=" * 60)
    print(f"  Deg types : {', '.join(deg_types)}")
    print(f"  Levels    : {', '.join(args.levels)}")
    print(f"  Dry run   : {args.dry_run}")
    print(f"  Overwrite : {args.overwrite}")
    print("=" * 60)

    for dt in deg_types:
        if dt in LEVELED_DEG_TYPES:
            for level in args.levels:
                generate_leveled_csv(dt, level, args.dry_run, args.overwrite)
        elif dt in FLAT_DEG_TYPES:
            generate_flat_csv(dt, args.dry_run, args.overwrite)
        else:
            print(f"  [WARN] Unknown deg_type '{dt}' — skipped")

    print("\n  ✅ Done.")


if __name__ == "__main__":
    main()

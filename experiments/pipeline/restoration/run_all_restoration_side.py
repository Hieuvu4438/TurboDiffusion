"""
Batch Face Restoration: Run inference on ALL side-view degraded CSV files.

Covers degradation types: downup, motion, screen, salt-pepper
(jpeg is intentionally excluded)

CSV file locations:
  - degradation_experiment/side/downup/{L1,L2,L3}/side_downup_{l1,l2,l3}.csv
  - degradation_experiment/side/motion/{L1,L2,L3}/side_motion_{l1,l2,l3}.csv
  - degradation_experiment/side/salt-pepper/{L1,L2,L3}/side_salt-pepper_{l1,l2,l3}.csv
  - degradation_experiment/side/screen/side_screen.csv

Outputs go to:
  output_full_new_combined/side/<deg_type>/<level>/   (leveled)
  output_full_new_combined/side/<deg_type>/           (flat, e.g. screen)

Usage:
    python scripts/run_all_restoration_side.py
    python scripts/run_all_restoration_side.py --dry_run
    python scripts/run_all_restoration_side.py --skip_existing
    python scripts/run_all_restoration_side.py --deg_types downup motion
    python scripts/run_all_restoration_side.py --levels L1 L2
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIDE_DIR     = PROJECT_ROOT / "degradation_experiment" / "side"

# Degradation types that have L1/L2/L3 sub-folders
LEVELED_DEG_TYPES = ["downup", "motion", "salt-pepper"]

# Degradation types with a flat folder (no level)
FLAT_DEG_TYPES = ["screen"]

LEVELS = ["L1", "L2", "L3"]

PROMPT_TEXT = (
    "Hyper-realistic forensic-level restoration of a human face, 8k resolution, extremely precise facial geometry. "
    "Strict structural coherence, exact mapping of original facial proportions. Photorealistic raw texture. "
    "The video dictates a deliberate motion ending in a static, straight-on frontal view. The final frames strictly "
    "lock into a direct frontal portrait, resolving all facial geometries symmetrically without altering the "
    "original identity, neutral unedited appearance."
)


# ── helpers ──────────────────────────────────────────────────────────────────

def find_all_csv_files(deg_types: list, levels: list) -> list:
    """
    Collect all CSV entries for the given deg_types.
    Returns a list of dicts with keys: csv_path, deg_type, level (None for flat).
    """
    csv_files = []

    for deg_type in deg_types:
        if deg_type in LEVELED_DEG_TYPES:
            for level in levels:
                folder   = SIDE_DIR / deg_type / level
                csv_name = f"side_{deg_type}_{level.lower()}.csv"
                csv_path = folder / csv_name
                if csv_path.exists():
                    csv_files.append({
                        "csv_path": csv_path,
                        "deg_type": deg_type,
                        "level":    level,
                        "is_flat":  False,
                    })
                else:
                    print(f"  [WARN] CSV not found: {csv_path}")
        elif deg_type in FLAT_DEG_TYPES:
            folder   = SIDE_DIR / deg_type
            csv_name = f"side_{deg_type}.csv"
            csv_path = folder / csv_name
            if csv_path.exists():
                csv_files.append({
                    "csv_path": csv_path,
                    "deg_type": deg_type,
                    "level":    None,
                    "is_flat":  True,
                })
            else:
                print(f"  [WARN] CSV not found: {csv_path}")
        else:
            print(f"  [WARN] Unknown deg_type '{deg_type}' — skipped")

    return csv_files


def build_output_dir(csv_info: dict) -> Path:
    """Build output directory path under output_full_new_combined/side/."""
    if csv_info["is_flat"]:
        return PROJECT_ROOT / "output_full_new_combined" / "side" / csv_info["deg_type"]
    else:
        return (
            PROJECT_ROOT
            / "output_full_new_combined"
            / "side"
            / csv_info["deg_type"]
            / csv_info["level"]
        )


def run_restoration_for_csv(csv_info: dict, args: argparse.Namespace) -> dict:
    """Call run_face_restoration.py for a single CSV file."""
    csv_path   = csv_info["csv_path"]
    deg_type   = csv_info["deg_type"]
    level      = csv_info["level"] or "—"
    output_dir = build_output_dir(csv_info)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "pipeline" / "restoration" / "run_face_restoration.py"),
        "--csv_path",   str(csv_path),
        "--output_dir", str(output_dir),
        "--prompt",     PROMPT_TEXT,
    ]

    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.dry_run:
        cmd.append("--dry_run")
    if args.num_steps:
        cmd.extend(["--num_steps", str(args.num_steps)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    start = time.time()
    env   = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "turbodiffusion")

    label = f"{deg_type}/{level}"
    try:
        result  = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
        elapsed = time.time() - start
        status  = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
        return {"csv": label, "status": status,
                "returncode": result.returncode, "elapsed": elapsed}
    except Exception as exc:
        elapsed = time.time() - start
        return {"csv": label, "status": f"ERROR: {exc}",
                "returncode": -1, "elapsed": elapsed}


def format_duration(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s   = divmod(rem, 60)
    if td.days > 0:
        return f"{td.days}d {h}h {m}m"
    elif h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    all_deg_types = LEVELED_DEG_TYPES + FLAT_DEG_TYPES

    parser = argparse.ArgumentParser(
        description="Run face restoration on ALL side-view degraded CSV files (no jpeg)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry_run",       action="store_true",
                        help="Preview commands without executing")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip videos that already exist (resume mode)")
    parser.add_argument("--deg_types",    type=str, nargs="+", default=None,
                        help=f"Specific deg types (default: all {all_deg_types})")
    parser.add_argument("--levels",       type=str, nargs="+", default=None,
                        help=f"Specific levels for leveled types (default: {LEVELS})")
    parser.add_argument("--num_steps",    type=int, default=4, choices=[1, 2, 3, 4],
                        help="Inference steps (default: 4)")
    parser.add_argument("--seed",         type=int, default=0,
                        help="Random seed (default: 0)")
    args = parser.parse_args()

    deg_types = args.deg_types if args.deg_types else all_deg_types
    levels    = args.levels    if args.levels    else LEVELS

    csv_files = find_all_csv_files(deg_types, levels)
    if not csv_files:
        print("ERROR: No CSV files found! Run generate_side_csvs.py first.")
        sys.exit(1)

    # ── header ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  🚀 Face Restoration (SIDE) — Full Batch Run")
    print("=" * 70)
    print(f"  Degradations : {', '.join(deg_types)}")
    print(f"  Levels       : {', '.join(levels)}")
    print(f"  CSV files    : {len(csv_files)}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Dry run      : {args.dry_run}")
    print(f"  Started at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\n  Queued jobs:")
    for i, cf in enumerate(csv_files, 1):
        lvl = cf["level"] or "(flat)"
        print(f"    {i:2d}. {cf['deg_type']}/{lvl}  →  {cf['csv_path'].name}")
    print()

    # ── run ──────────────────────────────────────────────────────────────────
    results     = []
    total_start = time.time()

    for i, csv_info in enumerate(csv_files, 1):
        deg_type = csv_info["deg_type"]
        level    = csv_info["level"] or "(flat)"

        print(f"\n{'╔' + '═'*68 + '╗'}")
        print(f"  [{i}/{len(csv_files)}] {deg_type.upper()} — level={level}")
        print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
        remaining = len(csv_files) - i
        if results:
            avg_time = sum(r["elapsed"] for r in results) / len(results)
            eta = format_duration(avg_time * remaining)
            print(f"  ETA for remaining {remaining} jobs: ~{eta}")
        print(f"{'╚' + '═'*68 + '╝'}")

        result = run_restoration_for_csv(csv_info, args)
        results.append(result)
        print(f"\n  → {result['status']} ({format_duration(result['elapsed'])})")

    # ── summary ──────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    ok_count      = sum(1 for r in results if r["returncode"] == 0)
    fail_count    = sum(1 for r in results if r["returncode"] != 0)

    print(f"\n{'='*70}")
    print(f"  ✅ ALL JOBS COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time   : {format_duration(total_elapsed)}")
    print(f"  Succeeded    : {ok_count}/{len(results)}")
    if fail_count:
        print(f"  Failed       : {fail_count}/{len(results)}")
    print(f"  Finished at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("  Results:")
    print(f"  {'Job':<30} {'Status':<20} {'Time':<12}")
    print(f"  {'─'*30} {'─'*20} {'─'*12}")
    for r in results:
        print(f"  {r['csv']:<30} {r['status']:<20} {format_duration(r['elapsed']):<12}")

    print(f"\n  Output locations:")
    for cf in csv_files:
        out_dir = build_output_dir(cf)
        print(f"    {out_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

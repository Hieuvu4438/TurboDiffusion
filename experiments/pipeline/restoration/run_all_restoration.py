"""
Batch Face Restoration: Run inference on ALL degraded CSV files automatically.

This script finds all CSV files under degradation_experiment/frontal/{deg_type}/{level}/
and runs run_face_restoration.py on each one sequentially.

Perfect for overnight / unattended runs.

Usage:
    python scripts/run_all_restoration.py
    python scripts/run_all_restoration.py --dry_run          # preview only
    python scripts/run_all_restoration.py --skip_existing    # resume after crash
    python scripts/run_all_restoration.py --deg_types motion # only specific degradation
    python scripts/run_all_restoration.py --levels L1 L2     # only specific levels
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BLURRED_OUTPUT = PROJECT_ROOT / "degradation_experiment" / "frontal"

ALL_SIGMAS = [3, 5, 8, 10, 12, 15]
# ALL_SIGMAS = [10, 12, 15]
EXPERIMENT_TYPES = ["side", "frontal"]    
DEGRADATION_TYPES = ["downup", "motion"]
LEVELS = ["L1", "L2", "L3"]

def find_all_csv_files(deg_types: list, levels: list) -> list:
    """Find all degraded CSV files, ordered by experiment type then sigma."""
    csv_files = []
    for deg_type in deg_types:
        for level in levels:
            folder = BLURRED_OUTPUT / deg_type / level
            csv_name = f"frontal_{deg_type}_{level.lower()}.csv"
            csv_path = folder / csv_name
            if csv_path.exists():
                csv_files.append({
                    "csv_path": csv_path,
                    "deg_type": deg_type,
                    "level": level,
                })
            else:
                print(f"CSV file not found: {csv_path}")
    return csv_files



def run_restoration_for_csv(csv_info: dict, args: argparse.Namespace) -> dict:
    """Run run_face_restoration.py for a single CSV file."""
    csv_path = csv_info["csv_path"]
    deg_type = csv_info["deg_type"]
    level = csv_info["level"]

    # Build output directory
    output_dir = PROJECT_ROOT / "output_full_new_combined" / deg_type / level

    # Determine prompt based on sigma
    # if sigma in [3, 5]:
    #     prompt_text = (
    #         "Ultra-high definition macro photography of a human face, 8k resolution, razor-sharp focus. "
    #         "Enhance micro-details, crystal clear optical sharpness. Authentic natural skin texture. "
    #         "The subject smoothly turns their head towards the lens, ending the video with a perfectly straight, "
    #         "direct frontal view portrait. Exact identity preservation maintained throughout the temporal motion, "
    #         "maintaining eye contact in the final frames."
    #     )
    # elif sigma in [8, 10]:
    #     prompt_text = (
    #         "Cinematic ultra-high definition portrait of a human, 8k resolution, raw photography. "
    #         "Exact structural reconstruction, authentic skin texture with natural real-world anomalies. "
    #         "Professional studio lighting. Dynamic temporal consistency: the camera naturally transitions "
    #         "to lock onto a strict frontal view. In the final frames, the face is perfectly aligned symmetrically "
    #         "towards the viewer, ensuring highly detailed, unmodified facial structure from the front."
    #     )
    # else:  # 12, 15
    #     prompt_text = (
    #         "Hyper-realistic forensic-level restoration of a human face, 8k resolution, extremely precise facial geometry. "
    #         "Strict structural coherence, exact mapping of original facial proportions. Photorealistic raw texture. "
    #         "The video dictates a deliberate motion ending in a static, straight-on frontal view. The final frames strictly "
    #         "lock into a direct frontal portrait, resolving all facial geometries symmetrically without altering the "
    #         "original identity, neutral unedited appearance."
    #     )
    prompt_text = (
        "Hyper-realistic forensic-level restoration of a human face, 8k resolution, extremely precise facial geometry. "
        "Strict structural coherence, exact mapping of original facial proportions. Photorealistic raw texture. "
        "The video dictates a deliberate motion ending in a static, straight-on frontal view. The final frames strictly "
        "lock into a direct frontal portrait, resolving all facial geometries symmetrically without altering the "
        "original identity, neutral unedited appearance."
    )

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "pipeline" / "restoration" / "run_face_restoration.py"),
        "--csv_path", str(csv_path),
        "--output_dir", str(output_dir),
        "--prompt", prompt_text,
    ]

    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.dry_run:
        cmd.append("--dry_run")
    if args.num_steps:
        cmd.extend(["--num_steps", str(args.num_steps)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    start_time = time.time()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "turbodiffusion")

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
        elapsed = time.time() - start_time
        return {
            "csv": f"{deg_type}/{level}",
            "status": "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})",
            "returncode": result.returncode,
            "elapsed": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "csv": f"{deg_type}/{level}",
            "status": f"ERROR: {e}",
            "returncode": -1,
            "elapsed": elapsed,
        }


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if td.days > 0:
        return f"{td.days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def main():
    parser = argparse.ArgumentParser(
        description="Run face restoration inference on ALL degraded CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry_run",        action="store_true", help="Preview commands without executing")
    parser.add_argument("--skip_existing",  action="store_true", help="Skip videos that already exist (resume mode)")
    parser.add_argument("--deg_types",      type=str, nargs="+", default=None, help=f"Specific degradation types (default: all {DEGRADATION_TYPES})")
    parser.add_argument("--levels",         type=str, nargs="+", default=None, help=f"Specific levels (default: all {LEVELS})")
    parser.add_argument("--num_steps",      type=int, default=4, choices=[1, 2, 3, 4], help="Inference steps (default: 4)")
    parser.add_argument("--seed",           type=int, default=0, help="Random seed (default: 0)")

    args = parser.parse_args()

    deg_types = args.deg_types if args.deg_types else DEGRADATION_TYPES
    levels = args.levels if args.levels else LEVELS

    # Find all CSV files
    csv_files = find_all_csv_files(deg_types, levels)

    if not csv_files:
        print("ERROR: No CSV files found!")
        sys.exit(1)

    # Print plan
    total_images = len(csv_files) * 100  # each CSV has 100 images
    print("=" * 70)
    print("  🚀 Face Restoration — Full Batch Run")
    print("=" * 70)
    print(f"  Degradations : {', '.join(deg_types)}")
    print(f"  Levels       : {', '.join(levels)}")
    print(f"  CSV files    : {len(csv_files)}")
    print(f"  Total images : ~{total_images}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Dry run      : {args.dry_run}")
    print(f"  Started at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\n  Queued jobs:")
    for i, cf in enumerate(csv_files, 1):
        print(f"    {i:2d}. {cf['deg_type']}/{cf['level']}  →  {cf['csv_path'].name}")
    print()

    # Run all
    results = []
    total_start = time.time()

    for i, csv_info in enumerate(csv_files, 1):
        deg_type = csv_info["deg_type"]
        level = csv_info["level"]

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

    # Final summary
    total_elapsed = time.time() - total_start
    ok_count = sum(1 for r in results if r["returncode"] == 0)
    fail_count = sum(1 for r in results if r["returncode"] != 0)

    print(f"\n{'='*70}")
    print(f"  ✅ ALL JOBS COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time   : {format_duration(total_elapsed)}")
    print(f"  Succeeded    : {ok_count}/{len(results)}")
    if fail_count > 0:
        print(f"  Failed       : {fail_count}/{len(results)}")
    print(f"  Finished at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("  Results:")
    print(f"  {'Job':<25} {'Status':<15} {'Time':<12}")
    print(f"  {'─'*25} {'─'*15} {'─'*12}")
    for r in results:
        print(f"  {r['csv']:<25} {r['status']:<15} {format_duration(r['elapsed']):<12}")

    # Output directories
    print(f"\n  Output locations:")
    for deg_type in deg_types:
        for level in levels:
            out_dir = PROJECT_ROOT / "output_full_new_combined" / deg_type / level
            print(f"    {out_dir}/")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()

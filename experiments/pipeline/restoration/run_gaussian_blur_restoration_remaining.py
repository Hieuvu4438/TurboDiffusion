"""
Generate missing Gaussian-blurred frontal inputs and run TurboDiffusion restoration.

This script uses lfw_frontal_test.csv as the canonical 250-person list, updates
blurred10/12/15 CSVs under degradation_experiment/blurred_output_new, then calls
experiments/pipeline/restoration/run_face_restoration.py to write videos under output_full_new/frontal.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_CSV = PROJECT_ROOT / "lfw_side_test.csv"
BLUR_ROOT = PROJECT_ROOT / "degradation_experiment" / "blurred_output_new" / "side"
OUTPUT_ROOT = PROJECT_ROOT / "output_full_new" / "side"
DEFAULT_SIGMAS = [10, 12, 15]

PROMPT_TEXT = (
    "Hyper-realistic forensic-level restoration of a human face, 8k resolution, extremely precise facial geometry. "
    "Strict structural coherence, exact mapping of original facial proportions. Photorealistic raw texture. "
    "The video dictates a deliberate motion ending in a static, straight-on frontal view. The final frames strictly "
    "lock into a direct frontal portrait, resolving all facial geometries symmetrically without altering the "
    "original identity, neutral unedited appearance."
)


REQUIRED_COLUMNS = {"person", "test_path", "ref_path"}


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def read_source_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Source CSV is missing columns: {', '.join(sorted(missing))}")

        rows = []
        seen = set()
        for row in reader:
            person = row["person"].strip()
            if not person:
                continue
            if person in seen:
                raise ValueError(f"Duplicate person in source CSV: {person}")
            seen.add(person)
            rows.append({
                "person": person,
                "test_path": row["test_path"].strip(),
                "ref_path": row["ref_path"].strip(),
            })

    if not rows:
        raise ValueError(f"Source CSV has no usable rows: {csv_path}")
    return rows


def blur_dir(sigma: int) -> Path:
    return BLUR_ROOT / f"blurred{sigma}"


def blur_image_path(person: str, sigma: int) -> Path:
    return blur_dir(sigma) / f"{person}_test_blurred_{sigma}.jpg"


def blur_csv_path(sigma: int) -> Path:
    return blur_dir(sigma) / f"frontal_blurred{sigma}.csv"


def output_dir(sigma: int) -> Path:
    return OUTPUT_ROOT / f"blurred{sigma}"


def output_video_path(person: str, sigma: int) -> Path:
    return output_dir(sigma) / f"{person}_frontal_blurred{sigma}.mp4"


def output_manifest_path(sigma: int) -> Path:
    return output_dir(sigma) / f"frontal_blurred{sigma}_full.csv"


def rel_to_project(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def apply_gaussian_blur(image, sigma: int):
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def count_existing(rows: list[dict[str, str]], sigma: int) -> dict[str, int]:
    blur_count = sum(1 for row in rows if blur_image_path(row["person"], sigma).exists())
    output_count = sum(1 for row in rows if output_video_path(row["person"], sigma).exists())
    return {
        "blur_existing": blur_count,
        "blur_missing": len(rows) - blur_count,
        "output_existing": output_count,
        "output_missing": len(rows) - output_count,
    }


def print_plan(rows: list[dict[str, str]], sigmas: list[int], args: argparse.Namespace) -> None:
    print("=" * 70)
    print("Gaussian blurred frontal restoration")
    print("=" * 70)
    print(f"Source CSV     : {args.source_csv}")
    print(f"Rows           : {len(rows)}")
    print(f"Sigmas         : {', '.join(str(s) for s in sigmas)}")
    print(f"Blur root      : {BLUR_ROOT}")
    print(f"Output root    : {OUTPUT_ROOT}")
    print(f"Skip existing  : {args.skip_existing}")
    print(f"Prepare only   : {args.prepare_only}")
    print(f"Dry run        : {args.dry_run}")
    print("=" * 70)

    for sigma in sigmas:
        counts = count_existing(rows, sigma)
        print(
            f"sigma={sigma}: "
            f"blur {counts['blur_existing']}/{len(rows)} existing "
            f"({counts['blur_missing']} missing), "
            f"output {counts['output_existing']}/{len(rows)} existing "
            f"({counts['output_missing']} missing)"
        )
    print()


def generate_blur_images(rows: list[dict[str, str]], sigma: int, args: argparse.Namespace) -> dict[str, int]:
    created = 0
    skipped = 0
    failed = 0

    if args.dry_run:
        missing = sum(
            1
            for row in rows
            if args.overwrite_blur or not blur_image_path(row["person"], sigma).exists()
        )
        print(f"[dry-run] sigma={sigma}: would create/overwrite {missing} blurred images")
        return {"created": 0, "skipped": len(rows) - missing, "failed": 0}

    folder = blur_dir(sigma)
    folder.mkdir(parents=True, exist_ok=True)

    for row in rows:
        person = row["person"]
        out_path = blur_image_path(person, sigma)
        if out_path.exists() and not args.overwrite_blur:
            skipped += 1
            continue

        image_path = resolve_project_path(row["test_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  [WARN] Cannot read image: {image_path}")
            failed += 1
            continue

        blurred = apply_gaussian_blur(image, sigma)
        if cv2.imwrite(str(out_path), blurred):
            created += 1
        else:
            print(f"  [WARN] Cannot write blurred image: {out_path}")
            failed += 1

    return {"created": created, "skipped": skipped, "failed": failed}


def write_blur_csv(rows: list[dict[str, str]], sigma: int, dry_run: bool) -> None:
    csv_path = blur_csv_path(sigma)
    fieldnames = ["person", "test_path", "ref_path", f"blurred_{sigma}"]

    if dry_run:
        print(f"[dry-run] sigma={sigma}: would write {csv_path} with {len(rows)} rows")
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "person": row["person"],
                "test_path": rel_to_project(blur_image_path(row["person"], sigma)),
                "ref_path": row["ref_path"],
                f"blurred_{sigma}": sigma,
            })

    print(f"  CSV updated: {csv_path} ({len(rows)} rows)")


def build_restoration_command(sigma: int, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "pipeline" / "restoration" / "run_face_restoration.py"),
        "--csv_path", str(blur_csv_path(sigma)),
        "--output_dir", str(output_dir(sigma)),
        "--prompt", PROMPT_TEXT,
        "--num_steps", str(args.num_steps),
        "--seed", str(args.seed),
    ]
    if args.skip_existing:
        cmd.append("--skip_existing")
    return cmd


def run_restoration(sigma: int, args: argparse.Namespace) -> int:
    cmd = build_restoration_command(sigma, args)
    if args.dry_run:
        print(f"[dry-run] sigma={sigma}: would run:")
        print("  " + " ".join(cmd))
        return 0

    output_dir(sigma).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "turbodiffusion")

    print(f"  Running restoration: sigma={sigma}")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    return result.returncode


def parse_output_person(video_path: Path, sigma: int) -> str | None:
    suffix = f"_frontal_blurred{sigma}.mp4"
    name = video_path.name
    if not name.endswith(suffix):
        return None
    return name[:-len(suffix)]


def write_output_manifest(rows: list[dict[str, str]], sigma: int, dry_run: bool) -> int:
    manifest_path = output_manifest_path(sigma)
    existing_outputs = {
        row["person"]: output_video_path(row["person"], sigma)
        for row in rows
        if output_video_path(row["person"], sigma).exists()
    }

    folder = output_dir(sigma)
    extra_outputs = []
    if folder.exists():
        source_people = {row["person"] for row in rows}
        for video_path in sorted(folder.glob("*.mp4")):
            person = parse_output_person(video_path, sigma)
            if person and person not in source_people:
                extra_outputs.append(video_path.name)

    if dry_run:
        print(
            f"[dry-run] sigma={sigma}: would write {manifest_path} "
            f"with {len(existing_outputs)} existing output rows"
        )
        if extra_outputs:
            print(f"[dry-run] sigma={sigma}: would ignore {len(extra_outputs)} output files not in source CSV")
        return len(existing_outputs)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    row_by_person = {row["person"]: row for row in rows}
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "person",
            "degraded_image_path",
            "output_video_path",
            "ref_image_path",
            "test_image_path",
        ])
        for row in rows:
            person = row["person"]
            video_path = existing_outputs.get(person)
            if not video_path:
                continue
            source_row = row_by_person[person]
            writer.writerow([
                person,
                str(blur_image_path(person, sigma)),
                str(video_path),
                str(resolve_project_path(source_row["ref_path"])),
                str(resolve_project_path(source_row["test_path"])),
            ])

    print(f"  Output manifest updated: {manifest_path} ({len(existing_outputs)} rows)")
    if extra_outputs:
        print(f"  [WARN] Ignored {len(extra_outputs)} output files not listed in source CSV")
    return len(existing_outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate missing frontal Gaussian blur inputs and run restoration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_gaussian_blur_restoration_remaining.py --dry_run --skip_existing
  python scripts/run_gaussian_blur_restoration_remaining.py --prepare_only --skip_existing
  python scripts/run_gaussian_blur_restoration_remaining.py --skip_existing
        """,
    )
    parser.add_argument("--source_csv", type=Path, default=SOURCE_CSV, help=f"Source 250-row CSV (default: {SOURCE_CSV})")
    parser.add_argument("--sigmas", type=int, nargs="+", default=DEFAULT_SIGMAS, help="Gaussian sigmas to run (default: 10 12 15)")
    parser.add_argument("--num_steps", type=int, default=4, choices=[1, 2, 3, 4], help="Inference steps (default: 4)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--skip_existing", "--skip_permissions", dest="skip_existing", action="store_true", help="Skip output videos that already exist")
    parser.add_argument("--overwrite_blur", action="store_true", help="Regenerate blurred JPGs even if they already exist")
    parser.add_argument("--prepare_only", action="store_true", help="Create/update blurred JPGs and CSVs without running inference")
    parser.add_argument("--dry_run", action="store_true", help="Print planned work without creating files or running inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sigmas = list(dict.fromkeys(args.sigmas))
    rows = read_source_rows(args.source_csv)

    print_plan(rows, sigmas, args)

    if not args.skip_existing and not args.prepare_only and not args.dry_run:
        print("[WARN] --skip_existing was not set; existing videos may be regenerated by run_face_restoration.py.")

    failed = False
    total_start = time.time()

    for sigma in sigmas:
        print("-" * 70)
        print(f"sigma={sigma}")

        blur_stats = generate_blur_images(rows, sigma, args)
        print(
            f"  Blur images: created={blur_stats['created']}, "
            f"skipped={blur_stats['skipped']}, failed={blur_stats['failed']}"
        )
        if blur_stats["failed"]:
            failed = True

        write_blur_csv(rows, sigma, args.dry_run)

        returncode = 0
        if args.prepare_only:
            print("  Restoration skipped (--prepare_only)")
        else:
            returncode = run_restoration(sigma, args)
            if returncode != 0:
                print(f"  [WARN] Restoration failed for sigma={sigma} with code {returncode}")
                failed = True

        write_output_manifest(rows, sigma, args.dry_run)

    elapsed = format_duration(time.time() - total_start)
    print("=" * 70)
    print(f"Done in {elapsed}")
    print("Status: FAILED" if failed else "Status: OK")
    print("=" * 70)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

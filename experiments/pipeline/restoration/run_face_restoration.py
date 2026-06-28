"""
Face Restoration Inference using TurboDiffusion Wan2.2 I2V.

Supports two modes:
  1. CSV mode  : Batch inference on all degraded images listed in a CSV file.
  2. Image mode: Single-image inference on an arbitrary image path.

Usage examples:
  # ── CSV batch mode ──────────────────────────────────────────────
  python experiments/pipeline/restoration/run_face_restoration.py \
      --csv_path degradation_experiment/blurred_output/side/blurred3/side_blurred3.csv

  # ── Single image mode ──────────────────────────────────────────
  python experiments/pipeline/restoration/run_face_restoration.py \
      --image_path path/to/degraded_image.jpg \
      --output_dir output/single_test

  # ── Dry-run (print commands without executing) ─────────────────
  python experiments/pipeline/restoration/run_face_restoration.py \
      --csv_path ... --dry_run
"""

import os
import sys
import csv
import argparse
import subprocess
from pathlib import Path

# Engine now lives under experiments/pipeline/restoration/; put the repo root
# on sys.path so the experiments package can be imported.
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.config import PROJECT_ROOT, LOW_MODEL, HIGH_MODEL, DEFAULT_PROMPT

# ─── Default paths (from experiments.config) ────────────────────────────────
DEFAULT_LOW_MODEL  = str(LOW_MODEL)
DEFAULT_HIGH_MODEL = str(HIGH_MODEL)


def build_i2v_command(
    image_path: str,
    save_path: str,
    prompt: str,
    low_model: str,
    high_model: str,
    resolution: str = "720p",
    num_steps: int = 4,
    num_frames: int = 81,
    num_samples: int = 1,
    seed: int = 0,
    attention_type: str = "sagesla",
    sla_topk: float = 0.1,
    use_ode: bool = True,
    use_quant_linear: bool = True,
    adaptive_resolution: bool = True,
) -> list:
    """Build the command list for wan2.2_i2v_infer.py."""
    cmd = [
        sys.executable,
        "turbodiffusion/inference/wan2.2_i2v_infer.py",
        "--model",          "Wan2.2-A14B",
        "--low_noise_model_path",  str(low_model),
        "--high_noise_model_path", str(high_model),
        "--resolution",     resolution,
        "--image_path",     str(image_path),
        "--prompt",         prompt,
        "--num_samples",    str(num_samples),
        "--num_steps",      str(num_steps),
        "--num_frames",     str(num_frames),
        "--seed",           str(seed),
        "--attention_type", attention_type,
        "--sla_topk",       str(sla_topk),
        "--save_path",      str(save_path),
    ]
    if adaptive_resolution:
        cmd.append("--adaptive_resolution")
    if use_ode:
        cmd.append("--ode")
    if use_quant_linear:
        cmd.append("--quant_linear")
    return cmd


def run_single_inference(
    image_path: str,
    save_path: str,
    args: argparse.Namespace,
    label: str = "",
    dry_run: bool = False,
) -> bool:
    """Run inference on a single image. Returns True on success."""
    cmd = build_i2v_command(
        image_path=image_path,
        save_path=save_path,
        prompt=args.prompt,
        low_model=args.low_model,
        high_model=args.high_model,
        resolution=args.resolution,
        num_steps=args.num_steps,
        num_frames=args.num_frames,
        num_samples=args.num_samples,
        seed=args.seed,
        attention_type=args.attention_type,
        sla_topk=args.sla_topk,
        use_ode=args.ode,
        use_quant_linear=args.quant_linear,
        adaptive_resolution=args.adaptive_resolution,
    )

    if label:
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")
    print(f"  image_path : {image_path}")
    print(f"  save_path  : {save_path}")
    print(f"  command    : {' '.join(cmd)}")

    if dry_run:
        print("  [DRY-RUN] Skipped.")
        return True

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "turbodiffusion")

    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)
        print(f"  ✓ Done → {save_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ FAILED (return code {e.returncode})")
        return False


def infer_from_csv(csv_path: str, output_dir: str, args: argparse.Namespace):
    """Batch inference over all rows in a degraded-image CSV."""
    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    # Read CSV
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("person", "").strip():
                rows.append(row)

    if not rows:
        print(f"ERROR: CSV is empty: {csv_path}")
        sys.exit(1)

    # Determine output directory
    if output_dir is None:
        # Auto-determine: output next to the csv's folder
        # e.g. .../blurred3/side_blurred3.csv → output/restoration/side/blurred3/
        csv_parent = csv_path.parent
        rel = csv_parent.relative_to(PROJECT_ROOT)
        output_dir = PROJECT_ROOT / "output" / "restoration" / rel
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(rows)
    # Apply skip/limit
    start_idx = args.skip
    end_idx = start_idx + args.limit if args.limit else total
    rows_to_process = rows[start_idx:end_idx]

    print(f"{'='*60}")
    print(f"  Face Restoration — CSV Batch Inference")
    print(f"{'='*60}")
    print(f"  CSV         : {csv_path}")
    print(f"  Total rows  : {total}")
    print(f"  Processing  : rows {start_idx+1}..{min(end_idx, total)} ({len(rows_to_process)} images)")
    print(f"  Output dir  : {output_dir}")
    print(f"{'='*60}")

    success = 0
    fail = 0
    for idx, row in enumerate(rows_to_process):
        global_idx = start_idx + idx
        person = row["person"]
        test_path = row["test_path"]

        # Make absolute if relative
        if not os.path.isabs(test_path):
            test_path = str(PROJECT_ROOT / test_path)

        # Output video filename
        # Extract sigma info from csv filename if present
        csv_stem = csv_path.stem  # e.g. "side_blurred3"
        video_name = f"{person}_{csv_stem}.mp4"
        save_path = output_dir / video_name

        # Skip if already exists and --skip_existing is set
        if args.skip_existing and save_path.exists():
            print(f"\n  [{global_idx+1}/{total}] {person} — SKIPPED (already exists)")
            success += 1
            continue

        ok = run_single_inference(
            image_path=test_path,
            save_path=str(save_path),
            args=args,
            label=f"[{global_idx+1}/{total}] {person}",
            dry_run=args.dry_run,
        )
        if ok:
            success += 1
        else:
            fail += 1

    print(f"\n{'='*60}")
    print(f"  Batch complete: {success} succeeded, {fail} failed out of {len(rows_to_process)}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


def infer_from_image(image_path: str, output_dir: str, args: argparse.Namespace):
    """Single-image inference."""
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "restoration" / "single"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f"{image_path.stem}_restored.mp4"

    print(f"{'='*60}")
    print(f"  Face Restoration — Single Image Inference")
    print(f"{'='*60}")
    print(f"  Image  : {image_path}")
    print(f"  Output : {save_path}")
    print(f"{'='*60}")

    run_single_inference(
        image_path=str(image_path),
        save_path=str(save_path),
        args=args,
        dry_run=args.dry_run,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Face Restoration inference using TurboDiffusion Wan2.2 I2V",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch inference from CSV
  python experiments/pipeline/restoration/run_face_restoration.py \\
      --csv_path degradation_experiment/blurred_output/side/blurred3/side_blurred3.csv

  # Single image inference
  python experiments/pipeline/restoration/run_face_restoration.py \\
      --image_path path/to/image.jpg

  # Dry-run (print commands only)
  python experiments/pipeline/restoration/run_face_restoration.py --csv_path ... --dry_run
        """,
    )

    # ── Input mode (mutually exclusive) ──────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--csv_path",
        type=str,
        help="Path to a degraded-image CSV file for batch inference",
    )
    input_group.add_argument(
        "--image_path",
        type=str,
        help="Path to a single image for inference",
    )

    # ── Output ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output videos (default: auto-determined)",
    )

    # ── Model ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--low_model",
        type=str,
        default=DEFAULT_LOW_MODEL,
        help=f"Path to low-noise model (default: {DEFAULT_LOW_MODEL})",
    )
    parser.add_argument(
        "--high_model",
        type=str,
        default=DEFAULT_HIGH_MODEL,
        help=f"Path to high-noise model (default: {DEFAULT_HIGH_MODEL})",
    )

    # ── Prompt ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text prompt for video generation",
    )

    # ── Generation parameters ────────────────────────────────────────────
    parser.add_argument("--resolution",     type=str,   default="720p", help="Resolution (default: 720p)")
    parser.add_argument("--num_steps",      type=int,   default=4,      choices=[1, 2, 3, 4], help="Number of inference steps (default: 4)")
    parser.add_argument("--num_frames",     type=int,   default=81,     help="Number of video frames (default: 81)")
    parser.add_argument("--num_samples",    type=int,   default=1,      help="Number of samples per image (default: 1)")
    parser.add_argument("--seed",           type=int,   default=0,      help="Random seed (default: 0)")
    parser.add_argument("--attention_type", type=str,   default="sagesla", choices=["sla", "sagesla", "original"], help="Attention mechanism (default: sagesla)")
    parser.add_argument("--sla_topk",       type=float, default=0.1,    help="Top-k ratio for SLA/SageSLA attention (default: 0.1)")
    parser.add_argument("--ode",            action="store_true", default=True, help="Use ODE sampling (default: True)")
    parser.add_argument("--no_ode",         action="store_true", help="Disable ODE sampling (use SDE instead)")
    parser.add_argument("--quant_linear",   action="store_true", default=True, help="Use quantized linear layers (default: True)")
    parser.add_argument("--no_quant_linear", action="store_true", help="Disable quantized linear layers")
    parser.add_argument("--adaptive_resolution", action="store_true", default=True, help="Adapt resolution to input aspect ratio (default: True)")

    # ── Batch control ────────────────────────────────────────────────────
    parser.add_argument("--skip",           type=int,   default=0,  help="Skip first N rows in CSV (default: 0)")
    parser.add_argument("--limit",          type=int,   default=0,  help="Limit to N rows (0 = all, default: 0)")
    parser.add_argument("--skip_existing",  action="store_true",    help="Skip images whose output video already exists")
    parser.add_argument("--dry_run",        action="store_true",    help="Print commands without executing")

    args = parser.parse_args()

    # Handle --no_ode / --no_quant_linear overrides
    if args.no_ode:
        args.ode = False
    if args.no_quant_linear:
        args.quant_linear = False

    # Dispatch
    if args.csv_path:
        infer_from_csv(args.csv_path, args.output_dir, args)
    elif args.image_path:
        infer_from_image(args.image_path, args.output_dir, args)


if __name__ == "__main__":
    main()

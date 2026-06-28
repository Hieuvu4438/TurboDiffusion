"""
Fast Batch Face Restoration — In-Memory Pipeline (No subprocess, No model reload).

Models are loaded ONCE into VRAM and reused across ALL images in ALL CSV files.

Optimization strategy:
  1. Pre-compute UMT5 text embedding → free ~11.4 GB VRAM (clear_umt5_memory).
  2. Pin high_noise_model, low_noise_model, VAE to CUDA (48 GB VRAM, no CPU swap).
  3. Iterate all CSV rows in-process, calling the sampling loop directly.

CSV locations (same as run_all_restoration_side.py):
  degradation_experiment/side/downup/{L1,L2,L3}/side_downup_{l1,l2,l3}.csv
  degradation_experiment/side/motion/{L1,L2,L3}/side_motion_{l1,l2,l3}.csv
  degradation_experiment/side/salt-pepper/{L1,L2,L3}/side_salt-pepper_{l1,l2,l3}.csv
  degradation_experiment/side/screen/side_screen.csv

Usage:
    python scripts/run_all_restoration_side_fast.py
    python scripts/run_all_restoration_side_fast.py --dry_run
    python scripts/run_all_restoration_side_fast.py --skip_existing
    python scripts/run_all_restoration_side_fast.py --deg_types downup motion --levels L1 L2
    python scripts/run_all_restoration_side_fast.py --num_steps 4 --seed 42
"""

import csv
import math
import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# ── Project root & sys.path ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "turbodiffusion"))

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
import torchvision.transforms.v2 as T
from tqdm import tqdm

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from modify_model import tensor_kwargs, create_model

torch._dynamo.config.suppress_errors = True

# ── Constants ─────────────────────────────────────────────────────────────────

LEVELED_DEG_TYPES = ["downup", "motion", "salt-pepper"]
FLAT_DEG_TYPES    = ["screen"]
LEVELS            = ["L1", "L2", "L3"]
SIDE_DIR          = PROJECT_ROOT / "degradation_experiment" / "side"
OUTPUT_BASE       = PROJECT_ROOT / "output_full_new_combined"

DEFAULT_HIGH_MODEL    = str(PROJECT_ROOT / "checkpoints" / "TurboWan2.2-I2V-A14B-high-720P-quant.pth")
DEFAULT_LOW_MODEL     = str(PROJECT_ROOT / "checkpoints" / "TurboWan2.2-I2V-A14B-low-720P-quant.pth")
DEFAULT_VAE_PATH      = str(PROJECT_ROOT / "checkpoints" / "Wan2.1_VAE.pth")
DEFAULT_TEXT_ENC_PATH = str(PROJECT_ROOT / "checkpoints" / "models_t5_umt5-xxl-enc-bf16.pth")

PROMPT_TEXT = (
    "Hyper-realistic forensic-level restoration of a human face, 8k resolution, extremely precise facial geometry. "
    "Strict structural coherence, exact mapping of original facial proportions. Photorealistic raw texture. "
    "The video dictates a deliberate motion ending in a static, straight-on frontal view. The final frames strictly "
    "lock into a direct frontal portrait, resolving all facial geometries symmetrically without altering the "
    "original identity, neutral unedited appearance."
)

SIGMA_MAX  = 200.0
BOUNDARY   = 0.9
NUM_FRAMES = 81


# ── CSV / path helpers ────────────────────────────────────────────────────────

def find_all_csv_files(deg_types: list, levels: list) -> list:
    csv_files = []
    for deg_type in deg_types:
        if deg_type in LEVELED_DEG_TYPES:
            for level in levels:
                folder   = SIDE_DIR / deg_type / level
                csv_name = f"side_{deg_type}_{level.lower()}.csv"
                csv_path = folder / csv_name
                if csv_path.exists():
                    csv_files.append({"csv_path": csv_path, "deg_type": deg_type,
                                      "level": level, "is_flat": False})
                else:
                    print(f"  [WARN] CSV not found: {csv_path}")
        elif deg_type in FLAT_DEG_TYPES:
            folder   = SIDE_DIR / deg_type
            csv_name = f"side_{deg_type}.csv"
            csv_path = folder / csv_name
            if csv_path.exists():
                csv_files.append({"csv_path": csv_path, "deg_type": deg_type,
                                  "level": None, "is_flat": True})
            else:
                print(f"  [WARN] CSV not found: {csv_path}")
        else:
            print(f"  [WARN] Unknown deg_type '{deg_type}' — skipped")
    return csv_files


def build_output_dir(csv_info: dict) -> Path:
    if csv_info["is_flat"]:
        return OUTPUT_BASE / "side" / csv_info["deg_type"]
    return OUTPUT_BASE / "side" / csv_info["deg_type"] / csv_info["level"]


def read_csv_rows(csv_path: Path) -> list:
    rows = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            if row.get("person", "").strip():
                rows.append(row)
    return rows


def resolve_path(p: str) -> str:
    if not os.path.isabs(p):
        return str(PROJECT_ROOT / p)
    return p


def format_duration(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s   = divmod(rem, 60)
    if td.days > 0:
        return f"{td.days}d {h}h {m}m"
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ── Model loading ─────────────────────────────────────────────────────────────

def build_model_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build a minimal Namespace compatible with create_model()."""
    import argparse as _ap
    ns = _ap.Namespace(
        model          = "Wan2.2-A14B",
        attention_type = args.attention_type,
        sla_topk       = args.sla_topk,
        quant_linear   = args.quant_linear,
        default_norm   = False,
    )
    return ns


def load_text_embedding(text_encoder_path: str, prompt: str) -> torch.Tensor:
    """Step 1: compute text embedding, then free the T5 encoder from VRAM."""
    log.info(f"[Step 1] Computing UMT5 embedding …")
    with torch.no_grad():
        text_emb = get_umt5_embedding(
            checkpoint_path=text_encoder_path,
            prompts=prompt,
        ).to(**tensor_kwargs)
    clear_umt5_memory()
    torch.cuda.empty_cache()
    log.success("[Step 1] UMT5 freed. VRAM reclaimed.")
    return text_emb


def load_dit_models(high_path: str, low_path: str, model_args: argparse.Namespace):
    """Step 2: load both DiT models directly to CUDA and keep them there."""
    log.info("[Step 2] Loading DiT High-noise model → CUDA …")
    high_noise_model = create_model(dit_path=high_path, args=model_args)
    # create_model already calls .to(cuda).eval() internally
    torch.cuda.empty_cache()

    log.info("[Step 2] Loading DiT Low-noise model → CUDA …")
    low_noise_model = create_model(dit_path=low_path, args=model_args)
    torch.cuda.empty_cache()

    log.success("[Step 2] Both DiT models pinned to VRAM.")
    return high_noise_model, low_noise_model


def load_vae(vae_path: str) -> Wan2pt1VAEInterface:
    log.info("[Step 2] Loading VAE …")
    tokenizer = Wan2pt1VAEInterface(vae_pth=vae_path)
    log.success("[Step 2] VAE ready.")
    return tokenizer


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(
    image_path: str,
    tokenizer: Wan2pt1VAEInterface,
    adaptive_resolution: bool,
    resolution: str,
    aspect_ratio: str,
) -> tuple:
    """
    Returns (y, lat_h, lat_w, lat_t, w, h) where y is the VAE-encoded
    conditioning tensor [1, C_lat+4, T_lat, H_lat, W_lat].
    """
    input_image = Image.open(image_path).convert("RGB")
    F = NUM_FRAMES

    if adaptive_resolution:
        base_w, base_h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
        max_area = base_w * base_h
        orig_w, orig_h = input_image.size
        ar = orig_h / orig_w
        ideal_w = math.sqrt(max_area / ar)
        ideal_h = math.sqrt(max_area * ar)
        stride  = tokenizer.spatial_compression_factor * 2
        lat_h   = round(ideal_h / stride)
        lat_w   = round(ideal_w / stride)
        h = lat_h * stride
        w = lat_w * stride
    else:
        w, h  = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
        lat_h = h // tokenizer.spatial_compression_factor
        lat_w = w // tokenizer.spatial_compression_factor

    lat_t = tokenizer.get_latent_num_frames(F)

    transforms = T.Compose([
        T.ToImage(),
        T.Resize(size=(h, w), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image_tensor = (
        transforms(input_image)
        .unsqueeze(0)
        .to(device=tensor_kwargs["device"], dtype=torch.float32)
    )

    with torch.no_grad():
        frames_to_encode = torch.cat(
            [image_tensor.unsqueeze(2),
             torch.zeros(1, 3, F - 1, h, w, device=image_tensor.device)],
            dim=2,
        )
        encoded_latents = tokenizer.encode(frames_to_encode)
        del frames_to_encode
        torch.cuda.empty_cache()

    msk = torch.zeros(
        1, 4, lat_t, lat_h, lat_w,
        device=tensor_kwargs["device"], dtype=tensor_kwargs["dtype"],
    )
    msk[:, :, 0, :, :] = 1.0
    y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)

    return y, lat_h, lat_w, lat_t, w, h


# ── Sampling loop ─────────────────────────────────────────────────────────────

def run_sampling(
    high_noise_model: torch.nn.Module,
    low_noise_model:  torch.nn.Module,
    tokenizer:        Wan2pt1VAEInterface,
    text_emb:         torch.Tensor,
    y:                torch.Tensor,
    lat_h: int, lat_w: int, lat_t: int,
    num_samples: int,
    num_steps:   int,
    seed:        int,
    use_ode:     bool,
    save_path:   str,
) -> None:
    """Run the rCM ODE/SDE sampling loop and save video. Models stay on VRAM."""

    y = y.repeat(num_samples, 1, 1, 1, 1)
    condition = {
        "crossattn_emb": repeat(
            text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=num_samples
        ),
        "y_B_C_T_H_W": y,
    }

    state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]
    generator   = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(seed)

    init_noise = torch.randn(
        num_samples, *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    mid_t   = [1.5, 1.4, 1.0][: num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(SIGMA_MAX), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )
    # Convert TrigFlow → RectifiedFlow
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    x        = init_noise.to(torch.float64) * t_steps[0]
    ones     = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1

    # Both models stay on VRAM; select active net by boundary
    net      = high_noise_model
    switched = False

    for t_cur, t_next in tqdm(
        list(zip(t_steps[:-1], t_steps[1:])),
        desc="Sampling", total=total_steps, leave=False,
    ):
        if t_cur.item() < BOUNDARY and not switched:
            net      = low_noise_model
            switched = True

        with torch.no_grad():
            v_pred = net(
                x_B_C_T_H_W  = x.to(**tensor_kwargs),
                timesteps_B_T = (t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition,
            ).to(torch.float64)

            if use_ode:
                x = x - (t_cur - t_next) * v_pred
            else:
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape,
                    dtype=torch.float32,
                    device=tensor_kwargs["device"],
                    generator=generator,
                )

    samples = x.float()

    with torch.no_grad():
        video = tokenizer.decode(samples)

    to_show = (1.0 + torch.stack([video.float().cpu()], dim=0).clamp(-1, 1)) / 2.0
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    save_image_or_video(
        rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
        save_path,
        fps=16,
    )


# ── Per-image inference ───────────────────────────────────────────────────────

def infer_one_image(
    image_path:       str,
    save_path:        str,
    high_noise_model: torch.nn.Module,
    low_noise_model:  torch.nn.Module,
    tokenizer:        Wan2pt1VAEInterface,
    text_emb:         torch.Tensor,
    args:             argparse.Namespace,
    label:            str = "",
) -> bool:
    try:
        if label:
            print(f"  {label}")
        print(f"    image : {image_path}")
        print(f"    output: {save_path}")

        y, lat_h, lat_w, lat_t, w, h = preprocess_image(
            image_path         = image_path,
            tokenizer          = tokenizer,
            adaptive_resolution= args.adaptive_resolution,
            resolution         = args.resolution,
            aspect_ratio       = args.aspect_ratio,
        )

        run_sampling(
            high_noise_model = high_noise_model,
            low_noise_model  = low_noise_model,
            tokenizer        = tokenizer,
            text_emb         = text_emb,
            y                = y,
            lat_h=lat_h, lat_w=lat_w, lat_t=lat_t,
            num_samples      = args.num_samples,
            num_steps        = args.num_steps,
            seed             = args.seed,
            use_ode          = args.use_ode,
            save_path        = save_path,
        )
        print(f"    ✓ saved → {save_path}")
        return True
    except Exception as exc:
        print(f"    ✗ FAILED: {exc}")
        import traceback; traceback.print_exc()
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    all_deg_types = LEVELED_DEG_TYPES + FLAT_DEG_TYPES

    parser = argparse.ArgumentParser(
        description="Fast in-memory batch face restoration (no subprocess, no model reload)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Batch control ────────────────────────────────────────────────────────
    parser.add_argument("--dry_run",       action="store_true",
                        help="Preview paths without running inference")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip images whose output video already exists")
    parser.add_argument("--deg_types",     nargs="+", default=None,
                        help=f"Degradation types (default: all {all_deg_types})")
    parser.add_argument("--levels",        nargs="+", default=None,
                        help=f"Levels for leveled types (default: {LEVELS})")

    # ── Generation parameters ────────────────────────────────────────────────
    parser.add_argument("--num_steps",     type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--seed",          type=int, default=0)
    parser.add_argument("--num_samples",   type=int, default=1)
    parser.add_argument("--resolution",    type=str, default="720p")
    parser.add_argument("--aspect_ratio",  type=str, default="16:9")
    parser.add_argument("--no_ode",        action="store_true",
                        help="Use SDE instead of ODE sampling")
    parser.add_argument("--adaptive_resolution", action="store_true", default=True)

    # ── Attention / quantization ─────────────────────────────────────────────
    parser.add_argument("--attention_type", default="sagesla",
                        choices=["sla", "sagesla", "original"])
    parser.add_argument("--sla_topk",       type=float, default=0.1)
    parser.add_argument("--quant_linear",   action="store_true", default=True)

    # ── Model checkpoints ────────────────────────────────────────────────────
    parser.add_argument("--high_model",      default=DEFAULT_HIGH_MODEL)
    parser.add_argument("--low_model",       default=DEFAULT_LOW_MODEL)
    parser.add_argument("--vae_path",        default=DEFAULT_VAE_PATH)
    parser.add_argument("--text_encoder_path", default=DEFAULT_TEXT_ENC_PATH)
    parser.add_argument("--prompt",          default=PROMPT_TEXT)

    args = parser.parse_args()
    args.use_ode = not args.no_ode

    deg_types = args.deg_types or all_deg_types
    levels    = args.levels    or LEVELS

    csv_files = find_all_csv_files(deg_types, levels)
    if not csv_files:
        print("ERROR: No CSV files found! Run generate_side_csvs.py first.")
        sys.exit(1)

    # ── Header ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  ⚡ Face Restoration (SIDE) — Fast In-Memory Batch Pipeline")
    print("=" * 70)
    print(f"  Degradations     : {', '.join(deg_types)}")
    print(f"  Levels           : {', '.join(levels)}")
    print(f"  CSV files        : {len(csv_files)}")
    print(f"  Skip existing    : {args.skip_existing}")
    print(f"  Dry run          : {args.dry_run}")
    print(f"  Num steps        : {args.num_steps}")
    print(f"  Seed             : {args.seed}")
    print(f"  Sampling         : {'ODE' if args.use_ode else 'SDE'}")
    print(f"  Attention        : {args.attention_type} (topk={args.sla_topk})")
    print(f"  Quant linear     : {args.quant_linear}")
    print(f"  Started at       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Listing all images that would be processed:\n")
        for cf in csv_files:
            output_dir = build_output_dir(cf)
            rows = read_csv_rows(cf["csv_path"])
            for row in rows:
                person     = row["person"]
                csv_stem   = cf["csv_path"].stem
                save_path  = output_dir / f"{person}_{csv_stem}.mp4"
                test_path  = resolve_path(row["test_path"])
                skip_mark  = " [SKIP]" if (args.skip_existing and save_path.exists()) else ""
                print(f"  {cf['deg_type']}/{cf['level'] or 'flat'}  {person}{skip_mark}")
                print(f"    {test_path}")
                print(f"    → {save_path}")
        print("\n[DRY RUN] Done.")
        return

    # ── Step 1: Text embedding ────────────────────────────────────────────────
    text_emb = load_text_embedding(args.text_encoder_path, args.prompt)

    # ── Step 2: Load models to VRAM ───────────────────────────────────────────
    model_args = build_model_args(args)
    high_noise_model, low_noise_model = load_dit_models(
        args.high_model, args.low_model, model_args
    )
    tokenizer = load_vae(args.vae_path)

    log.success("All models loaded. Starting inference loop …\n")

    # ── Step 3: Inference loop ────────────────────────────────────────────────
    total_images  = 0
    total_skipped = 0
    total_ok      = 0
    total_fail    = 0
    global_start  = time.time()

    for cf_idx, cf in enumerate(csv_files, 1):
        deg_type   = cf["deg_type"]
        level      = cf["level"] or "(flat)"
        csv_path   = cf["csv_path"]
        output_dir = build_output_dir(cf)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows     = read_csv_rows(csv_path)
        csv_stem = csv_path.stem

        print(f"\n{'╔' + '═'*68 + '╗'}")
        print(f"  [{cf_idx}/{len(csv_files)}] {deg_type.upper()} — level={level}  ({len(rows)} images)")
        print(f"  CSV: {csv_path.name}")
        print(f"  Output: {output_dir}")
        print(f"{'╚' + '═'*68 + '╝'}")

        csv_start = time.time()

        for row_idx, row in enumerate(rows, 1):
            person    = row["person"]
            test_path = resolve_path(row["test_path"])
            save_path = str(output_dir / f"{person}_{csv_stem}.mp4")

            total_images += 1

            if args.skip_existing and Path(save_path).exists():
                print(f"  [{row_idx}/{len(rows)}] {person} — SKIPPED (exists)")
                total_skipped += 1
                continue

            if not Path(test_path).exists():
                print(f"  [{row_idx}/{len(rows)}] {person} — SKIPPED (source missing: {test_path})")
                total_fail += 1
                continue

            t0 = time.time()
            ok = infer_one_image(
                image_path       = test_path,
                save_path        = save_path,
                high_noise_model = high_noise_model,
                low_noise_model  = low_noise_model,
                tokenizer        = tokenizer,
                text_emb         = text_emb,
                args             = args,
                label            = f"  [{row_idx}/{len(rows)}] {person}",
            )
            elapsed = time.time() - t0
            if ok:
                total_ok += 1
                print(f"    ⏱  {format_duration(elapsed)}")
            else:
                total_fail += 1

        csv_elapsed = time.time() - csv_start
        print(f"\n  CSV done in {format_duration(csv_elapsed)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - global_start
    print(f"\n{'='*70}")
    print(f"  ✅ ALL JOBS COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time    : {format_duration(total_elapsed)}")
    print(f"  Total images  : {total_images}")
    print(f"  Succeeded     : {total_ok}")
    print(f"  Skipped       : {total_skipped}")
    print(f"  Failed        : {total_fail}")
    print(f"  Finished at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output base   : {OUTPUT_BASE}/side/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

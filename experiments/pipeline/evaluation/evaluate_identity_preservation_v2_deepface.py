"""
evaluate_identity_preservation_v2_deepface.py

Same as evaluate_identity_preservation_v2.py but using DeepFace instead of InsightFace.
Output structure identical to evaluate_v2.

Usage:
  python scripts/evaluate_identity_preservation_v2_deepface.py --device gpu
  python scripts/evaluate_identity_preservation_v2_deepface.py --device cpu
  python scripts/evaluate_identity_preservation_v2_deepface.py --device gpu --skip-existing
"""

import os
import sys
import argparse

# ── Parse --device BEFORE importing DeepFace (controls GPU/CPU) ──
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
_pre_args, _ = _pre_parser.parse_known_args()

if _pre_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("[Device] CPU (CUDA_VISIBLE_DEVICES=-1)")
else:
    print("[Device] GPU")

import re
import cv2
import numpy as np
import pandas as pd
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.config import (
    INPUT_BASE,
    COSINE_OUTPUT,
    FRONTAL_REF,
    SIDE_REF,
    RECOGNITION_THRESHOLD,
    VIDEOS_PER_CATEGORY,
)
from experiments.faceid import cosine_similarity, extract_person_name
from experiments.faceid.embeddings_deepface import (
    get_embedding_robust,
    get_image_embedding,
    get_best_embedding_from_video,
)

# Per-script output directory under the shared cosine-similarity output root.
OUTPUT_BASE = str(COSINE_OUTPUT / "evaluate_v2_deepface")


def build_dataset_entries():
    entries = []

    # ── FRONTAL: downup / jpeg / motion / salt-pepper ──
    for noise in ["downup", "jpeg", "motion", "salt-pepper"]:
        for level in ["L1", "L2", "L3"]:
            video_dir = os.path.join(INPUT_BASE, "frontal", noise, level)
            if os.path.isdir(video_dir):
                entries.append({
                    "view": "frontal",
                    "noise_type": noise,
                    "level": level,
                    "video_dir": video_dir,
                    "ref_dir": FRONTAL_REF,
                })

    # ── FRONTAL: lowlight ──
    for level in ["L1", "L2", "L3"]:
        video_dir = os.path.join(INPUT_BASE, "frontal", "lowlight", level)
        if os.path.isdir(video_dir):
            entries.append({
                "view": "frontal",
                "noise_type": "lowlight",
                "level": level,
                "video_dir": video_dir,
                "ref_dir": FRONTAL_REF,
            })

    # ── FRONTAL: gaussian_blur ──
    for level in ["blurred10", "blurred12", "blurred15"]:
        video_dir = os.path.join(INPUT_BASE, "frontal", "gaussian_blur", level)
        if os.path.isdir(video_dir):
            entries.append({
                "view": "frontal",
                "noise_type": "gaussian_blur",
                "level": level,
                "video_dir": video_dir,
                "ref_dir": FRONTAL_REF,
            })

    # ── SIDE: downup / motion / salt-pepper ──
    for noise in ["downup", "motion", "salt-pepper"]:
        for level in ["L1", "L2", "L3"]:
            video_dir = os.path.join(INPUT_BASE, "side", noise, level)
            if os.path.isdir(video_dir):
                entries.append({
                    "view": "side",
                    "noise_type": noise,
                    "level": level,
                    "video_dir": video_dir,
                    "ref_dir": SIDE_REF,
                })

    # ── SIDE: lowlight ──
    for level in ["L1", "L2", "L3"]:
        video_dir = os.path.join(INPUT_BASE, "side", "lowlight", level)
        if os.path.isdir(video_dir):
            entries.append({
                "view": "side",
                "noise_type": "lowlight",
                "level": level,
                "video_dir": video_dir,
                "ref_dir": SIDE_REF,
            })

    return entries


# ─────────────────────────────────────────────
#  PROCESS A SINGLE ENTRY
# ─────────────────────────────────────────────
def process_entry(entry, model_name, summary_rows):
    view       = entry["view"]
    noise_type = entry["noise_type"]
    level      = entry["level"]
    video_dir  = entry["video_dir"]
    ref_dir    = entry["ref_dir"]

    out_dir = os.path.join(OUTPUT_BASE, view, noise_type, level)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  [{view}] {noise_type} / {level}")
    print(f"  Video dir: {video_dir}")
    print(f"  Ref dir:   {ref_dir}")
    print(f"{'='*60}")

    # Get person list from ref_dir
    ref_files = glob.glob(os.path.join(ref_dir, "*_ref.jpg"))
    known_persons = [os.path.basename(f).replace("_ref.jpg", "") for f in ref_files]
    if not known_persons:
        print(f"  [SKIP] No ref images in {ref_dir}")
        return

    # Scan video files and match with ref
    mp4_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not mp4_files:
        print(f"  [SKIP] No .mp4 files in {video_dir}")
        return

    rows = []
    for mp4_path in mp4_files:
        stem = os.path.splitext(os.path.basename(mp4_path))[0]
        person = extract_person_name(stem, known_persons)
        if person is None:
            print(f"  [WARN] Cannot parse person from: {os.path.basename(mp4_path)}")
            continue
        ref_path = os.path.join(ref_dir, f"{person}_ref.jpg")
        if not os.path.exists(ref_path):
            print(f"  [WARN] Ref not found: {ref_path}")
            continue
        rows.append({"person": person, "video_path": mp4_path, "ref_path": ref_path})

    if not rows:
        print(f"  [SKIP] No valid video-ref pairs")
        return

    unique_persons = sorted(set(r["person"] for r in rows))
    print(f"  Total persons: {len(unique_persons)}  |  Total videos: {len(rows)}")

    # ── Pass 1: Extract embeddings ──
    print(f"  Extracting embeddings (DeepFace: {model_name})...")
    embeddings = {}
    for i, r in enumerate(rows):
        person = r["person"]
        v_path = r["video_path"]
        ref_path = r["ref_path"]

        # Ref embedding (cache by person)
        if person not in embeddings:
            ref_emb = get_image_embedding(ref_path, model_name)
            embeddings[person] = {"ref_emb": ref_emb, "video_emb": None}
        else:
            ref_emb = embeddings[person]["ref_emb"]

        # Video embedding
        vid_emb = get_best_embedding_from_video(v_path, model_name, ref_emb)
        r["video_emb"] = vid_emb

        if (i + 1) % 10 == 0 or i == 0:
            print(f"    [{i+1:>4}/{len(rows)}] processed...")

    print(f"    [{len(rows)}/{len(rows)}] done.")

    # ── Pass 2: Build 1:1 balanced pairs ──
    print(f"  Building 1:1 balanced pairs...")
    np.random.seed(42)
    pairs = []

    # Positive pairs: video vs ref of same person
    for r in rows:
        sim = cosine_similarity(r["video_emb"], embeddings[r["person"]]["ref_emb"])
        pairs.append({
            "person_A": r["person"],
            "person_B": r["person"],
            "label": 1,
            "pair_type": "Positive",
            "cosine_similarity": sim,
        })

    # Negative pairs: video of person A vs ref of person B
    for r in rows:
        p_A = r["person"]
        available = [p for p in unique_persons if p != p_A]
        if not available:
            continue
        p_B = np.random.choice(available)
        sim = cosine_similarity(r["video_emb"], embeddings[p_B]["ref_emb"])
        pairs.append({
            "person_A": p_A,
            "person_B": p_B,
            "label": 0,
            "pair_type": "Negative",
            "cosine_similarity": sim,
        })

    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(os.path.join(out_dir, "pairs_evaluation.csv"), index=False)

    # ── Pass 3: Metrics ──
    y_true = df_pairs["label"].values
    y_scores = df_pairs["cosine_similarity"].values

    y_pred = (y_scores >= RECOGNITION_THRESHOLD).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    mean_cos = np.mean(y_scores)

    pos_mean = df_pairs[df_pairs["label"] == 1]["cosine_similarity"].mean()
    neg_mean = df_pairs[df_pairs["label"] == 0]["cosine_similarity"].mean()

    print(f"\n  Results (threshold={RECOGNITION_THRESHOLD}):")
    print(f"  Mean Cos: {mean_cos:.4f}  |  Pos mean: {pos_mean:.4f}  |  Neg mean: {neg_mean:.4f}")
    print(f"  Accuracy: {acc:.4f}  |  Precision: {prec:.4f}  |  Recall: {rec:.4f}  |  F1: {f1:.4f}")

    # Write summary report
    report = (
        f"View: {view} | Noise: {noise_type} | Level: {level}\n"
        f"Total Persons: {len(unique_persons)}\n"
        f"Total Pairs: {len(df_pairs)} (Pos: {len(df_pairs)//2}, Neg: {len(df_pairs)//2})\n"
        f"Threshold: {RECOGNITION_THRESHOLD}\n"
        f"Mean Cosine Similarity: {mean_cos:.4f}\n"
        f"Positive mean cos: {pos_mean:.4f}\n"
        f"Negative mean cos: {neg_mean:.4f}\n"
        f"Accuracy:  {acc:.4f}  ({acc*100:.1f}%)\n"
        f"Precision: {prec:.4f}  ({prec*100:.1f}%)\n"
        f"Recall:    {rec:.4f}  ({rec*100:.1f}%)\n"
        f"F1-Score:  {f1:.4f}\n"
    )
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(report)

    summary_rows.append({
        "View": view,
        "Noise": noise_type,
        "Level": level,
        "Face Recognition": "deepface",
        "Total Persons": len(unique_persons),
        "Cosine Similarity mean": round(mean_cos, 4),
        "Threshold": RECOGNITION_THRESHOLD,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
    })


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate identity preservation (v2 DeepFace version)"
    )
    parser.add_argument(
        "--device", choices=["gpu", "cpu"], default="gpu",
        help="Device for DeepFace (default: gpu)"
    )
    parser.add_argument(
        "--model-name", default="ArcFace",
        choices=[
            "VGG-Face", "Facenet", "Facenet512", "OpenFace",
            "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",
        ],
        help="DeepFace model name (default: ArcFace)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=False,
        help="Skip entries with existing pairs_evaluation.csv"
    )
    args = parser.parse_args()

    device_str = "GPU" if args.device == "gpu" else "CPU"
    print(f"[Config] Model: DeepFace/{args.model_name} | Device: {device_str}")
    print(f"[Config] Threshold: {RECOGNITION_THRESHOLD} | Top-K: {TOP_K_AVERAGE}")

    entries = build_dataset_entries()
    print(f"\nFound {len(entries)} combinations:\n")
    for e in entries:
        print(f"  [{e['view']:8s}] {e['noise_type']:15s} {e['level']}")

    summary_rows = []

    for entry in entries:
        label = f"{entry['view']}/{entry['noise_type']}/{entry['level']}"
        out_dir = os.path.join(OUTPUT_BASE, entry["view"], entry["noise_type"], entry["level"])
        result_csv = os.path.join(out_dir, "pairs_evaluation.csv")

        if args.skip_existing and os.path.exists(result_csv):
            print(f"\n[{label}] [SKIP] Already exists at {result_csv}")
            continue

        process_entry(entry, args.model_name, summary_rows)

    # ── Save summary ──
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        csv_path = os.path.join(OUTPUT_BASE, "evaluation_summary.csv")
        xlsx_path = os.path.join(OUTPUT_BASE, "evaluation_summary.xlsx")
        df_summary.to_csv(csv_path, index=False)
        df_summary.to_excel(xlsx_path, index=False)

        print(f"\n{'='*70}")
        print(f"  FINAL SUMMARY")
        print(f"{'='*70}")
        print(df_summary.to_string(index=False))
        print(f"\nSaved: {csv_path}")
        print(f"Saved: {xlsx_path}")

    print(f"\nAll done. Results in: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()

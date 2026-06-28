"""Evaluate identity preservation on restored videos (InsightFace backend).

Refactored: shared helpers are imported from ``experiments.faceid`` and all
paths from ``experiments.config``. CLI, runtime behavior, and the ``summary.txt``
report format are unchanged.

Run from the repo root:
    python -m experiments.pipeline.evaluation.evaluate_identity_preservation_v2
"""
import os
import re
import sys
import argparse
import cv2
import glob
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
from experiments.faceid.embeddings_insightface import (
    get_embedding_robust,
    get_image_embedding,
    get_best_embedding_from_video,
)

# Per-script output directory under the shared cosine-similarity output root.
OUTPUT_BASE = str(COSINE_OUTPUT / "evaluate_v2")


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
def process_entry(entry, app, summary_rows):
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

    # Lấy danh sách person từ ref_dir
    ref_files = glob.glob(os.path.join(ref_dir, "*_ref.jpg"))
    known_persons = [os.path.basename(f).replace("_ref.jpg", "") for f in ref_files]
    if not known_persons:
        print(f"  [SKIP] No ref images in {ref_dir}")
        return

    # Quét video files và match với ref
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
    print(f"  Extracting embeddings...")
    embeddings = {}
    for i, r in enumerate(rows):
        person = r["person"]
        v_path = r["video_path"]
        ref_path = r["ref_path"]

        # Ref embedding (cache by person)
        if person not in embeddings:
            ref_emb = get_image_embedding(ref_path, app)
            embeddings[person] = {"ref_emb": ref_emb, "video_emb": None}
        else:
            ref_emb = embeddings[person]["ref_emb"]

        # Video embedding (using test2.py approach)
        vid_emb = get_best_embedding_from_video(v_path, app, ref_emb)
        r["video_emb"] = vid_emb

        if (i + 1) % 50 == 0 or i == 0:
            print(f"    [{i+1:>4}/{len(rows)}] processed...")

    print(f"    [{len(rows)}/{len(rows)}] done.")

    # ── Pass 2: Build 1:1 balanced pairs (negativeSample.py approach) ──
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
        "Face Recognition": "insightface",
        "Total Persons": len(unique_persons),
        "Cosine Similarity mean": round(mean_cos, 4),
        "Threshold": RECOGNITION_THRESHOLD,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
    })


# ══════════════════════════════════════════════════════════════
#  --test MODE: Interactive single-video face recognition test
# ══════════════════════════════════════════════════════════════

def build_video_menu():
    """
    Build a dictionary: category_label -> list of (video_path, ref_dir, person_name, ref_path)
    with VIDEOS_PER_CATEGORY videos per (view, noise_type, level) combo.
    """
    menu = {}

    ref_persons_cache = {}

    def get_ref_persons(ref_dir):
        if ref_dir not in ref_persons_cache:
            ref_files = glob.glob(os.path.join(ref_dir, "*_ref.jpg"))
            ref_persons_cache[ref_dir] = sorted(
                [os.path.basename(f).replace("_ref.jpg", "") for f in ref_files],
                key=len, reverse=True
            )
        return ref_persons_cache[ref_dir]

    entries = build_dataset_entries()
    for entry in entries:
        view = entry["view"]
        noise = entry["noise_type"]
        level = entry["level"]
        video_dir = entry["video_dir"]
        ref_dir = entry["ref_dir"]

        label = f"[{view}] {noise} / {level}"

        mp4_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
        if not mp4_files:
            continue

        known_persons = get_ref_persons(ref_dir)

        # Sample VIDEOS_PER_CATEGORY evenly
        n = len(mp4_files)
        if n <= VIDEOS_PER_CATEGORY:
            sampled = mp4_files
        else:
            indices = np.linspace(0, n - 1, VIDEOS_PER_CATEGORY, dtype=int)
            sampled = [mp4_files[i] for i in indices]

        video_list = []
        for mp4_path in sampled:
            stem = os.path.splitext(os.path.basename(mp4_path))[0]
            person = extract_person_name(stem, known_persons)
            if person is None:
                continue
            ref_path = os.path.join(ref_dir, f"{person}_ref.jpg")
            if not os.path.exists(ref_path):
                continue
            video_list.append((mp4_path, ref_dir, person, ref_path))

        menu[label] = video_list

    return menu


def interactive_select_video(menu):
    """Two-level interactive menu. Returns (video_path, ref_dir, person_name, ref_path, category_label)."""
    categories = sorted(menu.keys())

    # ── Level 1: Select noise category ──
    print("\n" + "=" * 70)
    print("  SELECT NOISE CATEGORY")
    print("=" * 70)
    for i, cat in enumerate(categories):
        n_videos = len(menu[cat])
        print(f"  [{i:>3}] {cat}  ({n_videos} videos)")

    while True:
        try:
            choice = input(f"\n  Enter category number [0-{len(categories)-1}]: ").strip()
            idx = int(choice)
            if 0 <= idx < len(categories):
                cat = categories[idx]
                break
        except (ValueError, EOFError):
            pass
        print("  Invalid selection, try again.")

    # ── Level 2: Select video ──
    video_list = menu[cat]
    print(f"\n{'='*70}")
    print(f"  SELECT VIDEO — {cat}")
    print(f"{'='*70}")

    # Show first 20, last 5 if list is long
    display_n = min(20, len(video_list))
    for i in range(display_n):
        vpath, _, person, _ = video_list[i]
        fname = os.path.basename(vpath)
        print(f"  [{i:>3}] {person:40s}  ({fname})")
    if len(video_list) > display_n:
        print(f"  ... ({len(video_list) - display_n - 5} more)")
        for i in range(len(video_list) - 5, len(video_list)):
            vpath, _, person, _ = video_list[i]
            fname = os.path.basename(vpath)
            print(f"  [{i:>3}] {person:40s}  ({fname})")

    while True:
        try:
            choice = input(f"\n  Enter video number [0-{len(video_list)-1}]: ").strip()
            idx = int(choice)
            if 0 <= idx < len(video_list):
                vpath, ref_dir, person, ref_path = video_list[idx]
                # Extract noise_type from category label (e.g. "[side] lowlight / L2" -> "lowlight")
                noise_type = cat.split("] ")[1].split(" /")[0] if "] " in cat else None
                return vpath, ref_dir, person, ref_path, cat, noise_type
        except (ValueError, EOFError):
            pass
        print("  Invalid selection, try again.")


def visualize_face_recognition(video_path, ref_dir, person_name, ref_path, app, output_dir, noise_type=None):
    """
    Run face recognition on a single video with full visualization:
    - Extract all sampled frames, detect faces, draw bounding boxes
    - Show similarity scores
    - Highlight top-K frames
    - Plot embedding stats
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  TEST: Face Recognition Visualization")
    print(f"{'='*70}")
    print(f"  Video:    {video_path}")
    print(f"  Person:   {person_name}")
    print(f"  Ref:      {ref_path}")
    print(f"  Output:   {output_dir}")

    # Load reference image and get embedding
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f"  [ERROR] Cannot read ref image: {ref_path}")
        return

    ref_emb, ref_face = get_embedding_robust(ref_img, app)
    if ref_emb is None:
        print(f"  [ERROR] Cannot detect face in ref image: {ref_path}")
        return

    # Draw bounding box on ref image
    if ref_face is not None:
        bbox = ref_face.bbox.astype(int)
        cv2.rectangle(ref_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(ref_img, f"REF: {person_name}", (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    ref_out = os.path.join(output_dir, "00_ref_image.jpg")
    cv2.imwrite(ref_out, ref_img)
    print(f"  Saved ref image: {ref_out}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_f = int(total_frames * SKIP_START_PERCENT)
    end_f = int(total_frames * (1 - SKIP_END_PERCENT))

    print(f"\n  Video info:")
    print(f"    Total frames: {total_frames}")
    print(f"    FPS: {fps:.1f}")
    print(f"    Resolution: {width}x{height}")
    print(f"    Scan range: frame {start_f} -> {end_f} (step=5)")

    # ── Scan all sampled frames ──
    frames_dir = os.path.join(output_dir, "sampled_frames")
    os.makedirs(frames_dir, exist_ok=True)

    candidates = []
    frame_count = 0
    face_detected_count = 0

    for i in range(start_f, end_f, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Apply preprocessing for degraded inputs
        proc_frame = preprocess_for_detection(frame, noise_type)

        # Detect ALL faces for visualization (try preprocessed, then padding fallback)
        faces_all = app.get(proc_frame)
        base_for_drawing = proc_frame
        if not faces_all:
            for pad_size in [200, 300]:
                padded = cv2.copyMakeBorder(proc_frame, pad_size, pad_size, pad_size, pad_size,
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
                faces_all = app.get(padded)
                if faces_all:
                    base_for_drawing = padded
                    break

        faces_drawn = base_for_drawing.copy()

        # Now get best face matching ref (on preprocessed frame)
        emb, best_face = get_embedding_robust(frame, app, ref_emb, noise_type)

        if emb is not None and best_face is not None:
            face_detected_count += 1
            sim = cosine_similarity(emb, ref_emb)
            candidates.append((sim, emb, i, frame.copy(), best_face))

            # Draw ALL detected faces (thin box), highlight best match (thick green)
            if faces_all:
                for f in faces_all:
                    fb = f.bbox.astype(int)
                    is_best = (best_face is not None and
                               np.allclose(f.bbox, best_face.bbox, atol=5))
                    if is_best:
                        color = (0, 255, 0)  # Green thick for best match
                        thickness = 3
                        label = f"BEST sim={sim:.3f}"
                    else:
                        color = (0, 165, 255)  # Orange thin for other faces
                        thickness = 1
                        label = f"face det={f.det_score:.2f}"
                    cv2.rectangle(faces_drawn, (fb[0], fb[1]), (fb[2], fb[3]), color, thickness)
                    cv2.putText(faces_drawn, label, (fb[0], max(fb[1] - 5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Frame number and similarity overlay
            cv2.putText(faces_drawn, f"Frame {i} | sim={sim:.4f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Save only frames with detected faces
            out_path = os.path.join(frames_dir, f"frame_{i:06d}_sim_{sim:.4f}.jpg")
            cv2.imwrite(out_path, faces_drawn)

    cap.release()
    print(f"\n  Scan complete:")
    print(f"    Sampled frames: {frame_count}")
    print(f"    Frames with face detected: {face_detected_count}")
    print(f"    Candidates for top-K: {len(candidates)}")

    if not candidates:
        print("\n  [DIAGNOSTIC] No faces detected in any frame!")
        print(f"    Noise type: {noise_type or 'unknown'}")
        print(f"    Possible causes:")
        print(f"      1. Degradation too severe — facial features destroyed")
        print(f"      2. Frame too dark — try lower --det-thresh")
        print(f"      3. Face too small relative to frame size")
        print(f"    Suggested actions:")
        print(f"      - Try --det-thresh 0.3")
        print(f"      - For lowlight L3: consider using a dedicated enhancement model")

        # Save diagnostic: first, middle, last sampled frames with brightness stats
        diag_dir = os.path.join(output_dir, "diagnostic_no_faces")
        os.makedirs(diag_dir, exist_ok=True)

        cap2 = cv2.VideoCapture(video_path)
        for tag, f_idx in [("first", start_f), ("mid", (start_f + end_f) // 2), ("last", end_f - 1)]:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap2.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                txt = f"Frame {f_idx} | brightness={gray.mean():.0f} | {noise_type or 'unknown'}"
                cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(diag_dir, f"frame_{tag}_{f_idx}.jpg"), frame)

                # Also save preprocessed version
                enhanced = preprocess_for_detection(frame, noise_type)
                cv2.putText(enhanced, txt + " [preprocessed]", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(diag_dir, f"frame_{tag}_{f_idx}_enhanced.jpg"), enhanced)
        cap2.release()
        print(f"    Saved diagnostic frames to: {diag_dir}")

        # Still save a summary
        with open(os.path.join(output_dir, "summary_test.txt"), "w") as fh:
            fh.write(f"Video: {video_path}\nPerson: {person_name}\n"
                     f"Noise type: {noise_type or 'unknown'}\n"
                     f"ERROR: No faces detected in {frame_count} sampled frames.\n")
        return

    # ── Sort by similarity, get top-K ──
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_k = min(TOP_K_AVERAGE, len(candidates))
    top_candidates = candidates[:top_k]

    # ── Save top-K frames highlighted ──
    topk_dir = os.path.join(output_dir, "top_k_frames")
    os.makedirs(topk_dir, exist_ok=True)

    for rank, (sim, emb, frame_idx, frame_img, best_face) in enumerate(top_candidates):
        out_img = frame_img.copy()
        bbox = best_face.bbox.astype(int)
        cv2.rectangle(out_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)

        # Large overlay text
        info_lines = [
            f"TOP-{rank+1} | Frame {frame_idx}",
            f"Cosine Similarity: {sim:.4f}",
            f"Person: {person_name}",
        ]
        y0 = 30
        for line in info_lines:
            cv2.putText(out_img, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            y0 += 30

        cv2.putText(out_img, "SELECTED FOR EMBEDDING AVERAGE",
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)

        tk_path = os.path.join(topk_dir, f"top{rank+1}_frame{frame_idx:06d}_sim{sim:.4f}.jpg")
        cv2.imwrite(tk_path, out_img)

    # ── Save the first and last frame from scan range ──
    cap2 = cv2.VideoCapture(video_path)
    for tag, f_idx in [("first", start_f), ("last", end_f - 1)]:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap2.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, f"frame_{tag}.jpg"), frame)
    cap2.release()

    # ── Average embedding from top-K (normalized) ──
    top_embs = [c[1] for c in top_candidates]
    avg_emb = np.mean(top_embs, axis=0)
    norm = np.linalg.norm(avg_emb)
    avg_emb_norm = avg_emb / norm if norm > 0 else avg_emb

    final_sim = cosine_similarity(avg_emb_norm, ref_emb)

    # ── Statistics ──
    all_sims = [c[0] for c in candidates]
    sims_array = np.array(all_sims)

    # ── Plot 1: Similarity distribution ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram of all similarities
    ax = axes[0, 0]
    ax.hist(sims_array, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=RECOGNITION_THRESHOLD, color='red', linestyle='--', linewidth=2,
               label=f'Threshold ({RECOGNITION_THRESHOLD})')
    for rank, (sim, _, _, _, _) in enumerate(top_candidates):
        ax.axvline(x=sim, color='green', linestyle=':', alpha=0.7,
                   label=f'Top-{rank+1}' if rank == 0 else f'Top-{rank+1}' if rank <= 3 else None)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frame Count')
    ax.set_title(f'Similarity Distribution — {person_name}')
    ax.legend(fontsize=8)

    # Embedding vector visualization (first 128 dims)
    ax = axes[0, 1]
    ax.plot(ref_emb[:128], 'b-', alpha=0.6, linewidth=0.8, label='Ref embedding')
    ax.plot(avg_emb_norm[:128], 'r-', alpha=0.6, linewidth=0.8, label='Avg video embedding')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Value')
    ax.set_title(f'Embedding Vectors (first 128 dims) — sim={final_sim:.4f}')
    ax.legend(fontsize=8)

    # Bar chart: top-10 similarities
    ax = axes[1, 0]
    top_n_show = min(10, len(candidates))
    top_sims = sims_array[:top_n_show]
    colors = ['#2ca02c' if i < top_k else '#1f77b4' for i in range(top_n_show)]
    bars = ax.bar(range(top_n_show), top_sims, color=colors, edgecolor='white')
    ax.axhline(y=RECOGNITION_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold ({RECOGNITION_THRESHOLD})')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'Top {top_n_show} Frame Similarities (green = used for avg)')
    ax.legend(fontsize=8)
    ax.set_xticks(range(top_n_show))
    ax.set_xticklabels([f'#{i+1}' for i in range(top_n_show)])

    # Text summary
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = (
        f"=== FACE RECOGNITION SUMMARY ===\n\n"
        f"Person:        {person_name}\n"
        f"Video:         {os.path.basename(video_path)}\n"
        f"Total frames:  {total_frames}\n"
        f"Scan range:    [{start_f}, {end_f}) step=5\n"
        f"Sampled:       {frame_count}\n"
        f"Faces found:   {face_detected_count}\n"
        f"Candidates:    {len(candidates)}\n\n"
        f"--- Similarity Stats ---\n"
        f"Mean:          {sims_array.mean():.4f}\n"
        f"Std:           {sims_array.std():.4f}\n"
        f"Min:           {sims_array.min():.4f}\n"
        f"Max:           {sims_array.max():.4f}\n"
        f"Median:        {np.median(sims_array):.4f}\n\n"
        f"--- Top-{top_k} Average ---\n"
        f"Final sim:     {final_sim:.4f}\n"
        f"Decision:      {'MATCH' if final_sim >= RECOGNITION_THRESHOLD else 'NO MATCH'}\n"
        f"Threshold:     {RECOGNITION_THRESHOLD}\n\n"
        f"--- Embedding ---\n"
        f"Dimension:     {ref_emb.shape[0]}\n"
        f"Ref norm:      {np.linalg.norm(ref_emb):.4f}\n"
        f"Video norm:    {np.linalg.norm(avg_emb_norm):.4f}\n"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "statistics_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved statistics plot: {plot_path}")

    # ── Plot 2: Per-frame similarity over time ──
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    frame_indices = [c[2] for c in candidates]
    ax2.plot(frame_indices, sims_array, 'b-', alpha=0.6, linewidth=1)
    ax2.scatter(frame_indices, sims_array, c=sims_array, cmap='RdYlGn', s=20, alpha=0.7)
    ax2.axhline(y=RECOGNITION_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
                label=f'Threshold ({RECOGNITION_THRESHOLD})')
    # Highlight top-K
    for rank, (sim, _, f_idx, _, _) in enumerate(top_candidates):
        ax2.scatter([f_idx], [sim], color='green', s=120, zorder=5,
                    marker='*', edgecolors='darkgreen', linewidths=1.5)
        ax2.annotate(f'T{rank+1}', (f_idx, sim), textcoords="offset points",
                     xytext=(0, 10), fontsize=8, color='darkgreen', fontweight='bold')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title(f'Per-Frame Similarity Over Time — {person_name}')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    timeline_path = os.path.join(output_dir, "similarity_timeline.png")
    fig2.savefig(timeline_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved similarity timeline: {timeline_path}")

    # ── Save embedding arrays ──
    np.savez(os.path.join(output_dir, "embeddings.npz"),
             ref_embedding=ref_emb,
             avg_video_embedding=avg_emb_norm,
             top_k_embeddings=np.array(top_embs),
             all_similarities=sims_array,
             frame_indices=np.array([c[2] for c in candidates]))

    # ── Save detailed CSV ──
    rows = []
    for rank, (sim, _, f_idx, _, face) in enumerate(candidates):
        rows.append({
            "rank": rank + 1,
            "frame_index": f_idx,
            "cosine_similarity": round(sim, 6),
            "used_for_average": rank < top_k,
            "bbox_x1": face.bbox[0],
            "bbox_y1": face.bbox[1],
            "bbox_x2": face.bbox[2],
            "bbox_y2": face.bbox[3],
            "det_score": round(float(face.det_score), 4) if hasattr(face, 'det_score') else 'N/A',
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "frame_candidates.csv"), index=False)

    # ── Write summary ──
    summary_text = (
        f"FACE RECOGNITION VISUALIZATION SUMMARY\n"
        f"{'='*60}\n"
        f"Video:        {video_path}\n"
        f"Person:       {person_name}\n"
        f"Ref image:    {ref_path}\n\n"
        f"--- Video Info ---\n"
        f"Total frames: {total_frames}\n"
        f"Resolution:   {width}x{height}\n"
        f"FPS:          {fps:.1f}\n"
        f"Scan range:   [{start_f}, {end_f}) step=5\n"
        f"Sampled:      {frame_count}\n"
        f"Faces found:  {face_detected_count}\n\n"
        f"--- Face Recognition ---\n"
        f"Model:        InsightFace buffalo_l\n"
        f"Threshold:    {RECOGNITION_THRESHOLD}\n"
        f"Top-K avg:    {top_k}\n\n"
        f"--- Similarity Stats (all candidates) ---\n"
        f"Mean:         {sims_array.mean():.4f}\n"
        f"Std:          {sims_array.std():.4f}\n"
        f"Min:          {sims_array.min():.4f}\n"
        f"Max:          {sims_array.max():.4f}\n"
        f"Median:       {np.median(sims_array):.4f}\n\n"
        f"--- Final Result ---\n"
        f"Final sim:    {final_sim:.4f}\n"
        f"Decision:     {'MATCH' if final_sim >= RECOGNITION_THRESHOLD else 'NO MATCH'}\n\n"
        f"--- Top-{top_k} Frames Used ---\n"
    )
    for rank, (sim, _, f_idx, _, _) in enumerate(top_candidates):
        summary_text += f"  Rank {rank+1}: Frame {f_idx}, sim={sim:.4f}\n"

    with open(os.path.join(output_dir, "summary_test.txt"), "w") as fh:
        fh.write(summary_text)

    # ── Console output ──
    print(f"\n  {'='*50}")
    print(f"  RESULTS")
    print(f"  {'='*50}")
    print(f"  Mean similarity:   {sims_array.mean():.4f}")
    print(f"  Std similarity:    {sims_array.std():.4f}")
    print(f"  Max similarity:    {sims_array.max():.4f}")
    print(f"  Final sim (top-{top_k} avg): {final_sim:.4f}")
    print(f"  Decision:          {'MATCH' if final_sim >= RECOGNITION_THRESHOLD else 'NO MATCH'}")
    print(f"\n  Output saved to: {output_dir}")
    print(f"  Files:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {f} ({size_kb:.1f} KB)")
        elif os.path.isdir(fpath):
            n_files = len(os.listdir(fpath))
            print(f"    {f}/ ({n_files} files)")

    return final_sim


# ══════════════════════════════════════════════════════════════
#  --full-stats MODE: Full statistics with plots
# ══════════════════════════════════════════════════════════════

def run_full_stats(app, args):
    """
    Run evaluation on ALL entries, then generate aggregate plots.
    """
    output_dir = os.path.join(OUTPUT_BASE, "full_stats")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nFull stats output: {output_dir}")

    entries = build_dataset_entries()
    print(f"Found {len(entries)} combinations to evaluate.\n")

    summary_rows = []
    for entry in entries:
        process_entry(entry, app, summary_rows)

    if not summary_rows:
        print("No results collected.")
        return

    df = pd.DataFrame(summary_rows)

    # Save raw summary
    df.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)
    df.to_excel(os.path.join(output_dir, "all_results.xlsx"), index=False)

    # ── Plot 1: F1 Score by Noise Type (grouped bar by view) ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        pivot = df.pivot_table(values=metric, index="Level", columns=["View", "Noise"], aggfunc="mean")
        pivot.plot(kind="bar", ax=ax, edgecolor="white")
        ax.set_title(f"{metric} by Level, View & Noise Type")
        ax.set_ylabel(metric)
        ax.set_xlabel("Level")
        ax.legend(fontsize=7, loc='lower left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)

    # ── Plot: Cosine Similarity Mean ──
    ax = axes[4]
    pivot = df.pivot_table(values="Cosine Similarity mean", index="Level", columns=["View", "Noise"], aggfunc="mean")
    pivot.plot(kind="bar", ax=ax, edgecolor="white")
    ax.set_title("Mean Cosine Similarity by Level, View & Noise Type")
    ax.set_ylabel("Cosine Similarity")
    ax.set_xlabel("Level")
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(axis='y', alpha=0.3)

    # ── Plot: Summary heatmap ──
    ax = axes[5]
    heatmap_data = df.pivot_table(values="F1 Score", index="Noise", columns="Level", aggfunc="mean")
    im = ax.imshow(heatmap_data.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    ax.set_title("F1 Score Heatmap (Noise x Level)")
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            ax.text(j, i, f"{heatmap_data.values[i, j]:.3f}",
                    ha='center', va='center', fontsize=9, fontweight='bold')
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "aggregate_metrics.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved aggregate metrics plot: {plot_path}")

    # ── Plot 2: Box plot of cosine similarity distribution per noise type ──
    # Load all pairs_evaluation.csv files to build distributions
    all_pairs = []
    for entry in entries:
        csv_path = os.path.join(OUTPUT_BASE, entry["view"], entry["noise_type"],
                                entry["level"], "pairs_evaluation.csv")
        if os.path.exists(csv_path):
            pdf = pd.read_csv(csv_path)
            pdf["view"] = entry["view"]
            pdf["noise_type"] = entry["noise_type"]
            pdf["level"] = entry["level"]
            all_pairs.append(pdf)

    if all_pairs:
        df_pairs_all = pd.concat(all_pairs, ignore_index=True)

        fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6))

        # Positive pairs distribution
        ax = axes2[0]
        pos_data = df_pairs_all[df_pairs_all["label"] == 1]
        noise_types = sorted(pos_data["noise_type"].unique())
        pos_box_data = [pos_data[pos_data["noise_type"] == n]["cosine_similarity"].values
                        for n in noise_types]
        bp = ax.boxplot(pos_box_data, labels=noise_types, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
        ax.axhline(y=RECOGNITION_THRESHOLD, color='red', linestyle='--',
                   linewidth=1.5, label=f'Threshold ({RECOGNITION_THRESHOLD})')
        ax.set_title('Positive Pairs (Same Person) Cosine Similarity Distribution')
        ax.set_xlabel('Noise Type')
        ax.set_ylabel('Cosine Similarity')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Negative pairs distribution
        ax = axes2[1]
        neg_data = df_pairs_all[df_pairs_all["label"] == 0]
        neg_box_data = [neg_data[neg_data["noise_type"] == n]["cosine_similarity"].values
                        for n in noise_types]
        bp = ax.boxplot(neg_box_data, labels=noise_types, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
        ax.axhline(y=RECOGNITION_THRESHOLD, color='red', linestyle='--',
                   linewidth=1.5, label=f'Threshold ({RECOGNITION_THRESHOLD})')
        ax.set_title('Negative Pairs (Different Person) Cosine Similarity Distribution')
        ax.set_xlabel('Noise Type')
        ax.set_ylabel('Cosine Similarity')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        box_path = os.path.join(output_dir, "cosine_distribution_boxplot.png")
        fig2.savefig(box_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved cosine distribution boxplot: {box_path}")

    # ── Print final table ──
    print(f"\n{'='*80}")
    print(f"  FULL STATISTICS SUMMARY (sorted by F1)")
    print(f"{'='*80}")
    df_sorted = df.sort_values("F1 Score", ascending=False)
    print(df_sorted.to_string(index=False))

    print(f"\nAll results saved to: {output_dir}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate identity preservation (v2 - test2.py approach)"
    )
    parser.add_argument(
        "--device", choices=["gpu", "cpu"], default="gpu",
        help="Device for InsightFace"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=False,
        help="Skip entries with existing pairs_evaluation.csv"
    )
    parser.add_argument(
        "--gpu-mem-limit", type=int, default=2,
        help="GPU memory limit in GB for ONNX Runtime (GPU mode only, default: 2)"
    )
    parser.add_argument(
        "--det-size", type=int, default=640,
        help="Detection input size (default: 640). Lower = less VRAM but less accurate."
    )
    parser.add_argument(
        "--det-thresh", type=float, default=0.5,
        help="Face detection confidence threshold (default: 0.5). "
             "Lower to 0.25-0.35 for heavily degraded inputs (L3 lowlight, motion blur, etc.)."
    )
    parser.add_argument(
        "--test", action="store_true", default=False,
        help="Interactive test mode: select a video by number and visualize face recognition "
             "(bounding boxes, frame selection, embeddings)."
    )
    parser.add_argument(
        "--full-stats", action="store_true", default=False,
        help="Run evaluation on ALL entries and generate aggregate statistics with plots."
    )
    args = parser.parse_args()

    if args.device == "gpu":
        gpu_mem_bytes = args.gpu_mem_limit * 1024 * 1024 * 1024
        providers = [
            ('CUDAExecutionProvider', {
                'gpu_mem_limit': str(gpu_mem_bytes),
                'arena_extend_strategy': 'kSameAsRequested',
            }),
            'CPUExecutionProvider',
        ]
        ctx_id = 0
        print(f"[Device] GPU (CUDA) | mem: {args.gpu_mem_limit}GB | det_size: {args.det_size} | det_thresh: {args.det_thresh}")
    else:
        providers = ['CPUExecutionProvider']
        ctx_id = -1
        print(f"[Device] CPU | det_size: {args.det_size} | det_thresh: {args.det_thresh}")

    print("Loading InsightFace buffalo_l model...")
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(args.det_size, args.det_size),
                det_thresh=args.det_thresh)
    print(f"Model loaded (det_thresh={args.det_thresh}).\n")

    # ── --test mode ──
    if args.test:
        menu = build_video_menu()
        if not menu:
            print("No video categories found!")
            return

        print(f"\nLoaded {len(menu)} noise categories with up to {VIDEOS_PER_CATEGORY} videos each.")
        video_path, ref_dir, person_name, ref_path, cat_label, noise_type = interactive_select_video(menu)

        # Build output dir name from selection
        safe_cat = cat_label.replace("/", "_").replace(" ", "").replace("[", "").replace("]", "")
        safe_person = person_name.replace("/", "_")
        output_dir = os.path.join(OUTPUT_BASE, "test_visualization", f"{safe_cat}_{safe_person}")

        visualize_face_recognition(video_path, ref_dir, person_name, ref_path,
                                   app, output_dir, noise_type)
        return

    # ── --full-stats mode ──
    if args.full_stats:
        run_full_stats(app, args)
        return

    # ── Default mode: process all entries normally ──
    entries = build_dataset_entries()
    print(f"Found {len(entries)} combinations:\n")
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

        process_entry(entry, app, summary_rows)

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

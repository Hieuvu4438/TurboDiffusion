import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parents[3]) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
"""
evaluate_identity_preservation_combined_deepface.py

Đánh giá face identity preservation (cosine similarity) cho toàn bộ kết quả
inference trong /home/haipd/TurboDiffusion/output_full_new_combined.

*** Phiên bản này dùng DeepFace thay cho InsightFace ***
Model mặc định: ArcFace (backbone Facenet512 cũng được hỗ trợ qua --model)

Cấu trúc đầu vào:
  output_full_new_combined/
  ├── downup/{L1,L2,L3}/        → FRONTAL
  ├── jpeg/{L1,L2,L3}/          → FRONTAL
  ├── motion/{L1,L2,L3}/        → FRONTAL
  ├── salt-pepper/{L1,L2,L3}/   → FRONTAL
  ├── lowlight/
  │   └── side/
  │       ├── frontal/{L1,L2,L3}/ → FRONTAL (lowlight)
  │       └── {L1,L2,L3}/         → SIDE (lowlight)
  └── side/
      ├── downup/{L1,L2,L3}/    → SIDE
      ├── motion/{L1,L2,L3}/    → SIDE
      └── salt-pepper/{L1,L2,L3}/ → SIDE

Ref images:
  - FRONTAL → Experiment_Data_Split_Combined/Frontal_Exp/Ref/{person}_ref.jpg
  - SIDE    → Experiment_Data_Split_Combined/Side_Exp/Ref/{person}_ref.jpg

Đầu ra:
  cosine_similarity_output/evaluate_full_deepface/
  ├── frontal/{noise_type}/{level}/
  └── side/{noise_type}/{level}/
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
import glob
import tempfile
from deepface import DeepFace
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
from experiments.config import INPUT_BASE, COSINE_OUTPUT, FRONTAL_REF, SIDE_REF
OUTPUT_BASE = str(COSINE_OUTPUT / "evaluate_full_deepface")
FRONTAL_REF = str(FRONTAL_REF)
SIDE_REF = str(SIDE_REF)

# ─────────────────────────────────────────────
#  DATASET DEFINITION
#  Mỗi entry: (view, noise_type, level, video_dir, ref_dir)
#  view        : "frontal" | "side"
#  noise_type  : tên loại nhiễu (dùng làm tên thư mục con output)
#  level       : "L1" | "L2" | "L3"
#  video_dir   : thư mục chứa các file .mp4
#  ref_dir     : thư mục chứa ảnh ref
# ─────────────────────────────────────────────

def build_dataset_entries():
    entries = []

    # ── FRONTAL: downup / jpeg / motion / salt-pepper ──
    for noise in ["downup", "jpeg", "motion", "salt-pepper"]:
        for level in ["L1", "L2", "L3"]:
            video_dir = os.path.join(INPUT_BASE, noise, level)
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
        video_dir = os.path.join(INPUT_BASE, "lowlight", "side", "frontal", level)
        if os.path.isdir(video_dir):
            entries.append({
                "view": "frontal",
                "noise_type": "lowlight",
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
        video_dir = os.path.join(INPUT_BASE, "lowlight", "side", level)
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
#  PARSE PERSON NAME FROM FILENAME
# ─────────────────────────────────────────────

def extract_person_name(stem: str, known_persons: list) -> str | None:
    """
    Tìm person name trong stem bằng cách khớp prefix từ danh sách known_persons.
    known_persons: list các tên người (không có _ref.jpg)
    """
    # Sắp xếp dài trước để tránh match nhầm prefix ngắn hơn
    for person in sorted(known_persons, key=len, reverse=True):
        if stem.startswith(person):
            return person
    return None


# ─────────────────────────────────────────────
#  GENERATE CSV FOR A VIDEO DIRECTORY
# ─────────────────────────────────────────────

def generate_csv(entry: dict) -> pd.DataFrame:
    """
    Quét video_dir để tìm các file .mp4, ghép với ref tương ứng,
    trả về DataFrame với cột: person, output_video_path, ref_image_path
    """
    video_dir = entry["video_dir"]
    ref_dir   = entry["ref_dir"]

    # Lấy danh sách person từ ref_dir
    ref_files = glob.glob(os.path.join(ref_dir, "*_ref.jpg"))
    known_persons = [os.path.basename(f).replace("_ref.jpg", "") for f in ref_files]

    if not known_persons:
        print(f"  [WARN] Không tìm thấy ref images trong {ref_dir}")
        return pd.DataFrame()

    rows = []
    mp4_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

    if not mp4_files:
        print(f"  [WARN] Không tìm thấy file .mp4 trong {video_dir}")
        return pd.DataFrame()

    for mp4_path in mp4_files:
        filename = os.path.basename(mp4_path)
        stem = os.path.splitext(filename)[0]
        person = extract_person_name(stem, known_persons)

        if person is None:
            print(f"  [WARN] Không parse được person từ: {filename}")
            continue

        ref_path = os.path.join(ref_dir, f"{person}_ref.jpg")
        if not os.path.exists(ref_path):
            print(f"  [WARN] Ref không tồn tại: {ref_path}")
            continue

        rows.append({
            "person": person,
            "output_video_path": mp4_path,
            "ref_image_path": ref_path,
        })

    df = pd.DataFrame(rows)
    print(f"  Generated CSV: {len(df)} rows from {video_dir}")
    return df


# ─────────────────────────────────────────────
#  FACE EMBEDDING EXTRACTION (DeepFace)
# ─────────────────────────────────────────────

def extract_image_embedding(image_path: str, model_name: str):
    """Trích xuất embedding từ ảnh ref sử dụng DeepFace."""
    try:
        if not os.path.exists(image_path):
            return {"emb": None, "err": "File not found", "image": None}
        img = cv2.imread(image_path)
        if img is None:
            return {"emb": None, "err": "Failed to read image", "image": None}

        result = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            enforce_detection=False,
            detector_backend="retinaface",
        )
        if not result:
            return {"emb": None, "err": "No face detected", "image": None}

        # Chọn mặt có diện tích lớn nhất
        best = max(result, key=lambda r: (
            r["facial_area"]["w"] * r["facial_area"]["h"]
        ))
        emb = np.array(best["embedding"], dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-10)  # L2 normalize
        return {"emb": emb, "err": None, "image": img}
    except Exception as e:
        return {"emb": None, "err": str(e), "image": None}


def _frame_pose_score(face_area: dict) -> float:
    """
    Heuristic: ưu tiên mặt có vùng lớn và gần trung tâm (tương đương yaw~0).
    DeepFace không trả về pose, nên ta dùng diện tích mặt làm proxy.
    """
    return face_area["w"] * face_area["h"]


def extract_video_embedding(video_path: str, model_name: str):
    """
    Quét toàn bộ video, chọn frame có mặt phát hiện với diện tích lớn nhất
    (tương tự logic 'frontal frame selection' trong phiên bản InsightFace).
    """
    cap = None
    tmp_path = None
    try:
        if not os.path.exists(video_path):
            return {"emb": None, "err": "File not found", "image": None}
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"emb": None, "err": "Failed to open video", "image": None}

        best_score = -1.0
        best_emb   = None
        best_img   = None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Lưu frame tạm ra file để DeepFace đọc
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                tmp_path = tf.name
            cv2.imwrite(tmp_path, frame)

            try:
                result = DeepFace.represent(
                    img_path=tmp_path,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend="retinaface",
                )
            except Exception:
                result = []
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    tmp_path = None

            if not result:
                continue

            # Chọn mặt lớn nhất trong frame
            best_face = max(result, key=lambda r: _frame_pose_score(r["facial_area"]))
            score = _frame_pose_score(best_face["facial_area"])

            if score > best_score:
                best_score = score
                emb = np.array(best_face["embedding"], dtype=np.float32)
                best_emb = emb / (np.linalg.norm(emb) + 1e-10)
                best_img = frame.copy()

        cap.release()

        if best_emb is not None:
            return {"emb": best_emb, "err": None, "image": best_img}
        else:
            return {"emb": None, "err": "No face detected in any frame", "image": None}
    except Exception as e:
        if cap and cap.isOpened():
            cap.release()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return {"emb": None, "err": str(e), "image": None}


def cosine_sim(emb1, emb2) -> float:
    """Tính cosine similarity (đã normalize → dot product). Trả np.nan nếu lỗi."""
    if emb1 is None or emb2 is None:
        return np.nan
    return float(np.dot(emb1, emb2))


# ─────────────────────────────────────────────
#  REPORT GENERATION
# ─────────────────────────────────────────────

def _classification_metrics(df_eval: pd.DataFrame):
    y_true   = df_eval['label'].values
    y_scores = df_eval['cosine_similarity'].values

    fixed_threshold = 0.15
    f_met = {"acc": 0, "prec": 0, "rec": 0, "f1": 0}
    best_threshold, best_f1 = 0.0, -1.0
    b_met = {"acc": 0, "prec": 0, "rec": 0, "f1": 0}

    if len(y_true) > 0:
        y_pred_f = (y_scores >= fixed_threshold).astype(int)
        f_met = {
            "acc":  accuracy_score(y_true, y_pred_f),
            "prec": precision_score(y_true, y_pred_f, zero_division=0),
            "rec":  recall_score(y_true, y_pred_f, zero_division=0),
            "f1":   f1_score(y_true, y_pred_f, zero_division=0),
        }
        for t in np.arange(-0.5, 1.0, 0.01):
            y_pred = (y_scores >= t).astype(int)
            f = f1_score(y_true, y_pred, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_threshold = t
                b_met = {
                    "acc":  accuracy_score(y_true, y_pred),
                    "prec": precision_score(y_true, y_pred, zero_division=0),
                    "rec":  recall_score(y_true, y_pred, zero_division=0),
                    "f1":   f,
                }
    return fixed_threshold, f_met, best_threshold, b_met


def generate_report(out_dir: str, view: str, noise_type: str, level: str,
                    total_persons: int, df_pairs: pd.DataFrame):
    df_valid  = df_pairs.dropna(subset=['cosine_similarity'])
    n_total   = len(df_pairs)
    n_proc    = len(df_valid)
    n_fail    = n_total - n_proc

    pos_stats = df_valid[df_valid['label'] == 1]['cosine_similarity'].describe().to_dict()
    neg_stats = df_valid[df_valid['label'] == 0]['cosine_similarity'].describe().to_dict()

    f_thr_val, f_met_val, b_thr_val, b_met_val = _classification_metrics(df_valid)

    df_sys = df_pairs.copy()
    df_sys['cosine_similarity'] = df_sys['cosine_similarity'].fillna(-1.0)
    f_thr_sys, f_met_sys, b_thr_sys, b_met_sys = _classification_metrics(df_sys)

    def fmt(m):
        return (f"    - Accuracy  : {m['acc']:.4f}  ({m['acc']*100:.1f}%)\n"
                f"    - Precision : {m['prec']:.4f}  ({m['prec']*100:.1f}%)\n"
                f"    - Recall    : {m['rec']:.4f}  ({m['rec']*100:.1f}%)\n"
                f"    - F1-Score  : {m['f1']:.4f}")

    lines = [
        "=" * 70,
        "  END-TO-END FACE IDENTITY PRESERVATION (1:1 BALANCED SET)",
        f"  View: {view} | Noise: {noise_type} | Level: {level}",
        "=" * 70,
        "",
        f"  Total Persons        : {total_persons}",
        f"  Total Pairs Created  : {n_total}  (Pos: {n_total//2}, Neg: {n_total//2})",
        f"  Successfully Processed: {n_proc} pairs",
        f"  Failed pairs (NaN)   : {n_fail}  (penalized as -1.0 in System-Level)",
        "",
        f"  Positive Cosine Sim  : mean={pos_stats.get('mean', float('nan')):.4f}  "
        f"std={pos_stats.get('std', float('nan')):.4f}  "
        f"min={pos_stats.get('min', float('nan')):.4f}  "
        f"max={pos_stats.get('max', float('nan')):.4f}",
        f"  Negative Cosine Sim  : mean={neg_stats.get('mean', float('nan')):.4f}  "
        f"std={neg_stats.get('std', float('nan')):.4f}  "
        f"min={neg_stats.get('min', float('nan')):.4f}  "
        f"max={neg_stats.get('max', float('nan')):.4f}",
        "",
        "─" * 70,
        "  [TIER 2] INTERSECTION METRICS  (only on successfully detected faces)",
        "─" * 70,
        f"  Fixed Threshold  : {f_thr_val:.2f}",
        fmt(f_met_val),
        "",
        f"  Optimal Threshold: {b_thr_val:.2f}  (Maximizing F1)",
        fmt(b_met_val),
        "",
        "─" * 70,
        "  [TIER 3] SYSTEM-LEVEL METRICS  (failed → cosine = -1.0)",
        "─" * 70,
        f"  Fixed Threshold  : {f_thr_sys:.2f}",
        fmt(f_met_sys),
        "",
        f"  Optimal Threshold: {b_thr_sys:.2f}  (Maximizing F1)",
        fmt(b_met_sys),
        "",
        "=" * 70,
    ]

    report_text = "\n".join(lines)
    print(report_text)
    with open(os.path.join(out_dir, "summary_1_1_evaluation.txt"), "w", encoding="utf-8") as fh:
        fh.write(report_text)


# ─────────────────────────────────────────────
#  CORE PROCESSING FUNCTION
# ─────────────────────────────────────────────

def process_entry(entry: dict, model_name: str, df: pd.DataFrame):
    """
    Xử lý một (view, noise_type, level):
      1. Save CSV vào output dir
      2. Trích embedding toàn bộ rows
      3. Build 1:1 pairs (positive + negative)
      4. Tính cosine sim, save kết quả, in report
    """
    view       = entry["view"]
    noise_type = entry["noise_type"]
    level      = entry["level"]

    out_dir = os.path.join(OUTPUT_BASE, view, noise_type, level)
    os.makedirs(out_dir, exist_ok=True)

    # Lưu CSV nguồn vào output dir để tiện tra cứu
    csv_path = os.path.join(out_dir, "evaluation_data.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"  Processing  view={view}  noise={noise_type}  level={level}")
    print(f"  Rows: {len(df)}   Output: {out_dir}")
    print(f"{'='*60}")

    np.random.seed(42)

    best_faces_dir  = os.path.join(out_dir, "best_faces")
    comparisons_dir = os.path.join(out_dir, "comparisons")
    os.makedirs(best_faces_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    indices         = df.index.tolist()
    unique_persons  = df['person'].unique().tolist()
    embeddings_cache = {}

    # ── Pass 1: Extract embeddings ──
    print(f"  Extracting embeddings ({len(df)} videos) using DeepFace [{model_name}]...")
    for idx, row in df.iterrows():
        person     = row['person']
        video_path = row['output_video_path']
        ref_path   = row['ref_image_path']

        vid_res = extract_video_embedding(video_path, model_name)
        ref_res = extract_image_embedding(ref_path, model_name)

        embeddings_cache[idx] = {
            "person":    person,
            "video_emb": vid_res["emb"],
            "video_err": vid_res["err"],
            "video_img": vid_res["image"],
            "ref_emb":   ref_res["emb"],
            "ref_err":   ref_res["err"],
            "ref_img":   ref_res["image"],
        }

        if vid_res["image"] is not None:
            face_save = os.path.join(best_faces_dir, f"{person}_best_face.jpg")
            cv2.imwrite(face_save, vid_res["image"])

        status = "OK" if vid_res["emb"] is not None else f"FAIL({vid_res['err']})"
        print(f"    [{idx+1:>4}/{len(df)}] {person:40s} → {status}")

    # ── Pass 2: Build 1:1 balanced pairs ──
    print(f"\n  Constructing 1:1 Pairs...")
    pairs_data = []

    # Positive pairs
    for idx in indices:
        cache = embeddings_cache[idx]
        sim = cosine_sim(cache["video_emb"], cache["ref_emb"])
        pairs_data.append({
            "person_A":         cache["person"],
            "person_B":         cache["person"],
            "label":            1,
            "pair_type":        "Positive",
            "cosine_similarity": sim,
            "video_err":        cache["video_err"],
            "ref_err":          cache["ref_err"],
        })
        # Save comparison image
        vid_img = cache["video_img"]
        ref_img = cache["ref_img"]
        if vid_img is not None and ref_img is not None:
            h  = min(vid_img.shape[0], ref_img.shape[0])
            w1 = int(ref_img.shape[1] * h / ref_img.shape[0])
            w2 = int(vid_img.shape[1] * h / vid_img.shape[0])
            comp = np.hstack((
                cv2.resize(ref_img, (w1, h)),
                cv2.resize(vid_img, (w2, h)),
            ))
            if not np.isnan(sim):
                cv2.putText(comp, f"Sim: {sim:.4f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(
                os.path.join(comparisons_dir, f"{cache['person']}_comparison.jpg"),
                comp,
            )

    # Negative pairs
    for idx in indices:
        p_A = embeddings_cache[idx]["person"]
        available = [p for p in unique_persons if p != p_A]
        if not available:
            continue
        p_B     = np.random.choice(available)
        idx_B   = np.random.choice([i for i in indices if embeddings_cache[i]["person"] == p_B])
        sim = cosine_sim(embeddings_cache[idx]["video_emb"], embeddings_cache[idx_B]["ref_emb"])
        pairs_data.append({
            "person_A":         p_A,
            "person_B":         p_B,
            "label":            0,
            "pair_type":        "Negative",
            "cosine_similarity": sim,
            "video_err":        embeddings_cache[idx]["video_err"],
            "ref_err":          embeddings_cache[idx_B]["ref_err"],
        })

    df_pairs = pd.DataFrame(pairs_data)
    df_pairs.to_csv(os.path.join(out_dir, "pairs_evaluation_1_1.csv"), index=False)

    # ── Generate report ──
    print(f"\n  Generating Metrics Report...")
    generate_report(out_dir, view, noise_type, level, len(unique_persons), df_pairs)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate face identity preservation using DeepFace (cosine similarity)"
    )
    parser.add_argument(
        "--model",
        default="ArcFace",
        choices=[
            "VGG-Face", "Facenet", "Facenet512", "OpenFace",
            "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",
        ],
        help="DeepFace model name (mặc định: ArcFace)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Bỏ qua các entry đã có file kết quả pairs_evaluation_1_1.csv (không chạy lại).",
    )
    args = parser.parse_args()

    print(f"[Model] DeepFace with backbone: {args.model}")

    entries = build_dataset_entries()
    print(f"Found {len(entries)} (view, noise_type, level) combinations:\n")
    for e in entries:
        print(f"  [{e['view']:8s}] {e['noise_type']:15s} {e['level']}  →  {e['video_dir']}")

    print()

    for entry in entries:
        label = f"{entry['view']}/{entry['noise_type']}/{entry['level']}"

        out_dir = os.path.join(
            OUTPUT_BASE, entry["view"], entry["noise_type"], entry["level"]
        )
        result_csv = os.path.join(out_dir, "pairs_evaluation_1_1.csv")

        # ── Skip nếu đã có kết quả và cờ --skip_existing được bật ──
        if args.skip_existing and os.path.exists(result_csv):
            print(f"\n[{label}] [SKIP] Đã có kết quả tại {result_csv}")
            continue

        print(f"\n[{label}] Generating CSV from .mp4 files...")
        df = generate_csv(entry)

        if df.empty:
            print(f"  [SKIP] No data for {label}")
            continue

        process_entry(entry, args.model, df)

    print(f"\nAll done. Results saved in: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()

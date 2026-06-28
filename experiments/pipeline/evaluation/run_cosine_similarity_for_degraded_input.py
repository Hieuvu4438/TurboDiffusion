import os
import sys
import cv2
import numpy as np
import pandas as pd
import glob
import json
import time
from pathlib import Path
from insightface.app import FaceAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.config import (
    PROJECT_ROOT,
    COSINE_OUTPUT,
    BLURRED_OUTPUT,
    BLURRED_OUTPUT_OLD,
)

OUTPUT_BASE = str(COSINE_OUTPUT / "degradation_new_02")

def extract_image_embedding(image_path, app):
    """Trích xuất vector embedding 512-d từ ảnh gốc sử dụng InsightFace"""
    try:
        path = image_path
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
            
        if not os.path.exists(path):
            return {"emb": None, "err": "File not found", "image": None}
            
        img = cv2.imread(path)
        if img is None:
            return {"emb": None, "err": "Failed to read image", "image": None}
            
        faces = app.get(img)
        if len(faces) == 0:
            return {"emb": None, "err": "No face detected", "image": None}
            
        # Chọn mặt lớn nhất
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        return {"emb": faces[0].normed_embedding, "err": None, "image": img}
    except Exception as e:
        return {"emb": None, "err": str(e), "image": None}

def calculate_cosine_similarity(emb1, emb2):
    """Tính toàn cosine similarity phân chuẩn, trả về np.nan nếu có lỗi"""
    try:
         if emb1 is None or emb2 is None:
             return np.nan
         return float(np.dot(emb1, emb2))
    except Exception:
         return np.nan

def generate_report(out_dir, view, blur_level, total_persons, df_pairs):
    """Phân tích và xuất báo cáo metrics"""
    
    # Loại bỏ NaN
    df_valid = df_pairs.dropna(subset=['cosine_similarity'])
    processed_pairs = len(df_valid)
    failed_pairs = len(df_pairs) - processed_pairs
    
    pos_df = df_valid[df_valid['label'] == 1]
    neg_df = df_valid[df_valid['label'] == 0]
    
    # Descriptive Statistics
    def get_stats(data):
        if len(data) == 0:
            return {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0, "count": 0}
        return {
            "mean": np.mean(data),
            "median": np.median(data),
            "min": np.min(data),
            "max": np.max(data),
            "std": np.std(data),
            "count": len(data)
        }
        
    pos_stats = get_stats(pos_df['cosine_similarity'].values)
    neg_stats = get_stats(neg_df['cosine_similarity'].values)
    
    # Classification Metrics - Find Optimal Threshold for F1 Score
    y_true = df_valid['label'].values
    y_scores = df_valid['cosine_similarity'].values
    
    best_threshold = 0.0
    best_f1 = -1.0
    metrics_at_best = {"acc": 0, "prec": 0, "rec": 0, "f1": 0}
    
    fixed_threshold = 0.15
    metrics_at_fixed = {"acc": 0, "prec": 0, "rec": 0, "f1": 0}
    
    if len(y_true) > 0:
        y_pred_fixed = (y_scores >= fixed_threshold).astype(int)
        metrics_at_fixed = {
            "acc": accuracy_score(y_true, y_pred_fixed),
            "prec": precision_score(y_true, y_pred_fixed, zero_division=0),
            "rec": recall_score(y_true, y_pred_fixed, zero_division=0),
            "f1": f1_score(y_true, y_pred_fixed, zero_division=0)
        }
        
        for t in np.arange(-0.5, 1.0, 0.01):
            y_pred = (y_scores >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
                metrics_at_best = {
                    "acc": accuracy_score(y_true, y_pred),
                    "prec": precision_score(y_true, y_pred, zero_division=0),
                    "rec": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1
                }
    
    lines = []
    lines.append("======================================================================")
    lines.append("  END-TO-END FACE IDENTITY PRESERVATION (1:1 BALANCED SET) DEGRADED")
    lines.append(f"  View: {view} | Blur Level: {blur_level}")
    lines.append("======================================================================")
    lines.append("")
    lines.append(f"  Total Persons       : {total_persons}")
    lines.append(f"  Total Pairs Created : {len(df_pairs)} (Pos: {len(df_pairs)//2}, Neg: {len(df_pairs)//2})")
    lines.append(f"  Successfully Processed: {processed_pairs} pairs")
    lines.append(f"  Failed pairs (NaN)  : {failed_pairs}")
    lines.append("")
    lines.append("──────────────────────────────────────────────────────────────────────")
    lines.append("  DESCRIPTIVE STATISTICS (COSINE SIMILARITY)")
    lines.append("──────────────────────────────────────────────────────────────────────")
    lines.append(f"  [POSITIVE PAIRS (Label = 1) | Count: {pos_stats['count']}]")
    lines.append(f"    - Mean   : {pos_stats['mean']:.4f}")
    lines.append(f"    - Median : {pos_stats['median']:.4f}")
    lines.append(f"    - Min    : {pos_stats['min']:.4f}")
    lines.append(f"    - Max    : {pos_stats['max']:.4f}")
    lines.append(f"    - Std    : {pos_stats['std']:.4f}")
    lines.append("")
    lines.append(f"  [NEGATIVE PAIRS (Label = 0) | Count: {neg_stats['count']}]")
    lines.append(f"    - Mean   : {neg_stats['mean']:.4f}")
    lines.append(f"    - Median : {neg_stats['median']:.4f}")
    lines.append(f"    - Min    : {neg_stats['min']:.4f}")
    lines.append(f"    - Max    : {neg_stats['max']:.4f}")
    lines.append(f"    - Std    : {neg_stats['std']:.4f}")
    lines.append("")
    lines.append("──────────────────────────────────────────────────────────────────────")
    lines.append("  CLASSIFICATION METRICS")
    lines.append("──────────────────────────────────────────────────────────────────────")
    lines.append(f"  [FIXED THRESHOLD: {fixed_threshold:.2f}]")
    lines.append(f"    - Accuracy      : {metrics_at_fixed['acc']:.4f}  ({metrics_at_fixed['acc']*100:.1f}%)")
    lines.append(f"    - Precision     : {metrics_at_fixed['prec']:.4f}  ({metrics_at_fixed['prec']*100:.1f}%)")
    lines.append(f"    - Recall        : {metrics_at_fixed['rec']:.4f}  ({metrics_at_fixed['rec']*100:.1f}%)")
    lines.append(f"    - F1-Score      : {metrics_at_fixed['f1']:.4f}")
    lines.append("")
    lines.append(f"  [OPTIMAL THRESHOLD: {best_threshold:.2f} (Maximizing F1)]")
    lines.append(f"    - Accuracy      : {metrics_at_best['acc']:.4f}  ({metrics_at_best['acc']*100:.1f}%)")
    lines.append(f"    - Precision     : {metrics_at_best['prec']:.4f}  ({metrics_at_best['prec']*100:.1f}%)")
    lines.append(f"    - Recall        : {metrics_at_best['rec']:.4f}  ({metrics_at_best['rec']*100:.1f}%)")
    lines.append(f"    - F1-Score      : {metrics_at_best['f1']:.4f}")
    lines.append("======================================================================")
    
    # Save Report
    with open(os.path.join(out_dir, "summary_1_1_evaluation.txt"), "w", encoding='utf-8') as f:
        f.write("\n".join(lines))
        
    print("\n".join(lines))
        
    # JSON output
    json_data = {
        "view": view,
        "blur_level": blur_level,
        "total_persons": total_persons,
        "processed_pairs": int(processed_pairs),
        "failed_pairs": int(failed_pairs),
        "positive_stats": {k: float(v) for k, v in pos_stats.items()},
        "negative_stats": {k: float(v) for k, v in neg_stats.items()},
        "classification": {
            "fixed_threshold_0.15": {
                "accuracy": float(metrics_at_fixed['acc']),
                "precision": float(metrics_at_fixed['prec']),
                "recall": float(metrics_at_fixed['rec']),
                "f1_score": float(metrics_at_fixed['f1'])
            },
            "optimal_threshold": float(best_threshold),
            "accuracy": float(metrics_at_best['acc']),
            "precision": float(metrics_at_best['prec']),
            "recall": float(metrics_at_best['rec']),
            "f1_score": float(metrics_at_best['f1']),
        }
    }
    with open(os.path.join(out_dir, "identity_metrics_1_1.json"), "w", encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

def process_dataframe(df, view, blur_level, app):
    df = df.dropna(subset=['person', 'test_path', 'ref_path']).reset_index(drop=True)
    
    out_dir = os.path.join(OUTPUT_BASE, view, blur_level)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\nProcessing {view} / {blur_level} ({len(df)} rows)...")
    np.random.seed(42) # For reproducible negative pairs
    
    # Pass 1: Extract all embeddings first
    print("  Extracting embeddings...")
    embeddings_cache = {}
    indices = df.index.tolist()
    unique_persons = list(set(df['person'].tolist()))
    
    best_faces_dir = os.path.join(out_dir, "best_faces")
    comparisons_dir = os.path.join(out_dir, "comparisons")
    os.makedirs(best_faces_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)
    
    for idx, row in df.iterrows():
        p = row['person']
        test_path = row['test_path']
        ref_path = row['ref_path']
        
        test_res = extract_image_embedding(test_path, app)
        ref_res = extract_image_embedding(ref_path, app)
        
        embeddings_cache[idx] = {
            "person": p,
            "test_emb": test_res["emb"],
            "test_err": test_res["err"],
            "test_img": test_res["image"],
            "ref_emb": ref_res["emb"],
            "ref_err": ref_res["err"],
            "ref_img": ref_res["image"]
        }
        
        if test_res["image"] is not None:
            best_face_path = os.path.join(best_faces_dir, f"{p}_{view}_{blur_level}_best_face.jpg")
            cv2.imwrite(best_face_path, test_res["image"])
        
    # Pass 2: Create 1:1 Pairs (Positive and Negative)
    print("  Constructing 1:1 Pairs...")
    pairs_data = []
    
    # 1. Positives
    for idx in indices:
        p = embeddings_cache[idx]["person"]
        t_emb = embeddings_cache[idx]["test_emb"]
        r_emb = embeddings_cache[idx]["ref_emb"]
        
        sim = calculate_cosine_similarity(t_emb, r_emb)
        pairs_data.append({
            "person_A": p,
            "person_B": p,
            "label": 1,
            "pair_type": "Positive",
            "cosine_similarity": sim,
            "test_err": embeddings_cache[idx]["test_err"],
            "ref_err": embeddings_cache[idx]["ref_err"]
        })
        
        test_img = embeddings_cache[idx]["test_img"]
        ref_img = embeddings_cache[idx]["ref_img"]
        if test_img is not None and ref_img is not None:
            h = min(test_img.shape[0], ref_img.shape[0])
            w1 = int(ref_img.shape[1] * (h / ref_img.shape[0]))
            w2 = int(test_img.shape[1] * (h / test_img.shape[0]))
            ref_resized = cv2.resize(ref_img, (w1, h))
            test_resized = cv2.resize(test_img, (w2, h))
            comparison = np.hstack((ref_resized, test_resized))
            if not np.isnan(sim):
                cv2.putText(comparison, f"Sim: {sim:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            comp_path = os.path.join(comparisons_dir, f"{p}_{view}_{blur_level}_comparison.jpg")
            cv2.imwrite(comp_path, comparison)
        
    # 2. Negatives
    for idx in indices:
        p_A = embeddings_cache[idx]["person"]
        available_persons = list(set(unique_persons) - {p_A})
        p_B = np.random.choice(available_persons)
        
        idx_B_choices = [i for i in indices if embeddings_cache[i]["person"] == p_B]
        idx_B = np.random.choice(idx_B_choices)
        
        t_emb = embeddings_cache[idx]["test_emb"]
        r_emb_B = embeddings_cache[idx_B]["ref_emb"]
        
        sim = calculate_cosine_similarity(t_emb, r_emb_B)
        pairs_data.append({
            "person_A": p_A,
            "person_B": p_B,
            "label": 0,
            "pair_type": "Negative",
            "cosine_similarity": sim,
            "test_err": embeddings_cache[idx]["test_err"],
            "ref_err": embeddings_cache[idx_B]["ref_err"]
        })
        
    df_pairs = pd.DataFrame(pairs_data)
    df_pairs.to_csv(os.path.join(out_dir, "pairs_evaluation_1_1.csv"), index=False)
    
    # Compute and Save metrics
    print("  Generating Metrics Report...")
    generate_report(out_dir, view, blur_level, len(unique_persons), df_pairs)

def main():
    print("Loading InsightFace buffalo_l model...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model loaded successfully.")
    
    base_dirs = [
        str(BLURRED_OUTPUT_OLD),
        str(BLURRED_OUTPUT),
    ]
    
    combinations = set()
    for base_dir in base_dirs:
        for csv_path in glob.glob(os.path.join(base_dir, "*/*/*.csv")):
            parts = csv_path.split(os.sep)
            view = parts[-3]
            blur_level = parts[-2]
            combinations.add((view, blur_level))
            
    # Sorting to ensure deterministic order
    combinations = sorted(list(combinations))
    print(f"Found {len(combinations)} unique view/blur combinations.")
    
    for view, blur_level in combinations:
        dfs = []
        for base_dir in base_dirs:
            csv_path = os.path.join(base_dir, view, blur_level, f"{view}_{blur_level}.csv")
            if os.path.exists(csv_path):
                dfs.append(pd.read_csv(csv_path))
                
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            process_dataframe(combined_df, view, blur_level, app)
            
    print("Evaluation completed. Outputs saved in " + OUTPUT_BASE)

if __name__ == "__main__":
    main()

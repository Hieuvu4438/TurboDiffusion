import os
import cv2
import numpy as np
import pandas as pd
import glob
import json
import time
import math
from insightface.app import FaceAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

OUTPUT_BASE = "/home/haipd/TurboDiffusion/cosine_similarity_output/i2v_new_03"

def extract_image_embedding(image_path, app):
    """Trích xuất vector embedding 512-d từ ảnh gốc"""
    try:
        if not os.path.exists(image_path):
            return {"emb": None, "err": "File not found", "image": None}
            
        img = cv2.imread(image_path)
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

def extract_video_embedding(video_path, app):
    """Trích xuất vector embedding 512-d từ video (quét toàn bộ video)"""
    try:
        if not os.path.exists(video_path):
            return {"emb": None, "err": "File not found", "image": None}
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"emb": None, "err": "Failed to open video", "image": None}
            
        best_err = float('inf')
        best_emb = None
        best_img = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            faces = app.get(frame)
            if len(faces) > 0:
                # Lọc mặt lớn nhất
                faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
                face = faces[0]
                
                # Điều kiện det_score > 0.6
                if face.det_score > 0.6:
                    pitch, yaw, roll = face.pose
                    error = abs(yaw) * 1.5 + abs(pitch) + abs(roll) * 0.5
                    
                    if error < best_err:
                        best_err = error
                        best_emb = face.normed_embedding
                        best_img = frame.copy()
                        
        cap.release()
        
        if best_emb is not None:
             return {"emb": best_emb, "err": None, "image": best_img}
        else:
             return {"emb": None, "err": "No suitable face found (det_score > 0.6)", "image": None}
             
    except Exception as e:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return {"emb": None, "err": str(e), "image": None}

def calculate_cosine_similarity(emb1, emb2):
    """Tính toàn cosine similarity, trả về np.nan nếu có lỗi"""
    try:
         if emb1 is None or emb2 is None:
             return np.nan
         return float(np.dot(emb1, emb2))
    except Exception:
         return np.nan
def generate_report(out_dir, view, blur_level, total_persons, df_pairs):
    """Phân tích và xuất báo cáo metrics (Bao gồm cả System-Level)"""
    
    # 1. TẬP VALID (CHỈ CHẤM CÁC CA DETECT THÀNH CÔNG - TIER 2)
    df_valid = df_pairs.dropna(subset=['cosine_similarity'])
    processed_pairs = len(df_valid)
    failed_pairs = len(df_pairs) - processed_pairs
    
    pos_stats = df_valid[df_valid['label'] == 1]['cosine_similarity'].describe().to_dict()
    neg_stats = df_valid[df_valid['label'] == 0]['cosine_similarity'].describe().to_dict()
    
    # Hàm con tính toán metrics
    def calculate_classification_metrics(df_eval):
        y_true = df_eval['label'].values
        y_scores = df_eval['cosine_similarity'].values
        
        fixed_threshold = 0.15
        metrics_at_fixed = {"acc": 0, "prec": 0, "rec": 0, "f1": 0}
        best_threshold, best_f1 = 0.0, -1.0
        metrics_at_best = {"acc": 0, "prec": 0, "rec": 0, "f1": 0}
        
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
        return fixed_threshold, metrics_at_fixed, best_threshold, metrics_at_best

    # Tính metrics cho Tier 2 (Intersection)
    f_thr_val, f_met_val, b_thr_val, b_met_val = calculate_classification_metrics(df_valid)
    
    # 2. TẬP SYSTEM-LEVEL (PHẠT CÁC CA HỎNG NaN THÀNH -1.0 - TIER 3)
    df_system = df_pairs.copy()
    df_system['cosine_similarity'] = df_system['cosine_similarity'].fillna(-1.0)
    f_thr_sys, f_met_sys, b_thr_sys, b_met_sys = calculate_classification_metrics(df_system)

    # Khối in text Report
    lines = [
        "======================================================================",
        "  END-TO-END FACE IDENTITY PRESERVATION (1:1 BALANCED SET)",
        f"  View: {view} | Blur Level: {blur_level}",
        "======================================================================",
        "",
        f"  Total Persons       : {total_persons}",
        f"  Total Pairs Created : {len(df_pairs)} (Pos: {len(df_pairs)//2}, Neg: {len(df_pairs)//2})",
        f"  Successfully Processed: {processed_pairs} pairs",
        f"  Failed pairs (NaN)  : {failed_pairs} (These are penalized as -1.0 in System-Level Metrics)",
        "",
        "──────────────────────────────────────────────────────────────────────",
        "  [TIER 2] INTERSECTION METRICS (Calculated ONLY on successfully detected faces)",
        "──────────────────────────────────────────────────────────────────────",
        f"  Fixed Threshold   : {f_thr_val:.2f}",
        f"    - Accuracy      : {f_met_val['acc']:.4f}  ({f_met_val['acc']*100:.1f}%)",
        f"    - Precision     : {f_met_val['prec']:.4f}  ({f_met_val['prec']*100:.1f}%)",
        f"    - Recall        : {f_met_val['rec']:.4f}  ({f_met_val['rec']*100:.1f}%)",
        f"    - F1-Score      : {f_met_val['f1']:.4f}",
        "",
        f"  Optimal Threshold : {b_thr_val:.2f} (Maximizing F1)",
        f"    - Accuracy      : {b_met_val['acc']:.4f}  ({b_met_val['acc']*100:.1f}%)",
        f"    - Precision     : {b_met_val['prec']:.4f}  ({b_met_val['prec']*100:.1f}%)",
        f"    - Recall        : {b_met_val['rec']:.4f}  ({b_met_val['rec']*100:.1f}%)",
        f"    - F1-Score      : {b_met_val['f1']:.4f}",
        "",
        "──────────────────────────────────────────────────────────────────────",
        "  [TIER 3] SYSTEM-LEVEL METRICS (Failed detections penalized as Cosine = -1.0)",
        "──────────────────────────────────────────────────────────────────────",
        f"  Fixed Threshold   : {f_thr_sys:.2f}",
        f"    - Accuracy      : {f_met_sys['acc']:.4f}  ({f_met_sys['acc']*100:.1f}%)",
        f"    - Precision     : {f_met_sys['prec']:.4f}  ({f_met_sys['prec']*100:.1f}%)",
        f"    - Recall        : {f_met_sys['rec']:.4f}  ({f_met_sys['rec']*100:.1f}%)",
        f"    - F1-Score      : {f_met_sys['f1']:.4f}",
        "",
        f"  Optimal Threshold : {b_thr_sys:.2f} (Maximizing F1)",
        f"    - Accuracy      : {b_met_sys['acc']:.4f}  ({b_met_sys['acc']*100:.1f}%)",
        f"    - Precision     : {b_met_sys['prec']:.4f}  ({b_met_sys['prec']*100:.1f}%)",
        f"    - Recall        : {b_met_sys['rec']:.4f}  ({b_met_sys['rec']*100:.1f}%)",
        f"    - F1-Score      : {b_met_sys['f1']:.4f}",
        "======================================================================"
    ]
    
    report_text = "\n".join(lines)
    print(report_text)
    
    with open(os.path.join(out_dir, "summary_1_1_evaluation.txt"), "w", encoding='utf-8') as f:
        f.write(report_text)


def process_dataframe(df, view, blur_level, app):
    df = df.dropna(subset=['person', 'output_video_path', 'ref_image_path']).reset_index(drop=True)
    
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
        video_path = row['output_video_path']
        ref_path = row['ref_image_path']
        
        vid_res = extract_video_embedding(video_path, app)
        ref_res = extract_image_embedding(ref_path, app)
        
        embeddings_cache[idx] = {
            "person": p,
            "video_emb": vid_res["emb"],
            "video_err": vid_res["err"],
            "video_img": vid_res["image"],
            "ref_emb": ref_res["emb"],
            "ref_err": ref_res["err"],
            "ref_img": ref_res["image"]
        }
        
        if vid_res["image"] is not None:
            best_face_path = os.path.join(best_faces_dir, f"{p}_{view}_{blur_level}_best_face.jpg")
            cv2.imwrite(best_face_path, vid_res["image"])
        
    # Pass 2: Create 1:1 Pairs (Positive and Negative)
    print("  Constructing 1:1 Pairs...")
    pairs_data = []
    
    # 1. Positives
    for idx in indices:
        p = embeddings_cache[idx]["person"]
        v_emb = embeddings_cache[idx]["video_emb"]
        r_emb = embeddings_cache[idx]["ref_emb"]
        
        sim = calculate_cosine_similarity(v_emb, r_emb)
        pairs_data.append({
            "person_A": p,
            "person_B": p,
            "label": 1,
            "pair_type": "Positive",
            "cosine_similarity": sim,
            "video_err": embeddings_cache[idx]["video_err"],
            "ref_err": embeddings_cache[idx]["ref_err"]
        })
        
        vid_img = embeddings_cache[idx]["video_img"]
        ref_img = embeddings_cache[idx]["ref_img"]
        if vid_img is not None and ref_img is not None:
            h = min(vid_img.shape[0], ref_img.shape[0])
            w1 = int(ref_img.shape[1] * (h / ref_img.shape[0]))
            w2 = int(vid_img.shape[1] * (h / vid_img.shape[0]))
            ref_resized = cv2.resize(ref_img, (w1, h))
            vid_resized = cv2.resize(vid_img, (w2, h))
            comparison = np.hstack((ref_resized, vid_resized))
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
        
        v_emb = embeddings_cache[idx]["video_emb"]
        r_emb_B = embeddings_cache[idx_B]["ref_emb"]
        
        sim = calculate_cosine_similarity(v_emb, r_emb_B)
        pairs_data.append({
            "person_A": p_A,
            "person_B": p_B,
            "label": 0,
            "pair_type": "Negative",
            "cosine_similarity": sim,
            "video_err": embeddings_cache[idx]["video_err"],
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
        "/home/haipd/TurboDiffusion/output_full",
        "/home/haipd/TurboDiffusion/output_full_new"
    ]
    
    combinations = set()
    for base_dir in base_dirs:
        for csv_path in glob.glob(os.path.join(base_dir, "*/*/*_full.csv")):
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
            csv_path = os.path.join(base_dir, view, blur_level, f"{view}_{blur_level}_full.csv")
            if os.path.exists(csv_path):
                dfs.append(pd.read_csv(csv_path))
                
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            process_dataframe(combined_df, view, blur_level, app)
            
    print("Evaluation completed. Outputs saved in " + OUTPUT_BASE)

if __name__ == "__main__":
    main()


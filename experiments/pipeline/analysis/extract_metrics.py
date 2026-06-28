import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parents[3]) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import os
import csv
import re

from experiments.config import COSINE_OUTPUT
base_dir = str(COSINE_OUTPUT / "evaluate_full")
output_csv = str(COSINE_OUTPUT / "evaluation_summary.csv")

# Regex patterns
total_persons_pattern = re.compile(r"Total Persons\s*:\s*(\d+)")
cosine_mean_pattern = re.compile(r"Positive Cosine Sim\s*:\s*mean=([0-9.]+)")

headers = [
    "View", "Noise", "Level", "Face Recognition", "Total Persons", 
    "Cosine Similarity mean", "Threshold", "Accuracy", "Precision", "Recall", "F1 Score"
]

results = []

for root, dirs, files in os.walk(base_dir):
    if "summary_1_1_evaluation.txt" in files:
        filepath = os.path.join(root, "summary_1_1_evaluation.txt")
        
        # Extract view, noise, level from path
        rel_path = os.path.relpath(root, base_dir)
        parts = rel_path.split(os.sep)
        
        view = "Unknown"
        noise = "Unknown"
        level = "Unknown"
        
        if len(parts) >= 3:
            view = parts[-3].capitalize()
            noise = parts[-2]
            level = parts[-1]
            
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        total_persons = ""
        cosine_mean = ""
        accuracy = ""
        precision = ""
        recall = ""
        f1_score = ""
        
        in_system_metrics = False
        in_fixed_threshold = False
        
        for line in lines:
            m_persons = total_persons_pattern.search(line)
            if m_persons:
                total_persons = m_persons.group(1)
                
            m_cosine = cosine_mean_pattern.search(line)
            if m_cosine:
                cosine_mean = m_cosine.group(1)
                
            if "[TIER 3] SYSTEM-LEVEL METRICS" in line:
                in_system_metrics = True
                
            if in_system_metrics:
                if "Fixed Threshold" in line and "0.15" in line:
                    in_fixed_threshold = True
                elif "Optimal Threshold" in line or ("Fixed Threshold" in line and "0.15" not in line):
                    if in_fixed_threshold:
                        in_fixed_threshold = False
                        
                if in_fixed_threshold:
                    if "- Accuracy" in line:
                        accuracy = line.split(":")[1].split("(")[0].strip()
                    elif "- Precision" in line:
                        precision = line.split(":")[1].split("(")[0].strip()
                    elif "- Recall" in line:
                        recall = line.split(":")[1].split("(")[0].strip()
                    elif "- F1-Score" in line:
                        f1_score = line.split(":")[1].strip()
                        
        results.append({
            "View": view,
            "Noise": noise,
            "Level": level,
            "Face Recognition": "insightface",
            "Total Persons": total_persons,
            "Cosine Similarity mean": cosine_mean,
            "Threshold": "0.15",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score
        })

with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Extraction complete. Processed {len(results)} files.")
print(f"Results saved to {output_csv}")

import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parents[3]) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import os
import re
import csv
import pandas as pd

from experiments.config import COSINE_OUTPUT
base_dir = str(COSINE_OUTPUT / "evaluate_full_deepface")
output_csv = str(COSINE_OUTPUT / "evaluation_summary_deepface.csv")
output_xlsx = str(COSINE_OUTPUT / "evaluation_summary_deepface.xlsx")

data = []

for view in ['side', 'frontal']:
    view_dir = os.path.join(base_dir, view)
    if not os.path.exists(view_dir):
        continue
    for noise in os.listdir(view_dir):
        noise_dir = os.path.join(view_dir, noise)
        if not os.path.isdir(noise_dir):
            continue
        for level in ['L1', 'L2', 'L3']:
            level_dir = os.path.join(noise_dir, level)
            summary_path = os.path.join(level_dir, 'summary_1_1_evaluation.txt')
            if not os.path.exists(summary_path):
                continue
            
            with open(summary_path, 'r') as f:
                content = f.read()
                
            total_persons = re.search(r'Total Persons\s*:\s*(\d+)', content)
            total_persons = int(total_persons.group(1)) if total_persons else 0
            
            cosine_sim = re.search(r'Positive Cosine Sim\s*:\s*mean=([\d.]+)', content)
            cosine_sim = float(cosine_sim.group(1)) if cosine_sim else 0.0
            
            # Find system-level metrics
            system_level_idx = content.find('[TIER 3] SYSTEM-LEVEL METRICS')
            if system_level_idx == -1:
                # Fallback if TIER 3 is not there
                system_level_idx = content.find('INTERSECTION METRICS')
            
            sub_content = content[system_level_idx:]
            
            # Find Fixed Threshold: 0.15 block
            fixed_block_idx = sub_content.find('Fixed Threshold  : 0.15')
            if fixed_block_idx != -1:
                fixed_block = sub_content[fixed_block_idx:]
                # Read next lines until empty line or next block
                accuracy = re.search(r'Accuracy\s*:\s*([\d.]+)', fixed_block)
                precision = re.search(r'Precision\s*:\s*([\d.]+)', fixed_block)
                recall = re.search(r'Recall\s*:\s*([\d.]+)', fixed_block)
                f1_score = re.search(r'F1-Score\s*:\s*([\d.]+)', fixed_block)
                
                acc_val = float(accuracy.group(1)) if accuracy else 0.0
                prec_val = float(precision.group(1)) if precision else 0.0
                rec_val = float(recall.group(1)) if recall else 0.0
                f1_val = float(f1_score.group(1)) if f1_score else 0.0
                
                data.append([
                    view.capitalize(),
                    noise,
                    level,
                    'deepface',
                    total_persons,
                    f"{cosine_sim:.4f}",
                    "0.15",
                    f"{acc_val:.4f}",
                    f"{prec_val:.4f}",
                    f"{rec_val:.4f}",
                    f"{f1_val:.4f}"
                ])

# Sort data to mimic original structure: View (Side then Frontal), then by noise, then L2, L1, L3 maybe?
# Let's just output it as is, or sorted
data.sort(key=lambda x: (x[0] != 'Side', x[1], ['L2', 'L1', 'L3'].index(x[2]) if x[2] in ['L2', 'L1', 'L3'] else 0))

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['View', 'Noise', 'Level', 'Face Recognition', 'Total Persons', 'Cosine Similarity mean', 'Threshold', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    writer.writerows(data)

df = pd.DataFrame(data, columns=['View', 'Noise', 'Level', 'Face Recognition', 'Total Persons', 'Cosine Similarity mean', 'Threshold', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
# Convert numerical columns
df['Total Persons'] = pd.to_numeric(df['Total Persons'])
for col in ['Cosine Similarity mean', 'Threshold', 'Accuracy', 'Precision', 'Recall', 'F1 Score']:
    df[col] = pd.to_numeric(df[col])
    
df.to_excel(output_xlsx, index=False)

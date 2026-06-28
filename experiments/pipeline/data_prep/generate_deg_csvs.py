import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parents[3]) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import os
import pandas as pd
import re

from experiments.config import PROJECT_ROOT, DEGRADATION_EXP
base_ref_dir = str(PROJECT_ROOT / "Experiment_Data_Split_Combined" / "Frontal_Exp" / "Ref")

def get_person_name(filename, suffix):
    # Replaces the suffix, eg "_downup_L1.png" -> ""
    base = filename.replace(suffix, "")
    # Drops the trailing "_0001" or similar
    person = re.sub(r'_\d{4}$', '', base)
    return person

def create_csv_for_dir(deg_type, level):
    dir_path = str(DEGRADATION_EXP / "frontal" / deg_type / level)
    if not os.path.exists(dir_path):
        print(f"Skipping {dir_path} (does not exist)")
        return
        
    csv_file = os.path.join(dir_path, f"frontal_{deg_type}_{level.lower()}.csv")
    
    rows = []
    suffix = f"_{deg_type}_{level}.png"
    
    for f in os.listdir(dir_path):
        if not f.endswith(suffix):
            continue
            
        person = get_person_name(f, suffix)
        test_path = f"degradation_experiment/frontal/{deg_type}/{level}/{f}"
        ref_path = f"{base_ref_dir}/{person}_ref.jpg"
        
        col_name = f"{deg_type}_{level.lower()}"
        val = level.replace('L', '') # e.g., 'L1' -> '1'
        
        rows.append({
            "person": person,
            "test_path": test_path,
            "ref_path": ref_path,
            col_name: val
        })
        
    if not rows:
        print(f"No files found in {dir_path}")
        return
        
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    print(f"Created {csv_file} with {len(df)} rows.")

for deg_type in ["downup", "motion"]:
    for level in ["L1", "L2", "L3"]:
        create_csv_for_dir(deg_type, level)


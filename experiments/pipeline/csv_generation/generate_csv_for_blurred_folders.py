import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parents[3]) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import os
import glob
import csv

from experiments.config import PROJECT_ROOT, BLURRED_OUTPUT
base_output_dir = str(PROJECT_ROOT / "output_full_new")
base_degraded_dir = str(BLURRED_OUTPUT)
base_exp_dir = str(PROJECT_ROOT / "Experiment_Data_Split_New")

views = ["frontal", "side"]
blurs = ["blurred10", "blurred12", "blurred15"]

for view in views:
    view_exp = "Frontal" if view == "frontal" else "Side"
    exp_folder = f"{view_exp}_Exp_New"
    
    for blur in blurs:
        k_num = blur.replace("blurred", "")
        
        output_folder = os.path.join(base_output_dir, view, blur)
        if not os.path.exists(output_folder):
            print(f"Directory not found: {output_folder}")
            continue
            
        csv_path = os.path.join(output_folder, f"{view}_{blur}_full.csv")
        
        mp4_files = glob.glob(os.path.join(output_folder, "*.mp4"))
        
        csv_data = []
        for mp4_file in sorted(mp4_files):
            filename = os.path.basename(mp4_file)
            # format is <person>_<view>_<blur>.mp4
            # remove suffix
            suffix = f"_{view}_{blur}.mp4"
            if filename.endswith(suffix):
                person = filename[:-len(suffix)]
            else:
                # Fallback if name format is slightly different
                person = filename.rsplit(f"_{view}_", 1)[0]
                
            degraded_image_path = os.path.join(base_degraded_dir, view, blur, f"{person}_test_blurred_{k_num}.jpg")
            output_video_path = mp4_file
            ref_image_path = os.path.join(base_exp_dir, exp_folder, "Ref", f"{person}_ref.jpg")
            test_image_path = os.path.join(base_exp_dir, exp_folder, "Test", f"{person}_test.jpg")
            
            csv_data.append([person, degraded_image_path, output_video_path, ref_image_path, test_image_path])
            
        if csv_data:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["person", "degraded_image_path", "output_video_path", "ref_image_path", "test_image_path"])
                writer.writerows(csv_data)
            print(f"Created {csv_path} with {len(csv_data)} rows.")
        else:
            print(f"No mp4 files found in {output_folder}.")


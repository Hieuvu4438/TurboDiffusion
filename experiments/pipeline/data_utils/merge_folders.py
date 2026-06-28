import os
import shutil
import pandas as pd

src_dirs = {
    "/home/haipd/TurboDiffusion/Experiment_Data_Split/Frontal_Exp/Ref": "Frontal_Exp/Ref",
    "/home/haipd/TurboDiffusion/Experiment_Data_Split/Frontal_Exp/Test": "Frontal_Exp/Test",
    "/home/haipd/TurboDiffusion/Experiment_Data_Split/Side_Exp/Ref": "Side_Exp/Ref",
    "/home/haipd/TurboDiffusion/Experiment_Data_Split/Side_Exp/Test": "Side_Exp/Test",
    "/home/haipd/TurboDiffusion/Experiment_Data_Split_New/Frontal_Exp_New/Ref": "Frontal_Exp/Ref",
    "/home/haipd/TurboDiffusion/Experiment_Data_Split_New/Frontal_Exp_New/Test": "Frontal_Exp/Test",
    "/home/haipd/TurboDiffusion/Experiment_Data_Split_New/Side_Exp_New/Ref": "Side_Exp/Ref",
    "/home/haipd/TurboDiffusion/Experiment_Data_Split_New/Side_Exp_New/Test": "Side_Exp/Test",
}

out_dir = "/home/haipd/TurboDiffusion/Experiment_Data_Split_Combined"

print("====================================")
print("1. MERGING FOLDERS")
print("====================================")
for src, rel_dst in src_dirs.items():
    if not os.path.exists(src):
        print(f"Directory {src} does not exist, skipping.")
        continue
    
    dst = os.path.join(out_dir, rel_dst)
    os.makedirs(dst, exist_ok=True)
    
    count = 0
    for f in os.listdir(src):
        src_f = os.path.join(src, f)
        if os.path.isfile(src_f):
            dst_f = os.path.join(dst, f)
            if not os.path.exists(dst_f):
                shutil.copy2(src_f, dst_f)
            count += 1
    print(f"Copied/Checked {count} files from {src} -> {dst}")
                
print("\nMerge complete. Now verifying against CSV files.\n")

print("====================================")
print("2. VERIFYING CSV WITH COMBINED FOLDER")
print("====================================")
for csv_name, exp_type in [("lfw_frontal_test.csv", "Frontal_Exp"), ("lfw_side_test.csv", "Side_Exp")]:
    csv_path = os.path.join("/home/haipd/TurboDiffusion", csv_name)
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        continue
        
    df = pd.read_csv(csv_path)
    
    ref_dir = os.path.join(out_dir, exp_type, "Ref")
    test_dir = os.path.join(out_dir, exp_type, "Test")
    
    missing_refs = []
    missing_tests = []
    
    ref_files = set(os.listdir(ref_dir)) if os.path.exists(ref_dir) else set()
    test_files = set(os.listdir(test_dir)) if os.path.exists(test_dir) else set()
    
    for person in df['person'].values:
        expected_ref = f"{person}_ref.jpg"
        expected_test = f"{person}_test.jpg"
        
        if expected_ref not in ref_files:
            missing_refs.append(expected_ref)
        if expected_test not in test_files:
            missing_tests.append(expected_test)
            
    print(f"\n--- Checking {csv_name} ({exp_type}) ---")
    print(f"Total entries in CSV: {len(df)}")
    
    print(f"Missing Ref files: {len(missing_refs)} out of {len(df)}")
    if missing_refs:
        print(f"  Examples: {missing_refs[:5]}")
        
    print(f"Missing Test files: {len(missing_tests)} out of {len(df)}")
    if missing_tests:
        print(f"  Examples: {missing_tests[:5]}")
        
    if len(missing_refs) == 0 and len(missing_tests) == 0:
        print(f"[SUCCESS] All names in {csv_name} exactly match the `{exp_type}` folder in the combined directory!")
    else:
        print(f"[WARNING] Some files are missing for {csv_name}.")


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parents[3]) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from experiments.config import FRONTAL_TEST_CSV, SIDE_TEST_CSV
import pandas as pd

def update_csv(csv_path, exp_type):
    print(f"Updating {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Update paths based on compiled folder structure
    df['test_path'] = df['person'].apply(lambda p: f"Experiment_Data_Split_Combined/{exp_type}/Test/{p}_test.jpg")
    df['ref_path'] = df['person'].apply(lambda p: f"Experiment_Data_Split_Combined/{exp_type}/Ref/{p}_ref.jpg")
    
    # Save a backup of original
    backup_path = csv_path + ".bak"
    if not pd.io.common.file_exists(backup_path):
        import shutil
        shutil.copy2(csv_path, backup_path)
    
    # Save the updated csv, keeping the same structure (e.g. sharpness/yaw column preserved)
    df.to_csv(csv_path, index=False)
    print(f"Successfully updated {csv_path} (Backup kept at .bak). Showing first 3 rows:")
    print(df.head(3))
    print()

if __name__ == "__main__":
    update_csv(str(FRONTAL_TEST_CSV), "Frontal_Exp")
    update_csv(str(SIDE_TEST_CSV), "Side_Exp")


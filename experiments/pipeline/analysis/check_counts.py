import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parents[3]) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import os

from experiments.config import INPUT_BASE
base_path = str(INPUT_BASE)
issues = []
checked_dirs = 0

for root, dirs, files in os.walk(base_path):
    if not dirs: # We only care about leaf directories usually for such datasets
        num_files = len([f for f in files if os.path.isfile(os.path.join(root, f))])
        checked_dirs += 1
        if num_files != 250:
            issues.append((root, num_files))
            
print(f"Checked {checked_dirs} directories.")
if not issues:
    print("All leaf directories have exactly 250 files.")
else:
    print(f"Found {len(issues)} directories that do NOT have 250 files:")
    for root, count in issues:
        print(f"{root}: {count}")


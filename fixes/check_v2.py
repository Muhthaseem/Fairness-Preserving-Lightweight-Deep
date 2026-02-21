import os
import time
from datetime import datetime

base_dir = r"D:\M3 Projects\DeepFake_Research"
models_dir = os.path.join(base_dir, "outputs", "models")
results_dir = os.path.join(base_dir, "outputs", "results")

print("--- Models ---")
for f in os.listdir(models_dir):
    if "fair_student" in f:
        path = os.path.join(models_dir, f)
        mtime = os.path.getmtime(path)
        dt_mtime = datetime.fromtimestamp(mtime)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{f:30} | {dt_mtime} | {size_mb:6.2f} MB")

print("\n--- Results CSVs (Top 5 lines) ---")
for f in os.listdir(results_dir):
    if f.endswith(".csv") and "Fair_Distill" in f:
        path = os.path.join(results_dir, f)
        mtime = os.path.getmtime(path)
        dt_mtime = datetime.fromtimestamp(mtime)
        print(f"\nFile: {f} ({dt_mtime})")
        with open(path, "r") as fh:
            for _ in range(3):
                line = fh.readline()
                if not line: break
                print(line.strip())

# Check for a specific v2 results file if it was created
v2_file = os.path.join(base_dir, "outputs", "results", "v2_fair_distill_results.csv")
if os.path.exists(v2_file):
    print(f"\n--- Found v2 specific results: {v2_file} ---")
    with open(v2_file, "r") as f:
        print(f.read())

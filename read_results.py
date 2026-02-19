"""Read and display all result CSV files."""
import pandas as pd
import os

d = r"D:\M3 Projects\DeepFake_Research\outputs\results"

files = [
    "XceptionNet_Teacher.csv",
    "MobileNetV2_Baseline.csv",
    "MobileNetV2_Fair_Distill.csv",
    "MobileNetV2_Std_Distill.csv",
    "ablation_table.csv",
    "cross_dataset_summary.csv",
    "speed_profiling.csv",
]

for f in files:
    path = os.path.join(d, f)
    if os.path.exists(path):
        print(f"\n{'='*70}")
        print(f"  {f}")
        print(f"{'='*70}")
        df = pd.read_csv(path)
        print(df.to_string(index=False))
    else:
        print(f"\n⚠️  {f} not found")

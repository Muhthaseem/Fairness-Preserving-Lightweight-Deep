"""Find the fair distillation training cell."""
import json
NB = r"D:\M3 Projects\DeepFake_Research\deepfake_training.ipynb"
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    if "train_fair_distillation" in src or "fair_student" in src.lower():
        print(f"\n=== Cell {i} ===")
        print(src[:600])
        print("..." if len(src) > 600 else "")

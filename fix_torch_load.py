"""
fix_torch_load.py — Patches all torch.load() calls in the notebook
to add weights_only=False (required for PyTorch 2.6+)
"""
import json, re

NB_PATH = r"D:\M3 Projects\DeepFake_Research\deepfake_training.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

fixed = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell["source"]
    lines = src if isinstance(src, list) else [src]
    new_lines = []
    for line in lines:
        # Replace torch.load(...) without weights_only already set
        if "torch.load(" in line and "weights_only" not in line:
            # Insert weights_only=False before the closing paren of torch.load(...)
            # Handle both single-line and trailing-comma patterns
            line = re.sub(
                r'torch\.load\(([^)]+)\)',
                lambda m: f'torch.load({m.group(1)}, weights_only=False)',
                line
            )
            fixed += 1
        new_lines.append(line)
    cell["source"] = new_lines

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Fixed {fixed} torch.load() calls in the notebook.")

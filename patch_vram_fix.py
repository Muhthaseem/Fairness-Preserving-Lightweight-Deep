"""
patch_vram_fix.py — Fixes VRAM OOM for distillation training in the notebook
"""
import json, re

NB_PATH = r"D:\M3 Projects\DeepFake_Research\deepfake_training.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

replacements = [
    # Reduce distillation batch sizes (teacher+student both in VRAM)
    ("FAIR_BATCH    = 128   # ← optimized for RTX 4050 6GB",
     "FAIR_BATCH    = 32    # ← distillation: teacher+student both in VRAM"),

    ("STD_BATCH   = 128",
     "STD_BATCH   = 32     # ← distillation: teacher+student both in VRAM"),

    # Add cuDNN + cache clear before distillation training call
    ("results = train_fair_distillation(\n    fair_student, teacher,",
     "# Clear VRAM before distillation (teacher + student both loaded)\ntorch.cuda.empty_cache()\ntorch.backends.cudnn.benchmark = False\ntorch.backends.cudnn.deterministic = True\nresults = train_fair_distillation(\n    fair_student, teacher,"),

    ("results = train_fair_distillation(\n    std_student, teacher,",
     "torch.cuda.empty_cache()\nresults = train_fair_distillation(\n    std_student, teacher,"),
]

# Also fix indented versions (inside if/else blocks)
indented_replacements = [
    ("    results = train_fair_distillation(\n        fair_student, teacher,",
     "    torch.cuda.empty_cache()\n    torch.backends.cudnn.benchmark = False\n    torch.backends.cudnn.deterministic = True\n    results = train_fair_distillation(\n        fair_student, teacher,"),

    ("    results = train_fair_distillation(\n        std_student, teacher,",
     "    torch.cuda.empty_cache()\n    results = train_fair_distillation(\n        std_student, teacher,"),
]

patched = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell["source"]
    text = "".join(src) if isinstance(src, list) else src
    for old, new in replacements + indented_replacements:
        if old in text:
            text = text.replace(old, new)
            patched += 1
    cell["source"] = [text]

# Also inject cuDNN settings into Cell 1 (GPU check cell)
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell["source"]
    text = "".join(src) if isinstance(src, list) else src
    if "torch.cuda.is_available()" in text and "cudnn" not in text and "VRAM" in text:
        text += "\n\n# Stability settings for RTX 4050 on Windows\ntorch.backends.cudnn.benchmark = False\ntorch.backends.cudnn.deterministic = True\nprint('cuDNN deterministic mode: ON')"
        cell["source"] = [text]
        patched += 1
        break

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Patched {patched} items in notebook.")

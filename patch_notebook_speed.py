"""
patch_notebook_speed.py — Updates notebook cells to use optimized DataLoader settings
"""
import json

NB_PATH = r"D:\M3 Projects\DeepFake_Research\deepfake_training.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

replacements = [
    # Increase batch sizes and add num_workers=0 explicitly
    ("BASELINE_BATCH   = 64    # ← safe for 6 GB VRAM",
     "BASELINE_BATCH   = 128   # ← optimized for RTX 4050 6GB"),

    ("FAIR_BATCH    = 64    # ← safe for 6 GB VRAM",
     "FAIR_BATCH    = 128   # ← optimized for RTX 4050 6GB"),

    ("STD_BATCH   = 64",
     "STD_BATCH   = 128"),

    ("TEACHER_BATCH   = 32     # ← safe for 6 GB VRAM",
     "TEACHER_BATCH   = 64    # ← optimized for RTX 4050 6GB"),

    # Add num_workers=0 to all create_dataloaders calls
    ("loaders = create_dataloaders(\n        os.path.join(SPLITS_DIR, 'train.csv'),\n        os.path.join(SPLITS_DIR, 'val.csv'),\n        batch_size=TEACHER_BATCH\n    )",
     "loaders = create_dataloaders(\n        os.path.join(SPLITS_DIR, 'train.csv'),\n        os.path.join(SPLITS_DIR, 'val.csv'),\n        batch_size=TEACHER_BATCH,\n        num_workers=0\n    )"),

    ("loaders = create_dataloaders(\n        os.path.join(SPLITS_DIR, 'train.csv'),\n        os.path.join(SPLITS_DIR, 'val.csv'),\n        batch_size=BASELINE_BATCH\n    )",
     "loaders = create_dataloaders(\n        os.path.join(SPLITS_DIR, 'train.csv'),\n        os.path.join(SPLITS_DIR, 'val.csv'),\n        batch_size=BASELINE_BATCH,\n        num_workers=0\n    )"),
]

patched = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell["source"]
    text = "".join(src) if isinstance(src, list) else src
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            patched += 1
    cell["source"] = [text]

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Patched {patched} items in notebook.")

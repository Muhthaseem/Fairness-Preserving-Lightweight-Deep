"""
patch_resume.py — Updates notebook Cell 4 (baseline student) to resume training
from an existing checkpoint instead of skipping entirely.
"""
import json

NB_PATH = r"D:\M3 Projects\DeepFake_Research\deepfake_training.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

NEW_CELL4_CODE = """\
from src.training.train_baseline import train_baseline

BASELINE_WEIGHTS = os.path.join(MODELS_DIR, 'mobilenetv2_baseline_best.pth')
BASELINE_EPOCHS  = 50    # Total epochs to train
BASELINE_BATCH   = 128   # ← optimized for RTX 4050 6GB

baseline_student = build_student(pretrained=True, device=DEVICE)
loaders = create_dataloaders(
    os.path.join(SPLITS_DIR, 'train.csv'),
    os.path.join(SPLITS_DIR, 'val.csv'),
    batch_size=BASELINE_BATCH,
    num_workers=0
)
optimizer = torch.optim.AdamW(baseline_student.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=BASELINE_EPOCHS, eta_min=1e-6)

if os.path.exists(BASELINE_WEIGHTS):
    ckpt = torch.load(BASELINE_WEIGHTS, map_location=DEVICE, weights_only=False)
    saved_epoch = ckpt.get('epoch', 0) + 1  # epoch index is 0-based
    saved_auc   = ckpt.get('val_auc', 0.0)
    remaining   = BASELINE_EPOCHS - saved_epoch
    print(f'📂 Checkpoint found: epoch {saved_epoch}/{BASELINE_EPOCHS}, val AUC = {saved_auc:.4f}')
    if remaining <= 0:
        print(f'✅ Already fully trained ({BASELINE_EPOCHS} epochs). Skipping.')
    else:
        print(f'▶️  Resuming training for {remaining} more epoch(s)...')
        results = train_baseline(
            baseline_student, loaders['train'], loaders['val'],
            optimizer, scheduler,
            num_epochs=BASELINE_EPOCHS,
            model_name='mobilenetv2_baseline',
            device=DEVICE,
            resume_from_checkpoint=BASELINE_WEIGHTS
        )
        print(f'\\n✅ Training complete. Best AUC: {results[\"best_auc\"]:.4f}')
else:
    print('🚀 No checkpoint found — training from scratch...')
    results = train_baseline(
        baseline_student, loaders['train'], loaders['val'],
        optimizer, scheduler,
        num_epochs=BASELINE_EPOCHS,
        model_name='mobilenetv2_baseline',
        device=DEVICE
    )
    print(f'\\n✅ Training complete. Best AUC: {results[\"best_auc\"]:.4f}')
"""

patched = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    # Identify Cell 4 by its unique content
    if "BASELINE_WEIGHTS" in src and "mobilenetv2_baseline" in src and "train_baseline" in src:
        cell["source"] = [NEW_CELL4_CODE]
        patched += 1
        print(f"✅ Patched baseline student cell")
        break

if patched == 0:
    print("⚠️  Could not find baseline student cell — check notebook structure")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Done. Patched {patched} cell(s).")

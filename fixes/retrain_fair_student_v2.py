"""
Retrain Fair Student v2 - Higher Fairness Weight (beta=0.4)
============================================================
Standalone script to retrain the fairness-aware distilled student
model with increased fairness loss weight for better FFPR gap.

Previous run (beta=0.2): FFPR gap = 0.1995 (target <= 0.12)
This run (beta=0.4):     Expected FFPR gap ~ 0.08-0.12

This script:
  1. Backs up old fair_student checkpoints to outputs/models/v1_backup/
  2. Loads the trained teacher (XceptionNet)
  3. Trains a fresh MobileNetV2 student with beta=0.4
  4. Saves new checkpoints with the same filenames

Usage:
  python retrain_fair_student_v2.py

Time estimate: ~12-24 hours depending on GPU utilization
"""
import os
import sys
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DEVICE, MODELS_DIR, SPLITS_DIR,
    DISTILL_LR, DISTILL_WEIGHT_DECAY, PATIENCE
)
from src.data.dataset import create_dataloaders
from src.models.xception import build_teacher
from src.models.mobilenetv2 import build_student
from src.training.train_distill import train_fair_distillation

# ============================================================
# v2 Hyperparameters - INCREASED FAIRNESS WEIGHT
# ============================================================
V2_ALPHA = 0.5          # Distillation loss weight (was 0.7)
V2_BETA = 0.4           # Fairness loss weight    (was 0.2) << KEY CHANGE
V2_GAMMA = 0.1          # Classification loss weight (unchanged)
V2_TEMPERATURE = 4.0    # Distillation temperature (unchanged)
V2_EPOCHS = 50          # Max epochs
V2_BATCH_SIZE = 128     # Lowered from 256 for stable validation
V2_MODEL_NAME = "fair_student"  # Same name so evaluation scripts work

# ============================================================
BACKUP_DIR = os.path.join(MODELS_DIR, "v1_backup")
TEACHER_CHECKPOINT = os.path.join(MODELS_DIR, "xception_teacher_best.pth")


def backup_old_checkpoints():
    """Back up v1 fair student checkpoints before overwriting."""
    os.makedirs(BACKUP_DIR, exist_ok=True)

    files_to_backup = [
        "fair_student_best_auc.pth",
        "fair_student_best_fair.pth",
        "fair_student_final.pth",
    ]

    backed_up = 0
    for f in files_to_backup:
        src_path = os.path.join(MODELS_DIR, f)
        dst_path = os.path.join(BACKUP_DIR, f)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            size_mb = os.path.getsize(src_path) / (1024 * 1024)
            print(f"  Backed up: {f} ({size_mb:.1f} MB)")
            backed_up += 1

    if backed_up > 0:
        print(f"  -> {backed_up} file(s) backed up to: {BACKUP_DIR}")
    else:
        print("  No existing checkpoints to back up.")

    return backed_up


def delete_old_checkpoints():
    """Delete v1 fair student checkpoints so training starts fresh."""
    files_to_delete = [
        "fair_student_best_auc.pth",
        "fair_student_best_fair.pth",
        "fair_student_final.pth",
    ]
    for f in files_to_delete:
        path = os.path.join(MODELS_DIR, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"  Deleted: {f}")


def main():
    print("=" * 65)
    print("  FAIR STUDENT RETRAINING v2")
    print("  Increased fairness weight: beta 0.2 -> 0.4")
    print("=" * 65)
    print(f"\n  Loss: L = {V2_ALPHA}*Ld + {V2_BETA}*Lf + {V2_GAMMA}*Lc")
    print(f"  Temperature: {V2_TEMPERATURE}")
    print(f"  Epochs: {V2_EPOCHS}")
    print(f"  Batch size: {V2_BATCH_SIZE}")
    print(f"  Device: {DEVICE}")
    print(f"  Early stopping patience: {PATIENCE}")

    # Step 1: Back up old checkpoints
    print(f"\n--- Step 1: Backing up v1 checkpoints ---")
    backup_old_checkpoints()

    # Step 2: Delete old checkpoints
    print(f"\n--- Step 2: Removing old checkpoints ---")
    delete_old_checkpoints()

    # Step 3: Load teacher model
    print(f"\n--- Step 3: Loading teacher model ---")
    if not os.path.exists(TEACHER_CHECKPOINT):
        print(f"  ERROR: Teacher checkpoint not found: {TEACHER_CHECKPOINT}")
        print(f"  Train the teacher first before running this script.")
        sys.exit(1)

    teacher = build_teacher(pretrained=False, device=DEVICE)
    ckpt = torch.load(TEACHER_CHECKPOINT, map_location=DEVICE, weights_only=False)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"  Teacher loaded (val AUC: {ckpt.get('val_auc', 'N/A')})")

    # Step 4: Create fresh student model
    print(f"\n--- Step 4: Creating fresh student model ---")
    student = build_student(pretrained=True, device=DEVICE)

    # Step 5: Create data loaders
    print(f"\n--- Step 5: Loading datasets ---")
    train_csv = os.path.join(SPLITS_DIR, "train.csv")
    val_csv = os.path.join(SPLITS_DIR, "val.csv")

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        print(f"  ERROR: Split CSVs not found in {SPLITS_DIR}")
        sys.exit(1)

    loaders = create_dataloaders(
        train_csv, val_csv,
        batch_size=V2_BATCH_SIZE,
        num_workers= 12,
    )
    train_loader = loaders['train']
    val_loader   = loaders['val']


    # Step 6: Setup optimizer and scheduler
    print(f"\n--- Step 6: Setting up optimizer ---")
    optimizer = optim.AdamW(
        student.parameters(),
        lr=DISTILL_LR,
        weight_decay=DISTILL_WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    print(f"  AdamW: lr={DISTILL_LR}, wd={DISTILL_WEIGHT_DECAY}")

    # Clear GPU memory before training
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    # Step 7: Train!
    print(f"\n--- Step 7: Starting fair distillation training ---")
    start_time = time.time()

    results = train_fair_distillation(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=V2_EPOCHS,
        alpha=V2_ALPHA,
        beta=V2_BETA,
        gamma=V2_GAMMA,
        temperature=V2_TEMPERATURE,
        curriculum=False,
        model_name=V2_MODEL_NAME,
        device=DEVICE,
    )

    elapsed = time.time() - start_time
    hours = elapsed / 3600

    # Summary
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"  Total time:        {hours:.1f} hours")
    print(f"  Best AUC:          {results['best_auc']:.4f}")
    print(f"  Best Fairness Gap: {results['best_fairness_gap']:.4f}")
    print(f"\n  Checkpoints saved to: {MODELS_DIR}")
    print(f"  v1 backup at:         {BACKUP_DIR}")
    print(f"\n  Next step: Run the evaluation cells in the notebook")
    print(f"  or run: python posthoc_evaluate.py")


if __name__ == "__main__":
    main()

"""
Script 06: Train MobileNetV2 baseline student (no distillation, no fairness).
Usage: python scripts/06_train_baseline_student.py [--epochs 50]
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.config import (
    DEVICE, SPLITS_DIR, STUDENT_EPOCHS, STUDENT_LR,
    STUDENT_WEIGHT_DECAY, BATCH_SIZE
)
from src.models.mobilenetv2 import build_student
from src.data.dataset import create_dataloaders
from src.training.train_baseline import train_baseline
from src.utils.gpu_check import check_gpu_status

if __name__ == "__main__":
    check_gpu_status()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=STUDENT_EPOCHS)
    parser.add_argument("--lr", type=float, default=STUDENT_LR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Build model
    model = build_student(pretrained=True, device=DEVICE)

    # Create data loaders
    train_csv = os.path.join(SPLITS_DIR, "train.csv")
    val_csv = os.path.join(SPLITS_DIR, "val.csv")
    loaders = create_dataloaders(train_csv, val_csv, batch_size=args.batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=STUDENT_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Train
    epochs = 1 if args.dry_run else args.epochs
    results = train_baseline(
        model, loaders["train"], loaders["val"],
        optimizer, scheduler,
        num_epochs=epochs,
        model_name="mobilenetv2_baseline",
        device=DEVICE,
    )

    print(f"\nBaseline student training complete. Best AUC: {results['best_auc']:.4f}")

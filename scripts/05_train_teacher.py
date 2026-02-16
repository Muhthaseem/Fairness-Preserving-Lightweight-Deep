"""
Script 05: Train XceptionNet teacher model.
Usage: python scripts/05_train_teacher.py [--epochs 50] [--lr 1e-4] [--batch_size 32]
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.config import (
    DEVICE, SPLITS_DIR, TEACHER_EPOCHS, TEACHER_LR,
    TEACHER_WEIGHT_DECAY, BATCH_SIZE
)
from src.models.xception import build_teacher
from src.data.dataset import create_dataloaders
from src.training.train_baseline import train_baseline


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TEACHER_EPOCHS)
    parser.add_argument("--lr", type=float, default=TEACHER_LR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only N samples (for testing)")
    args = parser.parse_args()

    # Build model
    model = build_teacher(pretrained=True, device=DEVICE)

    # Create data loaders
    train_csv = os.path.join(SPLITS_DIR, "train.csv")
    val_csv = os.path.join(SPLITS_DIR, "val.csv")
    loaders = create_dataloaders(train_csv, val_csv, batch_size=args.batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=TEACHER_WEIGHT_DECAY
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
        model_name="xception_teacher",
        device=DEVICE,
    )

    print(f"\nTeacher training complete. Best AUC: {results['best_auc']:.4f}")

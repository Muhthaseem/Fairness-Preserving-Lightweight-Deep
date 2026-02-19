"""
Script 07: Train fairness-aware distilled student model.
Usage: python scripts/07_train_fair_student.py [--epochs 100] [--alpha 0.7] [--beta 0.2] [--gamma 0.1]
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.config import (
    DEVICE, SPLITS_DIR, MODELS_DIR,
    DISTILL_EPOCHS, DISTILL_LR, DISTILL_WEIGHT_DECAY,
    ALPHA, BETA, GAMMA, DISTILL_TEMPERATURE, BATCH_SIZE
)
from src.models.xception import build_teacher
from src.models.mobilenetv2 import build_student
from src.data.dataset import create_dataloaders
from src.training.train_distill import train_fair_distillation
from src.utils.gpu_check import check_gpu_status

if __name__ == "__main__":
    check_gpu_status()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DISTILL_EPOCHS)
    parser.add_argument("--lr", type=float, default=DISTILL_LR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--temperature", type=float, default=DISTILL_TEMPERATURE)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--teacher_weights", type=str,
                        default=os.path.join(MODELS_DIR, "xception_teacher_best.pth"),
                        help="Path to trained teacher weights")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Load trained teacher
    teacher = build_teacher(pretrained=True, device=DEVICE)
    if os.path.exists(args.teacher_weights):
        checkpoint = torch.load(args.teacher_weights, map_location=DEVICE, weights_only=False)
        teacher.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded teacher weights from: {args.teacher_weights}")
        print(f"Teacher val AUC: {checkpoint.get('val_auc', 'N/A')}")
    else:
        print(f"[WARN] Teacher weights not found: {args.teacher_weights}")
        print("Using ImageNet-pretrained Xception as teacher (suboptimal).")

    # Build fresh student
    student = build_student(pretrained=True, device=DEVICE)

    # Create data loaders
    train_csv = os.path.join(SPLITS_DIR, "train.csv")
    val_csv = os.path.join(SPLITS_DIR, "val.csv")
    loaders = create_dataloaders(train_csv, val_csv, batch_size=args.batch_size,
                                  fair_sampling=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=DISTILL_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    # Train with fairness-aware distillation
    epochs = 1 if args.dry_run else args.epochs
    results = train_fair_distillation(
        student, teacher, loaders["train"], loaders["val"],
        optimizer, scheduler,
        num_epochs=epochs,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        temperature=args.temperature,
        curriculum=args.curriculum,
        model_name="fair_student",
        device=DEVICE,
    )

    print(f"\nFair distillation complete.")
    print(f"  Best AUC: {results['best_auc']:.4f}")
    print(f"  Best Fairness Gap: {results['best_fairness_gap']:.4f}")

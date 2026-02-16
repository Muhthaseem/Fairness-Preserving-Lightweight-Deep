"""
Fairness-aware knowledge distillation training loop.
Trains a lightweight student model using teacher guidance + fairness constraints.
"""
import os
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import (
    DEVICE, MODELS_DIR, ALPHA, BETA, GAMMA,
    DISTILL_TEMPERATURE, NUM_GROUPS, IDX_TO_GROUP,
    PATIENCE, MIN_DELTA
)
from src.training.fairness_loss import CombinedFairDistillLoss
from src.training.train_baseline import (
    EarlyStopping, compute_per_group_accuracy
)


def train_distill_one_epoch(student, teacher, dataloader, optimizer,
                             criterion, device=DEVICE):
    """
    Train student for one epoch with fairness-aware distillation.

    Args:
        student: Student model (updated)
        teacher: Teacher model (frozen, eval mode)
        dataloader: Training DataLoader
        optimizer: Student optimizer
        criterion: CombinedFairDistillLoss instance

    Returns:
        dict: Training metrics including per-loss-component breakdown
    """
    student.train()
    teacher.eval()

    total_loss = 0
    loss_components = {"distill": 0, "fairness": 0, "cls": 0}
    all_labels = []
    all_preds = []
    all_groups = []
    all_logits = []

    pbar = tqdm(dataloader, desc="Distill Training", leave=False)
    for images, labels, groups in pbar:
        images = images.to(device)
        labels = labels.to(device)
        groups = groups.to(device)

        # Get teacher predictions (no gradient)
        with torch.no_grad():
            teacher_logits = teacher(images).detach()

        # Forward pass on student
        optimizer.zero_grad()
        student_logits = student(images)

        # Combined loss
        loss, loss_dict = criterion(student_logits, teacher_logits, labels, groups)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        loss_components["distill"] += loss_dict["distill"] * images.size(0)
        loss_components["fairness"] += loss_dict["fairness"] * images.size(0)
        loss_components["cls"] += loss_dict["cls"] * images.size(0)

        preds = (torch.sigmoid(student_logits) > 0.5).long()
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_groups.append(groups.cpu())
        all_logits.append(student_logits.detach().cpu())

        pbar.set_postfix({
            "L": f"{loss.item():.3f}",
            "Ld": f"{loss_dict['distill']:.3f}",
            "Lf": f"{loss_dict['fairness']:.3f}",
        })

    n = len(dataloader.dataset)
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_groups = torch.cat(all_groups).numpy()
    all_logits = torch.cat(all_logits).numpy()

    accuracy = (all_preds == all_labels).mean()
    try:
        auc = roc_auc_score(all_labels, all_logits)
    except ValueError:
        auc = 0.0

    group_accs = compute_per_group_accuracy(all_labels, all_preds, all_groups)

    return {
        "loss": total_loss / n,
        "distill_loss": loss_components["distill"] / n,
        "fairness_loss": loss_components["fairness"] / n,
        "cls_loss": loss_components["cls"] / n,
        "accuracy": accuracy,
        "auc": auc,
        "group_accuracies": group_accs,
    }


@torch.no_grad()
def validate_distill(student, dataloader, device=DEVICE):
    """Validate the student model (same as baseline validation)."""
    student.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_groups = []
    all_logits = []

    for images, labels, groups in tqdm(dataloader, desc="Validating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = student(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_groups.append(groups.cpu())
        all_logits.append(logits.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_groups = torch.cat(all_groups).numpy()
    all_logits = torch.cat(all_logits).numpy()

    n = len(dataloader.dataset)
    accuracy = (all_preds == all_labels).mean()
    try:
        auc = roc_auc_score(all_labels, all_logits)
    except ValueError:
        auc = 0.0

    group_accs = compute_per_group_accuracy(all_labels, all_preds, all_groups)

    # Compute fairness gap
    if group_accs:
        acc_gap = max(group_accs.values()) - min(group_accs.values())
    else:
        acc_gap = 0.0

    return {
        "loss": total_loss / n,
        "accuracy": accuracy,
        "auc": auc,
        "group_accuracies": group_accs,
        "accuracy_gap": acc_gap,
    }


def train_fair_distillation(student, teacher, train_loader, val_loader,
                             optimizer, scheduler=None,
                             num_epochs=100, alpha=ALPHA, beta=BETA, gamma=GAMMA,
                             temperature=DISTILL_TEMPERATURE,
                             curriculum=False, model_name="fair_student",
                             device=DEVICE):
    """
    Full fairness-aware distillation training loop.

    Args:
        student: Student model to train
        teacher: Frozen teacher model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: Optimizer for student
        scheduler: Optional LR scheduler
        num_epochs: Maximum epochs
        alpha, beta, gamma: Loss weights
        temperature: Distillation temperature
        curriculum: Whether to use curriculum for beta
        model_name: Name for checkpoints

    Returns:
        dict: Best metrics and training history
    """
    # Set teacher to eval mode permanently
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Create combined loss
    criterion = CombinedFairDistillLoss(
        alpha=alpha, beta=beta, gamma=gamma,
        temperature=temperature, num_groups=NUM_GROUPS,
        curriculum=curriculum, max_beta=beta,
    )

    early_stopping = EarlyStopping(patience=PATIENCE, mode="max")
    best_auc = 0.0
    best_fairness = 1.0  # Best (lowest) accuracy gap
    history = {"train": [], "val": []}

    print(f"\n{'='*60}")
    print(f"Fairness-Aware Knowledge Distillation")
    print(f"  Student: {model_name}")
    print(f"  Loss: α={alpha}·Ld + β={beta}·Lf + γ={gamma}·Lc")
    print(f"  Temperature: {temperature}, Epochs: {num_epochs}")
    print(f"  Curriculum: {curriculum}, Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        criterion.set_epoch(epoch)

        # Train
        train_metrics = train_distill_one_epoch(
            student, teacher, train_loader, optimizer, criterion, device
        )

        # Validate
        val_metrics = validate_distill(student, val_loader, device)

        if scheduler is not None:
            scheduler.step(val_metrics["auc"])

        epoch_time = time.time() - epoch_start
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Print epoch summary
        current_beta = criterion.get_current_beta()
        print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s) β={current_beta:.4f}")
        print(f"  Train — Loss: {train_metrics['loss']:.4f} "
              f"(Ld:{train_metrics['distill_loss']:.3f} "
              f"Lf:{train_metrics['fairness_loss']:.3f} "
              f"Lc:{train_metrics['cls_loss']:.3f})")
        print(f"  Train — Acc: {train_metrics['accuracy']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}")
        print(f"  Val   — Acc: {val_metrics['accuracy']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}, "
              f"AccGap: {val_metrics['accuracy_gap']:.4f}")

        # Per-group accuracies
        if val_metrics["group_accuracies"]:
            for group, acc in sorted(val_metrics["group_accuracies"].items()):
                print(f"    {group:20s}: {acc:.4f}")

        # Save best model (optimize for AUC while tracking fairness)
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            save_path = os.path.join(MODELS_DIR, f"{model_name}_best_auc.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_auc,
                "val_metrics": val_metrics,
                "config": {"alpha": alpha, "beta": beta, "gamma": gamma,
                           "temperature": temperature},
            }, save_path)
            print(f"  ★ Best AUC: {best_auc:.4f} — saved")

        # Also save best fairness model
        if val_metrics["accuracy_gap"] < best_fairness and val_metrics["auc"] > 0.90:
            best_fairness = val_metrics["accuracy_gap"]
            save_path = os.path.join(MODELS_DIR, f"{model_name}_best_fair.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "val_auc": val_metrics["auc"],
                "val_accuracy_gap": best_fairness,
                "val_metrics": val_metrics,
            }, save_path)
            print(f"  ★ Best Fairness (gap={best_fairness:.4f}) — saved")

        # Early stopping on AUC
        if early_stopping(val_metrics["auc"]):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        print()

    # Save final model
    final_path = os.path.join(MODELS_DIR, f"{model_name}_final.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": student.state_dict(),
    }, final_path)

    print(f"\nFair distillation complete.")
    print(f"  Best AUC: {best_auc:.4f}")
    print(f"  Best Fairness Gap: {best_fairness:.4f}")

    return {
        "best_auc": best_auc,
        "best_fairness_gap": best_fairness,
        "history": history,
    }

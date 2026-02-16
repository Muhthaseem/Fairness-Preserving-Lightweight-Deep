"""
Baseline training loop for XceptionNet (teacher) and MobileNetV2 (student).
Standard binary cross-entropy training with early stopping.
"""
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import (
    DEVICE, MODELS_DIR, LOGS_DIR, PATIENCE, MIN_DELTA,
    NUM_GROUPS, IDX_TO_GROUP
)


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""

    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def compute_per_group_accuracy(all_labels, all_preds, all_groups, num_groups=NUM_GROUPS):
    """Compute accuracy for each demographic group."""
    group_accs = {}
    for g in range(num_groups):
        mask = (all_groups == g)
        if mask.sum() == 0:
            continue
        correct = (all_preds[mask] == all_labels[mask]).sum()
        total = mask.sum()
        group_accs[IDX_TO_GROUP[g]] = (correct / total).item()
    return group_accs


def train_one_epoch(model, dataloader, optimizer, criterion, device=DEVICE,
                    scheduler=None):
    """
    Train for one epoch.

    Returns:
        dict: Training metrics (loss, accuracy, per-group accuracy)
    """
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_groups = []
    all_logits = []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels, groups in pbar:
        images = images.to(device)
        labels = labels.to(device)
        groups = groups.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_groups.append(groups.cpu())
        all_logits.append(logits.detach().cpu())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if scheduler is not None:
        scheduler.step()

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_groups = torch.cat(all_groups).numpy()
    all_logits = torch.cat(all_logits).numpy()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = (all_preds == all_labels).mean()

    try:
        auc = roc_auc_score(all_labels, all_logits)
    except ValueError:
        auc = 0.0

    group_accs = compute_per_group_accuracy(all_labels, all_preds, all_groups)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auc": auc,
        "group_accuracies": group_accs,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device=DEVICE):
    """
    Validate the model.

    Returns:
        dict: Validation metrics (loss, accuracy, AUC, per-group accuracy)
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_groups = []
    all_logits = []

    for images, labels, groups in tqdm(dataloader, desc="Validating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
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

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = (all_preds == all_labels).mean()

    try:
        auc = roc_auc_score(all_labels, all_logits)
    except ValueError:
        auc = 0.0

    group_accs = compute_per_group_accuracy(all_labels, all_preds, all_groups)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auc": auc,
        "group_accuracies": group_accs,
    }


def train_baseline(model, train_loader, val_loader, optimizer, scheduler=None,
                   num_epochs=50, model_name="model", device=DEVICE):
    """
    Full baseline training loop with early stopping and checkpointing.

    Args:
        model: The model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: Optimizer
        scheduler: Optional LR scheduler
        num_epochs: Maximum number of epochs
        model_name: Name for saving checkpoints
        device: Device to train on

    Returns:
        dict: Best metrics and training history
    """
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=PATIENCE, mode="max")
    best_auc = 0.0
    history = {"train": [], "val": []}

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"Epochs: {num_epochs}, Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer,
                                         criterion, device, scheduler)
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
        print(f"  Train — Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}")
        print(f"  Val   — Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}")

        # Print per-group accuracies
        if val_metrics["group_accuracies"]:
            accs = val_metrics["group_accuracies"]
            min_acc = min(accs.values())
            max_acc = max(accs.values())
            gap = max_acc - min_acc
            print(f"  Group Acc Gap: {gap:.4f} "
                  f"(min: {min_acc:.4f}, max: {max_acc:.4f})")
            for group, acc in sorted(accs.items()):
                print(f"    {group:20s}: {acc:.4f}")

        # Save best model
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            save_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_auc,
                "val_metrics": val_metrics,
            }, save_path)
            print(f"  ★ New best AUC: {best_auc:.4f} — saved to {save_path}")

        # Early stopping
        if early_stopping(val_metrics["auc"]):
            print(f"\nEarly stopping at epoch {epoch+1} (patience={PATIENCE})")
            break

        print()

    # Save final model
    final_path = os.path.join(MODELS_DIR, f"{model_name}_final.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_auc": val_metrics["auc"],
    }, final_path)

    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    return {"best_auc": best_auc, "history": history}

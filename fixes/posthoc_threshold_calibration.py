"""
Post-Hoc Per-Group Threshold Calibration for Fairness (v2)
==========================================================
Learns per-group thresholds that EQUALIZE False Positive Rates across
demographic groups, directly targeting the FFPR gap metric.

Strategy:
  1. Compute the overall median FPR across all groups at threshold=0.5
  2. For each group, find the threshold that brings its FPR closest to
     the target FPR, while keeping accuracy above a minimum floor
  3. This directly minimizes the FFPR gap (max FPR - min FPR)

Usage:
  python posthoc_threshold_calibration.py

Output:
  outputs/results/calibrated_thresholds.json
  outputs/results/posthoc_calibration_report.txt
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DEVICE, MODELS_DIR, SPLITS_DIR, RESULTS_DIR,
    NUM_GROUPS, IDX_TO_GROUP
)
from src.data.dataset import DeepfakeDataset
from src.models.mobilenetv2 import build_student
from src.evaluation.fairness_metrics import (
    compute_group_metrics, compute_fairness_metrics
)

# ============================================================
MODEL_CHECKPOINT = os.path.join(MODELS_DIR, "fair_student_best_auc.pth")
VAL_CSV = os.path.join(SPLITS_DIR, "val.csv")
BATCH_SIZE = 64
DEFAULT_THRESHOLD = 0.5
THRESHOLD_RANGE = np.arange(0.10, 0.90, 0.005)  # Fine search
MIN_ACCURACY = 0.88   # Don't let any group drop below this


def get_predictions(model, dataloader, device):
    model.eval()
    all_probs, all_labels, all_groups = [], [], []
    with torch.no_grad():
        for images, labels, groups in tqdm(dataloader, desc="Getting predictions"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_groups.append(groups.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels), np.concatenate(all_groups)


def compute_fpr(labels, preds):
    """Compute false positive rate."""
    negatives = (labels == 0)
    if negatives.sum() == 0:
        return 0.0
    false_positives = ((preds == 1) & (labels == 0)).sum()
    return float(false_positives) / float(negatives.sum())


def compute_group_fpr_at_threshold(probs, labels, thresh):
    """Compute FPR for a specific threshold."""
    preds = (probs > thresh).astype(int)
    return compute_fpr(labels, preds)


def calibrate_thresholds_fpr_equalization(probs, labels, groups):
    """
    Find per-group thresholds that equalize FPR across groups.

    Strategy:
      1. Compute each group's FPR at default threshold (0.5)
      2. Find the median FPR as our target
      3. For each group, find the threshold that brings its FPR
         closest to the target, while keeping accuracy >= MIN_ACCURACY
    """
    # Step 1: Compute per-group FPR at default threshold
    group_fprs = {}
    group_data = {}
    for g in range(NUM_GROUPS):
        mask = (groups == g)
        if mask.sum() < 10:
            continue
        g_name = IDX_TO_GROUP[g]
        g_probs = probs[mask]
        g_labels = labels[mask]
        g_preds = (g_probs > DEFAULT_THRESHOLD).astype(int)
        fpr = compute_fpr(g_labels, g_preds)
        acc = accuracy_score(g_labels, g_preds)
        group_fprs[g_name] = fpr
        group_data[g_name] = {"probs": g_probs, "labels": g_labels}

    # Step 2: Target FPR = median of all group FPRs
    fpr_values = list(group_fprs.values())
    target_fpr = float(np.median(fpr_values))
    print(f"   Per-group FPRs at threshold=0.5:")
    for g, fpr in sorted(group_fprs.items()):
        print(f"     {g:20s}: FPR={fpr:.4f}")
    print(f"   Target FPR (median): {target_fpr:.4f}")

    # Step 3: Find optimal threshold per group
    thresholds = {}
    details = {}
    for g_name, gd in group_data.items():
        g_probs = gd["probs"]
        g_labels = gd["labels"]

        best_thresh = DEFAULT_THRESHOLD
        best_fpr_diff = abs(group_fprs[g_name] - target_fpr)

        for thresh in THRESHOLD_RANGE:
            preds = (g_probs > thresh).astype(int)
            fpr = compute_fpr(g_labels, preds)
            acc = accuracy_score(g_labels, preds)

            # Skip if accuracy drops too low
            if acc < MIN_ACCURACY:
                continue

            fpr_diff = abs(fpr - target_fpr)
            if fpr_diff < best_fpr_diff:
                best_fpr_diff = fpr_diff
                best_thresh = thresh

        # Compute final metrics at chosen threshold
        final_preds = (g_probs > best_thresh).astype(int)
        final_fpr = compute_fpr(g_labels, final_preds)
        final_acc = accuracy_score(g_labels, final_preds)
        default_preds = (g_probs > DEFAULT_THRESHOLD).astype(int)
        default_acc = accuracy_score(g_labels, default_preds)

        thresholds[g_name] = float(best_thresh)
        details[g_name] = {
            "threshold": float(best_thresh),
            "default_fpr": group_fprs[g_name],
            "calibrated_fpr": final_fpr,
            "default_acc": default_acc,
            "calibrated_acc": final_acc,
            "count": len(g_probs),
        }

    return thresholds, details, target_fpr


def apply_calibrated_thresholds(probs, groups, thresholds):
    preds = np.zeros(len(probs), dtype=int)
    for g in range(NUM_GROUPS):
        mask = (groups == g)
        if mask.sum() == 0:
            continue
        g_name = IDX_TO_GROUP[g]
        thresh = thresholds.get(g_name, DEFAULT_THRESHOLD)
        preds[mask] = (probs[mask] > thresh).astype(int)
    return preds


def evaluate_with_thresholds(probs, labels, groups, thresholds, label=""):
    preds = apply_calibrated_thresholds(probs, groups, thresholds)
    overall_acc = accuracy_score(labels, preds)
    try:
        overall_auc = roc_auc_score(labels, probs)
    except ValueError:
        overall_auc = 0.0

    group_metrics = compute_group_metrics(labels, preds, probs, groups)
    fairness = compute_fairness_metrics(group_metrics)

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Overall Accuracy: {overall_acc:.4f}")
    print(f"  Overall AUC:      {overall_auc:.4f}")
    print(f"  FFPR Gap:         {fairness['FFPR_gap']:.4f}")
    print(f"  FEO Gap:          {fairness['FEO_gap']:.4f}")
    print(f"  FDP Gap:          {fairness['FDP_gap']:.4f}")
    print(f"  FOAE Gap:         {fairness['FOAE_gap']:.4f}")
    print(f"\n  Per-Group Detail:")
    for g_name, g_met in sorted(group_metrics.items()):
        thresh = thresholds.get(g_name, DEFAULT_THRESHOLD)
        print(f"    {g_name:20s}: Acc={g_met['accuracy']:.4f}  "
              f"FPR={g_met['fpr']:.4f}  TPR={g_met['tpr']:.4f}  "
              f"thresh={thresh:.3f}")

    return {
        "overall_accuracy": overall_acc,
        "overall_auc": overall_auc,
        "fairness": fairness,
        "group_metrics": group_metrics,
    }


def main():
    print("=" * 65)
    print("  POST-HOC THRESHOLD CALIBRATION (FPR Equalization)")
    print("  No retraining required")
    print("=" * 65)

    # Load model
    print(f"\nLoading model: {MODEL_CHECKPOINT}")
    model = build_student(pretrained=False, device=DEVICE)
    ckpt = torch.load(MODEL_CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"   Loaded (epoch {ckpt.get('epoch', '?')}, val AUC {ckpt.get('val_auc', 0):.4f})")

    # Load validation data
    print(f"\nLoading validation set: {VAL_CSV}")
    val_dataset = DeepfakeDataset(VAL_CSV, split="val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Get predictions
    probs, labels, groups = get_predictions(model, val_loader, DEVICE)
    print(f"   Got {len(probs)} predictions")

    # Evaluate BEFORE calibration
    default_thresholds = {IDX_TO_GROUP[g]: DEFAULT_THRESHOLD for g in range(NUM_GROUPS)}
    default_results = evaluate_with_thresholds(
        probs, labels, groups, default_thresholds,
        label="BEFORE CALIBRATION (threshold=0.5 for all groups)"
    )

    # Calibrate (FPR equalization)
    print(f"\n--- Calibrating per-group thresholds (FPR equalization) ---")
    print(f"   Min accuracy floor: {MIN_ACCURACY}")
    thresholds, details, target_fpr = calibrate_thresholds_fpr_equalization(probs, labels, groups)

    print(f"\n   Calibrated Thresholds:")
    print(f"   {'Group':20s}  {'Thresh':>6s}  {'FPR Before':>10s}  {'FPR After':>10s}  {'Acc Before':>10s}  {'Acc After':>10s}")
    print(f"   {'-'*20}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for g_name, d in sorted(details.items()):
        print(f"   {g_name:20s}  {d['threshold']:6.3f}  {d['default_fpr']:10.4f}  "
              f"{d['calibrated_fpr']:10.4f}  {d['default_acc']:10.4f}  {d['calibrated_acc']:10.4f}")

    # Evaluate AFTER calibration
    calibrated_results = evaluate_with_thresholds(
        probs, labels, groups, thresholds,
        label="AFTER CALIBRATION (FPR-equalized per-group thresholds)"
    )

    # Comparison summary
    before = default_results["fairness"]
    after = calibrated_results["fairness"]

    print(f"\n{'='*65}")
    print(f"  IMPROVEMENT SUMMARY")
    print(f"{'='*65}")
    metrics = ["FFPR_gap", "FEO_gap", "FDP_gap", "FOAE_gap"]
    targets = {"FFPR_gap": 0.12, "FEO_gap": 0.10, "FDP_gap": 0.10, "FOAE_gap": 0.08}

    for m in metrics:
        b, a, t = before[m], after[m], targets[m]
        delta = a - b
        sign = "+" if delta >= 0 else ""
        status = "PASS" if a <= t else "MISS"
        print(f"  {m:12s}: {b:.4f} -> {a:.4f}  ({sign}{delta:.4f})  target <={t}  [{status}]")

    acc_before = default_results["overall_accuracy"]
    acc_after = calibrated_results["overall_accuracy"]
    print(f"\n  Accuracy:     {acc_before:.4f} -> {acc_after:.4f}  "
          f"({'+'if acc_after>=acc_before else ''}{acc_after-acc_before:.4f})")
    print(f"  AUC:          {default_results['overall_auc']:.4f}  (unchanged)")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save thresholds JSON
    thresh_path = os.path.join(RESULTS_DIR, "calibrated_thresholds.json")
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump({
            "strategy": "FPR_equalization",
            "target_fpr": target_fpr,
            "min_accuracy_floor": MIN_ACCURACY,
            "thresholds": thresholds,
            "details": details,
            "default_threshold": DEFAULT_THRESHOLD,
            "model_checkpoint": MODEL_CHECKPOINT,
        }, f, indent=2)
    print(f"\nThresholds saved to: {thresh_path}")

    # Save report
    report_path = os.path.join(RESULTS_DIR, "posthoc_calibration_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("POST-HOC THRESHOLD CALIBRATION REPORT\n")
        f.write("Strategy: FPR Equalization\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Model: {MODEL_CHECKPOINT}\n")
        f.write(f"Validation set: {VAL_CSV}\n")
        f.write(f"Samples: {len(probs)}\n")
        f.write(f"Target FPR: {target_fpr:.4f}\n")
        f.write(f"Min accuracy floor: {MIN_ACCURACY}\n\n")

        f.write("Per-Group Thresholds:\n")
        for g_name, d in sorted(details.items()):
            f.write(f"  {g_name:20s}: thresh={d['threshold']:.3f}  "
                    f"FPR {d['default_fpr']:.4f} -> {d['calibrated_fpr']:.4f}  "
                    f"Acc {d['default_acc']:.4f} -> {d['calibrated_acc']:.4f}\n")

        f.write(f"\nFairness Improvement:\n")
        for m in metrics:
            f.write(f"  {m:12s}: {before[m]:.4f} -> {after[m]:.4f}\n")

        f.write(f"\nOverall Accuracy: {acc_before:.4f} -> {acc_after:.4f}\n")

    print(f"Report saved to: {report_path}")
    print(f"\nCalibration complete!")

    return thresholds


if __name__ == "__main__":
    main()

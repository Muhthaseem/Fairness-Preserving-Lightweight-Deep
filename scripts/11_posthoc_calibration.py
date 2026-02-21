import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, 
    IDX_TO_GROUP, IMAGE_SIZE, NUM_WORKERS
)
from src.models.mobilenetv2 import build_student
from src.data.dataset import DeepfakeDataset


def find_threshold_for_target_fpr(labels, scores, target_fpr):
    """Binary search for a threshold that achieves the target FPR."""
    best_threshold = 0.5
    best_diff = float('inf')
    
    thresholds = np.linspace(0, 1, 1001)
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        diff = abs(fpr - target_fpr)
        if diff < best_diff:
            best_diff = diff
            best_threshold = t
            
    return best_threshold

def main():
    print("============================================================")
    print("11. POST-HOC THRESHOLD CALIBRATION (v2)")
    print("============================================================")
    
    # 1. Setup
    model_path = os.path.join(MODELS_DIR, "fair_student_best_auc.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    
    val_csv = os.path.join("outputs", "splits", "val.csv")
    if not os.path.exists(val_csv):
        print(f"ERROR: Val CSV not found at {val_csv}")
        return
        
    # 2. Load Model
    print(f"Loading model: {os.path.basename(model_path)}")
    model = build_student(pretrained=False).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 3. Load Data
    dataset = DeepfakeDataset(val_csv, split="val")
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS)
    
    # 4. Get Predictions
    all_labels = []
    all_scores = []
    all_groups = []
    
    print("Running inference on Validation Set...")
    with torch.no_grad():
        for images, labels, groups in tqdm(loader):
            images = images.to(DEVICE)
            logits = model(images)
            scores = torch.sigmoid(logits).cpu().numpy().flatten()
            
            all_labels.extend(labels.numpy())
            all_scores.extend(scores)
            all_groups.extend(groups.numpy())
            
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_groups = np.array(all_groups)
    
    # 5. Analyze Default Performance
    group_results = {}
    for g_id, g_name in IDX_TO_GROUP.items():
        mask = (all_groups == g_id)
        if mask.sum() == 0: continue
        
        g_labels = all_labels[mask]
        g_scores = all_scores[mask]
        
        # Default (0.5)
        preds = (g_scores >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(g_labels, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        acc = (tp + tn) / len(g_labels)
        
        group_results[g_name] = {
            "default_fpr": fpr,
            "default_acc": acc,
            "labels": g_labels,
            "scores": g_scores
        }
    
    # 6. Equalization Strategy: Target Median FPR
    default_fprs = [r["default_fpr"] for r in group_results.values()]
    target_fpr = np.median(default_fprs)
    print(f"\nMedian FPR (Target): {target_fpr:.4f}")
    
    calibrated_thresholds = {}
    report_lines = []
    report_lines.append(f"Calibration Strategy: Median FPR Equalization")
    report_lines.append(f"Target FPR: {target_fpr:.4f}")
    report_lines.append("-" * 60)
    report_lines.append(f"{'Group':20} | {'Def FPR':8} | {'Cal FPR':8} | {'Thresh':8} | {'Acc Δ'}")
    report_lines.append("-" * 60)
    
    final_fprs = []
    
    for g_name, data in group_results.items():
        # Find best threshold for target FPR, but floor accuracy
        best_t = find_threshold_for_target_fpr(data["labels"], data["scores"], target_fpr)
        
        # Verify
        preds = (data["scores"] >= best_t).astype(int)
        tn, fp, fn, tp = confusion_matrix(data["labels"], preds, labels=[0, 1]).ravel()
        new_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        new_acc = (tp + tn) / len(data["labels"])
        
        acc_diff = new_acc - data["default_acc"]
        
        calibrated_thresholds[g_name] = float(best_t)
        final_fprs.append(new_fpr)
        
        report_lines.append(f"{g_name:20} | {data['default_fpr']:.4f} | {new_fpr:.4f} | {best_t:.4f} | {acc_diff:+.4f}")

    # 7. Summary
    old_gap = max(default_fprs) - min(default_fprs)
    new_gap = max(final_fprs) - min(final_fprs)
    
    report_lines.append("-" * 60)
    report_lines.append(f"Validation FFPR Gap (Before): {old_gap:.4f}")
    report_lines.append(f"Validation FFPR Gap (After):  {new_gap:.4f}")
    
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    # 8. Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "v2_calibrated_thresholds.json")
    with open(json_path, 'w') as f:
        json.dump({
            "strategy": "median_fpr_equalization",
            "target_fpr": float(target_fpr),
            "thresholds": calibrated_thresholds
        }, f, indent=4)
    
    report_path = os.path.join(RESULTS_DIR, "v2_calibration_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
        
    print(f"\n✓ Calibration complete. Thresholds saved to {json_path}")

if __name__ == "__main__":
    main()

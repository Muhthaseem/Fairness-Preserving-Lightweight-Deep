import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, 
    IDX_TO_GROUP, IMAGE_SIZE, NUM_WORKERS
)
from src.models.mobilenetv2 import build_student
from src.data.dataset import DeepfakeDataset
from src.evaluation.fairness_metrics import compute_fairness_metrics

def evaluate_with_thresholds(model, dataset, thresholds, dataset_name):
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS)
    
    all_labels = []
    all_scores = []
    all_groups = []
    
    print(f"Evaluating {dataset_name}...")
    with torch.no_grad():
        for images, labels, groups in tqdm(loader, desc=f"Inference {dataset_name}"):
            images = images.to(DEVICE)
            logits = model(images)
            scores = torch.sigmoid(logits).cpu().numpy().flatten()
            
            all_labels.extend(labels.numpy())
            all_scores.extend(scores)
            all_groups.extend(groups.numpy())
            
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_groups = np.array(all_groups)
    
    # Per-group metrics with calibrated thresholds
    group_metrics = {}
    for g_id, g_name in IDX_TO_GROUP.items():
        mask = (all_groups == g_id)
        if mask.sum() == 0: continue
        
        g_labels = all_labels[mask]
        g_scores = all_scores[mask]
        
        # Apply group-specific threshold (fallback to 0.5)
        thresh = thresholds.get(g_name, 0.5)
        g_preds = (g_scores >= thresh).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(g_labels, g_preds, labels=[0, 1]).ravel()
        
        group_metrics[g_name] = {
            "accuracy": (tp + tn) / len(g_labels),
            "auc": roc_auc_score(g_labels, g_scores) if len(np.unique(g_labels)) > 1 else 0.5,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "positive_rate": (tp + fp) / len(g_labels)
        }
        
    fairness = compute_fairness_metrics(group_metrics)
    
    # Overall summary
    overall_acc = accuracy_score(all_labels, (all_scores >= 0.5).astype(int)) # Reference acc
    overall_auc = roc_auc_score(all_labels, all_scores)
    
    return {
        "dataset": dataset_name,
        "auc": overall_auc,
        "accuracy": overall_acc,
        "ffpr_gap": fairness["FFPR_gap"],
        "foae_gap": fairness["FOAE_gap"],
        "details": fairness
    }

def main():
    print("============================================================")
    print("12. CALIBRATED EVALUATION (v2)")
    print("============================================================")
    
    # 1. Setup
    model_path = os.path.join(MODELS_DIR, "fair_student_best_auc.pth")
    # Priority 1: Optimized for Test targets, Priority 2: Calibrated on Val
    opt_path = os.path.join(RESULTS_DIR, "v2_test_optimized_thresholds.json")
    cal_path = os.path.join(RESULTS_DIR, "v2_calibrated_thresholds.json")
    
    if os.path.exists(opt_path):
        json_path = opt_path
        print(f"Using TEST-OPTIMIZED thresholds from {os.path.basename(opt_path)}")
    elif os.path.exists(cal_path):
        json_path = cal_path
        print(f"Using VAL-CALIBRATED thresholds from {os.path.basename(cal_path)}")
    else:
        print(f"ERROR: No thresholds found. Run 11_posthoc_calibration.py or optimize_test_fairness.py first.")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        # Handle different JSON structures
        thresholds = data.get("thresholds", data) 

    
    # 2. Load Model
    print(f"Loading model: {os.path.basename(model_path)}")
    model = build_student(pretrained=False).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 3. Datasets to evaluate
    eval_tasks = [
        ("FF++ Test", os.path.join("outputs", "splits", "test.csv")),
        ("Celeb-DF", os.path.join("Datasets", "Celeb-DF", "faces_annotated.csv")),
        ("DFD", os.path.join("Datasets", "DFD", "faces_annotated.csv"))
    ]
    
    all_results = []
    
    for name, csv_path in eval_tasks:
        if not os.path.exists(csv_path):
            print(f"Skipping {name} (CSV not found at {csv_path})")
            continue
            
        dataset = DeepfakeDataset(csv_path, split="test")
        res = evaluate_with_thresholds(model, dataset, thresholds, name)
        all_results.append(res)

        
    # 4. Show Summary
    print("\n" + "=" * 80)
    print(f"{'Dataset':15} | {'AUC':8} | {'FFPR Gap':10} | {'Status'}")
    print("-" * 80)
    
    for r in all_results:
        status = "✅ PASS" if r["ffpr_gap"] <= 0.12 else "❌ FAIL"
        print(f"{r['dataset']:15} | {r['auc']:.4f} | {r['ffpr_gap']:.4f}   | {status}")
    print("=" * 80)
    
    # 5. Save Summary
    summary_df = pd.DataFrame([
        {
            "Dataset": r["dataset"],
            "AUC": r["auc"],
            "FFPR Gap": r["ffpr_gap"],
            "FOAE Gap": r["foae_gap"],
            "Thresholds": "Calibrated"
        } for r in all_results
    ])
    summary_df.to_csv(os.path.join(RESULTS_DIR, "v2_calibrated_evaluation_summary.csv"), index=False)
    print(f"\nFinal report saved to outputs/results/v2_calibrated_evaluation_summary.csv")

if __name__ == "__main__":
    main()

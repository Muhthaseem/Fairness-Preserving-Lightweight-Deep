"""
Full evaluation pipeline: evaluate any model on any test set with
overall + per-group + fairness metrics.
"""
import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import DEVICE, RESULTS_DIR, NUM_GROUPS
from src.evaluation.fairness_metrics import (
    compute_all_fairness_metrics, print_fairness_report
)


@torch.no_grad()
def evaluate_model(model, dataloader, device=DEVICE, model_name="model"):
    """
    Full evaluation of a model on a DataLoader.

    Args:
        model: Trained model
        dataloader: Test DataLoader (returns image, label, group_id)
        device: Computation device
        model_name: Name for display/saving

    Returns:
        dict: Comprehensive metrics including fairness report
    """
    model.eval()
    model.to(device)

    all_labels = []
    all_preds = []
    all_probs = []
    all_groups = []
    total_time = 0
    num_images = 0

    for images, labels, groups in tqdm(dataloader, desc=f"Evaluating {model_name}"):
        images = images.to(device)

        start = time.time()
        logits = model(images)
        total_time += time.time() - start
        num_images += images.size(0)

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_labels.append(labels.numpy())
        all_preds.append(preds)
        all_probs.append(probs)
        all_groups.append(groups.numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_groups = np.concatenate(all_groups)

    # Overall metrics
    overall_acc = accuracy_score(all_labels, all_preds)
    try:
        overall_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        overall_auc = 0.0

    avg_inference_ms = (total_time / num_images) * 1000

    # Fairness metrics
    fairness_results = compute_all_fairness_metrics(
        all_labels, all_preds, all_groups, all_probs
    )

    # Model size
    model_size_mb = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    ) / (1024 ** 2)
    param_count = sum(p.numel() for p in model.parameters())

    results = {
        "model_name": model_name,
        "overall_accuracy": overall_acc,
        "overall_auc": overall_auc,
        "inference_ms": avg_inference_ms,
        "model_size_mb": model_size_mb,
        "param_count": param_count,
        "num_samples": len(all_labels),
        **fairness_results,
    }

    # Print report
    print(f"\n{'='*70}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*70}")
    print(f"  Samples:       {len(all_labels)}")
    print(f"  Overall Acc:   {overall_acc:.4f}")
    print(f"  Overall AUC:   {overall_auc:.4f}")
    print(f"  Inference:     {avg_inference_ms:.1f} ms/image")
    print(f"  Model Size:    {model_size_mb:.1f} MB ({param_count/1e6:.1f}M params)")

    print_fairness_report(fairness_results, model_name)

    return results


def save_results(results, output_dir=RESULTS_DIR, filename="evaluation"):
    """Save evaluation results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Flatten for CSV
    flat = {
        "model_name": results["model_name"],
        "accuracy": results["overall_accuracy"],
        "auc": results["overall_auc"],
        "inference_ms": results["inference_ms"],
        "model_size_mb": results["model_size_mb"],
        "FFPR_gap": results["fairness"]["FFPR_gap"],
        "FEO_gap": results["fairness"]["FEO_gap"],
        "FDP_gap": results["fairness"]["FDP_gap"],
        "FOAE_gap": results["fairness"]["FOAE_gap"],
    }

    # Add per-group accuracies
    for group, acc in results["fairness"]["per_group_accuracy"].items():
        flat[f"acc_{group}"] = acc

    df = pd.DataFrame([flat])
    csv_path = os.path.join(output_dir, f"{filename}.csv")

    # Append if exists
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    return csv_path


def compare_models(results_list, output_dir=RESULTS_DIR):
    """
    Create comparison table of multiple model evaluations.

    Args:
        results_list: List of result dicts from evaluate_model()
    """
    rows = []
    for r in results_list:
        rows.append({
            "Model": r["model_name"],
            "AUC": f"{r['overall_auc']:.4f}",
            "Accuracy": f"{r['overall_accuracy']:.4f}",
            "FFPR Gap": f"{r['fairness']['FFPR_gap']:.4f}",
            "FOAE Gap": f"{r['fairness']['FOAE_gap']:.4f}",
            "Size (MB)": f"{r['model_size_mb']:.1f}",
            "Inference (ms)": f"{r['inference_ms']:.1f}",
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    csv_path = os.path.join(output_dir, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved to: {csv_path}")
    return df

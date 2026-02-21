"""
Post-Hoc Evaluation with Calibrated Thresholds
================================================
Re-evaluates the Fair Distill model on the TEST SET and cross-datasets
using the per-group thresholds from posthoc_threshold_calibration.py.

Usage:
  python posthoc_evaluate.py

Prerequisites:
  Run posthoc_threshold_calibration.py first to generate calibrated_thresholds.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
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

MODEL_CHECKPOINT = os.path.join(MODELS_DIR, "fair_student_best_auc.pth")
THRESHOLDS_PATH = os.path.join(RESULTS_DIR, "calibrated_thresholds.json")
DEFAULT_THRESHOLD = 0.5
BATCH_SIZE = 64


def load_thresholds():
    """Load calibrated thresholds from JSON."""
    if not os.path.exists(THRESHOLDS_PATH):
        print(f"❌ Thresholds file not found: {THRESHOLDS_PATH}")
        print("   Run posthoc_threshold_calibration.py first!")
        sys.exit(1)

    with open(THRESHOLDS_PATH) as f:
        data = json.load(f)
    return data["thresholds"]


def get_predictions(model, dataloader, device):
    """Collect predictions."""
    model.eval()
    all_probs, all_labels, all_groups = [], [], []

    with torch.no_grad():
        for images, labels, groups in tqdm(dataloader, desc="Inference", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_groups.append(groups.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels), np.concatenate(all_groups)


def apply_thresholds(probs, groups, thresholds):
    """Apply per-group thresholds."""
    preds = np.zeros(len(probs), dtype=int)
    for g in range(NUM_GROUPS):
        mask = (groups == g)
        if mask.sum() == 0:
            continue
        g_name = IDX_TO_GROUP[g]
        thresh = thresholds.get(g_name, DEFAULT_THRESHOLD)
        preds[mask] = (probs[mask] > thresh).astype(int)
    return preds


def evaluate_dataset(model, csv_path, thresholds, dataset_name, device):
    """Evaluate on a dataset with both default and calibrated thresholds."""
    if not os.path.exists(csv_path):
        print(f"  ⚠️  {csv_path} not found, skipping.")
        return None

    ds = DeepfakeDataset(csv_path, split="test")
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    probs, labels, groups = get_predictions(model, dl, device)

    results = {}
    for mode, thresh_dict in [("default", {IDX_TO_GROUP[g]: 0.5 for g in range(NUM_GROUPS)}),
                                ("calibrated", thresholds)]:
        preds = apply_thresholds(probs, groups, thresh_dict)
        acc = accuracy_score(labels, preds)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.0
        gm = compute_group_metrics(labels, preds, probs, groups)
        fm = compute_fairness_metrics(gm)
        results[mode] = {
            "accuracy": acc, "auc": auc,
            "FFPR_gap": fm["FFPR_gap"], "FEO_gap": fm["FEO_gap"],
            "FDP_gap": fm["FDP_gap"], "FOAE_gap": fm["FOAE_gap"],
            "per_group": {g: m["accuracy"] for g, m in gm.items()},
        }

    # Print comparison
    d = results["default"]
    c = results["calibrated"]
    print(f"\n{'='*65}")
    print(f"  {dataset_name}")
    print(f"{'='*65}")
    print(f"  {'Metric':14s}  {'Default':>10s}  {'Calibrated':>10s}  {'Change':>10s}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}")

    for m in ["accuracy", "FFPR_gap", "FEO_gap", "FDP_gap", "FOAE_gap"]:
        dv, cv = d[m], c[m]
        delta = cv - dv
        sign = "+" if delta >= 0 else ""
        print(f"  {m:14s}  {dv:10.4f}  {cv:10.4f}  {sign}{delta:9.4f}")

    print(f"  {'AUC':14s}  {d['auc']:10.4f}  {c['auc']:10.4f}  {'(same)':>10s}")

    print(f"\n  Per-Group Accuracy (calibrated):")
    for g_name, acc in sorted(c["per_group"].items()):
        thresh = thresholds.get(g_name, 0.5)
        d_acc = d["per_group"].get(g_name, 0)
        delta = acc - d_acc
        print(f"    {g_name:20s}: {acc:.4f} (Δ {'+' if delta>=0 else ''}{delta:.4f}, thresh={thresh:.2f})")

    return results


def main():
    print("=" * 65)
    print("  POST-HOC EVALUATION WITH CALIBRATED THRESHOLDS")
    print("=" * 65)

    # Load model
    print(f"\n📂 Loading model: {os.path.basename(MODEL_CHECKPOINT)}")
    model = build_student(pretrained=False, device=DEVICE)
    ckpt = torch.load(MODEL_CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load calibrated thresholds
    thresholds = load_thresholds()
    print(f"\n📋 Per-Group Thresholds:")
    for g_name, t in sorted(thresholds.items()):
        print(f"   {g_name:20s}: {t:.2f}")

    # ── Evaluate on TEST set ──────────────────────────────────
    test_csv = os.path.join(SPLITS_DIR, "test.csv")
    test_results = evaluate_dataset(model, test_csv, thresholds, "FF++ TEST SET", DEVICE)

    # ── Evaluate on cross-datasets ────────────────────────────
    cross_results = {}
    for ds_name in ["Celeb-DF", "DFD"]:
        cross_csv = os.path.join(SPLITS_DIR, f"{ds_name}_test.csv")
        r = evaluate_dataset(model, cross_csv, thresholds, f"CROSS-DATASET: {ds_name}", DEVICE)
        if r:
            cross_results[ds_name] = r

    # ── Summary table ─────────────────────────────────────────
    targets = {"FFPR_gap": 0.12, "FEO_gap": 0.10, "FDP_gap": 0.10, "FOAE_gap": 0.08}

    print(f"\n{'='*65}")
    print(f"  FINAL SUMMARY — Fair Distill + Post-Hoc Calibration")
    print(f"{'='*65}")

    if test_results:
        c = test_results["calibrated"]
        d = test_results["default"]
        print(f"\n  FF++ Test Set:")
        print(f"    AUC:          {c['auc']:.4f}")
        print(f"    Accuracy:     {d['accuracy']:.4f} → {c['accuracy']:.4f}")
        for m, t in targets.items():
            status = "✅" if c[m] <= t else "⚠️"
            print(f"    {m:12s}:  {d[m]:.4f} → {c[m]:.4f}  (target ≤{t})  {status}")

    for ds_name, r in cross_results.items():
        if r:
            c = r["calibrated"]
            d = r["default"]
            print(f"\n  {ds_name}:")
            print(f"    AUC:          {c['auc']:.4f}")
            print(f"    Accuracy:     {d['accuracy']:.4f} → {c['accuracy']:.4f}")
            for m, t in targets.items():
                status = "✅" if c[m] <= t else "⚠️"
                print(f"    {m:12s}:  {d[m]:.4f} → {c[m]:.4f}  (target ≤{t})  {status}")

    # ── Save results CSV ──────────────────────────────────────
    rows = []
    all_datasets = {"FF++ Test": test_results}
    all_datasets.update(cross_results)

    for ds_name, r in all_datasets.items():
        if not r:
            continue
        for mode in ["default", "calibrated"]:
            row = {
                "Dataset": ds_name,
                "Mode": mode,
                "Accuracy": f"{r[mode]['accuracy']:.4f}",
                "AUC": f"{r[mode]['auc']:.4f}",
                "FFPR_gap": f"{r[mode]['FFPR_gap']:.4f}",
                "FEO_gap": f"{r[mode]['FEO_gap']:.4f}",
                "FDP_gap": f"{r[mode]['FDP_gap']:.4f}",
                "FOAE_gap": f"{r[mode]['FOAE_gap']:.4f}",
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "posthoc_evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")
    print(f"\n✅ Post-hoc evaluation complete!")


if __name__ == "__main__":
    main()

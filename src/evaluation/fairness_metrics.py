"""
Fairness metrics: FFPR (FPR Gap), FEO (Equalized Odds), FDP (Demographic Parity),
FOAE (Accuracy Equality).
"""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import NUM_GROUPS, IDX_TO_GROUP


def compute_group_metrics(y_true, y_pred, y_prob, group_ids, num_groups=NUM_GROUPS):
    """
    Compute per-group classification metrics.

    Args:
        y_true: Ground truth labels (0/1), shape (N,)
        y_pred: Predicted labels (0/1), shape (N,)
        y_prob: Predicted probabilities, shape (N,)
        group_ids: Demographic group IDs, shape (N,)

    Returns:
        dict of group_name -> {accuracy, fpr, fnr, tpr, auc, count, ...}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    group_ids = np.asarray(group_ids)

    group_metrics = {}

    for g in range(num_groups):
        mask = (group_ids == g)
        if mask.sum() < 5:
            continue

        gt = y_true[mask]
        pred = y_pred[mask]
        prob = y_prob[mask]
        group_name = IDX_TO_GROUP[g]

        # Accuracy
        acc = accuracy_score(gt, pred)

        # AUC
        try:
            auc = roc_auc_score(gt, prob)
        except ValueError:
            auc = 0.0

        # Confusion matrix: TN, FP, FN, TP
        if len(np.unique(gt)) >= 2:
            tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0, 1]).ravel()
        else:
            tn = fp = fn = tp = 0
            for g_true, g_pred in zip(gt, pred):
                if g_true == 0 and g_pred == 0:
                    tn += 1
                elif g_true == 0 and g_pred == 1:
                    fp += 1
                elif g_true == 1 and g_pred == 0:
                    fn += 1
                else:
                    tp += 1

        # Rates
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)
        tpr = tp / max(tp + fn, 1)  # sensitivity / recall
        tnr = tn / max(tn + fp, 1)  # specificity
        ppv = tp / max(tp + fp, 1)  # precision
        positive_rate = (pred == 1).mean()  # demographic parity metric

        group_metrics[group_name] = {
            "accuracy": acc,
            "auc": auc,
            "fpr": fpr,
            "fnr": fnr,
            "tpr": tpr,
            "tnr": tnr,
            "ppv": ppv,
            "positive_rate": positive_rate,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "count": int(mask.sum()),
        }

    return group_metrics


def compute_fairness_metrics(group_metrics):
    """
    Compute aggregate fairness metrics from per-group metrics.

    Args:
        group_metrics: Dict from compute_group_metrics()

    Returns:
        dict with FFPR, FEO, FDP, FOAE, and gap details
    """
    if len(group_metrics) < 2:
        return {"error": "Need at least 2 groups for fairness metrics"}

    fprs = {g: m["fpr"] for g, m in group_metrics.items()}
    tprs = {g: m["tpr"] for g, m in group_metrics.items()}
    accs = {g: m["accuracy"] for g, m in group_metrics.items()}
    pos_rates = {g: m["positive_rate"] for g, m in group_metrics.items()}
    aucs = {g: m["auc"] for g, m in group_metrics.items()}

    # FFPR: False Positive Rate Gap (max - min across groups)
    fpr_gap = max(fprs.values()) - min(fprs.values())
    fpr_worst = max(fprs, key=fprs.get)
    fpr_best = min(fprs, key=fprs.get)

    # FEO: Equalized Odds Gap (max disparity in TPR + FPR)
    tpr_gap = max(tprs.values()) - min(tprs.values())
    eo_gap = max(fpr_gap, tpr_gap)

    # FDP: Demographic Parity (max diff in positive prediction rates)
    dp_gap = max(pos_rates.values()) - min(pos_rates.values())

    # FOAE: Accuracy Equality (max accuracy gap)
    acc_gap = max(accs.values()) - min(accs.values())
    acc_worst = min(accs, key=accs.get)
    acc_best = max(accs, key=accs.get)

    # AUC gap
    auc_gap = max(aucs.values()) - min(aucs.values())

    return {
        "FFPR_gap": fpr_gap,
        "FFPR_worst_group": fpr_worst,
        "FFPR_best_group": fpr_best,
        "FEO_gap": eo_gap,
        "FEO_tpr_gap": tpr_gap,
        "FEO_fpr_gap": fpr_gap,
        "FDP_gap": dp_gap,
        "FOAE_gap": acc_gap,
        "FOAE_worst_group": acc_worst,
        "FOAE_best_group": acc_best,
        "AUC_gap": auc_gap,
        "per_group_fpr": fprs,
        "per_group_tpr": tprs,
        "per_group_accuracy": accs,
        "per_group_positive_rate": pos_rates,
        "per_group_auc": aucs,
    }


def compute_all_fairness_metrics(y_true, y_pred, group_ids, y_prob=None,
                                  num_groups=NUM_GROUPS):
    """
    One-shot function to compute everything.

    Args:
        y_true: Ground truth (0/1)
        y_pred: Predictions (0/1)
        group_ids: Demographic group IDs
        y_prob: Predicted probabilities (optional, defaults to y_pred)

    Returns:
        dict: Combined group and fairness metrics
    """
    if y_prob is None:
        y_prob = y_pred.astype(float)

    group_metrics = compute_group_metrics(y_true, y_pred, y_prob, group_ids,
                                           num_groups)
    fairness_metrics = compute_fairness_metrics(group_metrics)

    return {
        "group_metrics": group_metrics,
        "fairness": fairness_metrics,
    }


def print_fairness_report(results, model_name="Model"):
    """Pretty-print a fairness report."""
    gm = results["group_metrics"]
    fm = results["fairness"]

    print(f"\n{'='*70}")
    print(f"FAIRNESS REPORT: {model_name}")
    print(f"{'='*70}")

    # Per-group table
    print(f"\n{'Group':<22} {'Acc':>6} {'AUC':>6} {'FPR':>6} {'TPR':>6} "
          f"{'PPV':>6} {'Count':>7}")
    print("-" * 70)
    for group_name, m in sorted(gm.items()):
        print(f"{group_name:<22} {m['accuracy']:>6.3f} {m['auc']:>6.3f} "
              f"{m['fpr']:>6.3f} {m['tpr']:>6.3f} {m['ppv']:>6.3f} "
              f"{m['count']:>7d}")

    # Fairness summary
    print(f"\n{'Fairness Metric':<30} {'Value':>8} {'Target':>8} {'Pass':>6}")
    print("-" * 60)
    metrics_targets = [
        ("FFPR Gap (FPR)", fm["FFPR_gap"], 0.12),
        ("FEO Gap (Equalized Odds)", fm["FEO_gap"], 0.10),
        ("FDP Gap (Demographic Parity)", fm["FDP_gap"], 0.10),
        ("FOAE Gap (Accuracy)", fm["FOAE_gap"], 0.08),
        ("AUC Gap", fm["AUC_gap"], 0.05),
    ]
    for name, val, target in metrics_targets:
        passed = "✓" if val <= target else "✗"
        print(f"{name:<30} {val:>8.4f} {target:>8.2f} {passed:>6}")

    print(f"\nWorst group (accuracy): {fm['FOAE_worst_group']}")
    print(f"Best group (accuracy):  {fm['FOAE_best_group']}")
    print(f"{'='*70}")

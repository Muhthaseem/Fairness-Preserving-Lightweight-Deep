"""
Visualization utilities: bar charts, ROC curves, fairness comparison plots.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import FIGURES_DIR, DEMOGRAPHIC_GROUPS


def set_style():
    """Set publication-quality plot style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "figure.dpi": 150,
    })


def plot_per_group_accuracy(group_accs_dict, title="Per-Group Accuracy",
                             output_path=None):
    """
    Bar chart of per-group accuracy for multiple models.

    Args:
        group_accs_dict: {model_name: {group_name: accuracy}}
        title: Plot title
        output_path: Path to save (optional)
    """
    set_style()
    models = list(group_accs_dict.keys())
    groups = DEMOGRAPHIC_GROUPS

    x = np.arange(len(groups))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = sns.color_palette("husl", len(models))

    for i, model_name in enumerate(models):
        accs = [group_accs_dict[model_name].get(g, 0) for g in groups]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=model_name,
                       color=colors[i], alpha=0.85)

    ax.set_xlabel("Demographic Group")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="Target")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.savefig(os.path.join(FIGURES_DIR, "per_group_accuracy.png"),
                     dpi=150, bbox_inches="tight")
    plt.close()


def plot_fairness_gaps(fairness_dict, output_path=None):
    """
    Bar chart comparing fairness gaps across models.

    Args:
        fairness_dict: {model_name: {FFPR_gap, FEO_gap, FDP_gap, FOAE_gap}}
    """
    set_style()
    metrics = ["FFPR_gap", "FEO_gap", "FDP_gap", "FOAE_gap"]
    metric_labels = ["FPR Gap", "EO Gap", "DP Gap", "Acc Gap"]
    targets = [0.12, 0.10, 0.10, 0.08]

    models = list(fairness_dict.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("husl", len(models))

    for i, model_name in enumerate(models):
        vals = [fairness_dict[model_name].get(m, 0) for m in metrics]
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=model_name, color=colors[i], alpha=0.85)

    # Plot targets
    for j, target in enumerate(targets):
        ax.axhline(y=target, color="red", linestyle="--", alpha=0.3)

    ax.set_xlabel("Fairness Metric")
    ax.set_ylabel("Gap Value")
    ax.set_title("Fairness Gap Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()

    plt.tight_layout()
    save_path = output_path or os.path.join(FIGURES_DIR, "fairness_gaps.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_history(history, model_name="Model", output_path=None):
    """
    Plot training and validation curves (loss, accuracy, AUC).

    Args:
        history: {'train': [metrics_per_epoch], 'val': [metrics_per_epoch]}
    """
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train"]) + 1)

    # Loss
    axes[0].plot(epochs, [m["loss"] for m in history["train"]], label="Train")
    axes[0].plot(epochs, [m["loss"] for m in history["val"]], label="Val")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, [m["accuracy"] for m in history["train"]], label="Train")
    axes[1].plot(epochs, [m["accuracy"] for m in history["val"]], label="Val")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    # AUC
    axes[2].plot(epochs, [m["auc"] for m in history["train"]], label="Train")
    axes[2].plot(epochs, [m["auc"] for m in history["val"]], label="Val")
    axes[2].set_title(f"{model_name} — AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].legend()

    plt.tight_layout()
    save_path = output_path or os.path.join(FIGURES_DIR, f"history_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_per_group(y_true, y_prob, group_ids, model_name="Model",
                        output_path=None):
    """
    ROC curve per demographic group.

    Args:
        y_true, y_prob: Arrays of true labels and predicted probabilities
        group_ids: Demographic group IDs
    """
    from sklearn.metrics import roc_curve, auc

    set_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette("husl", len(DEMOGRAPHIC_GROUPS))

    for g, (group_name, color) in enumerate(zip(DEMOGRAPHIC_GROUPS, colors)):
        mask = (group_ids == g)
        if mask.sum() < 5:
            continue
        fpr, tpr, _ = roc_curve(y_true[mask], y_prob[mask])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{group_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves by Demographic Group — {model_name}")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    save_path = output_path or os.path.join(FIGURES_DIR, f"roc_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

"""
Cross-dataset evaluation: test models on Celeb-DF and DFD
to measure domain generalization and fairness robustness.
"""
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import DEVICE, SPLITS_DIR, RESULTS_DIR, BATCH_SIZE, NUM_WORKERS
from src.data.dataset import DeepfakeDataset
from src.evaluation.evaluate import evaluate_model, save_results


def evaluate_cross_dataset(model, model_name="model", device=DEVICE):
    """
    Evaluate model on cross-dataset test sets (Celeb-DF, DFD).

    Args:
        model: Trained model
        model_name: Name for results

    Returns:
        dict: Results for each cross-dataset
    """
    cross_results = {}

    for dataset_name in ["Celeb-DF", "DFD"]:
        test_csv = os.path.join(SPLITS_DIR, f"{dataset_name}_test.csv")
        if not os.path.exists(test_csv):
            print(f"[WARN] Cross-dataset test not found: {test_csv}")
            continue

        print(f"\n{'='*60}")
        print(f"CROSS-DATASET EVALUATION: {dataset_name}")
        print(f"{'='*60}")

        test_dataset = DeepfakeDataset(test_csv, split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        results = evaluate_model(
            model, test_loader, device,
            model_name=f"{model_name} on {dataset_name}"
        )
        save_results(results, RESULTS_DIR,
                      filename=f"cross_{dataset_name}_{model_name}")
        cross_results[dataset_name] = results

    return cross_results


def run_full_cross_dataset_evaluation(models_dict, device=DEVICE):
    """
    Run cross-dataset evaluation for multiple models.

    Args:
        models_dict: Dict of model_name -> model instance
    """
    all_results = {}
    for name, model in models_dict.items():
        results = evaluate_cross_dataset(model, name, device)
        all_results[name] = results

    # Summary table
    rows = []
    for model_name, datasets in all_results.items():
        for dataset_name, results in datasets.items():
            rows.append({
                "Model": model_name,
                "Dataset": dataset_name,
                "AUC": f"{results['overall_auc']:.4f}",
                "Accuracy": f"{results['overall_accuracy']:.4f}",
                "FFPR Gap": f"{results['fairness']['FFPR_gap']:.4f}",
                "FOAE Gap": f"{results['fairness']['FOAE_gap']:.4f}",
            })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("CROSS-DATASET SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))

    csv_path = os.path.join(RESULTS_DIR, "cross_dataset_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved to: {csv_path}")

    return all_results

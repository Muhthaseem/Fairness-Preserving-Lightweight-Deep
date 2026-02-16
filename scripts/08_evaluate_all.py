"""
Script 08: Run full evaluation + ablation study on all models.
Usage: python scripts/08_evaluate_all.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.config import DEVICE, SPLITS_DIR, MODELS_DIR, BATCH_SIZE, NUM_WORKERS
from src.models.xception import build_teacher
from src.models.mobilenetv2 import build_student
from src.data.dataset import DeepfakeDataset
from src.evaluation.evaluate import evaluate_model, save_results, compare_models
from src.evaluation.cross_dataset import evaluate_cross_dataset
from torch.utils.data import DataLoader


def load_model_weights(model, weights_path, device=DEVICE):
    """Load model weights from checkpoint."""
    if not os.path.exists(weights_path):
        print(f"[WARN] Weights not found: {weights_path}")
        return model
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded weights from: {weights_path}")
    return model


if __name__ == "__main__":
    # Test set
    test_csv = os.path.join(SPLITS_DIR, "test.csv")
    test_dataset = DeepfakeDataset(test_csv, split="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    all_results = []

    # --- 1. Teacher (XceptionNet) ---
    print("\n" + "=" * 70)
    print("1. EVALUATING TEACHER (XceptionNet)")
    print("=" * 70)
    teacher = build_teacher(pretrained=True, device=DEVICE)
    teacher = load_model_weights(
        teacher, os.path.join(MODELS_DIR, "xception_teacher_best.pth")
    )
    teacher_results = evaluate_model(teacher, test_loader, DEVICE, "XceptionNet (Teacher)")
    save_results(teacher_results, filename="teacher_evaluation")
    all_results.append(teacher_results)

    # --- 2. Baseline Student (MobileNetV2, no distillation) ---
    print("\n" + "=" * 70)
    print("2. EVALUATING BASELINE STUDENT (MobileNetV2)")
    print("=" * 70)
    baseline = build_student(pretrained=True, device=DEVICE)
    baseline = load_model_weights(
        baseline, os.path.join(MODELS_DIR, "mobilenetv2_baseline_best.pth")
    )
    baseline_results = evaluate_model(baseline, test_loader, DEVICE,
                                       "MobileNetV2 (Baseline)")
    save_results(baseline_results, filename="baseline_evaluation")
    all_results.append(baseline_results)

    # --- 3. Standard Distillation Student (for ablation) ---
    print("\n" + "=" * 70)
    print("3. EVALUATING STANDARD DISTILLED STUDENT")
    print("=" * 70)
    std_distill = build_student(pretrained=True, device=DEVICE)
    std_weights = os.path.join(MODELS_DIR, "std_distill_student_best_auc.pth")
    if os.path.exists(std_weights):
        std_distill = load_model_weights(std_distill, std_weights)
        std_results = evaluate_model(std_distill, test_loader, DEVICE,
                                      "MobileNetV2 (Std Distill)")
        save_results(std_results, filename="std_distill_evaluation")
        all_results.append(std_results)
    else:
        print("[SKIP] Standard distillation model not found.")

    # --- 4. Fair Distillation Student (OURS) ---
    print("\n" + "=" * 70)
    print("4. EVALUATING FAIR DISTILLED STUDENT (OURS)")
    print("=" * 70)
    fair = build_student(pretrained=True, device=DEVICE)
    fair = load_model_weights(
        fair, os.path.join(MODELS_DIR, "fair_student_best_auc.pth")
    )
    fair_results = evaluate_model(fair, test_loader, DEVICE,
                                   "MobileNetV2 (Fair Distill)")
    save_results(fair_results, filename="fair_evaluation")
    all_results.append(fair_results)

    # --- Comparison Table ---
    compare_models(all_results)

    # --- Cross-dataset evaluation for the fair model ---
    print("\n" + "=" * 70)
    print("5. CROSS-DATASET EVALUATION (Fair Student)")
    print("=" * 70)
    evaluate_cross_dataset(fair, model_name="fair_student", device=DEVICE)

    print("\n✓ All evaluations complete. Results saved to outputs/results/")

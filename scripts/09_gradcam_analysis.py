"""
Script 09: Generate Grad-CAM visualizations per demographic group.
Usage: python scripts/09_gradcam_analysis.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.config import DEVICE, SPLITS_DIR, MODELS_DIR, GRADCAM_DIR
from src.models.mobilenetv2 import build_student
from src.explainability.gradcam import generate_gradcam_per_group


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights", type=str,
                        default=os.path.join(MODELS_DIR, "fair_student_best_auc.pth"))
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    # Load model
    model = build_student(pretrained=True, device=DEVICE)
    if os.path.exists(args.model_weights):
        checkpoint = torch.load(args.model_weights, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from: {args.model_weights}")

    # Generate Grad-CAM
    test_csv = os.path.join(SPLITS_DIR, "test.csv")
    generate_gradcam_per_group(
        model, test_csv,
        num_samples_per_group=args.num_samples,
        output_dir=GRADCAM_DIR,
        device=DEVICE,
    )

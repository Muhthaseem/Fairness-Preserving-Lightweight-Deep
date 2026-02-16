# Fairness-Preserving Lightweight Deepfake Detection

A research implementation of **fairness-aware knowledge distillation** for deepfake detection, ensuring consistent performance across demographic groups while maintaining deployment-ready model size.

## Key Innovation

Novel fairness-aware loss function that minimizes accuracy disparity across 8 demographic groups (Male/Female × White/Black/Asian/Other) during knowledge distillation from a heavy teacher (XceptionNet, 71M params) to a lightweight student (MobileNetV2, 3.5M params).

```
L_total = α · L_distill(KL-div) + β · L_fairness(pairwise accuracy gap) + γ · L_cls(BCE)
```

## Project Structure

```
src/                    # Core library
  config.py             # Configuration (paths, hyperparams, demographics)
  data/                 # Data pipeline (frame extraction, face detection, demographics)
  models/               # XceptionNet (teacher) & MobileNetV2 (student)
  training/             # Baseline training & fairness-aware distillation
  evaluation/           # Metrics, fairness analysis, cross-dataset evaluation
  explainability/       # Grad-CAM visualizations
  utils/                # Visualization & plotting

scripts/                # Runner scripts (numbered 01-10)
demo/                   # Streamlit web demo
outputs/                # Generated outputs (models, results, figures)
```

## Quick Start

### Setup
See [INSTALL.md](INSTALL.md) for detailed setup instructions.

### Pipeline
Run the numbered scripts in order:

```bash
python scripts/01_extract_frames.py       # Extract frames from videos
python scripts/02_detect_faces.py          # Detect & crop faces (MTCNN)
python scripts/03_annotate_demographics.py # Annotate race/gender (FairFace)
python scripts/04_create_splits.py         # Create train/val/test splits
python scripts/05_train_teacher.py         # Train XceptionNet teacher
python scripts/06_train_baseline_student.py # Train MobileNetV2 baseline
python scripts/07_train_fair_student.py    # Train fair-distilled student
python scripts/08_evaluate_all.py          # Evaluate all models + ablation
python scripts/09_gradcam_analysis.py      # Generate Grad-CAM heatmaps
streamlit run demo/app.py                  # Launch web demo
```

## Datasets

| Dataset | Role | Size |
|---------|------|------|
| FaceForensics++ (C23) | Train/Val/Test | ~1.8M frames |
| Celeb-DF | Cross-dataset test | ~50K frames |
| DFD | Cross-dataset test | ~30K frames |

## Target Metrics

| Metric | Target |
|--------|--------|
| AUC (FF++) | ≥ 95% |
| FPR Gap | ≤ 12% |
| Accuracy Gap | ≤ 8% |
| Model Size | ≤ 15 MB |
| Inference | ≤ 30 ms |
| Cross-dataset AUC | ≥ 72% |

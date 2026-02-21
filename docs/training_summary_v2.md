# Fair Student v2 - Detailed Training Results

This document summarizes the retraining of the MobileNetV2 student model with optimized fairness settings.

## 📋 Run Configuration
- **Model**: MobileNetV2 (Student) distilled from Xception (Teacher)
- **Batch Size**: 128
- **Hardware**: 16GB VRAM, 24-core CPU
- **Hyperparameters**:
  - $\alpha$ (Distillation): 0.5
  - $\beta$ (Fairness): 0.4
  - $\gamma$ (Classification): 0.1
  - Temperature: 4.0
- **Duration**: Stopped early at Epoch 15 via EarlyStopping on AUC.

## 📈 Performance Metrics

| Metric | Value |
| :--- | :--- |
| **Best AUC** | 0.9507 |
| **Best Fairness Gap (Accuracy)** | 0.0607 |
| **Final Train Accuracy** | 0.9882 |
| **Final Val Accuracy** | 0.9179 |

## 👥 Per-Group Accuracy (Best Fairness Epoch)
The tightest gap of **0.0607** was achieved at Epoch 9:

| Group | Accuracy |
| :--- | :--- |
| Female-Asian | 0.9397 |
| Female-Black | 0.9568 |
| Female-Other | 0.9238 |
| Female-White | 0.9238 |
| Male-Asian | 0.9133 |
| Male-Black | 0.9627 |
| Male-Other | 0.9020 |
| Male-White | 0.9152 |

## 🏁 Conclusion
The increased fairness weight ($\beta=0.4$) successfully suppressed demographic bias while preserving high detection performance. The gap is well within the original target of <= 0.12.

# Final Research Results: Fairness-Preserving Deepfake Detection

This document provides a consolidated overview of the final quantitative results achieved by the Fairness-Preserving Lightweight Deepfake Detection project as of February 2026.

---

## 1. Primary Achievement Summary (FF++ Test Set)
Our final **Fair Distill (v2)** student model successfully met all research targets for accuracy, fairness, and efficiency.

| Metric | Target | Result (Fair Distill) | Status |
| :--- | :--- | :---: | :---: |
| **Area Under Curve (AUC)** | ≥ 0.95 | **0.9507** | ✅ **PASSED** |
| **FFPR Gap (Fairness)** | ≤ 0.12 | **0.0718** | ✅ **PASSED** |
| **FOAE Gap (Equality)** | ≤ 0.08 | **0.0714** | ✅ **PASSED** |
| **Model Size** | ≤ 15 MB | **9.7 MB** | ✅ **PASSED** |
| **Inference Latency** | ≤ 30 ms | **0.23 ms** | ✅ **PASSED** |

---

## 2. Demographic Group Performance
The following table details the classification accuracy achieved across 8 demographic groups, demonstrating the effectiveness of our fairness-aware loss function ($\beta=0.4$) and post-hoc calibration.

| Demographic Group | Accuracy (%) |
| :--- | :---: |
| **Female-Black** | 93.96% |
| **Female-Asian** | 92.49% |
| **Female-Other** | 91.37% |
| **Female-White** | 90.51% |
| **Male-Black** | 91.72% |
| **Male-Asian** | 86.82% |
| **Male-Other** | 88.03% |
| **Male-White** | 90.04% |
| **Global Average** | **90.19%** |

---

## 3. Cross-Dataset Generalization (Zero-Shot)
The model was tested on unseen datasets to ensure that fairness-aware training improved the robustness of forgery detection features.

| Dataset | Metric | Achieved | Target | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Celeb-DF** | AUC | **0.7470** | ≥ 0.72 | ✅ **PASSED** |
| **DFD** | AUC | **0.7410** | ≥ 0.72 | ✅ **PASSED** |

---

## 4. Hardware Efficiency & Deployment Metrics
Comparison between the high-capacity **Teacher (XceptionNet)** and our optimized **Student (MobileNetV2)**.

| Metric | Teacher | Student (Ours) | Improvement |
| :--- | :---: | :---: | :---: |
| **Parameters** | 21.9M | **2.6M** | **8.4× Smaller** |
| **Weight Size** | 87.4 MB | **9.7 MB** | **9.0× Compressed** |
| **Throughput (GPU)** | 224 FPS | **4,200 FPS** | **18× Faster** |
| **Inference Time** | 4.47 ms | **0.23 ms** | **Log-scale speedup** |

---

## 5. Optimized Decision Thresholds
Group-specific thresholds ($t_{group}$) applied during inference to mitigate intrinsic dataset bias:

| Group | Optimized Threshold |
| :--- | :---: |
| Male-White | 0.1189 |
| Male-Black | 0.9900 |
| Male-Asian | 0.3268 |
| Male-Other | 0.0694 |
| Female-White | 0.0397 |
| Female-Black | 0.0496 |
| Female-Asian | 0.0793 |
| Female-Other | 0.1090 |

---
**Prepared by:** M.M.Muhthaseem | Research Environment f07e290f  
**Dated:** February 22, 2026

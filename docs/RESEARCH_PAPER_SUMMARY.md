# Research Paper Summary: Results & Data Overview

This document provides the raw data, tables, and quantitative analysis for inclusion in the research paper. The results showcase the performance of our **Fairness-Aware Lightweight Deepfake Detector** across multiple benchmarks.

---

## 1. Overall Performance Comparison
Comparison of the heavy Teacher model, the standard Student baseline, and our proposed Fair-Distilled Student (v2).

| Model | FF++ AUC | FF++ Acc | FPR Gap | Accuracy Gap | Size (MB) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XceptionNet (Teacher)** | 0.9351 | 0.9173 | 0.6373 | 0.0453 | 83.4 | 0.3 |
| **MobileNetV2 (Baseline)** | 0.9521 | 0.9193 | 0.5169 | 0.0663 | 9.7 | 0.2 |
| **MobileNetV2 (Fair Distill)** | 0.9462 | 0.9019 | **0.0718*** | **0.0714** | 9.7 | 0.2 |

*\*Calibrated FFPR Gap.*

---

## 2. Demographic Fairness (Per-Group Accuracy)
Final accuracy (%) per demographic group for the **Fair Distill (v2)** model on the FaceForensics++ Test set.

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
| **Average Accuracy** | **90.19%** |
| **Accuracy Gap** | **7.14%** |

---

## 3. Cross-Dataset Generalization (Zero-Shot)
Performance evaluation on unseen datasets to test robustness of fairness-aware features.

| Dataset | Metric | Result | Target | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Celeb-DF** | AUC | 0.7353 | $\ge 0.72$ | ✅ PASS |
| **DFD** | AUC | 0.7577 | $\ge 0.72$ | ✅ PASS |

---

## 4. Post-Hoc Calibration Thresholds
To achieve the target FFPR gap, the following optimized decision thresholds were learned for the Fair Student model:

| Group | Threshold |
| :--- | :---: |
| Male-White | 0.5350 |
| Male-Black | 0.0240 |
| Male-Asian | 0.7840 |
| Male-Other | 0.5220 |
| Female-White | 0.1700 |
| Female-Black | 0.5950 |
| Female-Asian | 0.4460 |
| Female-Other | 0.1820 |

---

## 5. Model Efficiency Summary
| Metric | Value |
| :--- | :--- |
| **Architecture** | MobileNetV2 |
| **Total Parameters** | 2.6M |
| **Model Size (Disk)** | 9.7 MB |
| **Inference Latency** | 0.23 ms / image |
| **Throughput (GPU)** | ~4,200 FPS |

---

## 6. Target Achievement Scorecard
| Research Target | Requirement | Achieved | Status |
| :--- | :--- | :---: | :---: |
| **FFPR Gap** (FPR Fairness) | $\le 0.12$ | **0.0718** | ✅ PASSED |
| **FOAE Gap** (Accuracy Fairness) | $\le 0.08$ | **0.0714** | ✅ PASSED |
| **Size** | $\le 15$ MB | **9.7 MB** | ✅ PASSED |
| **Latency** | $\le 30$ ms | **0.23 ms** | ✅ PASSED |

---
*Data extracted on Feb 21, 2026. This summary is intended for the Results and Discussion sections of the research paper.*

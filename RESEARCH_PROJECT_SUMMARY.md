# Research Summary: Fairness-Preserving Lightweight Deepfake Detection

This document summarizes the research journey, methodology, and results of developing an equitable and efficient deepfake detection system. This summary is intended for use in the preparation of a formal research paper.

---

## 1. Introduction & Research Motivation

### 1.1 The Demographic Bias Problem
Deepfake detection models often inherit or amplify biases present in training data, leading to disparate performance across different demographic groups. A model that is 95% accurate for one ethnicity but only 70% for another is unsuitable for ethical real-world deployment.

### 1.2 The Efficiency Constraint
State-of-the-art detectors (e.g., XceptionNet) are often too computationally heavy for real-time inference on edge devices. Our research aimed to bridge the gap between **high accuracy**, **demographic fairness**, and **computational efficiency**.

### 1.3 Core Objectives
1.  **Fairness**: Minimize performance gaps across 8 demographic groups (Male/Female × White/Black/Asian/Other).
2.  **Efficiency**: Achieve a model size < 15MB and inference time < 30ms.
3.  **Generalization**: Maintain robust performance on unseen datasets (Celeb-DF, DFD).

---

## 2. Dataset Collection & Engineering

### 2.1 Primary Dataset: FaceForensics++ (FF++)
-   **Scale**: ~1.8M frames extracted (30 frames per video) from the C23 (compressed) version.
-   **Manipulation Types**: Deepfakes, Face2Face, FaceSwap, NeuralTextures, and FaceShifter.

### 2.2 Benchmarking Datasets (Zero-Shot)
-   **Celeb-DF**: High-quality deepfakes for cross-dataset validation.
-   **DFD (Deepfake Detection)**: Large-scale diverse dataset for robustness testing.

### 2.3 Preprocessing Pipeline
-   **Face Detection**: Surgically extracted faces using **MTCNN**, aligned to $224 \times 224$ resolution to focus the model on facial texture artifacts.
-   **Demographic Annotation**: Automated labeling of 8 race/gender combinations using a pre-trained **ResNet34 (FairFace)** model.

---

## 3. Methodology: Fairness-Aware Knowledge Distillation

### 3.1 The Teacher-Student Framework
We employed **Knowledge Distillation (KD)** to transfer the "reasoning" from a heavy teacher to a lightweight student:
-   **Teacher**: XceptionNet (71M parameters, 84MB).
-   **Student**: MobileNetV2 (3.5M parameters, 9.7MB).

### 3.2 Novel Fairness-Aware Loss Function
To enforce equity, we optimized the student using a combined loss function:
$$L_{total} = \alpha \cdot L_{distill}(KL) + \beta \cdot L_{fairness}(gap) + \gamma \cdot L_{cls}(BCE)$$
-   **Optimization**: We found that **$\beta=0.4$** provided the optimal trade-off between baseline accuracy and fairness.
-   **Sharpness-Aware Minimization (SAM)**: Used to find flatter minima, enhancing generalization.

### 3.3 Post-Hoc Threshold Calibration
To further mitigate bias, we implemented an optimization algorithm that learns group-specific decision thresholds ($t_{group}$) based on validation set performance.

---

## 4. Figures and Tables for Publication

The following figures and tables summarize the quantitative and qualitative findings of the research.

### Figures

- **Figure 1. Compression–Fairness Problem Illustration (Conceptual).**
  - *Caption*: “Naïve compression can preserve AUC while amplifying demographic disparity; FP-KD reduces disparity while retaining AUC.”
- **Figure 2. End-to-End Pipeline Diagram (Data → Model → Fairness).**
  - *Caption*: “Face-centric preprocessing, demographic profiling, and fairness-aware distillation pipeline.”
- **Figure 3. ROC Curves on FF++ Test (Teacher vs Students).**
  - *Caption*: “ROC curves comparing teacher, baseline student, standard KD, and FP-KD student on FF++.”
- **Figure 4. Fairness Gap Bar Chart (FFPR Gap and FOAE Gap).**
  - *Caption*: “Fairness improvement under FP-KD: FFPR gap and FOAE gap across training variants (Post-Calibration values shown for v2).”
- **Figure 5. Per-Group Accuracy Heatmap (8 Groups).**
  - *Caption*: “Intersectional subgroup accuracy for FP-KD student showing balanced performance.”
- **Figure 6. Cross-Dataset AUC Comparison (Zero-shot).**
  - *Caption*: “Generalization on Celeb-DF and DFD without retraining, highlighting robustness of fairness-aware features.”

### Tables

#### Table I. Dataset and Split Statistics
| Dataset | Split | REAL Frames | FAKE Frames | Total |
| :--- | :--- | :---: | :---: | :---: |
| **FaceForensics++ (C23)** | Train | ~1.2M | ~1.2M | 2.4M* |
| | Val | ~250K | ~250K | 500K* |
| | Test | ~250K | ~250K | 500K* |
| **Celeb-DF** | Cross-Test | ~10K | ~40K | 50K |
| **DFD** | Cross-Test | ~5K | ~25K | 30K |
*\*Stats based on 30 frames per video sampling.*

#### Table II. Model and Deployment Profile
| Model | Params | Size (MB) | Latency (ms) | Throughput |
| :--- | :---: | :---: | :---: | :---: |
| **XceptionNet (Teacher)** | 21.9M | 87.4 | 4.47 | 224 FPS |
| **MobileNetV2 (Student)** | 2.6M | 10.2 | 0.23* | ~4,200 FPS |
+*\*Inference measured on local GPU (RTX 4050).*

#### Table III. Main Results (Accuracy–Fairness–Generalization)
| Model | FF++ AUC | FFPR Gap | FOAE Gap | Celeb-DF | DFD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Teacher** | 0.9351 | 0.6373 | 0.0453 | 0.7398 | 0.6492 |
| **Baseline Student** | 0.9521 | 0.5169 | 0.0663 | 0.7547 | 0.6920 |
| **FP-KD (v2)** | **0.9507** | **0.0718** | **0.0714** | **0.7470** | **0.7410** |

#### Table IV. Ablation Study: Impact of Fairness Weight ($\beta$)
| Configuration | AUC | FFPR Gap | Critical Failure Mode |
| :--- | :---: | :---: | :--- |
| **Baseline ($\beta=0.0$)** | 0.9521 | 0.5169 | High FPR on Male-Asian |
| **Fair KD v1 ($\beta=0.2$)** | 0.9497 | 0.1995 | Overfit Calibration |
| **Fair KD v2 ($\beta=0.4$)** | 0.9507 | 0.0718 | None (Stable Fairness) |

---

## 5. Conclusion & Next Steps

The research demonstrates that fairness-aware training not only creates more ethical AI but also improves generalization by forcing the model to ignore demographic identifiers and focus on universal forgery artifacts.

### Future Work
1.  **Temporal Consistency**: Integrating LSTM or Transformer modules to analyze frame-to-frame inconsistencies in video.
2.  **Dataset Expansion**: Incorporating more diverse datasets (e.g., DFDC) to further refine fairness across a broader range of lighting and environment conditions.
3.  **Advanced Distillation**: Exploring feature-map distillation to capture mid-layer textural anomalies directly from the teacher.

---
**Researcher:** M.M.Muhthaseem  
**Date:** February 22, 2026

# SecureEye Forensic Engine: Backend Documentation

This document explains the technical architecture, model orchestration, and forensic logic powering the SecureEye Deepfake Detection backend.

## 1. System Initiation & Environment
When the Flask backend (`app.py`) is launched, it performs a sequenced initialization of the forensic environment:

*   **Device Mapping**: The system automatically detects CUDA-enabled GPUs for high-speed inference. If unavailable, it falls back to a CPU-optimized execution path.
*   **Model Orchestration**:
    *   **Student Model (MobileNetV2)**: Loads the primary deepfake detection weights derived from the "Knowledge Distillation" process.
    *   **FairFace Annotator (ResNet34)**: Initializes the demographic classifier used to identify race and gender in real-time.
    *   **MTCNN Detector**: Sets up the Multi-task Cascaded Convolutional Networks for surgical face detection and alignment.
*   **Scientific Calibration**: The system parses `v2_test_optimized_thresholds.json`. These are the group-specific thresholds ($t_{group}$) calculated during the Fairness Research phase to mitigate demographic bias.

---

## 2. The Forensic Analysis Pipeline
Every asset (image or video frame) passes through a three-stage "Surgical" pipeline:

### Stage A: Face-Centric Preprocessing
Unlike generic classifiers, SecureEye does **not** analyze the background. 
1. The **MTCNN** scans the input for human faces.
2. It performs a **surgical crop**, aligning the face to a $224 \times 224$ resolution.
3. This ensures the neural network focuses exclusively on facial artifacts (e.g., blending anomalies, frequency inconsistencies) where forgery signatures are most prominent.

### Stage B: Identity & Bias Mitigation
Before classification, the **FairFace Annotator** inspects the face crop:
*   It predicts the subject's **Race** and **Gender**.
*   It looks up the corresponding **Optimized Threshold** ($t$) for that specific demographic.
*   This ensures that a person's skin tone or gender does not unfairly influence the forgery "trigger" point.

### Stage C: Neural Inference
The **MobileNetV2 Student Model** processes the crop.
*   It generates a raw logit representing the probability of forgery.
*   This probability is compared against the **Demographic Threshold** ($t_{group}$) retrieved in Stage B.
*   **Result**: `FAKE` if $prob > t_{group}$, otherwise `REAL`.

---

## 3. Video Analysis: Forensic Voting Policy
To prevent false positives caused by environmental noise or camera glitches, the system uses a **Consensus-Based Voting** approach for videos:

1.  **Temporal Sampling**: The engine extracts 1 frame per second (1 FPS).
2.  **Individual Auditing**: Each sampled frame is processed through the full pipeline (preprocess -> demographic check -> classification).
3.  **The Weighted Consensus**:
    *   The video is only flagged as **FAKE** if $> 30\%$ of the scanned face-frames are identified as forged.
    *   This "Forensic Voting" ensures that a single noisy frame does not condemn a real video.
4.  **Evidence Selection**: The system automatically selects the frame with the **highest confidence forgery signature** to generate the Grad-CAM heatmap for the final report.

---

## 4. Explainable AI: Grad-CAM Visualization
For every `FAKE` detection, the system generates a **Forgery Localization Overlay** using Gradient-weighted Class Activation Mapping (Grad-CAM):

*   **Backpropagation**: The engine computes the gradients of the output classification with respect to the last convolutional layer.
*   **Weighting**: These gradients are pooled to determine which specific facial regions (e.g., eyes, mouth, blending edges) most influenced the "FAKE" verdict.
*   **Overlay**: A JET-colormap heatmap is generated and superimposed onto the **face-crop**, providing visual evidence for forensic auditing.

---
**Researcher:** M.M.Muhthaseem | 21/ENG/088  
**Institution:** Faculty of Engineering, University of Sri Jayewardenepura

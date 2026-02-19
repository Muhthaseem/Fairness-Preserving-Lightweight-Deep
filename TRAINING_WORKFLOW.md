# Deepfake Model Training Workflow

This guide details the step-by-step process to train the **Fairness-Preserving Lightweight Deepfake Detector**.

## 🚀 Automated Pipeline (Recommended)
You can run the entire pipeline (Demographics -> Splits -> Training -> Evaluation) with a single command. The script is optimized for **32GB VRAM** GPUs.

- **Command:** `python run_pipeline.py`
- **Features:** 
    - Auto-scales batch sizes (1024 for annotation, 512 for student training)
    - Parallel I/O for faster data loading
    - Automatic error handling and sequential execution

---

## 🛠️ Manual Execution (Step-by-Step)

### Phase 1: Data Processing

#### Step 1.1: Extract Frames
Extracts uniform frames from all video datasets (`FF++`, `Celeb-DF`, `DFD`).
- **Command:** `python scripts/01_extract_frames.py --num_frames 30`
- **Output:** `frames/` directory populated with `dataset/video/frame_001.jpg`

#### Step 1.2: Detect & Crop Faces
Uses MTCNN to detect faces in every extracted frame and crop them to 256x256.
- **Command:** `python scripts/02_detect_faces.py`
- **Optimization:** Uses parallel threads for image loading.
- **Output:** `faces/` directory and `faces.csv` manifest.

#### Step 1.3: Annotate Demographics
Uses a pre-trained FairFace model (Auto-downloaded) to classify Race and Gender.
- **Command:** `python scripts/03_annotate_demographics.py --batch_size 1024`
- **Weights:** Automatically downloads correct `res34_fair_align_multi_7_20190809.pt` from Google Drive.
- **Optimization:** Parallel loading + Batch 1024 for RTX 6000/A6000 class GPUs.
- **Output:** `faces_annotated.csv` with `race` and `gender` columns.

#### Step 1.4: Create Data Splits
Creates stratified Train (70%), Val (15%), Test (15%) splits, ensuring no video leakage.
- **Command:** `python scripts/04_create_splits.py`
- **Output:** `splits/train.csv`, `splits/val.csv`, `splits/test.csv`

---

### Phase 2: Baseline Training

#### Step 2.1: Train Teacher (XceptionNet)
Trains a heavy XceptionNet model to serve as the knowledge source.
- **Command:** `python scripts/05_train_teacher.py --epochs 20 --batch_size 64`
- **Note:** Batch size 64 is safe for 16GB VRAM.

#### Step 2.2: Train Baseline Student (MobileNetV2)
Trains a lightweight MobileNetV2 model *without* fairness components.
- **Command:** `python scripts/06_train_baseline_student.py --epochs 30 --batch_size 128`

---

### Phase 3: Fairness-Aware Distillation

#### Step 3.1: Train Fair Student
Distills knowledge from the Teacher to the Student while penalizing demographic accuracy gaps.
- **Command:** `python scripts/07_train_fair_student.py --epochs 50 --batch_size 128`
- **Key Args:** `--alpha 0.5 --beta 0.5 --gamma 0.1`

---

### Phase 4: Evaluation & Interpretation

#### Step 4.1: Comprehensive Evaluation
Evaluates all models on accuracy, fairness metrics (FPR Gap, EOD), and cross-dataset performance.
- **Command:** `python scripts/08_evaluate_all.py`
- **Output:** `outputs/results/evaluation_report.csv`

#### Step 4.2: Explainability (Grad-CAM)
Generates heatmaps to visualize model focus.
- **Command:** `python scripts/09_gradcam_analysis.py`

---

### Phase 5: Demo
Run the interactive web app to test the model on new images.
- **Command:** `streamlit run demo/app.py`

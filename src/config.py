"""
Global configuration for the Fairness-Preserving Lightweight Deepfake Detection project.
"""
import os
import torch

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "Datasets")

# FF++ paths
FF_ROOT = os.path.join(DATASET_ROOT, "FF++", "FaceForensics++_C23")
FF_ORIGINAL = os.path.join(FF_ROOT, "original")
FF_DEEPFAKES = os.path.join(FF_ROOT, "Deepfakes")
FF_FACE2FACE = os.path.join(FF_ROOT, "Face2Face")
FF_FACESWAP = os.path.join(FF_ROOT, "FaceSwap")
FF_FACESHIFTER = os.path.join(FF_ROOT, "FaceShifter")
FF_NEURALTEXTURES = os.path.join(FF_ROOT, "NeuralTextures")
FF_DEEPFAKEDETECTION = os.path.join(FF_ROOT, "DeepFakeDetection")
FF_CSV = os.path.join(FF_ROOT, "csv")

# Celeb-DF paths
CELEBDF_ROOT = os.path.join(DATASET_ROOT, "Celeb-DF")
CELEBDF_REAL = os.path.join(CELEBDF_ROOT, "Celeb-real")
CELEBDF_SYNTHESIS = os.path.join(CELEBDF_ROOT, "Celeb-synthesis")
CELEBDF_YOUTUBE = os.path.join(CELEBDF_ROOT, "YouTube-real")
CELEBDF_TEST_LIST = os.path.join(CELEBDF_ROOT, "List_of_testing_videos.txt")

# DFD paths
DFD_ROOT = os.path.join(DATASET_ROOT, "DFD")
DFD_MANIPULATED = os.path.join(DFD_ROOT, "DFD_manipulated_sequences")
DFD_ORIGINAL = os.path.join(DFD_ROOT, "DFD_original sequences")

# Output paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
FACES_DIR = os.path.join(OUTPUT_DIR, "faces")
SPLITS_DIR = os.path.join(OUTPUT_DIR, "splits")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
GRADCAM_DIR = os.path.join(OUTPUT_DIR, "gradcam")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Create output directories
for d in [OUTPUT_DIR, FRAMES_DIR, FACES_DIR, SPLITS_DIR, MODELS_DIR,
          RESULTS_DIR, FIGURES_DIR, GRADCAM_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Demographic Groups
# ============================================================
RACE_CATEGORIES = ["White", "Black", "Asian", "Other"]  # Simplified from FairFace
GENDER_CATEGORIES = ["Male", "Female"]
DEMOGRAPHIC_GROUPS = [
    f"{gender}-{race}" for gender in GENDER_CATEGORIES for race in RACE_CATEGORIES
]
# ['Male-White', 'Male-Black', 'Male-Asian', 'Male-Other',
#  'Female-White', 'Female-Black', 'Female-Asian', 'Female-Other']
NUM_GROUPS = len(DEMOGRAPHIC_GROUPS)  # 8
GROUP_TO_IDX = {g: i for i, g in enumerate(DEMOGRAPHIC_GROUPS)}
IDX_TO_GROUP = {i: g for i, g in enumerate(DEMOGRAPHIC_GROUPS)}

# FairFace race mapping (FairFace has 7 races, we map to 4)
FAIRFACE_RACE_MAP = {
    "White": "White",
    "Black": "Black",
    "East Asian": "Asian",
    "Southeast Asian": "Asian",
    "Indian": "Other",
    "Middle Eastern": "Other",
    "Latino_Hispanic": "Other",
}

# ============================================================
# Data Configuration
# ============================================================
IMAGE_SIZE = 224                  # MobileNetV2 native size (23% faster than 256)
FRAMES_PER_VIDEO = 30             # Frames to extract per video
FACE_DETECTION_BATCH_SIZE = 256   # Batch size for MTCNN (increased for GPU saturation)
MIN_FACE_SIZE = 60                # Minimum face size in pixels

# FF++ manipulation types for training
FF_FAKE_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]

# Dataset splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ============================================================
# Training Configuration
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Common training params
BATCH_SIZE = 32
NUM_WORKERS = 0   # Windows optimized (avoid multiprocessing spawn overhead)
PIN_MEMORY = True

# Teacher (XceptionNet)
TEACHER_EPOCHS = 50
TEACHER_LR = 1e-4
TEACHER_WEIGHT_DECAY = 1e-5

# Student Baseline (MobileNetV2)
STUDENT_EPOCHS = 50
STUDENT_LR = 1e-4
STUDENT_WEIGHT_DECAY = 1e-5

# Fair Distillation
DISTILL_EPOCHS = 100
DISTILL_LR = 1e-4
DISTILL_WEIGHT_DECAY = 1e-5
DISTILL_TEMPERATURE = 4.0         # Temperature for soft labels
ALPHA = 0.7                       # Weight for distillation loss (KL-div)
BETA = 0.2                        # Weight for fairness loss
GAMMA = 0.1                       # Weight for classification loss (BCE)
USE_SAM = True                    # Use Sharpness-Aware Minimization
SAM_RHO = 0.05                   # SAM perturbation radius

# Early stopping
PATIENCE = 10                    # Epochs without improvement before stopping
MIN_DELTA = 0.001                # Minimum improvement to reset patience

# ============================================================
# Evaluation Configuration
# ============================================================
# Fairness metric thresholds (targets from the research plan)
TARGET_AUC = 0.95
TARGET_FPR_GAP = 0.12             # ≤ 12% FPR gap between groups
TARGET_EO_GAP = 0.10              # ≤ 10% Equalized Odds gap
TARGET_DP_GAP = 0.10              # ≤ 10% Demographic Parity gap
TARGET_ACC_GAP = 0.08             # ≤ 8% Accuracy gap

# Cross-dataset target
TARGET_CROSS_AUC = 0.72

# Model size target
TARGET_MODEL_SIZE_MB = 15.0       # ≤ 15 MB
TARGET_INFERENCE_MS = 30.0        # ≤ 30 ms per image

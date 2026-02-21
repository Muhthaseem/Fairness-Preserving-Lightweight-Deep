
import json
import os

def create_retrain_v2():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🔄 Fair Student v2 — Retraining Notebook\n",
                    "### Optimized Training (Batch Size 128, 16GB VRAM)\n",
                    "This notebook retrains the fairness-aware MobileNetV2 student with a stronger fairness penalty (β=0.4) to reduce the FFPR gap.\n",
                    "\n",
                    "**Hardware Targets:** 16GB Dedicated VRAM, 24-core CPU.\n",
                    "\n",
                    "1. Setup & Imports\n",
                    "2. Configure v2 hyperparameters (Optimized BS=128)\n",
                    "3. Backup v1 checkpoints & load teacher\n",
                    "4. Create fresh student & data loaders (12 Workers)\n",
                    "5. **Train fair distillation v2** (~2-4 hours with optimization)\n",
                    "6. Plot training history\n",
                    "7. Evaluate on test + cross-datasets\n",
                    "8. Compare v1 vs v2"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## ⚙️ Cell 0 — Setup & Imports"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_0",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os, sys, time, shutil, warnings\n",
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "import torch.optim as optim\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import matplotlib.gridspec as gridspec\n",
                    "import seaborn as sns\n",
                    "from tqdm import tqdm\n",
                    "\n",
                    "# Add project root to path\n",
                    "PROJECT_ROOT = os.path.abspath(\".\")\n",
                    "if PROJECT_ROOT not in sys.path:\n",
                    "    sys.path.insert(0, PROJECT_ROOT)\n",
                    "\n",
                    "from src.config import (\n",
                    "    DEVICE, MODELS_DIR, SPLITS_DIR, FIGURES_DIR,\n",
                    "    DEMOGRAPHIC_GROUPS, BATCH_SIZE\n",
                    ")\n",
                    "from src.models.xception import build_teacher\n",
                    "from src.models.mobilenetv2 import build_student\n",
                    "from src.data.dataset import create_dataloaders, DeepfakeDataset\n",
                    "from src.training.train_distill import train_fair_distillation\n",
                    "\n",
                    "print('✅ Setup & Imports OK')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 🖥️ Cell 1 — GPU Check & v2 Config"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_1",
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('='*55)\n",
                    "print('  ENVIRONMENT')\n",
                    "print('='*55)\n",
                    "print(f'  PyTorch:      {torch.__version__}')\n",
                    "print(f'  Device:       {DEVICE}')\n",
                    "\n",
                    "if torch.cuda.is_available():\n",
                    "    gpu = torch.cuda.get_device_properties(0)\n",
                    "    print(f'  GPU:          {gpu.name}')\n",
                    "    print(f'  VRAM:         {gpu.total_memory / 1e9:.1f} GB')\n",
                    "else:\n",
                    "    print('  ⚠️  No GPU found!')\n",
                    "\n",
                    "# ── v2 Hyperparameters (AGGRESSIVE 16GB VRAM) ─────────────\n",
                    "V2_ALPHA       = 0.5    # distillation weight\n",
                    "V2_BETA        = 0.4    # fairness weight (KEY CHANGE)\n",
                    "V2_GAMMA       = 0.1    # classification weight\n",
                    "V2_TEMPERATURE = 4.0\n",
                    "V2_EPOCHS      = 50\n",
                    "V2_BATCH_SIZE  = 128    # Lowered from 256 for stable validation\n",
                    "V2_MODEL_NAME  = 'fair_student'\n",
                    "USE_CURRICULUM = False\n",
                    "\n",
                    "print() \n",
                    "print('='*55)\n",
                    "print('  v2 RETRAINING CONFIG (OPTIMIZED)')\n",
                    "print('='*55)\n",
                    "print(f'  Batch size:   {V2_BATCH_SIZE}')\n",
                    "print(f'  Workers:      4')\n",
                    "print(f'  Loss: L = {V2_ALPHA}·Ld + {V2_BETA}·Lf + {V2_GAMMA}·Lc')\n",
                    "\n",
                    "# Paths\n",
                    "TEACHER_CHECKPOINT = os.path.join(MODELS_DIR, 'xception_teacher_best.pth')\n",
                    "TRAIN_CSV = os.path.join(SPLITS_DIR, 'train.csv')\n",
                    "VAL_CSV   = os.path.join(SPLITS_DIR, 'val.csv')\n",
                    "TEST_CSV  = os.path.join(SPLITS_DIR, 'test.csv')\n",
                    "\n",
                    "torch.backends.cudnn.benchmark = True\n",
                    "print('\\ncuDNN benchmark mode: ON')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 💾 Cell 2 — Backup & Load Teacher"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_2",
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('--- Step 1: Backing up v1 checkpoints ---')\n",
                    "BACKUP_DIR = os.path.join(MODELS_DIR, 'v1_backup')\n",
                    "os.makedirs(BACKUP_DIR, exist_ok=True)\n",
                    "FAIR_FILES = ['fair_student_best_auc.pth', 'fair_student_best_fair.pth', 'fair_student_final.pth']\n",
                    "for fname in FAIR_FILES:\n",
                    "    src = os.path.join(MODELS_DIR, fname)\n",
                    "    if os.path.exists(src):\n",
                    "        shutil.copy2(src, os.path.join(BACKUP_DIR, fname))\n",
                    "        os.remove(src)\n",
                    "        print(f'  ✅ Backed up & Cleared: {fname}')\n",
                    "\n",
                    "print('\\n--- Step 2: Loading teacher ---')\n",
                    "teacher = build_teacher(pretrained=False, device=DEVICE)\n",
                    "ckpt = torch.load(TEACHER_CHECKPOINT, map_location=DEVICE, weights_only=False)\n",
                    "teacher.load_state_dict(ckpt['model_state_dict'])\n",
                    "teacher.eval()\n",
                    "print(f'  ✅ Teacher loaded — val AUC: {ckpt.get(\"val_auc\", \"N/A\")}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 📱 Cell 3 — Create Student & DataLoaders"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_3",
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('--- Loading Data (Parallel) ---')\n",
                    "loaders = create_dataloaders(TRAIN_CSV, VAL_CSV, batch_size=V2_BATCH_SIZE, num_workers=4)\n",
                    "train_loader = loaders['train']\n",
                    "val_loader   = loaders['val']\n",
                    "\n",
                    "assert not isinstance(train_loader, str), 'ERROR: Dataloader desync detected!'\n",
                    "\n",
                    "print('\\n--- Creating Student ---')\n",
                    "student = build_student(pretrained=True, device=DEVICE)\n",
                    "optimizer = optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-5)\n",
                    "scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)\n",
                    "\n",
                    "print(f'\\n✅ Ready: {len(train_loader)} batches (Size 128)')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 🚀 Cell 4 — Train Fair Distillation v2"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_4",
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('=================================================================')\n",
                    "print('  STARTING OPTIMIZED TRAINING (VRAM 16GB)')\n",
                    "print('=================================================================')\n",
                    "\n",
                    "results = train_fair_distillation(\n",
                    "    student=student, teacher=teacher,\n",
                    "    train_loader=train_loader, val_loader=val_loader,\n",
                    "    optimizer=optimizer, scheduler=scheduler,\n",
                    "    num_epochs=V2_EPOCHS, alpha=V2_ALPHA, beta=V2_BETA, gamma=V2_GAMMA,\n",
                    "    temperature=V2_TEMPERATURE, curriculum=USE_CURRICULUM,\n",
                    "    model_name=V2_MODEL_NAME, device=DEVICE\n",
                    ")\n",
                    "\n",
                    "print(f'\\n✅ TRAINING COMPLETE. Best Val AUC: {results[\"best_auc\"]:.4f}')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.8"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('retrain_v2.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print("Rebuilt retrain_v2.ipynb successfully.")

if __name__ == "__main__":
    create_retrain_v2()

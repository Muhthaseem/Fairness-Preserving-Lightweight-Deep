import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DEVICE, MODELS_DIR, IDX_TO_GROUP
from src.models.mobilenetv2 import build_student
from src.models.xception import build_teacher
from src.data.dataset import DeepfakeDataset

def get_fpr(labels, logits):
    scores = torch.sigmoid(logits).cpu().numpy()
    preds = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0

def check_split(split_name, csv_path):
    print(f"\n--- Checking {split_name} ---")
    dataset = DeepfakeDataset(csv_path, split="test") # no aug
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    
    # Load models
    fair_student = build_student(pretrained=False).to(DEVICE)
    fair_student.load_state_dict(torch.load(os.path.join(MODELS_DIR, "fair_student_best_auc.pth"), map_location=DEVICE, weights_only=False)["model_state_dict"])
    fair_student.eval()
    
    teacher = build_teacher().to(DEVICE)
    teacher.load_state_dict(torch.load(os.path.join(MODELS_DIR, "xception_teacher_best.pth"), map_location=DEVICE, weights_only=False)["model_state_dict"])
    teacher.eval()

    
    models = {"FairStudent": fair_student, "Teacher": teacher}
    
    for m_name, model in models.items():
        all_labels = []
        all_logits = []
        all_groups = []
        
        with torch.no_grad():
            for images, labels, groups in tqdm(loader, desc=f"{split_name} - {m_name}"):
                images = images.to(DEVICE)
                logits = model(images)
                all_labels.extend(labels.numpy())
                all_logits.extend(logits.cpu().numpy().flatten())
                all_groups.extend(groups.numpy())
        
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        all_groups = np.array(all_groups)
        
        results = []
        for g_id, g_name in IDX_TO_GROUP.items():
            mask = (all_groups == g_id)
            if mask.sum() == 0: continue
            fpr = get_fpr(all_labels[mask], torch.tensor(all_logits[mask]))
            results.append({"Group": g_name, "FPR": fpr})
        
        df = pd.DataFrame(results)
        print(f"\n{m_name} FPR on {split_name}:")
        print(df.to_string(index=False))

check_split("VAL", "outputs/splits/val.csv")
check_split("TEST", "outputs/splits/test.csv")

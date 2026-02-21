# Retraining Guide: Achieving FFPR Gap Target

## Why Retrain?

Our first fair distillation run used `beta=0.2` (fairness loss weight). Results:

| Metric | v1 Result | Target |
|--------|-----------|--------|
| **FFPR Gap** | 0.1995 | **<= 0.12** |
| AUC | 0.9497 | >= 0.95 |
| Model Size | 10.2 MB | <= 15 MB |

The FFPR gap is driven by **Male-Asian** having a much higher False Positive Rate (0.37) compared to other groups (~0.14-0.19). The model hasn't learned enough fairness — it needs stronger pressure during training.

**Post-hoc threshold calibration was attempted** but didn't generalize to the test set (thresholds overfit to validation data). The robust solution is to retrain with higher beta.

## What Changes

```
v1: L = 0.7*L_distill + 0.2*L_fairness + 0.1*L_cls
v2: L = 0.5*L_distill + 0.4*L_fairness + 0.1*L_cls
```

- **beta doubled** (0.2 -> 0.4): Forces the model to penalize FPR differences between groups more
- **alpha reduced** (0.7 -> 0.5): Slightly less emphasis on mimicking the teacher, more on fairness
- **gamma unchanged** (0.1): Same classification loss

### Expected Impact

| Metric | v1 (beta=0.2) | Expected v2 (beta=0.4) |
|--------|---------------|------------------------|
| **FFPR Gap** | 0.1995 | ~0.08-0.12 |
| AUC | 0.9497 | ~0.94-0.95 |
| Accuracy | 0.9189 | ~0.91-0.92 |
| FOAE Gap | 0.0702 | ~0.05-0.07 |

AUC may dip ~0.5%, but the fairness improvement should be significant.

## How to Retrain

### Prerequisites
- Teacher model trained: `outputs/models/xception_teacher_best.pth`
- Data splits exist: `outputs/splits/train.csv`, `val.csv`
- GPU available (RTX 4050 or similar)

### Step 1: Run the retraining script

```powershell
cd "D:\M3 Projects\DeepFake_Research"
python retrain_fair_student_v2.py
```

The script automatically:
1. **Backs up** old v1 checkpoints to `outputs/models/v1_backup/`
2. **Deletes** old fair_student checkpoint files
3. **Loads** the teacher model (frozen)
4. **Creates** a fresh MobileNetV2 student
5. **Trains** with the new hyperparameters (beta=0.4)
6. **Saves** new checkpoints with the same filenames

**Time estimate: ~12-24 hours** (same as original training)

### Step 2: Evaluate the new model

After training completes, run:

```powershell
python posthoc_evaluate.py
```

Or re-run the evaluation cells (Cells 8-10) in `deepfake_training.ipynb`.

### Step 3: Check results

Look at:
- `outputs/results/posthoc_evaluation_results.csv`
- Terminal output showing before/after fairness metrics

### If The Target Is Still Missed

If FFPR gap is still > 0.12 after beta=0.4:

1. **Try beta=0.6**: `V2_BETA = 0.6`, `V2_ALPHA = 0.3` in `retrain_fair_student_v2.py`
2. **Try curriculum learning**: Set `curriculum=True` in the script — this gradually increases beta during training
3. **Increase epochs**: Set `V2_EPOCHS = 100` to give the model more time to converge on fairness

## Files Created

| File | Purpose |
|------|---------|
| `retrain_fair_student_v2.py` | Standalone retraining script |
| `posthoc_threshold_calibration.py` | Post-hoc calibration (already tried) |
| `posthoc_evaluate.py` | Evaluation with calibrated thresholds |
| `RETRAINING_GUIDE.md` | This instruction file |

## Backup & Recovery

Your v1 checkpoints are safely backed up to `outputs/models/v1_backup/`:
- `fair_student_best_auc.pth`
- `fair_student_best_fair.pth`
- `fair_student_final.pth`

To restore v1 results, copy them back:
```powershell
Copy-Item "outputs\models\v1_backup\*" "outputs\models\" -Force
```

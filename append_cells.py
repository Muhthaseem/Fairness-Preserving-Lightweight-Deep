"""
append_cells.py — Appends evaluation, visualization, and Grad-CAM cells
to deepfake_training.ipynb
"""
import json, os

NB_PATH = r"D:\M3 Projects\DeepFake_Research\deepfake_training.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

def md(lines):
    return {"cell_type":"markdown","metadata":{},"source":lines if isinstance(lines,list) else [lines]}

def code(lines):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":lines if isinstance(lines,list) else [lines]}

new_cells = []

# ── Cell 7: Load all models for evaluation ──────────────────────────────────
new_cells.append(md("## 📋 Cell 7 — Load All Trained Models for Evaluation"))
new_cells.append(code("""\
def load_model(model, weights_path, device):
    if not os.path.exists(weights_path):
        print(f'[WARN] Not found: {weights_path}')
        return None
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

# Teacher
teacher_model = build_teacher(pretrained=True, device=DEVICE)
teacher_model = load_model(teacher_model, os.path.join(MODELS_DIR,'xception_teacher_best.pth'), DEVICE)

# Baseline student
baseline_model = build_student(pretrained=True, device=DEVICE)
baseline_model = load_model(baseline_model, os.path.join(MODELS_DIR,'mobilenetv2_baseline_best.pth'), DEVICE)

# Standard distillation (ablation)
std_model = build_student(pretrained=True, device=DEVICE)
std_model = load_model(std_model, os.path.join(MODELS_DIR,'std_distill_student_best_auc.pth'), DEVICE)

# Fair distillation (ours)
fair_model = build_student(pretrained=True, device=DEVICE)
fair_model = load_model(fair_model, os.path.join(MODELS_DIR,'fair_student_best_auc.pth'), DEVICE)

models = {
    'XceptionNet (Teacher)':       teacher_model,
    'MobileNetV2 (Baseline)':      baseline_model,
    'MobileNetV2 (Std Distill)':   std_model,
    'MobileNetV2 (Fair Distill)':  fair_model,
}
print('Models loaded:', [k for k,v in models.items() if v is not None])
"""))

# ── Cell 8: In-domain evaluation ────────────────────────────────────────────
new_cells.append(md("## 🧪 Cell 8 — In-Domain Evaluation (FF++ Test Set)"))
new_cells.append(code("""\
test_dataset = DeepfakeDataset(os.path.join(SPLITS_DIR,'test.csv'), split='test')
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

all_results = {}
for name, model in models.items():
    if model is None:
        print(f'[SKIP] {name} — no checkpoint')
        continue
    r = evaluate_model(model, test_loader, DEVICE, name)
    save_results(r, RESULTS_DIR, filename=name.replace(' ','_').replace('(','').replace(')',''))
    all_results[name] = r

print('\\n✅ In-domain evaluation complete')
"""))

# ── Cell 9: Cross-dataset evaluation ────────────────────────────────────────
new_cells.append(md("## 🌍 Cell 9 — Cross-Dataset Evaluation (Celeb-DF & DFD)"))
new_cells.append(code("""\
cross_models = {k: v for k, v in models.items() if v is not None}
cross_results = run_full_cross_dataset_evaluation(cross_models, device=DEVICE)
print('\\n✅ Cross-dataset evaluation complete')
"""))

# ── Cell 10: Ablation comparison table ──────────────────────────────────────
new_cells.append(md("## 📊 Cell 10 — Ablation Comparison Table"))
new_cells.append(code("""\
rows = []
for name, r in all_results.items():
    fm = r['fairness']['fairness']
    size_mb = sum(p.nelement()*p.element_size() for p in models[name].parameters()) / 1e6
    rows.append({
        'Method':          name,
        'AUC':             f\"{r['overall_auc']:.4f}\",
        'Accuracy':        f\"{r['overall_accuracy']:.4f}\",
        'FFPR Gap ↓':      f\"{fm['FFPR_gap']:.4f}\",
        'FOAE Gap ↓':      f\"{fm['FOAE_gap']:.4f}\",
        'FEO Gap ↓':       f\"{fm['FEO_gap']:.4f}\",
        'FDP Gap ↓':       f\"{fm['FDP_gap']:.4f}\",
        'Size (MB)':       f\"{size_mb:.1f}\",
        'Inference (ms)':  f\"{r['inference_ms']:.1f}\",
    })

df_ablation = pd.DataFrame(rows)

# Style the table
def highlight_best(s):
    is_best = s == s.min()
    return ['background-color: #d4edda; font-weight: bold' if v else '' for v in is_best]

styled = (df_ablation.style
    .set_caption('📋 Ablation Study — In-Domain (FF++) Results')
    .set_table_styles([{'selector':'caption','props':[('font-size','14px'),('font-weight','bold')]}])
    .apply(highlight_best, subset=['FFPR Gap ↓','FOAE Gap ↓','FEO Gap ↓','FDP Gap ↓'])
)
display(styled)

# Save
df_ablation.to_csv(os.path.join(RESULTS_DIR,'ablation_table.csv'), index=False)
print('Saved: outputs/results/ablation_table.csv')
"""))

# ── Cell 11: Per-group accuracy bar chart ───────────────────────────────────
new_cells.append(md("## 📈 Cell 11 — Per-Group Accuracy Bar Chart"))
new_cells.append(code("""\
groups = DEMOGRAPHIC_GROUPS
x = np.arange(len(groups))
width = 0.2
model_names = list(all_results.keys())
bar_colors = ['#3498db','#e74c3c','#f39c12','#2ecc71']

fig, ax = plt.subplots(figsize=(14, 6))
for i, (name, r) in enumerate(all_results.items()):
    accs = [r['fairness']['fairness']['per_group_accuracy'].get(g, 0) for g in groups]
    bars = ax.bar(x + i*width, accs, width, label=name, color=bar_colors[i % len(bar_colors)], alpha=0.85)

# Target line
ax.axhline(y=0.95, color='black', linestyle='--', linewidth=1.2, label='Target AUC (0.95)')

ax.set_xlabel('Demographic Group', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Per-Group Accuracy by Model', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * (len(all_results)-1)/2)
ax.set_xticklabels(groups, rotation=30, ha='right')
ax.set_ylim(0.5, 1.02)
ax.legend(loc='lower right', fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,'per_group_accuracy.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Saved: outputs/figures/per_group_accuracy.png')
"""))

# ── Cell 12: ROC curves per group ───────────────────────────────────────────
new_cells.append(md("## 📉 Cell 12 — Fairness Gap Summary Chart"))
new_cells.append(code("""\
metrics_labels = ['FFPR Gap', 'FOAE Gap', 'FEO Gap', 'FDP Gap']
metric_keys    = ['FFPR_gap', 'FOAE_gap', 'FEO_gap', 'FDP_gap']
targets        = [0.12, 0.08, 0.10, 0.10]

fig, axes = plt.subplots(1, len(metrics_labels), figsize=(16, 5))
fig.suptitle('Fairness Metrics Comparison (lower = fairer)', fontsize=13, fontweight='bold')

for ax, label, key, target in zip(axes, metrics_labels, metric_keys, targets):
    vals  = [all_results[n]['fairness']['fairness'][key] for n in all_results]
    names = [n.split('(')[-1].replace(')','') for n in all_results]
    colors_bar = ['#2ecc71' if v <= target else '#e74c3c' for v in vals]
    ax.bar(names, vals, color=colors_bar, alpha=0.85, edgecolor='white')
    ax.axhline(y=target, color='black', linestyle='--', linewidth=1.2, label=f'Target ({target})')
    ax.set_title(label, fontweight='bold')
    ax.set_ylabel('Gap')
    ax.set_ylim(0, max(vals)*1.3 + 0.02)
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,'fairness_gaps.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Saved: outputs/figures/fairness_gaps.png')
"""))

# ── Cell 13: Grad-CAM ───────────────────────────────────────────────────────
new_cells.append(md("## 🔥 Cell 13 — Grad-CAM Heatmaps per Demographic Group"))
new_cells.append(code("""\
from src.explainability.gradcam import generate_gradcam_per_group

if fair_model is not None:
    print('Generating Grad-CAM heatmaps for Fair Student...')
    generate_gradcam_per_group(
        fair_model,
        os.path.join(SPLITS_DIR, 'test.csv'),
        num_samples_per_group=3,
        output_dir=GRADCAM_DIR,
        device=DEVICE
    )
    # Display saved heatmaps
    import glob
    from PIL import Image
    heatmap_files = sorted(glob.glob(os.path.join(GRADCAM_DIR, '*.png')))[:8]
    if heatmap_files:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Grad-CAM Heatmaps — Fair Student (per demographic group)', fontsize=13, fontweight='bold')
        for ax, fpath in zip(axes.flat, heatmap_files):
            img = Image.open(fpath)
            ax.imshow(img)
            ax.set_title(os.path.basename(fpath).replace('.png',''), fontsize=8)
            ax.axis('off')
        for ax in axes.flat[len(heatmap_files):]:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR,'gradcam_grid.png'), dpi=150, bbox_inches='tight')
        plt.show()
else:
    print('[SKIP] Fair student model not loaded.')
"""))

# ── Cell 14: Model size & speed profiling ───────────────────────────────────
new_cells.append(md("## ⚡ Cell 14 — Model Size & Inference Speed Profiling"))
new_cells.append(code("""\
import time

dummy = torch.randn(1, 3, 256, 256).to(DEVICE)
WARMUP = 10
RUNS   = 100

rows = []
for name, model in models.items():
    if model is None:
        continue
    model.eval()
    size_mb = sum(p.nelement()*p.element_size() for p in model.parameters()) / 1e6
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(dummy)

    # Timed runs
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(RUNS):
            _ = model(dummy)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / RUNS * 1000

    rows.append({
        'Model':          name,
        'Params (M)':     f'{params_m:.1f}',
        'Size (MB)':      f'{size_mb:.1f}',
        'Inference (ms)': f'{elapsed_ms:.2f}',
        'FPS':            f'{1000/elapsed_ms:.0f}',
        '≤15 MB':         '✅' if size_mb <= 15 else '❌',
        '≤30 ms':         '✅' if elapsed_ms <= 30 else '❌',
    })

df_speed = pd.DataFrame(rows)
display(df_speed.style.set_caption('⚡ Model Efficiency Profiling').hide(axis='index'))
df_speed.to_csv(os.path.join(RESULTS_DIR,'speed_profiling.csv'), index=False)

# Bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Model Efficiency Comparison', fontsize=13, fontweight='bold')

names_short = [n.split('(')[-1].replace(')','') for n in df_speed['Model']]
sizes  = df_speed['Size (MB)'].astype(float)
speeds = df_speed['Inference (ms)'].astype(float)

ax1.barh(names_short, sizes, color=bar_colors[:len(sizes)], alpha=0.85)
ax1.axvline(x=15, color='red', linestyle='--', label='Target (15 MB)')
ax1.set_xlabel('Model Size (MB)')
ax1.set_title('Model Size')
ax1.legend()

ax2.barh(names_short, speeds, color=bar_colors[:len(speeds)], alpha=0.85)
ax2.axvline(x=30, color='red', linestyle='--', label='Target (30 ms)')
ax2.set_xlabel('Inference Time (ms)')
ax2.set_title('Inference Speed')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,'efficiency_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print('\\n✅ All done! Results saved to outputs/results/ and outputs/figures/')
"""))

# Append all new cells
nb["cells"].extend(new_cells)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Appended {len(new_cells)} cells. Total cells: {len(nb['cells'])}")

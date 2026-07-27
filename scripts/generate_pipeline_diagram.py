"""
Script to generate Figure 2: End-to-End Pipeline Diagram.
Uses Matplotlib to create a block-style flow chart.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow

def draw_box(ax, text, x, y, width=3, height=1, color='lightblue'):
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.1", 
                         linewidth=2, edgecolor='black', facecolor=color)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontweight='bold', fontsize=10)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

def main():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Row 1: Data & Preprocessing
    draw_box(ax, "FF++ Videos\n(Input Data)", 1, 8)
    draw_arrow(ax, 4.2, 8.5, 5, 8.5)
    
    draw_box(ax, "Frame Extraction\n(30 FPS)", 5, 8)
    draw_arrow(ax, 8.2, 8.5, 9, 8.5)
    
    draw_box(ax, "MTCNN Search\n& Alignment", 9, 8)
    draw_arrow(ax, 12.2, 8.5, 13, 8.5)
    
    draw_box(ax, "FairFace\nAnnotator", 13, 8, color='peachpuff')
    draw_arrow(ax, 16.2, 8.5, 17, 8.5)
    
    draw_box(ax, "Demographic\nLabels (8 Groups)", 17, 8, width=2.5, color='lightgreen')

    # Connecting row 1 to row 2
    draw_arrow(ax, 18, 8, 18, 5)

    # Row 2: Model Training
    draw_box(ax, "Stratified Split\n(70/15/15)", 16, 4)
    draw_arrow(ax, 16, 4.5, 14, 4.5)
    
    draw_box(ax, "Teacher Training\n(XceptionNet)", 11, 4, color='lightgray')
    draw_arrow(ax, 11, 4.5, 9, 4.5)
    
    draw_box(ax, "Fairness-Aware\nDistillation (FP-KD)", 5, 4, width=4, color='gold')
    draw_arrow(ax, 5, 4.5, 3, 4.5)
    
    draw_box(ax, "MobileNetV2\n(Fair Student)", 0.5, 4, color='skyblue')

    # Connecting row 2 to row 3
    draw_arrow(ax, 2, 4, 2, 1.5)

    # Row 3: Evaluation & Calibration
    draw_box(ax, "Multi-Metric\nEvaluation (AUC, Gaps)", 1, 0.5, width=4)
    draw_arrow(ax, 5.2, 1, 7, 1)
    
    draw_box(ax, "Post-hoc\nCalibration", 7, 0.5, color='lightsalmon')
    draw_arrow(ax, 10.2, 1, 12, 1)
    
    draw_box(ax, "Final Deployment\n& Explainability", 12, 0.5, width=4, color='lightgreen')

    plt.title("Figure 2: End-to-End Fairness-Preserving Pipeline Diagram", fontsize=16, fontweight='bold', pad=20)
    
    output_dir = 'd:\\M3 Projects\\DeepFake_Research\\outputs\\figures'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "fig2_pipeline_diagram.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 2 generated in outputs/figures/fig2_pipeline_diagram.png")

if __name__ == "__main__":
    main()

"""
SIFT Step 3.9: Total Detection from All Scales = 501 Keypoints
Shows the complete detection before filtering
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_step3_9_visual():
    """Create visual showing 501 total keypoints from all octaves before filtering"""
    
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Step 3.9: Total Detection from All Scales = 501 Keypoints', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Main axis
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # ============================================
    # PART 1: Three Octaves with Gaussian/DoG process
    # ============================================
    
    octaves = [
        ('OCTAVE 0', 'H × W', '640 × 480', 245, '#3498db', 12),
        ('OCTAVE 1', 'H/2 × W/2', '320 × 240', 178, '#e74c3c', 9.5),
        ('OCTAVE 2', 'H/4 × W/4', '160 × 120', 78, '#27ae60', 7),
    ]
    
    for name, size, pixels, count, color, y in octaves:
        # Octave box
        oct_box = FancyBboxPatch((0.5, y - 0.8), 2.5, 1.8, boxstyle="round,pad=0.05",
                                  facecolor=color, edgecolor='black', linewidth=2, alpha=0.9)
        ax.add_patch(oct_box)
        ax.text(1.75, y + 0.5, name, ha='center', fontsize=11, fontweight='bold', color='white')
        ax.text(1.75, y - 0.1, size, ha='center', fontsize=9, color='white')
        ax.text(1.75, y - 0.5, pixels, ha='center', fontsize=8, color='yellow')
        
        # Arrow to Gaussian
        ax.annotate('', xy=(3.3, y), xytext=(3.1, y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        # Gaussian scales box
        gauss_box = FancyBboxPatch((3.5, y - 0.6), 3, 1.4, boxstyle="round,pad=0.03",
                                    facecolor='#ecf0f1', edgecolor=color, linewidth=2)
        ax.add_patch(gauss_box)
        ax.text(5, y + 0.3, 'Gaussian Scales', ha='center', fontsize=9, fontweight='bold')
        ax.text(5, y - 0.2, 'σ₁ → σ₂ → σ₃ → σ₄', ha='center', fontsize=9, fontfamily='monospace')
        
        # Arrow to DoG
        ax.annotate('', xy=(6.8, y), xytext=(6.6, y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        # DoG box
        dog_box = FancyBboxPatch((7, y - 0.6), 2.2, 1.4, boxstyle="round,pad=0.03",
                                  facecolor='#fdebd0', edgecolor='#f39c12', linewidth=2)
        ax.add_patch(dog_box)
        ax.text(8.1, y + 0.3, 'DoG', ha='center', fontsize=10, fontweight='bold')
        ax.text(8.1, y - 0.2, '3 images', ha='center', fontsize=9)
        
        # Arrow to 26-neighbor
        ax.annotate('', xy=(9.5, y), xytext=(9.3, y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        # 26-neighbor box
        nbr_box = FancyBboxPatch((9.7, y - 0.6), 2.8, 1.4, boxstyle="round,pad=0.03",
                                  facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=2)
        ax.add_patch(nbr_box)
        ax.text(11.1, y + 0.3, '26-Neighbor', ha='center', fontsize=9, fontweight='bold')
        ax.text(11.1, y - 0.2, 'Extrema', ha='center', fontsize=9)
        
        # Arrow to count
        ax.annotate('', xy=(12.8, y), xytext=(12.6, y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        # Keypoint count box
        count_box = FancyBboxPatch((13, y - 0.6), 2.5, 1.4, boxstyle="round,pad=0.03",
                                    facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(count_box)
        ax.text(14.25, y + 0.2, f'{count}', ha='center', fontsize=16, fontweight='bold', color='white')
        ax.text(14.25, y - 0.3, 'keypoints', ha='center', fontsize=9, color='white')
    
    # ============================================
    # Arrows to combine
    # ============================================
    for y in [12, 9.5, 7]:
        ax.annotate('', xy=(14.25, 5.5), xytext=(14.25, y - 1),
                   arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2,
                                  connectionstyle="arc3,rad=0.1"))
    
    # ============================================
    # COMBINE box
    # ============================================
    combine_box = FancyBboxPatch((11.5, 4), 5.5, 1.8, boxstyle="round,pad=0.05",
                                  facecolor='#9b59b6', edgecolor='black', linewidth=3)
    ax.add_patch(combine_box)
    ax.text(14.25, 5.3, 'COMBINE ALL OCTAVES', ha='center', fontsize=11, fontweight='bold', color='white')
    ax.text(14.25, 4.6, '245 + 178 + 78 = 501', ha='center', fontsize=12, fontweight='bold', color='yellow')
    ax.text(14.25, 4.2, 'keypoints', ha='center', fontsize=10, color='white')
    
    # ============================================
    # Arrow to total
    # ============================================
    ax.annotate('', xy=(8.5, 3.5), xytext=(11.4, 4.5),
               arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=3))
    
    # ============================================
    # TOTAL 501 box (Step 3 Complete)
    # ============================================
    total_box = FancyBboxPatch((4, 2), 9, 2, boxstyle="round,pad=0.05",
                                facecolor='#2c3e50', edgecolor='black', linewidth=3)
    ax.add_patch(total_box)
    ax.text(8.5, 3.5, 'STEP 3 COMPLETE: 26-Neighbor Extrema Detection', 
           ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(8.5, 2.7, '501 KEYPOINTS DETECTED', ha='center', fontsize=18, 
           fontweight='bold', color='#2ecc71')
    ax.text(8.5, 2.2, '(Before filtering stages 2, 3, 4)', ha='center', fontsize=10, color='#bdc3c7')
    
    # ============================================
    # Next steps preview
    # ============================================
    ax.text(8.5, 1.2, '↓ Next: Filter through Stage 2, 3, 4 → 308 final keypoints', 
           ha='center', fontsize=11, style='italic', color='#7f8c8d')
    
    # ============================================
    # Summary box on left
    # ============================================
    summary_box = FancyBboxPatch((0.5, 2), 3, 4, boxstyle="round,pad=0.05",
                                  facecolor='#f8f9fa', edgecolor='#34495e', linewidth=2)
    ax.add_patch(summary_box)
    
    ax.text(2, 5.5, 'Detection Summary', ha='center', fontsize=11, fontweight='bold')
    
    summary_data = [
        ('Octave 0:', '245', '#3498db'),
        ('Octave 1:', '178', '#e74c3c'),
        ('Octave 2:', ' 78', '#27ae60'),
        ('─' * 12, '', 'gray'),
        ('TOTAL:', '501', '#2c3e50'),
    ]
    
    y_pos = 5.0
    for label, value, color in summary_data:
        if value:
            ax.text(1.2, y_pos, label, ha='left', fontsize=10, color='black')
            ax.text(2.8, y_pos, value, ha='right', fontsize=11, fontweight='bold', color=color)
        else:
            ax.text(2, y_pos, label, ha='center', fontsize=9, color=color)
        y_pos -= 0.5
    
    # Percentage breakdown
    ax.text(2, 2.7, 'Distribution:', ha='center', fontsize=9, fontweight='bold')
    ax.text(2, 2.3, 'Oct 0: 49%', ha='center', fontsize=8, color='#3498db')
    ax.text(2, 2.0, 'Oct 1: 35%', ha='center', fontsize=8, color='#e74c3c')
    ax.text(2, 1.7, 'Oct 2: 16%', ha='center', fontsize=8, color='#27ae60')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUT_DIR, 'sift_step3_9_total_501.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_501_to_308_flow():
    """Create visual showing 501 → filtering → 308"""
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('501 Keypoints (Step 3) → Filtering → 308 Final', 
                fontsize=18, fontweight='bold', pad=15)
    
    # Step 3 result
    step3_box = FancyBboxPatch((0.5, 7), 4, 2.5, boxstyle="round,pad=0.05",
                                facecolor='#9b59b6', edgecolor='black', linewidth=3)
    ax.add_patch(step3_box)
    ax.text(2.5, 8.8, 'Step 3 Complete', ha='center', fontsize=11, fontweight='bold', color='white')
    ax.text(2.5, 8.2, '26-Neighbor Extrema', ha='center', fontsize=10, color='white')
    ax.text(2.5, 7.5, '501', ha='center', fontsize=24, fontweight='bold', color='yellow')
    ax.text(2.5, 7.1, 'keypoints', ha='center', fontsize=10, color='white')
    
    # Arrow
    ax.annotate('', xy=(5, 8.2), xytext=(4.6, 8.2),
               arrowprops=dict(arrowstyle='->', color='black', lw=3))
    
    # Stage 2
    stage2_box = FancyBboxPatch((5.2, 7), 3, 2.5, boxstyle="round,pad=0.05",
                                 facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(stage2_box)
    ax.text(6.7, 8.8, 'Stage 2', ha='center', fontsize=10, fontweight='bold', color='white')
    ax.text(6.7, 8.3, 'Low Contrast', ha='center', fontsize=9, color='white')
    ax.text(6.7, 7.8, '|D(x̂)| < 0.03', ha='center', fontsize=8, color='yellow', fontfamily='monospace')
    ax.text(6.7, 7.2, '−114', ha='center', fontsize=14, fontweight='bold', color='#ffcccc')
    
    # Arrow
    ax.annotate('', xy=(8.5, 8.2), xytext=(8.3, 8.2),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(8.4, 8.6, '387', fontsize=10, fontweight='bold', color='#e74c3c')
    
    # Stage 3
    stage3_box = FancyBboxPatch((8.7, 7), 3, 2.5, boxstyle="round,pad=0.05",
                                 facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax.add_patch(stage3_box)
    ax.text(10.2, 8.8, 'Stage 3', ha='center', fontsize=10, fontweight='bold', color='white')
    ax.text(10.2, 8.3, 'Edge Response', ha='center', fontsize=9, color='white')
    ax.text(10.2, 7.8, 'Tr²/Det > 12.1', ha='center', fontsize=8, color='yellow', fontfamily='monospace')
    ax.text(10.2, 7.2, '−54', ha='center', fontsize=14, fontweight='bold', color='#fff3cd')
    
    # Arrow
    ax.annotate('', xy=(12, 8.2), xytext=(11.8, 8.2),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(11.9, 8.6, '333', fontsize=10, fontweight='bold', color='#f39c12')
    
    # Stage 4
    stage4_box = FancyBboxPatch((12.2, 7), 3, 2.5, boxstyle="round,pad=0.05",
                                 facecolor='#27ae60', edgecolor='black', linewidth=2)
    ax.add_patch(stage4_box)
    ax.text(13.7, 8.8, 'Stage 4', ha='center', fontsize=10, fontweight='bold', color='white')
    ax.text(13.7, 8.3, 'Sub-pixel', ha='center', fontsize=9, color='white')
    ax.text(13.7, 7.8, '|offset| > 0.5', ha='center', fontsize=8, color='yellow', fontfamily='monospace')
    ax.text(13.7, 7.2, '−25', ha='center', fontsize=14, fontweight='bold', color='#d5f5e3')
    
    # Arrow to final
    ax.annotate('', xy=(13.7, 6), xytext=(13.7, 6.9),
               arrowprops=dict(arrowstyle='->', color='#27ae60', lw=3))
    
    # Final result
    final_box = FancyBboxPatch((10.5, 3.5), 5, 2.3, boxstyle="round,pad=0.05",
                                facecolor='#2c3e50', edgecolor='black', linewidth=3)
    ax.add_patch(final_box)
    ax.text(13, 5.3, 'FINAL RESULT', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(13, 4.5, '308', ha='center', fontsize=28, fontweight='bold', color='#2ecc71')
    ax.text(13, 3.9, 'stable keypoints', ha='center', fontsize=10, color='white')
    
    # Summary table
    table_box = FancyBboxPatch((0.5, 1.5), 9, 4.5, boxstyle="round,pad=0.05",
                                facecolor='#f8f9fa', edgecolor='#34495e', linewidth=2)
    ax.add_patch(table_box)
    
    ax.text(5, 5.5, 'Filtering Summary: 501 → 308', ha='center', fontsize=12, fontweight='bold')
    
    table_data = [
        ('Step 3 (Detection):', '501', '100%', '#9b59b6'),
        ('After Stage 2:', '387', '77.2%', '#e74c3c'),
        ('After Stage 3:', '333', '66.5%', '#f39c12'),
        ('After Stage 4:', '308', '61.5%', '#27ae60'),
    ]
    
    y = 4.8
    for label, count, pct, color in table_data:
        ax.text(1, y, label, ha='left', fontsize=10)
        ax.text(6, y, count, ha='center', fontsize=12, fontweight='bold', color=color)
        ax.text(8, y, pct, ha='right', fontsize=10, color='gray')
        y -= 0.7
    
    ax.text(5, 1.8, 'Total removed: 114 + 54 + 25 = 193 keypoints (38.5%)', 
           ha='center', fontsize=10, color='#c0392b')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUT_DIR, 'sift_501_to_308_flow.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("Generating Step 3.9 visualizations (501 total keypoints)...")
    
    create_step3_9_visual()
    create_501_to_308_flow()
    
    print("\nDone!")

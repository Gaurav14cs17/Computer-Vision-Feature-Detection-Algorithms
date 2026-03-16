"""
SIFT Keypoint Filtering Flow:
Octave 0, 1, 2 → 1300 points → Stage 2, 3, 4 → 308 final points

Shows the complete flow from octave detection to final filtering
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_octave_to_filtering_flow():
    """Create visual showing octave detection → filtering stages"""
    
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('SIFT Keypoint Flow: Octaves → Filtering Stages\n501 Initial → 308 Final', 
                fontsize=18, fontweight='bold', pad=20)
    
    # ============================================
    # PART 1: Three Octaves (Top Section)
    # ============================================
    
    # Octave 0
    oct0_box = FancyBboxPatch((1, 10.5), 4, 2.5, boxstyle="round,pad=0.05",
                               facecolor='#3498db', edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(oct0_box)
    ax.text(3, 12.3, 'OCTAVE 0', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(3, 11.7, 'H × W (Full)', ha='center', fontsize=10, color='white')
    ax.text(3, 11.1, '640 × 480', ha='center', fontsize=9, color='white')
    
    # Octave 0 count
    oct0_count = FancyBboxPatch((1.5, 10.6), 3, 0.6, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='#3498db', linewidth=2)
    ax.add_patch(oct0_count)
    ax.text(3, 10.9, '245 keypoints', ha='center', fontsize=11, fontweight='bold', color='#3498db')
    
    # Octave 1
    oct1_box = FancyBboxPatch((6.5, 10.5), 4, 2.5, boxstyle="round,pad=0.05",
                               facecolor='#e74c3c', edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(oct1_box)
    ax.text(8.5, 12.3, 'OCTAVE 1', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(8.5, 11.7, 'H/2 × W/2', ha='center', fontsize=10, color='white')
    ax.text(8.5, 11.1, '320 × 240', ha='center', fontsize=9, color='white')
    
    # Octave 1 count
    oct1_count = FancyBboxPatch((7, 10.6), 3, 0.6, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(oct1_count)
    ax.text(8.5, 10.9, '178 keypoints', ha='center', fontsize=11, fontweight='bold', color='#e74c3c')
    
    # Octave 2
    oct2_box = FancyBboxPatch((12, 10.5), 4, 2.5, boxstyle="round,pad=0.05",
                               facecolor='#27ae60', edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(oct2_box)
    ax.text(14, 12.3, 'OCTAVE 2', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(14, 11.7, 'H/4 × W/4', ha='center', fontsize=10, color='white')
    ax.text(14, 11.1, '160 × 120', ha='center', fontsize=9, color='white')
    
    # Octave 2 count
    oct2_count = FancyBboxPatch((12.5, 10.6), 3, 0.6, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(oct2_count)
    ax.text(14, 10.9, '78 keypoints', ha='center', fontsize=11, fontweight='bold', color='#27ae60')
    
    # ============================================
    # Arrows from octaves to combine
    # ============================================
    ax.annotate('', xy=(8.5, 9.5), xytext=(3, 10.4),
               arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
    ax.annotate('', xy=(8.5, 9.5), xytext=(8.5, 10.4),
               arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
    ax.annotate('', xy=(8.5, 9.5), xytext=(14, 10.4),
               arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
    
    # ============================================
    # Combine box (Stage 1 complete)
    # ============================================
    combine_box = FancyBboxPatch((5.5, 8), 6, 1.5, boxstyle="round,pad=0.05",
                                  facecolor='#9b59b6', edgecolor='black', linewidth=3, alpha=0.9)
    ax.add_patch(combine_box)
    ax.text(8.5, 9, 'STEP 3: 26-Neighbor Extrema (All Octaves)', ha='center', 
           fontsize=11, fontweight='bold', color='white')
    ax.text(8.5, 8.4, '245 + 178 + 78 = 501 keypoints', ha='center', 
           fontsize=12, fontweight='bold', color='yellow')
    
    # ============================================
    # Arrow to filtering stages
    # ============================================
    ax.annotate('', xy=(8.5, 7), xytext=(8.5, 7.9),
               arrowprops=dict(arrowstyle='->', color='black', lw=3))
    
    # ============================================
    # PART 2: Filtering Stages (Bottom Section)
    # ============================================
    
    # Stage 2: Low Contrast
    stage2_box = FancyBboxPatch((1, 4), 5, 2.8, boxstyle="round,pad=0.05",
                                 facecolor='#e74c3c', edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(stage2_box)
    ax.text(3.5, 6.3, 'STAGE 2', ha='center', fontsize=11, fontweight='bold', color='white')
    ax.text(3.5, 5.7, 'Low Contrast Removal', ha='center', fontsize=10, color='white')
    ax.text(3.5, 5.1, '|D(x̂)| < 0.03', ha='center', fontsize=10, color='yellow', 
           fontfamily='monospace')
    
    # Removed count
    ax.text(3.5, 4.5, '−114 removed', ha='center', fontsize=12, fontweight='bold', color='#ffcccc')
    
    # Remaining count
    stage2_remain = FancyBboxPatch((1.5, 4.1), 4, 0.6, boxstyle="round,pad=0.02",
                                    facecolor='white', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(stage2_remain)
    ax.text(3.5, 4.4, '387 remaining', ha='center', fontsize=11, fontweight='bold', color='#e74c3c')
    
    # Stage 3: Edge Response
    stage3_box = FancyBboxPatch((7.5, 4), 5, 2.8, boxstyle="round,pad=0.05",
                                 facecolor='#f39c12', edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(stage3_box)
    ax.text(10, 6.3, 'STAGE 3', ha='center', fontsize=11, fontweight='bold', color='white')
    ax.text(10, 5.7, 'Edge Response Removal', ha='center', fontsize=10, color='white')
    ax.text(10, 5.1, 'Tr(H)²/Det(H) > 12.1', ha='center', fontsize=10, color='yellow',
           fontfamily='monospace')
    
    # Removed count
    ax.text(10, 4.5, '−54 removed', ha='center', fontsize=12, fontweight='bold', color='#fff3cd')
    
    # Remaining count
    stage3_remain = FancyBboxPatch((8, 4.1), 4, 0.6, boxstyle="round,pad=0.02",
                                    facecolor='white', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(stage3_remain)
    ax.text(10, 4.4, '333 remaining', ha='center', fontsize=11, fontweight='bold', color='#f39c12')
    
    # Stage 4: Sub-pixel
    stage4_box = FancyBboxPatch((14, 4), 5, 2.8, boxstyle="round,pad=0.05",
                                 facecolor='#27ae60', edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(stage4_box)
    ax.text(16.5, 6.3, 'STAGE 4', ha='center', fontsize=11, fontweight='bold', color='white')
    ax.text(16.5, 5.7, 'Sub-pixel Refinement', ha='center', fontsize=10, color='white')
    ax.text(16.5, 5.1, '|offset| > 0.5', ha='center', fontsize=10, color='yellow',
           fontfamily='monospace')
    
    # Removed count
    ax.text(16.5, 4.5, '−25 removed', ha='center', fontsize=12, fontweight='bold', color='#d5f5e3')
    
    # Remaining count
    stage4_remain = FancyBboxPatch((14.5, 4.1), 4, 0.6, boxstyle="round,pad=0.02",
                                    facecolor='white', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(stage4_remain)
    ax.text(16.5, 4.4, '308 remaining', ha='center', fontsize=11, fontweight='bold', color='#27ae60')
    
    # ============================================
    # Arrows between stages
    # ============================================
    # From combine to stage 2
    ax.annotate('', xy=(3.5, 6.9), xytext=(7, 7),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    ax.text(5.2, 7.3, '501', fontsize=10, fontweight='bold', color='#9b59b6')
    
    # Stage 2 to Stage 3
    ax.annotate('', xy=(7.4, 5.4), xytext=(6.1, 5.4),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    ax.text(6.7, 5.7, '387', fontsize=10, fontweight='bold', color='#e74c3c')
    
    # Stage 3 to Stage 4
    ax.annotate('', xy=(13.9, 5.4), xytext=(12.6, 5.4),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    ax.text(13.2, 5.7, '333', fontsize=10, fontweight='bold', color='#f39c12')
    
    # ============================================
    # Final result box
    # ============================================
    final_box = FancyBboxPatch((6, 0.8), 8, 2.5, boxstyle="round,pad=0.05",
                                facecolor='#2c3e50', edgecolor='black', linewidth=3)
    ax.add_patch(final_box)
    ax.text(10, 2.8, 'FINAL RESULT', ha='center', fontsize=14, fontweight='bold', color='white')
    ax.text(10, 2.1, '308 Stable Keypoints', ha='center', fontsize=16, fontweight='bold', color='#2ecc71')
    ax.text(10, 1.3, '(23.7% of initial 1300)', ha='center', fontsize=11, color='#bdc3c7')
    
    # Arrow to final
    ax.annotate('', xy=(10, 3.4), xytext=(16.5, 3.9),
               arrowprops=dict(arrowstyle='->', color='#27ae60', lw=3))
    ax.text(13.5, 3.5, '308', fontsize=11, fontweight='bold', color='#27ae60')
    
    # ============================================
    # Progress bar at very bottom
    # ============================================
    bar_y = 0.3
    bar_width = 16
    bar_x = 2
    
    # Background bar (501)
    bg_bar = FancyBboxPatch((bar_x, bar_y), bar_width, 0.4, boxstyle="round,pad=0.01",
                             facecolor='#ecf0f1', edgecolor='black')
    ax.add_patch(bg_bar)
    
    # Stage markers
    markers = [
        (0, '#9b59b6', '501'),
        (387/501, '#e74c3c', '387'),
        (333/501, '#f39c12', '333'),
        (308/501, '#27ae60', '308'),
    ]
    
    # Remaining bar (308/501)
    ratio = 308 / 501
    remain_bar = FancyBboxPatch((bar_x, bar_y), bar_width * ratio, 0.4, 
                                 boxstyle="round,pad=0.01",
                                 facecolor='#27ae60', edgecolor='black')
    ax.add_patch(remain_bar)
    
    ax.text(bar_x - 0.3, bar_y + 0.2, '0', ha='right', fontsize=9)
    ax.text(bar_x + bar_width + 0.3, bar_y + 0.2, '501', ha='left', fontsize=9)
    ax.text(bar_x + bar_width * ratio, bar_y + 0.6, '↑308', ha='center', fontsize=9, 
           fontweight='bold', color='#27ae60')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUT_DIR, 'sift_octave_filtering_flow.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_detailed_breakdown():
    """Create detailed breakdown table with per-octave filtering"""
    
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('Detailed Keypoint Filtering Breakdown by Octave', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Table data
    # Distribution based on 501 total keypoints
    data = [
        # Stage, Total, Oct0, Oct1, Oct2, Condition
        ('Step 3 (26-nbr)', 501, 245, 178, 78, 'max/min in 3×3×3'),
        ('Stage 2: Contrast', 387, 190, 138, 59, '|D(x̂)| < 0.03'),
        ('Stage 3: Edge', 333, 163, 119, 51, 'Tr²/Det > 12.1'),
        ('Stage 4: Refine', 308, 151, 110, 47, '|offset| > 0.5'),
    ]
    
    # Column headers
    headers = ['Stage', 'Total', 'Oct 0', 'Oct 1', 'Oct 2', 'Removed', 'Condition']
    col_x = [1.5, 4.5, 6.5, 8.5, 10.5, 12.5, 15.5]
    
    # Header row
    header_bg = FancyBboxPatch((0.3, 10), 17.4, 1.2, boxstyle="round,pad=0.02",
                                facecolor='#34495e', edgecolor='black')
    ax.add_patch(header_bg)
    
    for header, x in zip(headers, col_x):
        ax.text(x, 10.6, header, ha='center', va='center', fontsize=11, 
               fontweight='bold', color='white')
    
    # Data rows
    colors = ['#9b59b6', '#e74c3c', '#f39c12', '#27ae60']
    removed = ['—', '−114', '−54', '−25']
    
    for idx, (row, color, rem) in enumerate(zip(data, colors, removed)):
        y = 8.5 - idx * 2
        
        # Row background
        row_bg = FancyBboxPatch((0.3, y - 0.7), 17.4, 1.6, boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor='black', alpha=0.2)
        ax.add_patch(row_bg)
        
        # Stage name
        stage_box = FancyBboxPatch((0.5, y - 0.5), 2.5, 1.2, boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(stage_box)
        ax.text(col_x[0], y + 0.1, row[0], ha='center', va='center', fontsize=9, 
               fontweight='bold', color='white')
        
        # Total
        total_box = FancyBboxPatch((3.8, y - 0.4), 1.4, 1, boxstyle="round,pad=0.02",
                                    facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(total_box)
        ax.text(col_x[1], y + 0.1, str(row[1]), ha='center', va='center', fontsize=12, 
               fontweight='bold', color=color)
        
        # Octave counts
        ax.text(col_x[2], y + 0.1, str(row[2]), ha='center', va='center', fontsize=11, 
               color='#3498db', fontweight='bold')
        ax.text(col_x[3], y + 0.1, str(row[3]), ha='center', va='center', fontsize=11, 
               color='#e74c3c', fontweight='bold')
        ax.text(col_x[4], y + 0.1, str(row[4]), ha='center', va='center', fontsize=11, 
               color='#27ae60', fontweight='bold')
        
        # Removed
        if rem != '—':
            ax.text(col_x[5], y + 0.1, rem, ha='center', va='center', fontsize=11, 
                   fontweight='bold', color='#c0392b')
        else:
            ax.text(col_x[5], y + 0.1, rem, ha='center', va='center', fontsize=11, color='gray')
        
        # Condition
        ax.text(col_x[6], y + 0.1, row[5], ha='center', va='center', fontsize=9, 
               fontfamily='monospace')
    
    # Summary section
    summary_bg = FancyBboxPatch((0.3, 0.5), 17.4, 2, boxstyle="round,pad=0.02",
                                 facecolor='#2c3e50', edgecolor='black', linewidth=2)
    ax.add_patch(summary_bg)
    
    ax.text(9, 2, 'SUMMARY', ha='center', fontsize=14, fontweight='bold', color='white')
    ax.text(9, 1.3, 'Total Removed: 114 + 54 + 25 = 193 keypoints', ha='center', 
           fontsize=11, color='#e74c3c')
    ax.text(9, 0.7, 'Final Keypoints: 308  |  Retention Rate: 61.5%', ha='center', 
           fontsize=12, fontweight='bold', color='#2ecc71')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUT_DIR, 'sift_octave_filtering_detail.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_funnel_diagram():
    """Create funnel diagram showing keypoint reduction"""
    
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('Keypoint Filtering Funnel: 501 → 308', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Funnel stages (width proportional to keypoint count)
    stages = [
        ('Octaves 0+1+2\n26-Neighbor Extrema', 501, '#9b59b6', 11),
        ('After Stage 2\nLow Contrast (−114)', 387, '#e74c3c', 8.5),
        ('After Stage 3\nEdge Response (−54)', 333, '#f39c12', 6),
        ('After Stage 4\nSub-pixel (−25)', 308, '#27ae60', 3.5),
    ]
    
    max_count = 501
    center_x = 7
    
    for label, count, color, y in stages:
        # Width proportional to count
        width = 10 * (count / max_count)
        
        # Main box
        box = FancyBboxPatch((center_x - width/2, y - 1), width, 2, 
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=2, alpha=0.85)
        ax.add_patch(box)
        
        # Label
        ax.text(center_x, y + 0.3, label, ha='center', va='center', 
               fontsize=10, color='white', fontweight='bold')
        
        # Count
        ax.text(center_x, y - 0.5, f'{count}', ha='center', va='center', 
               fontsize=16, color='yellow', fontweight='bold')
        
        # Side count indicator
        ax.text(center_x + width/2 + 0.5, y, f'← {count}', ha='left', va='center',
               fontsize=11, fontweight='bold', color=color)
    
    # Arrows between stages
    for y in [9.8, 7.3, 4.8]:
        ax.annotate('', xy=(7, y - 0.8), xytext=(7, y + 0.3),
                   arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    
    # Final box
    final_box = FancyBboxPatch((4, 0.5), 6, 1.5, boxstyle="round,pad=0.05",
                                facecolor='#2c3e50', edgecolor='black', linewidth=3)
    ax.add_patch(final_box)
    ax.text(7, 1.5, 'FINAL: 308 Keypoints', ha='center', fontsize=14, 
           fontweight='bold', color='#2ecc71')
    ax.text(7, 0.9, '23.7% retention', ha='center', fontsize=11, color='#bdc3c7')
    
    # Arrow to final
    ax.annotate('', xy=(7, 2.1), xytext=(7, 2.4),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    
    plt.tight_layout()
    
    output_path = os.path.join(OUT_DIR, 'sift_filtering_funnel.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_5x5_filtering_example():
    """Create 5x5 integer example showing filtering through all stages"""
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('5×5 Integer Matrix: Complete Filtering Flow', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Sample 5x5 DoG matrix (integers)
    dog = np.array([
        [ 3,  4,  2, -1,  2],
        [ 5,  7,  5,  3,  0],
        [ 4, 12,  6,  4,  3],
        [ 3,  6,  4,  2,  1],
        [ 2,  3,  2,  0, -1]
    ], dtype=np.int32)
    
    # Keypoint at (2, 1), value = 12
    y, x = 2, 1
    center_val = dog[y, x]
    
    # Calculate derivatives
    Dx = (dog[y, x+1] - dog[y, x-1]) / 2  # (6 - 4) / 2 = 1
    Dy = (dog[y+1, x] - dog[y-1, x]) / 2  # (6 - 7) / 2 = -0.5
    
    Dxx = dog[y, x+1] + dog[y, x-1] - 2 * dog[y, x]  # 6 + 4 - 24 = -14
    Dyy = dog[y+1, x] + dog[y-1, x] - 2 * dog[y, x]  # 6 + 7 - 24 = -11
    Dxy = (dog[y+1, x+1] - dog[y+1, x-1] - dog[y-1, x+1] + dog[y-1, x-1]) / 4  # (4-3-5+5)/4 = 0.25
    
    det_H = Dxx * Dyy - Dxy * Dxy  # (-14)*(-11) - 0.0625 = 153.9375
    trace_H = Dxx + Dyy  # -14 + -11 = -25
    
    offset_x = -(Dyy * Dx - Dxy * Dy) / det_H
    offset_y = -(Dxx * Dy - Dxy * Dx) / det_H
    D_refined = center_val + 0.5 * (Dx * offset_x + Dy * offset_y)
    
    ratio = (trace_H ** 2) / det_H
    
    # Create 4 subplots for each stage
    stages_info = [
        ('Stage 1: 26-Neighbor\nExtrema Detection', '#9b59b6', 
         f'Center = {center_val}\nMax neighbor = 8\n{center_val} > 8 → MAXIMUM ✓'),
        ('Stage 2: Low Contrast\nRemoval', '#e74c3c',
         f'|D(x̂)| = |{D_refined:.1f}|\nThreshold = 3\n{abs(D_refined):.1f} ≥ 3 → KEEP ✓'),
        ('Stage 3: Edge Response\nRemoval', '#f39c12',
         f'Ratio = {ratio:.2f}\nThreshold = 12.1\n{ratio:.2f} ≤ 12.1 → KEEP ✓'),
        ('Stage 4: Sub-pixel\nRefinement', '#27ae60',
         f'offset = ({offset_x:.3f}, {offset_y:.3f})\n|offset| ≤ 0.5\n→ KEEP ✓'),
    ]
    
    for idx, (title, color, result) in enumerate(stages_info):
        ax = fig.add_subplot(1, 4, idx + 1)
        
        # Plot matrix
        im = ax.imshow(dog, cmap='RdBu', aspect='equal')
        
        # Add grid
        for i in range(6):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
            ax.axvline(i - 0.5, color='black', linewidth=0.5)
        
        # Add values
        for i in range(5):
            for j in range(5):
                text_color = 'white' if abs(dog[i, j]) > 6 else 'black'
                ax.text(j, i, str(dog[i, j]), ha='center', va='center', 
                       fontsize=12, color=text_color, fontweight='bold')
        
        # Highlight keypoint
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, 
                             edgecolor=color, linewidth=4)
        ax.add_patch(rect)
        
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([f'{i}' for i in range(5)], fontsize=8)
        ax.set_yticklabels([f'{i}' for i in range(5)], fontsize=8)
        
        # Result box below
        ax.text(2, 5.8, result, ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
               fontfamily='monospace')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUT_DIR, 'sift_5x5_all_stages.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("Generating Octave → Filtering flow visualizations...")
    
    create_octave_to_filtering_flow()
    create_detailed_breakdown()
    create_funnel_diagram()
    create_5x5_filtering_example()
    
    print("\nDone! Generated 4 images showing octave to filtering flow.")

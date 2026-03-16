"""
Visual diagram showing keypoint filtering process
From initial detection (501) to final keypoints (387)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os


def create_filtering_pipeline():
    """Create visual showing the keypoint filtering/reduction pipeline"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Keypoint Filtering Pipeline: 501 → 387', fontsize=18, fontweight='bold', pad=15)
    
    # ============================================
    # Stage 1: Initial Detection (26-neighbor extrema)
    # ============================================
    # Box
    stage1 = FancyBboxPatch((0.5, 7), 3.5, 2.2, boxstyle="round,pad=0.05",
                             facecolor='#3498db', edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(stage1)
    ax.text(2.25, 8.3, 'Stage 1', ha='center', fontsize=10, color='white', fontweight='bold')
    ax.text(2.25, 7.7, '26-Neighbor', ha='center', fontsize=11, color='white', fontweight='bold')
    ax.text(2.25, 7.2, 'Extrema Detection', ha='center', fontsize=10, color='white')
    
    # Count box
    count1 = FancyBboxPatch((1.2, 5.8), 2, 0.8, boxstyle="round,pad=0.03",
                             facecolor='white', edgecolor='#3498db', linewidth=2)
    ax.add_patch(count1)
    ax.text(2.2, 6.2, '501 KP', ha='center', fontsize=14, fontweight='bold', color='#3498db')
    
    # Arrow
    ax.annotate('', xy=(4.3, 8), xytext=(4.1, 8),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=3))
    
    # ============================================
    # Stage 2: Contrast Thresholding
    # ============================================
    stage2 = FancyBboxPatch((4.5, 7), 3.5, 2.2, boxstyle="round,pad=0.05",
                             facecolor='#e74c3c', edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(stage2)
    ax.text(6.25, 8.3, 'Stage 2', ha='center', fontsize=10, color='white', fontweight='bold')
    ax.text(6.25, 7.7, 'Low Contrast', ha='center', fontsize=11, color='white', fontweight='bold')
    ax.text(6.25, 7.2, 'Removal', ha='center', fontsize=10, color='white')
    
    # Threshold info
    ax.text(6.25, 5.5, '|D(x)| < 0.03', ha='center', fontsize=10, 
           bbox=dict(boxstyle='round', facecolor='#fadbd8', edgecolor='#e74c3c'))
    
    # Removed count
    removed2 = FancyBboxPatch((5.0, 4.3), 2.5, 0.8, boxstyle="round,pad=0.03",
                               facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=1.5)
    ax.add_patch(removed2)
    ax.text(6.25, 4.7, '−68 removed', ha='center', fontsize=11, color='#c0392b', fontweight='bold')
    
    # Count box
    count2 = FancyBboxPatch((5.2, 3.3), 2, 0.8, boxstyle="round,pad=0.03",
                             facecolor='white', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(count2)
    ax.text(6.2, 3.7, '433 KP', ha='center', fontsize=14, fontweight='bold', color='#e74c3c')
    
    # Arrow
    ax.annotate('', xy=(8.3, 8), xytext=(8.1, 8),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=3))
    
    # ============================================
    # Stage 3: Edge Removal (Hessian Ratio)
    # ============================================
    stage3 = FancyBboxPatch((8.5, 7), 3.5, 2.2, boxstyle="round,pad=0.05",
                             facecolor='#f39c12', edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(stage3)
    ax.text(10.25, 8.3, 'Stage 3', ha='center', fontsize=10, color='white', fontweight='bold')
    ax.text(10.25, 7.7, 'Edge Response', ha='center', fontsize=11, color='white', fontweight='bold')
    ax.text(10.25, 7.2, 'Removal', ha='center', fontsize=10, color='white')
    
    # Threshold info
    ax.text(10.25, 5.5, 'Tr(H)²/Det(H) > 12.1', ha='center', fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='#fdebd0', edgecolor='#f39c12'))
    
    # Removed count
    removed3 = FancyBboxPatch((9.0, 4.3), 2.5, 0.8, boxstyle="round,pad=0.03",
                               facecolor='#fdebd0', edgecolor='#f39c12', linewidth=1.5)
    ax.add_patch(removed3)
    ax.text(10.25, 4.7, '−31 removed', ha='center', fontsize=11, color='#d68910', fontweight='bold')
    
    # Count box
    count3 = FancyBboxPatch((9.2, 3.3), 2, 0.8, boxstyle="round,pad=0.03",
                             facecolor='white', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(count3)
    ax.text(10.2, 3.7, '402 KP', ha='center', fontsize=14, fontweight='bold', color='#f39c12')
    
    # Arrow
    ax.annotate('', xy=(12.3, 8), xytext=(12.1, 8),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=3))
    
    # ============================================
    # Stage 4: Sub-pixel Localization (may discard unstable)
    # ============================================
    stage4 = FancyBboxPatch((12.5, 7), 3, 2.2, boxstyle="round,pad=0.05",
                             facecolor='#27ae60', edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(stage4)
    ax.text(14, 8.3, 'Stage 4', ha='center', fontsize=10, color='white', fontweight='bold')
    ax.text(14, 7.7, 'Sub-pixel', ha='center', fontsize=11, color='white', fontweight='bold')
    ax.text(14, 7.2, 'Refinement', ha='center', fontsize=10, color='white')
    
    # Info
    ax.text(14, 5.5, 'Taylor expansion\n|offset| > 0.5', ha='center', fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='#d5f5e3', edgecolor='#27ae60'))
    
    # Removed count
    removed4 = FancyBboxPatch((12.8, 4.3), 2.5, 0.8, boxstyle="round,pad=0.03",
                               facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=1.5)
    ax.add_patch(removed4)
    ax.text(14.05, 4.7, '−15 removed', ha='center', fontsize=11, color='#1e8449', fontweight='bold')
    
    # Final count box
    final_box = FancyBboxPatch((12.8, 2.8), 2.5, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#27ae60', edgecolor='black', linewidth=3)
    ax.add_patch(final_box)
    ax.text(14.05, 3.6, '387 KP', ha='center', fontsize=16, fontweight='bold', color='white')
    ax.text(14.05, 3.1, 'FINAL', ha='center', fontsize=10, fontweight='bold', color='white')
    
    # ============================================
    # Summary bar at bottom
    # ============================================
    summary_box = FancyBboxPatch((0.5, 0.5), 15, 1.8, boxstyle="round,pad=0.05",
                                  facecolor='#ecf0f1', edgecolor='#34495e', linewidth=2)
    ax.add_patch(summary_box)
    
    ax.text(8, 1.9, 'Summary: Keypoint Reduction', ha='center', fontsize=12, fontweight='bold')
    
    # Progress bar visualization
    total_width = 12
    start_x = 2
    
    # Full bar (501)
    full_bar = FancyBboxPatch((start_x, 0.9), total_width, 0.5, boxstyle="round,pad=0.01",
                               facecolor='#bdc3c7', edgecolor='black', linewidth=1)
    ax.add_patch(full_bar)
    
    # Remaining bar (387/501 = 77.2%)
    remaining_ratio = 387 / 501
    remaining_bar = FancyBboxPatch((start_x, 0.9), total_width * remaining_ratio, 0.5, 
                                    boxstyle="round,pad=0.01",
                                    facecolor='#27ae60', edgecolor='black', linewidth=1)
    ax.add_patch(remaining_bar)
    
    ax.text(start_x - 0.3, 1.15, '501', ha='right', fontsize=10, fontweight='bold')
    ax.text(start_x + total_width + 0.3, 1.15, '387 (77%)', ha='left', fontsize=10, 
           fontweight='bold', color='#27ae60')
    ax.text(start_x + total_width * remaining_ratio, 1.6, '↑', ha='center', fontsize=14, color='#27ae60')
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), 'sift_filtering_pipeline.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


def create_filtering_breakdown():
    """Create detailed breakdown table visual"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title('Keypoint Filtering - Detailed Breakdown', fontsize=16, fontweight='bold', pad=15)
    
    # Table structure
    headers = ['Stage', 'Filter', 'Condition', 'Removed', 'Remaining']
    col_x = [1.5, 4, 7.5, 10.5, 12.5]
    
    # Header row
    header_bg = FancyBboxPatch((0.3, 7.2), 13.4, 1, boxstyle="round,pad=0.02",
                                facecolor='#34495e', edgecolor='black')
    ax.add_patch(header_bg)
    
    for header, x in zip(headers, col_x):
        ax.text(x, 7.7, header, ha='center', va='center', fontsize=11, 
               fontweight='bold', color='white')
    
    # Data rows
    rows = [
        ('Initial', '26-neighbor extrema', 'max/min in 3×3×3', '—', '501'),
        ('Stage 2', 'Low contrast', '|D(x)| < 0.03', '−68', '433'),
        ('Stage 3', 'Edge response', 'Tr²/Det > (r+1)²/r', '−31', '402'),
        ('Stage 4', 'Sub-pixel refine', '|offset| > 0.5', '−15', '387'),
    ]
    
    colors = ['#3498db', '#e74c3c', '#f39c12', '#27ae60']
    
    for idx, (row, color) in enumerate(zip(rows, colors)):
        y = 6.0 - idx * 1.4
        
        # Row background
        row_bg = FancyBboxPatch((0.3, y - 0.5), 13.4, 1.2, boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor='black', alpha=0.2)
        ax.add_patch(row_bg)
        
        # Stage indicator
        stage_box = FancyBboxPatch((0.5, y - 0.3), 2, 0.8, boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(stage_box)
        ax.text(col_x[0], y + 0.1, row[0], ha='center', va='center', fontsize=10, 
               fontweight='bold', color='white')
        
        # Other columns
        ax.text(col_x[1], y + 0.1, row[1], ha='center', va='center', fontsize=10)
        ax.text(col_x[2], y + 0.1, row[2], ha='center', va='center', fontsize=9, 
               family='monospace')
        
        # Removed (red text if applicable)
        if row[3] != '—':
            ax.text(col_x[3], y + 0.1, row[3], ha='center', va='center', fontsize=11, 
                   fontweight='bold', color='#c0392b')
        else:
            ax.text(col_x[3], y + 0.1, row[3], ha='center', va='center', fontsize=11, 
                   color='gray')
        
        # Remaining (bold)
        remaining_box = FancyBboxPatch((11.8, y - 0.25), 1.4, 0.7, boxstyle="round,pad=0.02",
                                        facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(remaining_box)
        ax.text(col_x[4], y + 0.1, row[4], ha='center', va='center', fontsize=12, 
               fontweight='bold', color=color)
    
    # Math formulas section
    formula_bg = FancyBboxPatch((0.3, 0.3), 13.4, 1.5, boxstyle="round,pad=0.03",
                                 facecolor='#fdf2e9', edgecolor='#e67e22', linewidth=2)
    ax.add_patch(formula_bg)
    
    ax.text(7, 1.5, 'Key Formulas:', ha='center', fontsize=11, fontweight='bold')
    ax.text(4, 0.9, 'Contrast: |D(x̂)| = |D + 0.5 × ∂D/∂x × x̂|', ha='center', fontsize=9, family='monospace')
    ax.text(10, 0.9, 'Edge: Tr(H)²/Det(H) < (r+1)²/r  [r=10]', ha='center', fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), 'sift_filtering_breakdown.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


if __name__ == "__main__":
    create_filtering_pipeline()
    create_filtering_breakdown()
    print("\nDone! Generated filtering visual diagrams.")

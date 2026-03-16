"""
Visual diagram showing math formulas for SIFT filtering stages
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_math_formulas_visual():
    """Create visual diagram with math formulas for each filtering stage"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # ============================================
    # Stage 1: 26-Neighbor Extrema
    # ============================================
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Stage 1: 26-Neighbor Extrema Detection', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    # Formula box
    formula_box = FancyBboxPatch((0.5, 4.5), 9, 3.5, boxstyle="round,pad=0.05",
                                  facecolor='#ebf5fb', edgecolor='#3498db', linewidth=2)
    ax.add_patch(formula_box)
    
    formulas = [
        r'$D(x,y,\sigma) = L(x,y,k\sigma) - L(x,y,\sigma)$',
        '',
        'Keypoint if:',
        r'$D(x,y,\sigma) >$ ALL 26 neighbors $\rightarrow$ Maximum',
        r'$D(x,y,\sigma) <$ ALL 26 neighbors $\rightarrow$ Minimum',
    ]
    
    y_pos = 7.5
    for formula in formulas:
        ax.text(5, y_pos, formula, ha='center', va='center', fontsize=11)
        y_pos -= 0.6
    
    # 3x3x3 cube illustration
    ax.text(5, 3.5, '26 neighbors = 8 (same σ) + 9 (σ-1) + 9 (σ+1)', 
           ha='center', fontsize=10, style='italic')
    
    # Visual cube
    cube_x, cube_y = 5, 1.8
    for layer, color, label in [(0, '#aed6f1', 'σ-1'), (0.8, '#5dade2', 'σ'), (1.6, '#2980b9', 'σ+1')]:
        rect = plt.Rectangle((cube_x - 1.5 + layer*0.3, cube_y - 0.5 + layer*0.3), 1.5, 1.5, 
                             fill=True, facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
        ax.add_patch(rect)
    ax.text(cube_x + 1.5, cube_y + 0.5, '3×3×3 cube', fontsize=9)
    
    # ============================================
    # Stage 2: Low Contrast Removal
    # ============================================
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Stage 2: Low Contrast Removal', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    # Formula box
    formula_box = FancyBboxPatch((0.5, 3), 9, 5, boxstyle="round,pad=0.05",
                                  facecolor='#fdedec', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.5, 'Taylor Expansion:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 6.8, r'$D(x) \approx D + (\nabla D)^T x + \frac{1}{2} x^T H x$', 
           ha='center', fontsize=12)
    
    ax.text(5, 5.8, 'Solve for extremum:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 5.1, r'$\hat{x} = -H^{-1} \nabla D$', 
           ha='center', fontsize=14)
    
    ax.text(5, 4.1, 'Contrast at refined location:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 3.4, r'$D(\hat{x}) = D + \frac{1}{2} (\nabla D)^T \hat{x}$', 
           ha='center', fontsize=12)
    
    # Rejection condition
    reject_box = FancyBboxPatch((2, 1), 6, 1.2, boxstyle="round,pad=0.05",
                                 facecolor='#f5b7b1', edgecolor='#c0392b', linewidth=2)
    ax.add_patch(reject_box)
    ax.text(5, 1.6, r'REJECT if:  $|D(\hat{x})| < 0.03$', 
           ha='center', fontsize=12, fontweight='bold', color='#922b21')
    
    # ============================================
    # Stage 3: Edge Response Removal
    # ============================================
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Stage 3: Edge Response Removal', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    # Formula box
    formula_box = FancyBboxPatch((0.5, 2.5), 9, 5.5, boxstyle="round,pad=0.05",
                                  facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.5, '2D Hessian Matrix:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 6.8, r'$H = \left[ D_{xx}, D_{xy} ; D_{xy}, D_{yy} \right]$', 
           ha='center', fontsize=12)
    ax.text(5, 6.2, r'$Tr(H) = D_{xx} + D_{yy}$  ,  $Det(H) = D_{xx} D_{yy} - D_{xy}^2$',
           ha='center', fontsize=11)
    
    ax.text(5, 5.2, 'Eigenvalue ratio test:', 
           ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 4.5, r'$\frac{Tr(H)^2}{Det(H)} = \frac{(\alpha+\beta)^2}{\alpha \cdot \beta} = \frac{(r+1)^2}{r}$', 
           ha='center', fontsize=12)
    ax.text(5, 3.7, r'where $r = \alpha / \beta$ (ratio of eigenvalues)', 
           ha='center', fontsize=10, style='italic')
    
    # Rejection condition
    reject_box = FancyBboxPatch((1.5, 0.8), 7, 1.2, boxstyle="round,pad=0.05",
                                 facecolor='#f9e79f', edgecolor='#d4ac0d', linewidth=2)
    ax.add_patch(reject_box)
    ax.text(5, 1.4, r'REJECT if:  $Tr(H)^2 / Det(H) > (r+1)^2 / r$  (r=10 → 12.1)', 
           ha='center', fontsize=11, fontweight='bold', color='#7d6608')
    
    # ============================================
    # Stage 4: Sub-pixel Refinement
    # ============================================
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#27ae60', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Stage 4: Sub-pixel Refinement', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    # Formula box
    formula_box = FancyBboxPatch((0.5, 2.5), 9, 5.5, boxstyle="round,pad=0.05",
                                  facecolor='#e9f7ef', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.5, 'Offset computation:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 6.7, r'$\hat{x} = -H^{-1} \nabla D$', 
           ha='center', fontsize=13)
    
    ax.text(5, 5.7, 'Expanded (2D case):', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 4.9, r'$offset_x = -(D_{yy} \cdot D_x - D_{xy} \cdot D_y) / Det(H)$', 
           ha='center', fontsize=11)
    ax.text(5, 4.1, r'$offset_y = -(D_{xx} \cdot D_y - D_{xy} \cdot D_x) / Det(H)$', 
           ha='center', fontsize=11)
    
    # Rejection condition
    reject_box = FancyBboxPatch((1.5, 0.8), 7, 1.2, boxstyle="round,pad=0.05",
                                 facecolor='#a9dfbf', edgecolor='#1e8449', linewidth=2)
    ax.add_patch(reject_box)
    ax.text(5, 1.4, r'REJECT if:  $|offset_x| > 0.5$  OR  $|offset_y| > 0.5$', 
           ha='center', fontsize=11, fontweight='bold', color='#145a32')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUT_DIR, 'sift_math_formulas.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_hessian_derivatives_visual():
    """Create visual showing how Hessian derivatives are computed"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Computing Derivatives (Finite Differences)', fontsize=16, fontweight='bold', pad=15)
    
    # First derivatives section
    section1 = FancyBboxPatch((0.5, 6.5), 6, 3, boxstyle="round,pad=0.05",
                               facecolor='#e8f8f5', edgecolor='#1abc9c', linewidth=2)
    ax.add_patch(section1)
    ax.text(3.5, 9, 'First Derivatives (Gradient)', ha='center', fontsize=12, fontweight='bold')
    
    ax.text(3.5, 8.2, r'$D_x = \frac{D(x+1,y) - D(x-1,y)}{2}$', ha='center', fontsize=12)
    ax.text(3.5, 7.4, r'$D_y = \frac{D(x,y+1) - D(x,y-1)}{2}$', ha='center', fontsize=12)
    ax.text(3.5, 6.8, r'$D_\sigma = \frac{D(x,y,\sigma+1) - D(x,y,\sigma-1)}{2}$', ha='center', fontsize=11)
    
    # Second derivatives section
    section2 = FancyBboxPatch((7.5, 6.5), 6, 3, boxstyle="round,pad=0.05",
                               facecolor='#fef5e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(section2)
    ax.text(10.5, 9, 'Second Derivatives (Hessian)', ha='center', fontsize=12, fontweight='bold')
    
    ax.text(10.5, 8.2, r'$D_{xx} = D(x+1,y) + D(x-1,y) - 2D(x,y)$', ha='center', fontsize=11)
    ax.text(10.5, 7.4, r'$D_{yy} = D(x,y+1) + D(x,y-1) - 2D(x,y)$', ha='center', fontsize=11)
    ax.text(10.5, 6.7, r'$D_{xy} = \frac{D(x+1,y+1) - D(x+1,y-1) - D(x-1,y+1) + D(x-1,y-1)}{4}$', 
           ha='center', fontsize=10)
    
    # Grid illustration
    grid_x, grid_y = 3.5, 3
    cell_size = 0.8
    
    # Draw 3x3 grid
    for i in range(3):
        for j in range(3):
            x = grid_x + (j - 1) * cell_size
            y = grid_y + (1 - i) * cell_size
            
            if i == 1 and j == 1:
                color = '#3498db'
                label = 'D(x,y)'
            elif (i == 1 and j == 0) or (i == 1 and j == 2):
                color = '#e74c3c'
                label = 'Dx' if j == 2 else '-'
            elif (i == 0 and j == 1) or (i == 2 and j == 1):
                color = '#27ae60'
                label = 'Dy' if i == 2 else '-'
            else:
                color = '#9b59b6'
                label = 'Dxy'
            
            rect = plt.Rectangle((x - cell_size/2, y - cell_size/2), cell_size, cell_size,
                                 fill=True, facecolor=color, edgecolor='black', linewidth=1, alpha=0.3)
            ax.add_patch(rect)
    
    ax.text(grid_x, grid_y - 1.5, '3×3 neighborhood for derivatives', ha='center', fontsize=10)
    
    # Legend
    legend_x = 8
    legend_y = 3.5
    colors = [('#3498db', 'Center D(x,y)'), ('#e74c3c', 'For Dx, Dxx'), 
              ('#27ae60', 'For Dy, Dyy'), ('#9b59b6', 'For Dxy')]
    
    for idx, (color, label) in enumerate(colors):
        rect = plt.Rectangle((legend_x, legend_y - idx * 0.6), 0.4, 0.4,
                             fill=True, facecolor=color, edgecolor='black', alpha=0.5)
        ax.add_patch(rect)
        ax.text(legend_x + 0.6, legend_y - idx * 0.6 + 0.2, label, fontsize=10, va='center')
    
    # Hessian matrix
    hess_box = FancyBboxPatch((0.5, 0.5), 13, 1.5, boxstyle="round,pad=0.05",
                               facecolor='#f4ecf7', edgecolor='#8e44ad', linewidth=2)
    ax.add_patch(hess_box)
    ax.text(7, 1.5, 'Hessian Matrix:', ha='center', fontsize=11, fontweight='bold')
    ax.text(7, 0.9, r'H = [Dxx, Dxy; Dxy, Dyy]    Tr(H) = Dxx + Dyy    Det(H) = Dxx·Dyy - Dxy²', 
           ha='center', fontsize=11)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUT_DIR, 'sift_derivatives.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    create_math_formulas_visual()
    create_hessian_derivatives_visual()
    print("\nDone! Generated math formula visuals.")

"""
Visual diagram showing math formulas for SURF algorithm
Including:
- Integral Image formulas
- Box filter patterns
- Hessian determinant
- Derivative computations
- Sub-pixel refinement
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


def create_integral_image_formula():
    """Create visual showing integral image computation"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Formula visualization
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Integral Image Definition', ha='center', va='center',
           fontsize=14, fontweight='bold', color='white')
    
    # Formula box
    formula_box = FancyBboxPatch((0.5, 4), 9, 4, boxstyle="round,pad=0.05",
                                  facecolor='#ebf5fb', edgecolor='#3498db', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.3, 'Definition:', ha='center', fontsize=12, fontweight='bold')
    ax.text(5, 6.5, r'$II(x,y) = \sum_{i=0}^{x} \sum_{j=0}^{y} I(i,j)$', 
           ha='center', fontsize=14)
    
    ax.text(5, 5.5, 'Recursive Formula:', ha='center', fontsize=12, fontweight='bold')
    ax.text(5, 4.7, r'$II(x,y) = I(x,y) + II(x-1,y) + II(x,y-1) - II(x-1,y-1)$', 
           ha='center', fontsize=12)
    
    # Box sum formula
    sum_box = FancyBboxPatch((0.5, 0.5), 9, 3, boxstyle="round,pad=0.05",
                              facecolor='#e8f8f5', edgecolor='#1abc9c', linewidth=2)
    ax.add_patch(sum_box)
    
    ax.text(5, 3, 'Box Sum Formula:', ha='center', fontsize=12, fontweight='bold')
    ax.text(5, 2.2, r'$\sum_{(x_1,y_1)}^{(x_2,y_2)} I = II(D) - II(B) - II(C) + II(A)$', 
           ha='center', fontsize=12)
    ax.text(5, 1.3, 'ANY rectangle sum = 4 lookups = O(1)!', 
           ha='center', fontsize=11, style='italic', color='#1e8449')
    
    # Right: Grid visualization
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Box Sum: 4 Lookups', fontsize=14, fontweight='bold')
    
    # Draw grid
    grid_x, grid_y = 2, 2
    cell_size = 1.2
    
    for i in range(5):
        for j in range(5):
            x = grid_x + j * cell_size
            y = grid_y + (4-i) * cell_size
            
            # Color the region we want to sum
            if 1 <= i <= 3 and 1 <= j <= 3:
                color = '#aed6f1'
            else:
                color = 'white'
            
            rect = Rectangle((x, y), cell_size, cell_size,
                            fill=True, facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
    
    # Mark corners A, B, C, D
    corners = [
        ('A', grid_x + 1*cell_size, grid_y + 4*cell_size, '#e74c3c'),
        ('B', grid_x + 4*cell_size, grid_y + 4*cell_size, '#e74c3c'),
        ('C', grid_x + 1*cell_size, grid_y + 1*cell_size, '#e74c3c'),
        ('D', grid_x + 4*cell_size, grid_y + 1*cell_size, '#27ae60'),
    ]
    
    for label, x, y, color in corners:
        circle = Circle((x, y), 0.2, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Formula below grid
    ax.text(5, 1, 'Sum = D - B - C + A', ha='center', fontsize=14, fontweight='bold')
    ax.text(5, 0.3, '(Blue region sum computed with only 4 lookups)', 
           ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    output_path = os.path.join(IMAGES_DIR, 'surf_integral_formula.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_hessian_formula():
    """Create visual showing Hessian matrix and determinant"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # ============================================
    # Hessian Matrix Definition
    # ============================================
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#9b59b6', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Hessian Matrix', ha='center', va='center',
           fontsize=14, fontweight='bold', color='white')
    
    formula_box = FancyBboxPatch((0.5, 3.5), 9, 4.5, boxstyle="round,pad=0.05",
                                  facecolor='#f5eef8', edgecolor='#9b59b6', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.3, '2nd Order Partial Derivatives:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 6.4, r'H = [Lxx, Lxy; Lxy, Lyy]', 
           ha='center', fontsize=13, family='monospace')
    
    ax.text(5, 5.2, 'SURF Approximation (Box Filters):', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 4.3, r'H $\approx$ [Dxx, Dxy; Dxy, Dyy]', 
           ha='center', fontsize=13)
    
    # Determinant formula
    det_box = FancyBboxPatch((0.5, 0.5), 9, 2.5, boxstyle="round,pad=0.05",
                              facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(det_box)
    
    ax.text(5, 2.5, 'Blob Response (Determinant):', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 1.5, r'$det(H) = D_{xx} \cdot D_{yy} - (0.9 \cdot D_{xy})^2$', 
           ha='center', fontsize=14, color='#d35400')
    
    # ============================================
    # Box Filter Patterns
    # ============================================
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#27ae60', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Box Filter Approximations', ha='center', va='center',
           fontsize=14, fontweight='bold', color='white')
    
    # Dxx filter
    ax.text(2.5, 7.8, 'Dxx Filter', ha='center', fontsize=11, fontweight='bold')
    for i, (x, color, label) in enumerate([(1, '#27ae60', '+1'), (2.5, '#e74c3c', '-2'), (4, '#27ae60', '+1')]):
        rect = Rectangle((x, 6.5), 1.2, 1, facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + 0.6, 7, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Dyy filter
    ax.text(7.5, 7.8, 'Dyy Filter', ha='center', fontsize=11, fontweight='bold')
    for i, (y, color, label) in enumerate([(7.2, '#27ae60', '+1'), (6.5, '#e74c3c', '-2'), (5.8, '#27ae60', '+1')]):
        rect = Rectangle((6.5, y), 2, 0.6, facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
        ax.add_patch(rect)
        ax.text(7.5, y + 0.3, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Dxy filter (checkerboard)
    ax.text(2.5, 5.2, 'Dxy Filter', ha='center', fontsize=11, fontweight='bold')
    dxy_pattern = [
        (1, 4, '#27ae60', '+1'), (2.5, 4, '#e74c3c', '-1'),
        (1, 3, '#e74c3c', '-1'), (2.5, 3, '#27ae60', '+1'),
    ]
    for x, y, color, label in dxy_pattern:
        rect = Rectangle((x, y), 1.2, 0.8, facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + 0.6, y + 0.4, label, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Formulas
    formula_box = FancyBboxPatch((0.5, 0.5), 9, 2.2, boxstyle="round,pad=0.05",
                                  facecolor='#e8f8f5', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 2.2, 'Using Integral Image:', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 1.5, r'$D_{xx} = LeftLobe - 2 \cdot CenterLobe + RightLobe$', ha='center', fontsize=10)
    ax.text(5, 0.9, 'Each lobe sum computed in O(1) with 4 lookups!', 
           ha='center', fontsize=9, style='italic', color='#1e8449')
    
    # ============================================
    # Derivative Formulas (Finite Differences)
    # ============================================
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'First Derivatives (Gradient)', ha='center', va='center',
           fontsize=14, fontweight='bold', color='white')
    
    formula_box = FancyBboxPatch((0.5, 4), 9, 4, boxstyle="round,pad=0.05",
                                  facecolor='#fdedec', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.5, 'Central Difference Formulas:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 6.6, r'$H_x = \frac{H(x+1,y,\sigma) - H(x-1,y,\sigma)}{2}$', ha='center', fontsize=12)
    ax.text(5, 5.6, r'$H_y = \frac{H(x,y+1,\sigma) - H(x,y-1,\sigma)}{2}$', ha='center', fontsize=12)
    ax.text(5, 4.6, r'$H_\sigma = \frac{H(x,y,\sigma+1) - H(x,y,\sigma-1)}{2}$', ha='center', fontsize=12)
    
    # Second derivatives
    second_box = FancyBboxPatch((0.5, 0.5), 9, 3, boxstyle="round,pad=0.05",
                                 facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(second_box)
    
    ax.text(5, 3, 'Second Derivatives (Curvature):', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 2.2, r'$H_{xx} = H(x+1,y) + H(x-1,y) - 2 \cdot H(x,y)$', ha='center', fontsize=11)
    ax.text(5, 1.3, r'$H_{xy} = \frac{H(x+1,y+1) - H(x+1,y-1) - H(x-1,y+1) + H(x-1,y-1)}{4}$', 
           ha='center', fontsize=10)
    
    # ============================================
    # 3×3 Neighborhood Visual
    # ============================================
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, '3×3 Neighborhood for Derivatives', ha='center', va='center',
           fontsize=14, fontweight='bold', color='white')
    
    # Draw 3x3 grid
    grid_x, grid_y = 3, 4
    cell_size = 1.3
    
    labels = [
        ['NW', 'N', 'NE'],
        ['W', 'C', 'E'],
        ['SW', 'S', 'SE']
    ]
    colors = [
        ['#9b59b6', '#27ae60', '#9b59b6'],
        ['#e74c3c', '#3498db', '#e74c3c'],
        ['#9b59b6', '#27ae60', '#9b59b6']
    ]
    
    for i in range(3):
        for j in range(3):
            x = grid_x + j * cell_size
            y = grid_y + (2-i) * cell_size
            
            rect = Rectangle((x, y), cell_size, cell_size,
                            fill=True, facecolor=colors[i][j], edgecolor='black', 
                            linewidth=2, alpha=0.6)
            ax.add_patch(rect)
            ax.text(x + cell_size/2, y + cell_size/2, labels[i][j], 
                   ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Legend
    legend_items = [
        ('#3498db', 'C = Center (keypoint)'),
        ('#e74c3c', 'E, W = For Hx, Hxx'),
        ('#27ae60', 'N, S = For Hy, Hyy'),
        ('#9b59b6', 'Corners = For Hxy'),
    ]
    
    for idx, (color, label) in enumerate(legend_items):
        y_pos = 3 - idx * 0.6
        rect = Rectangle((1, y_pos), 0.4, 0.4, facecolor=color, edgecolor='black', alpha=0.6)
        ax.add_patch(rect)
        ax.text(1.6, y_pos + 0.2, label, fontsize=9, va='center')
    
    plt.tight_layout()
    output_path = os.path.join(IMAGES_DIR, 'surf_math_formulas.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_derivatives_visual():
    """Create visual showing how derivatives are computed"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('Computing Derivatives Using Finite Differences', fontsize=16, fontweight='bold', pad=15)
    
    # First derivatives section
    section1 = FancyBboxPatch((0.5, 7.5), 7.5, 4), 
    section1_box = FancyBboxPatch((0.5, 7.5), 7.5, 4, boxstyle="round,pad=0.05",
                                   facecolor='#e8f8f5', edgecolor='#1abc9c', linewidth=2)
    ax.add_patch(section1_box)
    ax.text(4.25, 11, 'First Derivatives (Gradient)', ha='center', fontsize=13, fontweight='bold')
    
    ax.text(4.25, 10.2, r'$H_x = \frac{H(x+1,y) - H(x-1,y)}{2}$', ha='center', fontsize=12)
    ax.text(4.25, 9.4, r'$H_y = \frac{H(x,y+1) - H(x,y-1)}{2}$', ha='center', fontsize=12)
    ax.text(4.25, 8.6, r'$H_\sigma = \frac{H(x,y,\sigma+1) - H(x,y,\sigma-1)}{2}$', ha='center', fontsize=11)
    ax.text(4.25, 7.9, 'Measures slope in x, y, σ directions', ha='center', fontsize=10, style='italic')
    
    # Second derivatives section
    section2_box = FancyBboxPatch((8.5, 7.5), 7, 4, boxstyle="round,pad=0.05",
                                   facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(section2_box)
    ax.text(12, 11, 'Second Derivatives (Curvature)', ha='center', fontsize=13, fontweight='bold')
    
    ax.text(12, 10.2, r'$H_{xx} = H(E) + H(W) - 2 \cdot H(C)$', ha='center', fontsize=11)
    ax.text(12, 9.4, r'$H_{yy} = H(S) + H(N) - 2 \cdot H(C)$', ha='center', fontsize=11)
    ax.text(12, 8.6, r'$H_{xy} = \frac{H(SE) - H(NE) - H(SW) + H(NW)}{4}$', ha='center', fontsize=10)
    ax.text(12, 7.9, 'Measures how slope changes', ha='center', fontsize=10, style='italic')
    
    # 3D Hessian section
    section3_box = FancyBboxPatch((0.5, 3.5), 15, 3.5, boxstyle="round,pad=0.05",
                                   facecolor='#f4ecf7', edgecolor='#8e44ad', linewidth=2)
    ax.add_patch(section3_box)
    ax.text(8, 6.5, '3D Hessian Matrix (for Sub-pixel Refinement)', ha='center', fontsize=13, fontweight='bold')
    
    ax.text(8, 5.5, 'H_3D = [Hxx, Hxy, Hxs; Hxy, Hyy, Hys; Hxs, Hys, Hss]', 
           ha='center', fontsize=11, family='monospace')
    ax.text(8, 4.2, r'Sub-pixel offset: $\hat{x} = -(H^{3D})^{-1} \cdot \nabla H$', ha='center', fontsize=12)
    
    # Rejection condition
    reject_box = FancyBboxPatch((3, 0.5), 10, 2.5, boxstyle="round,pad=0.05",
                                 facecolor='#fadbd8', edgecolor='#c0392b', linewidth=2)
    ax.add_patch(reject_box)
    ax.text(8, 2.5, 'Stability Check:', ha='center', fontsize=12, fontweight='bold')
    ax.text(8, 1.7, r'REJECT if: $|offset_x| > 0.5$ OR $|offset_y| > 0.5$ OR $|offset_\sigma| > 0.5$', 
           ha='center', fontsize=12, color='#922b21')
    ax.text(8, 1, 'If offset > 0.5, the true extremum is in a neighboring pixel', 
           ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    output_path = os.path.join(IMAGES_DIR, 'surf_derivatives.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_hessian_computation_visual():
    """Create visual showing step-by-step Hessian computation"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # ============================================
    # Step 1: Box filter regions
    # ============================================
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Step 1: Define Box Filter Regions', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    # Show 9x9 filter layout for Dxx
    ax.text(5, 7.8, '9×9 Filter for Dxx:', ha='center', fontsize=11, fontweight='bold')
    
    # Draw filter regions
    for i, (x, w, color, label) in enumerate([
        (1.5, 1.5, '#27ae60', 'Left\n+1'),
        (3.2, 1.5, '#e74c3c', 'Center\n-2'),
        (4.9, 1.5, '#27ae60', 'Right\n+1')
    ]):
        rect = Rectangle((x, 5.5), w, 2, facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + w/2, 6.5, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    ax.text(5, 5, 'Each region: 3×9 pixels (for 9×9 filter)', ha='center', fontsize=10)
    
    # Formulas
    formula_box = FancyBboxPatch((0.5, 0.5), 9, 4, boxstyle="round,pad=0.05",
                                  facecolor='#ebf5fb', edgecolor='#3498db', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 4, 'Region coordinates (for point x,y):', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 3.2, 'half = filter_size / 2 = 4', ha='center', fontsize=10)
    ax.text(5, 2.4, 'lobe_width = filter_size / 3 = 3', ha='center', fontsize=10)
    ax.text(5, 1.6, 'Left:   (x-4, y-4) to (x-4+2, y+4)', ha='center', fontsize=10)
    ax.text(5, 0.9, 'Center: (x-1, y-4) to (x+1, y+4)', ha='center', fontsize=10)
    
    # ============================================
    # Step 2: Compute box sums
    # ============================================
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#27ae60', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Step 2: Compute Box Sums', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    example_box = FancyBboxPatch((0.5, 2), 9, 6, boxstyle="round,pad=0.05",
                                  facecolor='#e8f8f5', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(example_box)
    
    ax.text(5, 7.5, 'Example: Point (150, 200) with 9×9 filter', ha='center', fontsize=11, fontweight='bold')
    
    ax.text(5, 6.6, 'Left_sum = box_sum(II, 146, 196, 148, 204) = 450', ha='center', fontsize=10, family='monospace')
    ax.text(5, 5.9, 'Center_sum = box_sum(II, 149, 196, 151, 204) = 380', ha='center', fontsize=10, family='monospace')
    ax.text(5, 5.2, 'Right_sum = box_sum(II, 152, 196, 154, 204) = 420', ha='center', fontsize=10, family='monospace')
    
    ax.text(5, 4.2, 'Dxx_raw = Left - 2×Center + Right', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 3.4, '= 450 - 2×380 + 420 = 110', ha='center', fontsize=11)
    
    ax.text(5, 2.5, 'Dxx = Dxx_raw / area = 110 / 81 = 1.36', ha='center', fontsize=11, color='#1e8449')
    
    # ============================================
    # Step 3: Compute all derivatives
    # ============================================
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Step 3: Compute Dxx, Dyy, Dxy', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    result_box = FancyBboxPatch((0.5, 2), 9, 6, boxstyle="round,pad=0.05",
                                 facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(result_box)
    
    ax.text(5, 7.5, 'Same process for all three:', ha='center', fontsize=11, fontweight='bold')
    
    ax.text(5, 6.5, 'Dxx: Horizontal lobes (Left-Center-Right)', ha='center', fontsize=10)
    ax.text(5, 5.9, '     = 450 - 760 + 420 = 110  →  Dxx = 1.36', ha='center', fontsize=10, family='monospace')
    
    ax.text(5, 5.0, 'Dyy: Vertical lobes (Top-Middle-Bottom)', ha='center', fontsize=10)
    ax.text(5, 4.4, '     = 430 - 760 + 440 = 110  →  Dyy = 1.36', ha='center', fontsize=10, family='monospace')
    
    ax.text(5, 3.5, 'Dxy: Checkerboard (TL - TR - BL + BR)', ha='center', fontsize=10)
    ax.text(5, 2.9, '     = 200 - 180 - 190 + 210 = 40  →  Dxy = 0.49', ha='center', fontsize=10, family='monospace')
    
    # ============================================
    # Step 4: Compute determinant
    # ============================================
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Step 4: Compute det(H)', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    det_box = FancyBboxPatch((0.5, 4), 9, 4, boxstyle="round,pad=0.05",
                              facecolor='#fdedec', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(det_box)
    
    ax.text(5, 7.5, 'Hessian Determinant:', ha='center', fontsize=12, fontweight='bold')
    ax.text(5, 6.6, r'$det(H) = D_{xx} \times D_{yy} - (0.9 \times D_{xy})^2$', ha='center', fontsize=13)
    
    ax.text(5, 5.5, 'Numerical calculation:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 4.8, '= 1.36 × 1.36 - (0.9 × 0.49)²', ha='center', fontsize=11)
    ax.text(5, 4.2, '= 1.85 - (0.44)² = 1.85 - 0.19 = 1.66', ha='center', fontsize=11)
    
    # Result interpretation
    result_box = FancyBboxPatch((1, 0.5), 8, 3, boxstyle="round,pad=0.05",
                                 facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(result_box)
    
    ax.text(5, 3, 'Result: det(H) = 1.66', ha='center', fontsize=14, fontweight='bold', color='#1e8449')
    ax.text(5, 2.2, 'det(H) > 0  →  BLOB detected!', ha='center', fontsize=12, color='#1e8449')
    ax.text(5, 1.3, '(Dxx < 0 means bright blob / local maximum)', ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    output_path = os.path.join(IMAGES_DIR, 'surf_step2_computation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_subpixel_refinement_visual():
    """Create visual showing sub-pixel refinement process"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # ============================================
    # 3x3 Hessian neighborhood
    # ============================================
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Hessian Response Neighborhood', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    # Draw 3x3 grid with values
    grid_x, grid_y = 2.5, 4.5
    cell_size = 1.5
    
    values = [
        [1.45, 1.58, 1.42],
        [1.52, 1.66, 1.55],
        [1.48, 1.60, 1.46]
    ]
    
    for i in range(3):
        for j in range(3):
            x = grid_x + j * cell_size
            y = grid_y + (2-i) * cell_size
            
            color = '#3498db' if i == 1 and j == 1 else '#aed6f1'
            rect = Rectangle((x, y), cell_size, cell_size,
                            fill=True, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x + cell_size/2, y + cell_size/2, f'{values[i][j]:.2f}', 
                   ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Labels
    ax.text(grid_x - 0.3, grid_y + 2.25*cell_size, 'y-1', ha='right', fontsize=10)
    ax.text(grid_x - 0.3, grid_y + 1.25*cell_size, 'y', ha='right', fontsize=10)
    ax.text(grid_x - 0.3, grid_y + 0.25*cell_size, 'y+1', ha='right', fontsize=10)
    ax.text(grid_x + 0.5*cell_size, grid_y - 0.3, 'x-1', ha='center', fontsize=10)
    ax.text(grid_x + 1.5*cell_size, grid_y - 0.3, 'x', ha='center', fontsize=10)
    ax.text(grid_x + 2.5*cell_size, grid_y - 0.3, 'x+1', ha='center', fontsize=10)
    
    ax.text(5, 3.8, 'Keypoint at center: H(150,200) = 1.66', ha='center', fontsize=11)
    
    # ============================================
    # Gradient computation
    # ============================================
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#27ae60', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Step 1: Compute Gradient', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    grad_box = FancyBboxPatch((0.5, 3), 9, 5, boxstyle="round,pad=0.05",
                               facecolor='#e8f8f5', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(grad_box)
    
    ax.text(5, 7.5, r'$H_x = \frac{H(x+1,y) - H(x-1,y)}{2}$', ha='center', fontsize=12)
    ax.text(5, 6.7, '= (1.55 - 1.52) / 2 = 0.015', ha='center', fontsize=11, family='monospace')
    
    ax.text(5, 5.7, r'$H_y = \frac{H(x,y+1) - H(x,y-1)}{2}$', ha='center', fontsize=12)
    ax.text(5, 4.9, '= (1.60 - 1.58) / 2 = 0.01', ha='center', fontsize=11, family='monospace')
    
    ax.text(5, 3.7, r'$\nabla H = [0.015, 0.01]^T$', ha='center', fontsize=12, color='#1e8449')
    
    # ============================================
    # Hessian computation
    # ============================================
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Step 2: Compute 2nd Derivatives', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    hess_box = FancyBboxPatch((0.5, 2.5), 9, 5.5, boxstyle="round,pad=0.05",
                               facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(hess_box)
    
    ax.text(5, 7.5, r'$H_{xx} = H(E) + H(W) - 2 \cdot H(C)$', ha='center', fontsize=11)
    ax.text(5, 6.8, '= 1.55 + 1.52 - 2×1.66 = -0.25', ha='center', fontsize=10, family='monospace')
    
    ax.text(5, 5.9, r'$H_{yy} = H(S) + H(N) - 2 \cdot H(C)$', ha='center', fontsize=11)
    ax.text(5, 5.2, '= 1.60 + 1.58 - 2×1.66 = -0.14', ha='center', fontsize=10, family='monospace')
    
    ax.text(5, 4.3, r'$H_{xy} = \frac{H(SE) - H(NE) - H(SW) + H(NW)}{4}$', ha='center', fontsize=10)
    ax.text(5, 3.6, '= (1.46 - 1.42 - 1.48 + 1.45) / 4 = 0.0025', ha='center', fontsize=10, family='monospace')
    
    # ============================================
    # Offset computation
    # ============================================
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Step 3: Compute Offset', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    offset_box = FancyBboxPatch((0.5, 4), 9, 4, boxstyle="round,pad=0.05",
                                 facecolor='#fdedec', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(offset_box)
    
    ax.text(5, 7.5, r'$offset = -H^{-1} \cdot \nabla H$', ha='center', fontsize=13)
    
    ax.text(5, 6.5, 'Det(H²) = Hxx × Hyy - Hxy² = 0.035', ha='center', fontsize=10)
    ax.text(5, 5.7, 'offset_x = -(Hyy×Hx - Hxy×Hy)/Det = 0.06', ha='center', fontsize=10)
    ax.text(5, 5.0, 'offset_y = -(Hxx×Hy - Hxy×Hx)/Det = 0.07', ha='center', fontsize=10)
    
    # Result
    result_box = FancyBboxPatch((1, 0.5), 8, 3, boxstyle="round,pad=0.05",
                                 facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(result_box)
    
    ax.text(5, 3, '|offset_x| = 0.06 < 0.5  ✓', ha='center', fontsize=11)
    ax.text(5, 2.3, '|offset_y| = 0.07 < 0.5  ✓', ha='center', fontsize=11)
    ax.text(5, 1.4, 'KEEP: Refined position = (150.06, 200.07)', ha='center', fontsize=12, 
           fontweight='bold', color='#1e8449')
    
    plt.tight_layout()
    output_path = os.path.join(IMAGES_DIR, 'surf_subpixel_refinement.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_all_math_visuals():
    """Generate all math formula visualizations"""
    print("Generating SURF math formula visualizations...")
    create_integral_image_formula()
    create_hessian_formula()
    create_derivatives_visual()
    create_hessian_computation_visual()
    create_subpixel_refinement_visual()
    print("\nDone! Generated all math formula visuals.")


if __name__ == "__main__":
    create_all_math_visuals()

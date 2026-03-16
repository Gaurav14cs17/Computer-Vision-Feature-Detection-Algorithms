"""
SIFT Filtering Operations - Step by Step with 5x5 Matrix Example
Shows each operation applied to actual numbers
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_matrix_visual(ax, matrix, title, highlight_cell=None, highlight_neighbors=None, 
                         cmap='RdBu', show_values=True, value_fmt='d'):
    """Helper to visualize a matrix"""
    h, w = matrix.shape
    
    # Create heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='equal')
    
    # Add grid
    for i in range(h + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
    for j in range(w + 1):
        ax.axvline(j - 0.5, color='black', linewidth=0.5)
    
    # Add values
    if show_values:
        for i in range(h):
            for j in range(w):
                color = 'white' if abs(matrix[i, j]) > 0.5 * np.max(np.abs(matrix)) else 'black'
                val = int(matrix[i, j]) if value_fmt == 'd' else matrix[i, j]
                ax.text(j, i, f'{val}', ha='center', va='center', 
                       fontsize=10, color=color, fontweight='bold')
    
    # Highlight center cell
    if highlight_cell:
        y, x = highlight_cell
        rect = Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, 
                         edgecolor='lime', linewidth=3)
        ax.add_patch(rect)
    
    # Highlight neighbors
    if highlight_neighbors:
        for y, x in highlight_neighbors:
            rect = Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, 
                             edgecolor='yellow', linewidth=2, linestyle='--')
            ax.add_patch(rect)
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.set_xticklabels([f'x={i}' for i in range(w)], fontsize=7)
    ax.set_yticklabels([f'y={i}' for i in range(h)], fontsize=7)
    
    return im


def create_stage1_visual():
    """Stage 1: 26-Neighbor Extrema Detection with 5x5 example"""
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Stage 1: 26-Neighbor Extrema Detection - 5×5 Example', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Create 3 DoG scales (5x5 each) - INTEGER VALUES
    np.random.seed(42)
    dog_prev = np.array([
        [ 2,  3,  1, -2,  1],
        [ 4,  5,  3,  2, -1],
        [ 3,  6,  4,  3,  2],
        [ 2,  4,  3,  1,  0],
        [ 1,  2,  1, -1, -2]
    ], dtype=np.int32)
    
    dog_curr = np.array([
        [ 3,  4,  2, -1,  2],
        [ 5,  7,  5,  3,  0],
        [ 4, 12,  6,  4,  3],  # Center (2,1) = 12 is maximum
        [ 3,  6,  4,  2,  1],
        [ 2,  3,  2,  0, -1]
    ], dtype=np.int32)
    
    dog_next = np.array([
        [ 1,  2,  1, -3,  0],
        [ 3,  4,  3,  1, -2],
        [ 2,  8,  4,  2,  1],
        [ 1,  3,  2,  0, -1],
        [ 0,  1,  0, -2, -3]
    ], dtype=np.int32)
    
    # Plot 3 scales
    ax1 = fig.add_subplot(2, 3, 1)
    create_matrix_visual(ax1, dog_prev, 'DoG Scale σ-1', highlight_cell=None)
    
    ax2 = fig.add_subplot(2, 3, 2)
    # Highlight 8 neighbors in same scale
    neighbors_same = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 2), (3, 0), (3, 1), (3, 2)]
    create_matrix_visual(ax2, dog_curr, 'DoG Scale σ (Current)\nCenter (2,1) = 0.12', 
                        highlight_cell=(2, 1), highlight_neighbors=neighbors_same)
    
    ax3 = fig.add_subplot(2, 3, 3)
    create_matrix_visual(ax3, dog_next, 'DoG Scale σ+1', highlight_cell=None)
    
    # Calculation panel
    ax4 = fig.add_subplot(2, 3, (4, 6))
    ax4.axis('off')
    
    # Calculate all 26 neighbors
    center_val = dog_curr[2, 1]
    
    # 9 neighbors from prev scale
    prev_neighbors = [dog_prev[i, j] for i in range(1, 4) for j in range(0, 3)]
    # 8 neighbors from curr scale (exclude center)
    curr_neighbors = [dog_curr[i, j] for i in range(1, 4) for j in range(0, 3) if not (i == 2 and j == 1)]
    # 9 neighbors from next scale
    next_neighbors = [dog_next[i, j] for i in range(1, 4) for j in range(0, 3)]
    
    all_neighbors = prev_neighbors + curr_neighbors + next_neighbors
    
    text = f"""
    STEP-BY-STEP CALCULATION FOR PIXEL (2, 1):
    
    ═══════════════════════════════════════════════════════════════════════════
    
    Center Value: D(2, 1, σ) = {center_val}
    
    ═══════════════════════════════════════════════════════════════════════════
    
    NEIGHBORS FROM SCALE σ-1 (9 values):
    ┌─────────────────────────────────────────────────────────────┐
    │  D(1,0,σ-1)={dog_prev[1,0]:3d}   D(1,1,σ-1)={dog_prev[1,1]:3d}   D(1,2,σ-1)={dog_prev[1,2]:3d}   │
    │  D(2,0,σ-1)={dog_prev[2,0]:3d}   D(2,1,σ-1)={dog_prev[2,1]:3d}   D(2,2,σ-1)={dog_prev[2,2]:3d}   │
    │  D(3,0,σ-1)={dog_prev[3,0]:3d}   D(3,1,σ-1)={dog_prev[3,1]:3d}   D(3,2,σ-1)={dog_prev[3,2]:3d}   │
    └─────────────────────────────────────────────────────────────┘
    Max from σ-1: {max(prev_neighbors)}
    
    NEIGHBORS FROM SCALE σ (8 values, exclude center):
    ┌─────────────────────────────────────────────────────────────┐
    │  D(1,0,σ)={dog_curr[1,0]:3d}     D(1,1,σ)={dog_curr[1,1]:3d}     D(1,2,σ)={dog_curr[1,2]:3d}     │
    │  D(2,0,σ)={dog_curr[2,0]:3d}     [CENTER]     D(2,2,σ)={dog_curr[2,2]:3d}     │
    │  D(3,0,σ)={dog_curr[3,0]:3d}     D(3,1,σ)={dog_curr[3,1]:3d}     D(3,2,σ)={dog_curr[3,2]:3d}     │
    └─────────────────────────────────────────────────────────────┘
    Max from σ: {max(curr_neighbors)}
    
    NEIGHBORS FROM SCALE σ+1 (9 values):
    ┌─────────────────────────────────────────────────────────────┐
    │  D(1,0,σ+1)={dog_next[1,0]:3d}   D(1,1,σ+1)={dog_next[1,1]:3d}   D(1,2,σ+1)={dog_next[1,2]:3d}   │
    │  D(2,0,σ+1)={dog_next[2,0]:3d}   D(2,1,σ+1)={dog_next[2,1]:3d}   D(2,2,σ+1)={dog_next[2,2]:3d}   │
    │  D(3,0,σ+1)={dog_next[3,0]:3d}   D(3,1,σ+1)={dog_next[3,1]:3d}   D(3,2,σ+1)={dog_next[3,2]:3d}   │
    └─────────────────────────────────────────────────────────────┘
    Max from σ+1: {max(next_neighbors)}
    
    ═══════════════════════════════════════════════════════════════════════════
    
    COMPARISON:
    • Center = {center_val}
    • Max of ALL 26 neighbors = {max(all_neighbors)}
    
    Is Center > ALL neighbors?  {center_val} > {max(all_neighbors)}  →  {"YES ✓ MAXIMUM" if center_val > max(all_neighbors) else "NO"}
    Is Center < ALL neighbors?  {center_val} < {min(all_neighbors)}  →  {"YES ✓ MINIMUM" if center_val < min(all_neighbors) else "NO"}
    
    RESULT: {"✓ KEYPOINT DETECTED (Local Maximum)" if center_val > max(all_neighbors) else "✓ KEYPOINT DETECTED (Local Minimum)" if center_val < min(all_neighbors) else "✗ NOT A KEYPOINT"}
    """
    
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes, fontsize=9, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.tight_layout()
    output_path = os.path.join(OUT_DIR, 'sift_5x5_stage1.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return dog_curr, (2, 1), center_val


def create_stage2_visual(dog_matrix, kp_pos, kp_val):
    """Stage 2: Low Contrast Removal with 5x5 example"""
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Stage 2: Low Contrast Removal - 5×5 Example (Integer Values)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    y, x = kp_pos
    
    # Show DoG matrix
    ax1 = fig.add_subplot(1, 2, 1)
    create_matrix_visual(ax1, dog_matrix, f'DoG Matrix\nKeypoint at ({y},{x})', 
                        highlight_cell=kp_pos)
    
    # Calculation panel
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')
    
    # Calculate gradients using finite differences
    Dx = (dog_matrix[y, x+1] - dog_matrix[y, x-1]) / 2
    Dy = (dog_matrix[y+1, x] - dog_matrix[y-1, x]) / 2
    
    # Second derivatives
    Dxx = dog_matrix[y, x+1] + dog_matrix[y, x-1] - 2 * dog_matrix[y, x]
    Dyy = dog_matrix[y+1, x] + dog_matrix[y-1, x] - 2 * dog_matrix[y, x]
    Dxy = (dog_matrix[y+1, x+1] - dog_matrix[y+1, x-1] - dog_matrix[y-1, x+1] + dog_matrix[y-1, x-1]) / 4
    
    # Hessian determinant
    det_H = Dxx * Dyy - Dxy * Dxy
    
    # Offset (sub-pixel)
    if abs(det_H) > 1e-10:
        offset_x = -(Dyy * Dx - Dxy * Dy) / det_H
        offset_y = -(Dxx * Dy - Dxy * Dx) / det_H
    else:
        offset_x, offset_y = 0, 0
    
    # Contrast at refined location
    D_refined = kp_val + 0.5 * (Dx * offset_x + Dy * offset_y)
    
    # For threshold comparison, use scaled value (since we're using integers)
    # In real SIFT, values are normalized. Here we use threshold = 3 for integers
    threshold = 3
    
    text = f"""
    STEP-BY-STEP CALCULATION FOR KEYPOINT ({y}, {x}):
    
    ═══════════════════════════════════════════════════════════════
    
    STEP 1: Extract Values from 3×3 Neighborhood
    ┌───────────────────────────────────────────────────────┐
    │  D(1,0)={dog_matrix[1,0]:3d}     D(1,1)={dog_matrix[1,1]:3d}     D(1,2)={dog_matrix[1,2]:3d}    │
    │  D(2,0)={dog_matrix[2,0]:3d}     D(2,1)={dog_matrix[2,1]:3d}     D(2,2)={dog_matrix[2,2]:3d}    │
    │  D(3,0)={dog_matrix[3,0]:3d}     D(3,1)={dog_matrix[3,1]:3d}     D(3,2)={dog_matrix[3,2]:3d}    │
    └───────────────────────────────────────────────────────┘
    
    ═══════════════════════════════════════════════════════════════
    
    STEP 2: Compute First Derivatives (Gradient)
    
    Dx = [D(y,x+1) - D(y,x-1)] / 2
       = [{dog_matrix[y, x+1]} - {dog_matrix[y, x-1]}] / 2
       = {dog_matrix[y, x+1] - dog_matrix[y, x-1]} / 2
       = {Dx}
    
    Dy = [D(y+1,x) - D(y-1,x)] / 2
       = [{dog_matrix[y+1, x]} - {dog_matrix[y-1, x]}] / 2
       = {dog_matrix[y+1, x] - dog_matrix[y-1, x]} / 2
       = {Dy}
    
    ═══════════════════════════════════════════════════════════════
    
    STEP 3: Compute Second Derivatives (Hessian)
    
    Dxx = D(y,x+1) + D(y,x-1) - 2×D(y,x)
        = {dog_matrix[y, x+1]} + {dog_matrix[y, x-1]} - 2×{dog_matrix[y, x]}
        = {dog_matrix[y, x+1] + dog_matrix[y, x-1]} - {2 * dog_matrix[y, x]}
        = {Dxx}
    
    Dyy = D(y+1,x) + D(y-1,x) - 2×D(y,x)
        = {dog_matrix[y+1, x]} + {dog_matrix[y-1, x]} - 2×{dog_matrix[y, x]}
        = {dog_matrix[y+1, x] + dog_matrix[y-1, x]} - {2 * dog_matrix[y, x]}
        = {Dyy}
    
    Dxy = [D(y+1,x+1) - D(y+1,x-1) - D(y-1,x+1) + D(y-1,x-1)] / 4
        = [{dog_matrix[y+1, x+1]} - {dog_matrix[y+1, x-1]} - {dog_matrix[y-1, x+1]} + {dog_matrix[y-1, x-1]}] / 4
        = {dog_matrix[y+1, x+1] - dog_matrix[y+1, x-1] - dog_matrix[y-1, x+1] + dog_matrix[y-1, x-1]} / 4
        = {Dxy}
    
    ═══════════════════════════════════════════════════════════════
    
    STEP 4: Compute Contrast at Refined Location
    
    Det(H) = Dxx×Dyy - Dxy²
           = {Dxx} × {Dyy} - ({Dxy})²
           = {Dxx * Dyy} - {Dxy * Dxy}
           = {det_H}
    
    D(x̂) ≈ D + 0.5 × (Dx×offset_x + Dy×offset_y)
         = {kp_val} + 0.5 × ({Dx}×{offset_x:.3f} + {Dy}×{offset_y:.3f})
         = {D_refined:.2f}
    
    ═══════════════════════════════════════════════════════════════
    
    STEP 5: Apply Threshold (threshold = {threshold} for integer values)
    
    |D(x̂)| = |{D_refined:.2f}| = {abs(D_refined):.2f}
    
    Is |D(x̂)| < {threshold}?  →  {abs(D_refined):.2f} < {threshold}  →  {"YES → REJECT" if abs(D_refined) < threshold else "NO → KEEP ✓"}
    
    RESULT: {"✗ REJECTED (Low Contrast)" if abs(D_refined) < threshold else "✓ PASSED (Sufficient Contrast)"}
    """
    
    ax2.text(0.02, 0.98, text, transform=ax2.transAxes, fontsize=9, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.tight_layout()
    output_path = os.path.join(OUT_DIR, 'sift_5x5_stage2.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return Dxx, Dyy, Dxy, det_H


def create_stage3_visual(dog_matrix, kp_pos, Dxx, Dyy, Dxy, det_H):
    """Stage 3: Edge Response Removal with 5x5 example"""
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Stage 3: Edge Response Removal - 5×5 Example (Integer Values)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    y, x = kp_pos
    
    # Show DoG matrix
    ax1 = fig.add_subplot(1, 2, 1)
    create_matrix_visual(ax1, dog_matrix, f'DoG Matrix\nKeypoint at ({y},{x})', 
                        highlight_cell=kp_pos)
    
    # Calculation panel
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')
    
    # Calculate trace and ratio
    trace_H = Dxx + Dyy
    
    if det_H > 0:
        ratio = (trace_H ** 2) / det_H
    else:
        ratio = float('inf')
    
    r = 10  # edge threshold
    threshold = (r + 1) ** 2 / r
    
    text = f"""
    STEP-BY-STEP CALCULATION FOR KEYPOINT ({y}, {x}):
    
    ═══════════════════════════════════════════════════════════════
    
    STEP 1: Hessian Matrix (from Stage 2)
    
    H = | Dxx  Dxy |   =   | {int(Dxx):4d}   {Dxy:5.2f} |
        | Dxy  Dyy |       | {Dxy:5.2f}   {int(Dyy):4d} |
    
    ═══════════════════════════════════════════════════════════════
    
    STEP 2: Compute Trace and Determinant
    
    Tr(H) = Dxx + Dyy
          = {int(Dxx)} + {int(Dyy)}
          = {int(trace_H)}
    
    Det(H) = Dxx × Dyy - Dxy²
           = {int(Dxx)} × {int(Dyy)} - ({Dxy})²
           = {int(Dxx * Dyy)} - {Dxy * Dxy}
           = {det_H}
    
    ═══════════════════════════════════════════════════════════════
    
    STEP 3: Compute Edge Response Ratio
    
    Ratio = Tr(H)² / Det(H)
          = ({int(trace_H)})² / {det_H}
          = {trace_H**2} / {det_H}
          = {ratio:.4f}
    
    ═══════════════════════════════════════════════════════════════
    
    STEP 4: Compare with Threshold
    
    For edge threshold r = {r}:
    
    Threshold = (r+1)² / r
              = ({r}+1)² / {r}
              = {(r+1)**2} / {r}
              = {threshold:.2f}
    
    ═══════════════════════════════════════════════════════════════
    
    STEP 5: Decision
    
    Is Ratio > Threshold?
    
    {ratio:.4f} > {threshold:.2f}  →  {"YES → REJECT (Edge Response)" if ratio > threshold else "NO → KEEP ✓"}
    
    INTERPRETATION:
    • Ratio ≈ 1: Blob-like (equal curvature) → Good keypoint
    • Ratio >> 1: Edge-like (elongated) → Bad keypoint
    
    RESULT: {"✗ REJECTED (Edge Response)" if ratio > threshold else "✓ PASSED (Blob-like Response)"}
    """
    
    ax2.text(0.02, 0.98, text, transform=ax2.transAxes, fontsize=9, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.tight_layout()
    output_path = os.path.join(OUT_DIR, 'sift_5x5_stage3.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_stage4_visual(dog_matrix, kp_pos, Dxx, Dyy, Dxy, det_H):
    """Stage 4: Sub-pixel Refinement with 5x5 example"""
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Stage 4: Sub-pixel Refinement - 5×5 Example (Integer Values)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    y, x = kp_pos
    
    # Calculate gradients
    Dx = (dog_matrix[y, x+1] - dog_matrix[y, x-1]) / 2
    Dy = (dog_matrix[y+1, x] - dog_matrix[y-1, x]) / 2
    
    # Show DoG matrix with sub-pixel position
    ax1 = fig.add_subplot(1, 2, 1)
    create_matrix_visual(ax1, dog_matrix, f'DoG Matrix\nDetected at ({y},{x})', 
                        highlight_cell=kp_pos)
    
    # Calculate offset
    if abs(det_H) > 1e-10:
        offset_x = -(Dyy * Dx - Dxy * Dy) / det_H
        offset_y = -(Dxx * Dy - Dxy * Dx) / det_H
    else:
        offset_x, offset_y = 0, 0
    
    # Mark refined position
    refined_x = x + offset_x
    refined_y = y + offset_y
    ax1.plot(refined_x, refined_y, 'r*', markersize=15, markeredgecolor='black', 
            markeredgewidth=1, label=f'Refined: ({refined_y:.2f}, {refined_x:.2f})')
    ax1.legend(loc='upper right')
    
    # Calculation panel
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')
    
    text = f"""
    STEP-BY-STEP CALCULATION FOR KEYPOINT ({y}, {x}):
    
    ═══════════════════════════════════════════════════════════════
    
    STEP 1: Gradient Vector (from finite differences)
    
    Dx = (D(y,x+1) - D(y,x-1)) / 2 = ({dog_matrix[y, x+1]} - {dog_matrix[y, x-1]}) / 2 = {Dx}
    Dy = (D(y+1,x) - D(y-1,x)) / 2 = ({dog_matrix[y+1, x]} - {dog_matrix[y-1, x]}) / 2 = {Dy}
    
    ∇D = | Dx |   =   | {Dx:5.1f} |
         | Dy |       | {Dy:5.1f} |
    
    ═══════════════════════════════════════════════════════════════
    
    STEP 2: Hessian Matrix (from Stage 2)
    
    H = | Dxx  Dxy |   =   | {int(Dxx):4d}   {Dxy:5.2f} |
        | Dxy  Dyy |       | {Dxy:5.2f}   {int(Dyy):4d} |
    
    Det(H) = {det_H}
    
    ═══════════════════════════════════════════════════════════════
    
    STEP 3: Compute Offset  (x̂ = -H⁻¹ × ∇D)
    
    offset_x = -(Dyy × Dx - Dxy × Dy) / Det(H)
             = -({int(Dyy)} × {Dx} - {Dxy} × {Dy}) / {det_H}
             = -({Dyy * Dx} - {Dxy * Dy}) / {det_H}
             = -{Dyy * Dx - Dxy * Dy} / {det_H}
             = {offset_x:.4f}
    
    offset_y = -(Dxx × Dy - Dxy × Dx) / Det(H)
             = -({int(Dxx)} × {Dy} - {Dxy} × {Dx}) / {det_H}
             = -({Dxx * Dy} - {Dxy * Dx}) / {det_H}
             = -{Dxx * Dy - Dxy * Dx} / {det_H}
             = {offset_y:.4f}
    
    ═══════════════════════════════════════════════════════════════
    
    STEP 4: Check Stability
    
    |offset_x| = |{offset_x:.4f}| = {abs(offset_x):.4f}
    |offset_y| = |{offset_y:.4f}| = {abs(offset_y):.4f}
    
    Threshold = 0.5
    
    Is |offset_x| > 0.5?  →  {abs(offset_x):.4f} > 0.5  →  {"YES" if abs(offset_x) > 0.5 else "NO"}
    Is |offset_y| > 0.5?  →  {abs(offset_y):.4f} > 0.5  →  {"YES" if abs(offset_y) > 0.5 else "NO"}
    
    ═══════════════════════════════════════════════════════════════
    
    STEP 5: Final Refined Position
    
    x_refined = x + offset_x = {x} + {offset_x:.4f} = {refined_x:.4f}
    y_refined = y + offset_y = {y} + {offset_y:.4f} = {refined_y:.4f}
    
    Original:  ({y}, {x})
    Refined:   ({refined_y:.4f}, {refined_x:.4f})
    
    RESULT: {"✗ REJECTED (Offset > 0.5)" if abs(offset_x) > 0.5 or abs(offset_y) > 0.5 else "✓ PASSED → Refined position: (" + f"{refined_y:.2f}, {refined_x:.2f})"}
    """
    
    ax2.text(0.02, 0.98, text, transform=ax2.transAxes, fontsize=9, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.tight_layout()
    output_path = os.path.join(OUT_DIR, 'sift_5x5_stage4.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_complete_pipeline_visual():
    """Complete pipeline summary with 5x5 example"""
    
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('SIFT Filtering: Complete 5×5 Matrix Example (Integer Values)', 
                fontsize=18, fontweight='bold', y=0.99)
    
    # Sample DoG matrix - INTEGER VALUES
    dog = np.array([
        [ 3,  4,  2, -1,  2],
        [ 5,  7,  5,  3,  0],
        [ 4, 12,  6,  4,  3],
        [ 3,  6,  4,  2,  1],
        [ 2,  3,  2,  0, -1]
    ], dtype=np.int32)
    
    kp = (2, 1)
    y, x = kp
    
    # Calculations
    Dx = (dog[y, x+1] - dog[y, x-1]) / 2
    Dy = (dog[y+1, x] - dog[y-1, x]) / 2
    Dxx = dog[y, x+1] + dog[y, x-1] - 2 * dog[y, x]
    Dyy = dog[y+1, x] + dog[y-1, x] - 2 * dog[y, x]
    Dxy = (dog[y+1, x+1] - dog[y+1, x-1] - dog[y-1, x+1] + dog[y-1, x-1]) / 4
    det_H = Dxx * Dyy - Dxy * Dxy
    trace_H = Dxx + Dyy
    
    offset_x = -(Dyy * Dx - Dxy * Dy) / det_H if abs(det_H) > 1e-10 else 0
    offset_y = -(Dxx * Dy - Dxy * Dx) / det_H if abs(det_H) > 1e-10 else 0
    D_refined = dog[y, x] + 0.5 * (Dx * offset_x + Dy * offset_y)
    ratio = (trace_H ** 2) / det_H if det_H > 0 else float('inf')
    
    # 4 stages
    # Use threshold = 3 for integer values
    threshold_contrast = 3
    
    stages = [
        ('Stage 1: Extrema', '#3498db', f'Center={dog[y,x]}\nMax neighbor=8\n→ Is Maximum? YES ✓'),
        ('Stage 2: Contrast', '#e74c3c', f'|D(x̂)|={abs(D_refined):.1f}\nThreshold={threshold_contrast}\n→ Keep? {"YES ✓" if abs(D_refined) >= threshold_contrast else "NO ✗"}'),
        ('Stage 3: Edge', '#f39c12', f'Ratio={ratio:.2f}\nThreshold=12.1\n→ Keep? {"YES ✓" if ratio <= 12.1 else "NO ✗"}'),
        ('Stage 4: Refine', '#27ae60', f'offset=({offset_x:.3f},{offset_y:.3f})\n|offset|<0.5?\n→ Keep? {"YES ✓" if abs(offset_x) <= 0.5 and abs(offset_y) <= 0.5 else "NO ✗"}'),
    ]
    
    for i, (title, color, result) in enumerate(stages):
        ax = fig.add_subplot(2, 4, i + 1)
        create_matrix_visual(ax, dog, title, highlight_cell=kp)
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        
        # Add result box
        ax.text(0.5, -0.15, result, transform=ax.transAxes, ha='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    
    # Summary panel
    ax = fig.add_subplot(2, 1, 2)
    ax.axis('off')
    
    summary = f"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                              COMPLETE FILTERING SUMMARY (INTEGER VALUES)                               ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                         ║
    ║   INPUT: 5×5 DoG Matrix, Candidate Keypoint at (2, 1), Value = {dog[y,x]}                               ║
    ║                                                                                                         ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                         ║
    ║   STAGE 1: 26-Neighbor Extrema Detection                                                               ║
    ║   ────────────────────────────────────────────────────────────────────────────────────────────────     ║
    ║   • Center value: {dog[y,x]}                                                                            ║
    ║   • Maximum of 26 neighbors: 8                                                                         ║
    ║   • Test: {dog[y,x]} > 8 → YES, this is a local maximum                                                 ║
    ║   • RESULT: ✓ PASSED                                                                                   ║
    ║                                                                                                         ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                         ║
    ║   STAGE 2: Low Contrast Removal                                                                        ║
    ║   ────────────────────────────────────────────────────────────────────────────────────────────────     ║
    ║   • Gradient: ∇D = [{Dx}, {Dy}]                                                                         ║
    ║   • Hessian: Dxx={int(Dxx)}, Dyy={int(Dyy)}, Dxy={Dxy}                                                  ║
    ║   • Refined contrast |D(x̂)| = {abs(D_refined):.1f}                                                      ║
    ║   • Test: {abs(D_refined):.1f} ≥ {threshold_contrast} → {"YES" if abs(D_refined) >= threshold_contrast else "NO"}                                     ║
    ║   • RESULT: {"✓ PASSED" if abs(D_refined) >= threshold_contrast else "✗ REJECTED"}                      ║
    ║                                                                                                         ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                         ║
    ║   STAGE 3: Edge Response Removal                                                                       ║
    ║   ────────────────────────────────────────────────────────────────────────────────────────────────     ║
    ║   • Tr(H) = {int(trace_H)}, Det(H) = {det_H}                                                            ║
    ║   • Ratio = Tr(H)²/Det(H) = {int(trace_H)}² / {det_H} = {ratio:.2f}                                     ║
    ║   • Threshold = (10+1)²/10 = 12.1                                                                      ║
    ║   • Test: {ratio:.2f} ≤ 12.1 → {"YES" if ratio <= 12.1 else "NO"}                                       ║
    ║   • RESULT: {"✓ PASSED" if ratio <= 12.1 else "✗ REJECTED"}                                             ║
    ║                                                                                                         ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                         ║
    ║   STAGE 4: Sub-pixel Refinement                                                                        ║
    ║   ────────────────────────────────────────────────────────────────────────────────────────────────     ║
    ║   • Offset: ({offset_x:.4f}, {offset_y:.4f})                                                            ║
    ║   • Test: |{offset_x:.4f}| ≤ 0.5 AND |{offset_y:.4f}| ≤ 0.5 → {"YES" if abs(offset_x) <= 0.5 and abs(offset_y) <= 0.5 else "NO"}  ║
    ║   • Refined position: ({y + offset_y:.4f}, {x + offset_x:.4f})                                          ║
    ║   • RESULT: {"✓ PASSED" if abs(offset_x) <= 0.5 and abs(offset_y) <= 0.5 else "✗ REJECTED"}             ║
    ║                                                                                                         ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                         ║
    ║   FINAL RESULT: ✓ KEYPOINT ACCEPTED                                                                    ║
    ║   Final Position: ({y + offset_y:.4f}, {x + offset_x:.4f})                                              ║
    ║                                                                                                         ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, summary, transform=ax.transAxes, fontsize=9, 
            ha='center', va='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#34495e', linewidth=2))
    
    plt.tight_layout()
    output_path = os.path.join(OUT_DIR, 'sift_5x5_complete.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("Generating 5×5 matrix examples for each SIFT filtering stage...")
    
    # Stage 1
    dog_matrix, kp_pos, kp_val = create_stage1_visual()
    
    # Stage 2
    Dxx, Dyy, Dxy, det_H = create_stage2_visual(dog_matrix, kp_pos, kp_val)
    
    # Stage 3
    create_stage3_visual(dog_matrix, kp_pos, Dxx, Dyy, Dxy, det_H)
    
    # Stage 4
    create_stage4_visual(dog_matrix, kp_pos, Dxx, Dyy, Dxy, det_H)
    
    # Complete summary
    create_complete_pipeline_visual()
    
    print("\nDone! Generated 5 images showing 5×5 matrix examples.")

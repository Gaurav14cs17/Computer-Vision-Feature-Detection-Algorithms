"""
HOG Math Formulas Visualization
Visual representations of all mathematical formulas used in HOG
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')
os.makedirs(OUT_DIR, exist_ok=True)


def create_gradient_formulas_diagram():
    """Create diagram showing gradient computation formulas"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # --- Panel 1: Horizontal Gradient ---
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Horizontal Gradient: Gx', fontsize=14, fontweight='bold', color='blue')
    
    # 3x3 pixel grid
    for i in range(4):
        ax.axhline(y=2 + i*1.2, xmin=0.15, xmax=0.55, color='black', linewidth=1.5)
        ax.axvline(x=1.5 + i*1.2, ymin=0.25, ymax=0.7, color='black', linewidth=1.5)
    
    # Pixel labels
    pixel_labels = [
        [(1.9, 5.8), 'I(x-1,y-1)'], [(3.1, 5.8), 'I(x,y-1)'], [(4.3, 5.8), 'I(x+1,y-1)'],
        [(1.9, 4.6), 'I(x-1,y)'], [(3.1, 4.6), 'I(x,y)', 'red'], [(4.3, 4.6), 'I(x+1,y)'],
        [(1.9, 3.4), 'I(x-1,y+1)'], [(3.1, 3.4), 'I(x,y+1)'], [(4.3, 3.4), 'I(x+1,y+1)'],
    ]
    for item in pixel_labels:
        pos, label = item[0], item[1]
        color = item[2] if len(item) > 2 else 'black'
        ax.text(pos[0], pos[1], label, fontsize=8, ha='center', va='center', color=color)
    
    # Highlight used pixels
    ax.add_patch(Rectangle((1.5, 3.8), 1.2, 1.2, fill=True, facecolor='lightblue', edgecolor='blue', linewidth=2, alpha=0.5))
    ax.add_patch(Rectangle((3.9, 3.8), 1.2, 1.2, fill=True, facecolor='lightblue', edgecolor='blue', linewidth=2, alpha=0.5))
    
    # Formula
    ax.text(7.5, 5.5, r'$G_x = I(x+1, y) - I(x-1, y)$', fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='blue', linewidth=2))
    
    # Arrow showing direction
    ax.annotate('', xy=(5.3, 4.4), xytext=(1.3, 4.4),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(3.3, 4.0, 'Direction', fontsize=10, ha='center', color='blue')
    
    # Kernel
    ax.text(7.5, 3.5, 'Filter kernel:\n[-1,  0,  +1]', fontsize=11, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightgray'))
    
    # --- Panel 2: Vertical Gradient ---
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Vertical Gradient: Gy', fontsize=14, fontweight='bold', color='green')
    
    # 3x3 pixel grid
    for i in range(4):
        ax.axhline(y=2 + i*1.2, xmin=0.15, xmax=0.55, color='black', linewidth=1.5)
        ax.axvline(x=1.5 + i*1.2, ymin=0.25, ymax=0.7, color='black', linewidth=1.5)
    
    # Pixel labels
    for item in pixel_labels:
        pos, label = item[0], item[1]
        color = item[2] if len(item) > 2 else 'black'
        ax.text(pos[0], pos[1], label, fontsize=8, ha='center', va='center', color=color)
    
    # Highlight used pixels
    ax.add_patch(Rectangle((2.7, 5.0), 1.2, 1.2, fill=True, facecolor='lightgreen', edgecolor='green', linewidth=2, alpha=0.5))
    ax.add_patch(Rectangle((2.7, 2.6), 1.2, 1.2, fill=True, facecolor='lightgreen', edgecolor='green', linewidth=2, alpha=0.5))
    
    # Formula
    ax.text(7.5, 5.5, r'$G_y = I(x, y+1) - I(x, y-1)$', fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='green', linewidth=2))
    
    # Arrow showing direction
    ax.annotate('', xy=(3.3, 2.4), xytext=(3.3, 6.4),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(3.8, 4.4, 'Direction', fontsize=10, ha='center', color='green', rotation=90)
    
    # Kernel
    ax.text(7.5, 3.5, 'Filter kernel:\n[-1]\n[ 0]\n[+1]', fontsize=11, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightgray'))
    
    # --- Panel 3: Magnitude ---
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Gradient Magnitude', fontsize=14, fontweight='bold', color='purple')
    
    # Right triangle diagram
    ax.plot([2, 6, 6, 2], [3, 3, 6, 3], 'k-', linewidth=2)
    ax.annotate('', xy=(6, 3), xytext=(2, 3), arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.annotate('', xy=(6, 6), xytext=(6, 3), arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.annotate('', xy=(6, 6), xytext=(2, 3), arrowprops=dict(arrowstyle='->', color='purple', lw=3))
    
    ax.text(4, 2.5, r'$G_x$', fontsize=14, ha='center', color='blue', fontweight='bold')
    ax.text(6.5, 4.5, r'$G_y$', fontsize=14, ha='center', color='green', fontweight='bold')
    ax.text(3.5, 5.2, r'$M$', fontsize=14, ha='center', color='purple', fontweight='bold')
    
    # Formula
    ax.text(5, 7.2, r'$M = \sqrt{G_x^2 + G_y^2}$', fontsize=16, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='plum', edgecolor='purple', linewidth=2))
    
    # Example
    ax.text(8.5, 4.5, 'Example:\nGx = 0.3\nGy = 0.4\n\nM = √(0.09+0.16)\nM = √0.25\nM = 0.5', 
            fontsize=10, ha='center', va='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    # --- Panel 4: Direction ---
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Gradient Direction (Unsigned)', fontsize=14, fontweight='bold', color='orange')
    
    # Unit circle (half)
    theta = np.linspace(0, np.pi, 100)
    r = 2
    center = (5, 4)
    ax.plot(center[0] + r * np.cos(theta), center[1] + r * np.sin(theta), 'k-', linewidth=2)
    ax.plot([center[0] - r, center[0] + r], [center[1], center[1]], 'k-', linewidth=2)
    
    # Angle markers
    angles = [0, 20, 45, 90, 135, 160, 180]
    for angle in angles:
        rad = np.radians(angle)
        x = center[0] + r * np.cos(rad)
        y = center[1] + r * np.sin(rad)
        ax.plot([center[0], x], [center[1], y], 'gray', linewidth=1, linestyle='--')
        ax.text(center[0] + (r+0.5) * np.cos(rad), center[1] + (r+0.5) * np.sin(rad), 
               f'{angle}°', fontsize=9, ha='center', va='center')
    
    # Example gradient vector
    gx, gy = 0.3, 0.4
    angle = np.degrees(np.arctan2(gy, gx))
    ax.annotate('', xy=(center[0] + r * np.cos(np.radians(angle)), 
                        center[1] + r * np.sin(np.radians(angle))),
                xytext=center,
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax.text(center[0] + (r-0.5) * np.cos(np.radians(angle+15)), 
           center[1] + (r-0.5) * np.sin(np.radians(angle+15)),
           f'θ={angle:.1f}°', fontsize=11, color='red', fontweight='bold')
    
    # Formula
    ax.text(5, 7.2, r'$\theta = \arctan\left(\frac{G_y}{G_x}\right)$ mod 180°', fontsize=14, ha='center',
            bbox=dict(boxstyle='round', facecolor='moccasin', edgecolor='orange', linewidth=2))
    
    ax.text(5, 1.5, 'Range: 0° to 180° (unsigned)\nOpposite directions = same bin', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    plt.suptitle('HOG Gradient Computation Formulas', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_gradient_formulas.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_gradient_formulas.png")


def create_histogram_formulas_diagram():
    """Create diagram showing histogram construction formulas"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # --- Panel 1: 9-bin histogram ---
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('9-Bin Orientation Histogram', fontsize=14, fontweight='bold')
    
    # Draw bins
    bin_width = 0.8
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    for i in range(9):
        rect = Rectangle((0.5 + i*bin_width, 3), bin_width-0.05, 3, 
                         facecolor=colors[i], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(0.5 + i*bin_width + bin_width/2, 2.5, f'{i*20}°', fontsize=9, ha='center')
        ax.text(0.5 + i*bin_width + bin_width/2, 6.3, f'Bin {i}', fontsize=8, ha='center')
    
    # Formula
    ax.text(5, 7.5, 'Bin width = 180° / 9 = 20° per bin', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
    
    ax.text(5, 1.2, 'Bins: [0°-20°), [20°-40°), [40°-60°), ... [160°-180°)', fontsize=10, ha='center')
    
    # --- Panel 2: Bilinear interpolation ---
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Bilinear Interpolation (Soft Binning)', fontsize=14, fontweight='bold')
    
    # Two adjacent bins
    ax.add_patch(Rectangle((2, 3), 2.5, 3, facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.add_patch(Rectangle((4.5, 3), 2.5, 3, facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax.text(3.25, 6.3, 'Bin k', fontsize=12, ha='center', fontweight='bold')
    ax.text(5.75, 6.3, 'Bin k+1', fontsize=12, ha='center', fontweight='bold')
    ax.text(3.25, 2.5, f'{20}° center', fontsize=10, ha='center')
    ax.text(5.75, 2.5, f'{40}° center', fontsize=10, ha='center')
    
    # Gradient angle marker
    angle_pos = 4.2
    ax.plot([angle_pos, angle_pos], [3, 6], 'r-', linewidth=2)
    ax.plot(angle_pos, 4.5, 'ro', markersize=10)
    ax.text(angle_pos, 6.5, 'θ=35°', fontsize=11, ha='center', color='red', fontweight='bold')
    
    # Interpolation formula
    ax.text(5, 1.5, r'$w_{lower} = 1 - \frac{\theta - \theta_{lower}}{\Delta\theta}$', fontsize=12, ha='center')
    ax.text(5, 0.8, r'$w_{upper} = \frac{\theta - \theta_{lower}}{\Delta\theta}$', fontsize=12, ha='center')
    
    # Example
    ax.text(8.5, 5, 'Example:\nθ = 35°\n\nwₖ = 0.25\nwₖ₊₁ = 0.75', fontsize=10, ha='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    # --- Panel 3: Magnitude weighting ---
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Magnitude-Weighted Voting', fontsize=14, fontweight='bold')
    
    # Show pixel contributing to histogram
    ax.add_patch(Rectangle((1, 4), 1.5, 1.5, facecolor='lightyellow', edgecolor='black', linewidth=2))
    ax.text(1.75, 4.75, 'Pixel\n(x,y)', fontsize=10, ha='center')
    
    # Arrow to histogram
    ax.annotate('', xy=(5, 4.5), xytext=(2.7, 4.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # Mini histogram
    sample_hist = [0.1, 0.15, 0.25, 0.3, 0.15, 0.05, 0, 0, 0]
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    for i in range(9):
        height = sample_hist[i] * 4
        rect = Rectangle((5 + i*0.45, 3), 0.4, height, facecolor=colors[i], edgecolor='black')
        ax.add_patch(rect)
    ax.plot([5, 5 + 9*0.45], [3, 3], 'k-', linewidth=1)
    ax.text(7, 6.5, 'Cell Histogram', fontsize=11, ha='center', fontweight='bold')
    
    # Formula
    ax.text(5, 1.5, r'Vote = Magnitude × Interpolation Weight', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
    ax.text(5, 0.7, r'H[bin] += M(x,y) × w$_{bin}$', fontsize=13, ha='center')
    
    # --- Panel 4: Complete cell histogram ---
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Cell Histogram: 8×8 = 64 Pixels → 9 Values', fontsize=14, fontweight='bold')
    
    # 8x8 pixel cell
    cell_size = 0.35
    for i in range(8):
        for j in range(8):
            color = plt.cm.Blues(0.3 + 0.7 * np.random.random())
            rect = Rectangle((1 + j*cell_size, 2.5 + (7-i)*cell_size), cell_size, cell_size,
                            facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
    ax.text(2.4, 6.2, '8×8 Cell', fontsize=11, ha='center', fontweight='bold')
    ax.text(2.4, 2.1, '64 pixels', fontsize=10, ha='center')
    
    # Arrow
    ax.annotate('', xy=(5.5, 4), xytext=(4.2, 4), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(4.85, 4.5, 'Accumulate', fontsize=10, ha='center')
    
    # 9-bin histogram
    sample_hist = [0.2, 0.35, 0.5, 0.8, 0.6, 0.4, 0.25, 0.15, 0.1]
    bar_width = 0.35
    for i in range(9):
        height = sample_hist[i] * 3
        rect = Rectangle((5.5 + i*bar_width, 2.5), bar_width-0.02, height,
                         facecolor=plt.cm.hsv(i/18), edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
    ax.plot([5.5, 5.5 + 9*bar_width], [2.5, 2.5], 'k-', linewidth=1)
    ax.text(7.1, 6.2, '9-Bin Histogram', fontsize=11, ha='center', fontweight='bold')
    ax.text(7.1, 2.0, '9 values', fontsize=10, ha='center')
    
    # Dimension reduction
    ax.text(5, 7.2, 'Dimension: 64 → 9 (compression ratio ~7:1)', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgray'))
    
    plt.suptitle('HOG Histogram Construction Formulas', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_histogram_formulas.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_histogram_formulas.png")


def create_normalization_formulas_diagram():
    """Create diagram showing block normalization formulas"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # --- Panel 1: Block structure (2×2 cells) ---
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Block = 2×2 Cells', fontsize=14, fontweight='bold')
    
    # Draw 2×2 block
    cell_size = 1.8
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    labels = ['C₀', 'C₁', 'C₂', 'C₃']
    for i in range(2):
        for j in range(2):
            rect = Rectangle((2 + j*cell_size, 3 + (1-i)*cell_size), cell_size, cell_size,
                            facecolor=colors[i*2+j], edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(2 + j*cell_size + cell_size/2, 3 + (1-i)*cell_size + cell_size/2,
                   labels[i*2+j], fontsize=14, ha='center', va='center', fontweight='bold')
    
    ax.text(3.8, 6.8, '2×2 Block', fontsize=12, ha='center', fontweight='bold')
    ax.text(3.8, 2.5, 'Each cell: 9-bin histogram', fontsize=10, ha='center')
    
    # Arrow to vector
    ax.annotate('', xy=(7.5, 4.5), xytext=(6, 4.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # Block vector
    ax.text(8.3, 5.5, 'Block Vector:\n[C₀ | C₁ | C₂ | C₃]', fontsize=11, ha='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    ax.text(8.3, 3.5, '4 cells × 9 bins\n= 36 values', fontsize=10, ha='center')
    
    # --- Panel 2: L2 Normalization formula ---
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('L2 Normalization', fontsize=14, fontweight='bold', color='purple')
    
    # Formula box
    ax.text(5, 6, r'$v_{norm} = \frac{v}{\sqrt{\|v\|_2^2 + \epsilon^2}}$', fontsize=18, ha='center',
            bbox=dict(boxstyle='round', facecolor='plum', edgecolor='purple', linewidth=2))
    
    ax.text(5, 4.5, r'Where: $\|v\|_2 = \sqrt{\sum_{i=1}^{36} v_i^2}$', fontsize=14, ha='center')
    
    ax.text(5, 3.2, 'ε = small constant (e.g., 1e-6) for numerical stability', fontsize=11, ha='center')
    
    # Example
    ax.text(5, 1.5, 'Example: v = [0.3, 0.4, 0.5, ...]\n||v||₂ = 0.707\nvₙₒᵣₘ = v / 0.707', fontsize=10, ha='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    # --- Panel 3: Overlapping blocks ---
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Overlapping Blocks (50% Overlap)', fontsize=14, fontweight='bold')
    
    # Draw cell grid
    cell_size = 0.9
    for i in range(4):
        for j in range(5):
            rect = Rectangle((1 + j*cell_size, 2 + (3-i)*cell_size), cell_size, cell_size,
                            facecolor='white', edgecolor='black', linewidth=1)
            ax.add_patch(rect)
    
    # First block (red)
    rect = Rectangle((1, 2 + 2*cell_size), 2*cell_size, 2*cell_size,
                     facecolor='none', edgecolor='red', linewidth=3)
    ax.add_patch(rect)
    ax.text(1 + cell_size, 6.2, 'Block 1', fontsize=10, ha='center', color='red', fontweight='bold')
    
    # Second block (blue) - overlapping
    rect = Rectangle((1 + cell_size, 2 + 2*cell_size), 2*cell_size, 2*cell_size,
                     facecolor='none', edgecolor='blue', linewidth=3, linestyle='--')
    ax.add_patch(rect)
    ax.text(1 + 2*cell_size, 6.5, 'Block 2', fontsize=10, ha='center', color='blue', fontweight='bold')
    
    # Overlap area
    rect = Rectangle((1 + cell_size, 2 + 2*cell_size), cell_size, 2*cell_size,
                     facecolor='purple', alpha=0.3, edgecolor='none')
    ax.add_patch(rect)
    
    ax.text(5, 1.2, 'Stride = 1 cell = 50% overlap\nSame cells contribute to multiple blocks', fontsize=10, ha='center')
    
    # Formula for block count
    ax.text(7.5, 5, 'Number of blocks:', fontsize=11, ha='center', fontweight='bold')
    ax.text(7.5, 4, r'$n_x = (cells_x - 1)$', fontsize=12, ha='center')
    ax.text(7.5, 3, r'$n_y = (cells_y - 1)$', fontsize=12, ha='center')
    
    # --- Panel 4: Dimension calculation ---
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('HOG Descriptor Dimension (64×128 window)', fontsize=14, fontweight='bold')
    
    # Step-by-step calculation
    calc_text = """
    Image:     64 × 128 pixels
    
    Cells:     64/8 × 128/8 = 8 × 16 = 128 cells
    
    Blocks:    (8-1) × (16-1) = 7 × 15 = 105 blocks
    
    Features:  105 blocks × 4 cells × 9 bins
             = 105 × 36
             = 3780 dimensions
    """
    ax.text(5, 4.5, calc_text, fontsize=11, ha='center', va='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))
    
    ax.text(5, 1, 'Final HOG descriptor: 3780-D vector', fontsize=13, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', linewidth=2))
    
    plt.suptitle('HOG Block Normalization Formulas', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_normalization_formulas.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_normalization_formulas.png")


def create_math_summary_diagram():
    """Create summary of all HOG math formulas"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'HOG Algorithm - Complete Mathematical Summary', fontsize=16, 
            ha='center', fontweight='bold', bbox=dict(facecolor='lightblue', edgecolor='blue', linewidth=2))
    
    # Step 1: Preprocessing
    y = 8.2
    ax.text(0.5, y, 'Step 1: Preprocessing', fontsize=12, fontweight='bold', color='darkblue')
    ax.text(0.5, y-0.5, r'$I_{gamma} = I^{\gamma}$  (typically γ = 0.5)', fontsize=11)
    
    # Step 2: Gradients
    y = 7.0
    ax.text(0.5, y, 'Step 2: Gradient Computation', fontsize=12, fontweight='bold', color='darkblue')
    ax.text(0.5, y-0.4, r'$G_x = I(x+1, y) - I(x-1, y)$', fontsize=11)
    ax.text(0.5, y-0.8, r'$G_y = I(x, y+1) - I(x, y-1)$', fontsize=11)
    ax.text(0.5, y-1.2, r'$M = \sqrt{G_x^2 + G_y^2}$', fontsize=11)
    ax.text(0.5, y-1.6, r'$\theta = \arctan(G_y / G_x)$ mod 180°', fontsize=11)
    
    # Step 3: Histograms
    y = 4.6
    ax.text(0.5, y, 'Step 3: Cell Histograms', fontsize=12, fontweight='bold', color='darkblue')
    ax.text(0.5, y-0.4, r'Cell size: 8×8 pixels', fontsize=11)
    ax.text(0.5, y-0.8, r'Bins: 9 (each 20° wide, 0° to 180°)', fontsize=11)
    ax.text(0.5, y-1.2, r'Bilinear interpolation: $w = 1 - |(\theta - \theta_{center})| / \Delta\theta$', fontsize=11)
    ax.text(0.5, y-1.6, r'Voting: H[bin] += M × w', fontsize=11)
    
    # Step 4: Block normalization
    y = 2.2
    ax.text(0.5, y, 'Step 4: Block Normalization', fontsize=12, fontweight='bold', color='darkblue')
    ax.text(0.5, y-0.4, r'Block: 2×2 cells = 36 values', fontsize=11)
    ax.text(0.5, y-0.8, r'L2-norm: $v_{norm} = v / \sqrt{||v||_2^2 + \epsilon^2}$', fontsize=11)
    ax.text(0.5, y-1.2, r'Overlap: 50% (stride = 1 cell)', fontsize=11)
    
    # Dimension box
    ax.text(8.5, 5.5, '64×128 Window\nDimension Calculation', fontsize=12, ha='center', fontweight='bold',
            bbox=dict(facecolor='lightyellow', edgecolor='orange', linewidth=2))
    
    dim_text = """
    Cells:   8 × 16 = 128
    Blocks:  7 × 15 = 105
    Values:  105 × 36 = 3780-D
    """
    ax.text(8.5, 4, dim_text, fontsize=11, ha='center', family='monospace',
            bbox=dict(facecolor='white', edgecolor='gray'))
    
    # Key properties
    ax.text(8.5, 2.5, 'Key Properties', fontsize=12, ha='center', fontweight='bold',
            bbox=dict(facecolor='lightgreen', edgecolor='green', linewidth=2))
    props_text = """
    ✓ Illumination invariant (L2 norm)
    ✓ Captures local shape
    ✓ Unsigned gradients (0°-180°)
    ✓ Dense representation
    """
    ax.text(8.5, 1.3, props_text, fontsize=10, ha='center',
            bbox=dict(facecolor='white', edgecolor='gray'))
    
    plt.savefig(os.path.join(OUT_DIR, 'hog_math_summary.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_math_summary.png")


if __name__ == "__main__":
    print("Generating HOG Math Formula Visualizations...")
    print("-" * 50)
    create_gradient_formulas_diagram()
    create_histogram_formulas_diagram()
    create_normalization_formulas_diagram()
    create_math_summary_diagram()
    print("-" * 50)
    print("Done!")

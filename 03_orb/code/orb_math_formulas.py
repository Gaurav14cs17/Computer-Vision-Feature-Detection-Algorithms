"""
Visual diagram showing math formulas for ORB algorithm stages
With detailed step-by-step derivations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, '..', 'images')


def create_fast_formulas_visual():
    """Create visual diagram with FAST corner detection formulas."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    # ============================================
    # Panel 1: FAST Threshold Test
    # ============================================
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'FAST Threshold Test', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    formula_box = FancyBboxPatch((0.5, 2.5), 9, 5.5, boxstyle="round,pad=0.05",
                                  facecolor='#ebf5fb', edgecolor='#3498db', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.5, 'For center pixel p with intensity Iₚ:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 6.7, 'Upper bound: Iₚ + t', ha='center', fontsize=11)
    ax.text(5, 6.0, 'Lower bound: Iₚ - t', ha='center', fontsize=11)
    ax.text(5, 5.0, 'For each circle pixel c with intensity Ic:', ha='center', fontsize=10)
    ax.text(5, 4.2, 'BRIGHTER if Ic > Iₚ + t  (label = B)', ha='center', fontsize=10, color='#c0392b')
    ax.text(5, 3.5, 'DARKER if Ic < Iₚ - t  (label = D)', ha='center', fontsize=10, color='#2980b9')
    ax.text(5, 2.8, 'SIMILAR otherwise  (label = S)', ha='center', fontsize=10, color='#7f8c8d')
    
    note_box = FancyBboxPatch((1, 0.5), 8, 1.5, boxstyle="round,pad=0.05",
                               facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(note_box)
    ax.text(5, 1.25, 'Default threshold: t = 20 (for 0-255 images)\nt = 0.08 (for 0-1 normalized)', 
           ha='center', fontsize=10, style='italic')
    
    # ============================================
    # Panel 2: 16-Pixel Bresenham Circle
    # ============================================
    ax = axes[0, 1]
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title('16-Pixel Bresenham Circle', fontsize=12, fontweight='bold', pad=10)
    
    # Draw grid
    for i in range(-4, 5):
        ax.axhline(y=i, color='lightgray', linewidth=0.5)
        ax.axvline(x=i, color='lightgray', linewidth=0.5)
    
    # Circle offsets
    offsets = [
        (0, -3), (1, -3), (2, -2), (3, -1),
        (3, 0), (3, 1), (2, 2), (1, 3),
        (0, 3), (-1, 3), (-2, 2), (-3, 1),
        (-3, 0), (-3, -1), (-2, -2), (-1, -3)
    ]
    
    # Draw center
    center = Rectangle((-0.5, -0.5), 1, 1, fill=True, facecolor='red', edgecolor='black', linewidth=2)
    ax.add_patch(center)
    ax.text(0, 0, 'p', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Draw circle pixels
    colors = plt.cm.hsv(np.linspace(0, 1, 16))
    for i, (dx, dy) in enumerate(offsets):
        rect = Rectangle((dx - 0.4, -dy - 0.4), 0.8, 0.8, fill=True, 
                        facecolor=colors[i], edgecolor='black', linewidth=1, alpha=0.8)
        ax.add_patch(rect)
        ax.text(dx, -dy, str(i+1), ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw circle
    circle = Circle((0, 0), 3, fill=False, edgecolor='blue', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.set_xlabel('x offset', fontsize=10)
    ax.set_ylabel('y offset', fontsize=10)
    
    # ============================================
    # Panel 3: Corner Detection Condition
    # ============================================
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'FAST-9 Corner Condition', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    formula_box = FancyBboxPatch((0.5, 3), 9, 5, boxstyle="round,pad=0.05",
                                  facecolor='#fdedec', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.5, 'Corner detected if:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 6.5, '9 or more CONTIGUOUS pixels are B', ha='center', fontsize=11, color='#c0392b')
    ax.text(5, 5.8, 'OR', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 5.1, '9 or more CONTIGUOUS pixels are D', ha='center', fontsize=11, color='#2980b9')
    ax.text(5, 4.0, '(contiguous includes wrap-around:', ha='center', fontsize=10, style='italic')
    ax.text(5, 3.4, 'pixel 16 is adjacent to pixel 1)', ha='center', fontsize=10, style='italic')
    
    # High-speed test
    speedtest_box = FancyBboxPatch((0.5, 0.3), 9, 2.3, boxstyle="round,pad=0.05",
                                    facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(speedtest_box)
    ax.text(5, 2.2, 'High-Speed Pre-test:', ha='center', fontsize=10, fontweight='bold', color='#1e8449')
    ax.text(5, 1.5, 'Check pixels 1, 5, 9, 13 (N, E, S, W) first', ha='center', fontsize=9)
    ax.text(5, 0.8, 'Skip full test if < 3 of these match', ha='center', fontsize=9, color='#1e8449')
    
    # ============================================
    # Panel 4: Example Calculation
    # ============================================
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#9b59b6', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Example: FAST Test', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    example_box = FancyBboxPatch((0.5, 0.5), 9, 7.5, boxstyle="round,pad=0.05",
                                  facecolor='#f4ecf7', edgecolor='#9b59b6', linewidth=2)
    ax.add_patch(example_box)
    
    example_text = """Iₚ = 100, t = 20
Upper = 120, Lower = 80

Circle intensities:
Pos:  1   2   3   4   5   6   7   8
I:   60  55  58  62  65  70  68  72
     D   D   D   D   D   D   D   D

Pos:  9  10  11  12  13  14  15  16
I:  140 145 150 148 145 140  95  65
     B   B   B   B   B   B   S   D

Darker: positions 1-8 = 8 consecutive
Brighter: positions 9-14 = 6 consecutive

Result: 8 < 9 and 6 < 9 → NOT a corner"""
    
    ax.text(5, 4.5, example_text, ha='center', va='center', fontsize=9, 
           family='monospace', linespacing=1.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_fast_formulas.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_fast_formulas.png")


def create_harris_formulas_visual():
    """Create visual diagram with Harris corner response formulas."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    # ============================================
    # Panel 1: Gradient Computation
    # ============================================
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Step 1: Gradient Computation', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    formula_box = FancyBboxPatch((0.5, 1.5), 9, 6.5, boxstyle="round,pad=0.05",
                                  facecolor='#ebf5fb', edgecolor='#3498db', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.5, 'Sobel Operators:', ha='center', fontsize=11, fontweight='bold')
    
    ax.text(2.5, 6.2, 'Sobel X:', ha='center', fontsize=10)
    ax.text(2.5, 5.3, '[-1  0  +1]\n[-2  0  +2]\n[-1  0  +1]', ha='center', fontsize=9, family='monospace')
    
    ax.text(7.5, 6.2, 'Sobel Y:', ha='center', fontsize=10)
    ax.text(7.5, 5.3, '[-1  -2  -1]\n[ 0   0   0]\n[+1  +2  +1]', ha='center', fontsize=9, family='monospace')
    
    ax.text(5, 3.5, 'Gradients:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 2.7, 'Ix = Sobel_X * I(x, y)', ha='center', fontsize=10)
    ax.text(5, 2.0, 'Iy = Sobel_Y * I(x, y)', ha='center', fontsize=10)
    
    # ============================================
    # Panel 2: Structure Tensor
    # ============================================
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Step 2: Structure Tensor', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    formula_box = FancyBboxPatch((0.5, 1.5), 9, 6.5, boxstyle="round,pad=0.05",
                                  facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.5, 'Structure Tensor (2×2 matrix):', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 6.3, 'M = Σ w(u,v) [Ix²      Ix·Iy]\n              [Ix·Iy    Iy² ]', 
           ha='center', fontsize=10, family='monospace')
    
    ax.text(5, 4.5, 'Simplified notation:', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 3.5, 'M = [A   C]    where A = Σ w·Ix²\n    [C   B]          B = Σ w·Iy²\n                     C = Σ w·Ix·Iy', 
           ha='center', fontsize=10, family='monospace')
    
    ax.text(5, 1.8, 'w(u,v) = Gaussian weight', ha='center', fontsize=9, style='italic')
    
    # ============================================
    # Panel 3: Harris Response
    # ============================================
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Step 3: Harris Response', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    formula_box = FancyBboxPatch((0.5, 2.5), 9, 5.5, boxstyle="round,pad=0.05",
                                  facecolor='#fdedec', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.5, 'Harris Corner Response:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 6.5, 'R = det(M) - k × trace(M)²', ha='center', fontsize=13, fontweight='bold')
    
    ax.text(5, 5.2, 'where:', ha='center', fontsize=10)
    ax.text(5, 4.5, 'det(M) = A×B - C²', ha='center', fontsize=10)
    ax.text(5, 3.8, 'trace(M) = A + B', ha='center', fontsize=10)
    ax.text(5, 3.1, 'k = 0.04 (Harris constant)', ha='center', fontsize=10, style='italic')
    
    # Interpretation
    interp_box = FancyBboxPatch((1, 0.5), 8, 1.5, boxstyle="round,pad=0.05",
                                 facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(interp_box)
    ax.text(5, 1.25, 'R > 0 → Corner  |  R < 0 → Edge  |  R ≈ 0 → Flat', 
           ha='center', fontsize=10, fontweight='bold')
    
    # ============================================
    # Panel 4: Worked Example
    # ============================================
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#27ae60', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Example: Harris Calculation', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    example_box = FancyBboxPatch((0.5, 0.5), 9, 7.5, boxstyle="round,pad=0.05",
                                  facecolor='#e9f7ef', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(example_box)
    
    example_text = """Given weighted gradient sums:
A = Σ w·Ix² = 220
B = Σ w·Iy² = 180
C = Σ w·Ix·Iy = 150
k = 0.04

Step 1: det(M) = A×B - C²
        = 220×180 - 150²
        = 39600 - 22500 = 17100

Step 2: trace(M) = A + B
        = 220 + 180 = 400

Step 3: R = det(M) - k×trace(M)²
        = 17100 - 0.04×400²
        = 17100 - 6400 = 10700

Result: R = 10700 > 0 → CORNER ✓"""
    
    ax.text(5, 4.5, example_text, ha='center', va='center', fontsize=9, 
           family='monospace', linespacing=1.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_harris_formulas.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_harris_formulas.png")


def create_orientation_formulas_visual():
    """Create visual diagram with intensity centroid orientation formulas."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    # ============================================
    # Panel 1: Moment Definitions
    # ============================================
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#9b59b6', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Image Moments', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    formula_box = FancyBboxPatch((0.5, 2), 9, 6, boxstyle="round,pad=0.05",
                                  facecolor='#f4ecf7', edgecolor='#9b59b6', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.5, 'For patch around keypoint (xₖ, yₖ):', ha='center', fontsize=11, fontweight='bold')
    
    ax.text(5, 6.3, 'm₀₀ = Σ Σ I(x, y)', ha='center', fontsize=11)
    ax.text(5, 5.6, '(sum of all intensities)', ha='center', fontsize=9, style='italic', color='gray')
    
    ax.text(5, 4.6, 'm₁₀ = Σ Σ (x - xₖ) × I(x, y)', ha='center', fontsize=11)
    ax.text(5, 3.9, '(x-weighted intensity sum)', ha='center', fontsize=9, style='italic', color='gray')
    
    ax.text(5, 2.9, 'm₀₁ = Σ Σ (y - yₖ) × I(x, y)', ha='center', fontsize=11)
    ax.text(5, 2.2, '(y-weighted intensity sum)', ha='center', fontsize=9, style='italic', color='gray')
    
    note_box = FancyBboxPatch((1, 0.3), 8, 1.2, boxstyle="round,pad=0.05",
                               facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(note_box)
    ax.text(5, 0.9, 'Default patch radius: r = 15 pixels (31×31 patch)', 
           ha='center', fontsize=10)
    
    # ============================================
    # Panel 2: Orientation Formula
    # ============================================
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#1abc9c', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Orientation Computation', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    formula_box = FancyBboxPatch((0.5, 3), 9, 5, boxstyle="round,pad=0.05",
                                  facecolor='#e8f8f5', edgecolor='#1abc9c', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.5, 'Centroid Location:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 6.5, 'Cₓ = m₁₀ / m₀₀', ha='center', fontsize=11)
    ax.text(5, 5.8, 'Cy = m₀₁ / m₀₀', ha='center', fontsize=11)
    
    ax.text(5, 4.5, 'Orientation Angle:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 3.5, 'θ = atan2(m₀₁, m₁₀)', ha='center', fontsize=13, fontweight='bold', color='#16a085')
    
    note_box = FancyBboxPatch((1, 0.5), 8, 2, boxstyle="round,pad=0.05",
                               facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(note_box)
    ax.text(5, 1.8, 'θ points from keypoint toward', ha='center', fontsize=10)
    ax.text(5, 1.1, 'intensity centroid (bright region)', ha='center', fontsize=10)
    
    # ============================================
    # Panel 3: Visual Explanation
    # ============================================
    ax = axes[1, 0]
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.set_title('Intensity Centroid Visualization', fontsize=12, fontweight='bold', pad=10)
    
    # Draw patch boundary
    patch = Circle((0, 0), 4, fill=False, edgecolor='black', linewidth=2, linestyle='--')
    ax.add_patch(patch)
    
    # Draw intensity gradient (brighter toward upper-right)
    for i in range(-3, 4):
        for j in range(-3, 4):
            dist = np.sqrt(i**2 + j**2)
            if dist <= 3.5:
                intensity = 0.3 + 0.5 * (i + j + 6) / 12
                rect = Rectangle((i-0.4, j-0.4), 0.8, 0.8, 
                                facecolor=plt.cm.gray(intensity), edgecolor='lightgray', linewidth=0.5)
                ax.add_patch(rect)
    
    # Draw keypoint center
    ax.plot(0, 0, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2, label='Keypoint center')
    
    # Draw centroid (toward bright region)
    cx, cy = 1.5, 1.5
    ax.plot(cx, cy, 'b*', markersize=15, label='Intensity centroid')
    
    # Draw orientation arrow
    ax.annotate('', xy=(cx*1.5, cy*1.5), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax.text(1.2, 2.5, 'θ', fontsize=14, fontweight='bold', color='green')
    
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ============================================
    # Panel 4: Worked Example
    # ============================================
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Example: Orientation Calculation', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    example_box = FancyBboxPatch((0.5, 0.5), 9, 7.5, boxstyle="round,pad=0.05",
                                  facecolor='#fdedec', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(example_box)
    
    example_text = """Given 7×7 patch (simplified):
Brightness increases toward (+x, -y)

Step 1: Compute moments
m₀₀ = Σ I = 3675 (total intensity)
m₁₀ = Σ dx×I = +1500 (more bright on right)
m₀₁ = Σ dy×I = -800 (more bright on top)

Step 2: Compute orientation
θ = atan2(m₀₁, m₁₀)
  = atan2(-800, +1500)
  = -28.07°
  = -0.49 radians

Result: θ points toward upper-right
(matches bright region direction) ✓"""
    
    ax.text(5, 4.5, example_text, ha='center', va='center', fontsize=9, 
           family='monospace', linespacing=1.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_orientation_formulas.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_orientation_formulas.png")


def create_rbrief_formulas_visual():
    """Create visual diagram with rBRIEF descriptor formulas."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    # ============================================
    # Panel 1: Binary Test
    # ============================================
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'BRIEF Binary Test', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    formula_box = FancyBboxPatch((0.5, 3), 9, 5, boxstyle="round,pad=0.05",
                                  facecolor='#ebf5fb', edgecolor='#3498db', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.5, 'For point pair (p, q):', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 6.2, 'τ(p, q) = { 1,  if I(p) < I(q)\n          { 0,  otherwise', 
           ha='center', fontsize=11, family='monospace')
    
    ax.text(5, 4.5, '256 pairs → 256-bit descriptor:', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 3.5, 'D = [τ₀, τ₁, τ₂, ..., τ₂₅₅]', ha='center', fontsize=11)
    
    note_box = FancyBboxPatch((1, 0.5), 8, 2, boxstyle="round,pad=0.05",
                               facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(note_box)
    ax.text(5, 1.8, 'Storage: 256 bits = 32 bytes', ha='center', fontsize=10)
    ax.text(5, 1.1, '(vs SIFT: 128×4 = 512 bytes)', ha='center', fontsize=9, color='gray')
    
    # ============================================
    # Panel 2: Rotation Matrix
    # ============================================
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#9b59b6', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'rBRIEF: Pattern Rotation', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    formula_box = FancyBboxPatch((0.5, 2.5), 9, 5.5, boxstyle="round,pad=0.05",
                                  facecolor='#f4ecf7', edgecolor='#9b59b6', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.5, 'Rotation Matrix:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 6.3, 'R(θ) = [cos(θ)  -sin(θ)]\n       [sin(θ)   cos(θ)]', 
           ha='center', fontsize=11, family='monospace')
    
    ax.text(5, 4.5, 'Point Transformation:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 3.5, "x' = x×cos(θ) - y×sin(θ)\ny' = x×sin(θ) + y×cos(θ)", 
           ha='center', fontsize=10, family='monospace')
    
    note_box = FancyBboxPatch((1, 0.5), 8, 1.5, boxstyle="round,pad=0.05",
                               facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(note_box)
    ax.text(5, 1.25, 'θ = keypoint orientation (from Step 4)', 
           ha='center', fontsize=10)
    
    # ============================================
    # Panel 3: Rotation Example
    # ============================================
    ax = axes[1, 0]
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.set_title('Pattern Rotation Example (θ = 45°)', fontsize=12, fontweight='bold', pad=10)
    
    # Draw keypoint center
    ax.plot(0, 0, 'k*', markersize=15)
    
    # Original pattern (red)
    orig_pairs = [((3, 0), (-2, 1)), ((0, 3), (2, -2)), ((-3, 1), (1, 2))]
    for (px, py), (qx, qy) in orig_pairs:
        ax.plot(px, py, 'ro', markersize=10)
        ax.plot(qx, qy, 'rs', markersize=10)
        ax.plot([px, qx], [py, qy], 'r-', alpha=0.3, linewidth=1)
    
    # Rotated pattern (θ = 45°, blue)
    theta = np.radians(45)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for (px, py), (qx, qy) in orig_pairs:
        rpx = px * cos_t - py * sin_t
        rpy = px * sin_t + py * cos_t
        rqx = qx * cos_t - qy * sin_t
        rqy = qx * sin_t + qy * cos_t
        ax.plot(rpx, rpy, 'bo', markersize=10)
        ax.plot(rqx, rqy, 'bs', markersize=10)
        ax.plot([rpx, rqx], [rpy, rqy], 'b-', alpha=0.3, linewidth=1)
    
    # Legend
    ax.plot([], [], 'ro', markersize=8, label='Original p')
    ax.plot([], [], 'rs', markersize=8, label='Original q')
    ax.plot([], [], 'bo', markersize=8, label='Rotated p\'')
    ax.plot([], [], 'bs', markersize=8, label='Rotated q\'')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    
    # ============================================
    # Panel 4: Hamming Distance
    # ============================================
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 9.1, 'Hamming Distance Matching', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    formula_box = FancyBboxPatch((0.5, 0.5), 9, 7.5, boxstyle="round,pad=0.05",
                                  facecolor='#fdedec', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(5, 7.5, 'H(A, B) = popcount(A XOR B)', ha='center', fontsize=12, fontweight='bold')
    ax.text(5, 6.7, '= number of differing bits', ha='center', fontsize=10, style='italic')
    
    ax.text(5, 5.5, 'Example:', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 4.7, 'A = 1 0 1 1 0 0 1 0 ...\nB = 1 0 0 1 1 0 1 0 ...\n──────────────────\nXOR = 0 0 1 0 1 0 0 0 ...', 
           ha='center', fontsize=9, family='monospace')
    ax.text(5, 2.8, 'H = count of 1s = 2', ha='center', fontsize=10, fontweight='bold')
    
    ax.text(5, 1.5, 'Speed: ~8 CPU cycles (vs ~500 for L2)', ha='center', fontsize=9, color='#27ae60')
    ax.text(5, 0.8, 'Good match: H < 64 (25% threshold)', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_rbrief_formulas.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_rbrief_formulas.png")


def create_full_pipeline_diagram():
    """Create complete ORB pipeline diagram."""
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(10, 13.5, 'ORB Algorithm - Complete Pipeline', ha='center', fontsize=20, fontweight='bold')
    
    # Detection Phase Box
    det_box = FancyBboxPatch((0.5, 6.5), 9, 6.5, boxstyle="round,pad=0.1",
                              facecolor='#e8f8f5', edgecolor='#1abc9c', linewidth=3)
    ax.add_patch(det_box)
    ax.text(5, 12.5, '1️⃣ DETECTION PHASE', ha='center', fontsize=14, fontweight='bold', color='#1abc9c')
    
    # Detection steps
    det_steps = [
        ('Step 1', 'Scale Pyramid', 'scale_n = 1/1.2^n'),
        ('Step 2', 'FAST Corners', '9+ contiguous B or D'),
        ('Step 3', 'Harris Response', 'R = det(M) - k·tr(M)²'),
        ('Step 4', 'Orientation', 'θ = atan2(m₀₁, m₁₀)'),
    ]
    
    y_pos = 11.5
    for step, name, formula in det_steps:
        ax.text(1.5, y_pos, step, fontsize=10, fontweight='bold', color='#117a65')
        ax.text(3.5, y_pos, name, fontsize=10)
        ax.text(6, y_pos, formula, fontsize=9, style='italic', color='#566573')
        y_pos -= 1.1
    
    # Arrow between phases
    ax.annotate('', xy=(14.5, 9.5), xytext=(9.7, 9.5),
               arrowprops=dict(arrowstyle='->', color='black', lw=3))
    
    # Description Phase Box
    desc_box = FancyBboxPatch((10.5, 6.5), 9, 6.5, boxstyle="round,pad=0.1",
                               facecolor='#f4ecf7', edgecolor='#9b59b6', linewidth=3)
    ax.add_patch(desc_box)
    ax.text(15, 12.5, '2️⃣ DESCRIPTION PHASE', ha='center', fontsize=14, fontweight='bold', color='#9b59b6')
    
    # Description steps
    desc_steps = [
        ('Step 5', 'rBRIEF Descriptor', '256 binary tests'),
        ('Step 6', 'Matching', 'H = popcount(A XOR B)'),
    ]
    
    y_pos = 11
    for step, name, formula in desc_steps:
        ax.text(11.5, y_pos, step, fontsize=10, fontweight='bold', color='#6c3483')
        ax.text(13.5, y_pos, name, fontsize=10)
        ax.text(16.5, y_pos, formula, fontsize=9, style='italic', color='#566573')
        y_pos -= 1.1
    
    # Input/Output boxes
    input_box = FancyBboxPatch((0.5, 4.5), 6, 1.5, boxstyle="round,pad=0.05",
                                facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(input_box)
    ax.text(3.5, 5.25, 'INPUT: Image (H × W)', ha='center', fontsize=11, fontweight='bold')
    
    output_box = FancyBboxPatch((13.5, 4.5), 6, 1.5, boxstyle="round,pad=0.05",
                                 facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(output_box)
    ax.text(16.5, 5.25, 'OUTPUT: Keypoints + 256-bit', ha='center', fontsize=11, fontweight='bold')
    
    # Key formulas box
    formula_box = FancyBboxPatch((0.5, 0.5), 19, 3.5, boxstyle="round,pad=0.1",
                                  facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(formula_box)
    ax.text(10, 3.5, 'Key Formulas', ha='center', fontsize=12, fontweight='bold', color='#d68910')
    
    formulas = [
        'Scale: scale_n = 1/f^n (f=1.2)',
        'FAST: I_c > I_p + t (brighter) or I_c < I_p - t (darker)',
        'Harris: R = det(M) - k×trace(M)², where M = [Σw·Ix², Σw·Ix·Iy; Σw·Ix·Iy, Σw·Iy²]',
        "Orientation: θ = atan2(m₀₁, m₁₀), where m₁₀ = Σ dx×I, m₀₁ = Σ dy×I",
        "rBRIEF: bit_i = (I(p'_i) < I(q'_i)) ? 1 : 0, where (p', q') = R(θ)×(p, q)",
        'Hamming: H(A, B) = popcount(A XOR B)'
    ]
    
    y_pos = 2.8
    for formula in formulas:
        ax.text(1, y_pos, '•  ' + formula, fontsize=9, family='monospace')
        y_pos -= 0.4
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_pipeline_diagram.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_pipeline_diagram.png")


def main():
    """Generate all math formula visualizations."""
    print("=" * 60)
    print("ORB Math Formula Visualizations")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("\n1. Generating FAST formulas diagram...")
    create_fast_formulas_visual()
    
    print("\n2. Generating Harris formulas diagram...")
    create_harris_formulas_visual()
    
    print("\n3. Generating orientation formulas diagram...")
    create_orientation_formulas_visual()
    
    print("\n4. Generating rBRIEF formulas diagram...")
    create_rbrief_formulas_visual()
    
    print("\n5. Generating pipeline diagram...")
    create_full_pipeline_diagram()
    
    print("\n" + "=" * 60)
    print("Done! Generated all math formula visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()

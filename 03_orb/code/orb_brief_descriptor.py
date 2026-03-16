"""
rBRIEF Descriptor - Detailed Implementation
Rotated Binary Robust Independent Elementary Features
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Arrow, FancyArrowPatch
from PIL import Image

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, '..', 'images')


def generate_brief_pattern(n_pairs=256, patch_size=31, seed=42):
    """
    Generate BRIEF sampling pattern.
    """
    np.random.seed(seed)
    half = patch_size // 2
    pattern = []
    
    for _ in range(n_pairs):
        p_x = int(np.clip(np.random.randn() * half / 2, -half, half))
        p_y = int(np.clip(np.random.randn() * half / 2, -half, half))
        q_x = int(np.clip(np.random.randn() * half / 2, -half, half))
        q_y = int(np.clip(np.random.randn() * half / 2, -half, half))
        pattern.append(((p_x, p_y), (q_x, q_y)))
    
    return pattern


def rotate_point(x, y, theta):
    """Rotate point by angle theta."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return x * cos_t - y * sin_t, x * sin_t + y * cos_t


def load_image_patch():
    """Load a patch from the real input image."""
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    if os.path.exists(image_path):
        img_rgb = np.array(Image.open(image_path))
        if len(img_rgb.shape) == 3:
            img = (0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]) / 255.0
        else:
            img = img_rgb / 255.0
        # Extract a 31x31 patch from an interesting region
        cy, cx = img.shape[0] // 3, img.shape[1] // 3
        half = 15
        if cy - half >= 0 and cy + half + 1 <= img.shape[0] and cx - half >= 0 and cx + half + 1 <= img.shape[1]:
            return img[cy-half:cy+half+1, cx-half:cx+half+1]
    return None

def visualize_brief_concept():
    """
    Visualize the basic BRIEF concept: binary comparison of point pairs.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Try to load real patch, fallback to synthetic
    patch = load_image_patch()
    if patch is None:
        np.random.seed(42)
        patch = np.random.rand(31, 31) * 0.5 + 0.25
        patch[10:20, 10:20] = 0.8  # Brighter region
    patch[5:12, 18:25] = 0.2   # Darker region
    
    # Left: Show patch with sample points
    ax1 = axes[0]
    ax1.imshow(patch, cmap='gray', vmin=0, vmax=1)
    
    # Sample pairs (show 5)
    pairs = [
        ((5, 8), (20, 8)),
        ((10, 15), (25, 20)),
        ((15, 5), (15, 25)),
        ((8, 22), (22, 10)),
        ((12, 12), (18, 18)),
    ]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, ((px, py), (qx, qy)) in enumerate(pairs):
        ax1.plot(px, py, 'o', color=colors[i], markersize=10, markeredgecolor='white', markeredgewidth=1)
        ax1.plot(qx, qy, 's', color=colors[i], markersize=10, markeredgecolor='white', markeredgewidth=1)
        ax1.annotate('', xy=(qx, qy), xytext=(px, py),
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
        ax1.text(px-2, py-2, f'p{i+1}', fontsize=8, color=colors[i])
        ax1.text(qx+1, qy+1, f'q{i+1}', fontsize=8, color=colors[i])
    
    ax1.set_title('BRIEF: Sample Point Pairs\n○=p points, □=q points', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Middle: Binary test visualization
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#3498db', edgecolor='black', linewidth=2)
    ax2.add_patch(title_box)
    ax2.text(5, 9.1, 'Binary Test for Each Pair', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    
    y_pos = 7.5
    for i, ((px, py), (qx, qy)) in enumerate(pairs):
        p_val = patch[py, px]
        q_val = patch[qy, qx]
        bit = 1 if p_val < q_val else 0
        
        ax2.text(1, y_pos, f'Pair {i+1}:', fontsize=11, fontweight='bold', color=colors[i])
        ax2.text(3, y_pos, f'I(p)={p_val:.2f}  {"<" if bit else "≥"}  I(q)={q_val:.2f}', fontsize=10)
        ax2.text(8, y_pos, f'→ bit = {bit}', fontsize=11, fontweight='bold')
        y_pos -= 1.2
    
    formula_box = FancyBboxPatch((0.5, 0.5), 9, 1.5, boxstyle="round,pad=0.05",
                                  facecolor='#e8f8f5', edgecolor='#1abc9c', linewidth=2)
    ax2.add_patch(formula_box)
    ax2.text(5, 1.25, 'bit_i = 1 if I(p_i) < I(q_i) else 0', ha='center', fontsize=12, 
            family='monospace', fontweight='bold')
    
    # Right: Descriptor assembly
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#27ae60', edgecolor='black', linewidth=2)
    ax3.add_patch(title_box)
    ax3.text(5, 9.1, '256-bit Binary Descriptor', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    
    descriptor_text = """
BRIEF Descriptor = [b₀, b₁, b₂, ..., b₂₅₅]

For 256 pre-defined point pairs:
  Compare intensities at each pair
  → 256 binary comparisons
  → 256-bit descriptor

Example (first 16 bits):
[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1]

Storage: 256 bits = 32 bytes
(vs SIFT: 128 floats = 512 bytes)
"""
    ax3.text(0.5, 4, descriptor_text, fontsize=10, va='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_brief_concept.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_brief_concept.png")


def visualize_rbrief_rotation():
    """
    Visualize how rBRIEF rotates the sampling pattern.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Generate sample pattern (20 pairs for visualization)
    np.random.seed(42)
    pattern = []
    for _ in range(20):
        px = np.random.randint(-10, 11)
        py = np.random.randint(-10, 11)
        qx = np.random.randint(-10, 11)
        qy = np.random.randint(-10, 11)
        pattern.append(((px, py), (qx, qy)))
    
    # Original pattern (θ = 0)
    ax1 = axes[0]
    ax1.set_xlim(-15, 15)
    ax1.set_ylim(-15, 15)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.axvline(x=0, color='black', linewidth=0.5)
    
    for (px, py), (qx, qy) in pattern:
        ax1.plot(px, py, 'ro', markersize=6)
        ax1.plot(qx, qy, 'bs', markersize=6)
        ax1.plot([px, qx], [py, qy], 'g-', alpha=0.3, linewidth=1)
    
    ax1.plot(0, 0, 'k*', markersize=15)
    ax1.set_title('Original Pattern (θ = 0°)\n★ = keypoint center', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Rotated pattern (θ = 45°)
    ax2 = axes[1]
    ax2.set_xlim(-15, 15)
    ax2.set_ylim(-15, 15)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axvline(x=0, color='black', linewidth=0.5)
    
    theta = np.radians(45)
    for (px, py), (qx, qy) in pattern:
        rpx, rpy = rotate_point(px, py, theta)
        rqx, rqy = rotate_point(qx, qy, theta)
        ax2.plot(rpx, rpy, 'ro', markersize=6)
        ax2.plot(rqx, rqy, 'bs', markersize=6)
        ax2.plot([rpx, rqx], [rpy, rqy], 'g-', alpha=0.3, linewidth=1)
    
    ax2.plot(0, 0, 'k*', markersize=15)
    # Draw rotation arrow
    ax2.annotate('', xy=(10, 10), xytext=(14, 0),
                arrowprops=dict(arrowstyle='->', color='orange', lw=3, 
                               connectionstyle='arc3,rad=0.3'))
    ax2.text(12, 6, 'θ=45°', fontsize=12, color='orange', fontweight='bold')
    ax2.set_title('Rotated Pattern (θ = 45°)\n★ = keypoint center', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # Rotated pattern (θ = 135°)
    ax3 = axes[2]
    ax3.set_xlim(-15, 15)
    ax3.set_ylim(-15, 15)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.axvline(x=0, color='black', linewidth=0.5)
    
    theta = np.radians(135)
    for (px, py), (qx, qy) in pattern:
        rpx, rpy = rotate_point(px, py, theta)
        rqx, rqy = rotate_point(qx, qy, theta)
        ax3.plot(rpx, rpy, 'ro', markersize=6)
        ax3.plot(rqx, rqy, 'bs', markersize=6)
        ax3.plot([rpx, rqx], [rpy, rqy], 'g-', alpha=0.3, linewidth=1)
    
    ax3.plot(0, 0, 'k*', markersize=15)
    ax3.annotate('', xy=(-10, 10), xytext=(0, 14),
                arrowprops=dict(arrowstyle='->', color='orange', lw=3,
                               connectionstyle='arc3,rad=-0.3'))
    ax3.text(-5, 12, 'θ=135°', fontsize=12, color='orange', fontweight='bold')
    ax3.set_title('Rotated Pattern (θ = 135°)\n★ = keypoint center', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    
    plt.suptitle('rBRIEF: Pattern Rotation for Rotation Invariance\n'
                 '○=p points (red), □=q points (blue)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_rbrief_rotation.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_rbrief_rotation.png")


def visualize_rotation_formula():
    """
    Visualize the rotation formula used in rBRIEF.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    title_box = FancyBboxPatch((0.5, 8.5), 13, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#9b59b6', edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(7, 9.1, 'rBRIEF Rotation Formula', ha='center', va='center',
           fontsize=14, fontweight='bold', color='white')
    
    # Rotation matrix
    formula_box = FancyBboxPatch((0.5, 5), 6, 3, boxstyle="round,pad=0.05",
                                  facecolor='#f4ecf7', edgecolor='#9b59b6', linewidth=2)
    ax.add_patch(formula_box)
    
    ax.text(3.5, 7.5, 'Rotation Matrix:', ha='center', fontsize=12, fontweight='bold')
    ax.text(3.5, 6.7, r'$R(\theta) = [ [\cos\theta, -\sin\theta],$', ha='center', fontsize=12)
    ax.text(3.5, 6.2, r'$\quad\quad\quad\quad [\sin\theta, \cos\theta] ]$', ha='center', fontsize=12)
    ax.text(3.5, 5.5, r'where $\theta$ = keypoint orientation', ha='center', fontsize=10, style='italic')
    
    # Point transformation
    transform_box = FancyBboxPatch((7.5, 5), 6, 3, boxstyle="round,pad=0.05",
                                    facecolor='#e8f6f3', edgecolor='#1abc9c', linewidth=2)
    ax.add_patch(transform_box)
    
    ax.text(10.5, 7.5, 'Point Transformation:', ha='center', fontsize=12, fontweight='bold')
    ax.text(10.5, 6.5, r"$[x', y']^T = R(\theta) \cdot [x, y]^T$",
           ha='center', fontsize=13)
    ax.text(10.5, 5.5, 'Original (x,y) → Rotated (x\',y\')', ha='center', fontsize=10, style='italic')
    
    # Expanded formula
    expanded_box = FancyBboxPatch((0.5, 1.5), 13, 3, boxstyle="round,pad=0.05",
                                   facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(expanded_box)
    
    ax.text(7, 4, 'Expanded (what the code computes):', ha='center', fontsize=12, fontweight='bold')
    ax.text(7, 3, r"$x' = x \cdot \cos(\theta) - y \cdot \sin(\theta)$", ha='center', fontsize=13)
    ax.text(7, 2.2, r"$y' = x \cdot \sin(\theta) + y \cdot \cos(\theta)$", ha='center', fontsize=13)
    
    # Example
    ax.text(7, 0.8, 'Example: θ=45°, (x,y)=(3,0) → (x\',y\')=(2.12, 2.12)', 
           ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_rotation_formula.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_rotation_formula.png")


def visualize_hamming_distance():
    """
    Visualize Hamming distance matching.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Hamming distance concept
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Title
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax1.add_patch(title_box)
    ax1.text(5, 9.1, 'Hamming Distance', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    
    # Example
    ax1.text(5, 7.5, 'Descriptor A: 1 0 1 1 0 0 1 0 ...', ha='center', fontsize=12, family='monospace')
    ax1.text(5, 6.8, 'Descriptor B: 1 0 0 1 1 0 1 0 ...', ha='center', fontsize=12, family='monospace')
    ax1.text(5, 6.0, '──────────────────────────', ha='center', fontsize=12, family='monospace')
    ax1.text(5, 5.3, 'XOR result:   0 0 1 0 1 0 0 0 ...', ha='center', fontsize=12, family='monospace',
            color='red')
    
    formula_box = FancyBboxPatch((1, 3), 8, 1.8, boxstyle="round,pad=0.05",
                                  facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=2)
    ax1.add_patch(formula_box)
    ax1.text(5, 4.2, 'H(A, B) = popcount(A XOR B)', ha='center', fontsize=13, fontweight='bold')
    ax1.text(5, 3.4, '= number of differing bits', ha='center', fontsize=11)
    
    ax1.text(5, 1.5, 'In example: H = 2 (positions 3 and 5 differ)', ha='center', fontsize=11, 
            style='italic', color='gray')
    
    # Right: Why Hamming is fast
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#27ae60', edgecolor='black', linewidth=2)
    ax2.add_patch(title_box)
    ax2.text(5, 9.1, 'Why Hamming is Fast', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    
    content = """
Hardware Optimization:

1. XOR operation:
   • Single CPU instruction
   • Operates on 64 bits at once

2. POPCNT instruction:
   • Counts 1-bits in register
   • Single CPU cycle

256-bit descriptor comparison:
  • 4 × 64-bit XOR operations
  • 4 × POPCNT operations
  • Total: ~8 CPU cycles


vs SIFT L2 distance:
  • 128 subtractions
  • 128 multiplications  
  • 127 additions
  • 1 square root
  • Total: ~500+ CPU cycles


ORB is ~60× faster to match!
"""
    ax2.text(0.5, 4, content, fontsize=10, va='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_hamming_distance.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_hamming_distance.png")


def visualize_descriptor_comparison():
    """
    Compare binary (ORB) vs floating point (SIFT) descriptors.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Generate sample descriptors
    np.random.seed(42)
    orb_desc = np.random.randint(0, 2, 256)
    sift_desc = np.random.randn(128)
    sift_desc = sift_desc / np.linalg.norm(sift_desc)  # Normalize
    
    # ORB descriptor visualization
    ax1 = axes[0, 0]
    colors = ['#27ae60' if b == 1 else '#2c3e50' for b in orb_desc]
    ax1.bar(range(256), orb_desc, color=colors, width=1)
    ax1.set_xlim(-1, 256)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_title('ORB: 256-bit Binary Descriptor', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Bit position')
    ax1.set_ylabel('Bit value (0 or 1)')
    
    # SIFT descriptor visualization
    ax2 = axes[0, 1]
    ax2.bar(range(128), sift_desc, color='#3498db', width=1)
    ax2.set_xlim(-1, 128)
    ax2.set_title('SIFT: 128-D Float Descriptor', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Value (normalized)')
    
    # ORB as image (16×16)
    ax3 = axes[1, 0]
    orb_img = orb_desc.reshape(16, 16)
    ax3.imshow(orb_img, cmap='Greens', aspect='auto')
    ax3.set_title('ORB as 16×16 Binary Image\n(Green=1, Dark=0)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Comparison table
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#9b59b6', edgecolor='black', linewidth=2)
    ax4.add_patch(title_box)
    ax4.text(5, 9.1, 'ORB vs SIFT Comparison', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    
    comparison = """
Property          ORB              SIFT
──────────────────────────────────────────
Descriptor size   256 bits         128 floats
Storage           32 bytes         512 bytes
Distance metric   Hamming          L2 (Euclidean)
Computation       XOR + popcount   128 multiply-adds
Match time        ~8 cycles        ~500 cycles
Rotation inv.     rBRIEF           Gradient histogram
Scale inv.        Image pyramid    DoG pyramid
"""
    ax4.text(0.5, 4, comparison, fontsize=10, va='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_sift_comparison.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_sift_comparison.png")


def main():
    """Generate all BRIEF descriptor visualizations."""
    print("=" * 60)
    print("rBRIEF Descriptor Visualizations")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("\n1. Generating BRIEF concept diagram...")
    visualize_brief_concept()
    
    print("\n2. Generating rBRIEF rotation visualization...")
    visualize_rbrief_rotation()
    
    print("\n3. Generating rotation formula diagram...")
    visualize_rotation_formula()
    
    print("\n4. Generating Hamming distance visualization...")
    visualize_hamming_distance()
    
    print("\n5. Generating ORB vs SIFT comparison...")
    visualize_descriptor_comparison()
    
    print("\n" + "=" * 60)
    print("Done! Generated all rBRIEF descriptor visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()

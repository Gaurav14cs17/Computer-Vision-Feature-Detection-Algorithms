"""
SURF Final Summary Visualization
Complete pipeline overview with all steps
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


def load_image():
    """Load the input image."""
    img_path = os.path.join(IMAGES_DIR, 'input_image.jpg')
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('L')
        if img.size[0] > 800 or img.size[1] > 600:
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
        return np.array(img)
    else:
        img = np.zeros((480, 640), dtype=np.uint8)
        img[100:200, 100:250] = 180
        img[250:350, 300:450] = 220
        Image.fromarray(img).save(img_path)
        return img


def create_final_summary():
    """Create comprehensive SURF summary visualization"""
    
    img = load_image()
    H, W = img.shape
    
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.25)
    
    # 1. Input Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f'Step 0: Input\n{W}×{H}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Integral Image
    ax2 = fig.add_subplot(gs[0, 1])
    integral = np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)
    ii_norm = (integral - integral.min()) / (integral.max() - integral.min())
    ax2.imshow(ii_norm, cmap='viridis')
    ax2.set_title('Step 1: Integral Image\nII(x,y) = Σ I(i,j)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Hessian Response
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)
    ax3.axis('off')
    
    # Draw box filters
    rect1 = patches.Rectangle((20, 55), 20, 30, facecolor='green', edgecolor='black', alpha=0.7)
    rect2 = patches.Rectangle((40, 55), 20, 30, facecolor='red', edgecolor='black', alpha=0.7)
    rect3 = patches.Rectangle((60, 55), 20, 30, facecolor='green', edgecolor='black', alpha=0.7)
    ax3.add_patch(rect1)
    ax3.add_patch(rect2)
    ax3.add_patch(rect3)
    ax3.text(30, 70, '+1', fontsize=10, ha='center', va='center', fontweight='bold', color='white')
    ax3.text(50, 70, '-2', fontsize=10, ha='center', va='center', fontweight='bold', color='white')
    ax3.text(70, 70, '+1', fontsize=10, ha='center', va='center', fontweight='bold', color='white')
    ax3.text(50, 40, 'det(H) = Dxx·Dyy - (0.9·Dxy)²', fontsize=10, ha='center')
    ax3.set_title('Step 2: Hessian\nBox Filters', fontsize=12, fontweight='bold')
    
    # 4. Scale-Space
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 100)
    ax4.axis('off')
    
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']
    filter_sizes = [9, 15, 21, 27]
    for i, (fs, color) in enumerate(zip(filter_sizes, colors)):
        width = 15 + i * 8
        rect = patches.FancyBboxPatch((50 - width/2, 75 - i*18), width, 14,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax4.add_patch(rect)
        ax4.text(50, 82 - i*18, f'{fs}×{fs}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax4.text(50, 15, '26-neighbor extrema', fontsize=10, ha='center')
    ax4.set_title('Step 3: Scale-Space\nFilter Pyramid', fontsize=12, fontweight='bold')
    
    # 5. Keypoint Detection
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(img, cmap='gray')
    np.random.seed(42)
    colors_kp = ['red', 'lime', 'cyan']
    sizes = [4, 7, 11]
    for scale, (color, size) in enumerate(zip(colors_kp, sizes)):
        n_kps = [60, 40, 20][scale]
        margin = 20 + scale * 10
        for _ in range(n_kps):
            x = np.random.randint(margin, W - margin)
            y = np.random.randint(margin, H - margin)
            circle = plt.Circle((x, y), size, color=color, fill=False, linewidth=1.2)
            ax5.add_patch(circle)
    ax5.set_title('Step 3: Detected\nKeypoints', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 6. Filtering
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(img, cmap='gray')
    np.random.seed(123)
    for scale, (color, size) in enumerate(zip(colors_kp, sizes)):
        n_kps = [45, 30, 15][scale]
        margin = 25 + scale * 10
        for _ in range(n_kps):
            x = np.random.randint(margin, W - margin)
            y = np.random.randint(margin, H - margin)
            circle = plt.Circle((x, y), size, color=color, fill=False, linewidth=1.5)
            ax6.add_patch(circle)
    ax6.set_title('Step 4: Filtered\n(Sub-pixel refined)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # 7. Orientation
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(img, cmap='gray')
    np.random.seed(456)
    for _ in range(35):
        x = np.random.randint(50, W - 50)
        y = np.random.randint(50, H - 50)
        ori = np.random.uniform(0, 2 * np.pi)
        size = np.random.choice([5, 8, 11])
        circle = plt.Circle((x, y), size, color='red', fill=False, linewidth=1.5)
        ax7.add_patch(circle)
        dx = size * 1.5 * np.cos(ori)
        dy = size * 1.5 * np.sin(ori)
        ax7.arrow(x, y, dx, dy, head_width=4, head_length=3, fc='yellow', ec='yellow')
    ax7.set_title('Step 5: Orientation\n(Haar wavelets)', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    # 8. Descriptor
    ax8 = fig.add_subplot(gs[1, 3])
    np.random.seed(789)
    desc = np.random.randn(64)
    desc = desc / np.linalg.norm(desc)
    bar_colors = plt.cm.tab20(np.repeat(np.arange(16), 4) / 16)
    ax8.bar(range(64), desc, color=bar_colors, edgecolor='none', width=1)
    for i in range(1, 16):
        ax8.axvline(x=i*4 - 0.5, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax8.set_xlim(-1, 64)
    ax8.set_xlabel('Index (0-63)')
    ax8.set_ylabel('Value')
    ax8.set_title('Step 6: 64-D Descriptor\n[Σdx,Σdy,Σ|dx|,Σ|dy|]×16', fontsize=12, fontweight='bold')
    
    # 9. Pipeline Flow (bottom row)
    ax9 = fig.add_subplot(gs[2, :])
    ax9.axis('off')
    
    pipeline = """
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                           SURF COMPLETE PIPELINE SUMMARY                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                                    ║
║   INPUT: Image (H × W)                                                                                                            ║
║          │                                                                                                                        ║
║          ▼                                                                                                                        ║
║   ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    ║
║   │  1️⃣ DETECTION PHASE                                                                                                       │    ║
║   │     Step 1: Integral Image ──────► II(x,y) = Σ I(i,j)  [O(1) box sums enabled!]                                          │    ║
║   │     Step 2: Hessian Box Filters ─► det(H) = Dxx·Dyy - (0.9·Dxy)²  [Blob detection]                                       │    ║
║   │     Step 3: Scale-Space Extrema ─► 26-neighbor comparison at multiple filter sizes (9,15,21,27...)                       │    ║
║   │     Step 4: Keypoint Refinement ─► Sub-pixel localization, threshold filtering                                            │    ║
║   └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    ║
║          │                                                                                                                        ║
║          ▼                                                                                                                        ║
║   ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    ║
║   │  2️⃣ DESCRIPTION PHASE                                                                                                     │    ║
║   │     Step 5: Orientation ─────────► Haar wavelets in circular region, 60° sliding window                                   │    ║
║   │     Step 6: Descriptor (64-D) ───► 20s×20s region, 4×4 subregions, [Σdx', Σdy', Σ|dx'|, Σ|dy'|]×16                       │    ║
║   └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    ║
║          │                                                                                                                        ║
║          ▼                                                                                                                        ║
║   OUTPUT: N keypoints with (x, y, scale, orientation, 64-D descriptor)                                                            ║
║                                                                                                                                    ║
║   KEY INNOVATIONS: ① Integral Images → O(1) computation  ② Box Filters → Fast approximation  ③ Filter Pyramid → No resizing      ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
    ax9.text(0.02, 0.5, pipeline, fontsize=8, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))
    
    plt.suptitle('SURF (Speeded-Up Robust Features) - Complete Algorithm Overview', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_final_summary.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: surf_final_summary.png")


def create_quick_reference():
    """Create a quick reference card"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    
    reference = """
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                              SURF ALGORITHM - QUICK REFERENCE                             ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  PROPERTY           │ VALUE                                                              ║
║  ───────────────────┼────────────────────────────────────────────────────────────────────║
║  Full Name          │ Speeded-Up Robust Features                                         ║
║  Year               │ 2006 (Bay, Tuytelaars, Van Gool)                                   ║
║  Speed              │ ~3× faster than SIFT                                               ║
║  Descriptor Size    │ 64-D (or 128-D extended variant)                                   ║
║                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                    KEY FORMULAS                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  STEP 1 - INTEGRAL IMAGE:                                                                ║
║    II(x,y) = Σᵢ₌₀ˣ Σⱼ₌₀ʸ I(i,j)                                                         ║
║    Box_Sum = II(D) - II(B) - II(C) + II(A)   [O(1) for ANY rectangle]                   ║
║                                                                                          ║
║  STEP 2 - HESSIAN MATRIX:                                                                ║
║    H = | Dxx  Dxy |                                                                      ║
║        | Dxy  Dyy |                                                                      ║
║    det(H) = Dxx · Dyy - (0.9 · Dxy)²                                                    ║
║                                                                                          ║
║  STEP 3 - SCALE-SPACE:                                                                   ║
║    Filter sizes: 9×9, 15×15, 21×21, 27×27, ... (+6 each step)                           ║
║    σ ≈ 1.2 × (filter_size / 9)                                                          ║
║    Keypoint if: val > ALL 26 neighbors OR val < ALL 26 neighbors                        ║
║                                                                                          ║
║  STEP 4 - REFINEMENT:                                                                    ║
║    offset = -H⁻¹ × ∇H                                                                    ║
║    REJECT if: |offset| > 0.5                                                             ║
║                                                                                          ║
║  STEP 5 - ORIENTATION:                                                                   ║
║    Haar X: dx = I(right) - I(left)                                                      ║
║    Haar Y: dy = I(bottom) - I(top)                                                      ║
║    60° sliding window → dominant direction                                               ║
║                                                                                          ║
║  STEP 6 - DESCRIPTOR:                                                                    ║
║    20s × 20s region → 4×4 = 16 subregions                                               ║
║    Each subregion: [Σdx', Σdy', Σ|dx'|, Σ|dy'|]                                         ║
║    Total: 16 × 4 = 64 values (normalized)                                                ║
║                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                  SURF vs SIFT                                            ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  Feature          │ SIFT                    │ SURF                                       ║
║  ─────────────────┼─────────────────────────┼───────────────────────────────────────────║
║  Scale-Space      │ Image pyramid           │ Filter pyramid                             ║
║  Blob Detector    │ DoG                     │ Hessian det                                ║
║  Filter Type      │ Gaussian convolution    │ Box filters (O(1))                         ║
║  Descriptor       │ 128-D                   │ 64-D                                       ║
║  Orientation      │ Gradient histogram      │ Haar wavelet                               ║
║  Speed            │ Baseline                │ ~3× faster                                 ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""
    ax.text(0.02, 0.5, reference, fontsize=9, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))
    ax.set_title('SURF Quick Reference Card', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_quick_reference.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: surf_quick_reference.png")


if __name__ == "__main__":
    create_final_summary()
    create_quick_reference()
    print("\nFinal summary images generated!")

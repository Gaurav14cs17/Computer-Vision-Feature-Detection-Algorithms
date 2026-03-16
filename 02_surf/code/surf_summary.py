"""
SURF Summary and Comparison Visualizations
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
    input_path = os.path.join(IMAGES_DIR, 'input_image.jpg')
    if os.path.exists(input_path):
        img = Image.open(input_path).convert('L')
        if img.size[0] > 800 or img.size[1] > 600:
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
        return np.array(img)
    else:
        img = np.zeros((480, 640), dtype=np.uint8)
        img[100:200, 100:250] = 180
        img[250:350, 300:450] = 220
        Image.fromarray(img).save(input_path)
        return img


def visualize_complete_pipeline():
    """Complete pipeline overview"""
    print("=" * 60)
    print("SURF Complete Pipeline Summary")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.axis('off')
    
    pipeline = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         SURF COMPLETE PIPELINE                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

INPUT: Image (H × W)
        │
        ▼
═══════════════════════════════════════════════════════════════════════════════
                            1️⃣ DETECTION PHASE
═══════════════════════════════════════════════════════════════════════════════
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: INTEGRAL IMAGE                                                        │
│         II(x,y) = Σᵢ₌₀ˣ Σⱼ₌₀ʸ I(i,j)                                         │
│         └── One-time O(H×W) computation                                       │
│         └── Enables O(1) box sums for ALL subsequent operations               │
└───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: HESSIAN MATRIX (Box Filters)                                          │
│         H = | Dxx  Dxy |      det(H) = Dxx·Dyy - (0.9·Dxy)²                   │
│             | Dxy  Dyy |                                                      │
│         └── Box filter approximation of Gaussian 2nd derivatives              │
│         └── Applied at multiple filter sizes: 9×9, 15×15, 21×21, 27×27        │
│         └── O(1) per pixel using integral image!                              │
└───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: SCALE-SPACE EXTREMA                                                   │
│         For each point (x, y, scale):                                         │
│           Compare with 26 neighbors (8 same + 9 above + 9 below scale)        │
│           If val > ALL neighbors OR val < ALL neighbors → KEYPOINT            │
│         └── Same principle as SIFT, but using filter pyramid                  │
└───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: KEYPOINT LOCALIZATION                                                 │
│         Sub-pixel refinement: x̂ = x - H⁻¹·∇H (Taylor expansion)              │
│         Filtering: Remove weak responses |det(H)| < threshold                 │
│         └── Reject if offset > 0.5 (unstable)                                 │
└───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
OUTPUT: N stable keypoints with (x, y, scale)

═══════════════════════════════════════════════════════════════════════════════
                           2️⃣ DESCRIPTION PHASE
═══════════════════════════════════════════════════════════════════════════════
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: ORIENTATION ASSIGNMENT                                                │
│         Haar wavelet responses (dx, dy) in circular region (radius = 6s)      │
│         60° sliding window → find dominant gradient direction                 │
│         θ_dominant = atan2(Σdy_max, Σdx_max)                                  │
│         └── Achieves rotation invariance                                      │
└───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: DESCRIPTOR EXTRACTION (64-D)                                          │
│         Define 20s × 20s region, rotated by θ_dominant                        │
│         Divide into 4×4 = 16 subregions (each 5s × 5s)                        │
│         For each subregion: [Σdx, Σdy, Σ|dx|, Σ|dy|]                          │
│         Total: 16 × 4 = 64 values                                             │
│         Normalize: descriptor = descriptor / ||descriptor||                   │
└───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
OUTPUT: N keypoints with (x, y, scale, orientation, 64-D descriptor)
"""
    ax.text(0.01, 0.5, pipeline, fontsize=9, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))
    ax.set_title('SURF Complete Pipeline Flow', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_pipeline_flow.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_pipeline_flow.png")


def visualize_surf_vs_sift():
    """SURF vs SIFT comparison"""
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis('off')
    
    comparison = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           SURF vs SIFT COMPARISON                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Feature              │  SIFT (2004)               │  SURF (2006)            ║
║  ─────────────────────┼────────────────────────────┼─────────────────────────║
║  Full Name            │  Scale-Invariant Feature   │  Speeded-Up Robust      ║
║                       │  Transform                 │  Features               ║
║  ─────────────────────┼────────────────────────────┼─────────────────────────║
║  Speed                │  Slower                    │  ~3× FASTER             ║
║  ─────────────────────┼────────────────────────────┼─────────────────────────║
║  Scale-Space          │  IMAGE pyramid             │  FILTER pyramid         ║
║  Strategy             │  (resize image)            │  (resize filter)        ║
║  ─────────────────────┼────────────────────────────┼─────────────────────────║
║  Blob Detector        │  Difference of Gaussian    │  Hessian determinant    ║
║                       │  (DoG ≈ σ²∇²G)             │  det(H) = Dxx·Dyy-Dxy²  ║
║  ─────────────────────┼────────────────────────────┼─────────────────────────║
║  Filter Type          │  Gaussian convolution      │  BOX FILTERS            ║
║                       │  O(filter_size²) per pixel │  O(1) per pixel!        ║
║  ─────────────────────┼────────────────────────────┼─────────────────────────║
║  Fast Computation     │  No (direct convolution)   │  Yes (integral image)   ║
║  ─────────────────────┼────────────────────────────┼─────────────────────────║
║  Descriptor Size      │  128-D                     │  64-D (or 128-D)        ║
║  ─────────────────────┼────────────────────────────┼─────────────────────────║
║  Orientation          │  36-bin gradient histogram │  Haar wavelet +         ║
║  Assignment           │                            │  60° sliding window     ║
║  ─────────────────────┼────────────────────────────┼─────────────────────────║
║  Descriptor Method    │  4×4 subregions            │  4×4 subregions         ║
║                       │  8-bin gradient histograms │  Haar wavelet sums      ║
║                       │  16 × 8 = 128 values       │  16 × 4 = 64 values     ║
║  ─────────────────────┼────────────────────────────┼─────────────────────────║
║  Patent Status        │  Expired 2020              │  Expired 2020           ║
║  ─────────────────────┼────────────────────────────┼─────────────────────────║
║  Best For             │  Accuracy-critical         │  Real-time applications ║
║                       │  applications              │  Speed-critical tasks   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

KEY SURF INNOVATIONS:
━━━━━━━━━━━━━━━━━━━━━

1. INTEGRAL IMAGES → O(1) box filter computation
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ SIFT: Each pixel requires O(filter_size²) operations                   │
   │ SURF: Each pixel requires O(1) operations (4 lookups + 3 additions)    │
   └─────────────────────────────────────────────────────────────────────────┘

2. BOX FILTERS → Approximate Gaussian derivatives cheaply
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Gaussian 2nd derivative ≈ Simple box pattern (+1, -2, +1)              │
   │ Quality loss is minimal, speed gain is significant                     │
   └─────────────────────────────────────────────────────────────────────────┘

3. FILTER PYRAMID → No image resizing needed
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ SIFT: Resize image at each octave (expensive memory + computation)     │
   │ SURF: Apply larger filters to same image (cheap with integral image)   │
   └─────────────────────────────────────────────────────────────────────────┘

4. HAAR WAVELETS → Fast orientation and descriptor computation
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Simple +1/-1 patterns computed with integral image                     │
   │ 64-D descriptor is more compact than SIFT's 128-D                      │
   └─────────────────────────────────────────────────────────────────────────┘
"""
    ax.text(0.01, 0.5, comparison, fontsize=9, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))
    ax.set_title('SURF vs SIFT Detailed Comparison', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_vs_sift.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_vs_sift.png")


def visualize_speed_comparison():
    """Speed comparison visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # SIFT computation
    ax1 = axes[0]
    ax1.axis('off')
    sift_speed = """
SIFT COMPUTATION (Slow)
═══════════════════════════════════

Standard Convolution:

For EACH pixel in image:
  For EACH pixel in filter:
    multiply + add
    
= O(H × W × filter_size²)

Example (640×480 image, 9×9 filter):
  640 × 480 × 81 = 24,883,200 ops

For scale-space (4 octaves × 5 scales):
  Total ≈ 500 million operations!

Plus: Must RESIZE image at each octave
  - Additional memory allocation
  - Image interpolation overhead


Time complexity per scale:
  O(H × W × filter_size²)
  
For 21×21 filter:
  640 × 480 × 441 = 135,475,200 ops
"""
    ax1.text(0.05, 0.5, sift_speed, fontsize=11, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
    ax1.set_title('SIFT: O(filter_size²) per pixel', fontsize=14, fontweight='bold', color='red')
    
    # SURF computation
    ax2 = axes[1]
    ax2.axis('off')
    surf_speed = """
SURF COMPUTATION (Fast)
═══════════════════════════════════

Integral Image Magic:

For EACH pixel in image:
  4 array lookups
  3 additions
  
= O(H × W × 4)

Example (640×480 image, ANY filter):
  640 × 480 × 4 = 1,228,800 ops

For scale-space (same image, all scales):
  Total ≈ 5 million operations!

Plus: NO image resizing needed!
  - Single integral image
  - All filter sizes work


Time complexity per scale:
  O(H × W × 4) = O(H × W)
  
For 21×21 OR 99×99 filter:
  640 × 480 × 4 = 1,228,800 ops
  (CONSTANT regardless of filter size!)


Speed improvement:
  Per filter: 20× to 100× faster
  Overall:    ~3× faster end-to-end
"""
    ax2.text(0.05, 0.5, surf_speed, fontsize=11, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))
    ax2.set_title('SURF: O(1) per pixel with integral image', fontsize=14, fontweight='bold', color='green')
    
    plt.suptitle('Speed Comparison: Why SURF is Faster', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_speed_comparison.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_speed_comparison.png")


def visualize_summary():
    """Final summary table"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    summary = """
╔════════════════════════════════════════════════════════════════╗
║                      SURF ALGORITHM SUMMARY                     ║
╠════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  Property              │  Value                                 ║
║  ──────────────────────┼───────────────────────────────────────║
║  Full Name             │  Speeded-Up Robust Features           ║
║  Year                  │  2006 (Bay, Tuytelaars, Van Gool)     ║
║  Speed                 │  ~3× faster than SIFT                 ║
║  ──────────────────────┼───────────────────────────────────────║
║  1️⃣ DETECTION PHASE    │                                       ║
║     Integral Image     │  II(x,y) = Σ I(i,j) → O(1) box sums  ║
║     Blob Detector      │  det(H) = Dxx·Dyy - (0.9·Dxy)²       ║
║     Scale-Space        │  Filter pyramid (9,15,21,27,...)     ║
║     Localization       │  Sub-pixel refinement                 ║
║  ──────────────────────┼───────────────────────────────────────║
║  2️⃣ DESCRIPTION PHASE  │                                       ║
║     Orientation        │  Haar wavelets + 60° sliding window  ║
║     Descriptor Size    │  64-D (or 128-D extended)            ║
║     Descriptor Values  │  [Σdx, Σdy, Σ|dx|, Σ|dy|] × 16      ║
║  ──────────────────────┼───────────────────────────────────────║
║  Key Innovation        │  Integral images + Box filters        ║
║  Best For              │  Real-time applications               ║
║                                                                 ║
╚════════════════════════════════════════════════════════════════╝

QUICK REFERENCE - FORMULAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Detection:
  II(x,y) = I(x,y) + II(x-1,y) + II(x,y-1) - II(x-1,y-1)
  Box_sum = II(D) - II(B) - II(C) + II(A)
  det(H) = Dxx·Dyy - (0.9·Dxy)²

Description:
  Haar_dx = I(right) - I(left)
  Haar_dy = I(bottom) - I(top)
  Descriptor = [Σdx, Σdy, Σ|dx|, Σ|dy|] × 16 subregions
"""
    ax.text(0.05, 0.5, summary, fontsize=11, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title('SURF Algorithm Summary', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_summary.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_summary.png")


def visualize_complete_summary_visual():
    """Visual summary with images"""
    img = load_image()
    H, W = img.shape
    
    # Generate sample keypoints
    np.random.seed(42)
    keypoints = []
    for _ in range(50):
        kp = {
            'x': np.random.randint(50, W - 50),
            'y': np.random.randint(50, H - 50),
            'scale': np.random.randint(1, 4),
            'orientation': np.random.uniform(-np.pi, np.pi)
        }
        keypoints.append(kp)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.2)
    
    # Input
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f'1. Input Image\n{W} × {H}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Integral image
    ax2 = fig.add_subplot(gs[0, 1])
    integral = np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)
    ii_norm = (integral - integral.min()) / (integral.max() - integral.min())
    ax2.imshow(ii_norm, cmap='viridis')
    ax2.set_title('2. Integral Image\n(O(1) box sums)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Detection
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img, cmap='gray')
    colors = ['red', 'lime', 'blue']
    for kp in keypoints[:30]:
        scale = kp['scale'] - 1
        circle = plt.Circle((kp['x'], kp['y']), 4 + scale * 3, 
                           color=colors[scale], fill=False, linewidth=1.5)
        ax3.add_patch(circle)
    ax3.set_title(f'3. Detection\n{len(keypoints)} keypoints', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Orientation
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(img, cmap='gray')
    for kp in keypoints[::2]:
        x, y = kp['x'], kp['y']
        ori = kp['orientation']
        size = 4 + (kp['scale'] - 1) * 2
        circle = plt.Circle((x, y), size, color='red', fill=False, linewidth=1.5)
        ax4.add_patch(circle)
        ax4.arrow(x, y, size*1.8*np.cos(ori), size*1.8*np.sin(ori),
                 head_width=4, head_length=3, fc='yellow', ec='yellow')
    ax4.set_title('4. Orientation', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 4×4 grid
    ax5 = fig.add_subplot(gs[1, 1])
    kp_x, kp_y = W // 2, H // 2
    half = 30
    region = img[max(0, kp_y-half):min(H, kp_y+half), max(0, kp_x-half):min(W, kp_x+half)]
    if region.size > 0:
        ax5.imshow(region, cmap='gray', extent=[0, 4, 4, 0])
        for i in range(5):
            ax5.axhline(y=i, color='lime', linewidth=2)
            ax5.axvline(x=i, color='lime', linewidth=2)
    ax5.set_title('5. 4×4 Subregions', fontsize=12, fontweight='bold')
    
    # 64-D descriptor
    ax6 = fig.add_subplot(gs[1, 2])
    desc = np.random.randn(64)
    desc = desc / np.linalg.norm(desc)
    bar_colors = plt.cm.tab20(np.repeat(np.arange(16), 4) / 16)
    ax6.bar(range(64), desc, color=bar_colors, edgecolor='none', width=1)
    ax6.set_xlabel('Index (0-63)')
    ax6.set_ylabel('Value')
    ax6.set_title('6. 64-D Descriptor', fontsize=12, fontweight='bold')
    ax6.set_xlim(-1, 64)
    
    plt.suptitle('SURF Complete Pipeline: Detection → Description', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_complete_summary.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_complete_summary.png")


if __name__ == "__main__":
    visualize_complete_pipeline()
    visualize_surf_vs_sift()
    visualize_speed_comparison()
    visualize_summary()
    visualize_complete_summary_visual()
    print("\nSummary images generated successfully!")

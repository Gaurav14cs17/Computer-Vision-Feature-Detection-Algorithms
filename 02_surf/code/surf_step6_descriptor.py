"""
SURF Step 6: Descriptor Extraction (64-D) Visualization
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


def visualize_descriptor_region():
    """Visualize the 20s × 20s descriptor region"""
    print("=" * 60)
    print("SURF Step 6: Descriptor Extraction (64-D)")
    print("=" * 60)
    
    img = load_image()
    H, W = img.shape
    
    # Sample keypoint
    kp_x, kp_y = W // 2, H // 2
    scale = 2
    half = 15 * scale  # 20s total, half is 10s
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image with region
    ax1 = axes[0]
    ax1.imshow(img, cmap='gray')
    rect = patches.Rectangle((kp_x - half, kp_y - half), 2*half, 2*half,
                              linewidth=3, edgecolor='lime', facecolor='none')
    ax1.add_patch(rect)
    ax1.plot(kp_x, kp_y, 'r+', markersize=20, markeredgewidth=3)
    ax1.set_xlim(kp_x - half - 30, kp_x + half + 30)
    ax1.set_ylim(kp_y + half + 30, kp_y - half - 30)
    ax1.set_title('Step 6.1: 20s × 20s Region\n(Aligned with orientation)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 4×4 grid
    ax2 = axes[1]
    region = img[max(0, kp_y-half):min(H, kp_y+half), max(0, kp_x-half):min(W, kp_x+half)]
    if region.size > 0:
        ax2.imshow(region, cmap='gray', extent=[0, 4, 4, 0])
        for i in range(5):
            ax2.axhline(y=i, color='lime', linewidth=2)
            ax2.axvline(x=i, color='lime', linewidth=2)
        for i in range(4):
            for j in range(4):
                ax2.text(j + 0.5, i + 0.5, f'S{i*4+j}', fontsize=10, ha='center', va='center',
                        color='yellow', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    ax2.set_title('Step 6.2: 4×4 = 16 Subregions\n(Each 5s × 5s)', fontsize=14, fontweight='bold')
    
    # Descriptor structure
    ax3 = axes[2]
    ax3.axis('off')
    structure = """
Step 6.3: Per Subregion Values
════════════════════════════════════

For each of 16 subregions:
  • Sample 5×5 = 25 points
  • Compute Haar wavelet responses
  • Rotate responses by -θ (keypoint orientation)
  
Create 4-value vector:
  [Σdx, Σdy, Σ|dx|, Σ|dy|]

Meaning:
  Σdx   → Sum of horizontal gradients
          (Direction of intensity change)
          
  Σdy   → Sum of vertical gradients
          (Direction of intensity change)
          
  Σ|dx| → Sum of absolute horizontal gradients
          (Magnitude of horizontal change)
          
  Σ|dy| → Sum of absolute vertical gradients
          (Magnitude of vertical change)

Total: 16 subregions × 4 values = 64-D
"""
    ax3.text(0.02, 0.5, structure, fontsize=11, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax3.set_title('Subregion Values', fontsize=14, fontweight='bold')
    
    plt.suptitle('SURF Step 6: Descriptor Extraction (64-D)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step6_region.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step6_region.png")


def visualize_descriptor_vector():
    """Visualize the 64-D descriptor vector"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Generate sample descriptor
    np.random.seed(42)
    desc = np.random.randn(64)
    desc = desc / np.linalg.norm(desc)
    
    # 64-D bar chart
    ax1 = axes[0]
    bar_colors = plt.cm.tab20(np.repeat(np.arange(16), 4) / 16)
    ax1.bar(range(64), desc, color=bar_colors, edgecolor='none', width=1)
    
    # Add vertical lines to separate subregions
    for i in range(1, 16):
        ax1.axvline(x=i*4 - 0.5, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Descriptor Index (0-63)', fontsize=12)
    ax1.set_ylabel('Value (normalized)', fontsize=12)
    ax1.set_title('64-D SURF Descriptor Vector', fontsize=14, fontweight='bold')
    ax1.set_xlim(-1, 64)
    
    # Structure explanation
    ax2 = axes[1]
    ax2.axis('off')
    
    structure = """
SURF DESCRIPTOR STRUCTURE
══════════════════════════════════════════════════════════════════════════════

Index Layout:
┌──────────────────────────────────────────────────────────────────────────┐
│ S0      │ S1      │ S2      │ S3      │ ... │ S14     │ S15     │
│ [0-3]   │ [4-7]   │ [8-11]  │ [12-15] │     │ [56-59] │ [60-63] │
└──────────────────────────────────────────────────────────────────────────┘

Each Subregion (4 values):
┌─────────┬─────────┬─────────┬─────────┐
│  Σdx    │  Σdy    │ Σ|dx|   │ Σ|dy|   │
│ Index 0 │ Index 1 │ Index 2 │ Index 3 │
└─────────┴─────────┴─────────┴─────────┘

Spatial Layout (4×4 grid):
┌────────┬────────┬────────┬────────┐
│S0  0-3 │S1  4-7 │S2  8-11│S3 12-15│  Each subregion = 5s × 5s
├────────┼────────┼────────┼────────┤
│S4 16-19│S5 20-23│S6 24-27│S7 28-31│  Total region = 20s × 20s
├────────┼────────┼────────┼────────┤
│S8 32-35│S9 36-39│S10 40-43│S11 44-47│  s = keypoint scale
├────────┼────────┼────────┼────────┤
│S12 48-51│S13 52-55│S14 56-59│S15 60-63│
└────────┴────────┴────────┴────────┘

Normalization:
  descriptor = descriptor / ||descriptor||
  (Unit length vector for illumination invariance)
"""
    ax2.text(0.02, 0.5, structure, fontsize=10, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step6_descriptor.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step6_descriptor.png")


def visualize_descriptor_details():
    """Detailed explanation of descriptor computation"""
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.axis('off')
    
    details = """
SURF DESCRIPTOR EXTRACTION - Complete Details
══════════════════════════════════════════════════════════════════════════════

STEP 6.1: Define Descriptor Region
────────────────────────────────────────────────────────────────────────────
  • Size: 20s × 20s (s = keypoint scale)
  • Centered on keypoint (x, y)
  • ROTATED by dominant orientation θ (from Step 5)
  
  This rotation achieves ROTATION INVARIANCE!


STEP 6.2: Divide into 4×4 = 16 Subregions
────────────────────────────────────────────────────────────────────────────
  • Each subregion: 5s × 5s pixels
  • Total: 16 subregions arranged in 4×4 grid


STEP 6.3: Sample Points in Each Subregion
────────────────────────────────────────────────────────────────────────────
  For each subregion:
    • Sample 5×5 = 25 points at regular intervals
    • For each sample point:
      
      a) Compute Haar wavelet responses:
         dx = I(right) - I(left)   using 2s × 2s wavelet
         dy = I(bottom) - I(top)
      
      b) Rotate by keypoint orientation (-θ):
         dx' = dx·cos(θ) + dy·sin(θ)
         dy' = -dx·sin(θ) + dy·cos(θ)
      
      c) Weight by Gaussian centered at keypoint:
         weight = exp(-(dist²) / (2 × (3.3s)²))


STEP 6.4: Build Subregion Descriptor (4 values)
────────────────────────────────────────────────────────────────────────────
  For each subregion, compute:
  
    Σdx   = Σ (dx' × weight)    → Horizontal gradient sum
    Σdy   = Σ (dy' × weight)    → Vertical gradient sum
    Σ|dx| = Σ (|dx'| × weight)  → Horizontal gradient magnitude
    Σ|dy| = Σ (|dy'| × weight)  → Vertical gradient magnitude
  
  Subregion vector = [Σdx, Σdy, Σ|dx|, Σ|dy|]


STEP 6.5: Concatenate All Subregions
────────────────────────────────────────────────────────────────────────────
  Descriptor = [S0, S1, S2, ..., S15]
             = [Σdx₀, Σdy₀, Σ|dx|₀, Σ|dy|₀, Σdx₁, Σdy₁, ..., Σ|dy|₁₅]
             = 64 values


STEP 6.6: Normalize
────────────────────────────────────────────────────────────────────────────
  descriptor = descriptor / ||descriptor||
  
  This makes it invariant to:
    • Illumination changes (contrast)
    • Affine brightness changes


SURF-64 vs SURF-128:
────────────────────────────────────────────────────────────────────────────
  SURF-64:  [Σdx, Σdy, Σ|dx|, Σ|dy|] × 16 = 64 values (default)
  
  SURF-128: Split sums by sign of dy:
            [Σdx(dy<0), Σdx(dy≥0), Σ|dx|(dy<0), Σ|dx|(dy≥0),
             Σdy(dy<0), Σdy(dy≥0), Σ|dy|(dy<0), Σ|dy|(dy≥0)] × 16 = 128
            
            More distinctive, but slower to compute and match
"""
    ax.text(0.02, 0.5, details, fontsize=9.5, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title('SURF Step 6: Descriptor Extraction - Complete Details', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step6_details.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step6_details.png")


def visualize_numerical_example():
    """Show numerical example of descriptor computation"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    
    example = """
DESCRIPTOR COMPUTATION - Numerical Example
══════════════════════════════════════════════════════════════════════════════

Given: Keypoint at (200, 150), scale s=2, orientation θ=30°

STEP 1: Define Region
────────────────────────────────────────────────────────────────────────────
  Region size = 20s = 40 pixels
  Center = (200, 150)
  Corners (before rotation):
    Top-left:     (180, 130)
    Top-right:    (220, 130)
    Bottom-left:  (180, 170)
    Bottom-right: (220, 170)

STEP 2: For Subregion S5 (row 1, col 1)
────────────────────────────────────────────────────────────────────────────
  Subregion size = 5s = 10 pixels
  Subregion center ≈ (195, 145)
  
  Sample 5×5 = 25 points
  
  Sample point (193, 143):
    dx_raw = I(195,143) - I(191,143) = 120 - 85 = 35
    dy_raw = I(193,145) - I(193,141) = 95 - 100 = -5
    
  Rotate by -30°:
    cos(-30°) = 0.866,  sin(-30°) = -0.5
    dx' = 35×0.866 + (-5)×(-0.5) = 30.3 + 2.5 = 32.8
    dy' = -35×(-0.5) + (-5)×0.866 = 17.5 - 4.3 = 13.2
  
  Weight by Gaussian (dist from keypoint center):
    dist = √((193-200)² + (143-150)²) = √98 ≈ 9.9
    weight = exp(-9.9² / (2×(3.3×2)²)) = exp(-2.25) ≈ 0.11
    
    dx'_weighted = 32.8 × 0.11 = 3.6
    dy'_weighted = 13.2 × 0.11 = 1.5

STEP 3: Sum Over All 25 Points in S5
────────────────────────────────────────────────────────────────────────────
  Σdx   = 3.6 + 2.1 + 4.5 + ... = 45.2
  Σdy   = 1.5 + 0.8 + 2.3 + ... = 28.7
  Σ|dx| = |3.6| + |2.1| + |4.5| + ... = 52.3
  Σ|dy| = |1.5| + |0.8| + |2.3| + ... = 35.1

  S5 vector = [45.2, 28.7, 52.3, 35.1]

STEP 4: Final Descriptor (after all subregions)
────────────────────────────────────────────────────────────────────────────
  desc = [S0, S1, S2, S3, S4, S5, ..., S15]
       = [12.3, 8.5, 15.2, 10.1,  ...,  45.2, 28.7, 52.3, 35.1,  ...,  ...]
         └─────── S0 ────────┘          └─────── S5 ────────┘

  Normalize:
    ||desc|| = √(12.3² + 8.5² + ... + 35.1² + ...) = 245.8
    desc_normalized = desc / 245.8
    
    S5_normalized = [0.184, 0.117, 0.213, 0.143]
"""
    ax.text(0.02, 0.5, example, fontsize=9.5, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title('SURF Descriptor - Numerical Example', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step6_example.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step6_example.png")


if __name__ == "__main__":
    visualize_descriptor_region()
    visualize_descriptor_vector()
    visualize_descriptor_details()
    visualize_numerical_example()
    print("\nStep 6 images generated successfully!")

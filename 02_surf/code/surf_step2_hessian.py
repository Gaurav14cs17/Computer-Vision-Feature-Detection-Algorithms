"""
SURF Step 2: Hessian Matrix with Box Filters Visualization
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


def compute_integral_image(img):
    return np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)


def box_sum(integral, x1, y1, x2, y2):
    h, w = integral.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    D = integral[y2, x2]
    B = integral[y1-1, x2] if y1 > 0 else 0
    C = integral[y2, x1-1] if x1 > 0 else 0
    A = integral[y1-1, x1-1] if y1 > 0 and x1 > 0 else 0
    return D - B - C + A


def compute_hessian_response(integral, x, y, filter_size):
    """Compute Hessian determinant using box filters"""
    h, w = integral.shape
    half = filter_size // 2
    
    if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
        return 0, 0, 0
    
    lobe_w = filter_size // 3
    
    # Dxx: horizontal second derivative
    left = box_sum(integral, x - half, y - half, x - half + lobe_w - 1, y + half)
    center = box_sum(integral, x - lobe_w//2, y - half, x + lobe_w//2, y + half)
    right = box_sum(integral, x + half - lobe_w + 1, y - half, x + half, y + half)
    Dxx = left - 2 * center + right
    
    # Dyy: vertical second derivative
    top = box_sum(integral, x - half, y - half, x + half, y - half + lobe_w - 1)
    middle = box_sum(integral, x - half, y - lobe_w//2, x + half, y + lobe_w//2)
    bottom = box_sum(integral, x - half, y + half - lobe_w + 1, x + half, y + half)
    Dyy = top - 2 * middle + bottom
    
    # Dxy: mixed derivative (checkerboard pattern)
    tl = box_sum(integral, x - half, y - half, x - 1, y - 1)
    tr = box_sum(integral, x + 1, y - half, x + half, y - 1)
    bl = box_sum(integral, x - half, y + 1, x - 1, y + half)
    br = box_sum(integral, x + 1, y + 1, x + half, y + half)
    Dxy = tl - tr - bl + br
    
    # Normalize by area
    area = filter_size * filter_size
    Dxx /= area
    Dyy /= area
    Dxy /= area
    
    return Dxx, Dyy, Dxy


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


def visualize_box_filters():
    """Visualize Dxx, Dyy, Dxy box filter patterns"""
    print("=" * 60)
    print("SURF Step 2: Box Filters")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Dxx filter (9×9)
    ax1 = axes[0]
    ax1.set_xlim(0, 9)
    ax1.set_ylim(9, 0)
    
    # Define colors: green=+1, red=-2, white=0
    for i in range(9):
        for j in range(9):
            # Dxx pattern: vertical lobes
            if 3 <= j <= 5:  # center column (vertical stripe)
                if 3 <= i <= 5:
                    color, val = 'red', '-2'
                else:
                    color, val = 'green', '+1'
            elif 0 <= j <= 2 and 3 <= i <= 5:  # left lobe
                color, val = 'green', '+1'
            elif 6 <= j <= 8 and 3 <= i <= 5:  # right lobe  
                color, val = 'green', '+1'
            else:
                color, val = 'white', '0'
            rect = patches.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black', alpha=0.7)
            ax1.add_patch(rect)
    
    ax1.set_title('Dxx Filter (9×9)\n∂²L/∂x² approximation', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Dyy filter (9×9) - rotated version of Dxx
    ax2 = axes[1]
    ax2.set_xlim(0, 9)
    ax2.set_ylim(9, 0)
    
    for i in range(9):
        for j in range(9):
            # Dyy pattern: horizontal lobes
            if 3 <= i <= 5:  # center row (horizontal stripe)
                if 3 <= j <= 5:
                    color, val = 'red', '-2'
                else:
                    color, val = 'green', '+1'
            elif 0 <= i <= 2 and 3 <= j <= 5:  # top lobe
                color, val = 'green', '+1'
            elif 6 <= i <= 8 and 3 <= j <= 5:  # bottom lobe
                color, val = 'green', '+1'
            else:
                color, val = 'white', '0'
            rect = patches.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black', alpha=0.7)
            ax2.add_patch(rect)
    
    ax2.set_title('Dyy Filter (9×9)\n∂²L/∂y² approximation', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Dxy filter (9×9) - checkerboard pattern
    ax3 = axes[2]
    ax3.set_xlim(0, 9)
    ax3.set_ylim(9, 0)
    
    for i in range(9):
        for j in range(9):
            # Dxy pattern: checkerboard
            if i < 4 and j < 4:
                color = 'green'  # top-left: +1
            elif i < 4 and j > 4:
                color = 'red'    # top-right: -1
            elif i > 4 and j < 4:
                color = 'red'    # bottom-left: -1
            elif i > 4 and j > 4:
                color = 'green'  # bottom-right: +1
            else:
                color = 'white'  # center line: 0
            rect = patches.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black', alpha=0.7)
            ax3.add_patch(rect)
    
    ax3.set_title('Dxy Filter (9×9)\n∂²L/∂x∂y approximation', fontsize=14, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    plt.suptitle('SURF Box Filters (Approximate Gaussian Second Derivatives)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step2_boxfilters.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step2_boxfilters.png")


def visualize_hessian_response():
    """Visualize Hessian response maps at different scales"""
    img = load_image()
    H, W = img.shape
    integral = compute_integral_image(img)
    
    filter_sizes = [9, 15, 21, 27]
    responses = []
    
    print("Computing Hessian responses...")
    for fs in filter_sizes:
        print(f"  Filter size {fs}×{fs}...")
        response = np.zeros((H, W))
        margin = fs // 2 + 1
        for y in range(margin, H - margin, 2):
            for x in range(margin, W - margin, 2):
                Dxx, Dyy, Dxy = compute_hessian_response(integral, x, y, fs)
                response[y, x] = Dxx * Dyy - (0.9 * Dxy) ** 2
        responses.append(response)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    ax1 = axes[0, 0]
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Formula
    ax2 = axes[0, 1]
    ax2.axis('off')
    formula = """
Hessian Matrix:
        
    H = | Dxx  Dxy |
        | Dxy  Dyy |

Blob Response (Determinant):
        
    det(H) = Dxx · Dyy - (w · Dxy)²
    
    where w = 0.9 (correction factor)

• Positive det(H) → Blob-like structure
• Larger det(H) → Stronger blob response
• w = 0.9 compensates for box filter 
  approximation error

Box filters enable O(1) computation
using integral images!
"""
    ax2.text(0.05, 0.5, formula, fontsize=11, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax2.set_title('Hessian Formula', fontsize=12, fontweight='bold')
    
    # Filter size comparison
    ax3 = axes[0, 2]
    ax3.axis('off')
    comparison = """
Filter Sizes (Scale Pyramid):

┌─────────────────────────────┐
│  9×9   (σ ≈ 1.2)  Scale 1   │
│ 15×15  (σ ≈ 2.0)  Scale 2   │
│ 21×21  (σ ≈ 2.8)  Scale 3   │
│ 27×27  (σ ≈ 3.6)  Scale 4   │
└─────────────────────────────┘

SURF Innovation:
• Keep image size constant
• Increase filter size
• Faster than resizing image!

Scale relationship:
  σ = filter_size × 1.2 / 9
"""
    ax3.text(0.05, 0.5, comparison, fontsize=11, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    ax3.set_title('Scale-Space via Filter Pyramid', fontsize=12, fontweight='bold')
    
    # Hessian responses at different scales
    for i in range(3):
        ax = axes[1, i]
        resp = np.abs(responses[i])
        resp_norm = resp / (resp.max() + 1e-8)
        ax.imshow(resp_norm, cmap='hot')
        ax.set_title(f'|det(H)| at Filter {filter_sizes[i]}×{filter_sizes[i]}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('SURF Step 2: Hessian Matrix with Box Filters', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step2_hessian.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step2_hessian.png")
    
    return responses


def visualize_hessian_example():
    """Show numerical example of Hessian computation"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    
    example = """
HESSIAN COMPUTATION - Numerical Example
══════════════════════════════════════════════════════════════════════════════

STEP 1: Apply Box Filters at Point (x, y) with 9×9 filter
────────────────────────────────────────────────────────────────────────────

For Dxx (Horizontal 2nd Derivative):
┌─────────────────────────┐
│     │ +1  │     │       │   
│ +1  │ -2  │ +1  │       │  Dxx = Left_lobe - 2×Center + Right_lobe
│     │ +1  │     │       │      = 450 - 2×380 + 420
└─────────────────────────┘      = 450 - 760 + 420 = 110

For Dyy (Vertical 2nd Derivative):
┌─────────────────────────┐
│     │ +1  │     │       │   
│ +1  │ -2  │ +1  │       │  Dyy = Top_lobe - 2×Center + Bottom_lobe
│     │ +1  │     │       │      = 430 - 2×380 + 440
└─────────────────────────┘      = 430 - 760 + 440 = 110

For Dxy (Mixed Derivative - Checkerboard):
┌───────────────┐
│ +1  │    │ -1 │    Dxy = TL - TR - BL + BR
├─────┼────┼────┤        = 200 - 180 - 190 + 210
│     │    │    │        = 40
├─────┼────┼────┤
│ -1  │    │ +1 │
└───────────────┘

STEP 2: Compute Hessian Determinant
────────────────────────────────────────────────────────────────────────────

Normalize by filter area:
  Dxx_norm = 110 / 81 = 1.36
  Dyy_norm = 110 / 81 = 1.36
  Dxy_norm = 40 / 81 = 0.49

det(H) = Dxx × Dyy - (0.9 × Dxy)²
       = 1.36 × 1.36 - (0.9 × 0.49)²
       = 1.85 - 0.19
       = 1.66

STEP 3: Interpret Result
────────────────────────────────────────────────────────────────────────────

det(H) = 1.66 > 0  →  BLOB detected at this point!

• Higher det(H) = stronger blob response
• Compare across scales to find scale of blob
• Compare with 26 neighbors to find local maxima

KEY INSIGHT: Entire computation uses only integral image lookups!
             No convolution needed → O(1) per pixel!
"""
    ax.text(0.02, 0.5, example, fontsize=10, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title('SURF Step 2: Hessian Computation - Numerical Example', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step2_example.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step2_example.png")


if __name__ == "__main__":
    visualize_box_filters()
    visualize_hessian_response()
    visualize_hessian_example()
    print("\nStep 2 images generated successfully!")

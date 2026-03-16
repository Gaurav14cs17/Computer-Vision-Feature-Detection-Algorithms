"""
SURF Step 1: Integral Image Visualization
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
    """Compute integral image using cumulative sum"""
    return np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)


def box_sum(integral, x1, y1, x2, y2):
    """Compute sum of rectangle using integral image (O(1) operation)"""
    h, w = integral.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    
    D = integral[y2, x2]
    B = integral[y1-1, x2] if y1 > 0 else 0
    C = integral[y2, x1-1] if x1 > 0 else 0
    A = integral[y1-1, x1-1] if y1 > 0 and x1 > 0 else 0
    
    return D - B - C + A


def load_image():
    """Load input image"""
    input_path = os.path.join(IMAGES_DIR, 'input_image.jpg')
    if os.path.exists(input_path):
        img = Image.open(input_path).convert('L')
        if img.size[0] > 800 or img.size[1] > 600:
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
        return np.array(img)
    else:
        print("Creating sample image...")
        img = np.zeros((480, 640), dtype=np.uint8)
        img[100:200, 100:250] = 180
        img[250:350, 300:450] = 220
        img[150:250, 400:550] = 150
        Image.fromarray(img).save(input_path)
        return img


def visualize_integral_image():
    """Generate integral image visualization"""
    print("=" * 60)
    print("SURF Step 1: Integral Image")
    print("=" * 60)
    
    img = load_image()
    H, W = img.shape
    print(f"Image size: {W} × {H}")
    
    integral = compute_integral_image(img)
    
    # Main visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    ax1 = axes[0]
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Image I(x,y)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Integral image
    ax2 = axes[1]
    ii_norm = (integral - integral.min()) / (integral.max() - integral.min())
    ax2.imshow(ii_norm, cmap='viridis')
    ax2.set_title('Integral Image II(x,y)\nII(x,y) = Σᵢ₌₀ˣ Σⱼ₌₀ʸ I(i,j)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Box sum demonstration
    ax3 = axes[2]
    ax3.imshow(img, cmap='gray')
    x1, y1, x2, y2 = 100, 100, 200, 200
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                              linewidth=3, edgecolor='lime', facecolor='lime', alpha=0.3)
    ax3.add_patch(rect)
    region_sum = box_sum(integral, x1, y1, x2, y2)
    ax3.set_title(f'Box Sum = {region_sum:.0f}\n(Only 4 lookups!)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    plt.suptitle('SURF Step 1: Integral Image for O(1) Box Sums', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step1_integral.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step1_integral.png")
    
    return integral


def visualize_integral_example():
    """Generate numerical example of integral image computation"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    example = """
INTEGRAL IMAGE - Numerical Example
══════════════════════════════════════════════════════════════════════════════

STEP 1: Original Image (5×5)
┌───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │
├───┼───┼───┼───┼───┤
│ 2 │ 3 │ 4 │ 5 │ 6 │
├───┼───┼───┼───┼───┤
│ 3 │ 4 │ 5 │ 6 │ 7 │
├───┼───┼───┼───┼───┤
│ 4 │ 5 │ 6 │ 7 │ 8 │
├───┼───┼───┼───┼───┤
│ 5 │ 6 │ 7 │ 8 │ 9 │
└───┴───┴───┴───┴───┘

STEP 2: Compute Integral Image
                              
Formula: II(x,y) = I(x,y) + II(x-1,y) + II(x,y-1) - II(x-1,y-1)

Integral Image (5×5):
┌────┬────┬────┬────┬────┐
│  1 │  3 │  6 │ 10 │ 15 │  ← First row: cumulative sum
├────┼────┼────┼────┼────┤
│  3 │  8 │ 15 │ 24 │ 35 │
├────┼────┼────┼────┼────┤
│  6 │ 15 │ 27 │ 42 │ 60 │
├────┼────┼────┼────┼────┤
│ 10 │ 24 │ 42 │ 64 │ 90 │
├────┼────┼────┼────┼────┤
│ 15 │ 35 │ 60 │ 90 │125 │  ← II(4,4) = sum of entire image
└────┴────┴────┴────┴────┘

STEP 3: Box Sum Formula (O(1) computation!)
                    A ─────────── B
                    │   REGION   │
                    │             │
                    C ─────────── D

Sum = II(D) - II(B) - II(C) + II(A)

Example: Sum of 3×3 region (rows 1-3, cols 1-3)

    A=(0,0)   B=(0,3)
       ┌─────────┐
       │ 3 4 5   │   A = II(0,0) = 1
       │ 4 5 6   │   B = II(0,3) = 10
       │ 5 6 7   │   C = II(3,0) = 10
       └─────────┘   D = II(3,3) = 64
    C=(3,0)   D=(3,3)

Sum = 64 - 10 - 10 + 1 = 45  ✓ (Verify: 3+4+5+4+5+6+5+6+7 = 45)

KEY INSIGHT: ANY rectangle sum = 4 lookups = O(1)!
             Regardless of rectangle size!
"""
    ax.text(0.02, 0.5, example, fontsize=10, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title('SURF Step 1: Integral Image - Numerical Example', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step1_example.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step1_example.png")


if __name__ == "__main__":
    visualize_integral_image()
    visualize_integral_example()
    print("\nStep 1 images generated successfully!")

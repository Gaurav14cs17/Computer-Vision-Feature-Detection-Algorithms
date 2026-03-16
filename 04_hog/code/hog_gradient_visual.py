"""
HOG Gradient Visualization
Detailed visualizations for gradient computation step using real images
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')
os.makedirs(OUT_DIR, exist_ok=True)


def load_image():
    """Load the input image"""
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    img = np.array(Image.open(image_path))
    if len(img.shape) == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float64)
    return gray / 255.0, img


def compute_gradients(img):
    """Compute gradients"""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) * 180 / np.pi % 180
    return magnitude, direction, gx, gy


def create_gradient_computation_visual():
    """Detailed visualization of gradient computation with real image patch"""
    gray, _ = load_image()
    
    # Extract a small patch for detailed visualization
    patch_y, patch_x = 100, 150
    patch_size = 16
    patch = gray[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Compute gradients for patch
    gx = np.zeros_like(patch)
    gy = np.zeros_like(patch)
    gx[:, 1:-1] = patch[:, 2:] - patch[:, :-2]
    gy[1:-1, :] = patch[2:, :] - patch[:-2, :]
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) * 180 / np.pi % 180
    
    # Panel 1: Original patch
    ax = axes[0, 0]
    ax.imshow(patch, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Image Patch ({patch_size}×{patch_size})\nfrom position ({patch_x}, {patch_y})', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Panel 2: Gx
    ax = axes[0, 1]
    cmap_rg = LinearSegmentedColormap.from_list('rg', ['blue', 'white', 'red'])
    im = ax.imshow(gx, cmap=cmap_rg, vmin=-0.3, vmax=0.3)
    ax.set_title('Horizontal Gradient (Gx)\nGx = I(x+1,y) - I(x-1,y)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Panel 3: Gy
    ax = axes[0, 2]
    im = ax.imshow(gy, cmap=cmap_rg, vmin=-0.3, vmax=0.3)
    ax.set_title('Vertical Gradient (Gy)\nGy = I(x,y+1) - I(x,y-1)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Panel 4: Magnitude
    ax = axes[1, 0]
    im = ax.imshow(magnitude, cmap='hot')
    ax.set_title('Gradient Magnitude\nM = √(Gx² + Gy²)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Panel 5: Direction
    ax = axes[1, 1]
    im = ax.imshow(direction, cmap='hsv', vmin=0, vmax=180)
    ax.set_title('Gradient Direction (0°-180°)\nθ = arctan(Gy/Gx) mod 180°', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.8, label='degrees')
    
    # Panel 6: Gradient vectors
    ax = axes[1, 2]
    ax.imshow(patch, cmap='gray', vmin=0, vmax=1, alpha=0.5)
    for i in range(1, patch_size-1, 2):
        for j in range(1, patch_size-1, 2):
            mag = magnitude[i, j]
            angle = direction[i, j]
            if mag > 0.01:
                dx = mag * 4 * np.cos(np.radians(angle))
                dy = mag * 4 * np.sin(np.radians(angle))
                ax.arrow(j, i, dx, dy, head_width=0.3, head_length=0.2, fc='red', ec='red', linewidth=1)
    ax.set_title('Gradient Vectors\n(arrows show direction & magnitude)', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.5, patch_size-0.5)
    ax.set_ylim(patch_size-0.5, -0.5)
    
    plt.suptitle('HOG Step 2: Gradient Computation - Detailed View', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_gradient_computation.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_gradient_computation.png")


def create_edge_detection_visual():
    """Show how gradients reveal edges in real image"""
    gray, original_rgb = load_image()
    
    # Apply gamma correction
    gray = np.power(np.clip(gray, 1e-8, 1), 0.5)
    
    # Compute gradients
    magnitude, direction, gx, gy = compute_gradients(gray)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    ax = axes[0, 0]
    ax.imshow(original_rgb)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[0, 1]
    im = ax.imshow(gx, cmap='RdBu', vmin=-0.3, vmax=0.3)
    ax.set_title('Gx (Horizontal Gradient)\nDetects vertical edges', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[0, 2]
    im = ax.imshow(gy, cmap='RdBu', vmin=-0.3, vmax=0.3)
    ax.set_title('Gy (Vertical Gradient)\nDetects horizontal edges', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[1, 0]
    im = ax.imshow(magnitude, cmap='hot')
    ax.set_title('Gradient Magnitude\nEdge strength', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[1, 1]
    im = ax.imshow(direction, cmap='hsv', vmin=0, vmax=180)
    ax.set_title('Gradient Direction (0°-180°)\nEdge orientation', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7, label='degrees')
    
    # Gradient vectors overlay
    ax = axes[1, 2]
    ax.imshow(gray, cmap='gray', alpha=0.5)
    step = max(gray.shape[0] // 25, 10)
    for y in range(step, gray.shape[0]-step, step):
        for x in range(step, gray.shape[1]-step, step):
            mag = magnitude[y, x]
            angle = direction[y, x]
            if mag > 0.04:
                dx = mag * 12 * np.cos(np.radians(angle))
                dy = mag * 12 * np.sin(np.radians(angle))
                ax.arrow(x, y, dx, dy, head_width=3, head_length=1.5, fc='lime', ec='lime', linewidth=0.8)
    ax.set_title('Gradient Vectors on Image\n(sparse sampling)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG: How Gradients Detect Edges in Real Image', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_edge_detection.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_edge_detection.png")


def create_signed_vs_unsigned_visual():
    """Explain why HOG uses unsigned gradients (0-180) with real examples"""
    gray, _ = load_image()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Find two patches with opposite gradient directions (e.g., light-dark vs dark-light edges)
    # Use patches from different parts of the image
    
    # Panel 1: Explanation diagram
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Two edge examples
    ax.text(2.5, 7.2, 'Light → Dark', fontsize=11, ha='center', fontweight='bold')
    ax.text(7.5, 7.2, 'Dark → Light', fontsize=11, ha='center', fontweight='bold')
    
    # Draw gradient arrows
    ax.arrow(1, 5, 3, 0, head_width=0.3, head_length=0.2, fc='blue', ec='blue', linewidth=3)
    ax.text(2.5, 4.2, 'Gx > 0\nθ = 0°', fontsize=10, ha='center')
    
    ax.arrow(9, 5, -3, 0, head_width=0.3, head_length=0.2, fc='red', ec='red', linewidth=3)
    ax.text(7.5, 4.2, 'Gx < 0\nθ = 180°', fontsize=10, ha='center')
    
    # Explanation
    ax.text(5, 2.5, 'Both are VERTICAL EDGES!\nShould be treated the same.', fontsize=12, ha='center',
            bbox=dict(facecolor='lightyellow', edgecolor='orange', linewidth=2))
    
    ax.text(5, 1, 'Solution: Use unsigned gradients (0°-180°)\n180° → 0° (same bin)', fontsize=11, ha='center',
            bbox=dict(facecolor='lightgreen', edgecolor='green', linewidth=2))
    
    ax.set_title('Why Unsigned Gradients?', fontsize=12, fontweight='bold')
    
    # Panel 2: Signed histogram example
    ax = axes[1]
    signed_bins = np.arange(0, 360, 20)
    hist_signed = np.zeros(18)
    hist_signed[0] = 0.5   # 0° bin
    hist_signed[9] = 0.5   # 180° bin
    
    colors = plt.cm.hsv(np.linspace(0, 1, 18))
    ax.bar(range(18), hist_signed, color=colors, edgecolor='black')
    ax.set_xlabel('Orientation Bin (0°-360°)', fontsize=10)
    ax.set_ylabel('Vote', fontsize=10)
    ax.set_xticks(range(0, 18, 2))
    ax.set_xticklabels([f'{b}°' for b in signed_bins[::2]], fontsize=8, rotation=45)
    ax.set_title('Signed (0°-360°): Different Bins\n(Same edge type in different bins!)', fontsize=11, fontweight='bold', color='red')
    
    # Panel 3: Unsigned histogram example
    ax = axes[2]
    unsigned_bins = np.arange(0, 180, 20)
    hist_unsigned = np.zeros(9)
    hist_unsigned[0] = 1.0  # Both map to 0° bin
    
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    ax.bar(range(9), hist_unsigned, color=colors, edgecolor='black')
    ax.set_xlabel('Orientation Bin (0°-180°)', fontsize=10)
    ax.set_ylabel('Vote', fontsize=10)
    ax.set_xticks(range(9))
    ax.set_xticklabels([f'{b}°' for b in unsigned_bins], fontsize=9)
    ax.set_title('Unsigned (0°-180°): Same Bin\n(Both edge types combined!)', fontsize=11, fontweight='bold', color='green')
    
    plt.suptitle('HOG Uses Unsigned Gradients for Robustness', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_signed_vs_unsigned.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_signed_vs_unsigned.png")


if __name__ == "__main__":
    print("Generating HOG Gradient Visualizations...")
    print("-" * 50)
    create_gradient_computation_visual()
    create_edge_detection_visual()
    create_signed_vs_unsigned_visual()
    print("-" * 50)
    print("Done!")

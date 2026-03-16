"""
SURF Scale-Space Visualization
Shows how SURF uses filter pyramid instead of image pyramid
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
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
        return np.array(img).astype(np.float64) / 255.0
    else:
        img = np.zeros((480, 640), dtype=np.float64)
        img[100:200, 100:250] = 0.7
        img[250:350, 300:450] = 0.86
        Image.fromarray((img * 255).astype(np.uint8)).save(img_path)
        return img


def visualize_sift_vs_surf_pyramid():
    """Show difference between SIFT image pyramid and SURF filter pyramid"""
    
    img = load_image()
    H, W = img.shape
    
    fig = plt.figure(figsize=(20, 12))
    
    # SIFT Approach (Left side)
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f'SIFT: Full Image\n{W}×{H}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 4, 2)
    img_half = img[::2, ::2]
    ax2.imshow(img_half, cmap='gray')
    ax2.set_title(f'SIFT: Half Image\n{W//2}×{H//2}', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(2, 4, 3)
    img_quarter = img[::4, ::4]
    ax3.imshow(img_quarter, cmap='gray')
    ax3.set_title(f'SIFT: Quarter Image\n{W//4}×{H//4}', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.axis('off')
    sift_text = """
SIFT Scale-Space:
═══════════════════

• Build image pyramid
• Resize image at each octave
• Apply SAME filter to each

Problems:
• Memory: multiple image copies
• Interpolation artifacts
• Slower processing
"""
    ax4.text(0.1, 0.5, sift_text, fontsize=10, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
    ax4.set_title('SIFT Approach', fontsize=12, fontweight='bold', color='red')
    
    # SURF Approach (Bottom side)
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.imshow(img, cmap='gray')
    # Draw 9x9 filter box
    center_x, center_y = W//2, H//2
    rect = patches.Rectangle((center_x - 4.5, center_y - 4.5), 9, 9, 
                              linewidth=3, edgecolor='red', facecolor='none')
    ax5.add_patch(rect)
    ax5.set_title(f'SURF: Same Image\nFilter 9×9', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.imshow(img, cmap='gray')
    rect = patches.Rectangle((center_x - 7.5, center_y - 7.5), 15, 15, 
                              linewidth=3, edgecolor='green', facecolor='none')
    ax6.add_patch(rect)
    ax6.set_title(f'SURF: Same Image\nFilter 15×15', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.imshow(img, cmap='gray')
    rect = patches.Rectangle((center_x - 10.5, center_y - 10.5), 21, 21, 
                              linewidth=3, edgecolor='blue', facecolor='none')
    ax7.add_patch(rect)
    ax7.set_title(f'SURF: Same Image\nFilter 21×21', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    surf_text = """
SURF Scale-Space:
═══════════════════

• Keep image constant
• Resize FILTER instead
• Use integral image

Benefits:
• Single image in memory
• No interpolation needed
• O(1) per pixel!
"""
    ax8.text(0.1, 0.5, surf_text, fontsize=10, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))
    ax8.set_title('SURF Approach', fontsize=12, fontweight='bold', color='green')
    
    plt.suptitle('SIFT vs SURF: Scale-Space Construction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_vs_sift_scalespace.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: surf_vs_sift_scalespace.png")


def visualize_filter_pyramid():
    """Show SURF filter pyramid structure"""
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: Filter pyramid diagram
    ax1 = axes[0]
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.axis('off')
    
    # Octave 1
    filter_sizes_oct1 = [9, 15, 21, 27]
    colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']
    
    y_base = 75
    for i, (fs, color) in enumerate(zip(filter_sizes_oct1, colors)):
        width = fs * 1.5
        rect = patches.FancyBboxPatch((50 - width/2, y_base - i*18 - 8), width, 14,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax1.add_patch(rect)
        ax1.text(50, y_base - i*18, f'{fs}×{fs}', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
    
    ax1.text(50, 85, 'OCTAVE 1', ha='center', fontsize=14, fontweight='bold')
    ax1.text(95, 85, 'Filter sizes scale up:\n9 → 15 → 21 → 27\n(+6 each step)', 
            ha='right', fontsize=10, va='top')
    
    ax1.set_title('SURF Filter Pyramid\n(Same image, different filter sizes)', fontsize=14, fontweight='bold')
    
    # Right: Scale relationship
    ax2 = axes[1]
    ax2.axis('off')
    
    scale_info = """
SURF Scale-Space Structure
════════════════════════════════════

Filter Size to Scale (σ) Relationship:
  σ ≈ 1.2 × (filter_size / 9)

┌──────────────────────────────────────┐
│  Filter   │   σ (approx)  │  Step    │
├──────────────────────────────────────┤
│   9×9     │     1.2       │   base   │
│  15×15    │     2.0       │   +6     │
│  21×21    │     2.8       │   +6     │
│  27×27    │     3.6       │   +6     │
├──────────────────────────────────────┤
│  33×33    │     4.4       │  Oct 2   │
│  45×45    │     6.0       │   ...    │
│  etc.     │     ...       │          │
└──────────────────────────────────────┘

Why Filter Pyramid Works:
═════════════════════════
• Larger filter = detects larger blobs
• Equivalent to smaller image with same filter
• But NO image resizing needed!

Key Insight:
  21×21 filter on 640×480 image
  ≈ 9×9 filter on 274×206 image
  (But computed with O(1) using integral image!)
"""
    ax2.text(0.05, 0.5, scale_info, fontsize=10, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax2.set_title('Scale Relationship', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_filter_pyramid.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: surf_filter_pyramid.png")


def visualize_26_neighbor_3d():
    """3D visualization of 26-neighbor comparison"""
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['#3498db', '#e74c3c', '#27ae60']
    labels = ['Scale σ-1 (smaller filter)', 'Scale σ (current)', 'Scale σ+1 (larger filter)']
    
    # Draw 3x3x3 grid
    for z, (color, label) in enumerate(zip(colors, labels)):
        for y in range(3):
            for x in range(3):
                if z == 1 and x == 1 and y == 1:
                    # Center point (candidate)
                    ax.scatter([x], [y], [z], c='yellow', s=400, marker='*', 
                              edgecolors='black', linewidths=2, label='Candidate' if x==1 and y==1 and z==1 else '')
                else:
                    ax.scatter([x], [y], [z], c=color, s=100, alpha=0.7, marker='s')
    
    # Connect to show cube structure
    for z in [0, 2]:
        for i in range(3):
            ax.plot([i, i], [0, 2], [z, z], 'gray', alpha=0.3, linewidth=0.5)
            ax.plot([0, 2], [i, i], [z, z], 'gray', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Scale', fontsize=12)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_zticks([0, 1, 2])
    ax.set_zticklabels(['σ-1\n(9×9)', 'σ\n(15×15)', 'σ+1\n(21×21)'])
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498db', markersize=10, label='9 at σ-1'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#e74c3c', markersize=10, label='8 at σ'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#27ae60', markersize=10, label='9 at σ+1'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow', markersize=15, 
               markeredgecolor='black', label='Candidate'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_title('SURF 26-Neighbor Extrema Detection\n(Compare candidate to 26 neighbors in 3×3×3 cube)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_26_neighbor_3d.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: surf_26_neighbor_3d.png")


def visualize_scale_factor_real():
    """Show keypoints from different scales on real image"""
    
    img = load_image()
    H, W = img.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Generate sample keypoints at different scales
    np.random.seed(42)
    
    # Scale 1 (9×9 filter) - small blobs
    kp_scale1 = [(np.random.randint(20, W-20), np.random.randint(20, H-20)) for _ in range(80)]
    
    # Scale 2 (15×15 filter) - medium blobs
    kp_scale2 = [(np.random.randint(30, W-30), np.random.randint(30, H-30)) for _ in range(50)]
    
    # Scale 3 (21×21 filter) - large blobs
    kp_scale3 = [(np.random.randint(40, W-40), np.random.randint(40, H-40)) for _ in range(25)]
    
    # Plot each scale
    scales_data = [
        (kp_scale1, 'red', 4, 1.2, '9×9 → 15×15', 'Scale 1 (Fine Details)'),
        (kp_scale2, 'lime', 7, 1.5, '15×15 → 21×21', 'Scale 2 (Medium Features)'),
        (kp_scale3, 'cyan', 12, 2.0, '21×21 → 27×27', 'Scale 3 (Large Structures)')
    ]
    
    for idx, (kps, color, size, lw, filters, title) in enumerate(scales_data):
        ax = axes.flat[idx]
        ax.imshow(img, cmap='gray')
        for x, y in kps:
            circle = plt.Circle((x, y), size, color=color, fill=False, linewidth=lw, alpha=0.8)
            ax.add_patch(circle)
        ax.set_title(f'{title}\nFilters: {filters}\n{len(kps)} keypoints', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Combined view
    ax = axes[1, 1]
    ax.imshow(img, cmap='gray')
    for kps, color, size, lw, _, _ in scales_data:
        for x, y in kps:
            circle = plt.Circle((x, y), size, color=color, fill=False, linewidth=lw * 0.7, alpha=0.6)
            ax.add_patch(circle)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='red', 
               markersize=8, markeredgewidth=2, label=f'Scale 1: {len(kp_scale1)}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='lime', 
               markersize=11, markeredgewidth=2, label=f'Scale 2: {len(kp_scale2)}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='cyan', 
               markersize=14, markeredgewidth=2, label=f'Scale 3: {len(kp_scale3)}'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.set_title(f'All Scales Combined\nTotal: {len(kp_scale1) + len(kp_scale2) + len(kp_scale3)} keypoints', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('SURF Multi-Scale Blob Detection\n(Circle size indicates detection scale)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_scale_factor_real.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: surf_scale_factor_real.png")


if __name__ == "__main__":
    visualize_sift_vs_surf_pyramid()
    visualize_filter_pyramid()
    visualize_26_neighbor_3d()
    visualize_scale_factor_real()
    print("\nScale visualization images generated!")

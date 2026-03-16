"""
Generate detailed SURF Descriptor visualization images showing:
1. Keypoint with orientation
2. 20s × 20s region around keypoint
3. 4x4 subregions
4. Haar wavelet responses in each subregion
5. 4-value vectors per subregion [Σdx, Σdy, Σ|dx|, Σ|dy|]
6. Final 64-D descriptor
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
    raise FileNotFoundError("input_image.jpg not found")


def compute_integral_image(img):
    """Compute integral image."""
    return np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)


def box_sum(integral, x1, y1, x2, y2):
    """Compute sum of rectangular region using integral image."""
    h, w = integral.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    D = integral[y2, x2]
    B = integral[y1-1, x2] if y1 > 0 else 0
    C = integral[y2, x1-1] if x1 > 0 else 0
    A = integral[y1-1, x1-1] if y1 > 0 and x1 > 0 else 0
    return D - B - C + A


def compute_haar_response(integral, x, y, size):
    """Compute Haar wavelet response at a point."""
    half = size // 2
    h, w = integral.shape
    
    if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
        return 0, 0
    
    # Haar X: right - left
    left = box_sum(integral, x - half, y - half, x - 1, y + half)
    right = box_sum(integral, x, y - half, x + half, y + half)
    dx = right - left
    
    # Haar Y: bottom - top
    top = box_sum(integral, x - half, y - half, x + half, y - 1)
    bottom = box_sum(integral, x - half, y, x + half, y + half)
    dy = bottom - top
    
    return dx, dy


def plot_step1_keypoint_orientation(img, kp_x, kp_y, kp_orientation, scale=2):
    """Step 1: Show keypoint with its dominant orientation."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(img, cmap='gray')
    
    # Draw keypoint circle (radius = 6s in SURF)
    radius = 6 * scale
    circle = plt.Circle((kp_x, kp_y), radius, color='red', fill=False, linewidth=3)
    ax.add_patch(circle)
    
    # Draw orientation arrow
    arrow_len = radius * 1.5
    dx = arrow_len * np.cos(kp_orientation)
    dy = arrow_len * np.sin(kp_orientation)
    ax.arrow(kp_x, kp_y, dx, dy, head_width=8, head_length=5, fc='yellow', ec='yellow', linewidth=2)
    
    ax.plot(kp_x, kp_y, 'r+', markersize=15, markeredgewidth=3)
    
    ax.set_title(f'Step 1: Keypoint at ({kp_x}, {kp_y}) with Orientation {np.degrees(kp_orientation):.1f}°\n'
                 f'Scale s={scale}, Circular region radius = 6s = {radius}', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_desc_step1_keypoint.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: surf_desc_step1_keypoint.png')


def plot_step2_20s_region(img, kp_x, kp_y, kp_orientation, scale=2):
    """Step 2: Show 20s × 20s region around keypoint."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    region_size = 20 * scale
    half_size = region_size // 2
    
    # Left: Full image with region highlighted
    ax1 = axes[0]
    ax1.imshow(img, cmap='gray')
    
    rect = patches.Rectangle((kp_x - half_size, kp_y - half_size), region_size, region_size,
                              linewidth=3, edgecolor='cyan', facecolor='none')
    ax1.add_patch(rect)
    
    ax1.plot(kp_x, kp_y, 'r+', markersize=15, markeredgewidth=3)
    
    arrow_len = 25
    dx = arrow_len * np.cos(kp_orientation)
    dy = arrow_len * np.sin(kp_orientation)
    ax1.arrow(kp_x, kp_y, dx, dy, head_width=5, head_length=3, fc='yellow', ec='yellow', linewidth=2)
    
    ax1.set_title(f'20s × 20s Region Around Keypoint\n(s={scale}, region = {region_size}×{region_size})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlim(kp_x - 80, kp_x + 80)
    ax1.set_ylim(kp_y + 80, kp_y - 80)
    ax1.axis('off')
    
    # Right: Zoomed region with grid
    ax2 = axes[1]
    y_start = max(0, kp_y - half_size)
    y_end = min(img.shape[0], kp_y + half_size)
    x_start = max(0, kp_x - half_size)
    x_end = min(img.shape[1], kp_x + half_size)
    
    region = img[y_start:y_end, x_start:x_end]
    ax2.imshow(region, cmap='gray', extent=[0, 20, 20, 0])
    
    ax2.plot(10, 10, 'r+', markersize=20, markeredgewidth=3)
    ax2.set_title(f'Zoomed 20s×20s Region\n(Each unit = s = {scale} pixels)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (in units of s)')
    ax2.set_ylabel('Y (in units of s)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_desc_step2_20s_region.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: surf_desc_step2_20s_region.png')


def plot_step3_4x4_subregions(img, kp_x, kp_y, scale=2):
    """Step 3: Show 4x4 subregions (16 subregions total)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    region_size = 20 * scale
    half_size = region_size // 2
    subregion_size = 5 * scale
    
    # Left: Zoomed region with 4x4 grid
    ax1 = axes[0]
    y_start = max(0, kp_y - half_size)
    y_end = min(img.shape[0], kp_y + half_size)
    x_start = max(0, kp_x - half_size)
    x_end = min(img.shape[1], kp_x + half_size)
    
    region = img[y_start:y_end, x_start:x_end]
    ax1.imshow(region, cmap='gray', extent=[0, 4, 4, 0])
    
    colors = ['red', 'green', 'blue', 'orange']
    for i in range(4):
        for j in range(4):
            rect = patches.Rectangle((j, i), 1, 1, linewidth=2,
                                     edgecolor=colors[(i+j) % 4], facecolor='none')
            ax1.add_patch(rect)
            ax1.text(j + 0.5, i + 0.5, f'{i*4 + j}', fontsize=10, ha='center', va='center',
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=colors[(i+j) % 4], alpha=0.7))
    
    ax1.set_title(f'20s×20s Region Divided into 4×4 Grid\n(16 subregions, each 5s×5s = {subregion_size}×{subregion_size})',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Subregion column')
    ax1.set_ylabel('Subregion row')
    
    # Right: Schematic diagram
    ax2 = axes[1]
    ax2.set_xlim(0, 4)
    ax2.set_ylim(4, 0)
    
    for i in range(4):
        for j in range(4):
            rect = patches.Rectangle((j, i), 1, 1, linewidth=2,
                                     edgecolor='black', facecolor=colors[(i+j) % 4], alpha=0.3)
            ax2.add_patch(rect)
            ax2.text(j + 0.5, i + 0.5, f'Sub-\nregion\n{i*4 + j}', fontsize=9, ha='center', va='center',
                    fontweight='bold')
    
    ax2.set_title('4×4 Grid = 16 Subregions\nEach subregion = 5s × 5s', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.set_aspect('equal')
    ax2.grid(True, linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_desc_step3_4x4_subregions.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: surf_desc_step3_4x4_subregions.png')


def plot_step4_haar_wavelets(img, kp_x, kp_y, scale=2):
    """Step 4: Show Haar wavelet responses in subregions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Left: Haar X wavelet
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    rect1 = patches.Rectangle((2, 3), 2, 4, facecolor='red', edgecolor='black', linewidth=2)
    rect2 = patches.Rectangle((4, 3), 2, 4, facecolor='green', edgecolor='black', linewidth=2)
    ax1.add_patch(rect1)
    ax1.add_patch(rect2)
    ax1.text(3, 5, '-1', fontsize=14, fontweight='bold', color='white', ha='center', va='center')
    ax1.text(5, 5, '+1', fontsize=14, fontweight='bold', color='white', ha='center', va='center')
    ax1.text(5, 1.5, 'dx = right_sum - left_sum', fontsize=12, ha='center')
    ax1.set_title('Haar Wavelet X (dx)\nHorizontal Gradient', fontsize=12, fontweight='bold')
    ax1.axis('off')
    ax1.set_aspect('equal')
    
    # Middle: Haar Y wavelet
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    rect3 = patches.Rectangle((3, 5), 4, 2, facecolor='green', edgecolor='black', linewidth=2)
    rect4 = patches.Rectangle((3, 3), 4, 2, facecolor='red', edgecolor='black', linewidth=2)
    ax2.add_patch(rect3)
    ax2.add_patch(rect4)
    ax2.text(5, 6, '+1', fontsize=14, fontweight='bold', color='white', ha='center', va='center')
    ax2.text(5, 4, '-1', fontsize=14, fontweight='bold', color='white', ha='center', va='center')
    ax2.text(5, 1.5, 'dy = bottom_sum - top_sum', fontsize=12, ha='center')
    ax2.set_title('Haar Wavelet Y (dy)\nVertical Gradient', fontsize=12, fontweight='bold')
    ax2.axis('off')
    ax2.set_aspect('equal')
    
    # Right: Haar responses on image region
    ax3 = axes[2]
    integral = compute_integral_image(img)
    
    region_size = 20 * scale
    half_size = region_size // 2
    
    y_start = max(0, kp_y - half_size)
    y_end = min(img.shape[0], kp_y + half_size)
    x_start = max(0, kp_x - half_size)
    x_end = min(img.shape[1], kp_x + half_size)
    
    region = img[y_start:y_end, x_start:x_end]
    ax3.imshow(region, cmap='gray', extent=[0, region_size, region_size, 0])
    
    # Draw Haar response arrows
    haar_size = 2 * scale
    step = 4 * scale
    for py in range(step, region_size - step, step):
        for px in range(step, region_size - step, step):
            global_x = x_start + px
            global_y = y_start + py
            dx, dy = compute_haar_response(integral, global_x, global_y, haar_size)
            
            mag = np.sqrt(dx**2 + dy**2)
            if mag > 0:
                norm_dx = dx / mag * 8
                norm_dy = dy / mag * 8
                ax3.arrow(px, py, norm_dx, norm_dy, head_width=2, head_length=1,
                         fc='yellow', ec='yellow', linewidth=1)
    
    ax3.set_title('Haar Wavelet Responses in Region\n(Arrows show gradient direction)', 
                  fontsize=12, fontweight='bold')
    
    plt.suptitle('Step 4: Haar Wavelet Responses', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_desc_step4_haar_wavelets.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: surf_desc_step4_haar_wavelets.png')


def plot_step5_4value_vector():
    """Step 5: Show 4-value vector per subregion."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: 4-value vector explanation
    ax1 = axes[0]
    ax1.axis('off')
    
    explanation = """
SURF Subregion Descriptor (4 values)
═══════════════════════════════════════

For each of 16 subregions:

  Sample 5×5 = 25 points
  For each point, compute Haar (dx, dy)
  
  Rotate by keypoint orientation:
    dx' = dx·cos(θ) + dy·sin(θ)
    dy' = -dx·sin(θ) + dy·cos(θ)
  
  Weight by Gaussian centered at keypoint
  
  Sum to create 4 values:
  
    ┌────────────────────────────────┐
    │  Σdx'   = sum of horizontal   │
    │  Σdy'   = sum of vertical     │
    │  Σ|dx'| = sum of abs horiz    │
    │  Σ|dy'| = sum of abs vert     │
    └────────────────────────────────┘
  
  Σdx', Σdy' → DIRECTION information
  Σ|dx'|, Σ|dy'| → MAGNITUDE information
  
  Total per subregion: 4 values
  Total descriptor: 16 × 4 = 64 values
"""
    ax1.text(0.05, 0.5, explanation, fontsize=11, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax1.set_title('4-Value Subregion Descriptor', fontsize=14, fontweight='bold')
    
    # Right: Example histogram
    ax2 = axes[1]
    
    # Sample values for one subregion
    values = [15.2, -8.5, 22.1, 18.3]  # [Σdx', Σdy', Σ|dx'|, Σ|dy'|]
    labels = ['Σdx\'', 'Σdy\'', 'Σ|dx\'|', 'Σ|dy\'|']
    colors = ['blue', 'green', 'red', 'orange']
    
    bars = ax2.bar(labels, values, color=colors, edgecolor='black', linewidth=2)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Example: 4-Value Vector for One Subregion\n(Blue/Green = direction, Red/Orange = magnitude)',
                  fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5 if height >= 0 else height - 1.5,
                f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_desc_step5_4value_vector.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: surf_desc_step5_4value_vector.png')


def plot_step6_64d_descriptor():
    """Step 6: Show final 64-D descriptor."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Top: 16 subregions × 4 values = 64 values heatmap
    ax1 = axes[0]
    
    np.random.seed(42)
    descriptor = np.random.randn(16, 4) * 0.3
    
    im = ax1.imshow(descriptor.T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    
    ax1.set_xticks(range(16))
    ax1.set_xticklabels([f'S{i}' for i in range(16)], fontsize=9)
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(['Σdx\'', 'Σdy\'', 'Σ|dx\'|', 'Σ|dy\'|'])
    ax1.set_xlabel('16 Subregions', fontsize=12)
    ax1.set_ylabel('4 Values per Subregion', fontsize=12)
    ax1.set_title('64-D Descriptor: 16 Subregions × 4 Values = 64 Values\n(Heatmap showing Haar wavelet sums)',
                  fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Normalized Value')
    
    # Bottom: Flattened 64-D vector
    ax2 = axes[1]
    
    descriptor_flat = descriptor.flatten()
    descriptor_flat = descriptor_flat / (np.linalg.norm(descriptor_flat) + 1e-8)
    colors = plt.cm.tab20(np.repeat(np.arange(16), 4) / 16)
    
    bars = ax2.bar(range(64), descriptor_flat, color=colors, edgecolor='none', width=1)
    
    for i in range(1, 16):
        ax2.axvline(x=i*4 - 0.5, color='black', linewidth=1, linestyle='--', alpha=0.5)
    
    for i in range(16):
        ax2.text(i*4 + 2, max(descriptor_flat) + 0.05, f'S{i}', ha='center', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Descriptor Index (0-63)', fontsize=12)
    ax2.set_ylabel('Normalized Value', fontsize=12)
    ax2.set_title('Flattened 64-D Descriptor Vector\n(Each color = one subregion\'s 4 values)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlim(-1, 64)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_desc_step6_64d_descriptor.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: surf_desc_step6_64d_descriptor.png')


def plot_complete_descriptor_pipeline():
    """Create a complete pipeline visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Step 1: Keypoint
    ax = axes[0, 0]
    ax.text(0.5, 0.7, '●', fontsize=80, ha='center', va='center', color='red')
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(0.5 + 0.2*np.cos(theta), 0.7 + 0.15*np.sin(theta), 'r-', linewidth=2)
    ax.arrow(0.5, 0.7, 0.15, -0.08, head_width=0.05, head_length=0.03, fc='yellow', ec='yellow', linewidth=3)
    ax.text(0.5, 0.3, 'Step 1:\nKeypoint with\nOrientation θ', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Step 2: 20s region
    ax = axes[0, 1]
    rect = patches.Rectangle((0.2, 0.3), 0.6, 0.5, linewidth=3, edgecolor='cyan', facecolor='lightgray')
    ax.add_patch(rect)
    ax.text(0.5, 0.55, '20s×20s\nRegion', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(0.5, 0.15, 'Step 2:\nExtract 20s×20s\nRotated Region', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Step 3: 4x4 subregions
    ax = axes[0, 2]
    for i in range(4):
        for j in range(4):
            rect = patches.Rectangle((0.2 + j*0.15, 0.3 + i*0.125), 0.14, 0.12,
                                     linewidth=1, edgecolor='black', facecolor=f'C{(i+j)%4}', alpha=0.5)
            ax.add_patch(rect)
    ax.text(0.5, 0.15, 'Step 3:\nDivide into\n4×4 = 16 Subregions', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Step 4: Haar wavelets
    ax = axes[1, 0]
    for i in range(5):
        for j in range(5):
            angle = np.random.rand() * 2 * np.pi
            ax.arrow(0.15 + j*0.15, 0.35 + i*0.1, 0.05*np.cos(angle),
                    0.05*np.sin(angle), head_width=0.02, fc='red', ec='red')
    ax.text(0.5, 0.15, 'Step 4:\nCompute Haar\nWavelets (dx, dy)', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Step 5: 4-value vector
    ax = axes[1, 1]
    hist = np.random.rand(4) * 0.3 + 0.3
    ax.bar([0.2, 0.4, 0.6, 0.8], hist, width=0.12, color=['blue', 'green', 'red', 'orange'], edgecolor='black')
    ax.text(0.5, 0.8, '[Σdx,Σdy,Σ|dx|,Σ|dy|]', fontsize=10, ha='center', fontweight='bold')
    ax.text(0.5, 0.15, 'Step 5:\n4 Values per\nSubregion', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Step 6: 64-D
    ax = axes[1, 2]
    ax.bar(np.linspace(0.1, 0.9, 32), np.random.rand(32) * 0.3 + 0.35, width=0.02, color='purple', edgecolor='none')
    ax.text(0.5, 0.75, '16 × 4 = 64', fontsize=16, ha='center', va='center', fontweight='bold', color='purple')
    ax.text(0.5, 0.15, 'Step 6:\n64-D Descriptor\nVector', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.suptitle('SURF Descriptor Pipeline: 6 Steps to Create 64-D Feature Vector',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_desc_pipeline.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: surf_desc_pipeline.png')


def main():
    print("Loading image...")
    img = load_image()
    print(f"Image size: {img.shape}")
    
    kp_x, kp_y = 320, 240
    kp_orientation = np.radians(45)
    scale = 2
    
    print("\n=== Generating SURF Descriptor Visualization Images ===\n")
    
    print("Step 1: Keypoint with orientation...")
    plot_step1_keypoint_orientation(img, kp_x, kp_y, kp_orientation, scale)
    
    print("Step 2: 20s×20s region...")
    plot_step2_20s_region(img, kp_x, kp_y, kp_orientation, scale)
    
    print("Step 3: 4x4 subregions...")
    plot_step3_4x4_subregions(img, kp_x, kp_y, scale)
    
    print("Step 4: Haar wavelet responses...")
    plot_step4_haar_wavelets(img, kp_x, kp_y, scale)
    
    print("Step 5: 4-value vector...")
    plot_step5_4value_vector()
    
    print("Step 6: 64-D descriptor...")
    plot_step6_64d_descriptor()
    
    print("Complete pipeline diagram...")
    plot_complete_descriptor_pipeline()
    
    print("\n=== All images generated! ===")


if __name__ == '__main__':
    main()

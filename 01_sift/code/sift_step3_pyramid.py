"""
SIFT Step 3: Complete Multi-Octave Pyramid Visualization

Shows the full process:
H,W      → σ₁ → σ₂ → σ₃ → DoG → 26-neighbor → keypoints
   ↓
H/2,W/2  → σ₁ → σ₂ → σ₃ → DoG → 26-neighbor → keypoints
   ↓
H/4,W/4  → σ₁ → σ₂ → σ₃ → DoG → 26-neighbor → keypoints
   ↓
COMBINE ALL KEYPOINTS
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy import ndimage

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def gaussian_kernel(sigma, size=None):
    if size is None:
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1
    center = size // 2
    kernel = np.zeros((size, size))
    for y in range(size):
        for x in range(size):
            dx, dy = x - center, y - center
            kernel[y, x] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(img, sigma):
    kernel = gaussian_kernel(sigma)
    return ndimage.convolve(img, kernel, mode='reflect')


def detect_extrema(dog_below, dog_current, dog_above, threshold=0.01):
    """Detect local extrema using 26-neighbor comparison."""
    h, w = dog_current.shape
    keypoints = []
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            val = dog_current[y, x]
            if abs(val) < threshold:
                continue
            
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    neighbors.append(dog_below[y+di, x+dj])
                    neighbors.append(dog_above[y+di, x+dj])
                    if not (di == 0 and dj == 0):
                        neighbors.append(dog_current[y+di, x+dj])
            
            is_max = all(val > n for n in neighbors)
            is_min = all(val < n for n in neighbors)
            
            if is_max or is_min:
                keypoints.append((x, y, 'max' if is_max else 'min'))
    
    return keypoints


def process_octave(gray, sigma_base=1.6):
    """Process one octave: Gaussian → DoG → Keypoints."""
    k = 2 ** (1/3)
    sigmas = [sigma_base, sigma_base * k, sigma_base * k**2, sigma_base * k**3]
    gaussians = [gaussian_blur(gray, s) for s in sigmas]
    dogs = [gaussians[i+1] - gaussians[i] for i in range(len(gaussians)-1)]
    keypoints = detect_extrema(dogs[0], dogs[1], dogs[2])
    return gaussians, dogs, keypoints, sigmas


# =============================================================================
# STEP 3.1: Three Scales Visualization
# =============================================================================
def visualize_step3_1(dogs):
    """Show the three DoG scales."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['DoG Scale σ₁ (below)', 'DoG Scale σ₂ (current)', 'DoG Scale σ₃ (above)']
    
    for i, (ax, title) in enumerate(zip(axes, titles)):
        dog = dogs[i] if i < len(dogs) else dogs[-1]
        vmax = np.percentile(np.abs(dog), 95)
        im = ax.imshow(dog, cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    fig.suptitle('SIFT Step 3.1: Three DoG Scales for 26-Neighbor Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sift_step3_1_three_scales.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: sift_step3_1_three_scales.png")


# =============================================================================
# STEP 3.2: 26 Neighbors Concept
# =============================================================================
def visualize_step3_2():
    """Visualize 26-neighbor concept."""
    fig = plt.figure(figsize=(16, 6))
    
    # 2D view
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim(-0.5, 11.5)
    ax1.set_ylim(-0.5, 9.5)
    
    colors = ['#3498db', '#e74c3c', '#f39c12']
    labels = ['Scale σ-1\n(9 neighbors)', 'Scale σ\n(8 neighbors)', 'Scale σ+1\n(9 neighbors)']
    
    for scale_idx, (x_offset, color, label) in enumerate(zip([0, 4, 8], colors, labels)):
        for i in range(3):
            for j in range(3):
                x, y = x_offset + j, 6 - i
                if scale_idx == 1 and i == 1 and j == 1:
                    circle = Circle((x + 0.5, y + 0.5), 0.4, color='yellow', ec='black', linewidth=3)
                    ax1.add_patch(circle)
                    ax1.text(x + 0.5, y + 0.5, '?', ha='center', va='center', fontsize=16, fontweight='bold')
                else:
                    rect = Rectangle((x, y), 1, 1, linewidth=2, edgecolor='black', facecolor=color, alpha=0.6)
                    ax1.add_patch(rect)
        ax1.text(x_offset + 1.5, 3.5, label, ha='center', fontsize=10, fontweight='bold')
    
    ax1.set_title('26 Neighbors in Scale-Space\n(Yellow = pixel being tested)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    ax1.set_aspect('equal')
    
    # 3D view
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    for scale_idx, (z, color) in enumerate(zip([0, 1, 2], colors)):
        for i in range(3):
            for j in range(3):
                if scale_idx == 1 and i == 1 and j == 1:
                    ax2.scatter([j], [i], [z], c='yellow', s=300, edgecolors='black', linewidths=2, marker='o')
                else:
                    ax2.scatter([j], [i], [z], c=color, s=100, alpha=0.7, marker='s')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Scale')
    ax2.set_zticks([0, 1, 2])
    ax2.set_zticklabels(['σ-1', 'σ', 'σ+1'])
    ax2.set_title('3D View of Scale-Space', fontsize=12, fontweight='bold')
    
    fig.suptitle('SIFT Step 3.2: Understanding the 26-Neighbor Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sift_step3_2_26_neighbors.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: sift_step3_2_26_neighbors.png")


# =============================================================================
# STEP 3.3-3.5: Each Octave
# =============================================================================
def visualize_octave(gray, octave_num, scale_factor, sigma_base=1.6):
    """Visualize one octave completely."""
    h, w = gray.shape
    gaussians, dogs, keypoints, sigmas = process_octave(gray, sigma_base)
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.25)
    
    # Row 0: Title
    ax_title = fig.add_subplot(gs[0, 0])
    res_text = f'{h}×{w}'
    if scale_factor == 1:
        label = f'OCTAVE {octave_num}\n\nResolution:\n{res_text}\n(Full Size)'
        color = 'lightblue'
    elif scale_factor == 2:
        label = f'OCTAVE {octave_num}\n\nResolution:\n{res_text}\n(H/2 × W/2)'
        color = 'lightgreen'
    else:
        label = f'OCTAVE {octave_num}\n\nResolution:\n{res_text}\n(H/4 × W/4)'
        color = 'lightyellow'
    
    ax_title.text(0.5, 0.5, label, ha='center', va='center', fontsize=12, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor=color, edgecolor='black'))
    ax_title.axis('off')
    
    # Row 0: Gaussian images
    for i, (g, sigma) in enumerate(zip(gaussians, sigmas)):
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(g, cmap='gray')
        ax.set_title(f'G(σ={sigma:.2f})', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Row 1: DoG images
    ax_arrow = fig.add_subplot(gs[1, 0])
    ax_arrow.text(0.5, 0.5, 'Subtract\nadjacent\nGaussians\n↓\nDoG', ha='center', va='center', fontsize=11)
    ax_arrow.axis('off')
    
    for i, dog in enumerate(dogs):
        ax = fig.add_subplot(gs[1, i+1])
        vmax = np.percentile(np.abs(dog), 95)
        ax.imshow(dog, cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax.set_title(f'DoG {i+1}\n(σ{i+2} - σ{i+1})', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 4])
    ax.axis('off')
    
    # Row 2: 26-neighbor concept
    ax_info = fig.add_subplot(gs[2, 0])
    ax_info.text(0.5, 0.5, '26-Neighbor\nComparison\n↓\nFind Extrema', ha='center', va='center', fontsize=11)
    ax_info.axis('off')
    
    # 26-neighbor diagram
    ax_26 = fig.add_subplot(gs[2, 1:3])
    ax_26.set_xlim(0, 10)
    ax_26.set_ylim(0, 4)
    colors = ['#3498db', '#e74c3c', '#f39c12']
    for scale_idx, (x_off, clr) in enumerate(zip([0.5, 3.5, 6.5], colors)):
        for i in range(3):
            for j in range(3):
                x, y = x_off + j * 0.7, 2.5 - i * 0.7
                if scale_idx == 1 and i == 1 and j == 1:
                    circle = Circle((x, y), 0.25, color='yellow', ec='black', linewidth=2)
                    ax_26.add_patch(circle)
                else:
                    rect = Rectangle((x-0.25, y-0.25), 0.5, 0.5, fc=clr, ec='black', alpha=0.6)
                    ax_26.add_patch(rect)
    ax_26.text(5, 3.5, 'Compare center to all 26 neighbors', ha='center', fontsize=10, fontweight='bold')
    ax_26.axis('off')
    ax_26.set_aspect('equal')
    
    # Detection result on DoG
    ax_dog = fig.add_subplot(gs[2, 3:5])
    vmax = np.percentile(np.abs(dogs[1]), 95)
    ax_dog.imshow(dogs[1], cmap='RdBu', vmin=-vmax, vmax=vmax)
    if keypoints:
        xs = [kp[0] for kp in keypoints]
        ys = [kp[1] for kp in keypoints]
        ax_dog.scatter(xs, ys, c='lime', s=30, edgecolors='black', linewidths=0.5)
    ax_dog.set_title(f'DoG 2 with {len(keypoints)} extrema', fontsize=10, fontweight='bold')
    ax_dog.axis('off')
    
    # Row 3: Final keypoints on image
    ax_result = fig.add_subplot(gs[3, :])
    ax_result.imshow(gray, cmap='gray')
    
    maxima = [(x, y) for x, y, t in keypoints if t == 'max']
    minima = [(x, y) for x, y, t in keypoints if t == 'min']
    
    if maxima:
        xs, ys = zip(*maxima)
        ax_result.scatter(xs, ys, c='red', s=40, alpha=0.8, label=f'Maxima ({len(maxima)})', edgecolors='white', linewidths=0.5)
    if minima:
        xs, ys = zip(*minima)
        ax_result.scatter(xs, ys, c='blue', s=40, alpha=0.8, label=f'Minima ({len(minima)})', edgecolors='white', linewidths=0.5)
    
    ax_result.legend(loc='upper right', fontsize=10)
    ax_result.set_title(f'Detected Keypoints: {len(keypoints)} total (Red=Max, Blue=Min)', fontsize=12, fontweight='bold')
    ax_result.axis('off')
    
    fig.suptitle(f'SIFT Step 3.{octave_num+3}: Octave {octave_num} - Complete Process', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(OUT_DIR, f'sift_step3_{octave_num+3}_octave{octave_num}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: sift_step3_{octave_num+3}_octave{octave_num}.png")
    
    return keypoints


# =============================================================================
# STEP 3.6: Pyramid Structure
# =============================================================================
def visualize_pyramid_structure(gray):
    """Show complete pyramid structure."""
    h, w = gray.shape
    
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 6, figure=fig, hspace=0.4, wspace=0.3)
    
    # Title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.text(0.5, 0.5, 
        'SIFT Step 3.6: Complete Scale-Space Pyramid Structure\n\n'
        'Each octave: Gaussian blur → DoG subtraction → 26-neighbor detection → Keypoints',
        ha='center', va='center', fontsize=14, fontweight='bold')
    ax_title.axis('off')
    
    sigma_base = 1.6
    k = 2 ** (1/3)
    
    # Octave 0
    ax0 = fig.add_subplot(gs[1, 0])
    ax0.text(0.5, 0.5, f'OCTAVE 0\n{h}×{w}', ha='center', va='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax0.axis('off')
    
    for i in range(4):
        ax = fig.add_subplot(gs[1, i+1])
        sigma = sigma_base * (k ** i)
        g = gaussian_blur(gray, sigma)
        ax.imshow(g[::4, ::4], cmap='gray')
        ax.set_title(f'σ={sigma:.1f}', fontsize=9)
        ax.axis('off')
    
    ax_arr0 = fig.add_subplot(gs[1, 5])
    ax_arr0.text(0.5, 0.5, '→DoG→\n26-nbr\n→KP', ha='center', va='center', fontsize=9)
    ax_arr0.axis('off')
    
    # Octave 1
    gray1 = gray[::2, ::2]
    h1, w1 = gray1.shape
    
    ax1 = fig.add_subplot(gs[2, 0])
    ax1.text(0.5, 0.5, f'OCTAVE 1\n{h1}×{w1}\n(↓2)', ha='center', va='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax1.axis('off')
    
    for i in range(4):
        ax = fig.add_subplot(gs[2, i+1])
        sigma = sigma_base * (k ** i)
        g = gaussian_blur(gray1, sigma)
        ax.imshow(g[::2, ::2], cmap='gray')
        ax.set_title(f'σ={sigma:.1f}', fontsize=9)
        ax.axis('off')
    
    ax_arr1 = fig.add_subplot(gs[2, 5])
    ax_arr1.text(0.5, 0.5, '→DoG→\n26-nbr\n→KP', ha='center', va='center', fontsize=9)
    ax_arr1.axis('off')
    
    # Octave 2
    gray2 = gray[::4, ::4]
    h2, w2 = gray2.shape
    
    ax2 = fig.add_subplot(gs[3, 0])
    ax2.text(0.5, 0.5, f'OCTAVE 2\n{h2}×{w2}\n(↓4)', ha='center', va='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax2.axis('off')
    
    for i in range(4):
        ax = fig.add_subplot(gs[3, i+1])
        sigma = sigma_base * (k ** i)
        g = gaussian_blur(gray2, sigma)
        ax.imshow(g, cmap='gray')
        ax.set_title(f'σ={sigma:.1f}', fontsize=9)
        ax.axis('off')
    
    ax_arr2 = fig.add_subplot(gs[3, 5])
    ax_arr2.text(0.5, 0.5, '→DoG→\n26-nbr\n→KP', ha='center', va='center', fontsize=9)
    ax_arr2.axis('off')
    
    plt.savefig(os.path.join(OUT_DIR, 'sift_step3_6_pyramid_structure.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: sift_step3_6_pyramid_structure.png")


# =============================================================================
# STEP 3.7: Combined Keypoints
# =============================================================================
def visualize_combined(gray, kp0, kp1, kp2):
    """Show all keypoints combined."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    titles = ['Octave 0 (H×W)', 'Octave 1 (H/2×W/2)', 'Octave 2 (H/4×W/4)']
    keypoints_list = [kp0, kp1, kp2]
    colors = ['red', 'green', 'orange']
    
    for idx, (ax, kps, title, color) in enumerate(zip(axes.flat[:3], keypoints_list, titles, colors)):
        ax.imshow(gray, cmap='gray')
        if kps:
            xs = [x for x, y, t in kps]
            ys = [y for x, y, t in kps]
            ax.scatter(xs, ys, c=color, s=40, alpha=0.7, edgecolors='white', linewidths=0.5)
        ax.set_title(f'{title}\n{len(kps)} keypoints', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Combined
    ax = axes[1, 1]
    ax.imshow(gray, cmap='gray')
    
    total = 0
    for kps, color, label in zip(keypoints_list, colors, ['Oct 0', 'Oct 1', 'Oct 2']):
        if kps:
            xs = [x for x, y, t in kps]
            ys = [y for x, y, t in kps]
            ax.scatter(xs, ys, c=color, s=30, alpha=0.6, edgecolors='white', linewidths=0.3, 
                      label=f'{label} ({len(kps)})')
            total += len(kps)
    
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(f'ALL KEYPOINTS COMBINED\nTotal: {total}', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    fig.suptitle('SIFT Step 3.7: Keypoints from All Octaves Combined', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sift_step3_7_combined.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: sift_step3_7_combined.png (total: {total})")
    
    return total


# =============================================================================
# STEP 3.8: Final with Scale Circles
# =============================================================================
def visualize_final_scales(gray, kp0, kp1, kp2):
    """Final keypoints with scale indication."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(gray, cmap='gray')
    
    # Octave 0: small circles
    for x, y, t in kp0[:200]:
        color = 'red' if t == 'max' else 'blue'
        circle = Circle((x, y), 3, fill=False, edgecolor=color, linewidth=1, alpha=0.7)
        ax.add_patch(circle)
    
    # Octave 1: medium circles
    for x, y, t in kp1[:150]:
        color = 'lime' if t == 'max' else 'cyan'
        circle = Circle((x, y), 6, fill=False, edgecolor=color, linewidth=1.5, alpha=0.7)
        ax.add_patch(circle)
    
    # Octave 2: large circles
    for x, y, t in kp2[:100]:
        color = 'orange' if t == 'max' else 'purple'
        circle = Circle((x, y), 12, fill=False, edgecolor=color, linewidth=2, alpha=0.7)
        ax.add_patch(circle)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='red', 
               markersize=6, label='Octave 0 (fine)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='lime', 
               markersize=10, label='Octave 1 (medium)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='orange', 
               markersize=14, label='Octave 2 (coarse)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    total = len(kp0) + len(kp1) + len(kp2)
    ax.set_title(f'SIFT Step 3.8: Final Keypoints with Scale Indication\nCircle size = detection scale (Total: {total})',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sift_step3_8_final_scales.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: sift_step3_8_final_scales.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("SIFT Step 3: Complete Multi-Octave Pyramid Visualization")
    print("=" * 70)
    
    # Load or create image
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    if not os.path.exists(image_path):
        print("Creating test image...")
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img[:, :] = [100, 100, 100]
        img[40:90, 40:90] = [255, 255, 255]
        img[120:180, 50:150] = [200, 200, 200]
        img[50:100, 200:280] = [180, 180, 180]
        img[150:200, 250:350] = [220, 220, 220]
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    img[200 + i*20:200 + (i+1)*20, 50 + j*20:50 + (j+1)*20] = [240, 240, 240]
        Image.fromarray(img).save(image_path)
    
    img = np.array(Image.open(image_path))
    if len(img.shape) == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float64)
    gray = gray / 255.0
    
    h, w = gray.shape
    print(f"\nInput image: {h}×{w}")
    print("\nGenerating visualizations...\n")
    
    # Step 3.1: Three scales
    _, dogs, _, _ = process_octave(gray)
    visualize_step3_1(dogs)
    
    # Step 3.2: 26 neighbors
    visualize_step3_2()
    
    # Step 3.3: Octave 0 (full resolution)
    kp0 = visualize_octave(gray, 0, 1)
    
    # Step 3.4: Octave 1 (half resolution)
    gray1 = gray[::2, ::2]
    kp1_local = visualize_octave(gray1, 1, 2)
    kp1 = [(x*2, y*2, t) for x, y, t in kp1_local]  # Scale back
    
    # Step 3.5: Octave 2 (quarter resolution)
    gray2 = gray[::4, ::4]
    kp2_local = visualize_octave(gray2, 2, 4)
    kp2 = [(x*4, y*4, t) for x, y, t in kp2_local]  # Scale back
    
    # Step 3.6: Pyramid structure
    visualize_pyramid_structure(gray)
    
    # Step 3.7: Combined keypoints
    visualize_combined(gray, kp0, kp1, kp2)
    
    # Step 3.8: Final with scale circles
    visualize_final_scales(gray, kp0, kp1, kp2)
    
    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)
    print(f"\nTotal keypoints: {len(kp0) + len(kp1) + len(kp2)}")
    print(f"  - Octave 0: {len(kp0)}")
    print(f"  - Octave 1: {len(kp1)}")
    print(f"  - Octave 2: {len(kp2)}")


if __name__ == "__main__":
    main()

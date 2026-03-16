"""
SIFT Keypoint Filtering with Real Image
Shows how keypoints are reduced from initial detection to final count
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from scipy import ndimage
import urllib.request

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def download_image():
    """Download a real test image"""
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    if not os.path.exists(image_path):
        print("Downloading test image...")
        url = "https://picsum.photos/640/480"
        urllib.request.urlretrieve(url, image_path)
    return image_path


def gaussian_kernel(sigma):
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


def build_gaussian_pyramid(img, num_octaves=4, num_scales=5, sigma=1.6):
    k = 2 ** (1.0 / (num_scales - 3))
    pyramid = []
    current_img = img.copy()
    
    for octave in range(num_octaves):
        octave_images = []
        for scale in range(num_scales):
            if scale == 0:
                blurred = gaussian_blur(current_img, sigma)
            else:
                sigma_total = sigma * (k ** scale)
                sigma_prev = sigma * (k ** (scale - 1))
                sigma_blur = np.sqrt(max(0.01, sigma_total**2 - sigma_prev**2))
                blurred = gaussian_blur(octave_images[-1], sigma_blur)
            octave_images.append(blurred)
        pyramid.append(octave_images)
        if octave < num_octaves - 1:
            base_idx = min(num_scales - 3, len(octave_images) - 1)
            current_img = octave_images[base_idx][::2, ::2]
    
    return pyramid


def compute_dog_pyramid(gaussian_pyramid):
    dog_pyramid = []
    for octave_images in gaussian_pyramid:
        dog_octave = [octave_images[i + 1] - octave_images[i] for i in range(len(octave_images) - 1)]
        dog_pyramid.append(dog_octave)
    return dog_pyramid


def detect_extrema_all(dog_pyramid):
    """Stage 1: Detect ALL 26-neighbor extrema (no threshold)"""
    keypoints = []
    
    for octave_idx, dog_octave in enumerate(dog_pyramid):
        for scale_idx in range(1, len(dog_octave) - 1):
            prev_dog = dog_octave[scale_idx - 1]
            curr_dog = dog_octave[scale_idx]
            next_dog = dog_octave[scale_idx + 1]
            h, w = curr_dog.shape
            
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    val = curr_dog[y, x]
                    
                    # Get all 26 neighbors
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                neighbors.extend([prev_dog[y, x], next_dog[y, x]])
                            else:
                                neighbors.extend([
                                    curr_dog[y + dy, x + dx],
                                    prev_dog[y + dy, x + dx],
                                    next_dog[y + dy, x + dx]
                                ])
                    
                    # Check if extremum
                    is_max = all(val > n for n in neighbors)
                    is_min = all(val < n for n in neighbors)
                    
                    if is_max or is_min:
                        scale_factor = 2 ** octave_idx
                        keypoints.append({
                            'x': x * scale_factor,
                            'y': y * scale_factor,
                            'octave': octave_idx,
                            'scale': scale_idx,
                            'response': val,
                            'abs_response': abs(val),
                            'local_x': x,
                            'local_y': y,
                            'type': 'max' if is_max else 'min'
                        })
    
    return keypoints


def filter_low_contrast(keypoints, threshold=0.03):
    """Stage 2: Remove low contrast keypoints"""
    filtered = []
    removed = []
    for kp in keypoints:
        if kp['abs_response'] >= threshold:
            filtered.append(kp)
        else:
            removed.append(kp)
    return filtered, removed


def filter_edge_response(keypoints, dog_pyramid, edge_threshold=10.0):
    """Stage 3: Remove edge responses using Hessian"""
    filtered = []
    removed = []
    
    for kp in keypoints:
        dog = dog_pyramid[kp['octave']][kp['scale']]
        x, y = kp['local_x'], kp['local_y']
        h, w = dog.shape
        
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            removed.append(kp)
            continue
        
        # Hessian matrix elements
        dxx = dog[y, x + 1] + dog[y, x - 1] - 2 * dog[y, x]
        dyy = dog[y + 1, x] + dog[y - 1, x] - 2 * dog[y, x]
        dxy = (dog[y + 1, x + 1] - dog[y + 1, x - 1] - dog[y - 1, x + 1] + dog[y - 1, x - 1]) / 4
        
        det_h = dxx * dyy - dxy ** 2
        trace_h = dxx + dyy
        
        # Edge test: Tr(H)^2 / Det(H) < (r+1)^2 / r
        r = edge_threshold
        threshold_ratio = (r + 1) ** 2 / r
        
        if det_h <= 0:
            removed.append(kp)
        elif trace_h ** 2 / det_h > threshold_ratio:
            removed.append(kp)
        else:
            filtered.append(kp)
    
    return filtered, removed


def filter_subpixel_refinement(keypoints, dog_pyramid, offset_threshold=0.5):
    """Stage 4: Sub-pixel refinement (discard if offset too large)"""
    filtered = []
    removed = []
    
    for kp in keypoints:
        dog = dog_pyramid[kp['octave']][kp['scale']]
        x, y = kp['local_x'], kp['local_y']
        h, w = dog.shape
        
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            removed.append(kp)
            continue
        
        # First derivatives
        dx = (dog[y, x + 1] - dog[y, x - 1]) / 2
        dy = (dog[y + 1, x] - dog[y - 1, x]) / 2
        
        # Second derivatives
        dxx = dog[y, x + 1] + dog[y, x - 1] - 2 * dog[y, x]
        dyy = dog[y + 1, x] + dog[y - 1, x] - 2 * dog[y, x]
        dxy = (dog[y + 1, x + 1] - dog[y + 1, x - 1] - dog[y - 1, x + 1] + dog[y - 1, x - 1]) / 4
        
        # Hessian matrix
        H = np.array([[dxx, dxy], [dxy, dyy]])
        grad = np.array([dx, dy])
        
        # Solve for offset
        try:
            det = dxx * dyy - dxy * dxy
            if abs(det) < 1e-10:
                removed.append(kp)
                continue
            
            offset_x = -(dyy * dx - dxy * dy) / det
            offset_y = -(dxx * dy - dxy * dx) / det
            
            # Check if offset is within bounds
            if abs(offset_x) > offset_threshold or abs(offset_y) > offset_threshold:
                removed.append(kp)
            else:
                # Refine position
                kp_refined = kp.copy()
                scale_factor = 2 ** kp['octave']
                kp_refined['x'] = (x + offset_x) * scale_factor
                kp_refined['y'] = (y + offset_y) * scale_factor
                filtered.append(kp_refined)
        except:
            removed.append(kp)
    
    return filtered, removed


def visualize_filtering_stages(gray, stages_data):
    """Create visualization showing keypoints at each filtering stage"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    stage_names = [
        ('Stage 1: Initial Extrema Detection', '#3498db'),
        ('Stage 2: After Low Contrast Removal', '#e74c3c'),
        ('Stage 3: After Edge Response Removal', '#f39c12'),
        ('Stage 4: After Sub-pixel Refinement (FINAL)', '#27ae60')
    ]
    
    for idx, (ax, (title, color), data) in enumerate(zip(axes.flat, stage_names, stages_data)):
        keypoints, count, removed_count = data
        
        ax.imshow(gray, cmap='gray')
        
        # Draw keypoints
        for kp in keypoints[:500]:  # Limit for visibility
            circle = plt.Circle((kp['x'], kp['y']), 4, color=color, fill=False, linewidth=1.2)
            ax.add_patch(circle)
        
        ax.set_title(f"{title}\n{count} keypoints", fontsize=12, fontweight='bold', color=color)
        
        # Add removed count annotation
        if removed_count > 0:
            ax.text(0.98, 0.02, f"−{removed_count} removed", transform=ax.transAxes,
                   fontsize=11, ha='right', va='bottom', color='red',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUT_DIR, 'sift_filtering_stages_real.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_removed_keypoints(gray, stage2_removed, stage3_removed, stage4_removed):
    """Show what keypoints were removed at each stage"""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # Stage 2: Low contrast removed
    ax = axes[0]
    ax.imshow(gray, cmap='gray')
    for kp in stage2_removed[:200]:
        circle = plt.Circle((kp['x'], kp['y']), 3, color='red', fill=False, linewidth=0.8, alpha=0.6)
        ax.add_patch(circle)
    ax.set_title(f"Low Contrast Removed\n{len(stage2_removed)} keypoints", fontsize=12, fontweight='bold', color='#e74c3c')
    ax.axis('off')
    
    # Stage 3: Edge removed
    ax = axes[1]
    ax.imshow(gray, cmap='gray')
    for kp in stage3_removed[:200]:
        circle = plt.Circle((kp['x'], kp['y']), 3, color='orange', fill=False, linewidth=0.8, alpha=0.6)
        ax.add_patch(circle)
    ax.set_title(f"Edge Response Removed\n{len(stage3_removed)} keypoints", fontsize=12, fontweight='bold', color='#f39c12')
    ax.axis('off')
    
    # Stage 4: Sub-pixel removed
    ax = axes[2]
    ax.imshow(gray, cmap='gray')
    for kp in stage4_removed[:200]:
        circle = plt.Circle((kp['x'], kp['y']), 3, color='purple', fill=False, linewidth=0.8, alpha=0.6)
        ax.add_patch(circle)
    ax.set_title(f"Sub-pixel Refinement Removed\n{len(stage4_removed)} keypoints", fontsize=12, fontweight='bold', color='#9b59b6')
    ax.axis('off')
    
    plt.suptitle("Removed Keypoints at Each Stage", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUT_DIR, 'sift_removed_keypoints.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_summary_pipeline(counts):
    """Create summary pipeline diagram with actual counts"""
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    initial, after_contrast, after_edge, final = counts
    removed_contrast = initial - after_contrast
    removed_edge = after_contrast - after_edge
    removed_subpixel = after_edge - final
    
    ax.set_title(f'Keypoint Filtering Pipeline: {initial} → {final}', 
                fontsize=18, fontweight='bold', pad=15)
    
    # Stage boxes
    stages = [
        (1, 'Stage 1\n26-Neighbor\nExtrema', initial, '#3498db', '—'),
        (5, 'Stage 2\nLow Contrast\nRemoval', after_contrast, '#e74c3c', f'−{removed_contrast}'),
        (9, 'Stage 3\nEdge Response\nRemoval', after_edge, '#f39c12', f'−{removed_edge}'),
        (13, 'Stage 4\nSub-pixel\nRefinement', final, '#27ae60', f'−{removed_subpixel}')
    ]
    
    for x, label, count, color, removed in stages:
        # Main box
        box = FancyBboxPatch((x - 1.3, 4), 2.6, 2.5, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, 5.7, label, ha='center', va='center', fontsize=10, 
               color='white', fontweight='bold')
        
        # Count box
        count_box = FancyBboxPatch((x - 0.8, 2.8), 1.6, 0.9, boxstyle="round,pad=0.03",
                                    facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(count_box)
        ax.text(x, 3.25, f'{count}', ha='center', va='center', fontsize=16, 
               fontweight='bold', color=color)
        
        # Removed count
        if removed != '—':
            ax.text(x, 2.2, removed, ha='center', fontsize=12, fontweight='bold', color='#c0392b')
    
    # Arrows
    for x in [2.8, 6.8, 10.8]:
        ax.annotate('', xy=(x + 0.8, 5.25), xytext=(x, 5.25),
                   arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=3))
    
    # Progress bar
    bar_y = 1
    bar_width = 12
    bar_x = 2
    
    # Background bar
    bg_bar = FancyBboxPatch((bar_x, bar_y), bar_width, 0.5, boxstyle="round,pad=0.01",
                             facecolor='#bdc3c7', edgecolor='black')
    ax.add_patch(bg_bar)
    
    # Remaining bar
    ratio = final / initial
    remain_bar = FancyBboxPatch((bar_x, bar_y), bar_width * ratio, 0.5, boxstyle="round,pad=0.01",
                                 facecolor='#27ae60', edgecolor='black')
    ax.add_patch(remain_bar)
    
    ax.text(bar_x - 0.3, bar_y + 0.25, f'{initial}', ha='right', fontsize=11, fontweight='bold')
    ax.text(bar_x + bar_width + 0.3, bar_y + 0.25, f'{final} ({ratio*100:.1f}%)', 
           ha='left', fontsize=11, fontweight='bold', color='#27ae60')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUT_DIR, 'sift_filtering_pipeline_real.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def run_filtering_pipeline():
    """Run complete filtering pipeline with real image"""
    
    print("=" * 70)
    print("SIFT KEYPOINT FILTERING - REAL IMAGE")
    print("=" * 70)
    
    # Load image
    image_path = download_image()
    img = np.array(Image.open(image_path))
    if len(img.shape) == 3:
        gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]) / 255.0
    else:
        gray = img / 255.0
    
    print(f"Image size: {gray.shape[0]} × {gray.shape[1]}")
    
    # Build pyramids
    print("\nBuilding Gaussian pyramid...")
    gaussian_pyramid = build_gaussian_pyramid(gray)
    dog_pyramid = compute_dog_pyramid(gaussian_pyramid)
    
    # Stage 1: Detect all extrema
    print("\nStage 1: Detecting 26-neighbor extrema...")
    all_keypoints = detect_extrema_all(dog_pyramid)
    initial_count = len(all_keypoints)
    print(f"         Found {initial_count} initial keypoints")
    
    # Stage 2: Low contrast removal
    print("\nStage 2: Removing low contrast keypoints...")
    after_contrast, removed_contrast = filter_low_contrast(all_keypoints, threshold=0.03)
    contrast_count = len(after_contrast)
    print(f"         Removed {len(removed_contrast)}, remaining {contrast_count}")
    
    # Stage 3: Edge response removal
    print("\nStage 3: Removing edge responses...")
    after_edge, removed_edge = filter_edge_response(after_contrast, dog_pyramid, edge_threshold=10.0)
    edge_count = len(after_edge)
    print(f"         Removed {len(removed_edge)}, remaining {edge_count}")
    
    # Stage 4: Sub-pixel refinement
    print("\nStage 4: Sub-pixel refinement...")
    final_keypoints, removed_subpixel = filter_subpixel_refinement(after_edge, dog_pyramid, offset_threshold=0.5)
    final_count = len(final_keypoints)
    print(f"         Removed {len(removed_subpixel)}, remaining {final_count}")
    
    # Summary
    print("\n" + "=" * 70)
    print("FILTERING SUMMARY")
    print("=" * 70)
    print(f"Stage 1 (Initial):       {initial_count:5d} keypoints")
    print(f"Stage 2 (Low Contrast):  {contrast_count:5d} keypoints  (−{len(removed_contrast)})")
    print(f"Stage 3 (Edge Response): {edge_count:5d} keypoints  (−{len(removed_edge)})")
    print(f"Stage 4 (Sub-pixel):     {final_count:5d} keypoints  (−{len(removed_subpixel)})")
    print("-" * 70)
    print(f"TOTAL REMOVED:           {initial_count - final_count:5d}")
    print(f"RETENTION RATE:          {final_count/initial_count*100:5.1f}%")
    print("=" * 70)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    stages_data = [
        (all_keypoints, initial_count, 0),
        (after_contrast, contrast_count, len(removed_contrast)),
        (after_edge, edge_count, len(removed_edge)),
        (final_keypoints, final_count, len(removed_subpixel))
    ]
    
    visualize_filtering_stages(gray, stages_data)
    visualize_removed_keypoints(gray, removed_contrast, removed_edge, removed_subpixel)
    visualize_summary_pipeline([initial_count, contrast_count, edge_count, final_count])
    
    print("\nDone!")
    
    return {
        'initial': initial_count,
        'after_contrast': contrast_count,
        'after_edge': edge_count,
        'final': final_count
    }


if __name__ == "__main__":
    run_filtering_pipeline()

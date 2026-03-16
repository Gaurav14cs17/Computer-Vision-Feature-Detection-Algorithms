"""
ORB Pyramid and FAST Detailed Visualization
Creates SIFT-style comprehensive pyramid and detection images
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from scipy import ndimage

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, '..', 'images')

CIRCLE_OFFSETS = [
    (0, -3), (1, -3), (2, -2), (3, -1),
    (3, 0), (3, 1), (2, 2), (1, 3),
    (0, 3), (-1, 3), (-2, 2), (-3, 1),
    (-3, 0), (-3, -1), (-2, -2), (-1, -3)
]


def load_image():
    """Load the real input image."""
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    img = np.array(Image.open(image_path).convert('L')) / 255.0
    return img


def build_pyramid(img, n_levels=8, scale_factor=1.2):
    """Build image pyramid by downsampling."""
    pyramid = [img.copy()]
    scales = [1.0]
    
    for level in range(1, n_levels):
        scale = 1.0 / (scale_factor ** level)
        new_h = int(img.shape[0] * scale)
        new_w = int(img.shape[1] * scale)
        
        if new_h < 20 or new_w < 20:
            break
        
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        resized = np.array(pil_resized) / 255.0
        
        pyramid.append(resized)
        scales.append(scale)
    
    return pyramid, scales


def fast_detect(img, threshold=0.08):
    """Detect FAST corners."""
    h, w = img.shape
    corners = []
    
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            center = img[y, x]
            upper = center + threshold
            lower = center - threshold
            
            test_pos = [0, 4, 8, 12]
            n_b = sum(1 for p in test_pos if img[y + CIRCLE_OFFSETS[p][1], x + CIRCLE_OFFSETS[p][0]] > upper)
            n_d = sum(1 for p in test_pos if img[y + CIRCLE_OFFSETS[p][1], x + CIRCLE_OFFSETS[p][0]] < lower)
            
            if n_b < 3 and n_d < 3:
                continue
            
            labels = []
            for dx, dy in CIRCLE_OFFSETS:
                val = img[y + dy, x + dx]
                if val > upper:
                    labels.append('B')
                elif val < lower:
                    labels.append('D')
                else:
                    labels.append('S')
            
            labels_ext = labels + labels
            max_b = max_d = cnt_b = cnt_d = 0
            for l in labels_ext:
                if l == 'B':
                    cnt_b += 1
                    cnt_d = 0
                    max_b = max(max_b, cnt_b)
                elif l == 'D':
                    cnt_d += 1
                    cnt_b = 0
                    max_d = max(max_d, cnt_d)
                else:
                    cnt_b = cnt_d = 0
            
            if min(max_b, 16) >= 9 or min(max_d, 16) >= 9:
                corners.append({'x': x, 'y': y, 'response': max(max_b, max_d)})
    
    return corners


def compute_harris(img, keypoints, k=0.04):
    """Compute Harris response."""
    Ix = ndimage.sobel(img, axis=1)
    Iy = ndimage.sobel(img, axis=0)
    
    Ixx = ndimage.gaussian_filter(Ix * Ix, 1.5)
    Iyy = ndimage.gaussian_filter(Iy * Iy, 1.5)
    Ixy = ndimage.gaussian_filter(Ix * Iy, 1.5)
    
    for kp in keypoints:
        x, y = kp['x'], kp['y']
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            det = Ixx[y, x] * Iyy[y, x] - Ixy[y, x] ** 2
            trace = Ixx[y, x] + Iyy[y, x]
            kp['harris'] = det - k * trace ** 2
    
    return keypoints


# =============================================================================
# IMAGE 1: Full Pyramid like SIFT (showing all levels at actual resolution)
# =============================================================================
def generate_full_pyramid():
    """Generate full pyramid visualization showing all levels."""
    print("Generating: orb_step1_full_pyramid.png")
    img = load_image()
    pyramid, scales = build_pyramid(img, n_levels=8)
    
    fig = plt.figure(figsize=(20, 14))
    
    n_levels = len(pyramid)
    rows = (n_levels + 3) // 4
    cols = 4
    
    for i, (level_img, scale) in enumerate(zip(pyramid, scales)):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(level_img, cmap='gray')
        h, w = level_img.shape
        ax.set_title(f'Level {i}: {w}×{h}\nScale: {scale:.3f}', fontsize=10)
        ax.axis('off')
    
    fig.suptitle('ORB Step 1: Image Pyramid (Direct Downsampling)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step1_full_pyramid.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# IMAGE 2: Pyramid with Gaussian blur at each level
# =============================================================================
def generate_pyramid_gaussian():
    """Generate pyramid with Gaussian blur visualization."""
    print("Generating: orb_step1_pyramid_gaussian.png")
    img = load_image()
    pyramid, scales = build_pyramid(img, n_levels=4)
    
    # Apply Gaussian blur at different sigmas
    sigma_values = [0, 0.5, 1.0, 1.5]
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    
    for level_idx, (level_img, scale) in enumerate(zip(pyramid, scales)):
        for sigma_idx, sigma in enumerate(sigma_values):
            ax = axes[level_idx, sigma_idx]
            if sigma > 0:
                blurred = ndimage.gaussian_filter(level_img, sigma=sigma)
            else:
                blurred = level_img
            ax.imshow(blurred, cmap='gray')
            h, w = level_img.shape
            ax.set_title(f'L{level_idx}: {w}×{h}, σ={sigma}', fontsize=9)
            ax.axis('off')
    
    fig.suptitle('ORB Image Pyramid with Gaussian Blur\n(Rows: Pyramid Levels, Cols: Gaussian Sigma)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step1_pyramid_gaussian.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# IMAGE 3: FAST detection at each pyramid level (like sift_step3_3_octave0.png)
# =============================================================================
def generate_fast_per_level_detail():
    """Generate FAST corners shown on each pyramid level individually."""
    print("Generating: orb_step2_fast_level_0.png to orb_step2_fast_level_3.png")
    img = load_image()
    pyramid, scales = build_pyramid(img, n_levels=4)
    
    for level, (level_img, scale) in enumerate(zip(pyramid[:4], scales[:4])):
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(level_img, cmap='gray')
        
        kps = fast_detect(level_img, threshold=0.08)
        kps = compute_harris(level_img, kps)
        kps = [k for k in kps if k.get('harris', 0) > 0]
        kps = sorted(kps, key=lambda k: k.get('harris', 0), reverse=True)
        
        # Show more keypoints
        n_show = min(500, len(kps))
        for kp in kps[:n_show]:
            circle = Circle((kp['x'], kp['y']), 4, color='lime', fill=False, linewidth=1.2)
            ax.add_patch(circle)
        
        h, w = level_img.shape
        ax.set_title(f'ORB Level {level}: FAST Corners\nImage: {w}×{h}, Scale: {scale:.3f}, Keypoints: {len(kps)}', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'orb_step2_fast_level_{level}.png'), dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# IMAGE 4: Harris Response Map (like sift_step2_dog.png showing intermediate computation)
# =============================================================================
def generate_harris_response_maps():
    """Generate Harris corner response maps at each level."""
    print("Generating: orb_step3_harris_response.png")
    img = load_image()
    pyramid, scales = build_pyramid(img, n_levels=4)
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    
    for level, (level_img, scale) in enumerate(zip(pyramid, scales)):
        # Compute derivatives
        Ix = ndimage.sobel(level_img, axis=1)
        Iy = ndimage.sobel(level_img, axis=0)
        
        Ixx = ndimage.gaussian_filter(Ix * Ix, 1.5)
        Iyy = ndimage.gaussian_filter(Iy * Iy, 1.5)
        Ixy = ndimage.gaussian_filter(Ix * Iy, 1.5)
        
        # Harris response
        k = 0.04
        det = Ixx * Iyy - Ixy ** 2
        trace = Ixx + Iyy
        harris = det - k * trace ** 2
        
        # Plot: Original, Ix, Iy, Harris
        axes[level, 0].imshow(level_img, cmap='gray')
        axes[level, 0].set_title(f'L{level}: Original', fontsize=9)
        axes[level, 0].axis('off')
        
        axes[level, 1].imshow(Ix, cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[level, 1].set_title(f'L{level}: Gradient Ix', fontsize=9)
        axes[level, 1].axis('off')
        
        axes[level, 2].imshow(Iy, cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[level, 2].set_title(f'L{level}: Gradient Iy', fontsize=9)
        axes[level, 2].axis('off')
        
        # Normalize Harris for visualization
        harris_vis = harris.copy()
        harris_vis = np.clip(harris_vis, np.percentile(harris, 1), np.percentile(harris, 99))
        axes[level, 3].imshow(harris_vis, cmap='hot')
        axes[level, 3].set_title(f'L{level}: Harris Response', fontsize=9)
        axes[level, 3].axis('off')
    
    fig.suptitle('ORB Step 3: Harris Corner Response Computation\n(Rows: Pyramid Levels)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step3_harris_response.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# IMAGE 5: Pyramid Structure Diagram (like sift_step3_6_pyramid_structure.png)
# =============================================================================
def generate_pyramid_structure():
    """Generate pyramid structure diagram."""
    print("Generating: orb_pyramid_structure.png")
    img = load_image()
    pyramid, scales = build_pyramid(img, n_levels=8)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Calculate total width needed
    x_offset = 0
    spacing = 20
    
    for i, (level_img, scale) in enumerate(zip(pyramid[:6], scales[:6])):
        h, w = level_img.shape
        # Scale for display
        display_scale = 0.7
        dh, dw = int(h * display_scale), int(w * display_scale)
        
        # Center vertically
        y_offset = (pyramid[0].shape[0] * display_scale - dh) / 2
        
        # Resize for display
        pil_img = Image.fromarray((level_img * 255).astype(np.uint8))
        pil_resized = pil_img.resize((dw, dh), Image.LANCZOS)
        display_img = np.array(pil_resized)
        
        ax.imshow(display_img, cmap='gray', extent=[x_offset, x_offset + dw, y_offset + dh, y_offset])
        ax.text(x_offset + dw/2, y_offset + dh + 15, f'Level {i}\n{w}×{h}\nscale={scale:.3f}', 
               ha='center', fontsize=9)
        
        # Draw connecting arrow
        if i > 0:
            ax.annotate('', xy=(x_offset - 5, y_offset + dh/2), 
                       xytext=(prev_end + 5, y_offset + dh/2),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax.text((prev_end + x_offset) / 2, y_offset + dh/2 - 10, 
                   f'÷{1.2:.1f}', ha='center', fontsize=8, color='red')
        
        prev_end = x_offset + dw
        x_offset += dw + spacing
    
    ax.set_xlim(-20, x_offset)
    ax.set_ylim(pyramid[0].shape[0] * 0.7 + 50, -50)
    ax.set_title('ORB Image Pyramid Structure\n(Scale Factor = 1.2 between levels)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_pyramid_structure.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# IMAGE 6: FAST Circle Test Visualization on Real Image
# =============================================================================
def generate_fast_circle_real():
    """Generate FAST circle test visualization on real image patch."""
    print("Generating: orb_fast_circle_real.png")
    img = load_image()
    
    # Find a good corner point
    kps = fast_detect(img, threshold=0.08)
    kps = compute_harris(img, kps)
    kps = [k for k in kps if k.get('harris', 0) > 0]
    kps = sorted(kps, key=lambda k: k.get('harris', 0), reverse=True)
    
    # Pick a keypoint away from edges
    kp = None
    h, w = img.shape
    for k in kps:
        if 30 <= k['x'] < w - 30 and 30 <= k['y'] < h - 30:
            kp = k
            break
    
    if kp is None:
        kp = kps[0]
    
    cx, cy = kp['x'], kp['y']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Full image with keypoint location
    ax1 = axes[0]
    ax1.imshow(img, cmap='gray')
    ax1.plot(cx, cy, 'r+', markersize=20, markeredgewidth=3)
    rect_size = 30
    from matplotlib.patches import Rectangle
    rect = Rectangle((cx - rect_size, cy - rect_size), rect_size*2, rect_size*2, 
                     fill=False, color='yellow', linewidth=2)
    ax1.add_patch(rect)
    ax1.set_title('Full Image with Keypoint Location', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Zoomed patch with FAST circle
    ax2 = axes[1]
    patch_size = 20
    patch = img[max(0, cy-patch_size):min(h, cy+patch_size+1), 
                max(0, cx-patch_size):min(w, cx+patch_size+1)]
    ax2.imshow(patch, cmap='gray')
    
    center = patch_size  # Local center
    center_val = patch[center, center]
    threshold = 0.08
    
    # Draw circle points
    for i, (dx, dy) in enumerate(CIRCLE_OFFSETS):
        px, py = center + dx, center + dy
        if 0 <= px < patch.shape[1] and 0 <= py < patch.shape[0]:
            val = patch[py, px]
            if val > center_val + threshold:
                color = 'lime'  # Brighter
            elif val < center_val - threshold:
                color = 'red'  # Darker
            else:
                color = 'yellow'  # Similar
            ax2.plot(px, py, 'o', color=color, markersize=10, markeredgecolor='black')
            ax2.text(px + 0.5, py - 0.5, str(i), fontsize=8, color='white')
    
    ax2.plot(center, center, 's', color='blue', markersize=12)
    ax2.set_title(f'FAST Circle (16 points, radius=3)\nCenter intensity: {center_val:.3f}', 
                 fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Intensity profile
    ax3 = axes[2]
    intensities = []
    for dx, dy in CIRCLE_OFFSETS:
        px, py = center + dx, center + dy
        if 0 <= px < patch.shape[1] and 0 <= py < patch.shape[0]:
            intensities.append(patch[py, px])
        else:
            intensities.append(0)
    
    x = range(16)
    colors = []
    for i in intensities:
        if i > center_val + threshold:
            colors.append('lime')
        elif i < center_val - threshold:
            colors.append('red')
        else:
            colors.append('yellow')
    
    ax3.bar(x, intensities, color=colors, edgecolor='black')
    ax3.axhline(y=center_val, color='blue', linestyle='--', label=f'Center: {center_val:.3f}')
    ax3.axhline(y=center_val + threshold, color='lime', linestyle=':', label=f'Upper: {center_val+threshold:.3f}')
    ax3.axhline(y=center_val - threshold, color='red', linestyle=':', label=f'Lower: {center_val-threshold:.3f}')
    ax3.set_xlabel('Circle Position (0-15)')
    ax3.set_ylabel('Intensity')
    ax3.set_title('Intensity Profile Around Circle\nGreen=Brighter, Red=Darker, Yellow=Similar', 
                 fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    
    fig.suptitle('FAST Corner Detection on Real Image', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_fast_circle_real.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# IMAGE 7: Before/After Harris filtering comparison
# =============================================================================
def generate_harris_comparison():
    """Generate before/after Harris filtering comparison."""
    print("Generating: orb_harris_before_after.png")
    img = load_image()
    
    # Before: FAST only
    kps_before = fast_detect(img, threshold=0.08)
    
    # After: FAST + Harris filtering
    kps_after = fast_detect(img, threshold=0.08)
    kps_after = compute_harris(img, kps_after)
    kps_after = [k for k in kps_after if k.get('harris', 0) > 0]
    kps_after = sorted(kps_after, key=lambda k: k.get('harris', 0), reverse=True)[:500]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Before
    ax1 = axes[0]
    ax1.imshow(img, cmap='gray')
    n_show = min(800, len(kps_before))
    if len(kps_before) > n_show:
        indices = np.linspace(0, len(kps_before)-1, n_show, dtype=int)
        kps_show = [kps_before[i] for i in indices]
    else:
        kps_show = kps_before
    for kp in kps_show:
        circle = Circle((kp['x'], kp['y']), 3, color='red', fill=False, linewidth=0.5)
        ax1.add_patch(circle)
    ax1.set_title(f'Before: FAST Corners Only\n({len(kps_before)} keypoints)', 
                 fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # After
    ax2 = axes[1]
    ax2.imshow(img, cmap='gray')
    for kp in kps_after:
        circle = Circle((kp['x'], kp['y']), 4, color='lime', fill=False, linewidth=1.2)
        ax2.add_patch(circle)
    ax2.set_title(f'After: Harris Corner Response Filter\n({len(kps_after)} keypoints, top 500)', 
                 fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    fig.suptitle('ORB: Effect of Harris Corner Response Filtering', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_harris_before_after.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("ORB Pyramid and FAST Detailed Visualization")
    print("=" * 60)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    generate_full_pyramid()
    generate_pyramid_gaussian()
    generate_fast_per_level_detail()
    generate_harris_response_maps()
    generate_pyramid_structure()
    generate_fast_circle_real()
    generate_harris_comparison()
    
    print("\n" + "=" * 60)
    print("All pyramid/FAST visualizations generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
ORB Complete Visualization - Matching SIFT Style
Generates all step-by-step images with real image data
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image
from scipy import ndimage

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, '..', 'images')

# Bresenham circle offsets
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
    """Build image pyramid."""
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
            
            # High-speed test
            test_pos = [0, 4, 8, 12]
            n_b = sum(1 for p in test_pos if img[y + CIRCLE_OFFSETS[p][1], x + CIRCLE_OFFSETS[p][0]] > upper)
            n_d = sum(1 for p in test_pos if img[y + CIRCLE_OFFSETS[p][1], x + CIRCLE_OFFSETS[p][0]] < lower)
            
            if n_b < 3 and n_d < 3:
                continue
            
            # Full test
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


def compute_orientation(img, keypoints, radius=15):
    """Compute orientation using intensity centroid."""
    h, w = img.shape
    for kp in keypoints:
        x, y = kp['x'], kp['y']
        if x - radius < 0 or x + radius >= w or y - radius < 0 or y + radius >= h:
            kp['orientation'] = 0
            continue
        
        m10 = m01 = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    m10 += dx * img[y + dy, x + dx]
                    m01 += dy * img[y + dy, x + dx]
        
        kp['orientation'] = np.arctan2(m01, m10)
    return keypoints


def generate_descriptor(img, kp, pattern):
    """Generate rBRIEF descriptor."""
    h, w = img.shape
    x, y = kp['x'], kp['y']
    theta = kp.get('orientation', 0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    desc = np.zeros(len(pattern), dtype=np.uint8)
    for i, ((px, py), (qx, qy)) in enumerate(pattern):
        rpx = int(round(px * cos_t - py * sin_t))
        rpy = int(round(px * sin_t + py * cos_t))
        rqx = int(round(qx * cos_t - qy * sin_t))
        rqy = int(round(qx * sin_t + qy * cos_t))
        
        p_x, p_y = x + rpx, y + rpy
        q_x, q_y = x + rqx, y + rqy
        
        if 0 <= p_x < w and 0 <= p_y < h and 0 <= q_x < w and 0 <= q_y < h:
            if img[p_y, p_x] < img[q_y, q_x]:
                desc[i] = 1
    return desc


def get_pattern(n_pairs=256, patch_size=31, seed=42):
    """Generate BRIEF pattern."""
    np.random.seed(seed)
    half = patch_size // 2
    pattern = []
    for _ in range(n_pairs):
        px = int(np.clip(np.random.randn() * half/2, -half, half))
        py = int(np.clip(np.random.randn() * half/2, -half, half))
        qx = int(np.clip(np.random.randn() * half/2, -half, half))
        qy = int(np.clip(np.random.randn() * half/2, -half, half))
        pattern.append(((px, py), (qx, qy)))
    return pattern


# =============================================================================
# IMAGE 1: All Keypoints from All Levels (like sift_all_octaves_combined.png)
# =============================================================================
def generate_all_levels_combined():
    """Generate image showing keypoints from all pyramid levels with scale-based circles."""
    print("Generating: orb_all_levels_combined.png")
    img = load_image()
    pyramid, scales = build_pyramid(img)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    level_counts = []
    
    for level, (level_img, scale) in enumerate(zip(pyramid[:6], scales[:6])):
        kps = fast_detect(level_img, threshold=0.08)
        level_counts.append(len(kps))
        
        # Sample for display
        n_show = min(200, len(kps))
        if len(kps) > n_show:
            indices = np.linspace(0, len(kps)-1, n_show, dtype=int)
            kps = [kps[i] for i in indices]
        
        for kp in kps:
            # Map back to original coordinates
            orig_x = kp['x'] / scale
            orig_y = kp['y'] / scale
            radius = 4 + level * 4  # Larger circles for coarser scales
            circle = Circle((orig_x, orig_y), radius, 
                          color=colors[level % len(colors)], fill=False, linewidth=1.2)
            ax.add_patch(circle)
    
    # Legend
    for i, (count, color) in enumerate(zip(level_counts, colors[:len(level_counts)])):
        ax.plot([], [], 'o', color=color, markersize=8,
               label=f'Level {i}: {count} kps (scale={scales[i]:.2f})')
    
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    total = sum(level_counts)
    ax.set_title(f'All Detected Keypoints (Combined from All Levels)\n(Total: {total} keypoints)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_all_levels_combined.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# IMAGE 2: Stage 1 - After FAST Detection (like sift_stage1_low_contrast.png)
# =============================================================================
def generate_stage1_fast():
    """Generate image showing FAST corners at different scales."""
    print("Generating: orb_stage1_fast_detection.png")
    img = load_image()
    pyramid, scales = build_pyramid(img)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    colors = ['red', 'green', 'blue']
    labels = ['Level 0: Fine-scale', 'Level 1: Medium-scale', 'Level 2: Coarse-scale']
    level_kps = []
    
    for level in range(3):
        kps = fast_detect(pyramid[level], threshold=0.08)
        level_kps.append(kps)
        
        n_show = min(400, len(kps))
        if len(kps) > n_show:
            indices = np.linspace(0, len(kps)-1, n_show, dtype=int)
            kps_show = [kps[i] for i in indices]
        else:
            kps_show = kps
        
        for kp in kps_show:
            orig_x = kp['x'] / scales[level]
            orig_y = kp['y'] / scales[level]
            radius = 3 + level * 5
            circle = Circle((orig_x, orig_y), radius,
                          color=colors[level], fill=False, linewidth=1)
            ax.add_patch(circle)
    
    # Legend
    for i, (kps, color, label) in enumerate(zip(level_kps, colors, labels)):
        ax.plot([], [], 'o', color=color, markersize=8, label=f'{label} ({len(kps)} kps)')
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    total = sum(len(k) for k in level_kps)
    ax.set_title(f'Stage 1: After FAST Detection\n(Total: {total} keypoints)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_stage1_fast_detection.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# IMAGE 3: Stage 2 - After Harris Filtering (like sift_stage2_edge_response.png)
# =============================================================================
def generate_stage2_harris():
    """Generate image showing keypoints after Harris filtering."""
    print("Generating: orb_stage2_harris_filter.png")
    img = load_image()
    pyramid, scales = build_pyramid(img)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    colors = ['red', 'green', 'blue']
    labels = ['Level 0: Fine-scale', 'Level 1: Medium-scale', 'Level 2: Coarse-scale']
    level_kps = []
    
    for level in range(3):
        kps = fast_detect(pyramid[level], threshold=0.08)
        kps = compute_harris(pyramid[level], kps)
        # Filter by Harris response
        kps = [k for k in kps if k.get('harris', 0) > 0]
        kps = sorted(kps, key=lambda k: k.get('harris', 0), reverse=True)[:300]
        level_kps.append(kps)
        
        for kp in kps:
            orig_x = kp['x'] / scales[level]
            orig_y = kp['y'] / scales[level]
            radius = 3 + level * 5
            circle = Circle((orig_x, orig_y), radius,
                          color=colors[level], fill=False, linewidth=1.2)
            ax.add_patch(circle)
    
    for i, (kps, color, label) in enumerate(zip(level_kps, colors, labels)):
        ax.plot([], [], 'o', color=color, markersize=8, label=f'{label} ({len(kps)} kps)')
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    total = sum(len(k) for k in level_kps)
    ax.set_title(f'Stage 2: After Harris Corner Response Filtering\n(Total: {total} keypoints)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_stage2_harris_filter.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# IMAGE 4: Stage 3 - With Orientation (like sift_step5_orientation.png)
# =============================================================================
def generate_stage3_orientation():
    """Generate image showing keypoints with orientation."""
    print("Generating: orb_stage3_with_orientation.png")
    img = load_image()
    
    # Detect and filter
    kps = fast_detect(img, threshold=0.08)
    kps = compute_harris(img, kps)
    kps = [k for k in kps if k.get('harris', 0) > 0]
    kps = sorted(kps, key=lambda k: k.get('harris', 0), reverse=True)[:500]
    kps = compute_orientation(img, kps)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    for kp in kps:
        x, y = kp['x'], kp['y']
        theta = kp.get('orientation', 0)
        
        circle = Circle((x, y), 6, color='lime', fill=False, linewidth=1.5)
        ax.add_patch(circle)
        
        arrow_len = 12
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=4, head_length=3, fc='red', ec='red', linewidth=1)
    
    ax.set_title(f'Stage 3: {len(kps)} Keypoints with Orientation', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_stage3_with_orientation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return kps


# =============================================================================
# IMAGE 5: Final Keypoints with Descriptors (like sift_step6_descriptors.png)
# =============================================================================
def generate_final_descriptors(kps):
    """Generate image showing final keypoints with descriptors."""
    print("Generating: orb_final_with_descriptors.png")
    img = load_image()
    pattern = get_pattern()
    
    valid_kps = []
    h, w = img.shape
    for kp in kps:
        if 16 <= kp['x'] < w - 16 and 16 <= kp['y'] < h - 16:
            desc = generate_descriptor(img, kp, pattern)
            kp['descriptor'] = desc
            valid_kps.append(kp)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    for kp in valid_kps:
        x, y = kp['x'], kp['y']
        theta = kp.get('orientation', 0)
        
        circle = Circle((x, y), 6, color='lime', fill=False, linewidth=1.5)
        ax.add_patch(circle)
        
        arrow_len = 12
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=4, head_length=3, fc='red', ec='red', linewidth=1)
    
    ax.set_title(f'Final: {len(valid_kps)} Keypoints with 256-bit Descriptors', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_final_with_descriptors.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return valid_kps


# =============================================================================
# IMAGE 6: Descriptor Pipeline (like sift_desc_pipeline_real.png)
# =============================================================================
def generate_descriptor_pipeline_real(kps):
    """Generate descriptor creation pipeline using real image patch."""
    print("Generating: orb_desc_pipeline_real.png")
    img = load_image()
    pattern = get_pattern()
    
    # Find a good keypoint
    h, w = img.shape
    kp = None
    for k in kps:
        if 20 <= k['x'] < w - 20 and 20 <= k['y'] < h - 20:
            kp = k
            break
    
    if kp is None:
        kp = kps[0]
    
    x, y = kp['x'], kp['y']
    theta = kp.get('orientation', 0)
    
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Keypoint + Orientation on image
    ax1 = fig.add_subplot(2, 3, 1)
    patch_size = 50
    y1, y2 = max(0, y-patch_size), min(h, y+patch_size)
    x1, x2 = max(0, x-patch_size), min(w, x+patch_size)
    ax1.imshow(img[y1:y2, x1:x2], cmap='gray')
    
    # Draw keypoint
    local_x, local_y = x - x1, y - y1
    circle = Circle((local_x, local_y), 15, color='red', fill=False, linewidth=2)
    ax1.add_patch(circle)
    arrow_len = 20
    ax1.arrow(local_x, local_y, arrow_len*np.cos(theta), arrow_len*np.sin(theta),
             head_width=5, head_length=3, fc='yellow', ec='yellow', linewidth=2)
    ax1.set_title('1. Keypoint\n+ Orientation', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # 2. Extract 31x31 patch
    ax2 = fig.add_subplot(2, 3, 2)
    half = 15
    if y-half >= 0 and y+half+1 <= h and x-half >= 0 and x+half+1 <= w:
        patch = img[y-half:y+half+1, x-half:x+half+1]
        ax2.imshow(patch, cmap='gray')
    ax2.set_title('2. Extract\n31×31 Region', fontsize=11, fontweight='bold')
    ax2.axis('off')
    
    # 3. Show sampling pattern
    ax3 = fig.add_subplot(2, 3, 3)
    if y-half >= 0 and y+half+1 <= h and x-half >= 0 and x+half+1 <= w:
        ax3.imshow(patch, cmap='gray', alpha=0.5)
    
    # Draw some pattern points
    for i, ((px, py), (qx, qy)) in enumerate(pattern[:30]):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rpx = px * cos_t - py * sin_t + half
        rpy = px * sin_t + py * cos_t + half
        rqx = qx * cos_t - qy * sin_t + half
        rqy = qx * sin_t + qy * cos_t + half
        ax3.plot([rpx, rqx], [rpy, rqy], 'g-', alpha=0.5, linewidth=0.5)
        ax3.plot(rpx, rpy, 'ro', markersize=3)
        ax3.plot(rqx, rqy, 'bs', markersize=3)
    
    ax3.set_title('3. Rotated\nSampling Pattern', fontsize=11, fontweight='bold')
    ax3.set_xlim(-2, 33)
    ax3.set_ylim(33, -2)
    ax3.axis('off')
    
    # 4. Binary comparisons
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis('off')
    
    text = "4. Binary Comparisons\n" + "="*25 + "\n\n"
    text += "For each point pair (p, q):\n\n"
    
    for i in range(5):
        (px, py), (qx, qy) = pattern[i]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rpx = int(round(px * cos_t - py * sin_t))
        rpy = int(round(px * sin_t + py * cos_t))
        rqx = int(round(qx * cos_t - qy * sin_t))
        rqy = int(round(qx * sin_t + qy * cos_t))
        
        p_x, p_y = x + rpx, y + rpy
        q_x, q_y = x + rqx, y + rqy
        
        if 0 <= p_x < w and 0 <= p_y < h and 0 <= q_x < w and 0 <= q_y < h:
            Ip = img[p_y, p_x]
            Iq = img[q_y, q_x]
            bit = 1 if Ip < Iq else 0
            text += f"Pair {i+1}: I(p)={Ip:.2f}, I(q)={Iq:.2f} → bit={bit}\n"
    
    ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # 5. Build descriptor
    ax5 = fig.add_subplot(2, 3, 5)
    desc = generate_descriptor(img, kp, pattern)
    ax5.imshow(desc.reshape(16, 16), cmap='binary', aspect='auto')
    ax5.set_title('5. 256-bit Descriptor\n(16×16 visualization)', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Bit column')
    ax5.set_ylabel('Bit row')
    
    # 6. Final descriptor vector
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.bar(range(64), desc[:64], color='green', width=1)
    ax6.set_xlim(-1, 65)
    ax6.set_ylim(-0.1, 1.1)
    ax6.set_xlabel('Bit position (showing first 64)')
    ax6.set_ylabel('Bit value')
    ax6.set_title('6. Binary Vector\n(First 64 of 256 bits)', fontsize=11, fontweight='bold')
    
    fig.suptitle('ORB Descriptor Creation Pipeline (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_desc_pipeline_real.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# IMAGE 7: Refined Keypoints (like sift_step4_refined.png)
# =============================================================================
def generate_refined_keypoints():
    """Generate image showing refined keypoints."""
    print("Generating: orb_refined_keypoints.png")
    img = load_image()
    
    kps = fast_detect(img, threshold=0.08)
    kps = compute_harris(img, kps)
    kps = [k for k in kps if k.get('harris', 0) > 0]
    kps = sorted(kps, key=lambda k: k.get('harris', 0), reverse=True)[:400]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    for kp in kps:
        circle = Circle((kp['x'], kp['y']), 5, color='lime', fill=False, linewidth=1.2)
        ax.add_patch(circle)
    
    ax.set_title(f'ORB: {len(kps)} Refined Keypoints', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_refined_keypoints.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# IMAGE 8: Keypoints at Different Scales (like sift_step3_8_final_scales.png)
# =============================================================================
def generate_keypoints_at_scales():
    """Generate image showing keypoints at different pyramid levels."""
    print("Generating: orb_keypoints_at_scales.png")
    img = load_image()
    pyramid, scales = build_pyramid(img)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, (level_img, scale) in enumerate(zip(pyramid[:8], scales[:8])):
        ax = axes[i]
        ax.imshow(level_img, cmap='gray')
        
        kps = fast_detect(level_img, threshold=0.08)
        kps = compute_harris(level_img, kps)
        kps = [k for k in kps if k.get('harris', 0) > 0]
        kps = sorted(kps, key=lambda k: k.get('harris', 0), reverse=True)[:100]
        
        for kp in kps:
            circle = Circle((kp['x'], kp['y']), 3, color='lime', fill=False, linewidth=0.8)
            ax.add_patch(circle)
        
        h, w = level_img.shape
        ax.set_title(f'Level {i}: {w}×{h}\nscale={scale:.3f}, {len(kps)} kps', fontsize=10)
        ax.axis('off')
    
    fig.suptitle('ORB Keypoints at Different Pyramid Scales', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_keypoints_at_scales.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# IMAGE 9: Scale-Dependent Circle Sizes (like sift_scale_factor_real.png)
# =============================================================================
def generate_scale_circles():
    """Generate image showing scale-dependent keypoint circles."""
    print("Generating: orb_scale_factor_real.png")
    img = load_image()
    pyramid, scales = build_pyramid(img)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    all_kps = []
    for level, (level_img, scale) in enumerate(zip(pyramid[:5], scales[:5])):
        kps = fast_detect(level_img, threshold=0.08)
        kps = compute_harris(level_img, kps)
        kps = [k for k in kps if k.get('harris', 0) > 0]
        kps = sorted(kps, key=lambda k: k.get('harris', 0), reverse=True)[:80]
        
        for kp in kps:
            kp['level'] = level
            kp['scale'] = scale
            all_kps.append(kp)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, 5))
    
    for kp in all_kps:
        level = kp['level']
        scale = kp['scale']
        orig_x = kp['x'] / scale
        orig_y = kp['y'] / scale
        # Circle size proportional to scale
        radius = 5 * (1.2 ** level)
        circle = Circle((orig_x, orig_y), radius, 
                        color=colors[level], fill=False, linewidth=1.5)
        ax.add_patch(circle)
    
    for i in range(5):
        ax.plot([], [], 'o', color=colors[i], markersize=10,
               label=f'Level {i} (radius={5*(1.2**i):.1f}px)')
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_title('ORB Keypoints with Scale-Dependent Circle Sizes', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_scale_factor_real.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# IMAGE 10: Descriptor Vectors (like sift_step6_descriptor_vectors.png)
# =============================================================================
def generate_descriptor_vectors(kps):
    """Generate visualization of descriptor vectors."""
    print("Generating: orb_descriptor_vectors.png")
    img = load_image()
    pattern = get_pattern()
    
    h, w = img.shape
    valid_kps = []
    descriptors = []
    for kp in kps[:20]:
        if 16 <= kp['x'] < w - 16 and 16 <= kp['y'] < h - 16:
            desc = generate_descriptor(img, kp, pattern)
            valid_kps.append(kp)
            descriptors.append(desc)
    
    if len(descriptors) < 5:
        return
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 10))
    
    for i in range(5):
        ax = axes[i]
        desc = descriptors[i]
        colors = ['green' if b == 1 else 'white' for b in desc]
        ax.bar(range(256), np.ones(256), color=colors, width=1, edgecolor='gray', linewidth=0.1)
        ax.set_xlim(-1, 257)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel(f'KP {i+1}', fontsize=10)
        ax.set_yticks([])
        if i < 4:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Bit Position (0-255)', fontsize=10)
    
    fig.suptitle('ORB Descriptor Vectors (256-bit binary)\nGreen=1, White=0', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_descriptor_vectors.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("ORB Complete Visualization Generator")
    print("Generating all images with real image data")
    print("=" * 60)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Generate all images
    generate_all_levels_combined()
    generate_stage1_fast()
    generate_stage2_harris()
    kps = generate_stage3_orientation()
    valid_kps = generate_final_descriptors(kps)
    generate_descriptor_pipeline_real(kps)
    generate_refined_keypoints()
    generate_keypoints_at_scales()
    generate_scale_circles()
    generate_descriptor_vectors(kps)
    
    print("\n" + "=" * 60)
    print("All visualizations generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()

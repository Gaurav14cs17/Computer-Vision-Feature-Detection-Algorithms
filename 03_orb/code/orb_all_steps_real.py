"""
ORB All Steps - Real Image Visualization
Generates step-by-step images showing real results at each stage
Matches SIFT style with consistent keypoint visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
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
        pyramid.append(np.array(pil_resized) / 255.0)
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


# =============================================================================
# STEP 1: Input Image
# =============================================================================
def generate_step0_input():
    """Generate input image visualization."""
    print("Generating: orb_step0_input.png")
    img = load_image()
    
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img, cmap='gray')
    h, w = img.shape
    ax.set_title(f'ORB Step 0: Input Image ({w}×{h})', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step0_input.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# STEP 1: Pyramid - Show each level
# =============================================================================
def generate_step1_pyramid_levels():
    """Generate pyramid level images."""
    print("Generating: orb_step1_level_*.png")
    img = load_image()
    pyramid, scales = build_pyramid(img)
    
    for level, (level_img, scale) in enumerate(zip(pyramid, scales)):
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(level_img, cmap='gray')
        h, w = level_img.shape
        ax.set_title(f'ORB Step 1: Pyramid Level {level}\nSize: {w}×{h}, Scale: {scale:.3f}', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'orb_step1_level_{level}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    return pyramid, scales


# =============================================================================
# STEP 2: FAST Detection per level
# =============================================================================
def generate_step2_fast_all_levels(pyramid, scales):
    """Generate FAST detection images for each level."""
    print("Generating: orb_step2_fast_all.png and per-level")
    img = load_image()
    
    all_kps = []
    level_kps_list = []
    
    for level, (level_img, scale) in enumerate(zip(pyramid[:4], scales[:4])):
        kps = fast_detect(level_img, threshold=0.08)
        level_kps_list.append((kps, scale, level))
        
        # Per-level image
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(level_img, cmap='gray')
        
        n_show = min(500, len(kps))
        if len(kps) > n_show:
            indices = np.linspace(0, len(kps)-1, n_show, dtype=int)
            kps_show = [kps[i] for i in indices]
        else:
            kps_show = kps
        
        for kp in kps_show:
            circle = Circle((kp['x'], kp['y']), 3, color='lime', fill=False, linewidth=1)
            ax.add_patch(circle)
        
        h, w = level_img.shape
        ax.set_title(f'ORB Step 2: FAST Corners - Level {level}\n{len(kps)} keypoints detected ({w}×{h})', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'orb_step2_level_{level}_fast.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Collect for combined
        for kp in kps:
            all_kps.append({
                'x': kp['x'] / scale,
                'y': kp['y'] / scale,
                'level': level,
                'scale': scale
            })
    
    # Combined on original image
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img, cmap='gray')
    
    colors = ['red', 'green', 'blue', 'orange']
    for level, (kps, scale, lvl) in enumerate(level_kps_list):
        n_show = min(300, len(kps))
        if len(kps) > n_show:
            indices = np.linspace(0, len(kps)-1, n_show, dtype=int)
            kps_show = [kps[i] for i in indices]
        else:
            kps_show = kps
        
        for kp in kps_show:
            orig_x = kp['x'] / scale
            orig_y = kp['y'] / scale
            radius = 3 + level * 4
            circle = Circle((orig_x, orig_y), radius, color=colors[level], fill=False, linewidth=1)
            ax.add_patch(circle)
    
    # Legend
    for i, (kps, scale, lvl) in enumerate(level_kps_list):
        ax.plot([], [], 'o', color=colors[i], markersize=8, label=f'Level {i}: {len(kps)} kps')
    
    total = sum(len(kps) for kps, _, _ in level_kps_list)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_title(f'ORB Step 2: All FAST Corners Combined\nTotal: {total} keypoints', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step2_all_fast.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return level_kps_list


# =============================================================================
# STEP 3: Harris Filtering
# =============================================================================
def generate_step3_harris(pyramid, scales):
    """Generate Harris filtering images."""
    print("Generating: orb_step3_harris_*.png")
    img = load_image()
    
    all_harris_kps = []
    
    for level, (level_img, scale) in enumerate(zip(pyramid[:4], scales[:4])):
        kps = fast_detect(level_img, threshold=0.08)
        kps = compute_harris(level_img, kps)
        kps = [k for k in kps if k.get('harris', 0) > 0]
        kps = sorted(kps, key=lambda k: k.get('harris', 0), reverse=True)[:200]
        
        # Per-level image
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(level_img, cmap='gray')
        
        for kp in kps:
            circle = Circle((kp['x'], kp['y']), 4, color='lime', fill=False, linewidth=1.2)
            ax.add_patch(circle)
        
        h, w = level_img.shape
        ax.set_title(f'ORB Step 3: Harris Filtered - Level {level}\n{len(kps)} keypoints ({w}×{h})', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'orb_step3_level_{level}_harris.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        for kp in kps:
            kp['orig_x'] = kp['x'] / scale
            kp['orig_y'] = kp['y'] / scale
            kp['level'] = level
            kp['scale'] = scale
            all_harris_kps.append(kp)
    
    # Combined on original image
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img, cmap='gray')
    
    colors = ['red', 'green', 'blue', 'orange']
    level_counts = [0, 0, 0, 0]
    for kp in all_harris_kps:
        level = kp['level']
        level_counts[level] += 1
        radius = 4 + level * 5
        circle = Circle((kp['orig_x'], kp['orig_y']), radius, 
                        color=colors[level], fill=False, linewidth=1.2)
        ax.add_patch(circle)
    
    for i in range(4):
        ax.plot([], [], 'o', color=colors[i], markersize=8, label=f'Level {i}: {level_counts[i]} kps')
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_title(f'ORB Step 3: All Harris Filtered Keypoints\nTotal: {len(all_harris_kps)} keypoints', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step3_all_harris.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return all_harris_kps


# =============================================================================
# STEP 4: Orientation Assignment
# =============================================================================
def generate_step4_orientation(pyramid, scales):
    """Generate orientation assignment images."""
    print("Generating: orb_step4_orientation_*.png")
    img = load_image()
    
    # Get keypoints from level 0
    kps = fast_detect(img, threshold=0.08)
    kps = compute_harris(img, kps)
    kps = [k for k in kps if k.get('harris', 0) > 0]
    kps = sorted(kps, key=lambda k: k.get('harris', 0), reverse=True)[:500]
    kps = compute_orientation(img, kps)
    
    # Full image with orientation
    fig, ax = plt.subplots(figsize=(12, 9))
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
    
    ax.set_title(f'ORB Step 4: {len(kps)} Keypoints with Orientation', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step4_with_orientation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return kps


# =============================================================================
# STEP 5: Descriptors
# =============================================================================
def generate_step5_descriptors(kps):
    """Generate descriptor images."""
    print("Generating: orb_step5_*.png")
    img = load_image()
    h, w = img.shape
    pattern = get_pattern()
    
    valid_kps = []
    for kp in kps:
        if 16 <= kp['x'] < w - 16 and 16 <= kp['y'] < h - 16:
            desc = generate_descriptor(img, kp, pattern)
            kp['descriptor'] = desc
            valid_kps.append(kp)
    
    # Keypoints with descriptors
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img, cmap='gray')
    
    for kp in valid_kps:
        x, y = kp['x'], kp['y']
        theta = kp.get('orientation', 0)
        
        circle = Circle((x, y), 6, color='cyan', fill=False, linewidth=1.5)
        ax.add_patch(circle)
        
        arrow_len = 12
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=4, head_length=3, fc='yellow', ec='yellow', linewidth=1)
    
    ax.set_title(f'ORB Step 5: {len(valid_kps)} Keypoints with 256-bit Descriptors', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step5_with_descriptors.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return valid_kps


# =============================================================================
# STEP 6: Matching
# =============================================================================
def generate_step6_matching(kps):
    """Generate matching visualization."""
    print("Generating: orb_step6_matching.png")
    img = load_image()
    h, w = img.shape
    pattern = get_pattern()
    
    # Create rotated version
    angle = 15
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_rotated = pil_img.rotate(angle, expand=False)
    img_rotated = np.array(pil_rotated) / 255.0
    
    # Detect on rotated
    kps2 = fast_detect(img_rotated, threshold=0.08)
    kps2 = compute_harris(img_rotated, kps2)
    kps2 = [k for k in kps2 if k.get('harris', 0) > 0]
    kps2 = sorted(kps2, key=lambda k: k.get('harris', 0), reverse=True)[:300]
    kps2 = compute_orientation(img_rotated, kps2)
    
    # Generate descriptors
    valid_kps2 = []
    for kp in kps2:
        if 16 <= kp['x'] < w - 16 and 16 <= kp['y'] < h - 16:
            desc = generate_descriptor(img_rotated, kp, pattern)
            kp['descriptor'] = desc
            valid_kps2.append(kp)
    
    # Match
    matches = []
    for i, kp1 in enumerate(kps[:100]):
        if 'descriptor' not in kp1:
            continue
        best_dist = 256
        best_j = -1
        for j, kp2 in enumerate(valid_kps2):
            dist = np.sum(kp1['descriptor'] != kp2['descriptor'])
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_dist < 50 and best_j >= 0:
            matches.append((i, best_j, best_dist))
    
    # Visualization
    fig, ax = plt.subplots(figsize=(18, 9))
    
    combined = np.zeros((h, w * 2))
    combined[:, :w] = img
    combined[:, w:] = img_rotated
    
    ax.imshow(combined, cmap='gray')
    
    # Draw keypoints
    for kp in kps[:100]:
        if 'descriptor' in kp:
            circle = Circle((kp['x'], kp['y']), 4, color='lime', fill=False, linewidth=1)
            ax.add_patch(circle)
    
    for kp in valid_kps2:
        circle = Circle((kp['x'] + w, kp['y']), 4, color='cyan', fill=False, linewidth=1)
        ax.add_patch(circle)
    
    # Draw matches
    for i, j, dist in matches[:40]:
        x1, y1 = kps[i]['x'], kps[i]['y']
        x2, y2 = valid_kps2[j]['x'] + w, valid_kps2[j]['y']
        color = plt.cm.jet(1 - dist / 50)
        ax.plot([x1, x2], [y1, y2], '-', color=color, linewidth=1.5, alpha=0.7)
    
    ax.set_title(f'ORB Step 6: Feature Matching\nOriginal vs {angle}° Rotated | {len(matches)} matches found', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step6_matching.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Final Summary
# =============================================================================
def generate_final_summary(kps):
    """Generate final summary image."""
    print("Generating: orb_final_result.png")
    img = load_image()
    
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img, cmap='gray')
    
    valid_kps = [kp for kp in kps if 'descriptor' in kp]
    
    for kp in valid_kps:
        x, y = kp['x'], kp['y']
        theta = kp.get('orientation', 0)
        
        circle = Circle((x, y), 6, color='lime', fill=False, linewidth=1.5)
        ax.add_patch(circle)
        
        arrow_len = 12
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=4, head_length=3, fc='red', ec='red', linewidth=1)
    
    ax.set_title(f'ORB Final Result: {len(valid_kps)} Keypoints with Descriptors', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_final_result.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Intermediate sub-steps
# =============================================================================
def generate_step2_substeps():
    """Generate FAST detection substep visualizations."""
    print("Generating: orb_step2_1_*.png, orb_step2_2_*.png")
    img = load_image()
    h, w = img.shape
    
    # Find a good corner for detailed visualization
    kps = fast_detect(img, threshold=0.08)
    kps = compute_harris(img, kps)
    kps = sorted(kps, key=lambda k: k.get('harris', 0), reverse=True)
    
    kp = None
    for k in kps:
        if 30 <= k['x'] < w - 30 and 30 <= k['y'] < h - 30:
            kp = k
            break
    
    if kp is None:
        return
    
    cx, cy = kp['x'], kp['y']
    
    # Step 2.1: High-speed test visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Full image with location
    ax1 = axes[0]
    ax1.imshow(img, cmap='gray')
    ax1.plot(cx, cy, 'r+', markersize=20, markeredgewidth=3)
    from matplotlib.patches import Rectangle
    rect = Rectangle((cx - 20, cy - 20), 40, 40, fill=False, color='yellow', linewidth=2)
    ax1.add_patch(rect)
    ax1.set_title('Candidate Point Location', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Zoomed with circle
    ax2 = axes[1]
    patch = img[max(0, cy-15):min(h, cy+16), max(0, cx-15):min(w, cx+16)]
    ax2.imshow(patch, cmap='gray')
    center = 15
    
    # Draw circle points
    threshold = 0.08
    center_val = patch[center, center]
    
    # Cardinal points (1, 5, 9, 13)
    cardinal_pos = [0, 4, 8, 12]
    for i, (dx, dy) in enumerate(CIRCLE_OFFSETS):
        px, py = center + dx, center + dy
        if 0 <= px < patch.shape[1] and 0 <= py < patch.shape[0]:
            val = patch[py, px]
            if i in cardinal_pos:
                if val > center_val + threshold:
                    color = 'lime'
                elif val < center_val - threshold:
                    color = 'red'
                else:
                    color = 'yellow'
                ax2.plot(px, py, 'o', color=color, markersize=12, markeredgecolor='black', markeredgewidth=2)
    
    ax2.plot(center, center, 's', color='blue', markersize=10)
    ax2.set_title('Step 2.1: High-Speed Test\n(Cardinal Points 1,5,9,13)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Full circle test
    ax3 = axes[2]
    ax3.imshow(patch, cmap='gray')
    
    for i, (dx, dy) in enumerate(CIRCLE_OFFSETS):
        px, py = center + dx, center + dy
        if 0 <= px < patch.shape[1] and 0 <= py < patch.shape[0]:
            val = patch[py, px]
            if val > center_val + threshold:
                color = 'lime'
            elif val < center_val - threshold:
                color = 'red'
            else:
                color = 'yellow'
            ax3.plot(px, py, 'o', color=color, markersize=10, markeredgecolor='black')
            ax3.text(px + 0.5, py - 0.5, str(i+1), fontsize=7, color='white')
    
    ax3.plot(center, center, 's', color='blue', markersize=10)
    ax3.set_title('Step 2.2: Full 16-Pixel Test\nGreen=Brighter, Red=Darker', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    fig.suptitle('ORB Step 2: FAST Corner Detection Detail', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step2_detail.png'), dpi=150, bbox_inches='tight')
    plt.close()


def generate_step3_substeps():
    """Generate Harris filtering substep visualizations."""
    print("Generating: orb_step3_substeps.png")
    img = load_image()
    
    # Compute gradients
    Ix = ndimage.sobel(img, axis=1)
    Iy = ndimage.sobel(img, axis=0)
    Ixx = ndimage.gaussian_filter(Ix * Ix, 1.5)
    Iyy = ndimage.gaussian_filter(Iy * Iy, 1.5)
    Ixy = ndimage.gaussian_filter(Ix * Iy, 1.5)
    
    # Harris response
    k = 0.04
    det = Ixx * Iyy - Ixy ** 2
    trace = Ixx + Iyy
    harris = det - k * trace ** 2
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Step 3.1a: Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ix
    axes[0, 1].imshow(Ix, cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[0, 1].set_title('Step 3.1b: Gradient Ix (Sobel)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Iy
    axes[0, 2].imshow(Iy, cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[0, 2].set_title('Step 3.1c: Gradient Iy (Sobel)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Ixx
    axes[1, 0].imshow(Ixx, cmap='hot')
    axes[1, 0].set_title('Step 3.2a: Ixx = Gσ(Ix²)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Iyy
    axes[1, 1].imshow(Iyy, cmap='hot')
    axes[1, 1].set_title('Step 3.2b: Iyy = Gσ(Iy²)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Harris response
    harris_vis = np.clip(harris, np.percentile(harris, 1), np.percentile(harris, 99))
    axes[1, 2].imshow(harris_vis, cmap='hot')
    axes[1, 2].set_title('Step 3.3: Harris R = det(M) - k·tr(M)²', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    fig.suptitle('ORB Step 3: Harris Corner Response Computation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step3_substeps.png'), dpi=150, bbox_inches='tight')
    plt.close()


def generate_step4_substeps():
    """Generate orientation substep visualizations."""
    print("Generating: orb_step4_substeps.png")
    img = load_image()
    h, w = img.shape
    
    kps = fast_detect(img, threshold=0.08)
    kps = compute_harris(img, kps)
    kps = [k for k in kps if k.get('harris', 0) > 0]
    kps = sorted(kps, key=lambda k: k.get('harris', 0), reverse=True)
    
    # Find good keypoint
    kp = None
    for k in kps:
        if 25 <= k['x'] < w - 25 and 25 <= k['y'] < h - 25:
            kp = k
            break
    
    if kp is None:
        return
    
    cx, cy = kp['x'], kp['y']
    radius = 15
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Patch
    patch = img[cy-radius:cy+radius+1, cx-radius:cx+radius+1]
    
    axes[0, 0].imshow(patch, cmap='gray')
    circle = Circle((radius, radius), radius, fill=False, color='cyan', linewidth=2)
    axes[0, 0].add_patch(circle)
    axes[0, 0].plot(radius, radius, 'b+', markersize=15, markeredgewidth=2)
    axes[0, 0].set_title('Step 4.1a: Circular Patch (r=15)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # x-weighted
    weighted_x = np.zeros_like(patch, dtype=np.float64)
    m10 = m01 = m00 = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                intensity = patch[radius + dy, radius + dx]
                weighted_x[radius + dy, radius + dx] = dx * intensity
                m10 += dx * intensity
                m01 += dy * intensity
                m00 += intensity
    
    axes[0, 1].imshow(weighted_x, cmap='RdBu')
    axes[0, 1].set_title(f'Step 4.1b: x·I(x,y)\nm₁₀ = {m10:.1f}', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # y-weighted
    weighted_y = np.zeros_like(patch, dtype=np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                weighted_y[radius + dy, radius + dx] = dy * patch[radius + dy, radius + dx]
    
    axes[0, 2].imshow(weighted_y, cmap='RdBu')
    axes[0, 2].set_title(f'Step 4.1c: y·I(x,y)\nm₀₁ = {m01:.1f}', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Centroid
    centroid_x = m10 / m00 if m00 > 0 else 0
    centroid_y = m01 / m00 if m00 > 0 else 0
    theta = np.arctan2(m01, m10)
    
    axes[1, 0].imshow(patch, cmap='gray')
    circle = Circle((radius, radius), radius, fill=False, color='cyan', linewidth=2)
    axes[1, 0].add_patch(circle)
    axes[1, 0].plot(radius, radius, 'b+', markersize=15, markeredgewidth=2, label='Center')
    axes[1, 0].plot(radius + centroid_x, radius + centroid_y, 'r*', markersize=15, label='Centroid')
    axes[1, 0].legend(loc='upper right', fontsize=9)
    axes[1, 0].set_title(f'Step 4.2: Centroid\nC = ({centroid_x:.2f}, {centroid_y:.2f})', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Orientation arrow
    axes[1, 1].imshow(patch, cmap='gray')
    circle = Circle((radius, radius), radius, fill=False, color='cyan', linewidth=2)
    axes[1, 1].add_patch(circle)
    arrow_len = 12
    axes[1, 1].arrow(radius, radius, arrow_len * np.cos(theta), arrow_len * np.sin(theta),
                    head_width=2, head_length=1.5, fc='red', ec='red', linewidth=2)
    axes[1, 1].set_title(f'Step 4.3: Orientation\nθ = {np.degrees(theta):.1f}°', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Full image
    axes[1, 2].imshow(img, cmap='gray')
    axes[1, 2].plot(cx, cy, 'r+', markersize=20, markeredgewidth=3)
    rect = Rectangle((cx - radius, cy - radius), radius*2, radius*2, fill=False, color='yellow', linewidth=2)
    axes[1, 2].add_patch(rect)
    arrow_len_full = 25
    axes[1, 2].arrow(cx, cy, arrow_len_full * np.cos(theta), arrow_len_full * np.sin(theta),
                    head_width=8, head_length=5, fc='red', ec='red', linewidth=2)
    axes[1, 2].set_title('Keypoint on Full Image', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    fig.suptitle('ORB Step 4: Orientation Assignment (Intensity Centroid)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step4_substeps.png'), dpi=150, bbox_inches='tight')
    plt.close()


def generate_step5_substeps(kps):
    """Generate descriptor substep visualizations."""
    print("Generating: orb_step5_substeps.png")
    img = load_image()
    h, w = img.shape
    pattern = get_pattern()
    
    # Find a good keypoint
    kp = None
    for k in kps:
        if 20 <= k['x'] < w - 20 and 20 <= k['y'] < h - 20:
            kp = k
            break
    
    if kp is None:
        return
    
    cx, cy = kp['x'], kp['y']
    theta = kp.get('orientation', 0)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Patch location
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(img, cmap='gray')
    rect = Rectangle((cx - 15, cy - 15), 31, 31, fill=False, color='yellow', linewidth=2)
    ax1.add_patch(rect)
    ax1.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2)
    ax1.set_title('Step 5.1: 31×31 Patch Location', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Patch with pattern (unrotated)
    ax2 = fig.add_subplot(2, 3, 2)
    half = 15
    if cy-half >= 0 and cy+half+1 <= h and cx-half >= 0 and cx+half+1 <= w:
        patch = img[cy-half:cy+half+1, cx-half:cx+half+1]
        ax2.imshow(patch, cmap='gray', alpha=0.5)
    
    for i, ((px, py), (qx, qy)) in enumerate(pattern[:20]):
        ax2.plot([px + half, qx + half], [py + half, qy + half], 'g-', alpha=0.5, linewidth=0.5)
        ax2.plot(px + half, py + half, 'ro', markersize=3)
        ax2.plot(qx + half, qy + half, 'bs', markersize=3)
    
    ax2.set_xlim(-1, 32)
    ax2.set_ylim(32, -1)
    ax2.set_title('Step 5.1: Original Pattern\n(First 20 pairs)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Rotated pattern
    ax3 = fig.add_subplot(2, 3, 3)
    if cy-half >= 0 and cy+half+1 <= h and cx-half >= 0 and cx+half+1 <= w:
        ax3.imshow(patch, cmap='gray', alpha=0.5)
    
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for i, ((px, py), (qx, qy)) in enumerate(pattern[:20]):
        rpx = px * cos_t - py * sin_t + half
        rpy = px * sin_t + py * cos_t + half
        rqx = qx * cos_t - qy * sin_t + half
        rqy = qx * sin_t + qy * cos_t + half
        ax3.plot([rpx, rqx], [rpy, rqy], 'g-', alpha=0.5, linewidth=0.5)
        ax3.plot(rpx, rpy, 'ro', markersize=3)
        ax3.plot(rqx, rqy, 'bs', markersize=3)
    
    ax3.set_xlim(-1, 32)
    ax3.set_ylim(32, -1)
    ax3.set_title(f'Step 5.2: Rotated by θ={np.degrees(theta):.1f}°', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. Binary comparisons
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis('off')
    
    text = "Step 5.3: Binary Comparisons\n" + "="*30 + "\n\n"
    for i in range(8):
        (px, py), (qx, qy) = pattern[i]
        rpx = int(round(px * cos_t - py * sin_t))
        rpy = int(round(px * sin_t + py * cos_t))
        rqx = int(round(qx * cos_t - qy * sin_t))
        rqy = int(round(qx * sin_t + qy * cos_t))
        
        p_x, p_y = cx + rpx, cy + rpy
        q_x, q_y = cx + rqx, cy + rqy
        
        if 0 <= p_x < w and 0 <= p_y < h and 0 <= q_x < w and 0 <= q_y < h:
            Ip = img[p_y, p_x]
            Iq = img[q_y, q_x]
            bit = 1 if Ip < Iq else 0
            text += f"Pair {i+1}: I(p)={Ip:.2f} {'<' if bit else '≥'} I(q)={Iq:.2f} → bit={bit}\n"
    
    ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # 5. Descriptor visualization
    ax5 = fig.add_subplot(2, 3, 5)
    desc = generate_descriptor(img, kp, pattern)
    ax5.imshow(desc.reshape(16, 16), cmap='binary', aspect='auto')
    ax5.set_title('256-bit Descriptor\n(16×16 visualization)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Bit column')
    ax5.set_ylabel('Bit row')
    
    # 6. Binary vector
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.bar(range(64), desc[:64], color='green', width=1)
    ax6.set_xlim(-1, 65)
    ax6.set_ylim(-0.1, 1.1)
    ax6.set_xlabel('Bit position')
    ax6.set_ylabel('Bit value')
    ax6.set_title('First 64 bits of descriptor', fontsize=12, fontweight='bold')
    
    fig.suptitle('ORB Step 5: rBRIEF Descriptor Generation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step5_substeps.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("ORB All Steps - Real Image Visualization")
    print("Generating step-by-step images")
    print("=" * 60)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Step 0: Input
    generate_step0_input()
    
    # Step 1: Pyramid
    pyramid, scales = generate_step1_pyramid_levels()
    
    # Step 2: FAST
    generate_step2_fast_all_levels(pyramid, scales)
    generate_step2_substeps()
    
    # Step 3: Harris
    all_harris_kps = generate_step3_harris(pyramid, scales)
    generate_step3_substeps()
    
    # Step 4: Orientation
    kps = generate_step4_orientation(pyramid, scales)
    generate_step4_substeps()
    
    # Step 5: Descriptors
    valid_kps = generate_step5_descriptors(kps)
    generate_step5_substeps(kps)
    
    # Step 6: Matching
    generate_step6_matching(valid_kps)
    
    # Final
    generate_final_summary(valid_kps)
    
    print("\n" + "=" * 60)
    print("All step-by-step visualizations generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()

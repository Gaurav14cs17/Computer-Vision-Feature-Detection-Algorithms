"""
ORB Orientation and Descriptor Detailed Visualization
Creates SIFT-style comprehensive orientation and descriptor images
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyArrow, Rectangle
from matplotlib.gridspec import GridSpec
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
            kp['m10'] = 0
            kp['m01'] = 0
            continue
        
        m10 = m01 = m00 = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    intensity = img[y + dy, x + dx]
                    m10 += dx * intensity
                    m01 += dy * intensity
                    m00 += intensity
        
        kp['orientation'] = np.arctan2(m01, m10)
        kp['m10'] = m10
        kp['m01'] = m01
        kp['m00'] = m00
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
# IMAGE 1: Orientation with arrows (like sift_step5_orientation.png)
# =============================================================================
def generate_orientation_arrows():
    """Generate image showing keypoints with orientation arrows."""
    print("Generating: orb_step4_orientation_arrows.png")
    img = load_image()
    
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
    
    ax.set_title(f'ORB Step 4: {len(kps)} Keypoints with Orientation', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step4_orientation_arrows.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return kps


# =============================================================================
# IMAGE 2: Intensity Centroid Method Detailed
# =============================================================================
def generate_centroid_detail(kps):
    """Generate detailed visualization of intensity centroid method."""
    print("Generating: orb_centroid_method_real.png")
    img = load_image()
    h, w = img.shape
    
    # Find a good keypoint
    kp = None
    for k in kps:
        if 25 <= k['x'] < w - 25 and 25 <= k['y'] < h - 25:
            kp = k
            break
    if kp is None:
        kp = kps[0]
    
    cx, cy = kp['x'], kp['y']
    
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Patch with centroid visualization
    ax1 = fig.add_subplot(2, 3, 1)
    radius = 15
    patch = img[cy-radius:cy+radius+1, cx-radius:cx+radius+1]
    ax1.imshow(patch, cmap='gray')
    
    # Compute centroid for this patch
    m10 = m01 = m00 = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                intensity = patch[radius + dy, radius + dx]
                m10 += dx * intensity
                m01 += dy * intensity
                m00 += intensity
    
    # Centroid location relative to center
    if m00 > 0:
        centroid_x = m10 / m00
        centroid_y = m01 / m00
    else:
        centroid_x = centroid_y = 0
    
    # Draw circular region
    circle = Circle((radius, radius), radius, fill=False, color='cyan', linewidth=2)
    ax1.add_patch(circle)
    
    # Draw center
    ax1.plot(radius, radius, 'b+', markersize=15, markeredgewidth=2, label='Center (C)')
    
    # Draw centroid
    ax1.plot(radius + centroid_x, radius + centroid_y, 'r*', markersize=15, label='Centroid')
    
    # Draw orientation arrow
    theta = np.arctan2(m01, m10)
    arrow_len = 12
    ax1.arrow(radius, radius, arrow_len * np.cos(theta), arrow_len * np.sin(theta),
             head_width=2, head_length=1.5, fc='yellow', ec='yellow', linewidth=2)
    
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_title(f'Patch with Centroid\nθ = {np.degrees(theta):.1f}°', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # 2. Formula explanation
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.axis('off')
    
    text = "Intensity Centroid Method\n"
    text += "="*30 + "\n\n"
    text += "Image Moments:\n"
    text += f"  m₁₀ = Σ x·I(x,y) = {m10:.2f}\n"
    text += f"  m₀₁ = Σ y·I(x,y) = {m01:.2f}\n"
    text += f"  m₀₀ = Σ I(x,y) = {m00:.2f}\n\n"
    text += "Centroid:\n"
    text += f"  C = (m₁₀/m₀₀, m₀₁/m₀₀)\n"
    text += f"    = ({centroid_x:.2f}, {centroid_y:.2f})\n\n"
    text += "Orientation:\n"
    text += f"  θ = atan2(m₀₁, m₁₀)\n"
    text += f"    = atan2({m01:.2f}, {m10:.2f})\n"
    text += f"    = {np.degrees(theta):.1f}°"
    
    ax2.text(0.1, 0.95, text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # 3. Weighted intensity visualization
    ax3 = fig.add_subplot(2, 3, 3)
    weighted_x = np.zeros_like(patch, dtype=np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                weighted_x[radius + dy, radius + dx] = dx * patch[radius + dy, radius + dx]
    ax3.imshow(weighted_x, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax3.set_title('x·I(x,y) Contribution\n(For m₁₀ computation)', fontsize=11, fontweight='bold')
    ax3.axis('off')
    
    # 4. Full image with orientation
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(img, cmap='gray')
    ax4.plot(cx, cy, 'r+', markersize=20, markeredgewidth=3)
    rect = Rectangle((cx - radius, cy - radius), radius*2, radius*2, 
                     fill=False, color='yellow', linewidth=2)
    ax4.add_patch(rect)
    
    # Draw orientation arrow on full image
    arrow_len_full = 25
    ax4.arrow(cx, cy, arrow_len_full * np.cos(theta), arrow_len_full * np.sin(theta),
             head_width=8, head_length=5, fc='red', ec='red', linewidth=2)
    
    ax4.set_title('Keypoint Location on Full Image', fontsize=11, fontweight='bold')
    ax4.axis('off')
    
    # 5. Orientation histogram for all keypoints
    ax5 = fig.add_subplot(2, 3, 5)
    orientations = [np.degrees(k['orientation']) for k in kps if 'orientation' in k]
    ax5.hist(orientations, bins=36, range=(-180, 180), color='steelblue', edgecolor='black')
    ax5.axvline(x=np.degrees(theta), color='red', linestyle='--', linewidth=2, 
               label=f'Current: {np.degrees(theta):.1f}°')
    ax5.set_xlabel('Orientation (degrees)')
    ax5.set_ylabel('Count')
    ax5.set_title('Orientation Distribution\n(All keypoints)', fontsize=11, fontweight='bold')
    ax5.legend()
    
    # 6. y-weighted contribution
    ax6 = fig.add_subplot(2, 3, 6)
    weighted_y = np.zeros_like(patch, dtype=np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                weighted_y[radius + dy, radius + dx] = dy * patch[radius + dy, radius + dx]
    ax6.imshow(weighted_y, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax6.set_title('y·I(x,y) Contribution\n(For m₀₁ computation)', fontsize=11, fontweight='bold')
    ax6.axis('off')
    
    fig.suptitle('ORB Orientation: Intensity Centroid Method (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_centroid_method_real.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# IMAGE 3: Descriptor Overview (like sift_descriptor_overview.png)
# =============================================================================
def generate_descriptor_overview(kps):
    """Generate descriptor overview similar to SIFT style."""
    print("Generating: orb_descriptor_overview.png")
    img = load_image()
    h, w = img.shape
    pattern = get_pattern()
    
    fig = plt.figure(figsize=(16, 10), facecolor='#2b2b2b')
    
    # Title
    fig.text(0.5, 0.97, 'ORB Descriptor', fontsize=24, ha='center', color='cyan', fontweight='bold')
    fig.text(0.5, 0.92, 'Binary intensity comparisons with rotation compensation', 
            fontsize=14, ha='center', color='yellow', fontstyle='italic')
    
    # 1. Image with keypoints
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img, cmap='gray')
    
    n_show = min(100, len(kps))
    if len(kps) > n_show:
        indices = np.linspace(0, len(kps)-1, n_show, dtype=int)
        kps_show = [kps[i] for i in indices]
    else:
        kps_show = kps
    
    # Highlight one keypoint
    kp_highlight = None
    for k in kps:
        if 20 <= k['x'] < w - 20 and 20 <= k['y'] < h - 20:
            kp_highlight = k
            break
    
    for kp in kps_show:
        x, y = kp['x'], kp['y']
        theta = kp.get('orientation', 0)
        
        if kp == kp_highlight:
            circle = Circle((x, y), 10, color='blue', fill=False, linewidth=3)
            arrow_len = 15
        else:
            circle = Circle((x, y), 6, color='white', fill=False, linewidth=1, linestyle='--')
            arrow_len = 10
        ax1.add_patch(circle)
        
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        color = 'red' if kp == kp_highlight else 'red'
        ax1.arrow(x, y, dx, dy, head_width=4, head_length=2, fc=color, ec=color, linewidth=1)
    
    ax1.axis('off')
    ax1.set_facecolor('#2b2b2b')
    
    # 2. Sampling pattern visualization
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_xlim(-20, 20)
    ax2.set_ylim(-20, 20)
    ax2.set_aspect('equal')
    ax2.set_facecolor('white')
    
    # Draw grid
    for i in range(-15, 16, 5):
        ax2.axhline(y=i, color='lightgreen', linewidth=0.5)
        ax2.axvline(x=i, color='lightgreen', linewidth=0.5)
    
    # Draw circle
    circle = Circle((0, 0), 15, fill=False, color='blue', linewidth=2)
    ax2.add_patch(circle)
    
    # Draw 4 quadrants
    for color, quad in [('lime', (0, 0, 15, 15)), ('orange', (0, -15, 15, 0)),
                        ('pink', (-15, 0, 0, 15)), ('yellow', (-15, -15, 0, 0))]:
        rect = Rectangle((quad[0], quad[1]), quad[2]-quad[0], quad[3]-quad[1],
                         fill=True, alpha=0.2, color=color, linewidth=2, edgecolor=color)
        ax2.add_patch(rect)
    
    # Draw some sampling pairs
    for i, ((px, py), (qx, qy)) in enumerate(pattern[:20]):
        ax2.plot([px, qx], [py, qy], 'k-', alpha=0.3, linewidth=0.5)
        ax2.plot(px, py, 'ko', markersize=4)
        ax2.plot(qx, qy, 'k^', markersize=4)
    
    # Draw orientation arrow
    ax2.arrow(0, 0, 0, 12, head_width=2, head_length=1.5, fc='red', ec='red', linewidth=2)
    
    ax2.set_title('Binary sampling pattern', fontsize=12, color='black')
    
    # 3. Binary descriptor visualization
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Generate descriptor for highlighted keypoint
    if kp_highlight:
        desc = generate_descriptor(img, kp_highlight, pattern)
        
        # Show as colored bars
        n_bins = 32
        for i in range(8):
            for j in range(n_bins):
                idx = i * n_bins + j
                if idx < len(desc):
                    color = 'steelblue' if desc[idx] == 1 else 'lightgray'
                    rect = Rectangle((j, 7-i), 1, 1, fill=True, color=color, edgecolor='gray', linewidth=0.2)
                    ax3.add_patch(rect)
        
        ax3.set_xlim(0, n_bins)
        ax3.set_ylim(0, 8)
        ax3.set_xlabel('Bit position within row')
        ax3.set_ylabel('Row')
        ax3.set_title('256-bit Binary Descriptor\n(Blue=1, Gray=0)', fontsize=12)
        ax3.set_facecolor('#f0f0f0')
    
    # 4. Comparison equation
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    ax4.set_facecolor('#2b2b2b')
    
    text = "rBRIEF Binary Test:\n\n"
    text += "For each pair (p, q) in pattern:\n\n"
    text += "  1. Rotate by orientation θ:\n"
    text += "     p' = R(θ) · p\n"
    text += "     q' = R(θ) · q\n\n"
    text += "  2. Compare intensities:\n"
    text += "     bit = 1 if I(p') < I(q')\n"
    text += "     bit = 0 otherwise\n\n"
    text += "Result: 256-bit binary vector\n"
    text += "        (32 bytes)"
    
    ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            color='white', bbox=dict(boxstyle='round', facecolor='#404040', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_descriptor_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# IMAGE 4: Matching visualization
# =============================================================================
def generate_matching_visualization(kps):
    """Generate matching visualization between two versions of image."""
    print("Generating: orb_matching_visualization.png")
    img = load_image()
    h, w = img.shape
    pattern = get_pattern()
    
    # Create rotated version
    angle = 15
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_rotated = pil_img.rotate(angle, expand=False)
    img_rotated = np.array(pil_rotated) / 255.0
    
    # Detect keypoints on rotated image
    kps2 = fast_detect(img_rotated, threshold=0.08)
    kps2 = compute_harris(img_rotated, kps2)
    kps2 = [k for k in kps2 if k.get('harris', 0) > 0]
    kps2 = sorted(kps2, key=lambda k: k.get('harris', 0), reverse=True)[:300]
    kps2 = compute_orientation(img_rotated, kps2)
    
    # Generate descriptors
    descs1 = []
    valid_kps1 = []
    for kp in kps[:300]:
        if 16 <= kp['x'] < w - 16 and 16 <= kp['y'] < h - 16:
            desc = generate_descriptor(img, kp, pattern)
            descs1.append(desc)
            valid_kps1.append(kp)
    
    descs2 = []
    valid_kps2 = []
    for kp in kps2:
        if 16 <= kp['x'] < w - 16 and 16 <= kp['y'] < h - 16:
            desc = generate_descriptor(img_rotated, kp, pattern)
            descs2.append(desc)
            valid_kps2.append(kp)
    
    # Find matches using Hamming distance
    matches = []
    for i, d1 in enumerate(descs1[:100]):
        best_dist = 256
        best_j = -1
        for j, d2 in enumerate(descs2):
            dist = np.sum(d1 != d2)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_dist < 50:  # threshold
            matches.append((i, best_j, best_dist))
    
    # Visualization
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Combine images side by side
    combined = np.zeros((h, w * 2))
    combined[:, :w] = img
    combined[:, w:] = img_rotated
    
    ax.imshow(combined, cmap='gray')
    
    # Draw keypoints
    for kp in valid_kps1[:100]:
        circle = Circle((kp['x'], kp['y']), 4, color='lime', fill=False, linewidth=1)
        ax.add_patch(circle)
    
    for kp in valid_kps2:
        circle = Circle((kp['x'] + w, kp['y']), 4, color='cyan', fill=False, linewidth=1)
        ax.add_patch(circle)
    
    # Draw matches
    for i, j, dist in matches[:30]:
        x1, y1 = valid_kps1[i]['x'], valid_kps1[i]['y']
        x2, y2 = valid_kps2[j]['x'] + w, valid_kps2[j]['y']
        
        color = plt.cm.jet(1 - dist / 50)  # Better match = warmer color
        ax.plot([x1, x2], [y1, y2], '-', color=color, linewidth=1, alpha=0.7)
    
    ax.set_title(f'ORB Feature Matching\nOriginal vs {angle}° Rotated | {len(matches)} matches (Hamming distance < 50)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_matching_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# IMAGE 5: Final pipeline summary
# =============================================================================
def generate_final_pipeline(kps):
    """Generate final complete pipeline visualization."""
    print("Generating: orb_complete_pipeline.png")
    img = load_image()
    h, w = img.shape
    pattern = get_pattern()
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.25, wspace=0.15)
    
    # 1. Input Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img, cmap='gray')
    ax1.set_title('1. Input Image\n640×480', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # 2. Pyramid
    ax2 = fig.add_subplot(gs[0, 1])
    from PIL import Image as PILImage
    pyramid_viz = np.ones((h, w)) * 0.5
    pyramid_img = PILImage.fromarray((img * 255).astype(np.uint8))
    x_off, y_off = 0, 0
    for level in range(4):
        scale = 1.0 / (1.2 ** level)
        new_w, new_h = int(w * scale * 0.5), int(h * scale * 0.5)
        resized = pyramid_img.resize((new_w, new_h), PILImage.LANCZOS)
        arr = np.array(resized) / 255.0
        if y_off + new_h <= h and x_off + new_w <= w:
            pyramid_viz[y_off:y_off+new_h, x_off:x_off+new_w] = arr
        x_off += new_w + 5
        if x_off + new_w > w:
            x_off = 0
            y_off += new_h + 5
    ax2.imshow(pyramid_viz, cmap='gray')
    ax2.set_title('2. Scale Pyramid\n8 levels, factor=1.2', fontsize=11, fontweight='bold')
    ax2.axis('off')
    
    # 3. FAST Corners
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img, cmap='gray')
    kps_fast = fast_detect(img, threshold=0.08)
    n_show = min(500, len(kps_fast))
    indices = np.linspace(0, len(kps_fast)-1, n_show, dtype=int)
    for i in indices:
        kp = kps_fast[i]
        circle = Circle((kp['x'], kp['y']), 2, color='yellow', fill=False, linewidth=0.5)
        ax3.add_patch(circle)
    ax3.set_title(f'3. FAST Detection\n{len(kps_fast)} corners', fontsize=11, fontweight='bold')
    ax3.axis('off')
    
    # 4. Harris Filtered
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(img, cmap='gray')
    kps_harris = compute_harris(img, kps_fast)
    kps_harris = [k for k in kps_harris if k.get('harris', 0) > 0]
    kps_harris = sorted(kps_harris, key=lambda k: k.get('harris', 0), reverse=True)[:500]
    for kp in kps_harris:
        circle = Circle((kp['x'], kp['y']), 3, color='lime', fill=False, linewidth=0.8)
        ax4.add_patch(circle)
    ax4.set_title(f'4. Harris Filter\n{len(kps_harris)} keypoints', fontsize=11, fontweight='bold')
    ax4.axis('off')
    
    # 5. Orientation
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(img, cmap='gray')
    kps_orient = compute_orientation(img, kps_harris)
    for kp in kps_orient[:300]:
        x, y = kp['x'], kp['y']
        theta = kp.get('orientation', 0)
        circle = Circle((x, y), 4, color='lime', fill=False, linewidth=1)
        ax5.add_patch(circle)
        dx, dy = 8 * np.cos(theta), 8 * np.sin(theta)
        ax5.arrow(x, y, dx, dy, head_width=3, head_length=2, fc='red', ec='red', linewidth=0.8)
    ax5.set_title('5. Orientation\n(Intensity Centroid)', fontsize=11, fontweight='bold')
    ax5.axis('off')
    
    # 6. Descriptors
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(img, cmap='gray')
    
    valid_kps = []
    for kp in kps_orient:
        if 16 <= kp['x'] < w - 16 and 16 <= kp['y'] < h - 16:
            desc = generate_descriptor(img, kp, pattern)
            kp['descriptor'] = desc
            valid_kps.append(kp)
    
    for kp in valid_kps[:300]:
        x, y = kp['x'], kp['y']
        theta = kp.get('orientation', 0)
        circle = Circle((x, y), 4, color='cyan', fill=False, linewidth=1)
        ax6.add_patch(circle)
        dx, dy = 8 * np.cos(theta), 8 * np.sin(theta)
        ax6.arrow(x, y, dx, dy, head_width=3, head_length=2, fc='yellow', ec='yellow', linewidth=0.8)
    ax6.set_title(f'6. rBRIEF Descriptors\n{len(valid_kps)} with 256-bit desc', fontsize=11, fontweight='bold')
    ax6.axis('off')
    
    # 7. Sample descriptors visualization
    ax7 = fig.add_subplot(gs[1, 2])
    if len(valid_kps) >= 5:
        for i in range(5):
            desc = valid_kps[i]['descriptor']
            for j in range(64):
                color = 'green' if desc[j] == 1 else 'lightgray'
                rect = Rectangle((j, 4-i), 1, 0.8, fill=True, color=color, edgecolor='none')
                ax7.add_patch(rect)
        ax7.set_xlim(0, 64)
        ax7.set_ylim(0, 5)
        ax7.set_xlabel('Bit position (first 64)')
        ax7.set_ylabel('Keypoint')
        ax7.set_title('7. Sample Descriptors\n(Green=1, Gray=0)', fontsize=11, fontweight='bold')
    
    # 8. Summary stats
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    
    stats = f"""ORB Pipeline Summary
═══════════════════════

Input:     {w}×{h} image
Pyramid:   8 levels
FAST:      {len(kps_fast)} corners
Harris:    {len(kps_harris)} filtered
Final:     {len(valid_kps)} keypoints

Descriptor:
  • Size: 256 bits (32 bytes)
  • Type: Binary (rBRIEF)
  • Matching: Hamming distance

Advantages:
  • Very fast computation
  • Rotation invariant
  • Scale invariant
  • Compact storage"""
    
    ax8.text(0.05, 0.95, stats, transform=ax8.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    fig.suptitle('ORB Feature Extraction Pipeline', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_complete_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("ORB Orientation and Descriptor Visualization")
    print("=" * 60)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    kps = generate_orientation_arrows()
    generate_centroid_detail(kps)
    generate_descriptor_overview(kps)
    generate_matching_visualization(kps)
    generate_final_pipeline(kps)
    
    print("\n" + "=" * 60)
    print("All orientation/descriptor visualizations generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()

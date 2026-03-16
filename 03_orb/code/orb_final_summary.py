"""
Generate final ORB pipeline summary image showing detection + description results.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, sobel

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, '..', 'images')

# Bresenham circle offsets
CIRCLE_OFFSETS = [
    (0, -3), (1, -3), (2, -2), (3, -1),
    (3, 0), (3, 1), (2, 2), (1, 3),
    (0, 3), (-1, 3), (-2, 2), (-3, 1),
    (-3, 0), (-3, -1), (-2, -2), (-1, -3)
]


def load_real_image():
    """Load the real input image."""
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    if os.path.exists(image_path):
        img_rgb = np.array(Image.open(image_path))
        if len(img_rgb.shape) == 3:
            img = (0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]) / 255.0
        else:
            img = img_rgb / 255.0
        return img
    return None

def create_test_image():
    """Create a test image with features (fallback if no real image)."""
    # Try to load real image first
    real_img = load_real_image()
    if real_img is not None:
        return real_img
    
    # Fallback to synthetic
    img = np.zeros((240, 320), dtype=np.float64)
    img[:, :] = 0.5
    
    # Add shapes
    img[40:90, 40:90] = 0.9
    img[100:150, 180:250] = 0.3
    img[140:190, 60:120] = 0.75
    img[50:80, 200:260] = 0.85
    
    # Add texture
    for i in range(0, 240, 15):
        for j in range(0, 320, 15):
            img[i:i+8, j:j+8] += np.random.rand() * 0.08 - 0.04
    
    return np.clip(img, 0, 1)


def detect_fast_corners(img, threshold=0.08):
    """Simple FAST corner detection."""
    h, w = img.shape
    corners = []
    
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            center = img[y, x]
            upper = center + threshold
            lower = center - threshold
            
            circle_vals = [img[y + dy, x + dx] for dx, dy in CIRCLE_OFFSETS]
            
            labels = []
            for val in circle_vals:
                if val > upper:
                    labels.append('B')
                elif val < lower:
                    labels.append('D')
                else:
                    labels.append('S')
            
            labels_ext = labels + labels
            max_b = max_d = 0
            cnt_b = cnt_d = 0
            
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
    Ix = sobel(img, axis=1)
    Iy = sobel(img, axis=0)
    
    Ixx = gaussian_filter(Ix * Ix, 1.5)
    Iyy = gaussian_filter(Iy * Iy, 1.5)
    Ixy = gaussian_filter(Ix * Iy, 1.5)
    
    for kp in keypoints:
        x, y = kp['x'], kp['y']
        det = Ixx[y, x] * Iyy[y, x] - Ixy[y, x] ** 2
        trace = Ixx[y, x] + Iyy[y, x]
        kp['harris'] = det - k * trace ** 2
    
    return keypoints


def compute_orientation(img, keypoints, radius=15):
    """Compute orientation using intensity centroid."""
    h, w = img.shape
    
    for kp in keypoints:
        cx, cy = int(kp['x']), int(kp['y'])
        m10 = m01 = 0
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < w and 0 <= y < h:
                    intensity = img[y, x]
                    m10 += dx * intensity
                    m01 += dy * intensity
        
        kp['orientation'] = np.arctan2(m01, m10)
    
    return keypoints


def generate_descriptor(img, kp, n_pairs=256):
    """Generate rBRIEF descriptor."""
    np.random.seed(42)
    h, w = img.shape
    cx, cy = int(kp['x']), int(kp['y'])
    theta = kp.get('orientation', 0)
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    descriptor = np.zeros(n_pairs, dtype=np.uint8)
    
    for i in range(n_pairs):
        px = int(np.clip(np.random.randn() * 7, -15, 15))
        py = int(np.clip(np.random.randn() * 7, -15, 15))
        qx = int(np.clip(np.random.randn() * 7, -15, 15))
        qy = int(np.clip(np.random.randn() * 7, -15, 15))
        
        rpx = int(round(px * cos_t - py * sin_t))
        rpy = int(round(px * sin_t + py * cos_t))
        rqx = int(round(qx * cos_t - qy * sin_t))
        rqy = int(round(qx * sin_t + qy * cos_t))
        
        p_x, p_y = cx + rpx, cy + rpy
        q_x, q_y = cx + rqx, cy + rqy
        
        if 0 <= p_x < w and 0 <= p_y < h and 0 <= q_x < w and 0 <= q_y < h:
            if img[p_y, p_x] < img[q_y, q_x]:
                descriptor[i] = 1
    
    return descriptor


def create_final_summary():
    """Create final ORB summary visualization."""
    print("Creating test image...")
    img = create_test_image()
    
    print("Detecting FAST corners...")
    fast_corners = detect_fast_corners(img)
    print(f"  Found {len(fast_corners)} corners")
    
    print("Computing Harris response...")
    harris_corners = compute_harris(img, fast_corners)
    
    print("Selecting top keypoints...")
    harris_corners = sorted(harris_corners, key=lambda k: k.get('harris', 0), reverse=True)
    selected = harris_corners[:200]
    print(f"  Selected {len(selected)} keypoints")
    
    print("Computing orientations...")
    oriented = compute_orientation(img, selected)
    
    print("Generating descriptors...")
    valid_kps = []
    descriptors = []
    h, w = img.shape
    
    for kp in oriented:
        if 16 <= kp['x'] < w - 16 and 16 <= kp['y'] < h - 16:
            desc = generate_descriptor(img, kp)
            kp['descriptor'] = desc
            valid_kps.append(kp)
            descriptors.append(desc)
    
    print(f"  Generated {len(valid_kps)} descriptors")
    
    # Create visualization
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.2)
    
    # 1. Input Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f'1. Input Image\n{w}×{h} pixels', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. FAST Corners - sample evenly if too many
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img, cmap='gray', vmin=0, vmax=1)
    n_show = min(800, len(fast_corners))
    if len(fast_corners) > n_show:
        indices = np.linspace(0, len(fast_corners) - 1, n_show, dtype=int)
        fast_to_show = [fast_corners[i] for i in indices]
    else:
        fast_to_show = fast_corners
    for kp in fast_to_show:
        circle = plt.Circle((kp['x'], kp['y']), 2, color='yellow', fill=False, linewidth=0.5)
        ax2.add_patch(circle)
    ax2.set_title(f'2. FAST Corners\n{len(fast_corners)} detected', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Harris Filtered
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img, cmap='gray', vmin=0, vmax=1)
    for kp in selected:
        circle = plt.Circle((kp['x'], kp['y']), 4, color='cyan', fill=False, linewidth=1)
        ax3.add_patch(circle)
    ax3.set_title(f'3. Harris Filtered\n{len(selected)} keypoints', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. With Orientation
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(img, cmap='gray', vmin=0, vmax=1)
    for kp in valid_kps[:100]:
        x, y = kp['x'], kp['y']
        theta = kp.get('orientation', 0)
        
        circle = plt.Circle((x, y), 5, color='lime', fill=False, linewidth=1.5)
        ax4.add_patch(circle)
        
        arrow_len = 10
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        ax4.arrow(x, y, dx, dy, head_width=3, head_length=2, fc='red', ec='red')
    
    ax4.set_title(f'4. Orientation Assignment\n{len(valid_kps)} with θ', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 5. Descriptor visualization
    ax5 = fig.add_subplot(gs[1, 1])
    if len(descriptors) > 0:
        desc_img = np.array(descriptors[:16]).reshape(-1)[:256].reshape(16, 16)
        ax5.imshow(desc_img, cmap='Greens', aspect='auto')
        ax5.set_title('5. rBRIEF Descriptors\n256-bit binary (16×16 viz)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    ax6.axis('off')
    
    summary_text = f"""
ORB Pipeline Summary
════════════════════

Input: {w} × {h} pixels

1️⃣ DETECTION PHASE:
  • FAST corners: {len(fast_corners)}
  • Harris filtered: {len(selected)}
  • With orientation: {len(oriented)}

2️⃣ DESCRIPTION PHASE:
  • Valid keypoints: {len(valid_kps)}
  • Descriptor size: 256 bits
  • Storage: {len(valid_kps) * 32} bytes

Key Features:
  ✓ Fast: ~30× faster than SIFT
  ✓ Rotation invariant
  ✓ Scale invariant
  ✓ Compact binary descriptors
"""
    ax6.text(0.5, 5, summary_text, fontsize=10, va='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f8f5', edgecolor='#1abc9c', linewidth=2))
    ax6.set_title('6. Pipeline Summary', fontsize=12, fontweight='bold')
    
    plt.suptitle('ORB Complete Pipeline: Detection → Description', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(OUT_DIR, 'orb_complete_summary.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_complete_summary.png")


def create_orb_vs_sift_summary():
    """Create comparison summary between ORB and SIFT."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, 'ORB vs SIFT: Algorithm Comparison', ha='center', 
           fontsize=18, fontweight='bold')
    
    # ORB column
    orb_box = patches.FancyBboxPatch((0.5, 1), 7, 9.5, boxstyle="round,pad=0.1",
                                      facecolor='#e8f8f5', edgecolor='#1abc9c', linewidth=3)
    ax.add_patch(orb_box)
    ax.text(4, 10, 'ORB', ha='center', fontsize=16, fontweight='bold', color='#1abc9c')
    
    orb_content = """
Year: 2011 (Rublee et al.)

DETECTION:
  • FAST corners (16-pixel circle)
  • Harris corner response
  • Intensity centroid orientation
  • Image pyramid (f=1.2)

DESCRIPTION:
  • rBRIEF (256-bit binary)
  • Rotated sampling pattern
  • Pre-trained point pairs

MATCHING:
  • Hamming distance
  • XOR + popcount
  • ~8 CPU cycles

PERFORMANCE:
  • Speed: ~30 fps
  • Storage: 32 bytes/descriptor
  • Free (no patents)
"""
    ax.text(1, 9, orb_content, fontsize=9, va='top', family='monospace')
    
    # SIFT column
    sift_box = patches.FancyBboxPatch((8.5, 1), 7, 9.5, boxstyle="round,pad=0.1",
                                       facecolor='#fdedec', edgecolor='#e74c3c', linewidth=3)
    ax.add_patch(sift_box)
    ax.text(12, 10, 'SIFT', ha='center', fontsize=16, fontweight='bold', color='#e74c3c')
    
    sift_content = """
Year: 2004 (David Lowe)

DETECTION:
  • DoG extrema (26-neighbor)
  • Low contrast removal
  • Edge response removal
  • Gaussian scale space

DESCRIPTION:
  • 128-D float vector
  • 4×4 subregions × 8 bins
  • Gradient histograms

MATCHING:
  • L2 (Euclidean) distance
  • 128 multiply-adds
  • ~500 CPU cycles

PERFORMANCE:
  • Speed: ~1 fps
  • Storage: 512 bytes/descriptor
  • Patent expired (2020)
"""
    ax.text(9, 9, sift_content, fontsize=9, va='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_vs_sift_summary.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_vs_sift_summary.png")


def main():
    """Generate final summary visualizations."""
    print("=" * 60)
    print("ORB Final Summary Visualizations")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("\n1. Generating complete pipeline summary...")
    create_final_summary()
    
    print("\n2. Generating ORB vs SIFT comparison...")
    create_orb_vs_sift_summary()
    
    print("\n" + "=" * 60)
    print("Done! Generated all summary visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()

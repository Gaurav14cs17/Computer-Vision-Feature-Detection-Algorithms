"""
ORB Step-by-Step Visualization with Real Images
Generates detailed step images similar to SIFT implementation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy import ndimage

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, '..', 'images')

# Bresenham circle offsets (16 pixels)
CIRCLE_OFFSETS = [
    (0, -3), (1, -3), (2, -2), (3, -1),
    (3, 0), (3, 1), (2, 2), (1, 3),
    (0, 3), (-1, 3), (-2, 2), (-3, 1),
    (-3, 0), (-3, -1), (-2, -2), (-1, -3)
]


def load_image():
    """Load the real input image."""
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    img_rgb = np.array(Image.open(image_path))
    if len(img_rgb.shape) == 3:
        gray = (0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]) / 255.0
    else:
        gray = img_rgb / 255.0
    return gray, img_rgb


# =============================================================================
# STEP 1: Scale Pyramid - Detailed Visualization
# =============================================================================
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


def visualize_step1_pyramid_detail():
    """Step 1: Detailed pyramid visualization like SIFT."""
    print("Generating Step 1: Scale Pyramid (detailed)...")
    gray, _ = load_image()
    pyramid, scales = build_pyramid(gray)
    
    # Create detailed grid showing all levels
    n_levels = len(pyramid)
    fig = plt.figure(figsize=(20, 12))
    
    # Show pyramid levels in grid
    cols = 4
    rows = (n_levels + cols - 1) // cols
    
    for i, (level_img, scale) in enumerate(zip(pyramid, scales)):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(level_img, cmap='gray')
        h, w = level_img.shape
        ax.set_title(f'Level {i}\n{w}×{h}\nscale={scale:.3f}', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle('ORB Step 1: Scale-Space Pyramid\n(Scale Factor = 1.2, Direct Downsampling)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step1_pyramid_detail.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: orb_step1_pyramid_detail.png")
    
    return pyramid, scales


# =============================================================================
# STEP 2: FAST Corner Detection - Per Level Visualization
# =============================================================================
def fast_corner_test(img, x, y, threshold=20, n_contiguous=9):
    """FAST corner test at a single pixel."""
    h, w = img.shape
    if x < 3 or x >= w - 3 or y < 3 or y >= h - 3:
        return False, 0
    
    center = img[y, x]
    upper = center + threshold / 255.0
    lower = center - threshold / 255.0
    
    # High-speed test
    test_positions = [0, 4, 8, 12]
    n_brighter = sum(1 for pos in test_positions if img[y + CIRCLE_OFFSETS[pos][1], x + CIRCLE_OFFSETS[pos][0]] > upper)
    n_darker = sum(1 for pos in test_positions if img[y + CIRCLE_OFFSETS[pos][1], x + CIRCLE_OFFSETS[pos][0]] < lower)
    
    if n_brighter < 3 and n_darker < 3:
        return False, 0
    
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
    
    labels_extended = labels + labels
    max_b = max_d = count_b = count_d = 0
    
    for label in labels_extended:
        if label == 'B':
            count_b += 1
            count_d = 0
            max_b = max(max_b, count_b)
        elif label == 'D':
            count_d += 1
            count_b = 0
            max_d = max(max_d, count_d)
        else:
            count_b = count_d = 0
    
    max_b = min(max_b, 16)
    max_d = min(max_d, 16)
    response = max(max_b, max_d)
    
    return (max_b >= n_contiguous or max_d >= n_contiguous), response


def detect_fast_corners(img, threshold=20):
    """Detect FAST corners in image."""
    h, w = img.shape
    keypoints = []
    
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            is_corner, response = fast_corner_test(img, x, y, threshold)
            if is_corner:
                keypoints.append({'x': x, 'y': y, 'response': response})
    
    return keypoints


def visualize_step2_fast_per_level(pyramid, scales):
    """Step 2: FAST detection per pyramid level."""
    print("Generating Step 2: FAST Corners per Level...")
    
    n_levels = min(len(pyramid), 6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    all_keypoints = []
    
    for i in range(n_levels):
        ax = axes[i]
        level_img = pyramid[i]
        
        # Detect corners at this level
        kps = detect_fast_corners(level_img, threshold=25)
        all_keypoints.append(kps)
        
        ax.imshow(level_img, cmap='gray')
        
        # Sample keypoints for display
        n_show = min(500, len(kps))
        if len(kps) > n_show:
            indices = np.linspace(0, len(kps) - 1, n_show, dtype=int)
            kps_show = [kps[j] for j in indices]
        else:
            kps_show = kps
        
        for kp in kps_show:
            circle = Circle((kp['x'], kp['y']), 2, color='lime', fill=False, linewidth=0.8)
            ax.add_patch(circle)
        
        h, w = level_img.shape
        ax.set_title(f'Level {i}: {w}×{h}\n{len(kps)} corners (scale={scales[i]:.3f})', 
                     fontsize=10, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle('ORB Step 2: FAST Corner Detection Per Pyramid Level\n(Threshold = 25, FAST-9)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step2_fast_per_level.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: orb_step2_fast_per_level.png")
    
    return all_keypoints


def visualize_step2_fast_combined(gray, pyramid, scales):
    """Step 2b: Combined FAST corners from all levels."""
    print("Generating Step 2b: Combined FAST Corners...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    # Left: Original image with all corners
    ax1 = axes[0]
    ax1.imshow(gray, cmap='gray')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(pyramid)))
    total_corners = 0
    
    for i, (level_img, scale) in enumerate(zip(pyramid[:6], scales[:6])):
        kps = detect_fast_corners(level_img, threshold=25)
        total_corners += len(kps)
        
        # Map coordinates back to original scale
        n_show = min(300, len(kps))
        if len(kps) > n_show:
            indices = np.linspace(0, len(kps) - 1, n_show, dtype=int)
            kps_show = [kps[j] for j in indices]
        else:
            kps_show = kps
        
        for kp in kps_show:
            orig_x = kp['x'] / scale
            orig_y = kp['y'] / scale
            radius = 3 + i * 2
            circle = Circle((orig_x, orig_y), radius, color=colors[i], fill=False, linewidth=1)
            ax1.add_patch(circle)
    
    ax1.set_title(f'All FAST Corners Mapped to Original\n({total_corners} total from {len(pyramid)} levels)', 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Add legend
    for i in range(min(6, len(pyramid))):
        ax1.plot([], [], 'o', color=colors[i], label=f'Level {i} (scale={scales[i]:.2f})')
    ax1.legend(loc='upper right', fontsize=8)
    
    # Right: Statistics
    ax2 = axes[1]
    ax2.axis('off')
    
    # Create stats box
    stats_text = "ORB FAST Detection Summary\n" + "="*40 + "\n\n"
    stats_text += f"Pyramid Levels: {len(pyramid)}\n"
    stats_text += f"Scale Factor: 1.2\n"
    stats_text += f"FAST Threshold: 25\n"
    stats_text += f"Contiguous Pixels: 9 (FAST-9)\n\n"
    stats_text += "Corners per Level:\n"
    
    for i, (level_img, scale) in enumerate(zip(pyramid[:6], scales[:6])):
        kps = detect_fast_corners(level_img, threshold=25)
        h, w = level_img.shape
        stats_text += f"  Level {i}: {len(kps):5d} corners ({w}×{h})\n"
    
    stats_text += f"\nTotal: {total_corners} corners\n"
    
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    fig.suptitle('ORB Step 2: Multi-Scale FAST Corner Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_step2_fast_combined.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: orb_step2_fast_combined.png")


# =============================================================================
# STEP 3: Harris Corner Response
# =============================================================================
def compute_harris_response(img, keypoints, k=0.04):
    """Compute Harris response for keypoints."""
    Ix = ndimage.sobel(img, axis=1)
    Iy = ndimage.sobel(img, axis=0)
    
    Ixx = ndimage.gaussian_filter(Ix * Ix, 1.5)
    Iyy = ndimage.gaussian_filter(Iy * Iy, 1.5)
    Ixy = ndimage.gaussian_filter(Ix * Iy, 1.5)
    
    h, w = img.shape
    for kp in keypoints:
        x, y = kp['x'], kp['y']
        if 0 <= x < w and 0 <= y < h:
            det_M = Ixx[y, x] * Iyy[y, x] - Ixy[y, x] ** 2
            trace_M = Ixx[y, x] + Iyy[y, x]
            kp['harris'] = det_M - k * (trace_M ** 2)
        else:
            kp['harris'] = 0
    
    return keypoints


def visualize_step3_harris_detail(gray):
    """Step 3: Harris corner response visualization."""
    print("Generating Step 3: Harris Corner Response...")
    
    # Detect FAST corners
    fast_kps = detect_fast_corners(gray, threshold=25)
    print(f"  FAST corners: {len(fast_kps)}")
    
    # Compute Harris
    harris_kps = compute_harris_response(gray, fast_kps)
    
    # Compute gradient images for visualization
    Ix = ndimage.sobel(gray, axis=1)
    Iy = ndimage.sobel(gray, axis=0)
    Ixx = ndimage.gaussian_filter(Ix * Ix, 1.5)
    Iyy = ndimage.gaussian_filter(Iy * Iy, 1.5)
    Ixy = ndimage.gaussian_filter(Ix * Iy, 1.5)
    
    # Harris response image
    det_M = Ixx * Iyy - Ixy ** 2
    trace_M = Ixx + Iyy
    harris_response = det_M - 0.04 * (trace_M ** 2)
    
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    # Row 1: Gradient computation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(gray, cmap='gray')
    ax1.set_title('Original Image', fontsize=10, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(Ix, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax2.set_title('Gradient Ix (Sobel X)', fontsize=10, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(Iy, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax3.set_title('Gradient Iy (Sobel Y)', fontsize=10, fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(harris_response, cmap='hot', vmin=0)
    ax4.set_title('Harris Response R', fontsize=10, fontweight='bold')
    ax4.axis('off')
    
    # Row 2: Structure tensor components
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(Ixx, cmap='hot')
    ax5.set_title('Ixx (smoothed)', fontsize=10, fontweight='bold')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(Iyy, cmap='hot')
    ax6.set_title('Iyy (smoothed)', fontsize=10, fontweight='bold')
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(Ixy, cmap='RdBu')
    ax7.set_title('Ixy (smoothed)', fontsize=10, fontweight='bold')
    ax7.axis('off')
    
    # Formula box
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    formula_text = "Harris Corner Response\n" + "="*30 + "\n\n"
    formula_text += "Structure Tensor M:\n"
    formula_text += "  M = [Ixx  Ixy]\n"
    formula_text += "      [Ixy  Iyy]\n\n"
    formula_text += "Response:\n"
    formula_text += "  R = det(M) - k·trace(M)²\n"
    formula_text += "  R = Ixx·Iyy - Ixy² - k(Ixx+Iyy)²\n\n"
    formula_text += "k = 0.04 (Harris constant)\n\n"
    formula_text += "R > 0 → Corner\n"
    formula_text += "R < 0 → Edge\n"
    formula_text += "R ≈ 0 → Flat"
    ax8.text(0.1, 0.95, formula_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Row 3: FAST corners before and after Harris filtering
    ax9 = fig.add_subplot(gs[2, 0:2])
    ax9.imshow(gray, cmap='gray')
    n_show = min(800, len(fast_kps))
    indices = np.linspace(0, len(fast_kps) - 1, n_show, dtype=int) if len(fast_kps) > n_show else range(len(fast_kps))
    for i in indices:
        kp = fast_kps[i]
        circle = Circle((kp['x'], kp['y']), 2, color='yellow', fill=False, linewidth=0.5)
        ax9.add_patch(circle)
    ax9.set_title(f'FAST Corners (before Harris): {len(fast_kps)}', fontsize=11, fontweight='bold')
    ax9.axis('off')
    
    # Filter by Harris response
    harris_kps_sorted = sorted(harris_kps, key=lambda k: k.get('harris', 0), reverse=True)
    top_harris = harris_kps_sorted[:500]
    
    ax10 = fig.add_subplot(gs[2, 2:4])
    ax10.imshow(gray, cmap='gray')
    for kp in top_harris:
        circle = Circle((kp['x'], kp['y']), 3, color='lime', fill=False, linewidth=1)
        ax10.add_patch(circle)
    ax10.set_title(f'After Harris Filtering (top 500): {len(top_harris)}', fontsize=11, fontweight='bold')
    ax10.axis('off')
    
    fig.suptitle('ORB Step 3: Harris Corner Response for FAST Keypoints', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUT_DIR, 'orb_step3_harris_detail.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: orb_step3_harris_detail.png")
    
    return top_harris


# =============================================================================
# STEP 4: Orientation Assignment
# =============================================================================
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
                    intensity = img[y + dy, x + dx]
                    m10 += dx * intensity
                    m01 += dy * intensity
        
        kp['orientation'] = np.arctan2(m01, m10)
    
    return keypoints


def visualize_step4_orientation_detail(gray, keypoints):
    """Step 4: Orientation assignment visualization."""
    print("Generating Step 4: Orientation Assignment...")
    
    # Compute orientations
    oriented_kps = compute_orientation(gray, keypoints)
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.2)
    
    # Top left: Intensity centroid concept
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    concept_text = "Intensity Centroid Method\n" + "="*30 + "\n\n"
    concept_text += "For patch around keypoint:\n\n"
    concept_text += "Image Moments:\n"
    concept_text += "  m₀₀ = ΣΣ I(x,y)\n"
    concept_text += "  m₁₀ = ΣΣ x·I(x,y)\n"
    concept_text += "  m₀₁ = ΣΣ y·I(x,y)\n\n"
    concept_text += "Centroid Location:\n"
    concept_text += "  Cx = m₁₀ / m₀₀\n"
    concept_text += "  Cy = m₀₁ / m₀₀\n\n"
    concept_text += "Orientation Angle:\n"
    concept_text += "  θ = atan2(m₀₁, m₁₀)\n\n"
    concept_text += "Patch radius: 15 pixels"
    
    ax1.text(0.1, 0.95, concept_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    ax1.set_title('Orientation Method', fontsize=11, fontweight='bold')
    
    # Top middle: Example patch with centroid
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Get a keypoint and show its patch
    kp = oriented_kps[50] if len(oriented_kps) > 50 else oriented_kps[0]
    x, y = kp['x'], kp['y']
    radius = 15
    
    if x - radius >= 0 and x + radius < gray.shape[1] and y - radius >= 0 and y + radius < gray.shape[0]:
        patch = gray[y-radius:y+radius+1, x-radius:x+radius+1]
        ax2.imshow(patch, cmap='gray', extent=[-radius, radius, radius, -radius])
        
        # Compute centroid for visualization
        m10 = m01 = m00 = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    intensity = patch[dy + radius, dx + radius]
                    m00 += intensity
                    m10 += dx * intensity
                    m01 += dy * intensity
        
        if m00 > 0:
            cx = m10 / m00
            cy = m01 / m00
            ax2.plot(0, 0, 'ro', markersize=10, label='Keypoint center')
            ax2.plot(cx, cy, 'b*', markersize=12, label='Intensity centroid')
            ax2.arrow(0, 0, cx*2, cy*2, head_width=1.5, head_length=1, fc='green', ec='green', linewidth=2)
        
        # Draw patch boundary
        circle = plt.Circle((0, 0), radius, fill=False, color='red', linestyle='--', linewidth=2)
        ax2.add_patch(circle)
        ax2.set_xlim(-radius-2, radius+2)
        ax2.set_ylim(radius+2, -radius-2)
        ax2.legend(loc='upper right', fontsize=8)
    
    ax2.set_title(f'Example Patch at ({x}, {y})\nθ = {np.degrees(kp["orientation"]):.1f}°', 
                  fontsize=10, fontweight='bold')
    ax2.set_xlabel('x offset')
    ax2.set_ylabel('y offset')
    
    # Top right: Orientation histogram
    ax3 = fig.add_subplot(gs[0, 2])
    orientations = [np.degrees(kp.get('orientation', 0)) for kp in oriented_kps]
    ax3.hist(orientations, bins=36, range=(-180, 180), color='steelblue', edgecolor='black')
    ax3.set_xlabel('Orientation (degrees)', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('Orientation Distribution\n(36 bins, 10° each)', fontsize=10, fontweight='bold')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=1, label='0°')
    ax3.legend()
    
    # Bottom: Full image with oriented keypoints
    ax4 = fig.add_subplot(gs[1, :])
    ax4.imshow(gray, cmap='gray')
    
    for kp in oriented_kps[:300]:
        x, y = kp['x'], kp['y']
        theta = kp.get('orientation', 0)
        
        circle = Circle((x, y), 6, color='lime', fill=False, linewidth=1.2)
        ax4.add_patch(circle)
        
        arrow_len = 12
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        ax4.arrow(x, y, dx, dy, head_width=4, head_length=3, fc='red', ec='red', linewidth=1)
    
    ax4.set_title(f'Keypoints with Orientation Arrows ({len(oriented_kps)} keypoints)', 
                  fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    fig.suptitle('ORB Step 4: Orientation Assignment (Intensity Centroid Method)', 
                 fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUT_DIR, 'orb_step4_orientation_detail.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: orb_step4_orientation_detail.png")
    
    return oriented_kps


# =============================================================================
# STEP 5: rBRIEF Descriptor Extraction
# =============================================================================
def generate_brief_pattern(n_pairs=256, patch_size=31, seed=42):
    """Generate BRIEF sampling pattern."""
    np.random.seed(seed)
    half = patch_size // 2
    pattern = []
    
    for _ in range(n_pairs):
        p_x = int(np.clip(np.random.randn() * half / 2, -half, half))
        p_y = int(np.clip(np.random.randn() * half / 2, -half, half))
        q_x = int(np.clip(np.random.randn() * half / 2, -half, half))
        q_y = int(np.clip(np.random.randn() * half / 2, -half, half))
        pattern.append(((p_x, p_y), (q_x, q_y)))
    
    return pattern


def extract_rbrief_descriptor(img, kp, pattern):
    """Extract rotated BRIEF descriptor."""
    h, w = img.shape
    x, y = kp['x'], kp['y']
    theta = kp.get('orientation', 0)
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    descriptor = np.zeros(len(pattern), dtype=np.uint8)
    
    for i, ((px, py), (qx, qy)) in enumerate(pattern):
        # Rotate points
        rpx = int(round(px * cos_t - py * sin_t))
        rpy = int(round(px * sin_t + py * cos_t))
        rqx = int(round(qx * cos_t - qy * sin_t))
        rqy = int(round(qx * sin_t + qy * cos_t))
        
        p_x, p_y = x + rpx, y + rpy
        q_x, q_y = x + rqx, y + rqy
        
        if 0 <= p_x < w and 0 <= p_y < h and 0 <= q_x < w and 0 <= q_y < h:
            if img[p_y, p_x] < img[q_y, q_x]:
                descriptor[i] = 1
    
    return descriptor


def visualize_step5_descriptor_detail(gray, keypoints):
    """Step 5: rBRIEF descriptor visualization."""
    print("Generating Step 5: rBRIEF Descriptors...")
    
    pattern = generate_brief_pattern()
    
    # Extract descriptors
    valid_kps = []
    descriptors = []
    h, w = gray.shape
    
    for kp in keypoints:
        if 16 <= kp['x'] < w - 16 and 16 <= kp['y'] < h - 16:
            desc = extract_rbrief_descriptor(gray, kp, pattern)
            valid_kps.append(kp)
            descriptors.append(desc)
    
    print(f"  Valid keypoints: {len(valid_kps)}")
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # Top row: BRIEF concept
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    concept_text = "rBRIEF (Rotated BRIEF)\n" + "="*30 + "\n\n"
    concept_text += "Binary descriptor:\n"
    concept_text += "  256 point pairs (p, q)\n"
    concept_text += "  For each pair:\n"
    concept_text += "    bit = 1 if I(p) < I(q)\n"
    concept_text += "    bit = 0 otherwise\n\n"
    concept_text += "Rotation invariance:\n"
    concept_text += "  Rotate pattern by θ\n"
    concept_text += "  p' = R(θ) · p\n"
    concept_text += "  q' = R(θ) · q\n\n"
    concept_text += "R(θ) = [cos θ  -sin θ]\n"
    concept_text += "       [sin θ   cos θ]"
    
    ax1.text(0.1, 0.95, concept_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax1.set_title('rBRIEF Concept', fontsize=11, fontweight='bold')
    
    # Sampling pattern visualization
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Show first 50 point pairs
    for i, ((px, py), (qx, qy)) in enumerate(pattern[:50]):
        ax2.plot([px, qx], [py, qy], 'g-', alpha=0.3, linewidth=0.5)
        ax2.plot(px, py, 'ro', markersize=2)
        ax2.plot(qx, qy, 'bs', markersize=2)
    
    ax2.plot(0, 0, 'k*', markersize=15, label='Center')
    circle = plt.Circle((0, 0), 15, fill=False, color='black', linestyle='--', linewidth=2)
    ax2.add_patch(circle)
    ax2.set_xlim(-20, 20)
    ax2.set_ylim(-20, 20)
    ax2.set_aspect('equal')
    ax2.set_title('BRIEF Sampling Pattern\n(First 50 pairs)', fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Rotated pattern example
    ax3 = fig.add_subplot(gs[0, 2])
    theta = np.pi / 4  # 45 degrees
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    for i, ((px, py), (qx, qy)) in enumerate(pattern[:50]):
        rpx = px * cos_t - py * sin_t
        rpy = px * sin_t + py * cos_t
        rqx = qx * cos_t - qy * sin_t
        rqy = qx * sin_t + qy * cos_t
        ax3.plot([rpx, rqx], [rpy, rqy], 'b-', alpha=0.3, linewidth=0.5)
        ax3.plot(rpx, rpy, 'ro', markersize=2)
        ax3.plot(rqx, rqy, 'bs', markersize=2)
    
    ax3.plot(0, 0, 'k*', markersize=15)
    circle = plt.Circle((0, 0), 15, fill=False, color='black', linestyle='--', linewidth=2)
    ax3.add_patch(circle)
    ax3.set_xlim(-20, 20)
    ax3.set_ylim(-20, 20)
    ax3.set_aspect('equal')
    ax3.set_title('Rotated Pattern (θ = 45°)', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Middle row: Example descriptors
    ax4 = fig.add_subplot(gs[1, :])
    
    if len(descriptors) >= 5:
        desc_matrix = np.array(descriptors[:5])
        ax4.imshow(desc_matrix, cmap='binary', aspect='auto', interpolation='nearest')
        ax4.set_xlabel('Bit Position (0-255)', fontsize=10)
        ax4.set_ylabel('Keypoint', fontsize=10)
        ax4.set_yticks(range(5))
        ax4.set_yticklabels([f'KP {i}' for i in range(5)])
        ax4.set_title('Example rBRIEF Descriptors (256-bit binary)', fontsize=11, fontweight='bold')
    
    # Bottom row: Image with keypoints and descriptor stats
    ax5 = fig.add_subplot(gs[2, 0:2])
    ax5.imshow(gray, cmap='gray')
    
    for kp in valid_kps[:200]:
        x, y = kp['x'], kp['y']
        theta = kp.get('orientation', 0)
        circle = Circle((x, y), 5, color='lime', fill=False, linewidth=1)
        ax5.add_patch(circle)
        arrow_len = 10
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        ax5.arrow(x, y, dx, dy, head_width=3, head_length=2, fc='red', ec='red')
    
    ax5.set_title(f'Keypoints with Valid Descriptors ({len(valid_kps)} total)', 
                  fontsize=11, fontweight='bold')
    ax5.axis('off')
    
    # Stats
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    stats_text = "Descriptor Statistics\n" + "="*30 + "\n\n"
    stats_text += f"Total keypoints: {len(keypoints)}\n"
    stats_text += f"Valid descriptors: {len(valid_kps)}\n"
    stats_text += f"Descriptor size: 256 bits\n"
    stats_text += f"Storage: 32 bytes each\n"
    stats_text += f"Total storage: {len(valid_kps) * 32} bytes\n\n"
    
    if len(descriptors) > 0:
        avg_ones = np.mean([np.sum(d) for d in descriptors])
        stats_text += f"Avg bits set: {avg_ones:.1f}/256\n"
        stats_text += f"Bit ratio: {avg_ones/256:.2%}\n"
    
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    fig.suptitle('ORB Step 5: rBRIEF Descriptor Extraction', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUT_DIR, 'orb_step5_descriptor_detail.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: orb_step5_descriptor_detail.png")
    
    return valid_kps, descriptors


# =============================================================================
# STEP 6: Matching with Hamming Distance
# =============================================================================
def visualize_step6_matching_detail(descriptors):
    """Step 6: Hamming distance matching visualization."""
    print("Generating Step 6: Hamming Distance Matching...")
    
    if len(descriptors) < 10:
        print("  Not enough descriptors for matching visualization")
        return
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # Hamming distance concept
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    concept_text = "Hamming Distance\n" + "="*30 + "\n\n"
    concept_text += "For binary descriptors A, B:\n\n"
    concept_text += "  H(A,B) = popcount(A XOR B)\n\n"
    concept_text += "  = number of differing bits\n\n"
    concept_text += "Example:\n"
    concept_text += "  A = 10110010...\n"
    concept_text += "  B = 10011010...\n"
    concept_text += "  XOR= 00101000...\n"
    concept_text += "  H = count of 1s = 2\n\n"
    concept_text += "Advantages:\n"
    concept_text += "  • Single CPU instruction (POPCNT)\n"
    concept_text += "  • ~8 cycles for 256-bit\n"
    concept_text += "  • vs ~500 cycles for L2 (SIFT)"
    
    ax1.text(0.05, 0.95, concept_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax1.set_title('Hamming Distance', fontsize=11, fontweight='bold')
    
    # Distance matrix
    ax2 = fig.add_subplot(gs[0, 1:])
    
    n_desc = min(20, len(descriptors))
    dist_matrix = np.zeros((n_desc, n_desc))
    
    for i in range(n_desc):
        for j in range(n_desc):
            xor_result = np.bitwise_xor(descriptors[i], descriptors[j])
            dist_matrix[i, j] = np.sum(xor_result)
    
    im = ax2.imshow(dist_matrix, cmap='viridis')
    ax2.set_xlabel('Descriptor Index', fontsize=10)
    ax2.set_ylabel('Descriptor Index', fontsize=10)
    ax2.set_title(f'Hamming Distance Matrix ({n_desc}×{n_desc})', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Hamming Distance')
    
    # Distance histogram
    ax3 = fig.add_subplot(gs[1, 0])
    
    all_distances = []
    for i in range(min(100, len(descriptors))):
        for j in range(i + 1, min(100, len(descriptors))):
            xor_result = np.bitwise_xor(descriptors[i], descriptors[j])
            all_distances.append(np.sum(xor_result))
    
    ax3.hist(all_distances, bins=50, color='steelblue', edgecolor='black')
    ax3.axvline(x=64, color='red', linestyle='--', linewidth=2, label='Threshold (64 = 25%)')
    ax3.set_xlabel('Hamming Distance', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('Distance Distribution', fontsize=11, fontweight='bold')
    ax3.legend()
    
    # Matching criteria
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    match_text = "Matching Criteria\n" + "="*30 + "\n\n"
    match_text += "1. Distance Threshold:\n"
    match_text += "   H(A,B) < 64 (25% of bits)\n\n"
    match_text += "2. Ratio Test (Lowe's):\n"
    match_text += "   best_dist / 2nd_best < 0.75\n\n"
    match_text += "3. Cross-Check:\n"
    match_text += "   A's best match is B\n"
    match_text += "   AND B's best match is A\n\n"
    match_text += "Good match:\n"
    match_text += "   H < 64 bits different"
    
    ax4.text(0.05, 0.95, match_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    ax4.set_title('Matching Criteria', fontsize=11, fontweight='bold')
    
    # Speed comparison
    ax5 = fig.add_subplot(gs[1, 2])
    
    methods = ['ORB\n(Hamming)', 'SIFT\n(L2)']
    cycles = [8, 500]
    colors = ['green', 'orange']
    
    bars = ax5.bar(methods, cycles, color=colors)
    ax5.set_ylabel('CPU Cycles', fontsize=10)
    ax5.set_title('Matching Speed Comparison\n(per descriptor pair)', fontsize=11, fontweight='bold')
    
    for bar, cycle in zip(bars, cycles):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{cycle}', ha='center', fontsize=12, fontweight='bold')
    
    ax5.text(0.5, 0.7, f'ORB is {500/8:.0f}× faster!', transform=ax5.transAxes,
             ha='center', fontsize=14, fontweight='bold', color='green')
    
    fig.suptitle('ORB Step 6: Descriptor Matching with Hamming Distance', 
                 fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUT_DIR, 'orb_step6_matching_detail.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: orb_step6_matching_detail.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
def visualize_final_summary(gray, valid_kps, descriptors):
    """Create final pipeline summary."""
    print("Generating Final Summary...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    # Final keypoints on image
    ax1 = fig.add_subplot(gs[:2, 0:2])
    ax1.imshow(gray, cmap='gray')
    
    for kp in valid_kps[:300]:
        x, y = kp['x'], kp['y']
        theta = kp.get('orientation', 0)
        
        circle = Circle((x, y), 6, color='lime', fill=False, linewidth=1.5)
        ax1.add_patch(circle)
        
        arrow_len = 12
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        ax1.arrow(x, y, dx, dy, head_width=4, head_length=3, fc='red', ec='red', linewidth=1.2)
    
    ax1.set_title(f'ORB Final Output: {len(valid_kps)} Keypoints with Descriptors', 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Pipeline steps
    steps = [
        ('1. Scale Pyramid', 'Build 8 levels\nScale factor: 1.2'),
        ('2. FAST Corners', 'Detect corners\nThreshold: 25'),
        ('3. Harris Filter', 'Score & filter\nTop N keypoints'),
        ('4. Orientation', 'Intensity centroid\nθ = atan2(m01, m10)'),
        ('5. rBRIEF', '256-bit binary\nRotated pattern'),
        ('6. Matching', 'Hamming distance\n~8 CPU cycles'),
    ]
    
    for i, (title, desc) in enumerate(steps):
        row = i // 2
        col = 2 + i % 2
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
        
        color = plt.cm.viridis(i / len(steps))
        box = FancyBboxPatch((0.1, 0.2), 0.8, 0.6, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(box)
        ax.text(0.5, 0.7, title, ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        ax.text(0.5, 0.35, desc, ha='center', va='center', fontsize=9, color='white')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    fig.suptitle('ORB Algorithm: Complete Pipeline Summary', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(OUT_DIR, 'orb_final_pipeline_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: orb_final_pipeline_summary.png")


# =============================================================================
# Main Execution
# =============================================================================
def main():
    print("=" * 60)
    print("ORB Step-by-Step Visualization")
    print("=" * 60)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load image
    gray, img_rgb = load_image()
    print(f"\nInput image: {gray.shape[1]}×{gray.shape[0]}")
    
    # Step 1: Scale Pyramid
    pyramid, scales = visualize_step1_pyramid_detail()
    
    # Step 2: FAST Corners
    all_kps = visualize_step2_fast_per_level(pyramid, scales)
    visualize_step2_fast_combined(gray, pyramid, scales)
    
    # Step 3: Harris Corner Response
    top_harris = visualize_step3_harris_detail(gray)
    
    # Step 4: Orientation Assignment
    oriented_kps = visualize_step4_orientation_detail(gray, top_harris)
    
    # Step 5: rBRIEF Descriptors
    valid_kps, descriptors = visualize_step5_descriptor_detail(gray, oriented_kps)
    
    # Step 6: Matching
    visualize_step6_matching_detail(descriptors)
    
    # Final Summary
    visualize_final_summary(gray, valid_kps, descriptors)
    
    print("\n" + "=" * 60)
    print("All step-by-step visualizations generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
ORB Algorithm Pipeline - Step by Step
Oriented FAST and Rotated BRIEF
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, '..', 'images')


# =============================================================================
# STEP 1: Scale-Space Pyramid
# =============================================================================

def build_scale_pyramid(img, n_levels=8, scale_factor=1.2):
    """
    Build image pyramid for multi-scale detection.
    
    Unlike SIFT which uses Gaussian blur, ORB uses direct downsampling.
    
    Parameters:
    - img: Input grayscale image
    - n_levels: Number of pyramid levels (default: 8)
    - scale_factor: Scale ratio between levels (default: 1.2)
    
    Returns:
    - pyramid: List of images at different scales
    - scales: List of scale values
    """
    print("[ORB Step 1] Building scale pyramid...")
    pyramid = [img.copy()]
    scales = [1.0]
    
    current = img.copy()
    for level in range(1, n_levels):
        scale = 1.0 / (scale_factor ** level)
        new_h = int(img.shape[0] * scale)
        new_w = int(img.shape[1] * scale)
        
        if new_h < 10 or new_w < 10:
            break
        
        # Resize using PIL for quality
        pil_img = Image.fromarray((current * 255).astype(np.uint8))
        pil_resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        resized = np.array(pil_resized) / 255.0
        
        pyramid.append(resized)
        scales.append(scale)
        current = img.copy()
    
    print(f"         Built {len(pyramid)} levels")
    for i, (p, s) in enumerate(zip(pyramid, scales)):
        print(f"         Level {i}: {p.shape[1]}×{p.shape[0]} (scale={s:.3f})")
    
    return pyramid, scales


# =============================================================================
# STEP 2: FAST Corner Detection
# =============================================================================

# Bresenham circle offsets (16 pixels)
CIRCLE_OFFSETS = [
    (0, -3), (1, -3), (2, -2), (3, -1),    # positions 1-4
    (3, 0), (3, 1), (2, 2), (1, 3),         # positions 5-8
    (0, 3), (-1, 3), (-2, 2), (-3, 1),      # positions 9-12
    (-3, 0), (-3, -1), (-2, -2), (-1, -3)   # positions 13-16
]


def fast_corner_test(img, x, y, threshold=20, n_contiguous=9):
    """
    FAST corner test at a single pixel.
    
    Returns True if pixel is a corner (9+ contiguous brighter or darker pixels).
    """
    h, w = img.shape
    
    # Boundary check
    if x < 3 or x >= w - 3 or y < 3 or y >= h - 3:
        return False, 0
    
    center = img[y, x]
    upper = center + threshold / 255.0
    lower = center - threshold / 255.0
    
    # High-speed test: check positions 1, 5, 9, 13 first
    test_positions = [0, 4, 8, 12]  # 0-indexed
    n_brighter = 0
    n_darker = 0
    
    for pos in test_positions:
        dx, dy = CIRCLE_OFFSETS[pos]
        val = img[y + dy, x + dx]
        if val > upper:
            n_brighter += 1
        elif val < lower:
            n_darker += 1
    
    # If not at least 3 of 4 are brighter or darker, not a corner
    if n_brighter < 3 and n_darker < 3:
        return False, 0
    
    # Full 16-pixel test
    circle_values = []
    for dx, dy in CIRCLE_OFFSETS:
        circle_values.append(img[y + dy, x + dx])
    
    # Check for contiguous brighter pixels
    labels = []
    for val in circle_values:
        if val > upper:
            labels.append('B')
        elif val < lower:
            labels.append('D')
        else:
            labels.append('S')
    
    # Duplicate for wrap-around check
    labels_extended = labels + labels
    
    # Count max contiguous B's and D's
    max_brighter = 0
    max_darker = 0
    count_b = 0
    count_d = 0
    
    for label in labels_extended:
        if label == 'B':
            count_b += 1
            count_d = 0
            max_brighter = max(max_brighter, count_b)
        elif label == 'D':
            count_d += 1
            count_b = 0
            max_darker = max(max_darker, count_d)
        else:
            count_b = 0
            count_d = 0
    
    # Cap at 16 (full circle)
    max_brighter = min(max_brighter, 16)
    max_darker = min(max_darker, 16)
    
    # Corner response (for ranking)
    response = max(max_brighter, max_darker)
    
    return (max_brighter >= n_contiguous or max_darker >= n_contiguous), response


def detect_fast_corners(img, threshold=20, n_contiguous=9):
    """
    Detect FAST corners in image.
    
    Parameters:
    - img: Grayscale image (0-1 float)
    - threshold: Intensity threshold (default: 20)
    - n_contiguous: Minimum contiguous pixels (default: 9 for FAST-9)
    
    Returns:
    - keypoints: List of (x, y, response) tuples
    """
    print("[ORB Step 2] Detecting FAST corners...")
    h, w = img.shape
    keypoints = []
    
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            is_corner, response = fast_corner_test(img, x, y, threshold, n_contiguous)
            if is_corner:
                keypoints.append({'x': x, 'y': y, 'response': response})
    
    print(f"         Found {len(keypoints)} FAST corners")
    return keypoints


# =============================================================================
# STEP 3: Harris Corner Response
# =============================================================================

def compute_harris_response(img, keypoints, k=0.04, block_size=3):
    """
    Compute Harris corner response for each FAST keypoint.
    
    Harris Response: R = det(M) - k * trace(M)^2
    
    where M is the structure tensor:
    M = [Ix², Ix*Iy]
        [Ix*Iy, Iy²]
    """
    print("[ORB Step 3] Computing Harris corner response...")
    
    # Compute gradients
    Ix = ndimage.sobel(img, axis=1)
    Iy = ndimage.sobel(img, axis=0)
    
    # Structure tensor components
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # Gaussian smoothing for structure tensor
    sigma = block_size / 3.0
    Ixx = ndimage.gaussian_filter(Ixx, sigma)
    Iyy = ndimage.gaussian_filter(Iyy, sigma)
    Ixy = ndimage.gaussian_filter(Ixy, sigma)
    
    h, w = img.shape
    
    for kp in keypoints:
        x, y = kp['x'], kp['y']
        
        if 0 <= x < w and 0 <= y < h:
            # Harris response at keypoint location
            det_M = Ixx[y, x] * Iyy[y, x] - Ixy[y, x] ** 2
            trace_M = Ixx[y, x] + Iyy[y, x]
            R = det_M - k * (trace_M ** 2)
            kp['harris'] = R
        else:
            kp['harris'] = 0
    
    print(f"         Computed Harris response for {len(keypoints)} keypoints")
    return keypoints


def non_maximum_suppression(keypoints, radius=3):
    """
    Non-maximum suppression to keep only local maxima.
    """
    if len(keypoints) == 0:
        return []
    
    # Sort by Harris response (descending)
    keypoints = sorted(keypoints, key=lambda k: k.get('harris', 0), reverse=True)
    
    kept = []
    suppressed = set()
    
    for i, kp in enumerate(keypoints):
        if i in suppressed:
            continue
        
        kept.append(kp)
        
        # Suppress neighbors
        for j, other in enumerate(keypoints):
            if j != i and j not in suppressed:
                dist = np.sqrt((kp['x'] - other['x'])**2 + (kp['y'] - other['y'])**2)
                if dist < radius:
                    suppressed.add(j)
    
    return kept


def select_top_keypoints(keypoints, n_keypoints=500):
    """
    Select top N keypoints by Harris response.
    """
    print("[ORB Step 3b] Selecting top keypoints...")
    
    # Sort by Harris response (descending)
    keypoints = sorted(keypoints, key=lambda k: k.get('harris', 0), reverse=True)
    
    # Keep top N
    selected = keypoints[:n_keypoints]
    print(f"         Selected {len(selected)} keypoints (from {len(keypoints)})")
    
    return selected


# =============================================================================
# STEP 4: Orientation Assignment (Intensity Centroid)
# =============================================================================

def compute_orientation(img, keypoints, patch_radius=15):
    """
    Compute orientation using intensity centroid method.
    
    Orientation θ = atan2(m₀₁, m₁₀)
    
    where:
    - m₁₀ = Σ x * I(x, y)
    - m₀₁ = Σ y * I(x, y)
    """
    print("[ORB Step 4] Computing orientations (intensity centroid)...")
    
    h, w = img.shape
    
    for kp in keypoints:
        cx, cy = int(kp['x']), int(kp['y'])
        
        m10 = 0
        m01 = 0
        
        for dy in range(-patch_radius, patch_radius + 1):
            for dx in range(-patch_radius, patch_radius + 1):
                x, y = cx + dx, cy + dy
                
                if 0 <= x < w and 0 <= y < h:
                    intensity = img[y, x]
                    m10 += dx * intensity
                    m01 += dy * intensity
        
        # Orientation
        theta = np.arctan2(m01, m10)
        kp['orientation'] = theta
    
    print(f"         Computed orientation for {len(keypoints)} keypoints")
    return keypoints


# =============================================================================
# STEP 5: rBRIEF Descriptor
# =============================================================================

def generate_brief_pattern(n_pairs=256, patch_size=31, seed=42):
    """
    Generate BRIEF sampling pattern.
    
    Pattern is a list of (p, q) point pairs where p and q are (x, y) offsets.
    """
    np.random.seed(seed)
    
    half = patch_size // 2
    pattern = []
    
    for _ in range(n_pairs):
        # Gaussian distribution for better performance
        p_x = int(np.clip(np.random.randn() * half / 2, -half, half))
        p_y = int(np.clip(np.random.randn() * half / 2, -half, half))
        q_x = int(np.clip(np.random.randn() * half / 2, -half, half))
        q_y = int(np.clip(np.random.randn() * half / 2, -half, half))
        
        pattern.append(((p_x, p_y), (q_x, q_y)))
    
    return pattern


def rotate_pattern(pattern, theta):
    """
    Rotate BRIEF pattern by angle theta.
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    rotated = []
    for (px, py), (qx, qy) in pattern:
        # Rotate p
        rpx = int(round(px * cos_t - py * sin_t))
        rpy = int(round(px * sin_t + py * cos_t))
        
        # Rotate q
        rqx = int(round(qx * cos_t - qy * sin_t))
        rqy = int(round(qx * sin_t + qy * cos_t))
        
        rotated.append(((rpx, rpy), (rqx, rqy)))
    
    return rotated


def compute_rbrief_descriptor(img, keypoint, pattern, patch_radius=15):
    """
    Compute rBRIEF (rotated BRIEF) descriptor for a keypoint.
    
    Returns a 256-bit binary descriptor as numpy array of 0s and 1s.
    """
    h, w = img.shape
    cx, cy = int(keypoint['x']), int(keypoint['y'])
    theta = keypoint.get('orientation', 0)
    
    # Rotate pattern
    rotated_pattern = rotate_pattern(pattern, theta)
    
    descriptor = np.zeros(len(pattern), dtype=np.uint8)
    
    for i, ((px, py), (qx, qy)) in enumerate(rotated_pattern):
        # Point p
        p_x, p_y = cx + px, cy + py
        # Point q
        q_x, q_y = cx + qx, cy + qy
        
        # Boundary check
        if not (0 <= p_x < w and 0 <= p_y < h and 0 <= q_x < w and 0 <= q_y < h):
            continue
        
        # Binary test
        if img[p_y, p_x] < img[q_y, q_x]:
            descriptor[i] = 1
    
    return descriptor


def extract_descriptors(img, keypoints, n_pairs=256):
    """
    Extract rBRIEF descriptors for all keypoints.
    """
    print("[ORB Step 5] Extracting rBRIEF descriptors (256-bit)...")
    
    pattern = generate_brief_pattern(n_pairs)
    
    descriptors = []
    valid_keypoints = []
    
    h, w = img.shape
    
    for kp in keypoints:
        x, y = int(kp['x']), int(kp['y'])
        
        # Boundary check (need space for patch)
        if x < 16 or x >= w - 16 or y < 16 or y >= h - 16:
            continue
        
        desc = compute_rbrief_descriptor(img, kp, pattern)
        descriptors.append(desc)
        kp['descriptor'] = desc
        valid_keypoints.append(kp)
    
    print(f"         Extracted {len(descriptors)} descriptors")
    return valid_keypoints, np.array(descriptors)


# =============================================================================
# STEP 6: Hamming Distance Matching
# =============================================================================

def hamming_distance(desc1, desc2):
    """
    Compute Hamming distance between two binary descriptors.
    """
    return np.sum(desc1 != desc2)


def match_descriptors(desc1, desc2, threshold=64, ratio_threshold=0.8):
    """
    Match descriptors using Hamming distance.
    
    Returns list of (idx1, idx2, distance) tuples.
    """
    print("[ORB Step 6] Matching descriptors (Hamming distance)...")
    
    matches = []
    
    for i, d1 in enumerate(desc1):
        distances = []
        for j, d2 in enumerate(desc2):
            dist = hamming_distance(d1, d2)
            distances.append((j, dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        if len(distances) >= 2:
            best = distances[0]
            second_best = distances[1]
            
            # Ratio test
            if second_best[1] > 0 and best[1] / second_best[1] < ratio_threshold:
                if best[1] < threshold:
                    matches.append((i, best[0], best[1]))
    
    print(f"         Found {len(matches)} matches")
    return matches


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_pyramid(pyramid, scales, filename):
    """Visualize scale pyramid."""
    n_levels = min(len(pyramid), 8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(n_levels):
        row, col = i // 4, i % 4
        ax = axes[row, col]
        ax.imshow(pyramid[i], cmap='gray')
        ax.set_title(f'Level {i}\n{pyramid[i].shape[1]}×{pyramid[i].shape[0]}\nscale={scales[i]:.3f}', fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_levels, 8):
        row, col = i // 4, i % 4
        axes[row, col].axis('off')
    
    fig.suptitle('ORB Step 1: Scale-Space Pyramid', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"         Saved: {filename}")


def visualize_fast_corners(img, keypoints, filename, title="FAST Corners"):
    """Visualize FAST corner detections."""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img, cmap='gray')
    
    # If too many keypoints, sample evenly from all of them instead of first N
    n_show = min(800, len(keypoints))
    if len(keypoints) > n_show:
        # Sample evenly distributed keypoints
        indices = np.linspace(0, len(keypoints) - 1, n_show, dtype=int)
        kps_to_show = [keypoints[i] for i in indices]
    else:
        kps_to_show = keypoints
    
    for kp in kps_to_show:
        circle = plt.Circle((kp['x'], kp['y']), 3, color='lime', fill=False, linewidth=1)
        ax.add_patch(circle)
    
    ax.set_title(f'{title}\n({len(keypoints)} corners)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"         Saved: {filename}")


def visualize_keypoints_with_orientation(img, keypoints, filename, title="Keypoints"):
    """Visualize keypoints with orientation arrows."""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img, cmap='gray')
    
    for kp in keypoints[:300]:  # Limit for visualization
        x, y = kp['x'], kp['y']
        theta = kp.get('orientation', 0)
        
        # Draw circle
        circle = plt.Circle((x, y), 5, color='lime', fill=False, linewidth=1.5)
        ax.add_patch(circle)
        
        # Draw orientation arrow
        arrow_len = 12
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=3, head_length=2, fc='red', ec='red', linewidth=1)
    
    ax.set_title(f'{title}\n({len(keypoints)} keypoints)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"         Saved: {filename}")


def visualize_descriptors(descriptors, filename, n_show=5):
    """Visualize binary descriptors."""
    if len(descriptors) == 0:
        return
    
    n = min(n_show, len(descriptors))
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n))
    
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        ax = axes[i]
        desc = descriptors[i]
        
        # Show as bar chart
        colors = ['green' if b == 1 else 'darkblue' for b in desc]
        ax.bar(range(256), desc, color=colors, width=1)
        ax.set_xlim(-1, 256)
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel(f'KP {i}')
        
        if i == n - 1:
            ax.set_xlabel('Bit position (0-255)')
    
    fig.suptitle('ORB rBRIEF Descriptors (256-bit binary)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"         Saved: {filename}")


def visualize_complete_pipeline(img, pyramid, scales, fast_kps, harris_kps, final_kps, descriptors, filename):
    """Create complete pipeline summary visualization."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.2)
    
    # 1. Input Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f'1. Input Image\n{img.shape[1]}×{img.shape[0]} pixels', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Scale Pyramid (montage)
    ax2 = fig.add_subplot(gs[0, 1])
    # Create simple montage
    ax2.imshow(pyramid[0], cmap='gray', alpha=0.3)
    for i, p in enumerate(pyramid[::2]):  # Every other level
        size = int(50 * scales[i*2])
        if size > 5:
            ax2.add_patch(plt.Rectangle((10 + i*60, 10), size, size, 
                                        fill=False, edgecolor=['red', 'green', 'blue', 'orange'][i % 4], linewidth=2))
    ax2.set_title(f'2. Scale Pyramid\n{len(pyramid)} levels, factor=1.2', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. FAST Corners
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img, cmap='gray')
    for kp in fast_kps[:500]:
        circle = plt.Circle((kp['x'], kp['y']), 2, color='yellow', fill=False, linewidth=0.5)
        ax3.add_patch(circle)
    ax3.set_title(f'3. FAST Corners\n{len(fast_kps)} detected', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. Harris Filtered
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(img, cmap='gray')
    for kp in harris_kps[:300]:
        circle = plt.Circle((kp['x'], kp['y']), 4, color='cyan', fill=False, linewidth=1)
        ax4.add_patch(circle)
    ax4.set_title(f'4. Harris Filtered\n{len(harris_kps)} keypoints', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 5. Orientation Assignment
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(img, cmap='gray')
    for kp in final_kps[:200]:
        x, y = kp['x'], kp['y']
        theta = kp.get('orientation', 0)
        circle = plt.Circle((x, y), 5, color='lime', fill=False, linewidth=1.5)
        ax5.add_patch(circle)
        arrow_len = 10
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        ax5.arrow(x, y, dx, dy, head_width=3, head_length=2, fc='red', ec='red')
    ax5.set_title(f'5. With Orientation\n{len(final_kps)} keypoints', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 6. rBRIEF Descriptor
    ax6 = fig.add_subplot(gs[1, 2])
    if len(descriptors) > 0:
        # Show first descriptor as image-like representation
        desc_img = descriptors[0].reshape(16, 16)
        ax6.imshow(desc_img, cmap='binary', aspect='auto')
        ax6.set_title(f'6. rBRIEF Descriptor\n256-bit (16×16 visualization)', fontsize=12, fontweight='bold')
    else:
        ax6.set_title('6. rBRIEF Descriptor\n(no valid descriptors)', fontsize=12, fontweight='bold')
    
    plt.suptitle('ORB Complete Pipeline: Detection → Description', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"         Saved: {filename}")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_orb_pipeline():
    """Run complete ORB pipeline."""
    print("=" * 70)
    print("ORB ALGORITHM PIPELINE")
    print("Oriented FAST and Rotated BRIEF")
    print("=" * 70)
    
    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load or create test image
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    if not os.path.exists(image_path):
        print("Creating test image...")
        # Create a test image with features
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img[:, :] = [100, 100, 100]
        # Add some shapes for corner detection
        img[50:100, 50:100] = [255, 255, 255]
        img[150:200, 200:280] = [200, 200, 200]
        img[100:150, 300:350] = [180, 180, 180]
        # Add diagonal pattern
        for i in range(50):
            if 50+i < 300 and 150+i < 400:
                img[50+i, 150+i:155+i] = [220, 220, 220]
        Image.fromarray(img).save(image_path)
    
    # Load image
    img_rgb = np.array(Image.open(image_path))
    if len(img_rgb.shape) == 3:
        gray = (0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]) / 255.0
    else:
        gray = img_rgb / 255.0
    
    print(f"\nInput image: {gray.shape[1]}×{gray.shape[0]}")
    
    # Step 1: Build Scale Pyramid
    pyramid, scales = build_scale_pyramid(gray)
    visualize_pyramid(pyramid, scales, "orb_step1_pyramid.png")
    
    # Step 2: FAST Corner Detection (on first level for simplicity)
    fast_keypoints = detect_fast_corners(gray)
    visualize_fast_corners(gray, fast_keypoints, "orb_step2_fast.png", "ORB Step 2: FAST Corners")
    
    # Step 3: Harris Corner Response
    harris_keypoints = compute_harris_response(gray, fast_keypoints)
    harris_keypoints = non_maximum_suppression(harris_keypoints)
    selected_keypoints = select_top_keypoints(harris_keypoints, n_keypoints=500)
    visualize_fast_corners(gray, selected_keypoints, "orb_step3_harris.png", "ORB Step 3: Harris Filtered")
    
    # Step 4: Orientation Assignment
    oriented_keypoints = compute_orientation(gray, selected_keypoints)
    visualize_keypoints_with_orientation(gray, oriented_keypoints, "orb_step4_orientation.png", 
                                         "ORB Step 4: Orientation Assignment")
    
    # Step 5: rBRIEF Descriptor Extraction
    final_keypoints, descriptors = extract_descriptors(gray, oriented_keypoints)
    visualize_descriptors(descriptors, "orb_step5_descriptors.png")
    
    # Complete summary
    visualize_complete_pipeline(gray, pyramid, scales, fast_keypoints, selected_keypoints,
                               final_keypoints, descriptors, "orb_complete_summary.png")
    
    # Print summary
    print("\n" + "=" * 70)
    print("ORB PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n1️⃣ DETECTION PHASE:")
    print(f"   Step 1: Scale Pyramid - {len(pyramid)} levels")
    print(f"   Step 2: FAST Corners - {len(fast_keypoints)} detected")
    print(f"   Step 3: Harris Filter - {len(selected_keypoints)} keypoints")
    print(f"   Step 4: Orientation - {len(oriented_keypoints)} with θ")
    print(f"\n2️⃣ DESCRIPTION PHASE:")
    print(f"   Step 5: rBRIEF - {len(descriptors)} × 256-bit descriptors")
    print(f"\nFINAL OUTPUT: {len(final_keypoints)} keypoints with 256-bit descriptors")
    print("=" * 70)
    
    return final_keypoints, descriptors


if __name__ == "__main__":
    run_orb_pipeline()

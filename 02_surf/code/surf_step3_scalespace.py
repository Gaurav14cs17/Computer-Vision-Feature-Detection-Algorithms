"""
SURF Step 3: Scale-Space Extrema Detection Visualization
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


def compute_integral_image(img):
    return np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)


def box_sum(integral, x1, y1, x2, y2):
    h, w = integral.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    D = integral[y2, x2]
    B = integral[y1-1, x2] if y1 > 0 else 0
    C = integral[y2, x1-1] if x1 > 0 else 0
    A = integral[y1-1, x1-1] if y1 > 0 and x1 > 0 else 0
    return D - B - C + A


def compute_hessian_response(integral, x, y, filter_size):
    h, w = integral.shape
    half = filter_size // 2
    if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
        return 0
    lobe_w = filter_size // 3
    
    left = box_sum(integral, x - half, y - half, x - half + lobe_w - 1, y + half)
    center = box_sum(integral, x - lobe_w//2, y - half, x + lobe_w//2, y + half)
    right = box_sum(integral, x + half - lobe_w + 1, y - half, x + half, y + half)
    Dxx = left - 2 * center + right
    
    top = box_sum(integral, x - half, y - half, x + half, y - half + lobe_w - 1)
    middle = box_sum(integral, x - half, y - lobe_w//2, x + half, y + lobe_w//2)
    bottom = box_sum(integral, x - half, y + half - lobe_w + 1, x + half, y + half)
    Dyy = top - 2 * middle + bottom
    
    tl = box_sum(integral, x - half, y - half, x - 1, y - 1)
    tr = box_sum(integral, x + 1, y - half, x + half, y - 1)
    bl = box_sum(integral, x - half, y + 1, x - 1, y + half)
    br = box_sum(integral, x + 1, y + 1, x + half, y + half)
    Dxy = tl - tr - bl + br
    
    area = filter_size * filter_size
    Dxx /= area
    Dyy /= area
    Dxy /= area
    
    return Dxx * Dyy - (0.9 * Dxy) ** 2


def load_image():
    input_path = os.path.join(IMAGES_DIR, 'input_image.jpg')
    if os.path.exists(input_path):
        img = Image.open(input_path).convert('L')
        if img.size[0] > 800 or img.size[1] > 600:
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
        # IMPORTANT: Normalize to [0,1] range (same as SIFT)
        return np.array(img).astype(np.float64) / 255.0
    else:
        img = np.zeros((480, 640), dtype=np.float64)
        img[100:200, 100:250] = 0.7
        img[250:350, 300:450] = 0.86
        Image.fromarray((img * 255).astype(np.uint8)).save(input_path)
        return img


def detect_keypoints(responses, filter_sizes, threshold=0.0005):
    """Detect scale-space extrema with proper threshold"""
    keypoints = []
    
    for scale_idx in range(1, len(responses) - 1):
        prev_resp = responses[scale_idx - 1]
        curr_resp = responses[scale_idx]
        next_resp = responses[scale_idx + 1]
        h, w = curr_resp.shape
        
        for y in range(3, h - 3):
            for x in range(3, w - 3):
                val = curr_resp[y, x]
                if abs(val) < threshold:
                    continue
                
                # Check 26 neighbors in 3x3x3 cube
                is_max = True
                is_min = True
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        # Previous scale
                        if prev_resp[y + dy, x + dx] >= val:
                            is_max = False
                        if prev_resp[y + dy, x + dx] <= val:
                            is_min = False
                        # Current scale (excluding center)
                        if dy != 0 or dx != 0:
                            if curr_resp[y + dy, x + dx] >= val:
                                is_max = False
                            if curr_resp[y + dy, x + dx] <= val:
                                is_min = False
                        # Next scale
                        if next_resp[y + dy, x + dx] >= val:
                            is_max = False
                        if next_resp[y + dy, x + dx] <= val:
                            is_min = False
                
                if is_max or is_min:
                    keypoints.append({
                        'x': x, 'y': y, 
                        'scale': scale_idx, 
                        'filter_size': filter_sizes[scale_idx],
                        'response': val,
                        'type': 'max' if is_max else 'min'
                    })
    
    return keypoints


def visualize_scalespace():
    """Visualize scale-space extrema detection"""
    print("=" * 60)
    print("SURF Step 3: Scale-Space Extrema Detection")
    print("=" * 60)
    
    img = load_image()
    H, W = img.shape
    integral = compute_integral_image(img)
    
    filter_sizes = [9, 15, 21, 27]
    responses = []
    
    print("Computing Hessian responses...")
    for fs in filter_sizes:
        print(f"  Filter size {fs}×{fs}...")
        response = np.zeros((H, W))
        margin = fs // 2 + 1
        # Use step=1 for proper detection
        for y in range(margin, H - margin):
            for x in range(margin, W - margin):
                response[y, x] = compute_hessian_response(integral, x, y, fs)
        responses.append(response)
    
    print("Detecting keypoints...")
    keypoints = detect_keypoints(responses, filter_sizes)
    print(f"Detected: {len(keypoints)} keypoints")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # SURF vs SIFT scale-space
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.axis('off')
    ax1.set_title('SURF Filter Pyramid\n(vs SIFT Image Pyramid)', fontsize=12, fontweight='bold')
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    for i, (fs, color) in enumerate(zip(filter_sizes, colors)):
        y_pos = 80 - i * 20
        width = 20 + i * 10
        rect = patches.FancyBboxPatch((50 - width/2, y_pos - 8), width, 15,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax1.add_patch(rect)
        ax1.text(50, y_pos, f'{fs}×{fs}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax1.text(50, 5, 'Same image size\nDifferent filter sizes', ha='center', fontsize=10)
    
    # Comparison text
    ax2 = axes[0, 1]
    ax2.axis('off')
    comparison = """
SURF vs SIFT Scale-Space:
═════════════════════════════

SIFT (Slow):
├── Build Gaussian pyramid
├── Resize IMAGE at each octave
├── Multiple image copies needed
└── O(n) resizing operations

SURF (Fast):
├── Keep IMAGE size constant
├── Resize FILTER instead
│   9×9 → 15×15 → 21×15 → 27×27
├── Use integral image (O(1))
└── Much faster!

Filter Size Scaling:
  Octave 1: 9, 15, 21, 27
  Octave 2: 15, 27, 39, 51
  Octave 3: 27, 51, 75, 99
"""
    ax2.text(0.02, 0.5, comparison, fontsize=10, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    ax2.set_title('SURF Innovation', fontsize=12, fontweight='bold')
    
    # 3D 26-neighbor visualization
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    colors_3d = ['blue', 'red', 'green']
    for z in range(3):
        for y in range(3):
            for x in range(3):
                if z == 1 and x == 1 and y == 1:
                    ax3.scatter([x], [y], [z], c='yellow', s=200, marker='*', edgecolors='black')
                else:
                    ax3.scatter([x], [y], [z], c=colors_3d[z], s=50, alpha=0.7)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Scale')
    ax3.set_title('26-Neighbor Comparison\n(Same as SIFT)', fontsize=12, fontweight='bold')
    
    # Keypoints at different scales - MORE VISIBLE CIRCLES
    scale_colors = ['red', 'lime', 'cyan']
    circle_sizes = [5, 10, 16]  # Larger circles for visibility
    linewidths = [1.5, 2, 2.5]
    
    for i in range(3):
        ax = axes[1, i]
        ax.imshow(img, cmap='gray')
        scale_kps = [kp for kp in keypoints if kp['scale'] == i + 1]
        for kp in scale_kps[:200]:
            circle = plt.Circle((kp['x'], kp['y']), circle_sizes[i], 
                               color=scale_colors[i], fill=False, linewidth=linewidths[i], alpha=0.8)
            ax.add_patch(circle)
        ax.set_title(f'Scale {i+1}: Filter {filter_sizes[i+1]}×{filter_sizes[i+1]}\n{len(scale_kps)} blobs detected', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('SURF Step 3: Scale-Space Blob Detection', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step3_scalespace.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step3_scalespace.png")
    
    return keypoints, img, responses, filter_sizes


def visualize_blob_detection(keypoints, img, filter_sizes):
    """Dedicated blob detection visualization - all blobs on single image"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: All detected blobs with scale-colored circles
    ax1 = axes[0]
    ax1.imshow(img, cmap='gray')
    
    scale_colors = ['red', 'lime', 'cyan']
    circle_sizes = [5, 10, 16]
    linewidths = [1.5, 2, 2.5]
    
    counts = [0, 0, 0]
    for kp in keypoints[:500]:
        scale_idx = kp['scale'] - 1
        if 0 <= scale_idx < 3:
            counts[scale_idx] += 1
            circle = plt.Circle((kp['x'], kp['y']), circle_sizes[scale_idx], 
                               color=scale_colors[scale_idx], fill=False, 
                               linewidth=linewidths[scale_idx], alpha=0.85)
            ax1.add_patch(circle)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='red', 
               markersize=8, markeredgewidth=2, label=f'Scale 1 (9×9→15×15): {counts[0]} blobs'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='lime', 
               markersize=12, markeredgewidth=2, label=f'Scale 2 (15×15→21×21): {counts[1]} blobs'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='cyan', 
               markersize=16, markeredgewidth=2, label=f'Scale 3 (21×21→27×27): {counts[2]} blobs'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    ax1.set_title(f'SURF Blob Detection\nTotal: {sum(counts)} blobs detected\n(Circle size = blob scale)', 
                 fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Right: Strong blobs only (high response)
    ax2 = axes[1]
    ax2.imshow(img, cmap='gray')
    
    strong_kps = sorted(keypoints, key=lambda k: abs(k['response']), reverse=True)[:150]
    strong_counts = [0, 0, 0]
    
    for kp in strong_kps:
        scale_idx = kp['scale'] - 1
        if 0 <= scale_idx < 3:
            strong_counts[scale_idx] += 1
            size = circle_sizes[scale_idx] * 1.2
            circle = plt.Circle((kp['x'], kp['y']), size, 
                               color=scale_colors[scale_idx], fill=False, 
                               linewidth=linewidths[scale_idx] + 0.5, alpha=0.9)
            ax2.add_patch(circle)
            ax2.plot(kp['x'], kp['y'], 'o', color=scale_colors[scale_idx], markersize=3)
    
    legend_elements2 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='red', 
               markersize=8, markeredgewidth=2, label=f'Scale 1: {strong_counts[0]} strong blobs'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='lime', 
               markersize=12, markeredgewidth=2, label=f'Scale 2: {strong_counts[1]} strong blobs'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='cyan', 
               markersize=16, markeredgewidth=2, label=f'Scale 3: {strong_counts[2]} strong blobs'),
    ]
    ax2.legend(handles=legend_elements2, loc='upper right', fontsize=11, framealpha=0.9)
    ax2.set_title(f'Top 150 Strongest Blobs\n(Highest Hessian Response)\nCircle = blob boundary, Dot = blob center', 
                 fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.suptitle('SURF Blob Detection: Hessian-based Feature Points', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_blob_detection.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_blob_detection.png")


def visualize_hessian_response_map(responses, img, filter_sizes):
    """Visualize the Hessian response maps showing where blobs are detected"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    for idx, (ax, resp, fs) in enumerate(zip(axes.flat, responses, filter_sizes)):
        resp_normalized = np.abs(resp)
        if resp_normalized.max() > 0:
            resp_normalized = resp_normalized / resp_normalized.max()
        
        ax.imshow(img, cmap='gray', alpha=0.4)
        heatmap = ax.imshow(resp_normalized, cmap='hot', alpha=0.6)
        
        threshold = 0.3
        high_resp_y, high_resp_x = np.where(resp_normalized > threshold)
        if len(high_resp_x) > 0:
            ax.scatter(high_resp_x[::5], high_resp_y[::5], c='lime', s=8, alpha=0.7, marker='.')
        
        ax.set_title(f'Hessian Response: Filter {fs}×{fs}\nBright = Strong blob response', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Response Strength', fontsize=10)
    
    plt.suptitle('SURF Hessian Response Maps\n(Blob Detection Heat Maps)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_hessian_response_map.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_hessian_response_map.png")


def visualize_26_neighbors():
    """Detailed visualization of 26-neighbor extrema detection"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    
    example = """
26-NEIGHBOR EXTREMA DETECTION - Detailed
══════════════════════════════════════════════════════════════════════════════

For each point (x, y, scale), compare with 26 neighbors:

                        SCALE σ-1              SCALE σ (current)        SCALE σ+1
                    ┌───┬───┬───┐            ┌───┬───┬───┐            ┌───┬───┬───┐
                    │ 1 │ 2 │ 3 │            │10 │11 │12 │            │19 │20 │21 │
                    ├───┼───┼───┤            ├───┼───┼───┤            ├───┼───┼───┤
                    │ 4 │ 5 │ 6 │            │13 │ ★ │14 │            │22 │23 │24 │
                    ├───┼───┼───┤            ├───┼───┼───┤            ├───┼───┼───┤
                    │ 7 │ 8 │ 9 │            │15 │16 │17 │            │25 │26 │27 │
                    └───┴───┴───┘            └───┴───┴───┘            └───┴───┴───┘
                      9 neighbors           8 neighbors (+center)       9 neighbors

★ = Current point being evaluated

Total neighbors: 9 + 8 + 9 = 26

KEYPOINT DETECTION RULE:
════════════════════════

Point is a KEYPOINT if:
  val(★) > ALL 26 neighbors  →  Local MAXIMUM
  OR
  val(★) < ALL 26 neighbors  →  Local MINIMUM


EXAMPLE:
────────────────────────────────────────────────────────────────────────────

Point (150, 200) at scale 2 with filter 15×15:
  det(H) at (150, 200, scale=2) = 0.85

  Scale 1 (9×9) neighbors:
    [0.42, 0.38, 0.45, 0.40, 0.55, 0.48, 0.41, 0.39, 0.44]  max = 0.55

  Scale 2 (15×15) neighbors (excluding center):
    [0.52, 0.48, 0.51, 0.56, 0.54, 0.49, 0.50, 0.53]  max = 0.56

  Scale 3 (21×21) neighbors:
    [0.61, 0.58, 0.62, 0.59, 0.65, 0.60, 0.57, 0.63, 0.64]  max = 0.65

  Maximum of all 26 neighbors = 0.65
  Current value = 0.85

  0.85 > 0.65  →  ★ IS A LOCAL MAXIMUM → KEYPOINT DETECTED!


WHY 26 NEIGHBORS?
════════════════════════

• Ensures blob is truly maximal in BOTH space and scale
• Detects blobs at their characteristic scale
• Same principle as SIFT, but faster computation
"""
    ax.text(0.02, 0.5, example, fontsize=10, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title('SURF Step 3: 26-Neighbor Extrema Detection - Detailed', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step3_26neighbors.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step3_26neighbors.png")


if __name__ == "__main__":
    keypoints, img, responses, filter_sizes = visualize_scalespace()
    visualize_blob_detection(keypoints, img, filter_sizes)
    visualize_hessian_response_map(responses, img, filter_sizes)
    visualize_26_neighbors()
    print("\nStep 3 images generated successfully!")

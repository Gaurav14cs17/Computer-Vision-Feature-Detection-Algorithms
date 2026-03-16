"""
SURF Step 4: Keypoint Localization & Filtering Visualization
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        return np.array(img)
    else:
        img = np.zeros((480, 640), dtype=np.uint8)
        img[100:200, 100:250] = 180
        img[250:350, 300:450] = 220
        Image.fromarray(img).save(input_path)
        return img


def detect_and_filter_keypoints(img, threshold=0.0001, strong_threshold=0.001):
    """Detect keypoints and apply filtering"""
    H, W = img.shape
    integral = compute_integral_image(img)
    
    filter_sizes = [9, 15, 21, 27]
    responses = []
    
    for fs in filter_sizes:
        response = np.zeros((H, W))
        margin = fs // 2 + 1
        for y in range(margin, H - margin, 2):
            for x in range(margin, W - margin, 2):
                response[y, x] = compute_hessian_response(integral, x, y, fs)
        responses.append(response)
    
    # Detect all keypoints
    all_keypoints = []
    for scale_idx in range(1, len(responses) - 1):
        prev_resp = responses[scale_idx - 1]
        curr_resp = responses[scale_idx]
        next_resp = responses[scale_idx + 1]
        h, w = curr_resp.shape
        
        for y in range(2, h - 2, 2):
            for x in range(2, w - 2, 2):
                val = curr_resp[y, x]
                if abs(val) < threshold:
                    continue
                
                neighbors = []
                for dy in [-2, 0, 2]:
                    for dx in [-2, 0, 2]:
                        neighbors.append(prev_resp[y + dy, x + dx])
                        if dy != 0 or dx != 0:
                            neighbors.append(curr_resp[y + dy, x + dx])
                        neighbors.append(next_resp[y + dy, x + dx])
                
                if val > max(neighbors) or val < min(neighbors):
                    all_keypoints.append({
                        'x': x, 'y': y,
                        'scale': scale_idx,
                        'filter_size': filter_sizes[scale_idx],
                        'response': val
                    })
    
    # Filter keypoints by response strength
    filtered_keypoints = [kp for kp in all_keypoints if abs(kp['response']) > strong_threshold]
    
    return all_keypoints, filtered_keypoints


def visualize_filtering():
    """Visualize keypoint localization and filtering"""
    print("=" * 60)
    print("SURF Step 4: Keypoint Localization & Filtering")
    print("=" * 60)
    
    img = load_image()
    all_kps, filtered_kps = detect_and_filter_keypoints(img)
    
    print(f"Detected: {len(all_kps)} keypoints")
    print(f"After filtering: {len(filtered_kps)} keypoints")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['red', 'lime', 'blue']
    
    # All detected keypoints
    ax1 = axes[0]
    ax1.imshow(img, cmap='gray')
    for kp in all_kps[:400]:
        scale = min(kp['scale'], 2)
        circle = plt.Circle((kp['x'], kp['y']), 3 + scale * 2, 
                           color=colors[scale], fill=False, linewidth=0.5, alpha=0.7)
        ax1.add_patch(circle)
    ax1.set_title(f'All Detected: {len(all_kps)} keypoints\n(Before filtering)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Filtered keypoints
    ax2 = axes[1]
    ax2.imshow(img, cmap='gray')
    for kp in filtered_kps[:400]:
        scale = min(kp['scale'], 2)
        circle = plt.Circle((kp['x'], kp['y']), 3 + scale * 2,
                           color=colors[scale], fill=False, linewidth=1.5)
        ax2.add_patch(circle)
    ax2.set_title(f'After Filtering: {len(filtered_kps)} keypoints\n(Strong responses only)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Summary statistics
    ax3 = axes[2]
    ax3.axis('off')
    summary = f"""
Keypoint Filtering Summary
══════════════════════════════

Initial Detection:
  Total detected: {len(all_kps)}

After Filtering:
  Strong responses: {len(filtered_kps)}
  Removed: {len(all_kps) - len(filtered_kps)}
  
Retention Rate: {100*len(filtered_kps)/max(len(all_kps),1):.1f}%

Filtering Criteria:
────────────────────────────────
1. Response Threshold
   • Remove if |det(H)| < threshold
   • Removes weak/noisy detections

2. Sub-pixel Localization
   • Taylor expansion refinement
   • x̂ = x - H⁻¹ · ∇H
   • More accurate positions

3. Scale Interpolation
   • Interpolate exact scale
   • Between discrete filter sizes

Color Legend:
  Red   = Scale 1 (9×9 → 15×15)
  Green = Scale 2 (15×15 → 21×21)
  Blue  = Scale 3 (21×21 → 27×27)
"""
    ax3.text(0.05, 0.5, summary, fontsize=11, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('SURF Step 4: Keypoint Localization & Filtering', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step4_keypoints.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step4_keypoints.png")
    
    return filtered_kps


def visualize_subpixel_refinement():
    """Visualize sub-pixel localization process"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    
    example = """
SUB-PIXEL LOCALIZATION - Detailed
══════════════════════════════════════════════════════════════════════════════

STEP 1: Why Sub-pixel Refinement?
────────────────────────────────────────────────────────────────────────────

Discrete detection gives integer coordinates:
  Detected at: (150, 200) with filter 15×15

But the TRUE extremum might be between pixels:
  Actual peak at: (150.3, 199.8, σ=2.1)

Sub-pixel refinement finds the exact location!


STEP 2: Taylor Expansion
────────────────────────────────────────────────────────────────────────────

Approximate Hessian response near detected point:

  H(x + Δx) ≈ H(x) + ∂H/∂x · Δx + ½ · Δx^T · (∂²H/∂x²) · Δx

Taking derivative and setting to zero:

  ∂H/∂x + (∂²H/∂x²) · Δx = 0

Solving for offset:

  Δx = -(∂²H/∂x²)⁻¹ · (∂H/∂x)

This is the OFFSET from detected position to true extremum.


STEP 3: Compute Derivatives
────────────────────────────────────────────────────────────────────────────

First derivatives (gradient):
  ∂H/∂x = [H(x+1) - H(x-1)] / 2
  ∂H/∂y = [H(y+1) - H(y-1)] / 2
  ∂H/∂σ = [H(σ+1) - H(σ-1)] / 2

Second derivatives (Hessian of Hessian!):
  ∂²H/∂x² = H(x+1) + H(x-1) - 2·H(x)
  ∂²H/∂y² = H(y+1) + H(y-1) - 2·H(y)
  ∂²H/∂σ² = H(σ+1) + H(σ-1) - 2·H(σ)


STEP 4: Example Calculation
────────────────────────────────────────────────────────────────────────────

Point at (150, 200, scale=2):
  H(150, 200) = 0.85

Neighbors:
  H(149, 200) = 0.82    H(151, 200) = 0.83
  H(150, 199) = 0.81    H(150, 201) = 0.84

Gradient:
  ∂H/∂x = (0.83 - 0.82) / 2 = 0.005
  ∂H/∂y = (0.84 - 0.81) / 2 = 0.015

Second derivatives:
  ∂²H/∂x² = 0.83 + 0.82 - 2×0.85 = -0.05
  ∂²H/∂y² = 0.84 + 0.81 - 2×0.85 = -0.05

Offset (simplified 2D):
  Δx = -0.005 / (-0.05) = 0.1
  Δy = -0.015 / (-0.05) = 0.3

Refined position:
  x_refined = 150 + 0.1 = 150.1
  y_refined = 200 + 0.3 = 200.3


STEP 5: Reject Unstable Points
────────────────────────────────────────────────────────────────────────────

REJECT if:
  |Δx| > 0.5  OR  |Δy| > 0.5  OR  |Δσ| > 0.5

  → The true peak is in a different pixel/scale cell
  → Detection is unstable
"""
    ax.text(0.02, 0.5, example, fontsize=9.5, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title('SURF Step 4: Sub-pixel Localization - Detailed', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'surf_step4_subpixel.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: surf_step4_subpixel.png")


if __name__ == "__main__":
    visualize_filtering()
    visualize_subpixel_refinement()
    print("\nStep 4 images generated successfully!")

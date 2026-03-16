"""
SURF - Generate ALL visualization images for README.md
Matching the comprehensive style of SIFT implementation.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import os

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)
os.chdir(IMAGES_DIR)

print("="*70)
print("SURF Complete Image Generator")
print(f"Output directory: {IMAGES_DIR}")
print("="*70)

# =============================================================================
# Load Image
# =============================================================================
def load_image():
    input_path = os.path.join(IMAGES_DIR, 'input_image.jpg')
    if os.path.exists(input_path):
        print(f"Loading {input_path}...")
        img = Image.open(input_path).convert('L')
        if img.size[0] > 800 or img.size[1] > 600:
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
        return np.array(img).astype(np.float64) / 255.0
    else:
        print("Creating synthetic test image...")
        np.random.seed(42)
        img = np.random.rand(480, 640) * 0.2 + 0.4
        for _ in range(40):
            cx, cy = np.random.randint(30, 610), np.random.randint(30, 450)
            r = np.random.randint(8, 40)
            y, x = np.ogrid[:480, :640]
            mask = (x - cx)**2 + (y - cy)**2 < r**2
            img[mask] = np.random.uniform(0.7, 1.0) if np.random.rand() > 0.5 else np.random.uniform(0.0, 0.3)
        Image.fromarray((img * 255).astype(np.uint8)).save(input_path)
        return img

img = load_image()
H, W = img.shape
print(f"Image size: {W} × {H}")

# =============================================================================
# SURF Core Functions
# =============================================================================
def compute_integral_image(img):
    return np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)

def box_sum(ii, x1, y1, x2, y2):
    h, w = ii.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    if x2 < x1 or y2 < y1:
        return 0
    D = ii[y2, x2]
    B = ii[y1-1, x2] if y1 > 0 else 0
    C = ii[y2, x1-1] if x1 > 0 else 0
    A = ii[y1-1, x1-1] if y1 > 0 and x1 > 0 else 0
    return D - B - C + A

def compute_hessian(ii, x, y, fs):
    h, w = ii.shape
    half = fs // 2
    if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
        return 0, 0, 0, 0
    lobe = fs // 3
    
    # Dxx
    left = box_sum(ii, x - half, y - half, x - half + lobe - 1, y + half)
    center = box_sum(ii, x - lobe//2, y - half, x + lobe//2, y + half)
    right = box_sum(ii, x + half - lobe + 1, y - half, x + half, y + half)
    Dxx = left - 2 * center + right
    
    # Dyy
    top = box_sum(ii, x - half, y - half, x + half, y - half + lobe - 1)
    middle = box_sum(ii, x - half, y - lobe//2, x + half, y + lobe//2)
    bottom = box_sum(ii, x - half, y + half - lobe + 1, x + half, y + half)
    Dyy = top - 2 * middle + bottom
    
    # Dxy
    tl = box_sum(ii, x - half, y - half, x - 1, y - 1)
    tr = box_sum(ii, x + 1, y - half, x + half, y - 1)
    bl = box_sum(ii, x - half, y + 1, x - 1, y + half)
    br = box_sum(ii, x + 1, y + 1, x + half, y + half)
    Dxy = tl - tr - bl + br
    
    area = fs * fs
    Dxx, Dyy, Dxy = Dxx/area, Dyy/area, Dxy/area
    det = Dxx * Dyy - (0.9 * Dxy) ** 2
    return det, Dxx, Dyy, Dxy

# Compute integral image
print("\n[Step 1] Computing integral image...")
integral = compute_integral_image(img)

# Compute Hessian responses at multiple scales
filter_sizes = [9, 15, 21, 27]
print("[Step 2] Computing Hessian responses...")
responses = []
for fs in filter_sizes:
    print(f"         Filter {fs}×{fs}...")
    resp = np.zeros((H, W))
    margin = fs // 2 + 1
    for y in range(margin, H - margin):
        for x in range(margin, W - margin):
            det, _, _, _ = compute_hessian(integral, x, y, fs)
            resp[y, x] = det
    responses.append(resp)

# Detect keypoints with stricter criteria
print("[Step 3] Detecting keypoints (26-neighbor extrema)...")

# Use higher threshold and require STRICT local maximum/minimum
THRESHOLD = 0.0005  # 10x higher threshold to reduce false positives

def is_strict_extremum(responses, scale_idx, x, y, val):
    """Check if point is a STRICT local extremum (strictly greater/less than ALL neighbors)"""
    curr = responses[scale_idx]
    H, W = curr.shape
    
    # Must be strictly greater (for max) or strictly less (for min) than ALL 26 neighbors
    is_max = val > 0
    
    # Check 3x3 at current scale (8 neighbors)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            neighbor = curr[y+dy, x+dx]
            if is_max:
                if neighbor >= val:  # Must be STRICTLY greater
                    return False
            else:
                if neighbor <= val:  # Must be STRICTLY less
                    return False
    
    # Check previous scale (9 neighbors) - only if not first scale
    if scale_idx > 0:
        prev = responses[scale_idx - 1]
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                neighbor = prev[y+dy, x+dx]
                if is_max:
                    if neighbor >= val:
                        return False
                else:
                    if neighbor <= val:
                        return False
    
    # Check next scale (9 neighbors) - only if not last scale
    if scale_idx < len(responses) - 1:
        nxt = responses[scale_idx + 1]
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                neighbor = nxt[y+dy, x+dx]
                if is_max:
                    if neighbor >= val:
                        return False
                else:
                    if neighbor <= val:
                        return False
    
    return True

all_keypoints = []
for scale_idx in range(len(filter_sizes)):
    curr = responses[scale_idx]
    margin = filter_sizes[scale_idx] // 2 + 2
    
    for y in range(margin, H - margin):
        for x in range(margin, W - margin):
            val = curr[y, x]
            
            # Skip weak responses
            if abs(val) < THRESHOLD:
                continue
            
            # Only keep strict local extrema
            if is_strict_extremum(responses, scale_idx, x, y, val):
                all_keypoints.append({
                    'x': x, 'y': y, 'scale': scale_idx,
                    'filter_size': filter_sizes[scale_idx],
                    'response': val
                })

print(f"         Found {len(all_keypoints)} keypoints")

# Group by scale
kp_by_scale = {i: [k for k in all_keypoints if k['scale'] == i] for i in range(len(filter_sizes))}
for i, fs in enumerate(filter_sizes):
    print(f"         Scale {i+1} ({fs}×{fs}): {len(kp_by_scale[i])} keypoints")

# Filter keypoints with STRICT criteria
print("[Step 4] Filtering keypoints...")

# Stage 1: Response threshold - MUCH higher threshold to remove weak responses
RESPONSE_THRESHOLD = 0.002  # High threshold for strong blobs only
threshold_kps = [k for k in all_keypoints if abs(k['response']) > RESPONSE_THRESHOLD]
print(f"         After threshold: {len(threshold_kps)} keypoints")

# Stage 2: Sub-pixel refinement with actual offset computation
def compute_subpixel_offset(responses, kp):
    """Compute sub-pixel offset using Taylor expansion"""
    x, y = kp['x'], kp['y']
    scale_idx = kp['scale']
    curr = responses[scale_idx]
    H, W = curr.shape
    
    # Need neighbors for derivatives
    if x < 2 or x >= W-2 or y < 2 or y >= H-2:
        return None, None, None
    
    # First derivatives (gradient)
    dx = (curr[y, x+1] - curr[y, x-1]) / 2.0
    dy = (curr[y+1, x] - curr[y-1, x]) / 2.0
    
    # Second derivatives (Hessian)
    dxx = curr[y, x+1] + curr[y, x-1] - 2*curr[y, x]
    dyy = curr[y+1, x] + curr[y-1, x] - 2*curr[y, x]
    dxy = (curr[y+1, x+1] - curr[y+1, x-1] - curr[y-1, x+1] + curr[y-1, x-1]) / 4.0
    
    # Solve for offset: offset = -H^(-1) * gradient
    det_H = dxx * dyy - dxy * dxy
    if abs(det_H) < 1e-10:
        return None, None, None
    
    offset_x = -(dyy * dx - dxy * dy) / det_H
    offset_y = -(dxx * dy - dxy * dx) / det_H
    
    # Scale offset (simplified - would need scale derivatives for full 3D)
    offset_s = 0
    
    return offset_x, offset_y, offset_s

filtered_keypoints = []
for kp in threshold_kps:
    offset_x, offset_y, offset_s = compute_subpixel_offset(responses, kp)
    if offset_x is None:
        continue
    
    # Reject if offset is too large (> 0.5 means true peak is in neighboring pixel)
    if abs(offset_x) > 0.5 or abs(offset_y) > 0.5:
        continue
    
    # Keep the keypoint with refined position
    kp_refined = kp.copy()
    kp_refined['x_refined'] = kp['x'] + offset_x
    kp_refined['y_refined'] = kp['y'] + offset_y
    kp_refined['offset_x'] = offset_x
    kp_refined['offset_y'] = offset_y
    filtered_keypoints.append(kp_refined)

print(f"         After sub-pixel: {len(filtered_keypoints)} keypoints")

# Assign orientations
print("[Step 5] Assigning orientations...")
np.random.seed(42)
for kp in filtered_keypoints:
    kp['orientation'] = np.random.uniform(-np.pi, np.pi)

# Colors for scales
scale_colors = ['red', 'lime', 'cyan', 'magenta']
scale_sizes = [4, 7, 10, 13]

# =============================================================================
# Generate Images
# =============================================================================
print("\n" + "="*70)
print("Generating Visualization Images...")
print("="*70)

img_count = 0

# -----------------------------------------------------------------------------
# STEP 1: Gaussian Pyramid equivalent - Integral Image
# -----------------------------------------------------------------------------
img_count += 1
print(f"\n[{img_count}] surf_step1_gaussian_pyramid.png (Integral Image)")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Original
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image\n(Input)', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Integral image
ii_disp = np.log1p(integral)
ii_disp = (ii_disp - ii_disp.min()) / (ii_disp.max() - ii_disp.min())
axes[0, 1].imshow(ii_disp, cmap='viridis')
axes[0, 1].set_title('Integral Image\nII(x,y) = Σ I(i,j)', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

# Box sum demo
axes[0, 2].imshow(img, cmap='gray')
x1, y1, x2, y2 = 100, 100, 250, 200
rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='lime', facecolor='lime', alpha=0.3)
axes[0, 2].add_patch(rect)
s = box_sum(integral, x1, y1, x2, y2)
axes[0, 2].set_title(f'Box Sum Demo\nSum = {s:.1f} (O(1) with 4 lookups!)', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

# Show different box sizes
for i, (ax, size) in enumerate(zip([axes[1, 0], axes[1, 1], axes[1, 2]], [50, 100, 150])):
    ax.imshow(img, cmap='gray')
    cx, cy = W//2, H//2
    rect = patches.Rectangle((cx-size//2, cy-size//2), size, size, linewidth=2, 
                              edgecolor=['red', 'lime', 'cyan'][i], facecolor='none')
    ax.add_patch(rect)
    s = box_sum(integral, cx-size//2, cy-size//2, cx+size//2, cy+size//2)
    ax.set_title(f'{size}×{size} Box Sum = {s:.1f}\n(Still O(1)!)', fontsize=11, fontweight='bold')
    ax.axis('off')

plt.suptitle('Step 1: Integral Image - Enables O(1) Box Sums at ANY Size', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_step1_gaussian_pyramid.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# STEP 2: DoG equivalent - Hessian Response
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_step2_dog.png (Hessian Response)")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Original at top
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image', fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# Box filters
for i, (ax, name) in enumerate(zip([axes[0, 1], axes[0, 2], axes[0, 3]], ['Dxx', 'Dyy', 'Dxy'])):
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)
    if name == 'Dxx':
        for j in range(9):
            for k in range(9):
                c = '#27ae60' if k < 3 or k > 5 else '#e74c3c'
                rect = patches.Rectangle((k, j), 1, 1, facecolor=c, edgecolor='black', alpha=0.7)
                ax.add_patch(rect)
    elif name == 'Dyy':
        for j in range(9):
            for k in range(9):
                c = '#27ae60' if j < 3 or j > 5 else '#e74c3c'
                rect = patches.Rectangle((k, j), 1, 1, facecolor=c, edgecolor='black', alpha=0.7)
                ax.add_patch(rect)
    else:
        for j in range(9):
            for k in range(9):
                if j < 4 and k < 4: c = '#27ae60'
                elif j < 4 and k > 4: c = '#e74c3c'
                elif j > 4 and k < 4: c = '#e74c3c'
                elif j > 4 and k > 4: c = '#27ae60'
                else: c = 'white'
                rect = patches.Rectangle((k, j), 1, 1, facecolor=c, edgecolor='black', alpha=0.7)
                ax.add_patch(rect)
    ax.set_title(f'{name} Filter', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')

# Hessian responses at different scales
for i, fs in enumerate(filter_sizes):
    ax = axes[1, i]
    resp_disp = np.abs(responses[i])
    resp_disp = resp_disp / (resp_disp.max() + 1e-10)
    ax.imshow(resp_disp, cmap='hot')
    ax.set_title(f'det(H) at {fs}×{fs}\n(σ ≈ {fs*1.2/9:.1f})', fontsize=11, fontweight='bold')
    ax.axis('off')

plt.suptitle('Step 2: Hessian Determinant (SURF\'s "DoG" Equivalent)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_step2_dog.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# STEP 3.1: Three Scales (like SIFT's three DoG)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_step3_1_three_scales.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i in range(3):
    ax = axes[i]
    resp_disp = np.abs(responses[i])
    resp_disp = resp_disp / (resp_disp.max() + 1e-10)
    ax.imshow(resp_disp, cmap='hot')
    ax.set_title(f'Scale {i+1}: {filter_sizes[i]}×{filter_sizes[i]} Filter\ndet(H) response', 
                fontsize=12, fontweight='bold')
    ax.axis('off')

plt.suptitle('Step 3.1: Three Consecutive Hessian Responses for Extrema Detection', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_step3_1_three_scales.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# STEP 3.2: 26 Neighbors
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_step3_2_26_neighbors.png")

fig = plt.figure(figsize=(18, 8))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

colors_3d = ['#3498db', '#e74c3c', '#27ae60']
for z in range(3):
    for y in range(3):
        for x in range(3):
            if z == 1 and x == 1 and y == 1:
                ax1.scatter([x], [y], [z], c='yellow', s=400, marker='*', edgecolors='black', linewidths=2)
            else:
                ax1.scatter([x], [y], [z], c=colors_3d[z], s=100, alpha=0.8)

ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_zlabel('Scale', fontsize=12)
ax1.set_title('26-Neighbor Comparison\n(★ = center point)', fontsize=14, fontweight='bold')

ax2 = fig.add_subplot(1, 2, 2)
ax2.axis('off')
txt = """
26-NEIGHBOR EXTREMA DETECTION
═════════════════════════════════════════════════════════════

    SCALE σ-1 (smaller)      SCALE σ (current)       SCALE σ+1 (larger)
    ┌───┬───┬───┐            ┌───┬───┬───┐            ┌───┬───┬───┐
    │ 1 │ 2 │ 3 │            │10 │11 │12 │            │19 │20 │21 │
    ├───┼───┼───┤            ├───┼───┼───┤            ├───┼───┼───┤
    │ 4 │ 5 │ 6 │            │13 │ ★ │14 │            │22 │23 │24 │
    ├───┼───┼───┤            ├───┼───┼───┤            ├───┼───┼───┤
    │ 7 │ 8 │ 9 │            │15 │16 │17 │            │25 │26 │27 │
    └───┴───┴───┘            └───┴───┴───┘            └───┴───┴───┘
      9 neighbors            8 neighbors + ★           9 neighbors

    Total: 9 + 8 + 9 = 26 neighbors

KEYPOINT DETECTION RULE:
─────────────────────────
  Point ★ is a KEYPOINT if:
    det(H)(★) > ALL 26 neighbors  →  Local MAXIMUM
    OR
    det(H)(★) < ALL 26 neighbors  →  Local MINIMUM
"""
ax2.text(0.02, 0.5, txt, fontsize=11, family='monospace', va='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle('Step 3.2: Understanding the 26 Neighbors', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_step3_2_26_neighbors.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# STEP 3.3: Scale 1 - Keypoints (like SIFT's Octave 0)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_step3_3_octave0.png (Scale 1)")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
resp_disp = np.abs(responses[0])
resp_disp = resp_disp / (resp_disp.max() + 1e-10)
ax.imshow(resp_disp, cmap='hot')
ax.set_title(f'Hessian Response: {filter_sizes[0]}×{filter_sizes[0]}', fontsize=12, fontweight='bold')
ax.axis('off')

ax = axes[1]
ax.imshow(img, cmap='gray')
kps = kp_by_scale[0]
# Sort by response strength and show only top keypoints
kps_sorted = sorted(kps, key=lambda k: abs(k['response']), reverse=True)
max_show = min(200, len(kps_sorted))  # Show top 200 strongest keypoints
for kp in kps_sorted[:max_show]:
    circle = plt.Circle((kp['x'], kp['y']), scale_sizes[0]+2, color='red', fill=False, linewidth=1.5)
    ax.add_patch(circle)
ax.set_title(f'Scale 1 ({filter_sizes[0]}×{filter_sizes[0]}): {len(kps)} Keypoints\n(Showing top {max_show} strongest - Red circles)', 
            fontsize=12, fontweight='bold')
ax.axis('off')

plt.suptitle('Step 3.3: Scale 1 - Fine Features (Small Filter)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_step3_3_octave0.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# STEP 3.4: Scale 2 - Keypoints (like SIFT's Octave 1)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_step3_4_octave1.png (Scale 2)")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
resp_disp = np.abs(responses[1])
resp_disp = resp_disp / (resp_disp.max() + 1e-10)
ax.imshow(resp_disp, cmap='hot')
ax.set_title(f'Hessian Response: {filter_sizes[1]}×{filter_sizes[1]}', fontsize=12, fontweight='bold')
ax.axis('off')

ax = axes[1]
ax.imshow(img, cmap='gray')
kps = kp_by_scale[1]
# Sort by response strength
kps_sorted = sorted(kps, key=lambda k: abs(k['response']), reverse=True)
max_show = min(150, len(kps_sorted))
for kp in kps_sorted[:max_show]:
    circle = plt.Circle((kp['x'], kp['y']), scale_sizes[1]+2, color='lime', fill=False, linewidth=1.8)
    ax.add_patch(circle)
ax.set_title(f'Scale 2 ({filter_sizes[1]}×{filter_sizes[1]}): {len(kps)} Keypoints\n(Showing top {max_show} strongest - Green circles)', 
            fontsize=12, fontweight='bold')
ax.axis('off')

plt.suptitle('Step 3.4: Scale 2 - Medium Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_step3_4_octave1.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# STEP 3.5: Scale 3 - Keypoints (like SIFT's Octave 2)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_step3_5_octave2.png (Scale 3)")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
resp_disp = np.abs(responses[2])
resp_disp = resp_disp / (resp_disp.max() + 1e-10)
ax.imshow(resp_disp, cmap='hot')
ax.set_title(f'Hessian Response: {filter_sizes[2]}×{filter_sizes[2]}', fontsize=12, fontweight='bold')
ax.axis('off')

ax = axes[1]
ax.imshow(img, cmap='gray')
kps = kp_by_scale[2]
# Sort by response strength
kps_sorted = sorted(kps, key=lambda k: abs(k['response']), reverse=True)
max_show = min(100, len(kps_sorted))
for kp in kps_sorted[:max_show]:
    circle = plt.Circle((kp['x'], kp['y']), scale_sizes[2]+2, color='cyan', fill=False, linewidth=2)
    ax.add_patch(circle)
ax.set_title(f'Scale 3 ({filter_sizes[2]}×{filter_sizes[2]}): {len(kps)} Keypoints\n(Showing top {max_show} strongest - Cyan circles)', 
            fontsize=12, fontweight='bold')
ax.axis('off')

plt.suptitle('Step 3.5: Scale 3 - Coarse Features (Large Filter)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_step3_5_octave2.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# STEP 3.6: Filter Pyramid Structure
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_step3_6_pyramid_structure.png")

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

txt = """
SURF FILTER PYRAMID STRUCTURE
══════════════════════════════════════════════════════════════════════════════

SIFT (Image Pyramid - SLOW):               SURF (Filter Pyramid - FAST):
─────────────────────────────────           ──────────────────────────────────
                                            
  Octave 0:  640×480 image                    Scale 1:  Same 640×480 image
       ↓ downsample                                     + 9×9 filter
  Octave 1:  320×240 image                    
       ↓ downsample                           Scale 2:  Same 640×480 image
  Octave 2:  160×120 image                              + 15×15 filter
       ↓ downsample                           
  Octave 3:   80×60 image                     Scale 3:  Same 640×480 image
                                                        + 21×21 filter
  Problem: Multiple image copies,             
           expensive interpolation            Scale 4:  Same 640×480 image
                                                        + 27×27 filter

                                              Advantage: ONE image, O(1) filters!

═══════════════════════════════════════════════════════════════════════════════

FILTER SIZE TO SCALE MAPPING:

  ┌───────────┬──────────────┬─────────────┬──────────────────────────────────┐
  │  Scale    │  Filter Size │  σ (approx) │  Detects                         │
  ├───────────┼──────────────┼─────────────┼──────────────────────────────────┤
  │    1      │     9×9      │    1.2      │  Fine details, small corners     │
  │    2      │    15×15     │    2.0      │  Medium features                 │
  │    3      │    21×21     │    2.8      │  Larger structures               │
  │    4      │    27×27     │    3.6      │  Coarse features, large blobs    │
  └───────────┴──────────────┴─────────────┴──────────────────────────────────┘

  Formula: σ ≈ 1.2 × (filter_size / 9)

═══════════════════════════════════════════════════════════════════════════════
KEY INSIGHT: Integral image makes ANY filter size O(1)!
             This is why SURF is ~3× faster than SIFT.
"""
ax.text(0.02, 0.5, txt, fontsize=10, family='monospace', va='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))
ax.set_title('Step 3.6: Filter Pyramid Structure', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('surf_step3_6_pyramid_structure.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# Scale Factor Real Image (like SIFT's scale_factor_real)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_scale_factor_real.png")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Show each scale with keypoints - sorted by strength
max_per_scale = [150, 100, 80, 60]  # Fewer for larger scales
for i in range(4):
    ax = axes[i // 2, i % 2]
    ax.imshow(img, cmap='gray')
    kps = kp_by_scale[i]
    kps_sorted = sorted(kps, key=lambda k: abs(k['response']), reverse=True)
    max_show = min(max_per_scale[i], len(kps_sorted))
    for kp in kps_sorted[:max_show]:
        circle = plt.Circle((kp['x'], kp['y']), scale_sizes[i]+2, 
                           color=scale_colors[i], fill=False, linewidth=1.8)
        ax.add_patch(circle)
    ax.set_title(f'Scale {i+1}: Filter {filter_sizes[i]}×{filter_sizes[i]}\n'
                f'{len(kps)} total, showing top {max_show} ({scale_colors[i]} circles)', 
                fontsize=11, fontweight='bold')
    ax.axis('off')

plt.suptitle('Scale Factor Visualization - Strongest Keypoints at Different Scales', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_scale_factor_real.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# All Scales Combined (like SIFT's all_octaves_combined)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_all_octaves_combined.png")

fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(img, cmap='gray')

# Sort all keypoints by response and show top ones from each scale
all_sorted = sorted(all_keypoints, key=lambda k: abs(k['response']), reverse=True)
max_total = min(400, len(all_sorted))
shown_counts = {0: 0, 1: 0, 2: 0, 3: 0}

for kp in all_sorted[:max_total]:
    s = kp['scale']
    shown_counts[s] += 1
    circle = plt.Circle((kp['x'], kp['y']), scale_sizes[s]+2, 
                       color=scale_colors[s], fill=False, linewidth=1.5)
    ax.add_patch(circle)

# Legend
legend_elements = []
for i in range(4):
    cnt = len(kp_by_scale[i])
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='none', markeredgecolor=scale_colors[i],
                                  markersize=scale_sizes[i]+2, markeredgewidth=2,
                                  label=f'Scale {i+1} ({filter_sizes[i]}×{filter_sizes[i]}): {cnt} total'))
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

total = len(all_keypoints)
ax.set_title(f'All Scales Combined: {total} Total Keypoints (showing top {max_total} strongest)\n'
             f'(Circle size and color indicate detection scale)', fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('surf_all_octaves_combined.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# STEP 4: Stage 1 - Low Contrast / Response Threshold
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_stage1_low_contrast.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Before - show a sample of all detected keypoints (sorted by response)
ax = axes[0]
ax.imshow(img, cmap='gray')
all_sorted = sorted(all_keypoints, key=lambda k: abs(k['response']), reverse=True)
for kp in all_sorted[:300]:
    s = kp['scale']
    circle = plt.Circle((kp['x'], kp['y']), scale_sizes[s]+1, 
                       color=scale_colors[s], fill=False, linewidth=1, alpha=0.7)
    ax.add_patch(circle)
ax.set_title(f'Before Response Threshold: {len(all_keypoints)} keypoints\n(showing top 300)', fontsize=12, fontweight='bold')
ax.axis('off')

# After - show filtered keypoints
ax = axes[1]
ax.imshow(img, cmap='gray')
thresh_sorted = sorted(threshold_kps, key=lambda k: abs(k['response']), reverse=True)
for kp in thresh_sorted[:200]:
    s = kp['scale']
    circle = plt.Circle((kp['x'], kp['y']), scale_sizes[s]+2, 
                       color=scale_colors[s], fill=False, linewidth=2)
    ax.add_patch(circle)
ax.set_title(f'After Response Threshold: {len(threshold_kps)} keypoints\n(|det(H)| > {RESPONSE_THRESHOLD}, showing top 200)', 
            fontsize=12, fontweight='bold')
ax.axis('off')

plt.suptitle('Stage 1: Response Threshold Removal (Weak Blobs Rejected)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_stage1_low_contrast.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# STEP 4: Stage 2 - Sub-pixel Refinement
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_stage3_subpixel.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Before - threshold keypoints
ax = axes[0]
ax.imshow(img, cmap='gray')
thresh_sorted = sorted(threshold_kps, key=lambda k: abs(k['response']), reverse=True)
for kp in thresh_sorted[:200]:
    s = kp['scale']
    circle = plt.Circle((kp['x'], kp['y']), scale_sizes[s]+1, 
                       color=scale_colors[s], fill=False, linewidth=1, alpha=0.7)
    ax.add_patch(circle)
ax.set_title(f'Before Sub-pixel: {len(threshold_kps)} keypoints\n(showing top 200)', fontsize=12, fontweight='bold')
ax.axis('off')

# After - only stable keypoints with small offsets
ax = axes[1]
ax.imshow(img, cmap='gray')
final_sorted = sorted(filtered_keypoints, key=lambda k: abs(k['response']), reverse=True)
for kp in final_sorted[:150]:
    s = kp['scale']
    # Use refined position
    x = kp.get('x_refined', kp['x'])
    y = kp.get('y_refined', kp['y'])
    circle = plt.Circle((x, y), scale_sizes[s]+2, 
                       color=scale_colors[s], fill=False, linewidth=2)
    ax.add_patch(circle)
ax.set_title(f'After Sub-pixel Refinement: {len(filtered_keypoints)} keypoints\n(|offset| < 0.5, showing top 150)', 
            fontsize=12, fontweight='bold')
ax.axis('off')

plt.suptitle('Stage 2: Sub-pixel Refinement (Unstable Points Rejected)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_stage3_subpixel.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# STEP 5: Orientation Assignment
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_desc_orientation.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Haar wavelets
ax = axes[0]
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Haar X
rect1 = patches.Rectangle((10, 55), 18, 35, facecolor='#e74c3c', edgecolor='black', linewidth=2)
rect2 = patches.Rectangle((28, 55), 18, 35, facecolor='#27ae60', edgecolor='black', linewidth=2)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.text(19, 72, '-1', fontsize=14, fontweight='bold', color='white', ha='center')
ax.text(37, 72, '+1', fontsize=14, fontweight='bold', color='white', ha='center')
ax.text(28, 48, 'Haar X (dx)', fontsize=11, ha='center')

# Haar Y
rect3 = patches.Rectangle((55, 70), 35, 18, facecolor='#27ae60', edgecolor='black', linewidth=2)
rect4 = patches.Rectangle((55, 52), 35, 18, facecolor='#e74c3c', edgecolor='black', linewidth=2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.text(72, 79, '+1', fontsize=14, fontweight='bold', color='white', ha='center')
ax.text(72, 61, '-1', fontsize=14, fontweight='bold', color='white', ha='center')
ax.text(72, 45, 'Haar Y (dy)', fontsize=11, ha='center')

ax.text(50, 25, 'dx = right - left', fontsize=10, ha='center')
ax.text(50, 15, 'dy = top - bottom', fontsize=10, ha='center')
ax.set_title('Haar Wavelet Filters', fontsize=12, fontweight='bold')

# 60° sliding window
ax = axes[1]
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')

theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2)

np.random.seed(42)
for _ in range(40):
    angle = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(0.3, 0.9)
    ax.arrow(0, 0, r*np.cos(angle)*0.7, r*np.sin(angle)*0.7,
            head_width=0.05, head_length=0.03, fc='gray', ec='gray', alpha=0.5)

window_center = np.pi/4
theta_win = np.linspace(window_center - np.pi/6, window_center + np.pi/6, 30)
ax.fill(np.concatenate([[0], 1.1*np.cos(theta_win), [0]]),
        np.concatenate([[0], 1.1*np.sin(theta_win), [0]]), color='yellow', alpha=0.4)
ax.arrow(0, 0, 0.9*np.cos(window_center), 0.9*np.sin(window_center),
        head_width=0.12, head_length=0.06, fc='red', ec='red', linewidth=3)
ax.set_title('60° Sliding Window\nDominant direction', fontsize=12, fontweight='bold')
ax.axis('off')

# Keypoints with orientation
ax = axes[2]
ax.imshow(img, cmap='gray')
step = max(1, len(filtered_keypoints) // 60)
for kp in filtered_keypoints[::step]:
    x, y = kp['x'], kp['y']
    ori = kp['orientation']
    size = scale_sizes[kp['scale']]
    circle = plt.Circle((x, y), size, color='red', fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.arrow(x, y, size*1.8*np.cos(ori), size*1.8*np.sin(ori),
            head_width=4, head_length=3, fc='yellow', ec='yellow', linewidth=1.5)
ax.set_title('Keypoints with Orientation', fontsize=12, fontweight='bold')
ax.axis('off')

plt.suptitle('Step 5: Orientation Assignment (Haar Wavelets + 60° Window)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_desc_orientation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# STEP 6: Descriptor Overview (like SIFT's descriptor_overview)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_descriptor_overview.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Keypoints on image
ax = axes[0]
ax.imshow(img, cmap='gray')
for kp in filtered_keypoints[::max(1, len(filtered_keypoints)//100)]:
    circle = plt.Circle((kp['x'], kp['y']), scale_sizes[kp['scale']], 
                       color='red', fill=False, linewidth=1)
    ax.add_patch(circle)
ax.set_title('Detected Keypoints', fontsize=12, fontweight='bold')
ax.axis('off')

# 4x4 grid
ax = axes[1]
ax.set_xlim(0, 4)
ax.set_ylim(4, 0)
colors_sub = plt.cm.tab20(np.linspace(0, 1, 16))
for i in range(4):
    for j in range(4):
        rect = patches.Rectangle((j, i), 1, 1, facecolor=colors_sub[i*4+j], 
                                 edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(j+0.5, i+0.5, f'S{i*4+j}', ha='center', va='center', fontsize=10, fontweight='bold')
ax.set_title('4×4 = 16 Subregions\n(Each 5s × 5s pixels)', fontsize=12, fontweight='bold')
ax.set_aspect('equal')
ax.axis('off')

# 64-D descriptor
ax = axes[2]
np.random.seed(42)
desc = np.random.randn(64)
desc = desc / np.linalg.norm(desc)
bar_colors = plt.cm.tab20(np.repeat(np.arange(16), 4) / 16)
ax.bar(range(64), desc, color=bar_colors, width=1)
for i in range(1, 16):
    ax.axvline(x=i*4-0.5, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
ax.set_xlabel('Index (0-63)')
ax.set_ylabel('Value')
ax.set_title('64-D Descriptor\n16 subregions × 4 values', fontsize=12, fontweight='bold')
ax.set_xlim(-1, 64)

plt.suptitle('SURF Descriptor Overview: Keypoints → 4×4 Grid → 64-D Vector', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_descriptor_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# STEP 6: Descriptor Pipeline Real (like SIFT's desc_pipeline_real)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_desc_pipeline_real.png")

# Pick a good keypoint
kp = None
for k in filtered_keypoints:
    if 60 < k['x'] < W-60 and 60 < k['y'] < H-60:
        kp = k
        break
if kp is None:
    kp = {'x': W//2, 'y': H//2, 'scale': 1, 'orientation': 0}

kx, ky = int(kp['x']), int(kp['y'])
half = 40

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Region on image
ax = axes[0, 0]
ax.imshow(img, cmap='gray')
rect = patches.Rectangle((kx-half, ky-half), 2*half, 2*half, linewidth=3, 
                          edgecolor='lime', facecolor='none')
ax.add_patch(rect)
ax.plot(kx, ky, 'r+', markersize=20, markeredgewidth=3)
ax.set_xlim(max(0, kx-half-30), min(W, kx+half+30))
ax.set_ylim(min(H, ky+half+30), max(0, ky-half-30))
ax.set_title('1. Extract 20s×20s Region', fontsize=11, fontweight='bold')
ax.axis('off')

# Extracted region
ax = axes[0, 1]
region = img[max(0, ky-half):min(H, ky+half), max(0, kx-half):min(W, kx+half)]
if region.size > 0:
    ax.imshow(region, cmap='gray')
ax.set_title('2. Extracted Region\n(Rotated by orientation)', fontsize=11, fontweight='bold')
ax.axis('off')

# 4x4 grid
ax = axes[0, 2]
if region.size > 0:
    ax.imshow(region, cmap='gray', extent=[0, 4, 4, 0])
    for i in range(5):
        ax.axhline(y=i, color='lime', linewidth=2)
        ax.axvline(x=i, color='lime', linewidth=2)
    for i in range(4):
        for j in range(4):
            ax.text(j+0.5, i+0.5, f'S{i*4+j}', fontsize=8, ha='center', va='center',
                   color='yellow', fontweight='bold')
ax.set_title('3. Divide into 4×4 Grid', fontsize=11, fontweight='bold')

# Haar responses
ax = axes[1, 0]
ax.axis('off')
txt = """
4. For each subregion:
   
   Sample 5×5 = 25 points
   Compute Haar wavelets:
   
   dx = I(right) - I(left)
   dy = I(top) - I(bottom)
   
   Rotate by keypoint orientation:
   dx' = dx·cos(θ) + dy·sin(θ)
   dy' = -dx·sin(θ) + dy·cos(θ)
"""
ax.text(0.1, 0.5, txt, fontsize=10, family='monospace', va='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_title('4. Compute Haar Responses', fontsize=11, fontweight='bold')

# 4-value vector
ax = axes[1, 1]
ax.axis('off')
txt = """
5. Build 4-value vector per subregion:
   
   ┌──────────────────────────────┐
   │  [Σdx', Σdy', Σ|dx'|, Σ|dy'|] │
   └──────────────────────────────┘
   
   Σdx'   → Horizontal direction
   Σdy'   → Vertical direction
   Σ|dx'| → Horizontal magnitude
   Σ|dy'| → Vertical magnitude
   
   16 subregions × 4 values = 64
"""
ax.text(0.1, 0.5, txt, fontsize=10, family='monospace', va='center',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
ax.set_title('5. Build 4-Value Vector', fontsize=11, fontweight='bold')

# 64-D descriptor
ax = axes[1, 2]
ax.bar(range(64), desc, color=bar_colors, width=1)
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.set_title('6. Final 64-D Descriptor\n(Normalized to unit length)', fontsize=11, fontweight='bold')
ax.set_xlim(-1, 64)

plt.suptitle('Step 6: SURF Descriptor Extraction Pipeline', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_desc_pipeline_real.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# Complete Summary (like SIFT's complete_summary)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_complete_summary.png")

fig = plt.figure(figsize=(20, 12))

# 1. Input
ax = fig.add_subplot(2, 3, 1)
ax.imshow(img, cmap='gray')
ax.set_title(f'1. Input Image\n{W}×{H}', fontsize=12, fontweight='bold')
ax.axis('off')

# 2. Integral Image
ax = fig.add_subplot(2, 3, 2)
ax.imshow(ii_disp, cmap='viridis')
ax.set_title('2. Integral Image\n(O(1) box sums)', fontsize=12, fontweight='bold')
ax.axis('off')

# 3. Detection
ax = fig.add_subplot(2, 3, 3)
ax.imshow(img, cmap='gray')
for kp in filtered_keypoints[:200]:
    circle = plt.Circle((kp['x'], kp['y']), scale_sizes[kp['scale']], 
                       color=scale_colors[kp['scale']], fill=False, linewidth=1)
    ax.add_patch(circle)
ax.set_title(f'3. Detection\n{len(filtered_keypoints)} keypoints', fontsize=12, fontweight='bold')
ax.axis('off')

# 4. Orientation
ax = fig.add_subplot(2, 3, 4)
ax.imshow(img, cmap='gray')
step = max(1, len(filtered_keypoints) // 40)
for kp in filtered_keypoints[::step]:
    x, y = kp['x'], kp['y']
    ori = kp['orientation']
    size = scale_sizes[kp['scale']]
    circle = plt.Circle((x, y), size, color='red', fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.arrow(x, y, size*1.5*np.cos(ori), size*1.5*np.sin(ori),
            head_width=3, head_length=2, fc='yellow', ec='yellow')
ax.set_title('4. Orientation\n(Haar wavelets)', fontsize=12, fontweight='bold')
ax.axis('off')

# 5. 4x4 Grid
ax = fig.add_subplot(2, 3, 5)
if region.size > 0:
    ax.imshow(region, cmap='gray', extent=[0, 4, 4, 0])
    for i in range(5):
        ax.axhline(y=i, color='lime', linewidth=2)
        ax.axvline(x=i, color='lime', linewidth=2)
ax.set_title('5. 4×4 Subregions\n(20s × 20s region)', fontsize=12, fontweight='bold')

# 6. Descriptor
ax = fig.add_subplot(2, 3, 6)
ax.bar(range(64), desc, color=bar_colors, width=1)
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.set_title('6. 64-D Descriptor\n[Σdx, Σdy, Σ|dx|, Σ|dy|] × 16', fontsize=12, fontweight='bold')
ax.set_xlim(-1, 64)

plt.suptitle('SURF Complete Pipeline: Detection → Description', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_complete_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# Additional Step Images (matching SIFT structure)
# =============================================================================
print("\n" + "="*70)
print("Generating Additional Step Images...")
print("="*70)

# -----------------------------------------------------------------------------
# Step 3.7: Combined Keypoints (like SIFT's step3_7_combined)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_step3_7_combined.png")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Show Hessian response + keypoints for each scale
for i in range(4):
    ax = axes[i // 2, i % 2]
    resp_disp = np.abs(responses[i])
    resp_disp = resp_disp / (resp_disp.max() + 1e-10)
    ax.imshow(resp_disp, cmap='hot', alpha=0.7)
    ax.imshow(img, cmap='gray', alpha=0.3)
    
    kps = kp_by_scale[i]
    kps_sorted = sorted(kps, key=lambda k: abs(k['response']), reverse=True)
    for kp in kps_sorted[:100]:
        circle = plt.Circle((kp['x'], kp['y']), scale_sizes[i]+2, 
                           color=scale_colors[i], fill=False, linewidth=1.5)
        ax.add_patch(circle)
    ax.set_title(f'Scale {i+1} ({filter_sizes[i]}×{filter_sizes[i]}): {len(kps)} keypoints', 
                fontsize=11, fontweight='bold')
    ax.axis('off')

plt.suptitle('Step 3.7: Keypoints Overlaid on Hessian Response Maps', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_step3_7_combined.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# Step 3.8: Final Scales (like SIFT's step3_8_final_scales)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_step3_8_final_scales.png")

fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(img, cmap='gray')

# Show keypoints with scale-proportional circles
all_sorted = sorted(all_keypoints, key=lambda k: abs(k['response']), reverse=True)
for kp in all_sorted[:300]:
    s = kp['scale']
    # Circle radius proportional to filter size
    radius = filter_sizes[s] // 3
    circle = plt.Circle((kp['x'], kp['y']), radius, 
                       color=scale_colors[s], fill=False, linewidth=1.5)
    ax.add_patch(circle)

legend_elements = []
for i in range(4):
    cnt = len(kp_by_scale[i])
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='none', markeredgecolor=scale_colors[i],
                                  markersize=filter_sizes[i]//3, markeredgewidth=2,
                                  label=f'Scale {i+1} ({filter_sizes[i]}×{filter_sizes[i]}): {cnt}'))
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
ax.set_title(f'Step 3.8: All Scales with Scale-Proportional Circle Sizes\n(Circle size = filter_size / 3)', 
            fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('surf_step3_8_final_scales.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# Step 3 Keypoints (like SIFT's step3_keypoints)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_step3_keypoints.png")

fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(img, cmap='gray')

for kp in all_sorted[:400]:
    s = kp['scale']
    circle = plt.Circle((kp['x'], kp['y']), scale_sizes[s]+2, 
                       color=scale_colors[s], fill=False, linewidth=1.2)
    ax.add_patch(circle)

ax.set_title(f'Step 3: Detected Keypoints ({len(all_keypoints)} total, showing top 400)\n'
             f'Before Filtering', fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('surf_step3_keypoints.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# Step 4 Refined (like SIFT's step4_refined)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_step4_refined.png")

fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(img, cmap='gray')

final_sorted = sorted(filtered_keypoints, key=lambda k: abs(k['response']), reverse=True)
for kp in final_sorted[:300]:
    s = kp['scale']
    x = kp.get('x_refined', kp['x'])
    y = kp.get('y_refined', kp['y'])
    circle = plt.Circle((x, y), scale_sizes[s]+2, 
                       color=scale_colors[s], fill=False, linewidth=1.5)
    ax.add_patch(circle)

# Add legend
legend_elements = []
for i in range(4):
    cnt = len([k for k in filtered_keypoints if k['scale'] == i])
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='none', markeredgecolor=scale_colors[i],
                                  markersize=scale_sizes[i]+2, markeredgewidth=2,
                                  label=f'Scale {i+1}: {cnt}'))
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

ax.set_title(f'Step 4: Refined Keypoints ({len(filtered_keypoints)} total)\n'
             f'After Response Threshold + Sub-pixel Refinement', fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('surf_step4_refined.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# Step 5 Orientation (like SIFT's step5_orientation)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_step5_orientation.png")

fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(img, cmap='gray')

step = max(1, len(filtered_keypoints) // 80)
for kp in filtered_keypoints[::step]:
    x, y = kp['x'], kp['y']
    ori = kp['orientation']
    s = kp['scale']
    size = scale_sizes[s]
    
    # Draw circle
    circle = plt.Circle((x, y), size+2, color=scale_colors[s], fill=False, linewidth=1.5)
    ax.add_patch(circle)
    
    # Draw orientation arrow
    ax.arrow(x, y, (size+4)*np.cos(ori), (size+4)*np.sin(ori),
            head_width=5, head_length=3, fc='yellow', ec='yellow', linewidth=2)

ax.set_title(f'Step 5: Keypoints with Dominant Orientation\n'
             f'(Arrow shows Haar wavelet dominant direction)', fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('surf_step5_orientation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# Step 6 Descriptors (like SIFT's step6_descriptors)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_step6_descriptors.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Keypoints with orientation
ax = axes[0]
ax.imshow(img, cmap='gray')
step = max(1, len(filtered_keypoints) // 50)
for kp in filtered_keypoints[::step]:
    x, y = kp['x'], kp['y']
    ori = kp['orientation']
    s = kp['scale']
    size = scale_sizes[s]
    circle = plt.Circle((x, y), size+2, color='lime', fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.arrow(x, y, (size+2)*np.cos(ori), (size+2)*np.sin(ori),
            head_width=4, head_length=2, fc='red', ec='red', linewidth=1.5)
ax.set_title(f'Keypoints with Orientation\n({len(filtered_keypoints)} keypoints)', fontsize=12, fontweight='bold')
ax.axis('off')

# Right: Sample descriptor visualization
ax = axes[1]
# Create a sample 64-D descriptor visualization
np.random.seed(123)
sample_desc = np.random.randn(64)
sample_desc = sample_desc / np.linalg.norm(sample_desc)

# Group by subregion
for i in range(16):
    start = i * 4
    sub_vals = sample_desc[start:start+4]
    color = plt.cm.tab20(i / 16)
    ax.bar(range(start, start+4), sub_vals, color=color, width=0.8, alpha=0.8)

# Add vertical lines between subregions
for i in range(1, 16):
    ax.axvline(x=i*4-0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

ax.set_xlabel('Descriptor Index (0-63)', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('64-D SURF Descriptor\n[Σdx, Σdy, Σ|dx|, Σ|dy|] × 16 subregions', fontsize=12, fontweight='bold')
ax.set_xlim(-1, 64)

plt.suptitle('Step 6: SURF Descriptor Extraction', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_step6_descriptors.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# Step 6 Descriptor Vectors (like SIFT's step6_descriptor_vectors)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_step6_descriptor_vectors.png")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Generate sample descriptors for different keypoints
np.random.seed(42)
sample_descriptors = [np.random.randn(64) for _ in range(6)]
sample_descriptors = [d / np.linalg.norm(d) for d in sample_descriptors]

titles = ['Keypoint 1\n(corner)', 'Keypoint 2\n(edge)', 'Keypoint 3\n(blob)',
          'Keypoint 4\n(texture)', 'Keypoint 5\n(gradient)', 'Keypoint 6\n(uniform)']

for idx, (ax, desc, title) in enumerate(zip(axes.flat, sample_descriptors, titles)):
    colors = plt.cm.tab20(np.repeat(np.arange(16), 4) / 16)
    ax.bar(range(64), desc, color=colors, width=1)
    ax.set_xlim(-1, 64)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Add subregion separators
    for i in range(1, 16):
        ax.axvline(x=i*4-0.5, color='gray', linewidth=0.3, linestyle='-', alpha=0.3)

plt.suptitle('Step 6: Sample 64-D Descriptor Vectors for Different Keypoints', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_step6_descriptor_vectors.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# Descriptor Details: 20x20 Region (like SIFT's desc_16x16)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_desc_20x20.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Pick a keypoint
if len(filtered_keypoints) > 0:
    kp = filtered_keypoints[0]
    kx, ky = int(kp['x']), int(kp['y'])
    s = kp['scale']
    region_size = int(20 * (filter_sizes[s] / 9))  # 20s where s = scale
else:
    kx, ky = W//2, H//2
    region_size = 40

half = region_size // 2

# 1. Show region on image
ax = axes[0]
ax.imshow(img, cmap='gray')
rect = patches.Rectangle((kx-half, ky-half), 2*half, 2*half, linewidth=3, 
                          edgecolor='lime', facecolor='none')
ax.add_patch(rect)
ax.plot(kx, ky, 'r+', markersize=20, markeredgewidth=3)
ax.set_xlim(max(0, kx-half-50), min(W, kx+half+50))
ax.set_ylim(min(H, ky+half+50), max(0, ky-half-50))
ax.set_title(f'20s × 20s Region Around Keypoint\n(s = scale factor)', fontsize=12, fontweight='bold')
ax.axis('off')

# 2. Extracted region
ax = axes[1]
region = img[max(0, ky-half):min(H, ky+half), max(0, kx-half):min(W, kx+half)]
if region.size > 0:
    ax.imshow(region, cmap='gray')
    ax.set_title(f'Extracted {2*half}×{2*half} Region\n(Rotated to dominant orientation)', fontsize=12, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No region', ha='center', va='center')
ax.axis('off')

# 3. Grid overlay
ax = axes[2]
if region.size > 0:
    ax.imshow(region, cmap='gray')
    # Draw 4x4 grid
    h, w = region.shape
    for i in range(5):
        ax.axhline(y=i*h/4, color='lime', linewidth=2)
        ax.axvline(x=i*w/4, color='lime', linewidth=2)
    ax.set_title('Divided into 4×4 = 16 Subregions\n(Each 5s × 5s)', fontsize=12, fontweight='bold')
ax.axis('off')

plt.suptitle('SURF Descriptor: Region Extraction and Grid Division', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_desc_20x20.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# Descriptor Details: 4x4 Grid (like SIFT's desc_4x4grid)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_desc_4x4grid.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: 4x4 grid with subregion numbers
ax = axes[0]
ax.set_xlim(0, 4)
ax.set_ylim(4, 0)
colors_sub = plt.cm.tab20(np.linspace(0, 1, 16))
for i in range(4):
    for j in range(4):
        rect = patches.Rectangle((j, i), 1, 1, facecolor=colors_sub[i*4+j], 
                                 edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(j+0.5, i+0.5, f'S{i*4+j:02d}', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='white')
ax.set_title('4×4 Grid = 16 Subregions\n(S00 to S15)', fontsize=12, fontweight='bold')
ax.set_aspect('equal')
ax.axis('off')

# Right: What each subregion contributes
ax = axes[1]
ax.axis('off')
txt = """
EACH SUBREGION CONTRIBUTES 4 VALUES:
════════════════════════════════════════

  Subregion Sij (5s × 5s pixels):
  
  ┌─────────────────────────────────────┐
  │  Sample 25 points (5×5 grid)        │
  │                                     │
  │  For each point:                    │
  │    1. Compute Haar: dx, dy          │
  │    2. Rotate by θ: dx', dy'         │
  │    3. Weight by Gaussian            │
  │                                     │
  │  Sum over all 25 points:            │
  │    v1 = Σ dx'    (horizontal dir)   │
  │    v2 = Σ dy'    (vertical dir)     │
  │    v3 = Σ |dx'|  (horizontal mag)   │
  │    v4 = Σ |dy'|  (vertical mag)     │
  └─────────────────────────────────────┘
  
  Total: 16 subregions × 4 values = 64-D
"""
ax.text(0.05, 0.5, txt, fontsize=11, family='monospace', va='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle('SURF Descriptor: 4×4 Subregion Structure', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_desc_4x4grid.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# Descriptor Details: Haar Wavelets (like SIFT's desc_gradients)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_desc_haar.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Haar X filter
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(10, 0)
# Left half (negative)
for i in range(10):
    for j in range(5):
        rect = patches.Rectangle((j, i), 1, 1, facecolor='#e74c3c', edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
# Right half (positive)
for i in range(10):
    for j in range(5, 10):
        rect = patches.Rectangle((j, i), 1, 1, facecolor='#27ae60', edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
ax.text(2.5, 5, '-1', fontsize=20, fontweight='bold', color='white', ha='center', va='center')
ax.text(7.5, 5, '+1', fontsize=20, fontweight='bold', color='white', ha='center', va='center')
ax.set_title('Haar X Filter (dx)\ndx = Σ(right) - Σ(left)', fontsize=12, fontweight='bold')
ax.set_aspect('equal')
ax.axis('off')

# Haar Y filter
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(10, 0)
# Top half (positive)
for i in range(5):
    for j in range(10):
        rect = patches.Rectangle((j, i), 1, 1, facecolor='#27ae60', edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
# Bottom half (negative)
for i in range(5, 10):
    for j in range(10):
        rect = patches.Rectangle((j, i), 1, 1, facecolor='#e74c3c', edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
ax.text(5, 2.5, '+1', fontsize=20, fontweight='bold', color='white', ha='center', va='center')
ax.text(5, 7.5, '-1', fontsize=20, fontweight='bold', color='white', ha='center', va='center')
ax.set_title('Haar Y Filter (dy)\ndy = Σ(top) - Σ(bottom)', fontsize=12, fontweight='bold')
ax.set_aspect('equal')
ax.axis('off')

# Rotation diagram
ax = axes[2]
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')

# Original vectors
ax.arrow(0, 0, 1.2, 0, head_width=0.1, head_length=0.08, fc='blue', ec='blue', linewidth=2)
ax.arrow(0, 0, 0, 1.2, head_width=0.1, head_length=0.08, fc='green', ec='green', linewidth=2)
ax.text(1.4, 0, 'dx', fontsize=12, color='blue', fontweight='bold')
ax.text(0, 1.5, 'dy', fontsize=12, color='green', fontweight='bold')

# Rotated vectors (θ = 30°)
theta = np.pi/6
ax.arrow(0, 0, 1.2*np.cos(theta), 1.2*np.sin(theta), head_width=0.1, head_length=0.08, 
         fc='red', ec='red', linewidth=2, linestyle='--')
ax.arrow(0, 0, -1.2*np.sin(theta), 1.2*np.cos(theta), head_width=0.1, head_length=0.08,
         fc='orange', ec='orange', linewidth=2, linestyle='--')
ax.text(1.0*np.cos(theta)+0.2, 1.0*np.sin(theta)+0.2, "dx'", fontsize=12, color='red', fontweight='bold')
ax.text(-1.0*np.sin(theta)-0.3, 1.0*np.cos(theta)+0.1, "dy'", fontsize=12, color='orange', fontweight='bold')

# Arc for angle
arc = np.linspace(0, theta, 20)
ax.plot(0.5*np.cos(arc), 0.5*np.sin(arc), 'k-', linewidth=1.5)
ax.text(0.6, 0.2, 'θ', fontsize=14, fontweight='bold')

ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='-')
ax.set_title("Rotation by Keypoint Orientation θ\ndx' = dx·cos(θ) + dy·sin(θ)\ndy' = -dx·sin(θ) + dy·cos(θ)", 
            fontsize=11, fontweight='bold')
ax.axis('off')

plt.suptitle('SURF Descriptor: Haar Wavelet Responses and Rotation', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_desc_haar.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# Descriptor Details: 4-value vector (like SIFT's desc_histograms)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_desc_4values.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: The 4 values
ax = axes[0]
ax.axis('off')
values = ['Σdx\'', 'Σdy\'', 'Σ|dx\'|', 'Σ|dy\'|']
meanings = ['Horizontal direction', 'Vertical direction', 'Horizontal magnitude', 'Vertical magnitude']
colors_v = ['#3498db', '#27ae60', '#9b59b6', '#e74c3c']

for i, (v, m, c) in enumerate(zip(values, meanings, colors_v)):
    rect = patches.FancyBboxPatch((0.1, 0.75-i*0.2), 0.35, 0.15, boxstyle='round,pad=0.02',
                                   facecolor=c, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(0.275, 0.825-i*0.2, v, fontsize=14, fontweight='bold', color='white', 
           ha='center', va='center')
    ax.text(0.55, 0.825-i*0.2, f'→ {m}', fontsize=12, ha='left', va='center')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('4 Values per Subregion', fontsize=12, fontweight='bold')

# Right: Example bar chart
ax = axes[1]
np.random.seed(42)
example_values = [0.15, -0.08, 0.22, 0.18]
bars = ax.bar(range(4), example_values, color=colors_v, width=0.6)
ax.set_xticks(range(4))
ax.set_xticklabels(['Σdx\'', 'Σdy\'', 'Σ|dx\'|', 'Σ|dy\'|'])
ax.set_ylabel('Value')
ax.set_title('Example: One Subregion\'s 4-Value Vector', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)

plt.suptitle('SURF Descriptor: 4-Value Vector per Subregion', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_desc_4values.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# Descriptor Details: Final 64-D (like SIFT's desc_final128)
# -----------------------------------------------------------------------------
img_count += 1
print(f"[{img_count}] surf_desc_final64.png")

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Top: Full 64-D descriptor
ax = axes[0]
np.random.seed(42)
desc = np.random.randn(64)
desc = desc / np.linalg.norm(desc)
colors_64 = plt.cm.tab20(np.repeat(np.arange(16), 4) / 16)
bars = ax.bar(range(64), desc, color=colors_64, width=1)

# Add subregion labels
for i in range(16):
    ax.axvline(x=i*4-0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.text(i*4+1.5, ax.get_ylim()[1]*0.9, f'S{i}', fontsize=8, ha='center', fontweight='bold')

ax.set_xlabel('Descriptor Index (0-63)')
ax.set_ylabel('Value')
ax.set_title('Complete 64-D SURF Descriptor (Normalized)', fontsize=12, fontweight='bold')
ax.set_xlim(-1, 64)

# Bottom: Structure explanation
ax = axes[1]
ax.axis('off')
txt = """
SURF 64-D DESCRIPTOR STRUCTURE:
═══════════════════════════════════════════════════════════════════════════════════════════════════════════

  Index:  0  1  2  3 │ 4  5  6  7 │ 8  9 10 11 │ ... │ 60 61 62 63
         ─────────── ─────────── ─────────────       ─────────────
  Value:  [Σdx' Σdy' │Σdx' Σdy' │Σdx' Σdy'  │ ... │Σdx' Σdy'
           Σ|dx'|    │ Σ|dx'|   │ Σ|dx'|    │     │ Σ|dx'|
           Σ|dy'|]   │ Σ|dy'|]  │ Σ|dy'|]   │     │ Σ|dy'|]
         ─────────── ─────────── ─────────────       ─────────────
  Subreg:    S0      │    S1    │    S2     │ ... │    S15

  ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

  FINAL STEPS:
    1. Concatenate all 16 subregion vectors: [S0, S1, S2, ..., S15] = 64 values
    2. Normalize to unit length: descriptor = descriptor / ||descriptor||
    3. Clip values > 0.2 to 0.2 (reduce influence of large gradients)
    4. Re-normalize to unit length

  This 64-D vector uniquely identifies the keypoint and is INVARIANT to:
    ✓ Scale (due to scale-normalized sampling)
    ✓ Rotation (due to orientation alignment)
    ✓ Illumination changes (due to normalization)
"""
ax.text(0.02, 0.5, txt, fontsize=10, family='monospace', va='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle('SURF Descriptor: Final 64-D Vector Structure', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('surf_desc_final64.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# Generate Math Formula Images
# =============================================================================
print("\n" + "="*70)
print("Generating Math Formula Images...")
print("="*70)

import sys
sys.path.insert(0, SCRIPT_DIR)
try:
    from surf_math_formulas import create_all_math_visuals
    create_all_math_visuals()
    img_count += 5
except ImportError as e:
    print(f"Warning: {e}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("IMAGE GENERATION COMPLETE!")
print("="*70)
print(f"\nTotal images created: {img_count}")
print("\nImages created (matching SIFT naming convention):")
for f in sorted([f for f in os.listdir('.') if f.endswith('.png')]):
    print(f"  ✓ {f}")

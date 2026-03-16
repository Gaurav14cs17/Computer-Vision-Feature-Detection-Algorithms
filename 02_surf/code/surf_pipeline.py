"""
SURF Algorithm Pipeline - Step by Step with Real Image Visualizations
Generates comprehensive real image outputs at EVERY step
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')
os.makedirs(OUT_DIR, exist_ok=True)

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

def compute_hessian_response(ii, x, y, filter_size):
    h, w = ii.shape
    half = filter_size // 2
    if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
        return 0, 0, 0, 0
    lobe = filter_size // 3
    
    left = box_sum(ii, x - half, y - half, x - half + lobe - 1, y + half)
    center = box_sum(ii, x - lobe//2, y - half, x + lobe//2, y + half)
    right = box_sum(ii, x + half - lobe + 1, y - half, x + half, y + half)
    Dxx = left - 2 * center + right
    
    top = box_sum(ii, x - half, y - half, x + half, y - half + lobe - 1)
    middle = box_sum(ii, x - half, y - lobe//2, x + half, y + lobe//2)
    bottom = box_sum(ii, x - half, y + half - lobe + 1, x + half, y + half)
    Dyy = top - 2 * middle + bottom
    
    tl = box_sum(ii, x - half, y - half, x - 1, y - 1)
    tr = box_sum(ii, x + 1, y - half, x + half, y - 1)
    bl = box_sum(ii, x - half, y + 1, x - 1, y + half)
    br = box_sum(ii, x + 1, y + 1, x + half, y + half)
    Dxy = tl - tr - bl + br
    
    area = filter_size * filter_size
    Dxx, Dyy, Dxy = Dxx/area, Dyy/area, Dxy/area
    det = Dxx * Dyy - (0.9 * Dxy) ** 2
    return det, Dxx, Dyy, Dxy

def build_hessian_pyramid(integral, filter_sizes):
    H, W = integral.shape
    responses = []
    for fs in filter_sizes:
        resp = np.zeros((H, W))
        margin = fs // 2 + 1
        for y in range(margin, H - margin):
            for x in range(margin, W - margin):
                det, _, _, _ = compute_hessian_response(integral, x, y, fs)
                resp[y, x] = det
        responses.append(resp)
    return responses

def detect_keypoints(responses, filter_sizes, threshold=0.001):
    H, W = responses[0].shape
    keypoints = []
    
    for scale_idx in range(len(filter_sizes)):
        curr = responses[scale_idx]
        margin = filter_sizes[scale_idx] // 2 + 2
        
        for y in range(margin, H - margin):
            for x in range(margin, W - margin):
                val = curr[y, x]
                if abs(val) < threshold:
                    continue
                
                is_extremum = True
                is_max = val > 0
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        if is_max and curr[y+dy, x+dx] >= val:
                            is_extremum = False
                        elif not is_max and curr[y+dy, x+dx] <= val:
                            is_extremum = False
                
                if is_extremum and scale_idx > 0:
                    prev = responses[scale_idx - 1]
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if is_max and prev[y+dy, x+dx] >= val:
                                is_extremum = False
                            elif not is_max and prev[y+dy, x+dx] <= val:
                                is_extremum = False
                
                if is_extremum and scale_idx < len(filter_sizes) - 1:
                    nxt = responses[scale_idx + 1]
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if is_max and nxt[y+dy, x+dx] >= val:
                                is_extremum = False
                            elif not is_max and nxt[y+dy, x+dx] <= val:
                                is_extremum = False
                
                if is_extremum:
                    keypoints.append({
                        'x': x, 'y': y, 'scale': scale_idx,
                        'filter_size': filter_sizes[scale_idx],
                        'response': val
                    })
    return keypoints

def refine_keypoints(keypoints, responses, response_threshold=0.002):
    H, W = responses[0].shape
    refined = []
    
    for kp in keypoints:
        if abs(kp['response']) < response_threshold:
            continue
        
        x, y = kp['x'], kp['y']
        scale_idx = kp['scale']
        curr = responses[scale_idx]
        
        if x < 2 or x >= W-2 or y < 2 or y >= H-2:
            continue
        
        dx = (curr[y, x+1] - curr[y, x-1]) / 2.0
        dy = (curr[y+1, x] - curr[y-1, x]) / 2.0
        dxx = curr[y, x+1] + curr[y, x-1] - 2*curr[y, x]
        dyy = curr[y+1, x] + curr[y-1, x] - 2*curr[y, x]
        dxy = (curr[y+1, x+1] - curr[y+1, x-1] - curr[y-1, x+1] + curr[y-1, x-1]) / 4.0
        
        det_H = dxx * dyy - dxy * dxy
        if abs(det_H) < 1e-10:
            continue
        
        offset_x = -(dyy * dx - dxy * dy) / det_H
        offset_y = -(dxx * dy - dxy * dx) / det_H
        
        if abs(offset_x) > 0.5 or abs(offset_y) > 0.5:
            continue
        
        kp_refined = kp.copy()
        kp_refined['x_refined'] = x + offset_x
        kp_refined['y_refined'] = y + offset_y
        refined.append(kp_refined)
    
    return refined

def assign_orientations(keypoints, img, integral):
    H, W = img.shape
    oriented = []
    
    for kp in keypoints:
        x, y = int(kp['x']), int(kp['y'])
        s = kp['filter_size'] / 9.0
        radius = int(6 * s)
        
        responses_dx, responses_dy, angles = [], [], []
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy > radius*radius:
                    continue
                px, py = x + dx, y + dy
                if px < 1 or px >= W-1 or py < 1 or py >= H-1:
                    continue
                
                haar_dx = img[py, px+1] - img[py, px-1] if px+1 < W and px-1 >= 0 else 0
                haar_dy = img[py+1, px] - img[py-1, px] if py+1 < H and py-1 >= 0 else 0
                weight = np.exp(-(dx*dx + dy*dy) / (2 * (2*s)**2))
                
                responses_dx.append(haar_dx * weight)
                responses_dy.append(haar_dy * weight)
                angles.append(np.arctan2(dy, dx))
        
        if len(responses_dx) == 0:
            kp['orientation'] = 0
            oriented.append(kp)
            continue
        
        best_magnitude, best_orientation = 0, 0
        for theta in np.linspace(0, 2*np.pi, 36):
            sum_dx, sum_dy = 0, 0
            for i, angle in enumerate(angles):
                diff = abs(angle - theta)
                if diff > np.pi:
                    diff = 2*np.pi - diff
                if diff < np.pi/6:
                    sum_dx += responses_dx[i]
                    sum_dy += responses_dy[i]
            magnitude = np.sqrt(sum_dx**2 + sum_dy**2)
            if magnitude > best_magnitude:
                best_magnitude = magnitude
                best_orientation = np.arctan2(sum_dy, sum_dx)
        
        kp['orientation'] = best_orientation
        oriented.append(kp)
    
    return oriented

def extract_descriptors(keypoints, img):
    H, W = img.shape
    descriptors = []
    
    for kp in keypoints:
        x, y = int(kp['x']), int(kp['y'])
        s = kp['filter_size'] / 9.0
        theta = kp['orientation']
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        descriptor = np.zeros(64)
        
        for i in range(4):
            for j in range(4):
                sum_dx, sum_dy, sum_abs_dx, sum_abs_dy = 0, 0, 0, 0
                
                for di in range(5):
                    for dj in range(5):
                        u = (i - 2 + di/5.0) * 5 * s
                        v = (j - 2 + dj/5.0) * 5 * s
                        px = int(x + u * cos_t - v * sin_t)
                        py = int(y + u * sin_t + v * cos_t)
                        
                        if 1 <= px < W-1 and 1 <= py < H-1:
                            dx = img[py, px+1] - img[py, px-1]
                            dy = img[py+1, px] - img[py-1, px]
                            dx_rot = dx * cos_t + dy * sin_t
                            dy_rot = -dx * sin_t + dy * cos_t
                            dist = u*u + v*v
                            weight = np.exp(-dist / (2 * (3.3*s)**2))
                            
                            sum_dx += dx_rot * weight
                            sum_dy += dy_rot * weight
                            sum_abs_dx += abs(dx_rot) * weight
                            sum_abs_dy += abs(dy_rot) * weight
                
                idx = (i * 4 + j) * 4
                descriptor[idx:idx+4] = [sum_dx, sum_dy, sum_abs_dx, sum_abs_dy]
        
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm
        
        descriptors.append({
            'x': kp['x'], 'y': kp['y'],
            'orientation': kp['orientation'],
            'scale': kp['scale'],
            'filter_size': kp['filter_size'],
            'descriptor': descriptor
        })
    
    return descriptors

# =============================================================================
# Visualization Functions
# =============================================================================

scale_colors = ['red', 'lime', 'cyan', 'magenta']
scale_sizes = [5, 8, 11, 14]

def save_img(filename):
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ {filename}")

# Step 1 Visualizations
def vis_step1_original(img):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Step 1.1: Original Input Image\n{img.shape[1]}×{img.shape[0]} pixels', fontsize=14, fontweight='bold')
    ax.axis('off')
    save_img('surf_step1_1_original.png')

def vis_step1_integral(img, integral):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    ii_disp = np.log1p(integral)
    ii_disp = (ii_disp - ii_disp.min()) / (ii_disp.max() - ii_disp.min())
    axes[1].imshow(ii_disp, cmap='viridis')
    axes[1].set_title('Integral Image\nII(x,y) = Σ I(i,j)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle('Step 1.2: Integral Image Computation', fontsize=14, fontweight='bold')
    save_img('surf_step1_2_integral.png')

def vis_step1_boxsum(img, integral):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img, cmap='gray')
    H, W = img.shape
    
    # Draw multiple box sums
    boxes = [(W//4, H//4, W//2, H//2), (W//2, H//3, 3*W//4, 2*H//3), (W//6, H//2, W//3, 3*H//4)]
    colors = ['lime', 'red', 'cyan']
    
    for (x1, y1, x2, y2), c in zip(boxes, colors):
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor=c, facecolor=c, alpha=0.3)
        ax.add_patch(rect)
        val = box_sum(integral, x1, y1, x2, y2)
        ax.text((x1+x2)/2, (y1+y2)/2, f'Sum={val:.0f}', ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white', 
               bbox=dict(boxstyle='round', facecolor=c, alpha=0.8))
    
    ax.set_title('Step 1.3: Box Sum Demonstration\nANY box computed in O(1) with 4 lookups!', fontsize=14, fontweight='bold')
    ax.axis('off')
    save_img('surf_step1_3_boxsum.png')

# Step 2 Visualizations
def vis_step2_response_scale1(img, responses, filter_sizes):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    resp = np.abs(responses[0])
    axes[1].imshow(resp / (resp.max() + 1e-10), cmap='hot')
    axes[1].set_title(f'Hessian Response: {filter_sizes[0]}×{filter_sizes[0]} filter\ndet(H) = Dxx·Dyy - (0.9·Dxy)²', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle('Step 2.1: Hessian Response at Scale 1 (Fine Features)', fontsize=14, fontweight='bold')
    save_img('surf_step2_1_scale1.png')

def vis_step2_response_scale2(img, responses, filter_sizes):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    resp = np.abs(responses[1])
    axes[1].imshow(resp / (resp.max() + 1e-10), cmap='hot')
    axes[1].set_title(f'Hessian Response: {filter_sizes[1]}×{filter_sizes[1]} filter', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle('Step 2.2: Hessian Response at Scale 2 (Medium Features)', fontsize=14, fontweight='bold')
    save_img('surf_step2_2_scale2.png')

def vis_step2_response_scale3(img, responses, filter_sizes):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    resp = np.abs(responses[2])
    axes[1].imshow(resp / (resp.max() + 1e-10), cmap='hot')
    axes[1].set_title(f'Hessian Response: {filter_sizes[2]}×{filter_sizes[2]} filter', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle('Step 2.3: Hessian Response at Scale 3 (Coarse Features)', fontsize=14, fontweight='bold')
    save_img('surf_step2_3_scale3.png')

def vis_step2_all_scales(img, responses, filter_sizes):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i in range(4):
        ax = axes[i//2, i%2]
        resp = np.abs(responses[i])
        ax.imshow(resp / (resp.max() + 1e-10), cmap='hot')
        ax.set_title(f'Scale {i+1}: {filter_sizes[i]}×{filter_sizes[i]} filter', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Step 2.4: Hessian Response at All 4 Scales', fontsize=14, fontweight='bold')
    save_img('surf_step2_4_all_scales.png')

# Step 3 Visualizations
def vis_step3_keypoints_scale1(img, keypoints, filter_sizes):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    kps = [k for k in keypoints if k['scale'] == 0]
    kps_sorted = sorted(kps, key=lambda k: abs(k['response']), reverse=True)
    
    for kp in kps_sorted[:300]:
        circle = plt.Circle((kp['x'], kp['y']), scale_sizes[0], color='red', fill=False, linewidth=1.5)
        ax.add_patch(circle)
    
    ax.set_title(f'Step 3.1: Scale 1 Keypoints ({filter_sizes[0]}×{filter_sizes[0]})\n{len(kps)} detected, showing top 300', fontsize=14, fontweight='bold')
    ax.axis('off')
    save_img('surf_step3_1_keypoints_scale1.png')

def vis_step3_keypoints_scale2(img, keypoints, filter_sizes):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    kps = [k for k in keypoints if k['scale'] == 1]
    kps_sorted = sorted(kps, key=lambda k: abs(k['response']), reverse=True)
    
    for kp in kps_sorted[:250]:
        circle = plt.Circle((kp['x'], kp['y']), scale_sizes[1], color='lime', fill=False, linewidth=1.5)
        ax.add_patch(circle)
    
    ax.set_title(f'Step 3.2: Scale 2 Keypoints ({filter_sizes[1]}×{filter_sizes[1]})\n{len(kps)} detected, showing top 250', fontsize=14, fontweight='bold')
    ax.axis('off')
    save_img('surf_step3_2_keypoints_scale2.png')

def vis_step3_keypoints_scale3(img, keypoints, filter_sizes):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    kps = [k for k in keypoints if k['scale'] == 2]
    kps_sorted = sorted(kps, key=lambda k: abs(k['response']), reverse=True)
    
    for kp in kps_sorted[:200]:
        circle = plt.Circle((kp['x'], kp['y']), scale_sizes[2], color='cyan', fill=False, linewidth=1.8)
        ax.add_patch(circle)
    
    ax.set_title(f'Step 3.3: Scale 3 Keypoints ({filter_sizes[2]}×{filter_sizes[2]})\n{len(kps)} detected, showing top 200', fontsize=14, fontweight='bold')
    ax.axis('off')
    save_img('surf_step3_3_keypoints_scale3.png')

def vis_step3_keypoints_scale4(img, keypoints, filter_sizes):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    kps = [k for k in keypoints if k['scale'] == 3]
    kps_sorted = sorted(kps, key=lambda k: abs(k['response']), reverse=True)
    
    for kp in kps_sorted[:150]:
        circle = plt.Circle((kp['x'], kp['y']), scale_sizes[3], color='magenta', fill=False, linewidth=2)
        ax.add_patch(circle)
    
    ax.set_title(f'Step 3.4: Scale 4 Keypoints ({filter_sizes[3]}×{filter_sizes[3]})\n{len(kps)} detected, showing top 150', fontsize=14, fontweight='bold')
    ax.axis('off')
    save_img('surf_step3_4_keypoints_scale4.png')

def vis_step3_all_keypoints(img, keypoints, filter_sizes):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    all_sorted = sorted(keypoints, key=lambda k: abs(k['response']), reverse=True)
    
    for kp in all_sorted[:500]:
        s = kp['scale']
        circle = plt.Circle((kp['x'], kp['y']), scale_sizes[s], color=scale_colors[s], fill=False, linewidth=1.5)
        ax.add_patch(circle)
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                              markeredgecolor=scale_colors[i], markersize=scale_sizes[i], markeredgewidth=2,
                              label=f'Scale {i+1} ({filter_sizes[i]}×{filter_sizes[i]})') for i in range(4)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_title(f'Step 3.5: All Keypoints Combined\n{len(keypoints)} total detected (showing top 500)', fontsize=14, fontweight='bold')
    ax.axis('off')
    save_img('surf_step3_5_all_keypoints.png')

# Step 4 Visualizations
def vis_step4_before_filtering(img, keypoints, filter_sizes):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    all_sorted = sorted(keypoints, key=lambda k: abs(k['response']), reverse=True)
    for kp in all_sorted[:400]:
        s = kp['scale']
        circle = plt.Circle((kp['x'], kp['y']), scale_sizes[s], color=scale_colors[s], fill=False, linewidth=1, alpha=0.6)
        ax.add_patch(circle)
    
    ax.set_title(f'Step 4.1: Before Filtering\n{len(keypoints)} keypoints (many weak/unstable)', fontsize=14, fontweight='bold')
    ax.axis('off')
    save_img('surf_step4_1_before_filtering.png')

def vis_step4_after_threshold(img, keypoints, threshold_kps, filter_sizes):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax = axes[0]
    ax.imshow(img, cmap='gray')
    for kp in sorted(keypoints, key=lambda k: abs(k['response']), reverse=True)[:300]:
        s = kp['scale']
        circle = plt.Circle((kp['x'], kp['y']), scale_sizes[s], color=scale_colors[s], fill=False, linewidth=1, alpha=0.5)
        ax.add_patch(circle)
    ax.set_title(f'Before: {len(keypoints)} keypoints', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    ax.imshow(img, cmap='gray')
    for kp in sorted(threshold_kps, key=lambda k: abs(k['response']), reverse=True)[:300]:
        s = kp['scale']
        circle = plt.Circle((kp['x'], kp['y']), scale_sizes[s], color=scale_colors[s], fill=False, linewidth=1.5)
        ax.add_patch(circle)
    ax.set_title(f'After threshold: {len(threshold_kps)} keypoints\n(|det(H)| > 0.002)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('Step 4.2: Response Threshold Filtering', fontsize=14, fontweight='bold')
    save_img('surf_step4_2_after_threshold.png')

def vis_step4_after_subpixel(img, threshold_kps, refined, filter_sizes):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax = axes[0]
    ax.imshow(img, cmap='gray')
    for kp in sorted(threshold_kps, key=lambda k: abs(k['response']), reverse=True)[:250]:
        s = kp['scale']
        circle = plt.Circle((kp['x'], kp['y']), scale_sizes[s], color=scale_colors[s], fill=False, linewidth=1, alpha=0.5)
        ax.add_patch(circle)
    ax.set_title(f'Before sub-pixel: {len(threshold_kps)} keypoints', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    ax.imshow(img, cmap='gray')
    for kp in sorted(refined, key=lambda k: abs(k['response']), reverse=True)[:250]:
        s = kp['scale']
        x = kp.get('x_refined', kp['x'])
        y = kp.get('y_refined', kp['y'])
        circle = plt.Circle((x, y), scale_sizes[s], color=scale_colors[s], fill=False, linewidth=1.5)
        ax.add_patch(circle)
    ax.set_title(f'After sub-pixel: {len(refined)} keypoints\n(|offset| < 0.5)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('Step 4.3: Sub-pixel Refinement', fontsize=14, fontweight='bold')
    save_img('surf_step4_3_after_subpixel.png')

def vis_step4_final(img, refined, filter_sizes):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    for kp in sorted(refined, key=lambda k: abs(k['response']), reverse=True)[:350]:
        s = kp['scale']
        x = kp.get('x_refined', kp['x'])
        y = kp.get('y_refined', kp['y'])
        circle = plt.Circle((x, y), scale_sizes[s], color=scale_colors[s], fill=False, linewidth=2)
        ax.add_patch(circle)
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                              markeredgecolor=scale_colors[i], markersize=scale_sizes[i], markeredgewidth=2,
                              label=f'Scale {i+1}') for i in range(4)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_title(f'Step 4.4: Final Refined Keypoints\n{len(refined)} stable keypoints', fontsize=14, fontweight='bold')
    ax.axis('off')
    save_img('surf_step4_4_final_keypoints.png')

# Step 5 Visualizations
def vis_step5_orientation(img, oriented, filter_sizes):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    for kp in sorted(oriented, key=lambda k: abs(k.get('response', 0)), reverse=True)[:250]:
        s = kp['scale']
        x, y = kp['x'], kp['y']
        ori = kp['orientation']
        size = scale_sizes[s]
        
        circle = plt.Circle((x, y), size, color=scale_colors[s], fill=False, linewidth=1.5)
        ax.add_patch(circle)
        ax.arrow(x, y, size*1.8*np.cos(ori), size*1.8*np.sin(ori),
                head_width=4, head_length=3, fc='yellow', ec='yellow', linewidth=2)
    
    ax.set_title(f'Step 5: Keypoints with Orientation\n{len(oriented)} keypoints (Haar wavelets + 60° window)', fontsize=14, fontweight='bold')
    ax.axis('off')
    save_img('surf_step5_orientation.png')

# Step 6 Visualizations
def vis_step6_descriptors(img, descriptors, filter_sizes):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    for desc in sorted(descriptors, key=lambda k: np.linalg.norm(k['descriptor']), reverse=True)[:200]:
        s = desc['scale']
        x, y = desc['x'], desc['y']
        ori = desc['orientation']
        size = scale_sizes[s]
        
        circle = plt.Circle((x, y), size, color='lime', fill=False, linewidth=1.5)
        ax.add_patch(circle)
        ax.arrow(x, y, size*1.5*np.cos(ori), size*1.5*np.sin(ori),
                head_width=3, head_length=2, fc='red', ec='red', linewidth=1.5)
    
    ax.set_title(f'Step 6: Final Descriptors\n{len(descriptors)} keypoints with 64-D descriptors', fontsize=14, fontweight='bold')
    ax.axis('off')
    save_img('surf_step6_descriptors.png')

def vis_step6_vectors(descriptors):
    n = min(6, len(descriptors))
    if n == 0:
        return
    
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5*n))
    if n == 1:
        axes = [axes]
    
    colors = plt.cm.tab20(np.repeat(np.arange(16), 4) / 16)
    
    for i, desc in enumerate(descriptors[:n]):
        axes[i].bar(range(64), desc['descriptor'], color=colors, width=0.8)
        axes[i].set_xlim(-1, 64)
        axes[i].set_ylim(-0.4, 0.4)
        axes[i].set_ylabel(f'KP {i+1}')
        if i == n - 1:
            axes[i].set_xlabel('Descriptor Index (64-D)')
        for j in range(1, 16):
            axes[i].axvline(x=j*4-0.5, color='gray', linewidth=0.3, alpha=0.5)
    
    fig.suptitle('Step 6: Sample 64-D Descriptor Vectors\n[Σdx, Σdy, Σ|dx|, Σ|dy|] × 16 subregions', fontsize=14, fontweight='bold')
    save_img('surf_step6_vectors.png')

# Complete Pipeline
def vis_complete_pipeline(img, integral, responses, keypoints, refined, oriented, descriptors, filter_sizes):
    fig = plt.figure(figsize=(20, 12))
    
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(img, cmap='gray')
    ax.set_title(f'1. Input\n{img.shape[1]}×{img.shape[0]}', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(2, 3, 2)
    ii_disp = np.log1p(integral)
    ii_disp = (ii_disp - ii_disp.min()) / (ii_disp.max() - ii_disp.min())
    ax.imshow(ii_disp, cmap='viridis')
    ax.set_title('2. Integral Image', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(img, cmap='gray')
    for kp in sorted(keypoints, key=lambda k: abs(k['response']), reverse=True)[:200]:
        s = kp['scale']
        circle = plt.Circle((kp['x'], kp['y']), scale_sizes[s]-2, color=scale_colors[s], fill=False, linewidth=1)
        ax.add_patch(circle)
    ax.set_title(f'3. Detection\n{len(keypoints)} keypoints', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(2, 3, 4)
    ax.imshow(img, cmap='gray')
    for kp in sorted(refined, key=lambda k: abs(k['response']), reverse=True)[:150]:
        s = kp['scale']
        circle = plt.Circle((kp['x'], kp['y']), scale_sizes[s], color=scale_colors[s], fill=False, linewidth=1.5)
        ax.add_patch(circle)
    ax.set_title(f'4. Filtered\n{len(refined)} keypoints', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(2, 3, 5)
    ax.imshow(img, cmap='gray')
    for kp in sorted(oriented, key=lambda k: abs(k.get('response', 0)), reverse=True)[:100]:
        s = kp['scale']
        x, y, ori = kp['x'], kp['y'], kp['orientation']
        circle = plt.Circle((x, y), scale_sizes[s], color='red', fill=False, linewidth=1)
        ax.add_patch(circle)
        ax.arrow(x, y, scale_sizes[s]*1.3*np.cos(ori), scale_sizes[s]*1.3*np.sin(ori),
                head_width=2, head_length=1.5, fc='yellow', ec='yellow')
    ax.set_title(f'5. Orientation\n{len(oriented)} keypoints', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(2, 3, 6)
    if len(descriptors) > 0:
        colors = plt.cm.tab20(np.repeat(np.arange(16), 4) / 16)
        ax.bar(range(64), descriptors[0]['descriptor'], color=colors, width=1)
        ax.set_xlim(-1, 64)
        ax.set_xlabel('Index')
    ax.set_title(f'6. 64-D Descriptor\n{len(descriptors)} total', fontsize=11, fontweight='bold')
    
    plt.suptitle('SURF Complete Pipeline', fontsize=16, fontweight='bold')
    save_img('surf_complete_pipeline.png')

# =============================================================================
# Main Pipeline
# =============================================================================

def run_surf_pipeline():
    print("=" * 70)
    print("SURF ALGORITHM PIPELINE")
    print("Generating real image visualizations at EVERY step")
    print("=" * 70)
    
    # Load image
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    if not os.path.exists(image_path):
        print("Creating test image...")
        np.random.seed(42)
        img = np.random.rand(480, 640) * 0.2 + 0.4
        for _ in range(40):
            cx, cy = np.random.randint(30, 610), np.random.randint(30, 450)
            r = np.random.randint(8, 40)
            y, x = np.ogrid[:480, :640]
            mask = (x - cx)**2 + (y - cy)**2 < r**2
            img[mask] = np.random.uniform(0.7, 1.0) if np.random.rand() > 0.5 else np.random.uniform(0.0, 0.3)
        Image.fromarray((img * 255).astype(np.uint8)).save(image_path)
    
    img = np.array(Image.open(image_path).convert('L')).astype(np.float64) / 255.0
    H, W = img.shape
    print(f"Image: {W}×{H}")
    
    filter_sizes = [9, 15, 21, 27]
    
    # Step 1
    print("\n[Step 1] Integral Image")
    integral = compute_integral_image(img)
    vis_step1_original(img)
    vis_step1_integral(img, integral)
    vis_step1_boxsum(img, integral)
    
    # Step 2
    print("\n[Step 2] Hessian Response")
    responses = build_hessian_pyramid(integral, filter_sizes)
    vis_step2_response_scale1(img, responses, filter_sizes)
    vis_step2_response_scale2(img, responses, filter_sizes)
    vis_step2_response_scale3(img, responses, filter_sizes)
    vis_step2_all_scales(img, responses, filter_sizes)
    
    # Step 3
    print("\n[Step 3] Keypoint Detection")
    keypoints = detect_keypoints(responses, filter_sizes, threshold=0.0005)
    print(f"    Found {len(keypoints)} keypoints")
    vis_step3_keypoints_scale1(img, keypoints, filter_sizes)
    vis_step3_keypoints_scale2(img, keypoints, filter_sizes)
    vis_step3_keypoints_scale3(img, keypoints, filter_sizes)
    vis_step3_keypoints_scale4(img, keypoints, filter_sizes)
    vis_step3_all_keypoints(img, keypoints, filter_sizes)
    
    # Step 4
    print("\n[Step 4] Filtering & Refinement")
    threshold_kps = [k for k in keypoints if abs(k['response']) > 0.002]
    refined = refine_keypoints(keypoints, responses, response_threshold=0.002)
    print(f"    After threshold: {len(threshold_kps)}")
    print(f"    After sub-pixel: {len(refined)}")
    vis_step4_before_filtering(img, keypoints, filter_sizes)
    vis_step4_after_threshold(img, keypoints, threshold_kps, filter_sizes)
    vis_step4_after_subpixel(img, threshold_kps, refined, filter_sizes)
    vis_step4_final(img, refined, filter_sizes)
    
    # Step 5
    print("\n[Step 5] Orientation Assignment")
    oriented = assign_orientations(refined, img, integral)
    print(f"    {len(oriented)} keypoints with orientation")
    vis_step5_orientation(img, oriented, filter_sizes)
    
    # Step 6
    print("\n[Step 6] Descriptor Extraction")
    descriptors = extract_descriptors(oriented, img)
    print(f"    {len(descriptors)} descriptors (64-D)")
    vis_step6_descriptors(img, descriptors, filter_sizes)
    vis_step6_vectors(descriptors)
    
    # Complete
    print("\n[Summary] Complete Pipeline")
    vis_complete_pipeline(img, integral, responses, keypoints, refined, oriented, descriptors, filter_sizes)
    
    print("\n" + "=" * 70)
    print("COMPLETE! Generated all real image visualizations.")
    print("=" * 70)

if __name__ == "__main__":
    run_surf_pipeline()

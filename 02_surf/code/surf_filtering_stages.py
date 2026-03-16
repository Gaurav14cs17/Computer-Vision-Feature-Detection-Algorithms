"""
Generate separate images for each SURF filtering stage on real image.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


def load_image():
    """Load the input image."""
    img_path = os.path.join(IMAGES_DIR, 'input_image.jpg')
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('L')
        if img.size[0] > 800 or img.size[1] > 600:
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
        return np.array(img).astype(np.float64) / 255.0
    else:
        img = np.zeros((480, 640), dtype=np.float64)
        img[100:200, 100:250] = 0.7
        img[250:350, 300:450] = 0.86
        Image.fromarray((img * 255).astype(np.uint8)).save(img_path)
        return img


def compute_integral_image(img):
    """Compute integral image."""
    return np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)


def box_sum(integral, x1, y1, x2, y2):
    """Compute sum of rectangular region using integral image."""
    h, w = integral.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    D = integral[y2, x2]
    B = integral[y1-1, x2] if y1 > 0 else 0
    C = integral[y2, x1-1] if x1 > 0 else 0
    A = integral[y1-1, x1-1] if y1 > 0 and x1 > 0 else 0
    return D - B - C + A


def compute_hessian_response(integral, x, y, filter_size):
    """Compute Hessian determinant using box filter approximations."""
    h, w = integral.shape
    half = filter_size // 2
    if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
        return 0, 0, 0, 0
    lobe_w = filter_size // 3
    
    # Dxx
    left = box_sum(integral, x - half, y - half, x - half + lobe_w - 1, y + half)
    center = box_sum(integral, x - lobe_w//2, y - half, x + lobe_w//2, y + half)
    right = box_sum(integral, x + half - lobe_w + 1, y - half, x + half, y + half)
    Dxx = left - 2 * center + right
    
    # Dyy
    top = box_sum(integral, x - half, y - half, x + half, y - half + lobe_w - 1)
    middle = box_sum(integral, x - half, y - lobe_w//2, x + half, y + lobe_w//2)
    bottom = box_sum(integral, x - half, y + half - lobe_w + 1, x + half, y + half)
    Dyy = top - 2 * middle + bottom
    
    # Dxy
    tl = box_sum(integral, x - half, y - half, x - 1, y - 1)
    tr = box_sum(integral, x + 1, y - half, x + half, y - 1)
    bl = box_sum(integral, x - half, y + 1, x - 1, y + half)
    br = box_sum(integral, x + 1, y + 1, x + half, y + half)
    Dxy = tl - tr - bl + br
    
    area = filter_size * filter_size
    Dxx /= area
    Dyy /= area
    Dxy /= area
    
    det_H = Dxx * Dyy - (0.9 * Dxy) ** 2
    return det_H, Dxx, Dyy, Dxy


def build_hessian_responses(integral, filter_sizes=[9, 15, 21, 27]):
    """Build Hessian response maps at multiple scales."""
    h, w = integral.shape
    responses = []
    
    for fs in filter_sizes:
        print(f"  Computing filter {fs}×{fs}...")
        response = np.zeros((h, w))
        margin = fs // 2 + 1
        
        for y in range(margin, h - margin):
            for x in range(margin, w - margin):
                det_H, _, _, _ = compute_hessian_response(integral, x, y, fs)
                response[y, x] = det_H
        responses.append(response)
    
    return responses


def detect_keypoints(responses, threshold=0.0005):
    """Detect scale-space extrema using 26-neighbor comparison."""
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
                
                neighbors = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        neighbors.append(prev_resp[y + dy, x + dx])
                        neighbors.append(next_resp[y + dy, x + dx])
                        if dy != 0 or dx != 0:
                            neighbors.append(curr_resp[y + dy, x + dx])
                
                if val > max(neighbors) or val < min(neighbors):
                    keypoints.append({
                        'x': x, 'y': y, 'scale': scale_idx,
                        'response': val, 'response_map': curr_resp
                    })
    
    return keypoints


def filter_response_threshold(keypoints, threshold=0.001):
    """Stage 1: Remove weak response keypoints."""
    kept = []
    removed = []
    
    for kp in keypoints:
        kp['passed_threshold'] = abs(kp['response']) >= threshold
        if kp['passed_threshold']:
            kept.append(kp)
        else:
            removed.append(kp)
    
    return kept, removed


def filter_subpixel(keypoints, responses, offset_threshold=0.5):
    """Stage 2: Remove unstable keypoints with large sub-pixel offset."""
    kept = []
    removed = []
    
    for kp in keypoints:
        resp = responses[kp['scale']]
        x, y = kp['x'], kp['y']
        h, w = resp.shape
        
        if x < 2 or x >= w - 2 or y < 2 or y >= h - 2:
            removed.append(kp)
            continue
        
        # Compute derivatives
        dx = (resp[y, x + 1] - resp[y, x - 1]) / 2
        dy = (resp[y + 1, x] - resp[y - 1, x]) / 2
        dxx = resp[y, x + 1] + resp[y, x - 1] - 2 * resp[y, x]
        dyy = resp[y + 1, x] + resp[y - 1, x] - 2 * resp[y, x]
        dxy = (resp[y + 1, x + 1] - resp[y + 1, x - 1] - 
               resp[y - 1, x + 1] + resp[y - 1, x - 1]) / 4
        
        det = dxx * dyy - dxy ** 2
        
        if abs(det) < 1e-10:
            removed.append(kp)
            continue
        
        offset_x = -(dyy * dx - dxy * dy) / det
        offset_y = -(dxx * dy - dxy * dx) / det
        
        kp['offset_x'] = offset_x
        kp['offset_y'] = offset_y
        
        if abs(offset_x) <= offset_threshold and abs(offset_y) <= offset_threshold:
            kp['x_refined'] = x + offset_x
            kp['y_refined'] = y + offset_y
            kept.append(kp)
        else:
            removed.append(kp)
    
    return kept, removed


def plot_keypoints_on_image(img, keypoints, title, filename, color='red', removed_kps=None):
    """Plot keypoints on image and save."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(img, cmap='gray')
    
    colors = ['red', 'lime', 'blue']
    sizes = [5, 8, 12]
    
    for kp in keypoints:
        scale = min(kp['scale'], len(colors) - 1)
        circle = plt.Circle((kp['x'], kp['y']), sizes[scale], 
                            color=colors[scale], fill=False, linewidth=1.5)
        ax.add_patch(circle)
    
    if removed_kps:
        for kp in removed_kps:
            circle = plt.Circle((kp['x'], kp['y']), 5, color='gray', 
                               fill=False, linewidth=1, linestyle='--', alpha=0.5)
            ax.add_patch(circle)
    
    ax.set_title(f'{title}\n({len(keypoints)} keypoints)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {filename}')


def main():
    print("=" * 60)
    print("SURF Filtering Stages Visualization")
    print("=" * 60)
    
    print("\nLoading image...")
    img = load_image()
    print(f"Image size: {img.shape}")
    
    print("\nComputing integral image...")
    integral = compute_integral_image(img * 255)
    
    print("\nBuilding Hessian responses...")
    filter_sizes = [9, 15, 21, 27]
    responses = build_hessian_responses(integral, filter_sizes)
    
    print("\nDetecting keypoints...")
    all_keypoints = detect_keypoints(responses)
    print(f"Total detected: {len(all_keypoints)} keypoints")
    
    # Stage 0: All detected keypoints
    print("\n=== Stage 0: All Detected Keypoints ===")
    plot_keypoints_on_image(
        img, all_keypoints,
        f'Stage 0: All Detected Keypoints ({len(all_keypoints)})',
        'surf_stage0_detected.png',
        color='blue'
    )
    
    # Stage 1: Response Threshold Removal
    print("\n=== Stage 1: Response Threshold Removal ===")
    after_threshold, removed_threshold = filter_response_threshold(all_keypoints)
    print(f"Kept: {len(after_threshold)}, Removed: {len(removed_threshold)}")
    plot_keypoints_on_image(
        img, after_threshold,
        f'Stage 1: After Response Threshold ({len(after_threshold)})',
        'surf_stage1_threshold.png',
        color='green',
        removed_kps=removed_threshold
    )
    
    # Stage 2: Sub-pixel Refinement
    print("\n=== Stage 2: Sub-pixel Refinement ===")
    after_subpixel, removed_subpixel = filter_subpixel(after_threshold, responses)
    print(f"Kept: {len(after_subpixel)}, Removed: {len(removed_subpixel)}")
    plot_keypoints_on_image(
        img, after_subpixel,
        f'Stage 2: After Sub-pixel Refinement ({len(after_subpixel)})',
        'surf_stage2_subpixel.png',
        color='red',
        removed_kps=removed_subpixel
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("FILTERING SUMMARY")
    print("=" * 60)
    print(f"Stage 0 (Detected):          {len(all_keypoints)} keypoints")
    print(f"Stage 1 (Response Threshold): {len(after_threshold)} keypoints (-{len(removed_threshold)})")
    print(f"Stage 2 (Sub-pixel):         {len(after_subpixel)} keypoints (-{len(removed_subpixel)})")
    print(f"\nRetention: {len(after_subpixel)}/{len(all_keypoints)} = {100*len(after_subpixel)/max(len(all_keypoints),1):.1f}%")


if __name__ == '__main__':
    main()

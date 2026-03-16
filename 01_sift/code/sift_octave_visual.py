"""
Generate SIFT keypoint images with:
- Different circle sizes for different octaves
- Different colors for different octaves
- Combined view with all octaves in different colors
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

def download_image():
    """Use existing input image."""
    img_path = '../images/input_image.jpg'
    if os.path.exists(img_path):
        return np.array(Image.open(img_path).convert('L'))
    raise FileNotFoundError("input_image.jpg not found")

def build_gaussian_pyramid(img, num_octaves=3, num_scales=5, sigma=1.6, k=np.sqrt(2)):
    """Build Gaussian scale-space pyramid."""
    pyramid = []
    current_img = img.astype(np.float64)
    
    for octave in range(num_octaves):
        octave_images = []
        for scale in range(num_scales):
            sigma_scale = sigma * (k ** scale)
            blurred = gaussian_filter(current_img, sigma=sigma_scale)
            octave_images.append(blurred)
        pyramid.append(octave_images)
        current_img = current_img[::2, ::2]
    
    return pyramid

def compute_dog(pyramid):
    """Compute Difference of Gaussians."""
    dog_pyramid = []
    for octave_images in pyramid:
        dog_octave = []
        for i in range(len(octave_images) - 1):
            dog = octave_images[i + 1] - octave_images[i]
            dog_octave.append(dog)
        dog_pyramid.append(dog_octave)
    return dog_pyramid

def detect_extrema(dog_octave, threshold=0.03):
    """Detect local extrema in DoG images."""
    keypoints = []
    for s in range(1, len(dog_octave) - 1):
        current = dog_octave[s]
        below = dog_octave[s - 1]
        above = dog_octave[s + 1]
        
        h, w = current.shape
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                patch = current[y-1:y+2, x-1:x+2]
                patch_below = below[y-1:y+2, x-1:x+2]
                patch_above = above[y-1:y+2, x-1:x+2]
                
                center = current[y, x]
                all_neighbors = np.concatenate([
                    patch.flatten(),
                    patch_below.flatten(),
                    patch_above.flatten()
                ])
                all_neighbors = np.delete(all_neighbors, 4)
                
                if abs(center) > threshold:
                    if center > np.max(all_neighbors) or center < np.min(all_neighbors):
                        keypoints.append({'x': x, 'y': y, 'scale': s, 'response': center})
    return keypoints

def compute_derivatives(dog, x, y):
    """Compute gradient and Hessian at keypoint location."""
    h, w = dog.shape
    if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
        return None, None, None, None, None
    
    Dx = (dog[y, x + 1] - dog[y, x - 1]) / 2
    Dy = (dog[y + 1, x] - dog[y - 1, x]) / 2
    Dxx = dog[y, x + 1] + dog[y, x - 1] - 2 * dog[y, x]
    Dyy = dog[y + 1, x] + dog[y - 1, x] - 2 * dog[y, x]
    Dxy = (dog[y + 1, x + 1] - dog[y + 1, x - 1] - 
           dog[y - 1, x + 1] + dog[y - 1, x - 1]) / 4
    
    return Dx, Dy, Dxx, Dyy, Dxy

def filter_keypoints(keypoints, dog_octave):
    """Apply all filtering stages."""
    # Stage 1: Low Contrast
    kept = []
    for kp in keypoints:
        s = kp['scale']
        if s < 0 or s >= len(dog_octave):
            continue
        dog = dog_octave[s]
        x, y = kp['x'], kp['y']
        
        derivs = compute_derivatives(dog, x, y)
        if derivs[0] is None:
            continue
        
        Dx, Dy, Dxx, Dyy, Dxy = derivs
        det = Dxx * Dyy - Dxy * Dxy
        
        if abs(det) < 1e-10:
            continue
        
        offset_x = -(Dyy * Dx - Dxy * Dy) / det
        offset_y = -(Dxx * Dy - Dxy * Dx) / det
        
        # Stage 2: Edge Response
        trace = Dxx + Dyy
        if det <= 0:
            continue
        ratio = (trace ** 2) / det
        if ratio >= 12.1:
            continue
        
        # Stage 3: Sub-pixel
        if abs(offset_x) > 0.5 or abs(offset_y) > 0.5:
            continue
        
        kp['x_refined'] = x + offset_x
        kp['y_refined'] = y + offset_y
        kept.append(kp)
    
    return kept

# Colors for each octave
OCTAVE_COLORS = {
    0: '#FF0000',  # Red - Fine scale
    1: '#00FF00',  # Green - Medium scale
    2: '#0000FF',  # Blue - Coarse scale
}

# Circle radius for each octave
OCTAVE_RADIUS = {
    0: 4,   # Small - Fine scale
    1: 8,   # Medium - Medium scale
    2: 14,  # Large - Coarse scale
}

def plot_octave_keypoints(img, keypoints_by_octave, title, filename, show_legend=True):
    """Plot keypoints with different colors and sizes per octave."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    total = 0
    for octave, kps in keypoints_by_octave.items():
        scale_factor = 2 ** octave
        color = OCTAVE_COLORS[octave]
        radius = OCTAVE_RADIUS[octave]
        
        for kp in kps:
            x = kp['x'] * scale_factor
            y = kp['y'] * scale_factor
            circle = plt.Circle((x, y), radius, color=color, fill=False, linewidth=1.5)
            ax.add_patch(circle)
        total += len(kps)
    
    if show_legend:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                      markeredgecolor=OCTAVE_COLORS[0], markersize=8, markeredgewidth=2,
                      label=f'Octave 0: Fine-scale ({len(keypoints_by_octave.get(0, []))} kps)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                      markeredgecolor=OCTAVE_COLORS[1], markersize=12, markeredgewidth=2,
                      label=f'Octave 1: Medium-scale ({len(keypoints_by_octave.get(1, []))} kps)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                      markeredgecolor=OCTAVE_COLORS[2], markersize=16, markeredgewidth=2,
                      label=f'Octave 2: Coarse-scale ({len(keypoints_by_octave.get(2, []))} kps)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                 fancybox=True, framealpha=0.9)
    
    ax.set_title(f'{title}\n(Total: {total} keypoints)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {filename}')

def plot_single_octave(img, keypoints, octave, title, filename):
    """Plot keypoints from a single octave."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    scale_factor = 2 ** octave
    color = OCTAVE_COLORS[octave]
    radius = OCTAVE_RADIUS[octave]
    
    for kp in keypoints:
        x = kp['x'] * scale_factor
        y = kp['y'] * scale_factor
        circle = plt.Circle((x, y), radius, color=color, fill=False, linewidth=1.5)
        ax.add_patch(circle)
    
    ax.set_title(f'{title}\n({len(keypoints)} keypoints)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {filename}')

def plot_filtering_stages_by_octave(img, all_kps, filtered_kps, title, filename):
    """Plot filtered keypoints with octave colors and sizes."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    total = 0
    octave_counts = {0: 0, 1: 0, 2: 0}
    
    for kp in filtered_kps:
        octave = kp['octave']
        scale_factor = 2 ** octave
        color = OCTAVE_COLORS[octave]
        radius = OCTAVE_RADIUS[octave]
        
        x = kp['x'] * scale_factor
        y = kp['y'] * scale_factor
        circle = plt.Circle((x, y), radius, color=color, fill=False, linewidth=1.5)
        ax.add_patch(circle)
        total += 1
        octave_counts[octave] += 1
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                  markeredgecolor=OCTAVE_COLORS[0], markersize=8, markeredgewidth=2,
                  label=f'Octave 0: Small circles ({octave_counts[0]} kps)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                  markeredgecolor=OCTAVE_COLORS[1], markersize=12, markeredgewidth=2,
                  label=f'Octave 1: Medium circles ({octave_counts[1]} kps)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                  markeredgecolor=OCTAVE_COLORS[2], markersize=16, markeredgewidth=2,
                  label=f'Octave 2: Large circles ({octave_counts[2]} kps)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
             fancybox=True, framealpha=0.9)
    
    ax.set_title(f'{title}\n(Total: {total} keypoints)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {filename}')

def main():
    print("Loading image...")
    img = download_image()
    print(f"Image size: {img.shape}")
    
    print("\nBuilding Gaussian pyramid...")
    pyramid = build_gaussian_pyramid(img)
    
    print("Computing DoG...")
    dog_pyramid = compute_dog(pyramid)
    
    # Detect keypoints per octave
    print("\nDetecting keypoints per octave...")
    keypoints_by_octave = {}
    all_keypoints = []
    
    for octave_idx, dog_octave in enumerate(dog_pyramid):
        kps = detect_extrema(dog_octave)
        for kp in kps:
            kp['octave'] = octave_idx
            kp['dog'] = dog_octave[kp['scale']]
        keypoints_by_octave[octave_idx] = kps
        all_keypoints.extend(kps)
        print(f"  Octave {octave_idx}: {len(kps)} keypoints")
    
    print(f"Total detected: {len(all_keypoints)} keypoints")
    
    # 1. Plot each octave separately with its color
    print("\n=== Generating Octave-specific images ===")
    for octave in range(3):
        plot_single_octave(
            img, keypoints_by_octave[octave], octave,
            f'Octave {octave}: {"Fine" if octave==0 else "Medium" if octave==1 else "Coarse"}-scale Features',
            f'../images/sift_octave{octave}_keypoints.png'
        )
    
    # 2. Plot combined with different colors
    print("\n=== Generating Combined image (different colors) ===")
    plot_octave_keypoints(
        img, keypoints_by_octave,
        'All Detected Keypoints (Combined from All Octaves)',
        '../images/sift_all_octaves_combined.png'
    )
    
    # 3. Apply filtering and show with octave colors/sizes
    print("\n=== Applying filtering stages ===")
    
    # Filter all keypoints
    filtered_all = []
    for octave_idx, dog_octave in enumerate(dog_pyramid):
        kps = keypoints_by_octave[octave_idx]
        filtered = filter_keypoints(kps, dog_octave)
        for kp in filtered:
            kp['octave'] = octave_idx
        filtered_all.extend(filtered)
    
    print(f"After filtering: {len(filtered_all)} keypoints")
    
    # Count per octave after filtering
    filtered_by_octave = {0: [], 1: [], 2: []}
    for kp in filtered_all:
        filtered_by_octave[kp['octave']].append(kp)
    
    for octave in range(3):
        print(f"  Octave {octave}: {len(filtered_by_octave[octave])} keypoints")
    
    # 4. Plot filtered keypoints with octave colors
    print("\n=== Generating Filtered image (with octave colors) ===")
    plot_filtering_stages_by_octave(
        img, all_keypoints, filtered_all,
        'Final Keypoints After All Filtering Stages',
        '../images/sift_filtered_octaves.png'
    )
    
    # 5. Update the stage images with octave information
    print("\n=== Updating stage images with octave colors ===")
    
    # Stage 0: All detected
    plot_octave_keypoints(
        img, keypoints_by_octave,
        'Stage 0: All Detected Keypoints',
        '../images/sift_stage0_detected.png'
    )
    
    # For filtering stages, we need to track per stage
    # Stage 1: After low contrast (no change in our case)
    plot_octave_keypoints(
        img, keypoints_by_octave,
        'Stage 1: After Low Contrast Removal',
        '../images/sift_stage1_low_contrast.png'
    )
    
    # Stage 2: After edge response
    stage2_by_octave = {0: [], 1: [], 2: []}
    for octave_idx, dog_octave in enumerate(dog_pyramid):
        kps = keypoints_by_octave[octave_idx]
        for kp in kps:
            s = kp['scale']
            if s < 0 or s >= len(dog_octave):
                continue
            dog = dog_octave[s]
            x, y = kp['x'], kp['y']
            derivs = compute_derivatives(dog, x, y)
            if derivs[0] is None:
                continue
            Dx, Dy, Dxx, Dyy, Dxy = derivs
            det = Dxx * Dyy - Dxy * Dxy
            if det <= 0:
                continue
            trace = Dxx + Dyy
            ratio = (trace ** 2) / det
            if ratio < 12.1:
                kp['octave'] = octave_idx
                stage2_by_octave[octave_idx].append(kp)
    
    plot_octave_keypoints(
        img, stage2_by_octave,
        'Stage 2: After Edge Response Removal',
        '../images/sift_stage2_edge_response.png'
    )
    
    # Stage 3: Final filtered
    plot_octave_keypoints(
        img, filtered_by_octave,
        'Stage 3: After Sub-pixel Refinement (Final)',
        '../images/sift_stage3_subpixel.png'
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Octave 0 (Fine, Red, Small circles):   {len(keypoints_by_octave[0])} → {len(filtered_by_octave[0])}")
    print(f"Octave 1 (Medium, Green, Medium circles): {len(keypoints_by_octave[1])} → {len(filtered_by_octave[1])}")
    print(f"Octave 2 (Coarse, Blue, Large circles): {len(keypoints_by_octave[2])} → {len(filtered_by_octave[2])}")
    print(f"Total: {len(all_keypoints)} → {len(filtered_all)}")

if __name__ == '__main__':
    main()

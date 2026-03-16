"""
Generate separate images for each SIFT filtering stage on real image.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

def download_image():
    """Use existing input image or download one."""
    img_path = '../images/input_image.jpg'
    if os.path.exists(img_path):
        return np.array(Image.open(img_path).convert('L'))
    
    # Download sample image
    import urllib.request
    url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png'
    urllib.request.urlretrieve(url, img_path)
    return np.array(Image.open(img_path).convert('L'))

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
        # Downsample for next octave
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
                all_neighbors = np.delete(all_neighbors, 4)  # Remove center
                
                if abs(center) > threshold:
                    if center > np.max(all_neighbors) or center < np.min(all_neighbors):
                        keypoints.append({
                            'x': x, 'y': y, 'scale': s,
                            'response': center,
                            'dog': current
                        })
    return keypoints

def compute_derivatives(dog, x, y):
    """Compute gradient and Hessian at keypoint location."""
    h, w = dog.shape
    if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
        return None, None, None, None, None
    
    # Gradient
    Dx = (dog[y, x + 1] - dog[y, x - 1]) / 2
    Dy = (dog[y + 1, x] - dog[y - 1, x]) / 2
    
    # Hessian
    Dxx = dog[y, x + 1] + dog[y, x - 1] - 2 * dog[y, x]
    Dyy = dog[y + 1, x] + dog[y - 1, x] - 2 * dog[y, x]
    Dxy = (dog[y + 1, x + 1] - dog[y + 1, x - 1] - 
           dog[y - 1, x + 1] + dog[y - 1, x - 1]) / 4
    
    return Dx, Dy, Dxx, Dyy, Dxy

def filter_low_contrast(keypoints, threshold=0.03):
    """Stage 1: Remove low contrast keypoints."""
    kept = []
    removed = []
    
    for kp in keypoints:
        dog = kp['dog']
        x, y = kp['x'], kp['y']
        
        derivs = compute_derivatives(dog, x, y)
        if derivs[0] is None:
            removed.append(kp)
            continue
        
        Dx, Dy, Dxx, Dyy, Dxy = derivs
        
        # Compute contrast
        det = Dxx * Dyy - Dxy * Dxy
        if abs(det) < 1e-10:
            contrast = abs(dog[y, x])
        else:
            offset_x = -(Dyy * Dx - Dxy * Dy) / det
            offset_y = -(Dxx * Dy - Dxy * Dx) / det
            contrast = abs(dog[y, x] + 0.5 * (Dx * offset_x + Dy * offset_y))
        
        kp['contrast'] = contrast
        kp['Dxx'] = Dxx
        kp['Dyy'] = Dyy
        kp['Dxy'] = Dxy
        kp['Dx'] = Dx
        kp['Dy'] = Dy
        
        if contrast >= threshold:
            kept.append(kp)
        else:
            removed.append(kp)
    
    return kept, removed

def filter_edge_response(keypoints, r=10):
    """Stage 2: Remove edge responses."""
    threshold = ((r + 1) ** 2) / r
    kept = []
    removed = []
    
    for kp in keypoints:
        Dxx = kp.get('Dxx', 0)
        Dyy = kp.get('Dyy', 0)
        Dxy = kp.get('Dxy', 0)
        
        trace = Dxx + Dyy
        det = Dxx * Dyy - Dxy * Dxy
        
        if det <= 0:
            removed.append(kp)
            continue
        
        ratio = (trace ** 2) / det
        kp['edge_ratio'] = ratio
        
        if ratio < threshold:
            kept.append(kp)
        else:
            removed.append(kp)
    
    return kept, removed

def filter_subpixel(keypoints, offset_threshold=0.5):
    """Stage 3: Remove unstable keypoints with large sub-pixel offset."""
    kept = []
    removed = []
    
    for kp in keypoints:
        Dx = kp.get('Dx', 0)
        Dy = kp.get('Dy', 0)
        Dxx = kp.get('Dxx', 0)
        Dyy = kp.get('Dyy', 0)
        Dxy = kp.get('Dxy', 0)
        
        det = Dxx * Dyy - Dxy * Dxy
        
        if abs(det) < 1e-10:
            removed.append(kp)
            continue
        
        offset_x = -(Dyy * Dx - Dxy * Dy) / det
        offset_y = -(Dxx * Dy - Dxy * Dx) / det
        
        kp['offset_x'] = offset_x
        kp['offset_y'] = offset_y
        
        if abs(offset_x) <= offset_threshold and abs(offset_y) <= offset_threshold:
            kp['x_refined'] = kp['x'] + offset_x
            kp['y_refined'] = kp['y'] + offset_y
            kept.append(kp)
        else:
            removed.append(kp)
    
    return kept, removed

def plot_keypoints_on_image(img, keypoints, title, filename, scale_factor=1, color='red', removed_kps=None):
    """Plot keypoints on image and save."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(img, cmap='gray')
    
    # Plot kept keypoints
    for kp in keypoints:
        x = kp['x'] * scale_factor
        y = kp['y'] * scale_factor
        circle = plt.Circle((x, y), 5, color=color, fill=False, linewidth=1.5)
        ax.add_patch(circle)
    
    # Plot removed keypoints if provided
    if removed_kps:
        for kp in removed_kps:
            x = kp['x'] * scale_factor
            y = kp['y'] * scale_factor
            circle = plt.Circle((x, y), 5, color='gray', fill=False, linewidth=1, linestyle='--', alpha=0.5)
            ax.add_patch(circle)
    
    ax.set_title(f'{title}\n({len(keypoints)} keypoints)', fontsize=14, fontweight='bold')
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
    
    print("Detecting keypoints in all octaves...")
    all_keypoints = []
    for octave_idx, dog_octave in enumerate(dog_pyramid):
        scale_factor = 2 ** octave_idx
        kps = detect_extrema(dog_octave)
        for kp in kps:
            kp['octave'] = octave_idx
            kp['scale_factor'] = scale_factor
        all_keypoints.extend(kps)
    
    print(f"Total detected: {len(all_keypoints)} keypoints")
    
    # Stage 0: All detected keypoints
    print("\n=== Stage 0: All Detected Keypoints ===")
    plot_keypoints_on_image(
        img, all_keypoints, 
        f'Stage 0: All Detected Keypoints ({len(all_keypoints)})',
        '../images/sift_stage0_detected.png',
        color='blue'
    )
    
    # Stage 1: Low Contrast Removal
    print("\n=== Stage 1: Low Contrast Removal ===")
    after_low_contrast, removed_low_contrast = filter_low_contrast(all_keypoints)
    print(f"Kept: {len(after_low_contrast)}, Removed: {len(removed_low_contrast)}")
    plot_keypoints_on_image(
        img, after_low_contrast,
        f'Stage 1: After Low Contrast Removal ({len(after_low_contrast)})',
        '../images/sift_stage1_low_contrast.png',
        color='green',
        removed_kps=removed_low_contrast
    )
    
    # Stage 2: Edge Response Removal
    print("\n=== Stage 2: Edge Response Removal ===")
    after_edge, removed_edge = filter_edge_response(after_low_contrast)
    print(f"Kept: {len(after_edge)}, Removed: {len(removed_edge)}")
    plot_keypoints_on_image(
        img, after_edge,
        f'Stage 2: After Edge Response Removal ({len(after_edge)})',
        '../images/sift_stage2_edge_response.png',
        color='orange',
        removed_kps=removed_edge
    )
    
    # Stage 3: Sub-pixel Refinement
    print("\n=== Stage 3: Sub-pixel Refinement ===")
    after_subpixel, removed_subpixel = filter_subpixel(after_edge)
    print(f"Kept: {len(after_subpixel)}, Removed: {len(removed_subpixel)}")
    plot_keypoints_on_image(
        img, after_subpixel,
        f'Stage 3: After Sub-pixel Refinement ({len(after_subpixel)})',
        '../images/sift_stage3_subpixel.png',
        color='red',
        removed_kps=removed_subpixel
    )
    
    # Summary
    print("\n" + "="*50)
    print("FILTERING SUMMARY")
    print("="*50)
    print(f"Stage 0 (Detected):        {len(all_keypoints)} keypoints")
    print(f"Stage 1 (Low Contrast):    {len(after_low_contrast)} keypoints (-{len(removed_low_contrast)})")
    print(f"Stage 2 (Edge Response):   {len(after_edge)} keypoints (-{len(removed_edge)})")
    print(f"Stage 3 (Sub-pixel):       {len(after_subpixel)} keypoints (-{len(removed_subpixel)})")
    print(f"\nRetention: {len(after_subpixel)}/{len(all_keypoints)} = {100*len(after_subpixel)/len(all_keypoints):.1f}%")

if __name__ == '__main__':
    main()

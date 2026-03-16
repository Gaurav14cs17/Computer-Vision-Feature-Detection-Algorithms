"""
Generate final SIFT pipeline summary image showing detection + description results.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import os

def load_image():
    """Load the input image."""
    img_path = '../images/input_image.jpg'
    if os.path.exists(img_path):
        return np.array(Image.open(img_path).convert('L'))
    raise FileNotFoundError("input_image.jpg not found")

def compute_gradients(img):
    """Compute gradient magnitude and orientation."""
    gy, gx = np.gradient(img.astype(np.float64))
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi
    orientation = (orientation + 360) % 360
    return magnitude, orientation

def build_gaussian_pyramid(img, n_octaves=3, n_scales=4, sigma=1.6, k=np.sqrt(2)):
    """Build Gaussian scale-space pyramid."""
    pyramid = []
    current = img.astype(np.float64)
    
    for octave in range(n_octaves):
        octave_imgs = []
        for scale in range(n_scales):
            sig = sigma * (k ** scale)
            blurred = gaussian_filter(current, sigma=sig)
            octave_imgs.append(blurred)
        pyramid.append(octave_imgs)
        current = current[::2, ::2]
    
    return pyramid

def compute_dog(pyramid):
    """Compute Difference of Gaussians."""
    dog_pyramid = []
    for octave_imgs in pyramid:
        dog_octave = []
        for i in range(len(octave_imgs) - 1):
            dog = octave_imgs[i + 1] - octave_imgs[i]
            dog_octave.append(dog)
        dog_pyramid.append(dog_octave)
    return dog_pyramid

def detect_extrema(dog_pyramid):
    """Detect scale-space extrema."""
    keypoints = []
    
    for octave_idx, dog_octave in enumerate(dog_pyramid):
        scale_factor = 2 ** octave_idx
        
        for scale_idx in range(1, len(dog_octave) - 1):
            prev_dog = dog_octave[scale_idx - 1]
            curr_dog = dog_octave[scale_idx]
            next_dog = dog_octave[scale_idx + 1]
            
            h, w = curr_dog.shape
            
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    val = curr_dog[y, x]
                    
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            neighbors.append(prev_dog[y + dy, x + dx])
                            if dy != 0 or dx != 0:
                                neighbors.append(curr_dog[y + dy, x + dx])
                            neighbors.append(next_dog[y + dy, x + dx])
                    
                    if val > max(neighbors) or val < min(neighbors):
                        orig_x = x * scale_factor
                        orig_y = y * scale_factor
                        keypoints.append({
                            'x': orig_x,
                            'y': orig_y,
                            'octave': octave_idx,
                            'scale': scale_idx,
                            'response': abs(val)
                        })
    
    return keypoints

def filter_keypoints(keypoints, dog_pyramid, threshold=0.03, edge_threshold=10):
    """Filter keypoints."""
    filtered = []
    r = edge_threshold
    edge_limit = ((r + 1) ** 2) / r
    
    for kp in keypoints:
        octave = kp['octave']
        scale = kp['scale']
        scale_factor = 2 ** octave
        local_x = kp['x'] // scale_factor
        local_y = kp['y'] // scale_factor
        
        dog = dog_pyramid[octave][scale]
        h, w = dog.shape
        
        if not (1 <= local_x < w - 1 and 1 <= local_y < h - 1):
            continue
        
        # Edge response
        dxx = dog[local_y, local_x + 1] + dog[local_y, local_x - 1] - 2 * dog[local_y, local_x]
        dyy = dog[local_y + 1, local_x] + dog[local_y - 1, local_x] - 2 * dog[local_y, local_x]
        dxy = (dog[local_y + 1, local_x + 1] - dog[local_y + 1, local_x - 1] -
               dog[local_y - 1, local_x + 1] + dog[local_y - 1, local_x - 1]) / 4
        
        tr = dxx + dyy
        det = dxx * dyy - dxy ** 2
        
        if det <= 0:
            continue
        
        edge_ratio = (tr ** 2) / det
        if edge_ratio > edge_limit:
            continue
        
        filtered.append(kp)
    
    return filtered

def compute_orientation(magnitude, orientation, kp_x, kp_y, radius=8):
    """Compute dominant orientation."""
    hist = np.zeros(36)
    h, w = magnitude.shape
    
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y, x = kp_y + dy, kp_x + dx
            if 0 <= y < h and 0 <= x < w:
                mag = magnitude[y, x]
                ori = orientation[y, x]
                weight = np.exp(-(dx**2 + dy**2) / (2 * (radius/2)**2))
                bin_idx = int(ori / 10) % 36
                hist[bin_idx] += mag * weight
    
    dominant_bin = np.argmax(hist)
    return dominant_bin * 10 + 5

def plot_final_summary(img, keypoints, magnitude, orientation):
    """Create final summary image."""
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.2)
    
    # 1. Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img, cmap='gray')
    ax1.set_title('1. Input Image\n640 × 480 pixels', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Detection: All keypoints with octave colors
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img, cmap='gray')
    
    colors = ['red', 'lime', 'blue']
    sizes = [3, 6, 12]
    octave_counts = [0, 0, 0]
    
    for kp in keypoints:
        octave = kp['octave']
        octave_counts[octave] += 1
        circle = plt.Circle((kp['x'], kp['y']), sizes[octave], 
                           color=colors[octave], fill=False, linewidth=1)
        ax2.add_patch(circle)
    
    ax2.set_title(f'2. Detection: {len(keypoints)} Keypoints\nRed={octave_counts[0]} Green={octave_counts[1]} Blue={octave_counts[2]}', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Final keypoints with orientation arrows
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img, cmap='gray')
    
    # Sample keypoints for arrows (too many to show all)
    sample_kps = keypoints[::max(1, len(keypoints)//50)]
    
    for kp in sample_kps:
        x, y = kp['x'], kp['y']
        octave = kp['octave']
        
        if 0 <= y < magnitude.shape[0] and 0 <= x < magnitude.shape[1]:
            ori = compute_orientation(magnitude, orientation, int(x), int(y))
            
            arrow_len = sizes[octave] * 2
            dx = arrow_len * np.cos(np.radians(ori))
            dy = arrow_len * np.sin(np.radians(ori))
            
            circle = plt.Circle((x, y), sizes[octave], color=colors[octave], fill=False, linewidth=1.5)
            ax3.add_patch(circle)
            ax3.arrow(x, y, dx, dy, head_width=3, head_length=2, 
                     fc='yellow', ec='yellow', linewidth=1, alpha=0.8)
    
    ax3.set_title('3. Orientation Assignment\n(Arrows show gradient direction)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. Sample descriptor visualization
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Find a good keypoint
    sample_kp = None
    for kp in keypoints:
        x, y = kp['x'], kp['y']
        if 50 < x < img.shape[1] - 50 and 50 < y < img.shape[0] - 50:
            sample_kp = kp
            break
    
    if sample_kp:
        x, y = int(sample_kp['x']), int(sample_kp['y'])
        half = 8
        region = img[max(0,y-half):y+half, max(0,x-half):x+half]
        
        if region.shape[0] >= 16 and region.shape[1] >= 16:
            ax4.imshow(region[:16, :16], cmap='gray', extent=[0, 16, 16, 0])
            
            # Draw 4x4 grid
            for i in range(5):
                ax4.axhline(y=i*4, color='lime', linewidth=2)
                ax4.axvline(x=i*4, color='lime', linewidth=2)
            
            # Draw gradient arrows
            region_mag = magnitude[max(0,y-half):y+half, max(0,x-half):x+half]
            region_ori = orientation[max(0,y-half):y+half, max(0,x-half):x+half]
            
            if region_mag.shape[0] >= 16 and region_mag.shape[1] >= 16:
                max_mag = region_mag[:16, :16].max() + 1e-6
                for py in range(0, 16, 2):
                    for px in range(0, 16, 2):
                        mag = region_mag[py, px]
                        ori = region_ori[py, px]
                        arrow_len = 0.8 * (mag / max_mag)
                        adx = arrow_len * np.cos(np.radians(ori))
                        ady = arrow_len * np.sin(np.radians(ori))
                        ax4.arrow(px + 1, py + 1, adx, ady, head_width=0.3, head_length=0.15,
                                 fc='yellow', ec='yellow', linewidth=0.6)
    
    ax4.set_title('4. 16×16 Region with Gradients\n(4×4 = 16 subregions)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Pixels')
    ax4.set_ylabel('Pixels')
    
    # 5. 8-bin histogram example
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Generate sample histogram
    if sample_kp:
        x, y = int(sample_kp['x']), int(sample_kp['y'])
        half = 8
        region_mag = magnitude[max(0,y-half):y+half, max(0,x-half):x+half]
        region_ori = orientation[max(0,y-half):y+half, max(0,x-half):x+half]
        
        if region_mag.shape[0] >= 4 and region_mag.shape[1] >= 4:
            hist = np.zeros(8)
            for py in range(4):
                for px in range(4):
                    mag = region_mag[py, px]
                    ori = region_ori[py, px]
                    bin_idx = int(ori / 45) % 8
                    hist[bin_idx] += mag
            
            colors_hist = plt.cm.rainbow(np.linspace(0, 1, 8))
            ax5.bar(range(8), hist, color=colors_hist, edgecolor='black', linewidth=1)
            ax5.set_xticks(range(8))
            ax5.set_xticklabels([f'{i*45}°' for i in range(8)], fontsize=9)
    
    ax5.set_title('5. 8-bin Histogram\n(One per subregion)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Orientation Bin (45° each)')
    ax5.set_ylabel('Magnitude Sum')
    
    # 6. 128-D descriptor
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Generate full descriptor
    if sample_kp:
        x, y = int(sample_kp['x']), int(sample_kp['y'])
        half = 8
        region_mag = magnitude[max(0,y-half):y+half, max(0,x-half):x+half]
        region_ori = orientation[max(0,y-half):y+half, max(0,x-half):x+half]
        
        if region_mag.shape[0] >= 16 and region_mag.shape[1] >= 16:
            all_hist = []
            for row in range(4):
                for col in range(4):
                    hist = np.zeros(8)
                    for dy in range(4):
                        for dx in range(4):
                            py, px = row * 4 + dy, col * 4 + dx
                            if py < region_mag.shape[0] and px < region_mag.shape[1]:
                                mag = region_mag[py, px]
                                ori = region_ori[py, px]
                                bin_idx = int(ori / 45) % 8
                                hist[bin_idx] += mag
                    all_hist.extend(hist)
            
            desc = np.array(all_hist)
            desc = desc / (np.linalg.norm(desc) + 1e-6)
            desc = np.clip(desc, 0, 0.2)
            desc = desc / (np.linalg.norm(desc) + 1e-6)
            
            bar_colors = plt.cm.tab20(np.repeat(np.arange(16), 8) / 16)
            ax6.bar(range(128), desc, color=bar_colors, edgecolor='none', width=1)
            
            for i in range(1, 16):
                ax6.axvline(x=i*8 - 0.5, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    
    ax6.set_title('6. Final 128-D Descriptor\n(16 subregions × 8 bins)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Index (0-127)')
    ax6.set_ylabel('Normalized Value')
    ax6.set_xlim(-1, 128)
    
    plt.suptitle('SIFT Complete Pipeline: Detection → Description', fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig('../images/sift_complete_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_complete_summary.png')

def main():
    print("Loading image...")
    img = load_image()
    print(f"Image size: {img.shape}")
    
    print("Building Gaussian pyramid...")
    pyramid = build_gaussian_pyramid(img)
    
    print("Computing DoG...")
    dog_pyramid = compute_dog(pyramid)
    
    print("Detecting keypoints...")
    keypoints = detect_extrema(dog_pyramid)
    print(f"Detected: {len(keypoints)} keypoints")
    
    print("Filtering keypoints...")
    keypoints = filter_keypoints(keypoints, dog_pyramid)
    print(f"After filtering: {len(keypoints)} keypoints")
    
    print("Computing gradients...")
    magnitude, orientation = compute_gradients(img)
    
    print("\nGenerating final summary image...")
    plot_final_summary(img, keypoints, magnitude, orientation)
    
    print("\nDone!")

if __name__ == '__main__':
    main()

"""
Generate Scale Factor visualization using REAL image and keypoints.
Shows octave resolution hierarchy and coordinate transformation.
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

def detect_extrema_per_octave(dog_pyramid):
    """Detect scale-space extrema, return keypoints per octave."""
    keypoints_per_octave = [[], [], []]
    
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
                        # Store local coordinates and original coordinates
                        keypoints_per_octave[octave_idx].append({
                            'local_x': x,
                            'local_y': y,
                            'orig_x': x * scale_factor,
                            'orig_y': y * scale_factor,
                            'octave': octave_idx,
                            'response': abs(val)
                        })
    
    return keypoints_per_octave

def plot_scale_factor_visualization(img, pyramid, keypoints_per_octave):
    """Create scale factor visualization with real image."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # Main title
    fig.suptitle('SIFT Scale Factor: Octave Resolution & Coordinate Transformation', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Create grid layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25,
                          left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    # Colors for octaves
    colors = ['#E74C3C', '#27AE60', '#3498DB']  # Red, Green, Blue
    octave_names = ['OCTAVE 0\n(Full Resolution)', 'OCTAVE 1\n(Half Resolution)', 'OCTAVE 2\n(Quarter Resolution)']
    
    # ============================================
    # TOP LEFT: Octave Resolution Hierarchy
    # ============================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Octave Resolution Hierarchy', fontsize=14, fontweight='bold', pad=10)
    
    h, w = img.shape
    
    # Octave 0 - Full
    rect0 = patches.FancyBboxPatch((5, 55), 40, 40, boxstyle="round,pad=0.02",
                                    facecolor=colors[0], edgecolor='black', linewidth=2, alpha=0.8)
    ax1.add_patch(rect0)
    ax1.text(25, 75, f'OCTAVE 0\n{w}×{h}\n(Full)', ha='center', va='center', 
             fontsize=11, fontweight='bold', color='white')
    ax1.text(50, 75, 'Scale: ×1', ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Arrow down
    ax1.annotate('', xy=(25, 52), xytext=(25, 55),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax1.text(28, 53, '÷2', fontsize=10, fontweight='bold')
    
    # Octave 1 - Half
    rect1 = patches.FancyBboxPatch((15, 28), 30, 22, boxstyle="round,pad=0.02",
                                    facecolor=colors[1], edgecolor='black', linewidth=2, alpha=0.8)
    ax1.add_patch(rect1)
    ax1.text(30, 39, f'OCTAVE 1\n{w//2}×{h//2}', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='white')
    ax1.text(50, 39, 'Scale: ×2', ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Arrow down
    ax1.annotate('', xy=(30, 25), xytext=(30, 28),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax1.text(33, 26, '÷2', fontsize=10, fontweight='bold')
    
    # Octave 2 - Quarter
    rect2 = patches.FancyBboxPatch((22, 8), 20, 15, boxstyle="round,pad=0.02",
                                    facecolor=colors[2], edgecolor='black', linewidth=2, alpha=0.8)
    ax1.add_patch(rect2)
    ax1.text(32, 15, f'OCTAVE 2\n{w//4}×{h//4}', ha='center', va='center', 
             fontsize=9, fontweight='bold', color='white')
    ax1.text(50, 15, 'Scale: ×4', ha='left', va='center', fontsize=11, fontweight='bold')
    
    # ============================================
    # TOP RIGHT: Coordinate Transformation (Real)
    # ============================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Coordinate Transformation to Original Image', fontsize=14, fontweight='bold', pad=10)
    
    # Show original image
    ax2.imshow(img, cmap='gray', alpha=0.5)
    
    # Pick 3 sample keypoints (one from each octave)
    sample_kps = []
    for oct_idx in range(3):
        if keypoints_per_octave[oct_idx]:
            # Find a keypoint in the middle region
            kps = keypoints_per_octave[oct_idx]
            best_kp = None
            for kp in kps:
                x, y = kp['orig_x'], kp['orig_y']
                if 100 < x < w - 100 and 100 < y < h - 100:
                    best_kp = kp
                    break
            if best_kp is None and kps:
                best_kp = kps[len(kps)//2]
            if best_kp:
                sample_kps.append(best_kp)
    
    # Draw keypoints with transformation arrows
    sizes = [8, 12, 18]
    for i, kp in enumerate(sample_kps):
        octave = kp['octave']
        local_x, local_y = kp['local_x'], kp['local_y']
        orig_x, orig_y = kp['orig_x'], kp['orig_y']
        scale_factor = 2 ** octave
        
        # Draw on original image
        circle = plt.Circle((orig_x, orig_y), sizes[octave], 
                           color=colors[octave], fill=False, linewidth=3)
        ax2.add_patch(circle)
        ax2.plot(orig_x, orig_y, 'o', color=colors[octave], markersize=6)
        
        # Add label
        ax2.annotate(f'Oct {octave}: ({local_x},{local_y}) ×{scale_factor} → ({orig_x},{orig_y})',
                    xy=(orig_x, orig_y), xytext=(orig_x + 20, orig_y - 30),
                    fontsize=9, fontweight='bold', color=colors[octave],
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color=colors[octave], lw=1.5))
    
    ax2.set_xlim(0, w)
    ax2.set_ylim(h, 0)
    ax2.axis('off')
    
    # ============================================
    # BOTTOM LEFT: Scale Factor Table with Real Numbers
    # ============================================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)
    ax3.axis('off')
    ax3.set_title('Scale Factor Table (Real Image)', fontsize=14, fontweight='bold', pad=10)
    
    # Table header
    header_y = 90
    ax3.text(12, header_y, 'Octave', ha='center', va='center', fontsize=11, fontweight='bold')
    ax3.text(35, header_y, 'Resolution', ha='center', va='center', fontsize=11, fontweight='bold')
    ax3.text(58, header_y, 'Scale', ha='center', va='center', fontsize=11, fontweight='bold')
    ax3.text(82, header_y, 'Transform', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Header line
    ax3.axhline(y=85, xmin=0.02, xmax=0.98, color='black', linewidth=2)
    
    # Table rows
    rows = [
        ('0', f'{w} × {h}', '×1', '(x,y) → (x,y)'),
        ('1', f'{w//2} × {h//2}', '×2', '(x,y) → (2x,2y)'),
        ('2', f'{w//4} × {h//4}', '×4', '(x,y) → (4x,4y)'),
    ]
    
    for i, (oct, res, scale, transform) in enumerate(rows):
        y_pos = 72 - i * 22
        
        # Background
        rect = patches.FancyBboxPatch((2, y_pos - 8), 96, 18, boxstyle="round,pad=0.01",
                                       facecolor=colors[i], edgecolor='black', linewidth=1, alpha=0.3)
        ax3.add_patch(rect)
        
        ax3.text(12, y_pos, oct, ha='center', va='center', fontsize=12, fontweight='bold', color=colors[i])
        ax3.text(35, y_pos, res, ha='center', va='center', fontsize=10, fontweight='bold')
        ax3.text(58, y_pos, scale, ha='center', va='center', fontsize=11, fontweight='bold')
        ax3.text(82, y_pos, transform, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ============================================
    # BOTTOM RIGHT: Complete Keypoint Combination Flow
    # ============================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 100)
    ax4.axis('off')
    ax4.set_title('Complete Keypoint Combination Flow', fontsize=14, fontweight='bold', pad=10)
    
    # Count keypoints per octave
    counts = [len(kps) for kps in keypoints_per_octave]
    
    # Octave boxes (left side)
    box_x = 5
    for i in range(3):
        y_pos = 75 - i * 28
        
        # Octave box
        rect = patches.FancyBboxPatch((box_x, y_pos - 8), 22, 18, boxstyle="round,pad=0.02",
                                       facecolor=colors[i], edgecolor='black', linewidth=2, alpha=0.8)
        ax4.add_patch(rect)
        ax4.text(box_x + 11, y_pos, f'Octave {i}\n{counts[i]} KPs', 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # Arrow to scaling
        ax4.annotate('', xy=(35, y_pos), xytext=(28, y_pos),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        # Scale factor box
        scale_factor = 2 ** i
        rect2 = patches.FancyBboxPatch((36, y_pos - 6), 14, 12, boxstyle="round,pad=0.02",
                                        facecolor='white', edgecolor=colors[i], linewidth=2)
        ax4.add_patch(rect2)
        ax4.text(43, y_pos, f'×{scale_factor}', ha='center', va='center', 
                fontsize=11, fontweight='bold', color=colors[i])
        
        # Arrow to combine
        ax4.annotate('', xy=(58, y_pos), xytext=(51, y_pos),
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
    
    # Combine box (right side)
    rect_combine = patches.FancyBboxPatch((60, 35), 30, 35, boxstyle="round,pad=0.02",
                                           facecolor='#9B59B6', edgecolor='black', linewidth=3, alpha=0.8)
    ax4.add_patch(rect_combine)
    ax4.text(75, 52, 'COMBINE\nAll Keypoints\non Original\nImage', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Total count
    total = sum(counts)
    rect_total = patches.FancyBboxPatch((55, 8), 40, 12, boxstyle="round,pad=0.02",
                                         facecolor='#F39C12', edgecolor='black', linewidth=2)
    ax4.add_patch(rect_total)
    ax4.text(75, 14, f'Total = {counts[0]} + {counts[1]} + {counts[2]} = {total}', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow from combine to total
    ax4.annotate('', xy=(75, 21), xytext=(75, 34),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    plt.savefig('../images/sift_scale_factor_real.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_scale_factor_real.png')

def main():
    print("Loading image...")
    img = load_image()
    h, w = img.shape
    print(f"Image size: {w} × {h}")
    
    print("Building Gaussian pyramid...")
    pyramid = build_gaussian_pyramid(img)
    
    print("Computing DoG...")
    dog_pyramid = compute_dog(pyramid)
    
    print("Detecting keypoints per octave...")
    keypoints_per_octave = detect_extrema_per_octave(dog_pyramid)
    
    for i, kps in enumerate(keypoints_per_octave):
        print(f"  Octave {i}: {len(kps)} keypoints")
    
    print("\nGenerating scale factor visualization...")
    plot_scale_factor_visualization(img, pyramid, keypoints_per_octave)
    
    print("\nDone!")

if __name__ == '__main__':
    main()

"""
ORB Scale Pyramid Visualization
Detailed visualization of the image pyramid used in ORB
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, '..', 'images')


def build_pyramid(img, n_levels=8, scale_factor=1.2):
    """Build image pyramid."""
    pyramid = [img.copy()]
    scales = [1.0]
    
    for level in range(1, n_levels):
        scale = 1.0 / (scale_factor ** level)
        new_h = int(img.shape[0] * scale)
        new_w = int(img.shape[1] * scale)
        
        if new_h < 10 or new_w < 10:
            break
        
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        resized = np.array(pil_resized) / 255.0
        
        pyramid.append(resized)
        scales.append(scale)
    
    return pyramid, scales


def visualize_pyramid_concept():
    """Visualize the concept of image pyramid."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, 'ORB Scale-Space Pyramid', ha='center', fontsize=16, fontweight='bold')
    
    # Draw pyramid levels
    base_w, base_h = 6, 4
    x_start = 2
    y_base = 1
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 8))
    
    for level in range(8):
        scale = 1.0 / (1.2 ** level)
        w = base_w * scale
        h = base_h * scale
        
        x = x_start + (base_w - w) / 2
        y = y_base + level * 0.8
        
        rect = Rectangle((x, y), w, h * 0.6, facecolor=colors[level], 
                         edgecolor='black', linewidth=1, alpha=0.8)
        ax.add_patch(rect)
        
        # Label
        ax.text(x + w + 0.3, y + h * 0.3, f'Level {level}: scale = 1/{1.2**level:.2f}',
               fontsize=9, va='center')
    
    # Annotations
    info_box = FancyBboxPatch((10, 2), 5.5, 6, boxstyle="round,pad=0.1",
                               facecolor='#e8f8f5', edgecolor='#1abc9c', linewidth=2)
    ax.add_patch(info_box)
    
    info_text = """
Scale Pyramid Parameters:

• Scale factor: f = 1.2
• Number of levels: 8

Level 0: Original (H × W)
Level 1: H/1.2 × W/1.2
Level 2: H/1.44 × W/1.44
Level 3: H/1.73 × W/1.73
...
Level 7: H/3.58 × W/3.58

Each level detects features
at different scales!
"""
    ax.text(10.3, 7.5, info_text, fontsize=9, va='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_pyramid_concept.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_pyramid_concept.png")


def visualize_pyramid_comparison():
    """Compare ORB pyramid vs SIFT pyramid."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    # ORB Pyramid (left)
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    ax1.text(5, 9.5, 'ORB Scale Pyramid', ha='center', fontsize=14, fontweight='bold', color='#3498db')
    
    # Draw ORB pyramid
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, 8))
    base_w, base_h = 5, 3
    
    for level in range(8):
        scale = 1.0 / (1.2 ** level)
        w = base_w * scale
        h = base_h * scale
        x = 2.5 + (base_w - w) / 2
        y = 1 + level * 0.8
        
        rect = Rectangle((x, y), w, h * 0.5, facecolor=colors[level],
                         edgecolor='black', linewidth=1)
        ax1.add_patch(rect)
        ax1.text(x - 0.3, y + h * 0.25, f'{level}', fontsize=9, ha='right')
    
    ax1.text(5, 0.3, 'Simple downsampling\nScale factor: 1.2', ha='center', fontsize=10, style='italic')
    
    # SIFT Pyramid (right)
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    ax2.text(5, 9.5, 'SIFT Scale Pyramid', ha='center', fontsize=14, fontweight='bold', color='#e74c3c')
    
    # Draw SIFT pyramid (octaves with scales)
    octave_colors = ['#e74c3c', '#f39c12', '#27ae60']
    y_start = 1
    
    for octave in range(3):
        base_scale = 1.0 / (2 ** octave)
        w = base_w * base_scale
        h = base_h * base_scale
        x = 2.5 + (base_w - w) / 2
        
        # Draw 5 scales per octave
        for scale in range(5):
            y = y_start + octave * 2.5 + scale * 0.4
            rect = Rectangle((x, y), w, h * 0.3, 
                            facecolor=octave_colors[octave],
                            edgecolor='black', linewidth=0.5, alpha=0.6 + scale * 0.08)
            ax2.add_patch(rect)
        
        ax2.text(x - 0.3, y_start + octave * 2.5 + 1, f'Oct {octave}', fontsize=9, ha='right')
    
    ax2.text(5, 0.3, 'Gaussian blur + downsampling\nOctaves with multiple scales', ha='center', fontsize=10, style='italic')
    
    plt.suptitle('ORB vs SIFT: Scale Space Construction', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_sift_pyramid_comparison.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_sift_pyramid_comparison.png")


def visualize_pyramid_with_image():
    """Visualize pyramid with actual image resizing."""
    # Load real input image
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    if os.path.exists(image_path):
        img_rgb = np.array(Image.open(image_path))
        if len(img_rgb.shape) == 3:
            img = (0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]) / 255.0
        else:
            img = img_rgb / 255.0
    else:
        # Fallback to synthetic test image
        np.random.seed(42)
        img = np.zeros((240, 320), dtype=np.float64)
        img[:, :] = 0.5
        img[50:100, 50:100] = 0.9
        img[100:150, 200:280] = 0.3
        img[150:200, 80:130] = 0.7
        for i in range(0, 240, 20):
            for j in range(0, 320, 20):
                img[i:i+10, j:j+10] += np.random.rand() * 0.1 - 0.05
        img = np.clip(img, 0, 1)
    
    # Build pyramid
    pyramid, scales = build_pyramid(img)
    
    # Visualize
    n_levels = min(len(pyramid), 8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    
    for i in range(n_levels):
        row, col = i // 4, i % 4
        ax = axes[row, col]
        
        ax.imshow(pyramid[i], cmap='gray', vmin=0, vmax=1)
        h, w = pyramid[i].shape
        ax.set_title(f'Level {i}\n{w}×{h} (scale={scales[i]:.3f})', fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_levels, 8):
        row, col = i // 4, i % 4
        axes[row, col].axis('off')
    
    plt.suptitle('ORB Scale Pyramid: Actual Image Downsampling\n(Scale factor = 1.2)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_pyramid_images.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_pyramid_images.png")


def visualize_scale_factor_effects():
    """Show how different scale factors affect the pyramid."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    scale_factors = [1.1, 1.2, 1.4]
    titles = ['Fine (f=1.1)\nMore levels, slower', 
              'Default (f=1.2)\nBalanced', 
              'Coarse (f=1.4)\nFewer levels, faster']
    
    for idx, (f, title) in enumerate(zip(scale_factors, titles)):
        ax = axes[idx]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        ax.text(5, 9.5, title, ha='center', fontsize=11, fontweight='bold')
        
        # Draw pyramid levels
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, 10))
        base_w, base_h = 6, 3
        level = 0
        
        while True:
            scale = 1.0 / (f ** level)
            if scale < 0.1:  # Stop when too small
                break
            
            w = base_w * scale
            h = base_h * scale
            x = 2 + (base_w - w) / 2
            y = 1 + level * 0.7
            
            if y > 8.5:
                break
            
            rect = Rectangle((x, y), w, h * 0.4, 
                            facecolor=colors[min(level, 9)],
                            edgecolor='black', linewidth=1, alpha=0.8)
            ax.add_patch(rect)
            
            # Size text
            orig_h, orig_w = 480, 640
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)
            ax.text(x + w + 0.2, y + h * 0.2, f'{new_w}×{new_h}', fontsize=8)
            
            level += 1
        
        ax.text(5, 0.3, f'{level} levels total', ha='center', fontsize=10, style='italic')
    
    plt.suptitle('Effect of Scale Factor on Pyramid Structure\n(Original: 640×480)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_scale_factor_effects.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_scale_factor_effects.png")


def visualize_keypoint_scale_mapping():
    """Show how keypoints from different levels map to original image."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Pyramid with keypoints
    ax1 = axes[0]
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Keypoints in Pyramid Levels', fontsize=12, fontweight='bold')
    
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
    base_w, base_h = 6, 3
    
    # Sample keypoints at each level
    kps_per_level = [
        [(1, 1), (3, 2), (4, 1)],  # Level 0
        [(1.5, 1), (2.5, 2)],      # Level 1
        [(1, 1.5), (2, 1)],        # Level 2
        [(1, 1)],                   # Level 3
        [(0.8, 0.8)],              # Level 4
    ]
    
    for level in range(5):
        scale = 1.0 / (1.2 ** level)
        w = base_w * scale
        h = base_h * scale
        x = 3 + (base_w - w) / 2
        y = 1 + level * 1.6
        
        rect = Rectangle((x, y), w, h * 0.8, facecolor='lightgray',
                         edgecolor='black', linewidth=1)
        ax1.add_patch(rect)
        
        # Draw keypoints
        for kp_x, kp_y in kps_per_level[level]:
            kp_screen_x = x + kp_x * scale
            kp_screen_y = y + kp_y * scale * 0.4
            circle = plt.Circle((kp_screen_x, kp_screen_y), 0.15, 
                               facecolor=colors[level], edgecolor='black')
            ax1.add_patch(circle)
        
        ax1.text(x - 0.3, y + h * 0.4, f'L{level}', fontsize=9, ha='right')
    
    # Right: All keypoints mapped to original
    ax2 = axes[1]
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('All Keypoints Mapped to Original', fontsize=12, fontweight='bold')
    
    # Draw original image
    rect = Rectangle((3, 2), 6, 6, facecolor='lightgray',
                     edgecolor='black', linewidth=2)
    ax2.add_patch(rect)
    
    # Map keypoints
    for level, kps in enumerate(kps_per_level):
        scale = 1.0 / (1.2 ** level)
        size = 0.2 + level * 0.1  # Larger circles for higher levels
        
        for kp_x, kp_y in kps:
            # Scale back to original coordinates
            orig_x = kp_x / scale
            orig_y = kp_y / scale
            
            # Map to screen coordinates
            screen_x = 3 + orig_x * 1.5
            screen_y = 2 + orig_y * 1.5
            
            circle = plt.Circle((screen_x, screen_y), size,
                               facecolor=colors[level], edgecolor='black', alpha=0.8)
            ax2.add_patch(circle)
    
    # Legend
    for i, color in enumerate(colors):
        circle = plt.Circle((10, 7 - i * 0.6), 0.15, facecolor=color, edgecolor='black')
        ax2.add_patch(circle)
        ax2.text(10.3, 7 - i * 0.6, f'Level {i}', fontsize=9, va='center')
    
    ax2.text(6, 0.8, 'Coordinate transform:\nkp_orig = kp_level × (1.2^level)', 
            ha='center', fontsize=10, style='italic')
    
    plt.suptitle('ORB: Multi-Scale Keypoint Detection', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'orb_keypoint_scale_mapping.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: orb_keypoint_scale_mapping.png")


def main():
    """Generate all scale pyramid visualizations."""
    print("=" * 60)
    print("ORB Scale Pyramid Visualizations")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("\n1. Generating pyramid concept diagram...")
    visualize_pyramid_concept()
    
    print("\n2. Generating ORB vs SIFT comparison...")
    visualize_pyramid_comparison()
    
    print("\n3. Generating pyramid with images...")
    visualize_pyramid_with_image()
    
    print("\n4. Generating scale factor effects...")
    visualize_scale_factor_effects()
    
    print("\n5. Generating keypoint scale mapping...")
    visualize_keypoint_scale_mapping()
    
    print("\n" + "=" * 60)
    print("Done! Generated all scale pyramid visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()

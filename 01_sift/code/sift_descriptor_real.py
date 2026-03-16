"""
Generate REAL SIFT Descriptor visualization images using actual keypoints from the image.
Shows the step-by-step descriptor creation process on real data.
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
    return magnitude, orientation, gx, gy

def find_good_keypoint(img, magnitude):
    """Find a keypoint with good gradient response for visualization."""
    h, w = img.shape
    # Look for a region with strong gradients (interesting features)
    best_response = 0
    best_x, best_y = w//2, h//2
    
    margin = 50  # Stay away from edges
    for y in range(margin, h - margin, 20):
        for x in range(margin, w - margin, 20):
            region = magnitude[y-16:y+16, x-16:x+16]
            if region.shape == (32, 32):
                response = np.mean(region)
                if response > best_response:
                    best_response = response
                    best_x, best_y = x, y
    
    return best_x, best_y

def compute_dominant_orientation(magnitude, orientation, kp_x, kp_y, radius=8):
    """Compute dominant orientation using 36-bin histogram."""
    hist = np.zeros(36)
    
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y, x = kp_y + dy, kp_x + dx
            if 0 <= y < magnitude.shape[0] and 0 <= x < magnitude.shape[1]:
                mag = magnitude[y, x]
                ori = orientation[y, x]
                # Gaussian weight
                weight = np.exp(-(dx**2 + dy**2) / (2 * (radius/2)**2))
                bin_idx = int(ori / 10) % 36
                hist[bin_idx] += mag * weight
    
    dominant_bin = np.argmax(hist)
    dominant_orientation = dominant_bin * 10 + 5  # Center of bin
    return dominant_orientation, hist

def plot_step1_orientation_assignment(img, magnitude, orientation, kp_x, kp_y):
    """Step 5: Show orientation assignment with 36-bin histogram."""
    dom_ori, hist = compute_dominant_orientation(magnitude, orientation, kp_x, kp_y)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Left: Image with keypoint
    ax1 = axes[0]
    ax1.imshow(img, cmap='gray')
    circle = plt.Circle((kp_x, kp_y), 20, color='red', fill=False, linewidth=3)
    ax1.add_patch(circle)
    # Draw orientation arrow
    arrow_len = 35
    dx = arrow_len * np.cos(np.radians(dom_ori))
    dy = arrow_len * np.sin(np.radians(dom_ori))
    ax1.arrow(kp_x, kp_y, dx, dy, head_width=10, head_length=6, fc='yellow', ec='yellow', linewidth=3)
    ax1.plot(kp_x, kp_y, 'r+', markersize=15, markeredgewidth=3)
    ax1.set_title(f'Keypoint at ({kp_x}, {kp_y})\nDominant Orientation: {dom_ori:.1f}°', fontsize=12, fontweight='bold')
    ax1.set_xlim(kp_x - 60, kp_x + 60)
    ax1.set_ylim(kp_y + 60, kp_y - 60)
    ax1.axis('off')
    
    # Middle: Gradient arrows in region
    ax2 = axes[1]
    region_size = 16
    ax2.imshow(img[kp_y-region_size:kp_y+region_size, kp_x-region_size:kp_x+region_size], 
               cmap='gray', extent=[0, 32, 32, 0])
    
    # Draw gradient arrows
    for dy in range(0, 32, 4):
        for dx in range(0, 32, 4):
            y, x = kp_y - region_size + dy, kp_x - region_size + dx
            if 0 <= y < magnitude.shape[0] and 0 <= x < magnitude.shape[1]:
                mag = magnitude[y, x]
                ori = orientation[y, x]
                arrow_len = 1.5 * (mag / (magnitude[kp_y-16:kp_y+16, kp_x-16:kp_x+16].max() + 1e-6))
                adx = arrow_len * np.cos(np.radians(ori))
                ady = arrow_len * np.sin(np.radians(ori))
                ax2.arrow(dx + 2, dy + 2, adx, ady, head_width=0.8, head_length=0.4, 
                         fc='cyan', ec='cyan', linewidth=0.8, alpha=0.8)
    
    ax2.set_title('Gradient Directions in Region\n(Arrow = gradient direction)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # Right: 36-bin histogram
    ax3 = axes[2]
    bars = ax3.bar(range(36), hist, color='blue', edgecolor='black', alpha=0.7)
    bars[np.argmax(hist)].set_color('red')  # Highlight dominant
    ax3.axvline(x=np.argmax(hist), color='red', linestyle='--', linewidth=2, label=f'Dominant: {dom_ori:.0f}°')
    ax3.set_xlabel('Orientation Bin (each bin = 10°)', fontsize=11)
    ax3.set_ylabel('Weighted Magnitude Sum', fontsize=11)
    ax3.set_title('36-bin Orientation Histogram\nRed = Dominant Direction', fontsize=12, fontweight='bold')
    ax3.set_xticks([0, 9, 18, 27, 35])
    ax3.set_xticklabels(['0°', '90°', '180°', '270°', '360°'])
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('../images/sift_desc_orientation.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_desc_orientation.png')
    
    return dom_ori

def plot_step2_16x16_region(img, kp_x, kp_y, dom_ori):
    """Step 6.1: Show 16x16 region extraction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    half = 8
    
    # Left: Full image with region highlighted
    ax1 = axes[0]
    ax1.imshow(img, cmap='gray')
    rect = patches.Rectangle((kp_x - half, kp_y - half), 16, 16,
                              linewidth=3, edgecolor='lime', facecolor='none')
    ax1.add_patch(rect)
    circle = plt.Circle((kp_x, kp_y), 3, color='red', fill=True)
    ax1.add_patch(circle)
    # Orientation arrow
    arrow_len = 20
    dx = arrow_len * np.cos(np.radians(dom_ori))
    dy = arrow_len * np.sin(np.radians(dom_ori))
    ax1.arrow(kp_x, kp_y, dx, dy, head_width=5, head_length=3, fc='yellow', ec='yellow', linewidth=2)
    
    ax1.set_title(f'16×16 Region Around Keypoint\nGreen box = descriptor region', fontsize=12, fontweight='bold')
    ax1.set_xlim(kp_x - 40, kp_x + 40)
    ax1.set_ylim(kp_y + 40, kp_y - 40)
    ax1.axis('off')
    
    # Right: Zoomed region with pixel grid
    ax2 = axes[1]
    region = img[kp_y-half:kp_y+half, kp_x-half:kp_x+half]
    ax2.imshow(region, cmap='gray', extent=[0, 16, 16, 0])
    
    # Draw pixel grid
    for i in range(17):
        ax2.axhline(y=i, color='lime', linewidth=0.5, alpha=0.7)
        ax2.axvline(x=i, color='lime', linewidth=0.5, alpha=0.7)
    
    # Mark center
    ax2.plot(8, 8, 'r+', markersize=20, markeredgewidth=3)
    ax2.set_title('Zoomed 16×16 Region\n(256 pixels total)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    plt.savefig('../images/sift_desc_16x16.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_desc_16x16.png')

def plot_step3_4x4_grid(img, kp_x, kp_y):
    """Step 6.2: Show 4x4 subregion division."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    half = 8
    region = img[kp_y-half:kp_y+half, kp_x-half:kp_x+half]
    
    # Colors for subregions
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
              '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
              '#BB8FCE', '#85C1E9', '#F8B500', '#58D68D',
              '#EC7063', '#5DADE2', '#F4D03F', '#AF7AC5']
    
    # Left: Region with 4x4 grid overlay
    ax1 = axes[0]
    ax1.imshow(region, cmap='gray', extent=[0, 16, 16, 0])
    
    for i in range(4):
        for j in range(4):
            rect = patches.Rectangle((j*4, i*4), 4, 4, linewidth=2, 
                                     edgecolor=colors[i*4+j], facecolor=colors[i*4+j], alpha=0.3)
            ax1.add_patch(rect)
            ax1.text(j*4 + 2, i*4 + 2, f'{i*4+j}', fontsize=10, ha='center', va='center',
                    color='black', fontweight='bold')
    
    # Draw thick grid lines
    for i in range(5):
        ax1.axhline(y=i*4, color='black', linewidth=2)
        ax1.axvline(x=i*4, color='black', linewidth=2)
    
    ax1.set_title('4×4 Grid = 16 Subregions\nEach subregion = 4×4 pixels', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    # Right: Schematic
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 4.5)
    ax2.set_ylim(4.5, -0.5)
    
    for i in range(4):
        for j in range(4):
            rect = patches.Rectangle((j, i), 1, 1, linewidth=2, 
                                     edgecolor='black', facecolor=colors[i*4+j], alpha=0.5)
            ax2.add_patch(rect)
            ax2.text(j + 0.5, i + 0.5, f'Sub\n{i*4+j}', fontsize=9, ha='center', va='center',
                    fontweight='bold')
    
    ax2.set_title('16 Subregions Layout\n(Each will have 8-bin histogram)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax2.set_xticklabels(['0', '1', '2', '3'])
    ax2.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax2.set_yticklabels(['0', '1', '2', '3'])
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('../images/sift_desc_4x4grid.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_desc_4x4grid.png')

def plot_step4_gradients(img, magnitude, orientation, kp_x, kp_y):
    """Step 6.3: Show gradients in subregions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    half = 8
    region_img = img[kp_y-half:kp_y+half, kp_x-half:kp_x+half]
    region_mag = magnitude[kp_y-half:kp_y+half, kp_x-half:kp_x+half]
    region_ori = orientation[kp_y-half:kp_y+half, kp_x-half:kp_x+half]
    
    # Left: One subregion detailed
    ax1 = axes[0]
    sub_img = region_img[0:4, 0:4]
    sub_mag = region_mag[0:4, 0:4]
    sub_ori = region_ori[0:4, 0:4]
    
    ax1.imshow(sub_img, cmap='gray', extent=[0, 4, 4, 0])
    
    # Draw gradient arrows
    max_mag = sub_mag.max() + 1e-6
    for py in range(4):
        for px in range(4):
            mag = sub_mag[py, px]
            ori = sub_ori[py, px]
            arrow_len = 0.4 * (mag / max_mag)
            dx = arrow_len * np.cos(np.radians(ori))
            dy = arrow_len * np.sin(np.radians(ori))
            ax1.arrow(px + 0.5, py + 0.5, dx, dy, head_width=0.15, head_length=0.08, 
                     fc='red', ec='red', linewidth=1.5)
    
    # Grid
    for i in range(5):
        ax1.axhline(y=i, color='cyan', linewidth=1)
        ax1.axvline(x=i, color='cyan', linewidth=1)
    
    ax1.set_title('Subregion 0 (4×4 pixels)\nRed arrows = gradient directions', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Right: Full 16x16 with gradients
    ax2 = axes[1]
    ax2.imshow(region_img, cmap='gray', extent=[0, 16, 16, 0])
    
    # Draw gradients (sampled)
    max_mag = region_mag.max() + 1e-6
    for py in range(0, 16, 2):
        for px in range(0, 16, 2):
            mag = region_mag[py, px]
            ori = region_ori[py, px]
            arrow_len = 0.8 * (mag / max_mag)
            dx = arrow_len * np.cos(np.radians(ori))
            dy = arrow_len * np.sin(np.radians(ori))
            ax2.arrow(px + 1, py + 1, dx, dy, head_width=0.3, head_length=0.15, 
                     fc='yellow', ec='yellow', linewidth=0.8, alpha=0.9)
    
    # Draw 4x4 subregion grid
    for i in range(5):
        ax2.axhline(y=i*4, color='lime', linewidth=2)
        ax2.axvline(x=i*4, color='lime', linewidth=2)
    
    ax2.set_title('All 16×16 with Gradient Arrows\nGreen grid = 4×4 subregions', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('../images/sift_desc_gradients.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_desc_gradients.png')

def plot_step5_8bin_histograms(magnitude, orientation, kp_x, kp_y):
    """Step 6.4: Show 8-bin histograms for subregions."""
    half = 8
    region_mag = magnitude[kp_y-half:kp_y+half, kp_x-half:kp_x+half]
    region_ori = orientation[kp_y-half:kp_y+half, kp_x-half:kp_x+half]
    
    # Compute 8-bin histograms for all 16 subregions
    histograms = []
    for row in range(4):
        for col in range(4):
            hist = np.zeros(8)
            for dy in range(4):
                for dx in range(4):
                    py, px = row * 4 + dy, col * 4 + dx
                    mag = region_mag[py, px]
                    ori = region_ori[py, px]
                    bin_idx = int(ori / 45) % 8
                    hist[bin_idx] += mag
            histograms.append(hist)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: 8-bin explanation
    ax1 = axes[0, 0]
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, 8))
    bin_labels = ['0°-45°', '45°-90°', '90°-135°', '135°-180°',
                  '180°-225°', '225°-270°', '270°-315°', '315°-360°']
    
    for i in range(8):
        angle1 = np.radians(i * 45)
        angle2 = np.radians((i + 1) * 45)
        mid_angle = (angle1 + angle2) / 2
        
        ax1.plot([0, np.cos(angle1)], [0, np.sin(angle1)], 'k-', linewidth=1)
        
        theta_fill = np.linspace(angle1, angle2, 20)
        x_fill = np.concatenate([[0], 0.9*np.cos(theta_fill), [0]])
        y_fill = np.concatenate([[0], 0.9*np.sin(theta_fill), [0]])
        ax1.fill(x_fill, y_fill, color=colors[i], alpha=0.4)
        
        ax1.text(0.6*np.cos(mid_angle), 0.6*np.sin(mid_angle), f'B{i}',
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.set_title('8-bin Orientation Histogram\nEach bin = 45°', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Top-right: Example histogram for subregion 0
    ax2 = axes[0, 1]
    bars = ax2.bar(range(8), histograms[0], color=colors, edgecolor='black', linewidth=1)
    ax2.set_xticks(range(8))
    ax2.set_xticklabels([f'B{i}' for i in range(8)])
    ax2.set_xlabel('Orientation Bin (45° each)', fontsize=11)
    ax2.set_ylabel('Magnitude Sum', fontsize=11)
    ax2.set_title('Subregion 0: 8-bin Histogram\n(Sum of gradient magnitudes per direction)', fontsize=12, fontweight='bold')
    
    # Bottom-left: Grid of all 16 histograms (mini)
    ax3 = axes[1, 0]
    ax3.set_xlim(0, 4)
    ax3.set_ylim(4, 0)
    
    for row in range(4):
        for col in range(4):
            idx = row * 4 + col
            hist = histograms[idx]
            max_h = max(hist.max(), 1)
            for b in range(8):
                bar_h = 0.8 * hist[b] / max_h
                rect = patches.Rectangle((col + b*0.1 + 0.1, row + 0.9 - bar_h), 0.08, bar_h,
                                         facecolor=colors[b], edgecolor='none')
                ax3.add_patch(rect)
            # Subregion border
            rect = patches.Rectangle((col, row), 1, 1, linewidth=1, 
                                     edgecolor='black', facecolor='none')
            ax3.add_patch(rect)
            ax3.text(col + 0.5, row + 0.15, f'{idx}', fontsize=8, ha='center', va='center')
    
    ax3.set_title('All 16 Subregions with their 8-bin Histograms', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    ax3.set_aspect('equal')
    
    # Bottom-right: Total = 128 values
    ax4 = axes[1, 1]
    all_values = np.concatenate(histograms)
    
    # Normalize
    all_values = all_values / (np.linalg.norm(all_values) + 1e-6)
    all_values = np.clip(all_values, 0, 0.2)
    all_values = all_values / (np.linalg.norm(all_values) + 1e-6)
    
    bar_colors = np.repeat(np.arange(16), 8)
    cmap = plt.cm.tab20(bar_colors / 16)
    ax4.bar(range(128), all_values, color=cmap, edgecolor='none', width=1)
    
    for i in range(1, 16):
        ax4.axvline(x=i*8 - 0.5, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Descriptor Index (0-127)', fontsize=11)
    ax4.set_ylabel('Normalized Value', fontsize=11)
    ax4.set_title('128-D Descriptor Vector\n16 subregions × 8 bins = 128 values', fontsize=12, fontweight='bold')
    ax4.set_xlim(-1, 128)
    
    plt.tight_layout()
    plt.savefig('../images/sift_desc_histograms.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_desc_histograms.png')
    
    return histograms

def plot_step6_final_descriptor(histograms):
    """Step 6.5: Show final 128-D descriptor."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Flatten and normalize
    descriptor = np.concatenate(histograms)
    descriptor = descriptor / (np.linalg.norm(descriptor) + 1e-6)
    descriptor = np.clip(descriptor, 0, 0.2)
    descriptor = descriptor / (np.linalg.norm(descriptor) + 1e-6)
    
    # Top: Heatmap view
    ax1 = axes[0]
    desc_2d = descriptor.reshape(16, 8)
    im = ax1.imshow(desc_2d.T, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(16))
    ax1.set_xticklabels([f'S{i}' for i in range(16)], fontsize=9)
    ax1.set_yticks(range(8))
    ax1.set_yticklabels([f'B{i}' for i in range(8)])
    ax1.set_xlabel('16 Subregions', fontsize=12)
    ax1.set_ylabel('8 Orientation Bins', fontsize=12)
    ax1.set_title('128-D Descriptor as 16×8 Matrix\n(Heatmap: darker = higher value)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Normalized Value')
    
    # Bottom: Bar chart
    ax2 = axes[1]
    colors = plt.cm.tab20(np.repeat(np.arange(16), 8) / 16)
    ax2.bar(range(128), descriptor, color=colors, edgecolor='none', width=1)
    
    for i in range(1, 16):
        ax2.axvline(x=i*8 - 0.5, color='black', linewidth=1, linestyle='-', alpha=0.3)
    
    # Label subregions
    for i in range(16):
        ax2.text(i*8 + 4, max(descriptor) * 1.05, f'S{i}', ha='center', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Descriptor Index (0-127)', fontsize=12)
    ax2.set_ylabel('Normalized Value', fontsize=12)
    ax2.set_title('Final 128-D Descriptor Vector\n(Each color = one subregion\'s 8 histogram bins)', fontsize=14, fontweight='bold')
    ax2.set_xlim(-1, 128)
    
    plt.tight_layout()
    plt.savefig('../images/sift_desc_final128.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_desc_final128.png')

def plot_complete_pipeline(img, kp_x, kp_y, dom_ori):
    """Create a complete pipeline summary image."""
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    half = 8
    
    # 1. Keypoint
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img, cmap='gray')
    circle = plt.Circle((kp_x, kp_y), 15, color='red', fill=False, linewidth=2)
    ax1.add_patch(circle)
    ax1.arrow(kp_x, kp_y, 20*np.cos(np.radians(dom_ori)), 20*np.sin(np.radians(dom_ori)),
              head_width=6, head_length=4, fc='yellow', ec='yellow', linewidth=2)
    ax1.set_xlim(kp_x - 40, kp_x + 40)
    ax1.set_ylim(kp_y + 40, kp_y - 40)
    ax1.set_title('1. Keypoint\n+ Orientation', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # 2. 16x16 region
    ax2 = fig.add_subplot(gs[0, 1])
    region = img[kp_y-half:kp_y+half, kp_x-half:kp_x+half]
    ax2.imshow(region, cmap='gray')
    ax2.set_title('2. Extract\n16×16 Region', fontsize=11, fontweight='bold')
    ax2.axis('off')
    
    # 3. 4x4 grid
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(region, cmap='gray', extent=[0, 16, 16, 0])
    for i in range(5):
        ax3.axhline(y=i*4, color='lime', linewidth=2)
        ax3.axvline(x=i*4, color='lime', linewidth=2)
    ax3.set_title('3. Divide into\n4×4 = 16 Subregions', fontsize=11, fontweight='bold')
    ax3.axis('off')
    
    # 4. Gradients
    ax4 = fig.add_subplot(gs[0, 3])
    magnitude, orientation, _, _ = compute_gradients(img)
    region_mag = magnitude[kp_y-half:kp_y+half, kp_x-half:kp_x+half]
    region_ori = orientation[kp_y-half:kp_y+half, kp_x-half:kp_x+half]
    ax4.imshow(region, cmap='gray', extent=[0, 16, 16, 0])
    max_mag = region_mag.max() + 1e-6
    for py in range(0, 16, 3):
        for px in range(0, 16, 3):
            mag = region_mag[py, px]
            ori = region_ori[py, px]
            arrow_len = 1.2 * (mag / max_mag)
            dx = arrow_len * np.cos(np.radians(ori))
            dy = arrow_len * np.sin(np.radians(ori))
            ax4.arrow(px + 1.5, py + 1.5, dx, dy, head_width=0.4, head_length=0.2, 
                     fc='yellow', ec='yellow', linewidth=0.6)
    ax4.set_title('4. Compute\nGradients', fontsize=11, fontweight='bold')
    ax4.axis('off')
    
    # 5. 8-bin histogram (example)
    ax5 = fig.add_subplot(gs[1, 0:2])
    hist = np.zeros(8)
    for dy in range(4):
        for dx in range(4):
            mag = region_mag[dy, dx]
            ori = region_ori[dy, dx]
            bin_idx = int(ori / 45) % 8
            hist[bin_idx] += mag
    colors = plt.cm.rainbow(np.linspace(0, 1, 8))
    ax5.bar(range(8), hist, color=colors, edgecolor='black')
    ax5.set_xticks(range(8))
    ax5.set_xticklabels([f'{i*45}°-{(i+1)*45}°' for i in range(8)], rotation=45, ha='right', fontsize=8)
    ax5.set_title('5. Build 8-bin Histogram\n(for each of 16 subregions)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Magnitude')
    
    # 6. 128-D descriptor
    ax6 = fig.add_subplot(gs[1, 2:4])
    # Compute all histograms
    all_hist = []
    for row in range(4):
        for col in range(4):
            hist = np.zeros(8)
            for dy in range(4):
                for dx in range(4):
                    py, px = row * 4 + dy, col * 4 + dx
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
    ax6.set_title('6. Concatenate → 128-D Descriptor\n(16 subregions × 8 bins)', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Index (0-127)')
    ax6.set_ylabel('Value')
    ax6.set_xlim(-1, 128)
    
    plt.suptitle('SIFT Descriptor Creation Pipeline (Real Image)', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('../images/sift_desc_pipeline_real.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_desc_pipeline_real.png')

def main():
    print("Loading image...")
    img = load_image()
    print(f"Image size: {img.shape}")
    
    print("\nComputing gradients...")
    magnitude, orientation, gx, gy = compute_gradients(img)
    
    print("Finding good keypoint...")
    kp_x, kp_y = find_good_keypoint(img, magnitude)
    print(f"Selected keypoint: ({kp_x}, {kp_y})")
    
    print("\n=== Generating SIFT Descriptor Images (Real Data) ===\n")
    
    # Step 5: Orientation assignment
    print("Step 5: Orientation Assignment...")
    dom_ori = plot_step1_orientation_assignment(img, magnitude, orientation, kp_x, kp_y)
    print(f"Dominant orientation: {dom_ori:.1f}°")
    
    # Step 6.1: 16x16 region
    print("\nStep 6.1: 16x16 Region...")
    plot_step2_16x16_region(img, kp_x, kp_y, dom_ori)
    
    # Step 6.2: 4x4 grid
    print("Step 6.2: 4x4 Subregions...")
    plot_step3_4x4_grid(img, kp_x, kp_y)
    
    # Step 6.3: Gradients
    print("Step 6.3: Gradients...")
    plot_step4_gradients(img, magnitude, orientation, kp_x, kp_y)
    
    # Step 6.4: 8-bin histograms
    print("Step 6.4: 8-bin Histograms...")
    histograms = plot_step5_8bin_histograms(magnitude, orientation, kp_x, kp_y)
    
    # Step 6.5: Final 128-D
    print("Step 6.5: Final 128-D Descriptor...")
    plot_step6_final_descriptor(histograms)
    
    # Complete pipeline
    print("\nComplete Pipeline...")
    plot_complete_pipeline(img, kp_x, kp_y, dom_ori)
    
    print("\n=== All images generated! ===")

if __name__ == '__main__':
    main()

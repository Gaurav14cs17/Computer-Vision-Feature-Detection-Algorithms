"""
Generate detailed SIFT Descriptor visualization images showing:
1. Keypoint with orientation
2. 16x16 region around keypoint
3. 4x4 subregions
4. Gradient directions in each subregion
5. 8-bin histograms per subregion
6. Final 128-D descriptor
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
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
    # Compute gradients
    gy, gx = np.gradient(img.astype(np.float64))
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi  # Convert to degrees
    orientation = (orientation + 360) % 360  # Make positive [0, 360)
    return magnitude, orientation

def plot_step1_keypoint_orientation(img, kp_x, kp_y, kp_orientation):
    """Step 1: Show keypoint with its dominant orientation."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(img, cmap='gray')
    
    # Draw keypoint circle
    circle = plt.Circle((kp_x, kp_y), 20, color='red', fill=False, linewidth=3)
    ax.add_patch(circle)
    
    # Draw orientation arrow
    arrow_len = 30
    dx = arrow_len * np.cos(np.radians(kp_orientation))
    dy = arrow_len * np.sin(np.radians(kp_orientation))
    ax.arrow(kp_x, kp_y, dx, dy, head_width=8, head_length=5, fc='yellow', ec='yellow', linewidth=2)
    
    # Mark center
    ax.plot(kp_x, kp_y, 'r+', markersize=15, markeredgewidth=3)
    
    ax.set_title(f'Step 1: Keypoint at ({kp_x}, {kp_y}) with Orientation {kp_orientation:.1f}°\n'
                 f'Yellow arrow shows dominant gradient direction', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/sift_desc_step1_keypoint.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_desc_step1_keypoint.png')

def plot_step2_16x16_region(img, kp_x, kp_y, kp_orientation):
    """Step 2: Show 16x16 region around keypoint."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Full image with region highlighted
    ax1 = axes[0]
    ax1.imshow(img, cmap='gray')
    
    # Draw 16x16 region
    half_size = 8
    rect = patches.Rectangle((kp_x - half_size, kp_y - half_size), 16, 16,
                              linewidth=3, edgecolor='cyan', facecolor='none')
    ax1.add_patch(rect)
    
    # Draw keypoint
    ax1.plot(kp_x, kp_y, 'r+', markersize=15, markeredgewidth=3)
    
    # Draw orientation
    arrow_len = 20
    dx = arrow_len * np.cos(np.radians(kp_orientation))
    dy = arrow_len * np.sin(np.radians(kp_orientation))
    ax1.arrow(kp_x, kp_y, dx, dy, head_width=5, head_length=3, fc='yellow', ec='yellow', linewidth=2)
    
    ax1.set_title('16×16 Region Around Keypoint', fontsize=14, fontweight='bold')
    ax1.set_xlim(kp_x - 50, kp_x + 50)
    ax1.set_ylim(kp_y + 50, kp_y - 50)
    ax1.axis('off')
    
    # Right: Zoomed 16x16 region
    ax2 = axes[1]
    region = img[max(0, kp_y-half_size):kp_y+half_size, max(0, kp_x-half_size):kp_x+half_size]
    ax2.imshow(region, cmap='gray', extent=[0, 16, 16, 0])
    
    # Draw grid lines for 16x16
    for i in range(17):
        ax2.axhline(y=i, color='cyan', linewidth=0.5, alpha=0.5)
        ax2.axvline(x=i, color='cyan', linewidth=0.5, alpha=0.5)
    
    ax2.plot(8, 8, 'r+', markersize=20, markeredgewidth=3)
    ax2.set_title('Zoomed 16×16 Region (Each cell = 1 pixel)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    plt.savefig('../images/sift_desc_step2_16x16_region.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_desc_step2_16x16_region.png')

def plot_step3_4x4_subregions(img, kp_x, kp_y):
    """Step 3: Show 4x4 subregions (16 subregions total)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    half_size = 8
    
    # Left: Zoomed region with 4x4 grid
    ax1 = axes[0]
    region = img[max(0, kp_y-half_size):kp_y+half_size, max(0, kp_x-half_size):kp_x+half_size]
    ax1.imshow(region, cmap='gray', extent=[0, 16, 16, 0])
    
    # Draw 4x4 subregion grid (each subregion is 4x4 pixels)
    colors = ['red', 'green', 'blue', 'orange']
    for i in range(4):
        for j in range(4):
            rect = patches.Rectangle((j*4, i*4), 4, 4, linewidth=2, 
                                     edgecolor=colors[(i+j)%4], facecolor='none')
            ax1.add_patch(rect)
            # Label each subregion
            ax1.text(j*4 + 2, i*4 + 2, f'{i*4 + j}', fontsize=10, ha='center', va='center',
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=colors[(i+j)%4], alpha=0.7))
    
    ax1.set_title('16×16 Region Divided into 4×4 Grid\n(16 subregions, each 4×4 pixels)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    # Right: Schematic diagram
    ax2 = axes[1]
    ax2.set_xlim(0, 16)
    ax2.set_ylim(16, 0)
    
    for i in range(4):
        for j in range(4):
            rect = patches.Rectangle((j*4, i*4), 4, 4, linewidth=2, 
                                     edgecolor='black', facecolor=colors[(i+j)%4], alpha=0.3)
            ax2.add_patch(rect)
            ax2.text(j*4 + 2, i*4 + 2, f'Sub-\nregion\n{i*4 + j}', fontsize=9, ha='center', va='center',
                    fontweight='bold')
    
    ax2.set_title('4×4 Grid = 16 Subregions\nEach subregion = 4×4 pixels', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    ax2.grid(True, linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/sift_desc_step3_4x4_subregions.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_desc_step3_4x4_subregions.png')

def plot_step4_gradients_in_subregion(img, kp_x, kp_y):
    """Step 4: Show gradient directions in subregions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    half_size = 8
    magnitude, orientation = compute_gradients(img)
    
    # Extract region
    region_mag = magnitude[max(0, kp_y-half_size):kp_y+half_size, max(0, kp_x-half_size):kp_x+half_size]
    region_ori = orientation[max(0, kp_y-half_size):kp_y+half_size, max(0, kp_x-half_size):kp_x+half_size]
    region_img = img[max(0, kp_y-half_size):kp_y+half_size, max(0, kp_x-half_size):kp_x+half_size]
    
    # Left: One subregion with gradient arrows
    ax1 = axes[0]
    # Show one 4x4 subregion (top-left)
    sub_img = region_img[0:4, 0:4]
    sub_mag = region_mag[0:4, 0:4]
    sub_ori = region_ori[0:4, 0:4]
    
    ax1.imshow(sub_img, cmap='gray', extent=[0, 4, 4, 0])
    
    # Draw gradient arrows for each pixel
    for py in range(4):
        for px in range(4):
            mag = sub_mag[py, px]
            ori = sub_ori[py, px]
            # Normalize arrow length
            arrow_len = 0.3 * (mag / (sub_mag.max() + 1e-6))
            dx = arrow_len * np.cos(np.radians(ori))
            dy = arrow_len * np.sin(np.radians(ori))
            ax1.arrow(px + 0.5, py + 0.5, dx, dy, head_width=0.15, head_length=0.1, 
                     fc='red', ec='red', linewidth=1)
    
    # Draw grid
    for i in range(5):
        ax1.axhline(y=i, color='cyan', linewidth=1)
        ax1.axvline(x=i, color='cyan', linewidth=1)
    
    ax1.set_title('One 4×4 Subregion with Gradient Arrows\n(Arrow direction = gradient direction)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    # Right: Full 16x16 with all gradient arrows (sampled)
    ax2 = axes[1]
    ax2.imshow(region_img, cmap='gray', extent=[0, 16, 16, 0])
    
    # Draw gradient arrows (every 2 pixels for clarity)
    for py in range(0, 16, 2):
        for px in range(0, 16, 2):
            if py < region_mag.shape[0] and px < region_mag.shape[1]:
                mag = region_mag[py, px]
                ori = region_ori[py, px]
                arrow_len = 0.8 * (mag / (region_mag.max() + 1e-6))
                dx = arrow_len * np.cos(np.radians(ori))
                dy = arrow_len * np.sin(np.radians(ori))
                ax2.arrow(px + 1, py + 1, dx, dy, head_width=0.3, head_length=0.2, 
                         fc='yellow', ec='yellow', linewidth=0.5, alpha=0.8)
    
    # Draw 4x4 subregion grid
    for i in range(5):
        ax2.axhline(y=i*4, color='green', linewidth=2)
        ax2.axvline(x=i*4, color='green', linewidth=2)
    
    ax2.set_title('All 16×16 Region with Gradient Arrows\n(Sampled every 2 pixels)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    plt.savefig('../images/sift_desc_step4_gradients.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_desc_step4_gradients.png')

def plot_step5_8bin_histogram():
    """Step 5: Show 8-bin orientation histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Histogram bins explanation
    ax1 = axes[0]
    
    # Draw a circle with 8 bins
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2)
    
    # Draw bin boundaries
    bin_angles = np.linspace(0, 360, 9)
    colors = plt.cm.rainbow(np.linspace(0, 1, 8))
    
    for i in range(8):
        angle1 = np.radians(bin_angles[i])
        angle2 = np.radians(bin_angles[i+1])
        mid_angle = (angle1 + angle2) / 2
        
        # Draw bin boundary lines
        ax1.plot([0, np.cos(angle1)], [0, np.sin(angle1)], 'k-', linewidth=1)
        
        # Fill bin with color
        theta_fill = np.linspace(angle1, angle2, 20)
        x_fill = np.concatenate([[0], np.cos(theta_fill), [0]])
        y_fill = np.concatenate([[0], np.sin(theta_fill), [0]])
        ax1.fill(x_fill, y_fill, color=colors[i], alpha=0.3)
        
        # Label bin
        ax1.text(0.7*np.cos(mid_angle), 0.7*np.sin(mid_angle), 
                f'Bin {i}\n{int(bin_angles[i])}°-{int(bin_angles[i+1])}°',
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_aspect('equal')
    ax1.set_title('8-Bin Orientation Histogram\n(Each bin = 45°)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Right: Example histogram
    ax2 = axes[1]
    
    # Sample histogram data
    hist_values = [15, 8, 12, 25, 30, 18, 10, 5]
    bin_labels = ['0°-45°', '45°-90°', '90°-135°', '135°-180°', 
                  '180°-225°', '225°-270°', '270°-315°', '315°-360°']
    
    bars = ax2.bar(range(8), hist_values, color=colors, edgecolor='black', linewidth=1)
    ax2.set_xticks(range(8))
    ax2.set_xticklabels([f'Bin {i}' for i in range(8)], rotation=45, ha='right')
    ax2.set_ylabel('Weighted Magnitude Sum', fontsize=12)
    ax2.set_xlabel('Orientation Bins', fontsize=12)
    ax2.set_title('Example: 8-bin Histogram for One Subregion\n(Gradient magnitudes summed per direction)', 
                  fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, val in zip(bars, hist_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../images/sift_desc_step5_8bin_histogram.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_desc_step5_8bin_histogram.png')

def plot_step6_128d_descriptor():
    """Step 6: Show final 128-D descriptor."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Top: 16 subregions × 8 bins = 128 values
    ax1 = axes[0]
    
    # Generate sample descriptor values
    np.random.seed(42)
    descriptor = np.random.rand(16, 8) * 0.5  # Simulate normalized values
    
    # Create heatmap showing all 16 subregions and their 8 bins
    im = ax1.imshow(descriptor.T, cmap='YlOrRd', aspect='auto')
    
    ax1.set_xticks(range(16))
    ax1.set_xticklabels([f'Sub {i}' for i in range(16)], rotation=45, ha='right', fontsize=9)
    ax1.set_yticks(range(8))
    ax1.set_yticklabels([f'Bin {i}' for i in range(8)])
    ax1.set_xlabel('16 Subregions', fontsize=12)
    ax1.set_ylabel('8 Orientation Bins', fontsize=12)
    ax1.set_title('128-D Descriptor: 16 Subregions × 8 Bins = 128 Values\n(Heatmap showing histogram values)', 
                  fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Normalized Value')
    
    # Bottom: Flattened 128-D vector
    ax2 = axes[1]
    
    descriptor_flat = descriptor.flatten()
    colors = plt.cm.tab20(np.repeat(np.arange(16), 8) / 16)
    
    bars = ax2.bar(range(128), descriptor_flat, color=colors, edgecolor='none', width=1)
    
    # Add subregion separators
    for i in range(1, 16):
        ax2.axvline(x=i*8 - 0.5, color='black', linewidth=1, linestyle='--', alpha=0.5)
    
    # Label some subregions
    for i in range(16):
        ax2.text(i*8 + 4, max(descriptor_flat) + 0.05, f'S{i}', ha='center', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Descriptor Index (0-127)', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Flattened 128-D Descriptor Vector\n(Each color = one subregion\'s 8 bins)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlim(-1, 128)
    
    plt.tight_layout()
    plt.savefig('../images/sift_desc_step6_128d_descriptor.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_desc_step6_128d_descriptor.png')

def plot_complete_descriptor_pipeline():
    """Create a complete pipeline visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Step 1: Keypoint
    ax = axes[0, 0]
    ax.text(0.5, 0.7, '●', fontsize=80, ha='center', va='center', color='red')
    ax.arrow(0.5, 0.65, 0.2, -0.1, head_width=0.05, head_length=0.03, fc='yellow', ec='yellow', linewidth=3)
    ax.text(0.5, 0.3, 'Step 1:\nKeypoint with\nOrientation', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Step 2: 16x16 region
    ax = axes[0, 1]
    rect = patches.Rectangle((0.2, 0.3), 0.6, 0.5, linewidth=3, edgecolor='cyan', facecolor='lightgray')
    ax.add_patch(rect)
    ax.text(0.5, 0.55, '16×16\nRegion', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(0.5, 0.15, 'Step 2:\nExtract 16×16\nRegion', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Step 3: 4x4 subregions
    ax = axes[0, 2]
    for i in range(4):
        for j in range(4):
            rect = patches.Rectangle((0.2 + j*0.15, 0.3 + i*0.125), 0.14, 0.12, 
                                     linewidth=1, edgecolor='black', facecolor=f'C{(i+j)%4}', alpha=0.5)
            ax.add_patch(rect)
    ax.text(0.5, 0.15, 'Step 3:\nDivide into\n4×4 = 16 Subregions', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Step 4: Gradients
    ax = axes[1, 0]
    for i in range(5):
        for j in range(5):
            angle = np.random.rand() * 360
            ax.arrow(0.15 + j*0.15, 0.35 + i*0.1, 0.05*np.cos(np.radians(angle)), 
                    0.05*np.sin(np.radians(angle)), head_width=0.02, fc='red', ec='red')
    ax.text(0.5, 0.15, 'Step 4:\nCompute Gradient\nDirections', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Step 5: 8-bin histogram
    ax = axes[1, 1]
    hist = np.random.rand(8) * 0.4
    ax.bar(np.linspace(0.15, 0.85, 8), hist + 0.3, width=0.08, color='blue', edgecolor='black')
    ax.text(0.5, 0.15, 'Step 5:\n8-bin Histogram\nper Subregion', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Step 6: 128-D
    ax = axes[1, 2]
    ax.bar(np.linspace(0.1, 0.9, 32), np.random.rand(32) * 0.4 + 0.35, width=0.02, color='green', edgecolor='none')
    ax.text(0.5, 0.7, '16 × 8 = 128', fontsize=16, ha='center', va='center', fontweight='bold', color='green')
    ax.text(0.5, 0.15, 'Step 6:\n128-D Descriptor\nVector', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.suptitle('SIFT Descriptor Pipeline: 6 Steps to Create 128-D Feature Vector', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../images/sift_desc_pipeline.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: sift_desc_pipeline.png')

def main():
    print("Loading image...")
    img = load_image()
    print(f"Image size: {img.shape}")
    
    # Select a keypoint location
    kp_x, kp_y = 320, 240  # Center of image
    kp_orientation = 45  # Example orientation
    
    print("\n=== Generating SIFT Descriptor Visualization Images ===\n")
    
    # Generate all step images
    print("Step 1: Keypoint with orientation...")
    plot_step1_keypoint_orientation(img, kp_x, kp_y, kp_orientation)
    
    print("Step 2: 16x16 region...")
    plot_step2_16x16_region(img, kp_x, kp_y, kp_orientation)
    
    print("Step 3: 4x4 subregions...")
    plot_step3_4x4_subregions(img, kp_x, kp_y)
    
    print("Step 4: Gradients in subregions...")
    plot_step4_gradients_in_subregion(img, kp_x, kp_y)
    
    print("Step 5: 8-bin histogram...")
    plot_step5_8bin_histogram()
    
    print("Step 6: 128-D descriptor...")
    plot_step6_128d_descriptor()
    
    print("Complete pipeline diagram...")
    plot_complete_descriptor_pipeline()
    
    print("\n=== All images generated! ===")

if __name__ == '__main__':
    main()

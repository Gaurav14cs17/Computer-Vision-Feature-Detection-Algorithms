"""
HOG Cell Histogram Visualization
Detailed visualizations for histogram construction step using real images
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')
os.makedirs(OUT_DIR, exist_ok=True)


def load_image():
    """Load the input image"""
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    img = np.array(Image.open(image_path))
    if len(img.shape) == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float64)
    gray = gray / 255.0
    # Gamma correction
    gray = np.power(np.clip(gray, 1e-8, 1), 0.5)
    return gray, img


def compute_gradients(img):
    """Compute gradients"""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) * 180 / np.pi % 180
    return magnitude, direction


def compute_cell_histogram(cell_mag, cell_dir, num_bins=9):
    """Compute histogram for a single cell"""
    bin_width = 180.0 / num_bins
    histogram = np.zeros(num_bins)
    
    for py in range(cell_mag.shape[0]):
        for px in range(cell_mag.shape[1]):
            mag = cell_mag[py, px]
            angle = cell_dir[py, px]
            
            bin_idx = angle / bin_width
            lower_bin = int(bin_idx) % num_bins
            upper_bin = (lower_bin + 1) % num_bins
            
            upper_weight = bin_idx - int(bin_idx)
            lower_weight = 1 - upper_weight
            
            histogram[lower_bin] += mag * lower_weight
            histogram[upper_bin] += mag * upper_weight
    
    return histogram


def create_cell_histogram_detail():
    """Detailed visualization of how a cell histogram is built from real image"""
    gray, original_rgb = load_image()
    magnitude, direction = compute_gradients(gray)
    
    # Select a cell from an interesting region (with edges)
    cell_size = 8
    # Find cell with high gradient activity
    h, w = gray.shape
    cells_y, cells_x = h // cell_size, w // cell_size
    
    best_cell = (cells_y // 2, cells_x // 2)
    best_mag = 0
    for cy in range(1, cells_y - 1):
        for cx in range(1, cells_x - 1):
            y_start = cy * cell_size
            x_start = cx * cell_size
            cell_mag = magnitude[y_start:y_start+cell_size, x_start:x_start+cell_size]
            if cell_mag.sum() > best_mag:
                best_mag = cell_mag.sum()
                best_cell = (cy, cx)
    
    cy, cx = best_cell
    y_start = cy * cell_size
    x_start = cx * cell_size
    
    cell_img = gray[y_start:y_start+cell_size, x_start:x_start+cell_size]
    cell_mag = magnitude[y_start:y_start+cell_size, x_start:x_start+cell_size]
    cell_dir = direction[y_start:y_start+cell_size, x_start:x_start+cell_size]
    
    histogram = compute_cell_histogram(cell_mag, cell_dir)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Panel 1: Full image with cell highlighted
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(gray, cmap='gray')
    rect = Rectangle((x_start, y_start), cell_size, cell_size, 
                     fill=False, edgecolor='red', linewidth=3)
    ax1.add_patch(rect)
    ax1.set_title(f'Full Image with Selected Cell\n(cell at {cx}, {cy})', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Cell with gradient arrows
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(cell_img, cmap='gray', extent=[0, cell_size, cell_size, 0])
    for py in range(cell_size):
        for px in range(cell_size):
            mag = cell_mag[py, px]
            angle = cell_dir[py, px]
            if mag > 0.01:
                dx = mag * 3 * np.cos(np.radians(angle))
                dy = mag * 3 * np.sin(np.radians(angle))
                ax2.arrow(px + 0.5, py + 0.5, dx, dy, head_width=0.2, head_length=0.1, 
                         fc='lime', ec='lime', linewidth=1)
    ax2.set_title('8×8 Cell with Gradient Vectors\n(64 pixels → 9 histogram bins)', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, cell_size)
    ax2.set_ylim(cell_size, 0)
    
    # Panel 3: Magnitude heatmap
    ax3 = fig.add_subplot(2, 3, 3)
    im = ax3.imshow(cell_mag, cmap='hot', extent=[0, cell_size, cell_size, 0])
    ax3.set_title('Gradient Magnitude\n(voting weight)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax3, shrink=0.8)
    ax3.set_xlim(0, cell_size)
    ax3.set_ylim(cell_size, 0)
    
    # Panel 4: Direction heatmap
    ax4 = fig.add_subplot(2, 3, 4)
    im = ax4.imshow(cell_dir, cmap='hsv', vmin=0, vmax=180, extent=[0, cell_size, cell_size, 0])
    ax4.set_title('Gradient Direction (0°-180°)\n(determines which bin)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax4, shrink=0.8, label='degrees')
    ax4.set_xlim(0, cell_size)
    ax4.set_ylim(cell_size, 0)
    
    # Panel 5: Resulting histogram
    ax5 = fig.add_subplot(2, 3, 5)
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    bars = ax5.bar(range(9), histogram, color=colors, edgecolor='black', linewidth=1.5)
    ax5.set_xlabel('Orientation Bin (20° each)', fontsize=11)
    ax5.set_ylabel('Accumulated Magnitude', fontsize=11)
    ax5.set_xticks(range(9))
    ax5.set_xticklabels([f'{i*20}°-{(i+1)*20}°' for i in range(9)], fontsize=8, rotation=45)
    ax5.set_title('Resulting 9-Bin Histogram\n(sum of all 64 pixel votes)', fontsize=11, fontweight='bold')
    
    peak_idx = np.argmax(histogram)
    bars[peak_idx].set_edgecolor('red')
    bars[peak_idx].set_linewidth(3)
    ax5.axhline(y=histogram[peak_idx], color='red', linestyle='--', alpha=0.5)
    
    # Panel 6: HOG visualization of single cell
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_xlim(-1, 1)
    ax6.set_ylim(-1, 1)
    ax6.set_facecolor('black')
    
    max_hist = histogram.max() if histogram.max() > 0 else 1
    for bin_idx in range(9):
        if histogram[bin_idx] > max_hist * 0.1:
            angle = bin_idx * 20 + 10
            scale = (histogram[bin_idx] / max_hist) * 0.8
            dx = scale * np.cos(np.radians(angle))
            dy = scale * np.sin(np.radians(angle))
            ax6.plot([-dx, dx], [-dy, dy], color='white', linewidth=2 + scale * 2, alpha=0.8)
    
    ax6.set_title('HOG Visualization of This Cell\n(line length ∝ bin magnitude)', fontsize=11, fontweight='bold')
    ax6.set_aspect('equal')
    
    plt.suptitle('HOG Step 3: Cell Histogram Construction (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_cell_histogram_detail.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_cell_histogram_detail.png")


def create_interpolation_detail():
    """Detailed visualization of bilinear interpolation"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    examples = [
        {'angle': 35, 'mag': 0.4},
        {'angle': 70, 'mag': 0.5},
        {'angle': 160, 'mag': 0.3},
    ]
    
    bin_width = 20
    
    for ax, ex in zip(axes, examples):
        angle = ex['angle']
        mag = ex['mag']
        
        bin_float = angle / bin_width
        lower_bin = int(bin_float)
        upper_bin = (lower_bin + 1) % 9
        
        upper_weight = bin_float - int(bin_float)
        lower_weight = 1 - upper_weight
        
        hist = np.zeros(9)
        hist[lower_bin] += mag * lower_weight
        hist[upper_bin] += mag * upper_weight
        
        colors = ['lightgray'] * 9
        colors[lower_bin] = 'lightblue'
        if upper_bin < 9:
            colors[upper_bin] = 'lightgreen'
        
        bars = ax.bar(range(9), hist, color=colors, edgecolor='black', linewidth=1.5)
        bars[lower_bin].set_edgecolor('blue')
        bars[lower_bin].set_linewidth(2)
        if upper_bin < 9:
            bars[upper_bin].set_edgecolor('green')
            bars[upper_bin].set_linewidth(2)
        
        ax.set_xlabel('Bin', fontsize=10)
        ax.set_ylabel('Vote', fontsize=10)
        ax.set_xticks(range(9))
        ax.set_xticklabels([f'{i*20}°' for i in range(9)], fontsize=8)
        ax.set_ylim(0, mag * 1.3)
        
        ax.set_title(f'θ = {angle}°, M = {mag}', fontsize=12, fontweight='bold')
        
        vote_lower = mag * lower_weight
        vote_upper = mag * upper_weight
        
        ax.text(lower_bin, vote_lower + 0.02, f'{vote_lower:.3f}\n(w={lower_weight:.2f})', 
               fontsize=9, ha='center', color='blue', fontweight='bold')
        if upper_bin < 9 and vote_upper > 0.001:
            ax.text(upper_bin, vote_upper + 0.02, f'{vote_upper:.3f}\n(w={upper_weight:.2f})', 
                   fontsize=9, ha='center', color='green', fontweight='bold')
    
    plt.suptitle('Bilinear Interpolation: How Gradient Votes Are Split Between Bins', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_interpolation_detail.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_interpolation_detail.png")


def create_multiple_cells_visual():
    """Show histograms for multiple cells from real image"""
    gray, _ = load_image()
    magnitude, direction = compute_gradients(gray)
    
    cell_size = 8
    h, w = gray.shape
    cells_y, cells_x = h // cell_size, w // cell_size
    
    # Select 4 cells from different regions
    cell_positions = [
        (cells_y // 4, cells_x // 4, 'Top-Left'),
        (cells_y // 4, 3 * cells_x // 4, 'Top-Right'),
        (3 * cells_y // 4, cells_x // 4, 'Bottom-Left'),
        (3 * cells_y // 4, 3 * cells_x // 4, 'Bottom-Right'),
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for idx, (cy, cx, name) in enumerate(cell_positions):
        y_start = cy * cell_size
        x_start = cx * cell_size
        
        cell_img = gray[y_start:y_start+cell_size, x_start:x_start+cell_size]
        cell_mag = magnitude[y_start:y_start+cell_size, x_start:x_start+cell_size]
        cell_dir = direction[y_start:y_start+cell_size, x_start:x_start+cell_size]
        
        histogram = compute_cell_histogram(cell_mag, cell_dir)
        
        # Top row: cell image with gradients
        ax = axes[0, idx]
        ax.imshow(cell_img, cmap='gray', extent=[0, cell_size, cell_size, 0])
        for py in range(0, cell_size, 2):
            for px in range(0, cell_size, 2):
                mag = cell_mag[py, px]
                angle = cell_dir[py, px]
                if mag > 0.02:
                    dx = 0.8 * np.cos(np.radians(angle))
                    dy = 0.8 * np.sin(np.radians(angle))
                    ax.arrow(px + 0.5, py + 0.5, dx, dy, head_width=0.15, 
                            head_length=0.08, fc='lime', ec='lime', linewidth=0.8)
        ax.set_title(f'{name} Cell', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Bottom row: histogram
        ax = axes[1, idx]
        colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
        bars = ax.bar(range(9), histogram, color=colors, edgecolor='black')
        ax.set_xlabel('Bin', fontsize=9)
        ax.set_xticks(range(9))
        ax.set_xticklabels([f'{i*20}°' for i in range(9)], fontsize=7, rotation=45)
        
        peak = np.argmax(histogram)
        bars[peak].set_edgecolor('red')
        bars[peak].set_linewidth(2)
    
    plt.suptitle('Cell Histograms from Different Image Regions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_multiple_cells.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_multiple_cells.png")


if __name__ == "__main__":
    print("Generating HOG Cell Histogram Visualizations...")
    print("-" * 50)
    create_cell_histogram_detail()
    create_interpolation_detail()
    create_multiple_cells_visual()
    print("-" * 50)
    print("Done!")

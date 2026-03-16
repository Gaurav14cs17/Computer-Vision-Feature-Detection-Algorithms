"""
HOG Step 3 Detailed Visualization
Detailed step-by-step cell histogram visualization like SIFT step3_pyramid.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
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
    return magnitude, direction, gx, gy


def create_step3_1_cell_division():
    """Step 3.1: Show how image is divided into 8×8 cells"""
    gray, original_rgb = load_image()
    h, w = gray.shape
    cell_size = 8
    cells_y = h // cell_size
    cells_x = w // cell_size
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original image
    ax = axes[0]
    ax.imshow(original_rgb)
    ax.set_title(f'Original Image\n{w}×{h} pixels', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Image with cell grid
    ax = axes[1]
    ax.imshow(gray, cmap='gray')
    
    # Draw cell grid
    for cy in range(cells_y + 1):
        ax.axhline(y=cy * cell_size, color='lime', linewidth=0.5, alpha=0.8)
    for cx in range(cells_x + 1):
        ax.axvline(x=cx * cell_size, color='lime', linewidth=0.5, alpha=0.8)
    
    # Highlight a few cells with different colors
    highlight_cells = [(10, 10), (20, 30), (40, 50), (30, 70)]
    colors = ['red', 'blue', 'orange', 'purple']
    for (cy, cx), color in zip(highlight_cells, colors):
        if cy < cells_y and cx < cells_x:
            rect = Rectangle((cx * cell_size, cy * cell_size), cell_size, cell_size,
                             fill=False, edgecolor=color, linewidth=3)
            ax.add_patch(rect)
    
    ax.set_title(f'Image Divided into {cells_x}×{cells_y} = {cells_x*cells_y} Cells\nEach cell: 8×8 pixels', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 3.1: Cell Division', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step3_1_cell_division.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step3_1_cell_division.png")


def create_step3_2_single_cell_detail():
    """Step 3.2: Detailed view of a single 8×8 cell"""
    gray, original_rgb = load_image()
    magnitude, direction, gx, gy = compute_gradients(gray)
    
    # Select a cell with interesting content
    cell_size = 8
    h, w = gray.shape
    cells_y, cells_x = h // cell_size, w // cell_size
    
    # Find cell with high gradient activity
    best_cell = (cells_y // 2, cells_x // 2)
    best_mag = 0
    for cy in range(5, cells_y - 5):
        for cx in range(5, cells_x - 5):
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
    
    fig = plt.figure(figsize=(18, 10))
    
    # Panel 1: Full image with cell highlighted
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(gray, cmap='gray')
    rect = Rectangle((x_start, y_start), cell_size, cell_size, 
                     fill=False, edgecolor='red', linewidth=3)
    ax1.add_patch(rect)
    ax1.set_title(f'Full Image with Selected Cell\nCell ({cx}, {cy})', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Zoomed cell with pixel values
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(cell_img, cmap='gray', extent=[0, cell_size, cell_size, 0])
    for py in range(cell_size):
        for px in range(cell_size):
            val = cell_img[py, px]
            color = 'white' if val < 0.5 else 'black'
            ax2.text(px + 0.5, py + 0.5, f'{val:.2f}', ha='center', va='center', 
                    fontsize=7, color=color)
    ax2.set_title('8×8 Cell Pixel Values\n(64 intensity values)', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, cell_size)
    ax2.set_ylim(cell_size, 0)
    
    # Panel 3: Cell with gradient vectors
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(cell_img, cmap='gray', extent=[0, cell_size, cell_size, 0])
    for py in range(cell_size):
        for px in range(cell_size):
            mag = cell_mag[py, px]
            angle = cell_dir[py, px]
            if mag > 0.01:
                dx = mag * 3 * np.cos(np.radians(angle))
                dy = mag * 3 * np.sin(np.radians(angle))
                ax3.arrow(px + 0.5, py + 0.5, dx, dy, head_width=0.15, head_length=0.08, 
                         fc='lime', ec='lime', linewidth=0.8)
    ax3.set_title('Gradient Vectors (64 gradients)\nArrow = direction, length = magnitude', fontsize=11, fontweight='bold')
    ax3.set_xlim(0, cell_size)
    ax3.set_ylim(cell_size, 0)
    
    # Panel 4: Magnitude heatmap
    ax4 = fig.add_subplot(2, 3, 4)
    im = ax4.imshow(cell_mag, cmap='hot', extent=[0, cell_size, cell_size, 0])
    for py in range(cell_size):
        for px in range(cell_size):
            val = cell_mag[py, px]
            color = 'white' if val < 0.15 else 'black'
            ax4.text(px + 0.5, py + 0.5, f'{val:.2f}', ha='center', va='center', 
                    fontsize=6, color=color)
    ax4.set_title('Gradient Magnitude (M)\nM = √(Gx² + Gy²)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax4, shrink=0.8)
    ax4.set_xlim(0, cell_size)
    ax4.set_ylim(cell_size, 0)
    
    # Panel 5: Direction heatmap
    ax5 = fig.add_subplot(2, 3, 5)
    im = ax5.imshow(cell_dir, cmap='hsv', vmin=0, vmax=180, extent=[0, cell_size, cell_size, 0])
    for py in range(cell_size):
        for px in range(cell_size):
            val = cell_dir[py, px]
            ax5.text(px + 0.5, py + 0.5, f'{val:.0f}°', ha='center', va='center', 
                    fontsize=6, color='black')
    ax5.set_title('Gradient Direction (θ)\nθ = arctan(Gy/Gx) mod 180°', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax5, shrink=0.8, label='degrees')
    ax5.set_xlim(0, cell_size)
    ax5.set_ylim(cell_size, 0)
    
    # Panel 6: Summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary = f"""
    Single Cell Analysis
    ━━━━━━━━━━━━━━━━━━━━━
    
    Cell Position: ({cx}, {cy})
    Cell Size: 8 × 8 = 64 pixels
    
    For each pixel we have:
      • Intensity I(x,y)
      • Gradient Gx, Gy
      • Magnitude M = √(Gx² + Gy²)
      • Direction θ = arctan(Gy/Gx) mod 180°
    
    Total gradients: 64
    
    Next Step:
      64 gradients → 9-bin histogram
      (bin based on direction,
       weighted by magnitude)
    """
    ax6.text(0.1, 0.5, summary, fontsize=11, family='monospace', va='center',
            transform=ax6.transAxes, bbox=dict(facecolor='lightyellow', edgecolor='orange', linewidth=2))
    ax6.set_title('Summary', fontsize=11, fontweight='bold')
    
    plt.suptitle('HOG Step 3.2: Single Cell Analysis (8×8 pixels → 64 gradients)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step3_2_single_cell.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step3_2_single_cell.png")


def create_step3_3_histogram_bins():
    """Step 3.3: 9-bin histogram structure"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Bin structure diagram
    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw 9 bins
    bin_width = 1.0
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    
    for i in range(9):
        x = 1.5 + i * bin_width
        rect = Rectangle((x, 3), bin_width * 0.9, 4, facecolor=colors[i], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + bin_width/2, 2.5, f'Bin {i}', ha='center', fontsize=9, fontweight='bold')
        ax.text(x + bin_width/2, 7.5, f'{i*20}°-{(i+1)*20}°', ha='center', fontsize=8)
    
    ax.text(6, 9, '9 Bins Covering 0° to 180°', fontsize=14, ha='center', fontweight='bold')
    ax.text(6, 1, 'Bin Width = 180° / 9 = 20° each', fontsize=12, ha='center',
            bbox=dict(facecolor='lightyellow', edgecolor='black'))
    
    ax.set_title('9-Bin Histogram Structure', fontsize=12, fontweight='bold')
    
    # Panel 2: Angle to bin mapping
    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Example mappings
    examples = [
        (15, 0, 1, 0.75, 0.25),
        (45, 2, 3, 0.75, 0.25),
        (90, 4, 5, 0.5, 0.5),
        (170, 8, 0, 0.5, 0.5),  # wraps around
    ]
    
    ax.text(6, 9.3, 'Angle → Bin Mapping (Bilinear Interpolation)', fontsize=13, ha='center', fontweight='bold')
    
    y_pos = 7.5
    for angle, bin1, bin2, w1, w2 in examples:
        ax.text(1, y_pos, f'θ = {angle}°:', fontsize=11, fontweight='bold')
        ax.text(4, y_pos, f'→ Bin {bin1} (weight {w1:.2f})', fontsize=10, color='blue')
        ax.text(8, y_pos, f'+ Bin {bin2} (weight {w2:.2f})', fontsize=10, color='green')
        y_pos -= 1.5
    
    ax.text(6, 1.5, 'Vote = Magnitude × Weight', fontsize=12, ha='center',
            bbox=dict(facecolor='lightgreen', edgecolor='black'))
    
    ax.set_title('Bilinear Interpolation', fontsize=12, fontweight='bold')
    
    plt.suptitle('HOG Step 3.3: 9-Bin Histogram (0° to 180°)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step3_3_histogram_bins.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step3_3_histogram_bins.png")


def create_step3_4_voting_process():
    """Step 3.4: Show the voting process for one cell"""
    gray, _ = load_image()
    magnitude, direction, _, _ = compute_gradients(gray)
    
    cell_size = 8
    cells_y, cells_x = gray.shape[0] // cell_size, gray.shape[1] // cell_size
    
    # Find a good cell
    cy, cx = cells_y // 2, cells_x // 2
    y_start = cy * cell_size
    x_start = cx * cell_size
    
    cell_mag = magnitude[y_start:y_start+cell_size, x_start:x_start+cell_size]
    cell_dir = direction[y_start:y_start+cell_size, x_start:x_start+cell_size]
    
    # Compute histogram step by step
    bin_width = 20
    histogram = np.zeros(9)
    vote_details = []
    
    for py in range(cell_size):
        for px in range(cell_size):
            mag = cell_mag[py, px]
            angle = cell_dir[py, px]
            
            bin_idx = angle / bin_width
            lower_bin = int(bin_idx) % 9
            upper_bin = (lower_bin + 1) % 9
            
            upper_weight = bin_idx - int(bin_idx)
            lower_weight = 1 - upper_weight
            
            histogram[lower_bin] += mag * lower_weight
            histogram[upper_bin] += mag * upper_weight
            
            if mag > 0.05 and len(vote_details) < 8:
                vote_details.append((py, px, mag, angle, lower_bin, upper_bin, lower_weight, upper_weight))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Cell gradients
    ax = axes[0, 0]
    ax.imshow(cell_mag, cmap='hot', extent=[0, cell_size, cell_size, 0])
    for py in range(cell_size):
        for px in range(cell_size):
            mag = cell_mag[py, px]
            angle = cell_dir[py, px]
            if mag > 0.01:
                dx = mag * 2.5 * np.cos(np.radians(angle))
                dy = mag * 2.5 * np.sin(np.radians(angle))
                ax.arrow(px + 0.5, py + 0.5, dx, dy, head_width=0.15, head_length=0.08, 
                        fc='lime', ec='lime', linewidth=0.8)
    ax.set_title('Cell Gradients\n(64 pixels voting)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, cell_size)
    ax.set_ylim(cell_size, 0)
    
    # Panel 2: Voting examples
    ax = axes[0, 1]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    ax.text(5, 9.5, 'Sample Pixel Votes', fontsize=13, ha='center', fontweight='bold')
    
    y_pos = 8.5
    for py, px, mag, angle, lb, ub, lw, uw in vote_details[:6]:
        ax.text(0.5, y_pos, f'Pixel ({px},{py}): M={mag:.3f}, θ={angle:.0f}°', fontsize=9)
        ax.text(5.5, y_pos, f'→ Bin{lb}+={mag*lw:.3f}, Bin{ub}+={mag*uw:.3f}', fontsize=9, color='blue')
        y_pos -= 1.2
    
    ax.text(5, 1, '... (all 64 pixels vote similarly)', fontsize=10, ha='center', style='italic')
    ax.set_title('Voting Process', fontsize=11, fontweight='bold')
    
    # Panel 3: Resulting histogram
    ax = axes[1, 0]
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    bars = ax.bar(range(9), histogram, color=colors, edgecolor='black', linewidth=1.5)
    
    for i, h in enumerate(histogram):
        ax.text(i, h + 0.02, f'{h:.2f}', ha='center', fontsize=9)
    
    ax.set_xlabel('Orientation Bin', fontsize=11)
    ax.set_ylabel('Accumulated Magnitude', fontsize=11)
    ax.set_xticks(range(9))
    ax.set_xticklabels([f'{i*20}°-{(i+1)*20}°' for i in range(9)], fontsize=8, rotation=45)
    
    peak_idx = np.argmax(histogram)
    bars[peak_idx].set_edgecolor('red')
    bars[peak_idx].set_linewidth(3)
    
    ax.set_title(f'Resulting 9-Bin Histogram\nDominant orientation: {peak_idx*20}°-{(peak_idx+1)*20}°', 
                fontsize=11, fontweight='bold')
    
    # Panel 4: HOG visualization
    ax = axes[1, 1]
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_facecolor('black')
    
    max_hist = histogram.max() if histogram.max() > 0 else 1
    for bin_idx in range(9):
        if histogram[bin_idx] > max_hist * 0.1:
            angle = bin_idx * 20 + 10
            scale = (histogram[bin_idx] / max_hist) * 0.8
            dx = scale * np.cos(np.radians(angle))
            dy = scale * np.sin(np.radians(angle))
            lw = 2 + scale * 3
            ax.plot([-dx, dx], [-dy, dy], color='white', linewidth=lw, alpha=0.8)
    
    ax.set_title('HOG Visualization of This Cell\n(line = dominant orientation)', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.suptitle('HOG Step 3.4: Voting Process (64 pixels → 9-bin histogram)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step3_4_voting.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step3_4_voting.png")


def create_step3_5_all_cells():
    """Step 3.5: Show histograms for all cells"""
    gray, original_rgb = load_image()
    magnitude, direction, _, _ = compute_gradients(gray)
    
    cell_size = 8
    h, w = gray.shape
    cells_y = h // cell_size
    cells_x = w // cell_size
    
    # Compute all cell histograms
    histograms = np.zeros((cells_y, cells_x, 9))
    bin_width = 20
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            y_start = cy * cell_size
            x_start = cx * cell_size
            cell_mag = magnitude[y_start:y_start+cell_size, x_start:x_start+cell_size]
            cell_dir = direction[y_start:y_start+cell_size, x_start:x_start+cell_size]
            
            for py in range(cell_size):
                for px in range(cell_size):
                    mag = cell_mag[py, px]
                    angle = cell_dir[py, px]
                    bin_idx = angle / bin_width
                    lower_bin = int(bin_idx) % 9
                    upper_bin = (lower_bin + 1) % 9
                    upper_weight = bin_idx - int(bin_idx)
                    lower_weight = 1 - upper_weight
                    histograms[cy, cx, lower_bin] += mag * lower_weight
                    histograms[cy, cx, upper_bin] += mag * upper_weight
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel 1: Original with cell grid and HOG
    ax = axes[0]
    ax.imshow(gray, cmap='gray', alpha=0.3)
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = cx * cell_size + cell_size / 2
            
            max_val = hist.max() if hist.max() > 0 else 1
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.2:
                    angle = bin_idx * 20 + 10
                    scale = (hist[bin_idx] / max_val) * (cell_size / 2.5)
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax.plot([center_x - dx, center_x + dx],
                           [center_y - dy, center_y + dy],
                           color='red', linewidth=0.8, alpha=0.8)
    
    ax.set_title(f'HOG Visualization on Image\n{cells_x}×{cells_y} = {cells_x*cells_y} cells', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Panel 2: HOG only (black background)
    ax = axes[1]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_facecolor('black')
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = cx * cell_size + cell_size / 2
            
            max_val = hist.max() if hist.max() > 0 else 1
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.15:
                    angle = bin_idx * 20 + 10
                    scale = (hist[bin_idx] / max_val) * (cell_size / 2) * 0.9
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    alpha = 0.5 + 0.5 * (hist[bin_idx] / max_val)
                    ax.plot([center_x - dx, center_x + dx],
                           [center_y - dy, center_y + dy],
                           color='white', linewidth=0.8, alpha=alpha)
    
    ax.set_title('HOG Features Only\n(edge orientations per cell)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.suptitle(f'HOG Step 3.5: All Cell Histograms ({cells_x*cells_y} cells × 9 bins = {cells_x*cells_y*9} values)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step3_5_all_cells.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step3_5_all_cells.png")


def create_step3_6_compression():
    """Step 3.6: Show the compression from pixels to histograms"""
    gray, _ = load_image()
    h, w = gray.shape
    cell_size = 8
    cells_y = h // cell_size
    cells_x = w // cell_size
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    
    # Title
    ax.text(7, 9.5, 'HOG Step 3.6: Data Compression', fontsize=16, ha='center', fontweight='bold')
    
    # Input
    ax.text(2, 8, 'INPUT:', fontsize=12, fontweight='bold', color='blue')
    ax.text(2, 7.2, f'Image: {w} × {h} = {w*h:,} pixels', fontsize=11)
    ax.text(2, 6.6, f'Each pixel: 2 values (M, θ)', fontsize=11)
    ax.text(2, 6.0, f'Total: {w*h*2:,} gradient values', fontsize=11, fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(7, 5.5), xytext=(7, 7),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    ax.text(8, 6.2, 'Cell Histograms\n(8×8 → 9 bins)', fontsize=10, ha='left')
    
    # Output
    ax.text(2, 4.5, 'OUTPUT:', fontsize=12, fontweight='bold', color='red')
    ax.text(2, 3.7, f'Cells: {cells_x} × {cells_y} = {cells_x*cells_y} cells', fontsize=11)
    ax.text(2, 3.1, f'Each cell: 9-bin histogram', fontsize=11)
    ax.text(2, 2.5, f'Total: {cells_x*cells_y*9:,} histogram values', fontsize=11, fontweight='bold')
    
    # Compression ratio
    input_vals = w * h * 2
    output_vals = cells_x * cells_y * 9
    ratio = input_vals / output_vals
    
    ax.text(7, 1.5, f'Compression Ratio: {input_vals:,} → {output_vals:,}', fontsize=12, ha='center')
    ax.text(7, 0.8, f'≈ {ratio:.1f}:1 compression', fontsize=14, ha='center', fontweight='bold',
            bbox=dict(facecolor='lightgreen', edgecolor='green', linewidth=2))
    
    # Visual representation
    ax.add_patch(Rectangle((9.5, 6.5), 3, 2, facecolor='lightblue', edgecolor='blue', linewidth=2))
    ax.text(11, 7.5, 'Pixels\n(many)', fontsize=10, ha='center')
    
    ax.add_patch(Rectangle((9.5, 3), 1.5, 1, facecolor='lightcoral', edgecolor='red', linewidth=2))
    ax.text(10.25, 3.5, 'Hist\n(few)', fontsize=10, ha='center')
    
    ax.annotate('', xy=(10.25, 4.2), xytext=(11, 6.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    plt.savefig(os.path.join(OUT_DIR, 'hog_step3_6_compression.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step3_6_compression.png")


def create_step3_complete_summary():
    """Step 3 complete summary"""
    gray, original_rgb = load_image()
    magnitude, direction, gx, gy = compute_gradients(gray)
    
    cell_size = 8
    h, w = gray.shape
    cells_y = h // cell_size
    cells_x = w // cell_size
    
    # Compute histograms
    histograms = np.zeros((cells_y, cells_x, 9))
    bin_width = 20
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            y_start = cy * cell_size
            x_start = cx * cell_size
            cell_mag = magnitude[y_start:y_start+cell_size, x_start:x_start+cell_size]
            cell_dir = direction[y_start:y_start+cell_size, x_start:x_start+cell_size]
            
            for py in range(cell_size):
                for px in range(cell_size):
                    mag = cell_mag[py, px]
                    angle = cell_dir[py, px]
                    bin_idx = angle / bin_width
                    lower_bin = int(bin_idx) % 9
                    upper_bin = (lower_bin + 1) % 9
                    upper_weight = bin_idx - int(bin_idx)
                    lower_weight = 1 - upper_weight
                    histograms[cy, cx, lower_bin] += mag * lower_weight
                    histograms[cy, cx, upper_bin] += mag * upper_weight
    
    fig = plt.figure(figsize=(20, 10))
    
    # Panel 1: Original
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(original_rgb)
    ax1.set_title(f'Original Image\n{w}×{h}', fontsize=10, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Gradient Magnitude
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(magnitude, cmap='hot')
    ax2.set_title('Gradient Magnitude', fontsize=10, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: Gradient Direction
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(direction, cmap='hsv', vmin=0, vmax=180)
    ax3.set_title('Gradient Direction', fontsize=10, fontweight='bold')
    ax3.axis('off')
    
    # Panel 4: Cell grid
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(gray, cmap='gray')
    for cy in range(cells_y + 1):
        ax4.axhline(y=cy * cell_size, color='lime', linewidth=0.3)
    for cx in range(cells_x + 1):
        ax4.axvline(x=cx * cell_size, color='lime', linewidth=0.3)
    ax4.set_title(f'{cells_x}×{cells_y}={cells_x*cells_y} Cells', fontsize=10, fontweight='bold')
    ax4.axis('off')
    
    # Panel 5: Sample histogram
    ax5 = fig.add_subplot(2, 4, 5)
    sample_hist = histograms[cells_y//2, cells_x//2]
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    ax5.bar(range(9), sample_hist, color=colors, edgecolor='black')
    ax5.set_title('Sample Cell Histogram', fontsize=10, fontweight='bold')
    ax5.set_xlabel('Bin')
    ax5.set_xticks(range(9))
    ax5.set_xticklabels([f'{i*20}°' for i in range(9)], fontsize=7)
    
    # Panel 6: HOG overlay
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.imshow(gray, cmap='gray', alpha=0.3)
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = cx * cell_size + cell_size / 2
            max_val = hist.max() if hist.max() > 0 else 1
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.2:
                    angle = bin_idx * 20 + 10
                    scale = (hist[bin_idx] / max_val) * (cell_size / 2.5)
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax6.plot([center_x - dx, center_x + dx], [center_y - dy, center_y + dy],
                            color='red', linewidth=0.5, alpha=0.8)
    ax6.set_title('HOG on Image', fontsize=10, fontweight='bold')
    ax6.axis('off')
    
    # Panel 7: HOG only
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.set_xlim(0, w)
    ax7.set_ylim(h, 0)
    ax7.set_facecolor('black')
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = cx * cell_size + cell_size / 2
            max_val = hist.max() if hist.max() > 0 else 1
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.15:
                    angle = bin_idx * 20 + 10
                    scale = (hist[bin_idx] / max_val) * (cell_size / 2) * 0.9
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax7.plot([center_x - dx, center_x + dx], [center_y - dy, center_y + dy],
                            color='white', linewidth=0.5, alpha=0.8)
    ax7.set_title('HOG Features', fontsize=10, fontweight='bold')
    ax7.set_aspect('equal')
    
    # Panel 8: Statistics
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    stats = f"""
    Step 3 Summary
    ━━━━━━━━━━━━━━
    
    Input:
      {w}×{h} pixels
      {w*h*2:,} gradient values
    
    Output:
      {cells_x}×{cells_y} cells
      9 bins per cell
      {cells_x*cells_y*9:,} values
    
    Ready for Step 4:
      Block Normalization
    """
    ax8.text(0.1, 0.5, stats, fontsize=10, family='monospace', va='center',
            transform=ax8.transAxes, bbox=dict(facecolor='lightyellow', edgecolor='orange'))
    ax8.set_title('Statistics', fontsize=10, fontweight='bold')
    
    plt.suptitle('HOG Step 3: Cell Histograms - Complete Pipeline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step3_complete.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step3_complete.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating HOG Step 3 Detailed Visualizations")
    print("=" * 60)
    
    create_step3_1_cell_division()
    create_step3_2_single_cell_detail()
    create_step3_3_histogram_bins()
    create_step3_4_voting_process()
    create_step3_5_all_cells()
    create_step3_6_compression()
    create_step3_complete_summary()
    
    print("=" * 60)
    print("Done!")
    print("=" * 60)

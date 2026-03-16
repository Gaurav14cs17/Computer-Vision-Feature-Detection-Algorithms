"""
HOG Algorithm Pipeline - Step by Step
Histogram of Oriented Gradients for object detection

Uses real input image for visualization
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
    """Load and preprocess the input image"""
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    
    if not os.path.exists(image_path):
        print("Creating test image...")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :] = [100, 100, 100]
        # Add some features
        img[100:200, 100:200] = [200, 200, 200]
        img[200:350, 300:450] = [150, 150, 150]
        img[50:100, 400:500] = [220, 220, 220]
        Image.fromarray(img).save(image_path)
    
    img = np.array(Image.open(image_path))
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float64)
    
    # Normalize to [0, 1]
    gray = gray / 255.0
    
    return gray, img


def gamma_correction(img, gamma=0.5):
    """Apply gamma correction: I_corrected = I^gamma"""
    return np.power(np.clip(img, 1e-8, 1), gamma)


def compute_gradients(img):
    """
    Compute gradient magnitude and direction.
    
    Gx = I(x+1, y) - I(x-1, y)
    Gy = I(x, y+1) - I(x, y-1)
    
    Magnitude = sqrt(Gx^2 + Gy^2)
    Direction = arctan(Gy/Gx) in range [0, 180)
    """
    print("[HOG Step 2] Computing gradients...")
    
    h, w = img.shape
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    
    # Central difference
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    
    # Magnitude
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # Direction (unsigned: 0-180 degrees)
    direction = np.arctan2(gy, gx) * 180 / np.pi
    direction = direction % 180
    
    print(f"         Gradient magnitude range: [{magnitude.min():.4f}, {magnitude.max():.4f}]")
    return magnitude, direction, gx, gy


def compute_cell_histograms(magnitude, direction, cell_size=8, num_bins=9):
    """
    Compute histogram of oriented gradients for each cell.
    """
    print("[HOG Step 3] Computing cell histograms...")
    
    h, w = magnitude.shape
    cells_y = h // cell_size
    cells_x = w // cell_size
    
    bin_width = 180.0 / num_bins
    
    histograms = np.zeros((cells_y, cells_x, num_bins))
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            y_start = cy * cell_size
            y_end = y_start + cell_size
            x_start = cx * cell_size
            x_end = x_start + cell_size
            
            cell_mag = magnitude[y_start:y_end, x_start:x_end]
            cell_dir = direction[y_start:y_end, x_start:x_end]
            
            hist = np.zeros(num_bins)
            
            for py in range(cell_size):
                for px in range(cell_size):
                    mag = cell_mag[py, px]
                    angle = cell_dir[py, px]
                    
                    bin_idx = angle / bin_width
                    lower_bin = int(bin_idx) % num_bins
                    upper_bin = (lower_bin + 1) % num_bins
                    
                    upper_weight = bin_idx - int(bin_idx)
                    lower_weight = 1 - upper_weight
                    
                    hist[lower_bin] += mag * lower_weight
                    hist[upper_bin] += mag * upper_weight
            
            histograms[cy, cx, :] = hist
    
    print(f"         Created {cells_y}×{cells_x} = {cells_y * cells_x} cells")
    return histograms


def block_normalize(histograms, block_size=2, eps=1e-6):
    """
    Normalize histograms in overlapping blocks using L2-norm.
    """
    print("[HOG Step 4] Block normalization...")
    
    cells_y, cells_x, num_bins = histograms.shape
    
    blocks_y = cells_y - block_size + 1
    blocks_x = cells_x - block_size + 1
    
    block_features = []
    
    for by in range(blocks_y):
        for bx in range(blocks_x):
            block = histograms[by:by+block_size, bx:bx+block_size, :].flatten()
            norm = np.sqrt(np.sum(block**2) + eps**2)
            block_normalized = block / norm
            block_features.append(block_normalized)
    
    print(f"         Created {blocks_y}×{blocks_x} = {blocks_y * blocks_x} blocks")
    return np.array(block_features), (blocks_y, blocks_x)


def extract_hog_descriptor(img, cell_size=8, block_size=2, num_bins=9):
    """Extract complete HOG descriptor from image."""
    magnitude, direction, gx, gy = compute_gradients(img)
    histograms = compute_cell_histograms(magnitude, direction, cell_size, num_bins)
    block_features, block_shape = block_normalize(histograms, block_size)
    descriptor = block_features.flatten()
    
    return {
        'descriptor': descriptor,
        'magnitude': magnitude,
        'direction': direction,
        'gx': gx,
        'gy': gy,
        'histograms': histograms,
        'block_features': block_features,
        'block_shape': block_shape
    }


def visualize_step1_preprocessing(original_rgb, gray, gray_gamma, filename):
    """Visualize preprocessing step with real image"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    ax.set_title(f'Original RGB Image\n{original_rgb.shape[1]}×{original_rgb.shape[0]} pixels', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    ax.imshow(gray, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Grayscale Conversion\nI = 0.299R + 0.587G + 0.114B', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[2]
    ax.imshow(gray_gamma, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Gamma Correction (γ=0.5)\nI_γ = I^0.5', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 1: Preprocessing', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"         Saved: {filename}")


def visualize_step2_gradients(img, magnitude, direction, gx, gy, filename):
    """Visualize gradient computation with real image"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    ax = axes[0, 0]
    ax.imshow(img, cmap='gray')
    ax.set_title('Input Image', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = axes[0, 1]
    im = ax.imshow(gx, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title('Gx (Horizontal Gradient)\nGx = I(x+1,y) - I(x-1,y)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[0, 2]
    im = ax.imshow(gy, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title('Gy (Vertical Gradient)\nGy = I(x,y+1) - I(x,y-1)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[1, 0]
    im = ax.imshow(magnitude, cmap='hot')
    ax.set_title('Gradient Magnitude\nM = √(Gx² + Gy²)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[1, 1]
    im = ax.imshow(direction, cmap='hsv', vmin=0, vmax=180)
    ax.set_title('Gradient Direction (0°-180°)\nθ = arctan(Gy/Gx) mod 180°', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7, label='degrees')
    
    # Gradient vectors overlay
    ax = axes[1, 2]
    ax.imshow(img, cmap='gray', alpha=0.5)
    step = max(img.shape[0] // 30, 8)
    for y in range(step, img.shape[0] - step, step):
        for x in range(step, img.shape[1] - step, step):
            mag = magnitude[y, x]
            angle = direction[y, x]
            if mag > 0.03:
                dx = mag * 15 * np.cos(np.radians(angle))
                dy = mag * 15 * np.sin(np.radians(angle))
                ax.arrow(x, y, dx, dy, head_width=2, head_length=1, fc='lime', ec='lime', linewidth=0.8)
    ax.set_title('Gradient Vectors\n(showing direction & magnitude)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 2: Gradient Computation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"         Saved: {filename}")


def visualize_step3_cell_histograms(img, histograms, cell_size, filename):
    """Visualize cell histograms with real image"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    cells_y, cells_x = histograms.shape[:2]
    
    # Image with cell grid
    ax = axes[0]
    ax.imshow(img, cmap='gray')
    for cy in range(cells_y + 1):
        ax.axhline(y=cy * cell_size, color='lime', linewidth=0.5, alpha=0.7)
    for cx in range(cells_x + 1):
        ax.axvline(x=cx * cell_size, color='lime', linewidth=0.5, alpha=0.7)
    ax.set_title(f'Image with Cell Grid\n{cells_x}×{cells_y} = {cells_x*cells_y} cells (8×8 pixels each)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Sample histogram from center of image
    ax = axes[1]
    sample_cy, sample_cx = cells_y // 2, cells_x // 2
    hist = histograms[sample_cy, sample_cx]
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    bars = ax.bar(range(9), hist, color=colors, edgecolor='black', width=0.8)
    ax.set_xlabel('Orientation Bin (20° each)', fontsize=11)
    ax.set_ylabel('Weighted Magnitude Sum', fontsize=11)
    ax.set_xticks(range(9))
    ax.set_xticklabels([f'{i*20}°-{(i+1)*20}°' for i in range(9)], fontsize=8, rotation=45)
    ax.set_title(f'9-bin Histogram for Cell ({sample_cx}, {sample_cy})\n(center of image)', fontsize=11, fontweight='bold')
    
    peak_idx = np.argmax(hist)
    bars[peak_idx].set_edgecolor('red')
    bars[peak_idx].set_linewidth(3)
    
    # HOG visualization
    ax = axes[2]
    ax.imshow(img, cmap='gray', alpha=0.3)
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = cx * cell_size + cell_size / 2
            
            max_mag = hist.max() if hist.max() > 0 else 1
            for bin_idx, mag in enumerate(hist):
                if mag > max_mag * 0.2:
                    angle = bin_idx * 20 + 10
                    scale = (mag / max_mag) * (cell_size / 2.5)
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax.plot([center_x - dx, center_x + dx], 
                           [center_y - dy, center_y + dy],
                           color='red', linewidth=1, alpha=0.8)
    ax.set_title('HOG Visualization\n(lines show dominant gradient orientations)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 3: Cell Histograms (8×8 pixels → 9-bin histogram)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"         Saved: {filename}")


def visualize_step4_block_normalization(img, histograms, block_features, block_shape, cell_size, filename):
    """Visualize block normalization with real image"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    blocks_y, blocks_x = block_shape
    
    # Block layout overlay
    ax = axes[0]
    ax.imshow(img, cmap='gray', alpha=0.5)
    block_pixel_size = cell_size * 2
    
    # Show some blocks with different colors
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    num_show = min(12, blocks_y * blocks_x)
    for i in range(num_show):
        by = (i // blocks_x) % blocks_y
        bx = i % blocks_x
        rect = Rectangle((bx * cell_size, by * cell_size), 
                        block_pixel_size, block_pixel_size,
                        linewidth=2, edgecolor=colors[i % len(colors)], 
                        facecolor=colors[i % len(colors)], alpha=0.2)
        ax.add_patch(rect)
    ax.set_title(f'Block Layout (2×2 cells each)\n{blocks_x}×{blocks_y} = {blocks_x*blocks_y} blocks with 50% overlap', 
                fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Block vector visualization
    ax = axes[1]
    block_idx = (blocks_y // 2) * blocks_x + (blocks_x // 2)
    if block_idx < len(block_features):
        block_vector = block_features[block_idx]
        colors_bar = plt.cm.tab10(np.repeat(np.arange(4), 9) / 4)
        ax.bar(range(len(block_vector)), block_vector, color=colors_bar, edgecolor='none', width=0.8)
        for i in range(1, 4):
            ax.axvline(x=i*9 - 0.5, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Feature Index', fontsize=11)
        ax.set_ylabel('L2 Normalized Value', fontsize=11)
        ax.set_title('Single Block Vector (36 values)\n4 cells × 9 bins, L2 normalized', fontsize=11, fontweight='bold')
        ax.set_xlim(-1, 36)
        
        for i in range(4):
            ax.text(i*9 + 4, ax.get_ylim()[1] * 0.9, f'Cell {i}', ha='center', fontsize=9)
    
    # Final descriptor preview
    ax = axes[2]
    all_features = block_features.flatten()
    display_len = min(300, len(all_features))
    ax.bar(range(display_len), all_features[:display_len], 
           color='darkgreen', edgecolor='none', width=1)
    ax.set_xlabel('Feature Index', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'HOG Descriptor (first {display_len} of {len(all_features)})\n{blocks_x*blocks_y} blocks × 36 = {len(all_features)}-D vector', 
                fontsize=11, fontweight='bold')
    ax.set_xlim(-1, display_len)
    
    plt.suptitle('HOG Step 4: Block Normalization (L2-norm for illumination invariance)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"         Saved: {filename}")


def visualize_hog_final(img, histograms, cell_size, filename):
    """Create the classic HOG visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    cells_y, cells_x = histograms.shape[:2]
    
    # Original image
    ax = axes[0]
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Original Image ({img.shape[1]}×{img.shape[0]})', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # HOG visualization on black background
    ax = axes[1]
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_facecolor('black')
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = cx * cell_size + cell_size / 2
            
            max_mag = hist.max() if hist.max() > 0 else 1
            
            for bin_idx in range(9):
                magnitude_bin = hist[bin_idx]
                if magnitude_bin > max_mag * 0.1:
                    angle = bin_idx * 20 + 10
                    scale = (magnitude_bin / max_mag) * (cell_size / 2) * 0.9
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    
                    alpha = 0.4 + 0.6 * (magnitude_bin / max_mag)
                    ax.plot([center_x - dx, center_x + dx],
                           [center_y - dy, center_y + dy],
                           color='white', linewidth=1.2, alpha=alpha)
    
    ax.set_title('HOG Feature Visualization\n(edge orientations per cell)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.suptitle('HOG Algorithm - Final Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"         Saved: {filename}")


def visualize_complete_summary(original_rgb, gray_gamma, results, filename):
    """Complete pipeline summary visualization"""
    fig = plt.figure(figsize=(20, 12))
    
    cells_y, cells_x = results['histograms'].shape[:2]
    cell_size = 8
    
    # Step 1: Original
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(original_rgb)
    ax1.set_title('Step 1: Input Image', fontsize=10, fontweight='bold')
    ax1.axis('off')
    
    # Step 2a: Magnitude
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(results['magnitude'], cmap='hot')
    ax2.set_title('Step 2a: Gradient Magnitude', fontsize=10, fontweight='bold')
    ax2.axis('off')
    
    # Step 2b: Direction
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(results['direction'], cmap='hsv', vmin=0, vmax=180)
    ax3.set_title('Step 2b: Gradient Direction', fontsize=10, fontweight='bold')
    ax3.axis('off')
    
    # Step 3: Sample histogram
    ax4 = fig.add_subplot(2, 4, 4)
    sample_hist = results['histograms'][cells_y//2, cells_x//2]
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    ax4.bar(range(9), sample_hist, color=colors, edgecolor='black')
    ax4.set_title('Step 3: Cell Histogram\n(9 bins, 20° each)', fontsize=10, fontweight='bold')
    ax4.set_xticks(range(9))
    ax4.set_xticklabels([f'{i*20}°' for i in range(9)], fontsize=7)
    
    # Step 3: Cell grid
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.imshow(gray_gamma, cmap='gray')
    for cy in range(cells_y + 1):
        ax5.axhline(y=cy * cell_size, color='lime', linewidth=0.3)
    for cx in range(cells_x + 1):
        ax5.axvline(x=cx * cell_size, color='lime', linewidth=0.3)
    ax5.set_title(f'Step 3: {cells_x}×{cells_y}={cells_x*cells_y} Cells', fontsize=10, fontweight='bold')
    ax5.axis('off')
    
    # Step 4: Block vector
    ax6 = fig.add_subplot(2, 4, 6)
    block_idx = len(results['block_features']) // 2
    if block_idx < len(results['block_features']):
        block_vec = results['block_features'][block_idx]
        colors_bar = plt.cm.tab10(np.repeat(np.arange(4), 9) / 4)
        ax6.bar(range(36), block_vec, color=colors_bar, width=0.8)
        for i in range(1, 4):
            ax6.axvline(x=i*9-0.5, color='black', linestyle='--', linewidth=0.5)
    ax6.set_title('Step 4: Block Vector\n(36 values, L2 normalized)', fontsize=10, fontweight='bold')
    ax6.set_xlabel('Index')
    
    # HOG visualization
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.set_xlim(0, gray_gamma.shape[1])
    ax7.set_ylim(gray_gamma.shape[0], 0)
    ax7.set_facecolor('black')
    histograms = results['histograms']
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
                            color='white', linewidth=0.8, alpha=0.8)
    ax7.set_title('Step 5: HOG Visualization', fontsize=10, fontweight='bold')
    ax7.set_aspect('equal')
    
    # Final descriptor
    ax8 = fig.add_subplot(2, 4, 8)
    desc = results['descriptor']
    display_len = min(200, len(desc))
    ax8.bar(range(display_len), desc[:display_len], color='darkgreen', width=1)
    ax8.set_title(f'Final: {len(desc)}-D Descriptor', fontsize=10, fontweight='bold')
    ax8.set_xlabel('Index')
    ax8.set_xlim(-1, display_len)
    
    plt.suptitle(f'HOG Algorithm - Complete Pipeline ({gray_gamma.shape[1]}×{gray_gamma.shape[0]} image)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"         Saved: {filename}")


def run_hog_pipeline():
    """Run the complete HOG pipeline on real image"""
    print("=" * 70)
    print("HOG ALGORITHM PIPELINE")
    print("=" * 70)
    
    # Load real image
    print("\n[HOG Step 1] Preprocessing...")
    gray, original_rgb = load_image()
    h, w = gray.shape
    print(f"         Loaded image: {w} × {h} pixels")
    
    gray_gamma = gamma_correction(gray, gamma=0.5)
    print(f"         Applied gamma correction (γ=0.5)")
    
    # Extract HOG features
    results = extract_hog_descriptor(gray_gamma)
    
    print("\n[HOG Step 5] Concatenating feature vector...")
    print(f"         Final descriptor: {len(results['descriptor'])}-D")
    
    print("\n" + "-" * 70)
    print("Generating visualizations...")
    print("-" * 70)
    
    visualize_step1_preprocessing(original_rgb, gray, gray_gamma, 'hog_step1_preprocessing.png')
    
    visualize_step2_gradients(gray_gamma, results['magnitude'], results['direction'],
                             results['gx'], results['gy'], 'hog_step2_gradients.png')
    
    visualize_step3_cell_histograms(gray_gamma, results['histograms'], 8, 'hog_step3_cell_histograms.png')
    
    visualize_step4_block_normalization(gray_gamma, results['histograms'], 
                                        results['block_features'], results['block_shape'],
                                        8, 'hog_step4_block_normalization.png')
    
    visualize_hog_final(gray_gamma, results['histograms'], 8, 'hog_visualization.png')
    
    visualize_complete_summary(original_rgb, gray_gamma, results, 'hog_complete_summary.png')
    
    print("\n" + "=" * 70)
    print("HOG PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Image size:        {w} × {h}")
    print(f"  Cell size:         8 × 8 pixels")
    print(f"  Cells:             {results['histograms'].shape[1]} × {results['histograms'].shape[0]} = {results['histograms'].shape[0] * results['histograms'].shape[1]}")
    print(f"  Blocks:            {results['block_shape'][1]} × {results['block_shape'][0]} = {results['block_shape'][0] * results['block_shape'][1]}")
    print(f"  Bins per cell:     9")
    print(f"  Features/block:    36")
    print(f"  Total descriptor:  {len(results['descriptor'])}-D")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_hog_pipeline()

"""
HOG Step 4 Detailed Visualization
Block normalization step-by-step like SIFT filtering stages
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
    gray = np.power(np.clip(gray, 1e-8, 1), 0.5)
    return gray, img


def compute_cell_histograms(gray, cell_size=8, num_bins=9):
    """Compute cell histograms"""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) * 180 / np.pi % 180
    
    h, w = gray.shape
    cells_y = h // cell_size
    cells_x = w // cell_size
    bin_width = 180.0 / num_bins
    
    histograms = np.zeros((cells_y, cells_x, num_bins))
    
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
                    lower_bin = int(bin_idx) % num_bins
                    upper_bin = (lower_bin + 1) % num_bins
                    upper_weight = bin_idx - int(bin_idx)
                    lower_weight = 1 - upper_weight
                    histograms[cy, cx, lower_bin] += mag * lower_weight
                    histograms[cy, cx, upper_bin] += mag * upper_weight
    
    return histograms, magnitude, direction


def create_step4_1_block_definition():
    """Step 4.1: Define what a block is"""
    gray, original_rgb = load_image()
    histograms, _, _ = compute_cell_histograms(gray)
    
    cell_size = 8
    h, w = gray.shape
    cells_y, cells_x = histograms.shape[:2]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel 1: Cell grid with blocks highlighted
    ax = axes[0]
    ax.imshow(gray, cmap='gray')
    
    # Draw cell grid
    for cy in range(cells_y + 1):
        ax.axhline(y=cy * cell_size, color='gray', linewidth=0.3, alpha=0.5)
    for cx in range(cells_x + 1):
        ax.axvline(x=cx * cell_size, color='gray', linewidth=0.3, alpha=0.5)
    
    # Highlight 2×2 blocks
    block_positions = [(2, 2), (2, 4), (4, 2), (4, 4), (10, 20), (10, 22)]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    
    for (by, bx), color in zip(block_positions, colors):
        if by + 1 < cells_y and bx + 1 < cells_x:
            rect = Rectangle((bx * cell_size, by * cell_size), cell_size * 2, cell_size * 2,
                             fill=False, edgecolor=color, linewidth=3)
            ax.add_patch(rect)
    
    ax.set_title(f'Cells with 2×2 Blocks\nBlock = 2×2 cells = 16×16 pixels', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Panel 2: Block structure diagram
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Draw 2×2 block
    cell_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    cell_labels = ['Cell (0,0)\n9 bins', 'Cell (1,0)\n9 bins', 'Cell (0,1)\n9 bins', 'Cell (1,1)\n9 bins']
    positions = [(2, 5), (4.5, 5), (2, 2.5), (4.5, 2.5)]
    
    for (x, y), color, label in zip(positions, cell_colors, cell_labels):
        rect = Rectangle((x, y), 2, 2, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1, y + 1, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Block outline
    rect = Rectangle((2, 2.5), 4.5, 4.5, fill=False, edgecolor='red', linewidth=4)
    ax.add_patch(rect)
    
    ax.text(4.25, 7.5, '2×2 Block', fontsize=14, ha='center', fontweight='bold', color='red')
    
    ax.text(4.25, 1.5, 'Block vector = [C(0,0) | C(1,0) | C(0,1) | C(1,1)]', fontsize=10, ha='center')
    ax.text(4.25, 0.8, '= 4 cells × 9 bins = 36 values', fontsize=11, ha='center', fontweight='bold',
            bbox=dict(facecolor='lightyellow', edgecolor='orange'))
    
    ax.set_title('Block Structure: 2×2 Cells', fontsize=12, fontweight='bold')
    
    plt.suptitle('HOG Step 4.1: Block Definition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step4_1_block_definition.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step4_1_block_definition.png")


def create_step4_2_overlap():
    """Step 4.2: Show block overlap"""
    gray, _ = load_image()
    histograms, _, _ = compute_cell_histograms(gray)
    
    cell_size = 8
    cells_y, cells_x = histograms.shape[:2]
    blocks_y = cells_y - 1
    blocks_x = cells_x - 1
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Panel 1: Overlap visualization on image
    ax = axes[0]
    ax.imshow(gray, cmap='gray', alpha=0.3)
    
    # Draw first 4 blocks with overlap
    block_colors = ['red', 'blue', 'green', 'orange']
    for i, (by, bx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        color = block_colors[i]
        rect = Rectangle((bx * cell_size, by * cell_size), cell_size * 2, cell_size * 2,
                         fill=True, facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
    
    # Show overlap region
    rect = Rectangle((cell_size, cell_size), cell_size, cell_size,
                     fill=True, facecolor='purple', alpha=0.5, edgecolor='purple', linewidth=3)
    ax.add_patch(rect)
    
    ax.set_xlim(0, cell_size * 5)
    ax.set_ylim(cell_size * 5, 0)
    ax.set_title('Block Overlap (50%)\nPurple = shared by all 4 blocks', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Panel 2: Diagram
    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw cells grid
    for i in range(5):
        for j in range(5):
            rect = Rectangle((1 + j*1.5, 6 - i*1.5), 1.4, 1.4, 
                            facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            ax.text(1.7 + j*1.5, 6.7 - i*1.5, f'({j},{i})', fontsize=7, ha='center')
    
    # Block 1
    rect = Rectangle((1, 4.5), 3, 3, fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(rect)
    ax.text(2.5, 7.8, 'Block 1', fontsize=10, ha='center', color='red', fontweight='bold')
    
    # Block 2
    rect = Rectangle((2.5, 4.5), 3, 3, fill=False, edgecolor='blue', linewidth=3, linestyle='--')
    ax.add_patch(rect)
    ax.text(4, 8.1, 'Block 2', fontsize=10, ha='center', color='blue', fontweight='bold')
    
    # Explanation
    ax.text(9.5, 7, 'Stride = 1 cell', fontsize=11, ha='center')
    ax.text(9.5, 6, '= 50% overlap', fontsize=11, ha='center')
    ax.text(9.5, 4.5, f'Total Blocks:', fontsize=11, ha='center', fontweight='bold')
    ax.text(9.5, 3.5, f'{blocks_x} × {blocks_y} = {blocks_x*blocks_y}', fontsize=12, ha='center',
            bbox=dict(facecolor='lightyellow', edgecolor='orange'))
    
    ax.text(6, 1.5, 'Why overlap?\n→ Better feature continuity\n→ More robust detection', 
           fontsize=10, ha='center', bbox=dict(facecolor='lightgreen', edgecolor='green'))
    
    ax.set_title('50% Block Overlap', fontsize=12, fontweight='bold')
    
    plt.suptitle('HOG Step 4.2: Block Overlap (Stride = 1 cell)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step4_2_overlap.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step4_2_overlap.png")


def create_step4_3_l2_normalization():
    """Step 4.3: L2 normalization formula and example"""
    gray, _ = load_image()
    histograms, _, _ = compute_cell_histograms(gray)
    
    cells_y, cells_x = histograms.shape[:2]
    
    # Get a block
    by, bx = cells_y // 2, cells_x // 2
    block_raw = histograms[by:by+2, bx:bx+2, :].flatten()
    
    # L2 normalize
    eps = 1e-6
    l2_norm = np.sqrt(np.sum(block_raw**2) + eps**2)
    block_normalized = block_raw / l2_norm
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Raw block vector
    ax = axes[0, 0]
    colors = plt.cm.tab10(np.repeat(np.arange(4), 9) / 4)
    ax.bar(range(36), block_raw, color=colors, edgecolor='none', width=0.8)
    for i in range(1, 4):
        ax.axvline(x=i*9 - 0.5, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Feature Index', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Raw Block Vector (36 values)\n||v||₂ = {l2_norm:.4f}', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 36)
    
    for i in range(4):
        ax.text(i*9 + 4, ax.get_ylim()[1] * 0.95, f'Cell {i}', ha='center', fontsize=9, fontweight='bold')
    
    # Panel 2: L2 norm formula
    ax = axes[0, 1]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    ax.text(5, 9, 'L2 Normalization', fontsize=14, ha='center', fontweight='bold')
    
    ax.text(5, 7.5, r'$v_{normalized} = \frac{v}{\sqrt{||v||_2^2 + \epsilon^2}}$', fontsize=16, ha='center')
    
    ax.text(5, 5.5, r'$||v||_2 = \sqrt{\sum_{i=1}^{36} v_i^2}$', fontsize=14, ha='center')
    
    ax.text(5, 4, f'For this block:', fontsize=12, ha='center', fontweight='bold')
    ax.text(5, 3, f'||v||₂ = {l2_norm:.4f}', fontsize=12, ha='center')
    ax.text(5, 2, f'ε = {eps} (stability)', fontsize=11, ha='center')
    
    ax.text(5, 0.8, 'Result: All values scaled to [0, 1]', fontsize=11, ha='center',
            bbox=dict(facecolor='lightgreen', edgecolor='green'))
    
    ax.set_title('L2 Normalization Formula', fontsize=12, fontweight='bold')
    
    # Panel 3: Normalized block vector
    ax = axes[1, 0]
    ax.bar(range(36), block_normalized, color=colors, edgecolor='none', width=0.8)
    for i in range(1, 4):
        ax.axvline(x=i*9 - 0.5, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Feature Index', fontsize=11)
    ax.set_ylabel('Normalized Value', fontsize=11)
    
    new_norm = np.sqrt(np.sum(block_normalized**2))
    ax.set_title(f'Normalized Block Vector\n||v||₂ ≈ {new_norm:.4f} (≈ 1.0)', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 36)
    
    for i in range(4):
        ax.text(i*9 + 4, ax.get_ylim()[1] * 0.95, f'Cell {i}', ha='center', fontsize=9, fontweight='bold')
    
    # Panel 4: Before/After comparison
    ax = axes[1, 1]
    
    x = np.arange(9)
    width = 0.35
    
    ax.bar(x - width/2, block_raw[:9], width, label='Before (Cell 0)', color='lightblue', edgecolor='blue')
    ax.bar(x + width/2, block_normalized[:9], width, label='After (Cell 0)', color='lightcoral', edgecolor='red')
    
    ax.set_xlabel('Bin', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i*20}°' for i in range(9)], fontsize=8)
    ax.legend()
    ax.set_title('Before vs After Normalization\n(Cell 0 comparison)', fontsize=12, fontweight='bold')
    
    plt.suptitle('HOG Step 4.3: L2 Normalization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step4_3_l2_normalization.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step4_3_l2_normalization.png")


def create_step4_4_illumination_invariance():
    """Step 4.4: Show why L2 norm provides illumination invariance"""
    gray, _ = load_image()
    
    # Create dark and bright versions
    gray_dark = gray * 0.5
    gray_bright = np.clip(gray * 1.5, 0, 1)
    
    # Compute histograms for same region
    cell_size = 8
    cells_y, cells_x = gray.shape[0] // cell_size, gray.shape[1] // cell_size
    cy, cx = cells_y // 2, cells_x // 2
    
    def get_block_histogram(img, cy, cx, cell_size=8):
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
        gy[1:-1, :] = img[2:, :] - img[:-2, :]
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx) * 180 / np.pi % 180
        
        block = np.zeros(36)
        bin_width = 20
        
        for dy in range(2):
            for dx in range(2):
                cell_idx = dy * 2 + dx
                y_start = (cy + dy) * cell_size
                x_start = (cx + dx) * cell_size
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
                        block[cell_idx * 9 + lower_bin] += mag * lower_weight
                        block[cell_idx * 9 + upper_bin] += mag * upper_weight
        
        return block
    
    block_dark = get_block_histogram(gray_dark, cy, cx)
    block_bright = get_block_histogram(gray_bright, cy, cx)
    
    # Normalize
    block_dark_norm = block_dark / (np.sqrt(np.sum(block_dark**2)) + 1e-6)
    block_bright_norm = block_bright / (np.sqrt(np.sum(block_bright**2)) + 1e-6)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Dark image
    y_start = cy * cell_size
    x_start = cx * cell_size
    
    ax = axes[0, 0]
    ax.imshow(gray_dark, cmap='gray', vmin=0, vmax=1)
    rect = Rectangle((x_start, y_start), cell_size*2, cell_size*2, fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(rect)
    ax.set_title('Dark Image (×0.5)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = axes[0, 1]
    colors = plt.cm.tab10(np.repeat(np.arange(4), 9) / 4)
    ax.bar(range(36), block_dark, color=colors, edgecolor='none', width=0.8)
    ax.set_title(f'Raw Block Vector\nMax = {block_dark.max():.3f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature Index')
    
    ax = axes[0, 2]
    ax.bar(range(36), block_dark_norm, color=colors, edgecolor='none', width=0.8)
    ax.set_title('Normalized Block Vector', fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature Index')
    ax.set_ylim(0, 0.5)
    
    # Row 2: Bright image
    ax = axes[1, 0]
    ax.imshow(gray_bright, cmap='gray', vmin=0, vmax=1)
    rect = Rectangle((x_start, y_start), cell_size*2, cell_size*2, fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(rect)
    ax.set_title('Bright Image (×1.5)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.bar(range(36), block_bright, color=colors, edgecolor='none', width=0.8)
    ax.set_title(f'Raw Block Vector\nMax = {block_bright.max():.3f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature Index')
    
    ax = axes[1, 2]
    ax.bar(range(36), block_bright_norm, color=colors, edgecolor='none', width=0.8)
    ax.set_title('Normalized Block Vector', fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature Index')
    ax.set_ylim(0, 0.5)
    
    raw_diff = np.abs(block_dark - block_bright).sum()
    norm_diff = np.abs(block_dark_norm - block_bright_norm).sum()
    
    fig.text(0.5, 0.02, 
            f'Raw difference: {raw_diff:.3f}  →  Normalized difference: {norm_diff:.3f} (much smaller!)',
            ha='center', fontsize=13, fontweight='bold',
            bbox=dict(facecolor='lightgreen', edgecolor='green', linewidth=2))
    
    plt.suptitle('HOG Step 4.4: Illumination Invariance via L2 Normalization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(os.path.join(OUT_DIR, 'hog_step4_4_illumination.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step4_4_illumination.png")


def create_step4_5_final_descriptor():
    """Step 4.5: Final descriptor assembly"""
    gray, _ = load_image()
    histograms, _, _ = compute_cell_histograms(gray)
    
    cells_y, cells_x = histograms.shape[:2]
    blocks_y = cells_y - 1
    blocks_x = cells_x - 1
    
    # Compute all block features
    eps = 1e-6
    block_features = []
    
    for by in range(blocks_y):
        for bx in range(blocks_x):
            block = histograms[by:by+2, bx:bx+2, :].flatten()
            norm = np.sqrt(np.sum(block**2) + eps**2)
            block_features.append(block / norm)
    
    block_features = np.array(block_features)
    descriptor = block_features.flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Block layout
    ax = axes[0, 0]
    ax.imshow(gray, cmap='gray', alpha=0.3)
    
    # Show grid of blocks
    for by in range(min(blocks_y, 20)):
        for bx in range(min(blocks_x, 30)):
            color = plt.cm.tab20((by * blocks_x + bx) % 20)
            rect = Rectangle((bx * 8, by * 8), 16, 16, 
                             fill=True, facecolor=color, alpha=0.2, edgecolor=color, linewidth=0.5)
            ax.add_patch(rect)
    
    ax.set_title(f'All {blocks_x}×{blocks_y} = {blocks_x*blocks_y} Blocks', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Panel 2: Sample block vectors
    ax = axes[0, 1]
    
    # Show first 5 block vectors stacked
    n_show = 5
    for i in range(n_show):
        block_idx = i * (len(block_features) // n_show)
        y_offset = (n_show - 1 - i) * 0.6
        colors = plt.cm.tab10(np.repeat(np.arange(4), 9) / 4)
        ax.bar(range(36), block_features[block_idx] + y_offset, 
               color=colors, edgecolor='none', width=0.8, alpha=0.7)
        ax.axhline(y=y_offset, color='gray', linestyle='--', linewidth=0.5)
        ax.text(-2, y_offset + 0.1, f'B{block_idx}', fontsize=8)
    
    ax.set_xlabel('Feature Index (per block)', fontsize=11)
    ax.set_title(f'Sample Block Vectors (5 of {len(block_features)})', fontsize=12, fontweight='bold')
    ax.set_xlim(-3, 36)
    
    # Panel 3: Final descriptor
    ax = axes[1, 0]
    display_len = min(500, len(descriptor))
    ax.bar(range(display_len), descriptor[:display_len], color='darkgreen', width=1, edgecolor='none')
    ax.set_xlabel('Feature Index', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Final HOG Descriptor (first {display_len} of {len(descriptor):,})', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, display_len)
    
    # Panel 4: Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats = f"""
    HOG Descriptor Statistics
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Image size:        {gray.shape[1]} × {gray.shape[0]} pixels
    Cell size:         8 × 8 pixels
    Cells:             {cells_x} × {cells_y} = {cells_x*cells_y}
    
    Block size:        2 × 2 cells
    Block stride:      1 cell (50% overlap)
    Blocks:            {blocks_x} × {blocks_y} = {blocks_x*blocks_y}
    
    Values per block:  36 (4 cells × 9 bins)
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    FINAL DESCRIPTOR:  {len(descriptor):,} dimensions
                       = {blocks_x*blocks_y} blocks × 36
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Descriptor stats:
      Min:  {descriptor.min():.4f}
      Max:  {descriptor.max():.4f}
      Mean: {descriptor.mean():.4f}
    """
    ax.text(0.1, 0.5, stats, fontsize=11, family='monospace', va='center',
            transform=ax.transAxes, bbox=dict(facecolor='lightyellow', edgecolor='orange', linewidth=2))
    ax.set_title('Final Statistics', fontsize=12, fontweight='bold')
    
    plt.suptitle('HOG Step 4.5: Final Descriptor Assembly', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step4_5_final.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step4_5_final.png")


def create_step4_complete_summary():
    """Step 4 complete summary"""
    gray, original_rgb = load_image()
    histograms, magnitude, direction = compute_cell_histograms(gray)
    
    cells_y, cells_x = histograms.shape[:2]
    blocks_y = cells_y - 1
    blocks_x = cells_x - 1
    cell_size = 8
    
    # Compute block features
    eps = 1e-6
    block_features = []
    for by in range(blocks_y):
        for bx in range(blocks_x):
            block = histograms[by:by+2, bx:bx+2, :].flatten()
            norm = np.sqrt(np.sum(block**2) + eps**2)
            block_features.append(block / norm)
    block_features = np.array(block_features)
    descriptor = block_features.flatten()
    
    fig = plt.figure(figsize=(20, 10))
    
    # Panel 1: Original
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(original_rgb)
    ax1.set_title('Original Image', fontsize=10, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Cell histograms
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(gray, cmap='gray', alpha=0.3)
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
                    ax2.plot([center_x - dx, center_x + dx], [center_y - dy, center_y + dy],
                            color='red', linewidth=0.5, alpha=0.8)
    ax2.set_title(f'{cells_x*cells_y} Cell Histograms', fontsize=10, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: Block grid
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(gray, cmap='gray', alpha=0.3)
    for by in range(min(blocks_y, 15)):
        for bx in range(min(blocks_x, 20)):
            color = plt.cm.tab20((by * blocks_x + bx) % 20)
            rect = Rectangle((bx * 8, by * 8), 16, 16, 
                             fill=True, facecolor=color, alpha=0.3, edgecolor=color, linewidth=0.3)
            ax3.add_patch(rect)
    ax3.set_title(f'{blocks_x*blocks_y} Blocks (2×2 cells)', fontsize=10, fontweight='bold')
    ax3.axis('off')
    
    # Panel 4: Block vector
    ax4 = fig.add_subplot(2, 4, 4)
    block_idx = len(block_features) // 2
    colors = plt.cm.tab10(np.repeat(np.arange(4), 9) / 4)
    ax4.bar(range(36), block_features[block_idx], color=colors, width=0.8)
    for i in range(1, 4):
        ax4.axvline(x=i*9 - 0.5, color='black', linestyle='--', linewidth=0.5)
    ax4.set_title('Sample Block (36-D)', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Index')
    
    # Panel 5: Multiple blocks
    ax5 = fig.add_subplot(2, 4, 5)
    n_show = 4
    for i in range(n_show):
        idx = i * (len(block_features) // n_show)
        y_offset = (n_show - 1 - i) * 0.5
        ax5.bar(range(36), block_features[idx] + y_offset, color=colors, width=0.8, alpha=0.7)
    ax5.set_title(f'{n_show} Sample Blocks', fontsize=10, fontweight='bold')
    ax5.set_xlabel('Index')
    
    # Panel 6: L2 normalization effect
    ax6 = fig.add_subplot(2, 4, 6)
    raw_block = histograms[cells_y//2:cells_y//2+2, cells_x//2:cells_x//2+2, :].flatten()
    norm_block = block_features[len(block_features)//2]
    x = np.arange(9)
    ax6.bar(x - 0.2, raw_block[:9] / raw_block.max(), 0.4, label='Raw', color='blue', alpha=0.7)
    ax6.bar(x + 0.2, norm_block[:9] / norm_block.max(), 0.4, label='Normalized', color='red', alpha=0.7)
    ax6.legend(fontsize=8)
    ax6.set_title('L2 Normalization', fontsize=10, fontweight='bold')
    ax6.set_xlabel('Bin')
    
    # Panel 7: Final descriptor
    ax7 = fig.add_subplot(2, 4, 7)
    display_len = 200
    ax7.bar(range(display_len), descriptor[:display_len], color='darkgreen', width=1)
    ax7.set_title(f'Descriptor ({len(descriptor):,}-D)', fontsize=10, fontweight='bold')
    ax7.set_xlabel('Index')
    
    # Panel 8: Summary
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    summary = f"""
    Step 4 Complete
    ━━━━━━━━━━━━━━━
    
    Input:
      {cells_x*cells_y} cells
      × 9 bins each
    
    Process:
      Group into 2×2 blocks
      L2 normalize each
    
    Output:
      {blocks_x*blocks_y} blocks
      × 36 values each
      = {len(descriptor):,}-D
    """
    ax8.text(0.1, 0.5, summary, fontsize=10, family='monospace', va='center',
            transform=ax8.transAxes, bbox=dict(facecolor='lightyellow', edgecolor='orange'))
    ax8.set_title('Summary', fontsize=10, fontweight='bold')
    
    plt.suptitle('HOG Step 4: Block Normalization - Complete Pipeline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step4_complete.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step4_complete.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating HOG Step 4 Detailed Visualizations")
    print("=" * 60)
    
    create_step4_1_block_definition()
    create_step4_2_overlap()
    create_step4_3_l2_normalization()
    create_step4_4_illumination_invariance()
    create_step4_5_final_descriptor()
    create_step4_complete_summary()
    
    print("=" * 60)
    print("Done!")
    print("=" * 60)

"""
HOG Block Normalization Visualization
Detailed visualizations for block normalization step using real images
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


def compute_hog_features(gray, cell_size=8, num_bins=9, block_size=2):
    """Compute HOG features"""
    # Gradients
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
    
    # Cell histograms
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
    
    # Block normalization
    blocks_y = cells_y - block_size + 1
    blocks_x = cells_x - block_size + 1
    block_features = []
    
    eps = 1e-6
    for by in range(blocks_y):
        for bx in range(blocks_x):
            block = histograms[by:by+block_size, bx:bx+block_size, :].flatten()
            norm = np.sqrt(np.sum(block**2) + eps**2)
            block_features.append(block / norm)
    
    return histograms, np.array(block_features), (blocks_y, blocks_x)


def create_block_layout_visual():
    """Visualize how blocks are formed from cells with overlap"""
    gray, original_rgb = load_image()
    
    cell_size = 8
    h, w = gray.shape
    cells_y = h // cell_size
    cells_x = w // cell_size
    blocks_y = cells_y - 1
    blocks_x = cells_x - 1
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Cell grid on image
    ax = axes[0]
    ax.imshow(gray, cmap='gray')
    for cy in range(cells_y + 1):
        ax.axhline(y=cy * cell_size, color='lime', linewidth=0.5, alpha=0.7)
    for cx in range(cells_x + 1):
        ax.axvline(x=cx * cell_size, color='lime', linewidth=0.5, alpha=0.7)
    
    # Highlight some blocks
    colors = ['red', 'blue', 'green', 'orange']
    block_positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for (bx, by), color in zip(block_positions, colors):
        rect = Rectangle((bx * cell_size, by * cell_size), cell_size * 2, cell_size * 2,
                         fill=False, edgecolor=color, linewidth=3)
        ax.add_patch(rect)
    
    ax.set_title(f'Cell Grid ({cells_x}×{cells_y} cells)\nwith Example Blocks (colored rectangles)', 
                fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Panel 2: Block overlap detail
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Draw 4 cells
    cell_positions = [(1, 4), (3, 4), (1, 2), (3, 2)]
    cell_labels = ['C(0,0)', 'C(1,0)', 'C(0,1)', 'C(1,1)']
    
    for (x, y), label in zip(cell_positions, cell_labels):
        rect = Rectangle((x, y), 1.8, 1.8, facecolor='lightgray', edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.9, y + 0.9, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Block 1 (red)
    rect = Rectangle((1, 2), 3.8, 3.8, facecolor='none', edgecolor='red', linewidth=3)
    ax.add_patch(rect)
    ax.text(2.9, 6.2, 'Block 1', ha='center', fontsize=11, color='red', fontweight='bold')
    
    # Block 2 (blue) - overlapping
    rect = Rectangle((3, 2), 3.8, 3.8, facecolor='none', edgecolor='blue', linewidth=3, linestyle='--')
    ax.add_patch(rect)
    ax.text(5.1, 6.5, 'Block 2', ha='center', fontsize=11, color='blue', fontweight='bold')
    
    # Draw extra cells for block 2
    for x, y in [(5, 4), (5, 2)]:
        rect = Rectangle((x, y), 1.8, 1.8, facecolor='lightblue', edgecolor='black', linewidth=1, alpha=0.5)
        ax.add_patch(rect)
    
    # Overlap highlight
    rect = Rectangle((3, 2), 1.8, 3.8, facecolor='purple', alpha=0.3, edgecolor='none')
    ax.add_patch(rect)
    ax.text(3.9, 0.8, '50% Overlap\n(shared cells)', ha='center', fontsize=10, color='purple', fontweight='bold')
    
    ax.set_title('Block Overlap Diagram\n(stride = 1 cell = 50% overlap)', fontsize=11, fontweight='bold')
    
    # Panel 3: Statistics
    ax = axes[2]
    ax.axis('off')
    
    stats_text = f"""
    Block Layout Statistics
    ━━━━━━━━━━━━━━━━━━━━━━━
    
    Image size:     {w} × {h} pixels
    Cell size:      {cell_size} × {cell_size} pixels
    
    Number of cells:
      cells_x = {w} ÷ {cell_size} = {cells_x}
      cells_y = {h} ÷ {cell_size} = {cells_y}
      Total = {cells_x * cells_y} cells
    
    Block size:     2 × 2 cells
    Block stride:   1 cell (50% overlap)
    
    Number of blocks:
      blocks_x = {cells_x} - 1 = {blocks_x}
      blocks_y = {cells_y} - 1 = {blocks_y}
      Total = {blocks_x * blocks_y} blocks
    
    Values per block: 4 cells × 9 bins = 36
    
    HOG Descriptor: {blocks_x * blocks_y} × 36 = {blocks_x * blocks_y * 36}-D
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', va='center',
            transform=ax.transAxes, bbox=dict(facecolor='lightyellow', edgecolor='orange', linewidth=2))
    ax.set_title('Block Layout Statistics', fontsize=11, fontweight='bold')
    
    plt.suptitle('HOG Step 4: Block Layout and Overlap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_block_layout.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_block_layout.png")


def create_normalization_example():
    """Show L2 normalization with real block vector"""
    gray, _ = load_image()
    histograms, block_features, block_shape = compute_hog_features(gray)
    
    blocks_y, blocks_x = block_shape
    
    # Get a block from the center
    block_idx = (blocks_y // 2) * blocks_x + (blocks_x // 2)
    
    # Also get the raw (un-normalized) block
    by = block_idx // blocks_x
    bx = block_idx % blocks_x
    block_raw = histograms[by:by+2, bx:bx+2, :].flatten()
    block_normalized = block_features[block_idx]
    
    l2_norm = np.sqrt(np.sum(block_raw**2) + 1e-12)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel 1: Raw block vector
    ax = axes[0]
    colors = plt.cm.tab10(np.repeat(np.arange(4), 9) / 4)
    ax.bar(range(36), block_raw, color=colors, edgecolor='none', width=0.8)
    for i in range(1, 4):
        ax.axvline(x=i*9 - 0.5, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Feature Index', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Raw Block Vector (36 values)\nBefore Normalization', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 36)
    
    for i in range(4):
        ax.text(i*9 + 4, ax.get_ylim()[1] * 0.95, f'Cell {i}', ha='center', fontsize=9, fontweight='bold')
    
    ax.text(18, ax.get_ylim()[1] * 0.7, f'||v||₂ = {l2_norm:.3f}', 
           fontsize=10, ha='center', bbox=dict(facecolor='white', edgecolor='gray'))
    
    # Panel 2: Normalization formula
    ax = axes[1]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    ax.text(5, 8.5, 'L2 Normalization', fontsize=14, ha='center', fontweight='bold')
    
    ax.text(5, 7, r'$v_{norm} = \frac{v}{\sqrt{||v||_2^2 + \epsilon^2}}$', fontsize=16, ha='center')
    
    ax.text(5, 5.5, f'Calculation:', fontsize=12, ha='center', fontweight='bold')
    ax.text(5, 4.5, r'$||v||_2 = \sqrt{\sum_{i=1}^{36} v_i^2}$' + f' = {l2_norm:.4f}', fontsize=12, ha='center')
    ax.text(5, 3.5, r'$\epsilon$' + f' = 1e-6 (numerical stability)', fontsize=11, ha='center')
    
    ax.text(5, 2, 'Effect:', fontsize=12, ha='center', fontweight='bold')
    ax.text(5, 1, 'Normalizes to unit length\n→ Illumination invariant', fontsize=11, ha='center')
    
    ax.set_title('L2 Normalization Formula', fontsize=12, fontweight='bold')
    
    # Panel 3: Normalized block vector
    ax = axes[2]
    ax.bar(range(36), block_normalized, color=colors, edgecolor='none', width=0.8)
    for i in range(1, 4):
        ax.axvline(x=i*9 - 0.5, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Feature Index', fontsize=11)
    ax.set_ylabel('Normalized Value', fontsize=11)
    ax.set_title('Normalized Block Vector\nAfter L2 Normalization', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 36)
    
    for i in range(4):
        ax.text(i*9 + 4, ax.get_ylim()[1] * 0.95, f'Cell {i}', ha='center', fontsize=9, fontweight='bold')
    
    new_norm = np.sqrt(np.sum(block_normalized**2))
    ax.text(18, ax.get_ylim()[1] * 0.7, f'||v||₂ ≈ {new_norm:.3f}', 
           fontsize=10, ha='center', bbox=dict(facecolor='white', edgecolor='gray'))
    
    plt.suptitle('HOG Block Normalization: Real Example from Image', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_normalization_example.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_normalization_example.png")


def create_illumination_invariance_visual():
    """Show how normalization provides illumination invariance"""
    gray, _ = load_image()
    
    # Create bright and dark versions
    gray_dark = gray * 0.5
    gray_bright = np.clip(gray * 1.5, 0, 1)
    
    # Extract same cell from both
    cell_size = 8
    cy, cx = gray.shape[0] // (2 * cell_size), gray.shape[1] // (2 * cell_size)
    y_start = cy * cell_size
    x_start = cx * cell_size
    
    def get_cell_histogram(img, y_start, x_start, cell_size=8):
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
        gy[1:-1, :] = img[2:, :] - img[:-2, :]
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx) * 180 / np.pi % 180
        
        cell_mag = magnitude[y_start:y_start+cell_size, x_start:x_start+cell_size]
        cell_dir = direction[y_start:y_start+cell_size, x_start:x_start+cell_size]
        
        histogram = np.zeros(9)
        bin_width = 20
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
        return histogram
    
    hist_dark = get_cell_histogram(gray_dark, y_start, x_start)
    hist_bright = get_cell_histogram(gray_bright, y_start, x_start)
    
    hist_dark_norm = hist_dark / (np.sqrt(np.sum(hist_dark**2)) + 1e-6)
    hist_bright_norm = hist_bright / (np.sqrt(np.sum(hist_bright**2)) + 1e-6)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Dark image
    ax = axes[0, 0]
    ax.imshow(gray_dark[y_start:y_start+32, x_start:x_start+32], cmap='gray', vmin=0, vmax=1)
    ax.set_title('Dark Image (×0.5)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.bar(range(9), hist_dark, color='blue', alpha=0.7, edgecolor='black')
    ax.set_title(f'Raw Histogram\nMax = {hist_dark.max():.3f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Bin')
    ax.set_xticks(range(9))
    
    ax = axes[0, 2]
    ax.bar(range(9), hist_dark_norm, color='green', alpha=0.7, edgecolor='black')
    ax.set_title('Normalized Histogram', fontsize=11, fontweight='bold')
    ax.set_xlabel('Bin')
    ax.set_xticks(range(9))
    ax.set_ylim(0, 1)
    
    # Row 2: Bright image
    ax = axes[1, 0]
    ax.imshow(gray_bright[y_start:y_start+32, x_start:x_start+32], cmap='gray', vmin=0, vmax=1)
    ax.set_title('Bright Image (×1.5)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.bar(range(9), hist_bright, color='blue', alpha=0.7, edgecolor='black')
    ax.set_title(f'Raw Histogram\nMax = {hist_bright.max():.3f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Bin')
    ax.set_xticks(range(9))
    
    ax = axes[1, 2]
    ax.bar(range(9), hist_bright_norm, color='green', alpha=0.7, edgecolor='black')
    ax.set_title('Normalized Histogram', fontsize=11, fontweight='bold')
    ax.set_xlabel('Bin')
    ax.set_xticks(range(9))
    ax.set_ylim(0, 1)
    
    raw_diff = np.abs(hist_dark - hist_bright).sum()
    norm_diff = np.abs(hist_dark_norm - hist_bright_norm).sum()
    
    fig.text(0.5, 0.02, 
            f'Raw histogram difference: {raw_diff:.3f}  |  '
            f'Normalized difference: {norm_diff:.3f} (nearly identical!)',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='lightyellow', edgecolor='orange', linewidth=2))
    
    plt.suptitle('Illumination Invariance Through L2 Normalization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(os.path.join(OUT_DIR, 'hog_illumination_invariance.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_illumination_invariance.png")


def create_block_vector_structure():
    """Visualize how block vector is constructed from cells"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Draw 2×2 cells
    cell_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    cell_labels = ['Cell (0,0)', 'Cell (1,0)', 'Cell (0,1)', 'Cell (1,1)']
    positions = [(1, 5), (3, 5), (1, 3), (3, 3)]
    
    for (x, y), color, label in zip(positions, cell_colors, cell_labels):
        rect = Rectangle((x, y), 1.8, 1.8, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.9, y + 0.9, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.text(2.9, 7.2, '2×2 Block', ha='center', fontsize=12, fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(6.5, 4.5), xytext=(5.2, 4.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(5.85, 5, 'Concatenate', ha='center', fontsize=10)
    
    # Block vector
    block_start_x = 7
    block_width = 0.15
    
    np.random.seed(42)
    for cell_idx in range(4):
        start = block_start_x + cell_idx * 9 * block_width
        for bin_idx in range(9):
            x = start + bin_idx * block_width
            height = np.random.uniform(0.5, 2)
            rect = Rectangle((x, 3), block_width * 0.9, height, 
                            facecolor=cell_colors[cell_idx], edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        
        bracket_start = start
        bracket_end = start + 9 * block_width
        ax.plot([bracket_start, bracket_start], [2.5, 2.7], 'k-', linewidth=1)
        ax.plot([bracket_end, bracket_end], [2.5, 2.7], 'k-', linewidth=1)
        ax.plot([bracket_start, bracket_end], [2.5, 2.5], 'k-', linewidth=1)
        ax.text((bracket_start + bracket_end) / 2, 2.1, f'9 bins', ha='center', fontsize=8)
    
    ax.text(block_start_x + 18 * block_width, 5.5, '36-D Block Vector', ha='center', fontsize=12, fontweight='bold')
    
    ax.text(10.5, 1.2, '4 cells × 9 bins = 36 dimensions per block', ha='center', fontsize=11,
            bbox=dict(facecolor='lightgray', edgecolor='black'))
    
    plt.suptitle('Block Vector Structure: From 2×2 Cells to 36-D Vector', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUT_DIR, 'hog_block_vector_structure.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_block_vector_structure.png")


def create_dimension_calculation():
    """Visual calculation of final HOG descriptor dimension"""
    gray, _ = load_image()
    h, w = gray.shape
    cell_size = 8
    cells_x = w // cell_size
    cells_y = h // cell_size
    blocks_x = cells_x - 1
    blocks_y = cells_y - 1
    total_dim = blocks_x * blocks_y * 36
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    calc = f"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║              HOG DESCRIPTOR DIMENSION CALCULATION                 ║
    ║                   ({w}×{h} Input Image)                           ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  STEP 1: Count Cells                                              ║
    ║  ─────────────────                                                ║
    ║    Image: {w} × {h} pixels                                        ║
    ║    Cell:  8 × 8 pixels                                            ║
    ║                                                                   ║
    ║    cells_x = {w} ÷ 8 = {cells_x} cells                            ║
    ║    cells_y = {h} ÷ 8 = {cells_y} cells                            ║
    ║    Total cells = {cells_x} × {cells_y} = {cells_x * cells_y} cells                              ║
    ║                                                                   ║
    ║  STEP 2: Count Blocks (with 50% overlap)                          ║
    ║  ────────────────────────────────────────                         ║
    ║    Block: 2 × 2 cells                                             ║
    ║    Stride: 1 cell                                                 ║
    ║                                                                   ║
    ║    blocks_x = cells_x - 1 = {cells_x} - 1 = {blocks_x} blocks     ║
    ║    blocks_y = cells_y - 1 = {cells_y} - 1 = {blocks_y} blocks     ║
    ║    Total blocks = {blocks_x} × {blocks_y} = {blocks_x * blocks_y} blocks                        ║
    ║                                                                   ║
    ║  STEP 3: Calculate Features per Block                             ║
    ║  ────────────────────────────────────                             ║
    ║    Cells per block = 2 × 2 = 4 cells                              ║
    ║    Bins per cell = 9 bins                                         ║
    ║    Features per block = 4 × 9 = 36 values                         ║
    ║                                                                   ║
    ║  STEP 4: Final Descriptor Dimension                               ║
    ║  ──────────────────────────────────                               ║
    ║                                                                   ║
    ║    ┌─────────────────────────────────────────────────────┐        ║
    ║    │  TOTAL = blocks × features_per_block                │        ║
    ║    │       = {blocks_x * blocks_y} × 36                                    │        ║
    ║    │       = {total_dim:,} dimensions                            │        ║
    ║    └─────────────────────────────────────────────────────┘        ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, calc, fontsize=10, family='monospace', va='center', ha='center',
            transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black', linewidth=2))
    
    plt.savefig(os.path.join(OUT_DIR, 'hog_dimension_calculation.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_dimension_calculation.png")


if __name__ == "__main__":
    print("Generating HOG Block Normalization Visualizations...")
    print("-" * 50)
    create_block_layout_visual()
    create_normalization_example()
    create_illumination_invariance_visual()
    create_block_vector_structure()
    create_dimension_calculation()
    print("-" * 50)
    print("Done!")

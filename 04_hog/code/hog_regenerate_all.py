"""
HOG - Regenerate ALL images using REAL input image
Replaces synthetic/plain images with real image-based visualizations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.colors import Normalize
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')
os.makedirs(OUT_DIR, exist_ok=True)


def load_image():
    """Load the real input image"""
    image_path = os.path.join(OUT_DIR, "input_image.jpg")
    img = np.array(Image.open(image_path))
    if len(img.shape) == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float64)
    gray = gray / 255.0
    return gray, img


def gamma_correction(img, gamma=0.5):
    return np.power(np.clip(img, 1e-8, 1), gamma)


def compute_gradients(img):
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) * 180 / np.pi % 180
    return magnitude, direction, gx, gy


def compute_cell_histograms(magnitude, direction, cell_size=8, num_bins=9):
    h, w = magnitude.shape
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
    
    return histograms


def block_normalize(histograms, block_size=2, eps=1e-6):
    cells_y, cells_x, num_bins = histograms.shape
    blocks_y = cells_y - block_size + 1
    blocks_x = cells_x - block_size + 1
    
    block_features = []
    for by in range(blocks_y):
        for bx in range(blocks_x):
            block = histograms[by:by+block_size, bx:bx+block_size, :].flatten()
            norm = np.sqrt(np.sum(block**2) + eps**2)
            block_features.append(block / norm)
    
    return np.array(block_features), (blocks_y, blocks_x)


# =============================================================================
# STEP 1: PREPROCESSING - REAL IMAGE
# =============================================================================

def create_step1_preprocessing():
    """Step 1 preprocessing with real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    ax.set_title(f'Input: RGB Image\n{original_rgb.shape[1]}×{original_rgb.shape[0]} pixels', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    ax.imshow(gray, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Step 1.1: Grayscale\nI = 0.299R + 0.587G + 0.114B', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[2]
    ax.imshow(gray_gamma, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Step 1.2: Gamma Correction (γ=0.5)\nI_γ = I^0.5', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 1: Preprocessing (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step1_preprocessing.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step1_preprocessing.png")


# =============================================================================
# STEP 2: GRADIENTS - REAL IMAGE
# =============================================================================

def create_step2_gradients():
    """Step 2 gradients with real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, gx, gy = compute_gradients(gray_gamma)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    ax = axes[0, 0]
    ax.imshow(gray_gamma, cmap='gray')
    ax.set_title('Input (After Gamma)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = axes[0, 1]
    im = ax.imshow(gx, cmap='RdBu', vmin=-0.4, vmax=0.4)
    ax.set_title('Gx: Horizontal Gradient\nGx = I(x+1,y) - I(x-1,y)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[0, 2]
    im = ax.imshow(gy, cmap='RdBu', vmin=-0.4, vmax=0.4)
    ax.set_title('Gy: Vertical Gradient\nGy = I(x,y+1) - I(x,y-1)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[1, 0]
    ax.imshow(original_rgb)
    ax.set_title('Original Image', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1, 1]
    im = ax.imshow(magnitude, cmap='hot')
    ax.set_title('Magnitude: M = √(Gx² + Gy²)\nBright = strong edges', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[1, 2]
    dir_masked = np.where(magnitude > 0.02, direction, np.nan)
    im = ax.imshow(dir_masked, cmap='hsv', vmin=0, vmax=180)
    ax.set_title('Direction: θ = arctan(Gy/Gx) mod 180°\nColor = edge orientation', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7, label='Degrees')
    
    plt.suptitle('HOG Step 2: Gradient Computation (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step2_gradients.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step2_gradients.png")


def create_gradient_computation():
    """Gradient computation detail with real image patch"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, gx, gy = compute_gradients(gray_gamma)
    
    # Select a 5x5 patch from real image
    py, px = 200, 300
    patch_size = 5
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Show patch location
    ax = axes[0, 0]
    ax.imshow(original_rgb)
    rect = Rectangle((px-20, py-20), 40, 40, fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(rect)
    ax.set_title('Patch Location on Real Image', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Intensity values
    ax = axes[0, 1]
    patch = gray_gamma[py:py+patch_size, px:px+patch_size]
    im = ax.imshow(patch, cmap='gray', interpolation='nearest')
    for i in range(patch_size):
        for j in range(patch_size):
            ax.text(j, i, f'{patch[i,j]:.2f}', ha='center', va='center', fontsize=9, 
                   color='white' if patch[i,j] < 0.5 else 'black')
    ax.set_title('Intensity Values (I)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Gx values
    ax = axes[0, 2]
    patch_gx = gx[py:py+patch_size, px:px+patch_size]
    im = ax.imshow(patch_gx, cmap='RdBu', vmin=-0.3, vmax=0.3, interpolation='nearest')
    for i in range(patch_size):
        for j in range(patch_size):
            ax.text(j, i, f'{patch_gx[i,j]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_title('Gx = I(x+1) - I(x-1)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    # Gy values
    ax = axes[1, 0]
    patch_gy = gy[py:py+patch_size, px:px+patch_size]
    im = ax.imshow(patch_gy, cmap='RdBu', vmin=-0.3, vmax=0.3, interpolation='nearest')
    for i in range(patch_size):
        for j in range(patch_size):
            ax.text(j, i, f'{patch_gy[i,j]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_title('Gy = I(y+1) - I(y-1)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    # Magnitude
    ax = axes[1, 1]
    patch_mag = magnitude[py:py+patch_size, px:px+patch_size]
    im = ax.imshow(patch_mag, cmap='hot', interpolation='nearest')
    for i in range(patch_size):
        for j in range(patch_size):
            ax.text(j, i, f'{patch_mag[i,j]:.2f}', ha='center', va='center', fontsize=9,
                   color='white' if patch_mag[i,j] > 0.15 else 'black')
    ax.set_title('M = √(Gx² + Gy²)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    # Direction
    ax = axes[1, 2]
    patch_dir = direction[py:py+patch_size, px:px+patch_size]
    im = ax.imshow(patch_dir, cmap='hsv', vmin=0, vmax=180, interpolation='nearest')
    for i in range(patch_size):
        for j in range(patch_size):
            ax.text(j, i, f'{patch_dir[i,j]:.0f}°', ha='center', va='center', fontsize=8)
    ax.set_title('θ = arctan(Gy/Gx) mod 180°', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    plt.suptitle('HOG: Gradient Computation on Real Image Patch', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_gradient_computation.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_gradient_computation.png")


def create_edge_detection():
    """Edge detection on real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax = axes[0, 0]
    ax.imshow(original_rgb)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.imshow(magnitude, cmap='gray')
    ax.set_title('Edge Magnitude (Grayscale)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1, 0]
    threshold = 0.08
    edges = magnitude > threshold
    ax.imshow(edges, cmap='gray')
    ax.set_title(f'Binary Edges (M > {threshold})', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1, 1]
    overlay = original_rgb.copy().astype(float) / 255
    overlay[edges, 1] = 1.0  # Green edges
    overlay[edges, 0] *= 0.5
    overlay[edges, 2] *= 0.5
    ax.imshow(np.clip(overlay, 0, 1))
    ax.set_title('Edges on Original (Green)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG: Edge Detection on Real Image', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_edge_detection.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_edge_detection.png")


def create_signed_vs_unsigned():
    """Signed vs unsigned gradients on real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, gx, gy = compute_gradients(gray_gamma)
    
    signed_direction = np.arctan2(gy, gx) * 180 / np.pi
    signed_direction = (signed_direction + 360) % 360
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    mask = magnitude > 0.03
    signed_masked = np.where(mask, signed_direction, np.nan)
    im = ax.imshow(signed_masked, cmap='hsv', vmin=0, vmax=360)
    ax.set_title('Signed Direction (0°-360°)\nDifferent colors for opposite edges', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7, label='Degrees')
    
    ax = axes[2]
    unsigned_masked = np.where(mask, direction, np.nan)
    im = ax.imshow(unsigned_masked, cmap='hsv', vmin=0, vmax=180)
    ax.set_title('Unsigned Direction (0°-180°)\nOpposite edges same color', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7, label='Degrees')
    
    plt.suptitle('HOG: Signed vs Unsigned Gradients (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_signed_vs_unsigned.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_signed_vs_unsigned.png")


# =============================================================================
# STEP 3: CELL HISTOGRAMS - REAL IMAGE
# =============================================================================

def create_step3_cell_division():
    """Cell division on real image"""
    gray, original_rgb = load_image()
    
    cell_size = 8
    h, w = gray.shape
    cells_y = h // cell_size
    cells_x = w // cell_size
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    ax.set_title(f'Original: {w}×{h} pixels', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    ax.imshow(original_rgb, alpha=0.7)
    for cy in range(cells_y + 1):
        ax.axhline(y=cy * cell_size, color='lime', linewidth=0.5, alpha=0.7)
    for cx in range(cells_x + 1):
        ax.axvline(x=cx * cell_size, color='lime', linewidth=0.5, alpha=0.7)
    
    # Highlight some cells
    highlights = [(20, 30), (40, 50), (30, 60)]
    colors = ['red', 'blue', 'orange']
    for (cy, cx), color in zip(highlights, colors):
        rect = Rectangle((cx*cell_size, cy*cell_size), cell_size, cell_size,
                         fill=False, edgecolor=color, linewidth=3)
        ax.add_patch(rect)
    
    ax.set_title(f'Divided: {cells_x}×{cells_y} = {cells_x*cells_y} cells\nEach cell: 8×8 pixels', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 3.1: Cell Division (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step3_1_cell_division.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step3_1_cell_division.png")


def create_step3_single_cell():
    """Single cell analysis on real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, gx, gy = compute_gradients(gray_gamma)
    
    cell_size = 8
    cell_y, cell_x = 40, 50
    y_start = cell_y * cell_size
    x_start = cell_x * cell_size
    
    cell_img = original_rgb[y_start:y_start+cell_size, x_start:x_start+cell_size]
    cell_mag = magnitude[y_start:y_start+cell_size, x_start:x_start+cell_size]
    cell_dir = direction[y_start:y_start+cell_size, x_start:x_start+cell_size]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    rect = Rectangle((x_start, y_start), cell_size, cell_size, fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(rect)
    ax.set_title(f'Cell ({cell_x}, {cell_y}) Location', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    ax.imshow(cell_img, interpolation='nearest')
    ax.set_title('Cell Content (8×8)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = axes[2]
    im = ax.imshow(cell_mag, cmap='hot', interpolation='nearest')
    ax.set_title('Cell Magnitude', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[3]
    im = ax.imshow(cell_dir, cmap='hsv', vmin=0, vmax=180, interpolation='nearest')
    ax.set_title('Cell Direction (0°-180°)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    plt.suptitle('HOG Step 3.2: Single Cell Analysis (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step3_2_single_cell.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step3_2_single_cell.png")


def create_step3_histogram_bins():
    """Histogram bins with real cell"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cell_y, cell_x = 40, 50
    cell_size = 8
    hist = histograms[cell_y, cell_x]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Cell image
    ax = axes[0]
    y_start = cell_y * cell_size
    x_start = cell_x * cell_size
    cell_img = original_rgb[y_start:y_start+cell_size, x_start:x_start+cell_size]
    ax.imshow(cell_img, interpolation='nearest')
    ax.set_title(f'Cell ({cell_x}, {cell_y})', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Histogram
    ax = axes[1]
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    bars = ax.bar(range(9), hist, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(9))
    ax.set_xticklabels([f'{i*20}°-{(i+1)*20}°' for i in range(9)], rotation=45, ha='right')
    ax.set_xlabel('Orientation Bin')
    ax.set_ylabel('Vote Magnitude')
    ax.set_title('9-Bin Histogram (Real Data)', fontsize=12, fontweight='bold')
    
    dominant = np.argmax(hist)
    bars[dominant].set_edgecolor('red')
    bars[dominant].set_linewidth(3)
    ax.annotate(f'Dominant: {dominant*20}°-{(dominant+1)*20}°', 
               xy=(dominant, hist[dominant]), xytext=(dominant+1, hist[dominant]*1.1),
               fontsize=10, fontweight='bold', color='red',
               arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.suptitle('HOG Step 3.3: 9-Bin Histogram (Real Cell)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step3_3_histogram_bins.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step3_3_histogram_bins.png")


def create_step3_voting():
    """Voting process with real cell data"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    
    cell_size = 8
    cell_y, cell_x = 40, 50
    y_start = cell_y * cell_size
    x_start = cell_x * cell_size
    
    cell_mag = magnitude[y_start:y_start+cell_size, x_start:x_start+cell_size]
    cell_dir = direction[y_start:y_start+cell_size, x_start:x_start+cell_size]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Cell content
    ax = axes[0, 0]
    cell_img = original_rgb[y_start:y_start+cell_size, x_start:x_start+cell_size]
    ax.imshow(cell_img, interpolation='nearest')
    ax.set_title(f'Cell ({cell_x}, {cell_y})', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Magnitude with values
    ax = axes[0, 1]
    im = ax.imshow(cell_mag, cmap='hot', interpolation='nearest')
    for i in range(cell_size):
        for j in range(cell_size):
            color = 'white' if cell_mag[i,j] > 0.1 else 'black'
            ax.text(j, i, f'{cell_mag[i,j]:.2f}', ha='center', va='center', fontsize=7, color=color)
    ax.set_title('Magnitude (Vote Weight)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Direction with values
    ax = axes[1, 0]
    im = ax.imshow(cell_dir, cmap='hsv', vmin=0, vmax=180, interpolation='nearest')
    for i in range(cell_size):
        for j in range(cell_size):
            ax.text(j, i, f'{cell_dir[i,j]:.0f}', ha='center', va='center', fontsize=7)
    ax.set_title('Direction (Bin Selection)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Resulting histogram
    ax = axes[1, 1]
    hist = np.zeros(9)
    bin_width = 20.0
    for i in range(cell_size):
        for j in range(cell_size):
            m = cell_mag[i, j]
            d = cell_dir[i, j]
            bin_idx = d / bin_width
            lower_bin = int(bin_idx) % 9
            upper_bin = (lower_bin + 1) % 9
            upper_w = bin_idx - int(bin_idx)
            lower_w = 1 - upper_w
            hist[lower_bin] += m * lower_w
            hist[upper_bin] += m * upper_w
    
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    ax.bar(range(9), hist, color=colors, edgecolor='black')
    ax.set_xticks(range(9))
    ax.set_xticklabels([f'{i*20}°' for i in range(9)])
    ax.set_title('Resulting Histogram\n(Bilinear Interpolation)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Vote Sum')
    
    plt.suptitle('HOG Step 3.4: Voting Process (Real Cell Data)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step3_4_voting.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step3_4_voting.png")


def create_step3_all_cells():
    """All cells visualization on real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cells_y, cells_x = histograms.shape[:2]
    cell_size = 8
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original with grid
    ax = axes[0]
    ax.imshow(original_rgb)
    for cy in range(0, cells_y + 1, 10):
        ax.axhline(y=cy * cell_size, color='lime', linewidth=0.5)
    for cx in range(0, cells_x + 1, 10):
        ax.axvline(x=cx * cell_size, color='lime', linewidth=0.5)
    ax.set_title(f'Original with Grid\n{cells_x}×{cells_y} = {cells_x*cells_y} cells', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # HOG visualization
    ax = axes[1]
    ax.set_xlim(0, gray.shape[1])
    ax.set_ylim(gray.shape[0], 0)
    ax.set_facecolor('black')
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = cx * cell_size + cell_size / 2
            
            max_val = hist.max() if hist.max() > 0 else 1
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.2:
                    angle = bin_idx * 20 + 10
                    scale = (hist[bin_idx] / max_val) * (cell_size / 2) * 0.9
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax.plot([center_x - dx, center_x + dx],
                           [center_y - dy, center_y + dy],
                           color='white', linewidth=0.6, alpha=0.8)
    
    ax.set_title(f'HOG Visualization\n{cells_x*cells_y} cell histograms', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.suptitle('HOG Step 3.5: All Cell Histograms (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step3_5_all_cells.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step3_5_all_cells.png")


def create_step3_compression():
    """Compression achieved with real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cells_y, cells_x = histograms.shape[:2]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    input_size = gray.shape[0] * gray.shape[1] * 2
    ax.set_title(f'Input Gradients\n{gray.shape[1]}×{gray.shape[0]}×2 = {input_size:,} values', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    ax.set_xlim(0, gray.shape[1])
    ax.set_ylim(gray.shape[0], 0)
    ax.set_facecolor('black')
    
    cell_size = 8
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = cx * cell_size + cell_size / 2
            max_val = hist.max() if hist.max() > 0 else 1
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.2:
                    angle = bin_idx * 20 + 10
                    scale = (hist[bin_idx] / max_val) * (cell_size / 2) * 0.9
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax.plot([center_x - dx, center_x + dx], [center_y - dy, center_y + dy],
                           color='white', linewidth=0.6)
    
    output_size = cells_x * cells_y * 9
    ax.set_title(f'Cell Histograms\n{cells_x}×{cells_y}×9 = {output_size:,} values', 
                fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    ax = axes[2]
    ax.axis('off')
    ratio = input_size / output_size
    stats = f"""
    ╔═══════════════════════════════════╗
    ║     COMPRESSION ACHIEVED          ║
    ╠═══════════════════════════════════╣
    ║                                   ║
    ║  INPUT:  {input_size:>7,} gradient values  ║
    ║                                   ║
    ║  OUTPUT: {output_size:>7,} histogram values ║
    ║                                   ║
    ║  RATIO:  {ratio:>6.1f}:1                 ║
    ║                                   ║
    ╠═══════════════════════════════════╣
    ║  PRESERVED:                       ║
    ║  • Edge orientations              ║
    ║  • Edge strengths                 ║
    ║  • Spatial layout                 ║
    ╚═══════════════════════════════════╝
    """
    ax.text(0.1, 0.5, stats, fontsize=11, family='monospace', va='center',
           transform=ax.transAxes, bbox=dict(facecolor='lightgreen', edgecolor='darkgreen'))
    
    plt.suptitle('HOG Step 3.6: Compression (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step3_6_compression.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step3_6_compression.png")


def create_step3_complete():
    """Step 3 complete summary with real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cells_y, cells_x = histograms.shape[:2]
    cell_size = 8
    
    fig = plt.figure(figsize=(20, 10))
    
    # Original
    ax = fig.add_subplot(2, 4, 1)
    ax.imshow(original_rgb)
    ax.set_title('Input Image', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Magnitude
    ax = fig.add_subplot(2, 4, 2)
    ax.imshow(magnitude, cmap='hot')
    ax.set_title('Gradient Magnitude', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Direction
    ax = fig.add_subplot(2, 4, 3)
    ax.imshow(direction, cmap='hsv', vmin=0, vmax=180)
    ax.set_title('Gradient Direction', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Cell grid
    ax = fig.add_subplot(2, 4, 4)
    ax.imshow(gray_gamma, cmap='gray')
    for cy in range(0, cells_y + 1, 5):
        ax.axhline(y=cy * cell_size, color='lime', linewidth=0.3)
    for cx in range(0, cells_x + 1, 5):
        ax.axvline(x=cx * cell_size, color='lime', linewidth=0.3)
    ax.set_title(f'{cells_x*cells_y} Cells', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Sample histogram
    ax = fig.add_subplot(2, 4, 5)
    hist = histograms[cells_y//2, cells_x//2]
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    ax.bar(range(9), hist, color=colors, edgecolor='black')
    ax.set_title('Sample Cell Histogram', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 4, 8])
    ax.set_xticklabels(['0°', '80°', '160°'])
    
    # HOG visualization
    ax = fig.add_subplot(2, 4, 6)
    ax.set_xlim(0, gray.shape[1])
    ax.set_ylim(gray.shape[0], 0)
    ax.set_facecolor('black')
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = cx * cell_size + cell_size / 2
            max_val = hist.max() if hist.max() > 0 else 1
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.2:
                    angle = bin_idx * 20 + 10
                    scale = (hist[bin_idx] / max_val) * (cell_size / 2) * 0.8
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax.plot([center_x - dx, center_x + dx], [center_y - dy, center_y + dy],
                           color='white', linewidth=0.5)
    ax.set_title('HOG Visualization', fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    
    # HOG on image
    ax = fig.add_subplot(2, 4, 7)
    ax.imshow(original_rgb, alpha=0.5)
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = cx * cell_size + cell_size / 2
            max_val = hist.max() if hist.max() > 0 else 1
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.25:
                    angle = bin_idx * 20 + 10
                    scale = (hist[bin_idx] / max_val) * (cell_size / 2.5)
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax.plot([center_x - dx, center_x + dx], [center_y - dy, center_y + dy],
                           color='red', linewidth=0.6)
    ax.set_title('HOG on Image', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Stats
    ax = fig.add_subplot(2, 4, 8)
    ax.axis('off')
    stats = f"""Step 3 Output:
    
Cells: {cells_x} × {cells_y} = {cells_x*cells_y}
Bins per cell: 9
Total values: {cells_x*cells_y*9:,}

Next: Block Normalization"""
    ax.text(0.1, 0.5, stats, fontsize=11, family='monospace', va='center',
           transform=ax.transAxes, bbox=dict(facecolor='lightyellow', edgecolor='orange'))
    
    plt.suptitle('HOG Step 3 Complete: Cell Histograms (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step3_complete.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step3_complete.png")


def create_step3_cell_histograms():
    """Cell histograms detail with real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cell_positions = [(20, 25), (35, 45), (50, 60), (25, 70)]
    cell_size = 8
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for idx, (cy, cx) in enumerate(cell_positions):
        y_start = cy * cell_size
        x_start = cx * cell_size
        cell_img = original_rgb[y_start:y_start+cell_size, x_start:x_start+cell_size]
        hist = histograms[cy, cx]
        
        ax = axes[0, idx]
        ax.imshow(cell_img, interpolation='nearest')
        ax.set_title(f'Cell ({cx}, {cy})', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        ax = axes[1, idx]
        colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
        ax.bar(range(9), hist, color=colors, edgecolor='black')
        ax.set_xticks([0, 4, 8])
        ax.set_xticklabels(['0°', '80°', '160°'])
        dominant = np.argmax(hist)
        ax.set_title(f'Peak: Bin {dominant} ({dominant*20}°)', fontsize=9)
    
    plt.suptitle('HOG: Cell Histograms (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step3_cell_histograms.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step3_cell_histograms.png")


def create_cell_histogram_detail():
    """Detailed cell histogram with real data"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cell_y, cell_x = 40, 50
    cell_size = 8
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Cell in context
    ax = axes[0]
    ax.imshow(original_rgb)
    y_start = cell_y * cell_size
    x_start = cell_x * cell_size
    rect = Rectangle((x_start-2, y_start-2), cell_size+4, cell_size+4, 
                     fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(rect)
    ax.set_xlim(x_start-50, x_start+50)
    ax.set_ylim(y_start+50, y_start-50)
    ax.set_title(f'Cell ({cell_x}, {cell_y}) in Context', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Cell content
    ax = axes[1]
    cell_img = original_rgb[y_start:y_start+cell_size, x_start:x_start+cell_size]
    ax.imshow(cell_img, interpolation='nearest')
    ax.set_title('Cell Content (8×8 = 64 pixels)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Histogram
    ax = axes[2]
    hist = histograms[cell_y, cell_x]
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    bars = ax.bar(range(9), hist, color=colors, edgecolor='black', linewidth=2)
    ax.set_xticks(range(9))
    ax.set_xticklabels([f'{i*20}°' for i in range(9)])
    ax.set_xlabel('Orientation Bin', fontsize=11)
    ax.set_ylabel('Vote Magnitude', fontsize=11)
    ax.set_title('9-Bin Histogram (Real Data)', fontsize=12, fontweight='bold')
    
    for i, v in enumerate(hist):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    
    plt.suptitle('HOG: Cell Histogram Detail (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_cell_histogram_detail.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_cell_histogram_detail.png")


def create_multiple_cells():
    """Multiple cells comparison with real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cell_positions = [(15, 20), (25, 40), (35, 60), (45, 30), (50, 70), (20, 55)]
    cell_size = 8
    
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    
    # Row 0: Location on image
    ax = axes[0, 0]
    ax.imshow(original_rgb)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    for (cy, cx), color in zip(cell_positions, colors):
        rect = Rectangle((cx*cell_size, cy*cell_size), cell_size, cell_size,
                         fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
    ax.set_title('Cell Locations', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    for idx in range(1, 6):
        axes[0, idx].axis('off')
    
    # Row 1: Cell images
    for idx, ((cy, cx), color) in enumerate(zip(cell_positions, colors)):
        y_start = cy * cell_size
        x_start = cx * cell_size
        cell_img = original_rgb[y_start:y_start+cell_size, x_start:x_start+cell_size]
        
        ax = axes[1, idx]
        ax.imshow(cell_img, interpolation='nearest')
        ax.set_title(f'({cx},{cy})', fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
    
    # Row 2: Histograms
    for idx, ((cy, cx), color) in enumerate(zip(cell_positions, colors)):
        hist = histograms[cy, cx]
        
        ax = axes[2, idx]
        hist_colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
        ax.bar(range(9), hist, color=hist_colors, edgecolor='black', width=0.8)
        ax.set_xticks([0, 4, 8])
        ax.set_xticklabels(['0°', '80°', '160°'], fontsize=8)
        dominant = np.argmax(hist)
        ax.set_title(f'Peak: {dominant*20}°', fontsize=9)
    
    plt.suptitle('HOG: Multiple Cells Comparison (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_multiple_cells.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_multiple_cells.png")


def create_interpolation_detail():
    """Bilinear interpolation with real cell data"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    
    cell_size = 8
    cell_y, cell_x = 40, 50
    y_start = cell_y * cell_size
    x_start = cell_x * cell_size
    
    cell_mag = magnitude[y_start:y_start+cell_size, x_start:x_start+cell_size]
    cell_dir = direction[y_start:y_start+cell_size, x_start:x_start+cell_size]
    
    # Find pixels with interesting angles
    pixels = []
    for i in range(cell_size):
        for j in range(cell_size):
            if cell_mag[i, j] > 0.02:
                pixels.append((i, j, cell_mag[i, j], cell_dir[i, j]))
    
    pixels = sorted(pixels, key=lambda x: -x[2])[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, (py, px, mag, angle) in enumerate(pixels):
        ax = axes[idx // 2, idx % 2]
        
        bin_idx = angle / 20
        lower_bin = int(bin_idx) % 9
        upper_bin = (lower_bin + 1) % 9
        upper_w = bin_idx - int(bin_idx)
        lower_w = 1 - upper_w
        
        colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
        bars = ax.bar(range(9), np.zeros(9), color=colors, edgecolor='black', alpha=0.3)
        
        bars[lower_bin].set_height(mag * lower_w)
        bars[lower_bin].set_alpha(1.0)
        bars[upper_bin].set_height(mag * upper_w)
        bars[upper_bin].set_alpha(1.0)
        
        ax.set_xticks(range(9))
        ax.set_xticklabels([f'{i*20}°' for i in range(9)], fontsize=8)
        ax.set_title(f'Pixel ({px},{py}): θ={angle:.1f}°, M={mag:.3f}\n'
                    f'Bin {lower_bin}: {mag*lower_w:.3f}, Bin {upper_bin}: {mag*upper_w:.3f}',
                    fontsize=10, fontweight='bold')
        ax.set_ylabel('Vote')
        
        ax.axvline(x=bin_idx, color='red', linestyle='--', linewidth=2, label=f'θ/20={bin_idx:.2f}')
        ax.legend(fontsize=8)
    
    plt.suptitle('HOG: Bilinear Interpolation (Real Cell Data)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_interpolation_detail.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_interpolation_detail.png")


# =============================================================================
# STEP 4: BLOCK NORMALIZATION - REAL IMAGE
# =============================================================================

def create_step4_block_definition():
    """Block definition with real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cell_size = 8
    cells_y, cells_x = histograms.shape[:2]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    
    # Show some blocks
    block_positions = [(10, 20), (30, 40), (45, 60)]
    colors = ['red', 'blue', 'green']
    for (by, bx), color in zip(block_positions, colors):
        rect = Rectangle((bx*cell_size, by*cell_size), cell_size*2, cell_size*2,
                         fill=False, edgecolor=color, linewidth=3)
        ax.add_patch(rect)
    
    ax.set_title('Blocks on Image\nEach block = 2×2 cells = 16×16 pixels', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Block structure diagram
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    cell_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    positions = [(2, 5), (5, 5), (2, 2), (5, 2)]
    labels = ['Cell(0,0)\n9 bins', 'Cell(1,0)\n9 bins', 'Cell(0,1)\n9 bins', 'Cell(1,1)\n9 bins']
    
    for (x, y), color, label in zip(positions, cell_colors, labels):
        rect = Rectangle((x, y), 2.5, 2.5, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1.25, y + 1.25, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    rect = Rectangle((2, 2), 5.5, 5.5, fill=False, edgecolor='red', linewidth=4)
    ax.add_patch(rect)
    ax.text(4.75, 8, '2×2 Block = 4×9 = 36 values', fontsize=12, ha='center', fontweight='bold', color='red')
    
    ax.set_title('Block Structure', fontsize=12, fontweight='bold')
    
    plt.suptitle('HOG Step 4.1: Block Definition (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step4_1_block_definition.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step4_1_block_definition.png")


def create_step4_overlap():
    """Block overlap with real image"""
    gray, original_rgb = load_image()
    
    cell_size = 8
    h, w = gray.shape
    cells_y = h // cell_size
    cells_x = w // cell_size
    blocks_y = cells_y - 1
    blocks_x = cells_x - 1
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    
    # Show overlapping blocks in a small region
    start_by, start_bx = 20, 30
    colors = plt.cm.Set1(np.linspace(0, 1, 6))
    for i in range(6):
        by = start_by + i // 2
        bx = start_bx + i % 3
        rect = Rectangle((bx*cell_size, by*cell_size), cell_size*2, cell_size*2,
                         fill=True, facecolor=colors[i], alpha=0.4, edgecolor=colors[i], linewidth=2)
        ax.add_patch(rect)
    
    ax.set_xlim(start_bx*cell_size - 10, (start_bx+5)*cell_size + 10)
    ax.set_ylim((start_by+4)*cell_size + 10, start_by*cell_size - 10)
    ax.set_title('50% Overlap: Adjacent blocks share cells', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    ax.axis('off')
    stats = f"""
    Block Overlap Statistics:
    
    Image:    {w} × {h} pixels
    Cells:    {cells_x} × {cells_y} = {cells_x*cells_y}
    
    Block size:  2×2 cells
    Stride:      1 cell (50% overlap)
    
    Blocks:   {blocks_x} × {blocks_y} = {blocks_x*blocks_y}
    
    Each interior cell appears in 4 blocks!
    """
    ax.text(0.1, 0.5, stats, fontsize=12, family='monospace', va='center',
           transform=ax.transAxes, bbox=dict(facecolor='lightcyan', edgecolor='blue'))
    
    plt.suptitle('HOG Step 4.2: 50% Block Overlap (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step4_2_overlap.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step4_2_overlap.png")


def create_step4_l2_normalization():
    """L2 normalization with real block data"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    by, bx = 40, 50
    block_hists = []
    for dy in range(2):
        for dx in range(2):
            block_hists.append(histograms[by + dy, bx + dx])
    block_vector = np.concatenate(block_hists)
    
    norm = np.sqrt(np.sum(block_vector**2) + 1e-12)
    block_normalized = block_vector / norm
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax = axes[0]
    cell_colors = np.tile(plt.cm.tab10(np.arange(4) / 4), (9, 1)).T.flatten().reshape(-1, 4)
    ax.bar(range(36), block_vector, color=cell_colors[:36], edgecolor='black', width=0.8)
    ax.set_title(f'Before: L2 Norm = {norm:.3f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index (4 cells × 9 bins)')
    ax.set_ylabel('Value')
    
    ax = axes[1]
    ax.bar(range(36), block_normalized, color=cell_colors[:36], edgecolor='black', width=0.8)
    ax.set_title('After: L2 Norm ≈ 1.0', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index')
    ax.set_ylabel('Normalized Value')
    
    plt.suptitle(f'HOG Step 4.3: L2 Normalization - Block ({bx}, {by}) (Real Data)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step4_3_l2_normalization.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step4_3_l2_normalization.png")


def create_step4_illumination():
    """Illumination invariance with real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    # Simulate different brightness
    bright_factor = 1.5
    dark_factor = 0.5
    
    by, bx = 40, 50
    block_hists = []
    for dy in range(2):
        for dx in range(2):
            block_hists.append(histograms[by + dy, bx + dx])
    block_vector = np.concatenate(block_hists)
    
    block_bright = block_vector * bright_factor
    block_dark = block_vector * dark_factor
    
    norm_orig = np.sqrt(np.sum(block_vector**2) + 1e-12)
    norm_bright = np.sqrt(np.sum(block_bright**2) + 1e-12)
    norm_dark = np.sqrt(np.sum(block_dark**2) + 1e-12)
    
    normalized_orig = block_vector / norm_orig
    normalized_bright = block_bright / norm_bright
    normalized_dark = block_dark / norm_dark
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Original
    ax = axes[0, 0]
    ax.bar(range(36), block_vector, color='blue', width=0.8)
    ax.set_title(f'Original (×1.0)\nL2={norm_orig:.2f}', fontsize=11, fontweight='bold')
    
    ax = axes[1, 0]
    ax.bar(range(36), normalized_orig, color='blue', width=0.8)
    ax.set_title('Normalized', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 0.5)
    
    # Bright
    ax = axes[0, 1]
    ax.bar(range(36), block_bright, color='orange', width=0.8)
    ax.set_title(f'Bright (×{bright_factor})\nL2={norm_bright:.2f}', fontsize=11, fontweight='bold')
    
    ax = axes[1, 1]
    ax.bar(range(36), normalized_bright, color='orange', width=0.8)
    ax.set_title('Normalized (same!)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 0.5)
    
    # Dark
    ax = axes[0, 2]
    ax.bar(range(36), block_dark, color='green', width=0.8)
    ax.set_title(f'Dark (×{dark_factor})\nL2={norm_dark:.2f}', fontsize=11, fontweight='bold')
    
    ax = axes[1, 2]
    ax.bar(range(36), normalized_dark, color='green', width=0.8)
    ax.set_title('Normalized (same!)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 0.5)
    
    plt.suptitle('HOG Step 4.4: Illumination Invariance (Real Block Data)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step4_4_illumination.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step4_4_illumination.png")


def create_step4_final():
    """Final descriptor from real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    block_features, block_shape = block_normalize(histograms)
    descriptor = block_features.flatten()
    
    blocks_y, blocks_x = block_shape
    cells_y, cells_x = histograms.shape[:2]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax = axes[0, 0]
    ax.imshow(original_rgb)
    ax.set_title('Input Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.bar(range(min(400, len(descriptor))), descriptor[:400], color='darkblue', width=1)
    ax.set_title(f'Descriptor (first 400 of {len(descriptor):,})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index')
    
    ax = axes[1, 0]
    ax.hist(descriptor, bins=50, color='green', edgecolor='black', alpha=0.7)
    ax.set_title('Value Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    
    ax = axes[1, 1]
    ax.axis('off')
    stats = f"""
    Final HOG Descriptor:
    
    Image:      {gray.shape[1]} × {gray.shape[0]}
    Cells:      {cells_x} × {cells_y} = {cells_x*cells_y}
    Blocks:     {blocks_x} × {blocks_y} = {blocks_x*blocks_y}
    
    Descriptor: {len(descriptor):,} dimensions
    
    Min:  {descriptor.min():.4f}
    Max:  {descriptor.max():.4f}
    Mean: {descriptor.mean():.4f}
    """
    ax.text(0.1, 0.5, stats, fontsize=12, family='monospace', va='center',
           transform=ax.transAxes, bbox=dict(facecolor='lightyellow', edgecolor='orange'))
    
    plt.suptitle('HOG Step 4.5: Final Descriptor (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step4_5_final.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step4_5_final.png")


def create_step4_complete():
    """Step 4 complete summary"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    block_features, block_shape = block_normalize(histograms)
    descriptor = block_features.flatten()
    
    cells_y, cells_x = histograms.shape[:2]
    blocks_y, blocks_x = block_shape
    cell_size = 8
    
    fig = plt.figure(figsize=(20, 10))
    
    ax = fig.add_subplot(2, 4, 1)
    ax.imshow(original_rgb)
    ax.set_title('Input', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(2, 4, 2)
    ax.imshow(gray_gamma, cmap='gray')
    for cy in range(0, cells_y+1, 10):
        ax.axhline(y=cy*cell_size, color='lime', linewidth=0.3)
    for cx in range(0, cells_x+1, 10):
        ax.axvline(x=cx*cell_size, color='lime', linewidth=0.3)
    ax.set_title(f'{cells_x*cells_y} Cells', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(2, 4, 3)
    ax.imshow(original_rgb, alpha=0.5)
    for by in range(0, blocks_y, 5):
        for bx in range(0, blocks_x, 5):
            rect = Rectangle((bx*cell_size, by*cell_size), cell_size*2, cell_size*2,
                            fill=False, edgecolor='red', linewidth=0.5)
            ax.add_patch(rect)
    ax.set_title(f'{blocks_x*blocks_y} Blocks', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(2, 4, 4)
    sample_block = block_features[len(block_features)//2]
    colors = np.tile(plt.cm.tab10(np.arange(4)/4), (9,1)).T.flatten().reshape(-1,4)
    ax.bar(range(36), sample_block, color=colors[:36], width=0.8)
    ax.set_title('Sample Block (36-D)', fontsize=10, fontweight='bold')
    
    ax = fig.add_subplot(2, 4, 5)
    ax.set_xlim(0, gray.shape[1])
    ax.set_ylim(gray.shape[0], 0)
    ax.set_facecolor('black')
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size/2
            center_x = cx * cell_size + cell_size/2
            max_val = hist.max() if hist.max() > 0 else 1
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.2:
                    angle = bin_idx * 20 + 10
                    scale = (hist[bin_idx]/max_val) * (cell_size/2) * 0.8
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax.plot([center_x-dx, center_x+dx], [center_y-dy, center_y+dy],
                           color='white', linewidth=0.5)
    ax.set_title('HOG Visualization', fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    
    ax = fig.add_subplot(2, 4, 6)
    ax.bar(range(min(200, len(descriptor))), descriptor[:200], color='darkblue', width=1)
    ax.set_title(f'Descriptor ({len(descriptor):,}-D)', fontsize=10, fontweight='bold')
    
    ax = fig.add_subplot(2, 4, 7)
    ax.hist(descriptor, bins=30, color='green', edgecolor='black')
    ax.set_title('Distribution', fontsize=10, fontweight='bold')
    
    ax = fig.add_subplot(2, 4, 8)
    ax.axis('off')
    stats = f"""Output:
    
{blocks_x}×{blocks_y} = {blocks_x*blocks_y} blocks
× 36 values/block
= {len(descriptor):,} dimensions

Illumination Invariant ✓"""
    ax.text(0.1, 0.5, stats, fontsize=10, family='monospace', va='center',
           transform=ax.transAxes, bbox=dict(facecolor='lightgreen', edgecolor='darkgreen'))
    
    plt.suptitle('HOG Step 4 Complete: Block Normalization (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step4_complete.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step4_complete.png")


def create_block_normalization():
    """Block normalization overview with real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    block_features, block_shape = block_normalize(histograms)
    
    cells_y, cells_x = histograms.shape[:2]
    blocks_y, blocks_x = block_shape
    cell_size = 8
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    for by in range(0, blocks_y, 3):
        for bx in range(0, blocks_x, 3):
            rect = Rectangle((bx*cell_size, by*cell_size), cell_size*2, cell_size*2,
                            fill=False, edgecolor='red', linewidth=0.8)
            ax.add_patch(rect)
    ax.set_title(f'{blocks_x}×{blocks_y} = {blocks_x*blocks_y} Blocks', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    block_idx = len(block_features) // 2
    colors = np.tile(plt.cm.tab10(np.arange(4)/4), (9,1)).T.flatten().reshape(-1,4)
    ax.bar(range(36), block_features[block_idx], color=colors[:36], edgecolor='black', width=0.8)
    ax.set_title('Normalized Block Vector (36-D)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index')
    
    ax = axes[2]
    descriptor = block_features.flatten()
    ax.bar(range(min(300, len(descriptor))), descriptor[:300], color='darkgreen', width=1)
    ax.set_title(f'Final Descriptor (first 300 of {len(descriptor):,})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index')
    
    plt.suptitle('HOG: Block Normalization (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_step4_block_normalization.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_step4_block_normalization.png")


# =============================================================================
# FINAL VISUALIZATIONS - REAL IMAGE
# =============================================================================

def create_visualization():
    """Main HOG visualization with real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cells_y, cells_x = histograms.shape[:2]
    cell_size = 8
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    ax.set_title('Original Image', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    ax.set_xlim(0, gray.shape[1])
    ax.set_ylim(gray.shape[0], 0)
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
                           color='white', linewidth=0.7, alpha=alpha)
    
    ax.set_title('HOG Visualization', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.suptitle('HOG: Histogram of Oriented Gradients (Real Image)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_visualization.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_visualization.png")


def create_visualization_explained():
    """HOG visualization explained with real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cells_y, cells_x = histograms.shape[:2]
    cell_size = 8
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    ax.imshow(original_rgb, alpha=0.4)
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = cx * cell_size + cell_size / 2
            max_val = hist.max() if hist.max() > 0 else 1
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.25:
                    angle = bin_idx * 20 + 10
                    scale = (hist[bin_idx] / max_val) * (cell_size / 2.5)
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax.plot([center_x - dx, center_x + dx], [center_y - dy, center_y + dy],
                           color='red', linewidth=0.8)
    ax.set_title('HOG on Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[2]
    ax.set_xlim(0, gray.shape[1])
    ax.set_ylim(gray.shape[0], 0)
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
                    ax.plot([center_x - dx, center_x + dx], [center_y - dy, center_y + dy],
                           color='white', linewidth=0.7)
    ax.set_title('HOG Only', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.suptitle('HOG Visualization Explained (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_visualization_explained.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_visualization_explained.png")


def create_complete_summary():
    """Complete HOG pipeline summary with real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, gx, gy = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    block_features, block_shape = block_normalize(histograms)
    descriptor = block_features.flatten()
    
    cells_y, cells_x = histograms.shape[:2]
    blocks_y, blocks_x = block_shape
    cell_size = 8
    
    fig = plt.figure(figsize=(24, 16))
    
    # Row 1
    ax = fig.add_subplot(3, 4, 1)
    ax.imshow(original_rgb)
    ax.set_title('1. Input Image', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(3, 4, 2)
    ax.imshow(gray, cmap='gray')
    ax.set_title('2. Grayscale', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(3, 4, 3)
    ax.imshow(gray_gamma, cmap='gray')
    ax.set_title('3. Gamma (γ=0.5)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(3, 4, 4)
    ax.imshow(gx, cmap='RdBu', vmin=-0.3, vmax=0.3)
    ax.set_title('4. Gx', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Row 2
    ax = fig.add_subplot(3, 4, 5)
    ax.imshow(gy, cmap='RdBu', vmin=-0.3, vmax=0.3)
    ax.set_title('5. Gy', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(3, 4, 6)
    ax.imshow(magnitude, cmap='hot')
    ax.set_title('6. Magnitude', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(3, 4, 7)
    ax.imshow(direction, cmap='hsv', vmin=0, vmax=180)
    ax.set_title('7. Direction', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(3, 4, 8)
    ax.imshow(gray_gamma, cmap='gray')
    for cy in range(0, cells_y+1, 5):
        ax.axhline(y=cy*cell_size, color='lime', linewidth=0.3)
    for cx in range(0, cells_x+1, 5):
        ax.axvline(x=cx*cell_size, color='lime', linewidth=0.3)
    ax.set_title(f'8. {cells_x*cells_y} Cells', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Row 3
    ax = fig.add_subplot(3, 4, 9)
    hist = histograms[cells_y//2, cells_x//2]
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    ax.bar(range(9), hist, color=colors, edgecolor='black')
    ax.set_title('9. Cell Histogram', fontsize=11, fontweight='bold')
    ax.set_xticks([0, 4, 8])
    
    ax = fig.add_subplot(3, 4, 10)
    ax.set_xlim(0, gray.shape[1])
    ax.set_ylim(gray.shape[0], 0)
    ax.set_facecolor('black')
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = histograms[cy, cx]
            center_y = cy*cell_size + cell_size/2
            center_x = cx*cell_size + cell_size/2
            max_val = hist.max() if hist.max() > 0 else 1
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.2:
                    angle = bin_idx*20 + 10
                    scale = (hist[bin_idx]/max_val) * (cell_size/2) * 0.8
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax.plot([center_x-dx, center_x+dx], [center_y-dy, center_y+dy],
                           color='white', linewidth=0.5)
    ax.set_title('10. HOG Visualization', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    
    ax = fig.add_subplot(3, 4, 11)
    block_idx = len(block_features)//2
    colors_block = np.tile(plt.cm.tab10(np.arange(4)/4), (9,1)).T.flatten().reshape(-1,4)
    ax.bar(range(36), block_features[block_idx], color=colors_block[:36], width=0.8)
    ax.set_title('11. Block (36-D)', fontsize=11, fontweight='bold')
    
    ax = fig.add_subplot(3, 4, 12)
    ax.bar(range(min(150, len(descriptor))), descriptor[:150], color='darkgreen', width=1)
    ax.set_title(f'12. Descriptor ({len(descriptor):,}-D)', fontsize=11, fontweight='bold')
    
    plt.suptitle('HOG Complete Pipeline (Real Image)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_complete_summary.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_complete_summary.png")


if __name__ == "__main__":
    print("=" * 70)
    print("Regenerating ALL HOG images with REAL input image")
    print("=" * 70)
    
    print("\n--- Step 1: Preprocessing ---")
    create_step1_preprocessing()
    
    print("\n--- Step 2: Gradients ---")
    create_step2_gradients()
    create_gradient_computation()
    create_edge_detection()
    create_signed_vs_unsigned()
    
    print("\n--- Step 3: Cell Histograms ---")
    create_step3_cell_division()
    create_step3_single_cell()
    create_step3_histogram_bins()
    create_step3_voting()
    create_step3_all_cells()
    create_step3_compression()
    create_step3_complete()
    create_step3_cell_histograms()
    create_cell_histogram_detail()
    create_multiple_cells()
    create_interpolation_detail()
    
    print("\n--- Step 4: Block Normalization ---")
    create_step4_block_definition()
    create_step4_overlap()
    create_step4_l2_normalization()
    create_step4_illumination()
    create_step4_final()
    create_step4_complete()
    create_block_normalization()
    
    print("\n--- Final Visualizations ---")
    create_visualization()
    create_visualization_explained()
    create_complete_summary()
    
    print("\n" + "=" * 70)
    print("Done! Regenerated 28 images with real input image data.")
    print("=" * 70)

"""
HOG Real Image Step-by-Step Visualizations
Shows actual algorithm results on real image at each step
Similar to SIFT's step3_keypoints, step4_refined, etc.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import LineCollection
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
# STEP 1 VISUALIZATIONS
# =============================================================================

def create_step1_real_grayscale():
    """Step 1.1: Real grayscale conversion result"""
    gray, original_rgb = load_image()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    ax.set_title(f'Original RGB Image\n{original_rgb.shape[1]}×{original_rgb.shape[0]} pixels', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    ax.imshow(gray, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Step 1.1: Grayscale Conversion\nI = 0.299R + 0.587G + 0.114B', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 1.1: Grayscale Conversion (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step1_1_grayscale.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step1_1_grayscale.png")


def create_step1_real_gamma():
    """Step 1.2: Real gamma correction result"""
    gray, _ = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax = axes[0]
    ax.imshow(gray, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Before Gamma Correction', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    ax.imshow(gray_gamma, cmap='gray', vmin=0, vmax=1)
    ax.set_title('After Gamma Correction (γ=0.5)\nI_γ = I^0.5', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Difference
    ax = axes[2]
    diff = gray_gamma - gray
    im = ax.imshow(diff, cmap='RdBu', vmin=-0.3, vmax=0.3)
    ax.set_title('Difference (After - Before)\nDark regions enhanced', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    plt.suptitle('HOG Step 1.2: Gamma Correction (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step1_2_gamma.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step1_2_gamma.png")


# =============================================================================
# STEP 2 VISUALIZATIONS
# =============================================================================

def create_step2_real_gx():
    """Step 2.1: Real Gx gradient result"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    _, _, gx, _ = compute_gradients(gray_gamma)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ax.imshow(gray_gamma, cmap='gray')
    ax.set_title('Input (After Preprocessing)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    im = ax.imshow(gx, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title('Step 2.1: Horizontal Gradient (Gx)\nGx = I(x+1,y) - I(x-1,y)\nBlue=negative, Red=positive', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    plt.suptitle('HOG Step 2.1: Horizontal Gradient Gx (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step2_1_gx.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step2_1_gx.png")


def create_step2_real_gy():
    """Step 2.2: Real Gy gradient result"""
    gray, _ = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    _, _, _, gy = compute_gradients(gray_gamma)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ax.imshow(gray_gamma, cmap='gray')
    ax.set_title('Input (After Preprocessing)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    im = ax.imshow(gy, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title('Step 2.2: Vertical Gradient (Gy)\nGy = I(x,y+1) - I(x,y-1)\nBlue=negative, Red=positive', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    plt.suptitle('HOG Step 2.2: Vertical Gradient Gy (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step2_2_gy.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step2_2_gy.png")


def create_step2_real_magnitude():
    """Step 2.3: Real magnitude result"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, _, _, _ = compute_gradients(gray_gamma)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    im = ax.imshow(magnitude, cmap='hot')
    ax.set_title('Step 2.3: Gradient Magnitude\nM = √(Gx² + Gy²)\nBright = strong edges', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7, label='Magnitude')
    
    plt.suptitle('HOG Step 2.3: Gradient Magnitude (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step2_3_magnitude.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step2_3_magnitude.png")


def create_step2_real_direction():
    """Step 2.4: Real direction result"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    # Mask low magnitude areas
    direction_masked = np.where(magnitude > 0.02, direction, np.nan)
    im = ax.imshow(direction_masked, cmap='hsv', vmin=0, vmax=180)
    ax.set_title('Step 2.4: Gradient Direction (0°-180°)\nθ = arctan(Gy/Gx) mod 180°\nColor = edge orientation', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7, label='Direction (degrees)')
    
    plt.suptitle('HOG Step 2.4: Gradient Direction (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step2_4_direction.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step2_4_direction.png")


def create_step2_real_vectors():
    """Step 2.5: Gradient vectors on real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ax.imshow(original_rgb, alpha=0.7)
    
    # Draw gradient vectors
    step = max(gray.shape[0] // 40, 8)
    for y in range(step, gray.shape[0] - step, step):
        for x in range(step, gray.shape[1] - step, step):
            mag = magnitude[y, x]
            angle = direction[y, x]
            if mag > 0.03:
                dx = mag * 12 * np.cos(np.radians(angle))
                dy = mag * 12 * np.sin(np.radians(angle))
                ax.arrow(x, y, dx, dy, head_width=2, head_length=1, fc='lime', ec='lime', linewidth=0.8)
    
    ax.set_title('Step 2.5: Gradient Vectors on Image\nArrow direction = gradient direction, length = magnitude', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 2.5: Gradient Vectors (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step2_5_vectors.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step2_5_vectors.png")


# =============================================================================
# STEP 3 VISUALIZATIONS
# =============================================================================

def create_step3_real_cells():
    """Step 3.1: Real cell grid overlay"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    
    cell_size = 8
    h, w = gray.shape
    cells_y = h // cell_size
    cells_x = w // cell_size
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ax.imshow(original_rgb)
    
    # Draw cell grid
    for cy in range(cells_y + 1):
        ax.axhline(y=cy * cell_size, color='lime', linewidth=0.5, alpha=0.7)
    for cx in range(cells_x + 1):
        ax.axvline(x=cx * cell_size, color='lime', linewidth=0.5, alpha=0.7)
    
    ax.set_title(f'Step 3.1: Cell Grid Overlay\n{cells_x}×{cells_y} = {cells_x*cells_y} cells (8×8 pixels each)', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 3.1: Cell Division (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step3_1_cells.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step3_1_cells.png")


def create_step3_real_histograms():
    """Step 3.2-3.4: Real cell histograms as HOG visualization"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cells_y, cells_x = histograms.shape[:2]
    cell_size = 8
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original with HOG overlay
    ax = axes[0]
    ax.imshow(original_rgb, alpha=0.4)
    
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
    
    ax.set_title(f'Step 3.2-3.4: HOG on Image\n{cells_x*cells_y} cell histograms', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # HOG only
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
                           color='white', linewidth=0.8, alpha=alpha)
    
    ax.set_title('Step 3.5: HOG Visualization\nEdge orientations per cell', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.suptitle('HOG Step 3: Cell Histograms (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step3_2_histograms.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step3_2_histograms.png")


def create_step3_real_dominant():
    """Step 3.5: Show dominant orientation per cell"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cells_y, cells_x = histograms.shape[:2]
    cell_size = 8
    
    # Create dominant orientation map
    dominant_map = np.zeros((cells_y, cells_x))
    for cy in range(cells_y):
        for cx in range(cells_x):
            dominant_map[cy, cx] = np.argmax(histograms[cy, cx]) * 20 + 10
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    ax = axes[0]
    ax.imshow(original_rgb)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1]
    im = ax.imshow(dominant_map, cmap='hsv', vmin=0, vmax=180, 
                   extent=[0, gray.shape[1], gray.shape[0], 0])
    ax.set_title('Step 3.5: Dominant Orientation per Cell\nColor = most frequent edge direction', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7, label='Dominant angle (degrees)')
    
    plt.suptitle('HOG Step 3.5: Dominant Orientations (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step3_3_dominant.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step3_3_dominant.png")


# =============================================================================
# STEP 4 VISUALIZATIONS
# =============================================================================

def create_step4_real_blocks():
    """Step 4.1-4.2: Real block grid overlay"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    
    cell_size = 8
    cells_y, cells_x = histograms.shape[:2]
    blocks_y = cells_y - 1
    blocks_x = cells_x - 1
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ax.imshow(original_rgb, alpha=0.5)
    
    # Draw some blocks with colors
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i in range(min(100, blocks_y * blocks_x)):
        by = i // blocks_x
        bx = i % blocks_x
        if by < blocks_y and bx < blocks_x:
            color = colors[i % 20]
            rect = Rectangle((bx * cell_size, by * cell_size), cell_size * 2, cell_size * 2,
                             fill=True, facecolor=color, alpha=0.3, edgecolor=color, linewidth=0.5)
            ax.add_patch(rect)
    
    ax.set_title(f'Step 4.1-4.2: Block Grid (2×2 cells, 50% overlap)\n{blocks_x}×{blocks_y} = {blocks_x*blocks_y} blocks', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('HOG Step 4.1-4.2: Block Layout (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step4_1_blocks.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step4_1_blocks.png")


def create_step4_real_normalized():
    """Step 4.3-4.4: Normalized HOG visualization"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    block_features, block_shape = block_normalize(histograms)
    
    cell_size = 8
    cells_y, cells_x = histograms.shape[:2]
    blocks_y, blocks_x = block_shape
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Before normalization (raw histograms)
    ax = axes[0]
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
                    ax.plot([center_x - dx, center_x + dx],
                           [center_y - dy, center_y + dy],
                           color='white', linewidth=0.8, alpha=0.8)
    
    ax.set_title('Before Block Normalization\n(Raw cell histograms)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    # After normalization
    ax = axes[1]
    ax.set_xlim(0, gray.shape[1])
    ax.set_ylim(gray.shape[0], 0)
    ax.set_facecolor('black')
    
    # Reconstruct visualization from normalized blocks
    # Average the normalized values for cells that appear in multiple blocks
    cell_normalized = np.zeros((cells_y, cells_x, 9))
    cell_counts = np.zeros((cells_y, cells_x))
    
    for by in range(blocks_y):
        for bx in range(blocks_x):
            block_idx = by * blocks_x + bx
            block = block_features[block_idx].reshape(2, 2, 9)
            for dy in range(2):
                for dx in range(2):
                    cy, cx = by + dy, bx + dx
                    cell_normalized[cy, cx] += block[dy, dx]
                    cell_counts[cy, cx] += 1
    
    cell_counts[cell_counts == 0] = 1
    cell_normalized /= cell_counts[:, :, np.newaxis]
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            hist = cell_normalized[cy, cx]
            center_y = cy * cell_size + cell_size / 2
            center_x = cx * cell_size + cell_size / 2
            
            max_val = hist.max() if hist.max() > 0 else 1
            for bin_idx in range(9):
                if hist[bin_idx] > max_val * 0.15:
                    angle = bin_idx * 20 + 10
                    scale = (hist[bin_idx] / max_val) * (cell_size / 2) * 0.9
                    dx = scale * np.cos(np.radians(angle))
                    dy = scale * np.sin(np.radians(angle))
                    ax.plot([center_x - dx, center_x + dx],
                           [center_y - dy, center_y + dy],
                           color='cyan', linewidth=0.8, alpha=0.8)
    
    ax.set_title('After Block Normalization\n(L2 normalized, illumination invariant)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.suptitle('HOG Step 4.3-4.4: Block Normalization Effect (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step4_2_normalized.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step4_2_normalized.png")


def create_step4_real_descriptor():
    """Step 4.5: Final descriptor visualization"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, _, _ = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    block_features, block_shape = block_normalize(histograms)
    descriptor = block_features.flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original image
    ax = axes[0, 0]
    ax.imshow(original_rgb)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # HOG visualization
    cell_size = 8
    cells_y, cells_x = histograms.shape[:2]
    
    ax = axes[0, 1]
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
                    ax.plot([center_x - dx, center_x + dx],
                           [center_y - dy, center_y + dy],
                           color='white', linewidth=0.8, alpha=0.8)
    
    ax.set_title('HOG Visualization', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    # Descriptor vector (first 500)
    ax = axes[1, 0]
    display_len = min(500, len(descriptor))
    ax.bar(range(display_len), descriptor[:display_len], color='darkgreen', width=1)
    ax.set_title(f'Final Descriptor (first {display_len} of {len(descriptor):,})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Value')
    
    # Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    blocks_y, blocks_x = block_shape
    cells_y, cells_x = histograms.shape[:2]
    
    stats = f"""
    HOG Descriptor Statistics
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Image:       {gray.shape[1]} × {gray.shape[0]} pixels
    
    Cells:       {cells_x} × {cells_y} = {cells_x*cells_y}
    
    Blocks:      {blocks_x} × {blocks_y} = {blocks_x*blocks_y}
    
    Descriptor:  {len(descriptor):,} dimensions
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    Value Range:
      Min:  {descriptor.min():.4f}
      Max:  {descriptor.max():.4f}
      Mean: {descriptor.mean():.4f}
      Std:  {descriptor.std():.4f}
    """
    ax.text(0.1, 0.5, stats, fontsize=11, family='monospace', va='center',
            transform=ax.transAxes, bbox=dict(facecolor='lightyellow', edgecolor='orange', linewidth=2))
    ax.set_title('Statistics', fontsize=12, fontweight='bold')
    
    plt.suptitle('HOG Step 4.5: Final Descriptor (Real Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_step4_3_descriptor.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_step4_3_descriptor.png")


# =============================================================================
# COMPLETE PIPELINE VISUALIZATION
# =============================================================================

def create_complete_pipeline():
    """Complete step-by-step pipeline on real image"""
    gray, original_rgb = load_image()
    gray_gamma = gamma_correction(gray, 0.5)
    magnitude, direction, gx, gy = compute_gradients(gray_gamma)
    histograms = compute_cell_histograms(magnitude, direction)
    block_features, block_shape = block_normalize(histograms)
    descriptor = block_features.flatten()
    
    cell_size = 8
    cells_y, cells_x = histograms.shape[:2]
    blocks_y, blocks_x = block_shape
    
    fig = plt.figure(figsize=(24, 16))
    
    # Row 1: Preprocessing
    ax = fig.add_subplot(3, 4, 1)
    ax.imshow(original_rgb)
    ax.set_title('Input: RGB Image', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(3, 4, 2)
    ax.imshow(gray, cmap='gray')
    ax.set_title('Step 1.1: Grayscale', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(3, 4, 3)
    ax.imshow(gray_gamma, cmap='gray')
    ax.set_title('Step 1.2: Gamma (γ=0.5)', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Row 1 continued: Gradients
    ax = fig.add_subplot(3, 4, 4)
    ax.imshow(gx, cmap='RdBu', vmin=-0.3, vmax=0.3)
    ax.set_title('Step 2.1: Gx', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Row 2: Gradients continued
    ax = fig.add_subplot(3, 4, 5)
    ax.imshow(gy, cmap='RdBu', vmin=-0.3, vmax=0.3)
    ax.set_title('Step 2.2: Gy', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(3, 4, 6)
    ax.imshow(magnitude, cmap='hot')
    ax.set_title('Step 2.3: Magnitude', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(3, 4, 7)
    ax.imshow(direction, cmap='hsv', vmin=0, vmax=180)
    ax.set_title('Step 2.4: Direction', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Cell grid
    ax = fig.add_subplot(3, 4, 8)
    ax.imshow(gray_gamma, cmap='gray')
    for cy in range(cells_y + 1):
        ax.axhline(y=cy * cell_size, color='lime', linewidth=0.3)
    for cx in range(cells_x + 1):
        ax.axvline(x=cx * cell_size, color='lime', linewidth=0.3)
    ax.set_title(f'Step 3.1: {cells_x*cells_y} Cells', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Row 3: Histograms and final
    ax = fig.add_subplot(3, 4, 9)
    sample_hist = histograms[cells_y//2, cells_x//2]
    colors = plt.cm.hsv(np.linspace(0, 0.5, 9))
    ax.bar(range(9), sample_hist, color=colors, edgecolor='black')
    ax.set_title('Step 3.2-3.4: Cell Histogram', fontsize=10, fontweight='bold')
    ax.set_xticks(range(9))
    ax.set_xticklabels([f'{i*20}°' for i in range(9)], fontsize=7)
    
    # HOG visualization
    ax = fig.add_subplot(3, 4, 10)
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
                            color='white', linewidth=0.5, alpha=0.8)
    ax.set_title('Step 3.5: HOG Visualization', fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    
    # Block vector
    ax = fig.add_subplot(3, 4, 11)
    block_idx = len(block_features) // 2
    colors_bar = plt.cm.tab10(np.repeat(np.arange(4), 9) / 4)
    ax.bar(range(36), block_features[block_idx], color=colors_bar, width=0.8)
    ax.set_title('Step 4.3: Block Vector (36-D)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Index')
    
    # Final descriptor
    ax = fig.add_subplot(3, 4, 12)
    display_len = 200
    ax.bar(range(display_len), descriptor[:display_len], color='darkgreen', width=1)
    ax.set_title(f'Step 4.5: Descriptor ({len(descriptor):,}-D)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Index')
    
    plt.suptitle('HOG Algorithm: Complete Pipeline on Real Image', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hog_real_complete_pipeline.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hog_real_complete_pipeline.png")


if __name__ == "__main__":
    print("=" * 70)
    print("Generating HOG Real Image Step-by-Step Visualizations")
    print("=" * 70)
    
    print("\n--- Step 1: Preprocessing ---")
    create_step1_real_grayscale()
    create_step1_real_gamma()
    
    print("\n--- Step 2: Gradient Computation ---")
    create_step2_real_gx()
    create_step2_real_gy()
    create_step2_real_magnitude()
    create_step2_real_direction()
    create_step2_real_vectors()
    
    print("\n--- Step 3: Cell Histograms ---")
    create_step3_real_cells()
    create_step3_real_histograms()
    create_step3_real_dominant()
    
    print("\n--- Step 4: Block Normalization ---")
    create_step4_real_blocks()
    create_step4_real_normalized()
    create_step4_real_descriptor()
    
    print("\n--- Complete Pipeline ---")
    create_complete_pipeline()
    
    print("\n" + "=" * 70)
    print("Done! Generated 15 real image step visualizations.")
    print("=" * 70)
